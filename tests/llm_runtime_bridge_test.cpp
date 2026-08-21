#include "models/llm/runtime_bridge.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

using namespace std::chrono_literals;
using kimrt::Status;
using kimrt::StatusCode;
using kimrt::llm::AdmissionConfig;
using kimrt::llm::AdmissionSnapshot;
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationBackend;
using kimrt::llm::GenerationMailbox;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::GenerationRequest;
using kimrt::llm::GenerationRuntime;
using kimrt::llm::GenerationRuntimeConfig;
using kimrt::llm::RuntimeBridge;
using kimrt::llm::RuntimeBridgeConfig;
using kimrt::llm::TerminalEvent;
using namespace kimrt::llm::ipc;

void expect(bool condition, std::string_view message, int& failures) {
    if (condition) {
        return;
    }
    ++failures;
    std::cerr << "[FAIL] " << message << '\n';
}

[[nodiscard]] Status failure(StatusCode code, std::string message) {
    return Status::error(code, std::move(message));
}

template <typename Predicate>
[[nodiscard]] bool eventually(
    Predicate predicate,
    std::chrono::milliseconds timeout = 1s) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(2ms);
    }
    return predicate();
}

[[nodiscard]] bool resourcesAreZero(AdmissionSnapshot const& snapshot) {
    return snapshot.active_requests == 0 &&
        snapshot.reserved_input_tokens == 0 &&
        snapshot.reserved_output_tokens == 0;
}

class TemporarySocketDirectory final {
public:
    TemporarySocketDirectory() {
        char path[] = "/tmp/kimrt-bridge-XXXXXX";
        char* const created = ::mkdtemp(path);
        if (created == nullptr) {
            throw std::runtime_error(
                "failed to create bridge test directory");
        }
        directory_ = created;
        socket_path_ = directory_ + "/worker.sock";
    }

    ~TemporarySocketDirectory() {
        ::unlink(socket_path_.c_str());
        ::rmdir(directory_.c_str());
    }

    [[nodiscard]] std::string const& socketPath() const noexcept {
        return socket_path_;
    }

private:
    std::string directory_;
    std::string socket_path_;
};

struct ConnectionPair {
    UdsConnection client;
    UdsConnection server;
};

[[nodiscard]] ConnectionPair connectPair(UdsListener& listener) {
    UdsAcceptResult accepted;
    std::thread accept_thread([&] {
        accepted = listener.accept();
    });
    auto connected = UdsConnection::connect(listener.socketPath());
    accept_thread.join();
    if (!connected.ok()) {
        throw std::runtime_error(connected.status.message);
    }
    if (!accepted.ok()) {
        throw std::runtime_error(accepted.status.message);
    }
    return {
        std::move(connected.connection),
        std::move(accepted.connection),
    };
}

[[nodiscard]] ModelManifest makeManifest() {
    return ModelManifest{
        "tinyllama",
        "revision-1",
        "tokenizer-sha",
        "template-sha",
        "engine-sha",
        2,
        0,
        128,
        64,
        192,
        "fp16",
        2,
    };
}

[[nodiscard]] WorkerLimits makeLimits(
    std::uint32_t max_request_egress_frames = 8) {
    return WorkerLimits{
        2,
        64,
        64,
        kDefaultMaxFramePayloadBytes,
        128,
        4U * 1024U * 1024U,
        max_request_egress_frames,
        2U * 1024U * 1024U,
    };
}

[[nodiscard]] IpcSessionConfig makeSessionConfig(
    WorkerLimits const& limits) {
    IpcSessionConfig config;
    config.frame_codec.max_payload_bytes =
        limits.max_frame_payload_bytes;
    config.max_egress_frames = limits.max_session_egress_frames;
    config.max_egress_bytes = limits.max_session_egress_bytes;
    config.read_buffer_bytes = 4096;
    return config;
}

[[nodiscard]] HelloAck makeHelloAck(WorkerLimits const& limits) {
    return HelloAck{
        kProtocolVersion,
        77,
        makeManifest(),
        limits,
    };
}

[[nodiscard]] RuntimeBridgeConfig makeBridgeConfig(
    WorkerLimits limits) {
    RuntimeBridgeConfig config;
    config.worker_epoch = 77;
    config.limits = std::move(limits);
    config.control_reserve_frames = 8;
    config.control_reserve_bytes = 16U * 1024U;
    config.terminal_reserve_bytes = 2048;
    config.max_status_message_bytes = 1024;
    config.mailbox_wait_timeout = 10ms;
    config.session_send_retry = 2ms;
    config.session_stall_timeout = 1s;
    return config;
}

[[nodiscard]] GenerationRuntimeConfig makeRuntimeConfig(
    WorkerLimits const& limits) {
    GenerationRuntimeConfig config;
    config.admission = AdmissionConfig{
        limits.max_active_requests,
        static_cast<std::size_t>(limits.max_total_input_tokens),
        static_cast<std::size_t>(limits.max_reserved_output_tokens),
    };
    config.mailbox = GenerationMailboxConfig{16, 256};
    return config;
}

[[nodiscard]] Submit makeSubmit(std::uint64_t request_id) {
    Submit submit;
    submit.worker_epoch = 77;
    submit.request_id = request_id;
    submit.priority = 3;
    submit.timeout_ms = 5000;
    submit.trace_id = "trace-" + std::to_string(request_id);
    submit.input_token_ids = {1, 2, 3};
    submit.max_new_tokens = 8;
    submit.streaming = true;
    submit.sampling.temperature = 0.8;
    submit.sampling.top_k = 4;
    submit.sampling.top_p = 0.9;
    submit.sampling.random_seed = 123;
    submit.end_id = 2;
    submit.pad_id = 0;
    submit.stop_sequences = {{5, 6}};
    return submit;
}

[[nodiscard]] std::optional<std::uint64_t> requestId(
    Message const& message) {
    return std::visit(
        [](auto const& typed) -> std::optional<std::uint64_t> {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (
                std::is_same_v<T, Accepted> ||
                std::is_same_v<T, Rejected> ||
                std::is_same_v<T, TokenDelta> ||
                std::is_same_v<T, Terminal>) {
                return typed.request_id;
            }
            return std::nullopt;
        },
        message);
}

class FakeGenerationBackend final : public GenerationBackend {
public:
    Status start() override {
        std::lock_guard lock(mutex_);
        running_ = true;
        return Status::success();
    }

    Status submit(
        GenerationRequest request,
        std::shared_ptr<GenerationMailbox> mailbox) override {
        {
            std::lock_guard lock(mutex_);
            if (!running_) {
                return failure(
                    StatusCode::NotReady,
                    "Fake Backend is stopped");
            }
            auto const request_id = request.context.request_id;
            if (mailboxes_.find(request_id) != mailboxes_.end()) {
                return failure(
                    StatusCode::AlreadyExists,
                    "Fake Backend duplicate request");
            }
            requests_[request_id] = request;
            mailboxes_[request_id] = std::move(mailbox);
        }
        condition_.notify_all();
        return Status::success();
    }

    void cancel(std::uint64_t request_id) override {
        std::shared_ptr<GenerationMailbox> mailbox;
        std::size_t prompt_tokens{0};
        {
            std::lock_guard lock(mutex_);
            ++cancel_calls_[request_id];
            auto const mailbox_iterator = mailboxes_.find(request_id);
            if (mailbox_iterator == mailboxes_.end()) {
                return;
            }
            mailbox = mailbox_iterator->second;
            prompt_tokens = requests_[request_id].input_token_ids.size();
        }

        TerminalEvent terminal;
        terminal.request_id = request_id;
        terminal.status = failure(
            StatusCode::Cancelled,
            "request cancelled by Fake Backend");
        terminal.finish_reason = FinishReason::Cancelled;
        terminal.usage.prompt_tokens = prompt_tokens;
        if (mailbox->pushTerminal(std::move(terminal))) {
            eraseRequest(request_id, mailbox);
        }
    }

    void stop() override {
        std::vector<std::pair<
            std::uint64_t,
            std::shared_ptr<GenerationMailbox>>> active;
        {
            std::lock_guard lock(mutex_);
            running_ = false;
            for (auto const& [request_id, mailbox] : mailboxes_) {
                active.emplace_back(request_id, mailbox);
            }
        }
        for (auto const& [request_id, mailbox] : active) {
            TerminalEvent terminal;
            terminal.request_id = request_id;
            terminal.status = failure(
                StatusCode::Cancelled,
                "request stopped by Fake Backend");
            terminal.finish_reason = FinishReason::Cancelled;
            if (mailbox->pushTerminal(std::move(terminal))) {
                eraseRequest(request_id, mailbox);
            }
        }
    }

    [[nodiscard]] bool waitForSubmissions(
        std::size_t count,
        std::chrono::milliseconds timeout = 1s) const {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [this, count] {
            return requests_.size() >= count;
        });
    }

    [[nodiscard]] std::optional<GenerationRequest> request(
        std::uint64_t request_id) const {
        std::lock_guard lock(mutex_);
        auto const iterator = requests_.find(request_id);
        if (iterator == requests_.end()) {
            return std::nullopt;
        }
        return iterator->second;
    }

    [[nodiscard]] bool pushDelta(
        std::uint64_t request_id,
        std::uint64_t sequence_no,
        std::vector<std::int32_t> token_ids) {
        std::shared_ptr<GenerationMailbox> mailbox;
        {
            std::lock_guard lock(mutex_);
            auto const iterator = mailboxes_.find(request_id);
            if (iterator == mailboxes_.end()) {
                return false;
            }
            mailbox = iterator->second;
        }
        kimrt::llm::TokenDelta delta;
        delta.request_id = request_id;
        delta.sequence_no = sequence_no;
        delta.token_ids = std::move(token_ids);
        return mailbox->tryPushDelta(std::move(delta));
    }

    [[nodiscard]] bool finish(
        std::uint64_t request_id,
        std::size_t completion_tokens = 2) {
        std::shared_ptr<GenerationMailbox> mailbox;
        std::size_t prompt_tokens{0};
        {
            std::lock_guard lock(mutex_);
            auto const iterator = mailboxes_.find(request_id);
            if (iterator == mailboxes_.end()) {
                return false;
            }
            mailbox = iterator->second;
            prompt_tokens = requests_[request_id].input_token_ids.size();
        }

        TerminalEvent terminal;
        terminal.request_id = request_id;
        terminal.status = Status::success();
        terminal.finish_reason = FinishReason::Length;
        terminal.usage.prompt_tokens = prompt_tokens;
        terminal.usage.completion_tokens = completion_tokens;
        if (!mailbox->pushTerminal(std::move(terminal))) {
            return false;
        }
        eraseRequest(request_id, mailbox);
        return true;
    }

    [[nodiscard]] std::size_t cancelCalls(
        std::uint64_t request_id) const {
        std::lock_guard lock(mutex_);
        auto const iterator = cancel_calls_.find(request_id);
        return iterator == cancel_calls_.end() ? 0 : iterator->second;
    }

private:
    void eraseRequest(
        std::uint64_t request_id,
        std::shared_ptr<GenerationMailbox> const& mailbox) {
        std::lock_guard lock(mutex_);
        auto const iterator = mailboxes_.find(request_id);
        if (iterator != mailboxes_.end() && iterator->second == mailbox) {
            mailboxes_.erase(iterator);
        }
    }

    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    bool running_{false};
    std::unordered_map<std::uint64_t, GenerationRequest> requests_;
    std::unordered_map<
        std::uint64_t,
        std::shared_ptr<GenerationMailbox>> mailboxes_;
    std::unordered_map<std::uint64_t, std::size_t> cancel_calls_;
};

class BridgeFixture final {
public:
    explicit BridgeFixture(WorkerLimits limits = makeLimits())
        : limits_(std::move(limits)) {
        auto listening = UdsListener::listen(temporary_.socketPath());
        if (!listening.ok()) {
            throw std::runtime_error(listening.status.message);
        }
        listener_ = std::move(listening.listener);
        auto connections = connectPair(listener_);

        server_ = std::make_unique<IpcSession>(
            std::move(connections.server),
            makeSessionConfig(limits_),
            makeHelloAck(limits_));
        client_ = std::make_unique<IpcSession>(
            std::move(connections.client),
            makeSessionConfig(limits_),
            Hello{kProtocolVersion, "tinyllama", "revision-1"});

        auto backend = std::make_unique<FakeGenerationBackend>();
        backend_ = backend.get();
        runtime_ = std::make_unique<GenerationRuntime>(
            std::move(backend),
            makeRuntimeConfig(limits_));
        auto status = runtime_->start();
        if (!status.ok()) {
            throw std::runtime_error(status.message);
        }

        bridge_ = std::make_unique<RuntimeBridge>(
            *runtime_,
            *server_,
            makeBridgeConfig(limits_));
        status = bridge_->start();
        if (!status.ok()) {
            throw std::runtime_error(status.message);
        }
        status = client_->start([this](Message message) {
            {
                std::lock_guard lock(messages_mutex_);
                messages_.push_back(std::move(message));
            }
            messages_condition_.notify_all();
        });
        if (!status.ok()) {
            throw std::runtime_error(status.message);
        }
        if (!server_->waitUntilReady(1s).ok() ||
            !client_->waitUntilReady(1s).ok()) {
            throw std::runtime_error("RuntimeBridge handshake failed");
        }
    }

    ~BridgeFixture() {
        if (bridge_) {
            static_cast<void>(bridge_->stop());
        }
        if (client_) {
            static_cast<void>(client_->stop());
        }
        if (runtime_) {
            static_cast<void>(runtime_->stop());
        }
    }

    [[nodiscard]] Status send(Message message) {
        return client_->send(std::move(message));
    }

    [[nodiscard]] bool waitForRequestMessages(
        std::uint64_t request_id,
        std::size_t count,
        std::chrono::milliseconds timeout = 1s) const {
        std::unique_lock lock(messages_mutex_);
        return messages_condition_.wait_for(
            lock,
            timeout,
            [this, request_id, count] {
                std::size_t matching{0};
                for (auto const& message : messages_) {
                    if (requestId(message) == request_id) {
                        ++matching;
                    }
                }
                return matching >= count;
            });
    }

    [[nodiscard]] bool waitForStats(
        std::uint64_t probe_id,
        std::chrono::milliseconds timeout = 1s) const {
        std::unique_lock lock(messages_mutex_);
        return messages_condition_.wait_for(
            lock,
            timeout,
            [this, probe_id] {
                for (auto const& message : messages_) {
                    auto const* stats = std::get_if<Stats>(&message);
                    if (stats != nullptr && stats->probe_id == probe_id) {
                        return true;
                    }
                }
                return false;
            });
    }

    [[nodiscard]] std::vector<Message> requestMessages(
        std::uint64_t request_id) const {
        std::vector<Message> result;
        std::lock_guard lock(messages_mutex_);
        for (auto const& message : messages_) {
            if (requestId(message) == request_id) {
                result.push_back(message);
            }
        }
        return result;
    }

    [[nodiscard]] std::optional<Stats> stats(
        std::uint64_t probe_id) const {
        std::lock_guard lock(messages_mutex_);
        for (auto const& message : messages_) {
            auto const* stats = std::get_if<Stats>(&message);
            if (stats != nullptr && stats->probe_id == probe_id) {
                return *stats;
            }
        }
        return std::nullopt;
    }

    void disconnectClient() {
        static_cast<void>(client_->stop());
    }

    FakeGenerationBackend& backend() noexcept {
        return *backend_;
    }

    GenerationRuntime& runtime() noexcept {
        return *runtime_;
    }

    RuntimeBridge& bridge() noexcept {
        return *bridge_;
    }

private:
    WorkerLimits limits_;
    TemporarySocketDirectory temporary_;
    UdsListener listener_;
    std::unique_ptr<IpcSession> server_;
    std::unique_ptr<IpcSession> client_;
    FakeGenerationBackend* backend_{nullptr};
    std::unique_ptr<GenerationRuntime> runtime_;
    std::unique_ptr<RuntimeBridge> bridge_;
    mutable std::mutex messages_mutex_;
    mutable std::condition_variable messages_condition_;
    std::vector<Message> messages_;
};

void testTerminalAwareFairEgress(int& failures) {
    SessionEgressConfig config;
    config.max_frame_payload_bytes = 4096;
    config.max_session_frames = 10;
    config.max_session_bytes = 32U * 1024U;
    config.max_request_frames = 3;
    config.max_request_bytes = 8U * 1024U;
    config.control_reserve_frames = 2;
    config.control_reserve_bytes = 4U * 1024U;
    config.terminal_reserve_bytes = 1024;
    SessionEgress egress(config);

    expect(egress.registerRequest(1).ok(), "request 1 must reserve Terminal", failures);
    expect(egress.registerRequest(2).ok(), "request 2 must reserve Terminal", failures);
    expect(
        egress.enqueueControl(Message{Accepted{kProtocolVersion, 77, 1}}).enqueued(),
        "Accepted 1 must enter the control reserve",
        failures);
    expect(
        egress.enqueueControl(Message{Accepted{kProtocolVersion, 77, 2}}).enqueued(),
        "Accepted 2 must enter the control reserve",
        failures);

    Message popped;
    expect(
        egress.waitPop(popped, 0ms) == SessionEgressWaitResult::Message &&
            std::get<Accepted>(popped).request_id == 1,
        "control messages must be delivered first and in FIFO order",
        failures);
    expect(
        egress.waitPop(popped, 0ms) == SessionEgressWaitResult::Message &&
            std::get<Accepted>(popped).request_id == 2,
        "second control message must preserve FIFO order",
        failures);

    expect(egress.enqueueDelta(1, Message{TokenDelta{
        kProtocolVersion, 77, 1, 0, {10}}}).enqueued(),
        "request 1 first Delta must enqueue", failures);
    expect(egress.enqueueDelta(1, Message{TokenDelta{
        kProtocolVersion, 77, 1, 1, {11}}}).enqueued(),
        "request 1 second Delta must enqueue", failures);
    auto const full = egress.enqueueDelta(1, Message{TokenDelta{
        kProtocolVersion, 77, 1, 2, {12}}});
    expect(
        full.code == SessionEgressEnqueueCode::RequestFull,
        "ordinary Delta must not consume the Terminal frame reserve",
        failures);
    expect(egress.enqueueDelta(2, Message{TokenDelta{
        kProtocolVersion, 77, 2, 0, {20}}}).enqueued(),
        "request 2 Delta must enqueue", failures);
    expect(egress.enqueueTerminal(2, Message{Terminal{
        kProtocolVersion,
        77,
        2,
        Status::success(),
        kimrt::llm::ipc::FinishReason::Length,
        Usage{1, 1}}}).enqueued(),
        "request 2 Terminal must use its reserve", failures);
    expect(egress.enqueueTerminal(1, Message{Terminal{
        kProtocolVersion,
        77,
        1,
        Status::success(),
        kimrt::llm::ipc::FinishReason::Length,
        Usage{1, 2}}}).enqueued(),
        "request 1 Terminal must enqueue after a full data queue", failures);

    std::vector<std::pair<std::uint64_t, std::string>> order;
    while (egress.waitPop(popped, 0ms) ==
           SessionEgressWaitResult::Message) {
        if (auto const* delta = std::get_if<TokenDelta>(&popped)) {
            order.emplace_back(delta->request_id, "delta");
        } else if (auto const* terminal = std::get_if<Terminal>(&popped)) {
            order.emplace_back(terminal->request_id, "terminal");
        }
    }
    std::vector<std::pair<std::uint64_t, std::string>> const expected{
        {1, "delta"},
        {2, "delta"},
        {1, "delta"},
        {2, "terminal"},
        {1, "terminal"},
    };
    expect(
        order == expected,
        "request queues must round-robin while preserving per-request FIFO",
        failures);
    expect(
        egress.snapshot().registered_requests == 0,
        "Terminal delivery must retire request egress state",
        failures);
}

void testSubmitOrderingConversionAndHealth(int& failures) {
    BridgeFixture fixture;
    auto submit = makeSubmit(100);
    expect(
        fixture.send(Message{submit}).ok(),
        "client Submit must send",
        failures);
    expect(
        fixture.backend().waitForSubmissions(1),
        "RuntimeBridge must submit to the Backend",
        failures);

    auto converted = fixture.backend().request(100);
    expect(converted.has_value(), "Backend must retain converted request", failures);
    if (converted) {
        expect(converted->context.priority == 3, "priority must convert", failures);
        expect(
            converted->context.trace_id == "trace-100",
            "trace id must convert",
            failures);
        expect(
            converted->context.hasDeadline(),
            "timeout_ms must become a steady-clock deadline",
            failures);
        expect(
            converted->input_token_ids == std::vector<std::int32_t>({1, 2, 3}),
            "input tokens must convert without loss",
            failures);
        expect(
            converted->stop_sequences ==
                std::vector<std::vector<std::int32_t>>({{5, 6}}),
            "stop sequences must convert without loss",
            failures);
    }

    expect(
        fixture.backend().pushDelta(100, 0, {10, 11}),
        "Fake Backend must publish Delta",
        failures);
    expect(
        fixture.backend().finish(100, 2),
        "Fake Backend must publish Terminal",
        failures);
    expect(
        fixture.waitForRequestMessages(100, 3),
        "client must receive Accepted, Delta, and Terminal",
        failures);

    auto const messages = fixture.requestMessages(100);
    expect(
        messages.size() == 3 &&
            std::holds_alternative<Accepted>(messages[0]) &&
            std::holds_alternative<TokenDelta>(messages[1]) &&
            std::holds_alternative<Terminal>(messages[2]),
        "Accepted must precede every Delta and Terminal",
        failures);
    if (messages.size() == 3 && std::holds_alternative<Terminal>(messages[2])) {
        auto const& terminal = std::get<Terminal>(messages[2]);
        expect(terminal.status.ok(), "normal Terminal must succeed", failures);
        expect(
            terminal.finish_reason ==
                kimrt::llm::ipc::FinishReason::Length,
            "Terminal finish reason must convert",
            failures);
        expect(
            terminal.usage.prompt_tokens == 3 &&
                terminal.usage.completion_tokens == 2,
            "Terminal usage must convert",
            failures);
    }
    expect(
        eventually([&] {
            return resourcesAreZero(
                fixture.runtime().admissionSnapshot());
        }),
        "Terminal must release all Runtime resources",
        failures);

    expect(
        fixture.send(Message{Health{kProtocolVersion, 77, 9}}).ok(),
        "Health probe must send",
        failures);
    expect(
        fixture.waitForStats(9),
        "RuntimeBridge must answer Health with Stats",
        failures);
    auto const stats = fixture.stats(9);
    expect(
        stats.has_value() && stats->ready && stats->status.ok(),
        "Stats must report a ready RuntimeBridge",
        failures);
    if (stats) {
        expect(
            stats->active_requests == 0 &&
                stats->reserved_input_tokens == 0 &&
                stats->reserved_output_tokens == 0,
            "Stats must expose released Admission resources",
            failures);
    }
}

void testRejectedAndIdempotentCancel(int& failures) {
    BridgeFixture fixture;
    expect(
        fixture.send(Message{makeSubmit(200)}).ok() &&
            fixture.send(Message{makeSubmit(201)}).ok(),
        "two in-capacity requests must send",
        failures);
    expect(
        fixture.backend().waitForSubmissions(2),
        "two requests must reach the Backend",
        failures);
    expect(
        fixture.waitForRequestMessages(200, 1) &&
            fixture.waitForRequestMessages(201, 1),
        "both requests must be Accepted",
        failures);

    expect(
        fixture.send(Message{makeSubmit(202)}).ok(),
        "over-capacity Submit must reach RuntimeBridge",
        failures);
    expect(
        fixture.waitForRequestMessages(202, 1),
        "over-capacity request must receive Rejected",
        failures);
    auto const rejected_messages = fixture.requestMessages(202);
    expect(
        rejected_messages.size() == 1 &&
            std::holds_alternative<Rejected>(rejected_messages[0]) &&
            std::get<Rejected>(rejected_messages[0]).status.code ==
                StatusCode::ResourceExhausted,
        "Rejected must preserve the Runtime Admission status",
        failures);

    Cancel cancel{kProtocolVersion, 77, 200};
    expect(
        fixture.send(Message{cancel}).ok() &&
            fixture.send(Message{cancel}).ok(),
        "duplicate Cancel messages must send",
        failures);
    expect(
        fixture.waitForRequestMessages(200, 2),
        "cancelled Accepted request must receive one Terminal",
        failures);
    expect(
        eventually([&] {
            return fixture.backend().cancelCalls(200) == 1;
        }),
        "duplicate Cancel must reach the Backend only once",
        failures);

    expect(
        fixture.backend().finish(201),
        "other request must finish independently",
        failures);
    expect(
        fixture.waitForRequestMessages(201, 2),
        "other request must receive its Terminal",
        failures);
    expect(
        eventually([&] {
            return resourcesAreZero(
                fixture.runtime().admissionSnapshot());
        }),
        "reject and cancel paths must return all resources",
        failures);

    auto const request_200 = fixture.requestMessages(200);
    std::size_t terminals{0};
    for (auto const& message : request_200) {
        if (std::holds_alternative<Terminal>(message)) {
            ++terminals;
        }
    }
    expect(terminals == 1, "cancel must produce exactly one Terminal", failures);
    auto const bridge = fixture.bridge().snapshot();
    expect(
        bridge.rejected_requests == 1 &&
            bridge.cancelled_requests == 1,
        "RuntimeBridge counters must distinguish reject and cancel",
        failures);
}

void testBackpressurePreservesTerminal(int& failures) {
    BridgeFixture fixture{makeLimits(1)};
    expect(
        fixture.send(Message{makeSubmit(250)}).ok(),
        "backpressure request must send",
        failures);
    expect(
        fixture.backend().waitForSubmissions(1) &&
            fixture.waitForRequestMessages(250, 1),
        "backpressure request must be Accepted first",
        failures);
    expect(
        fixture.backend().pushDelta(250, 0, {42}),
        "Fake Backend must publish the saturating Delta",
        failures);
    expect(
        fixture.waitForRequestMessages(250, 2),
        "Terminal reserve must remain deliverable after data saturation",
        failures);

    auto const messages = fixture.requestMessages(250);
    expect(
        messages.size() == 2 &&
            std::holds_alternative<Accepted>(messages[0]) &&
            std::holds_alternative<Terminal>(messages[1]),
        "saturated request must receive Accepted then Terminal without Delta",
        failures);
    if (messages.size() == 2 && std::holds_alternative<Terminal>(messages[1])) {
        auto const& terminal = std::get<Terminal>(messages[1]);
        expect(
            terminal.status.code == StatusCode::QueueFull &&
                terminal.finish_reason ==
                    kimrt::llm::ipc::FinishReason::Backpressure,
            "egress saturation must terminate as Backpressure",
            failures);
    }
    expect(
        fixture.backend().cancelCalls(250) == 1,
        "backpressure must cancel only the saturated request",
        failures);
    expect(
        fixture.bridge().snapshot().backpressure_requests == 1,
        "backpressure counter must increment",
        failures);
    expect(
        eventually([&] {
            return resourcesAreZero(
                fixture.runtime().admissionSnapshot());
        }),
        "backpressure Terminal must release Runtime resources",
        failures);
}

void testDisconnectCancelsOwnedRequests(int& failures) {
    BridgeFixture fixture;
    expect(
        fixture.send(Message{makeSubmit(300)}).ok() &&
            fixture.send(Message{makeSubmit(301)}).ok(),
        "disconnect requests must send",
        failures);
    expect(
        fixture.backend().waitForSubmissions(2) &&
            fixture.waitForRequestMessages(300, 1) &&
            fixture.waitForRequestMessages(301, 1),
        "disconnect requests must be Accepted",
        failures);

    fixture.disconnectClient();
    expect(
        eventually([&] {
            return fixture.backend().cancelCalls(300) == 1 &&
                fixture.backend().cancelCalls(301) == 1;
        }),
        "peer disconnect must cancel every owned request",
        failures);
    expect(
        eventually([&] {
            return resourcesAreZero(
                fixture.runtime().admissionSnapshot());
        }),
        "disconnect cancellation must return all Runtime resources",
        failures);
    expect(
        eventually([&] { return !fixture.bridge().running(); }),
        "RuntimeBridge must leave Running after peer disconnect",
        failures);
    expect(
        fixture.bridge().terminalStatus().code == StatusCode::Unavailable,
        "peer disconnect must preserve Unavailable terminal status",
        failures);
    expect(
        fixture.bridge().snapshot().cancelled_requests == 2,
        "disconnect must count both owned cancellations",
        failures);
}

void testBridgeStopIsSuccessfulAndIdempotent(int& failures) {
    BridgeFixture fixture;
    auto const first_stop = fixture.bridge().stop();
    expect(
        first_stop.ok(),
        "intentional RuntimeBridge stop must not become Unavailable",
        failures);
    auto const second_stop = fixture.bridge().stop();
    expect(
        second_stop.ok(),
        "repeated RuntimeBridge stop must preserve success",
        failures);
    expect(
        !fixture.bridge().running(),
        "stopped RuntimeBridge must not report Running",
        failures);
}

} // namespace

int main() {
    int failures{0};
    try {
        testTerminalAwareFairEgress(failures);
        testSubmitOrderingConversionAndHealth(failures);
        testRejectedAndIdempotentCancel(failures);
        testBackpressurePreservesTerminal(failures);
        testDisconnectCancelsOwnedRequests(failures);
        testBridgeStopIsSuccessfulAndIdempotent(failures);
    } catch (std::exception const& exception) {
        ++failures;
        std::cerr << "[FAIL] unexpected exception: "
                  << exception.what() << '\n';
    }

    if (failures == 0) {
        std::cout << "llm_runtime_bridge_contract: PASS\n";
        return 0;
    }
    std::cerr << "llm_runtime_bridge_contract: "
              << failures << " failure(s)\n";
    return 1;
}
