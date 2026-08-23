#include "worker/worker_server.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationBackend;
using kimrt::llm::GenerationMailbox;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::GenerationRequest;
using kimrt::llm::GenerationRuntime;
using kimrt::llm::GenerationRuntimeConfig;
using kimrt::llm::RuntimeBridgeConfig;
using kimrt::llm::TerminalEvent;
using kimrt::llm::WorkerServer;
using kimrt::llm::WorkerServerConfig;
using namespace kimrt::llm::ipc;

void expect(bool condition, std::string_view message, int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "[FAIL] " << message << '\n';
    }
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

class TemporarySocketDirectory final {
public:
    TemporarySocketDirectory() {
        char path[] = "/tmp/kim-llm-worker-server-XXXXXX";
        char* const created = ::mkdtemp(path);
        if (created == nullptr) {
            throw std::runtime_error(
                "failed to create WorkerServer test directory");
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

class FakeBackend final : public GenerationBackend {
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
                return Status::error(
                    StatusCode::NotReady,
                    "FakeBackend is stopped");
            }
            mailboxes_[request.context.request_id] = std::move(mailbox);
            prompt_tokens_[request.context.request_id] =
                request.input_token_ids.size();
        }
        condition_.notify_all();
        return Status::success();
    }

    void cancel(std::uint64_t request_id) override {
        static_cast<void>(finish(
            request_id,
            Status::error(StatusCode::Cancelled, "request cancelled"),
            FinishReason::Cancelled));
    }

    void stop() override {
        std::vector<std::uint64_t> active;
        {
            std::lock_guard lock(mutex_);
            running_ = false;
            for (auto const& [request_id, mailbox] : mailboxes_) {
                (void)mailbox;
                active.push_back(request_id);
            }
        }
        for (auto const request_id : active) {
            cancel(request_id);
        }
    }

    [[nodiscard]] bool waitForRequest(
        std::uint64_t request_id,
        std::chrono::milliseconds timeout = 1s) const {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [this, request_id] {
            return mailboxes_.find(request_id) != mailboxes_.end();
        });
    }

    [[nodiscard]] bool complete(std::uint64_t request_id) {
        return finish(
            request_id,
            Status::success(),
            FinishReason::Length);
    }

private:
    [[nodiscard]] bool finish(
        std::uint64_t request_id,
        Status status,
        FinishReason reason) {
        std::shared_ptr<GenerationMailbox> mailbox;
        std::size_t prompt_tokens{0};
        {
            std::lock_guard lock(mutex_);
            auto const iterator = mailboxes_.find(request_id);
            if (iterator == mailboxes_.end()) {
                return false;
            }
            mailbox = iterator->second;
            prompt_tokens = prompt_tokens_[request_id];
        }

        TerminalEvent terminal;
        terminal.request_id = request_id;
        terminal.status = std::move(status);
        terminal.finish_reason = reason;
        terminal.usage.prompt_tokens = prompt_tokens;
        if (!mailbox->pushTerminal(std::move(terminal))) {
            return false;
        }

        std::lock_guard lock(mutex_);
        mailboxes_.erase(request_id);
        prompt_tokens_.erase(request_id);
        condition_.notify_all();
        return true;
    }

    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    bool running_{false};
    std::unordered_map<
        std::uint64_t,
        std::shared_ptr<GenerationMailbox>> mailboxes_;
    std::unordered_map<std::uint64_t, std::size_t> prompt_tokens_;
};

class MessageCollector final {
public:
    void push(Message message) {
        {
            std::lock_guard lock(mutex_);
            messages_.push_back(std::move(message));
        }
        condition_.notify_all();
    }

    template <typename T>
    [[nodiscard]] std::optional<T> waitFor(
        std::uint64_t request_id,
        std::chrono::milliseconds timeout = 1s) const {
        std::unique_lock lock(mutex_);
        auto const find_message = [this, request_id]() -> std::optional<T> {
            for (auto const& message : messages_) {
                auto const* typed = std::get_if<T>(&message);
                if (typed != nullptr && typed->request_id == request_id) {
                    return *typed;
                }
            }
            return std::nullopt;
        };
        if (!condition_.wait_for(lock, timeout, [&] {
                return find_message().has_value();
            })) {
            return std::nullopt;
        }
        return find_message();
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    std::vector<Message> messages_;
};

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

[[nodiscard]] WorkerLimits makeLimits() {
    return WorkerLimits{
        2,
        64,
        64,
        kDefaultMaxFramePayloadBytes,
        128,
        4U * 1024U * 1024U,
        32,
        2U * 1024U * 1024U,
    };
}

[[nodiscard]] WorkerServerConfig makeServerConfig(
    std::string socket_path,
    WorkerLimits limits) {
    WorkerServerConfig config;
    config.socket_path = std::move(socket_path);
    config.listen_backlog = 4;
    config.supervisor_poll_interval = 5ms;
    config.manifest = makeManifest();
    config.session.frame_codec.max_payload_bytes =
        limits.max_frame_payload_bytes;
    config.session.max_egress_frames =
        limits.max_session_egress_frames;
    config.session.max_egress_bytes = limits.max_session_egress_bytes;
    config.session.read_buffer_bytes = 4096;
    config.bridge.worker_epoch = 77;
    config.bridge.limits = std::move(limits);
    config.bridge.control_reserve_frames = 8;
    config.bridge.control_reserve_bytes = 16U * 1024U;
    config.bridge.terminal_reserve_bytes = 2048;
    config.bridge.mailbox_wait_timeout = 5ms;
    config.bridge.session_send_retry = 2ms;
    config.bridge.session_stall_timeout = 1s;
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

[[nodiscard]] IpcSessionConfig makeClientSessionConfig(
    WorkerLimits const& limits) {
    IpcSessionConfig config;
    config.frame_codec.max_payload_bytes = limits.max_frame_payload_bytes;
    config.max_egress_frames = limits.max_session_egress_frames;
    config.max_egress_bytes = limits.max_session_egress_bytes;
    config.read_buffer_bytes = 4096;
    return config;
}

[[nodiscard]] std::unique_ptr<IpcSession> connectClient(
    std::string const& socket_path,
    WorkerLimits const& limits,
    MessageCollector& collector) {
    auto connected = UdsConnection::connect(socket_path);
    if (!connected.ok()) {
        throw std::runtime_error(connected.status.message);
    }
    auto session = std::make_unique<IpcSession>(
        std::move(connected.connection),
        makeClientSessionConfig(limits),
        Hello{kProtocolVersion, "tinyllama", "revision-1"});
    auto const start_status = session->start(
        [&collector](Message message) {
            collector.push(std::move(message));
        });
    if (!start_status.ok()) {
        throw std::runtime_error(start_status.message);
    }
    return session;
}

[[nodiscard]] Submit makeSubmit(std::uint64_t request_id) {
    Submit submit;
    submit.worker_epoch = 77;
    submit.request_id = request_id;
    submit.timeout_ms = 5000;
    submit.trace_id = "worker-server-test";
    submit.input_token_ids = {1, 2, 3};
    submit.max_new_tokens = 8;
    submit.streaming = true;
    return submit;
}

} // namespace

int main() {
    int failures{0};
    try {
        TemporarySocketDirectory sockets;
        auto const limits = makeLimits();
        auto backend = std::make_unique<FakeBackend>();
        auto* const backend_view = backend.get();
        GenerationRuntime runtime(
            std::move(backend),
            makeRuntimeConfig(limits));
        expect(runtime.start().ok(), "Runtime must start", failures);

        WorkerServer server(
            runtime,
            makeServerConfig(sockets.socketPath(), limits));
        expect(server.start().ok(), "WorkerServer must start", failures);

        MessageCollector first_messages;
        auto first = connectClient(
            sockets.socketPath(),
            limits,
            first_messages);
        expect(
            first->waitUntilReady(1s).ok(),
            "first Gateway Session must complete handshake",
            failures);

        MessageCollector rejected_messages;
        auto rejected = connectClient(
            sockets.socketPath(),
            limits,
            rejected_messages);
        expect(
            !rejected->waitUntilReady(1s).ok(),
            "second active Gateway Session must be rejected",
            failures);
        static_cast<void>(rejected->stop());
        expect(
            eventually([&] {
                return server.snapshot().rejected_connections == 1;
            }),
            "WorkerServer must count the rejected second connection",
            failures);

        expect(
            first->send(Message{makeSubmit(100)}).ok(),
            "first Session Submit must enqueue",
            failures);
        expect(
            backend_view->waitForRequest(100),
            "WorkerServer must route Submit to Runtime",
            failures);
        expect(
            first_messages.waitFor<Accepted>(100).has_value(),
            "Accepted must return through WorkerServer",
            failures);
        expect(
            backend_view->complete(100),
            "FakeBackend must complete the request",
            failures);
        auto const terminal = first_messages.waitFor<Terminal>(100);
        expect(
            terminal.has_value() && terminal->status.ok(),
            "Terminal must return through WorkerServer",
            failures);

        static_cast<void>(first->stop());
        expect(
            eventually([&] {
                return !server.snapshot().active_session;
            }),
            "WorkerServer must reap a disconnected Session",
            failures);

        MessageCollector replacement_messages;
        auto replacement = connectClient(
            sockets.socketPath(),
            limits,
            replacement_messages);
        expect(
            replacement->waitUntilReady(1s).ok(),
            "a replacement Session must connect after cleanup",
            failures);
        expect(
            eventually([&] {
                auto const snapshot = server.snapshot();
                return snapshot.accepted_sessions == 2 &&
                    snapshot.active_session_ready;
            }),
            "WorkerServer must expose replacement Session state",
            failures);

        expect(server.stop().ok(), "WorkerServer stop must converge", failures);
        static_cast<void>(replacement->waitUntilClosed(1s));
        static_cast<void>(replacement->stop());
        expect(runtime.stop().ok(), "Runtime stop must converge", failures);
        expect(
            ::access(sockets.socketPath().c_str(), F_OK) != 0,
            "WorkerServer stop must remove the owned socket path",
            failures);
    } catch (std::exception const& exception) {
        ++failures;
        std::cerr << "[FAIL] unexpected exception: "
                  << exception.what() << '\n';
    }

    if (failures != 0) {
        std::cerr << "llm_worker_server_contract: "
                  << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "llm_worker_server_contract: PASS\n";
    return 0;
}
