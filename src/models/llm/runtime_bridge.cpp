#include "models/llm/runtime_bridge.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace kimrt::llm {
namespace {

[[nodiscard]] Status failure(StatusCode code, std::string message) {
    return Status::error(code, std::move(message));
}

[[nodiscard]] ipc::SessionEgressConfig makeEgressConfig(
    RuntimeBridgeConfig const& config) {
    ipc::SessionEgressConfig egress;
    egress.max_frame_payload_bytes = config.limits.max_frame_payload_bytes;
    egress.max_session_frames = config.limits.max_session_egress_frames;
    egress.max_session_bytes = config.limits.max_session_egress_bytes;
    egress.max_request_frames = config.limits.max_request_egress_frames;
    egress.max_request_bytes = config.limits.max_request_egress_bytes;
    egress.control_reserve_frames = config.control_reserve_frames;
    egress.control_reserve_bytes = config.control_reserve_bytes;
    egress.terminal_reserve_bytes = config.terminal_reserve_bytes;
    return egress;
}

} // namespace

struct RuntimeBridge::Impl {
    enum class State : std::uint8_t {
        Created,
        Running,
        Stopping,
        Stopped,
        Failed,
    };

    struct RequestRecord {
        RequestRecord(
            std::uint64_t input_request_id,
            std::shared_ptr<GenerationMailbox> input_mailbox)
            : request_id(input_request_id),
              mailbox(std::move(input_mailbox)) {}

        void allowForwarding() {
            {
                std::lock_guard lock(gate_mutex);
                forwarding_allowed = true;
            }
            gate_condition.notify_all();
        }

        void abandonForwarding() noexcept {
            try {
                {
                    std::lock_guard lock(gate_mutex);
                    abandoned = true;
                }
                gate_condition.notify_all();
            } catch (...) {
            }
        }

        [[nodiscard]] bool waitUntilForwardingAllowed() {
            std::unique_lock lock(gate_mutex);
            gate_condition.wait(lock, [this] {
                return forwarding_allowed || abandoned;
            });
            return forwarding_allowed && !abandoned;
        }

        std::uint64_t request_id{0};
        std::shared_ptr<GenerationMailbox> mailbox;
        std::thread pump;
        std::atomic<bool> cancel_requested{false};
        std::atomic<bool> backpressure_requested{false};
        std::atomic<bool> terminal_enqueued{false};
        std::atomic<bool> pump_finished{false};
        std::mutex gate_mutex;
        std::condition_variable gate_condition;
        bool forwarding_allowed{false};
        bool abandoned{false};
    };

    Impl(
        GenerationRuntime& input_runtime,
        ipc::IpcSession& input_session,
        RuntimeBridgeConfig input_config)
        : runtime(input_runtime),
          session(input_session),
          config(std::move(input_config)),
          egress(makeEgressConfig(config)) {}

    [[nodiscard]] Status validateConfiguration() const {
        if (config.worker_epoch == 0) {
            return failure(
                StatusCode::InvalidInput,
                "RuntimeBridge worker epoch must be positive");
        }
        if (config.max_status_message_bytes == 0 ||
            config.mailbox_wait_timeout.count() <= 0 ||
            config.session_send_retry.count() <= 0 ||
            config.session_stall_timeout.count() <= 0 ||
            config.session_send_retry > config.session_stall_timeout) {
            return failure(
                StatusCode::InvalidInput,
                "RuntimeBridge timeout and message limits are invalid");
        }
        if (session.role() != ipc::IpcSessionRole::Server) {
            return failure(
                StatusCode::InvalidInput,
                "RuntimeBridge requires a server IPC session");
        }
        if (!runtime.running()) {
            return failure(
                StatusCode::NotReady,
                "RuntimeBridge requires a running GenerationRuntime");
        }

        auto const admission = runtime.admissionSnapshot();
        if (config.limits.max_active_requests !=
                admission.max_active_requests ||
            config.limits.max_total_input_tokens !=
                admission.max_total_input_tokens ||
            config.limits.max_reserved_output_tokens !=
                admission.max_reserved_output_tokens) {
            return failure(
                StatusCode::InvalidInput,
                "RuntimeBridge limits do not match Runtime Admission");
        }
        return Status::success();
    }

    [[nodiscard]] Status start() {
        auto status = validateConfiguration();
        if (!status.ok()) {
            return status;
        }

        {
            std::lock_guard lock(mutex);
            if (state != State::Created) {
                return failure(
                    StatusCode::AlreadyExists,
                    "RuntimeBridge has already been started");
            }
            state = State::Running;
            stop_requested = false;
            terminal_status = Status::success();
        }

        try {
            writer_thread = std::thread(&Impl::writerLoop, this);
        } catch (std::exception const& exception) {
            status = failure(
                StatusCode::InternalError,
                std::string{"failed to start RuntimeBridge writer: "} +
                    exception.what());
            beginShutdown(status);
            finishStoppedState();
            return status;
        }

        status = session.start([this](ipc::Message message) {
            handleMessage(std::move(message));
        });
        if (!status.ok()) {
            beginShutdown(status);
            joinThread(writer_thread);
            finishStoppedState();
            return status;
        }

        try {
            monitor_thread = std::thread(&Impl::monitorLoop, this);
        } catch (std::exception const& exception) {
            status = failure(
                StatusCode::InternalError,
                std::string{"failed to start RuntimeBridge monitor: "} +
                    exception.what());
            beginShutdown(status);
            static_cast<void>(session.stop());
            joinThread(writer_thread);
            finishStoppedState();
            return status;
        }
        return Status::success();
    }

    [[nodiscard]] Status stop() noexcept {
        try {
            beginShutdown(Status::success());
            static_cast<void>(session.stop());
            joinThread(monitor_thread);
            joinThread(writer_thread);
            joinAllRequestPumps();
            finishStoppedState();

            std::lock_guard lock(mutex);
            return terminal_status;
        } catch (...) {
            return failure(
                StatusCode::InternalError,
                "RuntimeBridge stop failed");
        }
    }

    void finishStoppedState() noexcept {
        try {
            std::lock_guard lock(mutex);
            if (state != State::Failed) {
                state = State::Stopped;
            }
        } catch (...) {
        }
    }

    static void joinThread(std::thread& thread) noexcept {
        if (thread.joinable() &&
            thread.get_id() != std::this_thread::get_id()) {
            thread.join();
        }
    }

    void beginShutdown(Status status) noexcept {
        std::vector<std::shared_ptr<RequestRecord>> owned;
        try {
            {
                std::lock_guard lock(mutex);
                if (stop_requested) {
                    if (!status.ok() && terminal_status.ok()) {
                        terminal_status = std::move(status);
                        state = State::Failed;
                    }
                    return;
                }
                stop_requested = true;
                if (status.ok()) {
                    state = State::Stopping;
                } else {
                    terminal_status = std::move(status);
                    state = State::Failed;
                }
                owned.reserve(requests.size());
                for (auto const& [request_id, record] : requests) {
                    (void)request_id;
                    owned.push_back(record);
                }
            }

            egress.stop();
            shutdown_condition.notify_all();
            for (auto const& record : owned) {
                record->abandonForwarding();
                cancelRecord(record, true);
            }
        } catch (...) {
        }
    }

    void cancelRecord(
        std::shared_ptr<RequestRecord> const& record,
        bool count_cancel) noexcept {
        if (!record || record->terminal_enqueued.load()) {
            return;
        }
        if (record->cancel_requested.exchange(true)) {
            return;
        }
        if (count_cancel) {
            std::lock_guard lock(mutex);
            ++cancelled_requests;
        }
        try {
            runtime.cancel(record->request_id);
        } catch (...) {
            beginShutdown(failure(
                StatusCode::InternalError,
                "GenerationRuntime cancel threw an exception"));
        }
    }

    void handleMessage(ipc::Message message) {
        reapCompletedRequests();
        std::visit(
            [this](auto typed_message) {
                using T = std::decay_t<decltype(typed_message)>;
                if constexpr (std::is_same_v<T, ipc::Submit>) {
                    handleSubmit(std::move(typed_message));
                } else if constexpr (std::is_same_v<T, ipc::Cancel>) {
                    handleCancel(typed_message);
                } else if constexpr (std::is_same_v<T, ipc::Health>) {
                    handleHealth(typed_message);
                } else {
                    failFromHandler(failure(
                        StatusCode::InvalidInput,
                        "RuntimeBridge received an unsupported message"));
                }
            },
            std::move(message));
    }

    [[noreturn]] void failFromHandler(Status status) {
        auto const message = status.message;
        beginShutdown(std::move(status));
        throw std::runtime_error(message);
    }

    [[nodiscard]] Status convertSubmit(
        ipc::Submit const& input,
        GenerationRequest& output) const {
        if (input.worker_epoch != config.worker_epoch) {
            return failure(
                StatusCode::InvalidInput,
                "Submit worker epoch does not match the active Worker");
        }
        if (input.priority < std::numeric_limits<int>::min() ||
            input.priority > std::numeric_limits<int>::max()) {
            return failure(
                StatusCode::InvalidInput,
                "Submit priority is outside the Runtime range");
        }
        if (input.sampling.temperature >
                std::numeric_limits<float>::max() ||
            input.sampling.top_p > std::numeric_limits<float>::max()) {
            return failure(
                StatusCode::InvalidInput,
                "Submit sampling value is outside the Runtime range");
        }

        output.context.request_id = input.request_id;
        output.context.priority = static_cast<int>(input.priority);
        output.context.trace_id = input.trace_id;
        if (input.timeout_ms.has_value()) {
            using Milliseconds = std::chrono::milliseconds;
            using Rep = Milliseconds::rep;
            if (*input.timeout_ms >
                static_cast<std::uint64_t>(
                    std::numeric_limits<Rep>::max())) {
                return failure(
                    StatusCode::InvalidInput,
                    "Submit timeout exceeds the Runtime range");
            }
            auto const timeout = Milliseconds(
                static_cast<Rep>(*input.timeout_ms));
            auto const now = std::chrono::steady_clock::now();
            if (timeout > std::chrono::duration_cast<Milliseconds>(
                    std::chrono::steady_clock::time_point::max() - now)) {
                return failure(
                    StatusCode::InvalidInput,
                    "Submit deadline exceeds the Runtime range");
            }
            output.context.deadline = now + timeout;
        }

        output.input_token_ids = input.input_token_ids;
        output.max_new_tokens = input.max_new_tokens;
        output.streaming = input.streaming;
        output.sampling.temperature =
            static_cast<float>(input.sampling.temperature);
        output.sampling.top_k = input.sampling.top_k;
        output.sampling.top_p =
            static_cast<float>(input.sampling.top_p);
        output.sampling.random_seed = input.sampling.random_seed;
        output.end_id = input.end_id;
        output.pad_id = input.pad_id;
        output.stop_sequences = input.stop_sequences;
        return Status::success();
    }

    void handleSubmit(ipc::Submit message) {
        GenerationRequest request;
        auto conversion = convertSubmit(message, request);
        if (!conversion.ok()) {
            enqueueRejected(message.request_id, std::move(conversion));
            return;
        }

        auto submission = runtime.submit(std::move(request));
        if (!submission.accepted()) {
            enqueueRejected(
                message.request_id,
                std::move(submission.status));
            return;
        }

        auto registration = egress.registerRequest(message.request_id);
        if (!registration.ok()) {
            try {
                runtime.cancel(message.request_id);
            } catch (...) {
            }
            enqueueRejected(
                message.request_id,
                failure(
                    StatusCode::ResourceExhausted,
                    "RuntimeBridge could not reserve request egress"));
            return;
        }

        auto record = std::make_shared<RequestRecord>(
            message.request_id,
            std::move(submission.mailbox));
        bool inserted{false};
        {
            std::lock_guard lock(mutex);
            auto const result =
                requests.emplace(message.request_id, record);
            inserted = result.second;
        }
        if (!inserted) {
            static_cast<void>(egress.abandonRequest(message.request_id));
            try {
                runtime.cancel(message.request_id);
            } catch (...) {
            }
            enqueueRejected(
                message.request_id,
                failure(
                    StatusCode::AlreadyExists,
                    "RuntimeBridge request id is still owned"));
            return;
        }

        try {
            record->pump = std::thread(
                &Impl::requestPumpLoop,
                this,
                record);
        } catch (std::exception const& exception) {
            record->abandonForwarding();
            {
                std::lock_guard lock(mutex);
                requests.erase(message.request_id);
            }
            static_cast<void>(egress.abandonRequest(message.request_id));
            try {
                runtime.cancel(message.request_id);
            } catch (...) {
            }
            enqueueRejected(
                message.request_id,
                failure(
                    StatusCode::InternalError,
                    std::string{"failed to start EventForwarder: "} +
                        exception.what()));
            return;
        }

        auto accepted = egress.enqueueControl(ipc::Message{ipc::Accepted{
            ipc::kProtocolVersion,
            config.worker_epoch,
            message.request_id}});
        if (!accepted.enqueued()) {
            record->abandonForwarding();
            cancelRecord(record, true);
            failFromHandler(std::move(accepted.status));
        }
        record->allowForwarding();
    }

    void enqueueRejected(std::uint64_t request_id, Status status) {
        status = boundedStatus(std::move(status));
        {
            std::lock_guard lock(mutex);
            ++rejected_requests;
        }
        auto result = egress.enqueueControl(ipc::Message{ipc::Rejected{
            ipc::kProtocolVersion,
            config.worker_epoch,
            request_id,
            std::move(status)}});
        if (!result.enqueued()) {
            failFromHandler(std::move(result.status));
        }
    }

    void handleCancel(ipc::Cancel const& message) {
        if (message.worker_epoch != config.worker_epoch) {
            failFromHandler(failure(
                StatusCode::InvalidInput,
                "Cancel worker epoch does not match the active Worker"));
        }

        std::shared_ptr<RequestRecord> record;
        {
            std::lock_guard lock(mutex);
            auto const iterator = requests.find(message.request_id);
            if (iterator != requests.end()) {
                record = iterator->second;
            }
        }
        cancelRecord(record, true);
    }

    void handleHealth(ipc::Health const& message) {
        if (message.worker_epoch != config.worker_epoch) {
            failFromHandler(failure(
                StatusCode::InvalidInput,
                "Health worker epoch does not match the active Worker"));
        }

        auto const bridge = snapshot();
        auto const admission = runtime.admissionSnapshot();
        ipc::Stats stats;
        stats.worker_epoch = config.worker_epoch;
        stats.probe_id = message.probe_id;
        stats.ready = bridge.running && runtime.running() && session.ready();
        stats.status = stats.ready
            ? Status::success()
            : failure(StatusCode::NotReady, "Worker is not ready");
        stats.active_requests = admission.active_requests;
        stats.reserved_input_tokens = admission.reserved_input_tokens;
        stats.reserved_output_tokens = admission.reserved_output_tokens;
        stats.session_egress_frames =
            bridge.egress.queued_frames + session.queuedEgressFrames();
        stats.session_egress_bytes =
            bridge.egress.queued_bytes + session.queuedEgressBytes();
        stats.session_egress_high_watermark_frames = std::max(
            stats.session_egress_frames,
            bridge.egress.high_watermark_frames);
        stats.session_egress_high_watermark_bytes = std::max(
            stats.session_egress_bytes,
            bridge.egress.high_watermark_bytes);
        stats.rejected_requests = bridge.rejected_requests;
        stats.backpressure_requests = bridge.backpressure_requests;
        stats.cancelled_requests = bridge.cancelled_requests;

        auto result = egress.enqueueControl(ipc::Message{std::move(stats)});
        if (!result.enqueued()) {
            failFromHandler(std::move(result.status));
        }
    }

    [[nodiscard]] Status boundedStatus(Status status) const {
        if (status.message.size() > config.max_status_message_bytes) {
            status.message.resize(config.max_status_message_bytes);
        }
        return status;
    }

    [[nodiscard]] static ipc::FinishReason wireFinishReason(
        FinishReason reason) {
        switch (reason) {
        case FinishReason::Eos:
            return ipc::FinishReason::Eos;
        case FinishReason::Length:
            return ipc::FinishReason::Length;
        case FinishReason::Stop:
            return ipc::FinishReason::Stop;
        case FinishReason::Cancelled:
            return ipc::FinishReason::Cancelled;
        case FinishReason::Timeout:
            return ipc::FinishReason::Timeout;
        case FinishReason::Backpressure:
            return ipc::FinishReason::Backpressure;
        }
        return ipc::FinishReason::Backpressure;
    }

    void requestPumpLoop(
        std::shared_ptr<RequestRecord> record) noexcept {
        try {
            if (!record->waitUntilForwardingAllowed()) {
                record->pump_finished.store(true);
                return;
            }

            while (!stopping()) {
                GenerationEvent event;
                auto const result = record->mailbox->waitPop(
                    event,
                    config.mailbox_wait_timeout);
                if (result == MailboxWaitResult::Timeout) {
                    continue;
                }
                if (result == MailboxWaitResult::Closed) {
                    failFromPump(failure(
                        StatusCode::InternalError,
                        "Accepted request Mailbox closed without Terminal"));
                    break;
                }

                if (auto* delta = std::get_if<TokenDelta>(&event)) {
                    if (record->cancel_requested.load()) {
                        continue;
                    }
                    ipc::TokenDelta wire;
                    wire.worker_epoch = config.worker_epoch;
                    wire.request_id = delta->request_id;
                    wire.sequence_no = delta->sequence_no;
                    wire.token_ids = std::move(delta->token_ids);
                    auto enqueue = egress.enqueueDelta(
                        record->request_id,
                        ipc::Message{std::move(wire)});
                    if (enqueue.enqueued()) {
                        continue;
                    }
                    if (enqueue.code ==
                            ipc::SessionEgressEnqueueCode::RequestFull ||
                        enqueue.code ==
                            ipc::SessionEgressEnqueueCode::SessionFull) {
                        {
                            std::lock_guard lock(mutex);
                            ++backpressure_requests;
                        }
                        record->backpressure_requested.store(true);
                        cancelRecord(record, false);
                        continue;
                    }
                    if (enqueue.code !=
                        ipc::SessionEgressEnqueueCode::Stopped) {
                        failFromPump(std::move(enqueue.status));
                    }
                    break;
                }

                auto terminal = std::get<TerminalEvent>(std::move(event));
                ipc::Terminal wire;
                wire.worker_epoch = config.worker_epoch;
                wire.request_id = terminal.request_id;
                if (record->backpressure_requested.load()) {
                    wire.status = failure(
                        StatusCode::QueueFull,
                        "request cancelled because IPC egress was full");
                    wire.finish_reason = ipc::FinishReason::Backpressure;
                } else {
                    wire.status = boundedStatus(std::move(terminal.status));
                }
                if (!wire.finish_reason.has_value() &&
                    terminal.finish_reason.has_value()) {
                    wire.finish_reason =
                        wireFinishReason(*terminal.finish_reason);
                }
                wire.usage.prompt_tokens = terminal.usage.prompt_tokens;
                wire.usage.completion_tokens =
                    terminal.usage.completion_tokens;

                auto enqueue = egress.enqueueTerminal(
                    record->request_id,
                    ipc::Message{std::move(wire)});
                if (!enqueue.enqueued()) {
                    if (enqueue.code !=
                        ipc::SessionEgressEnqueueCode::Stopped) {
                        failFromPump(std::move(enqueue.status));
                    }
                    break;
                }
                record->terminal_enqueued.store(true);
                break;
            }
        } catch (std::exception const& exception) {
            failFromPump(failure(
                StatusCode::InternalError,
                std::string{"EventForwarder failed: "} + exception.what()));
        } catch (...) {
            failFromPump(failure(
                StatusCode::InternalError,
                "EventForwarder failed with an unknown exception"));
        }
        record->pump_finished.store(true);
    }

    void failFromPump(Status status) noexcept {
        beginShutdown(std::move(status));
        static_cast<void>(session.stop());
    }

    [[nodiscard]] bool stopping() const noexcept {
        std::lock_guard lock(mutex);
        return stop_requested;
    }

    void writerLoop() noexcept {
        try {
            while (!stopping()) {
                ipc::Message message;
                auto const popped = egress.waitPop(
                    message,
                    config.mailbox_wait_timeout);
                if (popped == ipc::SessionEgressWaitResult::Timeout) {
                    continue;
                }
                if (popped == ipc::SessionEgressWaitResult::Stopped) {
                    return;
                }

                auto const stall_started =
                    std::chrono::steady_clock::now();
                while (!stopping()) {
                    auto status = session.send(message);
                    if (status.ok()) {
                        break;
                    }
                    if (status.code != StatusCode::QueueFull ||
                        std::chrono::steady_clock::now() - stall_started >=
                            config.session_stall_timeout) {
                        beginShutdown(std::move(status));
                        static_cast<void>(session.stop());
                        return;
                    }

                    std::unique_lock lock(mutex);
                    shutdown_condition.wait_for(
                        lock,
                        config.session_send_retry,
                        [this] { return stop_requested; });
                }
            }
        } catch (std::exception const& exception) {
            beginShutdown(failure(
                StatusCode::InternalError,
                std::string{"RuntimeBridge writer failed: "} +
                    exception.what()));
            static_cast<void>(session.stop());
        } catch (...) {
            beginShutdown(failure(
                StatusCode::InternalError,
                "RuntimeBridge writer failed with an unknown exception"));
            static_cast<void>(session.stop());
        }
    }

    void monitorLoop() noexcept {
        try {
            while (true) {
                auto status = session.waitUntilClosed(
                    config.mailbox_wait_timeout);
                if (status.code == StatusCode::Timeout) {
                    if (stopping()) {
                        return;
                    }
                    continue;
                }
                if (status.ok()) {
                    if (stopping()) {
                        return;
                    }
                    status = failure(
                        StatusCode::Unavailable,
                        "IPC session closed");
                }
                beginShutdown(std::move(status));
                return;
            }
        } catch (...) {
            beginShutdown(failure(
                StatusCode::InternalError,
                "RuntimeBridge monitor failed"));
        }
    }

    void reapCompletedRequests() {
        std::vector<std::shared_ptr<RequestRecord>> completed;
        {
            std::lock_guard lock(mutex);
            for (auto iterator = requests.begin();
                 iterator != requests.end();) {
                auto const& record = iterator->second;
                if (record->pump_finished.load() &&
                    record->terminal_enqueued.load()) {
                    completed.push_back(record);
                    iterator = requests.erase(iterator);
                } else {
                    ++iterator;
                }
            }
        }
        for (auto const& record : completed) {
            joinThread(record->pump);
        }
    }

    void joinAllRequestPumps() noexcept {
        std::vector<std::shared_ptr<RequestRecord>> owned;
        try {
            {
                std::lock_guard lock(mutex);
                owned.reserve(requests.size());
                for (auto& [request_id, record] : requests) {
                    (void)request_id;
                    owned.push_back(std::move(record));
                }
                requests.clear();
            }
            for (auto const& record : owned) {
                record->abandonForwarding();
                joinThread(record->pump);
            }
        } catch (...) {
        }
    }

    [[nodiscard]] bool running() const noexcept {
        std::lock_guard lock(mutex);
        return state == State::Running && !stop_requested;
    }

    [[nodiscard]] Status terminalStatus() const {
        std::lock_guard lock(mutex);
        return terminal_status;
    }

    [[nodiscard]] RuntimeBridgeSnapshot snapshot() const noexcept {
        try {
            RuntimeBridgeSnapshot result;
            {
                std::lock_guard lock(mutex);
                result.running =
                    state == State::Running && !stop_requested;
                for (auto const& [request_id, record] : requests) {
                    (void)request_id;
                    if (!record->terminal_enqueued.load()) {
                        ++result.owned_requests;
                    }
                }
                result.rejected_requests = rejected_requests;
                result.backpressure_requests = backpressure_requests;
                result.cancelled_requests = cancelled_requests;
            }
            result.egress = egress.snapshot();
            return result;
        } catch (...) {
            return {};
        }
    }

    GenerationRuntime& runtime;
    ipc::IpcSession& session;
    RuntimeBridgeConfig config;
    ipc::SessionEgress egress;

    mutable std::mutex mutex;
    std::condition_variable shutdown_condition;
    std::unordered_map<
        std::uint64_t,
        std::shared_ptr<RequestRecord>> requests;
    std::thread writer_thread;
    std::thread monitor_thread;
    State state{State::Created};
    bool stop_requested{false};
    Status terminal_status;
    std::uint64_t rejected_requests{0};
    std::uint64_t backpressure_requests{0};
    std::uint64_t cancelled_requests{0};
};

RuntimeBridge::RuntimeBridge(
    GenerationRuntime& runtime,
    ipc::IpcSession& session,
    RuntimeBridgeConfig config)
    : impl_(std::make_unique<Impl>(
          runtime,
          session,
          std::move(config))) {}

RuntimeBridge::~RuntimeBridge() {
    static_cast<void>(stop());
}

Status RuntimeBridge::start() {
    return impl_->start();
}

Status RuntimeBridge::stop() noexcept {
    return impl_->stop();
}

bool RuntimeBridge::running() const noexcept {
    return impl_->running();
}

Status RuntimeBridge::terminalStatus() const {
    return impl_->terminalStatus();
}

RuntimeBridgeSnapshot RuntimeBridge::snapshot() const noexcept {
    return impl_->snapshot();
}

} // namespace kimrt::llm
