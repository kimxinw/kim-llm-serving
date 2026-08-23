#include "worker/worker_server.h"

#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/eventfd.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace kimrt::llm {
namespace {

[[nodiscard]] Status failure(StatusCode code, std::string message) {
    return Status::error(code, std::move(message));
}

[[nodiscard]] Status systemFailure(
    std::string message,
    int error_number) {
    return failure(
        StatusCode::Unavailable,
        std::move(message) + ": " + std::strerror(error_number));
}

} // namespace

struct WorkerServer::Impl {
    enum class State : std::uint8_t {
        Created,
        Running,
        Stopping,
        Stopped,
        Failed,
    };

    struct ActiveSession final {
        ActiveSession(
            GenerationRuntime& runtime,
            ipc::UdsConnection connection,
            ipc::IpcSessionConfig session_config,
            ipc::HelloAck hello_ack,
            RuntimeBridgeConfig bridge_config)
            : session(std::make_unique<ipc::IpcSession>(
                  std::move(connection),
                  std::move(session_config),
                  std::move(hello_ack))),
              bridge(std::make_unique<RuntimeBridge>(
                  runtime,
                  *session,
                  std::move(bridge_config))) {}

        [[nodiscard]] Status start() {
            return bridge->start();
        }

        [[nodiscard]] Status stop() noexcept {
            return bridge->stop();
        }

        [[nodiscard]] bool running() const noexcept {
            return bridge->running();
        }

        [[nodiscard]] bool ready() const noexcept {
            return session->ready();
        }

        [[nodiscard]] Status terminalStatus() const {
            return bridge->terminalStatus();
        }

        [[nodiscard]] RuntimeBridgeSnapshot snapshot() const noexcept {
            return bridge->snapshot();
        }

        std::unique_ptr<ipc::IpcSession> session;
        std::unique_ptr<RuntimeBridge> bridge;
    };

    Impl(
        GenerationRuntime& input_runtime,
        WorkerServerConfig input_config)
        : runtime(input_runtime),
          config(std::move(input_config)) {}

    ~Impl() {
        static_cast<void>(stop());
    }

    [[nodiscard]] ipc::HelloAck helloAck() const {
        return ipc::HelloAck{
            ipc::kProtocolVersion,
            config.bridge.worker_epoch,
            config.manifest,
            config.bridge.limits,
        };
    }

    [[nodiscard]] Status validateConfiguration() const {
        if (config.socket_path.empty()) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer socket path must not be empty");
        }
        if (config.listen_backlog <= 0 ||
            config.supervisor_poll_interval.count() <= 0) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer backlog and poll interval must be positive");
        }
        if (!runtime.running()) {
            return failure(
                StatusCode::NotReady,
                "WorkerServer requires a running GenerationRuntime");
        }

        auto const encoded_hello =
            ipc::encodePayload(ipc::Message{helloAck()});
        if (!encoded_hello.ok()) {
            return encoded_hello.status;
        }

        auto const& limits = config.bridge.limits;
        if (limits.max_active_requests > 8) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer v0.1 supports at most 8 active requests");
        }
        if (config.session.frame_codec.max_payload_bytes !=
                limits.max_frame_payload_bytes ||
            config.session.max_egress_frames !=
                limits.max_session_egress_frames ||
            config.session.max_egress_bytes !=
                limits.max_session_egress_bytes ||
            config.session.read_buffer_bytes == 0) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer Session limits do not match advertised limits");
        }

        if (config.bridge.control_reserve_frames == 0 ||
            config.bridge.control_reserve_bytes == 0 ||
            config.bridge.terminal_reserve_bytes == 0 ||
            config.bridge.control_reserve_frames >=
                limits.max_session_egress_frames ||
            config.bridge.control_reserve_bytes >=
                limits.max_session_egress_bytes ||
            limits.max_request_egress_frames >
                limits.max_session_egress_frames -
                    config.bridge.control_reserve_frames ||
            limits.max_request_egress_bytes >
                limits.max_session_egress_bytes -
                    config.bridge.control_reserve_bytes ||
            config.bridge.terminal_reserve_bytes >
                limits.max_request_egress_bytes) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer Bridge reserve hierarchy is invalid");
        }

        if (config.bridge.max_status_message_bytes == 0 ||
            config.bridge.mailbox_wait_timeout.count() <= 0 ||
            config.bridge.session_send_retry.count() <= 0 ||
            config.bridge.session_stall_timeout.count() <= 0 ||
            config.bridge.session_send_retry >
                config.bridge.session_stall_timeout) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer Bridge timeout configuration is invalid");
        }

        auto const admission = runtime.admissionSnapshot();
        if (limits.max_active_requests != admission.max_active_requests ||
            limits.max_total_input_tokens !=
                admission.max_total_input_tokens ||
            limits.max_reserved_output_tokens !=
                admission.max_reserved_output_tokens) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer limits do not match Runtime Admission");
        }
        return Status::success();
    }

    [[nodiscard]] Status start() {
        std::lock_guard lifecycle_lock(lifecycle_mutex);
        {
            std::lock_guard lock(mutex);
            if (state != State::Created) {
                return failure(
                    StatusCode::AlreadyExists,
                    "WorkerServer has already been started");
            }
        }

        auto status = validateConfiguration();
        if (!status.ok()) {
            return status;
        }

        auto listen_result = ipc::UdsListener::listen(
            config.socket_path,
            config.listen_backlog);
        if (!listen_result.ok()) {
            return listen_result.status;
        }

        int const new_wake_fd = ::eventfd(
            0,
            EFD_CLOEXEC | EFD_NONBLOCK);
        if (new_wake_fd < 0) {
            return systemFailure(
                "failed to create WorkerServer wake event",
                errno);
        }

        listener = std::move(listen_result.listener);
        wake_fd = new_wake_fd;
        {
            std::lock_guard lock(mutex);
            state = State::Running;
            stop_requested = false;
            terminal_status = Status::success();
        }

        try {
            accept_thread = std::thread(&Impl::acceptLoop, this);
        } catch (std::exception const& exception) {
            status = failure(
                StatusCode::InternalError,
                std::string{"failed to start WorkerServer accept thread: "} +
                    exception.what());
            fail(status);
            listener.close();
            closeWakeFd();
            return status;
        }
        return Status::success();
    }

    [[nodiscard]] Status waitUntilStopped(
        std::chrono::milliseconds timeout) const {
        if (timeout.count() < 0) {
            return failure(
                StatusCode::InvalidInput,
                "WorkerServer stop timeout must be non-negative");
        }
        std::unique_lock lock(mutex);
        bool const completed = state_condition.wait_for(
            lock,
            timeout,
            [this] {
                return state == State::Stopped || state == State::Failed;
            });
        if (!completed) {
            return failure(
                StatusCode::Timeout,
                "timed out waiting for WorkerServer to stop");
        }
        return terminal_status;
    }

    [[nodiscard]] Status stop() noexcept {
        try {
            std::lock_guard lifecycle_lock(lifecycle_mutex);
            {
                std::lock_guard lock(mutex);
                if (state == State::Created) {
                    state = State::Stopped;
                    state_condition.notify_all();
                    return Status::success();
                }
                if (state == State::Stopped) {
                    return terminal_status;
                }
                stop_requested = true;
                if (state != State::Failed) {
                    state = State::Stopping;
                }
            }

            signalWake();
            if (accept_thread.joinable() &&
                accept_thread.get_id() != std::this_thread::get_id()) {
                accept_thread.join();
            }
            listener.close();
            closeWakeFd();

            std::lock_guard lock(mutex);
            if (state != State::Failed) {
                state = State::Stopped;
            }
            state_condition.notify_all();
            return terminal_status;
        } catch (...) {
            return failure(
                StatusCode::InternalError,
                "WorkerServer stop failed");
        }
    }

    void acceptLoop() noexcept {
        try {
            while (!stopping()) {
                reapClosedSession();

                pollfd descriptors[2]{};
                descriptors[0].fd = listener.nativeHandle();
                descriptors[0].events = POLLIN;
                descriptors[1].fd = wake_fd;
                descriptors[1].events = POLLIN;

                int result{0};
                do {
                    result = ::poll(
                        descriptors,
                        2,
                        static_cast<int>(
                            config.supervisor_poll_interval.count()));
                } while (result < 0 && errno == EINTR);

                if (result < 0) {
                    fail(systemFailure(
                        "WorkerServer poll failed",
                        errno));
                    break;
                }
                if ((descriptors[1].revents & POLLIN) != 0) {
                    drainWake();
                    if (stopping()) {
                        break;
                    }
                }
                if ((descriptors[0].revents &
                     (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                    fail(failure(
                        StatusCode::Unavailable,
                        "WorkerServer listener became unavailable"));
                    break;
                }
                if ((descriptors[0].revents & POLLIN) != 0) {
                    auto accepted = listener.accept();
                    if (!accepted.ok()) {
                        if (!stopping()) {
                            fail(std::move(accepted.status));
                        }
                        break;
                    }
                    handleConnection(std::move(accepted.connection));
                }
            }
        } catch (std::exception const& exception) {
            fail(failure(
                StatusCode::InternalError,
                std::string{"WorkerServer accept loop failed: "} +
                    exception.what()));
        } catch (...) {
            fail(failure(
                StatusCode::InternalError,
                "WorkerServer accept loop failed with an unknown exception"));
        }

        stopActiveSession();
    }

    void handleConnection(ipc::UdsConnection connection) {
        {
            std::lock_guard lock(mutex);
            if (stop_requested || active_session) {
                ++rejected_connections;
                connection.shutdown();
                connection.close();
                return;
            }
        }

        std::shared_ptr<ActiveSession> candidate;
        try {
            candidate = std::make_shared<ActiveSession>(
                runtime,
                std::move(connection),
                config.session,
                helloAck(),
                config.bridge);
        } catch (std::exception const&) {
            std::lock_guard lock(mutex);
            ++failed_sessions;
            return;
        }

        auto status = candidate->start();
        if (!status.ok()) {
            static_cast<void>(candidate->stop());
            std::lock_guard lock(mutex);
            ++failed_sessions;
            return;
        }

        bool reject{false};
        {
            std::lock_guard lock(mutex);
            reject = stop_requested || static_cast<bool>(active_session);
            if (!reject) {
                active_session = candidate;
                ++accepted_sessions;
            } else {
                ++rejected_connections;
            }
        }
        if (reject) {
            static_cast<void>(candidate->stop());
        }
    }

    void reapClosedSession() noexcept {
        std::shared_ptr<ActiveSession> active;
        {
            std::lock_guard lock(mutex);
            active = active_session;
        }
        if (active && !active->running()) {
            stopActiveSession();
        }
    }

    void stopActiveSession() noexcept {
        std::shared_ptr<ActiveSession> active;
        {
            std::lock_guard lock(mutex);
            active = std::move(active_session);
        }
        if (!active) {
            return;
        }

        auto status = active->stop();
        std::lock_guard lock(mutex);
        ++completed_sessions;
        if (!status.ok()) {
            ++failed_sessions;
        }
    }

    [[nodiscard]] bool stopping() const noexcept {
        std::lock_guard lock(mutex);
        return stop_requested;
    }

    void fail(Status status) noexcept {
        try {
            if (status.ok()) {
                status = failure(
                    StatusCode::InternalError,
                    "WorkerServer failed without an error status");
            }
            {
                std::lock_guard lock(mutex);
                if (state == State::Failed || state == State::Stopped) {
                    return;
                }
                terminal_status = std::move(status);
                state = State::Failed;
                stop_requested = true;
            }
            signalWake();
            state_condition.notify_all();
        } catch (...) {
        }
    }

    void signalWake() noexcept {
        if (wake_fd < 0) {
            return;
        }
        std::uint64_t value{1};
        while (::write(wake_fd, &value, sizeof(value)) < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
    }

    void drainWake() noexcept {
        if (wake_fd < 0) {
            return;
        }
        std::uint64_t value{0};
        while (::read(wake_fd, &value, sizeof(value)) < 0 &&
               errno == EINTR) {
        }
    }

    void closeWakeFd() noexcept {
        if (wake_fd >= 0) {
            ::close(wake_fd);
            wake_fd = -1;
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

    [[nodiscard]] WorkerServerSnapshot snapshot() const noexcept {
        try {
            WorkerServerSnapshot result;
            std::shared_ptr<ActiveSession> active;
            {
                std::lock_guard lock(mutex);
                result.running =
                    state == State::Running && !stop_requested;
                result.accepting = result.running;
                result.active_session = static_cast<bool>(active_session);
                result.accepted_sessions = accepted_sessions;
                result.completed_sessions = completed_sessions;
                result.rejected_connections = rejected_connections;
                result.failed_sessions = failed_sessions;
                active = active_session;
            }
            if (active) {
                result.active_session_ready = active->ready();
                result.bridge = active->snapshot();
            }
            return result;
        } catch (...) {
            return {};
        }
    }

    GenerationRuntime& runtime;
    WorkerServerConfig config;

    mutable std::mutex lifecycle_mutex;
    mutable std::mutex mutex;
    mutable std::condition_variable state_condition;
    State state{State::Created};
    bool stop_requested{false};
    Status terminal_status;

    ipc::UdsListener listener;
    int wake_fd{-1};
    std::thread accept_thread;
    std::shared_ptr<ActiveSession> active_session;

    std::uint64_t accepted_sessions{0};
    std::uint64_t completed_sessions{0};
    std::uint64_t rejected_connections{0};
    std::uint64_t failed_sessions{0};
};

WorkerServer::WorkerServer(
    GenerationRuntime& runtime,
    WorkerServerConfig config)
    : impl_(std::make_unique<Impl>(runtime, std::move(config))) {}

WorkerServer::~WorkerServer() {
    static_cast<void>(stop());
}

Status WorkerServer::start() {
    return impl_->start();
}

Status WorkerServer::waitUntilStopped(
    std::chrono::milliseconds timeout) const {
    return impl_->waitUntilStopped(timeout);
}

Status WorkerServer::stop() noexcept {
    return impl_->stop();
}

bool WorkerServer::running() const noexcept {
    return impl_->running();
}

Status WorkerServer::terminalStatus() const {
    return impl_->terminalStatus();
}

WorkerServerSnapshot WorkerServer::snapshot() const noexcept {
    return impl_->snapshot();
}

} // namespace kimrt::llm
