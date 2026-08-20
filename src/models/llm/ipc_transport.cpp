#include "models/llm/ipc_transport.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace kimrt::llm::ipc {
namespace {

[[nodiscard]] Status invalid(std::string message) {
    return Status::error(StatusCode::InvalidInput, std::move(message));
}

[[nodiscard]] Status unavailable(std::string message) {
    return Status::error(StatusCode::Unavailable, std::move(message));
}

[[nodiscard]] Status systemError(
    StatusCode code,
    std::string const& operation,
    int error_number) {
    return Status::error(
        code,
        operation + ": " + std::strerror(error_number));
}

[[nodiscard]] Status makeSocketAddress(
    std::string const& socket_path,
    sockaddr_un& address,
    socklen_t& address_length) {
    if (socket_path.empty()) {
        return invalid("UDS socket path must not be empty");
    }
    if (socket_path.find('\0') != std::string::npos) {
        return invalid("UDS socket path must not contain NUL bytes");
    }
    if (socket_path.size() >= sizeof(address.sun_path)) {
        return invalid("UDS socket path is too long");
    }

    std::memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    std::memcpy(
        address.sun_path,
        socket_path.data(),
        socket_path.size());
    address.sun_path[socket_path.size()] = '\0';
    address_length = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + socket_path.size() + 1U);
    return Status::success();
}

[[nodiscard]] Status removeStaleSocket(std::string const& socket_path) {
    struct stat path_status {};
    if (::lstat(socket_path.c_str(), &path_status) == 0) {
        if (!S_ISSOCK(path_status.st_mode)) {
            return Status::error(
                StatusCode::AlreadyExists,
                "UDS path exists and is not a socket");
        }
        if (::unlink(socket_path.c_str()) != 0) {
            return systemError(
                StatusCode::Unavailable,
                "failed to remove stale UDS socket",
                errno);
        }
        return Status::success();
    }
    if (errno == ENOENT) {
        return Status::success();
    }
    return systemError(
        StatusCode::Unavailable,
        "failed to inspect UDS socket path",
        errno);
}

[[nodiscard]] bool isClientInbound(Message const& message) noexcept {
    return std::holds_alternative<Accepted>(message) ||
        std::holds_alternative<Rejected>(message) ||
        std::holds_alternative<TokenDelta>(message) ||
        std::holds_alternative<Terminal>(message) ||
        std::holds_alternative<Stats>(message);
}

[[nodiscard]] bool isServerInbound(Message const& message) noexcept {
    return std::holds_alternative<Submit>(message) ||
        std::holds_alternative<Cancel>(message) ||
        std::holds_alternative<Health>(message);
}

} // namespace

bool UdsReadResult::ok() const noexcept {
    return status.ok();
}

UdsReadResult::operator bool() const noexcept {
    return ok();
}

UdsConnection::UdsConnection(int file_descriptor) noexcept
    : file_descriptor_(file_descriptor) {}

UdsConnection::~UdsConnection() {
    close();
}

UdsConnection::UdsConnection(UdsConnection&& other) noexcept
    : file_descriptor_(std::exchange(other.file_descriptor_, -1)) {}

UdsConnection& UdsConnection::operator=(UdsConnection&& other) noexcept {
    if (this != &other) {
        close();
        file_descriptor_ = std::exchange(other.file_descriptor_, -1);
    }
    return *this;
}

UdsConnectResult UdsConnection::connect(
    std::string const& socket_path) {
    sockaddr_un address {};
    socklen_t address_length{0};
    auto status = makeSocketAddress(
        socket_path, address, address_length);
    if (!status.ok()) {
        return {std::move(status), {}};
    }

    int const file_descriptor =
        ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (file_descriptor < 0) {
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to create UDS client socket",
                errno),
            {}};
    }

    if (::connect(
            file_descriptor,
            reinterpret_cast<sockaddr const*>(&address),
            address_length) != 0) {
        int const error_number = errno;
        ::close(file_descriptor);
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to connect UDS socket",
                error_number),
            {}};
    }

    return {
        Status::success(),
        UdsConnection(file_descriptor)};
}

UdsReadResult UdsConnection::readSome(
    char* destination,
    std::size_t capacity) noexcept {
    if (!valid()) {
        return {
            Status::error(
                StatusCode::NotReady,
                "UDS connection is closed"),
            0,
            false};
    }
    if (destination == nullptr || capacity == 0) {
        return {
            invalid("UDS read buffer must be non-empty"),
            0,
            false};
    }

    while (true) {
        ssize_t const bytes_read =
            ::recv(file_descriptor_, destination, capacity, 0);
        if (bytes_read > 0) {
            return {
                Status::success(),
                static_cast<std::size_t>(bytes_read),
                false};
        }
        if (bytes_read == 0) {
            return {Status::success(), 0, true};
        }
        if (errno == EINTR) {
            continue;
        }
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to read UDS socket",
                errno),
            0,
            false};
    }
}

Status UdsConnection::writeAll(
    std::vector<std::uint8_t> const& bytes) noexcept {
    if (!valid()) {
        return Status::error(
            StatusCode::NotReady,
            "UDS connection is closed");
    }
    if (bytes.empty()) {
        return invalid("UDS write buffer must not be empty");
    }

    std::size_t written{0};
    while (written < bytes.size()) {
        ssize_t const result = ::send(
            file_descriptor_,
            bytes.data() + written,
            bytes.size() - written,
            MSG_NOSIGNAL);
        if (result > 0) {
            written += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        int const error_number = result == 0 ? EPIPE : errno;
        return systemError(
            StatusCode::Unavailable,
            "failed to write UDS socket",
            error_number);
    }
    return Status::success();
}

void UdsConnection::shutdown() noexcept {
    if (valid()) {
        ::shutdown(file_descriptor_, SHUT_RDWR);
    }
}

void UdsConnection::close() noexcept {
    if (valid()) {
        ::close(file_descriptor_);
        file_descriptor_ = -1;
    }
}

bool UdsConnection::valid() const noexcept {
    return file_descriptor_ >= 0;
}

int UdsConnection::nativeHandle() const noexcept {
    return file_descriptor_;
}

bool UdsConnectResult::ok() const noexcept {
    return status.ok() && connection.valid();
}

UdsConnectResult::operator bool() const noexcept {
    return ok();
}

UdsListener::UdsListener(
    int file_descriptor,
    std::string socket_path) noexcept
    : file_descriptor_(file_descriptor),
      socket_path_(std::move(socket_path)) {}

UdsListener::~UdsListener() {
    close();
}

UdsListener::UdsListener(UdsListener&& other) noexcept
    : file_descriptor_(std::exchange(other.file_descriptor_, -1)),
      socket_path_(std::move(other.socket_path_)) {
    other.socket_path_.clear();
}

UdsListener& UdsListener::operator=(UdsListener&& other) noexcept {
    if (this != &other) {
        close();
        file_descriptor_ = std::exchange(other.file_descriptor_, -1);
        socket_path_ = std::move(other.socket_path_);
        other.socket_path_.clear();
    }
    return *this;
}

UdsListenResult UdsListener::listen(
    std::string socket_path,
    int backlog) {
    if (backlog <= 0) {
        return {
            invalid("UDS listener backlog must be positive"),
            {}};
    }

    sockaddr_un address {};
    socklen_t address_length{0};
    auto status = makeSocketAddress(
        socket_path, address, address_length);
    if (!status.ok()) {
        return {std::move(status), {}};
    }
    status = removeStaleSocket(socket_path);
    if (!status.ok()) {
        return {std::move(status), {}};
    }

    int const file_descriptor =
        ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (file_descriptor < 0) {
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to create UDS listener socket",
                errno),
            {}};
    }

    if (::bind(
            file_descriptor,
            reinterpret_cast<sockaddr const*>(&address),
            address_length) != 0) {
        int const error_number = errno;
        ::close(file_descriptor);
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to bind UDS listener socket",
                error_number),
            {}};
    }
    if (::chmod(socket_path.c_str(), S_IRUSR | S_IWUSR) != 0) {
        int const error_number = errno;
        ::close(file_descriptor);
        ::unlink(socket_path.c_str());
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to set UDS socket permissions",
                error_number),
            {}};
    }
    if (::listen(file_descriptor, backlog) != 0) {
        int const error_number = errno;
        ::close(file_descriptor);
        ::unlink(socket_path.c_str());
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to listen on UDS socket",
                error_number),
            {}};
    }

    return {
        Status::success(),
        UdsListener(file_descriptor, std::move(socket_path))};
}

UdsAcceptResult UdsListener::accept() noexcept {
    if (!valid()) {
        return {
            Status::error(
                StatusCode::NotReady,
                "UDS listener is closed"),
            {}};
    }

    while (true) {
        int const accepted =
            ::accept4(file_descriptor_, nullptr, nullptr, SOCK_CLOEXEC);
        if (accepted >= 0) {
            return {
                Status::success(),
                UdsConnection(accepted)};
        }
        if (errno == EINTR) {
            continue;
        }
        return {
            systemError(
                StatusCode::Unavailable,
                "failed to accept UDS connection",
                errno),
            {}};
    }
}

void UdsListener::close() noexcept {
    if (valid()) {
        ::close(file_descriptor_);
        file_descriptor_ = -1;
    }
    removeOwnedSocketPath();
}

bool UdsListener::valid() const noexcept {
    return file_descriptor_ >= 0;
}

int UdsListener::nativeHandle() const noexcept {
    return file_descriptor_;
}

std::string const& UdsListener::socketPath() const noexcept {
    return socket_path_;
}

void UdsListener::removeOwnedSocketPath() noexcept {
    if (!socket_path_.empty()) {
        struct stat path_status {};
        if (::lstat(socket_path_.c_str(), &path_status) == 0 &&
            S_ISSOCK(path_status.st_mode)) {
            ::unlink(socket_path_.c_str());
        }
        socket_path_.clear();
    }
}

bool UdsListenResult::ok() const noexcept {
    return status.ok() && listener.valid();
}

UdsListenResult::operator bool() const noexcept {
    return ok();
}

bool UdsAcceptResult::ok() const noexcept {
    return status.ok() && connection.valid();
}

UdsAcceptResult::operator bool() const noexcept {
    return ok();
}

IpcSession::IpcSession(
    UdsConnection connection,
    IpcSessionConfig config,
    Hello client_hello)
    : connection_(std::move(connection)),
      config_(config),
      role_(IpcSessionRole::Client),
      client_hello_(std::move(client_hello)) {}

IpcSession::IpcSession(
    UdsConnection connection,
    IpcSessionConfig config,
    HelloAck server_hello_ack)
    : connection_(std::move(connection)),
      config_(config),
      role_(IpcSessionRole::Server),
      server_hello_ack_(std::move(server_hello_ack)) {}

IpcSession::~IpcSession() {
    static_cast<void>(stop());
}

Status IpcSession::start(MessageHandler handler) {
    auto status = validateConfiguration();
    if (!status.ok()) {
        return status;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != IpcSessionState::Created) {
            return Status::error(
                StatusCode::AlreadyExists,
                "IPC session has already been started");
        }
        message_handler_ = std::move(handler);
        state_ = IpcSessionState::Handshaking;
        stop_requested_ = false;
        terminal_status_ = Status::success();
    }

    try {
        writer_thread_ = std::thread(&IpcSession::writerLoop, this);
        reader_thread_ = std::thread(&IpcSession::readerLoop, this);
    } catch (std::exception const& exception) {
        status = Status::error(
            StatusCode::InternalError,
            std::string{"failed to start IPC session threads: "} +
                exception.what());
        fail(status);
        static_cast<void>(stop());
        return status;
    }

    if (role_ == IpcSessionRole::Client) {
        status = enqueueHandshake(Message{client_hello_});
        if (!status.ok()) {
            fail(status);
            static_cast<void>(stop());
            return status;
        }
    }
    return Status::success();
}

Status IpcSession::send(Message message) {
    if (!outboundMessageAllowed(message)) {
        return invalid("message type is not allowed for this IPC role");
    }
    return enqueueMessage(std::move(message), true);
}

Status IpcSession::waitUntilReady(
    std::chrono::milliseconds timeout) const {
    if (timeout.count() < 0) {
        return invalid("IPC ready timeout must be non-negative");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    bool const completed = state_condition_.wait_for(
        lock,
        timeout,
        [this] {
            return state_ == IpcSessionState::Ready ||
                state_ == IpcSessionState::Failed ||
                state_ == IpcSessionState::Stopped;
        });
    if (!completed) {
        return Status::error(
            StatusCode::Timeout,
            "timed out waiting for IPC handshake");
    }
    if (state_ == IpcSessionState::Ready) {
        return Status::success();
    }
    if (!terminal_status_.ok()) {
        return terminal_status_;
    }
    return unavailable("IPC session closed before handshake completed");
}

Status IpcSession::waitUntilClosed(
    std::chrono::milliseconds timeout) const {
    if (timeout.count() < 0) {
        return invalid("IPC close timeout must be non-negative");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    bool const completed = state_condition_.wait_for(
        lock,
        timeout,
        [this] {
            return state_ == IpcSessionState::Failed ||
                state_ == IpcSessionState::Stopped;
        });
    if (!completed) {
        return Status::error(
            StatusCode::Timeout,
            "timed out waiting for IPC session close");
    }
    return terminal_status_;
}

Status IpcSession::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == IpcSessionState::Created) {
            state_ = IpcSessionState::Stopped;
            state_condition_.notify_all();
            connection_.close();
            return Status::success();
        }
        if (state_ == IpcSessionState::Stopped) {
            return terminal_status_;
        }
        stop_requested_ = true;
        if (state_ != IpcSessionState::Failed) {
            state_ = IpcSessionState::Stopping;
        }
    }

    connection_.shutdown();
    writer_condition_.notify_all();
    state_condition_.notify_all();

    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    connection_.close();

    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != IpcSessionState::Failed) {
        state_ = IpcSessionState::Stopped;
    }
    egress_.clear();
    queued_egress_bytes_ = 0;
    state_condition_.notify_all();
    return terminal_status_;
}

IpcSessionRole IpcSession::role() const noexcept {
    return role_;
}

IpcSessionState IpcSession::state() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool IpcSession::ready() const noexcept {
    return state() == IpcSessionState::Ready;
}

Status IpcSession::terminalStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return terminal_status_;
}

std::uint64_t IpcSession::queuedEgressBytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return queued_egress_bytes_;
}

std::uint32_t IpcSession::queuedEgressFrames() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<std::uint32_t>(egress_.size());
}

Status IpcSession::validateConfiguration() const {
    if (!connection_.valid()) {
        return Status::error(
            StatusCode::NotReady,
            "IPC session requires an open UDS connection");
    }
    if (config_.frame_codec.max_payload_bytes == 0 ||
        config_.max_egress_frames == 0 ||
        config_.max_egress_bytes == 0 ||
        config_.read_buffer_bytes == 0) {
        return invalid("IPC session capacity values must be positive");
    }
    std::uint64_t const largest_frame =
        static_cast<std::uint64_t>(
            config_.frame_codec.max_payload_bytes) + 4U;
    if (config_.max_egress_bytes < largest_frame) {
        return invalid(
            "IPC egress byte limit must hold one maximum-sized frame");
    }
    if (role_ == IpcSessionRole::Server) {
        auto const& limits = server_hello_ack_.limits;
        if (limits.max_frame_payload_bytes !=
                config_.frame_codec.max_payload_bytes ||
            limits.max_session_egress_frames !=
                config_.max_egress_frames ||
            limits.max_session_egress_bytes !=
                config_.max_egress_bytes) {
            return invalid(
                "server IPC session config must match advertised WorkerLimits");
        }
    }

    Message handshake = role_ == IpcSessionRole::Client
        ? Message{client_hello_}
        : Message{server_hello_ack_};
    auto const encoded = encodePayload(handshake);
    return encoded.status;
}

Status IpcSession::enqueueHandshake(
    Message message,
    bool mark_ready_after_write) {
    return enqueueMessage(
        std::move(message), false, mark_ready_after_write);
}

Status IpcSession::enqueueMessage(
    Message message,
    bool require_ready,
    bool mark_ready_after_write) {
    auto payload = encodePayload(message);
    if (!payload.ok()) {
        return payload.status;
    }
    auto frame = encodeFrame(payload.payload, config_.frame_codec);
    if (!frame.ok()) {
        return frame.status;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_requested_ ||
        state_ == IpcSessionState::Failed ||
        state_ == IpcSessionState::Stopped) {
        return unavailable("IPC session is closed");
    }
    if (require_ready && state_ != IpcSessionState::Ready) {
        return Status::error(
            StatusCode::NotReady,
            "IPC handshake is not complete");
    }
    if (!require_ready && state_ != IpcSessionState::Handshaking) {
        return Status::error(
            StatusCode::NotReady,
            "IPC session is not handshaking");
    }
    if (egress_.size() >= config_.max_egress_frames ||
        frame.bytes.size() >
            config_.max_egress_bytes - queued_egress_bytes_) {
        return Status::error(
            StatusCode::QueueFull,
            "IPC session egress queue is full");
    }

    queued_egress_bytes_ += frame.bytes.size();
    egress_.push_back(PendingFrame{
        std::move(frame.bytes),
        mark_ready_after_write});
    writer_condition_.notify_one();
    return Status::success();
}

bool IpcSession::inboundMessageAllowed(
    Message const& message) const noexcept {
    return role_ == IpcSessionRole::Client
        ? isClientInbound(message)
        : isServerInbound(message);
}

bool IpcSession::outboundMessageAllowed(
    Message const& message) const noexcept {
    return role_ == IpcSessionRole::Client
        ? isServerInbound(message)
        : isClientInbound(message);
}

Status IpcSession::handleHandshake(Message message) {
    if (role_ == IpcSessionRole::Server) {
        auto const* hello = std::get_if<Hello>(&message);
        if (hello == nullptr) {
            return invalid("server expected Hello as the first IPC message");
        }
        if (hello->model_id != server_hello_ack_.manifest.model_id ||
            hello->revision != server_hello_ack_.manifest.revision) {
            return invalid("IPC client model manifest identity mismatch");
        }
        auto status = enqueueHandshake(
            Message{server_hello_ack_}, true);
        if (!status.ok()) {
            return status;
        }
        return Status::success();
    }

    auto const* hello_ack = std::get_if<HelloAck>(&message);
    if (hello_ack == nullptr) {
        return invalid("client expected HelloAck as the first IPC message");
    }
    if (hello_ack->manifest.model_id != client_hello_.model_id ||
        hello_ack->manifest.revision != client_hello_.revision) {
        return invalid("IPC worker model manifest identity mismatch");
    }
    server_hello_ack_ = *hello_ack;
    markReady();
    return Status::success();
}

Status IpcSession::dispatchReadyMessage(Message message) {
    MessageHandler handler;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != IpcSessionState::Ready) {
            return Status::error(
                StatusCode::NotReady,
                "IPC message arrived before handshake completion");
        }
        if (!inboundMessageAllowed(message)) {
            return invalid(
                "message type is not allowed from this IPC peer");
        }
        handler = message_handler_;
    }

    if (!handler) {
        return Status::success();
    }
    try {
        handler(std::move(message));
        return Status::success();
    } catch (std::exception const& exception) {
        return Status::error(
            StatusCode::InternalError,
            std::string{"IPC message handler failed: "} +
                exception.what());
    } catch (...) {
        return Status::error(
            StatusCode::InternalError,
            "IPC message handler failed with an unknown exception");
    }
}

void IpcSession::readerLoop() noexcept {
    try {
        FrameDecoder decoder(config_.frame_codec);
        std::vector<char> buffer(config_.read_buffer_bytes);
        while (true) {
            auto read_result =
                connection_.readSome(buffer.data(), buffer.size());
            if (!read_result.ok()) {
                bool stopping{false};
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping = stop_requested_;
                }
                if (stopping) {
                    return;
                }
                fail(read_result.status);
                return;
            }
            if (read_result.peer_closed) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stop_requested_) {
                        return;
                    }
                }
                if (decoder.hasPartialFrame()) {
                    fail(invalid(
                        "IPC peer disconnected with a partial frame"));
                } else {
                    fail(unavailable("IPC peer disconnected"));
                }
                return;
            }

            auto decoded_frames = decoder.feed(std::string_view(
                buffer.data(), read_result.bytes_read));
            if (!decoded_frames.ok()) {
                fail(decoded_frames.status);
                return;
            }
            bool batch_started_handshaking{false};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                batch_started_handshaking =
                    state_ == IpcSessionState::Handshaking;
            }
            for (std::size_t index = 0;
                 index < decoded_frames.payloads.size();
                 ++index) {
                if (batch_started_handshaking && index > 0) {
                    fail(invalid(
                        "IPC peer pipelined a message before handshake completion"));
                    return;
                }
                auto& payload = decoded_frames.payloads[index];
                auto decoded_message = decodePayload(payload);
                if (!decoded_message.ok()) {
                    fail(decoded_message.status);
                    return;
                }

                IpcSessionState current_state;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    current_state = state_;
                }
                Status status;
                if (current_state == IpcSessionState::Handshaking) {
                    status = handleHandshake(
                        std::move(*decoded_message.message));
                } else {
                    status = dispatchReadyMessage(
                        std::move(*decoded_message.message));
                }
                if (!status.ok()) {
                    fail(std::move(status));
                    return;
                }
            }
        }
    } catch (std::exception const& exception) {
        fail(Status::error(
            StatusCode::InternalError,
            std::string{"IPC reader failed: "} + exception.what()));
    } catch (...) {
        fail(Status::error(
            StatusCode::InternalError,
            "IPC reader failed with an unknown exception"));
    }
}

void IpcSession::writerLoop() noexcept {
    try {
        while (true) {
            PendingFrame frame;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                writer_condition_.wait(lock, [this] {
                    return stop_requested_ || !egress_.empty();
                });
                if (stop_requested_) {
                    return;
                }
                frame = std::move(egress_.front());
                egress_.pop_front();
                queued_egress_bytes_ -= frame.bytes.size();
            }

            auto status = connection_.writeAll(frame.bytes);
            if (!status.ok()) {
                bool stopping{false};
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping = stop_requested_;
                }
                if (stopping) {
                    return;
                }
                fail(std::move(status));
                return;
            }
            if (frame.mark_ready_after_write) {
                markReady();
            }
        }
    } catch (std::exception const& exception) {
        fail(Status::error(
            StatusCode::InternalError,
            std::string{"IPC writer failed: "} + exception.what()));
    } catch (...) {
        fail(Status::error(
            StatusCode::InternalError,
            "IPC writer failed with an unknown exception"));
    }
}

void IpcSession::fail(Status status) noexcept {
    if (status.ok()) {
        status = Status::error(
            StatusCode::InternalError,
            "IPC session failed without an error status");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == IpcSessionState::Failed ||
            state_ == IpcSessionState::Stopped ||
            state_ == IpcSessionState::Stopping) {
            return;
        }
        terminal_status_ = std::move(status);
        state_ = IpcSessionState::Failed;
        stop_requested_ = true;
    }
    connection_.shutdown();
    writer_condition_.notify_all();
    state_condition_.notify_all();
}

void IpcSession::markReady() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == IpcSessionState::Handshaking) {
        state_ = IpcSessionState::Ready;
        state_condition_.notify_all();
    }
}

} // namespace kimrt::llm::ipc
