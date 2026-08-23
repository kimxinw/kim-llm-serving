#pragma once

#include "common/status.h"
#include "ipc/ipc_protocol.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace kimrt::llm::ipc {

struct UdsConnectResult;
struct UdsListenResult;
struct UdsAcceptResult;

struct UdsReadResult {
    Status status;
    std::size_t bytes_read{0};
    bool peer_closed{false};

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

class UdsConnection final {
public:
    UdsConnection() noexcept = default;
    ~UdsConnection();

    UdsConnection(UdsConnection const&) = delete;
    UdsConnection& operator=(UdsConnection const&) = delete;
    UdsConnection(UdsConnection&& other) noexcept;
    UdsConnection& operator=(UdsConnection&& other) noexcept;

    [[nodiscard]] static UdsConnectResult connect(
        std::string const& socket_path);

    [[nodiscard]] UdsReadResult readSome(
        char* destination,
        std::size_t capacity) noexcept;
    [[nodiscard]] Status writeAll(
        std::vector<std::uint8_t> const& bytes) noexcept;

    void shutdown() noexcept;
    void close() noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] int nativeHandle() const noexcept;

private:
    explicit UdsConnection(int file_descriptor) noexcept;

    int file_descriptor_{-1};

    friend class UdsListener;
};

struct UdsConnectResult {
    Status status;
    UdsConnection connection;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

class UdsListener final {
public:
    UdsListener() noexcept = default;
    ~UdsListener();

    UdsListener(UdsListener const&) = delete;
    UdsListener& operator=(UdsListener const&) = delete;
    UdsListener(UdsListener&& other) noexcept;
    UdsListener& operator=(UdsListener&& other) noexcept;

    [[nodiscard]] static UdsListenResult listen(
        std::string socket_path,
        int backlog = 1);

    [[nodiscard]] UdsAcceptResult accept() noexcept;

    void close() noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] int nativeHandle() const noexcept;
    [[nodiscard]] std::string const& socketPath() const noexcept;

private:
    UdsListener(int file_descriptor, std::string socket_path) noexcept;

    void removeOwnedSocketPath() noexcept;

    int file_descriptor_{-1};
    std::string socket_path_;
};

struct UdsListenResult {
    Status status;
    UdsListener listener;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

struct UdsAcceptResult {
    Status status;
    UdsConnection connection;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

enum class IpcSessionRole : std::uint8_t {
    Client,
    Server,
};

enum class IpcSessionState : std::uint8_t {
    Created,
    Handshaking,
    Ready,
    Stopping,
    Stopped,
    Failed,
};

struct IpcSessionConfig {
    FrameCodecConfig frame_codec;
    std::uint32_t max_egress_frames{1024};
    std::uint64_t max_egress_bytes{4U * 1024U * 1024U};
    std::uint32_t read_buffer_bytes{16U * 1024U};
};

class IpcSession final {
public:
    using MessageHandler = std::function<void(Message)>;

    IpcSession(
        UdsConnection connection,
        IpcSessionConfig config,
        Hello client_hello);
    IpcSession(
        UdsConnection connection,
        IpcSessionConfig config,
        HelloAck server_hello_ack);
    ~IpcSession();

    IpcSession(IpcSession const&) = delete;
    IpcSession& operator=(IpcSession const&) = delete;
    IpcSession(IpcSession&&) = delete;
    IpcSession& operator=(IpcSession&&) = delete;

    [[nodiscard]] Status start(MessageHandler handler = {});
    [[nodiscard]] Status send(Message message);
    [[nodiscard]] Status waitUntilReady(
        std::chrono::milliseconds timeout) const;
    [[nodiscard]] Status waitUntilClosed(
        std::chrono::milliseconds timeout) const;
    [[nodiscard]] Status stop() noexcept;

    [[nodiscard]] IpcSessionRole role() const noexcept;
    [[nodiscard]] IpcSessionState state() const noexcept;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] Status terminalStatus() const;
    [[nodiscard]] std::uint64_t queuedEgressBytes() const noexcept;
    [[nodiscard]] std::uint32_t queuedEgressFrames() const noexcept;

private:
    struct PendingFrame {
        std::vector<std::uint8_t> bytes;
        bool mark_ready_after_write{false};
    };

    [[nodiscard]] Status validateConfiguration() const;
    [[nodiscard]] Status enqueueHandshake(
        Message message,
        bool mark_ready_after_write = false);
    [[nodiscard]] Status enqueueMessage(
        Message message,
        bool require_ready,
        bool mark_ready_after_write = false);
    [[nodiscard]] bool inboundMessageAllowed(
        Message const& message) const noexcept;
    [[nodiscard]] bool outboundMessageAllowed(
        Message const& message) const noexcept;
    [[nodiscard]] Status handleHandshake(Message message);
    [[nodiscard]] Status dispatchReadyMessage(Message message);

    void readerLoop() noexcept;
    void writerLoop() noexcept;
    void fail(Status status) noexcept;
    void markReady();

    UdsConnection connection_;
    IpcSessionConfig config_;
    IpcSessionRole role_;
    Hello client_hello_;
    HelloAck server_hello_ack_;
    MessageHandler message_handler_;

    mutable std::mutex mutex_;
    mutable std::condition_variable state_condition_;
    std::condition_variable writer_condition_;
    std::deque<PendingFrame> egress_;
    std::uint64_t queued_egress_bytes_{0};
    std::thread reader_thread_;
    std::thread writer_thread_;
    IpcSessionState state_{IpcSessionState::Created};
    bool stop_requested_{false};
    Status terminal_status_;
};

} // namespace kimrt::llm::ipc
