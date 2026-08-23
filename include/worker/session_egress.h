#pragma once

#include "common/status.h"
#include "ipc/ipc_protocol.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace kimrt::llm::ipc {

struct SessionEgressConfig {
    std::uint32_t max_frame_payload_bytes{kDefaultMaxFramePayloadBytes};
    std::uint32_t max_session_frames{1024};
    std::uint64_t max_session_bytes{4U * 1024U * 1024U};
    std::uint32_t max_request_frames{128};
    std::uint64_t max_request_bytes{2U * 1024U * 1024U};
    std::uint32_t control_reserve_frames{16};
    std::uint64_t control_reserve_bytes{64U * 1024U};
    std::uint64_t terminal_reserve_bytes{2048};
};

enum class SessionEgressEnqueueCode : std::uint8_t {
    Enqueued,
    Stopped,
    DuplicateRequest,
    UnknownRequest,
    InvalidMessage,
    ControlFull,
    RequestFull,
    SessionFull,
};

struct SessionEgressEnqueueResult {
    SessionEgressEnqueueCode code{
        SessionEgressEnqueueCode::InvalidMessage};
    Status status;

    [[nodiscard]] bool enqueued() const noexcept {
        return code == SessionEgressEnqueueCode::Enqueued && status.ok();
    }

    explicit operator bool() const noexcept {
        return enqueued();
    }
};

enum class SessionEgressWaitResult : std::uint8_t {
    Message,
    Timeout,
    Stopped,
};

struct SessionEgressSnapshot {
    bool accepting{false};
    std::uint64_t queued_frames{0};
    std::uint64_t queued_bytes{0};
    std::uint64_t high_watermark_frames{0};
    std::uint64_t high_watermark_bytes{0};
    std::uint64_t registered_requests{0};
    std::uint64_t reserved_terminal_frames{0};
};

class SessionEgress final {
public:
    explicit SessionEgress(SessionEgressConfig config);
    ~SessionEgress();

    SessionEgress(SessionEgress const&) = delete;
    SessionEgress& operator=(SessionEgress const&) = delete;
    SessionEgress(SessionEgress&&) = delete;
    SessionEgress& operator=(SessionEgress&&) = delete;

    [[nodiscard]] Status registerRequest(std::uint64_t request_id);
    [[nodiscard]] bool abandonRequest(std::uint64_t request_id) noexcept;

    [[nodiscard]] SessionEgressEnqueueResult enqueueControl(
        Message message);
    [[nodiscard]] SessionEgressEnqueueResult enqueueDelta(
        std::uint64_t request_id,
        Message message);
    [[nodiscard]] SessionEgressEnqueueResult enqueueTerminal(
        std::uint64_t request_id,
        Message message);

    [[nodiscard]] SessionEgressWaitResult waitPop(
        Message& message,
        std::chrono::milliseconds timeout);

    void stop() noexcept;

    [[nodiscard]] SessionEgressSnapshot snapshot() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kimrt::llm::ipc
