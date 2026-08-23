#pragma once

#include "common/status.h"
#include "runtime/generation_runtime.h"
#include "ipc/ipc_protocol.h"
#include "ipc/ipc_transport.h"
#include "worker/session_egress.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace kimrt::llm {

struct RuntimeBridgeConfig {
    std::uint64_t worker_epoch{0};
    ipc::WorkerLimits limits;
    std::uint32_t control_reserve_frames{16};
    std::uint64_t control_reserve_bytes{64U * 1024U};
    std::uint64_t terminal_reserve_bytes{2048};
    std::size_t max_status_message_bytes{1024};
    std::chrono::milliseconds mailbox_wait_timeout{50};
    std::chrono::milliseconds session_send_retry{5};
    std::chrono::milliseconds session_stall_timeout{2000};
};

struct RuntimeBridgeSnapshot {
    bool running{false};
    std::uint64_t owned_requests{0};
    std::uint64_t rejected_requests{0};
    std::uint64_t backpressure_requests{0};
    std::uint64_t cancelled_requests{0};
    ipc::SessionEgressSnapshot egress;
};

class RuntimeBridge final {
public:
    RuntimeBridge(
        GenerationRuntime& runtime,
        ipc::IpcSession& session,
        RuntimeBridgeConfig config);
    ~RuntimeBridge();

    RuntimeBridge(RuntimeBridge const&) = delete;
    RuntimeBridge& operator=(RuntimeBridge const&) = delete;
    RuntimeBridge(RuntimeBridge&&) = delete;
    RuntimeBridge& operator=(RuntimeBridge&&) = delete;

    [[nodiscard]] Status start();
    [[nodiscard]] Status stop() noexcept;

    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] Status terminalStatus() const;
    [[nodiscard]] RuntimeBridgeSnapshot snapshot() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kimrt::llm
