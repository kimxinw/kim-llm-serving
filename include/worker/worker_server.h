#pragma once

#include "common/status.h"
#include "ipc/ipc_protocol.h"
#include "ipc/ipc_transport.h"
#include "runtime/generation_runtime.h"
#include "worker/runtime_bridge.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace kimrt::llm {

struct WorkerServerConfig {
    std::string socket_path;
    int listen_backlog{8};
    std::chrono::milliseconds supervisor_poll_interval{50};
    ipc::ModelManifest manifest;
    ipc::IpcSessionConfig session;
    RuntimeBridgeConfig bridge;
};

struct WorkerServerSnapshot {
    bool running{false};
    bool accepting{false};
    bool active_session{false};
    bool active_session_ready{false};
    std::uint64_t accepted_sessions{0};
    std::uint64_t completed_sessions{0};
    std::uint64_t rejected_connections{0};
    std::uint64_t failed_sessions{0};
    RuntimeBridgeSnapshot bridge;
};

class WorkerServer final {
public:
    WorkerServer(
        GenerationRuntime& runtime,
        WorkerServerConfig config);
    ~WorkerServer();

    WorkerServer(WorkerServer const&) = delete;
    WorkerServer& operator=(WorkerServer const&) = delete;
    WorkerServer(WorkerServer&&) = delete;
    WorkerServer& operator=(WorkerServer&&) = delete;

    [[nodiscard]] Status start();
    [[nodiscard]] Status waitUntilStopped(
        std::chrono::milliseconds timeout) const;
    [[nodiscard]] Status stop() noexcept;

    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] Status terminalStatus() const;
    [[nodiscard]] WorkerServerSnapshot snapshot() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kimrt::llm
