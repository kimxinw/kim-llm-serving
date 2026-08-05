 #pragma once

#include "core/status.h"
#include "models/llm/admission_controller.h"
#include "models/llm/generation_backend.h"
#include "models/llm/generation_mailbox.h"
#include "models/llm/generation_types.h"

#include <cstdint>
#include <memory>
#include <shared_mutex>

namespace kimrt::llm {
struct GenerationRuntimeConfig {
    AdmissionConfig admission;
    GenerationMailboxConfig mailbox;
};

struct GenerationSubmission {
    Status status;
    std::shared_ptr<GenerationMailbox> mailbox;

    [[nodiscard]] bool accepted() const noexcept {
        return status.ok() && mailbox != nullptr;
    }

    explicit operator bool() const noexcept {
        return accepted();
    }
};

class GenerationRuntime final {
public:
    GenerationRuntime(
        std::unique_ptr<GenerationBackend> backend,
        GenerationRuntimeConfig config);

    ~GenerationRuntime();

    GenerationRuntime(GenerationRuntime const&) = delete;
    GenerationRuntime& operator=(GenerationRuntime const&) = delete;
    GenerationRuntime(GenerationRuntime&&) = delete;
    GenerationRuntime& operator=(GenerationRuntime&&) = delete;

    Status start();

    [[nodiscard]] GenerationSubmission submit(
        GenerationRequest request);

    void cancel(std::uint64_t requestId);

    Status stop() noexcept;

    [[nodiscard]] bool running() const noexcept;

    [[nodiscard]] AdmissionSnapshot admissionSnapshot() const noexcept;

private:
    enum class State : std::uint8_t {
        Stopped,
        Running,
        Failed,
    };

    std::unique_ptr<GenerationBackend> backend_;
    GenerationMailboxConfig mailboxConfig_;
    AdmissionController admission_;

    mutable std::shared_mutex lifecycleMutex_;
    State state_{State::Stopped};
};

}//namespace kimrt::llm