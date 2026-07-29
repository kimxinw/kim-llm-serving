#pragma once 

#include "models/llm/admission_controller.h"
#include "models/llm/generation_types.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstddef>
#include <deque>
#include <optional>

namespace kimrt::llm {
enum class MailboxWaitResult {
    Event,
    Timeout,
    Closed,
};

struct GenerationMailboxConfig {
    std::size_t max_queued_deltas{16};
    std::size_t max_queued_tokens{256};
};

class GenerationMailbox final {
public:
    explicit GenerationMailbox(GenerationMailboxConfig config);

    GenerationMailbox(
        GenerationMailboxConfig config,
        AdmissionLease admissionLease
    );

    GenerationMailbox(const GenerationMailbox&) = delete;
    GenerationMailbox& operator = (const GenerationMailbox&)=delete;

    bool tryPushDelta(TokenDelta delta)noexcept;
    bool pushTerminal(TerminalEvent terminal)noexcept;

    MailboxWaitResult waitPop(
        GenerationEvent&event,
        std::chrono::milliseconds timeout
    );

    void close()noexcept;
private:
    GenerationMailboxConfig config_;
    AdmissionLease admissionLease_;

    std::mutex gmmtx_;
    std::condition_variable gmcv_;
    std::deque<TokenDelta> deltas_;
    std::optional<TerminalEvent>terminal_;

    std::size_t queuedTokens_{0};
    bool terminalCommitted_{false};
    bool closed_{false};
};

}//namespace kimrt::llm