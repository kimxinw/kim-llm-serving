#pragma once

#include "models/llm/generation_mailbox.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace kimrt::llm::detail {
    enum class RequestPhase : std::uint8_t {
        Accepted,
        Running,
        Cancelling,
        Terminal,
    };

    enum class DeltaCommitResult : std::uint8_t {
        Committed,
        Empty,
        Backpressure,
        Cancelling,
        Terminal,
    };

    enum class TerminalCommitResult : std::uint8_t {
        Committed,
        AlreadyCommitted,
        MailboxRejected,
    };

    class RequestState final {
    public:
        using Clock = std::chrono::steady_clock;
        using TimePoint = Clock::time_point;

        RequestState(std::uint64_t externalId,
                     std::uint64_t executorId,
                     std::size_t promptTokens,
                    TimePoint deadline,
                    std::shared_ptr<GenerationMailbox>mailbox);
        
        RequestState(const RequestState&) = delete;
        RequestState& operator = (const RequestState&) = delete;
        [[nodiscard]] std::uint64_t externalId() const noexcept;
        [[nodiscard]] std::uint64_t executorId() const noexcept;
        [[nodiscard]] TimePoint deadline() const noexcept;

        [[nodiscard]] RequestPhase phase() const noexcept;
        [[nodiscard]] bool terminalCommitted() const noexcept;
        
        // Accepted -> Running。
        // Cancelling 或 Terminal 状态不会被重新改回 Running。
        bool markRunning() noexcept;

        // Accepted/Running -> Cancelling。
        // 重复取消或者已经 Terminal 时返回 false。
        bool markCancelling() noexcept;
        
        // Token 只有成功进入 Mailbox 后，sequence_no 和 usage 才会递增。
        DeltaCommitResult tryCommitDelta(std::vector<TokenId>tokenIds)noexcept;
        // 所有正常完成、取消、超时、背压和错误路径都必须经过这里。
        // 只有 Terminal 成功进入 Mailbox 的预留槽后，状态才进入 Terminal。
        TerminalCommitResult finalize(
            Status status,
            std::optional<FinishReason> finishReason
        )noexcept;
    private:
        std::uint64_t externalId_{0};
        std::uint64_t executorId_{0};
        std::size_t promptTokens_{0};
        TimePoint deadline_;
        std::shared_ptr<GenerationMailbox>mailbox_;

        mutable std::mutex mtx_;
        RequestPhase phase_{RequestPhase::Accepted};
        std::size_t completionTokens_{0};
        std::uint64_t nextSequenceNo_{0};
    };

}//namesapce kimrt::llm::detail