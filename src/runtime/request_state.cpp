#include "runtime/request_state.h"

#include <stdexcept>
#include <utility>

namespace kimrt::llm::detail {

    RequestState::RequestState(
        std::uint64_t externalId,
        std::uint64_t executorId,
        std::size_t promptTokens,
        TimePoint deadline,
        std::shared_ptr<GenerationMailbox>mailbox
    ):externalId_(externalId),
      executorId_(executorId),
      promptTokens_(promptTokens),
      deadline_(deadline),
      mailbox_(std::move(mailbox)){
        if(externalId_ == 0){
            throw std::invalid_argument("RequestState external request id must not be zero");
        }

        if(!mailbox_){
            throw std::invalid_argument("RequestState requires a GenerationMailbox");
        }
    }

    std::uint64_t RequestState::externalId() const noexcept{
        return externalId_;
    }

    std::uint64_t RequestState::executorId() const noexcept{
        return executorId_;
    }

    RequestState::TimePoint RequestState::deadline() const noexcept{
        return deadline_;
    }

    RequestPhase RequestState::phase() const noexcept{
        try{
            std::lock_guard<std::mutex>lock(mtx_);
            return phase_;
        }catch(...){
            //这里锁本身发生异常时采用fail-closed语义:
            //外部不能再把该请求当做可继续生成的活动请求
            return RequestPhase::Terminal;
        }
    }

    bool RequestState::terminalCommitted()const noexcept{
        try{
            std::lock_guard<std::mutex> lock(mtx_);
            return phase_ == RequestPhase::Terminal;
        }catch(...){
            return false;
        }
    }

    bool RequestState::markRunning()noexcept{
        try {
            std::lock_guard<std::mutex>lock(mtx_);

            if(phase_ == RequestPhase::Accepted){
                phase_ = RequestPhase::Running;
                return true;
            }

            return phase_ == RequestPhase::Running;
        }catch(...){
            return false;
        }
    }

    bool RequestState::markCancelling()noexcept{
        try {
            std::lock_guard<std::mutex>lock(mtx_);
            if(phase_ != RequestPhase::Accepted && phase_ != RequestPhase::Running){
                return false;
            }

            phase_ = RequestPhase::Cancelling;
            return true;
        }catch(...){
            return false;
        }
    }

    DeltaCommitResult RequestState::tryCommitDelta(std::vector<TokenId> tokenIds)noexcept{
        try {
            std::lock_guard<std::mutex>lock(mtx_);
            //Terminal是吸收态，进入后不能再交付Token
            if(phase_ == RequestPhase::Terminal){
                return DeltaCommitResult::Terminal;
            }

            // cancel() 已经生效后，Executor 返回的迟到 Token 不再交付。
            if (phase_ == RequestPhase::Cancelling) {
                return DeltaCommitResult::Cancelling;
            }

            if(tokenIds.empty()){
                return DeltaCommitResult::Empty;
            }

            std::size_t const tokenCount  = tokenIds.size();

            TokenDelta delta;
            delta.request_id = externalId_;
            delta.sequence_no = nextSequenceNo_;
            delta.token_ids = std::move(tokenIds);

            //这里等delta真正进入 Mailbox 后， sequence 和 usage 才能更新
            if(!mailbox_->tryPushDelta(std::move(delta))){
                return DeltaCommitResult::Backpressure;
            }

            ++nextSequenceNo_;
            completionTokens_ += tokenCount;

            if(phase_ == RequestPhase::Accepted){
                phase_ = RequestPhase::Running;
            }

            return DeltaCommitResult::Committed;
        }catch(...){
            //这里对于 Backend 来说，无法交付 Delta 与输出背压采取相同的
            // fail-closed 处理: 停止该请求并尝试提交 Backpressure Terminal。
            return DeltaCommitResult::Backpressure;
        }
    }

    TerminalCommitResult RequestState::finalize(
        Status status,
        std::optional<FinishReason>finishreason
    )noexcept {
    try{
        std::lock_guard<std::mutex>lock(mtx_);

        if(phase_ == RequestPhase::Terminal){
            return TerminalCommitResult::AlreadyCommitted;
        }

        TerminalEvent terminal;
        terminal.request_id = externalId_;
        terminal.status = std::move(status);
        terminal.finish_reason = finishreason;
        terminal.usage = GenerationUsage{
            promptTokens_,
            completionTokens_
        };


        if(!mailbox_->pushTerminal(std::move(terminal))){
            return TerminalCommitResult::MailboxRejected;
        }

        phase_ = RequestPhase::Terminal;
        return TerminalCommitResult::Committed;
    }catch(...){
        return TerminalCommitResult::MailboxRejected;
    }
    }

}//namespcae kimrt::llm::detail