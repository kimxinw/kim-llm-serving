#include "models/llm/generation_mailbox.h"

#include <stdexcept>
#include <utility>

namespace kimrt::llm {
    GenerationMailbox::GenerationMailbox(GenerationMailboxConfig config)
        :config_(std::move(config)){
            if(config_.max_queued_deltas == 0||config_.max_queued_tokens==0){
                throw std::invalid_argument(
                    "GenerationMailbox capacity must be greater than zero"
                );
            }
    }

    bool GenerationMailbox::tryPushDelta(TokenDelta delta)noexcept{
        if(delta.token_ids.empty()){
            return false;
        }

        try {
            auto const tokenCount = delta.token_ids.size();

            {
                std::lock_guard lock(gmmtx_);

                if(closed_ || terminalCommitted_){
                    return false;
                }

                if (deltas_.size() >= config_.max_queued_deltas) {
                    return false;
                }

                if (tokenCount > config_.max_queued_tokens - queuedTokens_) {
                    return false;
                }

                deltas_.push_back(std::move(delta));
                queuedTokens_ += tokenCount;
            }

            gmcv_.notify_one();
            return true;
        }catch(...){
            return false;
        }
    }

    bool GenerationMailbox::pushTerminal(TerminalEvent terminal) noexcept {
        try {
        {
            std::lock_guard lock(gmmtx_);

            if (closed_ || terminalCommitted_) {
            return false;
            }

            terminal_.emplace(std::move(terminal));
            terminalCommitted_ = true;
        }

        gmcv_.notify_all();
        return true;
        } catch (...) {
        return false;
        }
    }

    MailboxWaitResult GenerationMailbox::waitPop(
      GenerationEvent &event,
      std::chrono::milliseconds timeout){
        std::unique_lock<std::mutex>lock(gmmtx_);

        auto const ready = [this] {
            return !deltas_.empty() || terminal_.has_value() || closed_;
        };
        if (!ready() && !gmcv_.wait_for(lock, timeout, ready)) {
            return MailboxWaitResult::Timeout;
        }
        // Terminal 已提交时，也必须先交付此前成功入队的 Delta。
        if (!deltas_.empty()) {
            auto const tokenCount = deltas_.front().token_ids.size();

            event = std::move(deltas_.front());
            deltas_.pop_front();
            queuedTokens_ -= tokenCount;

            return MailboxWaitResult::Event;
        }

        if (terminal_) {
            event = std::move(*terminal_);
            terminal_.reset();

            // Terminal 是最后一个事件，交付后 Mailbox 进入关闭状态。
            closed_ = true;
            return MailboxWaitResult::Event;
        }

        return MailboxWaitResult::Closed;
    }

    void GenerationMailbox::close() noexcept {
        try {
        {
            std::lock_guard lock(gmmtx_);

            if (closed_) {
            return;
            }

            closed_ = true;
        }

        gmcv_.notify_all();
        } catch (...) {
        }
    }

}//namespace kimrt::llm