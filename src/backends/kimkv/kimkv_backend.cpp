#include "backends/kimkv/kimkv_backend.h"

#include "kim-kv/engine/iteration_scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kimrt::llm {

struct KimKvBackend::Impl final : kimkvcache::GenerationTokenSink {
    using RequestId = kimkvcache::RequestId;
    using Clock = std::chrono::steady_clock;

    struct PendingSubmission {
        GenerationRequest request;
        std::shared_ptr<GenerationMailbox> mailbox;
        std::shared_ptr<kimkvcache::GenerationCancellationToken> cancellation;
        std::promise<Status> result;
    };

    struct ActiveRequest {
        std::shared_ptr<GenerationMailbox> mailbox;
        std::shared_ptr<kimkvcache::GenerationCancellationToken> cancellation;
        Clock::time_point deadline;
        bool timed_out{false};
    };

    Impl(KimKvBackendConfig input_config, KimKvBackendResources input_resources)
        : config(std::move(input_config)), resources(std::move(input_resources)) {}

    [[nodiscard]] Status validateConfiguration() const {
        if (!resources.kv || !resources.runner ||
            config.max_input_tokens == 0 || config.max_output_tokens == 0 ||
            config.max_sequence_tokens == 0 || config.max_active_requests == 0 ||
            config.max_batched_tokens == 0 || config.prefill_chunk_size == 0 ||
            config.max_kv_tokens == 0 || config.model_max_batch_tokens == 0) {
            return Status::error(StatusCode::InvalidInput,
                "KimKV backend configuration is incomplete");
        }
        auto const model = resources.runner->generationConfig();
        if (!model.valid() || config.eos_token_id != model.eos_token_id ||
            config.max_sequence_tokens > model.max_position_embeddings ||
            config.max_input_tokens > config.max_sequence_tokens ||
            config.max_output_tokens > config.max_sequence_tokens ||
            config.model_max_batch_tokens > resources.runner->generationMaxBatchSize()) {
            return Status::error(StatusCode::InvalidInput,
                "KimKV model and service limits do not match");
        }
        return Status::success();
    }

    [[nodiscard]] Status validateRequest(
        GenerationRequest const& request,
        std::shared_ptr<GenerationMailbox> const& mailbox) const {
        if (!mailbox || request.context.request_id == 0 ||
            request.context.priority != 0 || request.input_token_ids.empty() ||
            request.input_token_ids.size() > config.max_input_tokens ||
            request.max_new_tokens == 0 ||
            request.max_new_tokens > config.max_output_tokens ||
            request.max_new_tokens >
                config.max_sequence_tokens - request.input_token_ids.size() ||
            request.max_new_tokens > std::numeric_limits<std::uint32_t>::max()) {
            return Status::error(StatusCode::InvalidInput,
                "KimKV request exceeds model limits or contains invalid fields");
        }
        auto const model = resources.runner->generationConfig();
        if (std::any_of(request.input_token_ids.begin(),
                request.input_token_ids.end(),
                [model](TokenId token) {
                    return token < 0 ||
                        static_cast<std::uint32_t>(token) >= model.vocabulary_size;
                }) ||
            (request.end_id && (*request.end_id < 0 ||
                static_cast<std::uint32_t>(*request.end_id) != config.eos_token_id)) ||
            (request.pad_id && (*request.pad_id < 0 ||
                static_cast<std::uint32_t>(*request.pad_id) >= model.vocabulary_size))) {
            return Status::error(StatusCode::InvalidInput,
                "KimKV request has invalid token ids");
        }
        if (request.sampling.top_k != 1 ||
            request.sampling.temperature != 1.0F ||
            request.sampling.top_p != 1.0F ||
            request.sampling.random_seed != 0 ||
            !request.stop_sequences.empty()) {
            return Status::error(StatusCode::InvalidInput,
                "KimKV backend currently supports greedy generation without stop sequences");
        }
        if (request.context.hasDeadline() &&
            request.context.deadline <= Clock::now()) {
            return Status::error(StatusCode::Timeout,
                "KimKV request deadline has expired");
        }
        return Status::success();
    }

    [[nodiscard]] Status start() {
        std::lock_guard lock(mutex);
        if (running) {
            return Status::success();
        }
        if (started_once) {
            return Status::error(StatusCode::NotReady,
                "KimKV backend cannot restart after stop");
        }
        if (auto status = validateConfiguration(); !status) {
            return status;
        }
        try {
            scheduler = std::make_unique<kimkvcache::IterationSchedulerRuntime>(
                *resources.kv, *resources.runner,
                kimkvcache::IterationSchedulerConfig{
                    config.max_active_requests, config.max_batched_tokens,
                    config.max_kv_tokens, config.prefill_chunk_size,
                }, this);
            worker = std::thread([this] { runLoop(); });
        } catch (std::exception const& error) {
            scheduler.reset();
            return Status::error(StatusCode::InternalError,
                std::string{"failed to start KimKV backend: "} + error.what());
        }
        running = true;
        started_once = true;
        return Status::success();
    }

    [[nodiscard]] Status submit(
        GenerationRequest request,
        std::shared_ptr<GenerationMailbox> mailbox) {
        if (auto status = validateRequest(request, mailbox); !status) {
            return status;
        }
        PendingSubmission pending;
        pending.request = std::move(request);
        pending.mailbox = std::move(mailbox);
        auto future = pending.result.get_future();
        auto const request_id = pending.request.context.request_id;
        try {
            pending.cancellation =
                std::make_shared<kimkvcache::GenerationCancellationToken>();
            {
                std::lock_guard lock(mutex);
                if (!running || stopping) {
                    return Status::error(StatusCode::NotReady,
                        "KimKV backend is not accepting requests");
                }
                if (cancellations.count(request_id) != 0) {
                    return Status::error(StatusCode::AlreadyExists,
                        "duplicate KimKV request id");
                }
                if (cancellations.size() >= config.max_active_requests) {
                    return Status::error(StatusCode::ResourceExhausted,
                        "KimKV active request capacity is exhausted");
                }
                cancellations.emplace(request_id, pending.cancellation);
                try {
                    submissions.push_back(std::move(pending));
                } catch (...) {
                    cancellations.erase(request_id);
                    throw;
                }
            }
            wake.notify_one();
            return future.get();
        } catch (std::exception const& error) {
            return Status::error(StatusCode::InternalError,
                std::string{"KimKV submit failed: "} + error.what());
        }
    }

    void cancel(std::uint64_t request_id) noexcept {
        try {
            std::lock_guard lock(mutex);
            auto const found = cancellations.find(request_id);
            if (found != cancellations.end()) {
                found->second->cancel();
            }
        } catch (...) {
        }
    }

    void stop() noexcept {
        try {
            {
                std::lock_guard lock(mutex);
                if (!running && !worker.joinable()) {
                    return;
                }
                stopping = true;
                for (auto const& entry : cancellations) {
                    entry.second->cancel();
                }
            }
            wake.notify_one();
            if (worker.joinable()) {
                worker.join();
            }
            scheduler.reset();
            std::lock_guard lock(mutex);
            running = false;
        } catch (...) {
        }
    }

    bool onToken(RequestId request_id, std::uint32_t token_id,
        std::uint64_t sequence_no) noexcept override {
        auto const found = active.find(request_id);
        if (found == active.end() ||
            token_id > static_cast<std::uint32_t>(
                std::numeric_limits<TokenId>::max())) {
            return false;
        }
        try {
            return found->second.mailbox->tryPushDelta(TokenDelta{
                request_id, sequence_no,
                {static_cast<TokenId>(token_id)},
            });
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] Status admissionStatus(
        kimkvcache::SchedulerAdmissionResult const& result) const {
        using Error = kimkvcache::SchedulerAdmissionError;
        switch (result.error) {
        case Error::InvalidArgument:
            return Status::error(StatusCode::InvalidInput, result.detail);
        case Error::DuplicateRequest:
            return Status::error(StatusCode::AlreadyExists, result.detail);
        case Error::MaxActiveRequests:
        case Error::KvTokenBudget:
            return Status::error(StatusCode::ResourceExhausted, result.detail);
        case Error::RuntimeStopped:
            return Status::error(StatusCode::NotReady, result.detail);
        case Error::InternalError:
        case Error::None:
            return Status::error(StatusCode::InternalError, result.detail);
        }
        return Status::error(StatusCode::InternalError,
            "unknown KimKV scheduler admission result");
    }

    void processSubmission(PendingSubmission pending) {
        auto const request_id = pending.request.context.request_id;
        Status status;
        if (stopping) {
            status = Status::error(StatusCode::NotReady,
                "KimKV backend is stopping");
        } else {
            try {
                kimkvcache::GenerationRequest request;
                request.request_id = request_id;
                request.prompt_token_ids.reserve(
                    pending.request.input_token_ids.size());
                for (TokenId token : pending.request.input_token_ids) {
                    request.prompt_token_ids.push_back(
                        static_cast<std::uint32_t>(token));
                }
                request.sampling.max_new_tokens = static_cast<std::uint32_t>(
                    pending.request.max_new_tokens);
                request.sampling.eos_token_id = config.eos_token_id;
                request.cancellation = pending.cancellation;

                active.emplace(request_id, ActiveRequest{
                    pending.mailbox, pending.cancellation,
                    pending.request.context.deadline,
                });
                auto const result = scheduler->submit(std::move(request));
                if (!result.ok()) {
                    active.erase(request_id);
                    static_cast<void>(scheduler->forgetRequest(request_id));
                    status = admissionStatus(result);
                }
            } catch (std::exception const& error) {
                active.erase(request_id);
                status = Status::error(StatusCode::InternalError,
                    std::string{"KimKV admission failed: "} + error.what());
            }
        }
        if (!status.ok()) {
            std::lock_guard lock(mutex);
            cancellations.erase(request_id);
        }
        pending.result.set_value(std::move(status));
    }

    void expireDeadlines() {
        auto const now = Clock::now();
        for (auto& entry : active) {
            auto& record = entry.second;
            if (!record.timed_out && record.deadline != Clock::time_point::max()
                && record.deadline <= now) {
                record.timed_out = true;
                record.cancellation->cancel();
            }
        }
    }

    [[nodiscard]] TerminalEvent terminalEvent(
        kimkvcache::GenerationTerminal const& terminal,
        bool timed_out) const {
        TerminalEvent event;
        event.request_id = terminal.request_id;
        event.usage = {
            terminal.usage.prompt_tokens, terminal.usage.completion_tokens,
        };
        using Reason = kimkvcache::GenerationTerminalReason;
        using Error = kimkvcache::GenerationError;
        if (terminal.ok()) {
            event.status = Status::success();
            event.finish_reason = terminal.reason == Reason::EosToken
                ? FinishReason::Eos : FinishReason::Length;
        } else if (terminal.error == Error::OutputBackpressure) {
            event.status = Status::error(StatusCode::QueueFull,
                "KimKV output mailbox is full");
            event.finish_reason = FinishReason::Backpressure;
        } else if (terminal.reason == Reason::Cancelled && timed_out) {
            event.status = Status::error(StatusCode::Timeout,
                "KimKV request timed out");
            event.finish_reason = FinishReason::Timeout;
        } else if (terminal.reason == Reason::Cancelled ||
            terminal.reason == Reason::RuntimeStopped) {
            event.status = Status::error(StatusCode::Cancelled,
                "KimKV request was cancelled");
            event.finish_reason = FinishReason::Cancelled;
        } else if (terminal.error == Error::KvBackendFailed ||
            terminal.error == Error::ModelFailed) {
            event.status = Status::error(StatusCode::CudaError, terminal.detail);
        } else {
            event.status = Status::error(StatusCode::InternalError,
                terminal.detail);
        }
        return event;
    }

    void deliverTerminals() {
        for (auto& terminal : scheduler->takeTerminals()) {
            auto const found = active.find(terminal.request_id);
            std::shared_ptr<GenerationMailbox> mailbox;
            TerminalEvent event;
            if (found != active.end()) {
                mailbox = found->second.mailbox;
                event = terminalEvent(terminal, found->second.timed_out);
                active.erase(found);
            }
            static_cast<void>(scheduler->forgetRequest(terminal.request_id));
            {
                std::lock_guard lock(mutex);
                cancellations.erase(terminal.request_id);
            }
            if (mailbox) {
                static_cast<void>(mailbox->pushTerminal(std::move(event)));
            }
        }
    }

    void failOutstanding() noexcept {
        try {
            {
                std::lock_guard lock(mutex);
                running = false;
                stopping = true;
            }
            for (auto& entry : active) {
                static_cast<void>(entry.second.mailbox->pushTerminal(
                    TerminalEvent{entry.first,
                        Status::error(StatusCode::InternalError,
                            "KimKV scheduler stopped unexpectedly"),
                        std::nullopt, {}}));
            }
            active.clear();
            std::lock_guard lock(mutex);
            while (!submissions.empty()) {
                auto pending = std::move(submissions.front());
                submissions.pop_front();
                pending.result.set_value(Status::error(
                    StatusCode::InternalError,
                    "KimKV scheduler stopped unexpectedly"));
            }
            cancellations.clear();
        } catch (...) {
        }
    }

    void runLoop() noexcept {
        try {
            while (true) {
                PendingSubmission pending;
                bool has_pending = false;
                {
                    std::unique_lock lock(mutex);
                    wake.wait(lock, [this] {
                        return stopping || !submissions.empty() ||
                            !scheduler->idle();
                    });
                    if (!submissions.empty()) {
                        pending = std::move(submissions.front());
                        submissions.pop_front();
                        has_pending = true;
                    }
                }
                if (has_pending) {
                    processSubmission(std::move(pending));
                }
                if (stopping) {
                    scheduler->stop();
                } else {
                    expireDeadlines();
                }
                if (!scheduler->idle()) {
                    static_cast<void>(scheduler->runIteration());
                }
                deliverTerminals();
                if (stopping && scheduler->idle()) {
                    std::lock_guard lock(mutex);
                    if (submissions.empty()) {
                        break;
                    }
                }
            }
        } catch (...) {
            failOutstanding();
        }
    }

    KimKvBackendConfig config;
    KimKvBackendResources resources;
    std::unique_ptr<kimkvcache::IterationSchedulerRuntime> scheduler;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<PendingSubmission> submissions;
    std::unordered_map<RequestId,
        std::shared_ptr<kimkvcache::GenerationCancellationToken>> cancellations;
    std::unordered_map<RequestId, ActiveRequest> active;
    bool running{false};
    std::atomic<bool> stopping{false};
    bool started_once{false};
};

KimKvBackend::KimKvBackend(KimKvBackendConfig config,
    KimKvBackendResources resources)
    : impl_(std::make_unique<Impl>(
        std::move(config), std::move(resources))) {}

KimKvBackend::~KimKvBackend() { stop(); }

Status KimKvBackend::start() { return impl_->start(); }

Status KimKvBackend::submit(GenerationRequest request,
    std::shared_ptr<GenerationMailbox> mailbox) {
    return impl_->submit(std::move(request), std::move(mailbox));
}

void KimKvBackend::cancel(std::uint64_t request_id) {
    impl_->cancel(request_id);
}

void KimKvBackend::stop() { impl_->stop(); }

} // namespace kimrt::llm
