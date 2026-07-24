#include "models/llm/trtllm_executor_backend.h"
#include "models/llm/request_state.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <stdexcept>

namespace tle = tensorrt_llm::executor;

namespace kimrt::llm {
namespace {

constexpr tle::SizeType32 kBeamWidth{1};
constexpr auto kPollInterval = std::chrono::milliseconds{50};

std::once_flag pluginInitFlag;

Status failure(StatusCode code, std::string message) {
  return Status::error(code, std::move(message));
}

std::optional<FinishReason> toFinishReason(tle::FinishReason reason) {
  switch (reason) {
  case tle::FinishReason::kEND_ID:
    return FinishReason::Eos;
  case tle::FinishReason::kSTOP_WORDS:
    return FinishReason::Stop;
  case tle::FinishReason::kLENGTH:
    return FinishReason::Length;
  case tle::FinishReason::kTIMED_OUT:
    return FinishReason::Timeout;
  case tle::FinishReason::kCANCELLED:
    return FinishReason::Cancelled;
  case tle::FinishReason::kNOT_FINISHED:
    return std::nullopt;
  }
  return std::nullopt;
}

Status statusForFinishReason(FinishReason reason) {
    switch (reason) {
    case FinishReason::Eos:
    case FinishReason::Length:
    case FinishReason::Stop:
      return Status::success();

    case FinishReason::Cancelled:
      return failure(
          StatusCode::Cancelled,
          "generation request was cancelled");

    case FinishReason::Timeout:
      return failure(
          StatusCode::Timeout,
          "generation request timed out");

    case FinishReason::Backpressure:
      return failure(
          StatusCode::QueueFull,
          "generation output mailbox is full");
    }

    return failure(
        StatusCode::InternalError,
        "unknown generation finish reason");
  }

bool hasNegativeToken(std::vector<TokenId> const &tokens) {
  return std::any_of(tokens.begin(), tokens.end(),
                     [](TokenId token) { return token < 0; });
}

bool hasInvalidStopSequence(StopSequence const & sequence){
  return sequence.empty() || hasNegativeToken(sequence);
}

bool hasInvalidStopSequences(
      std::vector<StopSequence> const &stopSequences) {
    return std::any_of(
        stopSequences.begin(), stopSequences.end(),
        [](StopSequence const &sequence) {
          return hasInvalidStopSequence(sequence);
        });
}

} // namespace

struct TrtLlmExecutorBackend::Impl {
  using RequestStatePtr = std::shared_ptr<detail::RequestState>;
public:
  explicit Impl(TrtLlmBackendConfig backendConfig)
      : config_(std::move(backendConfig)) {}

  Status start() {
    std::lock_guard lifecycleLock(lifecycleMutex_);

    {
      std::lock_guard lock(mutex_);
      if (state_ == State::Running) {
        return Status::success();
      }
      if (state_ != State::Stopped) {
        return failure(StatusCode::InternalError,
                       "backend must be stopped before restarting");
      }
      if (!requests_.empty() || !externalToExecutor_.empty()) {
        return failure(
        StatusCode::InternalError,
        "backend still contains unfinished requests");
      }
    }

    if (auto status = validateConfig(); !status) {
      return status;
    }

    try {
      std::call_once(pluginInitFlag, [] { initTrtLlmPlugins(); });

      auto executor = std::make_unique<tle::Executor>(
          config_.engine_dir, tle::ModelType::kDECODER_ONLY,
          tle::ExecutorConfig{kBeamWidth});
      {
        std::lock_guard lock(mutex_);
      executor_ = std::move(executor);
      responsePump_ = std::thread(&Impl::pumpResponses, this);
      state_ = State::Running;

      }
      return Status::success();
    } catch (std::exception const &exception) {
      resetAfterFailedStart();
      return failure(StatusCode::TensorRTError,
                     std::string{"failed to start TensorRT-LLM backend: "} +
                         exception.what());
    }
  }

  Status submit(GenerationRequest request,
                std::shared_ptr<GenerationMailbox> mailbox) {
    if (auto status = validateRequest(request, mailbox); !status) {
      return status;
    }

    std::optional<tle::Request> executorRequest;
    try {
      executorRequest.emplace(makeExecutorRequest(request));
    } catch (std::exception const &exception) {
      return failure(StatusCode::InvalidInput,
                     std::string{"failed to create TensorRT-LLM request: "} +
                         exception.what());
    }
    std::lock_guard lock(mutex_);

    if (state_ != State::Running || !executor_) {
      return failure(StatusCode::Cancelled, "backend is not running");
    }

    auto const externalId = request.context.request_id;
    if (externalToExecutor_.count(externalId) != 0) {
      return failure(StatusCode::InvalidInput,
                     "duplicate request id: " + std::to_string(externalId));
    }

    std::optional<tle::IdType> executorId;
    try {
      executorId = executor_->enqueueRequest(*executorRequest);

      auto requestState = std::make_shared<detail::RequestState>(
      externalId,
      *executorId,
      request.input_token_ids.size(),
      request.context.deadline,
      std::move(mailbox));

      auto const [requestIt, requestInserted] =
          requests_.emplace(*executorId, std::move(requestState));
      (void)requestIt;

      if (!requestInserted) {
        throw std::runtime_error(
            "TensorRT-LLM returned a duplicate executor request id");
      }

      auto const [externalIt, externalInserted] =
          externalToExecutor_.emplace(externalId, *executorId);
      (void)externalIt;

      if (!externalInserted) {
        throw std::runtime_error(
            "failed to register external request id");
      }
    } catch (std::exception const &exception) {
      externalToExecutor_.erase(externalId);
      if (executorId) {
        requests_.erase(*executorId);
        try {
          executor_->cancelRequest(*executorId);
        } catch (...) {
        }
      }
      return failure(StatusCode::TensorRTError,
                     std::string{"failed to enqueue request: "} +
                         exception.what());
    }

    return Status::success();
  }

  void cancel(std::uint64_t externalId) noexcept {
    std::lock_guard lock(mutex_);

    auto const externalIt = externalToExecutor_.find(externalId);

    if (externalIt == externalToExecutor_.end() || !executor_) {
      return;
    }

    auto const requestIt = requests_.find(externalIt->second);

    if (requestIt == requests_.end()) {
      return;
    }

    // 重复 cancel、已经进入 Cancelling 或 Terminal 都是安全 no-op。
    if (!requestIt->second->markCancelling()) {
      return;
    }
    try {
      executor_->cancelRequest(externalIt->second);
    } catch (...) {
      // cancelRequest 是 best-effort。
      // 如果请求仍然活动，后续 Response 或 stop() 会负责提交终态。
    }
  }

  void stop() noexcept {
    std::lock_guard lifecycleLock(lifecycleMutex_);

    tle::Executor *executor{nullptr};
    std::vector<tle::IdType> activeRequestIds;

    {
      std::lock_guard lock(mutex_);
      if (state_ == State::Stopped) {
        return;
      }

      // 从这一刻开始，submit() 将拒绝新请求。
      state_ = State::Stopping;
      executor = executor_.get();

      activeRequestIds.reserve(requests_.size());

      for (auto const &[executorId, request] : requests_) {
        if (request->markCancelling()) {
          activeRequestIds.push_back(executorId);
        }
      }
    }

    if (executor) {
      for (auto const executorId : activeRequestIds) {
        try {
          executor->cancelRequest(executorId);
        } catch (...) {
        }
      }

      try {
        executor->shutdown();
      } catch (...) {
      }
    }

    // Executor 在 responsePump_ 退出之前始终保持存活。
    if (responsePump_.joinable()) {
      responsePump_.join();
    }

    // shutdown 后仍残留在状态表中的请求统一收敛为 Cancelled。
    // 已经由 Response Pump 提交终态的请求会被 finalizeRequest()
    // 识别为 AlreadyCommitted，不会产生第二个 Terminal。
    for (auto const &request : snapshotRequests()) {
      finalizeRequest(
          request,
          failure(
              StatusCode::Cancelled,
              "backend stopped before request completed"),
          FinishReason::Cancelled);
    }

    {
      std::lock_guard lock(mutex_);

      executor_.reset();

      // 正常情况下两张表都应该为空。
      // 如果 Terminal 没能进入 Mailbox，则保留状态并进入 Failed，
      // 禁止在状态不完整的情况下直接重启。
      if (requests_.empty() && externalToExecutor_.empty()) {
        state_ = State::Stopped;
      } else {
        state_ = State::Failed;
      }
    }
  }

private:
  enum class State {
    Stopped,
    Running,
    Stopping,
    Failed,
  };

  Status validateConfig() const {
    std::error_code error;
    if (!std::filesystem::is_directory(config_.engine_dir, error) ||
        !std::filesystem::is_regular_file(config_.engine_dir / "config.json",
                                          error)) {
      return failure(StatusCode::EngineNotFound,
                     "invalid engine directory: " +
                         config_.engine_dir.string());
    }

    if (config_.max_input_tokens == 0 || config_.max_output_tokens == 0 ||
        config_.max_input_tokens > config_.max_sequence_tokens ||
        config_.max_output_tokens > config_.max_sequence_tokens) {
      return failure(StatusCode::InvalidInput, "invalid backend token limits");
    }
    return Status::success();
  }

  Status validateRequest(GenerationRequest const &request,
                         std::shared_ptr<GenerationMailbox> const &mailbox) const {
    if (!mailbox || request.context.request_id == 0) {
      return failure(StatusCode::InvalidInput,
                     "request id and mailbox are required");
    }
    if (request.context.priority != 0) {
      return failure(StatusCode::InvalidInput,
                     "request priority is not supported yet");
    }
    if (request.input_token_ids.empty() ||
        request.input_token_ids.size() > config_.max_input_tokens ||
        hasNegativeToken(request.input_token_ids)) {
      return failure(StatusCode::InvalidInput, "invalid input token ids");
    }
    if (request.input_token_ids.size() > config_.max_sequence_tokens ||
        request.max_new_tokens == 0 ||
        request.max_new_tokens > config_.max_output_tokens ||
        request.max_new_tokens >
            config_.max_sequence_tokens - request.input_token_ids.size()) {
      return failure(StatusCode::InvalidInput, "invalid max_new_tokens");
    }
    if (request.sampling.top_k <= 0 || !std::isfinite(request.sampling.top_p) ||
        request.sampling.top_p <= 0.0F || request.sampling.top_p > 1.0F ||
        !std::isfinite(request.sampling.temperature) ||
        request.sampling.temperature <= 0.0F) {
      return failure(StatusCode::InvalidInput, "invalid sampling parameters");
    }
    if ((request.end_id && *request.end_id < 0) ||
        (request.pad_id && *request.pad_id < 0)) {
      return failure(StatusCode::InvalidInput, "invalid special token id");
    }

    if (hasInvalidStopSequences(request.stop_sequences)) {
      return failure(StatusCode::InvalidInput, "invalid stop sequence");
    }
    
    if (request.context.hasDeadline() &&
        request.context.deadline <= std::chrono::steady_clock::now()) {
      return failure(StatusCode::Timeout, "request deadline has expired");
    }
    return Status::success();
  }

  tle::Request makeExecutorRequest(GenerationRequest const &request) const {
    tle::SamplingConfig sampling{kBeamWidth};
    sampling.setTopK(request.sampling.top_k);
    sampling.setTopP(request.sampling.top_p);
    sampling.setTemperature(request.sampling.temperature);
    sampling.setSeed(request.sampling.random_seed);
    tle::OutputConfig outputConfig;
    outputConfig.excludeInputFromOutput = true;

    tle::Request result{tle::VecTokens{request.input_token_ids.begin(),
                                       request.input_token_ids.end()},
                        static_cast<tle::SizeType32>(request.max_new_tokens),
                        request.streaming,sampling,outputConfig};

    result.setReturnAllGeneratedTokens(false);
    if (request.end_id) {
      result.setEndId(*request.end_id);
    }
    if (request.pad_id) {
      result.setPadId(*request.pad_id);
    }
    if (!request.stop_sequences.empty()) {
      std::list<tle::VecTokens> stopWords;

      for (auto const &sequence : request.stop_sequences) {
        stopWords.emplace_back(sequence.begin(), sequence.end());
      }
 
      result.setStopWords(stopWords);
    }

    if (request.context.hasDeadline()) {
      auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(
          request.context.deadline - std::chrono::steady_clock::now());
      result.setAllottedTimeMs(std::max(timeout, std::chrono::milliseconds{1}));
    }
    return result;
  }

  void pumpResponses() noexcept {
    tle::Executor *executor{nullptr};

    {
      std::lock_guard lock(mutex_);
      executor = executor_.get();
    }

    try {
      while (isRunning()) {
        auto responses = executor->awaitResponses(kPollInterval);

        for (auto const &response : responses) {
          handleResponse(response);
        }
      }
    } catch (std::exception const &exception) {
      if (markFailed()) {
        finalizeAll(
            failure(
                StatusCode::TensorRTError,
                std::string{"response pump failed: "} +
                    exception.what()),
            std::nullopt);
      }
    } catch (...) {
      if (markFailed()) {
        finalizeAll(
            failure(
                StatusCode::TensorRTError,
                "response pump failed with an unknown exception"),
            std::nullopt);
      }
    }
  }

  void handleResponse(tle::Response const &response) {
    auto request = findRequest(response.getRequestId());

    // 未知 ID 通常表示请求已经因 cancel、stop 或 backpressure
    // 提交过终态，迟到的 Executor Response 应安全忽略。
    if (!request) {
      return;
    }

    if (response.hasError()) {
      finalizeRequest(
          request,
          failure(
              StatusCode::TensorRTError,
              response.getErrorMsg()),
          std::nullopt);
      return;
    }

    auto const &result = response.getResult();

    // 当前 Backend 固定 beam width 为 1，因此每个结果必须只有一组
    // output token。该组 Token 本身允许为空，例如立即 EOS 或取消。
    if (result.outputTokenIds.size() != 1) {
      request->markCancelling();

      finalizeRequest(
          request,
          failure(
              StatusCode::TensorRTError,
              "TensorRT-LLM response has an invalid beam count"),
          std::nullopt);

      cancelExecutorRequest(response.getRequestId());
      return;
    }

    request->markRunning();

    auto const &responseTokens = result.outputTokenIds.front();

    if (!responseTokens.empty()) {
      std::vector<TokenId> deltaTokens{
          responseTokens.begin(),
          responseTokens.end(),
      };

      auto const deltaResult =
          request->tryCommitDelta(std::move(deltaTokens));

      if (deltaResult == detail::DeltaCommitResult::Backpressure) {
        request->markCancelling();

        finalizeRequest(
            request,
            statusForFinishReason(FinishReason::Backpressure),
            FinishReason::Backpressure);

        // Terminal 已经进入独立预留槽后，再取消 Executor 请求。
        cancelExecutorRequest(response.getRequestId());
        return;
      }

      if (deltaResult == detail::DeltaCommitResult::Terminal) {
        // 另一个竞态路径已经完成请求。
        return;
      }

      if (deltaResult == detail::DeltaCommitResult::Cancelling) {
        // cancel() 改变状态后，Executor 可能仍返回迟到 Token。
        // 不交付这些 Token；若这是 final response，继续处理下面的终态。
      }
    }

    // 非 final Response 的 Token 已作为一个增量事件交付，
    // RequestState 保留在映射表中等待后续 Response。
    if (!result.isFinal) {
      return;
    }

    if (result.finishReasons.size() != 1) {
      finalizeRequest(
          request,
          failure(
              StatusCode::TensorRTError,
              "final TensorRT-LLM response has no valid finish reason"),
          std::nullopt);
      return;
    }

    auto const finishReason =
        toFinishReason(result.finishReasons.front());

    if (!finishReason) {
      finalizeRequest(
          request,
          failure(
              StatusCode::TensorRTError,
              "final TensorRT-LLM response is not marked finished"),
          std::nullopt);
      return;
    }

    // final response 中的 Token 已经在上面成功进入 Mailbox，
    // 现在才能提交 Terminal。
    finalizeRequest(
        request,
        statusForFinishReason(*finishReason),
        *finishReason);
  }

  bool isRunning() const {
    std::lock_guard lock(mutex_);
    return state_ == State::Running;
  }

  bool markFailed() {
    std::lock_guard lock(mutex_);

    if (state_ != State::Running) {
      return false;
    }

    state_ = State::Failed;
    return true;
  }

  RequestStatePtr findRequest(tle::IdType executorId) {
    std::lock_guard lock(mutex_);

    auto const it = requests_.find(executorId);

    if (it == requests_.end()) {
      return {};
    }

    return it->second;
  }

  std::vector<RequestStatePtr> snapshotRequests() {
    std::vector<RequestStatePtr> result;

    std::lock_guard lock(mutex_);
    result.reserve(requests_.size());

    for (auto const &[executorId, request] : requests_) {
      (void)executorId;
      result.push_back(request);
    }

    return result;
  }

  void eraseRequest(RequestStatePtr const &request) {
    std::lock_guard lock(mutex_);

    auto const requestIt =
        requests_.find(request->executorId());

    if (requestIt == requests_.end()) {
      return;
    }

    // 防止极端情况下相同 Executor ID 已经对应另一个状态对象。
    if (requestIt->second != request) {
      return;
    }

    externalToExecutor_.erase(request->externalId());
    requests_.erase(requestIt);
  }

  void finalizeRequest(
      RequestStatePtr const &request,
      Status status,
      std::optional<FinishReason> finishReason) {
    auto const result = request->finalize(
        std::move(status),
        finishReason);

    if (result == detail::TerminalCommitResult::MailboxRejected) {
      // Terminal 尚未进入预留槽，不能删除两张 ID 表中的映射。
      return;
    }

    // Committed：本线程刚刚提交了 Terminal。
    // AlreadyCommitted：另一个竞态线程已经成功提交了 Terminal。
    // 两种情况下都允许幂等清理映射。
    eraseRequest(request);
  }

  void finalizeAll(
      Status status,
      std::optional<FinishReason> finishReason) {
    for (auto const &request : snapshotRequests()) {
      // 每个 TerminalEvent 都需要保留独立的 Status 内容，
      // 因此这里复制 status，不能在第一次循环中 move 掉。
      finalizeRequest(
          request,
          status,
          finishReason);
    }
  }

  void cancelExecutorRequest(tle::IdType executorId) noexcept {
    std::lock_guard lock(mutex_);

    if (!executor_) {
      return;
    }

    try {
      executor_->cancelRequest(executorId);
    } catch (...) {
    }
  }

   void resetAfterFailedStart() noexcept {
    {
      std::lock_guard lock(mutex_);
      state_ = State::Stopping;
    }

    try {
      if (executor_) {
        executor_->shutdown();
      }
    } catch (...) {
    }

    if (responsePump_.joinable()) {
      responsePump_.join();
    }

    std::lock_guard lock(mutex_);
    executor_.reset();

    if (requests_.empty() && externalToExecutor_.empty()) {
      state_ = State::Stopped;
    } else {
      state_ = State::Failed;
    }
  }

  TrtLlmBackendConfig config_;
  mutable std::mutex mutex_;
  std::mutex lifecycleMutex_;
  State state_{State::Stopped};

  std::unique_ptr<tle::Executor> executor_;
  std::thread responsePump_;

  std::unordered_map<tle::IdType, RequestStatePtr> requests_;
  std::unordered_map<std::uint64_t, tle::IdType> externalToExecutor_;
};

TrtLlmExecutorBackend::TrtLlmExecutorBackend(TrtLlmBackendConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

TrtLlmExecutorBackend::~TrtLlmExecutorBackend() { stop(); }

Status TrtLlmExecutorBackend::start() { return impl_->start(); }

Status TrtLlmExecutorBackend::submit(GenerationRequest request,
                                     std::shared_ptr<GenerationMailbox> mailbox) {
  return impl_->submit(std::move(request), std::move(mailbox));
}

void TrtLlmExecutorBackend::cancel(std::uint64_t requestId) {
  impl_->cancel(requestId);
}

void TrtLlmExecutorBackend::stop() {
  if (impl_) {
    impl_->stop();
  }
}

} // namespace kimrt::llm