#include "models/llm/trtllm_executor_backend.h"

#include "core/serial_executor.h"

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
    }

    if (auto status = validateConfig(); !status) {
      return status;
    }

    try {
      std::call_once(pluginInitFlag, [] { initTrtLlmPlugins(); });

      auto executor = std::make_unique<tle::Executor>(
          config_.engine_dir, tle::ModelType::kDECODER_ONLY,
          tle::ExecutorConfig{kBeamWidth});
      auto callbacks = std::make_unique<SerialExecutor>();

      std::lock_guard lock(mutex_);
      executor_ = std::move(executor);
      callbacks_ = std::move(callbacks);
      responsePump_ = std::thread(&Impl::pumpResponses, this);
      state_ = State::Running;
      return Status::success();
    } catch (std::exception const &exception) {
      resetAfterFailedStart();
      return failure(StatusCode::TensorRTError,
                     std::string{"failed to start TensorRT-LLM backend: "} +
                         exception.what());
    }
  }

  Status submit(GenerationRequest request,
                std::shared_ptr<GenerationSink> sink) {
    if (auto status = validateRequest(request, sink); !status) {
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
      externalToExecutor_.emplace(externalId, *executorId);
      requests_.emplace(*executorId,
                        PendingRequest{externalId, std::move(sink)});
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
    auto const it = externalToExecutor_.find(externalId);
    if (it == externalToExecutor_.end() || !executor_) {
      return;
    }

    try {
      executor_->cancelRequest(it->second);
    } catch (...) {
      // cancel is deliberately idempotent and non-throwing.
    }
  }

  void stop() noexcept {
    std::lock_guard lifecycleLock(lifecycleMutex_);

    tle::Executor *executor{nullptr};
    std::vector<tle::IdType> activeRequests;
    {
      std::lock_guard lock(mutex_);
      if (state_ == State::Stopped) {
        return;
      }

      state_ = State::Stopping;
      executor = executor_.get();
      activeRequests.reserve(requests_.size());
      for (auto const &[id, request] : requests_) {
        (void)request;
        activeRequests.push_back(id);
      }
    }

    if (executor) {
      for (auto id : activeRequests) {
        try {
          executor->cancelRequest(id);
        } catch (...) {
        }
      }
      try {
        executor->shutdown();
      } catch (...) {
      }
    }

    if (responsePump_.joinable()) {
      responsePump_.join();
    }

    for (auto &request : takeAllRequests()) {
      postCompletion(std::move(request), {}, FinishReason::Cancelled);
    }

    // SerialExecutor drains already-posted callbacks before returning.
    callbacks_.reset();

    std::lock_guard lock(mutex_);
    executor_.reset();
    state_ = State::Stopped;
  }

private:
  enum class State {
    Stopped,
    Running,
    Stopping,
    Failed,
  };

  struct PendingRequest {
    std::uint64_t externalId{0};
    std::shared_ptr<GenerationSink> sink;
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
                         std::shared_ptr<GenerationSink> const &sink) const {
    if (!sink || request.context.request_id == 0) {
      return failure(StatusCode::InvalidInput,
                     "request id and sink are required");
    }
    if (request.streaming) {
      return failure(StatusCode::InvalidInput,
                     "streaming is not implemented yet");
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

    tle::Request result{tle::VecTokens{request.input_token_ids.begin(),
                                       request.input_token_ids.end()},
                        static_cast<tle::SizeType32>(request.max_new_tokens),
                        false, sampling};

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
        for (auto const &response : executor->awaitResponses(kPollInterval)) {
          handleResponse(response);
        }
      }
    } catch (std::exception const &exception) {
      if (markFailed()) {
        failAll(
            failure(StatusCode::TensorRTError,
                    std::string{"response pump failed: "} + exception.what()));
      }
    }
  }

  void handleResponse(tle::Response const &response) {
    if (!response.hasError() && !response.getResult().isFinal) {
      return;
    }

    auto request = takeRequest(response.getRequestId());
    if (!request) {
      return;
    }
    if (response.hasError()) {
      postError(std::move(*request),
                failure(StatusCode::TensorRTError, response.getErrorMsg()));
      return;
    }

    auto const &result = response.getResult();
    if (result.outputTokenIds.empty() || result.finishReasons.empty()) {
      postError(std::move(*request), failure(StatusCode::TensorRTError,
                                             "incomplete final response"));
      return;
    }

    auto reason = toFinishReason(result.finishReasons.front());
    if (!reason) {
      postError(std::move(*request),
                failure(StatusCode::TensorRTError, "invalid finish reason"));
      return;
    }
    postCompletion(std::move(*request), result.outputTokenIds.front(), *reason);
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

  std::optional<PendingRequest> takeRequest(tle::IdType executorId) {
    std::lock_guard lock(mutex_);
    auto it = requests_.find(executorId);
    if (it == requests_.end()) {
      return std::nullopt;
    }

    auto request = std::move(it->second);
    requests_.erase(it);
    externalToExecutor_.erase(request.externalId);
    return request;
  }

  std::vector<PendingRequest> takeAllRequests() {
    std::vector<PendingRequest> result;
    std::lock_guard lock(mutex_);
    result.reserve(requests_.size());
    for (auto &[id, request] : requests_) {
      (void)id;
      result.push_back(std::move(request));
    }
    requests_.clear();
    externalToExecutor_.clear();
    return result;
  }

  void failAll(Status status) {
    for (auto &request : takeAllRequests()) {
      postError(std::move(request), status);
    }
  }

  void postCompletion(PendingRequest request, std::vector<TokenId> tokens,
                      FinishReason reason) {
    auto sink = std::move(request.sink);
    TokenChunk chunk{request.externalId, std::move(tokens)};
    callbacks_->post(
        [sink = std::move(sink), chunk = std::move(chunk), reason]() mutable {
          if (!chunk.token_ids.empty()) {
            try {
              sink->onTokens(std::move(chunk));
            } catch (...) {
            }
          }
          try {
            sink->onComplete(reason);
          } catch (...) {
            // A broken sink must not terminate the callback worker.
          }
        });
  }

  void postError(PendingRequest request, Status status) {
    auto sink = std::move(request.sink);
    callbacks_->post(
        [sink = std::move(sink), status = std::move(status)]() mutable {
          try {
            sink->onError(std::move(status));
          } catch (...) {
          }
        });
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
    callbacks_.reset();
    std::lock_guard lock(mutex_);
    executor_.reset();
    state_ = State::Stopped;
  }

  TrtLlmBackendConfig config_;
  mutable std::mutex mutex_;
  std::mutex lifecycleMutex_;
  State state_{State::Stopped};

  std::unique_ptr<tle::Executor> executor_;
  std::unique_ptr<SerialExecutor> callbacks_;
  std::thread responsePump_;

  std::unordered_map<tle::IdType, PendingRequest> requests_;
  std::unordered_map<std::uint64_t, tle::IdType> externalToExecutor_;
};

TrtLlmExecutorBackend::TrtLlmExecutorBackend(TrtLlmBackendConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

TrtLlmExecutorBackend::~TrtLlmExecutorBackend() { stop(); }

Status TrtLlmExecutorBackend::start() { return impl_->start(); }

Status TrtLlmExecutorBackend::submit(GenerationRequest request,
                                     std::shared_ptr<GenerationSink> sink) {
  return impl_->submit(std::move(request), std::move(sink));
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