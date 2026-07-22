#include "models/llm/trtllm_executor_backend.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using kimrt::Status;
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationRequest;
using kimrt::llm::GenerationSink;
using kimrt::llm::TokenChunk;
using kimrt::llm::TokenId;
using kimrt::llm::TrtLlmBackendConfig;
using kimrt::llm::TrtLlmExecutorBackend;

constexpr std::uint64_t kSmokeRequestId{100};
constexpr auto kTerminalTimeout = 180s;

struct SinkSnapshot {
  std::vector<TokenId> tokens;
  std::optional<FinishReason> finish_reason;
  std::optional<Status> error;
  std::size_t terminal_count{0};
  bool token_after_terminal{false};
  bool wrong_request_id{false};
};

class RecordingSink final : public GenerationSink {
public:
  explicit RecordingSink(std::uint64_t expectedRequestId)
      : expectedRequestId_(expectedRequestId) {}

  void onTokens(TokenChunk chunk) override {
    std::lock_guard lock(mutex_);
    wrongRequestId_ = wrongRequestId_ ||
                      chunk.request_id != expectedRequestId_;
    tokenAfterTerminal_ = tokenAfterTerminal_ || terminalCount_ != 0;
    tokens_.insert(tokens_.end(), chunk.token_ids.begin(),
                   chunk.token_ids.end());
  }

  void onComplete(FinishReason reason) override {
    {
      std::lock_guard lock(mutex_);
      if (terminalCount_ == 0) {
        finishReason_ = reason;
      }
      ++terminalCount_;
    }
    cv_.notify_all();
  }

  void onError(Status status) override {
    {
      std::lock_guard lock(mutex_);
      if (terminalCount_ == 0) {
        error_ = std::move(status);
      }
      ++terminalCount_;
    }
    cv_.notify_all();
  }

  bool waitForTerminal(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout,
                        [this] { return terminalCount_ != 0; });
  }

  SinkSnapshot snapshot() const {
    std::lock_guard lock(mutex_);
    return SinkSnapshot{tokens_,          finishReason_,
                        error_,           terminalCount_,
                        tokenAfterTerminal_, wrongRequestId_};
  }

private:
  std::uint64_t expectedRequestId_{0};
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<TokenId> tokens_;
  std::optional<FinishReason> finishReason_;
  std::optional<Status> error_;
  std::size_t terminalCount_{0};
  bool tokenAfterTerminal_{false};
  bool wrongRequestId_{false};
};

GenerationRequest makeRequest(std::uint64_t requestId) {
  GenerationRequest request;
  request.context.request_id = requestId;
  request.input_token_ids = {1, 2, 3, 4};
  request.max_new_tokens = 5;
  request.sampling.top_k = 1;
  request.sampling.random_seed = 0;
  return request;
}

bool expect(bool condition, std::string_view message, int &failures) {
  if (condition) {
    return true;
  }
  ++failures;
  std::cerr << "[FAIL] " << message << '\n';
  return false;
}

TrtLlmBackendConfig makeConfig(std::filesystem::path engineDir) {
  TrtLlmBackendConfig config;
  config.engine_dir = std::move(engineDir);
  config.max_input_tokens = 512;
  config.max_sequence_tokens = 544;
  config.max_output_tokens = 32;
  return config;
}

int run(std::filesystem::path const &engineDir) {
  int failures{0};
  auto const config = makeConfig(engineDir);
  TrtLlmExecutorBackend backend{config};

  auto beforeStartSink = std::make_shared<RecordingSink>(1);
  auto const beforeStart = backend.submit(makeRequest(1), beforeStartSink);
  expect(!beforeStart, "submit before start must be rejected", failures);
  expect(!beforeStartSink->waitForTerminal(100ms),
         "rejected submit must not produce callbacks", failures);

  auto const firstStart = backend.start();
  if (!expect(firstStart.ok(), "first start must succeed", failures)) {
    return failures;
  }
  expect(backend.start().ok(), "start while running must be idempotent",
         failures);

  auto smokeSink = std::make_shared<RecordingSink>(kSmokeRequestId);
  auto const submit = backend.submit(makeRequest(kSmokeRequestId), smokeSink);
  if (expect(submit.ok(), "smoke request must be accepted", failures)) {
    expect(smokeSink->waitForTerminal(
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   kTerminalTimeout)),
           "smoke request must reach terminal before timeout", failures);
  }

  backend.stop();
  backend.stop();

  auto const smoke = smokeSink->snapshot();
  std::vector<TokenId> const expectedTokens{
      1, 2, 3, 4, 3, 29966, 29989, 5205, 29989};
  expect(smoke.tokens == expectedTokens,
         "Backend output must match the saved L1 token baseline", failures);
  expect(smoke.finish_reason == FinishReason::Length,
         "fixed request must finish by length", failures);
  expect(!smoke.error.has_value(), "fixed request must not return an error",
         failures);
  expect(smoke.terminal_count == 1,
         "accepted request must produce exactly one terminal callback",
         failures);
  expect(!smoke.token_after_terminal,
         "no token callback is allowed after terminal", failures);
  expect(!smoke.wrong_request_id,
         "TokenChunk request id must match the external request id", failures);

  auto afterStopSink = std::make_shared<RecordingSink>(2);
  auto const afterStop = backend.submit(makeRequest(2), afterStopSink);
  expect(!afterStop, "submit after stop must be rejected", failures);
  expect(!afterStopSink->waitForTerminal(100ms),
         "submit rejected after stop must not produce callbacks", failures);

  auto const restart = backend.start();
  if (expect(restart.ok(), "start after stop must succeed", failures)) {
    backend.stop();
  }

  for (int cycle = 0; cycle < 2; ++cycle) {
    TrtLlmExecutorBackend candidate{config};
    auto const status = candidate.start();
    if (expect(status.ok(), "repeated Backend creation must start", failures)) {
      candidate.stop();
    }
  }

  if (failures == 0) {
    std::cout << "[PASS] LLM Backend A0 token and lifecycle baseline\n";
  }
  return failures;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <engine_dir>\n";
    return 2;
  }

  std::filesystem::path const engineDir{argv[1]};
  if (!std::filesystem::is_directory(engineDir) ||
      !std::filesystem::is_regular_file(engineDir / "config.json")) {
    std::cerr << "Invalid Engine directory: " << engineDir << '\n';
    return 2;
  }

  try {
    return run(engineDir) == 0 ? 0 : 1;
  } catch (std::exception const &exception) {
    std::cerr << "[FAIL] unexpected exception: " << exception.what() << '\n';
    return 1;
  }
}