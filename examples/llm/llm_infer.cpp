#include "models/llm/trtllm_executor_backend.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iostream>
#include <iterator>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMaxInputTokens{512};
constexpr std::size_t kMaxSequenceTokens{544};
constexpr std::size_t kMaxOutputTokens{32};
constexpr std::uint64_t kRequestId{1};
constexpr auto kRequestTimeout = std::chrono::seconds{180};

struct Options {
  std::filesystem::path engine;
  std::size_t maxNewTokens{0};
  std::vector<kimrt::llm::TokenId> inputTokens;
};

struct Outcome {
  std::vector<kimrt::llm::TokenId> tokens;
  std::optional<kimrt::llm::FinishReason> finishReason;
  std::optional<kimrt::Status> error;
};

class FutureSink final : public kimrt::llm::GenerationSink {
public:
  std::future<Outcome> getFuture() { return result_.get_future(); }

  void onTokens(kimrt::llm::TokenChunk chunk) override {
    tokens_.insert(tokens_.end(),
                   std::make_move_iterator(chunk.token_ids.begin()),
                   std::make_move_iterator(chunk.token_ids.end()));
  }

  void onComplete(kimrt::llm::FinishReason reason) override {
    result_.set_value(Outcome{std::move(tokens_), reason, std::nullopt});
  }

  void onError(kimrt::Status status) override {
    result_.set_value(Outcome{{}, std::nullopt, std::move(status)});
  }

private:
  std::vector<kimrt::llm::TokenId> tokens_;
  std::promise<Outcome> result_;
};

bool parseInt32(std::string_view text, std::int32_t &value) {
  auto const [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return !text.empty() && error == std::errc{} &&
         end == text.data() + text.size();
}

std::optional<Options> parseOptions(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_dir> <max_new_tokens> <token_id>...\n";
    return std::nullopt;
  }

  std::int32_t maxNewTokens{0};
  if (!parseInt32(argv[2], maxNewTokens) || maxNewTokens <= 0 ||
      maxNewTokens > static_cast<std::int32_t>(kMaxOutputTokens)) {
    std::cerr << "max_new_tokens must be in [1, " << kMaxOutputTokens << "]\n";
    return std::nullopt;
  }

  Options options;
  options.engine = argv[1];
  options.maxNewTokens = static_cast<std::size_t>(maxNewTokens);
  options.inputTokens.reserve(static_cast<std::size_t>(argc - 3));

  for (int index = 3; index < argc; ++index) {
    kimrt::llm::TokenId token{0};
    if (!parseInt32(argv[index], token) || token < 0) {
      std::cerr << "invalid token id: " << argv[index] << '\n';
      return std::nullopt;
    }
    options.inputTokens.push_back(token);
  }

  if (options.inputTokens.size() > kMaxInputTokens ||
      options.inputTokens.size() + options.maxNewTokens > kMaxSequenceTokens) {
    std::cerr << "input token count exceeds Engine limits\n";
    return std::nullopt;
  }
  return options;
}

std::string_view finishReasonName(kimrt::llm::FinishReason reason) {
  using kimrt::llm::FinishReason;
  switch (reason) {
  case FinishReason::Eos:
    return "end_id";
  case FinishReason::Length:
    return "length";
  case FinishReason::Stop:
    return "stop_words";
  case FinishReason::Cancelled:
    return "cancelled";
  case FinishReason::Timeout:
    return "timed_out";
  }
  return "unknown";
}

void printTokens(std::string_view label,
                 std::vector<kimrt::llm::TokenId> const &tokens) {
  std::cout << label << "=[";
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    std::cout << (index == 0 ? "" : ", ") << tokens[index];
  }
  std::cout << "]\n";
}

} // namespace

int main(int argc, char *argv[]) {
  auto options = parseOptions(argc, argv);
  if (!options) {
    return 2;
  }

  kimrt::llm::TrtLlmBackendConfig config;
  config.engine_dir = options->engine;
  config.max_input_tokens = kMaxInputTokens;
  config.max_sequence_tokens = kMaxSequenceTokens;
  config.max_output_tokens = kMaxOutputTokens;

  kimrt::llm::TrtLlmExecutorBackend backend{std::move(config)};

  auto const loadBegin = std::chrono::steady_clock::now();
  auto const startStatus = backend.start();
  auto const loadEnd = std::chrono::steady_clock::now();
  if (!startStatus) {
    std::cerr << "failed to start LLM backend: " << startStatus.message << '\n';
    return 3;
  }

  auto sink = std::make_shared<FutureSink>();
  auto result = sink->getFuture();

  kimrt::llm::GenerationRequest request;
  request.context.request_id = kRequestId;
  request.model_name = "tinyllama";
  request.input_token_ids = options->inputTokens;
  request.max_new_tokens = options->maxNewTokens;
  request.sampling.top_k = 1;
  request.sampling.random_seed = 0;

  auto const requestBegin = std::chrono::steady_clock::now();
  auto const submitStatus = backend.submit(std::move(request), sink);
  if (!submitStatus) {
    std::cerr << "failed to submit request: " << submitStatus.message << '\n';
    backend.stop();
    return 3;
  }

  if (result.wait_for(kRequestTimeout) != std::future_status::ready) {
    backend.cancel(kRequestId);
    backend.stop();
    std::cerr << "LLM request timed out\n";
    return 3;
  }

  auto outcome = result.get();
  auto const requestEnd = std::chrono::steady_clock::now();
  backend.stop();

  if (outcome.error) {
    std::cerr << "LLM inference failed: " << outcome.error->message << '\n';
    return 3;
  }
  if (!outcome.finishReason) {
    std::cerr << "LLM inference returned no finish reason\n";
    return 3;
  }

  std::cout << "request_id=" << kRequestId << '\n';
  printTokens("input_tokens", options->inputTokens);
  printTokens("output_tokens", outcome.tokens);
  std::cout << "finish_reason=" << finishReasonName(*outcome.finishReason)
            << '\n';
  std::cout
      << "engine_load_ms="
      << std::chrono::duration<double, std::milli>(loadEnd - loadBegin).count()
      << '\n';
  std::cout << "request_latency_ms="
            << std::chrono::duration<double, std::milli>(requestEnd -
                                                         requestBegin)
                   .count()
            << '\n';
  return 0;
}
