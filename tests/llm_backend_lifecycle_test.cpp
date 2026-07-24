#include "models/llm/trtllm_executor_backend.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace std::chrono_literals;
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationEvent;
using kimrt::llm::GenerationMailbox;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::GenerationRequest;
using kimrt::llm::MailboxWaitResult;
using kimrt::llm::TerminalEvent;
using kimrt::llm::TokenDelta;
using kimrt::llm::TokenId;
using kimrt::llm::TrtLlmBackendConfig;
using kimrt::llm::TrtLlmExecutorBackend;

constexpr std::uint64_t kSmokeRequestId{100};
constexpr auto kTerminalTimeout = 180s;

struct MailboxSnapshot {
  std::vector<TokenId> tokens;
  std::optional<TerminalEvent> terminal;
  bool timed_out{false};
  bool closed_before_terminal{false};
  bool closed_after_terminal{false};
  bool wrong_request_id{false};
  bool sequence_gap{false};
};

bool expect(bool condition, std::string_view message, int &failures) {
  if (condition) {
    return true;
  }
  ++failures;
  std::cerr << "[FAIL] " << message << '\n';
  return false;
}

std::shared_ptr<GenerationMailbox> makeMailbox() {
  return std::make_shared<GenerationMailbox>(
      GenerationMailboxConfig{32, 32});
}

GenerationRequest makeRequest(std::uint64_t requestId) {
  GenerationRequest request;
  request.context.request_id = requestId;
  request.context.deadline =
      std::chrono::steady_clock::now() + kTerminalTimeout;
  request.input_token_ids = {1, 2, 3, 4};
  request.max_new_tokens = 5;
  request.streaming = true;
  request.sampling.top_k = 1;
  request.sampling.random_seed = 0;
  return request;
}

MailboxSnapshot consumeMailbox(
    std::shared_ptr<GenerationMailbox> const &mailbox,
    std::uint64_t expectedRequestId) {
  MailboxSnapshot snapshot;
  std::uint64_t expectedSequenceNo{0};
  auto const deadline =
      std::chrono::steady_clock::now() + kTerminalTimeout;

  while (!snapshot.terminal) {
    auto const now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      snapshot.timed_out = true;
      break;
    }

    GenerationEvent event;
    auto const result = mailbox->waitPop(
        event,
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));

    if (result == MailboxWaitResult::Timeout) {
      snapshot.timed_out = true;
      break;
    }
    if (result == MailboxWaitResult::Closed) {
      snapshot.closed_before_terminal = true;
      break;
    }

    if (auto *delta = std::get_if<TokenDelta>(&event)) {
      snapshot.wrong_request_id = snapshot.wrong_request_id ||
                                  delta->request_id != expectedRequestId;
      snapshot.sequence_gap = snapshot.sequence_gap ||
                              delta->sequence_no != expectedSequenceNo;
      expectedSequenceNo = delta->sequence_no + 1;
      snapshot.tokens.insert(
          snapshot.tokens.end(),
          delta->token_ids.begin(),
          delta->token_ids.end());
      continue;
    }

    auto *terminal = std::get_if<TerminalEvent>(&event);
    if (!terminal) {
      snapshot.closed_before_terminal = true;
      break;
    }
    snapshot.wrong_request_id = snapshot.wrong_request_id ||
                                terminal->request_id != expectedRequestId;
    snapshot.terminal = std::move(*terminal);
  }

  if (snapshot.terminal) {
    GenerationEvent event;
    snapshot.closed_after_terminal =
        mailbox->waitPop(event, 0ms) == MailboxWaitResult::Closed;
  }
  return snapshot;
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
  TrtLlmExecutorBackend backend{makeConfig(engineDir)};

  auto beforeStartMailbox = makeMailbox();
  auto const beforeStart =
      backend.submit(makeRequest(1), beforeStartMailbox);
  expect(!beforeStart, "submit before start must be rejected", failures);
  GenerationEvent rejectedEvent;
  expect(beforeStartMailbox->waitPop(rejectedEvent, 0ms) ==
             MailboxWaitResult::Timeout,
         "rejected submit must not produce events", failures);

  auto const firstStart = backend.start();
  if (!expect(firstStart.ok(), "first start must succeed", failures)) {
    return failures;
  }
  expect(backend.start().ok(),
         "start while running must be idempotent", failures);

  auto smokeMailbox = makeMailbox();
  auto const submit =
      backend.submit(makeRequest(kSmokeRequestId), smokeMailbox);
  MailboxSnapshot smoke;
  if (expect(submit.ok(), "smoke request must be accepted", failures)) {
    smoke = consumeMailbox(smokeMailbox, kSmokeRequestId);
  }

  backend.stop();
  backend.stop();

  if (submit.ok()) {
    std::vector<TokenId> const expectedTokens{
        3, 29966, 29989, 5205, 29989};
    expect(!smoke.timed_out,
           "smoke request must finish before timeout", failures);
    expect(!smoke.closed_before_terminal,
           "Mailbox must not close before Terminal", failures);
    expect(smoke.closed_after_terminal,
           "Mailbox must close after Terminal", failures);
    expect(!smoke.wrong_request_id,
           "all events must preserve the external request id", failures);
    expect(!smoke.sequence_gap,
           "TokenDelta sequence numbers must be contiguous", failures);
    expect(smoke.tokens == expectedTokens,
           "delta-only output must match the saved generation baseline",
           failures);
    expect(smoke.terminal.has_value(),
           "accepted request must produce a Terminal", failures);

    if (smoke.terminal) {
      expect(smoke.terminal->status.ok(),
             "smoke Terminal must be successful", failures);
      expect(smoke.terminal->finish_reason == FinishReason::Length,
             "fixed request must finish by length", failures);
      expect(smoke.terminal->usage.prompt_tokens == 4,
             "prompt usage must equal the input length", failures);
      expect(smoke.terminal->usage.completion_tokens == 5,
             "completion usage must equal committed output tokens", failures);
    }
  }

  auto afterStopMailbox = makeMailbox();
  auto const afterStop = backend.submit(makeRequest(2), afterStopMailbox);
  expect(!afterStop, "submit after stop must be rejected", failures);
  expect(afterStopMailbox->waitPop(rejectedEvent, 0ms) ==
             MailboxWaitResult::Timeout,
         "submit rejected after stop must not produce events", failures);

  if (failures == 0) {
    std::cout << "[PASS] LLM Mailbox Backend lifecycle baseline\n";
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