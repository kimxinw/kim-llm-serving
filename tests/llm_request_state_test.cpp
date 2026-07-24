#include "models/llm/request_state.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using kimrt::Status;
using kimrt::StatusCode;
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationEvent;
using kimrt::llm::GenerationMailbox;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::MailboxWaitResult;
using kimrt::llm::TerminalEvent;
using kimrt::llm::TokenDelta;
using kimrt::llm::detail::DeltaCommitResult;
using kimrt::llm::detail::RequestPhase;
using kimrt::llm::detail::RequestState;
using kimrt::llm::detail::TerminalCommitResult;

bool expect(bool condition, std::string_view message, int &failures) {
  if (condition) {
    return true;
  }
  ++failures;
  std::cerr << "[FAIL] " << message << '\n';
  return false;
}

std::shared_ptr<GenerationMailbox> makeMailbox(
    std::size_t maxDeltas = 4,
    std::size_t maxTokens = 8) {
  return std::make_shared<GenerationMailbox>(
      GenerationMailboxConfig{maxDeltas, maxTokens});
}

void testDeltaAndTerminal(int &failures) {
  auto mailbox = makeMailbox();
  RequestState state{
      10,
      20,
      4,
      RequestState::TimePoint::max(),
      mailbox};

  expect(state.phase() == RequestPhase::Accepted,
         "new request must be Accepted", failures);
  expect(state.tryCommitDelta({7, 8}) == DeltaCommitResult::Committed,
         "first Delta must commit", failures);
  expect(state.tryCommitDelta({9}) == DeltaCommitResult::Committed,
         "second Delta must commit", failures);
  expect(state.phase() == RequestPhase::Running,
         "committed Delta must move request to Running", failures);

  expect(state.finalize(Status::success(), FinishReason::Length) ==
             TerminalCommitResult::Committed,
         "first Terminal must commit", failures);
  expect(state.finalize(
             Status::error(StatusCode::Cancelled, "late cancel"),
             FinishReason::Cancelled) ==
             TerminalCommitResult::AlreadyCommitted,
         "second Terminal must be rejected", failures);
  expect(state.tryCommitDelta({10}) == DeltaCommitResult::Terminal,
         "Delta after Terminal must be rejected", failures);

  GenerationEvent event;
  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Event,
         "first Delta must be readable", failures);
  auto const *first = std::get_if<TokenDelta>(&event);
  expect(first && first->sequence_no == 0 &&
             first->token_ids == std::vector<std::int32_t>({7, 8}),
         "first Delta contents must match", failures);

  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Event,
         "second Delta must be readable", failures);
  auto const *second = std::get_if<TokenDelta>(&event);
  expect(second && second->sequence_no == 1 &&
             second->token_ids == std::vector<std::int32_t>({9}),
         "second Delta contents must match", failures);

  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Event,
         "Terminal must be readable", failures);
  auto const *terminal = std::get_if<TerminalEvent>(&event);
  expect(terminal && terminal->status.ok() &&
             terminal->finish_reason == FinishReason::Length &&
             terminal->usage.prompt_tokens == 4 &&
             terminal->usage.completion_tokens == 3,
         "Terminal usage and finish reason must match", failures);

  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Closed,
         "Mailbox must close after Terminal", failures);
}

void testCancellingRejectsDelta(int &failures) {
  auto mailbox = makeMailbox();
  RequestState state{
      11,
      21,
      4,
      RequestState::TimePoint::max(),
      mailbox};

  expect(state.markCancelling(),
         "Accepted request must enter Cancelling", failures);
  expect(state.tryCommitDelta({7}) == DeltaCommitResult::Cancelling,
         "Cancelling request must reject late Delta", failures);
  expect(state.finalize(
             Status::error(StatusCode::Cancelled, "cancelled"),
             FinishReason::Cancelled) == TerminalCommitResult::Committed,
         "cancelled request must commit Terminal", failures);

  GenerationEvent event;
  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Event &&
             std::holds_alternative<TerminalEvent>(event),
         "cancelled request must not expose a TokenDelta", failures);
}

void testBackpressureReservesTerminal(int &failures) {
  auto mailbox = makeMailbox(1, 1);
  RequestState state{
      12,
      22,
      4,
      RequestState::TimePoint::max(),
      mailbox};

  expect(state.tryCommitDelta({7}) == DeltaCommitResult::Committed,
         "Delta within capacity must commit", failures);
  expect(state.tryCommitDelta({8}) == DeltaCommitResult::Backpressure,
         "full Delta queue must report Backpressure", failures);
  expect(state.finalize(
             Status::error(StatusCode::QueueFull, "output full"),
             FinishReason::Backpressure) == TerminalCommitResult::Committed,
         "Terminal must use its reserved slot", failures);

  GenerationEvent event;
  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Event &&
             std::holds_alternative<TokenDelta>(event),
         "queued Delta must be delivered before Terminal", failures);
  expect(mailbox->waitPop(event, std::chrono::milliseconds{0}) ==
             MailboxWaitResult::Event,
         "Backpressure Terminal must be delivered", failures);
  auto const *terminal = std::get_if<TerminalEvent>(&event);
  expect(terminal && terminal->status.code == StatusCode::QueueFull &&
             terminal->finish_reason == FinishReason::Backpressure &&
             terminal->usage.completion_tokens == 1,
         "Backpressure Terminal must preserve committed usage", failures);
}

} // namespace

int main() {
  int failures{0};
  testDeltaAndTerminal(failures);
  testCancellingRejectsDelta(failures);
  testBackpressureReservesTerminal(failures);

  if (failures == 0) {
    std::cout << "[PASS] LLM RequestState contract\n";
  }
  return failures == 0 ? 0 : 1;
}