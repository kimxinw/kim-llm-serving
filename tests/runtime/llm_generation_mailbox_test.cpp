 #include "runtime/generation_mailbox.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <iostream>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace {

using namespace std::chrono_literals;

using kimrt::Status;
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationEvent;
using kimrt::llm::GenerationMailbox;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::GenerationUsage;
using kimrt::llm::MailboxWaitResult;
using kimrt::llm::TerminalEvent;
using kimrt::llm::TokenDelta;
using kimrt::llm::TokenId;
using kimrt::llm::AdmissionController;

bool expect(bool condition, std::string_view message, int &failures) {
if (condition) {
    return true;
}

++failures;
std::cerr << "[FAIL] " << message << '\n';
return false;
}

TokenDelta makeDelta(
    std::uint64_t requestId,
    std::uint64_t sequenceNo,
    std::initializer_list<TokenId> tokens) {
TokenDelta delta;
delta.request_id = requestId;
delta.sequence_no = sequenceNo;
delta.token_ids = std::vector<TokenId>(tokens);
return delta;
}

TerminalEvent makeTerminal(
    std::uint64_t requestId,
    FinishReason reason = FinishReason::Length) {
TerminalEvent terminal;
terminal.request_id = requestId;
terminal.status = Status::success();
terminal.finish_reason = reason;
terminal.usage = GenerationUsage{3, 1};
return terminal;
}

void testCapacity(int &failures) {
GenerationMailbox deltaLimit{
    GenerationMailboxConfig{1, 8},
};

expect(deltaLimit.tryPushDelta(makeDelta(1, 0, {10})),
        "first Delta must fit into the queue", failures);

expect(!deltaLimit.tryPushDelta(makeDelta(1, 1, {11})),
        "Delta count limit must reject another Delta", failures);

GenerationMailbox tokenLimit{
    GenerationMailboxConfig{4, 3},
};

expect(tokenLimit.tryPushDelta(makeDelta(2, 0, {20, 21})),
        "first Delta must fit into the token budget", failures);

expect(!tokenLimit.tryPushDelta(makeDelta(2, 1, {22, 23})),
        "token budget must reject an oversized addition", failures);

GenerationEvent event;
expect(tokenLimit.waitPop(event, 0ms) == MailboxWaitResult::Event,
        "queued Delta must be available immediately", failures);

expect(tokenLimit.tryPushDelta(makeDelta(2, 1, {22, 23})),
        "popping a Delta must release its token budget", failures);
}

void testTerminalReservationAndOrdering(int &failures) {
GenerationMailbox mailbox{
    GenerationMailboxConfig{1, 1},
};

expect(mailbox.tryPushDelta(makeDelta(7, 0, {42})),
        "Delta must fill the normal queue", failures);

expect(mailbox.pushTerminal(makeTerminal(7)),
        "Terminal must use its reserved slot when Delta queue is full",
        failures);

expect(!mailbox.pushTerminal(makeTerminal(7)),
        "a second Terminal must be rejected", failures);

expect(!mailbox.tryPushDelta(makeDelta(7, 1, {43})),
        "Delta after Terminal commit must be rejected", failures);

GenerationEvent event;

expect(mailbox.waitPop(event, 0ms) == MailboxWaitResult::Event,
        "first waitPop must return the queued Delta", failures);

auto const *delta = std::get_if<TokenDelta>(&event);
expect(delta != nullptr,
        "Delta must be delivered before Terminal", failures);

if (delta != nullptr) {
    expect(delta->sequence_no == 0,
            "delivered Delta must preserve sequence number", failures);
    expect(delta->token_ids == std::vector<TokenId>({42}),
            "delivered Delta must preserve tokens", failures);
}

expect(mailbox.waitPop(event, 0ms) == MailboxWaitResult::Event,
        "second waitPop must return Terminal", failures);

auto const *terminal = std::get_if<TerminalEvent>(&event);
expect(terminal != nullptr,
        "Terminal must be delivered after queued Deltas", failures);

if (terminal != nullptr) {
    expect(terminal->request_id == 7,
            "Terminal must preserve request id", failures);
    expect(terminal->finish_reason == FinishReason::Length,
            "Terminal must preserve finish reason", failures);
}

expect(mailbox.waitPop(event, 0ms) == MailboxWaitResult::Closed,
        "Mailbox must be closed after Terminal delivery", failures);
}

void testTimeout(int &failures) {
GenerationMailbox mailbox{
    GenerationMailboxConfig{1, 1},
};

GenerationEvent event;

expect(mailbox.waitPop(event, 20ms) == MailboxWaitResult::Timeout,
        "empty Mailbox must time out", failures);
}

void testCloseWakesWaiter(int &failures) {
GenerationMailbox mailbox{
    GenerationMailboxConfig{1, 1},
};

auto waiter = std::async(std::launch::async, [&mailbox] {
    GenerationEvent event;
    return mailbox.waitPop(event, 5s);
});

std::this_thread::sleep_for(20ms);
mailbox.close();

expect(waiter.wait_for(1s) == std::future_status::ready,
        "close must wake a blocked consumer", failures);

if (waiter.wait_for(0ms) == std::future_status::ready) {
    expect(waiter.get() == MailboxWaitResult::Closed,
            "woken consumer must observe Closed", failures);
}

expect(!mailbox.tryPushDelta(makeDelta(9, 0, {90})),
        "closed Mailbox must reject Delta", failures);

expect(!mailbox.pushTerminal(makeTerminal(9)),
        "closed Mailbox must reject Terminal", failures);
}

void testTerminalReleasesAdmissionLease(int &failures) {
AdmissionController controller({1, 8, 8});

expect(
        controller.open(),
        "Admission controller must open",
        failures);

auto decision = controller.tryAcquire({100, 3, 4});

expect(
        decision.admitted(),
        "request must acquire Admission lease",
        failures);

GenerationMailbox mailbox(
        GenerationMailboxConfig{4, 8},
        std::move(decision.lease));

auto before_terminal = controller.snapshot();

expect(
        before_terminal.active_requests == 1 &&
        before_terminal.reserved_input_tokens == 3 &&
        before_terminal.reserved_output_tokens == 4,
        "Mailbox must retain Admission resources before Terminal",
        failures);

expect(
        mailbox.pushTerminal(makeTerminal(100)),
        "Terminal must commit successfully",
        failures);

/*
* 不调用 waitPop()。
* Admission 必须在 Terminal 提交时释放，而不是等消费者读取。
*/
auto after_terminal = controller.snapshot();

expect(
        after_terminal.active_requests == 0 &&
        after_terminal.reserved_input_tokens == 0 &&
        after_terminal.reserved_output_tokens == 0,
        "Terminal commit must release Admission resources",
        failures);

expect(
        !mailbox.pushTerminal(makeTerminal(100)),
        "duplicate Terminal must be rejected",
        failures);

auto after_duplicate = controller.snapshot();

expect(
        after_duplicate.active_requests == 0 &&
        after_duplicate.reserved_input_tokens == 0 &&
        after_duplicate.reserved_output_tokens == 0,
        "duplicate Terminal must not release resources twice",
        failures);
}

void testMailboxDestructionReleasesAdmissionLease(int &failures) {
        AdmissionController controller({1, 8, 8});

        expect(
                controller.open(),
                "Admission controller for destruction test must open",
                failures);

        {
                auto decision = controller.tryAcquire({101, 2, 3});

                expect(
                decision.admitted(),
                "destruction-test request must acquire Admission lease",
                failures);

                GenerationMailbox mailbox(
                GenerationMailboxConfig{4, 8},
                std::move(decision.lease));

                auto active = controller.snapshot();

                expect(
                active.active_requests == 1 &&
                        active.reserved_input_tokens == 2 &&
                        active.reserved_output_tokens == 3,
                "Mailbox must own Admission resources",
                failures);

                /*
                * 模拟 Backend submit 失败：
                * Mailbox 没有被交给 Backend，也没有产生 Terminal。
                */
        }

        auto released = controller.snapshot();

        expect(
                released.active_requests == 0 &&
                released.reserved_input_tokens == 0 &&
                released.reserved_output_tokens == 0,
                "Mailbox destruction must release Admission resources",
                failures);
}

int run() {
int failures{0};

testCapacity(failures);
testTerminalReservationAndOrdering(failures);
testTerminalReleasesAdmissionLease(failures);
testMailboxDestructionReleasesAdmissionLease(failures);
testTimeout(failures);
testCloseWakesWaiter(failures);

if (failures == 0) {
    std::cout << "[PASS] LLM generation mailbox contract\n";
}

return failures;
}

} // namespace

int main() {
return run() == 0 ? 0 : 1;
}