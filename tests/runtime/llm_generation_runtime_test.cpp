#include "runtime/generation_runtime.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace std::chrono_literals;

using kimrt::Status;
using kimrt::StatusCode;
using kimrt::llm::AdmissionConfig;
using kimrt::llm::AdmissionSnapshot;
using kimrt::llm::FinishReason;
using kimrt::llm::GenerationBackend;
using kimrt::llm::GenerationEvent;
using kimrt::llm::GenerationMailbox;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::GenerationRequest;
using kimrt::llm::GenerationRuntime;
using kimrt::llm::GenerationRuntimeConfig;
using kimrt::llm::GenerationSubmission;
using kimrt::llm::MailboxWaitResult;
using kimrt::llm::TerminalEvent;

bool expect(bool condition, std::string_view message, int& failures) {
    if (condition) {
        return true;
    }

    ++failures;
    std::cerr << "[FAIL] " << message << '\n';
    return false;
}

Status failure(StatusCode code, std::string message) {
    return Status::error(code, std::move(message));
}

GenerationRuntimeConfig makeRuntimeConfig(
    std::size_t maxActiveRequests = 2,
    std::size_t maxInputTokens = 16,
    std::size_t maxOutputTokens = 16) {

    GenerationRuntimeConfig config;
    config.admission = AdmissionConfig{
        maxActiveRequests,
        maxInputTokens,
        maxOutputTokens,
    };
    config.mailbox = GenerationMailboxConfig{4, 16};
    return config;
}

GenerationRequest makeRequest(
    std::uint64_t requestId,
    std::size_t inputTokens = 2,
    std::size_t maxNewTokens = 3) {

    GenerationRequest request;
    request.context.request_id = requestId;
    request.context.deadline =
        std::chrono::steady_clock::now() + 30s;
    request.input_token_ids.assign(inputTokens, 1);
    request.max_new_tokens = maxNewTokens;
    request.streaming = true;
    request.sampling.top_k = 1;
    request.sampling.top_p = 1.0F;
    request.sampling.temperature = 1.0F;
    return request;
}

bool resourcesAreZero(AdmissionSnapshot const& snapshot) {
    return snapshot.active_requests == 0 &&
           snapshot.reserved_input_tokens == 0 &&
           snapshot.reserved_output_tokens == 0;
}

std::optional<TerminalEvent> popTerminal(
    std::shared_ptr<GenerationMailbox> const& mailbox) {

    if (!mailbox) {
        return std::nullopt;
    }

    for (int eventIndex = 0; eventIndex < 8; ++eventIndex) {
        GenerationEvent event;
        auto const result = mailbox->waitPop(event, 0ms);

        if (result != MailboxWaitResult::Event) {
            return std::nullopt;
        }

        if (auto* terminal = std::get_if<TerminalEvent>(&event)) {
            return std::move(*terminal);
        }
    }

    return std::nullopt;
}

class FakeGenerationBackend final : public GenerationBackend {
public:
    Status start() override {
        std::lock_guard lock(mutex_);
        ++startCalls_;

        if (throwOnStart_) {
            throw std::runtime_error("configured start exception");
        }

        if (!startStatus_) {
            return startStatus_;
        }

        running_ = true;
        return Status::success();
    }

    Status submit(
        GenerationRequest request,
        std::shared_ptr<GenerationMailbox> mailbox) override {

        std::unique_lock lock(mutex_);
        ++submitCalls_;

        if (throwOnSubmit_) {
            throw std::runtime_error("configured submit exception");
        }

        if (!running_) {
            return failure(
                StatusCode::Cancelled,
                "Fake Backend is not running");
        }

        if (!submitStatus_) {
            return submitStatus_;
        }

        if (blockNextSubmit_) {
            blockedSubmitEntered_ = true;
            condition_.notify_all();
            condition_.wait(lock, [this] {
                return releaseBlockedSubmit_;
            });

            blockNextSubmit_ = false;
            releaseBlockedSubmit_ = false;
        }

        auto const requestId = request.context.request_id;
        auto const [iterator, inserted] =
            activeMailboxes_.emplace(requestId, std::move(mailbox));
        (void)iterator;

        if (!inserted) {
            return failure(
                StatusCode::AlreadyExists,
                "Fake Backend received a duplicate request id");
        }

        return Status::success();
    }

    void cancel(std::uint64_t requestId) override {
        std::shared_ptr<GenerationMailbox> mailbox;

        {
            std::lock_guard lock(mutex_);
            cancelledIds_.push_back(requestId);

            auto const iterator = activeMailboxes_.find(requestId);
            if (iterator != activeMailboxes_.end()) {
                mailbox = iterator->second;
            }
        }

        if (!mailbox) {
            return;
        }

        TerminalEvent terminal;
        terminal.request_id = requestId;
        terminal.status = failure(
            StatusCode::Cancelled,
            "request cancelled by Fake Backend");
        terminal.finish_reason = FinishReason::Cancelled;

        if (mailbox->pushTerminal(std::move(terminal))) {
            eraseIfSame(requestId, mailbox);
        }
    }

    void stop() override {
        std::unordered_map<
            std::uint64_t,
            std::shared_ptr<GenerationMailbox>> active;

        {
            std::lock_guard lock(mutex_);
            ++stopCalls_;

            if (throwOnStop_) {
                throw std::runtime_error("configured stop exception");
            }

            running_ = false;
            active.swap(activeMailboxes_);
        }

        for (auto& [requestId, mailbox] : active) {
            TerminalEvent terminal;
            terminal.request_id = requestId;
            terminal.status = failure(
                StatusCode::Cancelled,
                "request stopped by Fake Backend");
            terminal.finish_reason = FinishReason::Cancelled;
            (void)mailbox->pushTerminal(std::move(terminal));
        }
    }

    void setStartStatus(Status status) {
        std::lock_guard lock(mutex_);
        startStatus_ = std::move(status);
    }

    void setSubmitStatus(Status status) {
        std::lock_guard lock(mutex_);
        submitStatus_ = std::move(status);
    }

    void setThrowOnStart(bool enabled) {
        std::lock_guard lock(mutex_);
        throwOnStart_ = enabled;
    }

    void setThrowOnSubmit(bool enabled) {
        std::lock_guard lock(mutex_);
        throwOnSubmit_ = enabled;
    }

    void setThrowOnStop(bool enabled) {
        std::lock_guard lock(mutex_);
        throwOnStop_ = enabled;
    }

    bool finish(
        std::uint64_t requestId,
        Status status,
        std::optional<FinishReason> finishReason) {

        std::shared_ptr<GenerationMailbox> mailbox;

        {
            std::lock_guard lock(mutex_);
            auto const iterator = activeMailboxes_.find(requestId);

            if (iterator == activeMailboxes_.end()) {
                return false;
            }

            mailbox = iterator->second;
        }

        TerminalEvent terminal;
        terminal.request_id = requestId;
        terminal.status = std::move(status);
        terminal.finish_reason = finishReason;

        if (!mailbox->pushTerminal(std::move(terminal))) {
            return false;
        }

        eraseIfSame(requestId, mailbox);
        return true;
    }

    void blockNextSubmit() {
        std::lock_guard lock(mutex_);
        blockNextSubmit_ = true;
        blockedSubmitEntered_ = false;
        releaseBlockedSubmit_ = false;
    }

    bool waitUntilSubmitBlocked(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [this] {
            return blockedSubmitEntered_;
        });
    }

    void releaseSubmit() {
        std::lock_guard lock(mutex_);
        releaseBlockedSubmit_ = true;
        condition_.notify_all();
    }

    int startCalls() const {
        std::lock_guard lock(mutex_);
        return startCalls_;
    }

    int submitCalls() const {
        std::lock_guard lock(mutex_);
        return submitCalls_;
    }

    int stopCalls() const {
        std::lock_guard lock(mutex_);
        return stopCalls_;
    }

    std::vector<std::uint64_t> cancelledIds() const {
        std::lock_guard lock(mutex_);
        return cancelledIds_;
    }

private:
    void eraseIfSame(
        std::uint64_t requestId,
        std::shared_ptr<GenerationMailbox> const& mailbox) {

        std::lock_guard lock(mutex_);
        auto const iterator = activeMailboxes_.find(requestId);

        if (iterator != activeMailboxes_.end() &&
            iterator->second == mailbox) {
            activeMailboxes_.erase(iterator);
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;

    bool running_{false};
    bool throwOnStart_{false};
    bool throwOnSubmit_{false};
    bool throwOnStop_{false};

    bool blockNextSubmit_{false};
    bool blockedSubmitEntered_{false};
    bool releaseBlockedSubmit_{false};

    Status startStatus_;
    Status submitStatus_;

    int startCalls_{0};
    int submitCalls_{0};
    int stopCalls_{0};

    std::vector<std::uint64_t> cancelledIds_;
    std::unordered_map<
        std::uint64_t,
        std::shared_ptr<GenerationMailbox>> activeMailboxes_;
};

struct RuntimeFixture {
    RuntimeFixture()
        : RuntimeFixture(makeRuntimeConfig()) {}

    explicit RuntimeFixture(GenerationRuntimeConfig config)
        : RuntimeFixture(
              std::make_unique<FakeGenerationBackend>(),
              std::move(config)) {}

    FakeGenerationBackend* backend;
    GenerationRuntime runtime;

private:
    RuntimeFixture(
        std::unique_ptr<FakeGenerationBackend> ownedBackend,
        GenerationRuntimeConfig config)
        : backend(ownedBackend.get()),
          runtime(std::move(ownedBackend), std::move(config)) {}
};

void testConstructionValidation(int& failures) {
    bool nullBackendRejected{false};

    try {
        GenerationRuntime runtime{nullptr, makeRuntimeConfig()};
        (void)runtime;
    } catch (std::invalid_argument const&) {
        nullBackendRejected = true;
    }

    expect(
        nullBackendRejected,
        "Runtime must reject a null Backend",
        failures);

    bool invalidMailboxRejected{false};

    try {
        auto backend = std::make_unique<FakeGenerationBackend>();
        auto config = makeRuntimeConfig();
        config.mailbox.max_queued_deltas = 0;
        GenerationRuntime runtime{std::move(backend), config};
        (void)runtime;
    } catch (std::invalid_argument const&) {
        invalidMailboxRejected = true;
    }

    expect(
        invalidMailboxRejected,
        "Runtime must reject zero Mailbox capacity",
        failures);
}

void testStartFailureAndIdempotence(int& failures) {
    RuntimeFixture fixture;
    fixture.backend->setStartStatus(failure(
        StatusCode::TensorRTError,
        "configured start failure"));

    auto const failedStart = fixture.runtime.start();
    expect(!failedStart, "configured Backend start must fail", failures);
    expect(!fixture.runtime.running(), "failed Runtime must not run", failures);
    expect(
        !fixture.runtime.admissionSnapshot().accepting,
        "Admission must stay closed after start failure",
        failures);

    auto const rejected = fixture.runtime.submit(makeRequest(1));
    expect(
        rejected.status.code == StatusCode::NotReady,
        "submit after start failure must return NotReady",
        failures);
    expect(
        fixture.backend->submitCalls() == 0,
        "rejected Runtime submit must not reach Backend",
        failures);

    fixture.backend->setStartStatus(Status::success());
    expect(fixture.runtime.start().ok(), "Runtime retry must start", failures);
    expect(fixture.runtime.start().ok(), "repeated start must succeed", failures);
    expect(
        fixture.backend->startCalls() == 2,
        "running Runtime must not start Backend twice",
        failures);
    expect(
        fixture.runtime.admissionSnapshot().accepting,
        "Admission must open after Backend start",
        failures);

    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testStartAndSubmitExceptions(int& failures) {
    {
        RuntimeFixture fixture;
        fixture.backend->setThrowOnStart(true);

        auto const status = fixture.runtime.start();
        expect(
            status.code == StatusCode::InternalError,
            "Backend start exception must become InternalError",
            failures);
        expect(
            !fixture.runtime.admissionSnapshot().accepting,
            "start exception must not open Admission",
            failures);
    }

    {
        RuntimeFixture fixture;
        expect(fixture.runtime.start().ok(), "Runtime must start", failures);
        fixture.backend->setThrowOnSubmit(true);

        auto const submission = fixture.runtime.submit(makeRequest(2));
        expect(
            submission.status.code == StatusCode::InternalError,
            "Backend submit exception must become InternalError",
            failures);
        expect(
            !submission.mailbox,
            "submit exception must not expose a Mailbox",
            failures);
        expect(
            resourcesAreZero(fixture.runtime.admissionSnapshot()),
            "submit exception must release Admission Lease",
            failures);

        fixture.backend->setThrowOnSubmit(false);
        expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
    }
}

void testNormalTerminalReleasesAdmission(int& failures) {
    RuntimeFixture fixture{makeRuntimeConfig(2, 8, 8)};
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);

    auto submission = fixture.runtime.submit(makeRequest(10, 2, 3));
    expect(submission.accepted(), "valid request must be accepted", failures);

    auto const active = fixture.runtime.admissionSnapshot();
    expect(active.active_requests == 1, "one request must be active", failures);
    expect(active.reserved_input_tokens == 2, "input budget must be reserved", failures);
    expect(active.reserved_output_tokens == 3, "output budget must be reserved", failures);

    expect(
        fixture.backend->finish(10, Status::success(), FinishReason::Length),
        "Fake Backend must submit Terminal",
        failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "Terminal commit must immediately release Admission",
        failures);

    auto terminal = popTerminal(submission.mailbox);
    expect(terminal.has_value(), "consumer must receive Terminal", failures);
    if (terminal) {
        expect(terminal->request_id == 10, "Terminal must preserve request id", failures);
        expect(terminal->status.ok(), "normal Terminal must succeed", failures);
        expect(
            terminal->finish_reason == FinishReason::Length,
            "normal Terminal must preserve finish reason",
            failures);
    }

    GenerationEvent event;
    expect(
        submission.mailbox->waitPop(event, 0ms) == MailboxWaitResult::Closed,
        "Mailbox must close after Terminal delivery",
        failures);
    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testBackendSubmitFailureReleasesAdmission(int& failures) {
    RuntimeFixture fixture;
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);
    fixture.backend->setSubmitStatus(failure(
        StatusCode::TensorRTError,
        "configured submit failure"));

    auto const submission = fixture.runtime.submit(makeRequest(20));
    expect(!submission.accepted(), "Backend rejection must reject submission", failures);
    expect(
        submission.status.code == StatusCode::TensorRTError,
        "Backend rejection status must be preserved",
        failures);
    expect(!submission.mailbox, "rejected submission must not expose Mailbox", failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "Backend rejection must release Admission Lease",
        failures);

    fixture.backend->setSubmitStatus(Status::success());
    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testAdmissionRejections(int& failures) {
    RuntimeFixture fixture{makeRuntimeConfig(2, 4, 4)};
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);

    auto first = fixture.runtime.submit(makeRequest(30, 2, 2));
    expect(first.accepted(), "first request must be accepted", failures);

    auto const beforeDuplicate = fixture.runtime.admissionSnapshot();
    auto const duplicate = fixture.runtime.submit(makeRequest(30, 1, 1));
    expect(
        duplicate.status.code == StatusCode::AlreadyExists,
        "duplicate request id must return AlreadyExists",
        failures);
    auto const afterDuplicate = fixture.runtime.admissionSnapshot();
    expect(
        afterDuplicate.active_requests == beforeDuplicate.active_requests &&
            afterDuplicate.reserved_input_tokens ==
                beforeDuplicate.reserved_input_tokens &&
            afterDuplicate.reserved_output_tokens ==
                beforeDuplicate.reserved_output_tokens,
        "duplicate request must not change Admission usage",
        failures);

    auto const inputRejected = fixture.runtime.submit(makeRequest(31, 3, 1));
    expect(
        inputRejected.status.code == StatusCode::ResourceExhausted,
        "input capacity must return ResourceExhausted",
        failures);
    expect(
        inputRejected.status.message.find("input token") != std::string::npos,
        "input rejection must identify its capacity",
        failures);

    auto const outputRejected = fixture.runtime.submit(makeRequest(32, 1, 3));
    expect(
        outputRejected.status.code == StatusCode::ResourceExhausted,
        "output capacity must return ResourceExhausted",
        failures);
    expect(
        outputRejected.status.message.find("output token") != std::string::npos,
        "output rejection must identify its capacity",
        failures);

    auto second = fixture.runtime.submit(makeRequest(33, 1, 1));
    expect(second.accepted(), "second in-capacity request must be accepted", failures);

    auto const activeRejected = fixture.runtime.submit(makeRequest(34, 1, 1));
    expect(
        activeRejected.status.code == StatusCode::ResourceExhausted,
        "active request capacity must return ResourceExhausted",
        failures);
    expect(
        activeRejected.status.message.find("active request") != std::string::npos,
        "active rejection must identify its capacity",
        failures);
    expect(
        fixture.backend->submitCalls() == 2,
        "Admission rejections must not reach Backend",
        failures);

    (void)fixture.backend->finish(30, Status::success(), FinishReason::Length);
    (void)fixture.backend->finish(33, Status::success(), FinishReason::Length);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "all Admission resources must return to zero",
        failures);
    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testSloPolicyRejectsBeforeHardAdmission(int& failures) {
    auto config = makeRuntimeConfig(4, 128, 128);
    config.slo_policy = kimrt::llm::SloAdmissionPolicyConfig{
        "model",
        "revision",
        "engine",
        50.0,
        5.0,
        {
            {32, 1, 30.0},
            {32, 2, 48.0},
        },
    };
    RuntimeFixture fixture{std::move(config)};
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);

    auto first = fixture.runtime.submit(makeRequest(35, 17, 32));
    expect(first.accepted(), "profile-safe request must be accepted", failures);

    auto predicted_miss = fixture.runtime.submit(makeRequest(36, 17, 32));
    expect(
        predicted_miss.status.code == StatusCode::SloPredictedMiss,
        "profile-risk request must return SloPredictedMiss",
        failures);
    expect(
        fixture.backend->submitCalls() == 1,
        "soft-policy rejection must not reach Backend",
        failures);
    auto const snapshot = fixture.runtime.admissionSnapshot();
    expect(
        snapshot.active_requests == 1 &&
            snapshot.reserved_input_tokens == 17 &&
            snapshot.reserved_output_tokens == 32,
        "soft-policy rejection must not reserve hard capacity",
        failures);

    auto uncovered = fixture.runtime.submit(makeRequest(37, 33, 32));
    expect(
        uncovered.status.code == StatusCode::SloPredictedMiss,
        "uncovered input bucket must be conservatively rejected",
        failures);

    (void)fixture.backend->finish(35, Status::success(), FinishReason::Length);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "accepted policy request must release all resources",
        failures);
    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testExpiredDeadlineRejectedBeforeAdmission(int& failures) {
    RuntimeFixture fixture;
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);

    auto request = makeRequest(40);
    request.context.deadline = std::chrono::steady_clock::now() - 1ms;

    auto const submission = fixture.runtime.submit(std::move(request));
    expect(
        submission.status.code == StatusCode::Timeout,
        "expired request must return Timeout",
        failures);
    expect(
        fixture.backend->submitCalls() == 0,
        "expired request must not reach Backend",
        failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "expired request must not reserve Admission",
        failures);
    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testCancelTimeoutAndBackpressure(int& failures) {
    RuntimeFixture fixture{makeRuntimeConfig(3, 16, 16)};
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);

    auto cancelled = fixture.runtime.submit(makeRequest(50));
    expect(cancelled.accepted(), "cancel request must be accepted", failures);
    fixture.runtime.cancel(50);

    auto cancelledTerminal = popTerminal(cancelled.mailbox);
    expect(cancelledTerminal.has_value(), "cancel must produce Terminal", failures);
    if (cancelledTerminal) {
        expect(
            cancelledTerminal->status.code == StatusCode::Cancelled,
            "cancel Terminal must contain Cancelled",
            failures);
        expect(
            cancelledTerminal->finish_reason == FinishReason::Cancelled,
            "cancel Terminal must contain cancel finish reason",
            failures);
    }
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "cancel Terminal must release Admission",
        failures);

    fixture.runtime.cancel(50);
    GenerationEvent event;
    expect(
        cancelled.mailbox->waitPop(event, 0ms) == MailboxWaitResult::Closed,
        "repeated cancel must not produce a second Terminal",
        failures);

    auto timedOut = fixture.runtime.submit(makeRequest(51));
    expect(timedOut.accepted(), "timeout request must be accepted", failures);
    expect(
        fixture.backend->finish(
            51,
            failure(StatusCode::Timeout, "configured timeout"),
            FinishReason::Timeout),
        "Fake Backend must submit timeout Terminal",
        failures);
    auto timeoutTerminal = popTerminal(timedOut.mailbox);
    expect(
        timeoutTerminal && timeoutTerminal->status.code == StatusCode::Timeout,
        "timeout Terminal must be observable",
        failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "timeout Terminal must release Admission",
        failures);

    auto backpressured = fixture.runtime.submit(makeRequest(52));
    expect(backpressured.accepted(), "backpressure request must be accepted", failures);
    expect(
        fixture.backend->finish(
            52,
            failure(StatusCode::QueueFull, "configured backpressure"),
            FinishReason::Backpressure),
        "Fake Backend must submit backpressure Terminal",
        failures);
    auto backpressureTerminal = popTerminal(backpressured.mailbox);
    expect(
        backpressureTerminal &&
            backpressureTerminal->finish_reason == FinishReason::Backpressure,
        "backpressure Terminal must be observable",
        failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "backpressure Terminal must release Admission",
        failures);

    expect(fixture.runtime.stop().ok(), "Runtime stop must succeed", failures);
}

void testStopConvergesRequestsAndIsIdempotent(int& failures) {
    RuntimeFixture fixture{makeRuntimeConfig(2, 8, 8)};
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);

    auto first = fixture.runtime.submit(makeRequest(60, 2, 2));
    auto second = fixture.runtime.submit(makeRequest(61, 2, 2));
    expect(first.accepted() && second.accepted(), "two requests must be accepted", failures);

    auto const stopStatus = fixture.runtime.stop();
    expect(stopStatus.ok(), "stop must converge active requests", failures);
    expect(!fixture.runtime.running(), "stopped Runtime must not run", failures);
    expect(
        !fixture.runtime.admissionSnapshot().accepting,
        "stopped Runtime must close Admission",
        failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "stop must release all Admission resources",
        failures);

    auto firstTerminal = popTerminal(first.mailbox);
    auto secondTerminal = popTerminal(second.mailbox);
    expect(
        firstTerminal && firstTerminal->status.code == StatusCode::Cancelled,
        "stop must cancel first request",
        failures);
    expect(
        secondTerminal && secondTerminal->status.code == StatusCode::Cancelled,
        "stop must cancel second request",
        failures);

    expect(fixture.runtime.stop().ok(), "repeated stop must succeed", failures);
    expect(
        fixture.backend->stopCalls() == 1,
        "repeated Runtime stop must not stop Backend twice",
        failures);

    auto const submitCallsBefore = fixture.backend->submitCalls();
    auto const afterStop = fixture.runtime.submit(makeRequest(62));
    expect(
        afterStop.status.code == StatusCode::NotReady,
        "submit after stop must return NotReady",
        failures);
    expect(
        fixture.backend->submitCalls() == submitCallsBefore,
        "submit after stop must not reach Backend",
        failures);
}

void testSubmitStopRace(int& failures) {
    RuntimeFixture fixture{makeRuntimeConfig(1, 8, 8)};
    expect(fixture.runtime.start().ok(), "Runtime must start", failures);
    fixture.backend->blockNextSubmit();

    std::optional<GenerationSubmission> submission;
    Status stopStatus;
    std::atomic<bool> stopStarted{false};

    std::thread submitThread([&] {
        submission.emplace(fixture.runtime.submit(makeRequest(70)));
    });

    auto const submitBlocked = fixture.backend->waitUntilSubmitBlocked(2s);
    expect(
        submitBlocked,
        "Fake Backend submit must reach deterministic barrier",
        failures);

    std::thread stopThread([&] {
        stopStarted.store(true, std::memory_order_release);
        stopStatus = fixture.runtime.stop();
    });

    while (!stopStarted.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    fixture.backend->releaseSubmit();
    submitThread.join();
    stopThread.join();

    expect(submission && submission->accepted(),
           "submit already inside Runtime must finish before stop",
           failures);
    expect(stopStatus.ok(), "concurrent stop must succeed", failures);
    expect(
        resourcesAreZero(fixture.runtime.admissionSnapshot()),
        "submit/stop race must finish with zero resources",
        failures);

    if (submission && submission->mailbox) {
        auto terminal = popTerminal(submission->mailbox);
        expect(
            terminal && terminal->status.code == StatusCode::Cancelled,
            "stop must converge request accepted by racing submit",
            failures);
    }

    auto const submitCallsBefore = fixture.backend->submitCalls();
    auto const afterStop = fixture.runtime.submit(makeRequest(71));
    expect(
        afterStop.status.code == StatusCode::NotReady,
        "request after stop race must be rejected",
        failures);
    expect(
        fixture.backend->submitCalls() == submitCallsBefore,
        "closed Admission must prevent request from reaching Backend",
        failures);
}

} // namespace

int main() {
    int failures{0};

    try {
        testConstructionValidation(failures);
        testStartFailureAndIdempotence(failures);
        testStartAndSubmitExceptions(failures);
        testNormalTerminalReleasesAdmission(failures);
        testBackendSubmitFailureReleasesAdmission(failures);
        testAdmissionRejections(failures);
        testSloPolicyRejectsBeforeHardAdmission(failures);
        testExpiredDeadlineRejectedBeforeAdmission(failures);
        testCancelTimeoutAndBackpressure(failures);
        testStopConvergesRequestsAndIsIdempotent(failures);
        testSubmitStopRace(failures);
    } catch (std::exception const& exception) {
        ++failures;
        std::cerr << "[FAIL] unexpected exception: "
                  << exception.what() << '\n';
    } catch (...) {
        ++failures;
        std::cerr << "[FAIL] unexpected unknown exception\n";
    }

    if (failures == 0) {
        std::cout << "[PASS] GenerationRuntime contract\n";
        return 0;
    }

    std::cerr << failures << " GenerationRuntime contract checks failed\n";
    return 1;
}
