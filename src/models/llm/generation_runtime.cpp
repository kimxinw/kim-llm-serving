#include "models/llm/generation_runtime.h"

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace kimrt::llm {
namespace {

Status failure(StatusCode code, std::string message) {
    return Status::error(code, std::move(message));
}

Status validateRequestForAdmission(
    GenerationRequest const& request) {

    if (request.context.request_id == 0) {
        return failure(
            StatusCode::InvalidInput,
            "request id must be greater than zero");
    }

    if (request.input_token_ids.empty()) {
        return failure(
            StatusCode::InvalidInput,
            "input tokens must not be empty");
    }

    if (request.max_new_tokens == 0) {
        return failure(
            StatusCode::InvalidInput,
            "max_new_tokens must be greater than zero");
    }

    if (request.context.hasDeadline() &&
        request.context.deadline <=
            std::chrono::steady_clock::now()) {
        return failure(
            StatusCode::Timeout,
            "request deadline has expired");
    }

    return Status::success();
}

Status admissionFailure(AdmissionCode code) {
    switch (code) {
    case AdmissionCode::InvalidRequest:
        return failure(
            StatusCode::InvalidInput,
            "invalid Admission request");

    case AdmissionCode::NotAccepting:
        return failure(
            StatusCode::NotReady,
            "Generation Runtime is not accepting requests");

    case AdmissionCode::DuplicateRequestId:
        return failure(
            StatusCode::AlreadyExists,
            "request id is already active");

    case AdmissionCode::ActiveRequestLimit:
        return failure(
            StatusCode::ResourceExhausted,
            "active request capacity is exhausted");

    case AdmissionCode::InputTokenLimit:
        return failure(
            StatusCode::ResourceExhausted,
            "input token capacity is exhausted");

    case AdmissionCode::OutputTokenLimit:
        return failure(
            StatusCode::ResourceExhausted,
            "reserved output token capacity is exhausted");

    case AdmissionCode::Admitted:
        return failure(
            StatusCode::InternalError,
            "admitted request did not contain a valid lease");
    }

    return failure(
        StatusCode::InternalError,
        "unknown Admission result");
}

bool hasOutstandingResources(
    AdmissionSnapshot const& snapshot) {

    return snapshot.active_requests != 0 ||
            snapshot.reserved_input_tokens != 0 ||
            snapshot.reserved_output_tokens != 0;
}

} // namespace

GenerationRuntime::GenerationRuntime(
    std::unique_ptr<GenerationBackend> backend,
    GenerationRuntimeConfig config)
    : backend_(std::move(backend)),
    mailboxConfig_(config.mailbox),
    admission_(config.admission) {

    if (!backend_) {
        throw std::invalid_argument(
            "GenerationRuntime requires a Backend");
    }

    if (mailboxConfig_.max_queued_deltas == 0 ||
        mailboxConfig_.max_queued_tokens == 0) {
        throw std::invalid_argument(
            "GenerationRuntime Mailbox capacity must be positive");
    }
}

GenerationRuntime::~GenerationRuntime() {
    (void)stop();
}

Status GenerationRuntime::start() {
    std::unique_lock lock(lifecycleMutex_);

    if (state_ == State::Running) {
        return Status::success();
    }

    if (state_ == State::Failed) {
        return failure(
            StatusCode::NotReady,
            "failed GenerationRuntime must be stopped before restart");
    }

    Status backendStatus;

    try {
        backendStatus = backend_->start();
    } catch (std::exception const& exception) {
        return failure(
            StatusCode::InternalError,
            std::string{"Backend start threw an exception: "} +
                exception.what());
    } catch (...) {
        return failure(
            StatusCode::InternalError,
            "Backend start threw an unknown exception");
    }

    if (!backendStatus) {
        return backendStatus;
    }

    if (!admission_.open()) {
        try {
            backend_->stop();
        } catch (...) {
        }

        state_ = State::Failed;

        return failure(
            StatusCode::InternalError,
            "Admission could not open after Backend start");
    }

    state_ = State::Running;
    return Status::success();
}

GenerationSubmission GenerationRuntime::submit(
    GenerationRequest request) {

    if (auto status = validateRequestForAdmission(request);
        !status) {
        return {std::move(status), {}};
    }

    /*
    * submit 持有共享锁，允许多个请求并发提交。
    * stop/start 持有独占锁，因此不会插入一次提交过程。
    */
    std::shared_lock lock(lifecycleMutex_);

    if (state_ != State::Running) {
        return {
            failure(
                StatusCode::NotReady,
                "GenerationRuntime is not running"),
            {},
        };
    }

    auto decision = admission_.tryAcquire({
        request.context.request_id,
        request.input_token_ids.size(),
        request.max_new_tokens,
    });

    if (!decision.admitted()) {
        return {
            admissionFailure(decision.code),
            {},
        };
    }

    std::shared_ptr<GenerationMailbox> mailbox;

    try {
        mailbox = std::make_shared<GenerationMailbox>(
            mailboxConfig_,
            std::move(decision.lease));
    } catch (std::exception const& exception) {
        return {
            failure(
                StatusCode::InternalError,
                std::string{"failed to create Mailbox: "} +
                    exception.what()),
            {},
        };
    } catch (...) {
        return {
            failure(
                StatusCode::InternalError,
                "failed to create Mailbox"),
            {},
        };
    }

    Status backendStatus;

    try {
        backendStatus = backend_->submit(
            std::move(request),
            mailbox);
    } catch (std::exception const& exception) {
        return {
            failure(
                StatusCode::InternalError,
                std::string{"Backend submit threw an exception: "} +
                    exception.what()),
            {},
        };
    } catch (...) {
        return {
            failure(
                StatusCode::InternalError,
                "Backend submit threw an unknown exception"),
            {},
        };
    }

    if (!backendStatus) {
        /*
        * Backend 失败时不得保留 Mailbox。
        * 当前函数结束后 mailbox 析构，Lease 自动归还。
        */
        return {
            std::move(backendStatus),
            {},
        };
    }

    return {
        Status::success(),
        std::move(mailbox),
    };
}

void GenerationRuntime::cancel(
    std::uint64_t requestId) {

    if (requestId == 0) {
        return;
    }

    std::shared_lock lock(lifecycleMutex_);

    if (state_ != State::Running) {
        return;
    }

    backend_->cancel(requestId);
}

Status GenerationRuntime::stop() noexcept {
    std::unique_lock lock(lifecycleMutex_);

    if (state_ == State::Stopped) {
        return Status::success();
    }

    admission_.close();

    try {
        backend_->stop();
    } catch (...) {
        state_ = State::Failed;

        return failure(
            StatusCode::InternalError,
            "Backend stop threw an exception");
    }

    auto const snapshot = admission_.snapshot();

    if (hasOutstandingResources(snapshot)) {
        state_ = State::Failed;

        return failure(
            StatusCode::InternalError,
            "Backend stopped with outstanding Admission resources");
    }

    state_ = State::Stopped;
    return Status::success();
}

bool GenerationRuntime::running() const noexcept {
    std::shared_lock lock(lifecycleMutex_);
    return state_ == State::Running;
}

AdmissionSnapshot
GenerationRuntime::admissionSnapshot() const noexcept {
    return admission_.snapshot();
}

} // namespace kimrt::llm
