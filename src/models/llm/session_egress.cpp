#include "models/llm/session_egress.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

namespace kimrt::llm::ipc {
namespace {

[[nodiscard]] Status invalid(std::string message) {
    return Status::error(StatusCode::InvalidInput, std::move(message));
}

[[nodiscard]] Status queueFull(std::string message) {
    return Status::error(StatusCode::QueueFull, std::move(message));
}

[[nodiscard]] Status unavailable(std::string message) {
    return Status::error(StatusCode::Unavailable, std::move(message));
}

struct SizedMessage {
    Message message;
    std::uint64_t frame_bytes{0};
};

struct SizedMessageResult {
    Status status;
    SizedMessage message;
};

[[nodiscard]] SizedMessageResult sizeMessage(
    Message message,
    std::uint32_t max_frame_payload_bytes) {
    auto payload = encodePayload(message);
    if (!payload.ok()) {
        return {std::move(payload.status), {}};
    }
    auto frame = encodeFrame(
        payload.payload,
        FrameCodecConfig{max_frame_payload_bytes});
    if (!frame.ok()) {
        return {std::move(frame.status), {}};
    }
    return {
        Status::success(),
        {std::move(message), frame.bytes.size()},
    };
}

[[nodiscard]] SessionEgressEnqueueResult enqueueFailure(
    SessionEgressEnqueueCode code,
    Status status) {
    return {code, std::move(status)};
}

[[nodiscard]] SessionEgressEnqueueResult enqueued() {
    return {
        SessionEgressEnqueueCode::Enqueued,
        Status::success(),
    };
}

} // namespace

struct SessionEgress::Impl {
    struct RequestQueue {
        std::deque<SizedMessage> messages;
        std::uint64_t queued_bytes{0};
        bool scheduled{false};
        bool terminal_enqueued{false};
    };

    explicit Impl(SessionEgressConfig input_config)
        : config(std::move(input_config)) {
        validateConfiguration();
    }

    void validateConfiguration() const {
        if (config.max_frame_payload_bytes == 0 ||
            config.max_session_frames == 0 ||
            config.max_session_bytes == 0 ||
            config.max_request_frames == 0 ||
            config.max_request_bytes == 0 ||
            config.control_reserve_frames == 0 ||
            config.control_reserve_bytes == 0 ||
            config.terminal_reserve_bytes == 0) {
            throw std::invalid_argument(
                "SessionEgress capacity values must be positive");
        }
        if (config.control_reserve_frames >= config.max_session_frames ||
            config.control_reserve_bytes >= config.max_session_bytes) {
            throw std::invalid_argument(
                "SessionEgress control reserve must leave request capacity");
        }
        if (config.max_request_frames > requestFrameCapacity() ||
            config.max_request_bytes > requestByteCapacity()) {
            throw std::invalid_argument(
                "SessionEgress request capacity exceeds session data capacity");
        }
        if (config.terminal_reserve_bytes > config.max_request_bytes) {
            throw std::invalid_argument(
                "SessionEgress Terminal reserve exceeds request capacity");
        }
    }

    [[nodiscard]] std::uint64_t requestFrameCapacity() const noexcept {
        return static_cast<std::uint64_t>(config.max_session_frames) -
            config.control_reserve_frames;
    }

    [[nodiscard]] std::uint64_t requestByteCapacity() const noexcept {
        return config.max_session_bytes - config.control_reserve_bytes;
    }

    void updateHighWatermarks() noexcept {
        auto const frames = control_frames + request_frames;
        auto const bytes = control_bytes + request_bytes;
        high_watermark_frames = std::max(high_watermark_frames, frames);
        high_watermark_bytes = std::max(high_watermark_bytes, bytes);
    }

    void removeScheduledRequest(std::uint64_t request_id) {
        round_robin.erase(
            std::remove(
                round_robin.begin(),
                round_robin.end(),
                request_id),
            round_robin.end());
    }

    SessionEgressConfig config;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<SizedMessage> control;
    std::unordered_map<std::uint64_t, RequestQueue> requests;
    std::deque<std::uint64_t> round_robin;
    std::uint64_t control_frames{0};
    std::uint64_t control_bytes{0};
    std::uint64_t request_frames{0};
    std::uint64_t request_bytes{0};
    std::uint64_t terminal_reservations{0};
    std::uint64_t high_watermark_frames{0};
    std::uint64_t high_watermark_bytes{0};
    bool accepting{true};
};

SessionEgress::SessionEgress(SessionEgressConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

SessionEgress::~SessionEgress() {
    stop();
}

Status SessionEgress::registerRequest(std::uint64_t request_id) {
    if (request_id == 0) {
        return invalid("SessionEgress request id must be positive");
    }

    std::lock_guard lock(impl_->mutex);
    if (!impl_->accepting) {
        return unavailable("SessionEgress is stopped");
    }
    if (impl_->requests.find(request_id) != impl_->requests.end()) {
        return Status::error(
            StatusCode::AlreadyExists,
            "SessionEgress request id is already registered");
    }
    if (impl_->request_frames + impl_->terminal_reservations + 1 >
            impl_->requestFrameCapacity() ||
        impl_->request_bytes +
                (impl_->terminal_reservations + 1) *
                    impl_->config.terminal_reserve_bytes >
            impl_->requestByteCapacity()) {
        return queueFull(
            "SessionEgress cannot reserve capacity for another Terminal");
    }

    impl_->requests.emplace(request_id, Impl::RequestQueue{});
    ++impl_->terminal_reservations;
    return Status::success();
}

bool SessionEgress::abandonRequest(std::uint64_t request_id) noexcept {
    try {
        std::lock_guard lock(impl_->mutex);
        auto const iterator = impl_->requests.find(request_id);
        if (iterator == impl_->requests.end()) {
            return false;
        }

        auto const& queue = iterator->second;
        impl_->request_frames -= queue.messages.size();
        impl_->request_bytes -= queue.queued_bytes;
        if (!queue.terminal_enqueued) {
            --impl_->terminal_reservations;
        }
        impl_->removeScheduledRequest(request_id);
        impl_->requests.erase(iterator);
        return true;
    } catch (...) {
        return false;
    }
}

SessionEgressEnqueueResult SessionEgress::enqueueControl(
    Message message) {
    if (std::holds_alternative<TokenDelta>(message) ||
        std::holds_alternative<Terminal>(message) ||
        std::holds_alternative<Submit>(message) ||
        std::holds_alternative<Cancel>(message) ||
        std::holds_alternative<Health>(message) ||
        std::holds_alternative<Hello>(message) ||
        std::holds_alternative<HelloAck>(message)) {
        return enqueueFailure(
            SessionEgressEnqueueCode::InvalidMessage,
            invalid("SessionEgress control queue received a non-control message"));
    }

    auto sized = sizeMessage(
        std::move(message),
        impl_->config.max_frame_payload_bytes);
    if (!sized.status.ok()) {
        return enqueueFailure(
            SessionEgressEnqueueCode::InvalidMessage,
            std::move(sized.status));
    }

    {
        std::lock_guard lock(impl_->mutex);
        if (!impl_->accepting) {
            return enqueueFailure(
                SessionEgressEnqueueCode::Stopped,
                unavailable("SessionEgress is stopped"));
        }
        if (impl_->control_frames + 1 >
                impl_->config.control_reserve_frames ||
            sized.message.frame_bytes >
                impl_->config.control_reserve_bytes - impl_->control_bytes) {
            return enqueueFailure(
                SessionEgressEnqueueCode::ControlFull,
                queueFull("SessionEgress control reserve is full"));
        }

        impl_->control_bytes += sized.message.frame_bytes;
        ++impl_->control_frames;
        impl_->control.push_back(std::move(sized.message));
        impl_->updateHighWatermarks();
    }
    impl_->condition.notify_one();
    return enqueued();
}

SessionEgressEnqueueResult SessionEgress::enqueueDelta(
    std::uint64_t request_id,
    Message message) {
    auto const* delta = std::get_if<TokenDelta>(&message);
    if (delta == nullptr || delta->request_id != request_id) {
        return enqueueFailure(
            SessionEgressEnqueueCode::InvalidMessage,
            invalid("SessionEgress Delta request id mismatch"));
    }

    auto sized = sizeMessage(
        std::move(message),
        impl_->config.max_frame_payload_bytes);
    if (!sized.status.ok()) {
        return enqueueFailure(
            SessionEgressEnqueueCode::InvalidMessage,
            std::move(sized.status));
    }

    {
        std::lock_guard lock(impl_->mutex);
        if (!impl_->accepting) {
            return enqueueFailure(
                SessionEgressEnqueueCode::Stopped,
                unavailable("SessionEgress is stopped"));
        }
        auto iterator = impl_->requests.find(request_id);
        if (iterator == impl_->requests.end()) {
            return enqueueFailure(
                SessionEgressEnqueueCode::UnknownRequest,
                invalid("SessionEgress Delta request is not registered"));
        }
        auto& queue = iterator->second;
        if (queue.terminal_enqueued) {
            return enqueueFailure(
                SessionEgressEnqueueCode::InvalidMessage,
                invalid("SessionEgress cannot enqueue Delta after Terminal"));
        }

        if (queue.messages.size() + 2 >
                impl_->config.max_request_frames ||
            sized.message.frame_bytes + queue.queued_bytes +
                    impl_->config.terminal_reserve_bytes >
                impl_->config.max_request_bytes) {
            return enqueueFailure(
                SessionEgressEnqueueCode::RequestFull,
                queueFull("per-request SessionEgress queue is full"));
        }
        if (impl_->request_frames + 1 +
                impl_->terminal_reservations >
                impl_->requestFrameCapacity() ||
            sized.message.frame_bytes + impl_->request_bytes +
                    impl_->terminal_reservations *
                        impl_->config.terminal_reserve_bytes >
                impl_->requestByteCapacity()) {
            return enqueueFailure(
                SessionEgressEnqueueCode::SessionFull,
                queueFull("SessionEgress request partition is full"));
        }

        queue.queued_bytes += sized.message.frame_bytes;
        queue.messages.push_back(std::move(sized.message));
        ++impl_->request_frames;
        impl_->request_bytes += queue.messages.back().frame_bytes;
        if (!queue.scheduled) {
            queue.scheduled = true;
            impl_->round_robin.push_back(request_id);
        }
        impl_->updateHighWatermarks();
    }
    impl_->condition.notify_one();
    return enqueued();
}

SessionEgressEnqueueResult SessionEgress::enqueueTerminal(
    std::uint64_t request_id,
    Message message) {
    auto const* terminal = std::get_if<Terminal>(&message);
    if (terminal == nullptr || terminal->request_id != request_id) {
        return enqueueFailure(
            SessionEgressEnqueueCode::InvalidMessage,
            invalid("SessionEgress Terminal request id mismatch"));
    }

    auto sized = sizeMessage(
        std::move(message),
        impl_->config.max_frame_payload_bytes);
    if (!sized.status.ok()) {
        return enqueueFailure(
            SessionEgressEnqueueCode::InvalidMessage,
            std::move(sized.status));
    }

    {
        std::lock_guard lock(impl_->mutex);
        if (!impl_->accepting) {
            return enqueueFailure(
                SessionEgressEnqueueCode::Stopped,
                unavailable("SessionEgress is stopped"));
        }
        auto iterator = impl_->requests.find(request_id);
        if (iterator == impl_->requests.end()) {
            return enqueueFailure(
                SessionEgressEnqueueCode::UnknownRequest,
                invalid("SessionEgress Terminal request is not registered"));
        }
        auto& queue = iterator->second;
        if (queue.terminal_enqueued) {
            return enqueueFailure(
                SessionEgressEnqueueCode::InvalidMessage,
                invalid("SessionEgress request already has a Terminal"));
        }
        if (sized.message.frame_bytes >
            impl_->config.terminal_reserve_bytes) {
            return enqueueFailure(
                SessionEgressEnqueueCode::InvalidMessage,
                invalid("Terminal exceeds its SessionEgress byte reserve"));
        }
        if (queue.messages.size() + 1 >
                impl_->config.max_request_frames ||
            sized.message.frame_bytes >
                impl_->config.max_request_bytes - queue.queued_bytes) {
            return enqueueFailure(
                SessionEgressEnqueueCode::RequestFull,
                queueFull("per-request SessionEgress Terminal reserve is full"));
        }

        queue.terminal_enqueued = true;
        --impl_->terminal_reservations;
        queue.queued_bytes += sized.message.frame_bytes;
        queue.messages.push_back(std::move(sized.message));
        ++impl_->request_frames;
        impl_->request_bytes += queue.messages.back().frame_bytes;
        if (!queue.scheduled) {
            queue.scheduled = true;
            impl_->round_robin.push_back(request_id);
        }
        impl_->updateHighWatermarks();
    }
    impl_->condition.notify_one();
    return enqueued();
}

SessionEgressWaitResult SessionEgress::waitPop(
    Message& message,
    std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) {
        return SessionEgressWaitResult::Timeout;
    }

    std::unique_lock lock(impl_->mutex);
    auto const ready = [this] {
        return !impl_->control.empty() ||
            !impl_->round_robin.empty() || !impl_->accepting;
    };
    if (!ready() && !impl_->condition.wait_for(lock, timeout, ready)) {
        return SessionEgressWaitResult::Timeout;
    }

    if (!impl_->control.empty()) {
        auto queued = std::move(impl_->control.front());
        impl_->control.pop_front();
        --impl_->control_frames;
        impl_->control_bytes -= queued.frame_bytes;
        message = std::move(queued.message);
        return SessionEgressWaitResult::Message;
    }

    while (!impl_->round_robin.empty()) {
        auto const request_id = impl_->round_robin.front();
        impl_->round_robin.pop_front();
        auto iterator = impl_->requests.find(request_id);
        if (iterator == impl_->requests.end() ||
            iterator->second.messages.empty()) {
            continue;
        }

        auto& queue = iterator->second;
        auto queued = std::move(queue.messages.front());
        queue.messages.pop_front();
        queue.queued_bytes -= queued.frame_bytes;
        --impl_->request_frames;
        impl_->request_bytes -= queued.frame_bytes;

        bool const terminal =
            std::holds_alternative<Terminal>(queued.message);
        if (!queue.messages.empty()) {
            impl_->round_robin.push_back(request_id);
        } else if (terminal) {
            impl_->requests.erase(iterator);
        } else {
            queue.scheduled = false;
        }

        message = std::move(queued.message);
        return SessionEgressWaitResult::Message;
    }

    return impl_->accepting
        ? SessionEgressWaitResult::Timeout
        : SessionEgressWaitResult::Stopped;
}

void SessionEgress::stop() noexcept {
    try {
        {
            std::lock_guard lock(impl_->mutex);
            if (!impl_->accepting) {
                return;
            }
            impl_->accepting = false;
            impl_->control.clear();
            impl_->requests.clear();
            impl_->round_robin.clear();
            impl_->control_frames = 0;
            impl_->control_bytes = 0;
            impl_->request_frames = 0;
            impl_->request_bytes = 0;
            impl_->terminal_reservations = 0;
        }
        impl_->condition.notify_all();
    } catch (...) {
    }
}

SessionEgressSnapshot SessionEgress::snapshot() const noexcept {
    try {
        std::lock_guard lock(impl_->mutex);
        return {
            impl_->accepting,
            impl_->control_frames + impl_->request_frames,
            impl_->control_bytes + impl_->request_bytes,
            impl_->high_watermark_frames,
            impl_->high_watermark_bytes,
            impl_->requests.size(),
            impl_->terminal_reservations,
        };
    } catch (...) {
        return {};
    }
}

} // namespace kimrt::llm::ipc
