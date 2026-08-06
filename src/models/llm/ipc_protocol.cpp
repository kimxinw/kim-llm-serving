#include "models/llm/ipc_protocol.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kimrt::llm::ipc {
namespace {

using Json = nlohmann::json;

[[nodiscard]] bool statusOk(Status const& status) noexcept {
    return status.code == StatusCode::Ok;
}

[[nodiscard]] PayloadEncodeResult payloadError(
    std::string message) {
    return {
        Status::error(
            StatusCode::InvalidInput,
            std::move(message)),
        {},
    };
}

[[nodiscard]] HandshakeDecodeResult handshakeError(
    std::string message) {
    return {
        Status::error(
            StatusCode::InvalidInput,
            std::move(message)),
        std::nullopt,
    };
}

[[nodiscard]] std::optional<std::uint64_t> readUnsigned(
    Json const& object,
    char const* key) {
    auto const iterator = object.find(key);

    if (iterator == object.end() ||
        !iterator->is_number_unsigned()) {
        return std::nullopt;
    }

    return iterator->get<std::uint64_t>();
}

} // namespace

bool FrameEncodeResult::ok() const noexcept {
    return statusOk(status);
}

FrameEncodeResult::operator bool() const noexcept {
    return ok();
}

FrameEncodeResult encodeFrame(
    std::string_view payload,
    FrameCodecConfig config) {
    if (config.max_payload_bytes == 0) {
        return {
            Status::error(
                StatusCode::InvalidInput,
                "max_payload_bytes must be positive"),
            {},
        };
    }

    if (payload.empty()) {
        return {
            Status::error(
                StatusCode::InvalidInput,
                "frame payload must not be empty"),
            {},
        };
    }

    if (payload.size() > config.max_payload_bytes ||
        payload.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return {
            Status::error(
                StatusCode::ResourceExhausted,
                "frame payload exceeds max_payload_bytes"),
            {},
        };
    }

    auto const payload_size =
        static_cast<std::uint32_t>(payload.size());

    std::vector<std::uint8_t> frame;
    frame.reserve(4U + payload.size());

    frame.push_back(static_cast<std::uint8_t>(
        (payload_size >> 24U) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>(
        (payload_size >> 16U) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>(
        (payload_size >> 8U) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>(
        payload_size & 0xFFU));

    auto const* begin =
        reinterpret_cast<std::uint8_t const*>(payload.data());

    frame.insert(
        frame.end(),
        begin,
        begin + payload.size());

    return {
        Status::success(),
        std::move(frame),
    };
}

bool FrameDecodeResult::ok() const noexcept {
    return state != FrameDecodeState::Error &&
        statusOk(status);
}

FrameDecodeResult::operator bool() const noexcept {
    return ok();
}

FrameDecoder::FrameDecoder(FrameCodecConfig config)
    : config_(config) {
    if (config_.max_payload_bytes == 0) {
        throw std::invalid_argument(
            "max_payload_bytes must be positive");
    }
}

FrameDecodeResult FrameDecoder::feed(
    std::string_view bytes) {
    if (failed_) {
        return {
            FrameDecodeState::Error,
            failure_,
            {},
        };
    }

    std::vector<std::string> completed_payloads;

    auto const* data =
        reinterpret_cast<std::uint8_t const*>(bytes.data());

    std::size_t position = 0;

    while (position < bytes.size()) {
        while (prefix_size_ < prefix_.size() &&
                position < bytes.size()) {
            prefix_[prefix_size_++] = data[position++];
        }

        if (prefix_size_ < prefix_.size()) {
            break;
        }

        if (expected_payload_size_ == 0) {
            expected_payload_size_ =
                (static_cast<std::uint32_t>(
                    prefix_[0]) << 24U) |
                (static_cast<std::uint32_t>(
                    prefix_[1]) << 16U) |
                (static_cast<std::uint32_t>(
                    prefix_[2]) << 8U) |
                static_cast<std::uint32_t>(prefix_[3]);

            if (expected_payload_size_ == 0) {
                return fail(
                    StatusCode::InvalidInput,
                    "frame payload length must be positive");
            }

            if (expected_payload_size_ >
                config_.max_payload_bytes) {
                return fail(
                    StatusCode::ResourceExhausted,
                    "frame payload length exceeds "
                    "max_payload_bytes");
            }

            payload_.clear();
            payload_.reserve(expected_payload_size_);
        }

        auto const remaining_payload =
            static_cast<std::size_t>(
                expected_payload_size_) -
            payload_.size();

        auto const available =
            bytes.size() - position;

        auto const to_copy =
            std::min(remaining_payload, available);

        payload_.append(
            reinterpret_cast<char const*>(
                data + position),
            to_copy);

        position += to_copy;

        if (payload_.size() != expected_payload_size_) {
            break;
        }

        completed_payloads.push_back(
            std::move(payload_));

        payload_.clear();
        prefix_size_ = 0;
        expected_payload_size_ = 0;
    }

    auto const state = completed_payloads.empty()
        ? FrameDecodeState::NeedMore
        : FrameDecodeState::FramesReady;

    return {
        state,
        Status::success(),
        std::move(completed_payloads),
    };
}

void FrameDecoder::reset() noexcept {
    prefix_.fill(0);
    prefix_size_ = 0;
    expected_payload_size_ = 0;
    payload_.clear();

    failed_ = false;
    failure_ = Status::success();
}

bool FrameDecoder::failed() const noexcept {
    return failed_;
}

bool FrameDecoder::hasPartialFrame() const noexcept {
    return prefix_size_ != 0 ||
        expected_payload_size_ != 0;
}

FrameDecodeResult FrameDecoder::fail(
    StatusCode code,
    std::string message) {
    failed_ = true;
    failure_ = Status::error(
        code,
        std::move(message));

    prefix_.fill(0);
    prefix_size_ = 0;
    expected_payload_size_ = 0;
    payload_.clear();

    return {
        FrameDecodeState::Error,
        failure_,
        {},
    };
}

bool PayloadEncodeResult::ok() const noexcept {
    return statusOk(status);
}

PayloadEncodeResult::operator bool() const noexcept {
    return ok();
}

bool HandshakeDecodeResult::ok() const noexcept {
    return statusOk(status) &&
        message.has_value();
}

HandshakeDecodeResult::operator bool() const noexcept {
    return ok();
}

PayloadEncodeResult encodeHelloPayload(
    Hello const& hello) {
    if (hello.protocol_version != kProtocolVersion) {
        return payloadError(
            "unsupported Hello protocol_version");
    }

    Json const payload{
        {"type", "hello"},
        {"protocol_version", hello.protocol_version},
    };

    return {
        Status::success(),
        payload.dump(),
    };
}

PayloadEncodeResult encodeHelloAckPayload(
    HelloAck const& hello_ack) {
    if (hello_ack.protocol_version !=
        kProtocolVersion) {
        return payloadError(
            "unsupported HelloAck protocol_version");
    }

    if (hello_ack.worker_epoch == 0) {
        return payloadError(
            "HelloAck worker_epoch must be positive");
    }

    Json const payload{
        {"type", "hello_ack"},
        {
            "protocol_version",
            hello_ack.protocol_version,
        },
        {"worker_epoch", hello_ack.worker_epoch},
    };

    return {
        Status::success(),
        payload.dump(),
    };
}

HandshakeDecodeResult decodeHandshakePayload(
    std::string_view payload) {
    Json object;

    try {
        object = Json::parse(
            payload.begin(),
            payload.end());
    } catch (Json::exception const& exception) {
        return handshakeError(
            std::string{"invalid JSON payload: "} +
            exception.what());
    }

    if (!object.is_object()) {
        return handshakeError(
            "handshake payload must be a JSON object");
    }

    auto const type_iterator =
        object.find("type");

    if (type_iterator == object.end() ||
        !type_iterator->is_string()) {
        return handshakeError(
            "handshake type must be a string");
    }

    auto const type =
        type_iterator->get<std::string>();

    auto const protocol_version =
        readUnsigned(object, "protocol_version");

    if (!protocol_version.has_value() ||
        *protocol_version >
            std::numeric_limits<std::uint32_t>::max()) {
        return handshakeError(
            "protocol_version must be an unsigned "
            "32-bit integer");
    }

    if (*protocol_version != kProtocolVersion) {
        return handshakeError(
            "unsupported protocol_version");
    }

    if (type == "hello") {
        if (object.size() != 2 ||
            object.find("protocol_version") ==
                object.end()) {
            return handshakeError(
                "Hello contains unexpected fields");
        }

        return {
            Status::success(),
            HandshakeMessage{
                Hello{
                    static_cast<std::uint32_t>(
                        *protocol_version),
                },
            },
        };
    }

    if (type == "hello_ack") {
        if (object.size() != 3 ||
            object.find("protocol_version") ==
                object.end() ||
            object.find("worker_epoch") ==
                object.end()) {
            return handshakeError(
                "HelloAck contains unexpected fields");
        }

        auto const worker_epoch =
            readUnsigned(object, "worker_epoch");

        if (!worker_epoch.has_value() ||
            *worker_epoch == 0) {
            return handshakeError(
                "worker_epoch must be a positive "
                "unsigned integer");
        }

        return {
            Status::success(),
            HandshakeMessage{
                HelloAck{
                    static_cast<std::uint32_t>(
                        *protocol_version),
                    *worker_epoch,
                },
            },
        };
    }

    return handshakeError(
        "unknown handshake message type");
}

} // namespace kimrt::llm::ipc