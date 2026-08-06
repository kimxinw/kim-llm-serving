#include "models/llm/ipc_protocol.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <utility>

namespace {

using kimrt::StatusCode;
using kimrt::llm::ipc::FrameCodecConfig;
using kimrt::llm::ipc::FrameDecodeState;
using kimrt::llm::ipc::FrameDecoder;
using kimrt::llm::ipc::Hello;
using kimrt::llm::ipc::HelloAck;
using kimrt::llm::ipc::decodeHandshakePayload;
using kimrt::llm::ipc::encodeFrame;
using kimrt::llm::ipc::encodeHelloAckPayload;
using kimrt::llm::ipc::encodeHelloPayload;

bool expect(
    bool condition,
    std::string_view message,
    int& failures) {
    if (condition) {
        return true;
    }

    ++failures;
    std::cerr << "[FAIL] " << message << '\n';
    return false;
}

std::string_view bytesView(
    std::vector<std::uint8_t> const& bytes,
    std::size_t offset,
    std::size_t count) {
    return {
        reinterpret_cast<char const*>(
            bytes.data() + offset),
        count,
    };
}

std::vector<std::uint8_t> makeFrame(
    std::string_view payload,
    int& failures) {
    auto encoded = encodeFrame(payload);

    expect(
        encoded.ok(),
        "frame encoding must succeed",
        failures);

    return std::move(encoded.bytes);
}

void testNetworkByteOrder(int& failures) {
    // 258 == 0x00000102
    std::string const payload(258, 'x');

    auto encoded = encodeFrame(payload);

    if (!expect(
            encoded.ok(),
            "network-order frame encoding must succeed",
            failures)) {
        return;
    }

    if (!expect(
            encoded.bytes.size() == 4U + payload.size(),
            "encoded frame size must include four-byte prefix",
            failures)) {
        return;
    }

    expect(
        encoded.bytes[0] == 0U,
        "length prefix byte 0 must be network order",
        failures);

    expect(
        encoded.bytes[1] == 0U,
        "length prefix byte 1 must be network order",
        failures);

    expect(
        encoded.bytes[2] == 1U,
        "length prefix byte 2 must be network order",
        failures);

    expect(
        encoded.bytes[3] == 2U,
        "length prefix byte 3 must be network order",
        failures);
}

void testHelloRoundTrip(int& failures) {
    auto hello_payload =
        encodeHelloPayload(Hello{});

    expect(
        hello_payload.ok(),
        "Hello encoding must succeed",
        failures);

    auto frame = makeFrame(
        hello_payload.payload,
        failures);

    FrameDecoder decoder;

    auto decoded_frame = decoder.feed(
        bytesView(frame, 0, frame.size()));

    expect(
        decoded_frame.state ==
            FrameDecodeState::FramesReady,
        "complete Hello frame must be ready",
        failures);

    expect(
        decoded_frame.payloads.size() == 1,
        "complete Hello frame must produce one payload",
        failures);

    if (decoded_frame.payloads.size() != 1) {
        return;
    }

    auto decoded_message =
        decodeHandshakePayload(
            decoded_frame.payloads[0]);

    expect(
        decoded_message.ok(),
        "Hello payload must decode",
        failures);

    if (!decoded_message.message.has_value()) {
        return;
    }

    auto const* hello = std::get_if<Hello>(
        &*decoded_message.message);

    expect(
        hello != nullptr,
        "decoded message must be Hello",
        failures);

    if (hello != nullptr) {
        expect(
            hello->protocol_version == 1,
            "Hello protocol version must be preserved",
            failures);
    }
}

void testSplitPrefixAndPayload(int& failures) {
    auto ack_payload =
        encodeHelloAckPayload(HelloAck{1, 42});

    expect(
        ack_payload.ok(),
        "HelloAck encoding must succeed",
        failures);

    auto frame = makeFrame(
        ack_payload.payload,
        failures);

    FrameDecoder prefix_decoder;

    for (std::size_t index = 0; index < 3; ++index) {
        auto result = prefix_decoder.feed(
            bytesView(frame, index, 1));

        expect(
            result.state ==
                FrameDecodeState::NeedMore,
            "split length prefix must wait",
            failures);
    }

    auto prefix_completion = prefix_decoder.feed(
        bytesView(frame, 3, frame.size() - 3));

    expect(
        prefix_completion.state ==
            FrameDecodeState::FramesReady,
        "remaining bytes must complete frame",
        failures);

    FrameDecoder payload_decoder;

    auto const split =
        4U + ack_payload.payload.size() / 2U;

    auto first_half = payload_decoder.feed(
        bytesView(frame, 0, split));

    expect(
        first_half.state ==
            FrameDecodeState::NeedMore,
        "partial payload must wait",
        failures);

    expect(
        payload_decoder.hasPartialFrame(),
        "decoder must remember partial payload",
        failures);

    auto second_half = payload_decoder.feed(
        bytesView(
            frame,
            split,
            frame.size() - split));

    expect(
        second_half.state ==
            FrameDecodeState::FramesReady,
        "second half must complete frame",
        failures);

    if (second_half.payloads.size() != 1) {
        expect(
            false,
            "split frame must produce one payload",
            failures);
        return;
    }

    auto decoded = decodeHandshakePayload(
        second_half.payloads[0]);

    expect(
        decoded.ok(),
        "HelloAck payload must decode",
        failures);

    if (!decoded.message.has_value()) {
        return;
    }

    auto const* ack = std::get_if<HelloAck>(
        &*decoded.message);

    expect(
        ack != nullptr,
        "decoded message must be HelloAck",
        failures);

    if (ack != nullptr) {
        expect(
            ack->worker_epoch == 42,
            "worker epoch must be preserved",
            failures);
    }
}

void testMultipleFramesInOneRead(int& failures) {
    auto hello_payload =
        encodeHelloPayload(Hello{});

    auto ack_payload =
        encodeHelloAckPayload(HelloAck{1, 99});

    expect(
        hello_payload.ok(),
        "Hello encoding must succeed",
        failures);

    expect(
        ack_payload.ok(),
        "HelloAck encoding must succeed",
        failures);

    auto hello_frame = makeFrame(
        hello_payload.payload,
        failures);

    auto ack_frame = makeFrame(
        ack_payload.payload,
        failures);

    hello_frame.insert(
        hello_frame.end(),
        ack_frame.begin(),
        ack_frame.end());

    FrameDecoder decoder;

    auto result = decoder.feed(
        bytesView(
            hello_frame,
            0,
            hello_frame.size()));

    expect(
        result.state ==
            FrameDecodeState::FramesReady,
        "coalesced frames must decode",
        failures);

    expect(
        result.payloads.size() == 2,
        "coalesced input must produce two payloads",
        failures);
}

void testInvalidFrameLengths(int& failures) {
    FrameDecoder zero_length_decoder;

    std::string const zero_length(4, '\0');

    auto zero_result =
        zero_length_decoder.feed(zero_length);

    expect(
        zero_result.state ==
            FrameDecodeState::Error,
        "zero-length frame must fail",
        failures);

    expect(
        zero_result.status.code ==
            StatusCode::InvalidInput,
        "zero-length frame must be InvalidInput",
        failures);

    FrameDecoder oversized_decoder(
        FrameCodecConfig{16});

    std::string const oversized_prefix{
        static_cast<char>(0),
        static_cast<char>(0),
        static_cast<char>(0),
        static_cast<char>(17),
    };

    auto oversized_result =
        oversized_decoder.feed(
            oversized_prefix);

    expect(
        oversized_result.state ==
            FrameDecodeState::Error,
        "oversized frame must fail",
        failures);

    expect(
        oversized_result.status.code ==
            StatusCode::ResourceExhausted,
        "oversized frame must be ResourceExhausted",
        failures);

    auto failed_again =
        oversized_decoder.feed("anything");

    oversized_decoder.reset();

    expect(
        !oversized_decoder.failed(),
        "reset must clear decoder failure state",
        failures);

    auto recovered_frame = makeFrame(
        "ok",
        failures);

    auto recovered = oversized_decoder.feed(
        bytesView(
            recovered_frame,
            0,
            recovered_frame.size()));

    expect(
        recovered.state ==
            FrameDecodeState::FramesReady,
        "decoder must accept frames after reset",
        failures);

    expect(
        recovered.payloads.size() == 1 &&
            recovered.payloads[0] == "ok",
        "decoder must preserve payload after reset",
        failures);

    expect(
        failed_again.state ==
            FrameDecodeState::Error,
        "decoder must remain failed until reset",
        failures);
}

void testInvalidHandshakePayloads(int& failures) {
    auto malformed =
        decodeHandshakePayload("{not-json}");

    expect(
        !malformed.ok(),
        "malformed JSON must fail",
        failures);

    auto unsupported = decodeHandshakePayload(
        R"({"type":"hello","protocol_version":2})");

    expect(
        !unsupported.ok(),
        "unsupported version must fail",
        failures);

    auto unknown_field = decodeHandshakePayload(
        R"({"type":"hello","protocol_version":1,"extra":true})");

    expect(
        !unknown_field.ok(),
        "unexpected v1 field must fail",
        failures);

    auto zero_epoch = decodeHandshakePayload(
        R"({"type":"hello_ack","protocol_version":1,"worker_epoch":0})");

    expect(
        !zero_epoch.ok(),
        "zero worker epoch must fail",
        failures);
}

void testEncoderValidation(int& failures) {
    auto empty = encodeFrame("");

    expect(
        !empty.ok(),
        "empty frame payload must be rejected",
        failures);

    auto too_large = encodeFrame(
        "12345",
        FrameCodecConfig{4});

    expect(
        !too_large.ok(),
        "oversized payload must be rejected",
        failures);

    auto invalid_ack =
        encodeHelloAckPayload(
            HelloAck{1, 0});

    expect(
        !invalid_ack.ok(),
        "zero worker epoch must not encode",
        failures);
}

} // namespace

int main() {
    int failures = 0;

    testNetworkByteOrder(failures);
    testHelloRoundTrip(failures);
    testSplitPrefixAndPayload(failures);
    testMultipleFramesInOneRead(failures);
    testInvalidFrameLengths(failures);
    testInvalidHandshakePayloads(failures);
    testEncoderValidation(failures);

    if (failures == 0) {
        std::cout
            << "[PASS] LLM IPC protocol contract\n";
        return 0;
    }

    std::cerr
        << failures
        << " IPC protocol assertion(s) failed\n";

    return 1;
}