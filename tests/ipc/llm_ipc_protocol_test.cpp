#include "ipc/ipc_protocol.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace kimrt::llm::ipc;

bool expect(bool condition, std::string_view message, int& failures) {
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
        reinterpret_cast<char const*>(bytes.data() + offset),
        count,
    };
}

ModelManifest makeManifest() {
    return {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "main@abc123",
        "sha256:tokenizer",
        "sha256:chat-template",
        "sha256:engine",
        2,
        0,
        2016,
        128,
        2048,
        "fp16",
        8,
    };
}

WorkerLimits makeLimits() {
    return {
        8,
        4096,
        256,
        kDefaultMaxFramePayloadBytes,
        1024,
        4U * 1024U * 1024U,
        128,
        1024U * 1024U,
    };
}

Submit makeSubmit() {
    Submit message;
    message.worker_epoch = 42;
    message.request_id = 7;
    message.priority = 3;
    message.timeout_ms = 1500;
    message.trace_id = "trace-7";
    message.input_token_ids = {1, 2, 3};
    message.max_new_tokens = 32;
    message.streaming = true;
    message.sampling = {0.8, 40, 0.95, 1234};
    message.end_id = 2;
    message.pad_id = 0;
    message.stop_sequences = {{10, 11}, {12}};
    return message;
}

Stats makeStats() {
    Stats message;
    message.worker_epoch = 42;
    message.probe_id = 99;
    message.ready = true;
    message.status = kimrt::Status::success();
    message.uptime_ms = 5000;
    message.active_requests = 2;
    message.reserved_input_tokens = 128;
    message.reserved_output_tokens = 64;
    message.session_egress_frames = 3;
    message.session_egress_bytes = 1024;
    message.session_egress_high_watermark_frames = 7;
    message.session_egress_high_watermark_bytes = 4096;
    message.rejected_requests = 4;
    message.backpressure_requests = 1;
    message.cancelled_requests = 2;
    return message;
}

bool roundTrip(Message const& message, int& failures) {
    auto encoded = encodePayload(message);
    if (!expect(encoded.ok(), "message must encode", failures)) {
        return false;
    }
    auto decoded = decodePayload(encoded.payload);
    if (!decoded.ok()) {
        std::cerr << "decode failed for " << messageTypeName(message)
                  << ": " << decoded.status.message << '\n'
                  << encoded.payload << '\n';
    }
    if (!expect(decoded.ok(), "encoded message must decode", failures)) {
        return false;
    }
    return expect(
        messageTypeName(message) == messageTypeName(*decoded.message),
        "message type must survive round trip",
        failures);
}

void testAllMessageRoundTrips(int& failures) {
    std::vector<Message> const messages{
        Hello{1, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "main@abc123"},
        HelloAck{1, 42, makeManifest(), makeLimits()},
        makeSubmit(),
        Accepted{1, 42, 7},
        Rejected{
            1,
            42,
            8,
            kimrt::Status::error(
                kimrt::StatusCode::ResourceExhausted,
                "active request limit")},
        Cancel{1, 42, 7},
        TokenDelta{1, 42, 7, 0, {100, 101}},
        Terminal{
            1,
            42,
            7,
            kimrt::Status::success(),
            FinishReason::Length,
            Usage{3, 32}},
        Health{1, 42, 99},
        makeStats(),
    };

    for (auto const& message : messages) {
        roundTrip(message, failures);
    }
}

void testFieldPreservation(int& failures) {
    auto submit_encoded = encodePayload(Message{makeSubmit()});
    auto submit_decoded = decodePayload(submit_encoded.payload);
    if (!submit_decoded.ok()) {
        std::cerr << "Submit decode failed: "
                  << submit_decoded.status.message << '\n'
                  << submit_encoded.payload << '\n';
        return;
    }
    auto const* submit = std::get_if<Submit>(&*submit_decoded.message);
    expect(submit != nullptr, "Submit type must be preserved", failures);
    if (submit != nullptr) {
        expect(submit->request_id == 7, "Submit request id must survive", failures);
        expect(
            submit->timeout_ms == std::optional<std::uint64_t>{1500},
            "Submit timeout must survive",
            failures);
        expect(
            submit->input_token_ids == std::vector<std::int32_t>({1, 2, 3}),
            "Submit input tokens must survive",
            failures);
        expect(
            submit->stop_sequences.size() == 2,
            "Submit stop sequences must survive",
            failures);
        expect(
            submit->sampling.random_seed == 1234,
            "Submit sampling must survive",
            failures);
    }

    auto ack_encoded = encodePayload(
        Message{HelloAck{1, 42, makeManifest(), makeLimits()}});
    auto ack_decoded = decodePayload(ack_encoded.payload);
    auto const* ack = std::get_if<HelloAck>(&*ack_decoded.message);
    expect(ack != nullptr, "HelloAck type must be preserved", failures);
    if (ack != nullptr) {
        expect(
            ack->manifest.engine_fingerprint == "sha256:engine",
            "Manifest fingerprint must survive",
            failures);
        expect(
            ack->limits.max_active_requests == 8,
            "Worker limits must survive",
            failures);
    }

    auto terminal_encoded = encodePayload(Message{Terminal{
        1,
        42,
        7,
        kimrt::Status::error(kimrt::StatusCode::Timeout, "deadline"),
        FinishReason::Timeout,
        Usage{3, 4}}});
    auto terminal_decoded = decodePayload(terminal_encoded.payload);
    auto const* terminal = std::get_if<Terminal>(&*terminal_decoded.message);
    expect(terminal != nullptr, "Terminal type must be preserved", failures);
    if (terminal != nullptr) {
        expect(
            terminal->status.code == kimrt::StatusCode::Timeout,
            "Terminal status string mapping must survive",
            failures);
        expect(
            terminal->finish_reason == FinishReason::Timeout,
            "Terminal finish reason string mapping must survive",
            failures);
    }
}

void testStableNames(int& failures) {
    expect(
        statusCodeName(kimrt::StatusCode::ResourceExhausted) ==
            "resource_exhausted",
        "StatusCode must use stable snake-case string",
        failures);
    expect(
        parseStatusCode("resource_exhausted") ==
            kimrt::StatusCode::ResourceExhausted,
        "StatusCode string must parse",
        failures);
    expect(
        statusCodeName(kimrt::StatusCode::SloPredictedMiss) ==
            "slo_predicted_miss" &&
        parseStatusCode("slo_predicted_miss") ==
            kimrt::StatusCode::SloPredictedMiss,
        "SLO rejection must have a stable wire name",
        failures);
    expect(
        finishReasonName(FinishReason::Backpressure) == "backpressure",
        "FinishReason must use stable string",
        failures);
    expect(
        !parseFinishReason("7").has_value(),
        "numeric FinishReason representation must fail",
        failures);
}

void testStrictPayloadValidation(int& failures) {
    std::vector<std::string> const invalid_payloads{
        "{not-json}",
        R"([])",
        R"({"type":"unknown"})",
        R"({"type":"hello","protocol_version":1,"model_id":"m","revision":"r","extra":true})",
        R"({"type":"hello","protocol_version":2,"model_id":"m","revision":"r"})",
        R"({"type":"accepted","protocol_version":1,"worker_epoch":0,"request_id":1})",
        R"({"type":"accepted","protocol_version":1,"worker_epoch":1,"request_id":0})",
        R"({"type":"rejected","protocol_version":1,"worker_epoch":1,"request_id":1,"status":{"code":"ok","message":""}})",
        R"({"type":"rejected","protocol_version":1,"worker_epoch":1,"request_id":1,"status":{"code":"new_code","message":"x"}})",
        R"({"type":"token_delta","protocol_version":1,"worker_epoch":1,"request_id":1,"sequence_no":0,"token_ids":[]})",
        R"({"type":"terminal","protocol_version":1,"worker_epoch":1,"request_id":1,"status":{"code":"ok","message":""},"finish_reason":null,"usage":{"prompt_tokens":1,"completion_tokens":2}})",
        R"({"type":"terminal","protocol_version":1,"worker_epoch":1,"request_id":1,"status":{"code":"timeout","message":"x"},"finish_reason":"future_reason","usage":{"prompt_tokens":1,"completion_tokens":2}})",
        R"({"type":"health","protocol_version":1,"worker_epoch":1,"probe_id":0})",
    };

    for (auto const& payload : invalid_payloads) {
        expect(
            !decodePayload(payload).ok(),
            "strict decoder must reject invalid payload",
            failures);
    }
}

void testEncoderValidation(int& failures) {
    auto submit = makeSubmit();
    submit.timeout_ms = 0;
    expect(
        !encodePayload(Message{submit}).ok(),
        "zero timeout must not encode",
        failures);

    submit = makeSubmit();
    submit.input_token_ids.clear();
    expect(
        !encodePayload(Message{submit}).ok(),
        "empty input token list must not encode",
        failures);

    submit = makeSubmit();
    submit.sampling.top_p = 1.5;
    expect(
        !encodePayload(Message{submit}).ok(),
        "invalid top_p must not encode",
        failures);

    auto manifest = makeManifest();
    manifest.engine_fingerprint.clear();
    expect(
        !encodePayload(
             Message{HelloAck{1, 42, manifest, makeLimits()}})
             .ok(),
        "incomplete manifest must not encode",
        failures);

    auto stats = makeStats();
    stats.ready = false;
    expect(
        !encodePayload(Message{stats}).ok(),
        "inconsistent readiness must not encode",
        failures);
}

void testFrameCodec(int& failures) {
    std::string const payload(258, 'x');
    auto encoded = encodeFrame(payload);
    if (!expect(encoded.ok(), "frame encoding must succeed", failures)) {
        return;
    }
    expect(
        encoded.bytes.size() == 262,
        "frame must include four-byte prefix",
        failures);
    expect(
        encoded.bytes[0] == 0 && encoded.bytes[1] == 0 &&
            encoded.bytes[2] == 1 && encoded.bytes[3] == 2,
        "frame length must use network byte order",
        failures);

    FrameDecoder split_decoder;
    auto first = split_decoder.feed(bytesView(encoded.bytes, 0, 3));
    expect(
        first.state == FrameDecodeState::NeedMore,
        "split prefix must wait",
        failures);
    auto second = split_decoder.feed(
        bytesView(encoded.bytes, 3, encoded.bytes.size() - 3));
    expect(
        second.state == FrameDecodeState::FramesReady &&
            second.payloads.size() == 1 && second.payloads[0] == payload,
        "split frame must reassemble",
        failures);

    auto hello = encodePayload(
        Message{Hello{1, "model", "revision"}});
    auto health = encodePayload(Message{Health{1, 42, 9}});
    auto first_frame = encodeFrame(hello.payload).bytes;
    auto second_frame = encodeFrame(health.payload).bytes;
    first_frame.insert(
        first_frame.end(),
        second_frame.begin(),
        second_frame.end());
    FrameDecoder coalesced_decoder;
    auto coalesced = coalesced_decoder.feed(bytesView(
        first_frame,
        0,
        first_frame.size()));
    expect(
        coalesced.state == FrameDecodeState::FramesReady &&
            coalesced.payloads.size() == 2,
        "coalesced frames must decode independently",
        failures);

    std::string const zero_length(4, '\0');
    FrameDecoder zero_decoder;
    expect(
        zero_decoder.feed(zero_length).state == FrameDecodeState::Error,
        "zero-length frame must fail",
        failures);

    FrameDecoder oversized_decoder(FrameCodecConfig{16});
    std::string const oversized_prefix{
        static_cast<char>(0),
        static_cast<char>(0),
        static_cast<char>(0),
        static_cast<char>(17),
    };
    auto oversized = oversized_decoder.feed(oversized_prefix);
    expect(
        oversized.state == FrameDecodeState::Error &&
            oversized.status.code == kimrt::StatusCode::ResourceExhausted,
        "oversized frame must fail with ResourceExhausted",
        failures);
    oversized_decoder.reset();
    expect(
        !oversized_decoder.failed(),
        "reset must clear frame decoder failure",
        failures);
}

} // namespace

int main() {
    int failures = 0;
    testAllMessageRoundTrips(failures);
    testFieldPreservation(failures);
    testStableNames(failures);
    testStrictPayloadValidation(failures);
    testEncoderValidation(failures);
    testFrameCodec(failures);

    if (failures == 0) {
        std::cout << "[PASS] LLM IPC v1 protocol contract\n";
        return 0;
    }
    std::cerr << failures << " IPC protocol assertion(s) failed\n";
    return 1;
}
