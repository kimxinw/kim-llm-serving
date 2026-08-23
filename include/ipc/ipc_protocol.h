#pragma once

#include "common/status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kimrt::llm::ipc {

inline constexpr std::uint32_t kProtocolVersion{1};
inline constexpr std::uint32_t kDefaultMaxFramePayloadBytes{
    1024U * 1024U};

struct FrameCodecConfig {
    std::uint32_t max_payload_bytes{kDefaultMaxFramePayloadBytes};
};

struct FrameEncodeResult {
    Status status;
    std::vector<std::uint8_t> bytes;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

[[nodiscard]] FrameEncodeResult encodeFrame(
    std::string_view payload,
    FrameCodecConfig config = {});

enum class FrameDecodeState : std::uint8_t {
    NeedMore,
    FramesReady,
    Error,
};

struct FrameDecodeResult {
    FrameDecodeState state{FrameDecodeState::NeedMore};
    Status status;
    std::vector<std::string> payloads;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

class FrameDecoder final {
public:
    explicit FrameDecoder(FrameCodecConfig config = {});

    [[nodiscard]] FrameDecodeResult feed(std::string_view bytes);
    void reset() noexcept;

    [[nodiscard]] bool failed() const noexcept;
    [[nodiscard]] bool hasPartialFrame() const noexcept;

private:
    [[nodiscard]] FrameDecodeResult fail(
        StatusCode code,
        std::string message);

    FrameCodecConfig config_;
    std::array<std::uint8_t, 4> prefix_{};
    std::size_t prefix_size_{0};
    std::uint32_t expected_payload_size_{0};
    std::string payload_;
    bool failed_{false};
    Status failure_;
};

// Wire DTOs are deliberately independent from GenerationRequest and
// TensorRT-LLM types. The Runtime bridge performs the conversion later.
struct ModelManifest {
    std::string model_id;
    std::string revision;
    std::string tokenizer_fingerprint;
    std::string chat_template_fingerprint;
    std::string engine_fingerprint;
    std::int32_t eos_token_id{0};
    std::int32_t pad_token_id{0};
    std::uint32_t max_input_tokens{0};
    std::uint32_t max_output_tokens{0};
    std::uint32_t max_sequence_tokens{0};
    std::string precision;
    std::uint32_t max_batch_size{0};
};

struct WorkerLimits {
    std::uint32_t max_active_requests{0};
    std::uint64_t max_total_input_tokens{0};
    std::uint64_t max_reserved_output_tokens{0};
    std::uint32_t max_frame_payload_bytes{0};
    std::uint32_t max_session_egress_frames{0};
    std::uint64_t max_session_egress_bytes{0};
    std::uint32_t max_request_egress_frames{0};
    std::uint64_t max_request_egress_bytes{0};
};

struct SamplingParameters {
    double temperature{1.0};
    std::int32_t top_k{1};
    double top_p{1.0};
    std::uint64_t random_seed{0};
};

enum class FinishReason : std::uint8_t {
    Eos,
    Length,
    Stop,
    Cancelled,
    Timeout,
    Backpressure,
};

struct Usage {
    std::uint64_t prompt_tokens{0};
    std::uint64_t completion_tokens{0};
};

struct Hello {
    std::uint32_t protocol_version{kProtocolVersion};
    std::string model_id;
    std::string revision;
};

struct HelloAck {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    ModelManifest manifest;
    WorkerLimits limits;
};

struct Submit {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t request_id{0};
    std::int32_t priority{0};
    std::optional<std::uint64_t> timeout_ms;
    std::string trace_id;
    std::vector<std::int32_t> input_token_ids;
    std::uint32_t max_new_tokens{0};
    bool streaming{false};
    SamplingParameters sampling;
    std::optional<std::int32_t> end_id;
    std::optional<std::int32_t> pad_id;
    std::vector<std::vector<std::int32_t>> stop_sequences;
};

struct Accepted {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t request_id{0};
};

struct Rejected {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t request_id{0};
    Status status;
};

struct Cancel {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t request_id{0};
};

struct TokenDelta {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t request_id{0};
    std::uint64_t sequence_no{0};
    std::vector<std::int32_t> token_ids;
};

struct Terminal {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t request_id{0};
    Status status;
    std::optional<FinishReason> finish_reason;
    Usage usage;
};

struct Health {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t probe_id{0};
};

struct Stats {
    std::uint32_t protocol_version{kProtocolVersion};
    std::uint64_t worker_epoch{0};
    std::uint64_t probe_id{0};
    bool ready{false};
    Status status;
    std::uint64_t uptime_ms{0};
    std::uint64_t active_requests{0};
    std::uint64_t reserved_input_tokens{0};
    std::uint64_t reserved_output_tokens{0};
    std::uint64_t session_egress_frames{0};
    std::uint64_t session_egress_bytes{0};
    std::uint64_t session_egress_high_watermark_frames{0};
    std::uint64_t session_egress_high_watermark_bytes{0};
    std::uint64_t rejected_requests{0};
    std::uint64_t backpressure_requests{0};
    std::uint64_t cancelled_requests{0};
};

using Message = std::variant<
    Hello,
    HelloAck,
    Submit,
    Accepted,
    Rejected,
    Cancel,
    TokenDelta,
    Terminal,
    Health,
    Stats>;

struct PayloadEncodeResult {
    Status status;
    std::string payload;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

struct MessageDecodeResult {
    Status status;
    std::optional<Message> message;

    [[nodiscard]] bool ok() const noexcept;
    explicit operator bool() const noexcept;
};

[[nodiscard]] PayloadEncodeResult encodePayload(Message const& message);
[[nodiscard]] MessageDecodeResult decodePayload(std::string_view payload);

[[nodiscard]] std::string_view messageTypeName(
    Message const& message) noexcept;
[[nodiscard]] std::string_view statusCodeName(
    StatusCode code) noexcept;
[[nodiscard]] std::optional<StatusCode> parseStatusCode(
    std::string_view name) noexcept;
[[nodiscard]] std::string_view finishReasonName(
    FinishReason reason) noexcept;
[[nodiscard]] std::optional<FinishReason> parseFinishReason(
    std::string_view name) noexcept;

} // namespace kimrt::llm::ipc
