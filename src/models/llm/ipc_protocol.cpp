#include "models/llm/ipc_protocol.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace kimrt::llm::ipc {
namespace {

using Json = nlohmann::json;

[[nodiscard]] Status invalid(std::string message) {
    return Status::error(StatusCode::InvalidInput, std::move(message));
}

[[nodiscard]] PayloadEncodeResult encodeError(std::string message) {
    return {invalid(std::move(message)), {}};
}

[[nodiscard]] MessageDecodeResult decodeError(std::string message) {
    return {invalid(std::move(message)), std::nullopt};
}

[[nodiscard]] bool exactFields(
    Json const& object,
    std::initializer_list<char const*> fields) {
    if (!object.is_object() || object.size() != fields.size()) {
        return false;
    }

    for (auto const* field : fields) {
        if (object.find(field) == object.end()) {
            return false;
        }
    }
    return true;
}

template <typename T>
[[nodiscard]] bool readUnsigned(
    Json const& object,
    char const* key,
    T& value) {
    static_assert(std::is_unsigned_v<T>);
    auto const iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_number_unsigned()) {
        return false;
    }

    auto const raw = iterator->get<std::uint64_t>();
    if (raw > static_cast<std::uint64_t>(
                  std::numeric_limits<T>::max())) {
        return false;
    }
    value = static_cast<T>(raw);
    return true;
}

[[nodiscard]] bool readSigned32(
    Json const& object,
    char const* key,
    std::int32_t& value) {
    auto const iterator = object.find(key);
    if (iterator == object.end()) {
        return false;
    }

    if (iterator->is_number_unsigned()) {
        auto const raw = iterator->get<std::uint64_t>();
        if (raw > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int32_t>::max())) {
            return false;
        }
        value = static_cast<std::int32_t>(raw);
        return true;
    }

    if (!iterator->is_number_integer()) {
        return false;
    }
    auto const raw = iterator->get<std::int64_t>();
    if (raw < std::numeric_limits<std::int32_t>::min() ||
        raw > std::numeric_limits<std::int32_t>::max()) {
        return false;
    }
    value = static_cast<std::int32_t>(raw);
    return true;
}

[[nodiscard]] bool readString(
    Json const& object,
    char const* key,
    std::string& value) {
    auto const iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_string()) {
        return false;
    }
    value = iterator->get<std::string>();
    return true;
}

[[nodiscard]] bool readBoolean(
    Json const& object,
    char const* key,
    bool& value) {
    auto const iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_boolean()) {
        return false;
    }
    value = iterator->get<bool>();
    return true;
}

[[nodiscard]] bool readFiniteDouble(
    Json const& object,
    char const* key,
    double& value) {
    auto const iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_number()) {
        return false;
    }
    value = iterator->get<double>();
    return std::isfinite(value);
}

[[nodiscard]] bool readOptionalUnsigned64(
    Json const& object,
    char const* key,
    std::optional<std::uint64_t>& value) {
    auto const iterator = object.find(key);
    if (iterator == object.end()) {
        return false;
    }
    if (iterator->is_null()) {
        value.reset();
        return true;
    }
    if (!iterator->is_number_unsigned()) {
        return false;
    }
    value = iterator->get<std::uint64_t>();
    return true;
}

[[nodiscard]] bool readOptionalSigned32(
    Json const& object,
    char const* key,
    std::optional<std::int32_t>& value) {
    auto const iterator = object.find(key);
    if (iterator == object.end()) {
        return false;
    }
    if (iterator->is_null()) {
        value.reset();
        return true;
    }

    Json wrapper{{"value", *iterator}};
    std::int32_t parsed = 0;
    if (!readSigned32(wrapper, "value", parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

[[nodiscard]] bool readTokenIds(
    Json const& value,
    std::vector<std::int32_t>& token_ids) {
    if (!value.is_array()) {
        return false;
    }

    token_ids.clear();
    token_ids.reserve(value.size());
    for (auto const& item : value) {
        Json wrapper{{"value", item}};
        std::int32_t token_id = 0;
        if (!readSigned32(wrapper, "value", token_id)) {
            return false;
        }
        token_ids.push_back(token_id);
    }
    return true;
}

[[nodiscard]] bool readTokenIds(
    Json const& object,
    char const* key,
    std::vector<std::int32_t>& token_ids) {
    auto const iterator = object.find(key);
    return iterator != object.end() && readTokenIds(*iterator, token_ids);
}

[[nodiscard]] bool readStopSequences(
    Json const& object,
    char const* key,
    std::vector<std::vector<std::int32_t>>& sequences) {
    auto const iterator = object.find(key);
    if (iterator == object.end() || !iterator->is_array()) {
        return false;
    }

    sequences.clear();
    sequences.reserve(iterator->size());
    for (auto const& item : *iterator) {
        std::vector<std::int32_t> sequence;
        if (!readTokenIds(item, sequence)) {
            return false;
        }
        sequences.push_back(std::move(sequence));
    }
    return true;
}

[[nodiscard]] Status validateProtocol(std::uint32_t version) {
    if (version != kProtocolVersion) {
        return invalid("unsupported protocol_version");
    }
    return Status::success();
}

[[nodiscard]] Status validateEpoch(std::uint64_t epoch) {
    if (epoch == 0) {
        return invalid("worker_epoch must be positive");
    }
    return Status::success();
}

[[nodiscard]] Status validateRequestId(std::uint64_t request_id) {
    if (request_id == 0) {
        return invalid("request_id must be positive");
    }
    return Status::success();
}

[[nodiscard]] Status validateTokenId(
    std::int32_t token_id,
    std::string_view field) {
    if (token_id < 0) {
        return invalid(std::string{field} + " must be non-negative");
    }
    return Status::success();
}

[[nodiscard]] Status validateTokenIds(
    std::vector<std::int32_t> const& token_ids,
    bool require_non_empty,
    std::string_view field) {
    if (require_non_empty && token_ids.empty()) {
        return invalid(std::string{field} + " must not be empty");
    }
    for (auto const token_id : token_ids) {
        auto status = validateTokenId(token_id, field);
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

[[nodiscard]] Status validate(ModelManifest const& manifest) {
    if (manifest.model_id.empty() || manifest.revision.empty() ||
        manifest.tokenizer_fingerprint.empty() ||
        manifest.chat_template_fingerprint.empty() ||
        manifest.engine_fingerprint.empty() || manifest.precision.empty()) {
        return invalid("ModelManifest string fields must not be empty");
    }
    if (manifest.eos_token_id < 0 || manifest.pad_token_id < 0) {
        return invalid("ModelManifest token ids must be non-negative");
    }
    if (manifest.max_input_tokens == 0 ||
        manifest.max_output_tokens == 0 ||
        manifest.max_sequence_tokens == 0 ||
        manifest.max_batch_size == 0) {
        return invalid("ModelManifest limits must be positive");
    }
    if (manifest.max_input_tokens > manifest.max_sequence_tokens ||
        manifest.max_output_tokens > manifest.max_sequence_tokens) {
        return invalid("ModelManifest token limits exceed max_sequence_tokens");
    }
    return Status::success();
}

[[nodiscard]] Status validate(WorkerLimits const& limits) {
    if (limits.max_active_requests == 0 ||
        limits.max_total_input_tokens == 0 ||
        limits.max_reserved_output_tokens == 0 ||
        limits.max_frame_payload_bytes == 0 ||
        limits.max_session_egress_frames == 0 ||
        limits.max_session_egress_bytes == 0 ||
        limits.max_request_egress_frames == 0 ||
        limits.max_request_egress_bytes == 0) {
        return invalid("WorkerLimits values must be positive");
    }
    if (limits.max_request_egress_frames >
            limits.max_session_egress_frames ||
        limits.max_request_egress_bytes >
            limits.max_session_egress_bytes ||
        limits.max_frame_payload_bytes >
            limits.max_request_egress_bytes) {
        return invalid("WorkerLimits hierarchy is inconsistent");
    }
    return Status::success();
}

[[nodiscard]] Status validate(SamplingParameters const& sampling) {
    if (!std::isfinite(sampling.temperature) ||
        sampling.temperature <= 0.0) {
        return invalid("temperature must be finite and positive");
    }
    if (sampling.top_k < 0) {
        return invalid("top_k must be non-negative");
    }
    if (!std::isfinite(sampling.top_p) || sampling.top_p <= 0.0 ||
        sampling.top_p > 1.0) {
        return invalid("top_p must be in (0, 1]");
    }
    return Status::success();
}

[[nodiscard]] Status validateStatusValue(Status const& status) {
    if (statusCodeName(status.code).empty()) {
        return invalid("unknown StatusCode value");
    }
    return Status::success();
}

[[nodiscard]] Status validate(Hello const& message) {
    auto status = validateProtocol(message.protocol_version);
    if (!status.ok()) {
        return status;
    }
    if (message.model_id.empty() || message.revision.empty()) {
        return invalid("Hello model_id and revision must not be empty");
    }
    return Status::success();
}

[[nodiscard]] Status validate(HelloAck const& message) {
    auto status = validateProtocol(message.protocol_version);
    if (!status.ok()) {
        return status;
    }
    status = validateEpoch(message.worker_epoch);
    if (!status.ok()) {
        return status;
    }
    status = validate(message.manifest);
    if (!status.ok()) {
        return status;
    }
    return validate(message.limits);
}

[[nodiscard]] Status validate(Submit const& message) {
    auto status = validateProtocol(message.protocol_version);
    if (!status.ok()) {
        return status;
    }
    status = validateEpoch(message.worker_epoch);
    if (!status.ok()) {
        return status;
    }
    status = validateRequestId(message.request_id);
    if (!status.ok()) {
        return status;
    }
    if (message.timeout_ms.has_value() && *message.timeout_ms == 0) {
        return invalid("timeout_ms must be positive when present");
    }
    status = validateTokenIds(
        message.input_token_ids, true, "input_token_ids");
    if (!status.ok()) {
        return status;
    }
    if (message.max_new_tokens == 0) {
        return invalid("max_new_tokens must be positive");
    }
    status = validate(message.sampling);
    if (!status.ok()) {
        return status;
    }
    if (message.end_id.has_value()) {
        status = validateTokenId(*message.end_id, "end_id");
        if (!status.ok()) {
            return status;
        }
    }
    if (message.pad_id.has_value()) {
        status = validateTokenId(*message.pad_id, "pad_id");
        if (!status.ok()) {
            return status;
        }
    }
    for (auto const& sequence : message.stop_sequences) {
        status = validateTokenIds(sequence, true, "stop_sequences");
        if (!status.ok()) {
            return status;
        }
    }
    return Status::success();
}

template <typename T>
[[nodiscard]] Status validateRequestEnvelope(T const& message) {
    auto status = validateProtocol(message.protocol_version);
    if (!status.ok()) {
        return status;
    }
    status = validateEpoch(message.worker_epoch);
    if (!status.ok()) {
        return status;
    }
    return validateRequestId(message.request_id);
}

[[nodiscard]] Status validate(Accepted const& message) {
    return validateRequestEnvelope(message);
}

[[nodiscard]] Status validate(Rejected const& message) {
    auto status = validateRequestEnvelope(message);
    if (!status.ok()) {
        return status;
    }
    status = validateStatusValue(message.status);
    if (!status.ok()) {
        return status;
    }
    if (message.status.ok()) {
        return invalid("Rejected status must not be ok");
    }
    return Status::success();
}

[[nodiscard]] Status validate(Cancel const& message) {
    return validateRequestEnvelope(message);
}

[[nodiscard]] Status validate(TokenDelta const& message) {
    auto status = validateRequestEnvelope(message);
    if (!status.ok()) {
        return status;
    }
    return validateTokenIds(message.token_ids, true, "token_ids");
}

[[nodiscard]] Status validate(Terminal const& message) {
    auto status = validateRequestEnvelope(message);
    if (!status.ok()) {
        return status;
    }
    status = validateStatusValue(message.status);
    if (!status.ok()) {
        return status;
    }
    if (message.status.ok() && !message.finish_reason.has_value()) {
        return invalid("successful Terminal requires finish_reason");
    }
    if (message.finish_reason.has_value() &&
        finishReasonName(*message.finish_reason).empty()) {
        return invalid("unknown FinishReason value");
    }
    return Status::success();
}

[[nodiscard]] Status validate(Health const& message) {
    auto status = validateProtocol(message.protocol_version);
    if (!status.ok()) {
        return status;
    }
    status = validateEpoch(message.worker_epoch);
    if (!status.ok()) {
        return status;
    }
    if (message.probe_id == 0) {
        return invalid("probe_id must be positive");
    }
    return Status::success();
}

[[nodiscard]] Status validate(Stats const& message) {
    auto status = validateProtocol(message.protocol_version);
    if (!status.ok()) {
        return status;
    }
    status = validateEpoch(message.worker_epoch);
    if (!status.ok()) {
        return status;
    }
    if (message.probe_id == 0) {
        return invalid("probe_id must be positive");
    }
    status = validateStatusValue(message.status);
    if (!status.ok()) {
        return status;
    }
    if (message.ready != message.status.ok()) {
        return invalid("Stats ready and status are inconsistent");
    }
    if (message.session_egress_high_watermark_frames <
            message.session_egress_frames ||
        message.session_egress_high_watermark_bytes <
            message.session_egress_bytes) {
        return invalid("Stats egress high watermark is below current value");
    }
    return Status::success();
}

[[nodiscard]] Json optionalInteger(
    std::optional<std::uint64_t> const& value) {
    return value.has_value() ? Json(*value) : Json(nullptr);
}

[[nodiscard]] Json optionalInteger(
    std::optional<std::int32_t> const& value) {
    return value.has_value() ? Json(*value) : Json(nullptr);
}

[[nodiscard]] Json statusJson(Status const& status) {
    return {
        {"code", statusCodeName(status.code)},
        {"message", status.message},
    };
}

[[nodiscard]] Json manifestJson(ModelManifest const& manifest) {
    return {
        {"model_id", manifest.model_id},
        {"revision", manifest.revision},
        {"tokenizer_fingerprint", manifest.tokenizer_fingerprint},
        {"chat_template_fingerprint", manifest.chat_template_fingerprint},
        {"engine_fingerprint", manifest.engine_fingerprint},
        {"eos_token_id", manifest.eos_token_id},
        {"pad_token_id", manifest.pad_token_id},
        {"max_input_tokens", manifest.max_input_tokens},
        {"max_output_tokens", manifest.max_output_tokens},
        {"max_sequence_tokens", manifest.max_sequence_tokens},
        {"precision", manifest.precision},
        {"max_batch_size", manifest.max_batch_size},
    };
}

[[nodiscard]] Json limitsJson(WorkerLimits const& limits) {
    return {
        {"max_active_requests", limits.max_active_requests},
        {"max_total_input_tokens", limits.max_total_input_tokens},
        {"max_reserved_output_tokens", limits.max_reserved_output_tokens},
        {"max_frame_payload_bytes", limits.max_frame_payload_bytes},
        {"max_session_egress_frames", limits.max_session_egress_frames},
        {"max_session_egress_bytes", limits.max_session_egress_bytes},
        {"max_request_egress_frames", limits.max_request_egress_frames},
        {"max_request_egress_bytes", limits.max_request_egress_bytes},
    };
}

[[nodiscard]] Json samplingJson(SamplingParameters const& sampling) {
    return {
        {"temperature", sampling.temperature},
        {"top_k", sampling.top_k},
        {"top_p", sampling.top_p},
        {"random_seed", sampling.random_seed},
    };
}

[[nodiscard]] Json usageJson(Usage const& usage) {
    return {
        {"prompt_tokens", usage.prompt_tokens},
        {"completion_tokens", usage.completion_tokens},
    };
}

[[nodiscard]] Json messageJson(Hello const& message) {
    return {
        {"type", "hello"},
        {"protocol_version", message.protocol_version},
        {"model_id", message.model_id},
        {"revision", message.revision},
    };
}

[[nodiscard]] Json messageJson(HelloAck const& message) {
    return {
        {"type", "hello_ack"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"manifest", manifestJson(message.manifest)},
        {"limits", limitsJson(message.limits)},
    };
}

[[nodiscard]] Json messageJson(Submit const& message) {
    return {
        {"type", "submit"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"request_id", message.request_id},
        {"priority", message.priority},
        {"timeout_ms", optionalInteger(message.timeout_ms)},
        {"trace_id", message.trace_id},
        {"input_token_ids", message.input_token_ids},
        {"max_new_tokens", message.max_new_tokens},
        {"streaming", message.streaming},
        {"sampling", samplingJson(message.sampling)},
        {"end_id", optionalInteger(message.end_id)},
        {"pad_id", optionalInteger(message.pad_id)},
        {"stop_sequences", message.stop_sequences},
    };
}

[[nodiscard]] Json messageJson(Accepted const& message) {
    return {
        {"type", "accepted"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"request_id", message.request_id},
    };
}

[[nodiscard]] Json messageJson(Rejected const& message) {
    return {
        {"type", "rejected"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"request_id", message.request_id},
        {"status", statusJson(message.status)},
    };
}

[[nodiscard]] Json messageJson(Cancel const& message) {
    return {
        {"type", "cancel"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"request_id", message.request_id},
    };
}

[[nodiscard]] Json messageJson(TokenDelta const& message) {
    return {
        {"type", "token_delta"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"request_id", message.request_id},
        {"sequence_no", message.sequence_no},
        {"token_ids", message.token_ids},
    };
}

[[nodiscard]] Json messageJson(Terminal const& message) {
    Json finish_reason = nullptr;
    if (message.finish_reason.has_value()) {
        finish_reason = finishReasonName(*message.finish_reason);
    }
    return {
        {"type", "terminal"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"request_id", message.request_id},
        {"status", statusJson(message.status)},
        {"finish_reason", std::move(finish_reason)},
        {"usage", usageJson(message.usage)},
    };
}

[[nodiscard]] Json messageJson(Health const& message) {
    return {
        {"type", "health"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"probe_id", message.probe_id},
    };
}

[[nodiscard]] Json messageJson(Stats const& message) {
    return {
        {"type", "stats"},
        {"protocol_version", message.protocol_version},
        {"worker_epoch", message.worker_epoch},
        {"probe_id", message.probe_id},
        {"ready", message.ready},
        {"status", statusJson(message.status)},
        {"uptime_ms", message.uptime_ms},
        {"active_requests", message.active_requests},
        {"reserved_input_tokens", message.reserved_input_tokens},
        {"reserved_output_tokens", message.reserved_output_tokens},
        {"session_egress_frames", message.session_egress_frames},
        {"session_egress_bytes", message.session_egress_bytes},
        {"session_egress_high_watermark_frames",
         message.session_egress_high_watermark_frames},
        {"session_egress_high_watermark_bytes",
         message.session_egress_high_watermark_bytes},
        {"rejected_requests", message.rejected_requests},
        {"backpressure_requests", message.backpressure_requests},
        {"cancelled_requests", message.cancelled_requests},
    };
}

[[nodiscard]] bool readStatusJson(Json const& object, Status& status) {
    if (!exactFields(object, {"code", "message"})) {
        return false;
    }
    std::string code;
    if (!readString(object, "code", code) ||
        !readString(object, "message", status.message)) {
        return false;
    }
    auto const parsed = parseStatusCode(code);
    if (!parsed.has_value()) {
        return false;
    }
    status.code = *parsed;
    return true;
}

[[nodiscard]] bool readStatusJson(
    Json const& object,
    char const* key,
    Status& status) {
    auto const iterator = object.find(key);
    return iterator != object.end() && readStatusJson(*iterator, status);
}

[[nodiscard]] bool readManifestJson(
    Json const& object,
    ModelManifest& manifest) {
    if (!exactFields(
            object,
            {"model_id", "revision", "tokenizer_fingerprint",
             "chat_template_fingerprint", "engine_fingerprint",
             "eos_token_id", "pad_token_id", "max_input_tokens",
             "max_output_tokens", "max_sequence_tokens", "precision",
             "max_batch_size"})) {
        return false;
    }
    return readString(object, "model_id", manifest.model_id) &&
        readString(object, "revision", manifest.revision) &&
        readString(
            object,
            "tokenizer_fingerprint",
            manifest.tokenizer_fingerprint) &&
        readString(
            object,
            "chat_template_fingerprint",
            manifest.chat_template_fingerprint) &&
        readString(
            object,
            "engine_fingerprint",
            manifest.engine_fingerprint) &&
        readSigned32(object, "eos_token_id", manifest.eos_token_id) &&
        readSigned32(object, "pad_token_id", manifest.pad_token_id) &&
        readUnsigned(object, "max_input_tokens", manifest.max_input_tokens) &&
        readUnsigned(object, "max_output_tokens", manifest.max_output_tokens) &&
        readUnsigned(
            object,
            "max_sequence_tokens",
            manifest.max_sequence_tokens) &&
        readString(object, "precision", manifest.precision) &&
        readUnsigned(object, "max_batch_size", manifest.max_batch_size);
}

[[nodiscard]] bool readManifestJson(
    Json const& object,
    char const* key,
    ModelManifest& manifest) {
    auto const iterator = object.find(key);
    return iterator != object.end() &&
        readManifestJson(*iterator, manifest);
}

[[nodiscard]] bool readLimitsJson(
    Json const& object,
    WorkerLimits& limits) {
    if (!exactFields(
            object,
            {"max_active_requests", "max_total_input_tokens",
             "max_reserved_output_tokens", "max_frame_payload_bytes",
             "max_session_egress_frames", "max_session_egress_bytes",
             "max_request_egress_frames", "max_request_egress_bytes"})) {
        return false;
    }
    return readUnsigned(
               object,
               "max_active_requests",
               limits.max_active_requests) &&
        readUnsigned(
            object,
            "max_total_input_tokens",
            limits.max_total_input_tokens) &&
        readUnsigned(
            object,
            "max_reserved_output_tokens",
            limits.max_reserved_output_tokens) &&
        readUnsigned(
            object,
            "max_frame_payload_bytes",
            limits.max_frame_payload_bytes) &&
        readUnsigned(
            object,
            "max_session_egress_frames",
            limits.max_session_egress_frames) &&
        readUnsigned(
            object,
            "max_session_egress_bytes",
            limits.max_session_egress_bytes) &&
        readUnsigned(
            object,
            "max_request_egress_frames",
            limits.max_request_egress_frames) &&
        readUnsigned(
            object,
            "max_request_egress_bytes",
            limits.max_request_egress_bytes);
}

[[nodiscard]] bool readLimitsJson(
    Json const& object,
    char const* key,
    WorkerLimits& limits) {
    auto const iterator = object.find(key);
    return iterator != object.end() && readLimitsJson(*iterator, limits);
}

[[nodiscard]] bool readSamplingJson(
    Json const& object,
    SamplingParameters& sampling) {
    if (!exactFields(
            object,
            {"temperature", "top_k", "top_p", "random_seed"})) {
        return false;
    }
    return readFiniteDouble(object, "temperature", sampling.temperature) &&
        readSigned32(object, "top_k", sampling.top_k) &&
        readFiniteDouble(object, "top_p", sampling.top_p) &&
        readUnsigned(object, "random_seed", sampling.random_seed);
}

[[nodiscard]] bool readSamplingJson(
    Json const& object,
    char const* key,
    SamplingParameters& sampling) {
    auto const iterator = object.find(key);
    return iterator != object.end() &&
        readSamplingJson(*iterator, sampling);
}

[[nodiscard]] bool readUsageJson(Json const& object, Usage& usage) {
    if (!exactFields(object, {"prompt_tokens", "completion_tokens"})) {
        return false;
    }
    return readUnsigned(object, "prompt_tokens", usage.prompt_tokens) &&
        readUnsigned(
            object,
            "completion_tokens",
            usage.completion_tokens);
}

[[nodiscard]] bool readUsageJson(
    Json const& object,
    char const* key,
    Usage& usage) {
    auto const iterator = object.find(key);
    return iterator != object.end() && readUsageJson(*iterator, usage);
}

template <typename T>
[[nodiscard]] MessageDecodeResult decoded(T message) {
    auto status = validate(message);
    if (!status.ok()) {
        return {std::move(status), std::nullopt};
    }
    return {Status::success(), Message{std::move(message)}};
}

[[nodiscard]] bool readRequestEnvelope(
    Json const& object,
    std::uint32_t& protocol_version,
    std::uint64_t& worker_epoch,
    std::uint64_t& request_id) {
    return readUnsigned(object, "protocol_version", protocol_version) &&
        readUnsigned(object, "worker_epoch", worker_epoch) &&
        readUnsigned(object, "request_id", request_id);
}

[[nodiscard]] MessageDecodeResult decodeHello(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "model_id", "revision"})) {
        return decodeError("Hello contains missing or unexpected fields");
    }
    Hello message;
    if (!readUnsigned(
            object,
            "protocol_version",
            message.protocol_version) ||
        !readString(object, "model_id", message.model_id) ||
        !readString(object, "revision", message.revision)) {
        return decodeError("Hello contains invalid field types");
    }
    return decoded(std::move(message));
}

[[nodiscard]] MessageDecodeResult decodeHelloAck(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "manifest",
             "limits"})) {
        return decodeError("HelloAck contains missing or unexpected fields");
    }
    HelloAck message;
    if (!readUnsigned(
            object,
            "protocol_version",
            message.protocol_version) ||
        !readUnsigned(object, "worker_epoch", message.worker_epoch) ||
        !readManifestJson(object, "manifest", message.manifest) ||
        !readLimitsJson(object, "limits", message.limits)) {
        return decodeError("HelloAck contains invalid fields");
    }
    return decoded(std::move(message));
}

[[nodiscard]] MessageDecodeResult decodeSubmit(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "request_id",
             "priority", "timeout_ms", "trace_id", "input_token_ids",
             "max_new_tokens", "streaming", "sampling", "end_id",
             "pad_id", "stop_sequences"})) {
        return decodeError("Submit contains missing or unexpected fields");
    }
    Submit message;
    if (!readRequestEnvelope(
            object,
            message.protocol_version,
            message.worker_epoch,
            message.request_id) ||
        !readSigned32(object, "priority", message.priority) ||
        !readOptionalUnsigned64(object, "timeout_ms", message.timeout_ms) ||
        !readString(object, "trace_id", message.trace_id) ||
        !readTokenIds(object, "input_token_ids", message.input_token_ids) ||
        !readUnsigned(object, "max_new_tokens", message.max_new_tokens) ||
        !readBoolean(object, "streaming", message.streaming) ||
        !readSamplingJson(object, "sampling", message.sampling) ||
        !readOptionalSigned32(object, "end_id", message.end_id) ||
        !readOptionalSigned32(object, "pad_id", message.pad_id) ||
        !readStopSequences(
            object,
            "stop_sequences",
            message.stop_sequences)) {
        return decodeError("Submit contains invalid fields");
    }
    return decoded(std::move(message));
}

[[nodiscard]] MessageDecodeResult decodeAccepted(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "request_id"})) {
        return decodeError("Accepted contains missing or unexpected fields");
    }
    Accepted message;
    if (!readRequestEnvelope(
            object,
            message.protocol_version,
            message.worker_epoch,
            message.request_id)) {
        return decodeError("Accepted contains invalid fields");
    }
    return decoded(message);
}

[[nodiscard]] MessageDecodeResult decodeRejected(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "request_id",
             "status"})) {
        return decodeError("Rejected contains missing or unexpected fields");
    }
    Rejected message;
    if (!readRequestEnvelope(
            object,
            message.protocol_version,
            message.worker_epoch,
            message.request_id) ||
        !readStatusJson(object, "status", message.status)) {
        return decodeError("Rejected contains invalid fields");
    }
    return decoded(std::move(message));
}

[[nodiscard]] MessageDecodeResult decodeCancel(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "request_id"})) {
        return decodeError("Cancel contains missing or unexpected fields");
    }
    Cancel message;
    if (!readRequestEnvelope(
            object,
            message.protocol_version,
            message.worker_epoch,
            message.request_id)) {
        return decodeError("Cancel contains invalid fields");
    }
    return decoded(message);
}

[[nodiscard]] MessageDecodeResult decodeTokenDelta(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "request_id",
             "sequence_no", "token_ids"})) {
        return decodeError("TokenDelta contains missing or unexpected fields");
    }
    TokenDelta message;
    if (!readRequestEnvelope(
            object,
            message.protocol_version,
            message.worker_epoch,
            message.request_id) ||
        !readUnsigned(object, "sequence_no", message.sequence_no) ||
        !readTokenIds(object, "token_ids", message.token_ids)) {
        return decodeError("TokenDelta contains invalid fields");
    }
    return decoded(std::move(message));
}

[[nodiscard]] MessageDecodeResult decodeTerminal(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "request_id",
             "status", "finish_reason", "usage"})) {
        return decodeError("Terminal contains missing or unexpected fields");
    }
    Terminal message;
    if (!readRequestEnvelope(
            object,
            message.protocol_version,
            message.worker_epoch,
            message.request_id) ||
        !readStatusJson(object, "status", message.status) ||
        !readUsageJson(object, "usage", message.usage)) {
        return decodeError("Terminal contains invalid fields");
    }

    auto const finish_iterator = object.find("finish_reason");
    if (finish_iterator == object.end()) {
        return decodeError("Terminal finish_reason is missing");
    }
    if (finish_iterator->is_null()) {
        message.finish_reason.reset();
    } else if (finish_iterator->is_string()) {
        auto const parsed = parseFinishReason(
            finish_iterator->get<std::string>());
        if (!parsed.has_value()) {
            return decodeError("Terminal finish_reason is unknown");
        }
        message.finish_reason = *parsed;
    } else {
        return decodeError("Terminal finish_reason has invalid type");
    }
    return decoded(std::move(message));
}

[[nodiscard]] MessageDecodeResult decodeHealth(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "probe_id"})) {
        return decodeError("Health contains missing or unexpected fields");
    }
    Health message;
    if (!readUnsigned(
            object,
            "protocol_version",
            message.protocol_version) ||
        !readUnsigned(object, "worker_epoch", message.worker_epoch) ||
        !readUnsigned(object, "probe_id", message.probe_id)) {
        return decodeError("Health contains invalid fields");
    }
    return decoded(message);
}

[[nodiscard]] MessageDecodeResult decodeStats(Json const& object) {
    if (!exactFields(
            object,
            {"type", "protocol_version", "worker_epoch", "probe_id",
             "ready", "status", "uptime_ms", "active_requests",
             "reserved_input_tokens", "reserved_output_tokens",
             "session_egress_frames", "session_egress_bytes",
             "session_egress_high_watermark_frames",
             "session_egress_high_watermark_bytes", "rejected_requests",
             "backpressure_requests", "cancelled_requests"})) {
        return decodeError("Stats contains missing or unexpected fields");
    }
    Stats message;
    if (!readUnsigned(
            object,
            "protocol_version",
            message.protocol_version) ||
        !readUnsigned(object, "worker_epoch", message.worker_epoch) ||
        !readUnsigned(object, "probe_id", message.probe_id) ||
        !readBoolean(object, "ready", message.ready) ||
        !readStatusJson(object, "status", message.status) ||
        !readUnsigned(object, "uptime_ms", message.uptime_ms) ||
        !readUnsigned(object, "active_requests", message.active_requests) ||
        !readUnsigned(
            object,
            "reserved_input_tokens",
            message.reserved_input_tokens) ||
        !readUnsigned(
            object,
            "reserved_output_tokens",
            message.reserved_output_tokens) ||
        !readUnsigned(
            object,
            "session_egress_frames",
            message.session_egress_frames) ||
        !readUnsigned(
            object,
            "session_egress_bytes",
            message.session_egress_bytes) ||
        !readUnsigned(
            object,
            "session_egress_high_watermark_frames",
            message.session_egress_high_watermark_frames) ||
        !readUnsigned(
            object,
            "session_egress_high_watermark_bytes",
            message.session_egress_high_watermark_bytes) ||
        !readUnsigned(
            object,
            "rejected_requests",
            message.rejected_requests) ||
        !readUnsigned(
            object,
            "backpressure_requests",
            message.backpressure_requests) ||
        !readUnsigned(
            object,
            "cancelled_requests",
            message.cancelled_requests)) {
        return decodeError("Stats contains invalid fields");
    }
    return decoded(std::move(message));
}

} // namespace

bool FrameEncodeResult::ok() const noexcept {
    return status.ok();
}

FrameEncodeResult::operator bool() const noexcept {
    return ok();
}

FrameEncodeResult encodeFrame(
    std::string_view payload,
    FrameCodecConfig config) {
    if (config.max_payload_bytes == 0) {
        return {invalid("max_payload_bytes must be positive"), {}};
    }
    if (payload.empty()) {
        return {invalid("frame payload must not be empty"), {}};
    }
    if (payload.size() > config.max_payload_bytes ||
        payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        return {
            Status::error(
                StatusCode::ResourceExhausted,
                "frame payload exceeds max_payload_bytes"),
            {},
        };
    }

    auto const payload_size = static_cast<std::uint32_t>(payload.size());
    std::vector<std::uint8_t> frame;
    frame.reserve(4U + payload.size());
    frame.push_back(static_cast<std::uint8_t>((payload_size >> 24U) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>((payload_size >> 16U) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>((payload_size >> 8U) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>(payload_size & 0xFFU));
    auto const* begin =
        reinterpret_cast<std::uint8_t const*>(payload.data());
    frame.insert(frame.end(), begin, begin + payload.size());
    return {Status::success(), std::move(frame)};
}

bool FrameDecodeResult::ok() const noexcept {
    return state != FrameDecodeState::Error && status.ok();
}

FrameDecodeResult::operator bool() const noexcept {
    return ok();
}

FrameDecoder::FrameDecoder(FrameCodecConfig config) : config_(config) {
    if (config_.max_payload_bytes == 0) {
        throw std::invalid_argument("max_payload_bytes must be positive");
    }
}

FrameDecodeResult FrameDecoder::feed(std::string_view bytes) {
    if (failed_) {
        return {FrameDecodeState::Error, failure_, {}};
    }

    std::vector<std::string> completed_payloads;
    auto const* data =
        reinterpret_cast<std::uint8_t const*>(bytes.data());
    std::size_t position = 0;

    while (position < bytes.size()) {
        while (prefix_size_ < prefix_.size() && position < bytes.size()) {
            prefix_[prefix_size_++] = data[position++];
        }
        if (prefix_size_ < prefix_.size()) {
            break;
        }
        if (expected_payload_size_ == 0) {
            expected_payload_size_ =
                (static_cast<std::uint32_t>(prefix_[0]) << 24U) |
                (static_cast<std::uint32_t>(prefix_[1]) << 16U) |
                (static_cast<std::uint32_t>(prefix_[2]) << 8U) |
                static_cast<std::uint32_t>(prefix_[3]);
            if (expected_payload_size_ == 0) {
                return fail(
                    StatusCode::InvalidInput,
                    "frame payload length must be positive");
            }
            if (expected_payload_size_ > config_.max_payload_bytes) {
                return fail(
                    StatusCode::ResourceExhausted,
                    "frame payload length exceeds max_payload_bytes");
            }
            payload_.clear();
            payload_.reserve(expected_payload_size_);
        }

        auto const remaining =
            static_cast<std::size_t>(expected_payload_size_) - payload_.size();
        auto const to_copy = std::min(remaining, bytes.size() - position);
        payload_.append(
            reinterpret_cast<char const*>(data + position),
            to_copy);
        position += to_copy;

        if (payload_.size() != expected_payload_size_) {
            break;
        }
        completed_payloads.push_back(std::move(payload_));
        payload_.clear();
        prefix_size_ = 0;
        expected_payload_size_ = 0;
    }

    auto const state = completed_payloads.empty()
        ? FrameDecodeState::NeedMore
        : FrameDecodeState::FramesReady;
    return {state, Status::success(), std::move(completed_payloads)};
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
    return prefix_size_ != 0 || expected_payload_size_ != 0;
}

FrameDecodeResult FrameDecoder::fail(
    StatusCode code,
    std::string message) {
    failed_ = true;
    failure_ = Status::error(code, std::move(message));
    prefix_.fill(0);
    prefix_size_ = 0;
    expected_payload_size_ = 0;
    payload_.clear();
    return {FrameDecodeState::Error, failure_, {}};
}

bool PayloadEncodeResult::ok() const noexcept {
    return status.ok();
}

PayloadEncodeResult::operator bool() const noexcept {
    return ok();
}

bool MessageDecodeResult::ok() const noexcept {
    return status.ok() && message.has_value();
}

MessageDecodeResult::operator bool() const noexcept {
    return ok();
}

PayloadEncodeResult encodePayload(Message const& message) {
    auto status = std::visit(
        [](auto const& typed_message) { return validate(typed_message); },
        message);
    if (!status.ok()) {
        return {std::move(status), {}};
    }

    try {
        auto const object = std::visit(
            [](auto const& typed_message) {
                return messageJson(typed_message);
            },
            message);
        return {Status::success(), object.dump()};
    } catch (Json::exception const& exception) {
        return encodeError(
            std::string{"failed to encode JSON payload: "} +
            exception.what());
    }
}

MessageDecodeResult decodePayload(std::string_view payload) {
    Json object;
    try {
        object = Json::parse(payload.begin(), payload.end());
    } catch (Json::exception const& exception) {
        return decodeError(
            std::string{"invalid JSON payload: "} + exception.what());
    }

    if (!object.is_object()) {
        return decodeError("payload must be a JSON object");
    }
    std::string type;
    if (!readString(object, "type", type)) {
        return decodeError("message type must be a string");
    }
    if (type == "hello") {
        return decodeHello(object);
    }
    if (type == "hello_ack") {
        return decodeHelloAck(object);
    }
    if (type == "submit") {
        return decodeSubmit(object);
    }
    if (type == "accepted") {
        return decodeAccepted(object);
    }
    if (type == "rejected") {
        return decodeRejected(object);
    }
    if (type == "cancel") {
        return decodeCancel(object);
    }
    if (type == "token_delta") {
        return decodeTokenDelta(object);
    }
    if (type == "terminal") {
        return decodeTerminal(object);
    }
    if (type == "health") {
        return decodeHealth(object);
    }
    if (type == "stats") {
        return decodeStats(object);
    }
    return decodeError("unknown message type");
}

std::string_view messageTypeName(Message const& message) noexcept {
    return std::visit(
        [](auto const& typed_message) -> std::string_view {
            using T = std::decay_t<decltype(typed_message)>;
            if constexpr (std::is_same_v<T, Hello>) {
                return "hello";
            } else if constexpr (std::is_same_v<T, HelloAck>) {
                return "hello_ack";
            } else if constexpr (std::is_same_v<T, Submit>) {
                return "submit";
            } else if constexpr (std::is_same_v<T, Accepted>) {
                return "accepted";
            } else if constexpr (std::is_same_v<T, Rejected>) {
                return "rejected";
            } else if constexpr (std::is_same_v<T, Cancel>) {
                return "cancel";
            } else if constexpr (std::is_same_v<T, TokenDelta>) {
                return "token_delta";
            } else if constexpr (std::is_same_v<T, Terminal>) {
                return "terminal";
            } else if constexpr (std::is_same_v<T, Health>) {
                return "health";
            } else {
                return "stats";
            }
        },
        message);
}

std::string_view statusCodeName(StatusCode code) noexcept {
    switch (code) {
    case StatusCode::Ok:
        return "ok";
    case StatusCode::InvalidInput:
        return "invalid_input";
    case StatusCode::InvalidShape:
        return "invalid_shape";
    case StatusCode::AlreadyExists:
        return "already_exists";
    case StatusCode::NotReady:
        return "not_ready";
    case StatusCode::ResourceExhausted:
        return "resource_exhausted";
    case StatusCode::Unavailable:
        return "unavailable";
    case StatusCode::QueueFull:
        return "queue_full";
    case StatusCode::EngineNotFound:
        return "engine_not_found";
    case StatusCode::ContextUnavailable:
        return "context_unavailable";
    case StatusCode::CudaError:
        return "cuda_error";
    case StatusCode::TensorRTError:
        return "tensorrt_error";
    case StatusCode::InternalError:
        return "internal_error";
    case StatusCode::Timeout:
        return "timeout";
    case StatusCode::Cancelled:
        return "cancelled";
    }
    return {};
}

std::optional<StatusCode> parseStatusCode(std::string_view name) noexcept {
    for (auto const code : {
             StatusCode::Ok,
             StatusCode::InvalidInput,
             StatusCode::InvalidShape,
             StatusCode::AlreadyExists,
             StatusCode::NotReady,
             StatusCode::ResourceExhausted,
             StatusCode::Unavailable,
             StatusCode::QueueFull,
             StatusCode::EngineNotFound,
             StatusCode::ContextUnavailable,
             StatusCode::CudaError,
             StatusCode::TensorRTError,
             StatusCode::InternalError,
             StatusCode::Timeout,
             StatusCode::Cancelled}) {
        if (statusCodeName(code) == name) {
            return code;
        }
    }
    return std::nullopt;
}

std::string_view finishReasonName(FinishReason reason) noexcept {
    switch (reason) {
    case FinishReason::Eos:
        return "eos";
    case FinishReason::Length:
        return "length";
    case FinishReason::Stop:
        return "stop";
    case FinishReason::Cancelled:
        return "cancelled";
    case FinishReason::Timeout:
        return "timeout";
    case FinishReason::Backpressure:
        return "backpressure";
    }
    return {};
}

std::optional<FinishReason> parseFinishReason(
    std::string_view name) noexcept {
    for (auto const reason : {
             FinishReason::Eos,
             FinishReason::Length,
             FinishReason::Stop,
             FinishReason::Cancelled,
             FinishReason::Timeout,
             FinishReason::Backpressure}) {
        if (finishReasonName(reason) == name) {
            return reason;
        }
    }
    return std::nullopt;
}

} // namespace kimrt::llm::ipc
