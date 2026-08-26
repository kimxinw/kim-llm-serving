from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union


PROTOCOL_VERSION = 1
DEFAULT_MAX_FRAME_PAYLOAD_BYTES = 1024 * 1024

STATUS_CODES = frozenset(
    {
        "ok",
        "invalid_input",
        "invalid_shape",
        "already_exists",
        "not_ready",
        "resource_exhausted",
        "slo_predicted_miss",
        "unavailable",
        "queue_full",
        "engine_not_found",
        "context_unavailable",
        "cuda_error",
        "tensorrt_error",
        "internal_error",
        "timeout",
        "cancelled",
    }
)

FINISH_REASONS = frozenset(
    {"eos", "length", "stop", "cancelled", "timeout", "backpressure"}
)


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Status:
    code: str = "ok"
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.code == "ok"


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    revision: str
    tokenizer_fingerprint: str
    chat_template_fingerprint: str
    engine_fingerprint: str
    eos_token_id: int
    pad_token_id: int
    max_input_tokens: int
    max_output_tokens: int
    max_sequence_tokens: int
    precision: str
    max_batch_size: int


@dataclass(frozen=True)
class WorkerLimits:
    max_active_requests: int
    max_total_input_tokens: int
    max_reserved_output_tokens: int
    max_frame_payload_bytes: int
    max_session_egress_frames: int
    max_session_egress_bytes: int
    max_request_egress_frames: int
    max_request_egress_bytes: int


@dataclass(frozen=True)
class SamplingParameters:
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 1.0
    random_seed: int = 0


@dataclass(frozen=True)
class GenerationRequest:
    request_id: int
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    priority: int = 0
    timeout_ms: Optional[int] = None
    trace_id: str = ""
    streaming: bool = True
    sampling: SamplingParameters = field(default_factory=SamplingParameters)
    end_id: Optional[int] = None
    pad_id: Optional[int] = None
    stop_sequences: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class HelloAck:
    worker_epoch: int
    manifest: ModelManifest
    limits: WorkerLimits


@dataclass(frozen=True)
class Accepted:
    request_id: int


@dataclass(frozen=True)
class Rejected:
    request_id: int
    status: Status


@dataclass(frozen=True)
class TokenDelta:
    request_id: int
    sequence_no: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class Terminal:
    request_id: int
    status: Status
    finish_reason: Optional[str]
    usage: Usage


@dataclass(frozen=True)
class Stats:
    probe_id: int
    ready: bool
    status: Status
    uptime_ms: int
    active_requests: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    session_egress_frames: int
    session_egress_bytes: int
    session_egress_high_watermark_frames: int
    session_egress_high_watermark_bytes: int
    rejected_requests: int
    backpressure_requests: int
    cancelled_requests: int


ServerMessage = Union[HelloAck, Accepted, Rejected, TokenDelta, Terminal, Stats]
GenerationEvent = Union[TokenDelta, Terminal]


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"invalid JSON constant {value}")


def _require_exact_fields(
    value: Any,
    expected: set[str],
    description: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ProtocolError(
            f"{description} contains missing or unexpected fields"
        )
    return value


def _require_string(value: Any, description: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise ProtocolError(f"{description} must be a {qualifier}string")
    return value


def _require_bool(value: Any, description: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{description} must be a boolean")
    return value


def _require_int(
    value: Any,
    description: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 64) - 1,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ProtocolError(f"{description} is outside the supported integer range")
    return value


def _require_float(value: Any, description: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"{description} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ProtocolError(f"{description} must be finite")
    return converted


def _require_protocol(value: Any) -> None:
    version = _require_int(value, "protocol_version", maximum=(1 << 32) - 1)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")


def _require_epoch(value: Any, expected_epoch: Optional[int]) -> int:
    epoch = _require_int(value, "worker_epoch", minimum=1)
    if expected_epoch is not None and epoch != expected_epoch:
        raise ProtocolError("worker_epoch does not match the active Worker")
    return epoch


def _status_from_json(value: Any) -> Status:
    object_value = _require_exact_fields(
        value, {"code", "message"}, "status"
    )
    code = _require_string(object_value["code"], "status.code", non_empty=True)
    if code not in STATUS_CODES:
        raise ProtocolError(f"unknown status code {code}")
    return Status(code, _require_string(object_value["message"], "status.message"))


def _manifest_from_json(value: Any) -> ModelManifest:
    object_value = _require_exact_fields(
        value,
        {
            "model_id",
            "revision",
            "tokenizer_fingerprint",
            "chat_template_fingerprint",
            "engine_fingerprint",
            "eos_token_id",
            "pad_token_id",
            "max_input_tokens",
            "max_output_tokens",
            "max_sequence_tokens",
            "precision",
            "max_batch_size",
        },
        "manifest",
    )
    manifest = ModelManifest(
        model_id=_require_string(object_value["model_id"], "model_id", non_empty=True),
        revision=_require_string(object_value["revision"], "revision", non_empty=True),
        tokenizer_fingerprint=_require_string(
            object_value["tokenizer_fingerprint"],
            "tokenizer_fingerprint",
            non_empty=True,
        ),
        chat_template_fingerprint=_require_string(
            object_value["chat_template_fingerprint"],
            "chat_template_fingerprint",
            non_empty=True,
        ),
        engine_fingerprint=_require_string(
            object_value["engine_fingerprint"],
            "engine_fingerprint",
            non_empty=True,
        ),
        eos_token_id=_require_int(
            object_value["eos_token_id"], "eos_token_id", maximum=(1 << 31) - 1
        ),
        pad_token_id=_require_int(
            object_value["pad_token_id"], "pad_token_id", maximum=(1 << 31) - 1
        ),
        max_input_tokens=_require_int(
            object_value["max_input_tokens"],
            "max_input_tokens",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        max_output_tokens=_require_int(
            object_value["max_output_tokens"],
            "max_output_tokens",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        max_sequence_tokens=_require_int(
            object_value["max_sequence_tokens"],
            "max_sequence_tokens",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        precision=_require_string(
            object_value["precision"], "precision", non_empty=True
        ),
        max_batch_size=_require_int(
            object_value["max_batch_size"],
            "max_batch_size",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
    )
    if (
        manifest.max_input_tokens > manifest.max_sequence_tokens
        or manifest.max_output_tokens > manifest.max_sequence_tokens
    ):
        raise ProtocolError("manifest token limits exceed max_sequence_tokens")
    return manifest


def _limits_from_json(value: Any) -> WorkerLimits:
    object_value = _require_exact_fields(
        value,
        {
            "max_active_requests",
            "max_total_input_tokens",
            "max_reserved_output_tokens",
            "max_frame_payload_bytes",
            "max_session_egress_frames",
            "max_session_egress_bytes",
            "max_request_egress_frames",
            "max_request_egress_bytes",
        },
        "limits",
    )
    limits = WorkerLimits(
        max_active_requests=_require_int(
            object_value["max_active_requests"],
            "max_active_requests",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        max_total_input_tokens=_require_int(
            object_value["max_total_input_tokens"],
            "max_total_input_tokens",
            minimum=1,
        ),
        max_reserved_output_tokens=_require_int(
            object_value["max_reserved_output_tokens"],
            "max_reserved_output_tokens",
            minimum=1,
        ),
        max_frame_payload_bytes=_require_int(
            object_value["max_frame_payload_bytes"],
            "max_frame_payload_bytes",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        max_session_egress_frames=_require_int(
            object_value["max_session_egress_frames"],
            "max_session_egress_frames",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        max_session_egress_bytes=_require_int(
            object_value["max_session_egress_bytes"],
            "max_session_egress_bytes",
            minimum=1,
        ),
        max_request_egress_frames=_require_int(
            object_value["max_request_egress_frames"],
            "max_request_egress_frames",
            minimum=1,
            maximum=(1 << 32) - 1,
        ),
        max_request_egress_bytes=_require_int(
            object_value["max_request_egress_bytes"],
            "max_request_egress_bytes",
            minimum=1,
        ),
    )
    if (
        limits.max_request_egress_frames > limits.max_session_egress_frames
        or limits.max_request_egress_bytes > limits.max_session_egress_bytes
        or limits.max_frame_payload_bytes > limits.max_request_egress_bytes
    ):
        raise ProtocolError("Worker limit hierarchy is inconsistent")
    return limits


def _usage_from_json(value: Any) -> Usage:
    object_value = _require_exact_fields(
        value, {"prompt_tokens", "completion_tokens"}, "usage"
    )
    return Usage(
        _require_int(object_value["prompt_tokens"], "prompt_tokens"),
        _require_int(object_value["completion_tokens"], "completion_tokens"),
    )


def _request_envelope(
    message: Mapping[str, Any], expected_epoch: int
) -> int:
    _require_protocol(message["protocol_version"])
    _require_epoch(message["worker_epoch"], expected_epoch)
    return _require_int(message["request_id"], "request_id", minimum=1)


def encode_hello(manifest: ModelManifest) -> bytes:
    _validate_manifest(manifest)
    return encode_json(
        {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "model_id": manifest.model_id,
            "revision": manifest.revision,
        }
    )


def encode_submit(request: GenerationRequest, worker_epoch: int) -> bytes:
    _validate_request(request)
    _require_int(worker_epoch, "worker_epoch", minimum=1)
    return encode_json(
        {
            "type": "submit",
            "protocol_version": PROTOCOL_VERSION,
            "worker_epoch": worker_epoch,
            "request_id": request.request_id,
            "priority": request.priority,
            "timeout_ms": request.timeout_ms,
            "trace_id": request.trace_id,
            "input_token_ids": list(request.input_token_ids),
            "max_new_tokens": request.max_new_tokens,
            "streaming": request.streaming,
            "sampling": {
                "temperature": request.sampling.temperature,
                "top_k": request.sampling.top_k,
                "top_p": request.sampling.top_p,
                "random_seed": request.sampling.random_seed,
            },
            "end_id": request.end_id,
            "pad_id": request.pad_id,
            "stop_sequences": [list(sequence) for sequence in request.stop_sequences],
        }
    )


def encode_cancel(request_id: int, worker_epoch: int) -> bytes:
    _require_int(request_id, "request_id", minimum=1)
    _require_int(worker_epoch, "worker_epoch", minimum=1)
    return encode_json(
        {
            "type": "cancel",
            "protocol_version": PROTOCOL_VERSION,
            "worker_epoch": worker_epoch,
            "request_id": request_id,
        }
    )


def encode_health(probe_id: int, worker_epoch: int) -> bytes:
    _require_int(probe_id, "probe_id", minimum=1)
    _require_int(worker_epoch, "worker_epoch", minimum=1)
    return encode_json(
        {
            "type": "health",
            "protocol_version": PROTOCOL_VERSION,
            "worker_epoch": worker_epoch,
            "probe_id": probe_id,
        }
    )


def encode_json(message: Mapping[str, Any]) -> bytes:
    return json.dumps(
        message,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def encode_frame(payload: bytes, max_payload_bytes: int) -> bytes:
    if not payload:
        raise ProtocolError("frame payload must not be empty")
    if len(payload) > max_payload_bytes or len(payload) > (1 << 32) - 1:
        raise ProtocolError("frame payload exceeds max_payload_bytes")
    return struct.pack("!I", len(payload)) + payload


def decode_server_message(payload: bytes, expected_epoch: Optional[int]) -> ServerMessage:
    try:
        object_value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise ProtocolError(f"invalid JSON payload: {exception}") from exception
    if not isinstance(object_value, dict):
        raise ProtocolError("payload must be a JSON object")
    message_type = _require_string(
        object_value.get("type"), "message type", non_empty=True
    )

    if message_type == "hello_ack":
        message = _require_exact_fields(
            object_value,
            {"type", "protocol_version", "worker_epoch", "manifest", "limits"},
            "HelloAck",
        )
        _require_protocol(message["protocol_version"])
        epoch = _require_epoch(message["worker_epoch"], expected_epoch)
        return HelloAck(
            epoch,
            _manifest_from_json(message["manifest"]),
            _limits_from_json(message["limits"]),
        )

    if expected_epoch is None:
        raise ProtocolError("lifecycle message arrived before HelloAck")

    if message_type == "accepted":
        message = _require_exact_fields(
            object_value,
            {"type", "protocol_version", "worker_epoch", "request_id"},
            "Accepted",
        )
        return Accepted(_request_envelope(message, expected_epoch))

    if message_type == "rejected":
        message = _require_exact_fields(
            object_value,
            {"type", "protocol_version", "worker_epoch", "request_id", "status"},
            "Rejected",
        )
        status = _status_from_json(message["status"])
        if status.ok:
            raise ProtocolError("Rejected status must not be ok")
        return Rejected(_request_envelope(message, expected_epoch), status)

    if message_type == "token_delta":
        message = _require_exact_fields(
            object_value,
            {
                "type",
                "protocol_version",
                "worker_epoch",
                "request_id",
                "sequence_no",
                "token_ids",
            },
            "TokenDelta",
        )
        token_values = message["token_ids"]
        if not isinstance(token_values, list) or not token_values:
            raise ProtocolError("token_ids must be a non-empty array")
        tokens = tuple(
            _require_int(token, "token_id", maximum=(1 << 31) - 1)
            for token in token_values
        )
        return TokenDelta(
            _request_envelope(message, expected_epoch),
            _require_int(message["sequence_no"], "sequence_no"),
            tokens,
        )

    if message_type == "terminal":
        message = _require_exact_fields(
            object_value,
            {
                "type",
                "protocol_version",
                "worker_epoch",
                "request_id",
                "status",
                "finish_reason",
                "usage",
            },
            "Terminal",
        )
        status = _status_from_json(message["status"])
        finish_reason_value = message["finish_reason"]
        if finish_reason_value is None:
            finish_reason = None
        else:
            finish_reason = _require_string(
                finish_reason_value, "finish_reason", non_empty=True
            )
            if finish_reason not in FINISH_REASONS:
                raise ProtocolError(f"unknown finish reason {finish_reason}")
        if status.ok and finish_reason is None:
            raise ProtocolError("successful Terminal requires finish_reason")
        return Terminal(
            _request_envelope(message, expected_epoch),
            status,
            finish_reason,
            _usage_from_json(message["usage"]),
        )

    if message_type == "stats":
        message = _require_exact_fields(
            object_value,
            {
                "type",
                "protocol_version",
                "worker_epoch",
                "probe_id",
                "ready",
                "status",
                "uptime_ms",
                "active_requests",
                "reserved_input_tokens",
                "reserved_output_tokens",
                "session_egress_frames",
                "session_egress_bytes",
                "session_egress_high_watermark_frames",
                "session_egress_high_watermark_bytes",
                "rejected_requests",
                "backpressure_requests",
                "cancelled_requests",
            },
            "Stats",
        )
        _require_protocol(message["protocol_version"])
        _require_epoch(message["worker_epoch"], expected_epoch)
        status = _status_from_json(message["status"])
        ready = _require_bool(message["ready"], "ready")
        if ready != status.ok:
            raise ProtocolError("Stats ready and status are inconsistent")
        stats = Stats(
            probe_id=_require_int(message["probe_id"], "probe_id", minimum=1),
            ready=ready,
            status=status,
            uptime_ms=_require_int(message["uptime_ms"], "uptime_ms"),
            active_requests=_require_int(
                message["active_requests"], "active_requests"
            ),
            reserved_input_tokens=_require_int(
                message["reserved_input_tokens"], "reserved_input_tokens"
            ),
            reserved_output_tokens=_require_int(
                message["reserved_output_tokens"], "reserved_output_tokens"
            ),
            session_egress_frames=_require_int(
                message["session_egress_frames"], "session_egress_frames"
            ),
            session_egress_bytes=_require_int(
                message["session_egress_bytes"], "session_egress_bytes"
            ),
            session_egress_high_watermark_frames=_require_int(
                message["session_egress_high_watermark_frames"],
                "session_egress_high_watermark_frames",
            ),
            session_egress_high_watermark_bytes=_require_int(
                message["session_egress_high_watermark_bytes"],
                "session_egress_high_watermark_bytes",
            ),
            rejected_requests=_require_int(
                message["rejected_requests"], "rejected_requests"
            ),
            backpressure_requests=_require_int(
                message["backpressure_requests"], "backpressure_requests"
            ),
            cancelled_requests=_require_int(
                message["cancelled_requests"], "cancelled_requests"
            ),
        )
        if (
            stats.session_egress_high_watermark_frames
            < stats.session_egress_frames
            or stats.session_egress_high_watermark_bytes
            < stats.session_egress_bytes
        ):
            raise ProtocolError("Stats high watermark is below the current value")
        return stats

    raise ProtocolError(f"message type {message_type} is not allowed from Worker")


def _validate_manifest(manifest: ModelManifest) -> None:
    if not all(
        (
            manifest.model_id,
            manifest.revision,
            manifest.tokenizer_fingerprint,
            manifest.chat_template_fingerprint,
            manifest.engine_fingerprint,
            manifest.precision,
        )
    ):
        raise ProtocolError("ModelManifest string fields must not be empty")
    for token_id in (manifest.eos_token_id, manifest.pad_token_id):
        _require_int(token_id, "manifest token id", maximum=(1 << 31) - 1)
    for value, name in (
        (manifest.max_input_tokens, "max_input_tokens"),
        (manifest.max_output_tokens, "max_output_tokens"),
        (manifest.max_sequence_tokens, "max_sequence_tokens"),
        (manifest.max_batch_size, "max_batch_size"),
    ):
        _require_int(value, name, minimum=1, maximum=(1 << 32) - 1)
    if (
        manifest.max_input_tokens > manifest.max_sequence_tokens
        or manifest.max_output_tokens > manifest.max_sequence_tokens
    ):
        raise ProtocolError("manifest token limits exceed max_sequence_tokens")


def _validate_request(request: GenerationRequest) -> None:
    _require_int(request.request_id, "request_id", minimum=1)
    if not request.input_token_ids:
        raise ProtocolError("input_token_ids must not be empty")
    for token_id in request.input_token_ids:
        _require_int(token_id, "input_token_id", maximum=(1 << 31) - 1)
    _require_int(
        request.max_new_tokens,
        "max_new_tokens",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    _require_int(
        request.priority,
        "priority",
        minimum=-(1 << 31),
        maximum=(1 << 31) - 1,
    )
    if request.timeout_ms is not None:
        _require_int(request.timeout_ms, "timeout_ms", minimum=1)
    _require_string(request.trace_id, "trace_id")
    _require_bool(request.streaming, "streaming")
    temperature = _require_float(request.sampling.temperature, "temperature")
    if temperature <= 0:
        raise ProtocolError("temperature must be positive")
    _require_int(
        request.sampling.top_k,
        "top_k",
        minimum=0,
        maximum=(1 << 31) - 1,
    )
    top_p = _require_float(request.sampling.top_p, "top_p")
    if top_p <= 0 or top_p > 1:
        raise ProtocolError("top_p must be in (0, 1]")
    _require_int(request.sampling.random_seed, "random_seed")
    for optional_token in (request.end_id, request.pad_id):
        if optional_token is not None:
            _require_int(
                optional_token,
                "optional token id",
                maximum=(1 << 31) - 1,
            )
    for sequence in request.stop_sequences:
        if not sequence:
            raise ProtocolError("stop_sequences entries must not be empty")
        for token_id in sequence:
            _require_int(token_id, "stop token id", maximum=(1 << 31) - 1)


__all__ = [
    "Accepted",
    "DEFAULT_MAX_FRAME_PAYLOAD_BYTES",
    "GenerationEvent",
    "GenerationRequest",
    "HelloAck",
    "ModelManifest",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "Rejected",
    "SamplingParameters",
    "ServerMessage",
    "Stats",
    "Status",
    "Terminal",
    "TokenDelta",
    "Usage",
    "WorkerLimits",
    "decode_server_message",
    "encode_cancel",
    "encode_frame",
    "encode_health",
    "encode_hello",
    "encode_json",
    "encode_submit",
]
