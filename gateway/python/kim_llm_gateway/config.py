from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from kim_llm_client import ModelManifest


class GatewayConfigError(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise GatewayConfigError(f"invalid JSON constant {value}")


def _load_json_object(path: Path, description: str) -> Mapping[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exception:
        raise GatewayConfigError(f"failed to read {description}: {path}") from exception
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeError) as exception:
        raise GatewayConfigError(f"failed to parse {description}: {exception}") from exception
    if not isinstance(value, dict):
        raise GatewayConfigError(f"{description} must be a JSON object")
    return value


def _require_fields(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    description: str,
) -> None:
    fields = set(value)
    missing = required - fields
    unexpected = fields - required - optional
    if missing:
        raise GatewayConfigError(
            f"{description} is missing fields: {', '.join(sorted(missing))}"
        )
    if unexpected:
        raise GatewayConfigError(
            f"{description} contains unexpected fields: "
            f"{', '.join(sorted(unexpected))}"
        )


def _string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise GatewayConfigError(f"{description} must be a non-empty string")
    return value


def _integer(
    value: Any,
    description: str,
    *,
    minimum: int = 1,
    maximum: int = (1 << 31) - 1,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise GatewayConfigError(f"{description} is outside the supported range")
    return value


def _number(
    value: Any,
    description: str,
    *,
    minimum_exclusive: float = 0.0,
) -> float:
    if type(value) not in (int, float):
        raise GatewayConfigError(f"{description} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= minimum_exclusive:
        raise GatewayConfigError(f"{description} must be finite and positive")
    return converted


def _manifest(value: Any) -> ModelManifest:
    if not isinstance(value, dict):
        raise GatewayConfigError("worker manifest must be a JSON object")
    expected = {
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
    }
    _require_fields(value, expected, set(), "worker manifest")
    result = ModelManifest(
        model_id=_string(value["model_id"], "manifest.model_id"),
        revision=_string(value["revision"], "manifest.revision"),
        tokenizer_fingerprint=_string(
            value["tokenizer_fingerprint"], "manifest.tokenizer_fingerprint"
        ),
        chat_template_fingerprint=_string(
            value["chat_template_fingerprint"],
            "manifest.chat_template_fingerprint",
        ),
        engine_fingerprint=_string(
            value["engine_fingerprint"], "manifest.engine_fingerprint"
        ),
        eos_token_id=_integer(
            value["eos_token_id"], "manifest.eos_token_id", minimum=0
        ),
        pad_token_id=_integer(
            value["pad_token_id"], "manifest.pad_token_id", minimum=0
        ),
        max_input_tokens=_integer(
            value["max_input_tokens"], "manifest.max_input_tokens"
        ),
        max_output_tokens=_integer(
            value["max_output_tokens"], "manifest.max_output_tokens"
        ),
        max_sequence_tokens=_integer(
            value["max_sequence_tokens"], "manifest.max_sequence_tokens"
        ),
        precision=_string(value["precision"], "manifest.precision"),
        max_batch_size=_integer(
            value["max_batch_size"], "manifest.max_batch_size"
        ),
    )
    if (
        result.max_input_tokens > result.max_sequence_tokens
        or result.max_output_tokens > result.max_sequence_tokens
    ):
        raise GatewayConfigError("manifest token limits exceed max_sequence_tokens")
    return result


@dataclass(frozen=True)
class GatewayConfig:
    worker_config_path: Path
    tokenizer_path: Path
    socket_path: str
    manifest: ModelManifest
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    accept_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 120.0
    health_timeout_seconds: float = 1.0
    event_poll_seconds: float = 0.1
    cancel_drain_timeout_seconds: float = 30.0
    shutdown_grace_seconds: float = 30.0
    max_http_body_bytes: int = 1024 * 1024
    max_pending_requests: int = 8
    max_client_delta_events_per_request: int = 64
    max_sse_delta_events_per_request: int = 32
    sampling_top_k: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "GatewayConfig":
        config_path = Path(path).expanduser().resolve()
        root = _load_json_object(config_path, "Gateway configuration")
        required = {"worker_config_path", "tokenizer_path"}
        optional = {
            "host",
            "port",
            "log_level",
            "accept_timeout_seconds",
            "request_timeout_seconds",
            "health_timeout_seconds",
            "event_poll_seconds",
            "cancel_drain_timeout_seconds",
            "shutdown_grace_seconds",
            "max_http_body_bytes",
            "max_pending_requests",
            "max_client_delta_events_per_request",
            "max_sse_delta_events_per_request",
            "sampling_top_k",
        }
        _require_fields(root, required, optional, "Gateway configuration")

        base = config_path.parent
        worker_config_path = cls._resolve_path(
            base, _string(root["worker_config_path"], "worker_config_path")
        )
        tokenizer_path = cls._resolve_path(
            base, _string(root["tokenizer_path"], "tokenizer_path")
        )
        worker = _load_json_object(worker_config_path, "Worker configuration")
        if "socket_path" not in worker or "manifest" not in worker:
            raise GatewayConfigError(
                "Worker configuration must contain socket_path and manifest"
            )
        socket_value = _string(worker["socket_path"], "worker socket_path")
        socket_path = Path(socket_value).expanduser()
        if not socket_path.is_absolute():
            socket_path = worker_config_path.parent / socket_path

        result = cls(
            worker_config_path=worker_config_path,
            tokenizer_path=tokenizer_path,
            socket_path=str(socket_path.resolve()),
            manifest=_manifest(worker["manifest"]),
            host=_string(root.get("host", "127.0.0.1"), "host"),
            port=_integer(root.get("port", 8000), "port", maximum=65535),
            log_level=_string(root.get("log_level", "info"), "log_level"),
            accept_timeout_seconds=_number(
                root.get("accept_timeout_seconds", 10.0),
                "accept_timeout_seconds",
            ),
            request_timeout_seconds=_number(
                root.get("request_timeout_seconds", 120.0),
                "request_timeout_seconds",
            ),
            health_timeout_seconds=_number(
                root.get("health_timeout_seconds", 1.0),
                "health_timeout_seconds",
            ),
            event_poll_seconds=_number(
                root.get("event_poll_seconds", 0.1), "event_poll_seconds"
            ),
            cancel_drain_timeout_seconds=_number(
                root.get("cancel_drain_timeout_seconds", 30.0),
                "cancel_drain_timeout_seconds",
            ),
            shutdown_grace_seconds=_number(
                root.get("shutdown_grace_seconds", 30.0),
                "shutdown_grace_seconds",
            ),
            max_http_body_bytes=_integer(
                root.get("max_http_body_bytes", 1024 * 1024),
                "max_http_body_bytes",
            ),
            max_pending_requests=_integer(
                root.get("max_pending_requests", 8), "max_pending_requests"
            ),
            max_client_delta_events_per_request=_integer(
                root.get("max_client_delta_events_per_request", 64),
                "max_client_delta_events_per_request",
            ),
            max_sse_delta_events_per_request=_integer(
                root.get("max_sse_delta_events_per_request", 32),
                "max_sse_delta_events_per_request",
            ),
            sampling_top_k=_integer(
                root.get("sampling_top_k", 1),
                "sampling_top_k",
            ),
        )
        if (
            result.max_sse_delta_events_per_request
            > result.max_client_delta_events_per_request
        ):
            raise GatewayConfigError(
                "max_sse_delta_events_per_request must not exceed the "
                "GenerationClient delta capacity"
            )
        if result.log_level not in {"critical", "error", "warning", "info", "debug"}:
            raise GatewayConfigError("log_level is unsupported")
        return result

    @staticmethod
    def _resolve_path(base: Path, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()


__all__ = ["GatewayConfig", "GatewayConfigError"]
