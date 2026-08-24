from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from kim_llm_client import ModelManifest, Usage

from .runtime import GatewayError, GatewayRequest, GatewayTerminal


@dataclass(frozen=True)
class ChatCompletionCommand:
    model: str
    messages: tuple[Mapping[str, str], ...]
    stream: bool
    include_usage: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    random_seed: int
    stop: tuple[str, ...]

    def to_gateway_request(self, trace_id: str) -> GatewayRequest:
        return GatewayRequest(
            model=self.model,
            messages=self.messages,
            max_new_tokens=self.max_new_tokens,
            streaming=self.stream,
            temperature=self.temperature,
            top_p=self.top_p,
            random_seed=self.random_seed,
            stop=self.stop,
            trace_id=trace_id,
        )


def _invalid(message: str, param: Optional[str] = None) -> GatewayError:
    return GatewayError(
        message,
        http_status=400,
        error_type="invalid_request_error",
        code="invalid_request",
        param=param,
    )


def _string(value: Any, param: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise _invalid(f"{param} must be a {qualifier}string", param)
    return value


def _integer(
    value: Any,
    param: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise _invalid(f"{param} is outside the supported range", param)
    return value


def _number(
    value: Any,
    param: str,
    *,
    minimum_exclusive: float,
    maximum: float,
) -> float:
    if type(value) not in (int, float):
        raise _invalid(f"{param} must be numeric", param)
    converted = float(value)
    if (
        not math.isfinite(converted)
        or converted <= minimum_exclusive
        or converted > maximum
    ):
        raise _invalid(f"{param} is outside the supported range", param)
    return converted


def _messages(value: Any) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise _invalid("messages must be a non-empty array", "messages")
    result: list[Mapping[str, str]] = []
    for index, message in enumerate(value):
        param = f"messages[{index}]"
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise _invalid(
                f"{param} must contain exactly role and content",
                param,
            )
        role = _string(message["role"], f"{param}.role")
        if role not in {"system", "user", "assistant"}:
            raise _invalid(
                f"{param}.role is unsupported by the fixed chat template",
                f"{param}.role",
            )
        content = _string(message["content"], f"{param}.content", non_empty=False)
        result.append({"role": role, "content": content})
    return tuple(result)


def _stop(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise _invalid("stop must be a string or an array of strings", "stop")
    if not values or len(values) > 4:
        raise _invalid("stop must contain between 1 and 4 sequences", "stop")
    result = tuple(_string(item, "stop") for item in values)
    if len(set(result)) != len(result):
        raise _invalid("stop sequences must be unique", "stop")
    return result


def parse_chat_completion_request(
    value: Any,
    manifest: ModelManifest,
) -> ChatCompletionCommand:
    if not isinstance(value, dict):
        raise _invalid("request body must be a JSON object")
    supported = {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "n",
        "user",
        "frequency_penalty",
        "presence_penalty",
    }
    unexpected = set(value) - supported
    if unexpected:
        field = sorted(unexpected)[0]
        raise _invalid(f"unsupported request field {field}", field)
    if "model" not in value or "messages" not in value:
        missing = "model" if "model" not in value else "messages"
        raise _invalid(f"missing required field {missing}", missing)

    model = _string(value["model"], "model")
    messages = _messages(value["messages"])
    stream = value.get("stream", False)
    if type(stream) is not bool:
        raise _invalid("stream must be a boolean", "stream")
    if _integer(value.get("n", 1), "n", minimum=1, maximum=1) != 1:
        raise _invalid("only n=1 is supported", "n")

    if "max_tokens" in value and "max_completion_tokens" in value:
        raise _invalid(
            "max_tokens and max_completion_tokens are mutually exclusive",
            "max_tokens",
        )
    max_value = value.get(
        "max_completion_tokens",
        value.get("max_tokens", min(32, manifest.max_output_tokens)),
    )
    max_new_tokens = _integer(
        max_value,
        "max_tokens",
        minimum=1,
        maximum=manifest.max_output_tokens,
    )
    temperature = _number(
        value.get("temperature", 1.0),
        "temperature",
        minimum_exclusive=0.0,
        maximum=2.0,
    )
    top_p = _number(
        value.get("top_p", 1.0),
        "top_p",
        minimum_exclusive=0.0,
        maximum=1.0,
    )
    random_seed = _integer(
        value.get("seed", 0),
        "seed",
        minimum=0,
        maximum=(1 << 32) - 1,
    )
    stop = _stop(value.get("stop"))

    for penalty in ("frequency_penalty", "presence_penalty"):
        if penalty not in value:
            continue
        number = value[penalty]
        if type(number) not in (int, float) or not math.isfinite(float(number)):
            raise _invalid(f"{penalty} must be numeric", penalty)
        if float(number) != 0.0:
            raise _invalid(f"only {penalty}=0 is supported", penalty)
    if "user" in value:
        _string(value["user"], "user")

    include_usage = False
    stream_options = value.get("stream_options")
    if stream_options is not None:
        if not stream:
            raise _invalid("stream_options requires stream=true", "stream_options")
        if (
            not isinstance(stream_options, dict)
            or set(stream_options) != {"include_usage"}
            or type(stream_options["include_usage"]) is not bool
        ):
            raise _invalid(
                "stream_options must contain exactly boolean include_usage",
                "stream_options",
            )
        include_usage = stream_options["include_usage"]

    return ChatCompletionCommand(
        model=model,
        messages=messages,
        stream=stream,
        include_usage=include_usage,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        random_seed=random_seed,
        stop=stop,
    )


def openai_finish_reason(value: Optional[str]) -> str:
    if value == "length":
        return "length"
    return "stop"


def usage_json(usage: Usage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
    }


def error_json(error: GatewayError) -> dict[str, object]:
    return {
        "error": {
            "message": str(error),
            "type": error.error_type,
            "param": error.param,
            "code": error.code,
        }
    }


def chat_completion_json(
    completion_id: str,
    created: int,
    model: str,
    text: str,
    terminal: GatewayTerminal,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": openai_finish_reason(terminal.finish_reason),
            }
        ],
        "usage": usage_json(terminal.usage),
    }


def chat_completion_chunk_json(
    completion_id: str,
    created: int,
    model: str,
    *,
    delta: Mapping[str, str],
    finish_reason: Optional[str] = None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": dict(delta),
                "finish_reason": finish_reason,
            }
        ],
    }


def usage_chunk_json(
    completion_id: str,
    created: int,
    model: str,
    usage: Usage,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage_json(usage),
    }


def encode_sse(value: Mapping[str, object]) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n".encode("utf-8")


__all__ = [
    "ChatCompletionCommand",
    "chat_completion_chunk_json",
    "chat_completion_json",
    "encode_sse",
    "error_json",
    "openai_finish_reason",
    "parse_chat_completion_request",
    "usage_chunk_json",
    "usage_json",
]
