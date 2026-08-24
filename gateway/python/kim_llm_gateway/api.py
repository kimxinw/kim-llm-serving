from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from kim_llm_client import GenerationClient, GenerationClientConfig

try:
    from fastapi import Request as FastAPIRequest
except ImportError:
    FastAPIRequest = Any  # type: ignore[misc,assignment]

from .config import GatewayConfig
from .openai import (
    chat_completion_chunk_json,
    chat_completion_json,
    encode_sse,
    error_json,
    openai_finish_reason,
    parse_chat_completion_request,
    usage_chunk_json,
)
from .runtime import (
    GatewayError,
    GatewayRuntimeOptions,
    GatewayService,
    GatewaySession,
    GatewayTerminal,
    GatewayTextDelta,
    gateway_error_from_status,
)
from .tokenizer import HuggingFaceTokenizer


LOGGER = logging.getLogger("kim_llm_gateway")


@dataclass
class _ApplicationState:
    config: GatewayConfig
    service: Optional[GatewayService] = None
    startup_error: Optional[str] = None


def build_gateway_service(config: GatewayConfig) -> GatewayService:
    tokenizer = HuggingFaceTokenizer.load(config.tokenizer_path)
    tokenizer.validate_manifest(config.manifest)
    client = GenerationClient(
        GenerationClientConfig(
            socket_path=config.socket_path,
            expected_manifest=config.manifest,
            connect_timeout=config.accept_timeout_seconds,
            handshake_timeout=config.accept_timeout_seconds,
            max_pending_requests=config.max_pending_requests,
            max_delta_events_per_request=(
                config.max_client_delta_events_per_request
            ),
        )
    )
    return GatewayService(
        client,
        tokenizer,
        config.manifest,
        GatewayRuntimeOptions(
            accept_timeout_seconds=config.accept_timeout_seconds,
            request_timeout_seconds=config.request_timeout_seconds,
            health_timeout_seconds=config.health_timeout_seconds,
            cancel_drain_timeout_seconds=config.cancel_drain_timeout_seconds,
            shutdown_grace_seconds=config.shutdown_grace_seconds,
            max_sse_delta_events_per_request=(
                config.max_sse_delta_events_per_request
            ),
            sampling_top_k=config.sampling_top_k,
        ),
    )


def create_app(
    config: GatewayConfig,
    *,
    service: Optional[GatewayService] = None,
) -> Any:
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exception:
        raise RuntimeError("FastAPI is required to create the Gateway application") from exception

    state = _ApplicationState(
        config=config,
        service=service,
    )

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        try:
            if state.service is None:
                state.service = build_gateway_service(state.config)
            started = await state.service.start()
            if not started:
                state.startup_error = state.service.last_readiness_error
                LOGGER.error("Gateway started not-ready: %s", state.startup_error)
            else:
                state.startup_error = None
        except Exception as exception:
            state.startup_error = str(exception)
            LOGGER.exception("Gateway initialization failed")
        try:
            yield
        finally:
            if state.service is not None:
                await state.service.stop()

    app = FastAPI(
        title="kim-llm-serving Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    def unavailable_error() -> GatewayError:
        return GatewayError(
            state.startup_error or "Gateway is not initialized",
            http_status=503,
            error_type="server_error",
            code="not_ready",
        )

    def require_service() -> GatewayService:
        if state.service is None:
            raise unavailable_error()
        return state.service

    def error_response(error: GatewayError) -> Any:
        return JSONResponse(error_json(error), status_code=error.http_status)

    @app.get("/healthz")
    async def healthz() -> Any:
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    async def readyz() -> Any:
        try:
            gateway = require_service()
            stats = await gateway.check_readiness()
            state.startup_error = None
            return JSONResponse(
                {
                    "status": "ready",
                    "model": gateway.manifest.model_id,
                    "worker_connected": gateway.connected,
                    "active_requests": stats.active_requests,
                }
            )
        except GatewayError as exception:
            state.startup_error = str(exception)
            return JSONResponse(
                {"status": "not_ready", "reason": str(exception)},
                status_code=503,
            )

    @app.get("/metrics")
    async def metrics() -> Any:
        if state.service is None:
            return PlainTextResponse(
                "kim_llm_gateway_connected 0\nkim_llm_gateway_ready 0\n",
                media_type="text/plain; version=0.0.4",
            )
        return PlainTextResponse(
            state.service.render_metrics(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/models")
    async def models() -> Any:
        try:
            gateway = require_service()
        except GatewayError as exception:
            return error_response(exception)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": gateway.manifest.model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "kim-llm-serving",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: FastAPIRequest) -> Any:
        try:
            gateway = require_service()
            body = await _read_bounded_body(
                request,
                state.config.max_http_body_bytes,
            )
            try:
                value = json.loads(
                    body.decode("utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, UnicodeError, ValueError) as exception:
                raise GatewayError(
                    f"invalid JSON request body: {exception}",
                    http_status=400,
                    error_type="invalid_request_error",
                    code="invalid_json",
                ) from exception
            command = parse_chat_completion_request(value, gateway.manifest)
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            session = await gateway.submit(
                command.to_gateway_request(completion_id)
            )
        except GatewayError as exception:
            return error_response(exception)

        if not command.stream:
            try:
                text, terminal = await _collect_non_streaming(
                    request,
                    gateway,
                    session,
                    state.config.event_poll_seconds,
                )
                if not terminal.status.ok:
                    raise gateway_error_from_status(terminal.status)
                return JSONResponse(
                    chat_completion_json(
                        completion_id,
                        created,
                        command.model,
                        text,
                        terminal,
                    )
                )
            except GatewayError as exception:
                return error_response(exception)

        return StreamingResponse(
            _stream_chat_completion(
                request,
                gateway,
                session,
                completion_id,
                created,
                command.model,
                command.include_usage,
                state.config.event_poll_seconds,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


async def _read_bounded_body(request: Any, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise GatewayError(
                "request body exceeds the configured limit",
                http_status=413,
                error_type="invalid_request_error",
                code="request_too_large",
            )
        body.extend(chunk)
    if not body:
        raise GatewayError(
            "request body must not be empty",
            http_status=400,
            error_type="invalid_request_error",
            code="invalid_json",
        )
    return bytes(body)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


async def _collect_non_streaming(
    request: Any,
    gateway: GatewayService,
    session: GatewaySession,
    poll_seconds: float,
) -> tuple[str, GatewayTerminal]:
    chunks: list[str] = []
    terminal: Optional[GatewayTerminal] = None
    detached = False
    try:
        while True:
            if await request.is_disconnected():
                gateway.detach_cancel_and_drain(session, "http_disconnect")
                detached = True
                raise GatewayError(
                    "HTTP client disconnected",
                    http_status=499,
                    error_type="server_error",
                    code="client_disconnected",
                )
            try:
                event = await session.next_event(poll_seconds)
            except TimeoutError:
                continue
            if isinstance(event, GatewayTextDelta):
                chunks.append(event.text)
                continue
            terminal = event
            return "".join(chunks), terminal
    except asyncio.CancelledError:
        gateway.detach_cancel_and_drain(session, "http_disconnect")
        detached = True
        raise
    finally:
        if terminal is None and not detached:
            gateway.detach_cancel_and_drain(session, "http_disconnect")


async def _stream_chat_completion(
    request: Any,
    gateway: GatewayService,
    session: GatewaySession,
    completion_id: str,
    created: int,
    model: str,
    include_usage: bool,
    poll_seconds: float,
) -> AsyncIterator[bytes]:
    completed = False
    try:
        yield encode_sse(
            chat_completion_chunk_json(
                completion_id,
                created,
                model,
                delta={"role": "assistant"},
            )
        )
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await session.next_event(poll_seconds)
            except TimeoutError:
                continue
            if isinstance(event, GatewayTextDelta):
                yield encode_sse(
                    chat_completion_chunk_json(
                        completion_id,
                        created,
                        model,
                        delta={"content": event.text},
                    )
                )
                continue

            completed = True
            if not event.status.ok:
                yield encode_sse(error_json(gateway_error_from_status(event.status)))
                yield b"data: [DONE]\n\n"
                return
            yield encode_sse(
                chat_completion_chunk_json(
                    completion_id,
                    created,
                    model,
                    delta={},
                    finish_reason=openai_finish_reason(event.finish_reason),
                )
            )
            if include_usage:
                yield encode_sse(
                    usage_chunk_json(
                        completion_id,
                        created,
                        model,
                        event.usage,
                    )
                )
            yield b"data: [DONE]\n\n"
            return
    except asyncio.CancelledError:
        raise
    finally:
        if not completed:
            gateway.detach_cancel_and_drain(session, "http_disconnect")


__all__ = ["build_gateway_service", "create_app"]
