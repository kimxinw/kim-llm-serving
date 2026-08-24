from __future__ import annotations

import asyncio
import collections
import math
import secrets
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union

from kim_llm_client import (
    ClientCapacityError,
    GenerationClientError,
    GenerationHandle,
    GenerationRequest,
    ModelManifest,
    RequestRejectedError,
    SamplingParameters,
    Stats,
    Status,
    Terminal,
    TokenDelta,
    Usage,
    WorkerUnavailableError,
)

from .tokenizer import Tokenizer, TokenizerError


class GatewayError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int,
        error_type: str,
        code: str,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.error_type = error_type
        self.code = code
        self.param = param


def gateway_error_from_status(status: Status) -> GatewayError:
    http_status = {
        "invalid_input": 400,
        "invalid_shape": 400,
        "already_exists": 409,
        "not_ready": 503,
        "resource_exhausted": 429,
        "queue_full": 429,
        "unavailable": 503,
        "engine_not_found": 503,
        "context_unavailable": 503,
        "timeout": 504,
        "cancelled": 499,
    }.get(status.code, 500)
    error_type = (
        "invalid_request_error"
        if http_status in {400, 409, 429}
        else "server_error"
    )
    message = status.message or f"generation failed with status {status.code}"
    return GatewayError(
        message,
        http_status=http_status,
        error_type=error_type,
        code=status.code,
    )


@dataclass(frozen=True)
class GatewayRuntimeOptions:
    accept_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 120.0
    health_timeout_seconds: float = 1.0
    cancel_drain_timeout_seconds: float = 30.0
    shutdown_grace_seconds: float = 30.0
    max_sse_delta_events_per_request: int = 32
    sampling_top_k: int = 1

    def __post_init__(self) -> None:
        numeric = (
            self.accept_timeout_seconds,
            self.request_timeout_seconds,
            self.health_timeout_seconds,
            self.cancel_drain_timeout_seconds,
            self.shutdown_grace_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("Gateway Runtime timeouts must be finite and positive")
        if self.max_sse_delta_events_per_request <= 0:
            raise ValueError("Gateway SSE capacity must be positive")
        if self.sampling_top_k <= 0 or self.sampling_top_k > (1 << 31) - 1:
            raise ValueError("Gateway sampling_top_k is outside the supported range")


@dataclass(frozen=True)
class GatewayRequest:
    model: str
    messages: tuple[Mapping[str, str], ...]
    max_new_tokens: int
    streaming: bool
    temperature: float = 1.0
    top_p: float = 1.0
    random_seed: int = 0
    stop: tuple[str, ...] = ()
    trace_id: str = ""


@dataclass(frozen=True)
class GatewayTextDelta:
    text: str


@dataclass(frozen=True)
class GatewayTerminal:
    status: Status
    finish_reason: Optional[str]
    usage: Usage


GatewayEvent = Union[GatewayTextDelta, GatewayTerminal]


class GenerationClientLike(Protocol):
    @property
    def connected(self) -> bool:
        ...

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def health(self, timeout: float = 1.0) -> Stats:
        ...

    def submit(
        self,
        request: GenerationRequest,
        timeout: Optional[float] = None,
    ) -> GenerationHandle:
        ...


class GatewayMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offered = 0
        self._accepted = 0
        self._active = 0
        self._rejected: collections.Counter[str] = collections.Counter()
        self._terminals: collections.Counter[str] = collections.Counter()
        self._disconnect_cancels = 0
        self._backpressure_cancels = 0
        self._drain_timeouts = 0
        self._sse_buffered_events = 0
        self._sse_buffer_high_watermark = 0

    def offered(self) -> None:
        with self._lock:
            self._offered += 1

    def accepted(self) -> None:
        with self._lock:
            self._accepted += 1
            self._active += 1

    def rejected(self, code: str) -> None:
        with self._lock:
            self._rejected[code] += 1

    def terminal(self, code: str) -> None:
        with self._lock:
            self._terminals[code] += 1
            self._active = max(0, self._active - 1)

    def cancel(self, reason: str) -> None:
        with self._lock:
            if reason == "http_disconnect":
                self._disconnect_cancels += 1
            elif reason == "sse_backpressure":
                self._backpressure_cancels += 1

    def drain_timeout(self) -> None:
        with self._lock:
            self._drain_timeouts += 1

    def sse_delta_enqueued(self) -> None:
        with self._lock:
            self._sse_buffered_events += 1
            self._sse_buffer_high_watermark = max(
                self._sse_buffer_high_watermark,
                self._sse_buffered_events,
            )

    def sse_deltas_released(self, count: int) -> None:
        with self._lock:
            self._sse_buffered_events = max(
                0,
                self._sse_buffered_events - count,
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "offered": self._offered,
                "accepted": self._accepted,
                "active": self._active,
                "rejected": dict(self._rejected),
                "terminals": dict(self._terminals),
                "disconnect_cancels": self._disconnect_cancels,
                "backpressure_cancels": self._backpressure_cancels,
                "drain_timeouts": self._drain_timeouts,
                "sse_buffered_events": self._sse_buffered_events,
                "sse_buffer_high_watermark": self._sse_buffer_high_watermark,
            }

    def render_prometheus(self, *, connected: bool, ready: bool) -> str:
        snapshot = self.snapshot()
        lines = [
            "# HELP kim_llm_gateway_connected Whether the IPC client is connected.",
            "# TYPE kim_llm_gateway_connected gauge",
            f"kim_llm_gateway_connected {1 if connected else 0}",
            "# HELP kim_llm_gateway_ready Whether the Worker passed readiness.",
            "# TYPE kim_llm_gateway_ready gauge",
            f"kim_llm_gateway_ready {1 if ready else 0}",
            "# TYPE kim_llm_gateway_requests_offered_total counter",
            f"kim_llm_gateway_requests_offered_total {snapshot['offered']}",
            "# TYPE kim_llm_gateway_requests_accepted_total counter",
            f"kim_llm_gateway_requests_accepted_total {snapshot['accepted']}",
            "# TYPE kim_llm_gateway_active_requests gauge",
            f"kim_llm_gateway_active_requests {snapshot['active']}",
            "# TYPE kim_llm_gateway_http_disconnect_cancels_total counter",
            "kim_llm_gateway_http_disconnect_cancels_total "
            f"{snapshot['disconnect_cancels']}",
            "# TYPE kim_llm_gateway_sse_backpressure_cancels_total counter",
            "kim_llm_gateway_sse_backpressure_cancels_total "
            f"{snapshot['backpressure_cancels']}",
            "# TYPE kim_llm_gateway_terminal_drain_timeouts_total counter",
            "kim_llm_gateway_terminal_drain_timeouts_total "
            f"{snapshot['drain_timeouts']}",
            "# TYPE kim_llm_gateway_sse_buffered_events gauge",
            "kim_llm_gateway_sse_buffered_events "
            f"{snapshot['sse_buffered_events']}",
            "# TYPE kim_llm_gateway_sse_buffer_high_watermark gauge",
            "kim_llm_gateway_sse_buffer_high_watermark "
            f"{snapshot['sse_buffer_high_watermark']}",
        ]
        rejected = snapshot["rejected"]
        assert isinstance(rejected, dict)
        lines.append("# TYPE kim_llm_gateway_requests_rejected_total counter")
        for code, value in sorted(rejected.items()):
            lines.append(
                f'kim_llm_gateway_requests_rejected_total{{code="{code}"}} {value}'
            )
        terminals = snapshot["terminals"]
        assert isinstance(terminals, dict)
        lines.append("# TYPE kim_llm_gateway_terminals_total counter")
        for code, value in sorted(terminals.items()):
            lines.append(
                f'kim_llm_gateway_terminals_total{{code="{code}"}} {value}'
            )
        return "\n".join(lines) + "\n"


class _AsyncEventBuffer:
    def __init__(
        self,
        max_delta_events: int,
        metrics: GatewayMetrics,
    ) -> None:
        self._max_delta_events = max_delta_events
        self._metrics = metrics
        self._condition = asyncio.Condition()
        self._deltas: collections.deque[GatewayTextDelta] = collections.deque()
        self._terminal: Optional[GatewayTerminal] = None
        self._terminal_delivered = False
        self._abandoned = False

    async def put_delta(self, delta: GatewayTextDelta) -> bool:
        async with self._condition:
            if self._terminal is not None or self._abandoned:
                return False
            if len(self._deltas) >= self._max_delta_events:
                return False
            self._deltas.append(delta)
            self._metrics.sse_delta_enqueued()
            self._condition.notify_all()
            return True

    async def put_terminal(self, terminal: GatewayTerminal) -> bool:
        async with self._condition:
            if self._terminal is not None:
                return False
            self._terminal = terminal
            self._condition.notify_all()
            return True

    async def get(self, timeout: Optional[float]) -> GatewayEvent:
        async def wait_for_event() -> GatewayEvent:
            async with self._condition:
                while not self._deltas and self._terminal is None:
                    await self._condition.wait()
                if self._deltas:
                    result = self._deltas.popleft()
                    self._metrics.sse_deltas_released(1)
                    return result
                if self._terminal_delivered:
                    raise StopAsyncIteration
                self._terminal_delivered = True
                assert self._terminal is not None
                return self._terminal

        if timeout is None:
            return await wait_for_event()
        try:
            return await asyncio.wait_for(wait_for_event(), timeout)
        except asyncio.TimeoutError as exception:
            raise TimeoutError("timed out waiting for Gateway event") from exception

    async def abandon(self) -> None:
        async with self._condition:
            self._abandoned = True
            released = len(self._deltas)
            self._deltas.clear()
            if released:
                self._metrics.sse_deltas_released(released)
            self._condition.notify_all()


class _IncrementalDecoder:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._tokens: list[int] = []
        self._decoded = ""

    def append(self, token_ids: Sequence[int]) -> str:
        self._tokens.extend(token_ids)
        decoded = self._tokenizer.decode(self._tokens)
        if not decoded.startswith(self._decoded):
            raise TokenizerError(
                "incremental decode changed text that was already emitted"
            )
        delta = decoded[len(self._decoded) :]
        self._decoded = decoded
        return delta


class GatewaySession:
    def __init__(
        self,
        handle: GenerationHandle,
        tokenizer: Tokenizer,
        metrics: GatewayMetrics,
        max_delta_events: int,
        on_finished: Callable[["GatewaySession"], None],
    ) -> None:
        self._handle = handle
        self._decoder = _IncrementalDecoder(tokenizer)
        self._metrics = metrics
        self._buffer = _AsyncEventBuffer(max_delta_events, metrics)
        self._on_finished = on_finished
        self._cancel_lock = asyncio.Lock()
        self._cancel_sent = False
        self._done = asyncio.Event()
        self._pump_task: Optional[asyncio.Task[None]] = None
        self._terminal: Optional[GatewayTerminal] = None
        self._dropping_deltas = False

    @property
    def request_id(self) -> int:
        return self._handle.request_id

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def terminal(self) -> Optional[GatewayTerminal]:
        return self._terminal

    def start(self) -> None:
        if self._pump_task is not None:
            raise RuntimeError("Gateway Session pump already started")
        self._pump_task = asyncio.create_task(
            self._pump(),
            name=f"kim-llm-gateway-request-{self.request_id}",
        )

    async def next_event(self, timeout: Optional[float] = None) -> GatewayEvent:
        return await self._buffer.get(timeout)

    async def cancel(self, reason: str) -> bool:
        async with self._cancel_lock:
            if self._done.is_set():
                return False
            if self._cancel_sent:
                return True
            try:
                sent = await asyncio.to_thread(self._handle.cancel)
            except GenerationClientError:
                sent = False
            self._cancel_sent = sent
            if sent:
                self._metrics.cancel(reason)
            return sent

    async def cancel_and_drain(self, reason: str, timeout: float) -> bool:
        await self.cancel(reason)
        try:
            await asyncio.wait_for(self._done.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            self._metrics.drain_timeout()
            return False

    async def wait_done(self, timeout: Optional[float] = None) -> None:
        if timeout is None:
            await self._done.wait()
            return
        await asyncio.wait_for(self._done.wait(), timeout)

    async def abandon(self) -> None:
        await self._buffer.abandon()

    async def _pump(self) -> None:
        local_failure: Optional[Status] = None
        try:
            while True:
                event = await asyncio.to_thread(self._handle.next_event, None)
                if isinstance(event, TokenDelta):
                    if self._dropping_deltas:
                        continue
                    try:
                        text = self._decoder.append(event.token_ids)
                    except TokenizerError as exception:
                        local_failure = Status("internal_error", str(exception))
                        self._dropping_deltas = True
                        await self.cancel("decode_failure")
                        continue
                    if not text:
                        continue
                    accepted = await self._buffer.put_delta(GatewayTextDelta(text))
                    if not accepted:
                        self._dropping_deltas = True
                        await self.cancel("sse_backpressure")
                    continue

                assert isinstance(event, Terminal)
                terminal = GatewayTerminal(
                    local_failure or event.status,
                    None if local_failure is not None else event.finish_reason,
                    event.usage,
                )
                await self._finish(terminal)
                return
        except StopIteration:
            await self._finish(
                GatewayTerminal(
                    Status("internal_error", "generation ended without Terminal"),
                    None,
                    Usage(),
                )
            )
        except Exception as exception:
            code = (
                "unavailable"
                if isinstance(exception, WorkerUnavailableError)
                else "internal_error"
            )
            await self._finish(
                GatewayTerminal(Status(code, str(exception)), None, Usage())
            )
        finally:
            self._done.set()
            self._on_finished(self)

    async def _finish(self, terminal: GatewayTerminal) -> None:
        self._terminal = terminal
        if not await self._buffer.put_terminal(terminal):
            raise RuntimeError("Gateway Session received duplicate Terminal")
        self._metrics.terminal(terminal.status.code)


class _RequestIdAllocator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = secrets.randbits(62) + 1

    def next(self) -> int:
        with self._lock:
            result = self._next
            self._next += 1
            if self._next >= (1 << 63):
                self._next = 1
            return result


class GatewayService:
    def __init__(
        self,
        client: GenerationClientLike,
        tokenizer: Tokenizer,
        manifest: ModelManifest,
        options: GatewayRuntimeOptions,
        metrics: Optional[GatewayMetrics] = None,
    ) -> None:
        self._client = client
        self._tokenizer = tokenizer
        self._manifest = manifest
        self._options = options
        self._metrics = metrics or GatewayMetrics()
        self._request_ids = _RequestIdAllocator()
        self._connection_lock = asyncio.Lock()
        self._active: dict[int, GatewaySession] = {}
        self._detached_tasks: set[asyncio.Task[bool]] = set()
        self._stopping = False
        self._last_readiness_error: Optional[str] = "Gateway has not started"
        self._ready = False

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    @property
    def metrics(self) -> GatewayMetrics:
        return self._metrics

    @property
    def connected(self) -> bool:
        return self._client.connected

    @property
    def ready(self) -> bool:
        return self._ready and self._client.connected

    @property
    def last_readiness_error(self) -> Optional[str]:
        return self._last_readiness_error

    async def start(self) -> bool:
        self._stopping = False
        try:
            await self.check_readiness()
            return True
        except GatewayError:
            return False

    async def check_readiness(self) -> Stats:
        if self._stopping:
            raise GatewayError(
                "Gateway is stopping",
                http_status=503,
                error_type="server_error",
                code="not_ready",
            )
        async with self._connection_lock:
            try:
                if not self._client.connected:
                    await asyncio.to_thread(self._client.connect)
                stats = await asyncio.to_thread(
                    self._client.health,
                    self._options.health_timeout_seconds,
                )
                if not stats.ready or not stats.status.ok:
                    raise GatewayError(
                        stats.status.message or "Worker is not ready",
                        http_status=503,
                        error_type="server_error",
                        code=(
                            stats.status.code
                            if not stats.status.ok
                            else "not_ready"
                        ),
                    )
            except GatewayError as exception:
                self._ready = False
                self._last_readiness_error = str(exception)
                raise
            except Exception as exception:
                self._ready = False
                self._last_readiness_error = str(exception)
                raise GatewayError(
                    f"Worker readiness failed: {exception}",
                    http_status=503,
                    error_type="server_error",
                    code="unavailable",
                ) from exception
            self._ready = True
            self._last_readiness_error = None
            return stats

    async def submit(self, request: GatewayRequest) -> GatewaySession:
        self._metrics.offered()
        if self._stopping:
            self._metrics.rejected("not_ready")
            raise GatewayError(
                "Gateway is stopping",
                http_status=503,
                error_type="server_error",
                code="not_ready",
            )
        if request.model != self._manifest.model_id:
            self._metrics.rejected("model_not_found")
            raise GatewayError(
                f"model {request.model!r} is not served by this Gateway",
                http_status=404,
                error_type="invalid_request_error",
                code="model_not_found",
                param="model",
            )
        if not self.ready:
            try:
                await self.check_readiness()
            except GatewayError as exception:
                self._metrics.rejected(exception.code)
                raise

        try:
            input_token_ids = self._tokenizer.encode_chat(request.messages)
            stop_sequences = tuple(
                self._tokenizer.encode_text(value) for value in request.stop
            )
        except TokenizerError as exception:
            self._metrics.rejected("invalid_input")
            raise GatewayError(
                str(exception),
                http_status=400,
                error_type="invalid_request_error",
                code="invalid_input",
                param="messages",
            ) from exception
        if any(not sequence for sequence in stop_sequences):
            self._metrics.rejected("invalid_input")
            raise GatewayError(
                "stop sequence must produce at least one Token ID",
                http_status=400,
                error_type="invalid_request_error",
                code="invalid_input",
                param="stop",
            )
        if len(input_token_ids) > self._manifest.max_input_tokens:
            self._metrics.rejected("invalid_input")
            raise GatewayError(
                "tokenized prompt exceeds max_input_tokens",
                http_status=400,
                error_type="invalid_request_error",
                code="invalid_input",
                param="messages",
            )
        if request.max_new_tokens > self._manifest.max_output_tokens:
            self._metrics.rejected("invalid_input")
            raise GatewayError(
                "max_tokens exceeds the model output limit",
                http_status=400,
                error_type="invalid_request_error",
                code="invalid_input",
                param="max_tokens",
            )
        if len(input_token_ids) + request.max_new_tokens > self._manifest.max_sequence_tokens:
            self._metrics.rejected("invalid_input")
            raise GatewayError(
                "prompt and requested output exceed max_sequence_tokens",
                http_status=400,
                error_type="invalid_request_error",
                code="invalid_input",
                param="max_tokens",
            )

        request_id = self._request_ids.next()
        generation_request = GenerationRequest(
            request_id=request_id,
            input_token_ids=input_token_ids,
            max_new_tokens=request.max_new_tokens,
            timeout_ms=int(self._options.request_timeout_seconds * 1000),
            trace_id=request.trace_id,
            streaming=request.streaming,
            sampling=SamplingParameters(
                temperature=request.temperature,
                top_k=self._options.sampling_top_k,
                top_p=request.top_p,
                random_seed=request.random_seed,
            ),
            end_id=self._manifest.eos_token_id,
            pad_id=self._manifest.pad_token_id,
            stop_sequences=stop_sequences,
        )
        try:
            handle = await asyncio.to_thread(
                self._client.submit,
                generation_request,
                self._options.accept_timeout_seconds,
            )
        except RequestRejectedError as exception:
            self._metrics.rejected(exception.status.code)
            raise gateway_error_from_status(exception.status) from exception
        except ClientCapacityError as exception:
            self._metrics.rejected("queue_full")
            raise GatewayError(
                str(exception),
                http_status=429,
                error_type="invalid_request_error",
                code="queue_full",
            ) from exception
        except (WorkerUnavailableError, GenerationClientError) as exception:
            self._ready = False
            self._last_readiness_error = str(exception)
            self._metrics.rejected("unavailable")
            raise GatewayError(
                str(exception),
                http_status=503,
                error_type="server_error",
                code="unavailable",
            ) from exception
        except ValueError as exception:
            self._metrics.rejected("invalid_input")
            raise GatewayError(
                str(exception),
                http_status=400,
                error_type="invalid_request_error",
                code="invalid_input",
            ) from exception

        session = GatewaySession(
            handle,
            self._tokenizer,
            self._metrics,
            self._options.max_sse_delta_events_per_request,
            self._session_finished,
        )
        self._active[request_id] = session
        self._metrics.accepted()
        session.start()
        return session

    def detach_cancel_and_drain(self, session: GatewaySession, reason: str) -> None:
        task = asyncio.create_task(
            self._cancel_drain_and_abandon(session, reason),
            name=f"kim-llm-gateway-drain-{session.request_id}",
        )
        self._detached_tasks.add(task)
        task.add_done_callback(self._detached_tasks.discard)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self._ready = False
        sessions = list(self._active.values())
        if sessions:
            await asyncio.gather(
                *(
                    session.cancel_and_drain(
                        "gateway_shutdown",
                        self._options.shutdown_grace_seconds,
                    )
                    for session in sessions
                ),
                return_exceptions=True,
            )
        await asyncio.to_thread(self._client.close)
        for session in sessions:
            await session.abandon()
        if self._detached_tasks:
            await asyncio.gather(*tuple(self._detached_tasks), return_exceptions=True)

    def render_metrics(self) -> str:
        return self._metrics.render_prometheus(
            connected=self.connected,
            ready=self.ready,
        )

    def _session_finished(self, session: GatewaySession) -> None:
        current = self._active.get(session.request_id)
        if current is session:
            self._active.pop(session.request_id, None)

    async def _cancel_drain_and_abandon(
        self,
        session: GatewaySession,
        reason: str,
    ) -> bool:
        drained = await session.cancel_and_drain(
            reason,
            self._options.cancel_drain_timeout_seconds,
        )
        await session.abandon()
        return drained


__all__ = [
    "GatewayError",
    "GatewayEvent",
    "GatewayMetrics",
    "GatewayRequest",
    "GatewayRuntimeOptions",
    "GatewayService",
    "GatewaySession",
    "GatewayTerminal",
    "GatewayTextDelta",
    "gateway_error_from_status",
]
