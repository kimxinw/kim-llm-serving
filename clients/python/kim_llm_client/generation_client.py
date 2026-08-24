from __future__ import annotations

import collections
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from .protocol import (
    DEFAULT_MAX_FRAME_PAYLOAD_BYTES,
    Accepted,
    GenerationEvent,
    GenerationRequest,
    HelloAck,
    ModelManifest,
    ProtocolError,
    Rejected,
    Stats,
    Status,
    Terminal,
    TokenDelta,
    Usage,
    WorkerLimits,
    decode_server_message,
    encode_cancel,
    encode_frame,
    encode_health,
    encode_hello,
    encode_submit,
)


class GenerationClientError(RuntimeError):
    pass


class WorkerUnavailableError(GenerationClientError):
    pass


class ClientCapacityError(GenerationClientError):
    pass


class RequestRejectedError(GenerationClientError):
    def __init__(self, request_id: int, status: Status) -> None:
        super().__init__(
            f"request {request_id} was rejected: {status.code}: {status.message}"
        )
        self.request_id = request_id
        self.status = status


@dataclass(frozen=True)
class GenerationClientConfig:
    socket_path: str
    expected_manifest: ModelManifest
    connect_timeout: float = 2.0
    handshake_timeout: float = 2.0
    max_frame_payload_bytes: int = DEFAULT_MAX_FRAME_PAYLOAD_BYTES
    max_pending_requests: int = 8
    max_delta_events_per_request: int = 64


class _EventBuffer:
    def __init__(self, max_delta_events: int) -> None:
        self._max_delta_events = max_delta_events
        self._condition = threading.Condition()
        self._deltas: collections.deque[TokenDelta] = collections.deque()
        self._terminal: Optional[Terminal] = None
        self._terminal_delivered = False
        self._overflowed = False

    @property
    def overflowed(self) -> bool:
        with self._condition:
            return self._overflowed

    def put_delta(self, delta: TokenDelta) -> bool:
        with self._condition:
            if self._terminal is not None:
                return False
            if len(self._deltas) >= self._max_delta_events:
                self._overflowed = True
                return False
            self._deltas.append(delta)
            self._condition.notify_all()
            return True

    def put_terminal(self, terminal: Terminal) -> bool:
        with self._condition:
            if self._terminal is not None:
                return False
            self._terminal = terminal
            self._condition.notify_all()
            return True

    def get(self, timeout: Optional[float]) -> GenerationEvent:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._deltas and self._terminal is None:
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for generation event")
                self._condition.wait(remaining)

            if self._deltas:
                return self._deltas.popleft()
            if self._terminal_delivered:
                raise StopIteration
            self._terminal_delivered = True
            assert self._terminal is not None
            return self._terminal


@dataclass
class _RequestState:
    request_id: int
    events: _EventBuffer
    accepted: bool = False
    rejected: Optional[Status] = None
    failure: Optional[Status] = None
    next_sequence_no: int = 0
    terminal_received: bool = False
    cancel_sent: bool = False


class GenerationHandle:
    def __init__(
        self,
        request_id: int,
        events: _EventBuffer,
        cancel_callback: Callable[[int], bool],
    ) -> None:
        self._request_id = request_id
        self._events = events
        self._cancel_callback = cancel_callback

    @property
    def request_id(self) -> int:
        return self._request_id

    @property
    def overflowed(self) -> bool:
        return self._events.overflowed

    def cancel(self) -> bool:
        return self._cancel_callback(self._request_id)

    def next_event(self, timeout: Optional[float] = None) -> GenerationEvent:
        return self._events.get(timeout)

    def iter_events(self, timeout: Optional[float] = None) -> Iterator[GenerationEvent]:
        while True:
            event = self.next_event(timeout)
            yield event
            if isinstance(event, Terminal):
                return

    def collect(
        self,
        timeout: Optional[float] = None,
    ) -> tuple[tuple[int, ...], Terminal]:
        deadline = None if timeout is None else time.monotonic() + timeout
        tokens: list[int] = []
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("timed out collecting generation result")
            event = self.next_event(remaining)
            if isinstance(event, TokenDelta):
                tokens.extend(event.token_ids)
                continue
            return tuple(tokens), event


class GenerationClient:
    def __init__(self, config: GenerationClientConfig) -> None:
        self._validate_config(config)
        self._config = config
        self._lifecycle_lock = threading.Lock()
        self._state_condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._socket: Optional[socket.socket] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._hello_ack: Optional[HelloAck] = None
        self._connected = False
        self._requests: dict[int, _RequestState] = {}
        self._probes: dict[int, queue.Queue[Optional[Stats]]] = {}
        self._next_probe_id = 1

    def __enter__(self) -> "GenerationClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        with self._state_condition:
            return self._connected

    @property
    def worker_epoch(self) -> int:
        with self._state_condition:
            if self._hello_ack is None:
                raise WorkerUnavailableError("GenerationClient is not connected")
            return self._hello_ack.worker_epoch

    @property
    def manifest(self) -> ModelManifest:
        with self._state_condition:
            if self._hello_ack is None:
                raise WorkerUnavailableError("GenerationClient is not connected")
            return self._hello_ack.manifest

    @property
    def limits(self) -> WorkerLimits:
        with self._state_condition:
            if self._hello_ack is None:
                raise WorkerUnavailableError("GenerationClient is not connected")
            return self._hello_ack.limits

    def connect(self) -> None:
        with self._lifecycle_lock:
            with self._state_condition:
                if (
                    self._connected
                    or self._socket is not None
                    or (
                        self._reader_thread is not None
                        and self._reader_thread.is_alive()
                    )
                ):
                    raise GenerationClientError("GenerationClient is already connected")

            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(self._config.connect_timeout)
                connection.connect(self._config.socket_path)
                connection.settimeout(self._config.handshake_timeout)
                self._send_direct(
                    connection,
                    encode_hello(self._config.expected_manifest),
                    self._config.max_frame_payload_bytes,
                )
                payload = self._receive_frame(
                    connection,
                    self._config.max_frame_payload_bytes,
                )
                message = decode_server_message(payload, expected_epoch=None)
                if not isinstance(message, HelloAck):
                    raise ProtocolError("Worker did not reply with HelloAck")
                if message.manifest != self._config.expected_manifest:
                    raise ProtocolError("Worker ModelManifest does not match Gateway")
                if (
                    message.limits.max_frame_payload_bytes
                    > self._config.max_frame_payload_bytes
                ):
                    raise ProtocolError(
                        "Worker frame limit exceeds GenerationClient capacity"
                    )
                connection.settimeout(None)
            except Exception:
                connection.close()
                raise

            with self._state_condition:
                self._socket = connection
                self._hello_ack = message
                self._connected = True

            try:
                reader = threading.Thread(
                    target=self._reader_loop,
                    name="kim-llm-generation-client-reader",
                    daemon=True,
                )
                self._reader_thread = reader
                reader.start()
            except Exception as exception:
                self._fail(
                    Status("unavailable", f"failed to start reader: {exception}"),
                    join_reader=False,
                )
                raise

    def submit(
        self,
        request: GenerationRequest,
        timeout: Optional[float] = None,
    ) -> GenerationHandle:
        with self._state_condition:
            hello_ack = self._require_connected_locked()
            self._validate_request_against_worker(request, hello_ack)
            if request.request_id in self._requests:
                raise GenerationClientError(
                    f"request {request.request_id} is already active"
                )
            request_limit = min(
                self._config.max_pending_requests,
                hello_ack.limits.max_active_requests,
            )
            if len(self._requests) >= request_limit:
                raise ClientCapacityError(
                    "GenerationClient request capacity is full"
                )
            state = _RequestState(
                request.request_id,
                _EventBuffer(self._config.max_delta_events_per_request),
            )
            self._requests[request.request_id] = state
            epoch = hello_ack.worker_epoch

        try:
            self._send_payload(encode_submit(request, epoch))
        except Exception:
            with self._state_condition:
                self._requests.pop(request.request_id, None)
            raise

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_condition:
            while not state.accepted and state.rejected is None and state.failure is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._state_condition.wait(remaining)

            if state.accepted:
                return GenerationHandle(
                    request.request_id,
                    state.events,
                    self.cancel,
                )
            if state.rejected is not None:
                raise RequestRejectedError(request.request_id, state.rejected)
            if state.failure is not None:
                raise WorkerUnavailableError(state.failure.message)

        failure = Status(
            "unavailable",
            f"timed out waiting for request {request.request_id} acceptance",
        )
        self._fail(failure)
        raise WorkerUnavailableError(failure.message)

    def cancel(self, request_id: int) -> bool:
        with self._state_condition:
            hello_ack = self._require_connected_locked()
            state = self._requests.get(request_id)
            if state is None or not state.accepted or state.terminal_received:
                return False
            if state.cancel_sent:
                return True
            state.cancel_sent = True
            epoch = hello_ack.worker_epoch
        self._send_payload(encode_cancel(request_id, epoch))
        return True

    def health(self, timeout: float = 1.0) -> Stats:
        if timeout <= 0:
            raise ValueError("health timeout must be positive")
        response: queue.Queue[Optional[Stats]] = queue.Queue(maxsize=1)
        with self._state_condition:
            hello_ack = self._require_connected_locked()
            probe_id = self._next_probe_id
            self._next_probe_id += 1
            self._probes[probe_id] = response
            epoch = hello_ack.worker_epoch
        try:
            self._send_payload(encode_health(probe_id, epoch))
            result = response.get(timeout=timeout)
        except queue.Empty as exception:
            raise TimeoutError("timed out waiting for Worker Stats") from exception
        finally:
            with self._state_condition:
                self._probes.pop(probe_id, None)
        if result is None:
            raise WorkerUnavailableError("Worker disconnected during health probe")
        return result

    def close(self) -> None:
        with self._lifecycle_lock:
            self._fail(
                Status("unavailable", "GenerationClient closed"),
                join_reader=True,
            )

    def _reader_loop(self) -> None:
        try:
            while True:
                with self._state_condition:
                    if not self._connected or self._socket is None:
                        return
                    connection = self._socket
                    hello_ack = self._hello_ack
                assert hello_ack is not None
                payload = self._receive_frame(
                    connection,
                    hello_ack.limits.max_frame_payload_bytes,
                )
                message = decode_server_message(
                    payload,
                    expected_epoch=hello_ack.worker_epoch,
                )
                self._dispatch(message)
        except ProtocolError as exception:
            self._fail(
                Status("unavailable", f"IPC protocol failure: {exception}"),
                join_reader=False,
            )
        except (EOFError, OSError) as exception:
            self._fail(
                Status("unavailable", f"Worker IPC disconnected: {exception}"),
                join_reader=False,
            )
        except Exception as exception:
            self._fail(
                Status("unavailable", f"GenerationClient reader failed: {exception}"),
                join_reader=False,
            )

    def _dispatch(self, message: object) -> None:
        send_cancel: Optional[tuple[int, int]] = None
        terminal_to_deliver: Optional[tuple[_EventBuffer, Terminal]] = None
        with self._state_condition:
            hello_ack = self._require_connected_locked()
            if isinstance(message, Stats):
                response = self._probes.get(message.probe_id)
                if response is not None:
                    try:
                        response.put_nowait(message)
                    except queue.Full:
                        pass
                return
            if isinstance(message, HelloAck):
                raise ProtocolError("duplicate HelloAck")
            if not isinstance(message, (Accepted, Rejected, TokenDelta, Terminal)):
                raise ProtocolError("unsupported Worker message")

            state = self._requests.get(message.request_id)
            if state is None:
                raise ProtocolError(
                    f"message references unknown request {message.request_id}"
                )

            if isinstance(message, Accepted):
                if state.accepted or state.rejected is not None:
                    raise ProtocolError("request received duplicate acceptance")
                state.accepted = True
                self._state_condition.notify_all()
                return

            if isinstance(message, Rejected):
                if state.accepted:
                    raise ProtocolError("Accepted request was later Rejected")
                state.rejected = message.status
                self._requests.pop(message.request_id, None)
                self._state_condition.notify_all()
                return

            if not state.accepted:
                raise ProtocolError("request event arrived before Accepted")
            if state.terminal_received:
                raise ProtocolError("request event arrived after Terminal")

            if isinstance(message, TokenDelta):
                if message.sequence_no != state.next_sequence_no:
                    raise ProtocolError("TokenDelta sequence is not contiguous")
                state.next_sequence_no += 1
                if not state.events.put_delta(message) and not state.cancel_sent:
                    state.cancel_sent = True
                    send_cancel = (message.request_id, hello_ack.worker_epoch)
            else:
                state.terminal_received = True
                self._requests.pop(message.request_id, None)
                terminal_to_deliver = (state.events, message)

        if terminal_to_deliver is not None:
            events, terminal = terminal_to_deliver
            if not events.put_terminal(terminal):
                raise ProtocolError("request received duplicate Terminal")
        if send_cancel is not None:
            request_id, epoch = send_cancel
            self._send_payload(encode_cancel(request_id, epoch))

    def _send_payload(self, payload: bytes) -> None:
        with self._send_lock:
            with self._state_condition:
                hello_ack = self._require_connected_locked()
                assert self._socket is not None
                connection = self._socket
                max_payload = hello_ack.limits.max_frame_payload_bytes
            try:
                self._send_direct(connection, payload, max_payload)
            except OSError as exception:
                failure = Status("unavailable", f"Worker IPC write failed: {exception}")
                self._fail(failure, join_reader=False)
                raise WorkerUnavailableError(failure.message) from exception

    def _fail(self, status: Status, *, join_reader: bool = True) -> None:
        with self._state_condition:
            connection = self._socket
            reader = self._reader_thread
            self._socket = None
            self._connected = False
            self._hello_ack = None
            requests = list(self._requests.values())
            probes = list(self._probes.values())
            self._requests.clear()
            self._probes.clear()
            for request in requests:
                request.failure = status
            self._state_condition.notify_all()

        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

        for request in requests:
            if request.accepted and not request.terminal_received:
                request.events.put_terminal(
                    Terminal(
                        request.request_id,
                        Status("unavailable", status.message),
                        None,
                        Usage(),
                    )
                )
        for response in probes:
            try:
                response.put_nowait(None)
            except queue.Full:
                pass

        if (
            join_reader
            and reader is not None
            and reader.is_alive()
            and reader is not threading.current_thread()
        ):
            reader.join()
        if reader is not None and not reader.is_alive():
            with self._state_condition:
                if self._reader_thread is reader:
                    self._reader_thread = None

    def _require_connected_locked(self) -> HelloAck:
        if not self._connected or self._socket is None or self._hello_ack is None:
            raise WorkerUnavailableError("GenerationClient is not connected")
        return self._hello_ack

    def _validate_request_against_worker(
        self,
        request: GenerationRequest,
        hello_ack: HelloAck,
    ) -> None:
        manifest = hello_ack.manifest
        if len(request.input_token_ids) > manifest.max_input_tokens:
            raise ValueError("request input exceeds Worker max_input_tokens")
        if request.max_new_tokens > manifest.max_output_tokens:
            raise ValueError("request output exceeds Worker max_output_tokens")
        if (
            len(request.input_token_ids) + request.max_new_tokens
            > manifest.max_sequence_tokens
        ):
            raise ValueError("request exceeds Worker max_sequence_tokens")

    @staticmethod
    def _validate_config(config: GenerationClientConfig) -> None:
        if not config.socket_path:
            raise ValueError("socket_path must not be empty")
        if config.connect_timeout <= 0 or config.handshake_timeout <= 0:
            raise ValueError("connect and handshake timeouts must be positive")
        if (
            config.max_frame_payload_bytes <= 0
            or config.max_pending_requests <= 0
            or config.max_delta_events_per_request <= 0
        ):
            raise ValueError("GenerationClient capacities must be positive")

    @staticmethod
    def _send_direct(
        connection: socket.socket,
        payload: bytes,
        max_payload_bytes: int,
    ) -> None:
        connection.sendall(encode_frame(payload, max_payload_bytes))

    @staticmethod
    def _receive_frame(
        connection: socket.socket,
        max_payload_bytes: int,
    ) -> bytes:
        prefix = GenerationClient._receive_exact(connection, 4)
        payload_size = struct.unpack("!I", prefix)[0]
        if payload_size == 0:
            raise ProtocolError("frame payload length must be positive")
        if payload_size > max_payload_bytes:
            raise ProtocolError("frame payload length exceeds configured limit")
        return GenerationClient._receive_exact(connection, payload_size)

    @staticmethod
    def _receive_exact(connection: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                if data:
                    raise EOFError("Worker disconnected with a partial frame")
                raise EOFError("Worker disconnected")
            data.extend(chunk)
        return bytes(data)


__all__ = [
    "ClientCapacityError",
    "GenerationClient",
    "GenerationClientConfig",
    "GenerationClientError",
    "GenerationHandle",
    "RequestRejectedError",
    "WorkerUnavailableError",
]
