from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
import time
import unittest
from typing import Any, Optional

from kim_llm_client import (
    ClientCapacityError,
    GenerationClient,
    GenerationClientConfig,
    GenerationRequest,
    ModelManifest,
    ProtocolError,
    RequestRejectedError,
    Terminal,
    WorkerLimits,
)
from kim_llm_client.protocol import PROTOCOL_VERSION, encode_frame, encode_json


def make_manifest() -> ModelManifest:
    return ModelManifest(
        model_id="tinyllama",
        revision="revision-1",
        tokenizer_fingerprint="tokenizer-sha",
        chat_template_fingerprint="template-sha",
        engine_fingerprint="engine-sha",
        eos_token_id=2,
        pad_token_id=0,
        max_input_tokens=128,
        max_output_tokens=64,
        max_sequence_tokens=192,
        precision="fp16",
        max_batch_size=2,
    )


def make_limits() -> WorkerLimits:
    return WorkerLimits(
        max_active_requests=2,
        max_total_input_tokens=64,
        max_reserved_output_tokens=64,
        max_frame_payload_bytes=1024 * 1024,
        max_session_egress_frames=128,
        max_session_egress_bytes=4 * 1024 * 1024,
        max_request_egress_frames=32,
        max_request_egress_bytes=2 * 1024 * 1024,
    )


def manifest_json(manifest: ModelManifest) -> dict[str, Any]:
    return {
        "model_id": manifest.model_id,
        "revision": manifest.revision,
        "tokenizer_fingerprint": manifest.tokenizer_fingerprint,
        "chat_template_fingerprint": manifest.chat_template_fingerprint,
        "engine_fingerprint": manifest.engine_fingerprint,
        "eos_token_id": manifest.eos_token_id,
        "pad_token_id": manifest.pad_token_id,
        "max_input_tokens": manifest.max_input_tokens,
        "max_output_tokens": manifest.max_output_tokens,
        "max_sequence_tokens": manifest.max_sequence_tokens,
        "precision": manifest.precision,
        "max_batch_size": manifest.max_batch_size,
    }


def limits_json(limits: WorkerLimits) -> dict[str, Any]:
    return {
        "max_active_requests": limits.max_active_requests,
        "max_total_input_tokens": limits.max_total_input_tokens,
        "max_reserved_output_tokens": limits.max_reserved_output_tokens,
        "max_frame_payload_bytes": limits.max_frame_payload_bytes,
        "max_session_egress_frames": limits.max_session_egress_frames,
        "max_session_egress_bytes": limits.max_session_egress_bytes,
        "max_request_egress_frames": limits.max_request_egress_frames,
        "max_request_egress_bytes": limits.max_request_egress_bytes,
    }


class FakeWorker:
    def __init__(
        self,
        manifest: ModelManifest,
        limits: WorkerLimits,
        *,
        reject_request_ids: set[int] | None = None,
    ) -> None:
        self.manifest = manifest
        self.limits = limits
        self.worker_epoch = 77
        self.reject_request_ids = reject_request_ids or set()
        self.directory = tempfile.TemporaryDirectory(prefix="kim-llm-client-")
        self.socket_path = os.path.join(self.directory.name, "worker.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.ready = threading.Event()
        self.messages: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.send_lock = threading.Lock()

    def start(self) -> None:
        self.listener.bind(self.socket_path)
        self.listener.listen(1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.disconnect()
        self.listener.close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.directory.cleanup()

    def disconnect(self) -> None:
        connection = self.connection
        self.connection = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def send(self, message: dict[str, Any]) -> None:
        connection = self.connection
        if connection is None:
            raise RuntimeError("FakeWorker has no active connection")
        frame = encode_frame(
            encode_json(message),
            self.limits.max_frame_payload_bytes,
        )
        with self.send_lock:
            connection.sendall(frame)

    def wait_for_message(
        self,
        message_type: str,
        *,
        request_id: Optional[int] = None,
        timeout: float = 1.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                for message in self.messages:
                    if message.get("type") != message_type:
                        continue
                    if request_id is not None and message.get("request_id") != request_id:
                        continue
                    return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"missing {message_type} message")
                self.condition.wait(remaining)

    def _run(self) -> None:
        try:
            connection, _ = self.listener.accept()
            self.connection = connection
            hello = self._receive_json(connection)
            if hello.get("type") != "hello":
                return
            self.send(
                {
                    "type": "hello_ack",
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_epoch": self.worker_epoch,
                    "manifest": manifest_json(self.manifest),
                    "limits": limits_json(self.limits),
                }
            )
            self.ready.set()

            while True:
                message = self._receive_json(connection)
                with self.condition:
                    self.messages.append(message)
                    self.condition.notify_all()
                self._respond(message)
        except (EOFError, OSError):
            return

    def _respond(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "submit":
            request_id = message["request_id"]
            if request_id in self.reject_request_ids:
                self.send(
                    {
                        "type": "rejected",
                        "protocol_version": PROTOCOL_VERSION,
                        "worker_epoch": self.worker_epoch,
                        "request_id": request_id,
                        "status": {
                            "code": "resource_exhausted",
                            "message": "test rejection",
                        },
                    }
                )
            else:
                self.send(
                    {
                        "type": "accepted",
                        "protocol_version": PROTOCOL_VERSION,
                        "worker_epoch": self.worker_epoch,
                        "request_id": request_id,
                    }
                )
        elif message_type == "health":
            self.send(
                {
                    "type": "stats",
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_epoch": self.worker_epoch,
                    "probe_id": message["probe_id"],
                    "ready": True,
                    "status": {"code": "ok", "message": ""},
                    "uptime_ms": 100,
                    "active_requests": 0,
                    "reserved_input_tokens": 0,
                    "reserved_output_tokens": 0,
                    "session_egress_frames": 0,
                    "session_egress_bytes": 0,
                    "session_egress_high_watermark_frames": 1,
                    "session_egress_high_watermark_bytes": 128,
                    "rejected_requests": 0,
                    "backpressure_requests": 0,
                    "cancelled_requests": 0,
                }
            )

    @staticmethod
    def _receive_json(connection: socket.socket) -> dict[str, Any]:
        prefix = FakeWorker._receive_exact(connection, 4)
        payload_size = struct.unpack("!I", prefix)[0]
        payload = FakeWorker._receive_exact(connection, payload_size)
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("expected JSON object")
        return value

    @staticmethod
    def _receive_exact(connection: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                raise EOFError
            data.extend(chunk)
        return bytes(data)


class GenerationClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = make_manifest()
        self.limits = make_limits()
        self.workers: list[FakeWorker] = []
        self.clients: list[GenerationClient] = []

    def tearDown(self) -> None:
        for client in self.clients:
            client.close()
        for worker in self.workers:
            worker.stop()

    def make_client(
        self,
        worker: FakeWorker,
        *,
        max_delta_events: int = 64,
    ) -> GenerationClient:
        worker.start()
        self.workers.append(worker)
        client = GenerationClient(
            GenerationClientConfig(
                socket_path=worker.socket_path,
                expected_manifest=self.manifest,
                max_delta_events_per_request=max_delta_events,
            )
        )
        client.connect()
        self.clients.append(client)
        return client

    @staticmethod
    def request(request_id: int) -> GenerationRequest:
        return GenerationRequest(
            request_id=request_id,
            input_token_ids=(1, 2, 3),
            max_new_tokens=8,
            trace_id=f"trace-{request_id}",
        )

    @staticmethod
    def delta(request_id: int, sequence_no: int, tokens: list[int]) -> dict[str, Any]:
        return {
            "type": "token_delta",
            "protocol_version": PROTOCOL_VERSION,
            "worker_epoch": 77,
            "request_id": request_id,
            "sequence_no": sequence_no,
            "token_ids": tokens,
        }

    @staticmethod
    def terminal(
        request_id: int,
        *,
        code: str = "ok",
        finish_reason: Optional[str] = "length",
    ) -> dict[str, Any]:
        return {
            "type": "terminal",
            "protocol_version": PROTOCOL_VERSION,
            "worker_epoch": 77,
            "request_id": request_id,
            "status": {"code": code, "message": ""},
            "finish_reason": finish_reason,
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    def test_handshake_multiplex_and_health(self) -> None:
        worker = FakeWorker(self.manifest, self.limits)
        client = self.make_client(worker)
        first = client.submit(self.request(1), timeout=1.0)
        second = client.submit(self.request(2), timeout=1.0)

        worker.send(self.delta(2, 0, [20]))
        worker.send(self.delta(1, 0, [10, 11]))
        worker.send(self.terminal(2))
        worker.send(self.terminal(1))

        self.assertEqual(first.collect(timeout=1.0)[0], (10, 11))
        self.assertEqual(second.collect(timeout=1.0)[0], (20,))
        stats = client.health(timeout=1.0)
        self.assertTrue(stats.ready)
        self.assertEqual(stats.uptime_ms, 100)

    def test_rejected_request_never_creates_handle(self) -> None:
        worker = FakeWorker(
            self.manifest,
            self.limits,
            reject_request_ids={3},
        )
        client = self.make_client(worker)
        with self.assertRaises(RequestRejectedError) as context:
            client.submit(self.request(3), timeout=1.0)
        self.assertEqual(context.exception.status.code, "resource_exhausted")

    def test_disconnect_synthesizes_unavailable_terminal(self) -> None:
        worker = FakeWorker(self.manifest, self.limits)
        client = self.make_client(worker)
        handle = client.submit(self.request(4), timeout=1.0)
        worker.disconnect()
        event = handle.next_event(timeout=1.0)
        self.assertIsInstance(event, Terminal)
        assert isinstance(event, Terminal)
        self.assertEqual(event.status.code, "unavailable")
        with self.assertRaises(StopIteration):
            handle.next_event(timeout=0.1)

    def test_bounded_delta_buffer_cancels_only_that_request(self) -> None:
        worker = FakeWorker(self.manifest, self.limits)
        client = self.make_client(worker, max_delta_events=1)
        handle = client.submit(self.request(5), timeout=1.0)
        worker.send(self.delta(5, 0, [30]))
        worker.send(self.delta(5, 1, [31]))
        worker.wait_for_message("cancel", request_id=5, timeout=1.0)
        worker.send(
            self.terminal(
                5,
                code="cancelled",
                finish_reason="cancelled",
            )
        )
        tokens, terminal = handle.collect(timeout=1.0)
        self.assertEqual(tokens, (30,))
        self.assertEqual(terminal.status.code, "cancelled")
        self.assertTrue(handle.overflowed)

    def test_pending_request_capacity_has_a_typed_error(self) -> None:
        worker = FakeWorker(self.manifest, self.limits)
        client = self.make_client(worker)
        first = client.submit(self.request(6), timeout=1.0)
        second = client.submit(self.request(7), timeout=1.0)
        with self.assertRaises(ClientCapacityError):
            client.submit(self.request(8), timeout=1.0)
        worker.send(self.terminal(6))
        worker.send(self.terminal(7))
        self.assertTrue(first.collect(timeout=1.0)[1].status.ok)
        self.assertTrue(second.collect(timeout=1.0)[1].status.ok)

    def test_full_manifest_mismatch_rejects_handshake(self) -> None:
        mismatched = ModelManifest(
            **{
                **self.manifest.__dict__,
                "engine_fingerprint": "different-engine",
            }
        )
        worker = FakeWorker(mismatched, self.limits)
        worker.start()
        self.workers.append(worker)
        client = GenerationClient(
            GenerationClientConfig(worker.socket_path, self.manifest)
        )
        self.clients.append(client)
        with self.assertRaises(ProtocolError):
            client.connect()


if __name__ == "__main__":
    unittest.main()
