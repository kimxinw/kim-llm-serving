from __future__ import annotations

import json
import queue
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

from kim_llm_client import (
    GenerationRequest,
    ModelManifest,
    Stats,
    Status,
    Terminal,
    TokenDelta,
    Usage,
)
from kim_llm_gateway import (
    GatewayConfig,
    GatewayError,
    GatewayRequest,
    GatewayRuntimeOptions,
    GatewayService,
    GatewayTerminal,
    GatewayTextDelta,
    fingerprint_chat_template,
    fingerprint_tokenizer_assets,
    parse_chat_completion_request,
)
from kim_llm_gateway.api import _stream_chat_completion


def make_manifest() -> ModelManifest:
    return ModelManifest(
        model_id="tinyllama",
        revision="revision-1",
        tokenizer_fingerprint="tokenizer-sha",
        chat_template_fingerprint="template-sha",
        engine_fingerprint="engine-sha",
        eos_token_id=2,
        pad_token_id=2,
        max_input_tokens=32,
        max_output_tokens=16,
        max_sequence_tokens=48,
        precision="fp16",
        max_batch_size=2,
    )


def ready_stats() -> Stats:
    return Stats(
        probe_id=1,
        ready=True,
        status=Status(),
        uptime_ms=100,
        active_requests=0,
        reserved_input_tokens=0,
        reserved_output_tokens=0,
        session_egress_frames=0,
        session_egress_bytes=0,
        session_egress_high_watermark_frames=0,
        session_egress_high_watermark_bytes=0,
        rejected_requests=0,
        backpressure_requests=0,
        cancelled_requests=0,
    )


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2
    tokenizer_fingerprint = "tokenizer-sha"
    chat_template_fingerprint = "template-sha"

    def encode_chat(self, messages: Sequence[Mapping[str, str]]) -> tuple[int, ...]:
        if not messages:
            raise ValueError("messages must not be empty")
        return (10, 11)

    def encode_text(self, text: str) -> tuple[int, ...]:
        return (99,) if text else ()

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(ord("a") + token_id - 1) for token_id in token_ids)


class FakeHandle:
    def __init__(
        self,
        request_id: int,
        events: Sequence[object],
        *,
        terminal_on_cancel: bool = False,
    ) -> None:
        self.request_id = request_id
        self.events: queue.Queue[object] = queue.Queue()
        for event in events:
            self.events.put(event)
        self.terminal_on_cancel = terminal_on_cancel
        self.cancel_count = 0

    def next_event(self, timeout: Optional[float] = None) -> object:
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty as exception:
            raise TimeoutError from exception

    def cancel(self) -> bool:
        self.cancel_count += 1
        if self.terminal_on_cancel:
            self.events.put(
                Terminal(
                    self.request_id,
                    Status("cancelled", "cancelled by test"),
                    "cancelled",
                    Usage(prompt_tokens=2, completion_tokens=0),
                )
            )
            self.terminal_on_cancel = False
        return True


class FakeClient:
    def __init__(self) -> None:
        self.connected = False
        self.requests: list[GenerationRequest] = []
        self.handles: list[FakeHandle] = []
        self.next_events: list[object] = []
        self.terminal_on_cancel = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def health(self, timeout: float = 1.0) -> Stats:
        del timeout
        if not self.connected:
            raise RuntimeError("not connected")
        return ready_stats()

    def submit(
        self,
        request: GenerationRequest,
        timeout: Optional[float] = None,
    ) -> FakeHandle:
        del timeout
        self.requests.append(request)
        handle = FakeHandle(
            request.request_id,
            self.next_events,
            terminal_on_cancel=self.terminal_on_cancel,
        )
        self.handles.append(handle)
        self.next_events = []
        self.terminal_on_cancel = False
        return handle


class DisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


def make_request(*, streaming: bool = True) -> GatewayRequest:
    return GatewayRequest(
        model="tinyllama",
        messages=({"role": "user", "content": "hello"},),
        max_new_tokens=4,
        streaming=streaming,
        stop=("stop",),
        trace_id="chatcmpl-test",
    )


class GatewayConfigTest(unittest.TestCase):
    def test_config_reuses_worker_manifest_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-gateway-config-") as directory:
            root = Path(directory)
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            worker = {
                "engine_dir": "/engine",
                "socket_path": "worker.sock",
                "manifest": asdict(make_manifest()),
                "limits": {},
            }
            (root / "worker.json").write_text(
                json.dumps(worker), encoding="utf-8"
            )
            (root / "gateway.json").write_text(
                json.dumps(
                    {
                        "worker_config_path": "worker.json",
                        "tokenizer_path": "tokenizer",
                        "max_sse_delta_events_per_request": 8,
                        "max_client_delta_events_per_request": 16,
                    }
                ),
                encoding="utf-8",
            )
            config = GatewayConfig.load(root / "gateway.json")
            self.assertEqual(config.manifest, make_manifest())
            self.assertEqual(config.socket_path, str((root / "worker.sock").resolve()))
            self.assertEqual(config.tokenizer_path, tokenizer.resolve())

    def test_tokenizer_fingerprints_are_deterministic_and_domain_separated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-tokenizer-") as directory:
            root = Path(directory)
            (root / "tokenizer.model").write_bytes(b"tokenizer-model")
            (root / "tokenizer_config.json").write_text(
                '{"chat_template":"template"}', encoding="utf-8"
            )
            first = fingerprint_tokenizer_assets(root)
            second = fingerprint_tokenizer_assets(root)
            self.assertEqual(first, second)
            self.assertTrue(first.startswith("sha256:"))
            self.assertNotEqual(first, fingerprint_chat_template("template"))


class OpenAIContractTest(unittest.TestCase):
    def test_minimal_chat_request_and_stream_usage(self) -> None:
        command = parse_chat_completion_request(
            {
                "model": "tinyllama",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_completion_tokens": 8,
                "temperature": 1.0,
                "top_p": 0.9,
                "seed": 7,
                "stop": ["one", "two"],
            },
            make_manifest(),
        )
        self.assertTrue(command.stream)
        self.assertTrue(command.include_usage)
        self.assertEqual(command.max_new_tokens, 8)
        self.assertEqual(command.stop, ("one", "two"))

    def test_unknown_openai_field_is_rejected(self) -> None:
        with self.assertRaises(GatewayError):
            parse_chat_completion_request(
                {
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [],
                },
                make_manifest(),
            )


class GatewayRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = FakeClient()
        self.service = GatewayService(
            self.client,
            FakeTokenizer(),
            make_manifest(),
            GatewayRuntimeOptions(
                max_sse_delta_events_per_request=4,
                cancel_drain_timeout_seconds=1.0,
                shutdown_grace_seconds=1.0,
            ),
        )
        self.assertTrue(await self.service.start())

    async def asyncTearDown(self) -> None:
        await self.service.stop()

    async def test_generation_request_and_incremental_text_are_consistent(self) -> None:
        self.client.next_events = [
            TokenDelta(1, 0, (1, 2)),
            TokenDelta(1, 1, (3,)),
            Terminal(1, Status(), "length", Usage(2, 3)),
        ]
        session = await self.service.submit(make_request())
        first = await session.next_event(1.0)
        second = await session.next_event(1.0)
        terminal = await session.next_event(1.0)
        self.assertEqual(first, GatewayTextDelta("ab"))
        self.assertEqual(second, GatewayTextDelta("c"))
        self.assertIsInstance(terminal, GatewayTerminal)
        assert isinstance(terminal, GatewayTerminal)
        self.assertTrue(terminal.status.ok)
        submitted = self.client.requests[0]
        self.assertEqual(submitted.input_token_ids, (10, 11))
        self.assertEqual(submitted.stop_sequences, ((99,),))
        self.assertEqual(submitted.end_id, 2)
        self.assertEqual(submitted.pad_id, 2)

    async def test_sse_queue_overflow_cancels_only_request_and_keeps_terminal(self) -> None:
        overflow_service = GatewayService(
            self.client,
            FakeTokenizer(),
            make_manifest(),
            GatewayRuntimeOptions(
                max_sse_delta_events_per_request=1,
                cancel_drain_timeout_seconds=1.0,
                shutdown_grace_seconds=1.0,
            ),
        )
        self.client.next_events = [
            TokenDelta(1, 0, (1,)),
            TokenDelta(1, 1, (2,)),
            Terminal(1, Status(), "length", Usage(2, 2)),
        ]
        session = await overflow_service.submit(make_request())
        await session.wait_done(1.0)
        self.assertEqual(self.client.handles[-1].cancel_count, 1)
        self.assertIsInstance(await session.next_event(1.0), GatewayTextDelta)
        self.assertIsInstance(await session.next_event(1.0), GatewayTerminal)
        snapshot = overflow_service.metrics.snapshot()
        self.assertEqual(snapshot["backpressure_cancels"], 1)

    async def test_http_disconnect_cancel_continues_drain_to_terminal(self) -> None:
        self.client.next_events = []
        self.client.terminal_on_cancel = True
        session = await self.service.submit(make_request())
        self.assertTrue(
            await session.cancel_and_drain("http_disconnect", timeout=1.0)
        )
        terminal = await session.next_event(1.0)
        self.assertIsInstance(terminal, GatewayTerminal)
        assert isinstance(terminal, GatewayTerminal)
        self.assertEqual(terminal.status.code, "cancelled")
        snapshot = self.service.metrics.snapshot()
        self.assertEqual(snapshot["disconnect_cancels"], 1)

    async def test_stream_disconnect_adapter_cancels_and_drains_in_background(self) -> None:
        self.client.next_events = []
        self.client.terminal_on_cancel = True
        session = await self.service.submit(make_request())
        stream = _stream_chat_completion(
            DisconnectedRequest(),
            self.service,
            session,
            "chatcmpl-disconnect",
            1,
            "tinyllama",
            False,
            0.01,
        )
        first = await anext(stream)
        self.assertIn(b'"role":"assistant"', first)
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        await session.wait_done(1.0)
        self.assertEqual(self.client.handles[-1].cancel_count, 1)
        self.assertEqual(
            self.service.metrics.snapshot()["disconnect_cancels"],
            1,
        )

    async def test_wrong_model_is_rejected_before_worker_submit(self) -> None:
        request = GatewayRequest(
            **{**make_request().__dict__, "model": "other-model"}
        )
        with self.assertRaises(GatewayError) as context:
            await self.service.submit(request)
        self.assertEqual(context.exception.http_status, 404)
        self.assertFalse(self.client.requests)


if __name__ == "__main__":
    unittest.main()
