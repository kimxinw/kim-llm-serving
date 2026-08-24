from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    TestClient = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False

from kim_llm_gateway import GatewayConfig, GatewayRuntimeOptions, GatewayService
from kim_llm_gateway.api import create_app

from test_gateway_core import (
    FakeClient,
    FakeTokenizer,
    make_manifest,
)
from kim_llm_client import Status, Terminal, TokenDelta, Usage


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed")
class GatewayApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="kim-llm-gateway-api-")
        root = Path(self.directory.name)
        self.config = GatewayConfig(
            worker_config_path=root / "worker.json",
            tokenizer_path=root / "tokenizer",
            socket_path=str(root / "worker.sock"),
            manifest=make_manifest(),
            event_poll_seconds=0.01,
        )
        self.client_impl = FakeClient()
        self.service = GatewayService(
            self.client_impl,
            FakeTokenizer(),
            make_manifest(),
            GatewayRuntimeOptions(
                max_sse_delta_events_per_request=8,
                cancel_drain_timeout_seconds=1.0,
                shutdown_grace_seconds=1.0,
            ),
        )
        assert TestClient is not None
        self.client = TestClient(create_app(self.config, service=self.service))
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.directory.cleanup()

    def test_health_ready_models_and_non_streaming_completion(self) -> None:
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        models = self.client.get("/v1/models").json()
        self.assertEqual(models["data"][0]["id"], "tinyllama")

        self.client_impl.next_events = [
            TokenDelta(1, 0, (1, 2, 3)),
            Terminal(1, Status(), "length", Usage(2, 3)),
        ]
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "abc")
        self.assertEqual(payload["choices"][0]["finish_reason"], "length")
        self.assertEqual(payload["usage"]["total_tokens"], 5)

    def test_streaming_completion_is_openai_sse(self) -> None:
        self.client_impl.next_events = [
            TokenDelta(1, 0, (1,)),
            TokenDelta(1, 1, (2,)),
            Terminal(1, Status(), "eos", Usage(2, 2)),
        ]
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tinyllama",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 2,
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            lines = [line for line in response.iter_lines() if line]
        self.assertEqual(lines[-1], "data: [DONE]")
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in lines[:-1]
        ]
        self.assertEqual(payloads[0]["choices"][0]["delta"]["role"], "assistant")
        text = "".join(
            payload["choices"][0]["delta"].get("content", "")
            for payload in payloads
            if payload["choices"]
        )
        self.assertEqual(text, "ab")
        self.assertEqual(payloads[-1]["choices"], [])
        self.assertEqual(payloads[-1]["usage"]["completion_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
