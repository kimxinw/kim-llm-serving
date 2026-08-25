#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_benchmark() -> ModuleType:
    path = REPOSITORY_ROOT / "benchmark" / "llm_http_benchmark.py"
    spec = importlib.util.spec_from_file_location("kim_llm_http_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_benchmark()
CORE = sys.modules["http_benchmark_core"]


def completed_result(request_id: int, *, slow: bool = False) -> object:
    return BENCHMARK.RequestResult(
        request_id=request_id,
        outcome="completed",
        protocol_valid=True,
        accepted=True,
        slow_client=slow,
        http_status=200,
        status_code="ok",
        prompt_tokens=4,
        completion_tokens=2,
        output_text="ab",
        finish_reason="length",
        ttft_ms=10.0,
        tpot_ms=2.0,
        e2e_ms=20.0,
    )


def make_options(root: Path, *, mode: str = "open-loop") -> object:
    return BENCHMARK.BenchmarkOptions(
        worker=root / "llm_worker",
        engine_dir=root / "engine",
        tokenizer_path=root / "tokenizer",
        model="tinyllama",
        revision="revision",
        messages=({"role": "user", "content": "hello"},),
        mode=mode,
        concurrency=2,
        max_inflight=4,
        warmup_requests=2,
        measured_requests=3,
        max_new_tokens=2,
        offered_rate=10.0 if mode == "open-loop" else None,
        arrival_distribution="constant",
        arrival_seed=7,
        ttft_slo_ms=15.0,
        e2e_slo_ms=25.0,
        slow_request_every=0,
        slow_read_delay_ms=0.0,
        max_pending_requests=2,
        max_client_delta_events_per_request=4,
        max_sse_delta_events_per_request=2,
        startup_timeout=10.0,
        request_timeout=10.0,
        host="127.0.0.1",
        port=0,
        summary_path=root / "summary.json",
        request_csv_path=root / "requests.csv",
        worker_log_path=root / "worker.log",
        gateway_log_path=root / "gateway.log",
        allow_dirty=True,
    )


class FakeResponse:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._stream = io.BytesIO(payload)

    def readline(self, limit: int = -1) -> bytes:
        return self._stream.readline(limit)

    def read(self, limit: int = -1) -> bytes:
        return self._stream.read(limit)


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.sock = None
        self.closed = False

    def connect(self) -> None:
        return None

    def request(self, *_: object, **__: object) -> None:
        return None

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class HttpBenchmarkContractTest(unittest.TestCase):
    def test_sse_client_records_text_usage_and_terminal(self) -> None:
        events = (
            b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"b"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":4,'
            b'"completion_tokens":2,"total_tokens":6}}\n\n'
            b"data: [DONE]\n\n"
        )
        connection = FakeConnection(FakeResponse(200, events))
        client = BENCHMARK.OpenAiSseClient(
            "http://127.0.0.1:8000",
            "tinyllama",
            ({"role": "user", "content": "hello"},),
            2,
            4,
            10.0,
            0.0,
        )
        with mock.patch.object(
            CORE.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            result = client.run(7, time.perf_counter_ns(), False)

        self.assertEqual(result.outcome, "completed")
        self.assertTrue(result.protocol_valid)
        self.assertEqual(result.output_text, "ab")
        self.assertEqual(result.prompt_tokens, 4)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.finish_reason, "length")
        self.assertTrue(connection.closed)

    def test_http_rejection_is_a_valid_overload_observation(self) -> None:
        payload = b'{"error":{"code":"queue_full","message":"full"}}'
        connection = FakeConnection(FakeResponse(429, payload))
        client = BENCHMARK.OpenAiSseClient(
            "http://127.0.0.1:8000",
            "tinyllama",
            ({"role": "user", "content": "hello"},),
            2,
            4,
            10.0,
            0.0,
        )
        with mock.patch.object(
            CORE.http.client,
            "HTTPConnection",
            return_value=connection,
        ):
            result = client.run(8, time.perf_counter_ns(), False)
        self.assertEqual(result.outcome, "rejected")
        self.assertEqual(result.status_code, "queue_full")
        self.assertTrue(result.protocol_valid)
        self.assertFalse(result.accepted)

    def test_closed_loop_preserves_request_order_and_marks_slow_clients(self) -> None:
        def run_request(
            request_id: int,
            scheduled_ns: int,
            slow_client: bool,
        ) -> object:
            del scheduled_ns
            return completed_result(request_id, slow=slow_client)

        run = BENCHMARK.run_closed_loop(
            run_request,
            concurrency=2,
            total_requests=4,
            first_request_id=10,
            slow_request_every=2,
        )
        self.assertEqual(
            [request.request_id for request in run.requests],
            [10, 11, 12, 13],
        )
        self.assertEqual(
            [request.slow_client for request in run.requests],
            [False, True, False, True],
        )

    def test_open_loop_drops_at_client_boundary_instead_of_queueing(self) -> None:
        def slow_request(
            request_id: int,
            scheduled_ns: int,
            slow_client: bool,
        ) -> object:
            del scheduled_ns, slow_client
            time.sleep(0.05)
            return completed_result(request_id)

        run = BENCHMARK.run_open_loop(
            slow_request,
            max_inflight=1,
            total_requests=3,
            first_request_id=1,
            offered_rate=1000.0,
            distribution_name="constant",
            seed=0,
        )
        self.assertEqual(run.requests[0].outcome, "completed")
        self.assertEqual(
            [request.outcome for request in run.requests[1:]],
            ["client_overflow", "client_overflow"],
        )
        self.assertFalse(BENCHMARK.all_protocol_observations_valid(run))

    def test_summary_separates_goodput_rejection_and_resource_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-http-summary-") as directory:
            root = Path(directory)
            completed = completed_result(1)
            rejected = BENCHMARK.RequestResult(
                request_id=2,
                outcome="rejected",
                protocol_valid=True,
                http_status=429,
                status_code="queue_full",
                e2e_ms=1.0,
            )
            terminal_error = BENCHMARK.RequestResult(
                request_id=3,
                outcome="terminal_error",
                protocol_valid=True,
                accepted=True,
                http_status=200,
                status_code="timeout",
                e2e_ms=30.0,
            )
            before = {
                "kim_llm_gateway_requests_offered_total": 2.0,
            }
            after = {
                "kim_llm_gateway_requests_offered_total": 5.0,
                "kim_llm_gateway_active_requests": 0.0,
                "kim_llm_gateway_sse_buffered_events": 0.0,
                "kim_llm_worker_active_requests": 0.0,
                "kim_llm_worker_reserved_input_tokens": 0.0,
                "kim_llm_worker_reserved_output_tokens": 0.0,
                "kim_llm_worker_session_egress_frames": 0.0,
                "kim_llm_worker_session_egress_bytes": 0.0,
                "kim_llm_worker_session_egress_high_watermark_frames": 7.0,
            }
            summary = BENCHMARK.build_summary(
                make_options(root),
                (1, 2, 3, 4),
                BENCHMARK.RunResult(
                    (completed, rejected, terminal_error),
                    duration_seconds=1.0,
                    load_generation_seconds=0.2,
                ),
                worker_startup_ms=100.0,
                gateway_startup_ms=20.0,
                metrics_before=before,
                metrics_after=after,
            )

            self.assertEqual(summary["completed_requests"], 1)
            self.assertEqual(summary["rejected_requests"], 1)
            self.assertEqual(summary["terminal_error_requests"], 1)
            self.assertEqual(summary["good_requests"], 1)
            self.assertEqual(summary["goodput_rps"], 1.0)
            self.assertEqual(summary["rejections_by_code"], {"queue_full": 1})
            self.assertEqual(
                summary["measurement_counter_deltas"][
                    "kim_llm_gateway_requests_offered_total"
                ],
                3.0,
            )
            self.assertTrue(summary["resources_released"])


if __name__ == "__main__":
    unittest.main()
