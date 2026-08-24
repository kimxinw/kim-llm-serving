#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_runner() -> ModuleType:
    path = REPOSITORY_ROOT / "benchmark" / "run_direct_ipc_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "kim_llm_direct_ipc_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


def load_ipc_benchmark() -> ModuleType:
    path = REPOSITORY_ROOT / "benchmark" / "llm_ipc_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "kim_llm_ipc_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


IPC_BENCHMARK = load_ipc_benchmark()

from kim_llm_client import Status, Usage  # noqa: E402


def make_summary(path: str, scale: float = 1.0) -> dict[str, object]:
    return {
        "schema_version": 2,
        "benchmark_path": path,
        "engine_dir": "/engine",
        "concurrency": 2,
        "warmup_requests": 2,
        "measured_requests": 2,
        "successful_requests": 2,
        "failed_requests": 0,
        "input_tokens": 4,
        "input_token_ids": [1, 2, 3, 4],
        "max_new_tokens": 5,
        "streaming": True,
        "sampling": {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 0,
        },
        "request_throughput_rps": 10.0 / scale,
        "output_token_throughput_tps": 50.0 / scale,
        "ttft_ms": {"p50": 10.0 * scale, "p95": 12.0 * scale, "p99": 12.0 * scale},
        "tpot_ms": {"p50": 2.0 * scale, "p95": 3.0 * scale, "p99": 3.0 * scale},
        "e2e_ms": {"p50": 20.0 * scale, "p95": 24.0 * scale, "p99": 24.0 * scale},
        "measurement_counter_deltas": {
            "rejected_requests": 0,
            "backpressure_requests": 0,
            "cancelled_requests": 0,
        },
    }


def write_requests(path: Path, outputs: tuple[tuple[int, ...], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "request_id",
                "success",
                "prompt_tokens",
                "completion_tokens",
                "output_token_ids",
                "ttft_ms",
                "tpot_ms",
                "e2e_ms",
                "error",
            )
        )
        for index, token_ids in enumerate(outputs, start=3):
            writer.writerow(
                (
                    index,
                    1,
                    4,
                    len(token_ids),
                    " ".join(str(token) for token in token_ids),
                    10.0,
                    2.0,
                    20.0,
                    "",
                )
            )


class DirectIpcComparisonTest(unittest.TestCase):
    def test_ipc_closed_loop_records_tokens_and_request_order(self) -> None:
        class FakeHandle:
            def __init__(self, request_id: int) -> None:
                self._events = iter(
                    (
                        IPC_BENCHMARK.TokenDelta(request_id, 0, (3, 4)),
                        IPC_BENCHMARK.Terminal(
                            request_id,
                            Status(),
                            "length",
                            Usage(prompt_tokens=4, completion_tokens=2),
                        ),
                    )
                )

            def next_event(self, timeout: float) -> object:
                self.timeout = timeout
                return next(self._events)

            def cancel(self) -> bool:
                return True

        class FakeClient:
            def submit(self, request: object, timeout: float) -> FakeHandle:
                self.timeout = timeout
                return FakeHandle(request.request_id)

        run = IPC_BENCHMARK.run_closed_loop(
            FakeClient(),
            concurrency=2,
            total_requests=4,
            first_request_id=10,
            input_token_ids=(1, 2, 3, 4),
            max_new_tokens=2,
        )

        self.assertTrue(IPC_BENCHMARK.all_requests_successful(run))
        self.assertEqual(
            [request.request_id for request in run.requests],
            [10, 11, 12, 13],
        )
        self.assertEqual(
            [request.output_token_ids for request in run.requests],
            [(3, 4)] * 4,
        )

    def test_compare_and_aggregate_paired_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-compare-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            outputs = ((3, 4, 5), (3, 4, 5))
            write_requests(direct_requests, outputs)
            write_requests(ipc_requests, outputs)

            comparison = RUNNER.compare_run(
                make_summary("direct"),
                make_summary("ipc", scale=1.2),
                direct_requests,
                ipc_requests,
            )

            self.assertTrue(comparison["token_outputs_identical"])
            self.assertEqual(comparison["request_count"], 2)
            ttft = comparison["metrics"]["ttft_ms.p50"]
            self.assertAlmostEqual(ttft["ipc_minus_direct"], 2.0)
            self.assertAlmostEqual(ttft["relative_percent"], 20.0)

            aggregate = RUNNER.aggregate_runs((comparison, comparison))
            aggregate_ttft = aggregate["ttft_ms.p50"]
            self.assertAlmostEqual(aggregate_ttft["direct_mean"], 10.0)
            self.assertAlmostEqual(aggregate_ttft["ipc_mean"], 12.0)
            self.assertAlmostEqual(aggregate_ttft["paired_delta_mean"], 2.0)
            self.assertAlmostEqual(
                aggregate_ttft["paired_relative_percent_mean"],
                20.0,
            )

    def test_rejects_different_workloads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-compare-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            outputs = ((3,), (3,))
            write_requests(direct_requests, outputs)
            write_requests(ipc_requests, outputs)
            ipc_summary = make_summary("ipc")
            ipc_summary["max_new_tokens"] = 6

            with self.assertRaisesRegex(ValueError, "same workload"):
                RUNNER.compare_run(
                    make_summary("direct"),
                    ipc_summary,
                    direct_requests,
                    ipc_requests,
                )

    def test_rejects_different_output_tokens(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-compare-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            write_requests(direct_requests, ((3,), (3,)))
            write_requests(ipc_requests, ((3,), (4,)))

            with self.assertRaisesRegex(ValueError, "Token IDs"):
                RUNNER.compare_run(
                    make_summary("direct"),
                    make_summary("ipc"),
                    direct_requests,
                    ipc_requests,
                )

    def test_rejects_ipc_runs_with_control_events(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-compare-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            outputs = ((3,), (3,))
            write_requests(direct_requests, outputs)
            write_requests(ipc_requests, outputs)
            ipc_summary = make_summary("ipc")
            ipc_summary["measurement_counter_deltas"]["backpressure_requests"] = 1

            with self.assertRaisesRegex(ValueError, "backpressured"):
                RUNNER.compare_run(
                    make_summary("direct"),
                    ipc_summary,
                    direct_requests,
                    ipc_requests,
                )


if __name__ == "__main__":
    unittest.main()
