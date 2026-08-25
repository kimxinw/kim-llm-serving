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
    path = REPOSITORY_ROOT / "benchmark" / "run_direct_ipc_http_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "kim_llm_direct_ipc_http_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


class FakeTokenizer:
    def decode(self, token_ids: tuple[int, ...]) -> str:
        mapping = {
            (3, 4): "alpha",
            (5, 6): "beta",
        }
        return mapping[token_ids]


def make_summary(path: str, scale: float = 1.0) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_version": 1 if path == "http_sse" else 2,
        "benchmark_path": path,
        "engine_dir": "/engine",
        "concurrency": 2,
        "warmup_requests": 2,
        "measured_requests": 2,
        "successful_requests": 2,
        "failed_requests": 0,
        "input_tokens": 4,
        "input_token_ids": [1, 2, 3, 4],
        "max_new_tokens": 2,
        "streaming": True,
        "sampling": {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 0,
        },
        "request_throughput_rps": 10.0 / scale,
        "output_token_throughput_tps": 20.0 / scale,
        "ttft_ms": {
            "p50": 10.0 * scale,
            "p95": 12.0 * scale,
            "p99": 14.0 * scale,
        },
        "tpot_ms": {
            "p50": 2.0 * scale,
            "p95": 3.0 * scale,
            "p99": 4.0 * scale,
        },
        "e2e_ms": {
            "p50": 20.0 * scale,
            "p95": 24.0 * scale,
            "p99": 28.0 * scale,
        },
    }
    if path == "ipc":
        summary["measurement_counter_deltas"] = {
            "rejected_requests": 0,
            "backpressure_requests": 0,
            "cancelled_requests": 0,
        }
    if path == "http_sse":
        summary.update(
            {
                "mode": "closed-loop",
                "offered_requests": 2,
                "accepted_requests": 2,
                "completed_requests": 2,
                "rejected_requests": 0,
                "terminal_error_requests": 0,
                "client_failure_requests": 0,
                "slow_client_requests": 0,
                "resources_released": True,
                "load": {
                    "slow_request_every": 0,
                    "slow_read_delay_ms": 0.0,
                },
            }
        )
    return summary


def write_token_requests(
    path: Path,
    outputs: tuple[tuple[int, ...], ...],
) -> None:
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
        for request_id, token_ids in enumerate(outputs, start=3):
            writer.writerow(
                (
                    request_id,
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


def write_http_requests(
    path: Path,
    outputs: tuple[tuple[int, str], ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "request_id",
                "outcome",
                "protocol_valid",
                "completion_tokens",
                "output_text",
            )
        )
        for request_id, (completion_tokens, text) in enumerate(outputs, start=5):
            writer.writerow(
                (
                    request_id,
                    "completed",
                    True,
                    completion_tokens,
                    text,
                )
            )


class DirectIpcHttpComparisonTest(unittest.TestCase):
    def test_compare_and_aggregate_three_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-three-path-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            http_requests = root / "http.csv"
            token_outputs = ((3, 4), (5, 6))
            write_token_requests(direct_requests, token_outputs)
            write_token_requests(ipc_requests, token_outputs)
            write_http_requests(http_requests, ((2, "alpha"), (2, "beta")))

            comparison = RUNNER.compare_run(
                make_summary("direct"),
                make_summary("ipc", scale=1.2),
                make_summary("http_sse", scale=1.5),
                direct_requests,
                ipc_requests,
                http_requests,
                FakeTokenizer(),
            )

            self.assertTrue(comparison["token_outputs_identical_direct_ipc"])
            self.assertTrue(comparison["http_text_matches_token_decode"])
            self.assertTrue(comparison["completion_token_counts_identical"])
            self.assertEqual(comparison["request_count"], 2)
            ttft = comparison["metrics"]["ttft_ms.p50"]
            self.assertAlmostEqual(ttft["ipc_minus_direct"], 2.0)
            self.assertAlmostEqual(ttft["http_minus_ipc"], 3.0)

            aggregate = RUNNER.aggregate_runs((comparison, comparison))
            aggregate_ttft = aggregate["ttft_ms.p50"]
            self.assertAlmostEqual(aggregate_ttft["direct_mean"], 10.0)
            self.assertAlmostEqual(aggregate_ttft["ipc_mean"], 12.0)
            self.assertAlmostEqual(aggregate_ttft["http_mean"], 15.0)
            self.assertAlmostEqual(
                aggregate_ttft["http_relative_to_direct_percent_mean"],
                50.0,
            )

    def test_rejects_http_text_that_differs_from_token_decode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-three-path-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            http_requests = root / "http.csv"
            token_outputs = ((3, 4), (5, 6))
            write_token_requests(direct_requests, token_outputs)
            write_token_requests(ipc_requests, token_outputs)
            write_http_requests(http_requests, ((2, "alpha"), (2, "wrong")))

            with self.assertRaisesRegex(ValueError, "decoded output"):
                RUNNER.compare_run(
                    make_summary("direct"),
                    make_summary("ipc"),
                    make_summary("http_sse"),
                    direct_requests,
                    ipc_requests,
                    http_requests,
                    FakeTokenizer(),
                )

    def test_rejects_http_run_with_unreleased_resources(self) -> None:
        summary = make_summary("http_sse")
        summary["resources_released"] = False
        with self.assertRaisesRegex(ValueError, "return to zero"):
            RUNNER.validate_summary(summary, "http_sse")

    def test_rejects_mismatched_workload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-three-path-test-") as directory:
            root = Path(directory)
            direct_requests = root / "direct.csv"
            ipc_requests = root / "ipc.csv"
            http_requests = root / "http.csv"
            token_outputs = ((3, 4), (5, 6))
            write_token_requests(direct_requests, token_outputs)
            write_token_requests(ipc_requests, token_outputs)
            write_http_requests(http_requests, ((2, "alpha"), (2, "beta")))
            http_summary = make_summary("http_sse")
            http_summary["input_token_ids"] = [9, 9, 9, 9]

            with self.assertRaisesRegex(ValueError, "same workload"):
                RUNNER.compare_run(
                    make_summary("direct"),
                    make_summary("ipc"),
                    http_summary,
                    direct_requests,
                    ipc_requests,
                    http_requests,
                    FakeTokenizer(),
                )

    def test_execution_order_uses_balanced_rotation(self) -> None:
        self.assertEqual(
            [RUNNER.execution_order(index) for index in range(1, 7)],
            [
                ["direct", "ipc", "http"],
                ["ipc", "http", "direct"],
                ["http", "direct", "ipc"],
                ["direct", "ipc", "http"],
                ["ipc", "http", "direct"],
                ["http", "direct", "ipc"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
