#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Options:
    direct_benchmark: Path
    worker: Path
    engine_dir: Path
    output_dir: Path
    repetitions: int
    concurrency: int
    warmup_requests: int
    measured_requests: int
    max_new_tokens: int
    input_token_ids: tuple[int, ...]
    cuda_device: str
    startup_timeout: float
    allow_dirty: bool


@dataclass(frozen=True)
class MetricSpec:
    name: str
    path: tuple[str, ...]
    unit: str
    preference: str


METRICS = (
    MetricSpec("ttft_ms.p50", ("ttft_ms", "p50"), "ms", "lower"),
    MetricSpec("ttft_ms.p95", ("ttft_ms", "p95"), "ms", "lower"),
    MetricSpec("ttft_ms.p99", ("ttft_ms", "p99"), "ms", "lower"),
    MetricSpec("tpot_ms.p50", ("tpot_ms", "p50"), "ms", "lower"),
    MetricSpec("tpot_ms.p95", ("tpot_ms", "p95"), "ms", "lower"),
    MetricSpec("tpot_ms.p99", ("tpot_ms", "p99"), "ms", "lower"),
    MetricSpec("e2e_ms.p50", ("e2e_ms", "p50"), "ms", "lower"),
    MetricSpec("e2e_ms.p95", ("e2e_ms", "p95"), "ms", "lower"),
    MetricSpec("e2e_ms.p99", ("e2e_ms", "p99"), "ms", "lower"),
    MetricSpec(
        "request_throughput_rps",
        ("request_throughput_rps",),
        "request/s",
        "higher",
    ),
    MetricSpec(
        "output_token_throughput_tps",
        ("output_token_throughput_tps",),
        "token/s",
        "higher",
    ),
)


def parse_options(argv: Optional[Sequence[str]] = None) -> Options:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Direct/IPC benchmarks with alternating order and "
            "write an incremental-overhead report."
        )
    )
    parser.add_argument("--direct-benchmark", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--measured-requests", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--input-token-ids",
        required=True,
        type=int,
        nargs="+",
    )
    args = parser.parse_args(argv)

    if not args.direct_benchmark.is_file():
        parser.error(f"Direct benchmark does not exist: {args.direct_benchmark}")
    if not os.access(args.direct_benchmark, os.X_OK):
        parser.error("Direct benchmark is not executable")
    if not args.worker.is_file():
        parser.error(f"llm_worker does not exist: {args.worker}")
    if not os.access(args.worker, os.X_OK):
        parser.error("llm_worker is not executable")
    if not args.engine_dir.is_dir():
        parser.error(f"Engine directory does not exist: {args.engine_dir}")
    if not (args.engine_dir / "config.json").is_file():
        parser.error("Engine directory does not contain config.json")
    if args.output_dir.exists():
        parser.error("output_dir already exists; use a new evidence directory")
    if args.repetitions <= 0:
        parser.error("repetitions must be greater than zero")
    if not 1 <= args.concurrency <= 8:
        parser.error("concurrency must be in [1, 8]")
    if args.warmup_requests < 0:
        parser.error("warmup_requests must not be negative")
    if args.measured_requests <= 0:
        parser.error("measured_requests must be greater than zero")
    if not 1 <= args.max_new_tokens <= 32:
        parser.error("max_new_tokens must be in [1, 32]")
    if args.startup_timeout <= 0:
        parser.error("startup_timeout must be positive")
    if not args.cuda_device:
        parser.error("cuda_device must not be empty")
    if not args.input_token_ids or any(
        token_id < 0 for token_id in args.input_token_ids
    ):
        parser.error("input token IDs must be non-negative")
    if len(args.input_token_ids) > 512:
        parser.error("input token count exceeds Engine max_input_tokens")
    if len(args.input_token_ids) + args.max_new_tokens > 544:
        parser.error("request exceeds Engine max_sequence_tokens")

    return Options(
        direct_benchmark=args.direct_benchmark.resolve(),
        worker=args.worker.resolve(),
        engine_dir=args.engine_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        repetitions=args.repetitions,
        concurrency=args.concurrency,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        max_new_tokens=args.max_new_tokens,
        input_token_ids=tuple(args.input_token_ids),
        cuda_device=args.cuda_device,
        startup_timeout=args.startup_timeout,
        allow_dirty=args.allow_dirty,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def workload_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "engine_dir": summary.get("engine_dir"),
        "concurrency": summary.get("concurrency"),
        "warmup_requests": summary.get("warmup_requests"),
        "measured_requests": summary.get("measured_requests"),
        "input_tokens": summary.get("input_tokens"),
        "input_token_ids": summary.get("input_token_ids"),
        "max_new_tokens": summary.get("max_new_tokens"),
        "streaming": summary.get("streaming"),
        "sampling": summary.get("sampling"),
    }


def read_request_outputs(path: Path) -> list[tuple[int, tuple[int, ...]]]:
    results: list[tuple[int, tuple[int, ...]]] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"request_id", "success", "output_token_ids"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"request CSV is missing required columns: {path}")
        for row in reader:
            if row["success"] != "1":
                raise ValueError(
                    f"request {row['request_id']} is unsuccessful in {path}"
                )
            tokens = tuple(
                int(token) for token in row["output_token_ids"].split()
            )
            results.append((int(row["request_id"]), tokens))
    return results


def extract_metric(summary: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = summary
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"summary is missing metric {'.'.join(path)}")
        value = value[component]
    if type(value) not in (int, float):
        raise ValueError(f"metric {'.'.join(path)} must be numeric")
    return float(value)


def metric_delta(direct: float, ipc: float) -> dict[str, Optional[float]]:
    absolute = ipc - direct
    relative = None if direct == 0 else absolute / direct * 100.0
    return {
        "direct": direct,
        "ipc": ipc,
        "ipc_minus_direct": absolute,
        "relative_percent": relative,
    }


def compare_run(
    direct_summary: Mapping[str, Any],
    ipc_summary: Mapping[str, Any],
    direct_requests: Path,
    ipc_requests: Path,
) -> dict[str, Any]:
    if direct_summary.get("schema_version") != 2:
        raise ValueError("Direct summary schema_version must be 2")
    if ipc_summary.get("schema_version") != 2:
        raise ValueError("IPC summary schema_version must be 2")
    if direct_summary.get("benchmark_path") != "direct":
        raise ValueError("Direct summary has the wrong benchmark_path")
    if ipc_summary.get("benchmark_path") != "ipc":
        raise ValueError("IPC summary has the wrong benchmark_path")

    direct_workload = workload_from_summary(direct_summary)
    ipc_workload = workload_from_summary(ipc_summary)
    if direct_workload != ipc_workload:
        raise ValueError(
            "Direct and IPC summaries do not describe the same workload: "
            f"direct={direct_workload}, ipc={ipc_workload}"
        )
    measured_requests = direct_workload["measured_requests"]
    if direct_summary.get("successful_requests") != measured_requests:
        raise ValueError("Direct run did not complete every measured request")
    if ipc_summary.get("successful_requests") != measured_requests:
        raise ValueError("IPC run did not complete every measured request")
    if direct_summary.get("failed_requests") != 0:
        raise ValueError("Direct run contains failed requests")
    if ipc_summary.get("failed_requests") != 0:
        raise ValueError("IPC run contains failed requests")
    counter_deltas = ipc_summary.get("measurement_counter_deltas")
    if not isinstance(counter_deltas, Mapping):
        raise ValueError("IPC summary is missing measurement counter deltas")
    expected_counters = {
        "rejected_requests",
        "backpressure_requests",
        "cancelled_requests",
    }
    if set(counter_deltas) != expected_counters:
        raise ValueError("IPC summary has incomplete measurement counter deltas")
    unexpected_counters = {
        name: value
        for name, value in counter_deltas.items()
        if type(value) is not int or value != 0
    }
    if unexpected_counters:
        raise ValueError(
            "IPC measurement contains rejected, backpressured, or cancelled "
            f"requests: {unexpected_counters}"
        )

    direct_outputs = read_request_outputs(direct_requests)
    ipc_outputs = read_request_outputs(ipc_requests)
    if direct_outputs != ipc_outputs:
        raise ValueError("Direct and IPC output Token IDs are not identical")
    if len(direct_outputs) != measured_requests:
        raise ValueError("request CSV row count differs from measured_requests")

    metrics: dict[str, Any] = {}
    for spec in METRICS:
        comparison = metric_delta(
            extract_metric(direct_summary, spec.path),
            extract_metric(ipc_summary, spec.path),
        )
        comparison["unit"] = spec.unit
        comparison["preference"] = spec.preference
        metrics[spec.name] = comparison
    return {
        "token_outputs_identical": True,
        "request_count": len(direct_outputs),
        "metrics": metrics,
    }


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def sample_stddev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot aggregate an empty run list")
    aggregate: dict[str, Any] = {}
    for spec in METRICS:
        comparisons = [run["metrics"][spec.name] for run in runs]
        direct_values = [float(value["direct"]) for value in comparisons]
        ipc_values = [float(value["ipc"]) for value in comparisons]
        deltas = [float(value["ipc_minus_direct"]) for value in comparisons]
        percentages = [
            float(value["relative_percent"])
            for value in comparisons
            if value["relative_percent"] is not None
        ]
        aggregate[spec.name] = {
            "unit": spec.unit,
            "preference": spec.preference,
            "direct_mean": mean(direct_values),
            "direct_sample_stddev": sample_stddev(direct_values),
            "ipc_mean": mean(ipc_values),
            "ipc_sample_stddev": sample_stddev(ipc_values),
            "paired_delta_mean": mean(deltas),
            "paired_delta_sample_stddev": sample_stddev(deltas),
            "paired_relative_percent_mean": (
                mean(percentages) if percentages else None
            ),
            "paired_relative_percent_sample_stddev": (
                sample_stddev(percentages) if percentages else None
            ),
        }
    return aggregate


def repository_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def run_command(
    command: Sequence[str],
    log_path: Path,
    environment: Mapping[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        output.write(f"$ {shlex.join(command)}\n")
        output.flush()
        completed = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(environment),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited with {completed.returncode}; see {log_path}"
        )


def direct_command(options: Options, run_dir: Path) -> list[str]:
    return [
        str(options.direct_benchmark),
        str(options.engine_dir),
        str(options.concurrency),
        str(options.warmup_requests),
        str(options.measured_requests),
        str(options.max_new_tokens),
        str(run_dir / "direct-summary.json"),
        str(run_dir / "direct-requests.csv"),
        str(run_dir / "direct-iteration-stats.jsonl"),
        *(str(token) for token in options.input_token_ids),
    ]


def ipc_command(options: Options, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "benchmark" / "llm_ipc_benchmark.py"),
        "--worker",
        str(options.worker),
        "--engine-dir",
        str(options.engine_dir),
        "--concurrency",
        str(options.concurrency),
        "--warmup-requests",
        str(options.warmup_requests),
        "--measured-requests",
        str(options.measured_requests),
        "--max-new-tokens",
        str(options.max_new_tokens),
        "--summary",
        str(run_dir / "ipc-summary.json"),
        "--requests-csv",
        str(run_dir / "ipc-requests.csv"),
        "--worker-log",
        str(run_dir / "worker.log"),
        "--startup-timeout",
        str(options.startup_timeout),
        "--input-token-ids",
        *(str(token) for token in options.input_token_ids),
    ]


def print_aggregate(aggregate: Mapping[str, Mapping[str, Any]]) -> None:
    print("\nDirect / IPC paired comparison (mean across repetitions)")
    print(
        f"{'metric':32} {'direct':>12} {'ipc':>12} "
        f"{'ipc-direct':>12} {'change':>11}"
    )
    for spec in METRICS:
        result = aggregate[spec.name]
        percentage = result["paired_relative_percent_mean"]
        percentage_text = "n/a" if percentage is None else f"{percentage:+.2f}%"
        print(
            f"{spec.name:32} "
            f"{result['direct_mean']:12.3f} "
            f"{result['ipc_mean']:12.3f} "
            f"{result['paired_delta_mean']:+12.3f} "
            f"{percentage_text:>11}"
        )


def execute(options: Options) -> Path:
    commit, dirty = repository_state()
    if dirty and not options.allow_dirty:
        raise RuntimeError(
            "repository has tracked changes; commit them first or use --allow-dirty "
            "for a non-formal smoke run"
        )

    options.output_dir.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = options.cuda_device
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    run_reports: list[dict[str, Any]] = []
    execution_orders: list[list[str]] = []
    for repetition in range(1, options.repetitions + 1):
        run_dir = options.output_dir / f"run-{repetition:02d}"
        run_dir.mkdir()
        commands = {
            "direct": direct_command(options, run_dir),
            "ipc": ipc_command(options, run_dir),
        }
        order = ["direct", "ipc"] if repetition % 2 else ["ipc", "direct"]
        execution_orders.append(order)
        for benchmark_path in order:
            print(
                f"[run {repetition}/{options.repetitions}] "
                f"starting {benchmark_path}"
            )
            run_command(
                commands[benchmark_path],
                run_dir / f"{benchmark_path}-command.log",
                environment,
            )

        direct_summary_path = run_dir / "direct-summary.json"
        ipc_summary_path = run_dir / "ipc-summary.json"
        comparison = compare_run(
            load_json(direct_summary_path),
            load_json(ipc_summary_path),
            run_dir / "direct-requests.csv",
            run_dir / "ipc-requests.csv",
        )
        comparison.update(
            {
                "repetition": repetition,
                "execution_order": order,
                "direct_summary": str(
                    direct_summary_path.relative_to(options.output_dir)
                ),
                "ipc_summary": str(
                    ipc_summary_path.relative_to(options.output_dir)
                ),
            }
        )
        run_reports.append(comparison)

    aggregate = aggregate_runs(run_reports)
    report = {
        "schema_version": 1,
        "benchmark": "direct_vs_ipc_incremental_overhead",
        "interpretation": (
            "IPC minus Direct is the incremental cost of GenerationClient, "
            "serialization, UDS, WorkerServer, GenerationRuntime and bridge/egress; "
            "it is not a UDS syscall-only microbenchmark."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "engine_config_sha256": hashlib.sha256(
            (options.engine_dir / "config.json").read_bytes()
        ).hexdigest(),
        "cuda_visible_devices": options.cuda_device,
        "workload": {
            "engine_dir": str(options.engine_dir),
            "repetitions": options.repetitions,
            "concurrency": options.concurrency,
            "warmup_requests": options.warmup_requests,
            "measured_requests": options.measured_requests,
            "input_tokens": len(options.input_token_ids),
            "input_token_ids": list(options.input_token_ids),
            "max_new_tokens": options.max_new_tokens,
            "streaming": True,
            "sampling": {
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 0,
            },
        },
        "minimum_repetitions_met": options.repetitions >= 3,
        "execution_orders": execution_orders,
        "token_outputs_identical": True,
        "runs": run_reports,
        "aggregate": aggregate,
    }
    report_path = options.output_dir / "direct-ipc-comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print_aggregate(aggregate)
    print(f"\ncomparison report: {report_path}")
    return report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_options(argv)
    try:
        execute(options)
        return 0
    except Exception as exception:
        print(f"Direct/IPC benchmark failed: {exception}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
