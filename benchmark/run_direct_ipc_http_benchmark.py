#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "clients" / "python"))
sys.path.insert(0, str(REPOSITORY_ROOT / "gateway" / "python"))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmark"))

from http_benchmark_core import load_messages  # noqa: E402
from kim_llm_gateway import HuggingFaceTokenizer  # noqa: E402
from llm_ipc_benchmark import (  # noqa: E402
    MAX_CONCURRENCY,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_SEQUENCE_TOKENS,
)
from run_direct_ipc_benchmark import (  # noqa: E402
    METRICS,
    extract_metric,
    load_json,
    read_request_outputs,
    run_command,
)


DEFAULT_MODEL = "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_REVISION = "trtllm-0.16.0-engine"
PATHS = ("direct", "ipc", "http")
TRITON_BENCHMARK_REFERENCE = (
    "https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/"
    "tools/inflight_batcher_llm/benchmark_core_model.py"
)


@dataclass(frozen=True)
class Options:
    direct_benchmark: Path
    worker: Path
    engine_dir: Path
    tokenizer_path: Path
    output_dir: Path
    model: str
    revision: str
    messages: tuple[Mapping[str, str], ...]
    repetitions: int
    concurrency: int
    warmup_requests: int
    measured_requests: int
    max_new_tokens: int
    cuda_device: str
    startup_timeout: float
    request_timeout: float
    allow_dirty: bool


def parse_options(argv: Optional[Sequence[str]] = None) -> Options:
    parser = argparse.ArgumentParser(
        description=(
            "Run a rotating Direct/IPC/OpenAI HTTP-SSE closed-loop benchmark, "
            "verify equivalent outputs, and write a paired overhead report."
        )
    )
    parser.add_argument("--direct-benchmark", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    messages = parser.add_mutually_exclusive_group(required=True)
    messages.add_argument("--prompt")
    messages.add_argument("--messages-file", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--measured-requests", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    if not args.direct_benchmark.is_file():
        parser.error(f"Direct benchmark does not exist: {args.direct_benchmark}")
    if not os.access(args.direct_benchmark, os.X_OK):
        parser.error("Direct benchmark is not executable")
    if not args.worker.is_file():
        parser.error(f"llm_worker does not exist: {args.worker}")
    if not os.access(args.worker, os.X_OK):
        parser.error("llm_worker is not executable")
    if not args.engine_dir.is_dir() or not (args.engine_dir / "config.json").is_file():
        parser.error(f"invalid Engine directory: {args.engine_dir}")
    if not args.tokenizer_path.is_dir():
        parser.error(f"Tokenizer directory does not exist: {args.tokenizer_path}")
    if args.output_dir.exists():
        parser.error("output_dir already exists; use a new evidence directory")
    if not args.model or not args.revision:
        parser.error("model and revision must be non-empty")
    if args.repetitions <= 0:
        parser.error("repetitions must be greater than zero")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    if args.warmup_requests < 0:
        parser.error("warmup_requests must not be negative")
    if args.measured_requests <= 0:
        parser.error("measured_requests must be greater than zero")
    if not 1 <= args.max_new_tokens <= MAX_OUTPUT_TOKENS:
        parser.error(f"max_new_tokens must be in [1, {MAX_OUTPUT_TOKENS}]")
    if not args.cuda_device:
        parser.error("cuda_device must not be empty")
    for name in ("startup_timeout", "request_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")

    try:
        if args.prompt is not None:
            parsed_messages = load_messages(
                [{"role": "user", "content": args.prompt}]
            )
        else:
            assert args.messages_file is not None
            parsed_messages = load_messages(
                json.loads(args.messages_file.read_text(encoding="utf-8"))
            )
    except (OSError, UnicodeError, ValueError) as exception:
        parser.error(f"failed to load messages: {exception}")

    return Options(
        direct_benchmark=args.direct_benchmark.resolve(),
        worker=args.worker.resolve(),
        engine_dir=args.engine_dir.resolve(),
        tokenizer_path=args.tokenizer_path.resolve(),
        output_dir=args.output_dir.resolve(),
        model=args.model,
        revision=args.revision,
        messages=parsed_messages,
        repetitions=args.repetitions,
        concurrency=args.concurrency,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        max_new_tokens=args.max_new_tokens,
        cuda_device=args.cuda_device,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        allow_dirty=args.allow_dirty,
    )


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


def normalized_workload(summary: Mapping[str, Any]) -> dict[str, Any]:
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


def validate_summary(
    summary: Mapping[str, Any],
    benchmark_path: str,
) -> None:
    expected_schema = 1 if benchmark_path == "http_sse" else 2
    if summary.get("schema_version") != expected_schema:
        raise ValueError(
            f"{benchmark_path} summary schema_version must be {expected_schema}"
        )
    if summary.get("benchmark_path") != benchmark_path:
        raise ValueError(f"summary has the wrong benchmark_path: {benchmark_path}")
    measured = summary.get("measured_requests")
    if type(measured) is not int or measured <= 0:
        raise ValueError(f"{benchmark_path} summary has invalid measured_requests")
    if summary.get("successful_requests") != measured:
        raise ValueError(f"{benchmark_path} did not complete every measured request")
    if summary.get("failed_requests") != 0:
        raise ValueError(f"{benchmark_path} contains failed requests")

    if benchmark_path == "ipc":
        counter_deltas = summary.get("measurement_counter_deltas")
        expected = {
            "rejected_requests",
            "backpressure_requests",
            "cancelled_requests",
        }
        if not isinstance(counter_deltas, Mapping) or set(counter_deltas) != expected:
            raise ValueError("IPC summary has incomplete measurement counter deltas")
        nonzero = {
            name: value
            for name, value in counter_deltas.items()
            if type(value) is not int or value != 0
        }
        if nonzero:
            raise ValueError(f"IPC measurement contains control events: {nonzero}")

    if benchmark_path == "http_sse":
        if summary.get("mode") != "closed-loop":
            raise ValueError("HTTP comparison requires closed-loop mode")
        count_fields = (
            "offered_requests",
            "accepted_requests",
            "completed_requests",
        )
        if any(summary.get(name) != measured for name in count_fields):
            raise ValueError("HTTP run did not offer, accept, and complete every request")
        zero_fields = (
            "rejected_requests",
            "terminal_error_requests",
            "client_failure_requests",
            "slow_client_requests",
        )
        nonzero = {name: summary.get(name) for name in zero_fields if summary.get(name) != 0}
        if nonzero:
            raise ValueError(f"HTTP measurement contains unexpected outcomes: {nonzero}")
        if summary.get("resources_released") is not True:
            raise ValueError("HTTP resources did not return to zero")
        load = summary.get("load")
        if not isinstance(load, Mapping) or load.get("slow_request_every") != 0:
            raise ValueError("HTTP layered baseline must not include slow clients")


def read_http_outputs(path: Path) -> list[tuple[int, int, str]]:
    results: list[tuple[int, int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "request_id",
            "outcome",
            "protocol_valid",
            "completion_tokens",
            "output_text",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"HTTP request CSV is missing required columns: {path}")
        for row in reader:
            if row["outcome"] != "completed":
                raise ValueError(
                    f"HTTP request {row['request_id']} produced {row['outcome']}"
                )
            if row["protocol_valid"] not in {"1", "True", "true"}:
                raise ValueError(
                    f"HTTP request {row['request_id']} violated the SSE protocol"
                )
            results.append(
                (
                    int(row["request_id"]),
                    int(row["completion_tokens"]),
                    row["output_text"],
                )
            )
    return sorted(results)


def compare_run(
    direct_summary: Mapping[str, Any],
    ipc_summary: Mapping[str, Any],
    http_summary: Mapping[str, Any],
    direct_requests: Path,
    ipc_requests: Path,
    http_requests: Path,
    tokenizer: HuggingFaceTokenizer,
) -> dict[str, Any]:
    validate_summary(direct_summary, "direct")
    validate_summary(ipc_summary, "ipc")
    validate_summary(http_summary, "http_sse")

    workloads = {
        "direct": normalized_workload(direct_summary),
        "ipc": normalized_workload(ipc_summary),
        "http": normalized_workload(http_summary),
    }
    if len({json.dumps(value, sort_keys=True) for value in workloads.values()}) != 1:
        raise ValueError(f"Direct, IPC, and HTTP do not describe the same workload: {workloads}")

    direct_outputs = read_request_outputs(direct_requests)
    ipc_outputs = read_request_outputs(ipc_requests)
    if direct_outputs != ipc_outputs:
        raise ValueError("Direct and IPC output Token IDs are not identical")

    measured = direct_summary["measured_requests"]
    if len(direct_outputs) != measured:
        raise ValueError("Direct/IPC request CSV row count differs from measured_requests")
    http_outputs = read_http_outputs(http_requests)
    if len(http_outputs) != measured:
        raise ValueError("HTTP request CSV row count differs from measured_requests")

    expected_text = [tokenizer.decode(tokens) for _, tokens in direct_outputs]
    observed_text = [text for _, _, text in http_outputs]
    if observed_text != expected_text:
        mismatch = next(
            index
            for index, (expected, observed) in enumerate(
                zip(expected_text, observed_text),
                start=1,
            )
            if expected != observed
        )
        raise ValueError(
            "HTTP decoded output differs from Direct/IPC Token IDs at measured "
            f"request {mismatch}"
        )
    expected_counts = [len(tokens) for _, tokens in direct_outputs]
    observed_counts = [count for _, count, _ in http_outputs]
    if observed_counts != expected_counts:
        raise ValueError("HTTP completion token counts differ from Direct/IPC")

    metrics: dict[str, Any] = {}
    for spec in METRICS:
        direct = extract_metric(direct_summary, spec.path)
        ipc = extract_metric(ipc_summary, spec.path)
        http = extract_metric(http_summary, spec.path)
        metrics[spec.name] = {
            "unit": spec.unit,
            "preference": spec.preference,
            "direct": direct,
            "ipc": ipc,
            "http": http,
            "ipc_minus_direct": ipc - direct,
            "http_minus_direct": http - direct,
            "http_minus_ipc": http - ipc,
            "ipc_relative_to_direct_percent": (
                None if direct == 0 else (ipc - direct) / direct * 100.0
            ),
            "http_relative_to_direct_percent": (
                None if direct == 0 else (http - direct) / direct * 100.0
            ),
            "http_relative_to_ipc_percent": (
                None if ipc == 0 else (http - ipc) / ipc * 100.0
            ),
        }

    return {
        "token_outputs_identical_direct_ipc": True,
        "http_text_matches_token_decode": True,
        "completion_token_counts_identical": True,
        "request_count": measured,
        "unique_output_text_sha256": sorted(
            {
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in observed_text
            }
        ),
        "metrics": metrics,
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _sample_stddev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot aggregate an empty run list")
    aggregate: dict[str, Any] = {}
    numeric_fields = (
        "direct",
        "ipc",
        "http",
        "ipc_minus_direct",
        "http_minus_direct",
        "http_minus_ipc",
        "ipc_relative_to_direct_percent",
        "http_relative_to_direct_percent",
        "http_relative_to_ipc_percent",
    )
    for spec in METRICS:
        comparisons = [run["metrics"][spec.name] for run in runs]
        result: dict[str, Any] = {
            "unit": spec.unit,
            "preference": spec.preference,
        }
        for field in numeric_fields:
            values = [
                float(comparison[field])
                for comparison in comparisons
                if comparison[field] is not None
            ]
            result[f"{field}_mean"] = _mean(values) if values else None
            result[f"{field}_sample_stddev"] = (
                _sample_stddev(values) if values else None
            )
        aggregate[spec.name] = result
    return aggregate


def execution_order(repetition: int) -> list[str]:
    offset = (repetition - 1) % len(PATHS)
    return list(PATHS[offset:] + PATHS[:offset])


def direct_command(options: Options, run_dir: Path, token_ids: Sequence[int]) -> list[str]:
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
        *(str(token) for token in token_ids),
    ]


def ipc_command(options: Options, run_dir: Path, token_ids: Sequence[int]) -> list[str]:
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
        str(run_dir / "ipc-worker.log"),
        "--startup-timeout",
        str(options.startup_timeout),
        "--input-token-ids",
        *(str(token) for token in token_ids),
    ]


def http_command(
    options: Options,
    run_dir: Path,
    messages_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "benchmark" / "llm_http_benchmark.py"),
        "--worker",
        str(options.worker),
        "--engine-dir",
        str(options.engine_dir),
        "--tokenizer-path",
        str(options.tokenizer_path),
        "--model",
        options.model,
        "--revision",
        options.revision,
        "--messages-file",
        str(messages_path),
        "--mode",
        "closed-loop",
        "--concurrency",
        str(options.concurrency),
        "--max-inflight",
        str(options.concurrency),
        "--max-pending-requests",
        str(options.concurrency),
        "--warmup-requests",
        str(options.warmup_requests),
        "--measured-requests",
        str(options.measured_requests),
        "--max-new-tokens",
        str(options.max_new_tokens),
        "--startup-timeout",
        str(options.startup_timeout),
        "--request-timeout",
        str(options.request_timeout),
        "--summary",
        str(run_dir / "http-summary.json"),
        "--requests-csv",
        str(run_dir / "http-requests.csv"),
        "--worker-log",
        str(run_dir / "http-worker.log"),
        "--gateway-log",
        str(run_dir / "http-gateway.log"),
    ]
    if options.allow_dirty:
        command.append("--allow-dirty")
    return command


def print_aggregate(aggregate: Mapping[str, Mapping[str, Any]]) -> None:
    print("\nDirect / IPC / HTTP paired comparison (mean across repetitions)")
    print(
        f"{'metric':32} {'direct':>11} {'ipc':>11} {'http':>11} "
        f"{'ipc-direct':>12} {'http-ipc':>11}"
    )
    for spec in METRICS:
        result = aggregate[spec.name]
        print(
            f"{spec.name:32} "
            f"{result['direct_mean']:11.3f} "
            f"{result['ipc_mean']:11.3f} "
            f"{result['http_mean']:11.3f} "
            f"{result['ipc_minus_direct_mean']:+12.3f} "
            f"{result['http_minus_ipc_mean']:+11.3f}"
        )


def execute(options: Options) -> Path:
    commit, dirty = repository_state()
    if dirty and not options.allow_dirty:
        raise RuntimeError(
            "repository has tracked changes; commit them first or use --allow-dirty "
            "for a non-formal smoke run"
        )

    tokenizer = HuggingFaceTokenizer.load(options.tokenizer_path)
    input_token_ids = tokenizer.encode_chat(options.messages)
    if len(input_token_ids) > MAX_INPUT_TOKENS:
        raise ValueError("tokenized workload exceeds Engine max_input_tokens")
    if len(input_token_ids) + options.max_new_tokens > MAX_SEQUENCE_TOKENS:
        raise ValueError("workload exceeds Engine max_sequence_tokens")

    options.output_dir.mkdir(parents=True, exist_ok=False)
    messages_path = options.output_dir / "messages.json"
    messages_path.write_text(
        json.dumps(
            list(options.messages),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = options.cuda_device
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    run_reports: list[dict[str, Any]] = []
    execution_orders: list[list[str]] = []
    for repetition in range(1, options.repetitions + 1):
        run_dir = options.output_dir / f"run-{repetition:02d}"
        run_dir.mkdir()
        commands = {
            "direct": direct_command(options, run_dir, input_token_ids),
            "ipc": ipc_command(options, run_dir, input_token_ids),
            "http": http_command(options, run_dir, messages_path),
        }
        order = execution_order(repetition)
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
        http_summary_path = run_dir / "http-summary.json"
        comparison = compare_run(
            load_json(direct_summary_path),
            load_json(ipc_summary_path),
            load_json(http_summary_path),
            run_dir / "direct-requests.csv",
            run_dir / "ipc-requests.csv",
            run_dir / "http-requests.csv",
            tokenizer,
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
                "http_summary": str(
                    http_summary_path.relative_to(options.output_dir)
                ),
            }
        )
        run_reports.append(comparison)

    aggregate = aggregate_runs(run_reports)
    messages_bytes = messages_path.read_bytes()
    report = {
        "schema_version": 1,
        "benchmark": "direct_vs_ipc_vs_http_layered_overhead",
        "design_reference": {
            "project": "NVIDIA TensorRT-LLM Triton backend",
            "source": TRITON_BENCHMARK_REFERENCE,
            "reviewed_on": "2026-08-25",
            "adopted": (
                "separate the core inference path from end-to-end serving costs "
                "and preserve machine-readable workload metadata"
            ),
            "adjusted": (
                "use a three-path rotating paired experiment and require output "
                "equivalence plus resource-release evidence"
            ),
            "not_adopted": (
                "Triton/gRPC transport and backend-specific request schema"
            ),
        },
        "interpretation": {
            "ipc_minus_direct": (
                "GenerationClient, JSON serialization, UDS, WorkerServer, "
                "GenerationRuntime, and bridge/egress incremental cost"
            ),
            "http_minus_ipc": (
                "HTTP/1.1, OpenAI adaptation, SSE delivery, and Gateway-side "
                "tokenization/detokenization incremental cost"
            ),
            "metric_boundaries": (
                "Direct and IPC start at submit; HTTP starts before connection and "
                "ends after SSE [DONE], so HTTP deltas are end-to-end layer costs, "
                "not transport-only microbenchmarks"
            ),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "engine_config_sha256": hashlib.sha256(
            (options.engine_dir / "config.json").read_bytes()
        ).hexdigest(),
        "tokenizer_fingerprint": tokenizer.tokenizer_fingerprint,
        "chat_template_fingerprint": tokenizer.chat_template_fingerprint,
        "messages_sha256": hashlib.sha256(messages_bytes).hexdigest(),
        "cuda_visible_devices": options.cuda_device,
        "workload": {
            "engine_dir": str(options.engine_dir),
            "tokenizer_path": str(options.tokenizer_path),
            "model": options.model,
            "revision": options.revision,
            "repetitions": options.repetitions,
            "concurrency": options.concurrency,
            "warmup_requests": options.warmup_requests,
            "measured_requests": options.measured_requests,
            "input_tokens": len(input_token_ids),
            "input_token_ids": list(input_token_ids),
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
        "balanced_execution_order_cycle": options.repetitions % len(PATHS) == 0,
        "execution_orders": execution_orders,
        "token_outputs_identical_direct_ipc": True,
        "http_text_matches_token_decode": True,
        "completion_token_counts_identical": True,
        "runs": run_reports,
        "aggregate": aggregate,
    }
    report_path = options.output_dir / "direct-ipc-http-comparison.json"
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
        print(
            f"Direct/IPC/HTTP benchmark failed: {exception}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
