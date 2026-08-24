#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "clients" / "python"))

from kim_llm_client import (  # noqa: E402
    GenerationClient,
    GenerationClientConfig,
    GenerationHandle,
    GenerationRequest,
    ModelManifest,
    SamplingParameters,
    Stats,
    Terminal,
    TokenDelta,
    WorkerLimits,
)


MAX_CONCURRENCY = 8
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 32
MAX_SEQUENCE_TOKENS = 544
REQUEST_TIMEOUT_SECONDS = 180.0
ACCEPT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class BenchmarkOptions:
    worker: Path
    engine_dir: Path
    concurrency: int
    warmup_requests: int
    measured_requests: int
    max_new_tokens: int
    summary_path: Path
    request_csv_path: Path
    worker_log_path: Path
    startup_timeout: float
    input_token_ids: tuple[int, ...]


@dataclass
class RequestResult:
    request_id: int
    success: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output_token_ids: tuple[int, ...] = ()
    ttft_ms: float = -1.0
    tpot_ms: float = -1.0
    e2e_ms: float = -1.0
    error: str = ""


@dataclass(frozen=True)
class RunResult:
    requests: tuple[RequestResult, ...]
    duration_seconds: float


class ManagedWorker:
    def __init__(
        self,
        worker: Path,
        config_path: Path,
        socket_path: Path,
        log_path: Path,
        startup_timeout: float,
    ) -> None:
        self._worker = worker
        self._config_path = config_path
        self._socket_path = socket_path
        self._log_path = log_path
        self._startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen[str]] = None
        self._log = None

    def start(self) -> float:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._log_path.open("w", encoding="utf-8")
        begin = time.perf_counter_ns()
        self._process = subprocess.Popen(
            [str(self._worker), str(self._config_path)],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "llm_worker exited before readiness with code "
                    f"{return_code}; see {self._log_path}"
                )
            if self._socket_path.exists():
                return (time.perf_counter_ns() - begin) / 1_000_000.0
            time.sleep(0.01)
        raise TimeoutError(
            f"timed out waiting for llm_worker socket {self._socket_path}"
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is None:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired as exception:
                    process.kill()
                    process.wait(timeout=10.0)
                    raise RuntimeError(
                        "llm_worker did not stop after SIGTERM and was killed"
                    ) from exception
            if process.returncode != 0:
                raise RuntimeError(
                    f"llm_worker exited with code {process.returncode}; "
                    f"see {self._log_path}"
                )
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


def parse_options(argv: Optional[Sequence[str]] = None) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description="Measure the persistent Python/UDS/Worker IPC path."
    )
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--warmup-requests", required=True, type=int)
    parser.add_argument("--measured-requests", required=True, type=int)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--requests-csv", required=True, type=Path)
    parser.add_argument("--worker-log", required=True, type=Path)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument(
        "--input-token-ids",
        required=True,
        type=int,
        nargs="+",
    )
    args = parser.parse_args(argv)

    if not args.worker.is_file():
        parser.error(f"llm_worker does not exist: {args.worker}")
    if not args.engine_dir.is_dir():
        parser.error(f"Engine directory does not exist: {args.engine_dir}")
    if not (args.engine_dir / "config.json").is_file():
        parser.error("Engine directory does not contain config.json")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    if args.warmup_requests < 0:
        parser.error("warmup_requests must not be negative")
    if args.measured_requests <= 0:
        parser.error("measured_requests must be greater than zero")
    if not 1 <= args.max_new_tokens <= MAX_OUTPUT_TOKENS:
        parser.error(f"max_new_tokens must be in [1, {MAX_OUTPUT_TOKENS}]")
    if args.startup_timeout <= 0:
        parser.error("startup_timeout must be positive")
    if any(token_id < 0 for token_id in args.input_token_ids):
        parser.error("input token IDs must not be negative")
    if len(args.input_token_ids) > MAX_INPUT_TOKENS:
        parser.error("input token count exceeds Engine max_input_tokens")
    if len(args.input_token_ids) + args.max_new_tokens > MAX_SEQUENCE_TOKENS:
        parser.error("request exceeds Engine max_sequence_tokens")

    return BenchmarkOptions(
        worker=args.worker.resolve(),
        engine_dir=args.engine_dir.resolve(),
        concurrency=args.concurrency,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        max_new_tokens=args.max_new_tokens,
        summary_path=args.summary.resolve(),
        request_csv_path=args.requests_csv.resolve(),
        worker_log_path=args.worker_log.resolve(),
        startup_timeout=args.startup_timeout,
        input_token_ids=tuple(args.input_token_ids),
    )


def make_manifest(engine_dir: Path) -> ModelManifest:
    fingerprint = hashlib.sha256(
        (engine_dir / "config.json").read_bytes()
    ).hexdigest()
    return ModelManifest(
        model_id="TinyLlama-1.1B-Chat-v1.0",
        revision="trtllm-0.16.0-engine",
        tokenizer_fingerprint="token-id-benchmark",
        chat_template_fingerprint="token-id-benchmark",
        engine_fingerprint=fingerprint,
        eos_token_id=2,
        pad_token_id=2,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_sequence_tokens=MAX_SEQUENCE_TOKENS,
        precision="fp16",
        max_batch_size=MAX_CONCURRENCY,
    )


def make_limits() -> WorkerLimits:
    return WorkerLimits(
        max_active_requests=MAX_CONCURRENCY,
        max_total_input_tokens=MAX_CONCURRENCY * MAX_INPUT_TOKENS,
        max_reserved_output_tokens=MAX_CONCURRENCY * MAX_OUTPUT_TOKENS,
        max_frame_payload_bytes=1024 * 1024,
        max_session_egress_frames=1024,
        max_session_egress_bytes=16 * 1024 * 1024,
        max_request_egress_frames=64,
        max_request_egress_bytes=1024 * 1024,
    )


def write_worker_config(
    path: Path,
    engine_dir: Path,
    socket_path: Path,
    manifest: ModelManifest,
    limits: WorkerLimits,
) -> None:
    config = {
        "engine_dir": str(engine_dir),
        "socket_path": str(socket_path),
        "manifest": asdict(manifest),
        "limits": asdict(limits),
    }
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_one_request(
    client: GenerationClient,
    request_id: int,
    input_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> RequestResult:
    result = RequestResult(request_id=request_id)
    handle: Optional[GenerationHandle] = None
    output_tokens: list[int] = []
    first_token_ns: Optional[int] = None
    last_token_ns: Optional[int] = None
    begin_ns = time.perf_counter_ns()
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS

    try:
        handle = client.submit(
            GenerationRequest(
                request_id=request_id,
                input_token_ids=input_token_ids,
                max_new_tokens=max_new_tokens,
                timeout_ms=int(REQUEST_TIMEOUT_SECONDS * 1000),
                trace_id=f"ipc-benchmark-{request_id}",
                streaming=True,
                sampling=SamplingParameters(
                    temperature=1.0,
                    top_k=1,
                    top_p=1.0,
                    random_seed=0,
                ),
            ),
            timeout=ACCEPT_TIMEOUT_SECONDS,
        )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request timed out")
            event = handle.next_event(timeout=remaining)
            if isinstance(event, TokenDelta):
                receive_ns = time.perf_counter_ns()
                if first_token_ns is None:
                    first_token_ns = receive_ns
                last_token_ns = receive_ns
                output_tokens.extend(event.token_ids)
                continue

            if not isinstance(event, Terminal):
                raise RuntimeError("GenerationClient returned an unknown event")
            result.prompt_tokens = event.usage.prompt_tokens
            result.completion_tokens = event.usage.completion_tokens
            if not event.status.ok:
                raise RuntimeError(
                    f"Terminal {event.status.code}: {event.status.message}"
                )
            if result.prompt_tokens != len(input_token_ids):
                raise RuntimeError("Terminal prompt token usage is incorrect")
            if result.completion_tokens != len(output_tokens):
                raise RuntimeError(
                    "TokenDelta count differs from Terminal completion usage"
                )
            result.success = True
            break
    except Exception as exception:
        result.error = f"{type(exception).__name__}: {exception}"
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
    finally:
        end_ns = time.perf_counter_ns()
        result.e2e_ms = (end_ns - begin_ns) / 1_000_000.0
        result.output_token_ids = tuple(output_tokens)
        if first_token_ns is not None:
            result.ttft_ms = (first_token_ns - begin_ns) / 1_000_000.0
        if (
            first_token_ns is not None
            and last_token_ns is not None
            and len(output_tokens) > 1
        ):
            result.tpot_ms = (
                (last_token_ns - first_token_ns)
                / 1_000_000.0
                / (len(output_tokens) - 1)
            )

    return result


def run_closed_loop(
    client: GenerationClient,
    concurrency: int,
    total_requests: int,
    first_request_id: int,
    input_token_ids: tuple[int, ...],
    max_new_tokens: int,
) -> RunResult:
    if total_requests == 0:
        return RunResult((), 0.0)

    worker_count = min(concurrency, total_requests)
    results: list[Optional[RequestResult]] = [None] * total_requests
    next_index = 0
    index_lock = threading.Lock()
    start_barrier = threading.Barrier(worker_count + 1)

    def worker_loop() -> None:
        nonlocal next_index
        start_barrier.wait()
        while True:
            with index_lock:
                index = next_index
                next_index += 1
            if index >= total_requests:
                return
            results[index] = run_one_request(
                client,
                first_request_id + index,
                input_token_ids,
                max_new_tokens,
            )

    workers = [
        threading.Thread(
            target=worker_loop,
            name=f"kim-llm-ipc-benchmark-{index}",
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    begin_ns = time.perf_counter_ns()
    start_barrier.wait()
    for worker in workers:
        worker.join()
    duration_seconds = (time.perf_counter_ns() - begin_ns) / 1_000_000_000.0

    completed = tuple(result for result in results if result is not None)
    if len(completed) != total_requests:
        raise RuntimeError("IPC benchmark worker thread lost a request result")
    return RunResult(completed, duration_seconds)


def all_requests_successful(run: RunResult) -> bool:
    return all(request.success for request in run.requests)


def first_failure_message(phase: str, run: RunResult) -> str:
    for request in run.requests:
        if not request.success:
            suffix = f": {request.error}" if request.error else ""
            return f"{phase} request {request.request_id} failed{suffix}"
    return ""


def wait_for_resources_released(
    client: GenerationClient,
    timeout: float = 10.0,
) -> Stats:
    deadline = time.monotonic() + timeout
    last_snapshot: Optional[Stats] = None
    while time.monotonic() < deadline:
        last_snapshot = client.health(timeout=2.0)
        if (
            last_snapshot.active_requests == 0
            and last_snapshot.reserved_input_tokens == 0
            and last_snapshot.reserved_output_tokens == 0
        ):
            return last_snapshot
        time.sleep(0.05)
    raise RuntimeError(
        "Worker resources did not return to zero after benchmark: "
        f"{last_snapshot}"
    )


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    return ordered[max(1, rank) - 1]


def distribution(values: Sequence[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    return {
        "p50": nearest_rank_percentile(values, 0.50),
        "p95": nearest_rank_percentile(values, 0.95),
        "p99": nearest_rank_percentile(values, 0.99),
    }


def write_request_csv(path: Path, requests: Sequence[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for request in requests:
            writer.writerow(
                (
                    request.request_id,
                    int(request.success),
                    request.prompt_tokens,
                    request.completion_tokens,
                    " ".join(str(token) for token in request.output_token_ids),
                    f"{request.ttft_ms:.3f}" if request.ttft_ms >= 0 else "",
                    f"{request.tpot_ms:.3f}" if request.tpot_ms >= 0 else "",
                    f"{request.e2e_ms:.3f}",
                    request.error,
                )
            )


def write_summary(
    options: BenchmarkOptions,
    run: RunResult,
    worker_startup_ms: float,
    client_connect_ms: float,
    stats_before_measurement: Stats,
    final_stats: Stats,
) -> None:
    successful = [request for request in run.requests if request.success]
    completion_tokens = sum(request.completion_tokens for request in successful)
    request_throughput = (
        len(successful) / run.duration_seconds if run.duration_seconds > 0 else 0.0
    )
    token_throughput = (
        completion_tokens / run.duration_seconds
        if run.duration_seconds > 0
        else 0.0
    )
    summary = {
        "schema_version": 2,
        "benchmark_path": "ipc",
        "engine_dir": str(options.engine_dir),
        "concurrency": options.concurrency,
        "warmup_requests": options.warmup_requests,
        "measured_requests": options.measured_requests,
        "successful_requests": len(successful),
        "failed_requests": options.measured_requests - len(successful),
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
        "worker_startup_ms": worker_startup_ms,
        "client_connect_ms": client_connect_ms,
        "duration_seconds": run.duration_seconds,
        "request_throughput_rps": request_throughput,
        "output_token_throughput_tps": token_throughput,
        "ttft_ms": distribution(
            [request.ttft_ms for request in successful if request.ttft_ms >= 0]
        ),
        "tpot_ms": distribution(
            [request.tpot_ms for request in successful if request.tpot_ms >= 0]
        ),
        "e2e_ms": distribution([request.e2e_ms for request in successful]),
        "worker_stats_before_measurement": asdict(stats_before_measurement),
        "worker_stats_after_measurement": asdict(final_stats),
        "measurement_counter_deltas": {
            "rejected_requests": (
                final_stats.rejected_requests
                - stats_before_measurement.rejected_requests
            ),
            "backpressure_requests": (
                final_stats.backpressure_requests
                - stats_before_measurement.backpressure_requests
            ),
            "cancelled_requests": (
                final_stats.cancelled_requests
                - stats_before_measurement.cancelled_requests
            ),
        },
    }
    options.summary_path.parent.mkdir(parents=True, exist_ok=True)
    options.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def execute(options: BenchmarkOptions) -> int:
    manifest = make_manifest(options.engine_dir)
    limits = make_limits()
    measured: Optional[RunResult] = None
    worker_startup_ms = 0.0
    client_connect_ms = 0.0
    stats_before_measurement: Optional[Stats] = None
    final_stats: Optional[Stats] = None

    with tempfile.TemporaryDirectory(prefix="kim-llm-ipc-benchmark-") as directory:
        temporary = Path(directory)
        socket_path = temporary / "worker.sock"
        config_path = temporary / "worker-config.json"
        write_worker_config(
            config_path,
            options.engine_dir,
            socket_path,
            manifest,
            limits,
        )
        worker = ManagedWorker(
            options.worker,
            config_path,
            socket_path,
            options.worker_log_path,
            options.startup_timeout,
        )
        client: Optional[GenerationClient] = None
        failure: Optional[BaseException] = None
        try:
            worker_startup_ms = worker.start()
            client = GenerationClient(
                GenerationClientConfig(
                    socket_path=str(socket_path),
                    expected_manifest=manifest,
                    connect_timeout=2.0,
                    handshake_timeout=10.0,
                    max_pending_requests=options.concurrency,
                    max_delta_events_per_request=MAX_OUTPUT_TOKENS + 1,
                )
            )
            connect_begin_ns = time.perf_counter_ns()
            client.connect()
            client_connect_ms = (
                time.perf_counter_ns() - connect_begin_ns
            ) / 1_000_000.0
            ready = client.health(timeout=5.0)
            if not ready.ready or not ready.status.ok:
                raise RuntimeError(f"Worker is not ready: {ready}")

            warmup = run_closed_loop(
                client,
                options.concurrency,
                options.warmup_requests,
                1,
                options.input_token_ids,
                options.max_new_tokens,
            )
            if not all_requests_successful(warmup):
                raise RuntimeError(first_failure_message("warmup", warmup))
            stats_before_measurement = wait_for_resources_released(client)

            measured = run_closed_loop(
                client,
                options.concurrency,
                options.measured_requests,
                1 + options.warmup_requests,
                options.input_token_ids,
                options.max_new_tokens,
            )
            final_stats = wait_for_resources_released(client)
        except BaseException as exception:
            failure = exception
        finally:
            if client is not None:
                try:
                    client.close()
                except BaseException as exception:
                    if failure is None:
                        failure = exception
            try:
                worker.stop()
            except BaseException as exception:
                if failure is None:
                    failure = exception
        if failure is not None:
            raise failure

    assert measured is not None
    assert stats_before_measurement is not None
    assert final_stats is not None
    write_request_csv(options.request_csv_path, measured.requests)
    write_summary(
        options,
        measured,
        worker_startup_ms,
        client_connect_ms,
        stats_before_measurement,
        final_stats,
    )
    return 0 if all_requests_successful(measured) else 4


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_options(argv)
    try:
        return execute(options)
    except Exception as exception:
        print(f"IPC benchmark failed: {exception}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
