#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "clients" / "python"))
sys.path.insert(0, str(REPOSITORY_ROOT / "gateway" / "python"))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmark"))

from http_benchmark_core import (  # noqa: E402
    BenchmarkOptions,
    OpenAiSseClient,
    RequestResult,
    RunResult,
    all_protocol_observations_valid,
    all_requests_completed,
    arrival_offsets,
    build_summary,
    first_non_completed_request,
    load_messages,
    run_closed_loop,
    run_open_loop,
    wait_for_resources_released,
    write_request_csv,
    write_summary,
)
from kim_llm_client import ModelManifest  # noqa: E402
from kim_llm_gateway import HuggingFaceTokenizer  # noqa: E402
from llm_ipc_benchmark import (  # noqa: E402
    MAX_CONCURRENCY,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_SEQUENCE_TOKENS,
    ManagedWorker,
    make_limits,
    write_worker_config,
)


DEFAULT_MODEL = "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_REVISION = "trtllm-0.16.0-engine"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0


class ManagedGateway:
    def __init__(
        self,
        config_path: Path,
        base_url: str,
        log_path: Path,
        startup_timeout: float,
    ) -> None:
        self._config_path = config_path
        self._base_url = base_url
        self._log_path = log_path
        self._startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen[str]] = None
        self._log = None

    def start(self) -> float:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        python_paths = (
            str(REPOSITORY_ROOT / "clients" / "python"),
            str(REPOSITORY_ROOT / "gateway" / "python"),
        )
        current_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            python_paths + ((current_python_path,) if current_python_path else ())
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        begin = time.perf_counter_ns()
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "kim_llm_gateway",
                "serve",
                "--config",
                str(self._config_path),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        deadline = time.monotonic() + self._startup_timeout
        last_error = "Gateway did not answer readiness"
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    "Gateway exited before readiness with code "
                    f"{return_code}; see {self._log_path}"
                )
            try:
                with urllib.request.urlopen(
                    f"{self._base_url}/readyz",
                    timeout=1.0,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    if response.status == 200 and payload.get("status") == "ready":
                        return (time.perf_counter_ns() - begin) / 1_000_000.0
            except (OSError, ValueError, urllib.error.URLError) as exception:
                last_error = str(exception)
            time.sleep(0.05)
        raise TimeoutError(
            f"timed out waiting for Gateway readiness: {last_error}"
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        terminated_by_harness = False
        try:
            if process is None:
                return
            if process.poll() is None:
                terminated_by_harness = True
                process.terminate()
                try:
                    process.wait(timeout=45.0)
                except subprocess.TimeoutExpired as exception:
                    process.kill()
                    process.wait(timeout=10.0)
                    raise RuntimeError(
                        "Gateway did not stop after SIGTERM and was killed"
                    ) from exception
            expected_return_codes = {0}
            if terminated_by_harness:
                expected_return_codes.add(-signal.SIGTERM)
            if process.returncode not in expected_return_codes:
                raise RuntimeError(
                    f"Gateway exited with code {process.returncode}; "
                    f"see {self._log_path}"
                )
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


def parse_options(argv: Optional[Sequence[str]] = None) -> BenchmarkOptions:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the OpenAI HTTP/SSE path with bounded closed-loop or "
            "open-loop load. The benchmark starts an isolated Worker and Gateway."
        )
    )
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    messages = parser.add_mutually_exclusive_group(required=True)
    messages.add_argument("--prompt")
    messages.add_argument("--messages-file", type=Path)
    parser.add_argument(
        "--mode",
        choices=("closed-loop", "open-loop"),
        default="closed-loop",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--measured-requests", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--offered-rate", type=float)
    parser.add_argument(
        "--arrival-distribution",
        choices=("constant", "poisson"),
        default="constant",
    )
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--slow-request-every", type=int, default=0)
    parser.add_argument("--slow-read-delay-ms", type=float, default=0.0)
    parser.add_argument("--max-pending-requests", type=int, default=MAX_CONCURRENCY)
    parser.add_argument(
        "--max-client-delta-events-per-request",
        type=int,
        default=MAX_OUTPUT_TOKENS + 1,
    )
    parser.add_argument(
        "--max-sse-delta-events-per-request",
        type=int,
        default=MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--requests-csv", required=True, type=Path)
    parser.add_argument("--worker-log", required=True, type=Path)
    parser.add_argument("--gateway-log", required=True, type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    if not args.worker.is_file():
        parser.error(f"llm_worker does not exist: {args.worker}")
    if not args.engine_dir.is_dir() or not (args.engine_dir / "config.json").is_file():
        parser.error(f"invalid Engine directory: {args.engine_dir}")
    if not args.tokenizer_path.is_dir():
        parser.error(f"Tokenizer directory does not exist: {args.tokenizer_path}")
    if not args.model or not args.revision:
        parser.error("model and revision must be non-empty")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"concurrency must be in [1, {MAX_CONCURRENCY}]")
    if not 1 <= args.max_inflight <= 256:
        parser.error("max_inflight must be in [1, 256]")
    if args.warmup_requests < 0 or args.measured_requests <= 0:
        parser.error("warmup_requests must be non-negative and measured_requests positive")
    if not 1 <= args.max_new_tokens <= MAX_OUTPUT_TOKENS:
        parser.error(f"max_new_tokens must be in [1, {MAX_OUTPUT_TOKENS}]")
    if args.mode == "open-loop":
        if (
            args.offered_rate is None
            or not math.isfinite(args.offered_rate)
            or args.offered_rate <= 0
        ):
            parser.error("open-loop mode requires a finite positive offered_rate")
    elif args.offered_rate is not None:
        parser.error("offered_rate is only valid in open-loop mode")
    for name in ("ttft_slo_ms", "e2e_slo_ms"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if args.slow_request_every < 0:
        parser.error("slow_request_every must not be negative")
    if not math.isfinite(args.slow_read_delay_ms) or args.slow_read_delay_ms < 0:
        parser.error("slow_read_delay_ms must be finite and non-negative")
    if args.slow_request_every and args.slow_read_delay_ms <= 0:
        parser.error("slow_request_every requires a positive slow_read_delay_ms")
    if not 1 <= args.max_pending_requests <= MAX_CONCURRENCY:
        parser.error(f"max_pending_requests must be in [1, {MAX_CONCURRENCY}]")
    if args.max_client_delta_events_per_request <= 0:
        parser.error("max_client_delta_events_per_request must be positive")
    if not 1 <= args.max_sse_delta_events_per_request <= args.max_client_delta_events_per_request:
        parser.error("SSE capacity must be positive and no larger than client capacity")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.host != "127.0.0.1":
        parser.error("benchmark-managed Gateway must bind to 127.0.0.1")
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")

    evidence_paths = tuple(
        path.resolve()
        for path in (
            args.summary,
            args.requests_csv,
            args.worker_log,
            args.gateway_log,
        )
    )
    if len(set(evidence_paths)) != len(evidence_paths):
        parser.error("summary, CSV, and log paths must be distinct")
    existing_paths = [path for path in evidence_paths if path.exists()]
    if existing_paths:
        parser.error(
            "evidence paths already exist; use a new result directory: "
            + ", ".join(str(path) for path in existing_paths)
        )

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

    return BenchmarkOptions(
        worker=args.worker.resolve(),
        engine_dir=args.engine_dir.resolve(),
        tokenizer_path=args.tokenizer_path.resolve(),
        model=args.model,
        revision=args.revision,
        messages=parsed_messages,
        mode=args.mode,
        concurrency=args.concurrency,
        max_inflight=args.max_inflight,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        max_new_tokens=args.max_new_tokens,
        offered_rate=args.offered_rate,
        arrival_distribution=args.arrival_distribution,
        arrival_seed=args.arrival_seed,
        ttft_slo_ms=args.ttft_slo_ms,
        e2e_slo_ms=args.e2e_slo_ms,
        slow_request_every=args.slow_request_every,
        slow_read_delay_ms=args.slow_read_delay_ms,
        max_pending_requests=args.max_pending_requests,
        max_client_delta_events_per_request=args.max_client_delta_events_per_request,
        max_sse_delta_events_per_request=args.max_sse_delta_events_per_request,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        host=args.host,
        port=args.port,
        summary_path=evidence_paths[0],
        request_csv_path=evidence_paths[1],
        worker_log_path=evidence_paths[2],
        gateway_log_path=evidence_paths[3],
        allow_dirty=args.allow_dirty,
    )


def make_manifest(
    options: BenchmarkOptions,
    tokenizer: HuggingFaceTokenizer,
) -> ModelManifest:
    fingerprint = hashlib.sha256(
        (options.engine_dir / "config.json").read_bytes()
    ).hexdigest()
    return ModelManifest(
        model_id=options.model,
        revision=options.revision,
        tokenizer_fingerprint=tokenizer.tokenizer_fingerprint,
        chat_template_fingerprint=tokenizer.chat_template_fingerprint,
        engine_fingerprint=fingerprint,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_sequence_tokens=MAX_SEQUENCE_TOKENS,
        precision="fp16",
        max_batch_size=MAX_CONCURRENCY,
    )


def write_gateway_config(
    path: Path,
    worker_config_path: Path,
    options: BenchmarkOptions,
    port: int,
) -> None:
    config = {
        "worker_config_path": str(worker_config_path),
        "tokenizer_path": str(options.tokenizer_path),
        "host": options.host,
        "port": port,
        "log_level": "warning",
        "accept_timeout_seconds": 10.0,
        "request_timeout_seconds": options.request_timeout,
        "health_timeout_seconds": 2.0,
        "event_poll_seconds": 0.01,
        "cancel_drain_timeout_seconds": 30.0,
        "shutdown_grace_seconds": 30.0,
        "max_http_body_bytes": 1024 * 1024,
        "max_pending_requests": options.max_pending_requests,
        "max_client_delta_events_per_request": (
            options.max_client_delta_events_per_request
        ),
        "max_sse_delta_events_per_request": (
            options.max_sse_delta_events_per_request
        ),
        "sampling_top_k": 1,
    }
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reserve_loopback_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


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


def execute(options: BenchmarkOptions) -> int:
    commit, dirty = repository_state()
    if dirty and not options.allow_dirty:
        raise RuntimeError(
            "repository has tracked changes; commit them first or use "
            "--allow-dirty for a non-formal smoke run"
        )
    tokenizer = HuggingFaceTokenizer.load(options.tokenizer_path)
    input_token_ids = tokenizer.encode_chat(options.messages)
    if len(input_token_ids) > MAX_INPUT_TOKENS:
        raise ValueError("tokenized HTTP workload exceeds Engine max_input_tokens")
    if len(input_token_ids) + options.max_new_tokens > MAX_SEQUENCE_TOKENS:
        raise ValueError("HTTP workload exceeds Engine max_sequence_tokens")
    manifest = make_manifest(options, tokenizer)

    measured: Optional[RunResult] = None
    metrics_before: Optional[dict[str, float]] = None
    metrics_after: Optional[dict[str, float]] = None
    worker_startup_ms = 0.0
    gateway_startup_ms = 0.0

    with tempfile.TemporaryDirectory(prefix="kim-llm-http-benchmark-") as directory:
        temporary = Path(directory)
        socket_path = temporary / "worker.sock"
        worker_config = temporary / "worker-config.json"
        gateway_config = temporary / "gateway-config.json"
        port = options.port or reserve_loopback_port(options.host)
        base_url = f"http://{options.host}:{port}"
        write_worker_config(
            worker_config,
            options.engine_dir,
            socket_path,
            manifest,
            make_limits(),
        )
        write_gateway_config(gateway_config, worker_config, options, port)
        worker = ManagedWorker(
            options.worker,
            worker_config,
            socket_path,
            options.worker_log_path,
            options.startup_timeout,
        )
        gateway = ManagedGateway(
            gateway_config,
            base_url,
            options.gateway_log_path,
            options.startup_timeout,
        )
        client = OpenAiSseClient(
            base_url,
            options.model,
            options.messages,
            options.max_new_tokens,
            len(input_token_ids),
            options.request_timeout,
            options.slow_read_delay_ms,
        )
        failure: Optional[BaseException] = None
        try:
            worker_startup_ms = worker.start()
            gateway_startup_ms = gateway.start()
            warmup = run_closed_loop(
                client.run,
                options.concurrency,
                options.warmup_requests,
                1,
            )
            if not all_requests_completed(warmup):
                raise RuntimeError(
                    "HTTP warmup did not complete successfully: "
                    f"{first_non_completed_request(warmup)}"
                )
            metrics_before = wait_for_resources_released(base_url)
            first_request_id = 1 + options.warmup_requests
            if options.mode == "closed-loop":
                measured = run_closed_loop(
                    client.run,
                    options.concurrency,
                    options.measured_requests,
                    first_request_id,
                    options.slow_request_every,
                )
            else:
                assert options.offered_rate is not None
                measured = run_open_loop(
                    client.run,
                    options.max_inflight,
                    options.measured_requests,
                    first_request_id,
                    options.offered_rate,
                    options.arrival_distribution,
                    options.arrival_seed,
                    options.slow_request_every,
                )
            metrics_after = wait_for_resources_released(base_url)
        except BaseException as exception:
            failure = exception
        finally:
            try:
                gateway.stop()
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
    assert metrics_before is not None
    assert metrics_after is not None
    write_request_csv(options.request_csv_path, measured.requests)
    summary = build_summary(
        options,
        input_token_ids,
        measured,
        worker_startup_ms,
        gateway_startup_ms,
        metrics_before,
        metrics_after,
    )
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_dirty": dirty,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "manifest": asdict(manifest),
        }
    )
    write_summary(options.summary_path, summary)
    if not all_protocol_observations_valid(measured):
        return 4
    if options.mode == "closed-loop" and not all_requests_completed(measured):
        return 4
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_options(argv)
    try:
        return execute(options)
    except Exception as exception:
        print(f"HTTP benchmark failed: {exception}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
