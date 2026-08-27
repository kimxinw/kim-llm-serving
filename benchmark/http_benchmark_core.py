#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import http.client
import json
import math
import random
import socket
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmark"))

from llm_ipc_benchmark import distribution  # noqa: E402


MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class BenchmarkOptions:
    worker: Path
    engine_dir: Path
    tokenizer_path: Path
    model: str
    revision: str
    messages: tuple[Mapping[str, str], ...]
    mode: str
    concurrency: int
    max_inflight: int
    warmup_requests: int
    measured_requests: int
    max_new_tokens: int
    offered_rate: Optional[float]
    arrival_distribution: str
    arrival_seed: int
    ttft_slo_ms: Optional[float]
    e2e_slo_ms: Optional[float]
    slow_request_every: int
    slow_read_delay_ms: float
    max_pending_requests: int
    max_client_delta_events_per_request: int
    max_sse_delta_events_per_request: int
    startup_timeout: float
    request_timeout: float
    host: str
    port: int
    summary_path: Path
    request_csv_path: Path
    worker_log_path: Path
    gateway_log_path: Path
    allow_dirty: bool
    admission_strategy: str = "fixed-concurrency"
    worker_max_active_requests: int = 8
    worker_max_total_input_tokens: int = 4096
    worker_max_reserved_output_tokens: int = 256
    slo_policy: Optional[Mapping[str, object]] = None
    workload_variants: tuple[tuple[Mapping[str, str], ...], ...] = ()


@dataclass
class RequestResult:
    request_id: int
    outcome: str = "client_error"
    protocol_valid: bool = False
    accepted: bool = False
    slow_client: bool = False
    http_status: int = 0
    status_code: str = ""
    workload_index: int = 0
    input_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output_text: str = ""
    finish_reason: str = ""
    scheduled_offset_ms: float = 0.0
    scheduling_lag_ms: float = 0.0
    ttft_ms: float = -1.0
    tpot_ms: float = -1.0
    e2e_ms: float = -1.0
    error: str = ""


@dataclass(frozen=True)
class RunResult:
    requests: tuple[RequestResult, ...]
    duration_seconds: float
    load_generation_seconds: float


RequestRunner = Callable[[int, int, bool], RequestResult]


class OpenAiSseClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_new_tokens: int,
        expected_prompt_tokens: int,
        request_timeout: float,
        slow_read_delay_ms: float,
        workload_variants: Optional[
            Sequence[tuple[Sequence[Mapping[str, str]], int]]
        ] = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("HTTP benchmark base URL must use http://host[:port]")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        prefix = parsed.path.rstrip("/")
        self._path = f"{prefix}/v1/chat/completions"
        self._request_timeout = request_timeout
        self._slow_read_delay_seconds = slow_read_delay_ms / 1000.0
        variants = workload_variants or ((messages, expected_prompt_tokens),)
        self._variants = tuple(
            (
                json.dumps(
                    {
                        "model": model,
                        "messages": list(variant_messages),
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "max_tokens": max_new_tokens,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "seed": 0,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"),
                variant_tokens,
            )
            for variant_messages, variant_tokens in variants
        )
        if not self._variants:
            raise ValueError("HTTP benchmark requires at least one workload")

    def run(
        self,
        request_id: int,
        scheduled_ns: int,
        slow_client: bool,
    ) -> RequestResult:
        workload_index = (request_id - 1) % len(self._variants)
        body, expected_prompt_tokens = self._variants[workload_index]
        result = RequestResult(
            request_id=request_id,
            slow_client=slow_client,
            workload_index=workload_index,
            input_tokens=expected_prompt_tokens,
        )
        begin_ns = time.perf_counter_ns()
        result.scheduling_lag_ms = max(
            0.0,
            (begin_ns - scheduled_ns) / 1_000_000.0,
        )
        connection = http.client.HTTPConnection(
            self._host,
            self._port,
            timeout=self._request_timeout,
        )
        first_content_ns: Optional[int] = None
        last_content_ns: Optional[int] = None
        terminal_observed = False
        done_observed = False
        usage_observed = False
        terminal_error = ""
        try:
            connection.connect()
            if slow_client and connection.sock is not None:
                connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            connection.request(
                "POST",
                self._path,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Connection": "close",
                    "X-Kim-Trace-Id": f"http-benchmark-{request_id}",
                },
            )
            response = connection.getresponse()
            result.http_status = response.status
            if response.status != 200:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("HTTP error response exceeded size limit")
                result.outcome = "rejected"
                result.status_code = _error_code(payload) or f"http_{response.status}"
                result.protocol_valid = 400 <= response.status < 600
                return result

            result.accepted = True
            while True:
                line = response.readline(MAX_RESPONSE_BYTES + 1)
                if len(line) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("SSE line exceeded size limit")
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith(b"data: "):
                    raise RuntimeError("Gateway emitted a non-data SSE field")
                data = stripped[6:]
                if data == b"[DONE]":
                    done_observed = True
                    break
                value = json.loads(data.decode("utf-8"))
                if not isinstance(value, dict):
                    raise RuntimeError("Gateway emitted a non-object SSE payload")
                error = value.get("error")
                if error is not None:
                    if not isinstance(error, dict):
                        raise RuntimeError("Gateway emitted an invalid SSE error")
                    terminal_error = str(error.get("code") or "internal_error")
                    result.error = str(error.get("message") or terminal_error)
                    terminal_observed = True
                    continue

                usage = value.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise RuntimeError("Gateway emitted invalid usage")
                    result.prompt_tokens = _non_negative_int(
                        usage.get("prompt_tokens"),
                        "usage.prompt_tokens",
                    )
                    result.completion_tokens = _non_negative_int(
                        usage.get("completion_tokens"),
                        "usage.completion_tokens",
                    )
                    usage_observed = True

                choices = value.get("choices")
                if not isinstance(choices, list):
                    raise RuntimeError("Gateway emitted invalid choices")
                for choice in choices:
                    if not isinstance(choice, dict):
                        raise RuntimeError("Gateway emitted an invalid choice")
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None:
                        result.finish_reason = str(finish_reason)
                        terminal_observed = True
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        raise RuntimeError("Gateway emitted an invalid delta")
                    content = delta.get("content")
                    if content is None:
                        continue
                    if not isinstance(content, str):
                        raise RuntimeError("Gateway emitted non-string content")
                    if not content:
                        continue
                    received_ns = time.perf_counter_ns()
                    if first_content_ns is None:
                        first_content_ns = received_ns
                    last_content_ns = received_ns
                    result.output_text += content
                    if slow_client and self._slow_read_delay_seconds > 0:
                        time.sleep(self._slow_read_delay_seconds)

            if not done_observed:
                raise RuntimeError("SSE stream ended without [DONE]")
            if not terminal_observed:
                raise RuntimeError("SSE stream ended without a terminal event")
            if terminal_error:
                result.outcome = "terminal_error"
                result.status_code = terminal_error
            else:
                if not usage_observed:
                    raise RuntimeError("successful SSE stream did not report usage")
                if result.prompt_tokens != expected_prompt_tokens:
                    raise RuntimeError(
                        "HTTP usage prompt_tokens differs from the tokenized workload"
                    )
                result.outcome = "completed"
                result.status_code = "ok"
            result.protocol_valid = True
        except Exception as exception:
            result.outcome = "client_error"
            result.error = f"{type(exception).__name__}: {exception}"
        finally:
            end_ns = time.perf_counter_ns()
            result.e2e_ms = (end_ns - begin_ns) / 1_000_000.0
            if first_content_ns is not None:
                result.ttft_ms = (
                    first_content_ns - begin_ns
                ) / 1_000_000.0
            if (
                first_content_ns is not None
                and last_content_ns is not None
                and result.completion_tokens > 1
            ):
                result.tpot_ms = (
                    (last_content_ns - first_content_ns)
                    / 1_000_000.0
                    / (result.completion_tokens - 1)
                )
            connection.close()
        return result


def _non_negative_int(value: object, description: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{description} must be a non-negative integer")
    return value


def _error_code(payload: bytes) -> str:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError):
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("error"), dict):
        return ""
    code = value["error"].get("code")
    return code if isinstance(code, str) else ""


def load_messages(value: object) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty JSON array")
    messages: list[Mapping[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError(
                f"messages[{index}] must contain exactly role and content"
            )
        role = item["role"]
        content = item["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role is unsupported")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content must be a string")
        messages.append({"role": role, "content": content})
    return tuple(messages)


def is_slow_request(index: int, every: int) -> bool:
    return every > 0 and (index + 1) % every == 0


def run_closed_loop(
    run_request: RequestRunner,
    concurrency: int,
    total_requests: int,
    first_request_id: int,
    slow_request_every: int = 0,
) -> RunResult:
    if total_requests == 0:
        return RunResult((), 0.0, 0.0)
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
            scheduled_ns = time.perf_counter_ns()
            result = run_request(
                first_request_id + index,
                scheduled_ns,
                is_slow_request(index, slow_request_every),
            )
            result.scheduled_offset_ms = 0.0
            results[index] = result

    workers = [
        threading.Thread(
            target=worker_loop,
            name=f"kim-llm-http-closed-loop-{index}",
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    begin_ns = time.perf_counter_ns()
    start_barrier.wait()
    for worker in workers:
        worker.join()
    duration = (time.perf_counter_ns() - begin_ns) / 1_000_000_000.0
    completed = tuple(result for result in results if result is not None)
    if len(completed) != total_requests:
        raise RuntimeError("closed-loop scheduler lost a request result")
    return RunResult(completed, duration, duration)


def arrival_offsets(
    total_requests: int,
    offered_rate: float,
    distribution_name: str,
    seed: int,
) -> tuple[float, ...]:
    if total_requests <= 0:
        return ()
    generator = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, total_requests):
        if distribution_name == "constant":
            interval = 1.0 / offered_rate
        elif distribution_name == "poisson":
            interval = generator.expovariate(offered_rate)
        else:
            raise ValueError(f"unsupported arrival distribution {distribution_name}")
        offsets.append(offsets[-1] + interval)
    return tuple(offsets)


def run_open_loop(
    run_request: RequestRunner,
    max_inflight: int,
    total_requests: int,
    first_request_id: int,
    offered_rate: float,
    distribution_name: str,
    seed: int,
    slow_request_every: int = 0,
) -> RunResult:
    offsets = arrival_offsets(
        total_requests,
        offered_rate,
        distribution_name,
        seed,
    )
    if not offsets:
        return RunResult((), 0.0, 0.0)

    results: list[Optional[RequestResult]] = [None] * total_requests
    capacity = threading.BoundedSemaphore(max_inflight)
    futures: list[concurrent.futures.Future[RequestResult]] = []
    begin_ns = time.perf_counter_ns()

    def execute_one(index: int, scheduled_ns: int) -> RequestResult:
        try:
            return run_request(
                first_request_id + index,
                scheduled_ns,
                is_slow_request(index, slow_request_every),
            )
        finally:
            capacity.release()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_inflight,
        thread_name_prefix="kim-llm-http-open-loop",
    ) as executor:
        for index, offset in enumerate(offsets):
            scheduled_ns = begin_ns + int(offset * 1_000_000_000)
            remaining = (scheduled_ns - time.perf_counter_ns()) / 1_000_000_000.0
            if remaining > 0:
                time.sleep(remaining)
            if not capacity.acquire(blocking=False):
                results[index] = RequestResult(
                    request_id=first_request_id + index,
                    outcome="client_overflow",
                    slow_client=is_slow_request(index, slow_request_every),
                    scheduled_offset_ms=offset * 1000.0,
                    scheduling_lag_ms=max(
                        0.0,
                        (time.perf_counter_ns() - scheduled_ns) / 1_000_000.0,
                    ),
                    error="benchmark max_inflight capacity was exhausted",
                )
                continue
            future = executor.submit(execute_one, index, scheduled_ns)
            setattr(future, "kim_request_index", index)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            index = int(getattr(future, "kim_request_index"))
            result = future.result()
            result.scheduled_offset_ms = offsets[index] * 1000.0
            results[index] = result

    duration = (time.perf_counter_ns() - begin_ns) / 1_000_000_000.0
    completed = tuple(result for result in results if result is not None)
    if len(completed) != total_requests:
        raise RuntimeError("open-loop scheduler lost a request result")
    return RunResult(completed, duration, offsets[-1])


def fetch_prometheus_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5.0) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Gateway metrics response exceeded size limit")
    result: dict[str, float] = {}
    for raw_line in payload.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.rpartition(" ")
        if not separator or not name:
            raise RuntimeError(f"invalid Prometheus metric line: {line}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite Prometheus metric: {name}")
        result[name] = value
    return result


def wait_for_resources_released(
    base_url: str,
    timeout: float = 15.0,
) -> dict[str, float]:
    zero_metrics = (
        "kim_llm_gateway_active_requests",
        "kim_llm_gateway_sse_buffered_events",
        "kim_llm_worker_active_requests",
        "kim_llm_worker_reserved_input_tokens",
        "kim_llm_worker_reserved_output_tokens",
        "kim_llm_worker_session_egress_frames",
        "kim_llm_worker_session_egress_bytes",
    )
    deadline = time.monotonic() + timeout
    last_snapshot: dict[str, float] = {}
    while time.monotonic() < deadline:
        last_snapshot = fetch_prometheus_metrics(base_url)
        if all(last_snapshot.get(name) == 0 for name in zero_metrics):
            return last_snapshot
        time.sleep(0.05)
    remaining = {name: last_snapshot.get(name) for name in zero_metrics}
    raise RuntimeError(f"HTTP path resources did not return to zero: {remaining}")


def counter_deltas(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    names = {
        name
        for name in set(before) | set(after)
        if name.endswith("_total") or "_total{" in name
    }
    return {
        name: after.get(name, 0.0) - before.get(name, 0.0)
        for name in sorted(names)
    }


def request_meets_slo(
    request: RequestResult,
    ttft_slo_ms: Optional[float],
    e2e_slo_ms: Optional[float],
) -> bool:
    if request.outcome != "completed":
        return False
    if ttft_slo_ms is not None and (
        request.ttft_ms < 0 or request.ttft_ms > ttft_slo_ms
    ):
        return False
    if e2e_slo_ms is not None and request.e2e_ms > e2e_slo_ms:
        return False
    return True


def result_distribution(
    requests: Sequence[RequestResult],
    attribute: str,
) -> Optional[dict[str, float]]:
    values = [
        float(getattr(request, attribute))
        for request in requests
        if request.outcome == "completed" and getattr(request, attribute) >= 0
    ]
    return distribution(values)


def write_request_csv(path: Path, requests: Sequence[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        fields = tuple(RequestResult.__dataclass_fields__)
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for request in requests:
            writer.writerow(asdict(request))


def build_summary(
    options: BenchmarkOptions,
    input_token_ids: Sequence[int] | Sequence[Sequence[int]],
    run: RunResult,
    worker_startup_ms: float,
    gateway_startup_ms: float,
    metrics_before: Mapping[str, float],
    metrics_after: Mapping[str, float],
) -> dict[str, object]:
    requests = run.requests
    if input_token_ids and isinstance(input_token_ids[0], int):
        tokenized_workloads = (tuple(input_token_ids),)  # type: ignore[arg-type]
    else:
        tokenized_workloads = tuple(
            tuple(variant) for variant in input_token_ids  # type: ignore[union-attr]
        )
    completed = [request for request in requests if request.outcome == "completed"]
    healthy = [request for request in completed if not request.slow_client]
    slow = [request for request in completed if request.slow_client]
    rejected = [request for request in requests if request.outcome == "rejected"]
    terminal_errors = [
        request for request in requests if request.outcome == "terminal_error"
    ]
    client_failures = [
        request
        for request in requests
        if request.outcome in {"client_error", "client_overflow"}
    ]
    good = [
        request
        for request in requests
        if request_meets_slo(request, options.ttft_slo_ms, options.e2e_slo_ms)
    ]

    def counts_by_status(values: Sequence[RequestResult]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for request in values:
            code = request.status_code or "unknown"
            counts[code] = counts.get(code, 0) + 1
        return dict(sorted(counts.items()))

    workload_json = json.dumps(
        list(options.messages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    completion_tokens = sum(request.completion_tokens for request in completed)
    duration = run.duration_seconds
    workload_results = []
    for index, tokens in enumerate(tokenized_workloads):
        workload_requests = [
            request for request in requests if request.workload_index == index
        ]
        workload_completed = [
            request
            for request in workload_requests
            if request.outcome == "completed"
        ]
        workload_results.append(
            {
                "workload_index": index,
                "input_tokens": len(tokens),
                "offered_requests": len(workload_requests),
                "accepted_requests": sum(
                    request.accepted for request in workload_requests
                ),
                "completed_requests": len(workload_completed),
                "rejected_requests": sum(
                    request.outcome == "rejected"
                    for request in workload_requests
                ),
                "good_requests": sum(
                    request_meets_slo(
                        request,
                        options.ttft_slo_ms,
                        options.e2e_slo_ms,
                    )
                    for request in workload_requests
                ),
                "ttft_ms": result_distribution(
                    workload_completed,
                    "ttft_ms",
                ),
                "e2e_ms": result_distribution(
                    workload_completed,
                    "e2e_ms",
                ),
            }
        )
    return {
        "schema_version": 1,
        "benchmark_path": "http_sse",
        "mode": options.mode,
        "engine_dir": str(options.engine_dir),
        "model": options.model,
        "revision": options.revision,
        "workload_fingerprint": hashlib.sha256(workload_json).hexdigest(),
        "concurrency": options.concurrency,
        "max_inflight": options.max_inflight,
        "warmup_requests": options.warmup_requests,
        "measured_requests": options.measured_requests,
        "input_tokens": len(tokenized_workloads[0]),
        "input_token_ids": list(tokenized_workloads[0]),
        "workloads": [
            {
                "workload_index": index,
                "input_tokens": len(tokens),
                "input_token_ids": list(tokens),
                "offered_requests": sum(
                    request.workload_index == index for request in requests
                ),
            }
            for index, tokens in enumerate(tokenized_workloads)
        ],
        "workload_results": workload_results,
        "max_new_tokens": options.max_new_tokens,
        "streaming": True,
        "sampling": {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 0,
        },
        "admission": {
            "strategy": options.admission_strategy,
            "max_active_requests": options.worker_max_active_requests,
            "max_total_input_tokens": options.worker_max_total_input_tokens,
            "max_reserved_output_tokens": (
                options.worker_max_reserved_output_tokens
            ),
            "slo_policy": options.slo_policy,
        },
        "metric_boundaries": {
            "ttft": "client request start to first non-empty SSE content",
            "tpot": (
                "first-to-last SSE content duration divided by "
                "Terminal completion_tokens minus one"
            ),
            "e2e": "client request start through the SSE [DONE] marker",
            "connection_mode": "one HTTP/1.1 connection per request",
            "resource_watermarks": (
                "cumulative since Worker/Gateway start and therefore include warmup"
            ),
        },
        "load": {
            "target_offered_rate_rps": options.offered_rate,
            "arrival_distribution": options.arrival_distribution,
            "arrival_seed": options.arrival_seed,
            "load_generation_seconds": run.load_generation_seconds,
            "slow_request_every": options.slow_request_every,
            "slow_read_delay_ms": options.slow_read_delay_ms,
        },
        "slo": {
            "ttft_ms": options.ttft_slo_ms,
            "e2e_ms": options.e2e_slo_ms,
        },
        "worker_startup_ms": worker_startup_ms,
        "gateway_startup_ms": gateway_startup_ms,
        "duration_seconds": duration,
        "offered_requests": len(requests),
        "accepted_requests": sum(request.accepted for request in requests),
        "completed_requests": len(completed),
        "successful_requests": len(completed),
        "failed_requests": len(requests) - len(completed),
        "rejected_requests": len(rejected),
        "terminal_error_requests": len(terminal_errors),
        "client_failure_requests": len(client_failures),
        "slow_client_requests": sum(request.slow_client for request in requests),
        "good_requests": len(good),
        "slo_attainment": len(good) / len(requests) if requests else 0.0,
        "goodput_rps": len(good) / duration if duration > 0 else 0.0,
        "request_throughput_rps": (
            len(completed) / duration if duration > 0 else 0.0
        ),
        "output_token_throughput_tps": (
            completion_tokens / duration if duration > 0 else 0.0
        ),
        "rejections_by_code": counts_by_status(rejected),
        "terminal_errors_by_code": counts_by_status(terminal_errors),
        "ttft_ms": result_distribution(completed, "ttft_ms"),
        "tpot_ms": result_distribution(completed, "tpot_ms"),
        "e2e_ms": result_distribution(completed, "e2e_ms"),
        "scheduling_lag_ms": distribution(
            [request.scheduling_lag_ms for request in requests]
        ),
        "healthy_client_ttft_ms": result_distribution(healthy, "ttft_ms"),
        "healthy_client_e2e_ms": result_distribution(healthy, "e2e_ms"),
        "slow_client_ttft_ms": result_distribution(slow, "ttft_ms"),
        "slow_client_e2e_ms": result_distribution(slow, "e2e_ms"),
        "unique_output_text_sha256": sorted(
            {
                hashlib.sha256(request.output_text.encode("utf-8")).hexdigest()
                for request in completed
            }
        ),
        "gateway_metrics_before_measurement": dict(metrics_before),
        "gateway_metrics_after_measurement": dict(metrics_after),
        "measurement_counter_deltas": counter_deltas(
            metrics_before,
            metrics_after,
        ),
        "resources_released": all(
            metrics_after.get(name) == 0
            for name in (
                "kim_llm_gateway_active_requests",
                "kim_llm_gateway_sse_buffered_events",
                "kim_llm_worker_active_requests",
                "kim_llm_worker_reserved_input_tokens",
                "kim_llm_worker_reserved_output_tokens",
                "kim_llm_worker_session_egress_frames",
                "kim_llm_worker_session_egress_bytes",
            )
        ),
    }


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def all_protocol_observations_valid(run: RunResult) -> bool:
    return all(request.protocol_valid for request in run.requests)


def all_requests_completed(run: RunResult) -> bool:
    return all(request.outcome == "completed" for request in run.requests)


def first_non_completed_request(run: RunResult) -> str:
    for request in run.requests:
        if request.outcome != "completed":
            suffix = f": {request.error}" if request.error else ""
            return (
                f"request {request.request_id} produced {request.outcome}"
                f" ({request.status_code or 'no status'}){suffix}"
            )
    return ""
