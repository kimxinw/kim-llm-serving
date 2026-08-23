from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from kim_llm_client import (
    GenerationClient,
    GenerationClientConfig,
    GenerationRequest,
    ModelManifest,
    WorkerLimits,
)


REQUEST_ID = 100
INPUT_TOKEN_IDS = (1, 2, 3, 4)
MAX_NEW_TOKENS = 5
# Keep this in sync with llm_backend_lifecycle_test.cpp. It is the saved
# deterministic TinyLlama Direct Backend baseline for the request above.
EXPECTED_TOKEN_IDS = (3, 29966, 29989, 5205, 29989)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_manifest(engine_dir: Path) -> ModelManifest:
    engine_config = engine_dir / "config.json"
    fingerprint = hashlib.sha256(engine_config.read_bytes()).hexdigest()
    return ModelManifest(
        model_id="TinyLlama-1.1B-Chat-v1.0",
        revision="trtllm-0.16.0-engine",
        tokenizer_fingerprint="token-id-integration-test",
        chat_template_fingerprint="token-id-integration-test",
        engine_fingerprint=fingerprint,
        eos_token_id=2,
        pad_token_id=2,
        max_input_tokens=512,
        max_output_tokens=32,
        max_sequence_tokens=544,
        precision="fp16",
        max_batch_size=8,
    )


def make_limits() -> WorkerLimits:
    return WorkerLimits(
        max_active_requests=8,
        max_total_input_tokens=4096,
        max_reserved_output_tokens=256,
        max_frame_payload_bytes=1024 * 1024,
        max_session_egress_frames=256,
        max_session_egress_bytes=4 * 1024 * 1024,
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


def wait_for_socket(
    process: subprocess.Popen[str],
    socket_path: Path,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"llm_worker exited before readiness with code {return_code}"
            )
        if socket_path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(
        f"timed out waiting for llm_worker socket {socket_path}"
    )


def wait_for_resources_released(
    client: GenerationClient,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_snapshot = None
    while time.monotonic() < deadline:
        last_snapshot = client.health(timeout=2.0)
        if (
            last_snapshot.active_requests == 0
            and last_snapshot.reserved_input_tokens == 0
            and last_snapshot.reserved_output_tokens == 0
        ):
            return
        time.sleep(0.05)
    raise AssertionError(
        "Worker resources did not return to zero after Terminal: "
        f"{last_snapshot}"
    )


def stop_worker(
    process: subprocess.Popen[str],
    timeout: float,
) -> tuple[int, Optional[str]]:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
            return process.returncode, (
                "llm_worker did not stop after SIGTERM and was killed"
            )
    return_code = process.returncode
    if return_code is None:
        raise RuntimeError("llm_worker has no exit code after shutdown")
    return return_code, None


def run_test(
    worker: Path,
    engine_dir: Path,
    startup_timeout: float,
    inference_timeout: float,
) -> None:
    require(startup_timeout > 0, "startup timeout must be positive")
    require(inference_timeout > 0, "inference timeout must be positive")
    require(worker.is_file(), f"llm_worker does not exist: {worker}")
    require(os.access(worker, os.X_OK), f"llm_worker is not executable: {worker}")
    require(engine_dir.is_dir(), f"Engine directory does not exist: {engine_dir}")
    require(
        (engine_dir / "config.json").is_file(),
        f"Engine config.json does not exist: {engine_dir}",
    )

    manifest = make_manifest(engine_dir)
    limits = make_limits()

    with tempfile.TemporaryDirectory(prefix="kim-llm-e2e-") as directory:
        test_dir = Path(directory)
        socket_path = test_dir / "worker.sock"
        config_path = test_dir / "worker-config.json"
        log_path = test_dir / "worker.log"
        write_worker_config(
            config_path,
            engine_dir,
            socket_path,
            manifest,
            limits,
        )

        client: Optional[GenerationClient] = None
        test_failure: Optional[Exception] = None
        shutdown_failure: Optional[str] = None
        with log_path.open("w+", encoding="utf-8") as worker_log:
            process = subprocess.Popen(
                [str(worker), str(config_path)],
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for_socket(process, socket_path, startup_timeout)
                client = GenerationClient(
                    GenerationClientConfig(
                        socket_path=str(socket_path),
                        expected_manifest=manifest,
                        connect_timeout=2.0,
                        handshake_timeout=10.0,
                        max_pending_requests=1,
                        max_delta_events_per_request=MAX_NEW_TOKENS + 1,
                    )
                )
                client.connect()

                initial_stats = client.health(timeout=5.0)
                require(initial_stats.ready, "Worker must be ready after handshake")
                require(initial_stats.status.ok, "ready Worker status must be ok")

                handle = client.submit(
                    GenerationRequest(
                        request_id=REQUEST_ID,
                        input_token_ids=INPUT_TOKEN_IDS,
                        max_new_tokens=MAX_NEW_TOKENS,
                        timeout_ms=int(inference_timeout * 1000),
                        trace_id="real-worker-engine-e2e",
                        streaming=True,
                    ),
                    timeout=10.0,
                )
                tokens, terminal = handle.collect(timeout=inference_timeout)

                require(
                    tokens == EXPECTED_TOKEN_IDS,
                    "IPC output tokens do not match the saved Direct Backend "
                    f"baseline: expected {EXPECTED_TOKEN_IDS}, got {tokens}",
                )
                require(terminal.request_id == REQUEST_ID, "Terminal request_id changed")
                require(
                    terminal.status.ok,
                    "real Engine request failed: "
                    f"{terminal.status.code}: {terminal.status.message}",
                )
                require(
                    terminal.finish_reason == "length",
                    f"expected length finish, got {terminal.finish_reason}",
                )
                require(
                    terminal.usage.prompt_tokens == len(INPUT_TOKEN_IDS),
                    "Terminal prompt token usage is incorrect",
                )
                require(
                    terminal.usage.completion_tokens == len(EXPECTED_TOKEN_IDS),
                    "Terminal completion token usage is incorrect",
                )
                try:
                    handle.next_event(timeout=0.0)
                except StopIteration:
                    pass
                else:
                    raise AssertionError("an event was delivered after Terminal")

                wait_for_resources_released(client, timeout=10.0)
                require(client.connected, "GenerationClient disconnected after Terminal")
            except Exception as exception:
                test_failure = exception
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception as exception:
                        if test_failure is None:
                            test_failure = RuntimeError(
                                f"GenerationClient close failed: {exception}"
                            )
                return_code, shutdown_failure = stop_worker(process, timeout=30.0)
                socket_remained = socket_path.exists()
                worker_log.flush()
                worker_log.seek(0)
                worker_output = worker_log.read()

        if test_failure is not None:
            raise AssertionError(
                f"{test_failure}\nllm_worker output:\n{worker_output}"
            ) from test_failure
        require(shutdown_failure is None, shutdown_failure or "Worker shutdown failed")
        require(
            return_code == 0,
            f"llm_worker exited with code {return_code}\n{worker_output}",
        )
        require(not socket_remained, "llm_worker left its UDS path after shutdown")

    print(
        "[PASS] real llm_worker -> TensorRT-LLM Engine -> "
        f"Python GenerationClient: tokens={EXPECTED_TOKEN_IDS}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real cross-process Worker/Engine IPC integration test"
    )
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--inference-timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_test(
            args.worker.resolve(),
            args.engine_dir.resolve(),
            args.startup_timeout,
            args.inference_timeout,
        )
    except Exception as exception:
        print(f"[FAIL] {exception}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
