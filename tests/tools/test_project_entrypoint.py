#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPOSITORY_ROOT / "kim-llm"


def run_entrypoint(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


class ProjectEntrypointTest(unittest.TestCase):
    def test_help_lists_all_supported_workflows(self) -> None:
        result = run_entrypoint("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("build", "test", "serve", "benchmark"):
            self.assertIn(command, result.stdout)

    def test_cpu_build_dry_run_disables_gpu_dependencies(self) -> None:
        result = run_entrypoint(
            "build",
            "--build-dir",
            "build-entrypoint-test",
            "--parallel",
            "2",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DKIM_LLM_ENABLE_TRTLLM=OFF", result.stdout)
        self.assertIn("cmake --build", result.stdout)
        self.assertIn("--parallel 2", result.stdout)

    def test_gpu_build_dry_run_requires_fresh_config_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-entrypoint-build-") as directory:
            result = run_entrypoint(
                "build",
                "--gpu",
                "--build-dir",
                directory,
                "--dry-run",
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("fresh GPU configure requires", result.stderr)

    def test_cpu_test_dry_run_can_reuse_a_build(self) -> None:
        result = run_entrypoint(
            "test",
            "--build-dir",
            "build-entrypoint-test",
            "--no-build",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ctest --test-dir", result.stdout)
        self.assertIn("-LE gpu", result.stdout)

    def test_serve_dry_run_resolves_the_worker_config_and_socket(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-entrypoint-serve-") as directory:
            root = Path(directory)
            worker_config = root / "worker.json"
            gateway_config = root / "gateway.json"
            worker_config.write_text(
                json.dumps(
                    {
                        "engine_dir": "/engine",
                        "socket_path": "worker.sock",
                        "manifest": {},
                        "limits": {},
                    }
                ),
                encoding="utf-8",
            )
            gateway_config.write_text(
                json.dumps(
                    {
                        "worker_config_path": "worker.json",
                        "tokenizer_path": "/tokenizer",
                        "host": "127.0.0.1",
                        "port": 8123,
                    }
                ),
                encoding="utf-8",
            )
            result = run_entrypoint(
                "serve",
                "--gateway-config",
                str(gateway_config),
                "--worker",
                "build-llm/llm_worker",
                "--dry-run",
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(worker_config), result.stdout)
        self.assertIn("kim_llm_gateway serve", result.stdout)
        self.assertIn(str(gateway_config), result.stdout)

    def test_serve_supervises_readiness_and_graceful_shutdown(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kim-llm-entrypoint-supervisor-") as directory:
            root = Path(directory)
            socket_path = root / "worker.sock"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                port = int(listener.getsockname()[1])

            worker_config = root / "worker.json"
            gateway_config = root / "gateway.json"
            fake_worker = root / "fake-worker"
            fake_gateway = root / "fake-gateway"
            worker_config.write_text(
                json.dumps(
                    {
                        "engine_dir": "/engine",
                        "socket_path": str(socket_path),
                        "manifest": {},
                        "limits": {},
                    }
                ),
                encoding="utf-8",
            )
            gateway_config.write_text(
                json.dumps(
                    {
                        "worker_config_path": str(worker_config),
                        "tokenizer_path": "/tokenizer",
                        "host": "127.0.0.1",
                        "port": port,
                    }
                ),
                encoding="utf-8",
            )
            fake_worker.write_text(
                """#!/usr/bin/env python3
import json
import os
import signal
import socket
import sys

running = True
def stop(*_):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
config = json.load(open(sys.argv[1], encoding="utf-8"))
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(config["socket_path"])
listener.listen()
listener.settimeout(0.1)
while running:
    try:
        connection, _ = listener.accept()
        connection.close()
    except TimeoutError:
        pass
listener.close()
os.unlink(config["socket_path"])
""",
                encoding="utf-8",
            )
            fake_gateway.write_text(
                """#!/usr/bin/env python3
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

running = True
def stop(*_):
    global running
    running = False

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ready"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_):
        pass

signal.signal(signal.SIGTERM, stop)
config = json.load(open(sys.argv[-1], encoding="utf-8"))
server = HTTPServer((config.get("host", "127.0.0.1"), config.get("port", 8000)), Handler)
server.timeout = 0.1
while running:
    server.handle_request()
server.server_close()
""",
                encoding="utf-8",
            )
            fake_worker.chmod(0o755)
            fake_gateway.chmod(0o755)

            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(ENTRYPOINT),
                    "serve",
                    "--gateway-config",
                    str(gateway_config),
                    "--worker",
                    str(fake_worker),
                    "--python",
                    str(fake_gateway),
                    "--startup-timeout",
                    "5",
                    "--shutdown-timeout",
                    "2",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5.0
                output_lines: list[str] = []
                ready = False
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    assert process.stdout is not None
                    readable, _, _ = select.select([process.stdout], [], [], 0.1)
                    if readable:
                        line = process.stdout.readline()
                        output_lines.append(line)
                        if "kim-llm service ready" in line:
                            ready = True
                            break
                self.assertTrue(ready, "Gateway did not become ready")
                self.assertIsNone(process.poll())
                process.terminate()
                remaining_stdout, stderr = process.communicate(timeout=5.0)
                stdout = "".join(output_lines) + remaining_stdout
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2.0)

        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("kim-llm service ready", stdout)
        self.assertFalse(socket_path.exists())

    def test_benchmark_dry_run_uses_safe_smoke_defaults(self) -> None:
        result = run_entrypoint(
            "benchmark",
            "--engine-dir",
            "/engine",
            "--tokenizer-path",
            "/tokenizer",
            "--output-dir",
            "/tmp/kim-llm-entrypoint-smoke",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_direct_ipc_http_benchmark.py", result.stdout)
        self.assertIn("--repetitions 1", result.stdout)
        self.assertIn("--measured-requests 5", result.stdout)
        self.assertIn("--max-new-tokens 8", result.stdout)
        self.assertIn("--allow-dirty", result.stdout)

    def test_benchmark_formal_mode_does_not_allow_dirty_by_default(self) -> None:
        result = run_entrypoint(
            "benchmark",
            "--engine-dir",
            "/engine",
            "--tokenizer-path",
            "/tokenizer",
            "--output-dir",
            "/tmp/kim-llm-entrypoint-formal",
            "--formal",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--repetitions 3", result.stdout)
        self.assertIn("--measured-requests 200", result.stdout)
        self.assertNotIn("--allow-dirty", result.stdout)


if __name__ == "__main__":
    unittest.main()
