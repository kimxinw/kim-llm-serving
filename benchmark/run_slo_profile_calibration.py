#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HTTP_BENCHMARK = REPOSITORY_ROOT / "benchmark" / "llm_http_benchmark.py"


@dataclass(frozen=True)
class CalibrationOptions:
    worker: Path
    engine_dir: Path
    tokenizer_path: Path
    workloads_file: Path
    output_dir: Path
    active_buckets: tuple[int, ...]
    repetitions: int
    measured_requests: int
    max_new_tokens: int
    ttft_slo_ms: float
    safety_margin_ms: float


def parse_options(argv: Optional[Sequence[str]] = None) -> CalibrationOptions:
    parser = argparse.ArgumentParser(
        description="Build a manifest-bound TTFT P95 profile from closed-loop runs."
    )
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--workloads-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--active-buckets",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8),
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ttft-slo-ms", type=float, default=50.0)
    parser.add_argument("--safety-margin-ms", type=float, default=3.0)
    args = parser.parse_args(argv)
    required = (
        args.worker,
        args.engine_dir / "config.json",
        args.tokenizer_path,
        args.workloads_file,
    )
    if any(not path.exists() for path in required):
        parser.error("worker, Engine, Tokenizer, or workload path is missing")
    if args.output_dir.exists():
        parser.error("output_dir already exists; calibration evidence is immutable")
    buckets = tuple(args.active_buckets)
    if (
        args.repetitions < 3
        or args.measured_requests <= 0
        or args.max_new_tokens <= 0
        or args.ttft_slo_ms <= 0
        or args.safety_margin_ms < 0
        or not buckets
        or any(bucket <= 0 or bucket > 8 for bucket in buckets)
        or len(set(buckets)) != len(buckets)
    ):
        parser.error("invalid calibration repetitions, limits, SLO, or buckets")
    return CalibrationOptions(
        worker=args.worker.resolve(),
        engine_dir=args.engine_dir.resolve(),
        tokenizer_path=args.tokenizer_path.resolve(),
        workloads_file=args.workloads_file.resolve(),
        output_dir=args.output_dir.resolve(),
        active_buckets=tuple(sorted(buckets)),
        repetitions=args.repetitions,
        measured_requests=args.measured_requests,
        max_new_tokens=args.max_new_tokens,
        ttft_slo_ms=args.ttft_slo_ms,
        safety_margin_ms=args.safety_margin_ms,
    )


class SloProfileCalibration:
    def __init__(self, options: CalibrationOptions) -> None:
        self._options = options
        self._commit = self._git("rev-parse", "HEAD").strip()

    def run(self) -> Mapping[str, object]:
        if self._git("status", "--porcelain", "--untracked-files=no").strip():
            raise RuntimeError("formal calibration requires a clean Git worktree")
        self._options.output_dir.mkdir(parents=True)
        runs: dict[tuple[int, int], Mapping[str, object]] = {}
        for active in self._options.active_buckets:
            for repetition in range(1, self._options.repetitions + 1):
                runs[(active, repetition)] = self._run_one(active, repetition)
        profile, report = self._aggregate(runs)
        self._write_json(self._options.output_dir / "slo-profile.json", profile)
        self._write_json(
            self._options.output_dir / "calibration-report.json",
            report,
        )
        return report

    def _run_one(self, active: int, repetition: int) -> Mapping[str, object]:
        run_dir = self._options.output_dir / f"active-{active}" / f"run-{repetition:02d}"
        run_dir.mkdir(parents=True)
        command = [
            sys.executable,
            str(HTTP_BENCHMARK),
            "--worker", str(self._options.worker),
            "--engine-dir", str(self._options.engine_dir),
            "--tokenizer-path", str(self._options.tokenizer_path),
            "--workloads-file", str(self._options.workloads_file),
            "--mode", "closed-loop",
            "--admission-strategy", "fixed-concurrency",
            "--worker-max-active-requests", str(active),
            "--worker-max-total-input-tokens", str(active * 512),
            "--worker-max-reserved-output-tokens",
            str(active * self._options.max_new_tokens),
            "--concurrency", str(active),
            "--max-inflight", str(active),
            "--max-pending-requests", str(active),
            "--warmup-requests", str(max(4, active)),
            "--measured-requests", str(self._options.measured_requests),
            "--max-new-tokens", str(self._options.max_new_tokens),
            "--ttft-slo-ms", str(self._options.ttft_slo_ms),
            "--summary", str(run_dir / "summary.json"),
            "--requests-csv", str(run_dir / "requests.csv"),
            "--worker-log", str(run_dir / "worker.log"),
            "--gateway-log", str(run_dir / "gateway.log"),
        ]
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (run_dir / "controller.log").write_text(
            "$ " + " ".join(command) + "\n" + result.stdout,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"calibration active={active} repetition={repetition} failed; "
                f"see {run_dir / 'controller.log'}"
            )
        return self._load_object(run_dir / "summary.json")

    def _aggregate(
        self,
        runs: Mapping[tuple[int, int], Mapping[str, object]],
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        first = next(iter(runs.values()))
        manifest = first.get("manifest")
        if not isinstance(manifest, dict):
            raise RuntimeError("calibration summary is missing manifest")
        entries = []
        evidence = []
        for active in self._options.active_buckets:
            by_workload: dict[int, list[float]] = {}
            token_counts: dict[int, int] = {}
            for repetition in range(1, self._options.repetitions + 1):
                summary = runs[(active, repetition)]
                workloads = summary.get("workload_results")
                if not isinstance(workloads, list):
                    raise RuntimeError("calibration summary lacks workload_results")
                for workload in workloads:
                    if not isinstance(workload, dict):
                        raise RuntimeError("invalid workload result")
                    distribution = workload.get("ttft_ms")
                    if not isinstance(distribution, dict):
                        raise RuntimeError("calibration workload lacks TTFT")
                    index = int(workload["workload_index"])
                    token_counts[index] = int(workload["input_tokens"])
                    by_workload.setdefault(index, []).append(
                        float(distribution["p95"])
                    )
            for index in sorted(by_workload):
                samples = by_workload[index]
                predicted = max(samples)
                entries.append(
                    {
                        "max_input_tokens": token_counts[index],
                        "active_requests": active,
                        "predicted_ttft_p95_ms": predicted,
                    }
                )
                evidence.append(
                    {
                        "workload_index": index,
                        "max_input_tokens": token_counts[index],
                        "active_requests": active,
                        "ttft_p95_ms_samples": samples,
                        "ttft_p95_ms_mean": statistics.fmean(samples),
                        "ttft_p95_ms_sample_stdev": statistics.stdev(samples),
                        "profile_prediction_ms": predicted,
                    }
                )
        profile = {
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "engine_fingerprint": manifest["engine_fingerprint"],
            "ttft_slo_ms": self._options.ttft_slo_ms,
            "safety_margin_ms": self._options.safety_margin_ms,
            "entries": entries,
        }
        report = {
            "schema_version": 1,
            "benchmark": "closed_loop_slo_profile_calibration",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._commit,
            "git_dirty": False,
            "repetitions": self._options.repetitions,
            "measured_requests_per_run": self._options.measured_requests,
            "active_buckets": list(self._options.active_buckets),
            "profile": profile,
            "evidence": evidence,
            "all_resources_released": all(
                bool(summary.get("resources_released"))
                for summary in runs.values()
            ),
        }
        return profile, report

    @staticmethod
    def _load_object(path: Path) -> Mapping[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("summary must be a JSON object")
        return value

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, object]) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        SloProfileCalibration(parse_options(argv)).run()
        return 0
    except Exception as exception:
        print(f"SLO profile calibration failed: {exception}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
