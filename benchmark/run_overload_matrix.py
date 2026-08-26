#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HTTP_BENCHMARK = REPOSITORY_ROOT / "benchmark" / "llm_http_benchmark.py"
STRATEGY_ORDERS = (
    ("fixed-concurrency", "token-budget", "profile-guided"),
    ("token-budget", "profile-guided", "fixed-concurrency"),
    ("profile-guided", "fixed-concurrency", "token-budget"),
)


@dataclass(frozen=True)
class MatrixOptions:
    worker: Path
    engine_dir: Path
    tokenizer_path: Path
    workloads_file: Path
    slo_profile: Path
    output_dir: Path
    offered_rates: tuple[float, ...]
    repetitions: int
    warmup_requests: int
    measured_requests: int
    max_new_tokens: int
    max_inflight: int
    max_pending_requests: int
    ttft_slo_ms: float
    e2e_slo_ms: float
    fixed_max_active_requests: int
    token_max_active_requests: int
    token_input_budget: int
    token_output_budget: int
    arrival_distribution: str
    arrival_seed: int


@dataclass(frozen=True)
class Strategy:
    name: str
    max_active_requests: int
    max_total_input_tokens: int
    max_reserved_output_tokens: int
    slo_profile: Optional[Path] = None


def parse_options(argv: Optional[Sequence[str]] = None) -> MatrixOptions:
    parser = argparse.ArgumentParser(
        description="Run the formal three-strategy HTTP open-loop overload matrix."
    )
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--workloads-file", required=True, type=Path)
    parser.add_argument("--slo-profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--offered-rates", required=True, nargs="+", type=float)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--measured-requests", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--max-pending-requests", type=int, default=64)
    parser.add_argument("--ttft-slo-ms", type=float, default=50.0)
    parser.add_argument("--e2e-slo-ms", type=float, default=400.0)
    parser.add_argument("--fixed-max-active-requests", type=int, default=4)
    parser.add_argument("--token-max-active-requests", type=int, default=8)
    parser.add_argument("--token-input-budget", type=int, default=256)
    parser.add_argument("--token-output-budget", type=int, default=256)
    parser.add_argument(
        "--arrival-distribution",
        choices=("constant", "poisson"),
        default="poisson",
    )
    parser.add_argument("--arrival-seed", type=int, default=20260826)
    args = parser.parse_args(argv)

    files = (
        args.worker,
        args.engine_dir / "config.json",
        args.tokenizer_path,
        args.workloads_file,
        args.slo_profile,
    )
    if any(not path.exists() for path in files):
        parser.error("worker, Engine, Tokenizer, workload, or profile path is missing")
    if args.output_dir.exists():
        parser.error("output_dir already exists; formal evidence is never overwritten")
    if args.repetitions < 3:
        parser.error("formal matrix requires at least three repetitions")
    positive = (
        *args.offered_rates,
        args.measured_requests,
        args.max_new_tokens,
        args.max_inflight,
        args.max_pending_requests,
        args.ttft_slo_ms,
        args.e2e_slo_ms,
        args.fixed_max_active_requests,
        args.token_max_active_requests,
        args.token_input_budget,
        args.token_output_budget,
    )
    if any(value <= 0 for value in positive) or args.warmup_requests < 0:
        parser.error("rates, limits, SLOs, and request counts must be positive")
    if args.fixed_max_active_requests > 8 or args.token_max_active_requests > 8:
        parser.error("current Engine baseline supports at most eight active requests")
    return MatrixOptions(
        worker=args.worker.resolve(),
        engine_dir=args.engine_dir.resolve(),
        tokenizer_path=args.tokenizer_path.resolve(),
        workloads_file=args.workloads_file.resolve(),
        slo_profile=args.slo_profile.resolve(),
        output_dir=args.output_dir.resolve(),
        offered_rates=tuple(args.offered_rates),
        repetitions=args.repetitions,
        warmup_requests=args.warmup_requests,
        measured_requests=args.measured_requests,
        max_new_tokens=args.max_new_tokens,
        max_inflight=args.max_inflight,
        max_pending_requests=args.max_pending_requests,
        ttft_slo_ms=args.ttft_slo_ms,
        e2e_slo_ms=args.e2e_slo_ms,
        fixed_max_active_requests=args.fixed_max_active_requests,
        token_max_active_requests=args.token_max_active_requests,
        token_input_budget=args.token_input_budget,
        token_output_budget=args.token_output_budget,
        arrival_distribution=args.arrival_distribution,
        arrival_seed=args.arrival_seed,
    )


class OverloadMatrixRunner:
    def __init__(self, options: MatrixOptions) -> None:
        self._options = options
        self._commit = self._repository_commit()
        self._profile = self._load_object(options.slo_profile, "SLO profile")
        self._workloads = self._load_array(options.workloads_file, "workloads")
        self._strategies = self._make_strategies()

    def run(self) -> Mapping[str, object]:
        self._require_clean_repository()
        self._options.output_dir.mkdir(parents=True)
        summaries: dict[tuple[float, int, str], Mapping[str, object]] = {}
        execution_orders: list[Mapping[str, object]] = []
        for rate_index, rate in enumerate(self._options.offered_rates):
            for repetition in range(1, self._options.repetitions + 1):
                order = STRATEGY_ORDERS[(repetition - 1 + rate_index) % 3]
                execution_orders.append(
                    {
                        "offered_rate_rps": rate,
                        "repetition": repetition,
                        "order": list(order),
                    }
                )
                for strategy_name in order:
                    strategy = self._strategies[strategy_name]
                    summary = self._run_one(rate, repetition, strategy)
                    summaries[(rate, repetition, strategy_name)] = summary

        report = self._build_report(summaries, execution_orders)
        self._write_json(self._options.output_dir / "overload-matrix.json", report)
        return report

    def _make_strategies(self) -> Mapping[str, Strategy]:
        fixed = self._options.fixed_max_active_requests
        token = self._options.token_max_active_requests
        return {
            "fixed-concurrency": Strategy(
                "fixed-concurrency",
                fixed,
                fixed * 512,
                fixed * self._options.max_new_tokens,
            ),
            "token-budget": Strategy(
                "token-budget",
                token,
                self._options.token_input_budget,
                self._options.token_output_budget,
            ),
            "profile-guided": Strategy(
                "profile-guided",
                token,
                self._options.token_input_budget,
                self._options.token_output_budget,
                self._options.slo_profile,
            ),
        }

    def _run_one(
        self,
        rate: float,
        repetition: int,
        strategy: Strategy,
    ) -> Mapping[str, object]:
        run_dir = (
            self._options.output_dir
            / f"rate-{rate:g}"
            / f"run-{repetition:02d}-{strategy.name}"
        )
        run_dir.mkdir(parents=True)
        command = [
            sys.executable,
            str(HTTP_BENCHMARK),
            "--worker", str(self._options.worker),
            "--engine-dir", str(self._options.engine_dir),
            "--tokenizer-path", str(self._options.tokenizer_path),
            "--workloads-file", str(self._options.workloads_file),
            "--mode", "open-loop",
            "--admission-strategy", strategy.name,
            "--worker-max-active-requests", str(strategy.max_active_requests),
            "--worker-max-total-input-tokens", str(strategy.max_total_input_tokens),
            "--worker-max-reserved-output-tokens",
            str(strategy.max_reserved_output_tokens),
            # Warmup must validate the path without exercising the strategy;
            # the open-loop measurement below supplies the actual overload.
            "--concurrency", "1",
            "--max-inflight", str(self._options.max_inflight),
            "--max-pending-requests", str(self._options.max_pending_requests),
            "--warmup-requests", str(self._options.warmup_requests),
            "--measured-requests", str(self._options.measured_requests),
            "--max-new-tokens", str(self._options.max_new_tokens),
            "--offered-rate", str(rate),
            "--arrival-distribution", self._options.arrival_distribution,
            "--arrival-seed", str(self._options.arrival_seed + repetition - 1),
            "--ttft-slo-ms", str(self._options.ttft_slo_ms),
            "--e2e-slo-ms", str(self._options.e2e_slo_ms),
            "--summary", str(run_dir / "summary.json"),
            "--requests-csv", str(run_dir / "requests.csv"),
            "--worker-log", str(run_dir / "worker.log"),
            "--gateway-log", str(run_dir / "gateway.log"),
        ]
        if strategy.slo_profile is not None:
            command.extend(("--slo-profile", str(strategy.slo_profile)))
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
                f"{strategy.name} rate {rate:g} repetition {repetition} failed; "
                f"see {run_dir / 'controller.log'}"
            )
        return self._load_object(run_dir / "summary.json", "run summary")

    def _build_report(
        self,
        summaries: Mapping[tuple[float, int, str], Mapping[str, object]],
        execution_orders: Sequence[Mapping[str, object]],
    ) -> Mapping[str, object]:
        rate_reports = []
        for rate in self._options.offered_rates:
            strategy_reports: dict[str, Mapping[str, object]] = {}
            for strategy_name in self._strategies:
                runs = [
                    summaries[(rate, repetition, strategy_name)]
                    for repetition in range(1, self._options.repetitions + 1)
                ]
                strategy_reports[strategy_name] = self._aggregate_strategy(
                    runs,
                    strategy_name,
                )
            rate_reports.append(
                {
                    "offered_rate_rps": rate,
                    "strategies": strategy_reports,
                    "paired": {
                        "token_budget_minus_fixed_goodput_rps":
                            self._paired_mean(
                                summaries,
                                rate,
                                "token-budget",
                                "fixed-concurrency",
                                "goodput_rps",
                            ),
                        "profile_minus_token_budget_goodput_rps":
                            self._paired_mean(
                                summaries,
                                rate,
                                "profile-guided",
                                "token-budget",
                                "goodput_rps",
                            ),
                    },
                }
            )
        return {
            "schema_version": 1,
            "benchmark": "three_strategy_open_loop_overload_matrix",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": self._commit,
            "git_dirty": False,
            "minimum_repetitions_met": self._options.repetitions >= 3,
            "balanced_execution_order_cycle": self._options.repetitions % 3 == 0,
            "options": {
                **asdict(self._options),
                "worker": str(self._options.worker),
                "engine_dir": str(self._options.engine_dir),
                "tokenizer_path": str(self._options.tokenizer_path),
                "workloads_file": str(self._options.workloads_file),
                "slo_profile": str(self._options.slo_profile),
                "output_dir": str(self._options.output_dir),
            },
            "workloads_sha256": hashlib.sha256(
                self._options.workloads_file.read_bytes()
            ).hexdigest(),
            "profile_sha256": hashlib.sha256(
                self._options.slo_profile.read_bytes()
            ).hexdigest(),
            "strategies": {
                name: {
                    **asdict(strategy),
                    "slo_profile": (
                        str(strategy.slo_profile)
                        if strategy.slo_profile is not None
                        else None
                    ),
                }
                for name, strategy in self._strategies.items()
            },
            "execution_orders": list(execution_orders),
            "rates": rate_reports,
        }

    def _aggregate_strategy(
        self,
        runs: Sequence[Mapping[str, object]],
        strategy_name: str,
    ) -> Mapping[str, object]:
        metrics = (
            "goodput_rps",
            "slo_attainment",
            "accepted_requests",
            "completed_requests",
            "rejected_requests",
            "good_requests",
        )
        result: dict[str, object] = {
            metric: self._mean_stdev(float(run[metric]) for run in runs)
            for metric in metrics
        }
        result["ttft_p95_ms"] = self._mean_stdev(
            float(self._distribution(run, "ttft_ms")["p95"])
            for run in runs
        )
        result["e2e_p95_ms"] = self._mean_stdev(
            float(self._distribution(run, "e2e_ms")["p95"])
            for run in runs
        )
        result["accepted_slo_misses"] = self._mean_stdev(
            float(run["accepted_requests"]) - float(run["good_requests"])
            for run in runs
        )
        result["all_resources_released"] = all(
            bool(run["resources_released"]) for run in runs
        )
        rejection_counts: dict[str, int] = {}
        for run in runs:
            for code, count in self._mapping(run, "rejections_by_code").items():
                rejection_counts[code] = rejection_counts.get(code, 0) + int(count)
        result["rejections_by_code"] = dict(sorted(rejection_counts.items()))
        if strategy_name == "profile-guided":
            result["profile_ttft_p95_error_ms"] = self._mean_stdev(
                self._profile_prediction_errors(runs)
            )
        return result

    def _profile_prediction_errors(
        self,
        runs: Sequence[Mapping[str, object]],
    ) -> Sequence[float]:
        entries = self._profile.get("entries")
        safety_margin = self._profile.get("safety_margin_ms")
        ttft_slo = self._profile.get("ttft_slo_ms")
        if (
            not isinstance(entries, list)
            or not isinstance(safety_margin, (int, float))
            or not isinstance(ttft_slo, (int, float))
        ):
            raise RuntimeError("SLO profile is missing prediction metadata")
        errors: list[float] = []
        for run in runs:
            workload_results = run.get("workload_results")
            if not isinstance(workload_results, list):
                raise RuntimeError("run summary is missing workload_results")
            for workload in workload_results:
                if not isinstance(workload, dict):
                    raise RuntimeError("invalid workload result")
                distribution = workload.get("ttft_ms")
                if not isinstance(distribution, dict):
                    continue
                input_tokens = int(workload["input_tokens"])
                predictions = [
                    float(entry["predicted_ttft_p95_ms"])
                    for entry in entries
                    if isinstance(entry, dict)
                    and input_tokens <= int(entry["max_input_tokens"])
                    and float(entry["predicted_ttft_p95_ms"])
                    + float(safety_margin)
                    <= float(ttft_slo)
                ]
                if not predictions:
                    continue
                errors.append(float(distribution["p95"]) - max(predictions))
        if not errors:
            raise RuntimeError("profile run produced no comparable TTFT predictions")
        return errors

    def _paired_mean(
        self,
        summaries: Mapping[tuple[float, int, str], Mapping[str, object]],
        rate: float,
        left: str,
        right: str,
        metric: str,
    ) -> Mapping[str, float]:
        differences = [
            float(summaries[(rate, repetition, left)][metric])
            - float(summaries[(rate, repetition, right)][metric])
            for repetition in range(1, self._options.repetitions + 1)
        ]
        return self._mean_stdev(differences)

    @staticmethod
    def _mean_stdev(values: Iterable[float]) -> Mapping[str, float]:
        materialized = list(values)
        return {
            "mean": statistics.fmean(materialized),
            "sample_stdev": (
                statistics.stdev(materialized)
                if len(materialized) > 1
                else 0.0
            ),
        }

    @staticmethod
    def _distribution(
        value: Mapping[str, object],
        key: str,
    ) -> Mapping[str, object]:
        result = value.get(key)
        if not isinstance(result, dict):
            raise RuntimeError(f"summary {key} distribution is missing")
        return result

    @staticmethod
    def _mapping(
        value: Mapping[str, object],
        key: str,
    ) -> Mapping[str, object]:
        result = value.get(key)
        if not isinstance(result, dict):
            raise RuntimeError(f"summary {key} mapping is missing")
        return result

    @staticmethod
    def _load_object(path: Path, description: str) -> Mapping[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"{description} must be a JSON object")
        return value

    @staticmethod
    def _load_array(path: Path, description: str) -> Sequence[object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not value:
            raise RuntimeError(f"{description} must be a non-empty JSON array")
        return value

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, object]) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _repository_commit() -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    @staticmethod
    def _require_clean_repository() -> None:
        status = subprocess.run(
            [
                "git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
        if status.strip():
            raise RuntimeError("formal overload matrix requires a clean Git worktree")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        options = parse_options(argv)
        OverloadMatrixRunner(options).run()
        return 0
    except Exception as exception:
        print(f"overload matrix failed: {exception}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
