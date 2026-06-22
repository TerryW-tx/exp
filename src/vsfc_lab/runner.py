from __future__ import annotations

import random
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .interfaces import Metrics, Solver, Topology
from .io import append_jsonl, dump_json, ensure_dir, setup_logger, write_summary_csv
from .models import PlacementSolution, SFCRequest


class ExperimentRunner:
    """Run repeated experiments and persist reproducible artifacts."""

    def __init__(
        self,
        config: ExperimentConfig,
        topology: Topology,
        requests: list[SFCRequest],
        solver: Solver,
        metrics: Metrics,
    ) -> None:
        self.config = config
        self.topology = topology
        self.requests = requests
        self.solver = solver
        self.metrics = metrics

    def _build_exp_id(self) -> str:
        if self.config.exp_id:
            return self.config.exp_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.exp_name}_{ts}"

    def run(self, seeds: list[int], config_snapshot: dict[str, Any]) -> Path:
        exp_id = self._build_exp_id()
        result_dir = Path(self.config.output_root) / exp_id
        ensure_dir(result_dir)

        log_file = result_dir / "run.log"
        logger = setup_logger(log_file)
        logger.info("Experiment started. exp_id=%s", exp_id)

        dump_json(result_dir / "config.snapshot.json", config_snapshot)

        jsonl_path = result_dir / "runs.jsonl"
        per_run_metrics: list[dict[str, float]] = []

        for seed in seeds:
            run_payload: dict[str, Any] = {
                "seed": seed,
                "status": "ok",
                "metrics": {},
                "error": None,
            }
            try:
                random.seed(seed)
                started_at = time.perf_counter()
                solutions: list[PlacementSolution] = self.solver.solve(
                    topology=self.topology,
                    requests=self.requests,
                    seed=seed,
                    **self.config.run_kwargs,
                )
                solve_runtime = time.perf_counter() - started_at
                metric_values = self.metrics.evaluate(
                    topology=self.topology,
                    requests=self.requests,
                    solutions=solutions,
                )
                metric_values["algorithm_runtime_seconds"] = solve_runtime
                run_payload["metrics"] = metric_values
                per_run_metrics.append(metric_values)
                run_payload["solutions"] = [asdict(solution) for solution in solutions]
            except NotImplementedError as exc:
                run_payload["status"] = "not_implemented"
                run_payload["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                run_payload["status"] = "failed"
                run_payload["error"] = repr(exc)

            append_jsonl(jsonl_path, run_payload)
            logger.info("Run finished. seed=%s status=%s", seed, run_payload["status"])

        summary_rows = self._summarize(per_run_metrics)
        write_summary_csv(
            result_dir / "summary.csv",
            summary_rows,
            default_fieldnames=["metric", "mean", "std", "n_runs"],
        )

        logger.info("Experiment finished. result_dir=%s", result_dir)
        return result_dir

    @staticmethod
    def _summarize(per_run_metrics: list[dict[str, float]]) -> list[dict[str, Any]]:
        if not per_run_metrics:
            return []

        metric_names = sorted(set().union(*[row.keys() for row in per_run_metrics]))
        summary: list[dict[str, Any]] = []
        for name in metric_names:
            values = [row[name] for row in per_run_metrics if name in row]
            if not values:
                continue
            mean_val = statistics.fmean(values)
            std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary.append(
                {
                    "metric": name,
                    "mean": mean_val,
                    "std": std_val,
                    "n_runs": len(values),
                }
            )
        return summary
