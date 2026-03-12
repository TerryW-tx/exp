from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    exp_name: str = "vsfc_exp"
    exp_id: str = ""
    output_root: str = "results"
    solver_name: str = "placeholder_solver"
    metrics_name: str = "placeholder_metrics"
    topology_name: str = "placeholder_topology"
    run_kwargs: dict[str, Any] = field(default_factory=dict)
    plot: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeOptions:
    seeds: list[int]
    config_path: Path


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML object must be a mapping/dict.")
    return data


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        exp_name=str(raw.get("exp_name", "vsfc_exp")),
        exp_id=str(raw.get("exp_id", "")),
        output_root=str(raw.get("output_root", "results")),
        solver_name=str(raw.get("solver_name", "placeholder_solver")),
        metrics_name=str(raw.get("metrics_name", "placeholder_metrics")),
        topology_name=str(raw.get("topology_name", "placeholder_topology")),
        run_kwargs=dict(raw.get("run_kwargs", {})),
        plot=dict(raw.get("plot", {})),
    )
