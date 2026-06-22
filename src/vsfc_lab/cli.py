from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .config import load_yaml_config, parse_config
from .mock_components import (
    PlaceholderMetrics,
    PlaceholderSolver,
    build_mock_topology,
    build_mock_requests,
    load_scenario_config,
    save_scenario_config,
)
from .runner import ExperimentRunner

SOLVER_REGISTRY = {
    "placeholder_solver": PlaceholderSolver,
    "milp_baseline": PlaceholderSolver,
}

METRICS_REGISTRY = {
    "placeholder_metrics": PlaceholderMetrics,
    "network_metrics": PlaceholderMetrics,
}


def _parse_kv_overrides(pairs: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected key=value.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid override '{pair}'. Empty key is not allowed.")
        overrides[key] = yaml.safe_load(value)
    return overrides


def _apply_overrides(raw_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(raw_cfg)
    for key, value in overrides.items():
        if "." not in key:
            updated[key] = value
            continue
        cursor = updated
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return updated


def _build_component(name: str, registry: dict[str, type]) -> Any:
    try:
        return registry[name]()
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown component '{name}'. Available: {available}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vSFC experiment runner")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds for repeated runs.")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Override config values via key=value, support dotted key e.g. run_kwargs.alpha=0.1",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=None,
        help="Load topology + requests from a saved scenario JSON.",
    )
    parser.add_argument(
        "--save-scenario-config",
        type=Path,
        default=None,
        help="Save generated topology + requests to scenario JSON for reproducible reruns.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw_cfg = load_yaml_config(args.config)
    overrides = _parse_kv_overrides(args.overrides)
    merged_cfg = _apply_overrides(raw_cfg, overrides)
    config = parse_config(merged_cfg)

    if args.scenario_config is not None:
        topology, requests = load_scenario_config(args.scenario_config)
    else:
        topology = build_mock_topology(seed=args.seeds[0] if args.seeds else 0)
        requests = build_mock_requests(node_ids=topology.node_ids)

    if args.save_scenario_config is not None:
        save_scenario_config(args.save_scenario_config, topology, requests)
    solver = _build_component(config.solver_name, SOLVER_REGISTRY)
    metrics = _build_component(config.metrics_name, METRICS_REGISTRY)

    runner = ExperimentRunner(
        config=config,
        topology=topology,
        requests=requests,
        solver=solver,
        metrics=metrics,
    )

    config_snapshot = {
        "config_path": str(args.config),
        "seeds": args.seeds,
        "scenario_config": None if args.scenario_config is None else str(args.scenario_config),
        "save_scenario_config": (
            None if args.save_scenario_config is None else str(args.save_scenario_config)
        ),
        "merged_config": merged_cfg,
        "parsed_config": asdict(config),
        "available_solvers": sorted(SOLVER_REGISTRY),
        "available_metrics": sorted(METRICS_REGISTRY),
    }
    result_dir = runner.run(seeds=args.seeds, config_snapshot=config_snapshot)
    print(f"Experiment completed. Results in: {result_dir}")


if __name__ == "__main__":
    main()
