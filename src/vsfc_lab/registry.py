from __future__ import annotations

from collections.abc import Callable

from .interfaces import Metrics, Solver
from .mock_components import PlaceholderMetrics, PlaceholderSolver

SolverFactory = Callable[[], Solver]
MetricsFactory = Callable[[], Metrics]

SOLVER_FACTORIES: dict[str, SolverFactory] = {
    "milp_baseline": PlaceholderSolver,
    "placeholder_solver": PlaceholderSolver,
}

METRICS_FACTORIES: dict[str, MetricsFactory] = {
    "network_metrics": PlaceholderMetrics,
    "placeholder_metrics": PlaceholderMetrics,
}


def _normalise_name(name: str) -> str:
    return name.strip().lower()


def create_solver(name: str) -> Solver:
    key = _normalise_name(name)
    try:
        return SOLVER_FACTORIES[key]()
    except KeyError as exc:
        available = ", ".join(sorted(SOLVER_FACTORIES))
        raise ValueError(f"Unknown solver_name '{name}'. Available solvers: {available}") from exc


def create_metrics(name: str) -> Metrics:
    key = _normalise_name(name)
    try:
        return METRICS_FACTORIES[key]()
    except KeyError as exc:
        available = ", ".join(sorted(METRICS_FACTORIES))
        raise ValueError(f"Unknown metrics_name '{name}'. Available metrics: {available}") from exc
