from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_thesis_figures import (  # noqa: E402
    ALGORITHMS,
    MARKERS,
    RL_ALGORITHMS,
    algorithm_kwargs,
    build_requests,
    build_synthetic_topology,
    extract_rewards,
    loss_proxy,
    unified_solution_cost,
)
from vsfc_lab.mock_components import PlaceholderMetrics, save_scenario_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run non-repeated RL hyperparameter convergence curves for PAS, TD3-DDQN, and Dual."
    )
    parser.add_argument("--output-root", type=Path, default=Path("results") / "rl_hyperparam_curves")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes per setting; must be >=1000.")
    parser.add_argument("--node-count", type=int, default=10)
    parser.add_argument("--request-count", type=int, default=20)
    parser.add_argument("--time-slots", type=int, default=3)
    parser.add_argument("--n-domains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--smooth-window", type=int, default=25, help="Rolling mean window used only for plotting.")
    parser.add_argument("--trend-window", type=int, default=100, help="Tail window used for convergence checks.")
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.32,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def run_episode(
    algorithm: str,
    solver: object,
    topology: Any,
    requests: list[Any],
    seed: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    episode_kwargs = dict(kwargs)
    episode_kwargs["seed"] = seed
    metrics = PlaceholderMetrics()
    started = time.perf_counter()
    try:
        solutions = solver.solve(topology=topology, requests=requests, **episode_kwargs)  # type: ignore[attr-defined]
        runtime_s = time.perf_counter() - started
        metric_values = metrics.evaluate(topology=topology, requests=requests, solutions=solutions)
        rewards = extract_rewards(solutions)
        total_cost = unified_solution_cost(
            topology=topology,
            requests=requests,
            solutions=solutions,
            offloading_unit_cost=float(episode_kwargs.get("offloading_unit_cost", 5.0)),
        )
        return {
            "status": "ok",
            "error": "",
            "algorithm": algorithm,
            "cumulative_reward": float(sum(rewards)) if rewards else math.nan,
            "loss": loss_proxy(rewards, float(episode_kwargs.get("gamma", 0.95))),
            "avg_latency_ms": float(metric_values.get("avg_latency_ms", math.nan)),
            "max_latency_ms": float(metric_values.get("max_latency_ms", math.nan)),
            "total_cost": total_cost,
            "runtime_s": runtime_s,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error": repr(exc),
            "algorithm": algorithm,
            "cumulative_reward": math.nan,
            "loss": math.nan,
            "avg_latency_ms": math.nan,
            "max_latency_ms": math.nan,
            "total_cost": math.nan,
            "runtime_s": time.perf_counter() - started,
        }


def kwargs_for_sweep(algorithm: str, sweep: str, value: float | int) -> dict[str, Any]:
    kwargs = algorithm_kwargs(algorithm, opt_time_limit=1.0)
    if sweep == "learning_rate":
        learning_rate = float(value)
        kwargs.update(
            {
                "learning_rate": learning_rate,
                "critic_lr": learning_rate,
                "actor_lr": max(learning_rate / 3.0, 0.0005),
            }
        )
    elif sweep == "batch_size":
        kwargs["batch_size"] = int(value)
    elif sweep == "buffer_size":
        kwargs["replay_capacity"] = int(value)
    else:
        raise ValueError(f"Unsupported sweep: {sweep}")
    return kwargs


def run_curves(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    topology = build_synthetic_topology(args.node_count, args.n_domains, args.seed)
    requests = build_requests(topology.node_ids, args.request_count, args.time_slots, args.seed)
    save_scenario_config(output_dir / "scenario.json", topology, requests)

    sweep_values: dict[str, list[float | int]] = {
        "learning_rate": args.learning_rates,
        "batch_size": args.batch_sizes,
        "buffer_size": args.buffer_sizes,
    }
    rows: list[dict[str, Any]] = []
    total_settings = len(RL_ALGORITHMS) * sum(len(values) for values in sweep_values.values())
    setting_idx = 0
    for algorithm in RL_ALGORITHMS:
        _, factory = ALGORITHMS[algorithm]
        for sweep, values in sweep_values.items():
            for value in values:
                setting_idx += 1
                solver = factory()
                kwargs = kwargs_for_sweep(algorithm, sweep, value)
                for episode in range(1, args.episodes + 1):
                    result = run_episode(
                        algorithm=algorithm,
                        solver=solver,
                        topology=topology,
                        requests=requests,
                        seed=args.seed + 100000 * setting_idx + episode,
                        kwargs=kwargs,
                    )
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "sweep": sweep,
                            "parameter_value": value,
                            "episode": episode,
                            "learning_rate": kwargs.get("learning_rate"),
                            "critic_lr": kwargs.get("critic_lr"),
                            "actor_lr": kwargs.get("actor_lr"),
                            "batch_size": kwargs.get("batch_size"),
                            "buffer_size": kwargs.get("replay_capacity"),
                            **result,
                        }
                    )
                print(
                    f"Completed setting {setting_idx}/{total_settings}: "
                    f"algorithm={algorithm}, {sweep}={value}, episodes={args.episodes}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def rolling_values(values: pd.Series, window: int) -> pd.Series:
    return values.astype(float).rolling(window=max(1, window), min_periods=1).mean()


def plot_curve(
    df: pd.DataFrame,
    algorithm: str,
    sweep: str,
    metric: str,
    output: Path,
    smooth_window: int,
) -> None:
    subset = df[(df["algorithm"] == algorithm) & (df["sweep"] == sweep) & (df["status"] == "ok")]
    if subset.empty:
        return
    ylabel = "Cumulative reward" if metric == "cumulative_reward" else "Loss"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for value, group in subset.groupby("parameter_value", sort=True):
        group = group.sort_values("episode")
        x_values = group["episode"].astype(int)
        y_values = rolling_values(group[metric], smooth_window)
        label = f"{sweep.replace('_', ' ')}={value:g}" if isinstance(value, float) else f"{sweep.replace('_', ' ')}={value}"
        ax.plot(
            x_values,
            y_values,
            marker=MARKERS.get(algorithm, "o") if len(x_values) <= 50 else None,
            linewidth=1.5,
            color=None,
            label=label,
        )
    ax.set_title(f"{algorithm} {ylabel} under different {sweep.replace('_', ' ')} values")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def tail_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    denominator = sum((idx - x_mean) ** 2 for idx in range(len(values)))
    if denominator == 0.0:
        return 0.0
    numerator = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values))
    return float(numerator / denominator)


def convergence_rows(df: pd.DataFrame, trend_window: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ok_df = df[df["status"] == "ok"].copy()
    for (algorithm, sweep, value), group in ok_df.groupby(["algorithm", "sweep", "parameter_value"], sort=True):
        group = group.sort_values("episode")
        window = max(20, min(int(trend_window), max(20, len(group) // 10)))
        for metric, higher_is_better in [("cumulative_reward", True), ("loss", False)]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().astype(float).tolist()
            if not values:
                continue
            head = values[:window]
            tail = values[-window:]
            start_mean = statistics.fmean(head)
            end_mean = statistics.fmean(tail)
            slope = tail_slope(tail)
            denominator = max(abs(start_mean), 1e-9)
            if higher_is_better:
                improvement = end_mean - start_mean
                relative_change = improvement / denominator
                direction_ok = improvement >= 0.0
                slope_ok = abs(slope) * window <= max(abs(end_mean) * 0.05, 1e-9)
            else:
                improvement = start_mean - end_mean
                relative_change = improvement / denominator
                direction_ok = improvement >= 0.0
                slope_ok = abs(slope) * window <= max(abs(end_mean) * 0.05, 1e-9)
            tail_std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
            tail_cv = tail_std / max(abs(end_mean), 1e-9)
            appears_convergent = bool(direction_ok and slope_ok)
            rows.append(
                {
                    "algorithm": algorithm,
                    "sweep": sweep,
                    "parameter_value": value,
                    "metric": metric,
                    "episodes": len(values),
                    "window": window,
                    "start_mean": start_mean,
                    "end_mean": end_mean,
                    "relative_change": relative_change,
                    "tail_slope_per_episode": slope,
                    "tail_cv": tail_cv,
                    "direction_ok": direction_ok,
                    "tail_slope_ok": slope_ok,
                    "appears_convergent": appears_convergent,
                }
            )
    return rows


def plot_all(df: pd.DataFrame, figure_dir: Path, smooth_window: int) -> None:
    ensure_dir(figure_dir)
    for algorithm in RL_ALGORITHMS:
        safe_algorithm = algorithm.lower().replace("-", "_").replace("/", "_")
        for sweep in ["learning_rate", "batch_size", "buffer_size"]:
            plot_curve(
                df=df,
                algorithm=algorithm,
                sweep=sweep,
                metric="cumulative_reward",
                output=figure_dir / f"{safe_algorithm}_reward_{sweep}.png",
                smooth_window=smooth_window,
            )
            plot_curve(
                df=df,
                algorithm=algorithm,
                sweep=sweep,
                metric="loss",
                output=figure_dir / f"{safe_algorithm}_loss_{sweep}.png",
                smooth_window=smooth_window,
            )


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1000:
        raise ValueError("--episodes must be at least 1000 for this experiment.")
    configure_style()
    timestamp = datetime.now().strftime("rl_hparam_%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    figure_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(figure_dir)

    started_at = time.perf_counter()
    raw_df = run_curves(args, output_dir)
    convergence_df = pd.DataFrame(convergence_rows(raw_df, args.trend_window))
    plot_all(raw_df, figure_dir, args.smooth_window)

    raw_df.to_csv(output_dir / "raw_rl_hyperparam_curves.csv", index=False)
    convergence_df.to_csv(output_dir / "convergence_summary.csv", index=False)
    metadata = {
        "output_dir": str(output_dir),
        "elapsed_seconds": time.perf_counter() - started_at,
        "episodes": args.episodes,
        "repeats": 1,
        "algorithms": RL_ALGORITHMS,
        "learning_rates": args.learning_rates,
        "batch_sizes": args.batch_sizes,
        "buffer_sizes": args.buffer_sizes,
        "node_count": args.node_count,
        "request_count": args.request_count,
        "time_slots": args.time_slots,
        "smooth_window": args.smooth_window,
        "trend_window": args.trend_window,
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not convergence_df.empty:
        print(convergence_df.groupby(["algorithm", "metric"])["appears_convergent"].mean().to_string())


if __name__ == "__main__":
    main()