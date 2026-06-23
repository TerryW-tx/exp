from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_thesis_figures import (  # noqa: E402
    ALGORITHMS,
    COLORS,
    MARKERS,
    algorithm_kwargs,
    build_requests,
    build_synthetic_topology,
    ensure_dir,
    run_solver,
    save_scenario_config,
)


RL_ALGORITHMS = ["PAS", "TD3-DDQN", "Dual"]
PARAMETER_VALUES = {
    "learning_rate": [0.001, 0.003, 0.01, 0.03],
    "batch_size": [4, 8, 16, 32],
    "buffer_size": [32, 64, 128, 256],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-repeat RL hyperparameter convergence curves for PAS, TD3-DDQN, and Dual."
    )
    parser.add_argument("--output-root", type=Path, default=Path("results") / "rl_hyperparam_convergence")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes per parameter value; must be >=1000.")
    parser.add_argument("--node-count", type=int, default=10)
    parser.add_argument("--request-count", type=int, default=10)
    parser.add_argument("--time-slots", type=int, default=3)
    parser.add_argument("--n-domains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=PARAMETER_VALUES["learning_rate"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=PARAMETER_VALUES["batch_size"])
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=PARAMETER_VALUES["buffer_size"])
    parser.add_argument("--smooth-window", type=int, default=25, help="Rolling window used only for plotting trend lines.")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser


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


def parameter_sets(args: argparse.Namespace) -> dict[str, list[float | int]]:
    return {
        "learning_rate": [float(value) for value in args.learning_rates],
        "batch_size": [int(value) for value in args.batch_sizes],
        "buffer_size": [int(value) for value in args.buffer_sizes],
    }


def apply_parameter(algorithm: str, parameter: str, value: float | int) -> dict[str, Any]:
    kwargs = algorithm_kwargs(algorithm, opt_time_limit=1.0)
    if parameter == "learning_rate":
        learning_rate = float(value)
        kwargs.update(
            {
                "learning_rate": learning_rate,
                "critic_lr": learning_rate,
                "actor_lr": max(learning_rate / 3.0, 0.0005),
            }
        )
    elif parameter == "batch_size":
        kwargs["batch_size"] = int(value)
    elif parameter == "buffer_size":
        kwargs["replay_capacity"] = int(value)
    else:
        raise ValueError(f"Unsupported parameter: {parameter}")
    return kwargs


def stable_value(value: float | int) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def rolling_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values.astype(float)
    return values.astype(float).rolling(window=window, min_periods=max(1, window // 5)).mean()


def run_curves(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    topology = build_synthetic_topology(args.node_count, args.n_domains, args.seed)
    requests = build_requests(topology.node_ids, args.request_count, args.time_slots, args.seed)
    save_scenario_config(output_dir / "scenario.json", topology, requests)

    rows: list[dict[str, Any]] = []
    values_by_parameter = parameter_sets(args)
    total_curves = len(RL_ALGORITHMS) * sum(len(values) for values in values_by_parameter.values())
    curve_idx = 0
    for algorithm in RL_ALGORITHMS:
        for parameter, values in values_by_parameter.items():
            for value in values:
                curve_idx += 1
                solver = ALGORITHMS[algorithm][1]()
                kwargs = apply_parameter(algorithm, parameter, value)
                started = time.perf_counter()
                print(
                    f"[{curve_idx}/{total_curves}] algorithm={algorithm} parameter={parameter} value={stable_value(value)}"
                )
                for episode in range(1, args.episodes + 1):
                    result = run_solver(
                        algorithm=algorithm,
                        topology=topology,
                        requests=requests,
                        seed=args.seed + episode,
                        opt_time_limit=1.0,
                        extra_kwargs=kwargs,
                        solver_instance=solver,
                    )
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "parameter": parameter,
                            "parameter_value": value,
                            "learning_rate": kwargs.get("learning_rate", kwargs.get("critic_lr")),
                            "batch_size": kwargs.get("batch_size"),
                            "buffer_size": kwargs.get("replay_capacity"),
                            "episode": episode,
                            "status": result["status"],
                            "error": result["error"],
                            "loss": result["loss"],
                            "cumulative_reward": result["cumulative_reward"],
                            "avg_latency_ms": result["avg_latency_ms"],
                            "max_latency_ms": result["max_latency_ms"],
                            "total_cost": result["total_cost"],
                            "runtime_s": result["runtime_s"],
                        }
                    )
                    if args.progress_interval > 0 and episode % args.progress_interval == 0:
                        print(
                            f"  episode={episode}/{args.episodes} elapsed_s={time.perf_counter() - started:.1f}"
                        )
    return pd.DataFrame(rows)


def plot_single_algorithm_parameter(
    df: pd.DataFrame,
    algorithm: str,
    parameter: str,
    metric: str,
    output: Path,
    smooth_window: int,
) -> None:
    subset = df[(df["algorithm"] == algorithm) & (df["parameter"] == parameter)].copy()
    subset = subset[subset["status"] == "ok"].dropna(subset=[metric])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for value, group in subset.groupby("parameter_value", sort=True):
        group = group.sort_values("episode")
        y_values = rolling_series(group[metric], smooth_window)
        label = f"{parameter}={stable_value(value)}"
        ax.plot(group["episode"], y_values, linewidth=1.6, label=label)
    ax.set_title(f"{algorithm} {metric.replace('_', ' ')} under different {parameter}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward" if metric == "cumulative_reward" else "Loss")
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_all_curves(df: pd.DataFrame, figure_dir: Path, smooth_window: int) -> None:
    for algorithm in RL_ALGORITHMS:
        for parameter in PARAMETER_VALUES:
            plot_single_algorithm_parameter(
                df=df,
                algorithm=algorithm,
                parameter=parameter,
                metric="cumulative_reward",
                output=figure_dir / f"{slug(algorithm)}_{parameter}_reward.png",
                smooth_window=smooth_window,
            )
            plot_single_algorithm_parameter(
                df=df,
                algorithm=algorithm,
                parameter=parameter,
                metric="loss",
                output=figure_dir / f"{slug(algorithm)}_{parameter}_loss.png",
                smooth_window=smooth_window,
            )


def finite_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean[np.isfinite(clean)]
    return float(clean.mean()) if not clean.empty else math.nan


def tail_slope(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return math.nan
    x_values = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x_values, values.astype(float), deg=1)
    return float(slope)


def summarize_convergence(df: pd.DataFrame, episodes: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    window = max(20, min(100, episodes // 10))
    for (algorithm, parameter, value), group in df[df["status"] == "ok"].groupby(
        ["algorithm", "parameter", "parameter_value"], sort=True
    ):
        group = group.sort_values("episode")
        loss = pd.to_numeric(group["loss"], errors="coerce")
        reward = pd.to_numeric(group["cumulative_reward"], errors="coerce")
        first_loss = finite_mean(loss.head(window))
        final_loss = finite_mean(loss.tail(window))
        first_reward = finite_mean(reward.head(window))
        final_reward = finite_mean(reward.tail(window))
        tail_loss_values = loss.tail(window).dropna().to_numpy(dtype=float)
        slope = tail_slope(tail_loss_values)
        normalized_tail_change = (
            abs(slope) * window / max(abs(final_loss), 1e-9) if math.isfinite(slope) and math.isfinite(final_loss) else math.nan
        )
        loss_change_pct = (
            (final_loss - first_loss) / max(abs(first_loss), 1e-9)
            if math.isfinite(first_loss) and math.isfinite(final_loss)
            else math.nan
        )
        reward_change_pct = (
            (final_reward - first_reward) / max(abs(first_reward), 1e-9)
            if math.isfinite(first_reward) and math.isfinite(final_reward)
            else math.nan
        )
        loss_decreased = bool(math.isfinite(loss_change_pct) and loss_change_pct < -0.02)
        tail_stable = bool(math.isfinite(normalized_tail_change) and normalized_tail_change < 0.10)
        if loss_decreased and tail_stable:
            trend = "convergent"
        elif loss_decreased:
            trend = "decreasing_but_unstable_tail"
        elif tail_stable:
            trend = "stable_no_clear_decrease"
        else:
            trend = "no_clear_convergence"
        rows.append(
            {
                "algorithm": algorithm,
                "parameter": parameter,
                "parameter_value": value,
                "episodes": int(group["episode"].max()),
                "window": window,
                "first_loss_mean": first_loss,
                "final_loss_mean": final_loss,
                "loss_change_pct": loss_change_pct,
                "tail_loss_slope": slope,
                "normalized_tail_change": normalized_tail_change,
                "first_reward_mean": first_reward,
                "final_reward_mean": final_reward,
                "reward_change_pct": reward_change_pct,
                "trend": trend,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1000:
        raise ValueError("--episodes must be at least 1000 for this convergence experiment.")
    configure_style()
    timestamp = datetime.now().strftime("rl_%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    figure_dir = output_dir / "figures"
    ensure_dir(figure_dir)

    started = time.perf_counter()
    curves = run_curves(args, output_dir)
    summary = summarize_convergence(curves, args.episodes)
    plot_all_curves(curves, figure_dir, args.smooth_window)

    curves.to_csv(output_dir / "rl_hyperparam_curves.csv", index=False)
    summary.to_csv(output_dir / "convergence_summary.csv", index=False)
    manifest = {
        "output_dir": str(output_dir),
        "elapsed_seconds": time.perf_counter() - started,
        "algorithms": RL_ALGORITHMS,
        "episodes": args.episodes,
        "node_count": args.node_count,
        "request_count": args.request_count,
        "time_slots": args.time_slots,
        "n_domains": args.n_domains,
        "seed": args.seed,
        "parameter_values": parameter_sets(args),
        "smooth_window": args.smooth_window,
        "raw_rows": int(len(curves)),
        "summary_rows": int(len(summary)),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()