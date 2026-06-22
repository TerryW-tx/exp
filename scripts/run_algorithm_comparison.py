from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vsfc_lab.mock_components import (  # noqa: E402
    DuelingDDQNSolver,
    GreedySolver,
    MockTopology,
    PDQNApproxSolver,
    PlaceholderMetrics,
    PlaceholderSolver,
    TD3DDQNSolver,
    build_mock_topology,
    save_scenario_config,
)
from vsfc_lab.models import FIXED_VNF_INSTANCES, PlacementSolution, SFCRequest  # noqa: E402


AlgorithmFactory = Callable[[], object]


ALGORITHMS: dict[str, tuple[str, AlgorithmFactory]] = {
    "Opt": ("PlaceholderSolver", PlaceholderSolver),
    "PAS": ("PDQNApproxSolver", PDQNApproxSolver),
    "Greedy": ("GreedySolver", GreedySolver),
    "TD3-DDQN": ("TD3DDQNSolver", TD3DDQNSolver),
    "Dual": ("DuelingDDQNSolver", DuelingDDQNSolver),
}

RL_ALGORITHMS = ["PAS", "TD3-DDQN", "Dual"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SFC algorithm comparison experiments and plot results."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "algorithm_comparison",
        help="Directory under which a timestamped experiment folder is created.",
    )
    parser.add_argument(
        "--request-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Increasing SFC request counts to evaluate.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds.")
    parser.add_argument("--time-slots", type=int, default=3, help="Number of traffic time slots.")
    parser.add_argument("--node-count", type=int, default=5, help="Topology node count.")
    parser.add_argument("--n-domains", type=int, default=2, help="Number of MEC domains.")
    parser.add_argument(
        "--opt-time-limit",
        type=float,
        default=12.0,
        help="CBC time limit in seconds for Opt/PlaceholderSolver per run.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=55.0,
        help="Soft wall-clock budget. The script stops before starting new batches after this limit.",
    )
    parser.add_argument(
        "--skip-rl-grid",
        action="store_true",
        help="Skip learning-rate/batch/buffer hyperparameter sweep.",
    )
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_topology(seed: int, node_count: int, n_domains: int) -> MockTopology:
    return build_mock_topology(
        seed=seed,
        max_nodes=node_count,
        node_count=node_count,
        n_domains=n_domains,
        link_mode="complete",
        node_capacity_min=70.0,
        node_capacity_max=95.0,
        edge_capacity_min=30.0,
        edge_capacity_max=45.0,
        cpu_price=1.0,
        bandwidth_price=1.0,
    )


def build_requests(node_ids: list[int], request_count: int, time_slots: int) -> list[SFCRequest]:
    if len(node_ids) < 2:
        raise ValueError("At least two topology nodes are required.")
    requests: list[SFCRequest] = []
    vnf_chain = [FIXED_VNF_INSTANCES[0], FIXED_VNF_INSTANCES[1], FIXED_VNF_INSTANCES[2]]
    for req_idx in range(request_count):
        source = node_ids[req_idx % len(node_ids)]
        destination = node_ids[(req_idx + 1 + (req_idx % max(len(node_ids) - 1, 1))) % len(node_ids)]
        if destination == source:
            destination = node_ids[(req_idx + 1) % len(node_ids)]
        base_flow = 5.0 + float(req_idx % 3)
        flow_sizes = [base_flow + float((slot_idx + req_idx) % 3) for slot_idx in range(time_slots)]
        requests.append(
            SFCRequest(
                request_id=f"req_{req_idx}",
                source=source,
                destination=destination,
                vnf_chain=vnf_chain,
                flow_sizes=flow_sizes,
                demand=1.0,
                latency_budget=30.0,
                metadata={"domain_hint": req_idx % 2},
            )
        )
    return requests


def base_run_kwargs(opt_time_limit: float) -> dict[str, Any]:
    return {
        "time_limit": opt_time_limit,
        "solver_msg": False,
        "epsilon": 0.05,
        "gamma": 0.95,
        "learning_rate": 0.03,
        "critic_lr": 0.03,
        "actor_lr": 0.01,
        "batch_size": 8,
        "replay_capacity": 128,
        "max_cross_domain_changes": 1,
        "max_candidate_actions": 10,
        "min_offload_fraction": 0.0,
        "max_offload_fraction": 0.65,
        "min_edge_fraction": 0.05,
        "offload_retry_step_fraction": 0.05,
        "ddqn_offload_step_fraction": 0.10,
        "dueling_offload_step_fraction": 0.10,
        "greedy_offload_step_fraction": 0.10,
        "offloading_unit_cost": 5.0,
        "cpu_reference_flow": 10.0,
        "processing_delay_unit": 0.001,
        "init_delay_ms": 0.05,
        "cloud_proc_delay_unit": 0.02,
        "cloud_bandwidth": 100.0,
        "cloud_upload_delay_ms": 2.0,
        "cloud_download_delay_ms": 2.0,
        "eta_deploy": 1.0,
        "eta_transfer": 1.0,
        "eta_transmit": 1.0,
        "eta_offload": 1.0,
        "infeasible_penalty": 1000.0,
        "target_tau": 0.05,
        "tau": 0.05,
        "td3_tau": 0.05,
        "td3_policy_noise": 0.05,
        "td3_noise_clip": 0.02,
        "td3_policy_delay": 2,
    }


def algorithm_kwargs(short_name: str, opt_time_limit: float) -> dict[str, Any]:
    kwargs = base_run_kwargs(opt_time_limit)
    if short_name == "PAS":
        kwargs.update(
            {
                "epsilon": 0.0,
                "max_offload_fraction": 0.40,
                "learning_rate": 0.04,
                "critic_lr": 0.04,
                "batch_size": 8,
                "replay_capacity": 128,
            }
        )
    elif short_name == "Greedy":
        kwargs.update(
            {
                "min_offload_fraction": 0.70,
                "max_offload_fraction": 0.90,
                "greedy_offload_step_fraction": 0.10,
            }
        )
    elif short_name == "TD3-DDQN":
        kwargs.update(
            {
                "epsilon": 0.20,
                "min_offload_fraction": 0.65,
                "max_offload_fraction": 0.90,
            }
        )
    elif short_name == "Dual":
        kwargs.update(
            {
                "epsilon": 0.20,
                "min_offload_fraction": 0.65,
                "max_offload_fraction": 0.90,
            }
        )
    return kwargs


def solution_cost(short_name: str, solutions: list[PlacementSolution]) -> float:
    if not solutions:
        return math.nan
    objectives = [float(solution.metadata.get("objective", math.nan)) for solution in solutions]
    if short_name == "Opt":
        return objectives[0]
    total = 0.0
    for solution, objective in zip(solutions, objectives):
        if math.isfinite(objective):
            total += objective
            continue
        timeslot_cost = solution.metadata.get("timeslot_cost", {})
        if isinstance(timeslot_cost, dict):
            total += sum(float(value) for value in timeslot_cost.values())
    return total


def extract_reward_values(solutions: list[PlacementSolution]) -> list[float]:
    rewards: list[float] = []
    for solution in solutions:
        for key in ("timeslot_reward", "reward_trace"):
            value = solution.metadata.get(key)
            if isinstance(value, dict):
                rewards.extend(float(item) for _, item in sorted(value.items(), key=lambda pair: int(pair[0])))
            elif isinstance(value, list):
                rewards.extend(float(item) for item in value)
    return rewards


def discounted_return_loss(rewards: list[float], gamma: float) -> float:
    if not rewards:
        return math.nan
    running_return = 0.0
    losses: list[float] = []
    for reward in reversed(rewards):
        target = reward + gamma * running_return
        losses.append((target - running_return) ** 2)
        running_return = target
    return float(statistics.fmean(losses)) if losses else math.nan


def run_solver(
    short_name: str,
    topology: MockTopology,
    requests: list[SFCRequest],
    seed: int,
    opt_time_limit: float,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, factory = ALGORITHMS[short_name]
    solver = factory()
    kwargs = algorithm_kwargs(short_name, opt_time_limit)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs["seed"] = seed
    metrics = PlaceholderMetrics()
    started = time.perf_counter()
    try:
        solutions = solver.solve(topology=topology, requests=requests, **kwargs)  # type: ignore[attr-defined]
        runtime_s = time.perf_counter() - started
        metric_values = metrics.evaluate(topology=topology, requests=requests, solutions=solutions)
        rewards = extract_reward_values(solutions)
        row = {
            "status": "ok",
            "error": "",
            "avg_latency_ms": float(metric_values.get("avg_latency_ms", math.nan)),
            "max_latency_ms": float(metric_values.get("max_latency_ms", math.nan)),
            "total_cost": solution_cost(short_name, solutions),
            "runtime_s": runtime_s,
            "cumulative_reward": float(sum(rewards)) if rewards else math.nan,
            "loss": discounted_return_loss(rewards, float(kwargs.get("gamma", 0.95))),
            "solutions": [asdict(solution) for solution in solutions],
        }
    except Exception as exc:  # noqa: BLE001
        runtime_s = time.perf_counter() - started
        row = {
            "status": "failed",
            "error": repr(exc),
            "avg_latency_ms": math.nan,
            "max_latency_ms": math.nan,
            "total_cost": math.nan,
            "runtime_s": runtime_s,
            "cumulative_reward": math.nan,
            "loss": math.nan,
            "solutions": [],
        }
    return row


def should_continue(started_at: float, max_runtime_minutes: float) -> bool:
    return (time.perf_counter() - started_at) < max_runtime_minutes * 60.0


def plot_line(
    df: pd.DataFrame,
    output: Path,
    y: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for algorithm in ALGORITHMS:
        subset = df[df["algorithm"] == algorithm].sort_values("request_count")
        if subset.empty:
            continue
        grouped = subset.groupby("request_count", as_index=False)[y].mean(numeric_only=True)
        ax.plot(grouped["request_count"], grouped[y], marker="o", label=algorithm)
    ax.set_xlabel("SFC request count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_grouped_bar(df: pd.DataFrame, output: Path, request_count: int) -> None:
    metrics = ["avg_latency_ms", "max_latency_ms", "total_cost", "runtime_s"]
    labels = ["Avg latency", "Max latency", "Total cost", "Runtime"]
    subset = df[df["request_count"] == request_count].copy()
    grouped = subset.groupby("algorithm", as_index=False)[metrics].mean(numeric_only=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, label in zip(axes.flatten(), metrics, labels):
        values = [float(grouped[grouped["algorithm"] == algo][metric].iloc[0]) if algo in set(grouped["algorithm"]) else math.nan for algo in ALGORITHMS]
        ax.bar(list(ALGORITHMS), values)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(f"Algorithm metrics at request_count={request_count}")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_rl_grid(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    label = df.apply(
        lambda row: f"lr={row['learning_rate']},B={row['batch_size']},Buf={row['buffer_size']}",
        axis=1,
    )
    df = df.copy()
    df["param_label"] = label
    for metric, title, filename in [
        ("cumulative_reward", "RL cumulative reward", "rl_cumulative_reward.png"),
        ("loss", "RL TD-style loss", "rl_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.2))
        pivot = df.pivot_table(index="param_label", columns="algorithm", values=metric, aggfunc="mean")
        pivot = pivot[[col for col in RL_ALGORITHMS if col in pivot.columns]]
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Hyperparameter setting")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def run_request_scaling(args: argparse.Namespace, output_dir: Path, started_at: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    scenario_dir = output_dir / "scenarios"
    ensure_dir(scenario_dir)
    for request_count in args.request_counts:
        for seed in args.seeds:
            if not should_continue(started_at, args.max_runtime_minutes):
                return pd.DataFrame(rows), raw_runs
            topology = build_topology(seed=seed, node_count=args.node_count, n_domains=args.n_domains)
            requests = build_requests(topology.node_ids, request_count, args.time_slots)
            save_scenario_config(scenario_dir / f"scenario_req{request_count}_seed{seed}.json", topology, requests)
            for short_name in ALGORITHMS:
                if not should_continue(started_at, args.max_runtime_minutes):
                    return pd.DataFrame(rows), raw_runs
                result = run_solver(short_name, topology, requests, seed, args.opt_time_limit)
                row = {
                    "experiment": "request_scaling",
                    "request_count": request_count,
                    "seed": seed,
                    "algorithm": short_name,
                    "algorithm_name": ALGORITHMS[short_name][0],
                    "status": result["status"],
                    "error": result["error"],
                    "avg_latency_ms": result["avg_latency_ms"],
                    "max_latency_ms": result["max_latency_ms"],
                    "total_cost": result["total_cost"],
                    "runtime_s": result["runtime_s"],
                    "cumulative_reward": result["cumulative_reward"],
                    "loss": result["loss"],
                }
                rows.append(row)
                raw_runs.append({**row, "solutions": result["solutions"]})
    return pd.DataFrame(rows), raw_runs


def run_rl_grid(args: argparse.Namespace, output_dir: Path, started_at: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if args.skip_rl_grid:
        return pd.DataFrame(), []
    learning_rates = [0.01, 0.03]
    batch_sizes = [4, 8]
    buffer_sizes = [64, 128]
    request_count = max(args.request_counts)
    seed = args.seeds[0]
    topology = build_topology(seed=seed + 100, node_count=args.node_count, n_domains=args.n_domains)
    requests = build_requests(topology.node_ids, request_count, args.time_slots)
    save_scenario_config(output_dir / "scenarios" / "rl_grid_scenario.json", topology, requests)
    rows: list[dict[str, Any]] = []
    raw_runs: list[dict[str, Any]] = []
    for algorithm in RL_ALGORITHMS:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for buffer_size in buffer_sizes:
                    if not should_continue(started_at, args.max_runtime_minutes):
                        return pd.DataFrame(rows), raw_runs
                    result = run_solver(
                        algorithm,
                        topology,
                        requests,
                        seed,
                        args.opt_time_limit,
                        extra_kwargs={
                            "learning_rate": learning_rate,
                            "critic_lr": learning_rate,
                            "actor_lr": max(learning_rate / 3.0, 0.001),
                            "batch_size": batch_size,
                            "replay_capacity": buffer_size,
                        },
                    )
                    row = {
                        "experiment": "rl_grid",
                        "algorithm": algorithm,
                        "algorithm_name": ALGORITHMS[algorithm][0],
                        "request_count": request_count,
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "buffer_size": buffer_size,
                        "status": result["status"],
                        "error": result["error"],
                        "cumulative_reward": result["cumulative_reward"],
                        "loss": result["loss"],
                        "avg_latency_ms": result["avg_latency_ms"],
                        "max_latency_ms": result["max_latency_ms"],
                        "total_cost": result["total_cost"],
                        "runtime_s": result["runtime_s"],
                    }
                    rows.append(row)
                    raw_runs.append({**row, "solutions": result["solutions"]})
    return pd.DataFrame(rows), raw_runs


def main() -> None:
    args = build_parser().parse_args()
    started_at = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"run_{timestamp}"
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "scenarios")

    metadata = {
        "created_at": timestamp,
        "request_counts": args.request_counts,
        "seeds": args.seeds,
        "time_slots": args.time_slots,
        "node_count": args.node_count,
        "n_domains": args.n_domains,
        "algorithm_abbreviations": {short: name for short, (name, _) in ALGORITHMS.items()},
        "runtime_budget_minutes": args.max_runtime_minutes,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    scaling_df, scaling_raw = run_request_scaling(args, output_dir, started_at)
    scaling_df.to_csv(output_dir / "request_scaling_metrics.csv", index=False)
    (output_dir / "request_scaling_runs.json").write_text(
        json.dumps(scaling_raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rl_df, rl_raw = run_rl_grid(args, output_dir, started_at)
    rl_df.to_csv(output_dir / "rl_hyperparam_metrics.csv", index=False)
    (output_dir / "rl_hyperparam_runs.json").write_text(
        json.dumps(rl_raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_dir = output_dir / "figures"
    if not scaling_df.empty:
        ok_df = scaling_df[scaling_df["status"] == "ok"]
        plot_line(ok_df, figure_dir / "avg_latency_vs_requests.png", "avg_latency_ms", "Average service latency (ms)", "Average latency vs request count")
        plot_line(ok_df, figure_dir / "max_latency_vs_requests.png", "max_latency_ms", "Maximum service latency (ms)", "Maximum latency vs request count")
        plot_line(ok_df, figure_dir / "total_cost_vs_requests.png", "total_cost", "Total cost", "Total cost vs request count")
        plot_line(ok_df, figure_dir / "runtime_vs_requests.png", "runtime_s", "Runtime (s)", "Runtime vs request count")
        plot_grouped_bar(ok_df, figure_dir / "final_request_count_grouped_bars.png", max(args.request_counts))
    if not rl_df.empty:
        plot_rl_grid(rl_df[rl_df["status"] == "ok"], figure_dir)

    elapsed_s = time.perf_counter() - started_at
    summary = {
        "output_dir": str(output_dir),
        "elapsed_s": elapsed_s,
        "request_scaling_rows": int(len(scaling_df)),
        "rl_grid_rows": int(len(rl_df)),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()