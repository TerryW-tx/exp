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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
ALGORITHM_ORDER = list(ALGORITHMS)
COLORS = {
    "Opt": "#1f77b4",
    "PAS": "#d62728",
    "Greedy": "#2ca02c",
    "TD3-DDQN": "#9467bd",
    "Dual": "#ff7f0e",
}
MARKERS = {
    "Opt": "o",
    "PAS": "s",
    "Greedy": "^",
    "TD3-DDQN": "D",
    "Dual": "v",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run thesis-scale experiments and generate figures.")
    parser.add_argument("--output-root", type=Path, default=Path("results") / "thesis_figures")
    parser.add_argument("--node-counts", type=int, nargs="+", default=list(range(2, 101, 2)))
    parser.add_argument("--request-counts", type=int, nargs="+", default=list(range(10, 201, 2)))
    parser.add_argument("--repeats", type=int, default=5, help="Repeated runs per experiment point; must be >=5.")
    parser.add_argument("--time-slots", type=int, default=3)
    parser.add_argument("--n-domains", type=int, default=4)
    parser.add_argument("--fixed-request-count", type=int, default=10)
    parser.add_argument("--fixed-node-count", type=int, default=10)
    parser.add_argument("--opt-time-limit", type=float, default=3.0)
    parser.add_argument("--max-runtime-minutes", type=float, default=55.0)
    parser.add_argument("--rl-iterations", type=int, default=12)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run every node_count/request_count pair. The default runs the two thesis sweeps needed for line figures.",
    )
    parser.add_argument(
        "--save-raw-solutions",
        action="store_true",
        help="Persist full per-run placement solutions. Disabled by default for large thesis sweeps.",
    )
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def should_continue(started_at: float, max_runtime_minutes: float) -> bool:
    return (time.perf_counter() - started_at) < max_runtime_minutes * 60.0


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def build_synthetic_topology(node_count: int, n_domains: int, seed: int) -> MockTopology:
    rng = __import__("random").Random(seed * 1009 + node_count)
    node_ids = list(range(node_count))
    node_domains = {node_id: (node_id % n_domains) + 1 for node_id in node_ids}
    edge_set: set[tuple[int, int]] = set()
    for node_id in node_ids:
        edge_set.add(canonical_edge(node_id, (node_id + 1) % node_count))
        if node_count > 5:
            edge_set.add(canonical_edge(node_id, (node_id + 2) % node_count))
        if node_id + 5 < node_count:
            edge_set.add(canonical_edge(node_id, node_id + 5))
    edges = sorted(edge_set)
    node_capacities = {
        node_id: round(rng.uniform(170.0, 340.0) + 1.5 * math.sqrt(node_count), 3)
        for node_id in node_ids
    }
    edge_capacities = {
        edge: round(rng.uniform(110.0, 260.0) + 1.2 * math.sqrt(node_count), 3)
        for edge in edges
    }
    edge_delay: dict[tuple[int, int], float] = {}
    for u, v in edges:
        hop_distance = min(abs(u - v), node_count - abs(u - v))
        domain_penalty = 0.04 if node_domains[u] != node_domains[v] else 0.0
        edge_delay[(u, v)] = round(0.15 + 0.012 * hop_distance + domain_penalty + rng.uniform(0.0, 0.015), 4)
    return MockTopology(
        node_ids=node_ids,
        edge_list=edges,
        node_capacities=node_capacities,
        edge_capacities=edge_capacities,
        edge_delay=edge_delay,
        node_domains=node_domains,
        cpu_price=1.0,
        bandwidth_price=1.0,
    )


def build_requests(node_ids: list[int], request_count: int, time_slots: int, seed: int) -> list[SFCRequest]:
    rng = __import__("random").Random(seed * 7919 + request_count)
    chains = [
        [FIXED_VNF_INSTANCES[0], FIXED_VNF_INSTANCES[1]],
        [FIXED_VNF_INSTANCES[0], FIXED_VNF_INSTANCES[1], FIXED_VNF_INSTANCES[2]],
        [FIXED_VNF_INSTANCES[1], FIXED_VNF_INSTANCES[2]],
    ]
    requests: list[SFCRequest] = []
    for req_idx in range(request_count):
        source = node_ids[(req_idx * 3 + seed) % len(node_ids)]
        destination = node_ids[(req_idx * 5 + 7 + seed) % len(node_ids)]
        if destination == source:
            destination = node_ids[(node_ids.index(source) + max(1, len(node_ids) // 3)) % len(node_ids)]
        traffic_mean = 4.0 + 0.18 * request_count + 0.35 * (req_idx % 5)
        traffic_std = max(1.8, 0.35 * traffic_mean)
        flow_sizes = [
            round(max(0.5, rng.gauss(traffic_mean, traffic_std)), 3)
            for _ in range(time_slots)
        ]
        requests.append(
            SFCRequest(
                request_id=f"req_{req_idx}",
                source=source,
                destination=destination,
                vnf_chain=chains[req_idx % len(chains)],
                flow_sizes=flow_sizes,
                demand=1.0,
                latency_budget=35.0,
                metadata={
                    "request_index": req_idx,
                    "traffic_distribution": "truncated_normal",
                    "traffic_mean": traffic_mean,
                    "traffic_std": traffic_std,
                },
            )
        )
    return requests


def base_kwargs(opt_time_limit: float) -> dict[str, Any]:
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
        "max_candidate_actions": 8,
        "min_offload_fraction": 0.0,
        "max_offload_fraction": 0.65,
        "min_edge_fraction": 0.05,
        "offload_retry_step_fraction": 0.05,
        "ddqn_offload_step_fraction": 0.1,
        "dueling_offload_step_fraction": 0.1,
        "greedy_offload_step_fraction": 0.1,
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


def algorithm_kwargs(algorithm: str, opt_time_limit: float) -> dict[str, Any]:
    kwargs = base_kwargs(opt_time_limit)
    if algorithm == "PAS":
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
    elif algorithm == "Greedy":
        kwargs.update({"min_offload_fraction": 0.70, "max_offload_fraction": 0.90})
    elif algorithm == "TD3-DDQN":
        kwargs.update({"epsilon": 0.20, "min_offload_fraction": 0.65, "max_offload_fraction": 0.90})
    elif algorithm == "Dual":
        kwargs.update({"epsilon": 0.20, "min_offload_fraction": 0.65, "max_offload_fraction": 0.90})
    return kwargs


def can_run_opt(node_count: int, request_count: int) -> bool:
    return node_count <= 20 and request_count <= 30


def solution_cost(algorithm: str, solutions: list[PlacementSolution]) -> float:
    if not solutions:
        return math.nan
    objectives = [float(solution.metadata.get("objective", math.nan)) for solution in solutions]
    if algorithm == "Opt":
        return objectives[0]
    total = 0.0
    for solution, objective in zip(solutions, objectives):
        if math.isfinite(objective):
            total += objective
        else:
            costs = solution.metadata.get("timeslot_cost", {})
            if isinstance(costs, dict):
                total += sum(float(value) for value in costs.values())
    return total


def segment_flow_coefficients(request: SFCRequest) -> list[float]:
    coeffs = [1.0]
    acc = 1.0
    for vnf in request.vnf_chain:
        acc *= vnf.flow_scaling_factor
        coeffs.append(acc)
    return coeffs


def unified_solution_cost(
    topology: MockTopology,
    requests: list[SFCRequest],
    solutions: list[PlacementSolution],
    offloading_unit_cost: float,
) -> float:
    solution_by_request = {solution.request_id: solution for solution in solutions}
    total = 0.0
    for request in requests:
        solution = solution_by_request.get(request.request_id)
        if solution is None:
            return math.nan
        coeffs = segment_flow_coefficients(request)
        for t_idx, flow_size in enumerate(request.timeslot_flow_sizes()):
            offloading_flow = float(solution.timeslot_offloading_flow.get(t_idx, solution.offloading_flow))
            edge_flow = max(float(flow_size) - offloading_flow, 0.0)
            vnf_to_node = solution.timeslot_vnf_to_node.get(t_idx, solution.vnf_to_node)
            path_mapping = solution.timeslot_path_mapping.get(t_idx, solution.path_mapping)
            for vnf in request.vnf_chain:
                if vnf.instance_id not in vnf_to_node:
                    continue
                total += topology.cpu_price * vnf.cpu_demand
            for seg_idx in range(len(request.vnf_chain) + 1):
                segment_flow = edge_flow * coeffs[seg_idx]
                path = path_mapping.get(f"segment_{seg_idx}", [])
                total += topology.bandwidth_price * segment_flow * max(len(path) - 1, 0)
            total += offloading_unit_cost * offloading_flow
    return float(total)


def extract_rewards(solutions: list[PlacementSolution]) -> list[float]:
    rewards: list[float] = []
    for solution in solutions:
        for key in ("timeslot_reward", "reward_trace"):
            value = solution.metadata.get(key)
            if isinstance(value, dict):
                rewards.extend(float(item) for item in value.values())
            elif isinstance(value, list):
                rewards.extend(float(item) for item in value)
    return rewards


def loss_proxy(rewards: list[float], gamma: float = 0.95) -> float:
    if not rewards:
        return math.nan
    running_return = 0.0
    losses: list[float] = []
    for reward in reversed(rewards):
        target = reward + gamma * running_return
        losses.append((target - running_return) ** 2)
        running_return = target
    return float(statistics.fmean(losses))


def run_solver(
    algorithm: str,
    topology: MockTopology,
    requests: list[SFCRequest],
    seed: int,
    opt_time_limit: float,
    extra_kwargs: dict[str, Any] | None = None,
    solver_instance: object | None = None,
) -> dict[str, Any]:
    kwargs = algorithm_kwargs(algorithm, opt_time_limit)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs["seed"] = seed
    solver = solver_instance if solver_instance is not None else ALGORITHMS[algorithm][1]()
    metrics = PlaceholderMetrics()
    started = time.perf_counter()
    try:
        solutions = solver.solve(topology=topology, requests=requests, **kwargs)  # type: ignore[attr-defined]
        runtime_s = time.perf_counter() - started
        metric_values = metrics.evaluate(topology=topology, requests=requests, solutions=solutions)
        rewards = extract_rewards(solutions)
        cost_value = unified_solution_cost(
            topology=topology,
            requests=requests,
            solutions=solutions,
            offloading_unit_cost=float(kwargs.get("offloading_unit_cost", 5.0)),
        )
        return {
            "status": "ok",
            "error": "",
            "avg_latency_ms": float(metric_values.get("avg_latency_ms", math.nan)),
            "max_latency_ms": float(metric_values.get("max_latency_ms", math.nan)),
            "total_cost": cost_value,
            "runtime_s": runtime_s,
            "cumulative_reward": float(sum(rewards)) if rewards else math.nan,
            "loss": loss_proxy(rewards, float(kwargs.get("gamma", 0.95))),
            "solutions": [asdict(solution) for solution in solutions],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error": repr(exc),
            "avg_latency_ms": math.nan,
            "max_latency_ms": math.nan,
            "total_cost": math.nan,
            "runtime_s": time.perf_counter() - started,
            "cumulative_reward": math.nan,
            "loss": math.nan,
            "solutions": [],
        }


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df.loc[~np.isfinite(df[metric]), metric] = math.nan
    grouped = df.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else math.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"{metric}_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_line_with_band(
    summary: pd.DataFrame,
    x_col: str,
    metric: str,
    output: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for algorithm in ALGORITHM_ORDER:
        subset = summary[summary["algorithm"] == algorithm].sort_values(x_col)
        subset = subset.dropna(subset=[f"{metric}_mean"])
        if subset.empty:
            continue
        x_values = subset[x_col].astype(float).to_numpy()
        means = subset[f"{metric}_mean"].astype(float).to_numpy()
        stds = subset[f"{metric}_std"].astype(float).fillna(0.0).to_numpy()
        ax.plot(
            x_values,
            means,
            marker=MARKERS.get(algorithm, "o"),
            color=COLORS.get(algorithm),
            linewidth=1.8,
            markersize=4.5,
            label=algorithm,
        )
        ax.fill_between(x_values, means - stds, means + stds, color=COLORS.get(algorithm), alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_metric_bars(summary: pd.DataFrame, output: Path, node_count: int, request_count: int) -> None:
    subset = summary[(summary["node_count"] == node_count) & (summary["request_count"] == request_count)]
    metrics = ["avg_latency_ms", "max_latency_ms", "total_cost", "runtime_s"]
    labels = ["Average latency (ms)", "Maximum latency (ms)", "Total cost", "Runtime (s)"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, metric, label in zip(axes.flatten(), metrics, labels):
        values = []
        errors = []
        for algorithm in ALGORITHM_ORDER:
            row = subset[subset["algorithm"] == algorithm]
            values.append(float(row[f"{metric}_mean"].iloc[0]) if not row.empty else math.nan)
            errors.append(float(row[f"{metric}_std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(ALGORITHM_ORDER, values, yerr=errors, capsize=4, color=[COLORS[a] for a in ALGORITHM_ORDER])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Grouped metric bars (nodes={node_count}, requests={request_count})")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_heatmap_grid(
    summary: pd.DataFrame,
    metric: str,
    output: Path,
    title: str,
    colorbar_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes_flat = axes.flatten()
    for ax, algorithm in zip(axes_flat, ALGORITHM_ORDER):
        subset = summary[summary["algorithm"] == algorithm]
        pivot = subset.pivot(index="request_count", columns="node_count", values=f"{metric}_mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        if pivot.empty:
            ax.axis("off")
            continue
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(algorithm)
        ax.set_xlabel("Node count")
        ax.set_ylabel("SFC request count")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(value)) for value in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(value)) for value in pivot.index])
        fig.colorbar(image, ax=ax, shrink=0.82, label=colorbar_label)
    axes_flat[-1].axis("off")
    fig.suptitle(title)
    fig.savefig(output)
    plt.close(fig)


def plot_pareto(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for algorithm in ALGORITHM_ORDER:
        subset = summary[summary["algorithm"] == algorithm].dropna(
            subset=["avg_latency_ms_mean", "total_cost_mean", "runtime_s_mean"]
        )
        if subset.empty:
            continue
        sizes = 35.0 + 260.0 * subset["runtime_s_mean"] / max(summary["runtime_s_mean"].max(), 1e-9)
        ax.scatter(
            subset["avg_latency_ms_mean"],
            subset["total_cost_mean"],
            s=sizes,
            alpha=0.68,
            color=COLORS.get(algorithm),
            label=algorithm,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_title("Pareto trade-off between latency and cost")
    ax.set_xlabel("Average service latency (ms)")
    ax.set_ylabel("Total cost")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_training_curve(
    df: pd.DataFrame,
    parameter: str,
    metric: str,
    output: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
) -> None:
    if df.empty or parameter not in df.columns or metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for value, group in df.groupby(parameter):
        summary = summarize_mean_std(group, ["iteration"], [metric])
        summary = summary.sort_values("iteration")
        x_values = summary["iteration"].astype(float).to_numpy()
        means = summary[f"{metric}_mean"].astype(float).to_numpy()
        stds = summary[f"{metric}_std"].astype(float).to_numpy()
        label_value = f"{value:g}" if isinstance(value, float) else str(value)
        ax.plot(x_values, means, marker="o", linewidth=1.7, markersize=3.2, label=f"{xlabel}={label_value}")
        ax.fill_between(x_values, means - stds, means + stds, alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend()
    if log_x:
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_final_loss(
    df: pd.DataFrame,
    parameter: str,
    output: Path,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    if df.empty or parameter not in df.columns or "final_loss" not in df.columns:
        return
    summary = summarize_mean_std(df, [parameter], ["final_loss"])
    if summary.empty:
        return
    summary = summary.sort_values(parameter)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.errorbar(
        summary[parameter].astype(float),
        summary["final_loss_mean"],
        yerr=summary["final_loss_std"],
        marker="o",
        capsize=4,
        linewidth=1.8,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final loss")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_param_heatmap(df: pd.DataFrame, x: str, y: str, output: Path, title: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns or "final_loss" not in df.columns:
        return
    pivot = df.pivot_table(index=y, columns=x, values="final_loss", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns], rotation=35)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    fig.colorbar(image, ax=ax, label="Final loss")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def run_scalability(args: argparse.Namespace, output_dir: Path, started_at: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    scenario_dir = output_dir / "scenarios"
    ensure_dir(scenario_dir)
    if args.full_grid:
        experiment_points = [
            (node_count, request_count)
            for node_count in args.node_counts
            for request_count in args.request_counts
        ]
    else:
        experiment_points = sorted(
            {
                *((node_count, args.fixed_request_count) for node_count in args.node_counts),
                *((args.fixed_node_count, request_count) for request_count in args.request_counts),
            }
        )
    total_points = len(experiment_points) * args.repeats
    point_idx = 0
    for node_count, request_count in experiment_points:
        for repeat in range(args.repeats):
            point_idx += 1
            if not should_continue(started_at, args.max_runtime_minutes):
                return pd.DataFrame(rows), raw
            seed = 1000 + 31 * repeat + node_count * 3 + request_count
            topology = build_synthetic_topology(node_count, args.n_domains, seed)
            requests = build_requests(topology.node_ids, request_count, args.time_slots, seed)
            if repeat == 0:
                save_scenario_config(scenario_dir / f"n{node_count}_r{request_count}.json", topology, requests)
            for algorithm in ALGORITHM_ORDER:
                if algorithm == "Opt" and not can_run_opt(node_count, request_count):
                    rows.append(
                        {
                            "node_count": node_count,
                            "request_count": request_count,
                            "repeat": repeat,
                            "algorithm": algorithm,
                            "status": "skipped",
                            "error": "Opt skipped for node_count>20 or request_count>30",
                            "avg_latency_ms": math.nan,
                            "max_latency_ms": math.nan,
                            "total_cost": math.nan,
                            "runtime_s": math.nan,
                            "cumulative_reward": math.nan,
                            "loss": math.nan,
                        }
                    )
                    continue
                result = run_solver(algorithm, topology, requests, seed, args.opt_time_limit)
                row = {
                    "node_count": node_count,
                    "request_count": request_count,
                    "repeat": repeat,
                    "algorithm": algorithm,
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
                if args.save_raw_solutions:
                    raw.append({**row, "solutions": result["solutions"]})
            print(f"Completed scalability point {point_idx}/{total_points}: nodes={node_count}, requests={request_count}, repeat={repeat}")
    return pd.DataFrame(rows), raw


def run_persistent_rl_iteration(
    solver: object,
    topology: MockTopology,
    requests: list[SFCRequest],
    algorithm: str,
    seed: int,
    kwargs: dict[str, Any],
) -> dict[str, float]:
    result = run_solver(
        algorithm,
        topology,
        requests,
        seed,
        opt_time_limit=1.0,
        extra_kwargs=kwargs,
        solver_instance=solver,
    )
    return {
        "loss": float(result["loss"]),
        "cumulative_reward": float(result["cumulative_reward"]),
        "avg_latency_ms": float(result["avg_latency_ms"]),
        "total_cost": float(result["total_cost"]),
    }


def run_rl_curves(args: argparse.Namespace, output_dir: Path, started_at: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    topology = build_synthetic_topology(25, args.n_domains, 4242)
    requests = build_requests(topology.node_ids, 25, args.time_slots, 4242)
    save_scenario_config(output_dir / "scenarios" / "rl_training_fixed_scenario.json", topology, requests)
    base = algorithm_kwargs("PAS", opt_time_limit=1.0)
    learning_rates = [0.001, 0.003, 0.01, 0.03]
    batch_sizes = [4, 8, 16, 32]
    buffer_sizes = [32, 64, 128, 256]
    curve_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    single_sweeps = [
        ("learning_rate", learning_rates),
        ("batch_size", batch_sizes),
        ("buffer_size", buffer_sizes),
    ]
    for parameter, values in single_sweeps:
        for value in values:
            for repeat in range(args.repeats):
                if not should_continue(started_at, args.max_runtime_minutes):
                    return pd.DataFrame(curve_rows), pd.DataFrame(final_rows)
                solver = PDQNApproxSolver()
                kwargs = dict(base)
                if parameter == "learning_rate":
                    kwargs.update({"learning_rate": value, "critic_lr": value, "actor_lr": max(float(value) / 3.0, 0.0005)})
                elif parameter == "batch_size":
                    kwargs["batch_size"] = int(value)
                elif parameter == "buffer_size":
                    kwargs["replay_capacity"] = int(value)
                losses: list[float] = []
                for iteration in range(1, args.rl_iterations + 1):
                    result = run_persistent_rl_iteration(
                        solver=solver,
                        topology=topology,
                        requests=requests,
                        algorithm="PAS",
                        seed=9000 + repeat * 100 + iteration,
                        kwargs=kwargs,
                    )
                    losses.append(result["loss"])
                    curve_rows.append(
                        {
                            "sweep": parameter,
                            "parameter": parameter,
                            parameter: value,
                            "repeat": repeat,
                            "iteration": iteration,
                            **result,
                        }
                    )
                final_rows.append(
                    {
                        "sweep": parameter,
                        "learning_rate": kwargs.get("learning_rate"),
                        "batch_size": kwargs.get("batch_size"),
                        "buffer_size": kwargs.get("replay_capacity"),
                        "repeat": repeat,
                        "final_loss": float(statistics.fmean(losses[-3:])),
                    }
                )

    pair_sweeps = [
        ("learning_rate", learning_rates, "batch_size", batch_sizes),
        ("learning_rate", learning_rates, "buffer_size", buffer_sizes),
        ("batch_size", batch_sizes, "buffer_size", buffer_sizes),
    ]
    pair_iterations = max(4, min(args.rl_iterations, 8))
    for x_name, x_values, y_name, y_values in pair_sweeps:
        for x_value in x_values:
            for y_value in y_values:
                for repeat in range(args.repeats):
                    if not should_continue(started_at, args.max_runtime_minutes):
                        return pd.DataFrame(curve_rows), pd.DataFrame(final_rows)
                    solver = PDQNApproxSolver()
                    kwargs = dict(base)
                    if x_name == "learning_rate" or y_name == "learning_rate":
                        lr = float(x_value if x_name == "learning_rate" else y_value)
                        kwargs.update({"learning_rate": lr, "critic_lr": lr, "actor_lr": max(lr / 3.0, 0.0005)})
                    if x_name == "batch_size" or y_name == "batch_size":
                        kwargs["batch_size"] = int(x_value if x_name == "batch_size" else y_value)
                    if x_name == "buffer_size" or y_name == "buffer_size":
                        kwargs["replay_capacity"] = int(x_value if x_name == "buffer_size" else y_value)
                    losses = []
                    for iteration in range(1, pair_iterations + 1):
                        result = run_persistent_rl_iteration(
                            solver=solver,
                            topology=topology,
                            requests=requests,
                            algorithm="PAS",
                            seed=12000 + repeat * 100 + iteration,
                            kwargs=kwargs,
                        )
                        losses.append(result["loss"])
                    final_rows.append(
                        {
                            "sweep": f"{x_name}_{y_name}",
                            "learning_rate": kwargs.get("learning_rate"),
                            "batch_size": kwargs.get("batch_size"),
                            "buffer_size": kwargs.get("replay_capacity"),
                            "repeat": repeat,
                            "final_loss": float(statistics.fmean(losses[-3:])),
                        }
                    )
    return pd.DataFrame(curve_rows), pd.DataFrame(final_rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 5:
        raise ValueError("--repeats must be at least 5 for this experiment design.")
    configure_style()
    started_at = time.perf_counter()
    exp_id = datetime.now().strftime("thesis_%Y%m%d_%H%M%S")
    output_dir = args.output_root / exp_id
    figure_dir = output_dir / "figures"
    ensure_dir(figure_dir)
    ensure_dir(output_dir / "scenarios")

    scaling_df, scaling_raw = run_scalability(args, output_dir, started_at)
    ok_scaling = scaling_df[scaling_df["status"] == "ok"].copy()
    summary = summarize_mean_std(
        ok_scaling,
        ["node_count", "request_count", "algorithm"],
        ["avg_latency_ms", "max_latency_ms", "total_cost", "runtime_s"],
    )
    fixed_request_summary = summary[summary["request_count"] == args.fixed_request_count]
    fixed_node_summary = summary[summary["node_count"] == args.fixed_node_count]

    plot_line_with_band(fixed_request_summary, "node_count", "avg_latency_ms", figure_dir / "01_node_avg_latency.png", "Average latency under different node counts", "Node count", "Average service latency (ms)")
    plot_line_with_band(fixed_request_summary, "node_count", "max_latency_ms", figure_dir / "02_node_max_latency.png", "Maximum latency under different node counts", "Node count", "Maximum service latency (ms)")
    plot_line_with_band(fixed_request_summary, "node_count", "runtime_s", figure_dir / "03_node_runtime.png", "Runtime under different node counts", "Node count", "Runtime (s)")
    plot_line_with_band(fixed_request_summary, "node_count", "total_cost", figure_dir / "04_node_total_cost.png", "Total cost under different node counts", "Node count", "Total cost")
    plot_line_with_band(fixed_node_summary, "request_count", "avg_latency_ms", figure_dir / "05_request_avg_latency.png", "Average latency under different request counts", "SFC request count", "Average service latency (ms)")
    plot_line_with_band(fixed_node_summary, "request_count", "max_latency_ms", figure_dir / "06_request_max_latency.png", "Maximum latency under different request counts", "SFC request count", "Maximum service latency (ms)")
    plot_line_with_band(fixed_node_summary, "request_count", "runtime_s", figure_dir / "07_request_runtime.png", "Runtime under different request counts", "SFC request count", "Runtime (s)")
    plot_line_with_band(fixed_node_summary, "request_count", "total_cost", figure_dir / "08_request_total_cost.png", "Total cost under different request counts", "SFC request count", "Total cost")

    plot_metric_bars(summary, figure_dir / "09_fixed_scenario_metric_bars.png", args.fixed_node_count, args.fixed_request_count)
    plot_heatmap_grid(summary, "avg_latency_ms", figure_dir / "10a_heatmap_avg_latency_all_algorithms.png", "Average latency heatmaps", "Average service latency (ms)")
    plot_heatmap_grid(summary, "total_cost", figure_dir / "10b_heatmap_total_cost_all_algorithms.png", "Total cost heatmaps", "Total cost")
    plot_heatmap_grid(summary, "runtime_s", figure_dir / "10c_heatmap_runtime_all_algorithms.png", "Runtime heatmaps", "Runtime (s)")

    rl_curve_df, rl_final_df = run_rl_curves(args, output_dir, started_at)
    if not rl_curve_df.empty and "sweep" in rl_curve_df.columns:
        plot_training_curve(rl_curve_df[rl_curve_df["sweep"] == "learning_rate"], "learning_rate", "cumulative_reward", figure_dir / "10_reward_curve_learning_rate.png", "Reward under different learning rates", "Learning rate", "Cumulative reward")
        plot_training_curve(rl_curve_df[rl_curve_df["sweep"] == "batch_size"], "batch_size", "cumulative_reward", figure_dir / "10_reward_curve_batch_size.png", "Reward under different batch sizes", "Batch size", "Cumulative reward")
        plot_training_curve(rl_curve_df[rl_curve_df["sweep"] == "buffer_size"], "buffer_size", "cumulative_reward", figure_dir / "10_reward_curve_buffer_size.png", "Reward under different buffer sizes", "Buffer size", "Cumulative reward")
        plot_training_curve(rl_curve_df[rl_curve_df["sweep"] == "learning_rate"], "learning_rate", "loss", figure_dir / "11_loss_curve_learning_rate.png", "Loss convergence under different learning rates", "Learning rate", "Loss")
        plot_training_curve(rl_curve_df[rl_curve_df["sweep"] == "batch_size"], "batch_size", "loss", figure_dir / "11_loss_curve_batch_size.png", "Loss convergence under different batch sizes", "Batch size", "Loss")
        plot_training_curve(rl_curve_df[rl_curve_df["sweep"] == "buffer_size"], "buffer_size", "loss", figure_dir / "11_loss_curve_buffer_size.png", "Loss convergence under different buffer sizes", "Buffer size", "Loss")
    if not rl_final_df.empty and "sweep" in rl_final_df.columns:
        plot_final_loss(rl_final_df[rl_final_df["sweep"] == "learning_rate"], "learning_rate", figure_dir / "15_final_loss_learning_rate_bar.png", "Final loss sensitivity to learning rate", "Learning rate", log_x=True)
        plot_final_loss(rl_final_df[rl_final_df["sweep"] == "batch_size"], "batch_size", figure_dir / "16_final_loss_batch_size_bar.png", "Final loss sensitivity to batch size", "Batch size")
        plot_final_loss(rl_final_df[rl_final_df["sweep"] == "buffer_size"], "buffer_size", figure_dir / "17_final_loss_buffer_size_bar.png", "Final loss sensitivity to buffer size", "Buffer size")
        plot_param_heatmap(rl_final_df[rl_final_df["sweep"] == "learning_rate_batch_size"], "learning_rate", "batch_size", figure_dir / "18_heatmap_lr_batch_final_loss.png", "Final loss: learning rate vs batch size")
        plot_param_heatmap(rl_final_df[rl_final_df["sweep"] == "learning_rate_buffer_size"], "learning_rate", "buffer_size", figure_dir / "19_heatmap_lr_buffer_final_loss.png", "Final loss: learning rate vs buffer size")
        plot_param_heatmap(rl_final_df[rl_final_df["sweep"] == "batch_size_buffer_size"], "batch_size", "buffer_size", figure_dir / "20_heatmap_batch_buffer_final_loss.png", "Final loss: batch size vs buffer size")

    scaling_df.to_csv(output_dir / "raw_scalability_results.csv", index=False)
    summary.to_csv(output_dir / "summary_scalability_mean_std.csv", index=False)
    rl_curve_df.to_csv(output_dir / "raw_rl_training_curves.csv", index=False)
    rl_final_df.to_csv(output_dir / "summary_rl_final_loss.csv", index=False)
    (output_dir / "raw_scalability_runs.json").write_text(json.dumps(scaling_raw, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "output_dir": str(output_dir),
        "elapsed_seconds": time.perf_counter() - started_at,
        "node_counts": args.node_counts,
        "request_counts": args.request_counts,
        "repeats": args.repeats,
        "fixed_node_count": args.fixed_node_count,
        "fixed_request_count": args.fixed_request_count,
        "scaling_mode": "full_grid" if args.full_grid else "node sweep at fixed request count plus request sweep at fixed node count",
        "opt_rule": "Opt is only run when node_count <= 20 and request_count <= 30.",
        "save_raw_solutions": bool(args.save_raw_solutions),
        "traffic_distribution": "Each SFC request traffic time series is sampled from a truncated normal distribution with higher variance than the previous deterministic/uniform setup.",
        "cost_policy": "All algorithms use the same post-hoc evaluation cost for thesis comparison: fixed VNF deployment cost + routed bandwidth cost + offloading cost. This avoids comparing solver-specific internal objectives directly.",
        "figure_types": "Line charts, grouped bar charts, reward/loss training curves, and auxiliary heatmaps are generated.",
        "algorithm_abbreviations": {key: value[0] for key, value in ALGORITHMS.items()},
        "figure_count": len(list(figure_dir.glob("*.png"))),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()