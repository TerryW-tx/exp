from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pulp
from geopy.distance import geodesic

from .interfaces import Metrics, Solver, Topology
from .models import FIXED_VNF_INSTANCES, PlacementSolution, SFCRequest, VNFInstance


@dataclass(slots=True)
class MockTopology(Topology):
    """Minimal runnable topology stub.

    Replace with your own topology class/loader.
    """

    node_ids: list[int]
    edge_list: list[tuple[int, int]] = field(default_factory=list)
    node_capacities: dict[int, float] = field(default_factory=dict)
    edge_capacities: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_delay: dict[tuple[int, int], float] = field(default_factory=dict)
    cpu_price: float = 1.0
    bandwidth_price: float = 1.0

    def nodes(self) -> list[int]:
        return self.node_ids

    def links(self) -> list[tuple[int, int]]:
        return self.edge_list

    def get_node_capacity(self, node_id: int) -> float:
        return self.node_capacities.get(node_id, 0.0)

    def get_edge_capacity(self, edge: tuple[int, int]) -> float:
        canonical_edge = edge if edge[0] <= edge[1] else (edge[1], edge[0])
        return self.edge_capacities.get(canonical_edge, 0.0)

    def get_edge_delay(self, edge: tuple[int, int]) -> float:
        canonical_edge = edge if edge[0] <= edge[1] else (edge[1], edge[0])
        return self.edge_delay.get(canonical_edge, 0.0)


def build_mock_topology(
    seed: int = 0,
    max_nodes: int = 9,
    csv_path: Path | None = None,
) -> MockTopology:
    """Build a topology from site CSV with random node sampling.

    Args:
        seed: Random seed for reproducibility.
        max_nodes: Maximum node count (capped to 10).
        csv_path: Path to site CSV file. Defaults to project root file.

    Returns:
        MockTopology: Randomly generated topology from sampled CSV rows.
    """
    rng = random.Random(seed)
    upper_bound = min(max_nodes, 10)
    if upper_bound < 2:
        raise ValueError("max_nodes must be >= 2")

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[2] / "site-optus-melbCBD.csv"

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lat = row.get("LATITUDE")
            lon = row.get("LONGITUDE")
            site_id = row.get("SITE_ID")
            if not lat or not lon or not site_id:
                continue
            try:
                float(lat)
                float(lon)
                int(float(site_id))
            except ValueError:
                continue
            rows.append(row)

    if len(rows) < 2:
        raise ValueError(f"CSV file does not contain enough valid site rows: {csv_path}")

    node_count = rng.randint(2, min(upper_bound, len(rows)))
    sampled_rows = rng.sample(rows, node_count)

    node_ids = [int(float(item["SITE_ID"])) for item in sampled_rows]
    node_positions = {
        int(float(item["SITE_ID"])): (float(item["LATITUDE"]), float(item["LONGITUDE"]))
        for item in sampled_rows
    }

    edges: set[tuple[int, int]] = set()
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            u = node_ids[i]
            v = node_ids[j]
            edges.add((u, v) if u <= v else (v, u))

    node_capacities = {node_id: round(rng.uniform(10.0, 100.0), 2) for node_id in node_ids}
    edge_capacities = {edge: round(rng.uniform(5.0, 50.0), 2) for edge in edges}
    edge_delay = {}
    for u, v in edges:
        distance_km = geodesic(node_positions[u], node_positions[v]).km
        delay_ms = round(0.2 + 0.02 * distance_km, 4)
        edge_delay[(u, v)] = delay_ms

    cpu_price = round(rng.uniform(1.0, 5.0), 2)
    bandwidth_price = round(rng.uniform(2.0, 10.0), 2)
    return MockTopology(
        node_ids=node_ids,
        edge_list=sorted(edges),
        node_capacities=node_capacities,
        edge_capacities=edge_capacities,
        edge_delay=edge_delay,
        cpu_price=cpu_price,
        bandwidth_price=bandwidth_price,
    )


class PlaceholderSolver(Solver):
    """MILP-based global solver baseline for vSFC placement/mapping."""

    def segment_flow_coefficients(self, request: SFCRequest) -> list[float]:
        coeffs = [1.0]
        acc = 1.0
        for vnf in request.vnf_chain:
            acc *= vnf.flow_scaling_factor
            coeffs.append(acc)
        return coeffs

    def _segment_flow_coefficients(self, request: SFCRequest) -> list[float]:
        return self.segment_flow_coefficients(request)

    def _build_total_objective(
        self,
        topology: Topology,
        requests: list[SFCRequest],
        nodes: list[int],
        arcs: list[tuple[int, int]],
        x: dict[tuple[int, int, int], pulp.LpVariable],
        q: dict[tuple[int, int, int, int], pulp.LpVariable],
        offloading_flow: dict[int, pulp.LpVariable],
        offloading_unit_cost: float,
    ) -> pulp.LpAffineExpression:
        cpu_cost_term = pulp.lpSum(
            topology.cpu_price * request.vnf_chain[v_idx].cpu_demand * x[(r_idx, v_idx, node_id)]
            for r_idx, request in enumerate(requests)
            for v_idx in range(len(request.vnf_chain))
            for node_id in nodes
        )
        bandwidth_cost_term = pulp.lpSum(
            topology.bandwidth_price * q[(r_idx, seg_idx, u, v)]
            for r_idx, request in enumerate(requests)
            for seg_idx in range(len(request.vnf_chain) + 1)
            for (u, v) in arcs
        )
        offloading_cost_term = pulp.lpSum(
            offloading_unit_cost * offloading_flow[r_idx] for r_idx, _ in enumerate(requests)
        )
        return cpu_cost_term + bandwidth_cost_term + offloading_cost_term

    @staticmethod
    def _extract_path_nodes(
        selected_arcs: set[tuple[int, int]],
        start_node: int,
        end_node: int,
    ) -> list[int]:
        if start_node == end_node:
            return [start_node]
        adjacency: dict[int, list[int]] = defaultdict(list)
        for u, v in selected_arcs:
            adjacency[u].append(v)

        path = [start_node]
        current = start_node
        visited = {start_node}
        max_steps = len(selected_arcs) + 2
        for _ in range(max_steps):
            if current == end_node:
                return path
            candidates = [nxt for nxt in adjacency.get(current, []) if nxt not in visited]
            if not candidates:
                break
            nxt = candidates[0]
            path.append(nxt)
            visited.add(nxt)
            current = nxt
        if path[-1] != end_node:
            path.append(end_node)
        return path

    def solve(
        self,
        topology: Topology,
        requests: list[SFCRequest],
        **kwargs,
    ) -> list[PlacementSolution]:
        if not requests:
            return []

        nodes = topology.nodes()
        undirected_edges = topology.links()
        arcs: list[tuple[int, int]] = []
        for u, v in undirected_edges:
            arcs.append((u, v))
            arcs.append((v, u))

        offloading_cost_factor = float(kwargs.get("offloading_cost_factor", 100.0))
        offloading_unit_cost = float(
            kwargs.get("offloading_unit_cost", topology.bandwidth_price * offloading_cost_factor)
        )

        model = pulp.LpProblem("vsfc_milp_global", pulp.LpMinimize)

        offloading_flow: dict[int, pulp.LpVariable] = {}
        x: dict[tuple[int, int, int], pulp.LpVariable] = {}
        y: dict[tuple[int, int, int, int], pulp.LpVariable] = {}
        q: dict[tuple[int, int, int, int], pulp.LpVariable] = {}

        for r_idx, request in enumerate(requests):
            offloading_flow[r_idx] = pulp.LpVariable(
                f"offloading_flow_r{r_idx}",
                lowBound=0.0,
                upBound=float(request.flow_size),
                cat=pulp.LpContinuous,
            )
            for v_idx in range(len(request.vnf_chain)):
                for node_id in nodes:
                    x[(r_idx, v_idx, node_id)] = pulp.LpVariable(
                        f"x_r{r_idx}_v{v_idx}_n{node_id}", cat=pulp.LpBinary
                    )
            for seg_idx in range(len(request.vnf_chain) + 1):
                for (u, v) in arcs:
                    y[(r_idx, seg_idx, u, v)] = pulp.LpVariable(
                        f"y_r{r_idx}_s{seg_idx}_a{u}_{v}", cat=pulp.LpBinary
                    )
                    q[(r_idx, seg_idx, u, v)] = pulp.LpVariable(
                        f"q_r{r_idx}_s{seg_idx}_a{u}_{v}", lowBound=0.0, cat=pulp.LpContinuous
                    )

        model += self._build_total_objective(
            topology=topology,
            requests=requests,
            nodes=nodes,
            arcs=arcs,
            x=x,
            q=q,
            offloading_flow=offloading_flow,
            offloading_unit_cost=offloading_unit_cost,
        )

        for r_idx, request in enumerate(requests):
            for v_idx in range(len(request.vnf_chain)):
                model += (
                    pulp.lpSum(x[(r_idx, v_idx, node_id)] for node_id in nodes) == 1,
                    f"vnf_assign_r{r_idx}_v{v_idx}",
                )

        for node_id in nodes:
            model += (
                pulp.lpSum(
                    request.vnf_chain[v_idx].cpu_demand * x[(r_idx, v_idx, node_id)]
                    for r_idx, request in enumerate(requests)
                    for v_idx in range(len(request.vnf_chain))
                )
                <= topology.get_node_capacity(node_id),
                f"cpu_cap_node_{node_id}",
            )

        for r_idx, request in enumerate(requests):
            coeffs = self.segment_flow_coefficients(request)
            for seg_idx in range(len(request.vnf_chain) + 1):
                flow_expr = coeffs[seg_idx] * (request.flow_size - offloading_flow[r_idx])
                max_flow = coeffs[seg_idx] * request.flow_size

                for (u, v) in arcs:
                    model += (
                        q[(r_idx, seg_idx, u, v)] <= max_flow * y[(r_idx, seg_idx, u, v)],
                        f"lin1_r{r_idx}_s{seg_idx}_a{u}_{v}",
                    )
                    model += (
                        q[(r_idx, seg_idx, u, v)] <= flow_expr,
                        f"lin2_r{r_idx}_s{seg_idx}_a{u}_{v}",
                    )
                    model += (
                        q[(r_idx, seg_idx, u, v)] >= flow_expr - max_flow * (1 - y[(r_idx, seg_idx, u, v)]),
                        f"lin3_r{r_idx}_s{seg_idx}_a{u}_{v}",
                    )

            for seg_idx in range(len(request.vnf_chain) + 1):
                for node_id in nodes:
                    outgoing = pulp.lpSum(
                        y[(r_idx, seg_idx, u, v)] for (u, v) in arcs if u == node_id
                    )
                    incoming = pulp.lpSum(
                        y[(r_idx, seg_idx, u, v)] for (u, v) in arcs if v == node_id
                    )

                    if seg_idx == 0:
                        rhs = (1 if node_id == request.source else 0) - x[(r_idx, 0, node_id)]
                    elif seg_idx == len(request.vnf_chain):
                        rhs = x[(r_idx, len(request.vnf_chain) - 1, node_id)] - (
                            1 if node_id == request.destination else 0
                        )
                    else:
                        rhs = x[(r_idx, seg_idx - 1, node_id)] - x[(r_idx, seg_idx, node_id)]

                    model += (
                        outgoing - incoming == rhs,
                        f"path_flow_r{r_idx}_s{seg_idx}_n{node_id}",
                    )

        for u, v in undirected_edges:
            model += (
                pulp.lpSum(
                    q[(r_idx, seg_idx, u, v)] + q[(r_idx, seg_idx, v, u)]
                    for r_idx, request in enumerate(requests)
                    for seg_idx in range(len(request.vnf_chain) + 1)
                )
                <= topology.get_edge_capacity((u, v)),
                f"bw_cap_edge_{u}_{v}",
            )

        for r_idx, request in enumerate(requests):
            if request.latency_budget is None:
                continue
            model += (
                pulp.lpSum(
                    topology.get_edge_delay((u, v)) * y[(r_idx, seg_idx, u, v)]
                    for seg_idx in range(len(request.vnf_chain) + 1)
                    for (u, v) in arcs
                )
                <= float(request.latency_budget),
                f"latency_budget_r{r_idx}",
            )

        time_limit = kwargs.get("time_limit")
        msg = bool(kwargs.get("solver_msg", False))
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        model.solve(solver)

        if pulp.LpStatus[model.status] not in {"Optimal", "Feasible"}:
            raise RuntimeError(f"MILP solver failed with status: {pulp.LpStatus[model.status]}")

        solutions: list[PlacementSolution] = []
        for r_idx, request in enumerate(requests):
            vnf_to_node: dict[str, int] = {}
            for v_idx, vnf in enumerate(request.vnf_chain):
                selected_node = next(
                    node_id
                    for node_id in nodes
                    if pulp.value(x[(r_idx, v_idx, node_id)]) is not None
                    and pulp.value(x[(r_idx, v_idx, node_id)]) > 0.5
                )
                vnf_to_node[vnf.instance_id] = selected_node

            path_mapping: dict[str, list[int]] = {}
            for seg_idx in range(len(request.vnf_chain) + 1):
                if seg_idx == 0:
                    start_node = request.source
                    end_node = vnf_to_node[request.vnf_chain[0].instance_id]
                elif seg_idx == len(request.vnf_chain):
                    start_node = vnf_to_node[request.vnf_chain[-1].instance_id]
                    end_node = request.destination
                else:
                    start_node = vnf_to_node[request.vnf_chain[seg_idx - 1].instance_id]
                    end_node = vnf_to_node[request.vnf_chain[seg_idx].instance_id]

                selected_arcs = {
                    (u, v)
                    for (u, v) in arcs
                    if pulp.value(y[(r_idx, seg_idx, u, v)]) is not None
                    and pulp.value(y[(r_idx, seg_idx, u, v)]) > 0.5
                }
                path_mapping[f"segment_{seg_idx}"] = self._extract_path_nodes(
                    selected_arcs=selected_arcs,
                    start_node=start_node,
                    end_node=end_node,
                )

            solutions.append(
                PlacementSolution(
                    request_id=request.request_id,
                    offloading_flow=float(pulp.value(offloading_flow[r_idx]) or 0.0),
                    vnf_to_node=vnf_to_node,
                    path_mapping=path_mapping,
                    metadata={
                        "solver_status": pulp.LpStatus[model.status],
                        "objective": float(pulp.value(model.objective) or 0.0),
                        "offloading_unit_cost": offloading_unit_cost,
                    },
                )
            )
        return solutions


class PlaceholderMetrics(Metrics):
    """Metrics computed from solver objective reconstruction and network KPIs."""

    @staticmethod
    def _canonical_edge(edge: tuple[int, int]) -> tuple[int, int]:
        return edge if edge[0] <= edge[1] else (edge[1], edge[0])

    @staticmethod
    def _jain_fairness(values: list[float]) -> float:
        if not values:
            return 0.0
        squared_sum = sum(values) ** 2
        sum_of_squares = sum(value**2 for value in values)
        if sum_of_squares == 0:
            return 1.0
        return squared_sum / (len(values) * sum_of_squares)

    def evaluate(
        self,
        topology: Topology,
        requests: list[SFCRequest],
        solutions: list[PlacementSolution],
    ) -> dict[str, float]:
        if not requests:
            return {"objective_total": 0.0}

        solution_by_req = {solution.request_id: solution for solution in solutions}
        missing = [request.request_id for request in requests if request.request_id not in solution_by_req]
        if missing:
            raise ValueError(f"Missing PlacementSolution for request_id(s): {missing}")

        nodes = topology.nodes()
        undirected_edges = topology.links()
        arcs: list[tuple[int, int]] = []
        for u, v in undirected_edges:
            arcs.append((u, v))
            arcs.append((v, u))

        solver = PlaceholderSolver()

        offloading_unit_cost = float(
            solutions[0].metadata.get("offloading_unit_cost", topology.bandwidth_price * 100.0)
        )

        offloading_flow: dict[int, pulp.LpVariable] = {}
        x: dict[tuple[int, int, int], pulp.LpVariable] = {}
        q: dict[tuple[int, int, int, int], pulp.LpVariable] = {}

        for r_idx, request in enumerate(requests):
            solution = solution_by_req[request.request_id]

            offloading_flow[r_idx] = pulp.LpVariable(
                f"metric_offloading_flow_r{r_idx}",
                lowBound=0.0,
                upBound=float(request.flow_size),
                cat=pulp.LpContinuous,
            )
            offloading_flow[r_idx].varValue = float(solution.offloading_flow)

            for v_idx, vnf in enumerate(request.vnf_chain):
                assigned_node = solution.vnf_to_node[vnf.instance_id]
                for node_id in nodes:
                    x[(r_idx, v_idx, node_id)] = pulp.LpVariable(
                        f"metric_x_r{r_idx}_v{v_idx}_n{node_id}", cat=pulp.LpBinary
                    )
                    x[(r_idx, v_idx, node_id)].varValue = 1.0 if node_id == assigned_node else 0.0

            coeffs = solver.segment_flow_coefficients(request)
            for seg_idx in range(len(request.vnf_chain) + 1):
                segment_flow = coeffs[seg_idx] * (request.flow_size - solution.offloading_flow)
                path_nodes = solution.path_mapping.get(f"segment_{seg_idx}", [])
                selected_arcs = set(zip(path_nodes[:-1], path_nodes[1:]))

                for u, v in arcs:
                    q[(r_idx, seg_idx, u, v)] = pulp.LpVariable(
                        f"metric_q_r{r_idx}_s{seg_idx}_a{u}_{v}", lowBound=0.0, cat=pulp.LpContinuous
                    )
                    q[(r_idx, seg_idx, u, v)].varValue = (
                        float(segment_flow) if (u, v) in selected_arcs else 0.0
                    )

        objective_expr = solver._build_total_objective(
            topology=topology,
            requests=requests,
            nodes=nodes,
            arcs=arcs,
            x=x,
            q=q,
            offloading_flow=offloading_flow,
            offloading_unit_cost=offloading_unit_cost,
        )

        objective_total = float(pulp.value(objective_expr) or 0.0)
        avg_offloading_flow = float(sum(solution.offloading_flow for solution in solutions) / len(solutions))
        total_input_flow = sum(request.flow_size for request in requests)
        processed_flows = [
            max(0.0, request.flow_size - solution_by_req[request.request_id].offloading_flow)
            for request in requests
        ]
        throughput = sum(processed_flows)
        packet_loss_rate = (
            max(0.0, total_input_flow - throughput) / total_input_flow if total_input_flow > 0 else 0.0
        )

        edge_loads = {edge: 0.0 for edge in undirected_edges}
        request_delays: list[float] = []
        for request, processed_flow in zip(requests, processed_flows, strict=True):
            solution = solution_by_req[request.request_id]
            coeffs = solver.segment_flow_coefficients(request)
            # Current request/VNF data models do not include processing delay.
            request_delay = 0.0
            for seg_idx in range(len(request.vnf_chain) + 1):
                segment_flow = coeffs[seg_idx] * processed_flow
                path_nodes = solution.path_mapping.get(f"segment_{seg_idx}", [])
                for edge in zip(path_nodes[:-1], path_nodes[1:], strict=True):
                    canonical_edge = self._canonical_edge(edge)
                    edge_loads[canonical_edge] = edge_loads.get(canonical_edge, 0.0) + segment_flow
                    request_delay += topology.get_edge_delay(canonical_edge)
            request_delays.append(request_delay)

        link_utilizations = [
            load / capacity
            for edge, load in edge_loads.items()
            if (capacity := topology.get_edge_capacity(edge)) > 0
        ]
        avg_link_utilization = (
            sum(link_utilizations) / len(link_utilizations) if link_utilizations else 0.0
        )
        max_link_utilization = max(link_utilizations, default=0.0)
        avg_end_to_end_delay_ms = (
            sum(request_delays) / len(request_delays) if request_delays else 0.0
        )
        fairness_index = self._jain_fairness(processed_flows)

        return {
            "objective_total": objective_total,
            "avg_offloading_flow": avg_offloading_flow,
            "n_solutions": float(len(solutions)),
            "throughput": throughput,
            "avg_end_to_end_delay_ms": avg_end_to_end_delay_ms,
            "packet_loss_rate": packet_loss_rate,
            "avg_link_utilization": avg_link_utilization,
            "max_link_utilization": max_link_utilization,
            "fairness_index": fairness_index,
        }


def build_mock_requests(node_ids: list[int] | None = None) -> list[SFCRequest]:
    """Return small dummy requests for pipeline smoke test.

    TODO: replace with your real request generation/loading logic.
    """
    if node_ids is None or len(node_ids) < 2:
        source, destination = 0, 1
    else:
        source, destination = node_ids[0], node_ids[1]

    return [
        SFCRequest(
            request_id="req_0",
            source=source,
            destination=destination,
            vnf_chain=[FIXED_VNF_INSTANCES[0], FIXED_VNF_INSTANCES[1]],
            flow_size=10.0,
            demand=1.0,
            latency_budget=10.0,
        )
    ]


def _vnf_lookup() -> dict[str, VNFInstance]:
    return {instance.instance_id: instance for instance in FIXED_VNF_INSTANCES}


def serialize_topology(topology: MockTopology) -> dict[str, object]:
    return {
        "node_ids": topology.node_ids,
        "edge_list": [list(edge) for edge in topology.edge_list],
        "node_capacities": {str(k): v for k, v in topology.node_capacities.items()},
        "edge_capacities": {f"{u},{v}": cap for (u, v), cap in topology.edge_capacities.items()},
        "edge_delay": {f"{u},{v}": delay for (u, v), delay in topology.edge_delay.items()},
        "cpu_price": topology.cpu_price,
        "bandwidth_price": topology.bandwidth_price,
    }


def serialize_requests(requests: list[SFCRequest]) -> list[dict[str, object]]:
    return [
        {
            "request_id": request.request_id,
            "source": request.source,
            "destination": request.destination,
            "vnf_chain": [vnf.instance_id for vnf in request.vnf_chain],
            "flow_size": request.flow_size,
            "demand": request.demand,
            "latency_budget": request.latency_budget,
            "metadata": request.metadata,
        }
        for request in requests
    ]


def save_scenario_config(path: Path, topology: MockTopology, requests: list[SFCRequest]) -> None:
    payload = {
        "format": "vsfc_scenario_v1",
        "topology": serialize_topology(topology),
        "requests": serialize_requests(requests),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_scenario_config(path: Path) -> tuple[MockTopology, list[SFCRequest]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    topology_data = payload["topology"]

    edge_list = [tuple(int(x) for x in edge) for edge in topology_data["edge_list"]]
    node_capacities = {int(k): float(v) for k, v in topology_data["node_capacities"].items()}

    edge_capacities: dict[tuple[int, int], float] = {}
    for key, value in topology_data["edge_capacities"].items():
        u_str, v_str = key.split(",")
        edge_capacities[(int(u_str), int(v_str))] = float(value)

    edge_delay: dict[tuple[int, int], float] = {}
    for key, value in topology_data["edge_delay"].items():
        u_str, v_str = key.split(",")
        edge_delay[(int(u_str), int(v_str))] = float(value)

    topology = MockTopology(
        node_ids=[int(node_id) for node_id in topology_data["node_ids"]],
        edge_list=edge_list,
        node_capacities=node_capacities,
        edge_capacities=edge_capacities,
        edge_delay=edge_delay,
        cpu_price=float(topology_data["cpu_price"]),
        bandwidth_price=float(topology_data["bandwidth_price"]),
    )

    vnf_map = _vnf_lookup()
    requests: list[SFCRequest] = []
    for request_data in payload["requests"]:
        chain_ids = request_data["vnf_chain"]
        vnf_chain = [vnf_map[str(vnf_id)] for vnf_id in chain_ids]
        requests.append(
            SFCRequest(
                request_id=str(request_data["request_id"]),
                source=int(request_data["source"]),
                destination=int(request_data["destination"]),
                vnf_chain=vnf_chain,
                flow_size=float(request_data["flow_size"]),
                demand=(
                    None
                    if request_data.get("demand") is None
                    else float(request_data["demand"])
                ),
                latency_budget=(
                    None
                    if request_data.get("latency_budget") is None
                    else float(request_data["latency_budget"])
                ),
                metadata=dict(request_data.get("metadata", {})),
            )
        )

    return topology, requests
