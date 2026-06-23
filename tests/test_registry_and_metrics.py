from __future__ import annotations

import unittest

from vsfc_lab.mock_components import MockTopology, PlaceholderMetrics, PlaceholderSolver
from vsfc_lab.models import FIXED_VNF_INSTANCES, PlacementSolution, SFCRequest
from vsfc_lab.registry import create_metrics, create_solver


class RegistryTest(unittest.TestCase):
    def test_create_registered_baseline_components(self) -> None:
        self.assertIsInstance(create_solver("milp_baseline"), PlaceholderSolver)
        self.assertIsInstance(create_solver("placeholder_solver"), PlaceholderSolver)
        self.assertIsInstance(create_metrics("network_metrics"), PlaceholderMetrics)
        self.assertIsInstance(create_metrics("placeholder_metrics"), PlaceholderMetrics)

    def test_unknown_solver_lists_available_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Available solvers"):
            create_solver("missing_solver")


class NetworkMetricsTest(unittest.TestCase):
    def test_network_metrics_are_computed_from_solution_paths(self) -> None:
        topology = MockTopology(
            node_ids=[0, 1],
            edge_list=[(0, 1)],
            node_capacities={0: 100.0, 1: 100.0},
            edge_capacities={(0, 1): 20.0},
            edge_delay={(0, 1): 5.0},
            cpu_price=1.0,
            bandwidth_price=1.0,
        )
        request = SFCRequest(
            request_id="req_0",
            source=0,
            destination=1,
            vnf_chain=[FIXED_VNF_INSTANCES[0]],
            flow_size=10.0,
        )
        solution = PlacementSolution(
            request_id="req_0",
            offloading_flow=2.0,
            vnf_to_node={"fw_0": 1},
            path_mapping={"segment_0": [0, 1], "segment_1": [1]},
        )

        metrics = create_metrics("network_metrics").evaluate(topology, [request], [solution])

        self.assertEqual(metrics["throughput"], 8.0)
        self.assertEqual(metrics["offloading_ratio"], 0.2)
        self.assertEqual(metrics["avg_end_to_end_delay_ms"], 5.0)
        self.assertEqual(metrics["avg_link_utilization"], 0.4)
        self.assertEqual(metrics["max_link_utilization"], 0.4)
        self.assertEqual(metrics["jain_fairness"], 1.0)


if __name__ == "__main__":
    unittest.main()
