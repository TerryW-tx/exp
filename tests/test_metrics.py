from __future__ import annotations

import unittest

from vsfc_lab.mock_components import MockTopology, PlaceholderMetrics
from vsfc_lab.models import FIXED_VNF_INSTANCES, PlacementSolution, SFCRequest


class PlaceholderMetricsTest(unittest.TestCase):
    def test_network_kpis_are_computed_from_solution_paths(self) -> None:
        topology = MockTopology(
            node_ids=[1, 2],
            edge_list=[(1, 2)],
            node_capacities={1: 10.0, 2: 10.0},
            edge_capacities={(1, 2): 20.0},
            edge_delay={(1, 2): 5.0},
            cpu_price=1.0,
            bandwidth_price=1.0,
        )
        request = SFCRequest(
            request_id="r0",
            source=1,
            destination=2,
            vnf_chain=[FIXED_VNF_INSTANCES[0]],
            flow_size=10.0,
        )
        solution = PlacementSolution(
            request_id="r0",
            offloading_flow=2.0,
            vnf_to_node={"fw_0": 1},
            path_mapping={"segment_0": [1], "segment_1": [1, 2]},
            metadata={"offloading_unit_cost": 1.0},
        )

        metrics = PlaceholderMetrics().evaluate(topology, [request], [solution])

        self.assertEqual(metrics["throughput"], 8.0)
        self.assertEqual(metrics["packet_loss_rate"], 0.2)
        self.assertEqual(metrics["avg_end_to_end_delay_ms"], 5.0)
        self.assertEqual(metrics["avg_link_utilization"], 0.4)
        self.assertEqual(metrics["max_link_utilization"], 0.4)
        self.assertEqual(metrics["fairness_index"], 1.0)


if __name__ == "__main__":
    unittest.main()
