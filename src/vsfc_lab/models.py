from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class VNFInstance:
    """VNF instance data contract.

    Attributes:
        instance_id: Unique identifier of this VNF instance.
        cpu_demand: CPU resource required for deploying this VNF instance.
        bandwidth_demand: Bandwidth resource required by this VNF instance.
        flow_scaling_factor: Traffic scaling ratio after this VNF processes flow.
            Example: 0.5 means 10 Mb -> 5 Mb to the next VNF.
    """

    instance_id: str
    cpu_demand: float
    bandwidth_demand: float
    flow_scaling_factor: float


FIXED_VNF_INSTANCES: list[VNFInstance] = [
    VNFInstance(
        instance_id="fw_0",
        cpu_demand=2.0,
        bandwidth_demand=10.0,
        flow_scaling_factor=1.0,
    ),
    VNFInstance(
        instance_id="dpi_0",
        cpu_demand=4.0,
        bandwidth_demand=8.0,
        flow_scaling_factor=0.8,
    ),
    VNFInstance(
        instance_id="nat_0",
        cpu_demand=1.5,
        bandwidth_demand=10.0,
        flow_scaling_factor=1.0,
    ),
]


@dataclass(slots=True)
class SFCRequest:
    """SFC request data contract.

    You can extend this structure based on your model.

    Example
    -------
    {
      "request_id": "r_001",
      "source": "edge_1",
      "destination": "cloud_1",
    "vnf_chain": [
    {"instance_id": "fw_0", "cpu_demand": 2.0, "bandwidth_demand": 10.0, "flow_scaling_factor": 1.0},
    {"instance_id": "dpi_0", "cpu_demand": 4.0, "bandwidth_demand": 8.0, "flow_scaling_factor": 0.8}
    ],
    "flow_size": 10.0,
      "demand": 8.0,
      "latency_budget": 20.0,
      "metadata": {"slice": "uRLLC"}
    }
    """

    request_id: str
    source: int
    destination: int
    vnf_chain: list[VNFInstance]
    flow_size: float
    demand: float | None = None
    latency_budget: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fixed_ids = {vnf.instance_id for vnf in FIXED_VNF_INSTANCES}
        chain_ids = [vnf.instance_id for vnf in self.vnf_chain]
        invalid_ids = [vnf_id for vnf_id in chain_ids if vnf_id not in fixed_ids]
        if invalid_ids:
            raise ValueError(
                "All VNF instances in SFCRequest.vnf_chain must be selected from "
                f"FIXED_VNF_INSTANCES. Invalid instance_id(s): {invalid_ids}"
            )


@dataclass(slots=True)
class PlacementSolution:
    """Minimal placement result contract.

    Required fields are intentionally minimal:
    - `request_id`: Corresponding request identifier.
    - `vnf_to_node`: Mapping each VNF to a node ID.
    - `path_mapping`: Mapping for virtual links/path segments.
    - `offloading_flow`: Offloaded portion of request flow.
      The actually processed flow can be interpreted as `flow_size - offloading_flow`.

    You can add more fields later (objective, feasibility, runtime details, etc.).
    """

    request_id: str
    offloading_flow: float
    vnf_to_node: dict[str, int]
    path_mapping: dict[str, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)
