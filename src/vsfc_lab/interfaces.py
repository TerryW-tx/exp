from __future__ import annotations

from abc import ABC, abstractmethod

from .models import PlacementSolution, SFCRequest


class Topology(ABC):
    """Abstract topology interface.

    Implementer responsibility (TODO by researcher):
    - Define concrete topology representation and data loading.
    - Provide query methods used by solver and metrics.
    """

    @property
    @abstractmethod
    def cpu_price(self) -> float:
        """Unit CPU price coefficient."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bandwidth_price(self) -> float:
        """Unit bandwidth price coefficient."""
        raise NotImplementedError

    @property
    @abstractmethod
    def edge_delay(self) -> dict[tuple[int, int], float]:
        """Edge delay map in milliseconds, keyed by undirected edge tuple."""
        raise NotImplementedError

    @abstractmethod
    def nodes(self) -> list[int]:
        """Return all node identifiers.

        Returns:
            list[int]: Node IDs in topology.
        """
        raise NotImplementedError

    @abstractmethod
    def links(self) -> list[tuple[int, int]]:
        """Return all directed/undirected links as endpoint tuples.

        Returns:
            list[tuple[int, int]]: Link list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_node_capacity(self, node_id: int) -> float:
        """Query capacity of a given node.

        Args:
            node_id: Node identifier.

        Returns:
            float: Available/total capacity value (your convention).
        """
        raise NotImplementedError

    @abstractmethod
    def get_edge_capacity(self, edge: tuple[int, int]) -> float:
        """Query capacity of a given edge.

        Args:
            edge: Edge represented as a tuple of node identifiers.

        Returns:
            float: Available/total capacity value (your convention).
        """
        raise NotImplementedError

    @abstractmethod
    def get_edge_delay(self, edge: tuple[int, int]) -> float:
        """Query propagation/transmission delay of a given edge in ms.

        Args:
            edge: Edge represented as a tuple of node identifiers.

        Returns:
            float: Edge delay in milliseconds.
        """
        raise NotImplementedError


class Solver(ABC):
    """Abstract solver interface for vSFC placement/mapping."""

    @abstractmethod
    def solve(
        self,
        topology: Topology,
        requests: list[SFCRequest],
        **kwargs,
    ) -> list[PlacementSolution]:
        """Generate placement solutions for given requests.

        Args:
            topology: Concrete topology object.
            requests: List of SFC requests.
            **kwargs: Extra algorithm-specific runtime options.

        Returns:
            list[PlacementSolution]: One or multiple solutions.

        Note:
            Core algorithm should be implemented by researcher.
        """
        raise NotImplementedError


class Metrics(ABC):
    """Abstract metrics interface for experiment evaluation."""

    @abstractmethod
    def evaluate(
        self,
        topology: Topology,
        requests: list[SFCRequest],
        solutions: list[PlacementSolution],
    ) -> dict[str, float]:
        """Evaluate one run and return scalar metrics.

        Args:
            topology: Topology used in this run.
            requests: Input SFC requests.
            solutions: Output solutions from solver.

        Returns:
            dict[str, float]: Metric name to value mapping.

        Note:
            Core metric definitions/calculations should be implemented by researcher.
        """
        raise NotImplementedError
