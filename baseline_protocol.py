"""
Shared protocol objects for comparison algorithms.

The baseline folders use this module to keep experiment outputs consistent
without coupling unfinished comparison algorithms into the main PPO workflow.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class AlgorithmResult:
    """Unified result record for DRL and metaheuristic comparison algorithms."""

    algorithm_name: str
    category: str
    instance_id: str
    n_usvs: int
    n_tasks: int
    makespan: float
    success: bool
    runtime_sec: float
    seed: int

    def to_dict(self) -> Dict[str, object]:
        """Convert result to a CSV-friendly dictionary."""
        return asdict(self)

    @classmethod
    def not_implemented(cls, algorithm_name: str, category: str,
                        instance: dict, seed: int = 0) -> "AlgorithmResult":
        """Create a placeholder result for algorithms that are registered but unfinished."""
        return cls(
            algorithm_name=algorithm_name,
            category=category,
            instance_id=str(instance.get("instance_id", "unknown")),
            n_usvs=int(instance.get("n_usvs", 0)),
            n_tasks=int(instance.get("n_tasks", 0)),
            makespan=float("inf"),
            success=False,
            runtime_sec=0.0,
            seed=int(seed),
        )
