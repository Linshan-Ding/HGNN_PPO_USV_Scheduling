"""Registry for DRL comparison algorithms."""

from .a2c import A2CBaseline
from .ddqn import DDQNBaseline
from .dqn import DQNBaseline
from .reinforce import REINFORCEBaseline


ALGORITHMS = {
    A2CBaseline.algorithm_name: A2CBaseline,
    DDQNBaseline.algorithm_name: DDQNBaseline,
    DQNBaseline.algorithm_name: DQNBaseline,
    REINFORCEBaseline.algorithm_name: REINFORCEBaseline,
}


def list_algorithms(include_unimplemented: bool = False):
    """List registered algorithm names."""
    if include_unimplemented:
        return sorted(ALGORITHMS.keys())
    return sorted(
        name for name, cls in ALGORITHMS.items()
        if getattr(cls, "implemented", False)
    )


def get_algorithm(name: str, **kwargs):
    """Instantiate a registered DRL algorithm by name."""
    if name not in ALGORITHMS:
        valid = ", ".join(sorted(ALGORITHMS))
        raise KeyError(f"Unknown DRL algorithm '{name}'. Valid options: {valid}")
    return ALGORITHMS[name](**kwargs)
