"""Registry for metaheuristic comparison algorithms."""

from .aco import ACOBaseline
from .ga import GABaseline
from .pso import PSOBaseline
from .sa import SABaseline


ALGORITHMS = {
    ACOBaseline.algorithm_name: ACOBaseline,
    GABaseline.algorithm_name: GABaseline,
    PSOBaseline.algorithm_name: PSOBaseline,
    SABaseline.algorithm_name: SABaseline,
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
    """Instantiate a registered metaheuristic algorithm by name."""
    if name not in ALGORITHMS:
        valid = ", ".join(sorted(ALGORITHMS))
        raise KeyError(
            f"Unknown metaheuristic algorithm '{name}'. Valid options: {valid}"
        )
    return ALGORITHMS[name](**kwargs)
