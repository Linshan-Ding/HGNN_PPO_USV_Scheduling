"""Base interface for metaheuristic comparison algorithms."""

from baseline_protocol import AlgorithmResult


class MetaheuristicBaseline:
    """Common interface for metaheuristic baselines."""

    algorithm_name = "MetaheuristicBase"
    category = "metaheuristic"
    implemented = False

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def solve(self, instance: dict, cfg=None) -> AlgorithmResult:
        """Solve one scheduling instance."""
        return AlgorithmResult.not_implemented(
            self.algorithm_name, self.category, instance, self.seed
        )
