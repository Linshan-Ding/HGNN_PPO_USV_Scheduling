"""Base interface for deep reinforcement learning comparison algorithms."""

from baseline_protocol import AlgorithmResult


class DRLBaseline:
    """Common interface for DRL baselines."""

    algorithm_name = "DRLBase"
    category = "drl"
    implemented = False

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def train(self, instance: dict, cfg=None) -> AlgorithmResult:
        """Train the DRL baseline on one instance."""
        return AlgorithmResult.not_implemented(
            self.algorithm_name, self.category, instance, self.seed
        )

    def evaluate(self, instance: dict, cfg=None) -> AlgorithmResult:
        """Evaluate the DRL baseline on one instance."""
        return AlgorithmResult.not_implemented(
            self.algorithm_name, self.category, instance, self.seed
        )

    def save(self, path: str):
        """Save model state."""
        raise NotImplementedError(f"{self.algorithm_name} save() is not implemented.")

    def load(self, path: str):
        """Load model state."""
        raise NotImplementedError(f"{self.algorithm_name} load() is not implemented.")
