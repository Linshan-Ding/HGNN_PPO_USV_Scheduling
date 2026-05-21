"""Smoke tests for comparison algorithm registries and result protocol."""

import unittest
import importlib.util
import tempfile

from baseline_protocol import AlgorithmResult
from drl_baselines import registry as drl_registry
from metaheuristic_baselines import registry as meta_registry


EXPECTED_RESULT_FIELDS = {
    "algorithm_name",
    "category",
    "instance_id",
    "n_usvs",
    "n_tasks",
    "makespan",
    "success",
    "runtime_sec",
    "seed",
}


class BaselineInterfaceSmokeTest(unittest.TestCase):
    def setUp(self):
        self.instance = {
            "instance_id": "smoke",
            "n_usvs": 2,
            "n_tasks": 5,
        }

    def assert_result_shape(self, result: AlgorithmResult):
        self.assertIsInstance(result, AlgorithmResult)
        self.assertEqual(set(result.to_dict().keys()), EXPECTED_RESULT_FIELDS)

    def test_drl_registry_and_result_shape(self):
        self.assertEqual(
            drl_registry.list_algorithms(include_unimplemented=True),
            ["A2C", "DDQN", "DQN", "REINFORCE"],
        )
        self.assertEqual(
            drl_registry.list_algorithms(),
            ["A2C", "DDQN", "DQN", "REINFORCE"],
        )

        for name in drl_registry.list_algorithms(include_unimplemented=True):
            algorithm = drl_registry.get_algorithm(name, seed=7)
            self.assertTrue(algorithm.implemented)

    def test_metaheuristic_registry_and_result_shape(self):
        self.assertEqual(
            meta_registry.list_algorithms(include_unimplemented=True),
            ["ACO", "GA", "PSO", "SA"],
        )
        self.assertEqual(meta_registry.list_algorithms(), [])

        for name in meta_registry.list_algorithms(include_unimplemented=True):
            algorithm = meta_registry.get_algorithm(name, seed=11)
            self.assertFalse(algorithm.implemented)
            self.assert_result_shape(algorithm.solve(self.instance))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed")
    def test_drl_minimal_training_with_torch(self):
        from config import get_config
        from utils import load_instance_from_config

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = get_config(
                n_usvs=2,
                n_tasks=20,
                data_dir="data/public",
                max_epochs=1,
                n_trajectories=1,
                eval_interval=1,
                hidden_dim=16,
                hgnn_layers=1,
                n_heads=4,
                dropout=0.0,
                use_visdom=False,
                model_dir=tmpdir,
                result_dir=tmpdir,
            )
            instance = load_instance_from_config(cfg)
            for name in drl_registry.list_algorithms():
                algorithm = drl_registry.get_algorithm(name, seed=3)
                result = algorithm.train(instance, cfg)
                self.assert_result_shape(result)
                self.assertEqual(result.algorithm_name, name)
                self.assertEqual(result.category, "drl")


if __name__ == "__main__":
    unittest.main()
