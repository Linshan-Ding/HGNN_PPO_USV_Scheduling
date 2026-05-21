"""Smoke tests for parallel rollout utilities."""

import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed")
class ParallelRolloutUtilityTest(unittest.TestCase):
    def test_resolve_rollout_workers(self):
        from parallel_rollout import resolve_rollout_workers

        with self.assertRaises(ValueError):
            resolve_rollout_workers(1, 0)
        self.assertEqual(resolve_rollout_workers(8, 4), 4)
        self.assertEqual(resolve_rollout_workers(4, 10), 4)
        self.assertGreaterEqual(resolve_rollout_workers(2, 0), 2)


if __name__ == "__main__":
    unittest.main()
