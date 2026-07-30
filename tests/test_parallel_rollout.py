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

    def test_merge_rollout_results_counts_total_steps(self):
        from unittest.mock import MagicMock

        import numpy as np
        from parallel_rollout import merge_rollout_results

        def make_transition():
            return {
                'state': {},
                'action': (0, 0),
                'log_prob': 0.0,
                'reward': 0.0,
                'done': False,
                'value': 0.0,
                'task_mask': np.ones(2, dtype=bool),
                'usv_masks': np.ones((2, 2), dtype=bool),
            }

        agent = MagicMock()
        agent.device = 'cpu'
        results = [
            {'ep_reward': -1.0, 'success': True, 'makespan': 10.0,
             'transitions': [make_transition() for _ in range(3)]},
            {'ep_reward': -2.0, 'success': False, 'makespan': None,
             'transitions': [make_transition() for _ in range(5)]},
        ]
        merged = merge_rollout_results(agent, results)
        self.assertEqual(merged['total_steps'], 8)
        self.assertEqual(len(merged['epoch_makespans']), 1)
        self.assertEqual(agent.store_transition.call_count, 8)


if __name__ == "__main__":
    unittest.main()
