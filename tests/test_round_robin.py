"""In-process smoke test for the round-robin multi-instance trainer."""

import csv
import importlib.util
import os
import tempfile
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed")
class RoundRobinTrainerSmokeTest(unittest.TestCase):
    def test_round_robin_cycles_instances_and_saves_per_instance_bests(self):
        from config import get_config
        from multi_train import (ensure_rules_results, load_public_instances,
                                 train_round_robin)

        instances = load_public_instances(
            os.path.join('data', 'public'), ['u2_t20', 'u2_t40'])
        self.assertEqual([i['instance_id'] for i in instances],
                         ['u2_t20', 'u2_t40'])

        with tempfile.TemporaryDirectory() as tmpdir:
            rules = ensure_rules_results(
                instances, path=os.path.join(tmpdir, 'rules_results.csv'))
            self.assertIn('u2_t20', rules)
            self.assertEqual(len(rules['u2_t20']['rules']), 5)
            # Cache round-trip
            cached = ensure_rules_results(
                instances, path=os.path.join(tmpdir, 'rules_results.csv'))
            self.assertEqual(
                cached['u2_t40']['best_rule_name'],
                rules['u2_t40']['best_rule_name'])

            cfg = get_config(
                n_usvs=2, n_tasks=20,
                max_epochs=4, seed=0,
                hidden_dim=16, hgnn_layers=1, n_heads=2,
                n_trajectories=2,
                use_visdom=False,
                model_dir=os.path.join(tmpdir, 'models'),
                result_dir=os.path.join(tmpdir, 'results'),
                training_log_dir=os.path.join(tmpdir, 'logs'),
                rollout_torch_threads=1,
            )
            info = train_round_robin(cfg, instances, rules)

            # 4 epochs over 2 instances -> visits 1,1,2,2
            with open(info['training_log_path'], newline='',
                      encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual([r['instance_id'] for r in rows],
                             ['u2_t20', 'u2_t40', 'u2_t20', 'u2_t40'])
            self.assertEqual([r['visit_index'] for r in rows],
                             ['1', '1', '2', '2'])
            for row in rows:
                self.assertEqual(row['protocol'], 'round_robin')
                self.assertNotEqual(row['eval_solve_time_sec'], '')
                self.assertNotEqual(row['steps_collected'], '')

            # Per-instance best isolation: each instance gets its own checkpoint
            for instance_id, path in info['best_paths'].items():
                self.assertIn(instance_id, path)
                self.assertTrue(os.path.isfile(path), path)
            self.assertLessEqual(set(info['best_makespans']),
                                 {'u2_t20', 'u2_t40'})


if __name__ == "__main__":
    unittest.main()
