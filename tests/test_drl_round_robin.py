"""In-process smoke test for the DRL-baseline round-robin trainer."""

import csv
import importlib.util
import os
import tempfile
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed")
class DRLRoundRobinSmokeTest(unittest.TestCase):
    def test_a2c_round_robin_cycles_and_logs(self):
        from config import get_config
        from drl_baselines.registry import get_algorithm
        from multi_train import ensure_rules_results, load_public_instances

        instances = load_public_instances(
            os.path.join('data', 'public'), ['u2_t20', 'u2_t40'])

        with tempfile.TemporaryDirectory() as tmpdir:
            rules = ensure_rules_results(
                instances, path=os.path.join(tmpdir, 'rules_results.csv'))
            cfg = get_config(
                n_usvs=2, n_tasks=20,
                max_epochs=4, seed=0,
                hidden_dim=16, hgnn_layers=1, n_heads=2,
                n_trajectories=2,
                use_visdom=False,
                model_dir=os.path.join(tmpdir, 'models'),
                result_dir=os.path.join(tmpdir, 'results'),
                training_log_dir=os.path.join(tmpdir, 'logs'),
            )
            algorithm = get_algorithm('A2C', seed=0)
            info = algorithm.train_round_robin(instances, cfg, rules)

            with open(info['training_log_path'], newline='',
                      encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual([r['instance_id'] for r in rows],
                             ['u2_t20', 'u2_t40', 'u2_t20', 'u2_t40'])
            self.assertEqual({r['algorithm'] for r in rows}, {'A2C'})
            self.assertEqual({r['protocol'] for r in rows}, {'round_robin'})
            for row in rows:
                self.assertNotEqual(row['eval_solve_time_sec'], '')

            for instance_id, path in info['best_paths'].items():
                self.assertIn('A2C', path)
                self.assertIn(instance_id, path)
                self.assertTrue(os.path.isfile(path), path)

    def test_dqn_per_instance_replay_isolation(self):
        from config import get_config
        from drl_baselines.registry import get_algorithm
        from multi_train import load_public_instances

        instances = load_public_instances(
            os.path.join('data', 'public'), ['u2_t20', 'u2_t40'])
        cfg = get_config(n_usvs=2, n_tasks=20, hidden_dim=16,
                         hgnn_layers=1, n_heads=2, drl_rr_replay_size=50)
        algorithm = get_algorithm('DQN', seed=0)
        algorithm._build(cfg, 2, 20)

        algorithm.set_instance(instances[0])
        replay_a = algorithm.replay
        algorithm.set_instance(instances[1])
        replay_b = algorithm.replay
        self.assertIsNot(replay_a, replay_b)
        algorithm.set_instance(instances[0])
        self.assertIs(algorithm.replay, replay_a)
        self.assertEqual(algorithm.n_tasks, 20)


if __name__ == "__main__":
    unittest.main()
