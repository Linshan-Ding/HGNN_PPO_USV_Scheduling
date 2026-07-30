"""Smoke tests for realtime training CSV logging."""

import csv
import os
import tempfile
import unittest

from training_logger import TrainingCSVLogger, make_training_run_id


class TrainingCSVLoggerSmokeTest(unittest.TestCase):
    def test_logger_writes_header_and_flushes_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = make_training_run_id('PPO', 'full', 'u2_t20', 0, 'smoke')
            logger = TrainingCSVLogger(tmpdir, run_id)
            logger.log({
                'run_id': run_id,
                'algorithm': 'PPO',
                'variant': 'full',
                'instance_id': 'u2_t20',
                'n_usvs': 2,
                'n_tasks': 20,
                'seed': 0,
                'epoch': 1,
                'best_eval_makespan': 100.0,
            })

            path = logger.path
            self.assertTrue(os.path.isfile(path))
            with open(path, newline='', encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['run_id'], run_id)
            self.assertEqual(rows[0]['best_eval_makespan'], '100.0')
            logger.close()

    def test_round_robin_fields_present_and_unknown_keys_dropped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = make_training_run_id('PPO', 'full', 'public25', 0, 'smoke')
            logger = TrainingCSVLogger(tmpdir, run_id)
            logger.log({
                'run_id': run_id,
                'epoch': 3,
                'protocol': 'round_robin',
                'visit_index': 1,
                'steps_collected': 40,
                'rollout_time_per_decision_ms': 2.5,
                'eval_steps': 20,
                'eval_solve_time_sec': 0.05,
                'eval_time_per_decision_ms': 2.5,
                'exploration_epsilon': 0.7,
                'not_a_real_field': 'must_be_dropped',
            })
            with open(logger.path, newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                header = reader.fieldnames
            self.assertNotIn('not_a_real_field', header)
            for field in ('protocol', 'visit_index', 'steps_collected',
                          'rollout_time_per_decision_ms', 'eval_steps',
                          'eval_solve_time_sec', 'eval_time_per_decision_ms',
                          'exploration_epsilon'):
                self.assertIn(field, header)
            self.assertEqual(rows[0]['protocol'], 'round_robin')
            self.assertEqual(rows[0]['visit_index'], '1')
            self.assertEqual(rows[0]['steps_collected'], '40')
            logger.close()


if __name__ == "__main__":
    unittest.main()
