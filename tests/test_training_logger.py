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


if __name__ == "__main__":
    unittest.main()
