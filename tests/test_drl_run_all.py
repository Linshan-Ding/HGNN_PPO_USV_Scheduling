"""Tests for the serial all-DRL-baselines runner."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from drl_baselines import run_all


class DRLRunAllTest(unittest.TestCase):
    def test_default_order_and_process_isolation(self):
        completed = subprocess.CompletedProcess([], 0)
        with mock.patch.object(run_all.subprocess, 'run', return_value=completed) as runner:
            exit_code = run_all.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(runner.call_count, 4)
        algorithms = [
            call.args[0][call.args[0].index('--algorithm') + 1]
            for call in runner.call_args_list
        ]
        self.assertEqual(algorithms, ['A2C', 'DQN', 'DDQN', 'REINFORCE'])
        for call in runner.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:3], [
                sys.executable, '-m', 'drl_baselines.multi_run'])
            self.assertEqual(call.kwargs['cwd'], str(run_all.PROJECT_ROOT))
            self.assertFalse(call.kwargs['check'])

    def test_all_training_arguments_are_forwarded(self):
        args = run_all.build_parser().parse_args([
            '--start-from', 'DQN',
            '--data-dir', r'D:\data folder\public',
            '--result-dir', r'D:\result folder',
            '--model-dir', r'D:\model folder',
            '--max-epochs', '12',
            '--seed', '7',
            '--instances', 'u2_t20,u2_t40',
            '--hidden-dim', '32',
            '--hgnn-layers', '1',
            '--n-heads', '2',
            '--dropout', '0.2',
            '--n-trajectories', '3',
            '--epsilon-decay-epochs', '99',
            '--rr-replay-size', '101',
            '--no-visdom',
            '--visdom-env', 'batch_test',
            '--no-training-csv',
            '--rules-seed', '123',
        ])

        command = run_all.build_child_command('DQN', args)
        self.assertEqual(command, [
            sys.executable, '-m', 'drl_baselines.multi_run',
            '--algorithm', 'DQN',
            '--data-dir', r'D:\data folder\public',
            '--result-dir', r'D:\result folder',
            '--model-dir', r'D:\model folder',
            '--max-epochs', '12',
            '--seed', '7',
            '--hidden-dim', '32',
            '--hgnn-layers', '1',
            '--n-heads', '2',
            '--dropout', '0.2',
            '--n-trajectories', '3',
            '--epsilon-decay-epochs', '99',
            '--rr-replay-size', '101',
            '--visdom-env', 'batch_test',
            '--rules-seed', '123',
            '--instances', 'u2_t20,u2_t40',
            '--no-visdom',
            '--no-training-csv',
        ])

    def test_start_from_runs_only_requested_suffix(self):
        completed = subprocess.CompletedProcess([], 0)
        with mock.patch.object(run_all.subprocess, 'run', return_value=completed) as runner:
            exit_code = run_all.main(['--start-from', 'DDQN'])

        self.assertEqual(exit_code, 0)
        algorithms = [
            call.args[0][call.args[0].index('--algorithm') + 1]
            for call in runner.call_args_list
        ]
        self.assertEqual(algorithms, ['DDQN', 'REINFORCE'])

    def test_resume_command_preserves_training_arguments(self):
        args = run_all.build_parser().parse_args([
            '--result-dir', r'D:\result folder',
            '--max-epochs', '12',
            '--no-visdom',
        ])

        command = run_all.build_resume_command('DQN', args)
        self.assertEqual(command[:5], [
            sys.executable, '-m', 'drl_baselines.run_all',
            '--start-from', 'DQN',
        ])
        self.assertEqual(
            command[command.index('--result-dir') + 1],
            r'D:\result folder',
        )
        self.assertEqual(command[command.index('--max-epochs') + 1], '12')
        self.assertIn('--no-visdom', command)

    def test_failure_stops_batch_and_propagates_exit_code(self):
        results = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 17),
        ]
        with mock.patch.object(run_all.subprocess, 'run', side_effect=results) as runner:
            exit_code = run_all.main([])

        self.assertEqual(exit_code, 17)
        self.assertEqual(runner.call_count, 2)
        second_command = runner.call_args_list[1].args[0]
        self.assertEqual(
            second_command[second_command.index('--algorithm') + 1], 'DQN')

    def test_project_root_is_package_parent(self):
        self.assertEqual(
            run_all.PROJECT_ROOT,
            Path(run_all.__file__).resolve().parents[1],
        )


if __name__ == '__main__':
    unittest.main()
