"""Run all DRL baselines serially in isolated child processes.

Usage:
    python -m drl_baselines.run_all
    python -m drl_baselines.run_all --start-from DQN --no-visdom
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .multi_run import add_training_arguments, training_args_to_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ORDER = ('A2C', 'DQN', 'DDQN', 'REINFORCE')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--start-from',
        choices=ALGORITHM_ORDER,
        default=ALGORITHM_ORDER[0],
        help='Resume the serial batch from this algorithm (default: A2C).',
    )
    add_training_arguments(parser)
    return parser


def build_child_command(algorithm, args):
    """Build one isolated ``multi_run`` command using the current interpreter."""
    return [
        sys.executable,
        '-m',
        'drl_baselines.multi_run',
        '--algorithm',
        algorithm,
        *training_args_to_cli(args),
    ]


def build_resume_command(algorithm, args):
    """Build a batch command that resumes with the same training arguments."""
    return [
        sys.executable,
        '-m',
        'drl_baselines.run_all',
        '--start-from',
        algorithm,
        *training_args_to_cli(args),
    ]


def format_command(command):
    """Format an argv sequence for the current platform's shell."""
    if os.name == 'nt':
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_serial(args):
    """Run the selected suffix of ``ALGORITHM_ORDER`` and return an exit code."""
    start_index = ALGORITHM_ORDER.index(args.start_from)
    selected = ALGORITHM_ORDER[start_index:]
    batch_start = time.monotonic()

    print(
        f'[DRLBatch] Serial run: {" -> ".join(selected)} '
        f'(start-from={args.start_from})',
        flush=True,
    )

    for algorithm in selected:
        position = ALGORITHM_ORDER.index(algorithm) + 1
        algorithm_start = time.monotonic()
        print(
            f'\n[DRLBatch] [{position}/{len(ALGORITHM_ORDER)}] '
            f'Starting {algorithm}',
            flush=True,
        )
        completed = subprocess.run(
            build_child_command(algorithm, args),
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if completed.returncode != 0:
            print(
                f'[DRLBatch] {algorithm} failed with exit code '
                f'{completed.returncode}; stopping the batch.',
                file=sys.stderr,
                flush=True,
            )
            print(
                '[DRLBatch] Resume after fixing the issue with: '
                f'{format_command(build_resume_command(algorithm, args))}',
                file=sys.stderr,
                flush=True,
            )
            return completed.returncode

        elapsed = time.monotonic() - algorithm_start
        print(
            f'[DRLBatch] [{position}/{len(ALGORITHM_ORDER)}] '
            f'{algorithm} completed in {elapsed:.1f}s',
            flush=True,
        )

    elapsed = time.monotonic() - batch_start
    print(
        f'\n[DRLBatch] All selected baselines completed in {elapsed:.1f}s.',
        flush=True,
    )
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_serial(args)


if __name__ == '__main__':
    raise SystemExit(main())
