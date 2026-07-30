"""
Round-robin multi-instance training driver for the DRL baselines.

Runs one baseline algorithm (A2C / DQN / DDQN / REINFORCE) under the exact
protocol of the main method (see multi_train.py): one policy, seed 0,
--max-epochs epochs cycling the 25 public instances, deterministic evaluation
every visit, per-instance best checkpoints
(models/best_{ALG}_{instance_id}_seed{seed}.pth), and per-epoch CSV logs that
extract_results.py turns into the paper tables.

Usage:
    python -m drl_baselines.multi_run --algorithm A2C
    python -m drl_baselines.multi_run --algorithm DDQN --instances u2_t20,u2_t40 --max-epochs 4
"""

import argparse
import os

from config import get_config
from multi_train import ensure_rules_results, load_public_instances

from .registry import get_algorithm, list_algorithms


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', required=True,
                        help=f'One of: {", ".join(list_algorithms())}')
    parser.add_argument('--data-dir', default=os.path.join('data', 'public'))
    parser.add_argument('--result-dir', default='results')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--max-epochs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instances', default=None,
                        help='Comma-separated instance IDs (smoke runs); default all 25')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hgnn-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n-trajectories', type=int, default=8)
    parser.add_argument('--epsilon-decay-epochs', type=int, default=2000,
                        help='DQN/DDQN epsilon decay horizon in global epochs')
    parser.add_argument('--rr-replay-size', type=int, default=2000,
                        help='DQN/DDQN per-instance replay capacity')
    parser.add_argument('--no-visdom', action='store_true')
    parser.add_argument('--visdom-env', default='usv_rr')
    parser.add_argument('--no-training-csv', action='store_true')
    parser.add_argument('--rules-seed', type=int, default=20260519)
    return parser


def main(args) -> dict:
    instance_ids = ([item.strip() for item in args.instances.split(',') if item.strip()]
                    if args.instances else None)
    instances = load_public_instances(args.data_dir, instance_ids)
    rules_by_instance = ensure_rules_results(
        instances,
        path=os.path.join(args.result_dir, 'rules_results.csv'),
        random_seed=args.rules_seed,
    )

    cfg = get_config(
        n_usvs=instances[0]['n_usvs'],
        n_tasks=instances[0]['n_tasks'],
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        hgnn_layers=args.hgnn_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr_encoder=1e-4,
        lr_actor=3e-4,
        lr_critic=3e-4,
        n_trajectories=args.n_trajectories,
        entropy_coef=0.01,
        drl_epsilon_decay_epochs=args.epsilon_decay_epochs,
        drl_rr_replay_size=args.rr_replay_size,
        use_visdom=not args.no_visdom,
        visdom_env=args.visdom_env,
        save_training_csv=not args.no_training_csv,
        training_log_dir=os.path.join(args.result_dir, 'training_logs'),
    )

    algorithm = get_algorithm(args.algorithm, seed=args.seed)
    return algorithm.train_round_robin(instances, cfg, rules_by_instance)


if __name__ == '__main__':
    main(build_parser().parse_args())
