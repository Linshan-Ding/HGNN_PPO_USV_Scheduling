"""
Run PPO ablation variants on one selected public instance.

Variants:
- full: current HGNN dual-encoder Pairwise PPO
- no_hgnn: Simple node MLP encoder + Pairwise PPO
- shared_encoder: shared HGNN encoder for actor and critic
- no_reward_norm: full architecture with unnormalized rewards
"""

import argparse
import csv
import math
import os
from typing import List

from config import get_config


VALID_VARIANTS = ['full', 'no_hgnn', 'shared_encoder', 'no_reward_norm']


def _parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(',') if item.strip()]


def _parse_seeds(text: str) -> List[int]:
    return [int(item) for item in _parse_csv_list(text)]


def _validate_variants(variants: List[str]) -> List[str]:
    unknown = [variant for variant in variants if variant not in VALID_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown ablation variants: {unknown}. "
            f"Expected one or more of {VALID_VARIANTS}."
        )
    return variants


def run_ablation(args) -> List[dict]:
    from main import train

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    variants = _validate_variants(_parse_csv_list(args.variants))
    seeds = _parse_seeds(args.seeds)
    rows = []

    for variant in variants:
        for seed in seeds:
            visdom_env = f"usv_ablation_{variant}_u{args.n_usvs}_t{args.n_tasks}"
            cfg = get_config(
                n_usvs=args.n_usvs,
                n_tasks=args.n_tasks,
                instance_id=args.instance_id,
                data_dir=args.data_dir,
                result_dir=args.result_dir,
                model_dir=args.model_dir,
                max_epochs=args.max_epochs,
                seed=seed,
                hidden_dim=args.hidden_dim,
                hgnn_layers=args.hgnn_layers,
                n_heads=args.n_heads,
                dropout=args.dropout,
                lr_encoder=args.lr_encoder,
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic,
                n_trajectories=args.n_trajectories,
                ppo_epochs=args.ppo_epochs,
                entropy_coef=args.entropy_coef,
                eval_interval=args.eval_interval,
                log_interval=args.log_interval,
                use_visdom=args.visdom,
                visdom_server=args.visdom_server,
                visdom_port=args.visdom_port,
                visdom_env=visdom_env,
                ablation_variant=variant,
                save_training_csv=not args.no_training_csv,
                training_log_dir=os.path.join(args.result_dir, 'training_logs'),
                training_log_interval=args.training_log_interval,
                rollout_num_workers=args.rollout_num_workers,
                rollout_device=args.rollout_device,
                rollout_torch_threads=args.rollout_torch_threads,
                vectorized_update=not args.legacy_update,
                update_batch_size=args.update_batch_size,
                update_shuffle=not args.no_update_shuffle,
            )

            print("=" * 80)
            print(
                f"[Ablation] variant={variant}, seed={seed}, "
                f"USVs={args.n_usvs}, tasks={args.n_tasks}"
            )
            _, instance, train_info = train(cfg)

            baseline = train_info['baseline']
            best_eval = float(train_info['best_eval_makespan'])
            best_rule = float(baseline['best_rule_makespan'])
            gap_to_rule = (
                (best_eval - best_rule) / best_rule * 100.0
                if math.isfinite(best_eval) and best_rule > 0
                else float('inf')
            )

            rows.append({
                'variant': variant,
                'instance_id': instance.get(
                    'instance_id',
                    f"u{instance['n_usvs']}_t{instance['n_tasks']}"
                ),
                'n_usvs': instance['n_usvs'],
                'n_tasks': instance['n_tasks'],
                'seed': seed,
                'best_eval_makespan': best_eval,
                'best_rule_name': baseline['best_rule_name'],
                'best_rule_makespan': best_rule,
                'gap_to_rule_percent': gap_to_rule,
                'gap_to_full_percent': '',
                'success': math.isfinite(best_eval),
            })

    full_lookup = {
        (row['instance_id'], row['seed']): row['best_eval_makespan']
        for row in rows
        if row['variant'] == 'full' and math.isfinite(row['best_eval_makespan'])
    }
    for row in rows:
        full_value = full_lookup.get((row['instance_id'], row['seed']))
        if full_value and full_value > 0 and math.isfinite(row['best_eval_makespan']):
            row['gap_to_full_percent'] = (
                (row['best_eval_makespan'] - full_value) / full_value * 100.0
            )

    output_path = os.path.join(args.result_dir, args.output)
    fieldnames = [
        'variant', 'instance_id', 'n_usvs', 'n_tasks', 'seed',
        'best_eval_makespan', 'best_rule_name', 'best_rule_makespan',
        'gap_to_rule_percent', 'gap_to_full_percent', 'success'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Ablation] Summary saved: {output_path}")
    return rows


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default=','.join(VALID_VARIANTS))
    parser.add_argument('--n-usvs', type=int, default=2)
    parser.add_argument('--n-tasks', type=int, default=20)
    parser.add_argument('--instance-id', default=None)
    parser.add_argument('--data-dir', default='data/public')
    parser.add_argument('--result-dir', default='results')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--output', default='ablation_summary.csv')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hgnn-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr-encoder', type=float, default=1e-4)
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)
    parser.add_argument('--n-trajectories', type=int, default=8)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--visdom-server', default='http://localhost')
    parser.add_argument('--visdom-port', type=int, default=8097)
    parser.add_argument('--no-training-csv', action='store_true')
    parser.add_argument('--training-log-interval', type=int, default=1)
    parser.add_argument('--rollout-num-workers', type=int, default=0)
    parser.add_argument('--rollout-device', default='cpu')
    parser.add_argument('--rollout-torch-threads', type=int, default=1)
    parser.add_argument('--legacy-update', action='store_true')
    parser.add_argument('--update-batch-size', type=int, default=128)
    parser.add_argument('--no-update-shuffle', action='store_true')
    return parser


if __name__ == '__main__':
    run_ablation(build_parser().parse_args())
