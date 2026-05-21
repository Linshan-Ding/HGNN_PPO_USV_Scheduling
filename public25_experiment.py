"""
Run the public 25-instance PPO experiment.

This script is intentionally conservative: each public instance is trained on
multiple seeds, the best deterministic-eval checkpoint per seed is used, and
the final CSV reports whether the seed-mean PPO makespan beats the best rule.
"""

import argparse
import csv
import math
import os
from typing import List, Tuple

import numpy as np

from config import get_config


def _rank_abs(values: List[float]) -> List[float]:
    """Average ranks for absolute non-zero differences."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank_less(ppo_values: List[float],
                              rule_values: List[float]) -> Tuple[float, float]:
    """
    One-sided Wilcoxon signed-rank test for H1: PPO < best rule.

    Returns:
        statistic_w_plus, p_value
    """
    diffs = [p - r for p, r in zip(ppo_values, rule_values) if abs(p - r) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = [abs(d) for d in diffs]
    ranks = _rank_abs(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return w_plus, 1.0

    # Continuity-corrected normal approximation. Small W+ supports PPO < rule.
    z = (w_plus - mean_w + 0.5) / math.sqrt(var_w)
    p_value = 0.5 * math.erfc(-z / math.sqrt(2.0))
    return w_plus, p_value


def _parse_seeds(seed_text: str) -> List[int]:
    return [int(item.strip()) for item in seed_text.split(',') if item.strip()]


def run_public25(args) -> List[dict]:
    from main import train

    manifest_path = os.path.join(args.data_dir, 'manifest.csv')
    os.makedirs(args.result_dir, exist_ok=True)

    with open(manifest_path, newline='', encoding='utf-8-sig') as f:
        manifest_rows = list(csv.DictReader(f))

    seeds = _parse_seeds(args.seeds)
    summary_rows = []

    for row in manifest_rows:
        instance_id = row['instance_id']
        n_usvs = int(row['n_usvs'])
        n_tasks = int(row['n_tasks'])
        seed_makespans = []
        best_rule_name = None
        best_rule_makespan = None

        print("=" * 80)
        print(f"[Public25] Instance {instance_id}: USVs={n_usvs}, Tasks={n_tasks}")

        for seed in seeds:
            cfg = get_config(
                n_usvs=n_usvs,
                n_tasks=n_tasks,
                instance_id=instance_id,
                data_dir=args.data_dir,
                result_dir=args.result_dir,
                model_dir=args.model_dir,
                max_epochs=args.max_epochs,
                seed=seed,
                hidden_dim=args.hidden_dim,
                hgnn_layers=args.hgnn_layers,
                n_heads=args.n_heads,
                dropout=args.dropout,
                lr_encoder=1e-4,
                lr_actor=3e-4,
                lr_critic=3e-4,
                n_trajectories=args.n_trajectories,
                ppo_epochs=4,
                entropy_coef=0.01,
                eval_interval=10,
                use_visdom=args.visdom,
                visdom_env='usv_training',
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

            _, _, train_info = train(cfg)
            seed_makespans.append(train_info['best_eval_makespan'])
            best_rule_name = train_info['baseline']['best_rule_name']
            best_rule_makespan = train_info['baseline']['best_rule_makespan']

        ppo_mean = float(np.mean(seed_makespans))
        ppo_std = float(np.std(seed_makespans))
        gap_percent = (ppo_mean - best_rule_makespan) / best_rule_makespan * 100.0
        pass_instance = ppo_mean < best_rule_makespan

        summary_rows.append({
            'instance_id': instance_id,
            'n_usvs': n_usvs,
            'n_tasks': n_tasks,
            'best_rule_name': best_rule_name,
            'best_rule_makespan': best_rule_makespan,
            'ppo_mean': ppo_mean,
            'ppo_std': ppo_std,
            'gap_percent': gap_percent,
            'pass_instance': pass_instance,
        })

    ppo_values = [row['ppo_mean'] for row in summary_rows]
    rule_values = [row['best_rule_makespan'] for row in summary_rows]
    statistic, p_value = wilcoxon_signed_rank_less(ppo_values, rule_values)
    avg_improvement = float(np.mean([
        (r - p) / r * 100.0 for p, r in zip(ppo_values, rule_values)
    ]))

    summary_path = os.path.join(args.result_dir, 'public25_summary.csv')
    fieldnames = [
        'instance_id', 'n_usvs', 'n_tasks', 'best_rule_name',
        'best_rule_makespan', 'ppo_mean', 'ppo_std',
        'gap_percent', 'pass_instance'
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    stats_path = os.path.join(args.result_dir, 'public25_wilcoxon.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"wilcoxon_w_plus={statistic:.6f}\n")
        f.write(f"p_value={p_value:.8f}\n")
        f.write(f"avg_improvement_percent={avg_improvement:.6f}\n")
        f.write(f"all_instances_pass={all(row['pass_instance'] for row in summary_rows)}\n")

    print(f"[Public25] Summary saved: {summary_path}")
    print(f"[Public25] Wilcoxon p={p_value:.8f}, avg improvement={avg_improvement:.2f}%")
    return summary_rows


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/public')
    parser.add_argument('--result-dir', default='results')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hgnn-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n-trajectories', type=int, default=8)
    parser.add_argument('--visdom', action='store_true')
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
    run_public25(build_parser().parse_args())
