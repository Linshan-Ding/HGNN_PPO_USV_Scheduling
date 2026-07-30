"""
Zero-shot generalization and scalability experiment.

Policies trained on a single public instance (default: u10_t100, the largest
training scale) are evaluated zero-shot -- no fine-tuning -- on larger unseen
instances generated with the same distribution and battery-sizing rule as the
public 25-instance benchmark. All five scheduling rules run directly on each
instance as baselines. Solution times follow the timing-fair protocol of the
paper (CPU, single torch thread, one untimed warm-up episode).

Outputs (paper Sec. "Generalization and scalability"):
    results/scalability_summary.csv   -- one row per unseen instance
    results/scalability_wilcoxon.txt  -- one-sided Wilcoxon over paired makespans
"""

import argparse
import csv
import os
from typing import List

import numpy as np

from config import get_config
from stats_utils import parse_seeds, wilcoxon_signed_rank_less

# Unseen evaluation scales: fleet-size-only control (u12_t100) plus the
# {10,12,15} x {150,200} grid beyond the largest training scale.
SCALABILITY_GRID = [
    (12, 100),
    (10, 150), (12, 150), (15, 150),
    (10, 200), (12, 200), (15, 200),
]

DRL_BASELINES = ['A2C', 'DQN', 'DDQN', 'REINFORCE']

SUMMARY_FIELDS = [
    'instance_id', 'n_usvs', 'n_tasks', 'source_instance',
    'best_rule_name', 'best_rule_makespan', 'best_rule_solve_time_sec',
    'ppo_zero_shot_makespan', 'ppo_gap_percent',
    'ppo_solve_time_sec', 'ppo_time_per_decision_ms',
    'a2c_zero_shot_makespan', 'a2c_gap_percent', 'a2c_solve_time_sec',
    'dqn_zero_shot_makespan', 'dqn_gap_percent', 'dqn_solve_time_sec',
    'ddqn_zero_shot_makespan', 'ddqn_gap_percent', 'ddqn_solve_time_sec',
    'reinforce_zero_shot_makespan', 'reinforce_gap_percent', 'reinforce_solve_time_sec',
]


def generate_scalability_instances(data_dir: str, seed_start: int) -> List[dict]:
    """Generate and persist the unseen instances (same recipe as public 25)."""
    from instance_generator import (
        DEFAULT_BATTERY_POLICY, DEFAULT_BATTERY_SAFETY_FACTOR,
        InstanceConfig, InstanceGenerator,
    )
    import pandas as pd

    os.makedirs(data_dir, exist_ok=True)
    instances = []
    manifest_rows = []

    for idx, (n_usvs, n_tasks) in enumerate(SCALABILITY_GRID):
        seed = seed_start + idx
        cfg = InstanceConfig(n_usvs=n_usvs, n_tasks=n_tasks)
        generator = InstanceGenerator(
            cfg, battery_safety_factor=DEFAULT_BATTERY_SAFETY_FACTOR)
        instance = generator.generate(seed=seed)
        instance_id = f"u{n_usvs}_t{n_tasks}"
        instance['instance_id'] = instance_id
        filename = f"{instance_id}.csv"
        generator.save_to_csv(instance, os.path.join(data_dir, filename))
        validation = InstanceGenerator.validate_battery_capacity(instance)
        manifest_rows.append({
            'instance_id': instance_id,
            'filename': filename,
            'n_usvs': n_usvs,
            'n_tasks': n_tasks,
            'seed': seed,
            'battery_capacity': validation['battery_capacity'],
            'battery_policy': DEFAULT_BATTERY_POLICY,
            'battery_safety_factor': DEFAULT_BATTERY_SAFETY_FACTOR,
            'max_single_trip_energy': validation['max_single_trip_energy'],
            'farthest_task_id': validation['farthest_task_id'],
            'farthest_task_energy': validation['farthest_task_energy'],
        })
        instances.append(instance)

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(data_dir, 'manifest.csv'), index=False)
    return instances


def evaluate_rules_timed(instance: dict) -> dict:
    """Run all rules once, then warm-up + timed re-run of the winner."""
    from env import USVSchedulingEnv
    from scheduling_rules import get_all_rules, run_scheduling

    best_name, best_makespan = None, float('inf')
    for rule in get_all_rules():
        if rule.name == 'Random':
            np.random.seed(0)
        result = run_scheduling(USVSchedulingEnv(instance), rule)
        makespan = result['makespan'] if result['success'] else float('inf')
        if makespan < best_makespan:
            best_name, best_makespan = rule.name, makespan

    rule = next(r for r in get_all_rules() if r.name == best_name)
    if rule.name == 'Random':
        np.random.seed(0)
    run_scheduling(USVSchedulingEnv(instance), rule)  # warm-up
    if rule.name == 'Random':
        np.random.seed(0)
    timed = run_scheduling(USVSchedulingEnv(instance), rule)
    return {
        'best_rule_name': best_name,
        'best_rule_makespan': best_makespan,
        'best_rule_solve_time_sec': timed['solve_time_sec'],
    }


def _make_eval_cfg(args, instance: dict):
    return get_config(
        n_usvs=instance['n_usvs'],
        n_tasks=instance['n_tasks'],
        model_dir=args.model_dir,
        hidden_dim=args.hidden_dim,
        hgnn_layers=args.hgnn_layers,
        n_heads=args.n_heads,
    )


def evaluate_ppo_zero_shot(args, instance: dict, seed: int,
                           best_rule_makespan: float) -> dict:
    """Timed zero-shot evaluation of the proposed method's checkpoint."""
    import torch
    from main import build_agent, evaluate_agent_once, timed_evaluation

    cfg = _make_eval_cfg(args, instance)
    ckpt = os.path.join(
        args.model_dir, f'best_{args.source_instance}_seed{seed}.pth')

    if args.smoke:
        torch.set_num_threads(1)
        agent = build_agent(cfg, instance, device='cpu')
        evaluate_agent_once(agent, instance)  # warm-up
        result = evaluate_agent_once(agent, instance)
    elif os.path.exists(ckpt):
        result = timed_evaluation(cfg, instance, ckpt)
    else:
        print(f"  [Skip] missing checkpoint: {ckpt}")
        return {'ppo_zero_shot_makespan': float('nan'),
                'ppo_gap_percent': float('nan'),
                'ppo_solve_time_sec': float('nan'),
                'ppo_time_per_decision_ms': float('nan')}

    makespan = result['makespan'] if result['success'] else float('nan')
    gap = (makespan - best_rule_makespan) / best_rule_makespan * 100.0 \
        if np.isfinite(makespan) and best_rule_makespan > 0 else float('nan')
    return {
        'ppo_zero_shot_makespan': makespan,
        'ppo_gap_percent': gap,
        'ppo_solve_time_sec': result['solve_time_sec'],
        'ppo_time_per_decision_ms': result['time_per_decision_ms'],
    }


def evaluate_baselines_zero_shot(args, instance: dict, seed: int,
                                 best_rule_makespan: float) -> dict:
    """Timed zero-shot evaluation of every DRL baseline's checkpoint."""
    from drl_baselines.common import (build_baseline_for_eval,
                                      evaluate_pairwise_policy_timed,
                                      timed_baseline_evaluation)

    cfg = _make_eval_cfg(args, instance)
    stats = {}
    for alg in DRL_BASELINES:
        key = alg.lower()
        ckpt = os.path.join(
            args.model_dir, f'best_{alg}_{args.source_instance}_seed{seed}.pth')
        if args.smoke:
            algorithm = build_baseline_for_eval(alg, cfg, instance)
            evaluate_pairwise_policy_timed(algorithm, instance)  # warm-up
            result = evaluate_pairwise_policy_timed(algorithm, instance)
        elif os.path.exists(ckpt):
            result = timed_baseline_evaluation(alg, cfg, instance, ckpt)
        else:
            print(f"  [Skip] missing checkpoint: {ckpt}")
            stats[f'{key}_zero_shot_makespan'] = float('nan')
            stats[f'{key}_gap_percent'] = float('nan')
            stats[f'{key}_solve_time_sec'] = float('nan')
            continue

        makespan = result['makespan'] if result['success'] else float('nan')
        gap = (makespan - best_rule_makespan) / best_rule_makespan * 100.0 \
            if np.isfinite(makespan) and best_rule_makespan > 0 else float('nan')
        stats[f'{key}_zero_shot_makespan'] = makespan
        stats[f'{key}_gap_percent'] = gap
        stats[f'{key}_solve_time_sec'] = result['solve_time_sec']
    return stats


def run_scalability(args) -> List[dict]:
    seeds = parse_seeds(args.seeds)
    seed = seeds[0]
    os.makedirs(args.result_dir, exist_ok=True)

    instances = generate_scalability_instances(args.data_dir, args.seed_start)
    summary_rows = []

    for instance in instances:
        instance_id = instance['instance_id']
        print("=" * 80)
        print(f"[Scalability] {instance_id}: USVs={instance['n_usvs']}, "
              f"Tasks={instance['n_tasks']} (zero-shot from {args.source_instance})")

        rule_stats = evaluate_rules_timed(instance)
        best_rule_makespan = rule_stats['best_rule_makespan']

        ppo_stats = evaluate_ppo_zero_shot(args, instance, seed, best_rule_makespan)
        drl_stats = evaluate_baselines_zero_shot(args, instance, seed,
                                                 best_rule_makespan)

        summary_rows.append({
            'instance_id': instance_id,
            'n_usvs': instance['n_usvs'],
            'n_tasks': instance['n_tasks'],
            'source_instance': args.source_instance,
            **rule_stats,
            **ppo_stats,
            **drl_stats,
        })
        print(f"  Best rule: {rule_stats['best_rule_name']} "
              f"({best_rule_makespan:.1f}), PPO zero-shot: "
              f"{ppo_stats['ppo_zero_shot_makespan']:.1f}, "
              f"gap: {ppo_stats['ppo_gap_percent']:+.2f}%")

    summary_path = os.path.join(args.result_dir, 'scalability_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    paired = [
        (row['ppo_zero_shot_makespan'], row['best_rule_makespan'])
        for row in summary_rows
        if np.isfinite(row['ppo_zero_shot_makespan'])
    ]
    if paired:
        ppo_values, rule_values = zip(*paired)
        statistic, p_value = wilcoxon_signed_rank_less(
            list(ppo_values), list(rule_values))
        avg_improvement = float(np.mean([
            (r - p) / r * 100.0 for p, r in paired
        ]))
        stats_path = os.path.join(args.result_dir, 'scalability_wilcoxon.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"wilcoxon_w_plus={statistic:.6f}\n")
            f.write(f"p_value={p_value:.8f}\n")
            f.write(f"avg_improvement_percent={avg_improvement:.6f}\n")
            f.write(f"n_instances={len(paired)}\n")
        print(f"[Scalability] Wilcoxon p={p_value:.8f}, "
              f"avg improvement={avg_improvement:.2f}%")

    print(f"[Scalability] Summary saved: {summary_path}")
    return summary_rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-instance', default='u10_t100',
                        help='Training instance whose checkpoints are evaluated zero-shot')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--data-dir', default=os.path.join('data', 'scalability'))
    parser.add_argument('--result-dir', default='results')
    parser.add_argument('--seed-start', type=int, default=2026052600)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hgnn-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true',
                        help='Pipeline check with a randomly initialized agent '
                             '(no checkpoints needed), first seed only')
    return parser


if __name__ == '__main__':
    run_scalability(build_parser().parse_args())
