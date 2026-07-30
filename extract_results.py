"""
Extract paper-facing result CSVs from round-robin training logs.

Per-instance results follow the paper protocol: mean/std/min over the
deterministic evaluation makespans of the instance's LAST 10 visits; average
epoch time is the mean epoch_time_sec over ALL visits of that instance.

Inputs:
    results/training_logs/*.csv   (multi_train.py / drl_baselines/multi_run.py)
    results/rules_results.csv     (multi_train.ensure_rules_results)

Outputs (each schema is cited 1:1 by a paper table note):
    results/main_results.csv      rules vs ours          -> tab:main_results
    results/drl_results.csv       DRL baselines vs ours  -> tab:drl_results
    results/ablation_results.csv  ablation variants      -> tab:ablation
    results/decision_time_grid.csv  U x T decision time  -> fig:decision_time_heatmap
    results/wilcoxon_results.csv  8 one-sided tests      -> tab:wilcoxon
"""

import argparse
import os
from typing import Dict, Optional

import pandas as pd

from stats_utils import wilcoxon_signed_rank_less

LAST_N_VISITS = 10

RULE_SHORT_NAMES = {
    'MinBattery_NearestTask': 'R1',
    'MaxBattery_NearestTask': 'R2',
    'NearestOrigin_NearestTask': 'R3',
    'FarthestOrigin_NearestTask': 'R4',
    'Random': 'R5',
}

DRL_METHODS = ['a2c', 'dqn', 'ddqn', 'reinforce']
ABLATION_VARIANTS = ['full', 'no_hgnn', 'shared_encoder', 'no_reward_norm']


def load_round_robin_logs(log_dir: str) -> pd.DataFrame:
    """Concatenate round-robin rows from all training logs, deduped to the
    latest run per (algorithm, variant)."""
    frames = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith('.csv') or name == 'summary.csv':
            continue
        df = pd.read_csv(os.path.join(log_dir, name), encoding='utf-8-sig')
        if df.empty or 'protocol' not in df.columns:
            continue
        df = df[df['protocol'] == 'round_robin']
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise ValueError(f'No round-robin training logs under {log_dir}')
    merged = pd.concat(frames, ignore_index=True)

    for col in ('epoch', 'visit_index', 'eval_makespan', 'epoch_time_sec',
                'eval_time_per_decision_ms', 'eval_steps', 'n_usvs', 'n_tasks'):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    # Keep only the most recent run_id per (algorithm, variant)
    latest = merged.groupby(['algorithm', 'variant'])['run_id'].transform('max')
    return merged[merged['run_id'] == latest].copy()


def method_run(logs: pd.DataFrame, algorithm: str,
               variant: str) -> Optional[pd.DataFrame]:
    run = logs[(logs['algorithm'] == algorithm) & (logs['variant'] == variant)]
    return run if not run.empty else None


def per_instance_stats(run: pd.DataFrame) -> Dict[str, dict]:
    """Last-10-visit eval stats + all-visit mean epoch time per instance."""
    stats = {}
    for instance_id, group in run.groupby('instance_id'):
        group = group.sort_values('epoch')
        evals = group.dropna(subset=['eval_makespan']).tail(LAST_N_VISITS)
        entry = {
            'n_usvs': int(group['n_usvs'].iloc[0]),
            'n_tasks': int(group['n_tasks'].iloc[0]),
            'last10_mean': float(evals['eval_makespan'].mean()) if not evals.empty else float('nan'),
            'last10_std': float(evals['eval_makespan'].std(ddof=0)) if not evals.empty else float('nan'),
            'last10_min': float(evals['eval_makespan'].min()) if not evals.empty else float('nan'),
            'epoch_time_mean_sec': float(group['epoch_time_sec'].mean()),
        }
        if 'eval_time_per_decision_ms' in group:
            tail = group.dropna(subset=['eval_time_per_decision_ms']).tail(LAST_N_VISITS)
            entry['eval_time_per_decision_ms_mean'] = (
                float(tail['eval_time_per_decision_ms'].mean()) if not tail.empty else float('nan'))
            entry['eval_steps_mean'] = (
                float(tail['eval_steps'].mean()) if not tail.empty else float('nan'))
        stats[instance_id] = entry
    return stats


def load_rules(rules_csv: str) -> pd.DataFrame:
    rules = pd.read_csv(rules_csv, encoding='utf-8-sig')
    missing = set(RULE_SHORT_NAMES) - set(rules['rule_name'].unique())
    if missing:
        raise ValueError(f'rules_results.csv missing rules: {sorted(missing)}')
    return rules


def instance_order(rules: pd.DataFrame) -> list:
    ordered = rules[['instance_id', 'n_usvs', 'n_tasks']].drop_duplicates()
    ordered = ordered.sort_values(['n_usvs', 'n_tasks'])
    return list(ordered['instance_id'])


def build_main_results(rules: pd.DataFrame, ours: Dict[str, dict],
                       out_path: str) -> pd.DataFrame:
    rows = []
    for instance_id in instance_order(rules):
        inst_rules = rules[rules['instance_id'] == instance_id]
        row = {
            'instance_id': instance_id,
            'n_usvs': int(inst_rules['n_usvs'].iloc[0]),
            'n_tasks': int(inst_rules['n_tasks'].iloc[0]),
        }
        best_name, best_cmax = None, float('inf')
        for rule_name, short in RULE_SHORT_NAMES.items():
            rule_row = inst_rules[inst_rules['rule_name'] == rule_name].iloc[0]
            row[f'{short}_cmax'] = float(rule_row['makespan'])
            row[f'{short}_time_sec'] = float(rule_row['solve_time_sec'])
            if rule_row['makespan'] < best_cmax:
                best_cmax = float(rule_row['makespan'])
                best_name = rule_name
        row['best_rule_name'] = best_name
        row['best_rule_cmax'] = best_cmax

        stats = ours.get(instance_id, {})
        row['ours_last10_mean'] = stats.get('last10_mean')
        row['ours_last10_std'] = stats.get('last10_std')
        row['ours_last10_min'] = stats.get('last10_min')
        row['ours_epoch_time_mean_sec'] = stats.get('epoch_time_mean_sec')
        mean = stats.get('last10_mean')
        row['gap_percent'] = ((mean - best_cmax) / best_cmax * 100.0
                              if mean is not None and pd.notna(mean) else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return df


def build_drl_results(logs: pd.DataFrame, rules: pd.DataFrame,
                      ours: Dict[str, dict], out_path: str) -> pd.DataFrame:
    method_stats = {'ours': ours}
    for method in DRL_METHODS:
        run = method_run(logs, method.upper(), 'baseline')
        method_stats[method] = per_instance_stats(run) if run is not None else {}

    rows = []
    for instance_id in instance_order(rules):
        inst_rules = rules[rules['instance_id'] == instance_id]
        row = {
            'instance_id': instance_id,
            'n_usvs': int(inst_rules['n_usvs'].iloc[0]),
            'n_tasks': int(inst_rules['n_tasks'].iloc[0]),
            'best_rule_cmax': float(inst_rules['makespan'].min()),
        }
        for method in DRL_METHODS + ['ours']:
            stats = method_stats[method].get(instance_id, {})
            row[f'{method}_last10_mean'] = stats.get('last10_mean')
            row[f'{method}_last10_std'] = stats.get('last10_std')
            row[f'{method}_epoch_time_mean_sec'] = stats.get('epoch_time_mean_sec')

        baseline_means = [row[f'{m}_last10_mean'] for m in DRL_METHODS
                          if row[f'{m}_last10_mean'] is not None
                          and pd.notna(row[f'{m}_last10_mean'])]
        ours_mean = row['ours_last10_mean']
        row['ours_gap_vs_best_drl_percent'] = (
            (ours_mean - min(baseline_means)) / min(baseline_means) * 100.0
            if baseline_means and ours_mean is not None and pd.notna(ours_mean)
            else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return df


def build_ablation_results(logs: pd.DataFrame, rules: pd.DataFrame,
                           out_path: str) -> pd.DataFrame:
    variant_stats = {}
    for variant in ABLATION_VARIANTS:
        run = method_run(logs, 'PPO', variant)
        variant_stats[variant] = per_instance_stats(run) if run is not None else {}

    rows = []
    for instance_id in instance_order(rules):
        inst_rules = rules[rules['instance_id'] == instance_id]
        row = {
            'instance_id': instance_id,
            'n_usvs': int(inst_rules['n_usvs'].iloc[0]),
            'n_tasks': int(inst_rules['n_tasks'].iloc[0]),
        }
        full_mean = variant_stats['full'].get(instance_id, {}).get('last10_mean')
        for variant in ABLATION_VARIANTS:
            stats = variant_stats[variant].get(instance_id, {})
            row[f'{variant}_last10_mean'] = stats.get('last10_mean')
            row[f'{variant}_last10_std'] = stats.get('last10_std')
            row[f'{variant}_epoch_time_mean_sec'] = stats.get('epoch_time_mean_sec')
            if variant != 'full':
                mean = stats.get('last10_mean')
                row[f'{variant}_gap_to_full_percent'] = (
                    (mean - full_mean) / full_mean * 100.0
                    if mean is not None and full_mean is not None
                    and pd.notna(mean) and pd.notna(full_mean) else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return df


def build_decision_time_grid(ours: Dict[str, dict], out_path: str) -> pd.DataFrame:
    rows = []
    for instance_id, stats in sorted(
            ours.items(), key=lambda kv: (kv[1]['n_usvs'], kv[1]['n_tasks'])):
        rows.append({
            'instance_id': instance_id,
            'n_usvs': stats['n_usvs'],
            'n_tasks': stats['n_tasks'],
            'eval_time_per_decision_ms_mean': stats.get('eval_time_per_decision_ms_mean'),
            'eval_steps_mean': stats.get('eval_steps_mean'),
            'epoch_time_mean_sec': stats.get('epoch_time_mean_sec'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return df


def build_wilcoxon(logs: pd.DataFrame, rules: pd.DataFrame,
                   ours: Dict[str, dict], out_path: str,
                   alpha: float = 0.05) -> pd.DataFrame:
    order = instance_order(rules)
    ours_means = [ours.get(i, {}).get('last10_mean') for i in order]

    comparisons = []

    best_rule = [float(rules[rules['instance_id'] == i]['makespan'].min())
                 for i in order]
    comparisons.append(('ours_vs_best_rule', ours_means, best_rule))

    for method in DRL_METHODS:
        run = method_run(logs, method.upper(), 'baseline')
        stats = per_instance_stats(run) if run is not None else {}
        comparisons.append((f'ours_vs_{method}',
                            ours_means,
                            [stats.get(i, {}).get('last10_mean') for i in order]))

    full_run = method_run(logs, 'PPO', 'full')
    full_stats = per_instance_stats(full_run) if full_run is not None else {}
    full_means = [full_stats.get(i, {}).get('last10_mean') for i in order]
    for variant in ABLATION_VARIANTS[1:]:
        run = method_run(logs, 'PPO', variant)
        stats = per_instance_stats(run) if run is not None else {}
        comparisons.append((f'full_vs_{variant}',
                            full_means,
                            [stats.get(i, {}).get('last10_mean') for i in order]))

    rows = []
    for name, ours_vals, other_vals in comparisons:
        paired = [(a, b) for a, b in zip(ours_vals, other_vals)
                  if a is not None and b is not None
                  and pd.notna(a) and pd.notna(b)]
        if paired:
            a_vals, b_vals = zip(*paired)
            w_plus, p_value = wilcoxon_signed_rank_less(list(a_vals), list(b_vals))
            rows.append({'comparison': name, 'w_plus': w_plus,
                         'p_value': p_value,
                         'significant': int(p_value < alpha),
                         'n_pairs': len(paired)})
        else:
            rows.append({'comparison': name, 'w_plus': None, 'p_value': None,
                         'significant': None, 'n_pairs': 0})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return df


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logs = load_round_robin_logs(args.log_dir)
    rules = load_rules(args.rules_csv)

    ours_run = method_run(logs, 'PPO', 'full')
    if ours_run is None:
        raise ValueError('No PPO/full round-robin run found in the training logs')
    ours = per_instance_stats(ours_run)

    outputs = {
        'main_results.csv': build_main_results(
            rules, ours, os.path.join(args.out_dir, 'main_results.csv')),
        'drl_results.csv': build_drl_results(
            logs, rules, ours, os.path.join(args.out_dir, 'drl_results.csv')),
        'ablation_results.csv': build_ablation_results(
            logs, rules, os.path.join(args.out_dir, 'ablation_results.csv')),
        'decision_time_grid.csv': build_decision_time_grid(
            ours, os.path.join(args.out_dir, 'decision_time_grid.csv')),
        'wilcoxon_results.csv': build_wilcoxon(
            logs, rules, ours, os.path.join(args.out_dir, 'wilcoxon_results.csv')),
    }
    for name, df in outputs.items():
        print(f'[Extract] {name}: {len(df)} rows')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default=os.path.join('results', 'training_logs'))
    parser.add_argument('--rules-csv', default=os.path.join('results', 'rules_results.csv'))
    parser.add_argument('--out-dir', default='results')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
