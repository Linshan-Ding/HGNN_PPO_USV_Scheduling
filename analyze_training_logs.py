"""Analyze epoch-level training CSV logs and generate candidate figures."""

import argparse
import os
import re
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    'n_usvs',
    'n_tasks',
    'seed',
    'epoch',
    'elapsed_sec',
    'train_reward_avg',
    'train_reward_std',
    'train_makespan_avg',
    'train_makespan_min',
    'train_makespan_std',
    'success_rate',
    'n_trajectories',
    'n_success',
    'n_failed',
    'eval_makespan',
    'best_eval_makespan',
    'best_eval_epoch',
    'gap_to_best_rule_percent',
    'best_rule_makespan',
    'random_makespan',
    'actor_loss',
    'critic_loss',
    'entropy',
    'rollout_time_sec',
    'update_time_sec',
    'epoch_time_sec',
    'batch_prepare_time_sec',
    'actor_update_time_sec',
    'critic_update_time_sec',
    'vectorized_update',
    'update_batch_size',
    'update_micro_batch_size',
    'max_update_pairs',
    'update_shuffle',
    'effective_update_batch_size',
    'effective_update_micro_batch_size',
    'pairs_per_state',
]


def _safe_name(text: object) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_')


def load_logs(log_dir: str, instance_id: str = None, variant: str = None,
              algorithm: str = None) -> List[pd.DataFrame]:
    frames = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith('.csv') or name == 'summary.csv':
            continue
        path = os.path.join(log_dir, name)
        df = pd.read_csv(path, encoding='utf-8-sig')
        if df.empty or 'run_id' not in df.columns:
            continue
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.attrs['training_log_path'] = path
        first = df.iloc[0]
        if instance_id and str(first.get('instance_id')) != instance_id:
            continue
        if variant and str(first.get('variant')) != variant:
            continue
        if algorithm and str(first.get('algorithm')) != algorithm:
            continue
        frames.append(df)
    return frames


def build_summary(frames: List[pd.DataFrame], summary_path: str) -> pd.DataFrame:
    rows = []
    for df in frames:
        first = df.iloc[0]
        last = df.iloc[-1]
        best_series = df['best_eval_makespan'].dropna()
        best_epoch_series = df['best_eval_epoch'].dropna()
        best_rule = float(first.get('best_rule_makespan')) if pd.notna(first.get('best_rule_makespan')) else float('nan')
        final_best = float(best_series.iloc[-1]) if not best_series.empty else float('nan')
        gap = (
            (final_best - best_rule) / best_rule * 100.0
            if pd.notna(final_best) and pd.notna(best_rule) and best_rule > 0
            else float('nan')
        )
        rows.append({
            'run_id': first.get('run_id'),
            'algorithm': first.get('algorithm'),
            'variant': first.get('variant'),
            'instance_id': first.get('instance_id'),
            'n_usvs': int(first.get('n_usvs')) if pd.notna(first.get('n_usvs')) else '',
            'n_tasks': int(first.get('n_tasks')) if pd.notna(first.get('n_tasks')) else '',
            'seed': int(first.get('seed')) if pd.notna(first.get('seed')) else '',
            'final_best_eval_makespan': final_best,
            'best_eval_epoch': int(best_epoch_series.iloc[-1]) if not best_epoch_series.empty else '',
            'best_rule_makespan': best_rule,
            'gap_to_best_rule_percent': gap,
            'final_success_rate': last.get('success_rate'),
            'training_log_path': df.attrs.get('training_log_path'),
        })

    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(summary_path) or '.', exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    return summary


def plot_single_run(df: pd.DataFrame, output_dir: str):
    first = df.iloc[0]
    run_id = first.get('run_id')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df['epoch'], df['train_makespan_avg'], label='Train Makespan Avg', alpha=0.8)
    ax.plot(df['epoch'], df['best_eval_makespan'], label='Best Eval Makespan', linewidth=2)
    if 'eval_makespan' in df:
        eval_rows = df.dropna(subset=['eval_makespan'])
        ax.scatter(eval_rows['epoch'], eval_rows['eval_makespan'], label='Eval Makespan', s=20)
    if pd.notna(first.get('best_rule_makespan')):
        ax.axhline(first.get('best_rule_makespan'), color='tab:red', linestyle='--',
                   label='Best Rule')
    ax.set_title(str(run_id))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Makespan')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'run_{_safe_name(run_id)}_curves.png'), dpi=200)
    plt.close(fig)


def plot_mean_curves(frames: List[pd.DataFrame], output_dir: str):
    grouped = {}
    for df in frames:
        first = df.iloc[0]
        key = (first.get('instance_id'), first.get('variant'))
        grouped.setdefault(key, []).append(df[['epoch', 'best_eval_makespan']].copy())

    for (instance_id, variant), dfs in grouped.items():
        merged = pd.concat(dfs, ignore_index=True)
        stats = merged.groupby('epoch')['best_eval_makespan'].agg(['mean', 'std']).reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(stats['epoch'], stats['mean'], label=f'{variant} mean', linewidth=2)
        if stats['std'].notna().any():
            ax.fill_between(
                stats['epoch'],
                stats['mean'] - stats['std'].fillna(0),
                stats['mean'] + stats['std'].fillna(0),
                alpha=0.2,
                label='std'
            )
        ax.set_title(f'{instance_id} - {variant}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Eval Makespan')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f'mean_curve_{_safe_name(instance_id)}_{_safe_name(variant)}.png'),
            dpi=200
        )
        plt.close(fig)


def plot_ablation_curves(frames: List[pd.DataFrame], output_dir: str):
    by_instance = {}
    for df in frames:
        by_instance.setdefault(df.iloc[0].get('instance_id'), []).append(df)

    for instance_id, instance_frames in by_instance.items():
        variants = sorted({df.iloc[0].get('variant') for df in instance_frames})
        if len(variants) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for variant in variants:
            variant_rows = []
            for df in instance_frames:
                if df.iloc[0].get('variant') == variant:
                    variant_rows.append(df[['epoch', 'best_eval_makespan']])
            merged = pd.concat(variant_rows, ignore_index=True)
            stats = merged.groupby('epoch')['best_eval_makespan'].mean().reset_index()
            ax.plot(stats['epoch'], stats['best_eval_makespan'], label=str(variant), linewidth=2)
        ax.set_title(f'Ablation Curves - {instance_id}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Best Eval Makespan')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'ablation_{_safe_name(instance_id)}.png'), dpi=200)
        plt.close(fig)


def plot_gap_by_tasks(summary: pd.DataFrame, output_dir: str):
    if summary.empty or 'n_tasks' not in summary or summary['n_tasks'].isna().all():
        return
    plot_df = summary.dropna(subset=['gap_to_best_rule_percent', 'n_tasks'])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = [
        group['gap_to_best_rule_percent'].values
        for _, group in plot_df.groupby('n_tasks')
    ]
    labels = [str(int(n)) for n in sorted(plot_df['n_tasks'].dropna().unique())]
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.axhline(0, color='tab:red', linestyle='--', linewidth=1)
    ax.set_title('Gap to Best Rule by Task Scale')
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Gap to Best Rule (%)')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gap_by_tasks.png'), dpi=200)
    plt.close(fig)


def analyze(args):
    os.makedirs(args.output_dir, exist_ok=True)
    frames = load_logs(args.log_dir, args.instance_id, args.variant, args.algorithm)
    if not frames:
        raise ValueError(f'No training log CSV files found under {args.log_dir}')

    summary_path = os.path.join(args.log_dir, 'summary.csv')
    summary = build_summary(frames, summary_path)

    for df in frames[:max(args.max_run_plots, 0)]:
        plot_single_run(df, args.output_dir)
    plot_mean_curves(frames, args.output_dir)
    plot_ablation_curves(frames, args.output_dir)
    plot_gap_by_tasks(summary, args.output_dir)

    print(f"[Analyze] Loaded runs: {len(frames)}")
    print(f"[Analyze] Summary saved: {summary_path}")
    print(f"[Analyze] Figures saved under: {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='results/training_logs')
    parser.add_argument('--output-dir', default='results/figures')
    parser.add_argument('--instance-id', default=None)
    parser.add_argument('--variant', default=None)
    parser.add_argument('--algorithm', default=None)
    parser.add_argument('--max-run-plots', type=int, default=20)
    return parser


if __name__ == '__main__':
    analyze(build_parser().parse_args())
