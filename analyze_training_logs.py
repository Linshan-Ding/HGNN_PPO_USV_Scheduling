"""Analyze training CSV logs (round-robin aware) and generate the paper figures.

Round-robin logs interleave 25 instances in one CSV; every per-instance view
here groups by the `instance_id` column and plots against `visit_index`.

Figure inventory (paper figures/fig_*.tex cite these outputs):
    training_curves.pdf        2x2 representative-instance convergence
    all25_convergence.pdf      5x5 small multiples, gap%% vs visit
    ablation_curves.pdf        2x2 instances x 4 ablation variants
    gap_heatmap.pdf            U x T heatmap of ours gap%%
    decision_time_heatmap.pdf  U x T heatmap of ms/decision
    drl_gap_violin.pdf         per-method violin of last-10 gap%%
    gap_by_tasks.pdf           boxplot of gap%% grouped by task count
    scalability.pdf            (a) gap vs M per method; (b) log solve time vs M
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Output format for all figures ('pdf' feeds the paper's figures/generated/
# directory directly; override with --format png for quick previews).
FIG_FORMAT = 'pdf'

REPRESENTATIVE_INSTANCES = ['u2_t20', 'u4_t60', 'u8_t80', 'u10_t100']
ABLATION_VARIANTS = ['full', 'no_hgnn', 'shared_encoder', 'no_reward_norm']
DRL_METHODS = ['A2C', 'DQN', 'DDQN', 'REINFORCE']
LAST_N_VISITS = 10

NUMERIC_COLUMNS = [
    'n_usvs', 'n_tasks', 'seed', 'epoch', 'elapsed_sec',
    'train_reward_avg', 'train_reward_std',
    'train_makespan_avg', 'train_makespan_min', 'train_makespan_std',
    'success_rate', 'n_trajectories', 'n_success', 'n_failed',
    'eval_makespan', 'best_eval_makespan', 'best_eval_epoch',
    'gap_to_best_rule_percent', 'best_rule_makespan', 'random_makespan',
    'actor_loss', 'critic_loss', 'entropy',
    'rollout_time_sec', 'update_time_sec', 'epoch_time_sec',
    'batch_prepare_time_sec', 'actor_update_time_sec', 'critic_update_time_sec',
    'visit_index', 'steps_collected', 'rollout_time_per_decision_ms',
    'eval_steps', 'eval_solve_time_sec', 'eval_time_per_decision_ms',
    'exploration_epsilon',
]


def _fig_path(output_dir: str, stem: str) -> str:
    return os.path.join(output_dir, f'{stem}.{FIG_FORMAT}')


def _safe_name(text: object) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_')


def load_logs(log_dir: str, algorithm: str = None,
              variant: str = None) -> List[pd.DataFrame]:
    """Load per-run DataFrames; a run is labeled by (algorithm, variant)."""
    frames = []
    if not os.path.isdir(log_dir):
        return frames
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
        if algorithm and str(first.get('algorithm')) != algorithm:
            continue
        if variant and str(first.get('variant')) != variant:
            continue
        frames.append(df)
    return frames


def latest_run(frames: List[pd.DataFrame], algorithm: str,
               variant: str) -> Optional[pd.DataFrame]:
    """Most recent run for (algorithm, variant); run_ids embed timestamps."""
    candidates = [df for df in frames
                  if str(df.iloc[0].get('algorithm')) == algorithm
                  and str(df.iloc[0].get('variant')) == variant]
    if not candidates:
        return None
    return max(candidates, key=lambda df: str(df.iloc[0].get('run_id')))


def build_summary(frames: List[pd.DataFrame], summary_path: str) -> pd.DataFrame:
    """One row per (run x instance): last-10 stats + mean timings."""
    rows = []
    for df in frames:
        first = df.iloc[0]
        for instance_id, group in df.groupby('instance_id'):
            group = group.sort_values('epoch')
            evals = group.dropna(subset=['eval_makespan'])
            tail = evals.tail(LAST_N_VISITS)
            best_rule = group['best_rule_makespan'].dropna()
            best_rule = float(best_rule.iloc[0]) if not best_rule.empty else float('nan')
            last10_mean = float(tail['eval_makespan'].mean()) if not tail.empty else float('nan')
            rows.append({
                'run_id': first.get('run_id'),
                'algorithm': first.get('algorithm'),
                'variant': first.get('variant'),
                'instance_id': instance_id,
                'n_usvs': int(group['n_usvs'].iloc[0]),
                'n_tasks': int(group['n_tasks'].iloc[0]),
                'seed': first.get('seed'),
                'last10_mean': last10_mean,
                'last10_std': float(tail['eval_makespan'].std(ddof=0)) if not tail.empty else float('nan'),
                'last10_min': float(tail['eval_makespan'].min()) if not tail.empty else float('nan'),
                'best_eval_makespan': float(evals['eval_makespan'].min()) if not evals.empty else float('nan'),
                'best_rule_makespan': best_rule,
                'gap_to_best_rule_percent': (
                    (last10_mean - best_rule) / best_rule * 100.0
                    if pd.notna(last10_mean) and pd.notna(best_rule) and best_rule > 0
                    else float('nan')),
                'mean_epoch_time_sec': float(group['epoch_time_sec'].mean()),
                'mean_rollout_time_sec': float(group['rollout_time_sec'].mean()) if 'rollout_time_sec' in group else float('nan'),
                'mean_update_time_sec': float(group['update_time_sec'].mean()) if 'update_time_sec' in group else float('nan'),
                'total_train_time_sec': float(group['elapsed_sec'].dropna().iloc[-1]) if not group['elapsed_sec'].dropna().empty else float('nan'),
                'training_log_path': df.attrs.get('training_log_path'),
            })

    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(summary_path) or '.', exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    return summary


def _instance_curve(df: pd.DataFrame, instance_id: str) -> pd.DataFrame:
    group = df[df['instance_id'] == instance_id].sort_values('epoch')
    return group.dropna(subset=['eval_makespan'])


def plot_training_curves_grid(run: pd.DataFrame, output_dir: str,
                              instances: List[str] = None):
    """2x2 grid: eval makespan vs visit, best-rule line, last-10 shaded."""
    instances = instances or REPRESENTATIVE_INSTANCES
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, instance_id in zip(axes.flat, instances):
        curve = _instance_curve(run, instance_id)
        if curve.empty:
            ax.set_visible(False)
            continue
        ax.plot(curve['visit_index'], curve['eval_makespan'],
                linewidth=1.2, label='Eval makespan')
        best_rule = curve['best_rule_makespan'].dropna()
        if not best_rule.empty:
            ax.axhline(best_rule.iloc[0], color='tab:red', linestyle='--',
                       linewidth=1, label='Best rule')
        last10_start = curve['visit_index'].max() - LAST_N_VISITS + 1
        ax.axvspan(last10_start, curve['visit_index'].max(),
                   alpha=0.12, color='tab:green', label='Last 10 visits')
        ax.set_title(instance_id)
        ax.set_xlabel('Visit')
        ax.set_ylabel('Makespan')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_fig_path(output_dir, 'training_curves'), dpi=200)
    plt.close(fig)


def plot_all25_convergence_grid(run: pd.DataFrame, output_dir: str):
    """5x5 small multiples: gap%% vs visit for every instance."""
    instances = sorted(
        run['instance_id'].unique(),
        key=lambda iid: (run[run['instance_id'] == iid]['n_usvs'].iloc[0],
                         run[run['instance_id'] == iid]['n_tasks'].iloc[0]))
    n = len(instances)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows),
                             sharex=True)
    axes_flat = axes.flat if n > 1 else [axes]
    for ax, instance_id in zip(axes_flat, instances):
        curve = run[run['instance_id'] == instance_id].sort_values('epoch')
        curve = curve.dropna(subset=['gap_to_best_rule_percent'])
        ax.plot(curve['visit_index'], curve['gap_to_best_rule_percent'],
                linewidth=0.9)
        ax.axhline(0, color='tab:red', linestyle='--', linewidth=0.7)
        ax.set_title(instance_id, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)
    fig.supxlabel('Visit')
    fig.supylabel('Gap to Best Rule (%)')
    fig.tight_layout()
    fig.savefig(_fig_path(output_dir, 'all25_convergence'), dpi=200)
    plt.close(fig)


def plot_ablation_curves_grid(frames: List[pd.DataFrame], output_dir: str,
                              instances: List[str] = None):
    """2x2 instances, one curve per ablation variant."""
    instances = instances or REPRESENTATIVE_INSTANCES
    runs = {variant: latest_run(frames, 'PPO', variant)
            for variant in ABLATION_VARIANTS}
    if runs.get('full') is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, instance_id in zip(axes.flat, instances):
        plotted = False
        for variant, run in runs.items():
            if run is None:
                continue
            curve = _instance_curve(run, instance_id)
            if curve.empty:
                continue
            ax.plot(curve['visit_index'], curve['eval_makespan'],
                    linewidth=1.1, label=variant)
            plotted = True
        if not plotted:
            ax.set_visible(False)
            continue
        ax.set_title(instance_id)
        ax.set_xlabel('Visit')
        ax.set_ylabel('Eval Makespan')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_fig_path(output_dir, 'ablation_curves'), dpi=200)
    plt.close(fig)


def _annotated_heatmap(pivot: pd.DataFrame, title: str, cbar_label: str,
                       out_path: str, fmt: str = '{:.1f}'):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)),
                  [str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(int(i)) for i in pivot.index])
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Number of USVs')
    ax.set_title(title)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, fmt.format(value), ha='center', va='center',
                        fontsize=9)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gap_heatmap(main_results_csv: str, output_dir: str):
    """Annotated U x T heatmap of ours gap%% (from main_results.csv)."""
    df = pd.read_csv(main_results_csv, encoding='utf-8-sig')
    pivot = df.pivot_table(index='n_usvs', columns='n_tasks',
                           values='gap_percent')
    _annotated_heatmap(pivot, 'Gap to Best Rule across Scales',
                       'Gap (%)', _fig_path(output_dir, 'gap_heatmap'),
                       fmt='{:+.1f}')


def plot_decision_time_heatmap(decision_time_csv: str, output_dir: str):
    """Annotated U x T heatmap of avg per-decision time (ms)."""
    df = pd.read_csv(decision_time_csv, encoding='utf-8-sig')
    pivot = df.pivot_table(index='n_usvs', columns='n_tasks',
                           values='eval_time_per_decision_ms_mean')
    _annotated_heatmap(pivot, 'Average Decision Time across Scales',
                       'ms / decision',
                       _fig_path(output_dir, 'decision_time_heatmap'),
                       fmt='{:.2f}')


def plot_drl_gap_violin(frames: List[pd.DataFrame], output_dir: str):
    """Violin per method of pooled last-10 gap%% (25 instances x 10 visits)."""
    methods = [('Ours', latest_run(frames, 'PPO', 'full'))]
    methods += [(alg, latest_run(frames, alg, 'baseline'))
                for alg in DRL_METHODS]

    data, labels = [], []
    for label, run in methods:
        if run is None:
            continue
        pooled = []
        for _, group in run.groupby('instance_id'):
            tail = group.sort_values('epoch').dropna(
                subset=['gap_to_best_rule_percent']).tail(LAST_N_VISITS)
            pooled.extend(tail['gap_to_best_rule_percent'].tolist())
        if pooled:
            data.append(pooled)
            labels.append(label)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, showmedians=True)
    ax.axhline(0, color='tab:red', linestyle='--', linewidth=1)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel('Gap to Best Rule (%)')
    ax.set_title('Distribution of Last-10-Visit Gaps per Method')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(_fig_path(output_dir, 'drl_gap_violin'), dpi=200)
    plt.close(fig)


def plot_gap_by_tasks(summary: pd.DataFrame, output_dir: str):
    """Boxplot of ours gap%% grouped by task count (PPO/full rows only)."""
    plot_df = summary[(summary['algorithm'] == 'PPO') &
                      (summary['variant'] == 'full')]
    plot_df = plot_df.dropna(subset=['gap_to_best_rule_percent', 'n_tasks'])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = [group['gap_to_best_rule_percent'].values
              for _, group in plot_df.groupby('n_tasks')]
    labels = [str(int(n)) for n in sorted(plot_df['n_tasks'].unique())]
    ax.boxplot(groups, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.axhline(0, color='tab:red', linestyle='--', linewidth=1)
    ax.set_title('Gap to Best Rule by Task Scale')
    ax.set_xlabel('Number of Tasks')
    ax.set_ylabel('Gap to Best Rule (%)')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(_fig_path(output_dir, 'gap_by_tasks'), dpi=200)
    plt.close(fig)


def plot_scalability(main_results_csv: str, scalability_csv: str,
                     output_dir: str):
    """(a) per-method gap%% vs M (zero-shot shaded); (b) log solve time vs M."""
    frames = []
    if main_results_csv and os.path.exists(main_results_csv):
        df = pd.read_csv(main_results_csv, encoding='utf-8-sig')
        df = df.rename(columns={
            'gap_percent': 'ours_gap',
            'ours_epoch_time_mean_sec': '_unused',
        })
        frames.append(df.assign(regime='training'))
    scal = None
    if scalability_csv and os.path.exists(scalability_csv):
        scal = pd.read_csv(scalability_csv, encoding='utf-8-sig')
    if not frames and scal is None:
        return

    train_max_tasks = frames[0]['n_tasks'].max() if frames else None

    fig, (ax_gap, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) gap vs M per method (mean over fleet sizes)
    if frames is not None and frames:
        train_gap = frames[0].groupby('n_tasks')['ours_gap'].mean()
        ax_gap.plot(train_gap.index, train_gap.values, marker='o',
                    label='Ours (training scales)')
    if scal is not None:
        methods = [('ppo_gap_percent', 'Ours (zero-shot)'),
                   ('a2c_gap_percent', 'A2C'), ('dqn_gap_percent', 'DQN'),
                   ('ddqn_gap_percent', 'DDQN'),
                   ('reinforce_gap_percent', 'REINFORCE')]
        for col, label in methods:
            if col in scal:
                series = scal.groupby('n_tasks')[col].mean().dropna()
                if not series.empty:
                    ax_gap.plot(series.index, series.values, marker='s',
                                linewidth=1.1, label=label)
    ax_gap.axhline(0, color='tab:red', linestyle='--', linewidth=1)
    if train_max_tasks is not None and scal is not None:
        ax_gap.axvspan(train_max_tasks, scal['n_tasks'].max(),
                       alpha=0.08, color='tab:orange')
    ax_gap.set_xlabel('Number of Tasks')
    ax_gap.set_ylabel('Gap to Best Rule (%)')
    ax_gap.set_title('(a) Solution quality across scales')
    ax_gap.legend(fontsize=7)
    ax_gap.grid(alpha=0.25)

    # (b) solve time vs M, log axis
    if scal is not None:
        time_cols = [('best_rule_solve_time_sec', 'Best rule'),
                     ('ppo_solve_time_sec', 'Ours'),
                     ('a2c_solve_time_sec', 'A2C'),
                     ('dqn_solve_time_sec', 'DQN'),
                     ('ddqn_solve_time_sec', 'DDQN'),
                     ('reinforce_solve_time_sec', 'REINFORCE')]
        for col, label in time_cols:
            if col in scal:
                series = scal.groupby('n_tasks')[col].mean().dropna()
                if not series.empty:
                    ax_time.plot(series.index, series.values, marker='o',
                                 linewidth=1.1, label=label)
        ax_time.set_yscale('log')
    ax_time.set_xlabel('Number of Tasks')
    ax_time.set_ylabel('Solution Time (s, log scale)')
    ax_time.set_title('(b) Zero-shot solution time across scales')
    ax_time.legend(fontsize=7)
    ax_time.grid(alpha=0.25, which='both')

    fig.tight_layout()
    fig.savefig(_fig_path(output_dir, 'scalability'), dpi=200)
    plt.close(fig)


def analyze(args):
    global FIG_FORMAT
    FIG_FORMAT = args.format
    os.makedirs(args.output_dir, exist_ok=True)

    if args.main_results_csv or args.scalability_csv:
        plot_scalability(args.main_results_csv, args.scalability_csv,
                         args.output_dir)
    if args.main_results_csv and os.path.exists(args.main_results_csv):
        plot_gap_heatmap(args.main_results_csv, args.output_dir)
    if args.decision_time_csv and os.path.exists(args.decision_time_csv):
        plot_decision_time_heatmap(args.decision_time_csv, args.output_dir)

    frames = load_logs(args.log_dir, args.algorithm, args.variant)
    if not frames:
        print(f"[Analyze] No training logs under {args.log_dir}; "
              f"CSV-based figures saved under: {args.output_dir}")
        return

    summary_path = os.path.join(args.log_dir, 'summary.csv')
    summary = build_summary(frames, summary_path)

    ours = latest_run(frames, 'PPO', 'full')
    if ours is not None:
        plot_training_curves_grid(ours, args.output_dir)
        plot_all25_convergence_grid(ours, args.output_dir)
    plot_ablation_curves_grid(frames, args.output_dir)
    plot_drl_gap_violin(frames, args.output_dir)
    plot_gap_by_tasks(summary, args.output_dir)

    print(f"[Analyze] Loaded runs: {len(frames)}")
    print(f"[Analyze] Summary saved: {summary_path}")
    print(f"[Analyze] Figures saved under: {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', default='results/training_logs')
    parser.add_argument('--output-dir', default='results/figures')
    parser.add_argument('--algorithm', default=None)
    parser.add_argument('--variant', default=None)
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png'],
                        help='Figure file format (pdf feeds the paper directly)')
    parser.add_argument('--main-results-csv', default=None,
                        help='results/main_results.csv (gap heatmap + scalability)')
    parser.add_argument('--scalability-csv', default=None,
                        help='results/scalability_summary.csv')
    parser.add_argument('--decision-time-csv', default=None,
                        help='results/decision_time_grid.csv (decision-time heatmap)')
    return parser


if __name__ == '__main__':
    analyze(build_parser().parse_args())
