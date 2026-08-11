"""Generate the paper figures from training logs and result CSVs.

One figure per command (all defaults point at the standard result paths, so a
bare command works on any OS without line continuations):

    python analyze_training_logs.py training_curves
    python analyze_training_logs.py convergence_all25
    python analyze_training_logs.py ablation_curves
    python analyze_training_logs.py gap_heatmap
    python analyze_training_logs.py decision_time_heatmap
    python analyze_training_logs.py drl_gap_violin
    python analyze_training_logs.py gap_by_tasks
    python analyze_training_logs.py scalability
    python analyze_training_logs.py gantt --gantt-instance u6_t60
    python analyze_training_logs.py summary
    python analyze_training_logs.py all

Outputs land in results/figures/*.pdf and map 1:1 to the paper's
figures/fig_*.tex placeholders.

Style: Okabe-Ito colorblind-safe categorical palette with a fixed
method-to-color mapping (identity is never re-assigned across figures),
per-method markers as a secondary encoding, a diverging RdBu map centered at
zero for gap polarity, a perceptually uniform sequential map for magnitudes,
serif/STIX typography matched to the LaTeX body, and embedded (Type-42) fonts.
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches

FIG_FORMAT = 'pdf'

REPRESENTATIVE_INSTANCES = ['u2_t20', 'u4_t60', 'u8_t80', 'u10_t100']
ABLATION_VARIANTS = ['full', 'no_hgnn', 'shared_encoder', 'no_reward_norm']
DRL_METHODS = ['A2C', 'DQN', 'DDQN', 'REINFORCE']
LAST_N_VISITS = 10

# Okabe-Ito palette, validated (CVD-adjacent worst pair sits in the 6-8 band,
# which is legal only with a secondary encoding -> every multi-series line/
# violin figure also assigns a fixed per-method marker/position).
METHOD_COLORS = {
    'Ours': '#0072B2',
    'A2C': '#E69F00',
    'DQN': '#56B4E9',
    'DDQN': '#009E73',
    'REINFORCE': '#CC79A7',
    'Best rule': '#555555',
}
METHOD_MARKERS = {
    'Ours': 'o', 'A2C': 's', 'DQN': '^', 'DDQN': 'D', 'REINFORCE': 'v',
    'Best rule': 'x',
}
VARIANT_COLORS = {
    'full': '#0072B2',
    'no_hgnn': '#D55E00',
    'shared_encoder': '#009E73',
    'no_reward_norm': '#CC79A7',
}
VARIANT_LABELS = {
    'full': 'Full (proposed)',
    'no_hgnn': 'NoHGNN',
    'shared_encoder': 'SharedEncoder',
    'no_reward_norm': 'NoRewardNorm',
}
VARIANT_STYLES = {
    'full': '-', 'no_hgnn': '--', 'shared_encoder': '-.', 'no_reward_norm': ':',
}
REFERENCE_COLOR = '#555555'
ZERO_LINE_COLOR = '#999999'

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


def _set_style():
    """Journal figure style: serif/STIX, embedded fonts, recessive axes."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 8.5,
        'axes.titlesize': 9,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'grid.color': '#c9c9c9',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


def _fig_path(output_dir: str, stem: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f'{stem}.{FIG_FORMAT}')


def _save(fig, output_dir: str, stem: str):
    path = _fig_path(output_dir, stem)
    fig.savefig(path)
    plt.close(fig)
    print(f'[Figure] {path}')


def _safe_name(text: object) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_')


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_logs(log_dir: str, algorithm: str = None, variant: str = None,
              protocol: str = 'round_robin') -> List[pd.DataFrame]:
    """Load per-run DataFrames; a run is labeled by (algorithm, variant).

    Only rows matching `protocol` are kept (default: round_robin, the paper
    protocol). Logs written before the round-robin upgrade have no `protocol`
    column and are skipped entirely, so stale single-instance runs in the same
    directory can never leak into the figures.
    """
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
        if protocol:
            if 'protocol' not in df.columns:
                continue
            df = df[df['protocol'] == protocol]
            if df.empty:
                continue
            df = df.reset_index(drop=True)
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


def _run_timestamp(df: pd.DataFrame) -> str:
    """Trailing YYYYMMDD_HHMMSS of the run_id (falls back to the whole id)."""
    run_id = str(df.iloc[0].get('run_id'))
    match = re.search(r'(\d{8}_\d{6})$', run_id)
    return match.group(1) if match else run_id


def latest_run(frames: List[pd.DataFrame], algorithm: str,
               variant: str) -> Optional[pd.DataFrame]:
    """Most recent run for (algorithm, variant), by run-id timestamp."""
    candidates = [df for df in frames
                  if str(df.iloc[0].get('algorithm')) == algorithm
                  and str(df.iloc[0].get('variant')) == variant]
    if not candidates:
        return None
    return max(candidates, key=_run_timestamp)


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
    print(f'[Summary] {summary_path} ({len(summary)} rows)')
    return summary


def _instance_curve(df: pd.DataFrame, instance_id: str) -> pd.DataFrame:
    group = df[df['instance_id'] == instance_id].sort_values('epoch')
    return group.dropna(subset=['eval_makespan'])


def _require_run(args, algorithm='PPO', variant='full') -> pd.DataFrame:
    frames = load_logs(args.log_dir)
    run = latest_run(frames, algorithm, variant)
    if run is None:
        raise SystemExit(
            f'No round-robin training log for ({algorithm}, {variant}) '
            f'under {args.log_dir}')
    if 'visit_index' not in run.columns or run['visit_index'].isna().all():
        raise SystemExit(
            f'Selected log {run.attrs.get("training_log_path")} has no '
            f'visit_index data -- it is not a round-robin training log')
    return run


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_training_curves_grid(run: pd.DataFrame, output_dir: str,
                              instances: List[str] = None):
    """2x2 grid: eval makespan vs visit, best-rule line, last-10 shaded."""
    instances = instances or REPRESENTATIVE_INSTANCES
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for ax, instance_id in zip(axes.flat, instances):
        curve = _instance_curve(run, instance_id)
        if curve.empty:
            ax.set_visible(False)
            continue
        last10_start = curve['visit_index'].max() - LAST_N_VISITS + 1
        ax.axvspan(last10_start, curve['visit_index'].max(),
                   color='#0072B2', alpha=0.08, lw=0, label='Last 10 visits')
        best_rule = curve['best_rule_makespan'].dropna()
        if not best_rule.empty:
            ax.axhline(best_rule.iloc[0], color=REFERENCE_COLOR,
                       linestyle='--', linewidth=1.0, label='Best rule')
        ax.plot(curve['visit_index'], curve['eval_makespan'],
                color=METHOD_COLORS['Ours'], linewidth=1.2,
                label='Eval makespan')
        ax.set_title(instance_id.replace('_', r'\_')
                     if plt.rcParams['text.usetex'] else instance_id)
        ax.set_xlabel('Visit')
        ax.set_ylabel('Makespan')
        ax.grid(True, axis='y')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = [labels.index(l) for l in
             ['Eval makespan', 'Best rule', 'Last 10 visits'] if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, output_dir, 'training_curves')


def plot_all25_convergence_grid(run: pd.DataFrame, output_dir: str):
    """5x5 small multiples: gap% vs visit for every instance."""
    meta = (run.groupby('instance_id')
            .agg(n_usvs=('n_usvs', 'first'), n_tasks=('n_tasks', 'first')))
    instances = list(meta.sort_values(['n_usvs', 'n_tasks']).index)
    n = len(instances)
    ncols = 5
    nrows = max((n + ncols - 1) // ncols, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 1.35 * nrows + 0.5),
                             sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, instance_id in zip(axes_flat, instances):
        curve = run[run['instance_id'] == instance_id].sort_values('epoch')
        curve = curve.dropna(subset=['gap_to_best_rule_percent'])
        ax.axhline(0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=0.6)
        ax.plot(curve['visit_index'], curve['gap_to_best_rule_percent'],
                color=METHOD_COLORS['Ours'], linewidth=0.8)
        ax.set_title(instance_id, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis='y')
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.supxlabel('Visit', fontsize=8.5)
    fig.supylabel('Gap to best rule (%)', fontsize=8.5)
    fig.tight_layout()
    _save(fig, output_dir, 'all25_convergence')


def plot_ablation_curves_grid(frames: List[pd.DataFrame], output_dir: str,
                              instances: List[str] = None):
    """2x2 instances, one curve per ablation variant."""
    instances = instances or REPRESENTATIVE_INSTANCES
    runs = {variant: latest_run(frames, 'PPO', variant)
            for variant in ABLATION_VARIANTS}
    if runs.get('full') is None:
        raise SystemExit('No PPO/full run found for the ablation figure')
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for ax, instance_id in zip(axes.flat, instances):
        plotted = False
        for variant in ABLATION_VARIANTS:
            run = runs.get(variant)
            if run is None:
                continue
            curve = _instance_curve(run, instance_id)
            if curve.empty:
                continue
            ax.plot(curve['visit_index'], curve['eval_makespan'],
                    color=VARIANT_COLORS[variant],
                    linestyle=VARIANT_STYLES[variant],
                    linewidth=1.2, label=VARIANT_LABELS[variant])
            plotted = True
        if not plotted:
            ax.set_visible(False)
            continue
        ax.set_title(instance_id)
        ax.set_xlabel('Visit')
        ax.set_ylabel('Eval makespan')
        ax.grid(True, axis='y')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center',
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, output_dir, 'ablation_curves')


def _annotated_heatmap(pivot: pd.DataFrame, cbar_label: str, out_stem: str,
                       output_dir: str, cmap, norm=None, fmt='{:.1f}'):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)),
                  [str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(int(i)) for i in pivot.index])
    ax.set_xlabel('Number of tasks $M$')
    ax.set_ylabel('Number of USVs $N$')
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if pd.isna(value):
                continue
            r, g, b, _ = im.cmap(im.norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, fmt.format(value), ha='center', va='center',
                    fontsize=7.5,
                    color='white' if luminance < 0.5 else '#1a1a1a')
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, output_dir, out_stem)


def plot_gap_heatmap(main_results_csv: str, output_dir: str):
    """Diverging U x T heatmap of ours gap% (polarity around zero)."""
    df = pd.read_csv(main_results_csv, encoding='utf-8-sig')
    pivot = df.pivot_table(index='n_usvs', columns='n_tasks',
                           values='gap_percent')
    span = float(np.nanmax(np.abs(pivot.values))) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    _annotated_heatmap(pivot, 'Gap to best rule (%)', 'gap_heatmap',
                       output_dir, cmap='RdBu_r', norm=norm, fmt='{:+.1f}')


def plot_decision_time_heatmap(decision_time_csv: str, output_dir: str):
    """Sequential U x T heatmap of avg per-decision time (ms)."""
    df = pd.read_csv(decision_time_csv, encoding='utf-8-sig')
    pivot = df.pivot_table(index='n_usvs', columns='n_tasks',
                           values='eval_time_per_decision_ms_mean')
    _annotated_heatmap(pivot, 'Decision time (ms)', 'decision_time_heatmap',
                       output_dir, cmap='viridis', fmt='{:.2f}')


def plot_drl_gap_violin(frames: List[pd.DataFrame], output_dir: str):
    """Violin per method of pooled last-10 gap% (25 instances x 10 visits)."""
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
        raise SystemExit('No gap data found for the violin figure')

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=0.8)
    parts = ax.violinplot(data, showmedians=True, showextrema=False,
                          widths=0.75)
    for body, label in zip(parts['bodies'], labels):
        body.set_facecolor(METHOD_COLORS[label])
        body.set_alpha(0.75)
        body.set_edgecolor('#333333')
        body.set_linewidth(0.6)
    parts['cmedians'].set_color('#1a1a1a')
    parts['cmedians'].set_linewidth(1.0)
    for idx, values in enumerate(data, start=1):
        ax.text(idx + 0.42, np.median(values), f'{np.median(values):+.1f}',
                fontsize=7, va='center', ha='left', color='#1a1a1a')
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel('Gap to best rule (%)')
    ax.grid(True, axis='y')
    fig.tight_layout()
    _save(fig, output_dir, 'drl_gap_violin')


def plot_gap_by_tasks(summary: pd.DataFrame, output_dir: str):
    """Boxplot of ours gap% grouped by task count (PPO/full rows only)."""
    plot_df = summary[(summary['algorithm'] == 'PPO') &
                      (summary['variant'] == 'full')]
    plot_df = plot_df.dropna(subset=['gap_to_best_rule_percent', 'n_tasks'])
    if plot_df.empty:
        raise SystemExit('No PPO/full summary rows for the boxplot')
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axhline(0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=0.8)
    task_counts = sorted(plot_df['n_tasks'].unique())
    groups = [plot_df[plot_df['n_tasks'] == n]['gap_to_best_rule_percent'].values
              for n in task_counts]
    box = ax.boxplot(groups, showmeans=True, widths=0.55, patch_artist=True,
                     medianprops=dict(color='#1a1a1a', linewidth=1.0),
                     meanprops=dict(marker='D', markerfacecolor='#ffffff',
                                    markeredgecolor='#1a1a1a', markersize=4),
                     flierprops=dict(marker='o', markersize=3,
                                     markerfacecolor='none',
                                     markeredgecolor='#777777'))
    for patch in box['boxes']:
        patch.set_facecolor(METHOD_COLORS['Ours'])
        patch.set_alpha(0.55)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(0.7)
    for whisk in box['whiskers'] + box['caps']:
        whisk.set_color('#333333')
        whisk.set_linewidth(0.7)
    ax.set_xticks(range(1, len(task_counts) + 1),
                  [str(int(n)) for n in task_counts])
    ax.set_xlabel('Number of tasks $M$')
    ax.set_ylabel('Gap to best rule (%)')
    ax.grid(True, axis='y')
    fig.tight_layout()
    _save(fig, output_dir, 'gap_by_tasks')


def plot_scalability(main_results_csv: str, scalability_csv: str,
                     output_dir: str):
    """(a) per-method gap% vs M (zero-shot shaded); (b) log solve time vs M."""
    train_df = (pd.read_csv(main_results_csv, encoding='utf-8-sig')
                if main_results_csv and os.path.exists(main_results_csv)
                else None)
    scal = (pd.read_csv(scalability_csv, encoding='utf-8-sig')
            if scalability_csv and os.path.exists(scalability_csv) else None)
    if train_df is None and scal is None:
        raise SystemExit('Neither main_results.csv nor scalability_summary.csv found')

    fig, (ax_gap, ax_time) = plt.subplots(1, 2, figsize=(6.8, 2.9))

    train_max_tasks = train_df['n_tasks'].max() if train_df is not None else None
    zero_shot_max = scal['n_tasks'].max() if scal is not None else None

    for ax in (ax_gap, ax_time):
        if train_max_tasks is not None and zero_shot_max is not None:
            ax.axvspan(train_max_tasks, zero_shot_max, color='#E69F00',
                       alpha=0.07, lw=0)

    # (a) gap vs M per method (mean over fleet sizes)
    ax_gap.axhline(0, color=ZERO_LINE_COLOR, linestyle='--', linewidth=0.8)
    if train_df is not None:
        series = train_df.groupby('n_tasks')['gap_percent'].mean()
        ax_gap.plot(series.index, series.values,
                    color=METHOD_COLORS['Ours'], marker=METHOD_MARKERS['Ours'],
                    markersize=3.5, linewidth=1.3, label='Ours')
    if scal is not None:
        gap_cols = [('ppo_gap_percent', 'Ours'), ('a2c_gap_percent', 'A2C'),
                    ('dqn_gap_percent', 'DQN'), ('ddqn_gap_percent', 'DDQN'),
                    ('reinforce_gap_percent', 'REINFORCE')]
        for col, label in gap_cols:
            if col not in scal:
                continue
            series = scal.groupby('n_tasks')[col].mean().dropna()
            if series.empty:
                continue
            ax_gap.plot(series.index, series.values,
                        color=METHOD_COLORS[label],
                        marker=METHOD_MARKERS[label], markersize=3.5,
                        linewidth=1.1, linestyle='--' if label == 'Ours' else '-',
                        label=f'{label} (zero-shot)' if label == 'Ours' else label)
    ax_gap.set_xlabel('Number of tasks $M$')
    ax_gap.set_ylabel('Gap to best rule (%)')
    ax_gap.set_title('(a) Solution quality across scales', fontsize=8.5)
    ax_gap.legend(ncol=2, columnspacing=0.8, handlelength=1.6)
    ax_gap.grid(True, axis='y')

    # (b) solve time vs M, log axis
    if scal is not None:
        time_cols = [('best_rule_solve_time_sec', 'Best rule'),
                     ('ppo_solve_time_sec', 'Ours'),
                     ('a2c_solve_time_sec', 'A2C'),
                     ('dqn_solve_time_sec', 'DQN'),
                     ('ddqn_solve_time_sec', 'DDQN'),
                     ('reinforce_solve_time_sec', 'REINFORCE')]
        for col, label in time_cols:
            if col not in scal:
                continue
            series = scal.groupby('n_tasks')[col].mean().dropna()
            if series.empty:
                continue
            ax_time.plot(series.index, series.values,
                         color=METHOD_COLORS[label],
                         marker=METHOD_MARKERS[label], markersize=3.5,
                         linewidth=1.1,
                         linestyle='--' if label == 'Best rule' else '-',
                         label=label)
        ax_time.set_yscale('log')
    ax_time.set_xlabel('Number of tasks $M$')
    ax_time.set_ylabel('Solution time (s)')
    ax_time.set_title('(b) Zero-shot solution time', fontsize=8.5)
    ax_time.legend(ncol=2, columnspacing=0.8, handlelength=1.6)
    ax_time.grid(True, which='both', axis='y')

    fig.tight_layout()
    _save(fig, output_dir, 'scalability')


# --------------------------------------------------------------------------
# Gantt comparison (best rule vs ours) -- needs torch + a trained checkpoint
# --------------------------------------------------------------------------

def _draw_gantt_panel(ax, env, title: str):
    """Render one schedule from env.usv_history onto ax."""
    cmap = plt.get_cmap('tab20')
    max_time = 0.0
    for usv_id in range(env.n_usvs):
        for event in env.usv_history[usv_id]:
            start, end = event['start'], event['end']
            duration = end - start
            max_time = max(max_time, end)
            if duration <= 0.01:
                continue
            if event['type'] == 'task':
                tid = (int(event['info'].replace('T', ''))
                       if 'T' in str(event['info']) else 0)
                ax.barh(usv_id, duration, left=start, height=0.62,
                        color=cmap(tid % 20), edgecolor='white',
                        linewidth=0.4)
            elif event['type'] == 'move':
                ax.barh(usv_id, duration, left=start, height=0.36,
                        color='#c9c9c9', edgecolor='none', hatch='////',
                        alpha=0.85)
            elif event['type'] == 'charge':
                ax.barh(usv_id, duration, left=start, height=0.62,
                        color='#F0E442', edgecolor='#8a7d00', linewidth=0.5)
    ax.set_yticks(range(env.n_usvs), [f'USV {i}' for i in range(env.n_usvs)])
    ax.set_xlim(0, max_time * 1.03)
    ax.set_title(title, fontsize=8.5, loc='left')
    ax.grid(True, axis='x')
    ax.tick_params(length=0)
    return max_time


def plot_gantt_comparison(args, output_dir: str):
    """Two-panel Gantt: (a) best rule schedule, (b) learned schedule."""
    import torch

    from env import USVSchedulingEnv
    from main import build_agent, evaluate_agent_once
    from multi_train import load_public_instances
    from scheduling_rules import get_all_rules, run_scheduling
    from config import get_config

    instance = load_public_instances(args.data_dir,
                                     [args.gantt_instance])[0]

    # Best rule: pick by makespan, keep the winning env for its history.
    best_env, best_name, best_makespan = None, None, float('inf')
    for rule in get_all_rules():
        if rule.name == 'Random':
            np.random.seed(0)
        env = USVSchedulingEnv(instance)
        result = run_scheduling(env, rule)
        makespan = result['makespan'] if result['success'] else float('inf')
        if makespan < best_makespan:
            best_env, best_name, best_makespan = env, rule.name, makespan

    # Ours: deterministic episode with the instance's best checkpoint.
    cfg = get_config(n_usvs=instance['n_usvs'], n_tasks=instance['n_tasks'],
                     model_dir=args.model_dir, hidden_dim=args.hidden_dim,
                     hgnn_layers=args.hgnn_layers, n_heads=args.n_heads)
    ckpt = os.path.join(args.model_dir,
                        f'best_{args.gantt_instance}_seed{args.seed}.pth')
    if not os.path.exists(ckpt):
        raise SystemExit(f'Checkpoint not found: {ckpt}')
    torch.set_num_threads(1)
    agent = build_agent(cfg, instance, device='cpu')
    agent.load(ckpt)

    ours_env = USVSchedulingEnv(instance)
    state = ours_env.reset()
    done, step = False, 0
    with torch.no_grad():
        while not done and step < ours_env.n_tasks * 10:
            task_mask, _ = agent._get_masks(ours_env)
            if task_mask.sum() == 0:
                break
            action, _, _ = agent.select_action(ours_env, state,
                                               deterministic=True)
            state, _, done, info = ours_env.step(action[0], action[1])
            step += 1
    ours_makespan = info.get('makespan', float('nan'))

    fig, (ax_rule, ax_ours) = plt.subplots(
        2, 1, figsize=(6.8, 1.2 + 0.42 * instance['n_usvs']), sharex=True)
    t1 = _draw_gantt_panel(
        ax_rule, best_env,
        f'(a) Best rule ({best_name.replace("_", "--")}), '
        f'$C_{{\\max}}$ = {best_makespan:.1f}')
    t2 = _draw_gantt_panel(
        ax_ours, ours_env,
        f'(b) Proposed method, $C_{{\\max}}$ = {ours_makespan:.1f}')
    xmax = max(t1, t2) * 1.03
    ax_rule.set_xlim(0, xmax)
    ax_ours.set_xlim(0, xmax)
    ax_ours.set_xlabel('Time')

    legend_patches = [
        mpatches.Patch(facecolor=plt.get_cmap('tab20')(0),
                       edgecolor='white', label='Task'),
        mpatches.Patch(facecolor='#c9c9c9', hatch='////', alpha=0.85,
                       label='Travel'),
        mpatches.Patch(facecolor='#F0E442', edgecolor='#8a7d00',
                       label='Battery swap'),
    ]
    fig.legend(handles=legend_patches, ncol=3, loc='lower center',
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, output_dir, 'gantt_comparison')


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

FIGURES = ['training_curves', 'convergence_all25', 'ablation_curves',
           'gap_heatmap', 'decision_time_heatmap', 'drl_gap_violin',
           'gap_by_tasks', 'scalability', 'gantt', 'summary']


def run_figure(name: str, args):
    if name == 'training_curves':
        plot_training_curves_grid(_require_run(args), args.output_dir)
    elif name == 'convergence_all25':
        plot_all25_convergence_grid(_require_run(args), args.output_dir)
    elif name == 'ablation_curves':
        plot_ablation_curves_grid(load_logs(args.log_dir), args.output_dir)
    elif name == 'gap_heatmap':
        plot_gap_heatmap(args.main_results_csv, args.output_dir)
    elif name == 'decision_time_heatmap':
        plot_decision_time_heatmap(args.decision_time_csv, args.output_dir)
    elif name == 'drl_gap_violin':
        plot_drl_gap_violin(load_logs(args.log_dir), args.output_dir)
    elif name == 'gap_by_tasks':
        frames = load_logs(args.log_dir)
        summary = build_summary(
            frames, os.path.join(args.log_dir, 'summary.csv'))
        plot_gap_by_tasks(summary, args.output_dir)
    elif name == 'scalability':
        plot_scalability(args.main_results_csv, args.scalability_csv,
                         args.output_dir)
    elif name == 'gantt':
        plot_gantt_comparison(args, args.output_dir)
    elif name == 'summary':
        build_summary(load_logs(args.log_dir),
                      os.path.join(args.log_dir, 'summary.csv'))
    else:
        raise SystemExit(f'Unknown figure: {name}')


def main(args):
    global FIG_FORMAT
    FIG_FORMAT = args.format
    _set_style()

    if args.figure == 'all':
        for name in FIGURES:
            if name == 'gantt' and not os.path.exists(os.path.join(
                    args.model_dir,
                    f'best_{args.gantt_instance}_seed{args.seed}.pth')):
                print('[Skip] gantt: checkpoint not found '
                      '(run it separately once models exist)')
                continue
            try:
                run_figure(name, args)
            except SystemExit as exc:
                print(f'[Skip] {name}: {exc}')
    else:
        run_figure(args.figure, args)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('figure', choices=FIGURES + ['all'],
                        help='Which figure to generate')
    parser.add_argument('--log-dir', default=os.path.join('results', 'training_logs'))
    parser.add_argument('--output-dir', default=os.path.join('results', 'figures'))
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png'])
    parser.add_argument('--main-results-csv',
                        default=os.path.join('results', 'main_results.csv'))
    parser.add_argument('--scalability-csv',
                        default=os.path.join('results', 'scalability_summary.csv'))
    parser.add_argument('--decision-time-csv',
                        default=os.path.join('results', 'decision_time_grid.csv'))
    parser.add_argument('--data-dir', default=os.path.join('data', 'public'))
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--gantt-instance', default='u6_t60',
                        help='Instance for the Gantt comparison figure')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hgnn-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
