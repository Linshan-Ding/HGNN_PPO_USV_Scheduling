"""
Round-robin multi-instance PPO training over the 25 public instances.

Protocol (paper Sec. "Protocol and metrics"): one training run per learning
method, seed 0 by default, K epochs cycling the manifest-ordered instances
(epoch k trains instance (k-1) mod 25; K=5000 gives 200 visits per instance).
Every epoch ends with one deterministic evaluation of the trained instance;
per-instance best checkpoints are saved as models/best_{instance_id}_seed{s}.pth
(the exact names scalability_experiment.py loads for zero-shot evaluation).
Per-instance reported results are extracted later from the training log
(last 10 visits) by extract_results.py -- no separate evaluation pass.

Ablation variants reuse this trainer via --variant.
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Optional

import numpy as np

from config import get_config

RULES_RESULTS_FIELDS = [
    'instance_id', 'n_usvs', 'n_tasks', 'rule_name',
    'makespan', 'solve_time_sec', 'is_best_rule',
]


def load_public_instances(data_dir: str,
                          instance_ids: Optional[List[str]] = None) -> List[dict]:
    """Load benchmark instances in manifest order without mutating any cfg."""
    from instance_generator import InstanceGenerator

    manifest_path = os.path.join(data_dir, 'manifest.csv')
    with open(manifest_path, newline='', encoding='utf-8-sig') as f:
        manifest_rows = list(csv.DictReader(f))

    wanted = set(instance_ids) if instance_ids else None
    instances = []
    for row in manifest_rows:
        if wanted and row['instance_id'] not in wanted:
            continue
        instance = InstanceGenerator.load_from_csv(
            os.path.join(data_dir, row['filename']))
        instance['instance_id'] = row['instance_id']
        instances.append(instance)

    if wanted and len(instances) != len(wanted):
        found = {inst['instance_id'] for inst in instances}
        raise ValueError(f"Instances not in manifest: {sorted(wanted - found)}")
    if not instances:
        raise ValueError(f"No instances loaded from {manifest_path}")
    return instances


def evaluate_all_rules_timed(instance: dict, random_seed: int) -> dict:
    """Run every rule with one untimed warm-up plus one timed run."""
    from env import USVSchedulingEnv
    from scheduling_rules import get_all_rules, run_scheduling

    rules = {}
    for rule in get_all_rules():
        if rule.name == 'Random':
            np.random.seed(random_seed)
        run_scheduling(USVSchedulingEnv(instance), rule)  # warm-up
        if rule.name == 'Random':
            np.random.seed(random_seed)
        result = run_scheduling(USVSchedulingEnv(instance), rule)
        makespan = result['makespan'] if result['success'] else float('inf')
        rules[rule.name] = {
            'makespan': makespan,
            'solve_time_sec': result['solve_time_sec'],
        }

    best_name = min(rules, key=lambda name: rules[name]['makespan'])
    return {
        'rules': rules,
        'best_rule_name': best_name,
        'best_rule_makespan': rules[best_name]['makespan'],
        'random_makespan': rules['Random']['makespan'],
    }


def ensure_rules_results(instances: List[dict],
                         path: str = os.path.join('results', 'rules_results.csv'),
                         random_seed: int = 20260519,
                         force: bool = False) -> Dict[str, dict]:
    """Compute (or load cached) timed rule baselines for every instance."""
    by_instance: Dict[str, dict] = {}

    if os.path.isfile(path) and not force:
        with open(path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                entry = by_instance.setdefault(row['instance_id'], {
                    'rules': {}, 'best_rule_name': None,
                    'best_rule_makespan': float('inf'), 'random_makespan': None,
                })
                makespan = float(row['makespan'])
                entry['rules'][row['rule_name']] = {
                    'makespan': makespan,
                    'solve_time_sec': float(row['solve_time_sec']),
                }
                if row['rule_name'] == 'Random':
                    entry['random_makespan'] = makespan
                if makespan < entry['best_rule_makespan']:
                    entry['best_rule_makespan'] = makespan
                    entry['best_rule_name'] = row['rule_name']
        if all(inst['instance_id'] in by_instance for inst in instances):
            print(f"[Rules] Loaded cached baselines: {path}")
            return by_instance
        by_instance = {}

    print(f"[Rules] Computing timed rule baselines for {len(instances)} instances")
    rows = []
    for instance in instances:
        stats = evaluate_all_rules_timed(instance, random_seed)
        by_instance[instance['instance_id']] = stats
        for rule_name, rule_stats in stats['rules'].items():
            rows.append({
                'instance_id': instance['instance_id'],
                'n_usvs': instance['n_usvs'],
                'n_tasks': instance['n_tasks'],
                'rule_name': rule_name,
                'makespan': rule_stats['makespan'],
                'solve_time_sec': rule_stats['solve_time_sec'],
                'is_best_rule': int(rule_name == stats['best_rule_name']),
            })
        print(f"  {instance['instance_id']}: best={stats['best_rule_name']} "
              f"({stats['best_rule_makespan']:.1f})")

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=RULES_RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Rules] Saved: {path}")
    return by_instance


def train_round_robin(cfg, instances: List[dict],
                      rules_by_instance: Dict[str, dict]) -> dict:
    """Train one PPO policy over all instances in round-robin order."""
    import torch

    from main import (VisdomLogger, evaluate_agent_once, set_global_seed)
    from parallel_rollout import ParallelRolloutCollector
    from ppo import PPOAgent
    from training_logger import TrainingCSVLogger, make_training_run_id

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    seed = cfg.train.get('seed', 0)
    variant = cfg.network.get('ablation_variant', 'full')
    variant_tag = '' if variant == 'full' else f'_{variant}'
    reward_norm = getattr(cfg.instance, 'reward_normalization', True)
    for instance in instances:
        instance['config'].reward_normalization = reward_norm

    set_global_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    print(f"[RoundRobin] {len(instances)} instances x "
          f"{cfg.train.max_epochs // len(instances)} visits "
          f"({cfg.train.max_epochs} epochs), variant={variant}, seed={seed}")

    agent = PPOAgent(cfg, instances[0]['n_usvs'], instances[0]['n_tasks'])
    n_trajectories = cfg.train.get('n_trajectories', 8)
    rollout_collector = ParallelRolloutCollector(cfg, n_trajectories)

    viz = None
    if cfg.train.get('use_visdom', True):
        viz = VisdomLogger(env=cfg.train.get('visdom_env', 'usv_round_robin'))

    run_id = make_training_run_id('PPO', variant, 'public25', seed)
    csv_logger = None
    training_log_path = None
    if cfg.train.get('save_training_csv', True):
        log_dir = cfg.train.get('training_log_dir',
                                os.path.join(cfg.result_dir, 'training_logs'))
        csv_logger = TrainingCSVLogger(log_dir, run_id)
        training_log_path = csv_logger.path
        print(f"[TrainingLog] {training_log_path}")

    best_eval_makespan: Dict[str, float] = {}
    best_eval_epoch: Dict[str, int] = {}
    best_paths: Dict[str, str] = {}
    training_start_time = time.monotonic()

    for epoch in range(1, cfg.train.max_epochs + 1):
        epoch_start = time.monotonic()
        instance = instances[(epoch - 1) % len(instances)]
        instance_id = instance['instance_id']
        visit = (epoch - 1) // len(instances) + 1
        baseline = rules_by_instance[instance_id]

        # The networks are size-agnostic; only the mask/index bounds change.
        agent.n_usvs = instance['n_usvs']
        agent.n_tasks = instance['n_tasks']

        rollout_start = time.monotonic()
        rollout_info = rollout_collector.collect(
            agent=agent, instance=instance, epoch=epoch, train_seed=seed)
        rollout_time_sec = time.monotonic() - rollout_start

        update_start = time.monotonic()
        loss_info = agent.update()
        update_time_sec = time.monotonic() - update_start
        if epoch % cfg.train.get('lr_decay_step', 250) == 0:
            agent.decay_lr()

        epoch_rewards = rollout_info['epoch_rewards']
        epoch_makespans = rollout_info['epoch_makespans']
        avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        avg_makespan = float(np.mean(epoch_makespans)) if epoch_makespans else float('nan')
        n_success = len(epoch_makespans)
        success_rate = n_success / n_trajectories

        # Deterministic evaluation EVERY visit of the trained instance
        # (a fixed epoch stride would only ever evaluate gcd-aligned instances).
        eval_result = evaluate_agent_once(agent, instance)
        eval_makespan = eval_result['makespan']
        gap_percent = None
        if eval_result['success']:
            gap_percent = (
                (eval_makespan - baseline['best_rule_makespan']) /
                baseline['best_rule_makespan'] * 100.0
            )
            if eval_makespan < best_eval_makespan.get(instance_id, float('inf')):
                best_eval_makespan[instance_id] = eval_makespan
                best_eval_epoch[instance_id] = epoch
                path = os.path.join(
                    cfg.model_dir,
                    f'best_{instance_id}{variant_tag}_seed{seed}.pth')
                best_paths[instance_id] = agent.save(path)

        epoch_time_sec = time.monotonic() - epoch_start

        if viz is not None:
            viz.plot('Eval Makespan by Instance', epoch,
                     eval_makespan if eval_result['success'] else float('nan'),
                     trace=instance_id)
            if gap_percent is not None:
                viz.plot('Gap vs Best Rule (%) by Instance', epoch,
                         gap_percent, trace=instance_id)
            viz.log_metrics(epoch, {
                'Actor Loss': loss_info.get('actor_loss'),
                'Critic Loss': loss_info.get('critic_loss'),
                'Entropy': loss_info.get('entropy'),
                'Success Rate': success_rate,
                'Epoch Time (s)': epoch_time_sec,
                'Rollout Time (s)': rollout_time_sec,
                'Update Time (s)': update_time_sec,
            })

        if csv_logger is not None:
            lr_info = agent.get_lr_info()
            total_steps = rollout_info.get('total_steps')
            csv_logger.log({
                'run_id': run_id,
                'algorithm': 'PPO',
                'variant': variant,
                'instance_id': instance_id,
                'n_usvs': instance['n_usvs'],
                'n_tasks': instance['n_tasks'],
                'seed': seed,
                'epoch': epoch,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_sec': time.monotonic() - training_start_time,
                'train_reward_avg': avg_reward,
                'train_reward_std': float(np.std(epoch_rewards)) if epoch_rewards else 0.0,
                'train_makespan_avg': avg_makespan,
                'train_makespan_min': float(np.min(epoch_makespans)) if epoch_makespans else None,
                'train_makespan_std': float(np.std(epoch_makespans)) if epoch_makespans else 0.0,
                'success_rate': success_rate,
                'n_trajectories': n_trajectories,
                'n_success': n_success,
                'n_failed': n_trajectories - n_success,
                'eval_makespan': eval_makespan if eval_result['success'] else None,
                'eval_success': eval_result['success'],
                'best_eval_makespan': best_eval_makespan.get(instance_id),
                'best_eval_epoch': best_eval_epoch.get(instance_id),
                'gap_to_best_rule_percent': gap_percent,
                'best_rule_name': baseline['best_rule_name'],
                'best_rule_makespan': baseline['best_rule_makespan'],
                'random_makespan': baseline['random_makespan'],
                'actor_loss': loss_info.get('actor_loss'),
                'critic_loss': loss_info.get('critic_loss'),
                'entropy': loss_info.get('entropy'),
                'lr_actor_encoder': lr_info.get('LR Actor Encoder'),
                'lr_actor': lr_info.get('LR Actor'),
                'lr_critic_encoder': lr_info.get('LR Critic Encoder'),
                'lr_critic': lr_info.get('LR Critic'),
                'lr_shared_encoder': lr_info.get('LR Shared Encoder'),
                'hidden_dim': cfg.network.hidden_dim,
                'hgnn_layers': cfg.network.hgnn_layers,
                'n_heads': cfg.network.get('n_heads', 4),
                'ppo_epochs': cfg.train.ppo_epochs,
                'vectorized_update': cfg.train.get('vectorized_update', True),
                'update_batch_size': cfg.train.get('update_batch_size', 128),
                'update_micro_batch_size': cfg.train.get('update_micro_batch_size', 0),
                'max_update_pairs': cfg.train.get('max_update_pairs', 32768),
                'update_shuffle': cfg.train.get('update_shuffle', True),
                'gamma': cfg.train.gamma,
                'gae_lambda': cfg.train.gae_lambda,
                'clip_epsilon': cfg.train.epsilon,
                'entropy_coef': cfg.train.entropy_coef,
                'reward_normalization': reward_norm,
                'best_model_path': best_paths.get(instance_id),
                'rollout_time_sec': rollout_time_sec,
                'update_time_sec': update_time_sec,
                'epoch_time_sec': epoch_time_sec,
                'batch_prepare_time_sec': loss_info.get('batch_prepare_time_sec'),
                'actor_update_time_sec': loss_info.get('actor_update_time_sec'),
                'critic_update_time_sec': loss_info.get('critic_update_time_sec'),
                'effective_update_batch_size': loss_info.get('effective_update_batch_size'),
                'effective_update_micro_batch_size': loss_info.get('effective_update_micro_batch_size'),
                'pairs_per_state': loss_info.get('pairs_per_state'),
                'protocol': 'round_robin',
                'visit_index': visit,
                'steps_collected': total_steps,
                'rollout_time_per_decision_ms': (
                    rollout_time_sec / total_steps * 1000.0 if total_steps else None
                ),
                'eval_steps': eval_result['steps'],
                'eval_solve_time_sec': eval_result['solve_time_sec'],
                'eval_time_per_decision_ms': eval_result['time_per_decision_ms'],
            })

        if epoch % cfg.train.get('log_interval', 10) == 0 or epoch == 1:
            eval_str = (f"{eval_makespan:7.1f}" if eval_result['success']
                        else "  fail")
            gap_str = f"{gap_percent:+6.2f}%" if gap_percent is not None else "   n/a"
            print(f"Ep {epoch:4d} [{instance_id:9s} v{visit:3d}] | "
                  f"Eval:{eval_str} Gap:{gap_str} | "
                  f"Best:{best_eval_makespan.get(instance_id, float('inf')):7.1f} | "
                  f"SR:{success_rate:.0%} | T:{epoch_time_sec:.1f}s")

    rollout_collector.close()
    if csv_logger is not None:
        csv_logger.close()

    n_solved = len(best_paths)
    print("-" * 70)
    print(f"[Done] Per-instance best checkpoints: {n_solved}/{len(instances)}")
    return {
        'best_paths': best_paths,
        'best_makespans': dict(best_eval_makespan),
        'best_epochs': dict(best_eval_epoch),
        'training_log_path': training_log_path,
        'run_id': run_id,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=os.path.join('data', 'public'))
    parser.add_argument('--result-dir', default='results')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--max-epochs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant', default='full',
                        choices=['full', 'no_hgnn', 'shared_encoder', 'no_reward_norm'])
    parser.add_argument('--instances', default=None,
                        help='Comma-separated instance IDs (smoke runs); default all 25')
    parser.add_argument('--lr-decay-step', type=int, default=250)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--hgnn-layers', type=int, default=3)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n-trajectories', type=int, default=8)
    parser.add_argument('--no-visdom', action='store_true',
                        help='Disable the (default-on) live Visdom dashboard')
    parser.add_argument('--visdom-env', default='usv_round_robin')
    parser.add_argument('--no-training-csv', action='store_true')
    parser.add_argument('--rollout-num-workers', type=int, default=0)
    parser.add_argument('--rollout-device', default='cpu')
    parser.add_argument('--rollout-torch-threads', type=int, default=1)
    parser.add_argument('--legacy-update', action='store_true')
    parser.add_argument('--update-batch-size', type=int, default=128)
    parser.add_argument('--update-micro-batch-size', type=int, default=0)
    parser.add_argument('--max-update-pairs', type=int, default=32768)
    parser.add_argument('--rules-seed', type=int, default=20260519)
    parser.add_argument('--force-rules', action='store_true',
                        help='Recompute the rules_results.csv cache')
    return parser


def main(args) -> dict:
    instance_ids = ([item.strip() for item in args.instances.split(',') if item.strip()]
                    if args.instances else None)
    instances = load_public_instances(args.data_dir, instance_ids)
    rules_by_instance = ensure_rules_results(
        instances,
        path=os.path.join(args.result_dir, 'rules_results.csv'),
        random_seed=args.rules_seed,
        force=args.force_rules,
    )

    cfg = get_config(
        n_usvs=instances[0]['n_usvs'],
        n_tasks=instances[0]['n_tasks'],
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        model_dir=args.model_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        ablation_variant=args.variant,
        lr_decay_step=args.lr_decay_step,
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
        use_visdom=not args.no_visdom,
        visdom_env=args.visdom_env,
        save_training_csv=not args.no_training_csv,
        training_log_dir=os.path.join(args.result_dir, 'training_logs'),
        rollout_num_workers=args.rollout_num_workers,
        rollout_device=args.rollout_device,
        rollout_torch_threads=args.rollout_torch_threads,
        vectorized_update=not args.legacy_update,
        update_batch_size=args.update_batch_size,
        update_micro_batch_size=args.update_micro_batch_size,
        max_update_pairs=args.max_update_pairs,
    )
    return train_round_robin(cfg, instances, rules_by_instance)


if __name__ == '__main__':
    main(build_parser().parse_args())
