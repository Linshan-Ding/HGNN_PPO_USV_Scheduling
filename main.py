"""
USV Scheduling Training System with Dual Encoder Architecture.

Architecture:
- Actor Encoder (HGNN) + Actor (MLP): Policy network
- Critic Encoder (HGNN) + Critic (MLP): Value network

Key Features:
1. Separate encoders prevent gradient interference
2. Multi-trajectory collection for stable updates
3. Entropy regularization for exploration
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import numpy as np

from config import get_config
from env import USVSchedulingEnv
from ppo import PPOAgent
from utils import load_instance_from_config, plot_gantt_chart


class VisdomLogger:
    """Visdom logger for training visualization."""
    
    def __init__(
        self,
        server: str = 'http://localhost',
        port: int = 8097,
        env: str = 'usv_training',
    ):
        self.enabled = False
        self.wins = {}
        self.server = server
        self.port = port
        self.env = env
        try:
            from visdom import Visdom
            self.viz = Visdom(server=server, port=port, env=env)
            if self.viz.check_connection():
                self.enabled = True
                print(f"[Visdom] Connected: {server}:{port}, env={env}")
                print(f"[Visdom] Dashboard: {server}:{port}")
            else:
                print(f"[Visdom] Server not reachable at {server}:{port}")
                print(f"[Visdom] Start it with: python -m visdom.server -port {port}")
        except ImportError:
            print("[Visdom] Package not installed. Install with: pip install visdom")
        except Exception as exc:
            print(f"[Visdom] Not available: {exc}")
    
    def plot(self, name: str, x: int, y: float):
        """Plot a single point."""
        if not self.enabled:
            return
        
        y_arr = np.array([float(y)])
        x_arr = np.array([int(x)])
        
        if name not in self.wins:
            self.wins[name] = self.viz.line(
                X=x_arr, Y=y_arr,
                opts=dict(title=name, xlabel='Epoch', ylabel=name)
            )
        else:
            self.viz.line(X=x_arr, Y=y_arr, win=self.wins[name], update='append')
    
    def text(self, name: str, content: str):
        """Show a text panel in Visdom."""
        if not self.enabled:
            return
        self.wins[name] = self.viz.text(content, win=self.wins.get(name))
    
    def log_metrics(self, epoch: int, metrics: dict):
        """Plot a dictionary of scalar metrics."""
        for name, value in metrics.items():
            if value is not None:
                self.plot(name, epoch, value)


def set_global_seed(seed: int):
    """Set Python, NumPy, and Torch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_baselines(instance: dict, random_seed: int = 0) -> dict:
    """
    Run all heuristic rules before training.

    Every rule, including Random, is run once. This keeps the baseline protocol
    consistent and avoids selecting the best result from repeated random trials.
    """
    from scheduling_rules import get_all_rules, run_scheduling

    stats = {}
    best_candidates = []

    for rule in get_all_rules():
        is_random = rule.name == 'Random'
        actual_runs = 1
        makespans = []
        for run_idx in range(actual_runs):
            if is_random:
                run_seed = random_seed + run_idx
                np.random.seed(run_seed)

            env = USVSchedulingEnv(instance)
            result = run_scheduling(env, rule)
            makespan = result['makespan'] if result['success'] else float('inf')
            makespans.append(makespan)

        stats[rule.name] = {
            'mean': float(np.mean(makespans)),
            'std': float(np.std(makespans)) if len(makespans) > 1 else 0.0,
            'min': float(np.min(makespans)),
            'max': float(np.max(makespans)),
            'runs': actual_runs,
        }

        candidate_value = stats[rule.name]['mean']
        best_candidates.append((candidate_value, rule.name))

    best_value, best_name = min(best_candidates, key=lambda item: item[0])
    return {
        'stats': stats,
        'best_rule_name': best_name,
        'best_rule_makespan': float(best_value),
        'random_mean': stats['Random']['mean'],
        'random_makespan': stats['Random']['mean'],
        'random_min': stats['Random']['min'],
    }


def evaluate_agent_once(agent: PPOAgent, instance: dict) -> dict:
    """Run one deterministic policy evaluation episode."""
    modes = {
        'actor_encoder': agent.actor_encoder.training,
        'actor': agent.actor.training,
        'critic_encoder': agent.critic_encoder.training,
        'critic': agent.critic.training,
    }
    agent.actor_encoder.eval()
    agent.actor.eval()
    agent.critic_encoder.eval()
    agent.critic.eval()

    env = USVSchedulingEnv(instance)
    state = env.reset()
    done = False
    info = {}
    step = 0
    max_steps = env.n_tasks * 10

    with torch.no_grad():
        while not done and step < max_steps:
            task_mask, _ = agent._get_masks(env)
            if task_mask.sum() == 0:
                break
            action, _, _ = agent.select_action(env, state, deterministic=True)
            state, _, done, info = env.step(action[0], action[1])
            step += 1

    if modes['actor_encoder']:
        agent.actor_encoder.train()
    if modes['actor']:
        agent.actor.train()
    if modes['critic_encoder']:
        agent.critic_encoder.train()
    if modes['critic']:
        agent.critic.train()

    success = env.n_scheduled_tasks == env.n_tasks
    makespan = info.get('makespan', float('inf')) if success else float('inf')
    return {'makespan': makespan, 'success': success, 'steps': step}


def collect_trajectory(env: USVSchedulingEnv, agent: PPOAgent):
    """
    Collect a single trajectory.
    
    Returns:
        ep_reward: Episode reward
        makespan: Final makespan
        success: Whether all tasks scheduled
    """
    state = env.reset()
    done = False
    ep_reward = 0
    step = 0
    max_steps = env.n_tasks * 10
    info = {}
    
    while not done and step < max_steps:
        task_mask, usv_masks = agent._get_masks(env)
        action, log_prob, value = agent.select_action(env, state, deterministic=False)
        task_id, usv_id = action
        
        next_state, reward, done, info = env.step(task_id, usv_id)
        
        agent.store_transition(
            state_dict=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
            task_mask=task_mask,
            usv_masks=usv_masks
        )
        
        state = next_state
        ep_reward += reward
        step += 1
    
    success = env.n_scheduled_tasks == env.n_tasks
    makespan = info.get('makespan', 5000.0) if success else 5000.0
    
    return ep_reward, makespan, success


def train(cfg):
    """Main PPO training function with deterministic evaluation."""
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    
    set_global_seed(cfg.train.get('seed', 0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    variant = cfg.network.get('ablation_variant', 'full')
    
    # Load selected public CSV instance
    fixed_instance = load_instance_from_config(cfg)
    instance_id = fixed_instance.get(
        'instance_id',
        f"u{fixed_instance['n_usvs']}_t{fixed_instance['n_tasks']}"
    )
    instance_tag = f"u{fixed_instance['n_usvs']}_t{fixed_instance['n_tasks']}"
    
    # Baselines are part of the training contract.
    baseline = evaluate_baselines(
        fixed_instance,
        random_seed=cfg.train.get('baseline_seed', 20260519)
    )
    
    # Initialize agent
    agent = PPOAgent(cfg, fixed_instance['n_usvs'], fixed_instance['n_tasks'])
    
    # Visdom logger
    viz = None
    if cfg.train.use_visdom:
        visdom_env = cfg.train.visdom_env
        if variant != 'full' and variant not in visdom_env:
            visdom_env = f"{visdom_env}_{variant}"
        if instance_tag not in visdom_env:
            visdom_env = f"{visdom_env}_{instance_tag}"
        viz = VisdomLogger(
            server=cfg.train.visdom_server,
            port=cfg.train.visdom_port,
            env=visdom_env,
        )
    
    # Training tracking
    best_eval_makespan = float('inf')
    n_trajectories = cfg.train.get('n_trajectories', 8)
    eval_interval = cfg.train.get('eval_interval', 10)
    seed = cfg.train.get('seed', 0)
    variant_tag = '' if variant == 'full' else f'_{variant}'
    best_model_path = os.path.join(cfg.model_dir, f'best_{instance_id}{variant_tag}_seed{seed}.pth')
    
    # Print configuration
    print(f"\n[Training] Instance ID={fixed_instance.get('instance_id', 'N/A')}")
    print(f"[Training] Instance seed={fixed_instance.get('seed', 'N/A')}")
    print(f"[Training] Train seed={seed}")
    print(f"[Training] Variant={variant}")
    print(f"[Config] USVs={fixed_instance['n_usvs']}, Tasks={fixed_instance['n_tasks']}")
    print(f"[Config] Hidden={cfg.network.hidden_dim}, HGNN_layers={cfg.network.hgnn_layers}")
    print(f"[Config] Reward normalization={getattr(fixed_instance['config'], 'reward_normalization', True)}")
    print(f"[Config] PPO_epochs={cfg.train.ppo_epochs}, Trajectories={n_trajectories}")
    print(f"[Config] LR: encoder={cfg.train.lr_encoder}, actor={cfg.train.lr_actor}, critic={cfg.train.lr_critic}")
    print(f"[Config] Entropy_coef={cfg.train.entropy_coef}")
    print(f"[Baseline] Best={baseline['best_rule_name']} {baseline['best_rule_makespan']:.2f}")
    print(f"[Baseline] Random={baseline['random_mean']:.2f}")
    print("-" * 70)
    
    if viz and viz.enabled:
        viz.text(
            'Training Config',
            '<br>'.join([
                '<b>USV Scheduling PPO Training</b>',
                f'Variant: {variant}',
                f'Instance ID: {fixed_instance.get("instance_id", "N/A")}',
                f'Instance seed: {fixed_instance.get("seed", "N/A")}',
                f'Train seed: {seed}',
                f'USVs: {fixed_instance["n_usvs"]}',
                f'Tasks: {fixed_instance["n_tasks"]}',
                f'Best rule: {baseline["best_rule_name"]} = {baseline["best_rule_makespan"]:.2f}',
                f'Random: {baseline["random_makespan"]:.2f}',
                f'Hidden dim: {cfg.network.hidden_dim}',
                f'HGNN layers: {cfg.network.hgnn_layers}',
                f'Reward normalization: {getattr(fixed_instance["config"], "reward_normalization", True)}',
                f'PPO epochs: {cfg.train.ppo_epochs}',
                f'Trajectories/update: {n_trajectories}',
                f'LR encoder: {cfg.train.lr_encoder}',
                f'LR actor: {cfg.train.lr_actor}',
                f'LR critic: {cfg.train.lr_critic}',
                f'Entropy coef: {cfg.train.entropy_coef}',
            ])
        )

    initial_eval = evaluate_agent_once(agent, fixed_instance)
    if initial_eval['success']:
        best_eval_makespan = initial_eval['makespan']
        agent.save(best_model_path)
        agent.save(os.path.join(cfg.model_dir, 'best_model.pth'))
        initial_gap = (
            (best_eval_makespan - baseline['best_rule_makespan']) /
            baseline['best_rule_makespan'] * 100.0
        )
        print(f"[Eval@0] Makespan={best_eval_makespan:.2f}, Gap={initial_gap:.2f}%")
        if viz and viz.enabled:
            viz.log_metrics(0, {
                'Eval Makespan': best_eval_makespan,
                'Best Rule Makespan': baseline['best_rule_makespan'],
                'Random Makespan': baseline['random_makespan'],
                'Gap vs Best Rule (%)': initial_gap,
            })
    
    # Training loop
    for epoch in range(1, cfg.train.max_epochs + 1):
        # ============ COLLECT MULTIPLE TRAJECTORIES ============
        epoch_rewards = []
        epoch_makespans = []
        
        for _ in range(n_trajectories):
            env = USVSchedulingEnv(fixed_instance)
            ep_reward, makespan, success = collect_trajectory(env, agent)
            epoch_rewards.append(ep_reward)
            if success:
                epoch_makespans.append(makespan)
        
        # ============ PPO UPDATE ============
        loss_info = agent.update()
        
        # Learning rate decay
        if epoch % cfg.train.lr_decay_step == 0:
            agent.decay_lr()
        
        # Statistics
        avg_reward = np.mean(epoch_rewards)
        avg_makespan = np.mean(epoch_makespans) if epoch_makespans else 5000.0
        min_makespan = np.min(epoch_makespans) if epoch_makespans else 5000.0
        success_rate = len(epoch_makespans) / n_trajectories
        eval_makespan = None
        gap_percent = None
        
        # ============ DETERMINISTIC EVALUATION ============
        if epoch == 1 or epoch % eval_interval == 0:
            eval_result = evaluate_agent_once(agent, fixed_instance)
            eval_makespan = eval_result['makespan']
            gap_percent = (
                (eval_makespan - baseline['best_rule_makespan']) /
                baseline['best_rule_makespan'] * 100.0
            )
            
            if eval_result['success'] and eval_makespan < best_eval_makespan:
                best_eval_makespan = eval_makespan
                agent.save(best_model_path)
                agent.save(os.path.join(cfg.model_dir, 'best_model.pth'))
        
        # Visdom logging
        if viz and loss_info:
            metrics = {
                'Reward (Avg)': avg_reward,
                'Train Makespan': avg_makespan,
                'Eval Makespan': eval_makespan,
                'Best Rule Makespan': baseline['best_rule_makespan'],
                'Random Makespan': baseline['random_makespan'],
                'Gap vs Best Rule (%)': gap_percent,
                'Success Rate': success_rate,
                'Actor Loss': loss_info['actor_loss'],
                'Critic Loss': loss_info['critic_loss'],
                'Entropy': loss_info.get('entropy', 0),
            }
            metrics.update(agent.get_lr_info())
            viz.log_metrics(epoch, metrics)
        
        # Console logging
        if epoch % cfg.train.log_interval == 0:
            loss_str = ""
            if loss_info:
                loss_str = (f"| A:{loss_info['actor_loss']:.4f} "
                           f"C:{loss_info['critic_loss']:.4f} "
                           f"E:{loss_info.get('entropy', 0):.3f}")
            eval_str = ""
            if eval_makespan is not None:
                eval_str = f" | Eval:{eval_makespan:7.1f} Gap:{gap_percent:6.2f}%"
            print(f"Ep {epoch:4d} | R:{avg_reward:7.3f} | "
                  f"MS:{avg_makespan:7.1f} (min:{min_makespan:7.1f}) | "
                  f"BestEval:{best_eval_makespan:7.1f}{eval_str} | "
                  f"SR:{success_rate:.0%} {loss_str}")
    
    print("-" * 70)
    print(f"[Done] Best Eval Makespan: {best_eval_makespan:.2f}")
    print(f"[Done] Best Rule Makespan: {baseline['best_rule_makespan']:.2f}")
    
    return agent, fixed_instance, {
        'baseline': baseline,
        'best_eval_makespan': best_eval_makespan,
        'best_model_path': best_model_path,
        'variant': variant,
    }


def evaluate(cfg, agent=None, instance=None, n_episodes=10):
    """
    Evaluate trained agent.
    """
    if instance is None:
        instance = load_instance_from_config(cfg)
    
    if agent is None:
        agent = PPOAgent(cfg, instance['n_usvs'], instance['n_tasks'])
        agent.load(os.path.join(cfg.model_dir, 'best_model.pth'))
    
    print(f"\n[Evaluation] {n_episodes} episodes (deterministic)")
    
    makespans = []
    for ep in range(n_episodes):
        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        
        while not done:
            task_mask, usv_masks = agent._get_masks(env)
            if task_mask.sum() == 0:
                break
            
            action, _, _ = agent.select_action(env, state, deterministic=True)
            state, _, done, info = env.step(action[0], action[1])
        
        if env.n_scheduled_tasks == env.n_tasks:
            makespans.append(info['makespan'])
            print(f"  Episode {ep+1}: Makespan = {info['makespan']:.2f}")
    
    if makespans:
        print(f"\n[Result] Mean: {np.mean(makespans):.2f}, "
              f"Std: {np.std(makespans):.2f}, "
              f"Min: {np.min(makespans):.2f}")
    
    return makespans


def demo(cfg, agent=None, instance=None):
    """Run single demonstration with visualization."""
    if instance is None:
        instance = load_instance_from_config(cfg)
    
    if agent is None:
        agent = PPOAgent(cfg, instance['n_usvs'], instance['n_tasks'])
        agent.load(os.path.join(cfg.model_dir, 'best_model.pth'))
    
    env = USVSchedulingEnv(instance)
    state = env.reset()
    
    print(f"\n[Demo] {env.n_usvs} USVs, {env.n_tasks} Tasks")
    print(f"[Demo] Using dual encoder architecture")
    
    done = False
    step = 0
    info = {}
    
    while not done:
        task_mask, usv_masks = agent._get_masks(env)
        if task_mask.sum() == 0:
            break
        
        action, _, _ = agent.select_action(env, state, deterministic=True)
        task_id, usv_id = action
        
        step += 1
        print(f"  Step {step:2d}: Task {task_id:2d} -> USV {usv_id}")
        
        state, _, done, info = env.step(task_id, usv_id)
    
    print(f"\n[Result] Makespan: {info.get('makespan', 'N/A'):.2f}")
    print(f"[Result] Tasks: {env.n_scheduled_tasks}/{env.n_tasks}")
    
    plot_gantt_chart(env, os.path.join(cfg.result_dir, 'gantt.png'))
    print(f"[Result] Gantt chart saved to {cfg.result_dir}/gantt.png")


def compare_with_heuristics(cfg, agent=None, instance=None):
    """Compare trained agent with heuristic baselines."""
    from scheduling_rules import get_all_rules, run_scheduling
    
    if instance is None:
        instance = load_instance_from_config(cfg)
    
    if agent is None:
        agent = PPOAgent(cfg, instance['n_usvs'], instance['n_tasks'])
        agent.load(os.path.join(cfg.model_dir, 'best_model.pth'))
    
    print(f"\n[Comparison] PPO vs Heuristics")
    print("-" * 50)
    
    # PPO agent
    env = USVSchedulingEnv(instance)
    state = env.reset()
    done = False
    
    while not done:
        task_mask, usv_masks = agent._get_masks(env)
        if task_mask.sum() == 0:
            break
        action, _, _ = agent.select_action(env, state, deterministic=True)
        state, _, done, info = env.step(action[0], action[1])
    
    ppo_makespan = info.get('makespan', float('inf'))
    print(f"{'PPO (Dual Encoder)':<30} {ppo_makespan:>10.2f}")
    
    # Heuristic baselines
    rules = get_all_rules()
    for rule in rules:
        env = USVSchedulingEnv(instance)
        result = run_scheduling(env, rule)
        makespan = result['makespan'] if result['success'] else float('inf')
        print(f"{rule.name:<30} {makespan:>10.2f}")
    
    print("-" * 50)


if __name__ == "__main__":
    cfg = get_config(
        # Problem size
        n_usvs=2,
        n_tasks=20,
        data_dir='data/public',
        max_epochs=500,
        seed=0,
        
        # Network architecture
        hidden_dim=256,
        hgnn_layers=3,
        n_heads=4,
        dropout=0.1,
        
        # Training parameters
        ppo_epochs=4,
        n_trajectories=8,
        lr_encoder=1e-4,
        lr_actor=3e-4,
        lr_critic=3e-4,
        entropy_coef=0.01,
        eval_interval=10,
        # Visualization
        use_visdom=True,
        visdom_server='http://localhost',
        visdom_port=8097,
        visdom_env='usv_training'
    )
    
    # Train
    agent, instance, train_info = train(cfg)
    #
    # Evaluate
    evaluate(cfg, agent, instance)

    # Demo
    demo(cfg)
    
    # # Compare with heuristics
    # compare_with_heuristics(cfg, agent, instance)
