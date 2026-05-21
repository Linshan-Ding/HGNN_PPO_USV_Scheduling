"""Parallel rollout collection for PPO training."""

import copy
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import torch

from env import USVSchedulingEnv
from ppo import PPOAgent


def resolve_rollout_workers(n_trajectories: int, requested_workers: int) -> int:
    """Resolve configured worker count."""
    if n_trajectories < 2:
        raise ValueError(
            "Parallel rollout requires n_trajectories >= 2. "
            f"Got n_trajectories={n_trajectories}."
        )
    if requested_workers and requested_workers > 0:
        workers = max(1, min(int(requested_workers), int(n_trajectories)))
    else:
        workers = max(1, min(int(n_trajectories), os.cpu_count() or 1))
    if workers < 2:
        raise ValueError(
            "Parallel rollout requires at least 2 worker processes. "
            f"Resolved rollout_num_workers={workers}."
        )
    return workers


def build_policy_snapshot(agent: PPOAgent) -> Dict:
    """Copy policy/value weights to CPU for worker inference."""
    def cpu_state(module):
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    return {
        'actor_encoder': cpu_state(agent.actor_encoder),
        'critic_encoder': None if agent.use_shared_encoder else cpu_state(agent.critic_encoder),
        'actor': cpu_state(agent.actor),
        'critic': cpu_state(agent.critic),
    }


def _trajectory_to_serializable(trajectory: Dict) -> Dict:
    """Convert torch masks in a collected trajectory to numpy arrays."""
    transitions = []
    for transition in trajectory['transitions']:
        transitions.append({
            'state': transition['state'],
            'action': transition['action'],
            'log_prob': transition['log_prob'],
            'reward': transition['reward'],
            'done': transition['done'],
            'value': transition['value'],
            'task_mask': transition['task_mask'].detach().cpu().numpy().astype(bool),
            'usv_masks': transition['usv_masks'].detach().cpu().numpy().astype(bool),
        })
    return {
        'transitions': transitions,
        'ep_reward': trajectory['ep_reward'],
        'makespan': trajectory['makespan'],
        'success': trajectory['success'],
    }


def _collect_single_trajectory(instance: dict, agent: PPOAgent, seed: int) -> Dict:
    """Collect one trajectory without mutating the training agent buffer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = USVSchedulingEnv(instance)
    state = env.reset()
    done = False
    ep_reward = 0.0
    step = 0
    max_steps = env.n_tasks * 10
    info = {}
    transitions = []

    while not done and step < max_steps:
        task_mask, usv_masks = agent._get_masks(env)
        action, log_prob, value = agent.select_action(env, state, deterministic=False)
        task_id, usv_id = action
        next_state, reward, done, info = env.step(task_id, usv_id)

        transitions.append({
            'state': copy.deepcopy(state),
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'done': done,
            'value': value,
            'task_mask': task_mask,
            'usv_masks': usv_masks,
        })

        state = next_state
        ep_reward += reward
        step += 1

    success = env.n_scheduled_tasks == env.n_tasks
    makespan = info.get('makespan', 5000.0) if success else 5000.0
    return {
        'transitions': transitions,
        'ep_reward': float(ep_reward),
        'makespan': float(makespan),
        'success': bool(success),
    }


def _worker_collect_rollouts(payload: Dict) -> List[Dict]:
    """Worker entrypoint; must remain top-level for Windows multiprocessing."""
    torch.set_num_threads(max(int(payload.get('torch_threads', 1)), 1))
    cfg = payload['cfg']
    cfg.train.use_visdom = False
    cfg.train.save_training_csv = False

    agent = PPOAgent(
        cfg,
        payload['instance']['n_usvs'],
        payload['instance']['n_tasks'],
        device=payload.get('device', 'cpu'),
        verbose=False,
    )
    snapshot = payload['snapshot']
    agent.actor_encoder.load_state_dict(snapshot['actor_encoder'])
    if not agent.use_shared_encoder and snapshot.get('critic_encoder') is not None:
        agent.critic_encoder.load_state_dict(snapshot['critic_encoder'])
    agent.actor.load_state_dict(snapshot['actor'])
    agent.critic.load_state_dict(snapshot['critic'])
    agent.actor_encoder.eval()
    agent.critic_encoder.eval()
    agent.actor.eval()
    agent.critic.eval()

    results = []
    for local_idx in range(payload['n_rollouts']):
        seed = payload['base_seed'] + local_idx
        trajectory = _collect_single_trajectory(payload['instance'], agent, seed)
        results.append(_trajectory_to_serializable(trajectory))
    return results


def merge_rollout_results(agent: PPOAgent, results: List[Dict]) -> Dict[str, List]:
    """Append worker transitions to the main agent buffer."""
    epoch_rewards = []
    epoch_makespans = []

    for result in results:
        epoch_rewards.append(result['ep_reward'])
        if result['success']:
            epoch_makespans.append(result['makespan'])
        for transition in result['transitions']:
            task_mask = torch.as_tensor(
                transition['task_mask'],
                dtype=torch.bool,
                device=agent.device,
            )
            usv_masks = torch.as_tensor(
                transition['usv_masks'],
                dtype=torch.bool,
                device=agent.device,
            )
            agent.store_transition(
                state_dict=transition['state'],
                action=tuple(transition['action']),
                log_prob=float(transition['log_prob']),
                reward=float(transition['reward']),
                done=bool(transition['done']),
                value=float(transition['value']),
                task_mask=task_mask,
                usv_masks=usv_masks,
            )

    return {
        'epoch_rewards': epoch_rewards,
        'epoch_makespans': epoch_makespans,
    }


class ParallelRolloutCollector:
    """Persistent process pool for collecting PPO trajectories."""

    def __init__(self, cfg, n_trajectories: int):
        self.cfg = cfg
        self.n_trajectories = int(n_trajectories)
        self.num_workers = resolve_rollout_workers(
            self.n_trajectories,
            cfg.train.get('rollout_num_workers', 0),
        )
        self.device = cfg.train.get('rollout_device', 'cpu')
        self.torch_threads = cfg.train.get('rollout_torch_threads', 1)
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    def collect(self, agent: PPOAgent, instance: dict, epoch: int, train_seed: int) -> Dict[str, List]:
        snapshot = build_policy_snapshot(agent)
        base = self.n_trajectories // self.num_workers
        remainder = self.n_trajectories % self.num_workers
        futures = []

        for worker_id in range(self.num_workers):
            n_rollouts = base + (1 if worker_id < remainder else 0)
            if n_rollouts <= 0:
                continue
            payload = {
                'cfg': copy.deepcopy(self.cfg),
                'instance': copy.deepcopy(instance),
                'snapshot': snapshot,
                'n_rollouts': n_rollouts,
                'base_seed': int(train_seed) + int(epoch) * 100000 + worker_id * 1000,
                'device': self.device,
                'torch_threads': self.torch_threads,
            }
            futures.append(self.executor.submit(_worker_collect_rollouts, payload))

        worker_results = []
        for future in as_completed(futures):
            worker_results.extend(future.result())

        return merge_rollout_results(agent, worker_results)
