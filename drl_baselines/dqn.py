"""DQN and Double DQN comparison algorithms."""

import os
import random
from collections import deque

import numpy as np

from .base import DRLBaseline
from .common import (
    build_actor_components,
    checkpoint_path,
    copy_state,
    evaluate_pairwise_policy,
    flat_to_pair,
    get_action_masks,
    get_cfg_attr,
    make_result,
    make_visdom_logger,
    now,
    pair_to_flat,
    prepare_state,
    random_legal_action,
    require_torch,
    set_seed,
)


class ReplayBuffer:
    """Small replay buffer for pairwise Q-learning."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(int(batch_size), len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class DQNBaseline(DRLBaseline):
    """DQN baseline using pairwise Q scores over legal task-USV actions."""

    algorithm_name = "DQN"
    implemented = True
    double_q = False

    def _build(self, cfg, n_usvs: int, n_tasks: int):
        torch, nn, optim = require_torch()
        self.torch = torch
        self.nn = nn
        self.n_usvs = n_usvs
        self.n_tasks = n_tasks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = get_cfg_attr(cfg, "train", "gamma", 0.99)
        self.grad_clip = get_cfg_attr(cfg, "train", "grad_clip", 0.5)
        self.batch_size = get_cfg_attr(cfg, "train", "drl_batch_size", 64)
        self.replay_size = get_cfg_attr(cfg, "train", "drl_replay_size", 5000)
        self.target_update_interval = get_cfg_attr(
            cfg, "train", "drl_target_update_interval", 10
        )
        self.updates_per_epoch = get_cfg_attr(cfg, "train", "drl_updates_per_epoch", 4)
        self.epsilon_start = get_cfg_attr(cfg, "train", "drl_epsilon_start", 1.0)
        self.epsilon_end = get_cfg_attr(cfg, "train", "drl_epsilon_end", 0.05)
        self.epsilon_decay_epochs = max(
            1, get_cfg_attr(cfg, "train", "drl_epsilon_decay_epochs", 200)
        )

        self.online_encoder, self.online_head = build_actor_components(cfg, self.device)
        self.target_encoder, self.target_head = build_actor_components(cfg, self.device)
        self._sync_target()

        lr_actor = get_cfg_attr(cfg, "train", "lr_actor", 3e-4)
        lr_encoder = get_cfg_attr(cfg, "train", "lr_encoder", 1e-4)
        self.optimizer = optim.AdamW(
            [
                {"params": self.online_encoder.parameters(), "lr": lr_encoder},
                {"params": self.online_head.parameters(), "lr": lr_actor},
            ],
            weight_decay=1e-4,
            eps=1e-5,
        )
        self.replay = ReplayBuffer(self.replay_size)

    def _sync_target(self):
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_head.load_state_dict(self.online_head.state_dict())
        self.target_encoder.eval()
        self.target_head.eval()

    def _q_logits(self, encoder, head, state_dict, pair_mask):
        state_tensor = prepare_state(state_dict, self.device)
        encoded = encoder(state_tensor)
        logits, _ = head(encoded, state_tensor["edge_features"], pair_mask)
        return logits

    def _epsilon(self, epoch: int) -> float:
        progress = min(float(epoch) / self.epsilon_decay_epochs, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, env, state_dict, deterministic: bool = False):
        """Select action by greedy Q value or epsilon-greedy exploration."""
        _, pair_mask = get_action_masks(env, self.n_tasks, self.n_usvs, self.device)
        if pair_mask.sum() == 0:
            return 0, 0

        if not deterministic and random.random() < getattr(self, "current_epsilon", 0.0):
            return random_legal_action(pair_mask)

        with self.torch.no_grad():
            logits = self._q_logits(self.online_encoder, self.online_head, state_dict, pair_mask)
            flat_index = self.torch.argmax(logits.reshape(-1)).item()
        return flat_to_pair(flat_index, self.n_usvs)

    def _collect_trajectory(self, instance: dict, epoch: int):
        from env import USVSchedulingEnv

        self.current_epsilon = self._epsilon(epoch)
        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        makespan = float("inf")
        success = False
        step = 0
        max_steps = env.n_tasks * 10

        while not done and step < max_steps:
            _, pair_mask = get_action_masks(env, self.n_tasks, self.n_usvs, self.device)
            if pair_mask.sum() == 0:
                break

            action = self.select_action(env, state, deterministic=False)
            next_state, reward, done, info = env.step(action[0], action[1])

            if not done:
                _, next_pair_mask = get_action_masks(env, self.n_tasks, self.n_usvs, self.device)
            else:
                next_pair_mask = self.torch.zeros_like(pair_mask)

            self.replay.push({
                "state": copy_state(state),
                "action": action,
                "reward": float(reward),
                "next_state": copy_state(next_state),
                "done": bool(done),
                "pair_mask": pair_mask.detach().cpu(),
                "next_pair_mask": next_pair_mask.detach().cpu(),
            })

            state = next_state
            step += 1
            makespan = info.get("makespan", makespan)

        success = env.n_scheduled_tasks == env.n_tasks
        return makespan if success else float("inf"), success

    def _transition_target(self, transition):
        torch = self.torch
        reward = torch.tensor(float(transition["reward"]), dtype=torch.float32, device=self.device)
        if transition["done"]:
            return reward

        next_mask = transition["next_pair_mask"].to(self.device)
        if next_mask.sum() == 0:
            return reward

        with torch.no_grad():
            if self.double_q:
                online_next = self._q_logits(
                    self.online_encoder,
                    self.online_head,
                    transition["next_state"],
                    next_mask,
                )
                next_flat = torch.argmax(online_next.reshape(-1)).item()
                target_next = self._q_logits(
                    self.target_encoder,
                    self.target_head,
                    transition["next_state"],
                    next_mask,
                ).reshape(-1)[next_flat]
            else:
                target_next = self._q_logits(
                    self.target_encoder,
                    self.target_head,
                    transition["next_state"],
                    next_mask,
                ).reshape(-1).max()
        return reward + self.gamma * target_next

    def _update_batch(self):
        if len(self.replay) == 0:
            return 0.0

        batch = self.replay.sample(self.batch_size)
        losses = []
        for transition in batch:
            pair_mask = transition["pair_mask"].to(self.device)
            logits = self._q_logits(
                self.online_encoder,
                self.online_head,
                transition["state"],
                pair_mask,
            )
            action_flat = pair_to_flat(transition["action"], self.n_usvs)
            q_pred = logits.reshape(-1)[action_flat]
            target = self._transition_target(transition)
            losses.append(self.nn.SmoothL1Loss()(q_pred, target))

        loss = sum(losses) / len(losses)
        self.optimizer.zero_grad()
        loss.backward()
        params = list(self.online_encoder.parameters()) + list(self.online_head.parameters())
        self.torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def train(self, instance: dict, cfg=None):
        """Train DQN on one public instance."""
        set_seed(self.seed)
        start = now()
        self._build(cfg, instance["n_usvs"], instance["n_tasks"])

        max_epochs = get_cfg_attr(cfg, "train", "max_epochs", 500)
        n_trajectories = get_cfg_attr(cfg, "train", "n_trajectories", 8)
        eval_interval = get_cfg_attr(cfg, "train", "eval_interval", 10)
        best_makespan = float("inf")
        best_success = False
        best_path = checkpoint_path(cfg, self.algorithm_name, instance, self.seed)
        viz = make_visdom_logger(cfg, self.algorithm_name, instance)
        if viz.enabled:
            viz.text(
                "Training Config",
                "<br>".join([
                    f"<b>{self.algorithm_name} DRL Baseline</b>",
                    f"Instance: {instance.get('instance_id', 'unknown')}",
                    f"USVs: {instance['n_usvs']}",
                    f"Tasks: {instance['n_tasks']}",
                    f"Seed: {self.seed}",
                    f"Trajectories/update: {n_trajectories}",
                    f"Replay size: {self.replay_size}",
                    f"Batch size: {self.batch_size}",
                ]),
            )

        for epoch in range(1, max_epochs + 1):
            train_makespans = []
            for _ in range(n_trajectories):
                makespan, success = self._collect_trajectory(instance, epoch)
                if success:
                    train_makespans.append(makespan)

            batch_losses = []
            for _ in range(self.updates_per_epoch):
                if len(self.replay) >= 1:
                    batch_losses.append(self._update_batch())

            if epoch % self.target_update_interval == 0:
                self._sync_target()

            eval_makespan = None
            if epoch == 1 or epoch % eval_interval == 0:
                makespan, success = evaluate_pairwise_policy(self, instance)
                eval_makespan = makespan if success else None
                if success and makespan < best_makespan:
                    best_makespan = makespan
                    best_success = True
                    self.save(best_path)

            viz.log_metrics(epoch, {
                "Train Makespan": float(np.mean(train_makespans)) if train_makespans else None,
                "Success Rate": len(train_makespans) / max(n_trajectories, 1),
                "Eval Makespan": eval_makespan,
                "Best Eval Makespan": best_makespan if best_success else None,
                "Q Loss": float(np.mean(batch_losses)) if batch_losses else None,
                "Replay Size": len(self.replay),
                "Epsilon": self._epsilon(epoch),
            })

        return make_result(
            self.algorithm_name, self.category, instance,
            best_makespan, best_success, now() - start, self.seed
        )

    def evaluate(self, instance: dict, cfg=None):
        """Evaluate deterministic DQN policy."""
        start = now()
        makespan, success = evaluate_pairwise_policy(self, instance)
        return make_result(
            self.algorithm_name, self.category, instance,
            makespan, success, now() - start, self.seed
        )

    def save(self, path: str):
        """Save DQN checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.torch.save(
            {
                "online_encoder": self.online_encoder.state_dict(),
                "online_head": self.online_head.state_dict(),
                "target_encoder": self.target_encoder.state_dict(),
                "target_head": self.target_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load DQN checkpoint. Models must be built first via train() or _build()."""
        checkpoint = self.torch.load(path, map_location=self.device)
        self.online_encoder.load_state_dict(checkpoint["online_encoder"])
        self.online_head.load_state_dict(checkpoint["online_head"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder"])
        self.target_head.load_state_dict(checkpoint["target_head"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
