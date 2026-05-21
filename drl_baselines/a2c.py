"""A2C comparison algorithm."""

import os

import numpy as np

from .base import DRLBaseline
from .common import (
    build_actor_components,
    build_critic_components,
    checkpoint_path,
    discounted_returns,
    evaluate_pairwise_policy,
    get_action_masks,
    get_cfg_attr,
    make_result,
    make_visdom_logger,
    normalize_tensor,
    now,
    prepare_state,
    require_torch,
    set_seed,
)


class A2CBaseline(DRLBaseline):
    """Advantage Actor-Critic baseline with pairwise legal action distribution."""

    algorithm_name = "A2C"
    implemented = True

    def _build(self, cfg, n_usvs: int, n_tasks: int):
        torch, nn, optim = require_torch()
        self.torch = torch
        self.nn = nn
        self.n_usvs = n_usvs
        self.n_tasks = n_tasks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = get_cfg_attr(cfg, "train", "gamma", 0.99)
        self.entropy_coef = get_cfg_attr(cfg, "train", "entropy_coef", 0.01)
        self.value_coef = get_cfg_attr(cfg, "train", "value_coef", 0.5)
        self.grad_clip = get_cfg_attr(cfg, "train", "grad_clip", 0.5)

        self.actor_encoder, self.actor = build_actor_components(cfg, self.device)
        self.critic_encoder, self.critic = build_critic_components(cfg, self.device)

        lr_actor = get_cfg_attr(cfg, "train", "lr_actor", 3e-4)
        lr_critic = get_cfg_attr(cfg, "train", "lr_critic", 3e-4)
        lr_encoder = get_cfg_attr(cfg, "train", "lr_encoder", 1e-4)
        self.optimizer = optim.AdamW(
            [
                {"params": self.actor_encoder.parameters(), "lr": lr_encoder},
                {"params": self.actor.parameters(), "lr": lr_actor},
                {"params": self.critic_encoder.parameters(), "lr": lr_encoder},
                {"params": self.critic.parameters(), "lr": lr_critic},
            ],
            weight_decay=1e-4,
            eps=1e-5,
        )

    def select_action(self, env, state_dict, deterministic: bool = False):
        """Select a legal task-USV pair."""
        _, pair_mask = get_action_masks(env, self.n_tasks, self.n_usvs, self.device)
        state_tensor = prepare_state(state_dict, self.device)
        with self.torch.set_grad_enabled(not deterministic):
            encoded = self.actor_encoder(state_tensor)
            action, _ = self.actor.select_action(
                encoded, state_tensor["edge_features"], pair_mask, deterministic
            )
        return action

    def _sample_trajectory(self, instance: dict):
        from env import USVSchedulingEnv

        env = USVSchedulingEnv(instance)
        state = env.reset()
        done = False
        rewards = []
        log_probs = []
        values = []
        entropies = []
        makespan = float("inf")
        step = 0
        max_steps = env.n_tasks * 10

        while not done and step < max_steps:
            _, pair_mask = get_action_masks(env, self.n_tasks, self.n_usvs, self.device)
            if pair_mask.sum() == 0:
                break

            state_tensor = prepare_state(state, self.device)
            actor_encoded = self.actor_encoder(state_tensor)
            action, log_prob = self.actor.select_action(
                actor_encoded, state_tensor["edge_features"], pair_mask, deterministic=False
            )
            log_prob_checked, entropy, _ = self.actor.get_action_log_prob(
                actor_encoded, state_tensor["edge_features"], pair_mask, action
            )
            critic_encoded = self.critic_encoder(state_tensor)
            value = self.critic(critic_encoded["graph_embed"])

            next_state, reward, done, info = env.step(action[0], action[1])
            rewards.append(float(reward))
            log_probs.append(log_prob_checked if log_prob_checked is not None else log_prob)
            values.append(value)
            entropies.append(entropy)
            state = next_state
            step += 1
            makespan = info.get("makespan", makespan)

        success = env.n_scheduled_tasks == env.n_tasks
        return {
            "rewards": rewards,
            "log_probs": log_probs,
            "values": values,
            "entropies": entropies,
            "makespan": makespan if success else float("inf"),
            "success": success,
        }

    def _update(self, trajectories):
        torch = self.torch
        all_log_probs = []
        all_values = []
        all_returns = []
        all_entropies = []

        for traj in trajectories:
            if not traj["rewards"]:
                continue
            returns = discounted_returns(traj["rewards"], self.gamma, self.device)
            all_returns.append(returns)
            all_log_probs.extend(traj["log_probs"])
            all_values.extend(traj["values"])
            all_entropies.extend(traj["entropies"])

        if not all_log_probs:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        returns = torch.cat(all_returns)
        values = torch.stack(all_values)
        log_probs = torch.stack(all_log_probs)
        entropies = torch.stack(all_entropies)

        advantages = normalize_tensor(returns - values.detach())
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropies.mean()
        critic_loss = self.nn.SmoothL1Loss()(values, returns)
        loss = actor_loss + self.value_coef * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        params = (
            list(self.actor_encoder.parameters()) + list(self.actor.parameters()) +
            list(self.critic_encoder.parameters()) + list(self.critic.parameters())
        )
        self.torch.nn.utils.clip_grad_norm_(params, max_norm=self.grad_clip)
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropies.mean().item()),
        }

    def train(self, instance: dict, cfg=None):
        """Train A2C on one public instance."""
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
                ]),
            )

        for epoch in range(1, max_epochs + 1):
            trajectories = [self._sample_trajectory(instance) for _ in range(n_trajectories)]
            loss_info = self._update(trajectories)
            train_makespans = [t["makespan"] for t in trajectories if t["success"]]
            train_makespan = float(np.mean(train_makespans)) if train_makespans else None
            success_rate = len(train_makespans) / max(n_trajectories, 1)
            eval_makespan = None

            if epoch == 1 or epoch % eval_interval == 0:
                makespan, success = evaluate_pairwise_policy(self, instance)
                eval_makespan = makespan if success else None
                if success and makespan < best_makespan:
                    best_makespan = makespan
                    best_success = True
                    self.save(best_path)

            viz.log_metrics(epoch, {
                "Train Makespan": train_makespan,
                "Success Rate": success_rate,
                "Eval Makespan": eval_makespan,
                "Best Eval Makespan": best_makespan if best_success else None,
                "Actor Loss": loss_info.get("actor_loss"),
                "Critic Loss": loss_info.get("critic_loss"),
                "Entropy": loss_info.get("entropy"),
            })

        return make_result(
            self.algorithm_name, self.category, instance,
            best_makespan, best_success, now() - start, self.seed
        )

    def evaluate(self, instance: dict, cfg=None):
        """Evaluate the deterministic A2C policy."""
        start = now()
        makespan, success = evaluate_pairwise_policy(self, instance)
        return make_result(
            self.algorithm_name, self.category, instance,
            makespan, success, now() - start, self.seed
        )

    def save(self, path: str):
        """Save A2C checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.torch.save(
            {
                "actor_encoder": self.actor_encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_encoder": self.critic_encoder.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load A2C checkpoint. Models must be built first via train() or _build()."""
        checkpoint = self.torch.load(path, map_location=self.device)
        self.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
