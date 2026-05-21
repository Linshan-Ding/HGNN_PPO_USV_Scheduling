"""
PPO Agent for USV Scheduling with Dual Encoders.

Architecture:
- Actor Encoder (HGNN): Dedicated encoder for policy network
- Critic Encoder (HGNN): Dedicated encoder for value network
- Actor: Pairwise policy for legal (task, USV) selection
- Critic: State value estimator

Benefits of dual encoders:
1. Actor and Critic can learn different state representations
2. No gradient interference between actor and critic
3. More stable training dynamics
4. Better convergence properties
"""

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List

from hgnn import HGNNEncoder
from mlp import PairwiseActor, Critic


VALID_ABLATION_VARIANTS = {'full', 'no_hgnn', 'shared_encoder', 'no_reward_norm'}


class SimpleGraphEncoder(nn.Module):
    """
    Lightweight non-HGNN encoder for the NoHGNN ablation.

    It maps normalized USV and task node features independently, then builds a
    graph embedding by mean-pooling both node sets. The output keys and tensor
    dimensions match HGNNEncoder so the PairwiseActor and Critic are unchanged.
    """

    def __init__(self, usv_feat_dim: int = 7, task_feat_dim: int = 8,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.usv_encoder = nn.Sequential(
            nn.Linear(usv_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, state_dict: dict) -> dict:
        usv_embed = self.usv_encoder(state_dict['usv_features'])
        task_embed = self.task_encoder(state_dict['task_features'])
        pool_dim = 1 if usv_embed.dim() == 3 else 0
        graph_embed = torch.cat(
            [usv_embed.mean(dim=pool_dim), task_embed.mean(dim=pool_dim)],
            dim=-1,
        )
        return {
            'usv_embed': usv_embed,
            'task_embed': task_embed,
            'graph_embed': graph_embed,
        }


class PPOAgent:
    """PPO agent with separate encoders for actor and critic."""
    
    def __init__(self, config, n_usvs: int, n_tasks: int,
                 device: str = None, verbose: bool = True):
        """
        Initialize PPO agent with dual HGNN encoders.
        
        Args:
            config: Configuration object
            n_usvs: Number of USVs
            n_tasks: Number of tasks
        """
        self.config = config
        self.n_usvs = n_usvs
        self.n_tasks = n_tasks
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.variant = config.network.get('ablation_variant', 'full')
        if self.variant not in VALID_ABLATION_VARIANTS:
            raise ValueError(
                f"Unknown ablation_variant={self.variant}. "
                f"Expected one of {sorted(VALID_ABLATION_VARIANTS)}"
            )
        self.use_shared_encoder = self.variant == 'shared_encoder'
        
        # Hyperparameters
        self.gamma = config.train.gamma
        self.gae_lambda = config.train.gae_lambda
        self.clip_epsilon = config.train.epsilon
        self.entropy_coef = config.train.get('entropy_coef', 0.01)
        self.value_coef = config.train.get('value_coef', 0.5)
        self.grad_clip = config.train.grad_clip
        self.ppo_epochs = config.train.ppo_epochs
        
        hidden_dim = config.network.hidden_dim
        
        # ============ ENCODER ARCHITECTURE ============
        if self.use_shared_encoder:
            self.actor_encoder = self._build_encoder(hidden_dim).to(self.device)
            self.critic_encoder = self.actor_encoder
        else:
            self.actor_encoder = self._build_encoder(hidden_dim).to(self.device)
            self.critic_encoder = self._build_encoder(hidden_dim).to(self.device)
        
        # Actor network (joint legal pair policy)
        self.actor = PairwiseActor(
            hidden_dim=hidden_dim,
            edge_feat_dim=4,
            graph_dim=hidden_dim * 2,
            hidden_dims=config.network.get('mlp_hidden_dims', [128, 64]),
            dropout=config.network.get('dropout', 0.1)
        ).to(self.device)
        
        # Critic network (value function)
        self.critic = Critic(
            state_dim=hidden_dim * 2,
            hidden_dims=config.network.get('mlp_hidden_dims', [128, 64]),
            dropout=config.network.get('dropout', 0.1)
        ).to(self.device)
        
        # ============ OPTIMIZERS ============
        lr_actor = config.train.lr_actor
        lr_critic = config.train.get('lr_critic', lr_actor)
        lr_encoder = config.train.get('lr_encoder', lr_actor * 0.5)

        if self.use_shared_encoder:
            self.shared_optimizer = optim.AdamW([
                {'params': self.actor_encoder.parameters(), 'lr': lr_encoder, 'name': 'shared_encoder'},
                {'params': self.actor.parameters(), 'lr': lr_actor, 'name': 'actor'},
                {'params': self.critic.parameters(), 'lr': lr_critic, 'name': 'critic'},
            ], weight_decay=1e-4, eps=1e-5)
            self.actor_optimizer = self.shared_optimizer
            self.critic_optimizer = None
            self.actor_scheduler = optim.lr_scheduler.StepLR(
                self.shared_optimizer,
                step_size=config.train.get('lr_decay_step', 100),
                gamma=config.train.get('lr_decay_gamma', 0.95)
            )
            self.critic_scheduler = None
        else:
            # Actor optimizer: actor_encoder + actor
            self.actor_optimizer = optim.AdamW([
                {'params': self.actor_encoder.parameters(), 'lr': lr_encoder, 'name': 'actor_encoder'},
                {'params': self.actor.parameters(), 'lr': lr_actor, 'name': 'actor'}
            ], weight_decay=1e-4, eps=1e-5)

            # Critic optimizer: critic_encoder + critic
            self.critic_optimizer = optim.AdamW([
                {'params': self.critic_encoder.parameters(), 'lr': lr_encoder, 'name': 'critic_encoder'},
                {'params': self.critic.parameters(), 'lr': lr_critic, 'name': 'critic'}
            ], weight_decay=1e-4, eps=1e-5)

            # Learning rate schedulers
            self.actor_scheduler = optim.lr_scheduler.StepLR(
                self.actor_optimizer,
                step_size=config.train.get('lr_decay_step', 100),
                gamma=config.train.get('lr_decay_gamma', 0.95)
            )
            self.critic_scheduler = optim.lr_scheduler.StepLR(
                self.critic_optimizer,
                step_size=config.train.get('lr_decay_step', 100),
                gamma=config.train.get('lr_decay_gamma', 0.95)
            )
        
        # Experience buffer
        self.reset_buffer()
        self.update_count = 0
        
        if verbose:
            print(f"[PPO] Variant: {self.variant}")
            print(f"[PPO] Encoder Architecture: {self._encoder_name()}")
            print(f"  - Actor Encoder: {sum(p.numel() for p in self.actor_encoder.parameters())} params")
            if self.use_shared_encoder:
                print(f"  - Critic Encoder: shared with Actor Encoder")
            else:
                print(f"  - Critic Encoder: {sum(p.numel() for p in self.critic_encoder.parameters())} params")
            print(f"  - Actor: {sum(p.numel() for p in self.actor.parameters())} params")
            print(f"  - Critic: {sum(p.numel() for p in self.critic.parameters())} params")

    def _build_encoder(self, hidden_dim: int) -> nn.Module:
        """Build the encoder selected by the current ablation variant."""
        if self.variant == 'no_hgnn':
            return SimpleGraphEncoder(
                usv_feat_dim=7,
                task_feat_dim=8,
                hidden_dim=hidden_dim,
                dropout=self.config.network.get('dropout', 0.1)
            )
        return HGNNEncoder(
            usv_feat_dim=7,
            task_feat_dim=8,
            edge_feat_dim=4,
            hidden_dim=hidden_dim,
            num_layers=self.config.network.hgnn_layers,
            num_heads=self.config.network.get('n_heads', 4),
            dropout=self.config.network.get('dropout', 0.1)
        )

    def _encoder_name(self) -> str:
        if self.variant == 'no_hgnn':
            return 'SimpleGraphEncoder'
        if self.variant == 'shared_encoder':
            return 'Shared HGNNEncoder'
        return 'Dual HGNNEncoder'
    
    def reset_buffer(self):
        """Clear experience buffer."""
        self.states: List[Dict] = []
        self.actions: List[Tuple[int, int]] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.task_masks: List[torch.Tensor] = []
        self.usv_masks_list: List[torch.Tensor] = []
    
    def _prepare_state(self, state_dict: Dict) -> Dict[str, torch.Tensor]:
        """Convert numpy state to torch tensors."""
        return {
            'usv_features': torch.FloatTensor(state_dict['usv_features']).to(self.device),
            'task_features': torch.FloatTensor(state_dict['task_features']).to(self.device),
            'edge_features': torch.FloatTensor(state_dict['edge_features']).to(self.device)
        }
    
    def _get_masks(self, env) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action masks from environment."""
        task_mask = torch.zeros(self.n_tasks, dtype=torch.bool, device=self.device)
        usv_masks = torch.zeros(self.n_tasks, self.n_usvs, dtype=torch.bool, device=self.device)
        
        available_tasks = env.get_available_tasks()
        
        for task_id in available_tasks:
            task_mask[task_id] = True
            available_usvs = env.get_available_usvs_for_task(task_id)
            for usv_id in available_usvs:
                usv_masks[task_id, usv_id] = True
        
        return task_mask, usv_masks
    
    def select_action(self, env, state_dict: Dict, deterministic: bool = False
                      ) -> Tuple[Tuple[int, int], float, float]:
        """
        Select action using actor encoder and actor network.
        Value estimated using critic encoder and critic network.
        """
        task_mask, usv_masks = self._get_masks(env)
        
        if task_mask.sum() == 0:
            for t in range(self.n_tasks):
                if env.task_states[t, 0] == 0:
                    for u in range(self.n_usvs):
                        return (t, u), -100.0, 0.0
            return (0, 0), -100.0, 0.0
        
        state_tensor = self._prepare_state(state_dict)
        
        with torch.no_grad():
            # Actor uses actor_encoder
            actor_encoded = self.actor_encoder(state_tensor)
            action, log_prob = self.actor.select_action(
                actor_encoded,
                state_tensor['edge_features'],
                usv_masks,
                deterministic
            )
            
            # Critic uses critic_encoder
            critic_encoded = self.critic_encoder(state_tensor)
            critic_embed = critic_encoded['graph_embed']
            value = self.critic(critic_embed)
        
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state_dict: Dict, action: Tuple[int, int],
                        log_prob: float, reward: float, done: bool,
                        value: float, task_mask: torch.Tensor,
                        usv_masks: torch.Tensor):
        """Store transition in buffer."""
        self.states.append(copy.deepcopy(state_dict))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.task_masks.append(task_mask.clone())
        self.usv_masks_list.append(usv_masks.clone())
    
    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Run the configured PPO update implementation."""
        if len(self.states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        if not self.config.train.get('vectorized_update', True):
            return self.legacy_update()
        return self._update_vectorized()

    def _build_update_batch(self) -> Dict[str, torch.Tensor]:
        """Convert the rollout buffer into CPU batched tensors.

        Keeping the full rollout batch on CPU avoids CUDA advanced-indexing
        kernels during mini-batch slicing. Each mini-batch is moved to the
        training device only after CPU slicing, which is more stable on Windows
        display-GPU setups and keeps persistent GPU memory lower.
        """
        advantages, returns = self.compute_gae()
        usv_features = np.stack([s['usv_features'] for s in self.states], axis=0)
        task_features = np.stack([s['task_features'] for s in self.states], axis=0)
        edge_features = np.stack([s['edge_features'] for s in self.states], axis=0)
        actions_flat = np.array(
            [task_id * self.n_usvs + usv_id for task_id, usv_id in self.actions],
            dtype=np.int64
        )

        batch = {
            'usv_features': torch.as_tensor(usv_features, dtype=torch.float32),
            'task_features': torch.as_tensor(task_features, dtype=torch.float32),
            'edge_features': torch.as_tensor(edge_features, dtype=torch.float32),
            'pair_masks': torch.stack(self.usv_masks_list).detach().cpu(),
            'actions_flat': torch.as_tensor(actions_flat, dtype=torch.long),
            'old_log_probs': torch.as_tensor(self.log_probs, dtype=torch.float32),
            'advantages': advantages.detach().cpu(),
            'returns': returns.detach().cpu(),
        }
        self._validate_update_batch(batch)
        return batch

    def _validate_update_batch(self, batch: Dict[str, torch.Tensor]):
        """Fail early on CPU before invalid data can trigger opaque CUDA errors."""
        n_samples = batch['actions_flat'].numel()
        max_action = self.n_tasks * self.n_usvs
        if n_samples == 0:
            raise ValueError("PPO update received an empty rollout batch.")

        min_action = int(batch['actions_flat'].min().item())
        max_seen_action = int(batch['actions_flat'].max().item())
        if min_action < 0 or max_seen_action >= max_action:
            raise ValueError(
                "Invalid flattened action index in PPO buffer: "
                f"min={min_action}, max={max_seen_action}, "
                f"valid_range=[0,{max_action - 1}], "
                f"n_tasks={self.n_tasks}, n_usvs={self.n_usvs}"
            )

        for name in (
            'usv_features', 'task_features', 'edge_features',
            'old_log_probs', 'advantages', 'returns'
        ):
            if not torch.isfinite(batch[name]).all():
                raise ValueError(f"Non-finite values detected in PPO update batch: {name}")

    def _slice_update_batch(self, batch: Dict[str, torch.Tensor],
                            indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Slice state tensors on CPU for one mini-batch."""
        return {
            'usv_features': batch['usv_features'][indices],
            'task_features': batch['task_features'][indices],
            'edge_features': batch['edge_features'][indices],
        }

    def _to_device_state(self, state_tensor: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move one sliced mini-batch state to the training device."""
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in state_tensor.items()
        }

    def _update_vectorized(self) -> Dict[str, float]:
        """Batched mini-batch PPO update."""
        prepare_start = time.monotonic()
        batch = self._build_update_batch()
        batch_prepare_time = time.monotonic() - prepare_start

        n_samples = batch['actions_flat'].size(0)
        batch_size = max(int(self.config.train.get('update_batch_size', 128)), 1)
        update_shuffle = self.config.train.get('update_shuffle', True)

        actor_losses = []
        critic_losses = []
        entropies = []
        actor_update_time = 0.0
        critic_update_time = 0.0

        if self.use_shared_encoder:
            for _ in range(self.ppo_epochs):
                order = (
                    torch.randperm(n_samples)
                    if update_shuffle else torch.arange(n_samples)
                )
                for start in range(0, n_samples, batch_size):
                    idx = order[start:start + batch_size]
                    step_start = time.monotonic()
                    self.shared_optimizer.zero_grad()

                    state_tensor = self._to_device_state(self._slice_update_batch(batch, idx))
                    returns = batch['returns'][idx].to(self.device, non_blocking=True)
                    advantages = batch['advantages'][idx].to(self.device, non_blocking=True)
                    old_log_probs = batch['old_log_probs'][idx].to(self.device, non_blocking=True)
                    actions_flat = batch['actions_flat'][idx].to(self.device, non_blocking=True)
                    pair_masks = batch['pair_masks'][idx].to(self.device, non_blocking=True)
                    encoded = self.actor_encoder(state_tensor)
                    values = self.critic(encoded['graph_embed'])
                    critic_loss = nn.SmoothL1Loss()(values, returns)

                    new_log_probs, entropy, _ = self.actor.get_batch_action_log_prob(
                        encoded,
                        state_tensor['edge_features'],
                        pair_masks,
                        actions_flat,
                    )
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon
                    ) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                    total_loss = actor_loss + critic_loss
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor_encoder.parameters()) +
                        list(self.actor.parameters()) +
                        list(self.critic.parameters()),
                        max_norm=self.grad_clip
                    )
                    self.shared_optimizer.step()
                    elapsed = time.monotonic() - step_start
                    # Shared encoder uses one combined backward pass; split timing
                    # only for log readability so actor+critic roughly matches update time.
                    actor_update_time += elapsed * 0.5
                    critic_update_time += elapsed * 0.5
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropies.append(entropy.mean().item())
        else:
            for _ in range(self.ppo_epochs):
                order = (
                    torch.randperm(n_samples)
                    if update_shuffle else torch.arange(n_samples)
                )

                for start in range(0, n_samples, batch_size):
                    idx = order[start:start + batch_size]
                    step_start = time.monotonic()
                    self.critic_optimizer.zero_grad()
                    state_tensor = self._to_device_state(self._slice_update_batch(batch, idx))
                    returns = batch['returns'][idx].to(self.device, non_blocking=True)
                    critic_encoded = self.critic_encoder(state_tensor)
                    values = self.critic(critic_encoded['graph_embed'])
                    critic_loss = nn.SmoothL1Loss()(values, returns)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.critic_encoder.parameters()) + list(self.critic.parameters()),
                        max_norm=self.grad_clip
                    )
                    self.critic_optimizer.step()
                    critic_update_time += time.monotonic() - step_start
                    critic_losses.append(critic_loss.item())

                for start in range(0, n_samples, batch_size):
                    idx = order[start:start + batch_size]
                    step_start = time.monotonic()
                    self.actor_optimizer.zero_grad()
                    state_tensor = self._to_device_state(self._slice_update_batch(batch, idx))
                    advantages = batch['advantages'][idx].to(self.device, non_blocking=True)
                    old_log_probs = batch['old_log_probs'][idx].to(self.device, non_blocking=True)
                    actions_flat = batch['actions_flat'][idx].to(self.device, non_blocking=True)
                    pair_masks = batch['pair_masks'][idx].to(self.device, non_blocking=True)
                    actor_encoded = self.actor_encoder(state_tensor)
                    new_log_probs, entropy, _ = self.actor.get_batch_action_log_prob(
                        actor_encoded,
                        state_tensor['edge_features'],
                        pair_masks,
                        actions_flat,
                    )
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon
                    ) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
                        max_norm=self.grad_clip
                    )
                    self.actor_optimizer.step()
                    actor_update_time += time.monotonic() - step_start
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy.mean().item())

        self.update_count += 1
        self.reset_buffer()

        return {
            'actor_loss': np.mean(actor_losses) if actor_losses else 0.0,
            'critic_loss': np.mean(critic_losses) if critic_losses else 0.0,
            'entropy': np.mean(entropies) if entropies else 0.0,
            'batch_prepare_time_sec': batch_prepare_time,
            'actor_update_time_sec': actor_update_time,
            'critic_update_time_sec': critic_update_time,
        }

    def legacy_update(self) -> Dict[str, float]:
        """
        Perform the original sample-by-sample PPO update.
        
        Actor and Critic are updated separately with their own encoders.
        This prevents gradient interference and ensures stable learning.
        """
        if len(self.states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}

        if self.use_shared_encoder:
            return self._update_shared_encoder()
        
        # Compute GAE
        advantages, returns = self.compute_gae()
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        
        # Deep copy data
        all_states = [copy.deepcopy(s) for s in self.states]
        all_actions = self.actions.copy()
        all_task_masks = [m.clone() for m in self.task_masks]
        all_usv_masks = [m.clone() for m in self.usv_masks_list]
        
        # Statistics
        actor_losses = []
        critic_losses = []
        entropies = []
        
        n_samples = len(all_states)
        
        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            # ============ UPDATE CRITIC ============
            self.critic_optimizer.zero_grad()
            
            total_critic_loss = 0
            for idx in range(n_samples):
                state_tensor = self._prepare_state(all_states[idx])
                
                # Critic uses its own encoder
                critic_encoded = self.critic_encoder(state_tensor)
                critic_embed = critic_encoded['graph_embed']
                
                new_value = self.critic(critic_embed)
                critic_loss = nn.SmoothL1Loss()(new_value, returns[idx])
                total_critic_loss += critic_loss
            
            total_critic_loss = total_critic_loss / n_samples
            total_critic_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.critic_encoder.parameters()) + list(self.critic.parameters()),
                max_norm=self.grad_clip
            )
            self.critic_optimizer.step()
            critic_losses.append(total_critic_loss.item())
            
            # ============ UPDATE ACTOR ============
            self.actor_optimizer.zero_grad()
            
            total_actor_loss = 0
            total_entropy = 0
            
            for idx in range(n_samples):
                state_tensor = self._prepare_state(all_states[idx])
                action = all_actions[idx]
                task_mask = all_task_masks[idx]
                usv_masks = all_usv_masks[idx]
                
                # Actor uses its own encoder
                actor_encoded = self.actor_encoder(state_tensor)
                # Get joint pair log probability and entropy
                new_log_prob, entropy, _ = self.actor.get_action_log_prob(
                    actor_encoded,
                    state_tensor['edge_features'],
                    usv_masks,
                    action
                )
                
                # Importance ratio
                ratio = torch.exp(new_log_prob - old_log_probs[idx])
                
                # PPO clipped objective
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2)
                
                # Entropy bonus
                entropy_bonus = -self.entropy_coef * entropy
                
                # Total actor loss
                actor_loss = policy_loss + entropy_bonus
                total_actor_loss += actor_loss
                total_entropy += entropy.detach()
            
            total_actor_loss = total_actor_loss / n_samples
            total_actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
                max_norm=self.grad_clip
            )
            self.actor_optimizer.step()
            
            actor_losses.append(total_actor_loss.item())
            entropies.append((total_entropy / n_samples).item())
        
        self.update_count += 1
        self.reset_buffer()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies)
        }

    def _update_shared_encoder(self) -> Dict[str, float]:
        """
        PPO update for the SharedEncoder ablation.

        The encoder appears in a single optimizer and receives the combined
        actor and critic gradients once per PPO epoch.
        """
        advantages, returns = self.compute_gae()
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)

        all_states = [copy.deepcopy(s) for s in self.states]
        all_actions = self.actions.copy()
        all_usv_masks = [m.clone() for m in self.usv_masks_list]

        actor_losses = []
        critic_losses = []
        entropies = []
        n_samples = len(all_states)

        for _ in range(self.ppo_epochs):
            self.shared_optimizer.zero_grad()

            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0

            for idx in range(n_samples):
                state_tensor = self._prepare_state(all_states[idx])
                action = all_actions[idx]
                usv_masks = all_usv_masks[idx]

                encoded = self.actor_encoder(state_tensor)

                new_value = self.critic(encoded['graph_embed'])
                critic_loss = nn.SmoothL1Loss()(new_value, returns[idx])

                new_log_prob, entropy, _ = self.actor.get_action_log_prob(
                    encoded,
                    state_tensor['edge_features'],
                    usv_masks,
                    action
                )
                ratio = torch.exp(new_log_prob - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy.detach()

            total_actor_loss = total_actor_loss / n_samples
            total_critic_loss = total_critic_loss / n_samples
            total_loss = total_actor_loss + total_critic_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.actor_encoder.parameters()) +
                list(self.actor.parameters()) +
                list(self.critic.parameters()),
                max_norm=self.grad_clip
            )
            self.shared_optimizer.step()

            actor_losses.append(total_actor_loss.item())
            critic_losses.append(total_critic_loss.item())
            entropies.append((total_entropy / n_samples).item())

        self.update_count += 1
        self.reset_buffer()

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies)
        }

    def decay_lr(self):
        """Step learning rate schedulers."""
        self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
    
    def get_lr(self) -> Tuple[float, float]:
        """Get current learning rates."""
        return (
            self.actor_optimizer.param_groups[0]['lr'],
            self.actor_optimizer.param_groups[-1]['lr'] if self.use_shared_encoder
            else self.critic_optimizer.param_groups[0]['lr']
        )

    def get_lr_info(self) -> Dict[str, float]:
        """Return named learning rates for logging."""
        if self.use_shared_encoder:
            return {
                'LR Shared Encoder': self.shared_optimizer.param_groups[0]['lr'],
                'LR Actor': self.shared_optimizer.param_groups[1]['lr'],
                'LR Critic': self.shared_optimizer.param_groups[2]['lr'],
            }
        return {
            'LR Actor Encoder': self.actor_optimizer.param_groups[0]['lr'],
            'LR Actor': self.actor_optimizer.param_groups[1]['lr'],
            'LR Critic Encoder': self.critic_optimizer.param_groups[0]['lr'],
            'LR Critic': self.critic_optimizer.param_groups[1]['lr'],
        }
    
    def save(self, path: str):
        """Save checkpoint robustly and return the actual saved path."""
        payload = {
            'variant': self.variant,
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': None if self.use_shared_encoder else self.critic_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': None if self.use_shared_encoder else self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': None if self.use_shared_encoder else self.critic_scheduler.state_dict(),
            'update_count': self.update_count
        }

        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_path = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, path)
            return path
        except (OSError, RuntimeError) as exc:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            root, ext = os.path.splitext(path)
            fallback_path = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}{ext or '.pth'}"
            try:
                torch.save(payload, fallback_path)
                print(
                    f"[Checkpoint Warning] Could not save to {path}: {exc}. "
                    f"Saved fallback checkpoint to {fallback_path}."
                )
                return fallback_path
            except (OSError, RuntimeError) as fallback_exc:
                raise RuntimeError(
                    f"Failed to save checkpoint to {path}. "
                    f"Primary error: {exc}. "
                    f"Fallback error: {fallback_exc}."
                ) from fallback_exc
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        if not self.use_shared_encoder and checkpoint.get('critic_encoder') is not None:
            self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            if not self.use_shared_encoder and checkpoint.get('critic_optimizer') is not None:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'actor_scheduler' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
            if not self.use_shared_encoder and checkpoint.get('critic_scheduler') is not None:
                self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
            self.update_count = checkpoint.get('update_count', 0)
    
    def get_actor_embedding(self, state_dict: Dict) -> torch.Tensor:
        """Get actor's state embedding (for visualization/analysis)."""
        state_tensor = self._prepare_state(state_dict)
        with torch.no_grad():
            encoded = self.actor_encoder(state_tensor)
        return encoded['graph_embed']
    
    def get_critic_embedding(self, state_dict: Dict) -> torch.Tensor:
        """Get critic's state embedding (for visualization/analysis)."""
        state_tensor = self._prepare_state(state_dict)
        with torch.no_grad():
            encoded = self.critic_encoder(state_tensor)
        return encoded['graph_embed']
