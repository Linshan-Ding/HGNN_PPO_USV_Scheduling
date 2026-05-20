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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List

from hgnn import HGNNEncoder
from mlp import PairwiseActor, Critic


class PPOAgent:
    """PPO agent with separate encoders for actor and critic."""
    
    def __init__(self, config, n_usvs: int, n_tasks: int):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.gamma = config.train.gamma
        self.gae_lambda = config.train.gae_lambda
        self.clip_epsilon = config.train.epsilon
        self.entropy_coef = config.train.get('entropy_coef', 0.01)
        self.grad_clip = config.train.grad_clip
        self.ppo_epochs = config.train.ppo_epochs
        
        hidden_dim = config.network.hidden_dim
        
        # ============ DUAL ENCODER ARCHITECTURE ============
        # Actor Encoder - dedicated HGNN for policy
        self.actor_encoder = HGNNEncoder(
            usv_feat_dim=7,
            task_feat_dim=8,
            edge_feat_dim=4,
            hidden_dim=hidden_dim,
            num_layers=config.network.hgnn_layers,
            num_heads=config.network.get('n_heads', 4),
            dropout=config.network.get('dropout', 0.1)
        ).to(self.device)
        
        # Critic Encoder - dedicated HGNN for value estimation
        self.critic_encoder = HGNNEncoder(
            usv_feat_dim=7,
            task_feat_dim=8,
            edge_feat_dim=4,
            hidden_dim=hidden_dim,
            num_layers=config.network.hgnn_layers,
            num_heads=config.network.get('n_heads', 4),
            dropout=config.network.get('dropout', 0.1)
        ).to(self.device)
        
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
        
        # ============ SEPARATE OPTIMIZERS ============
        lr_actor = config.train.lr_actor
        lr_critic = config.train.get('lr_critic', lr_actor)
        lr_encoder = config.train.get('lr_encoder', lr_actor * 0.5)
        
        # Actor optimizer: actor_encoder + actor
        self.actor_optimizer = optim.AdamW([
            {'params': self.actor_encoder.parameters(), 'lr': lr_encoder},
            {'params': self.actor.parameters(), 'lr': lr_actor}
        ], weight_decay=1e-4, eps=1e-5)
        
        # Critic optimizer: critic_encoder + critic
        self.critic_optimizer = optim.AdamW([
            {'params': self.critic_encoder.parameters(), 'lr': lr_encoder},
            {'params': self.critic.parameters(), 'lr': lr_critic}
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
        
        print(f"[PPO] Dual Encoder Architecture:")
        print(f"  - Actor Encoder: {sum(p.numel() for p in self.actor_encoder.parameters())} params")
        print(f"  - Critic Encoder: {sum(p.numel() for p in self.critic_encoder.parameters())} params")
        print(f"  - Actor: {sum(p.numel() for p in self.actor.parameters())} params")
        print(f"  - Critic: {sum(p.numel() for p in self.critic.parameters())} params")
    
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
        """
        Perform PPO update with dual encoders.
        
        Actor and Critic are updated separately with their own encoders.
        This prevents gradient interference and ensures stable learning.
        """
        if len(self.states) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
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

    def decay_lr(self):
        """Step learning rate schedulers."""
        self.actor_scheduler.step()
        self.critic_scheduler.step()
    
    def get_lr(self) -> Tuple[float, float]:
        """Get current learning rates."""
        return (
            self.actor_optimizer.param_groups[0]['lr'],
            self.critic_optimizer.param_groups[0]['lr']
        )
    
    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'actor_encoder': self.actor_encoder.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_encoder.load_state_dict(checkpoint['actor_encoder'])
        self.critic_encoder.load_state_dict(checkpoint['critic_encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        
        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'actor_scheduler' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
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
