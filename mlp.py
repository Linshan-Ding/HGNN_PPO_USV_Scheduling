"""
MLP Networks for PPO Agent.

This module implements:
- HierarchicalActor: Two-level actor for task and USV selection
- Critic: State value estimator

The hierarchical actor uses:
- Actor1: Selects task given graph state
- Actor2: Selects USV given graph state and selected task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MLP(nn.Module):
    """Multi-layer perceptron with normalization and dropout."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = 'relu', dropout: float = 0.1):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HierarchicalActor(nn.Module):
    """
    Two-level hierarchical actor network.
    
    Level 1 (Actor1): Select task given graph embedding
    Level 2 (Actor2): Select USV given graph embedding and selected task
    
    Joint probability: P(task, usv) = P(task) * P(usv | task)
    """
    
    def __init__(self, state_dim: int, n_tasks: int, n_usvs: int,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        """
        Args:
            state_dim: Global state feature dimension (typically 2 * hidden_dim)
            n_tasks: Number of tasks (excluding charging)
            n_usvs: Number of USVs
            hidden_dims: MLP hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_tasks = n_tasks
        self.n_usvs = n_usvs
        
        # Actor1: Task selection
        self.actor1 = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=n_tasks,
            dropout=dropout
        )
        
        # Actor2: USV selection (conditioned on task)
        self.actor2 = MLP(
            input_dim=state_dim + n_tasks,  # State + task one-hot
            hidden_dims=hidden_dims,
            output_dim=n_usvs,
            dropout=dropout
        )
    
    def forward(self, graph_embed: torch.Tensor, task_mask: torch.Tensor,
                usv_masks: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute action probabilities for all valid actions.
        
        Args:
            graph_embed: Global graph embedding [state_dim]
            task_mask: Valid task mask [n_tasks] (True = available)
            usv_masks: Valid USV masks per task [n_tasks, n_usvs]
            
        Returns:
            task_logits: Masked task logits [n_tasks]
            task_probs: Task probabilities [n_tasks]
            usv_logits_all: USV logits for each task [n_tasks, n_usvs]
            usv_probs_all: USV probabilities for each task [n_tasks, n_usvs]
        """
        device = graph_embed.device
        
        # Actor1: Compute task scores
        task_logits = self.actor1(graph_embed)
        task_logits_masked = task_logits.clone()
        task_logits_masked[~task_mask] = float('-inf')
        task_probs = F.softmax(task_logits_masked, dim=-1)
        
        # Actor2: Compute USV scores for each valid task
        usv_logits_all = torch.full(
            (self.n_tasks, self.n_usvs), float('-inf'), device=device
        )
        usv_probs_all = torch.zeros(
            (self.n_tasks, self.n_usvs), device=device
        )
        
        for task_idx in range(self.n_tasks):
            if not task_mask[task_idx]:
                continue
            
            # Create task one-hot encoding
            task_onehot = torch.zeros(self.n_tasks, device=device)
            task_onehot[task_idx] = 1.0
            
            # Compute USV scores
            actor2_input = torch.cat([graph_embed, task_onehot], dim=-1)
            usv_logits = self.actor2(actor2_input)
            
            # Apply USV mask
            usv_mask = usv_masks[task_idx]
            usv_logits_masked = usv_logits.clone()
            usv_logits_masked[~usv_mask] = float('-inf')
            
            if usv_mask.sum() > 0:
                usv_probs = F.softmax(usv_logits_masked, dim=-1)
            else:
                usv_probs = torch.zeros_like(usv_logits)
            
            usv_logits_all[task_idx] = usv_logits_masked
            usv_probs_all[task_idx] = usv_probs
        
        return task_logits_masked, task_probs, usv_logits_all, usv_probs_all
    
    def get_action_log_prob(self, graph_embed: torch.Tensor, task_mask: torch.Tensor,
                            usv_masks: torch.Tensor, action: Tuple[int, int]
                            ) -> Tuple[torch.Tensor, ...]:
        """
        Compute log probability and entropy for a given action.
        
        Args:
            graph_embed: Global graph embedding [state_dim]
            task_mask: Valid task mask [n_tasks]
            usv_masks: Valid USV masks per task [n_tasks, n_usvs]
            action: (task_id, usv_id) tuple
            
        Returns:
            log_prob: Joint log probability log P(task, usv)
            log_prob_task: Task log probability
            log_prob_usv: USV log probability (given task)
            entropy_task: Actor1 entropy
            entropy_usv: Actor2 entropy
            task_probs: Task probability distribution
            usv_probs: USV probability distribution (for selected task)
        """
        task_id, usv_id = action
        device = graph_embed.device
        
        # Get probability distributions
        task_logits, task_probs, usv_logits_all, usv_probs_all = self.forward(
            graph_embed, task_mask, usv_masks
        )
        
        # Actor1 log probability
        log_prob_task = torch.log(task_probs[task_id] + 1e-10)
        
        # Actor2 log probability
        usv_probs = usv_probs_all[task_id]
        log_prob_usv = torch.log(usv_probs[usv_id] + 1e-10)
        
        # Joint log probability
        log_prob = log_prob_task + log_prob_usv
        
        # Compute entropies
        valid_task_probs = task_probs[task_mask]
        if len(valid_task_probs) > 0 and valid_task_probs.sum() > 0:
            entropy_task = -(valid_task_probs * torch.log(valid_task_probs + 1e-10)).sum()
        else:
            entropy_task = torch.tensor(0.0, device=device)
        
        usv_mask = usv_masks[task_id]
        valid_usv_probs = usv_probs[usv_mask]
        if len(valid_usv_probs) > 0 and valid_usv_probs.sum() > 0:
            entropy_usv = -(valid_usv_probs * torch.log(valid_usv_probs + 1e-10)).sum()
        else:
            entropy_usv = torch.tensor(0.0, device=device)
        
        return log_prob, log_prob_task, log_prob_usv, entropy_task, entropy_usv, task_probs, usv_probs
    
    def select_action(self, graph_embed: torch.Tensor, task_mask: torch.Tensor,
                      usv_masks: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            graph_embed: Global graph embedding [state_dim]
            task_mask: Valid task mask [n_tasks]
            usv_masks: Valid USV masks per task [n_tasks, n_usvs]
            deterministic: If True, select argmax; else sample
            
        Returns:
            action: (task_id, usv_id) tuple
            log_prob: Joint log probability
        """
        device = graph_embed.device
        
        # Check for valid actions
        if task_mask.sum() == 0:
            return (0, 0), torch.tensor(-100.0, device=device)
        
        # Get probability distributions
        task_logits, task_probs, usv_logits_all, usv_probs_all = self.forward(
            graph_embed, task_mask, usv_masks
        )
        
        # Select task
        if deterministic:
            task_id = torch.argmax(task_probs).item()
        else:
            try:
                valid_probs = task_probs.clone()
                valid_probs[~task_mask] = 0
                prob_sum = valid_probs.sum()
                if prob_sum > 0:
                    normalized_probs = valid_probs / prob_sum
                    task_id = torch.multinomial(normalized_probs, 1).item()
                else:
                    task_id = torch.argmax(task_probs).item()
            except RuntimeError:
                task_id = torch.argmax(task_probs).item()
        
        # Select USV for chosen task
        usv_probs = usv_probs_all[task_id]
        usv_mask = usv_masks[task_id]
        
        if usv_mask.sum() == 0:
            # Fallback: find any valid task-USV pair
            for t_idx in range(self.n_tasks):
                if task_mask[t_idx] and usv_masks[t_idx].sum() > 0:
                    task_id = t_idx
                    usv_probs = usv_probs_all[t_idx]
                    usv_mask = usv_masks[t_idx]
                    break
            else:
                return (0, 0), torch.tensor(-100.0, device=device)
        
        if deterministic:
            usv_id = torch.argmax(usv_probs).item()
        else:
            try:
                valid_probs = usv_probs.clone()
                valid_probs[~usv_mask] = 0
                prob_sum = valid_probs.sum()
                if prob_sum > 0:
                    normalized_probs = valid_probs / prob_sum
                    usv_id = torch.multinomial(normalized_probs, 1).item()
                else:
                    usv_id = torch.argmax(usv_probs).item()
            except RuntimeError:
                usv_id = torch.argmax(usv_probs).item()
        
        # Compute joint log probability
        log_prob_task = torch.log(task_probs[task_id] + 1e-10)
        log_prob_usv = torch.log(usv_probs[usv_id] + 1e-10)
        log_prob = log_prob_task + log_prob_usv
        
        return (task_id, usv_id), log_prob


class PairwiseActor(nn.Module):
    """
    Pairwise actor for direct (task, USV) scoring.

    The policy forms one masked categorical distribution over every legal
    task-USV pair. This avoids the old factorization error where task and USV
    were scored separately without seeing their local edge compatibility.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int = 4,
                 graph_dim: int = None, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.graph_dim = graph_dim if graph_dim is not None else hidden_dim * 2
        pair_input_dim = hidden_dim * 2 + edge_feat_dim + self.graph_dim
        self.scorer = MLP(
            input_dim=pair_input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout
        )

    def forward(self, encoded: dict, edge_features: torch.Tensor,
                pair_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute masked logits and probabilities over legal task-USV pairs.

        Args:
            encoded: HGNN output with task_embed [T,H], usv_embed [U,H],
                graph_embed [G].
            edge_features: Normalized edge features [U,T,E].
            pair_mask: Legal action mask [T,U].

        Returns:
            logits: Masked logits [T,U].
            probs: Joint pair probabilities [T,U].
        """
        task_embed = encoded['task_embed']
        usv_embed = encoded['usv_embed']
        graph_embed = encoded['graph_embed']
        n_tasks = task_embed.size(0)
        n_usvs = usv_embed.size(0)
        device = task_embed.device

        task_exp = task_embed.unsqueeze(1).expand(n_tasks, n_usvs, -1)
        usv_exp = usv_embed.unsqueeze(0).expand(n_tasks, n_usvs, -1)
        edge_tu = edge_features.permute(1, 0, 2)
        graph_exp = graph_embed.view(1, 1, -1).expand(n_tasks, n_usvs, -1)

        pair_inputs = torch.cat([task_exp, usv_exp, edge_tu, graph_exp], dim=-1)
        logits = self.scorer(pair_inputs).squeeze(-1)

        masked_logits = logits.clone()
        masked_logits[~pair_mask] = float('-inf')

        flat_mask = pair_mask.reshape(-1)
        flat_probs = torch.zeros(n_tasks * n_usvs, device=device)
        if flat_mask.sum() > 0:
            flat_logits = masked_logits.reshape(-1)
            flat_probs = F.softmax(flat_logits, dim=0)

        return masked_logits, flat_probs.view(n_tasks, n_usvs)

    def get_action_log_prob(self, encoded: dict, edge_features: torch.Tensor,
                            pair_mask: torch.Tensor, action: Tuple[int, int]
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log probability, entropy, and pair probabilities."""
        task_id, usv_id = action
        logits, probs = self.forward(encoded, edge_features, pair_mask)
        flat_probs = probs.reshape(-1)
        flat_mask = pair_mask.reshape(-1)
        flat_idx = task_id * probs.size(1) + usv_id

        log_prob = torch.log(flat_probs[flat_idx] + 1e-10)
        valid_probs = flat_probs[flat_mask]
        if valid_probs.numel() > 0 and valid_probs.sum() > 0:
            entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum()
        else:
            entropy = torch.tensor(0.0, device=probs.device)

        return log_prob, entropy, probs

    def select_action(self, encoded: dict, edge_features: torch.Tensor,
                      pair_mask: torch.Tensor, deterministic: bool = False
                      ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """Sample or greedily select one legal task-USV pair."""
        if pair_mask.sum() == 0:
            device = encoded['graph_embed'].device
            return (0, 0), torch.tensor(-100.0, device=device)

        _, probs = self.forward(encoded, edge_features, pair_mask)
        flat_probs = probs.reshape(-1)

        if deterministic:
            flat_idx = torch.argmax(flat_probs).item()
        else:
            try:
                flat_idx = torch.multinomial(flat_probs, 1).item()
            except RuntimeError:
                flat_idx = torch.argmax(flat_probs).item()

        n_usvs = probs.size(1)
        task_id = flat_idx // n_usvs
        usv_id = flat_idx % n_usvs
        log_prob = torch.log(flat_probs[flat_idx] + 1e-10)
        return (int(task_id), int(usv_id)), log_prob


class Critic(nn.Module):
    """State value estimator network."""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1):
        """
        Args:
            state_dim: Global state feature dimension
            hidden_dims: MLP hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.value_net = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout
        )
    
    def forward(self, graph_embed: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.
        
        Args:
            graph_embed: Global graph embedding [state_dim] or [batch, state_dim]
            
        Returns:
            State value estimate (scalar or [batch])
        """
        return self.value_net(graph_embed).squeeze(-1)
