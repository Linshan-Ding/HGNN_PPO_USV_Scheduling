"""
Heterogeneous Graph Neural Network Encoder.

This module implements a heterogeneous GNN for encoding the USV scheduling state:
- USV nodes with position, battery, and status features
- Task nodes with position, duration, and scheduling features
- Edge features representing USV-task relationships (distance, energy cost, etc.)

Architecture:
1. Feature normalization
2. Initial embedding with edge feature aggregation
3. Multi-layer bipartite attention message passing
4. Graph-level pooling for global representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNormalizer(nn.Module):
    """
    Feature normalizer with support for batch normalization.
    Handles variable batch sizes and different tensor shapes.
    """
    
    def __init__(self, feat_dim: int, norm_type: str = 'batch'):
        """
        Args:
            feat_dim: Feature dimension to normalize
            norm_type: Normalization type ('batch' or 'layer')
        """
        super().__init__()
        self.norm_type = norm_type
        self.feat_dim = feat_dim
        
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(feat_dim)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(feat_dim)
        else:
            self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization.
        
        Args:
            x: Input tensor [N, feat_dim] or [N, M, feat_dim]
            
        Returns:
            Normalized tensor with same shape
        """
        if self.norm is None:
            return x
            
        original_shape = x.shape
        
        if self.norm_type == 'layer':
            return self.norm(x)
        
        # Batch normalization
        if len(original_shape) == 3:
            # Reshape [N, M, D] -> [N*M, D] for BatchNorm
            x_flat = x.reshape(-1, self.feat_dim)
            if x_flat.size(0) > 1:
                x_norm = self.norm(x_flat)
            else:
                x_norm = x_flat
            return x_norm.reshape(original_shape)
        else:
            if x.size(0) > 1:
                return self.norm(x)
            return x


class BipartiteAttentionLayer(nn.Module):
    """
    Bipartite graph attention layer for USV-Task message passing.
    Uses multi-head attention to aggregate information from source to target nodes.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        
        # Self-loop projection
        self.W_self = nn.Linear(in_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor, 
                adj_mask: torch.Tensor) -> torch.Tensor:
        """
        Message passing from source to target nodes.
        
        Args:
            src_feats: Source node features [num_src, in_dim]
            tgt_feats: Target node features [num_tgt, in_dim]
            adj_mask: Adjacency mask [num_src, num_tgt] (True = connected)
            
        Returns:
            Updated target features [num_tgt, out_dim]
        """
        squeeze_batch = False
        if src_feats.dim() == 2:
            src_feats = src_feats.unsqueeze(0)
            tgt_feats = tgt_feats.unsqueeze(0)
            if adj_mask.dim() == 2:
                adj_mask = adj_mask.unsqueeze(0)
            squeeze_batch = True

        batch_size = src_feats.size(0)
        num_src = src_feats.size(1)
        num_tgt = tgt_feats.size(1)
        
        # Compute Q, K, V
        Q = self.W_q(tgt_feats).view(batch_size, num_tgt, self.num_heads, self.head_dim)
        K = self.W_k(src_feats).view(batch_size, num_src, self.num_heads, self.head_dim)
        V = self.W_v(src_feats).view(batch_size, num_src, self.num_heads, self.head_dim)
        
        # Transpose to [batch, heads, nodes, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores [batch, heads, num_tgt, num_src]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply adjacency mask
        if adj_mask.dim() == 2:
            adj_mask = adj_mask.unsqueeze(0).expand(batch_size, -1, -1)
        adj_mask_t = adj_mask.transpose(-2, -1)  # [batch, num_tgt, num_src]
        mask_exp = adj_mask_t.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(~mask_exp, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate neighbor information
        neighbor_info = torch.matmul(attn_weights, V)  # [batch, heads, num_tgt, head_dim]
        neighbor_info = (
            neighbor_info.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tgt, -1)
        )
        
        # Combine with self information
        self_info = self.W_self(tgt_feats)
        output = neighbor_info + self_info
        
        # Residual connection and layer norm
        if tgt_feats.size(-1) == self.out_dim:
            output = self.layer_norm(output + tgt_feats)
        else:
            output = self.layer_norm(output)
        
        return output.squeeze(0) if squeeze_batch else output


class HGNNEncoder(nn.Module):
    """
    Heterogeneous Graph Neural Network Encoder.
    
    Encodes USV-Task scheduling state into:
    - USV embeddings [n_usvs, hidden_dim]
    - Task embeddings [n_tasks, hidden_dim]
    - Graph embedding [2 * hidden_dim]
    """
    
    def __init__(self, usv_feat_dim: int = 7, task_feat_dim: int = 8, 
                 edge_feat_dim: int = 4, hidden_dim: int = 64,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            usv_feat_dim: USV feature dimension
            task_feat_dim: Task feature dimension  
            edge_feat_dim: Edge feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of message passing layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature normalizers
        self.usv_normalizer = FeatureNormalizer(usv_feat_dim, norm_type='layer')
        self.task_normalizer = FeatureNormalizer(task_feat_dim, norm_type='layer')
        self.edge_normalizer = FeatureNormalizer(edge_feat_dim, norm_type='layer')
        
        # Initial projection layers (with edge feature concatenation)
        self.usv_init_projection = nn.Linear(
            usv_feat_dim + task_feat_dim + edge_feat_dim,
            hidden_dim
        )
        self.task_init_projection = nn.Linear(
            task_feat_dim + usv_feat_dim + edge_feat_dim,
            hidden_dim
        )
        
        # Bipartite attention layers
        self.task_to_usv_layers = nn.ModuleList([
            BipartiteAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.usv_to_task_layers = nn.ModuleList([
            BipartiteAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Graph pooling layers
        self.usv_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.task_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def _aggregate_with_edges(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor,
                              edge_feats: torch.Tensor, adj_mask: torch.Tensor,
                              is_task_to_usv: bool) -> torch.Tensor:
        """
        Aggregate source features with edge features for initial embedding.
        
        Args:
            src_feats: Source node features
            tgt_feats: Target node features (unused, for signature consistency)
            edge_feats: Edge features [n_usvs, n_tasks, edge_dim]
            adj_mask: Adjacency mask [n_usvs, n_tasks]
            is_task_to_usv: Direction of aggregation
            
        Returns:
            Aggregated features for each target node
        """
        squeeze_batch = False
        if src_feats.dim() == 2:
            src_feats = src_feats.unsqueeze(0)
            tgt_feats = tgt_feats.unsqueeze(0)
            edge_feats = edge_feats.unsqueeze(0)
            squeeze_batch = True

        batch_size = src_feats.size(0)
        num_tgt = tgt_feats.size(1)

        if is_task_to_usv:
            # USV aggregates the mean task feature plus mean edge feature per USV.
            src_mean = src_feats.mean(dim=1, keepdim=True).expand(batch_size, num_tgt, -1)
            edge_mean = edge_feats.mean(dim=2)
        else:
            # Task aggregates the mean USV feature plus mean edge feature per task.
            src_mean = src_feats.mean(dim=1, keepdim=True).expand(batch_size, num_tgt, -1)
            edge_mean = edge_feats.mean(dim=1)

        aggregated = torch.cat([src_mean, edge_mean], dim=-1)
        return aggregated.squeeze(0) if squeeze_batch else aggregated
    
    def forward(self, state_dict: dict) -> dict:
        """
        Encode scheduling state.
        
        Args:
            state_dict: {
                'usv_features': [n_usvs, usv_feat_dim],
                'task_features': [n_tasks, task_feat_dim],
                'edge_features': [n_usvs, n_tasks, edge_feat_dim]
            }
            
        Returns:
            {
                'usv_embed': [n_usvs, hidden_dim],
                'task_embed': [n_tasks, hidden_dim],
                'graph_embed': [2 * hidden_dim]
            }
        """
        usv_feats = state_dict['usv_features']
        task_feats = state_dict['task_features']
        edge_feats = state_dict['edge_features']
        
        batched = usv_feats.dim() == 3
        num_usvs = usv_feats.size(-2)
        num_tasks = task_feats.size(-2)
        device = usv_feats.device
        
        # Step 1: Normalize features
        usv_feats = self.usv_normalizer(usv_feats)
        task_feats = self.task_normalizer(task_feats)
        edge_feats = self.edge_normalizer(edge_feats)
        
        # Fully connected adjacency
        if batched:
            adj_matrix = torch.ones(
                usv_feats.size(0), num_usvs, num_tasks,
                dtype=torch.bool,
                device=device
            )
        else:
            adj_matrix = torch.ones(num_usvs, num_tasks, dtype=torch.bool, device=device)
        
        # Step 2: Initial embedding with edge aggregation
        usv_agg = self._aggregate_with_edges(
            task_feats, usv_feats, edge_feats, adj_matrix, is_task_to_usv=True
        )
        usv_embeds = self.usv_init_projection(torch.cat([usv_feats, usv_agg], dim=-1))
        
        task_agg = self._aggregate_with_edges(
            usv_feats, task_feats, edge_feats, adj_matrix, is_task_to_usv=False
        )
        task_embeds = self.task_init_projection(torch.cat([task_feats, task_agg], dim=-1))
        
        # Step 3: Multi-layer bipartite attention
        for layer_idx in range(self.num_layers):
            # Task -> USV message passing
            usv_embeds_new = self.task_to_usv_layers[layer_idx](
                src_feats=task_embeds,
                tgt_feats=usv_embeds,
                adj_mask=adj_matrix.transpose(-2, -1)
            )
            
            # USV -> Task message passing
            task_embeds_new = self.usv_to_task_layers[layer_idx](
                src_feats=usv_embeds_new,
                tgt_feats=task_embeds,
                adj_mask=adj_matrix
            )
            
            usv_embeds = usv_embeds_new
            task_embeds = task_embeds_new
        
        # Step 4: Graph-level pooling
        pool_dim = 1 if batched else 0
        usv_pool = self.usv_pooling(usv_embeds.mean(dim=pool_dim))
        task_pool = self.task_pooling(task_embeds.mean(dim=pool_dim))
        graph_embed = torch.cat([usv_pool, task_pool], dim=-1)
        
        return {
            'usv_embed': usv_embeds,
            'task_embed': task_embeds,
            'graph_embed': graph_embed
        }
