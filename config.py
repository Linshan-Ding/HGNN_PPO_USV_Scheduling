"""
Configuration Management for USV Scheduling.

Contains dataclass-based configurations for:
- InstanceConfig: Problem instance parameters
- DataConfig: Data loading/saving paths
- NetworkConfig: Neural network architecture (dual encoder support)
- TrainConfig: Training hyperparameters
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InstanceConfig:
    """Problem instance configuration."""
    n_usvs: int = 2                                             # Number of USVs
    n_tasks: int = 20                                           # Number of tasks
    map_size: Tuple[int, int] = (1000, 1000)                   # Map dimensions
    battery_capacity: float = 3000.0                            # Max battery
    usv_speed: float = 5.0                                      # USV speed (units/time)
    charge_time: float = 10.0                                   # Charging duration
    energy_cost_per_distance: float = 1.0                       # Energy per distance
    energy_cost_per_task_time: float = 5.0                      # Energy per task time

    def get(self, key: str, default=None):
        """Get attribute with default fallback."""
        return getattr(self, key, default)


@dataclass
class DataConfig:
    """Data directory configuration."""
    data_dir: str = 'data/public'
    instance_id: Optional[str] = None
    instance_file: Optional[str] = None
    instance_index: int = 0


@dataclass 
class NetworkConfig:
    """
    Neural network architecture configuration.
    
    Supports dual encoder architecture:
    - Actor Encoder (HGNN): Dedicated for policy network
    - Critic Encoder (HGNN): Dedicated for value network
    """
    hidden_dim: int = 64                                        # HGNN hidden dimension
    hgnn_layers: int = 3                                        # Number of HGNN layers
    n_heads: int = 4                                            # Number of attention heads
    dropout: float = 0.1                                        # Dropout rate
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    
    # Dual encoder architecture is enabled by default
    # Both actor and critic have their own HGNN encoder

    def get(self, key: str, default=None):
        """Get attribute with default fallback."""
        return getattr(self, key, default)


@dataclass
class TrainConfig:
    """
    Training hyperparameters.
    
    Key parameters for dual encoder architecture:
    - lr_encoder: Learning rate for both encoders (lower for stability)
    - lr_actor: Learning rate for actor MLP
    - lr_critic: Learning rate for critic MLP
    """
    # Training schedule
    max_epochs: int = 500
    seed: int = 0
    
    # Learning rates for dual encoder architecture
    lr_actor: float = 3e-4                    # Actor MLP learning rate
    lr_critic: float = 3e-4                   # Critic MLP learning rate
    lr_encoder: float = 1e-4                  # Encoder learning rate (both encoders)
    lr_decay_step: int = 100
    lr_decay_gamma: float = 0.95

    # PPO hyperparameters
    gamma: float = 0.99                       # Discount factor
    gae_lambda: float = 0.95                  # GAE lambda
    epsilon: float = 0.2                      # PPO clip epsilon
    value_coef: float = 0.5                   # Value loss coefficient
    entropy_coef: float = 0.01                # Entropy bonus coefficient
    grad_clip: float = 0.5                    # Gradient clipping threshold
    ppo_epochs: int = 4                       # PPO update epochs per batch
    
    # Multi-trajectory collection
    n_trajectories: int = 8                   # Trajectories per update

    # Deterministic evaluation/checkpointing
    eval_interval: int = 10
    baseline_seed: int = 20260519
    train_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # Logging
    use_visdom: bool = True
    visdom_server: str = 'http://localhost'
    visdom_port: int = 8097
    visdom_env: str = 'usv_training'
    log_interval: int = 10
    save_interval: int = 50

    def get(self, key: str, default=None):
        """Get attribute with default fallback."""
        return getattr(self, key, default)


@dataclass
class Config:
    """Combined configuration."""
    instance: InstanceConfig
    data: DataConfig
    network: NetworkConfig
    train: TrainConfig
    model_dir: str = 'models'
    result_dir: str = 'results'


def get_config(**kwargs) -> Config:
    """
    Create configuration with optional overrides.
    
    Args:
        **kwargs: Override parameters
        
    Example:
        cfg = get_config(
            n_usvs=4, n_tasks=40,
            hidden_dim=128, hgnn_layers=3,
            lr_actor=3e-4, entropy_coef=0.01
        )
    """
    instance_fields = InstanceConfig.__dataclass_fields__
    data_fields = DataConfig.__dataclass_fields__
    network_fields = NetworkConfig.__dataclass_fields__
    train_fields = TrainConfig.__dataclass_fields__
    
    # Instance config
    instance_kwargs = {k: v for k, v in kwargs.items() if k in instance_fields}
    instance = InstanceConfig(**instance_kwargs)
    
    # Data config
    data_kwargs = {k: v for k, v in kwargs.items() if k in data_fields}
    data = DataConfig(**data_kwargs)
    
    # Network config
    network_kwargs = {k: v for k, v in kwargs.items() if k in network_fields}
    network = NetworkConfig(**network_kwargs)
    
    # Train config
    train_kwargs = {k: v for k, v in kwargs.items() if k in train_fields}
    train = TrainConfig(**train_kwargs)
    
    return Config(
        instance=instance,
        data=data,
        network=network,
        train=train,
        model_dir=kwargs.get('model_dir', 'models'),
        result_dir=kwargs.get('result_dir', 'results')
    )
