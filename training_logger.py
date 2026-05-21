"""CSV logging utilities for training runs."""

import csv
import math
import os
import re
from datetime import datetime
from typing import Dict, Optional


TRAINING_LOG_FIELDS = [
    'run_id',
    'algorithm',
    'variant',
    'instance_id',
    'n_usvs',
    'n_tasks',
    'seed',
    'epoch',
    'timestamp',
    'elapsed_sec',
    'train_reward_avg',
    'train_reward_std',
    'train_makespan_avg',
    'train_makespan_min',
    'train_makespan_std',
    'success_rate',
    'n_trajectories',
    'n_success',
    'n_failed',
    'eval_makespan',
    'eval_success',
    'best_eval_makespan',
    'best_eval_epoch',
    'gap_to_best_rule_percent',
    'best_rule_name',
    'best_rule_makespan',
    'random_makespan',
    'actor_loss',
    'critic_loss',
    'entropy',
    'lr_actor_encoder',
    'lr_actor',
    'lr_critic_encoder',
    'lr_critic',
    'lr_shared_encoder',
    'hidden_dim',
    'hgnn_layers',
    'n_heads',
    'ppo_epochs',
    'vectorized_update',
    'update_batch_size',
    'update_shuffle',
    'gamma',
    'gae_lambda',
    'clip_epsilon',
    'entropy_coef',
    'reward_normalization',
    'best_model_path',
    'rollout_time_sec',
    'update_time_sec',
    'epoch_time_sec',
    'batch_prepare_time_sec',
    'actor_update_time_sec',
    'critic_update_time_sec',
]


def timestamp_for_id() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def safe_run_id(*parts: object) -> str:
    """Build a compact filesystem-safe run ID."""
    text = '_'.join(str(part) for part in parts if part not in (None, ''))
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text)
    return text.strip('_')


def _clean_value(value):
    if value is None:
        return ''
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ''
    if isinstance(value, bool):
        return int(value)
    return value


class TrainingCSVLogger:
    """Append epoch-level training metrics to a CSV file and flush immediately."""

    def __init__(self, log_dir: str, run_id: str, fieldnames=None):
        self.log_dir = log_dir
        self.run_id = run_id
        self.fieldnames = fieldnames or TRAINING_LOG_FIELDS
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f'{run_id}.csv')
        self.file = open(self.path, 'w', newline='', encoding='utf-8-sig')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict):
        clean_row = {field: _clean_value(row.get(field)) for field in self.fieldnames}
        self.writer.writerow(clean_row)
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def make_training_run_id(
    algorithm: str,
    variant: str,
    instance_id: str,
    seed: int,
    created_at: Optional[str] = None,
) -> str:
    """Create the standard training run ID."""
    return safe_run_id(
        algorithm,
        variant,
        instance_id,
        f'seed{seed}',
        created_at or timestamp_for_id(),
    )
