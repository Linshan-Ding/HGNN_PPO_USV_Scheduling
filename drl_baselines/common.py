"""Shared helpers for DRL comparison algorithms."""

import copy
import os
import random
import time
from typing import Dict, Tuple

import numpy as np

from baseline_protocol import AlgorithmResult


def require_torch():
    """Import PyTorch lazily so registry tests can run without torch installed."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to train/evaluate DRL baselines. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch, nn, optim


def get_cfg_attr(cfg, section: str, name: str, default):
    """Read cfg.section.name with a default fallback."""
    obj = getattr(cfg, section, None)
    if obj is None:
        return default
    return getattr(obj, name, default)


class DRLVisdomLogger:
    """Small Visdom logger shared by DRL baseline algorithms."""

    def __init__(self, cfg, algorithm_name: str, instance: dict):
        self.enabled = False
        self.wins = {}
        self.algorithm_name = algorithm_name
        self.instance_id = str(instance.get("instance_id", "unknown"))
        self.server = get_cfg_attr(cfg, "train", "visdom_server", "http://localhost")
        self.port = get_cfg_attr(cfg, "train", "visdom_port", 8097)
        base_env = get_cfg_attr(cfg, "train", "visdom_env", "drl_baselines")
        self.env = f"{base_env}_{algorithm_name}_{self.instance_id}"

        if not get_cfg_attr(cfg, "train", "use_visdom", False):
            return

        try:
            from visdom import Visdom
            self.viz = Visdom(server=self.server, port=self.port, env=self.env)
            if self.viz.check_connection():
                self.enabled = True
                print(f"[Visdom:{algorithm_name}] Connected: {self.server}:{self.port}, env={self.env}")
            else:
                print(f"[Visdom:{algorithm_name}] Server not reachable at {self.server}:{self.port}")
                print(f"[Visdom:{algorithm_name}] Start it with: python -m visdom.server -port {self.port}")
        except ImportError:
            print("[Visdom] Package not installed. Install with: pip install visdom")
        except Exception as exc:
            print(f"[Visdom:{algorithm_name}] Not available: {exc}")

    def plot(self, name: str, x: int, y):
        """Plot one scalar if Visdom is enabled."""
        if not self.enabled or y is None:
            return
        y_arr = np.array([float(y)])
        x_arr = np.array([int(x)])
        if name not in self.wins:
            self.wins[name] = self.viz.line(
                X=x_arr,
                Y=y_arr,
                opts=dict(title=name, xlabel="Epoch", ylabel=name),
            )
        else:
            self.viz.line(X=x_arr, Y=y_arr, win=self.wins[name], update="append")

    def log_metrics(self, epoch: int, metrics: Dict[str, object]):
        """Plot all numeric scalar metrics."""
        for name, value in metrics.items():
            if value is not None:
                self.plot(name, epoch, value)

    def text(self, name: str, content: str):
        """Show a text panel if Visdom is enabled."""
        if not self.enabled:
            return
        self.wins[name] = self.viz.text(content, win=self.wins.get(name))


def make_visdom_logger(cfg, algorithm_name: str, instance: dict) -> DRLVisdomLogger:
    """Create a DRL baseline Visdom logger."""
    return DRLVisdomLogger(cfg, algorithm_name, instance)


def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and Torch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_result(algorithm_name: str, category: str, instance: dict,
                makespan: float, success: bool, runtime_sec: float,
                seed: int) -> AlgorithmResult:
    """Build a unified AlgorithmResult from an instance and metrics."""
    return AlgorithmResult(
        algorithm_name=algorithm_name,
        category=category,
        instance_id=str(instance.get("instance_id", "unknown")),
        n_usvs=int(instance.get("n_usvs", 0)),
        n_tasks=int(instance.get("n_tasks", 0)),
        makespan=float(makespan),
        success=bool(success),
        runtime_sec=float(runtime_sec),
        seed=int(seed),
    )


def prepare_state(state_dict: Dict, device):
    """Convert environment numpy state into torch tensors."""
    torch, _, _ = require_torch()
    return {
        "usv_features": torch.FloatTensor(state_dict["usv_features"]).to(device),
        "task_features": torch.FloatTensor(state_dict["task_features"]).to(device),
        "edge_features": torch.FloatTensor(state_dict["edge_features"]).to(device),
    }


def get_action_masks(env, n_tasks: int, n_usvs: int, device):
    """Return valid task mask and valid task-USV pair mask."""
    torch, _, _ = require_torch()
    task_mask = torch.zeros(n_tasks, dtype=torch.bool, device=device)
    pair_mask = torch.zeros(n_tasks, n_usvs, dtype=torch.bool, device=device)

    for task_id in env.get_available_tasks():
        task_mask[task_id] = True
        for usv_id in env.get_available_usvs_for_task(task_id):
            pair_mask[task_id, usv_id] = True

    return task_mask, pair_mask


def pair_to_flat(action: Tuple[int, int], n_usvs: int) -> int:
    """Convert (task_id, usv_id) to flattened pair index."""
    task_id, usv_id = action
    return int(task_id) * int(n_usvs) + int(usv_id)


def flat_to_pair(flat_index: int, n_usvs: int) -> Tuple[int, int]:
    """Convert flattened pair index to (task_id, usv_id)."""
    return int(flat_index) // int(n_usvs), int(flat_index) % int(n_usvs)


def random_legal_action(pair_mask) -> Tuple[int, int]:
    """Sample a legal pair uniformly from a pair mask."""
    legal = pair_mask.reshape(-1).nonzero(as_tuple=False).view(-1)
    if legal.numel() == 0:
        return 0, 0
    flat_index = legal[random.randrange(legal.numel())].item()
    return flat_to_pair(flat_index, pair_mask.size(1))


def build_actor_components(cfg, device):
    """Create HGNN encoder plus pairwise actor."""
    from hgnn import HGNNEncoder
    from mlp import PairwiseActor

    hidden_dim = get_cfg_attr(cfg, "network", "hidden_dim", 64)
    encoder = HGNNEncoder(
        usv_feat_dim=7,
        task_feat_dim=8,
        edge_feat_dim=4,
        hidden_dim=hidden_dim,
        num_layers=get_cfg_attr(cfg, "network", "hgnn_layers", 3),
        num_heads=get_cfg_attr(cfg, "network", "n_heads", 4),
        dropout=get_cfg_attr(cfg, "network", "dropout", 0.1),
    ).to(device)
    actor = PairwiseActor(
        hidden_dim=hidden_dim,
        edge_feat_dim=4,
        graph_dim=hidden_dim * 2,
        hidden_dims=get_cfg_attr(cfg, "network", "mlp_hidden_dims", [128, 64]),
        dropout=get_cfg_attr(cfg, "network", "dropout", 0.1),
    ).to(device)
    return encoder, actor


def build_critic_components(cfg, device):
    """Create HGNN encoder plus critic."""
    from hgnn import HGNNEncoder
    from mlp import Critic

    hidden_dim = get_cfg_attr(cfg, "network", "hidden_dim", 64)
    encoder = HGNNEncoder(
        usv_feat_dim=7,
        task_feat_dim=8,
        edge_feat_dim=4,
        hidden_dim=hidden_dim,
        num_layers=get_cfg_attr(cfg, "network", "hgnn_layers", 3),
        num_heads=get_cfg_attr(cfg, "network", "n_heads", 4),
        dropout=get_cfg_attr(cfg, "network", "dropout", 0.1),
    ).to(device)
    critic = Critic(
        state_dim=hidden_dim * 2,
        hidden_dims=get_cfg_attr(cfg, "network", "mlp_hidden_dims", [128, 64]),
        dropout=get_cfg_attr(cfg, "network", "dropout", 0.1),
    ).to(device)
    return encoder, critic


def discounted_returns(rewards, gamma: float, device):
    """Compute discounted returns for one episode."""
    torch, _, _ = require_torch()
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.insert(0, running)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def normalize_tensor(values):
    """Normalize a tensor when it has enough samples."""
    if values.numel() <= 1:
        return values
    return (values - values.mean()) / (values.std() + 1e-8)


def checkpoint_path(cfg, algorithm_name: str, instance: dict, seed: int) -> str:
    """Build an algorithm-specific best checkpoint path."""
    model_dir = getattr(cfg, "model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)
    instance_id = str(instance.get("instance_id", "unknown"))
    return os.path.join(model_dir, f"best_{algorithm_name}_{instance_id}_seed{seed}.pth")


def evaluate_pairwise_policy(agent, instance: dict) -> Tuple[float, bool]:
    """Deterministically evaluate any agent exposing select_action(...)."""
    from env import USVSchedulingEnv

    module_names = [
        "actor_encoder",
        "actor",
        "critic_encoder",
        "critic",
        "online_encoder",
        "online_head",
        "target_encoder",
        "target_head",
    ]
    modules = []
    for name in module_names:
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "training"):
            modules.append((module, module.training))
            module.eval()

    env = USVSchedulingEnv(instance)
    state = env.reset()
    done = False
    info = {}
    step = 0
    max_steps = env.n_tasks * 10

    while not done and step < max_steps:
        _, pair_mask = get_action_masks(env, env.n_tasks, env.n_usvs, agent.device)
        if pair_mask.sum() == 0:
            break
        action = agent.select_action(env, state, deterministic=True)
        state, _, done, info = env.step(action[0], action[1])
        step += 1

    success = env.n_scheduled_tasks == env.n_tasks
    makespan = info.get("makespan", float("inf")) if success else float("inf")
    for module, was_training in modules:
        module.train(was_training)
    return makespan, success


def now() -> float:
    """Return a monotonic-ish wall clock timestamp."""
    return time.time()


def copy_state(state: Dict) -> Dict:
    """Deep-copy state dictionaries stored in replay buffers."""
    return copy.deepcopy(state)
