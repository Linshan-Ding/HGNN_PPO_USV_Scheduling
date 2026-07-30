"""Base interface for deep reinforcement learning comparison algorithms."""

import os
import time
from typing import Dict, List

from baseline_protocol import AlgorithmResult


class DRLBaseline:
    """Common interface for DRL baselines."""

    algorithm_name = "DRLBase"
    category = "drl"
    implemented = False

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def set_instance(self, instance: dict):
        """Re-target mask/index bounds to `instance` (networks are size-agnostic)."""
        self.n_usvs = instance["n_usvs"]
        self.n_tasks = instance["n_tasks"]

    def train_epoch(self, instance: dict, cfg, epoch: int) -> dict:
        """Run one training epoch on `instance`.

        Returns a dict with keys: train_makespans (list of successful episode
        makespans), steps_collected (env decisions this epoch), loss_info
        (dict; keys among actor_loss/critic_loss/entropy/q_loss), and
        optionally exploration_epsilon.
        """
        raise NotImplementedError(
            f"{self.algorithm_name} train_epoch() is not implemented.")

    def train(self, instance: dict, cfg=None) -> AlgorithmResult:
        """Train the DRL baseline on one instance."""
        return AlgorithmResult.not_implemented(
            self.algorithm_name, self.category, instance, self.seed
        )

    def evaluate(self, instance: dict, cfg=None) -> AlgorithmResult:
        """Evaluate the DRL baseline on one instance."""
        return AlgorithmResult.not_implemented(
            self.algorithm_name, self.category, instance, self.seed
        )

    def train_round_robin(self, instances: List[dict], cfg,
                          rules_by_instance: Dict[str, dict]) -> dict:
        """Train one policy over all instances in round-robin order.

        Mirrors multi_train.train_round_robin: epoch k trains instance
        (k-1) mod N, every visit ends with a deterministic evaluation, and
        per-instance best checkpoints are saved via checkpoint_path().
        """
        import numpy as np

        from training_logger import TrainingCSVLogger, make_training_run_id

        from .common import (checkpoint_path, evaluate_pairwise_policy_timed,
                             get_cfg_attr, make_visdom_logger, set_seed)

        set_seed(self.seed)
        self._build(cfg, instances[0]["n_usvs"], instances[0]["n_tasks"])

        max_epochs = get_cfg_attr(cfg, "train", "max_epochs", 5000)
        n_trajectories = get_cfg_attr(cfg, "train", "n_trajectories", 8)
        log_interval = get_cfg_attr(cfg, "train", "log_interval", 10)

        run_id = make_training_run_id(
            self.algorithm_name, "baseline", "public25", self.seed)
        csv_logger = None
        training_log_path = None
        if get_cfg_attr(cfg, "train", "save_training_csv", True):
            log_dir = get_cfg_attr(
                cfg, "train", "training_log_dir",
                os.path.join("results", "training_logs"))
            csv_logger = TrainingCSVLogger(log_dir, run_id)
            training_log_path = csv_logger.path
            print(f"[TrainingLog] {training_log_path}")

        viz = make_visdom_logger(cfg, self.algorithm_name,
                                 {"instance_id": "public25"})

        best_eval_makespan: Dict[str, float] = {}
        best_eval_epoch: Dict[str, int] = {}
        best_paths: Dict[str, str] = {}
        training_start_time = time.monotonic()
        print(f"[RoundRobin:{self.algorithm_name}] {len(instances)} instances x "
              f"{max_epochs // len(instances)} visits ({max_epochs} epochs), "
              f"seed={self.seed}")

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.monotonic()
            instance = instances[(epoch - 1) % len(instances)]
            instance_id = instance["instance_id"]
            visit = (epoch - 1) // len(instances) + 1
            baseline = rules_by_instance[instance_id]

            self.set_instance(instance)
            epoch_info = self.train_epoch(instance, cfg, epoch)
            train_makespans = epoch_info.get("train_makespans", [])
            loss_info = epoch_info.get("loss_info", {})
            n_success = len(train_makespans)
            success_rate = n_success / max(n_trajectories, 1)

            eval_result = evaluate_pairwise_policy_timed(self, instance)
            eval_makespan = eval_result["makespan"]
            gap_percent = None
            if eval_result["success"]:
                gap_percent = (
                    (eval_makespan - baseline["best_rule_makespan"]) /
                    baseline["best_rule_makespan"] * 100.0
                )
                if eval_makespan < best_eval_makespan.get(instance_id, float("inf")):
                    best_eval_makespan[instance_id] = eval_makespan
                    best_eval_epoch[instance_id] = epoch
                    path = checkpoint_path(cfg, self.algorithm_name,
                                           instance, self.seed)
                    self.save(path)
                    best_paths[instance_id] = path

            epoch_time_sec = time.monotonic() - epoch_start

            viz.plot("Eval Makespan by Instance", epoch,
                     eval_makespan if eval_result["success"] else None,
                     trace=instance_id)
            if gap_percent is not None:
                viz.plot("Gap vs Best Rule (%) by Instance", epoch,
                         gap_percent, trace=instance_id)
            viz.log_metrics(epoch, {
                "Actor Loss": loss_info.get("actor_loss"),
                "Critic Loss": loss_info.get("critic_loss"),
                "Q Loss": loss_info.get("q_loss"),
                "Entropy": loss_info.get("entropy"),
                "Success Rate": success_rate,
                "Epoch Time (s)": epoch_time_sec,
            })

            if csv_logger is not None:
                steps_collected = epoch_info.get("steps_collected")
                csv_logger.log({
                    "run_id": run_id,
                    "algorithm": self.algorithm_name,
                    "variant": "baseline",
                    "instance_id": instance_id,
                    "n_usvs": instance["n_usvs"],
                    "n_tasks": instance["n_tasks"],
                    "seed": self.seed,
                    "epoch": epoch,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_sec": time.monotonic() - training_start_time,
                    "train_makespan_avg": float(np.mean(train_makespans)) if train_makespans else None,
                    "train_makespan_min": float(np.min(train_makespans)) if train_makespans else None,
                    "train_makespan_std": float(np.std(train_makespans)) if train_makespans else None,
                    "success_rate": success_rate,
                    "n_trajectories": n_trajectories,
                    "n_success": n_success,
                    "n_failed": max(n_trajectories - n_success, 0),
                    "eval_makespan": eval_makespan if eval_result["success"] else None,
                    "eval_success": eval_result["success"],
                    "best_eval_makespan": best_eval_makespan.get(instance_id),
                    "best_eval_epoch": best_eval_epoch.get(instance_id),
                    "gap_to_best_rule_percent": gap_percent,
                    "best_rule_name": baseline["best_rule_name"],
                    "best_rule_makespan": baseline["best_rule_makespan"],
                    "random_makespan": baseline.get("random_makespan"),
                    "actor_loss": loss_info.get("actor_loss"),
                    "critic_loss": loss_info.get("critic_loss", loss_info.get("q_loss")),
                    "entropy": loss_info.get("entropy"),
                    "hidden_dim": get_cfg_attr(cfg, "network", "hidden_dim", 64),
                    "hgnn_layers": get_cfg_attr(cfg, "network", "hgnn_layers", 3),
                    "n_heads": get_cfg_attr(cfg, "network", "n_heads", 4),
                    "gamma": get_cfg_attr(cfg, "train", "gamma", 0.99),
                    "entropy_coef": get_cfg_attr(cfg, "train", "entropy_coef", 0.01),
                    "best_model_path": best_paths.get(instance_id),
                    "epoch_time_sec": epoch_time_sec,
                    "protocol": "round_robin",
                    "visit_index": visit,
                    "steps_collected": steps_collected,
                    "eval_steps": eval_result["steps"],
                    "eval_solve_time_sec": eval_result["solve_time_sec"],
                    "eval_time_per_decision_ms": eval_result["time_per_decision_ms"],
                    "exploration_epsilon": epoch_info.get("exploration_epsilon"),
                })

            if epoch % log_interval == 0 or epoch == 1:
                eval_str = (f"{eval_makespan:7.1f}" if eval_result["success"]
                            else "  fail")
                gap_str = (f"{gap_percent:+6.2f}%" if gap_percent is not None
                           else "   n/a")
                print(f"[{self.algorithm_name}] Ep {epoch:4d} "
                      f"[{instance_id:9s} v{visit:3d}] | Eval:{eval_str} "
                      f"Gap:{gap_str} | SR:{success_rate:.0%} | "
                      f"T:{epoch_time_sec:.1f}s")

        if csv_logger is not None:
            csv_logger.close()

        print(f"[{self.algorithm_name}] Done. Per-instance best checkpoints: "
              f"{len(best_paths)}/{len(instances)}")
        return {
            "best_paths": best_paths,
            "best_makespans": dict(best_eval_makespan),
            "best_epochs": dict(best_eval_epoch),
            "training_log_path": training_log_path,
            "run_id": run_id,
        }

    def save(self, path: str):
        """Save model state."""
        raise NotImplementedError(f"{self.algorithm_name} save() is not implemented.")

    def load(self, path: str):
        """Load model state."""
        raise NotImplementedError(f"{self.algorithm_name} load() is not implemented.")
