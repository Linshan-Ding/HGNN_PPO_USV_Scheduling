"""
Heuristic Scheduling Rules for Baseline Comparison.

This module implements several scheduling heuristics:
1. MinBattery + NearestTask: Prioritize low-battery USVs, nearest tasks
2. MaxBattery + NearestTask: Prioritize high-battery USVs, nearest tasks
3. NearestOrigin + NearestTask: Prioritize USVs near origin
4. FarthestOrigin + NearestTask: Prioritize USVs far from origin
5. Random: Random selection

These rules serve as baselines for comparing PPO performance.

Note: Charging is handled automatically by the environment when USVs
cannot execute any remaining task.
"""

import os
import numpy as np
from typing import List, Tuple, Optional

from env import USVSchedulingEnv
from config import get_config
from utils import load_instance_from_config, plot_gantt_chart


class SchedulingRule:
    """Base class for scheduling rules."""
    
    def __init__(self, name: str):
        self.name = name
    
    def select_action(self, env: USVSchedulingEnv) -> Tuple[Optional[int], Optional[int]]:
        """
        Select action based on rule.
        
        Returns:
            (task_id, usv_id) or (None, None) if no valid action
        """
        raise NotImplementedError


class MinBatteryNearestTask(SchedulingRule):
    """Rule 1: Select lowest battery USV, assign nearest task."""
    
    def __init__(self):
        super().__init__("MinBattery_NearestTask")
    
    def select_action(self, env: USVSchedulingEnv) -> Tuple[Optional[int], Optional[int]]:
        available_tasks = env.get_available_tasks()
        if not available_tasks:
            return None, None
        
        idle_usvs = env.get_idle_usvs()
        if not idle_usvs:
            return None, None
        
        # Sort USVs by battery (ascending)
        sorted_usvs = sorted(idle_usvs, key=lambda u: env.usv_states[u, 2])
        
        for usv_id in sorted_usvs:
            usv_tasks = [t for t in available_tasks if env._can_usv_do_task(usv_id, t)]
            if usv_tasks:
                task_id = self._select_nearest_task(env, usv_tasks, usv_id)
                return task_id, usv_id
        
        return None, None
    
    def _select_nearest_task(self, env: USVSchedulingEnv, 
                             tasks: List[int], usv_id: int) -> int:
        """Select task nearest to USV."""
        usv_pos = env.usv_states[usv_id, 0:2]
        min_dist = float('inf')
        best_task = tasks[0]
        
        for task_id in tasks:
            task_pos = env.task_coords[task_id]
            dist = np.linalg.norm(task_pos - usv_pos)
            if dist < min_dist:
                min_dist = dist
                best_task = task_id
        
        return best_task


class MaxBatteryNearestTask(SchedulingRule):
    """Rule 2: Select highest battery USV, assign nearest task."""
    
    def __init__(self):
        super().__init__("MaxBattery_NearestTask")
    
    def select_action(self, env: USVSchedulingEnv) -> Tuple[Optional[int], Optional[int]]:
        available_tasks = env.get_available_tasks()
        if not available_tasks:
            return None, None
        
        idle_usvs = env.get_idle_usvs()
        if not idle_usvs:
            return None, None
        
        # Sort USVs by battery (descending)
        sorted_usvs = sorted(idle_usvs, key=lambda u: env.usv_states[u, 2], reverse=True)
        
        for usv_id in sorted_usvs:
            usv_tasks = [t for t in available_tasks if env._can_usv_do_task(usv_id, t)]
            if usv_tasks:
                task_id = self._select_nearest_task(env, usv_tasks, usv_id)
                return task_id, usv_id
        
        return None, None
    
    def _select_nearest_task(self, env: USVSchedulingEnv,
                             tasks: List[int], usv_id: int) -> int:
        usv_pos = env.usv_states[usv_id, 0:2]
        min_dist = float('inf')
        best_task = tasks[0]
        
        for task_id in tasks:
            task_pos = env.task_coords[task_id]
            dist = np.linalg.norm(task_pos - usv_pos)
            if dist < min_dist:
                min_dist = dist
                best_task = task_id
        
        return best_task


class NearestOriginNearestTask(SchedulingRule):
    """Rule 3: Select USV nearest to origin, assign nearest task."""
    
    def __init__(self):
        super().__init__("NearestOrigin_NearestTask")
    
    def select_action(self, env: USVSchedulingEnv) -> Tuple[Optional[int], Optional[int]]:
        available_tasks = env.get_available_tasks()
        if not available_tasks:
            return None, None
        
        idle_usvs = env.get_idle_usvs()
        if not idle_usvs:
            return None, None
        
        # Sort USVs by distance to origin (ascending)
        sorted_usvs = sorted(idle_usvs, key=lambda u: np.linalg.norm(env.usv_states[u, 0:2]))
        
        for usv_id in sorted_usvs:
            usv_tasks = [t for t in available_tasks if env._can_usv_do_task(usv_id, t)]
            if usv_tasks:
                task_id = self._select_nearest_task(env, usv_tasks, usv_id)
                return task_id, usv_id
        
        return None, None
    
    def _select_nearest_task(self, env: USVSchedulingEnv,
                             tasks: List[int], usv_id: int) -> int:
        usv_pos = env.usv_states[usv_id, 0:2]
        min_dist = float('inf')
        best_task = tasks[0]
        
        for task_id in tasks:
            task_pos = env.task_coords[task_id]
            dist = np.linalg.norm(task_pos - usv_pos)
            if dist < min_dist:
                min_dist = dist
                best_task = task_id
        
        return best_task


class FarthestOriginNearestTask(SchedulingRule):
    """Rule 4: Select USV farthest from origin, assign nearest task."""
    
    def __init__(self):
        super().__init__("FarthestOrigin_NearestTask")
    
    def select_action(self, env: USVSchedulingEnv) -> Tuple[Optional[int], Optional[int]]:
        available_tasks = env.get_available_tasks()
        if not available_tasks:
            return None, None
        
        idle_usvs = env.get_idle_usvs()
        if not idle_usvs:
            return None, None
        
        # Sort USVs by distance to origin (descending)
        sorted_usvs = sorted(
            idle_usvs,
            key=lambda u: np.linalg.norm(env.usv_states[u, 0:2]),
            reverse=True
        )
        
        for usv_id in sorted_usvs:
            usv_tasks = [t for t in available_tasks if env._can_usv_do_task(usv_id, t)]
            if usv_tasks:
                task_id = self._select_nearest_task(env, usv_tasks, usv_id)
                return task_id, usv_id
        
        return None, None
    
    def _select_nearest_task(self, env: USVSchedulingEnv,
                             tasks: List[int], usv_id: int) -> int:
        usv_pos = env.usv_states[usv_id, 0:2]
        min_dist = float('inf')
        best_task = tasks[0]
        
        for task_id in tasks:
            task_pos = env.task_coords[task_id]
            dist = np.linalg.norm(task_pos - usv_pos)
            if dist < min_dist:
                min_dist = dist
                best_task = task_id
        
        return best_task


class RandomRule(SchedulingRule):
    """Rule 5: Random task and USV selection."""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, env: USVSchedulingEnv) -> Tuple[Optional[int], Optional[int]]:
        available_tasks = env.get_available_tasks()
        if not available_tasks:
            return None, None
        
        idle_usvs = env.get_idle_usvs()
        if not idle_usvs:
            return None, None
        
        # Random shuffle
        shuffled_usvs = list(idle_usvs)
        np.random.shuffle(shuffled_usvs)
        
        for usv_id in shuffled_usvs:
            usv_tasks = [t for t in available_tasks if env._can_usv_do_task(usv_id, t)]
            if usv_tasks:
                task_id = np.random.choice(usv_tasks)
                return task_id, usv_id
        
        return None, None


def run_scheduling(env: USVSchedulingEnv, rule: SchedulingRule,
                   verbose: bool = False) -> dict:
    """
    Run scheduling with specified rule.
    
    The environment handles auto-charging automatically when USVs
    cannot execute any remaining task.
    
    Args:
        env: Environment instance
        rule: Scheduling rule to use
        verbose: Print step details
        
    Returns:
        Dict with makespan, steps, and success status
    """
    state = env.reset()
    done = False
    step = 0
    max_steps = env.n_tasks * 10
    info = {}
    
    while not done and step < max_steps:
        task_id, usv_id = rule.select_action(env)
        
        if task_id is None:
            # No valid action - check if all tasks done
            if env.n_scheduled_tasks >= env.n_tasks:
                break
            # Environment should have prepared valid actions
            # If not, we're in a deadlock
            break
        
        step += 1
        
        if verbose:
            print(f"  Step {step:2d}: Task {task_id:2d} -> USV {usv_id}")
        
        state, reward, done, info = env.step(task_id, usv_id)
    
    success = env.n_scheduled_tasks == env.n_tasks
    makespan = info.get('makespan', float('inf')) if success else float('inf')
    
    return {'makespan': makespan, 'steps': step, 'success': success}


def evaluate_rules(instance: dict, rules: List[SchedulingRule],
                   n_runs: int = 1, verbose: bool = False) -> dict:
    """
    Evaluate multiple scheduling rules.
    
    Args:
        instance: Problem instance
        rules: List of rules to evaluate
        n_runs: Kept for backward compatibility; all rules now run once.
        verbose: Print details
        
    Returns:
        Dict mapping rule names to statistics
    """
    results = {}
    
    for rule in rules:
        makespans = []
        actual_runs = 1
        
        for run in range(actual_runs):
            env = USVSchedulingEnv(instance)
            result = run_scheduling(env, rule, verbose=(verbose and run == 0))
            
            if result['success']:
                makespans.append(result['makespan'])
            else:
                makespans.append(float('inf'))
        
        if makespans:
            results[rule.name] = {
                'mean': np.mean(makespans),
                'std': np.std(makespans) if len(makespans) > 1 else 0.0,
                'min': np.min(makespans),
                'max': np.max(makespans),
                'runs': actual_runs
            }
    
    return results


def get_all_rules() -> List[SchedulingRule]:
    """Get list of all scheduling rules."""
    return [
        MinBatteryNearestTask(),
        MaxBatteryNearestTask(),
        NearestOriginNearestTask(),
        FarthestOriginNearestTask(),
        RandomRule()
    ]


def print_results(results: dict):
    """Print evaluation results in formatted table."""
    print("\n" + "=" * 70)
    print("Scheduling Rule Evaluation Results")
    print("=" * 70)
    print(f"{'Rule Name':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for name, stats in sorted_results:
        mean_str = f"{stats['mean']:.2f}" if stats['mean'] < float('inf') else "inf"
        std_str = f"{stats['std']:.2f}" if stats['std'] < float('inf') else "-"
        min_str = f"{stats['min']:.2f}" if stats['min'] < float('inf') else "inf"
        max_str = f"{stats['max']:.2f}" if stats['max'] < float('inf') else "inf"
        print(f"{name:<30} {mean_str:>10} {std_str:>10} {min_str:>10} {max_str:>10}")
    
    print("=" * 70)
    
    best_rule = sorted_results[0][0]
    best_makespan = sorted_results[0][1]['mean']
    print(f"Best Rule: {best_rule} (Makespan: {best_makespan:.2f})")


def main():
    """Main function for baseline evaluation."""
    cfg = get_config(
        n_usvs=2,
        n_tasks=20,
        data_dir='data/public',
    )
    
    # Load selected public CSV instance
    instance = load_instance_from_config(cfg)
    print(f"[Instance] ID={instance.get('instance_id', 'N/A')}")
    print(f"[Instance] USVs={instance['n_usvs']}, Tasks={instance['n_tasks']}")
    print(f"[Instance] Seed={instance.get('seed', 'N/A')}")
    
    # Evaluate rules
    rules = get_all_rules()
    print(f"\n[Rules] {len(rules)} scheduling rules:")
    for i, rule in enumerate(rules, 1):
        print(f"  {i}. {rule.name}")
    
    print("\n[Evaluating]...")
    results = evaluate_rules(instance, rules, n_runs=1)
    print_results(results)
    
    # Demo best rule
    best_rule_name = min(results.keys(), key=lambda k: results[k]['mean'])
    best_rule = next(r for r in rules if r.name == best_rule_name)
    
    print(f"\n[Demo] {best_rule.name} scheduling:")
    env = USVSchedulingEnv(instance)
    result = run_scheduling(env, best_rule, verbose=True)
    print(f"[Result] Makespan: {result['makespan']:.2f}")
    
    # Save Gantt chart
    os.makedirs(cfg.result_dir, exist_ok=True)
    plot_gantt_chart(env, os.path.join(cfg.result_dir, f'gantt_{best_rule.name}.png'))
    
    return results


if __name__ == "__main__":
    main()
