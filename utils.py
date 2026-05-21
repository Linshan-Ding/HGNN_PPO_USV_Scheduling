"""
Utility Functions for USV Scheduling.

This module provides:
- InstanceLoader: Load problem instances from CSV files
- plot_gantt_chart: Visualize scheduling results
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional

from instance_generator import InstanceGenerator


class InstanceLoader:
    """Loader for USV scheduling problem instances."""
    
    def __init__(
        self,
        data_dir: str,
        n_usvs: Optional[int] = None,
        n_tasks: Optional[int] = None,
        instance_id: Optional[str] = None,
        instance_file: Optional[str] = None,
    ):
        """
        Initialize loader and find matching instance files.
        
        Args:
            data_dir: Directory containing CSV instance files
            n_usvs: Number of USVs to filter by when no explicit instance is given
            n_tasks: Number of tasks to filter by when no explicit instance is given
            instance_id: Public benchmark instance ID, e.g. u4_t40
            instance_file: Explicit CSV path or filename under data_dir
            
        Raises:
            ValueError: If no matching files found
        """
        self.data_dir = data_dir
        self.files = self._resolve_files(
            data_dir=data_dir,
            n_usvs=n_usvs,
            n_tasks=n_tasks,
            instance_id=instance_id,
            instance_file=instance_file,
        )
        
        if not self.files:
            raise ValueError(
                "No instance CSV files found. "
                f"data_dir={data_dir}, n_usvs={n_usvs}, n_tasks={n_tasks}, "
                f"instance_id={instance_id}, instance_file={instance_file}"
            )
        
        print(f"Loaded {len(self.files)} instance file(s):")
        for fpath in self.files:
            print(f"  - {fpath}")
    
    @staticmethod
    def _resolve_files(
        data_dir: str,
        n_usvs: Optional[int],
        n_tasks: Optional[int],
        instance_id: Optional[str],
        instance_file: Optional[str],
    ) -> List[str]:
        """Resolve CSV files from explicit file, instance ID, or size filters."""
        if instance_file:
            candidates = []
            if os.path.isabs(instance_file):
                candidates.append(instance_file)
            else:
                candidates.append(instance_file)
                candidates.append(os.path.join(data_dir, instance_file))
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return [candidate]
            raise ValueError(f"Instance file not found: {instance_file}")
        
        if instance_id:
            csv_name = instance_id if instance_id.endswith('.csv') else f"{instance_id}.csv"
            direct_path = os.path.join(data_dir, csv_name)
            if os.path.isfile(direct_path):
                return [direct_path]
            
            manifest_path = os.path.join(data_dir, 'manifest.csv')
            if os.path.isfile(manifest_path):
                try:
                    import pandas as pd
                    manifest = pd.read_csv(manifest_path)
                    matched = manifest[manifest['instance_id'] == instance_id]
                    if not matched.empty:
                        filename = matched.iloc[0]['filename']
                        fpath = os.path.join(data_dir, filename)
                        if os.path.isfile(fpath):
                            return [fpath]
                except Exception as exc:
                    raise ValueError(f"Failed reading manifest {manifest_path}: {exc}")
            
            recursive_pattern = os.path.join(data_dir, '**', csv_name)
            matches = sorted(glob.glob(recursive_pattern, recursive=True))
            if matches:
                return matches[:1]
            raise ValueError(f"Instance ID not found under {data_dir}: {instance_id}")
        
        if n_usvs is None or n_tasks is None:
            pattern = os.path.join(data_dir, '*.csv')
            files = [f for f in sorted(glob.glob(pattern)) if not f.endswith('manifest.csv')]
            return files
        
        patterns = [
            os.path.join(data_dir, f"u{n_usvs}_t{n_tasks}.csv"),
            os.path.join(data_dir, f"u{n_usvs:02d}_t{n_tasks:03d}_*.csv"),
            os.path.join(data_dir, f"u{n_usvs}_t{n_tasks}_*.csv"),
        ]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        return sorted(set(files))
    
    def get_instance(self, index: int = 0) -> dict:
        """
        Load a specific instance.
        
        Args:
            index: Instance index (default: 0)
            
        Returns:
            Instance dictionary
        """
        if index < 0 or index >= len(self.files):
            raise IndexError(f"Instance index {index} out of range for {len(self.files)} files")
        fpath = self.files[index]
        return InstanceGenerator.load_from_csv(fpath)


def load_instance_from_config(cfg) -> dict:
    """
    Load the instance selected by cfg.data and synchronize cfg.instance.

    Selection priority:
    1. cfg.data.instance_file
    2. cfg.data.instance_id
    3. cfg.instance.n_usvs + cfg.instance.n_tasks + cfg.data.instance_index
    """
    reward_normalization = getattr(cfg.instance, 'reward_normalization', True)
    loader = InstanceLoader(
        data_dir=cfg.data.data_dir,
        n_usvs=cfg.instance.n_usvs,
        n_tasks=cfg.instance.n_tasks,
        instance_id=cfg.data.instance_id,
        instance_file=cfg.data.instance_file,
    )
    instance = loader.get_instance(cfg.data.instance_index)
    cfg.instance = instance['config']
    cfg.instance.reward_normalization = reward_normalization
    instance['config'].reward_normalization = reward_normalization
    return instance


def plot_gantt_chart(env, save_path: str = './results/gantt.png'):
    """
    Generate and save Gantt chart for scheduling results.
    
    Displays:
    - Task execution blocks (colored by task ID)
    - Movement blocks (gray hatched)
    - Charging blocks (yellow)
    
    Args:
        env: Environment with completed schedule
        save_path: Output file path
    """
    print(f"Generating Gantt chart -> {save_path}")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap('tab20')
    colors = {'move': '#D3D3D3', 'charge': '#FFD700'}
    max_time = 0
    
    for usv_id in range(env.n_usvs):
        for event in env.usv_history[usv_id]:
            start, end = event['start'], event['end']
            duration = end - start
            max_time = max(max_time, end)
            
            if duration <= 0.01:
                continue
            
            etype = event['type']
            
            if etype == 'task':
                # Task block with task-specific color
                tid = int(event['info'].replace('T', '')) if 'T' in event['info'] else 0
                color = cmap(tid % 20)
                ax.barh(usv_id, duration, left=start, height=0.6,
                       color=color, edgecolor='k', alpha=0.9)
                ax.text(start + duration / 2, usv_id, event['info'],
                       ha='center', va='center', color='white',
                       fontsize=7, fontweight='bold')
            
            elif etype == 'move':
                # Movement block (gray hatched)
                ax.barh(usv_id, duration, left=start, height=0.4,
                       color=colors['move'], edgecolor='gray',
                       alpha=0.6, hatch='///')
            
            elif etype == 'charge':
                # Charging block (yellow)
                ax.barh(usv_id, duration, left=start, height=0.6,
                       color=colors['charge'], edgecolor='k')
                ax.text(start + duration / 2, usv_id, "⚡",
                       ha='center', va='center', fontsize=10)
    
    # Configure axes
    ax.set_yticks(range(env.n_usvs))
    ax.set_yticklabels([f'USV-{i}' for i in range(env.n_usvs)])
    ax.set_xlabel('Time (s)')
    ax.set_title('USV Schedule Gantt Chart')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.set_xlim(0, max_time * 1.05)
    
    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=cmap(0), edgecolor='k', label='Task'),
        mpatches.Patch(facecolor=colors['move'], edgecolor='gray',
                      hatch='///', alpha=0.6, label='Move'),
        mpatches.Patch(facecolor=colors['charge'], edgecolor='k', label='Charge')
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
