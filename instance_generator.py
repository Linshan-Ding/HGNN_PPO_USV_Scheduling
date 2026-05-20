"""
Instance Generator for USV Scheduling Problem.

This module generates random problem instances with:
- Random task locations within map boundaries
- Triangular fuzzy processing times
- Automatic battery capacity adjustment to ensure feasibility
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

from config import InstanceConfig, DataConfig


DEFAULT_BATTERY_POLICY = "max_single_trip_energy"
DEFAULT_BATTERY_SAFETY_FACTOR = 1.20


class InstanceGenerator:
    """Generator for USV scheduling problem instances."""
    
    def __init__(
        self,
        config: InstanceConfig,
        battery_safety_factor: float = DEFAULT_BATTERY_SAFETY_FACTOR,
    ):
        """
        Initialize generator with configuration.
        
        Args:
            config: Instance configuration parameters
            battery_safety_factor: Multiplier for maximum single-task round-trip energy
        """
        self.config = config
        self.battery_safety_factor = battery_safety_factor

    @staticmethod
    def expected_durations(fuzzy_times: np.ndarray) -> np.ndarray:
        """Calculate expected durations from triangular fuzzy processing times."""
        return (
            fuzzy_times[:, 0] +
            2 * fuzzy_times[:, 1] +
            fuzzy_times[:, 2]
        ) / 4.0

    @staticmethod
    def single_trip_energies(
        task_coords: np.ndarray,
        fuzzy_times: np.ndarray,
        config: InstanceConfig,
    ) -> np.ndarray:
        """
        Compute per-task energy for origin -> task -> origin with service.

        This is the feasibility baseline for battery sizing. If the battery is
        at least the maximum value here, every individual task can be completed
        from the charging station and safely returned.
        """
        task_durations = InstanceGenerator.expected_durations(fuzzy_times)
        distances = np.linalg.norm(task_coords, axis=1)
        return (
            2 * distances * config.energy_cost_per_distance +
            task_durations * config.energy_cost_per_task_time
        )

    @staticmethod
    def required_battery_capacity(
        task_coords: np.ndarray,
        fuzzy_times: np.ndarray,
        config: InstanceConfig,
        safety_factor: float = DEFAULT_BATTERY_SAFETY_FACTOR,
    ) -> Tuple[float, float, int]:
        """
        Calculate battery capacity from the maximum single-task round-trip energy.

        Returns:
            battery_capacity, max_single_trip_energy, farthest_task_id
        """
        single_trip_energy = InstanceGenerator.single_trip_energies(
            task_coords, fuzzy_times, config
        )
        max_single_trip_energy = float(np.max(single_trip_energy))
        distances = np.linalg.norm(task_coords, axis=1)
        farthest_task_id = int(np.argmax(distances))
        return (
            safety_factor * max_single_trip_energy,
            max_single_trip_energy,
            farthest_task_id,
        )
    
    def generate(self, seed: int = None) -> dict:
        """
        Generate a random problem instance.
        
        Ensures battery capacity is sufficient for every single task:
        origin -> task -> origin plus service energy.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Instance dictionary with task coordinates, fuzzy times, and config
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate task coordinates (uniform distribution)
        task_coords = np.random.uniform(
            0, self.config.map_size[0],
            (self.config.n_tasks, 2)
        )
        
        # Generate triangular fuzzy processing times (t1 < t2 < t3)
        t2 = np.random.uniform(5, 20, self.config.n_tasks)
        t1 = t2 * np.random.uniform(0.7, 0.9, self.config.n_tasks)
        t3 = t2 * np.random.uniform(1.1, 1.3, self.config.n_tasks)
        fuzzy_times = np.stack([t1, t2, t3], axis=1)
        
        actual_battery, max_single_trip_energy, farthest_task_id = (
            self.required_battery_capacity(
                task_coords,
                fuzzy_times,
                self.config,
                self.battery_safety_factor,
            )
        )
        
        # Create updated config
        config_copy = InstanceConfig(
            n_usvs=self.config.n_usvs,
            n_tasks=self.config.n_tasks,
            map_size=self.config.map_size,
            battery_capacity=actual_battery,
            usv_speed=self.config.usv_speed,
            charge_time=self.config.charge_time,
            energy_cost_per_distance=self.config.energy_cost_per_distance,
            energy_cost_per_task_time=self.config.energy_cost_per_task_time
        )
        instance_id = f"u{self.config.n_usvs}_t{self.config.n_tasks}_{seed}"
        
        return {
            'instance_id': instance_id,
            'n_usvs': self.config.n_usvs,
            'n_tasks': self.config.n_tasks,
            'task_coords': task_coords,
            'fuzzy_times': fuzzy_times,
            'config': config_copy,
            'seed': seed,
            'battery_policy': DEFAULT_BATTERY_POLICY,
            'battery_safety_factor': self.battery_safety_factor,
            'max_single_trip_energy': max_single_trip_energy,
            'farthest_task_id': farthest_task_id
        }
    
    def save_to_csv(self, instance: dict, filepath: str):
        """
        Save instance to CSV file with metadata header.
        
        Args:
            instance: Instance dictionary
            filepath: Output file path
        """
        df = pd.DataFrame({
            'task_id': range(instance['n_tasks']),
            'x': instance['task_coords'][:, 0],
            'y': instance['task_coords'][:, 1],
            't1': instance['fuzzy_times'][:, 0],
            't2': instance['fuzzy_times'][:, 1],
            't3': instance['fuzzy_times'][:, 2]
        })
        
        cfg = instance['config']
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Write metadata header
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            f.write(
                f"# instance_id={instance.get('instance_id', '')},"
                f"n_usvs={instance['n_usvs']},"
                f"n_tasks={instance['n_tasks']},"
                f"seed={instance['seed']}\n"
            )
            f.write(
                f"# battery_capacity={cfg.battery_capacity:.10f},"
                f"battery_policy={instance.get('battery_policy', DEFAULT_BATTERY_POLICY)},"
                f"battery_safety_factor={instance.get('battery_safety_factor', DEFAULT_BATTERY_SAFETY_FACTOR):.2f},"
                f"max_single_trip_energy={instance.get('max_single_trip_energy', 0.0):.10f},"
                f"farthest_task_id={instance.get('farthest_task_id', -1)},"
                f"map_width={cfg.map_size[0]},"
                f"map_height={cfg.map_size[1]},"
                f"speed={cfg.usv_speed},"
                f"charge_time={cfg.charge_time},"
                f"energy_cost_per_distance={cfg.energy_cost_per_distance},"
                f"energy_cost_per_task_time={cfg.energy_cost_per_task_time}\n"
            )
        
        df.to_csv(filepath, mode='a', index=False)
    
    @staticmethod
    def load_from_csv(filepath: str, config: InstanceConfig = None) -> dict:
        """
        Load instance from CSV file.
        
        Args:
            filepath: Input file path
            config: Optional config override
            
        Returns:
            Instance dictionary
        """
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            line1 = f.readline().strip('# \n')
            line2 = f.readline().strip('# \n')
        
        def _parse_meta(line: str) -> dict:
            metadata = {}
            for item in line.split(','):
                if '=' not in item:
                    continue
                key, value = item.split('=', 1)
                metadata[key.strip()] = value.strip()
            return metadata
        
        meta1 = _parse_meta(line1)
        meta2 = _parse_meta(line2)
        
        df = pd.read_csv(filepath, comment='#', encoding='utf-8-sig')
        
        # Build config from metadata
        if config is None:
            battery_capacity = float(
                meta2.get('battery_capacity', meta2.get('battery', 3000.0))
            )
            map_width = int(float(meta2.get('map_width', 1000)))
            map_height = int(float(meta2.get('map_height', map_width)))
            energy_cost_per_task_time = float(
                meta2.get('energy_cost_per_task_time', meta2.get('task_energy', 5.0))
            )
            config = InstanceConfig(
                n_usvs=int(meta1['n_usvs']),
                n_tasks=int(meta1['n_tasks']),
                map_size=(map_width, map_height),
                battery_capacity=battery_capacity,
                usv_speed=float(meta2['speed']),
                charge_time=float(meta2['charge_time']),
                energy_cost_per_distance=float(meta2.get('energy_cost_per_distance', 1.0)),
                energy_cost_per_task_time=energy_cost_per_task_time
            )
        
        single_trip_energy = InstanceGenerator.single_trip_energies(
            df[['x', 'y']].values,
            df[['t1', 't2', 't3']].values,
            config,
        )
        max_single_trip_energy = float(np.max(single_trip_energy))
        farthest_task_id = int(np.argmax(np.linalg.norm(df[['x', 'y']].values, axis=1)))
        seed_value = meta1.get('seed')
        seed = int(seed_value) if seed_value not in (None, 'None', '') else None
        
        return {
            'instance_id': meta1.get('instance_id', os.path.splitext(os.path.basename(filepath))[0]),
            'n_usvs': int(meta1['n_usvs']),
            'n_tasks': int(meta1['n_tasks']),
            'task_coords': df[['x', 'y']].values,
            'fuzzy_times': df[['t1', 't2', 't3']].values,
            'config': config,
            'seed': seed,
            'battery_policy': meta2.get('battery_policy', 'legacy'),
            'battery_safety_factor': float(meta2.get('battery_safety_factor', 0.0)),
            'max_single_trip_energy': float(
                meta2.get('max_single_trip_energy', max_single_trip_energy)
            ),
            'farthest_task_id': int(meta2.get('farthest_task_id', farthest_task_id))
        }

    @staticmethod
    def validate_battery_capacity(instance: dict, atol: float = 1e-6) -> dict:
        """
        Validate that battery capacity can complete every single task round trip.

        Raises:
            ValueError: If the instance violates the battery sizing rule.
        """
        config = instance['config']
        single_trip_energy = InstanceGenerator.single_trip_energies(
            instance['task_coords'], instance['fuzzy_times'], config
        )
        max_energy = float(np.max(single_trip_energy))
        farthest_task_id = int(np.argmax(np.linalg.norm(instance['task_coords'], axis=1)))
        farthest_energy = float(single_trip_energy[farthest_task_id])
        battery_capacity = float(config.battery_capacity)
        
        if max_energy - battery_capacity > atol:
            raise ValueError(
                f"Instance {instance.get('instance_id', '<unknown>')} has "
                f"battery_capacity={battery_capacity:.6f}, but max single-trip "
                f"energy={max_energy:.6f}"
            )
        if farthest_energy - battery_capacity > atol:
            raise ValueError(
                f"Instance {instance.get('instance_id', '<unknown>')} cannot "
                f"complete farthest task {farthest_task_id}: "
                f"required={farthest_energy:.6f}, battery={battery_capacity:.6f}"
            )
        
        return {
            'max_single_trip_energy': max_energy,
            'farthest_task_id': farthest_task_id,
            'farthest_task_energy': farthest_energy,
            'battery_capacity': battery_capacity,
        }
    
    def save_instances(self, num_instances: int, save_dir: str):
        """
        Generate and save multiple instances.
        
        Args:
            num_instances: Number of instances to generate
            save_dir: Output directory
        """
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"u{self.config.n_usvs}_t{self.config.n_tasks}"
        print(f"Generating {num_instances} instances: {prefix}")
        
        for i in range(num_instances):
            instance = self.generate(seed=1000 + i)
            filepath = os.path.join(save_dir, f"{prefix}_{i}.csv")
            self.save_to_csv(instance, filepath)
        
        print(f"Saved to {save_dir}")


def generate_benchmark_datasets(data_cfg: DataConfig = None, 
                                instance_cfg: InstanceConfig = None):
    """
    Generate standard benchmark datasets.
    
    Creates instances for common problem sizes:
    - 4 USVs, 20 tasks
    - 4 USVs, 40 tasks
    - 6 USVs, 60 tasks
    - 8 USVs, 80 tasks
    
    Args:
        data_cfg: Data configuration
        instance_cfg: Instance configuration template
    """
    if data_cfg is None:
        data_cfg = DataConfig()
    
    configs = [(4, 20), (4, 40), (6, 60), (8, 80)]
    
    for n_usvs, n_tasks in configs:
        base_cfg = instance_cfg or InstanceConfig()
        cfg = InstanceConfig(
            n_usvs=n_usvs,
            n_tasks=n_tasks,
            map_size=base_cfg.map_size,
            battery_capacity=base_cfg.battery_capacity,
            usv_speed=base_cfg.usv_speed,
            charge_time=base_cfg.charge_time,
            energy_cost_per_distance=base_cfg.energy_cost_per_distance,
            energy_cost_per_task_time=base_cfg.energy_cost_per_task_time,
        )
        
        generator = InstanceGenerator(cfg)
        generator.save_instances(1, save_dir=data_cfg.data_dir)


def generate_public_25_dataset(
    save_dir: str = os.path.join('data', 'public'),
    seed_start: int = 2026051900,
    battery_safety_factor: float = DEFAULT_BATTERY_SAFETY_FACTOR,
) -> pd.DataFrame:
    """
    Generate the fixed 25-instance public benchmark set.

    The set is the full Cartesian product of:
    n_usvs = [2, 4, 6, 8, 10]
    n_tasks = [20, 40, 60, 80, 100]
    """
    os.makedirs(save_dir, exist_ok=True)
    usv_counts = [2, 4, 6, 8, 10]
    task_counts = [20, 40, 60, 80, 100]
    manifest_rows = []
    idx = 0
    
    for n_usvs in usv_counts:
        for n_tasks in task_counts:
            seed = seed_start + idx
            cfg = InstanceConfig(n_usvs=n_usvs, n_tasks=n_tasks)
            generator = InstanceGenerator(
                cfg, battery_safety_factor=battery_safety_factor
            )
            instance = generator.generate(seed=seed)
            instance_id = f"u{n_usvs}_t{n_tasks}"
            instance['instance_id'] = instance_id
            filename = f"{instance_id}.csv"
            filepath = os.path.join(save_dir, filename)
            generator.save_to_csv(instance, filepath)
            validation = InstanceGenerator.validate_battery_capacity(instance)
            
            manifest_rows.append({
                'instance_id': instance_id,
                'filename': filename,
                'n_usvs': n_usvs,
                'n_tasks': n_tasks,
                'seed': seed,
                'battery_capacity': validation['battery_capacity'],
                'battery_policy': DEFAULT_BATTERY_POLICY,
                'battery_safety_factor': battery_safety_factor,
                'max_single_trip_energy': validation['max_single_trip_energy'],
                'farthest_task_id': validation['farthest_task_id'],
                'farthest_task_energy': validation['farthest_task_energy'],
            })
            idx += 1
    
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(save_dir, 'manifest.csv'), index=False)
    return manifest


if __name__ == "__main__":
    generate_public_25_dataset()
