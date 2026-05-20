"""
USV Scheduling Environment.

This environment implements a multi-USV task scheduling problem where:
- USVs start at origin (0, 0) and must complete all tasks
- Each task has a location and fuzzy processing time
- USVs have limited battery and must return to origin when depleted
- The goal is to minimize makespan (total completion time)

Key Design:
- Actions: Select (task_id, usv_id) pairs
- Charging is handled automatically when a USV cannot execute any task
- Environment ensures valid actions are always available (or episode ends)
"""

import numpy as np
from typing import Dict, List, Tuple


class USVSchedulingEnv:
    """
    Multi-USV task scheduling environment.
    
    State Space:
        - USV features: position, battery, busy status, etc.
        - Task features: position, duration, scheduling status, etc.
        - Edge features: distance, travel time, energy cost between USV-task pairs
    
    Action Space:
        - (task_id, usv_id): Assign task to USV
        - Only unscheduled tasks and idle USVs with sufficient energy are valid
    
    Reward:
        - Negative change in estimated makespan (encourage reducing makespan)
    """
    
    def __init__(self, instance: Dict):
        """
        Initialize environment from problem instance.
        
        Args:
            instance: Dict containing problem data:
                - n_usvs: Number of USVs
                - n_tasks: Number of tasks  
                - task_coords: Task locations [n_tasks, 2]
                - fuzzy_times: Fuzzy processing times [n_tasks, 3] (t1, t2, t3)
                - config: InstanceConfig object
        """
        self.instance = instance
        self.config = instance['config']
        self.n_usvs = instance['n_usvs']
        self.n_tasks = instance['n_tasks']
        
        self.task_coords = instance['task_coords']
        self.fuzzy_times = instance['fuzzy_times']
        
        # Calculate expected task durations using triangular fuzzy numbers
        # E[T] = (t1 + 2*t2 + t3) / 4
        self.task_durations = (
            self.fuzzy_times[:, 0] + 
            2 * self.fuzzy_times[:, 1] + 
            self.fuzzy_times[:, 2]
        ) / 4.0
        
        self._precompute_constants()
        self.reset()
    
    def _precompute_constants(self):
        """Precompute task-related constants for efficiency."""
        # Distance from each task to origin
        self.task_dist_to_origin = np.linalg.norm(self.task_coords, axis=1)
        
        # Base energy for round trip: go + execute + return
        self.task_base_energy = (
            self.task_dist_to_origin * self.config.energy_cost_per_distance +
            self.task_durations * self.config.energy_cost_per_task_time +
            self.task_dist_to_origin * self.config.energy_cost_per_distance
        )
        
        # Normalization constants
        self.map_width = max(float(self.config.map_size[0]), 1.0)
        self.map_height = max(float(self.config.map_size[1]), 1.0)
        self.max_dist = max(np.sqrt(self.map_width ** 2 + self.map_height ** 2), 1.0)
        self.max_duration = np.max(self.task_durations) if len(self.task_durations) > 0 else 1.0
        self.max_duration = max(float(self.max_duration), 1.0)
        self.max_energy = max(float(self.config.battery_capacity), 1.0)
        avg_service_load = np.sum(self.task_durations) / max(self.n_usvs, 1)
        avg_route_load = (
            2.0 * np.mean(self.task_dist_to_origin) / self.config.usv_speed *
            max(self.n_tasks / max(self.n_usvs, 1), 1.0)
        )
        self.scale_time = max(float(avg_service_load + avg_route_load), 1.0)
        self.avg_tasks_per_usv = max(self.n_tasks / max(self.n_usvs, 1), 1.0)

    def reset(self) -> Dict:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state dictionary with guaranteed available actions
        """
        # USV states: [x, y, battery, busy_until]
        self.usv_states = np.zeros((self.n_usvs, 4))
        self.usv_states[:, 2] = self.config.battery_capacity  # Full battery
        
        # Task states: [is_scheduled, start_time, end_time, assigned_usv]
        self.task_states = np.zeros((self.n_tasks, 4))
        self.task_states[:, 3] = -1  # No USV assigned
        
        # Tracking variables
        self.usv_task_counts = np.zeros(self.n_usvs)
        self.current_time = 0.0
        self.n_scheduled_tasks = 0
        self.usv_history = {i: [] for i in range(self.n_usvs)}
        self.last_makespan = 0.0
        
        return self._get_state()

    def _get_state(self) -> Dict:
        """Get current state as feature dictionary."""
        return {
            'usv_features': self._get_usv_features(),
            'task_features': self._get_task_features(),
            'edge_features': self._get_edge_features(),
            'n_scheduled': self.n_scheduled_tasks,
            'current_time': self.current_time
        }

    def _get_usv_features(self) -> np.ndarray:
        """
        Extract USV features (7 dimensions per USV):
        [x, y, battery, remaining_busy_time, dist_to_origin, task_count, is_idle]
        """
        features = np.zeros((self.n_usvs, 7))
        
        for i in range(self.n_usvs):
            pos = self.usv_states[i, 0:2]
            battery = self.usv_states[i, 2]
            busy_until = self.usv_states[i, 3]
            
            features[i, 0] = pos[0] / self.map_width                   # x position
            features[i, 1] = pos[1] / self.map_height                  # y position
            features[i, 2] = battery / self.max_energy                 # current battery
            features[i, 3] = max(busy_until - self.current_time, 0.0) / self.scale_time
            features[i, 4] = np.linalg.norm(pos) / self.max_dist       # distance to origin
            features[i, 5] = self.usv_task_counts[i] / self.avg_tasks_per_usv
            features[i, 6] = 1.0 if busy_until <= self.current_time else 0.0  # idle flag
        
        return features

    def _get_task_features(self) -> np.ndarray:
        """
        Extract task features (8 dimensions per task):
        [x, y, duration, dist_to_origin, energy_demand, is_scheduled, feasible_usv_count, progress]
        """
        features = np.zeros((self.n_tasks, 8))
        
        for i in range(self.n_tasks):
            features[i, 0] = self.task_coords[i, 0] / self.map_width
            features[i, 1] = self.task_coords[i, 1] / self.map_height
            features[i, 2] = self.task_durations[i] / self.max_duration
            features[i, 3] = self.task_dist_to_origin[i] / self.max_dist
            features[i, 4] = self.task_base_energy[i] / self.max_energy
            features[i, 5] = self.task_states[i, 0]          # is scheduled (0/1)
            
            # Count USVs that can execute this task
            feasible_count = len(self.get_available_usvs_for_task(i))
            features[i, 6] = feasible_count / max(self.n_usvs, 1)
            
            # Scheduling progress
            features[i, 7] = self.n_scheduled_tasks / max(self.n_tasks, 1)
        
        return features

    def _get_edge_features(self) -> np.ndarray:
        """
        Extract edge features for USV-task pairs (4 dimensions):
        [distance, travel_time, energy_cost, remaining_battery_after]
        """
        edge_features = np.zeros((self.n_usvs, self.n_tasks, 4))
        
        for u in range(self.n_usvs):
            usv_pos = self.usv_states[u, 0:2]
            usv_battery = self.usv_states[u, 2]
            
            for t in range(self.n_tasks):
                task_pos = self.task_coords[t]
                dist = np.linalg.norm(task_pos - usv_pos)
                
                travel_energy = dist * self.config.energy_cost_per_distance
                task_energy = self.task_durations[t] * self.config.energy_cost_per_task_time
                return_energy = self.task_dist_to_origin[t] * self.config.energy_cost_per_distance
                total_energy = travel_energy + task_energy + return_energy
                
                edge_features[u, t, 0] = dist / self.max_dist
                edge_features[u, t, 1] = (dist / self.config.usv_speed) / self.scale_time
                edge_features[u, t, 2] = total_energy / self.max_energy
                edge_features[u, t, 3] = (usv_battery - total_energy) / self.max_energy
        
        return edge_features

    def _compute_energy_for_task(self, usv_id: int, task_id: int) -> float:
        """
        Calculate total energy needed for USV to complete task and return to origin.
        
        Energy = travel_to_task + execute_task + return_to_origin
        """
        usv_pos = self.usv_states[usv_id, 0:2]
        task_pos = self.task_coords[task_id]
        
        dist_to_task = np.linalg.norm(task_pos - usv_pos)
        travel_energy = dist_to_task * self.config.energy_cost_per_distance
        task_energy = self.task_durations[task_id] * self.config.energy_cost_per_task_time
        return_energy = self.task_dist_to_origin[task_id] * self.config.energy_cost_per_distance
        
        return travel_energy + task_energy + return_energy

    def _is_usv_idle(self, usv_id: int) -> bool:
        """Check if USV is currently idle (not busy)."""
        return self.usv_states[usv_id, 3] <= self.current_time

    def _can_usv_do_task(self, usv_id: int, task_id: int) -> bool:
        """Check if USV can execute a specific task."""
        # USV must be idle
        if not self._is_usv_idle(usv_id):
            return False
        
        # Task must not be scheduled
        if self.task_states[task_id, 0] == 1:
            return False
        
        # USV must have sufficient energy
        energy_needed = self._compute_energy_for_task(usv_id, task_id)
        return self.usv_states[usv_id, 2] >= energy_needed

    def _usv_needs_charging(self, usv_id: int) -> bool:
        """
        Check if USV needs to return for charging.
        Returns True if USV is idle and cannot execute any remaining task.
        """
        # If USV is not idle, it doesn't need charging decision now
        if not self._is_usv_idle(usv_id):
            return False
            
        for task_id in range(self.n_tasks):
            if self.task_states[task_id, 0] == 0:  # Unscheduled task
                if self._can_usv_do_task(usv_id, task_id):
                    return False
        return True

    def _can_usv_return_to_origin(self, usv_id: int) -> bool:
        """Check if USV has enough energy to return to origin."""
        usv_pos = self.usv_states[usv_id, 0:2]
        dist_to_origin = np.linalg.norm(usv_pos)
        return_energy = dist_to_origin * self.config.energy_cost_per_distance
        return self.usv_states[usv_id, 2] >= return_energy

    def _execute_charging(self, usv_id: int):
        """
        Execute automatic charging: return to origin and recharge.
        Called when USV cannot execute any remaining task.
        """
        usv_pos = self.usv_states[usv_id, 0:2]
        dist_to_origin = np.linalg.norm(usv_pos)
        
        # Return to origin
        if dist_to_origin > 1e-3:
            return_time = dist_to_origin / self.config.usv_speed
            return_energy = dist_to_origin * self.config.energy_cost_per_distance
            return_start = max(self.current_time, self.usv_states[usv_id, 3])
            return_end = return_start + return_time
            
            self.usv_history[usv_id].append({
                'type': 'move',
                'start': return_start,
                'end': return_end,
                'info': 'Return(Charge)'
            })
            
            self.usv_states[usv_id, 2] -= return_energy
        else:
            return_end = max(self.current_time, self.usv_states[usv_id, 3])
        
        # Charge
        charge_end = return_end + self.config.charge_time
        self.usv_history[usv_id].append({
            'type': 'charge',
            'start': return_end,
            'end': charge_end,
            'info': 'Charge'
        })
        
        # Update USV state
        self.usv_states[usv_id, 0:2] = [0, 0]
        self.usv_states[usv_id, 2] = self.config.battery_capacity
        self.usv_states[usv_id, 3] = charge_end

    def _handle_auto_charging(self):
        """
        Automatically send idle USVs for charging if they cannot execute any remaining task.
        """
        for usv_id in range(self.n_usvs):
            if self._usv_needs_charging(usv_id) and self._can_usv_return_to_origin(usv_id):
                self._execute_charging(usv_id)

    def _advance_time_once(self) -> bool:
        """
        Advance time to next USV availability if no USV is currently idle.
        
        Returns:
            True if time was advanced, False otherwise
        """
        idle_usvs = [u for u in range(self.n_usvs) if self._is_usv_idle(u)]
        
        if len(idle_usvs) == 0 and self.n_scheduled_tasks < self.n_tasks:
            self.current_time = np.min(self.usv_states[:, 3])
            return True
        return False

    def ensure_ready_state(self) -> bool:
        """
        Ensure environment is in a state where valid actions exist.
        
        This method handles:
        1. Auto-charging for USVs that cannot execute any task
        2. Time advancement when all USVs are busy
        3. Iterates until valid actions exist or episode should end
        
        Returns:
            True if valid actions exist, False if episode should end
        """
        max_iterations = self.n_usvs * self.n_tasks + 100  # Safety limit
        
        for _ in range(max_iterations):
            # Check if all tasks are scheduled
            if self.n_scheduled_tasks >= self.n_tasks:
                return False
            
            # Handle auto-charging for idle USVs that need it
            self._handle_auto_charging()
            
            # Check if we have valid actions now
            available_tasks = self.get_available_tasks()
            if len(available_tasks) > 0:
                return True
            
            # No valid actions - check if all USVs are idle
            idle_usvs = [u for u in range(self.n_usvs) if self._is_usv_idle(u)]
            busy_usvs = [u for u in range(self.n_usvs) if not self._is_usv_idle(u)]
            
            if len(busy_usvs) == 0:
                # All USVs are idle but none can do any task - deadlock
                # This should not happen if instance is feasible
                return False
            
            # Advance time to next busy USV completion
            self.current_time = np.min(self.usv_states[:, 3])
        
        # Safety: should not reach here in normal operation
        return False

    def get_available_tasks(self) -> List[int]:
        """
        Get list of tasks that can be executed by at least one idle USV.
        """
        available = []
        
        for task_id in range(self.n_tasks):
            if self.task_states[task_id, 0] == 1:  # Already scheduled
                continue
            
            # Check if any idle USV can execute this task
            for usv_id in range(self.n_usvs):
                if self._can_usv_do_task(usv_id, task_id):
                    available.append(task_id)
                    break
        
        return available

    def get_available_usvs_for_task(self, task_id: int) -> List[int]:
        """Get list of USVs that can execute the specified task."""
        return [
            usv_id for usv_id in range(self.n_usvs)
            if self._can_usv_do_task(usv_id, task_id)
        ]

    def get_idle_usvs(self) -> List[int]:
        """Get list of currently idle USVs."""
        return [u for u in range(self.n_usvs) if self._is_usv_idle(u)]

    def step(self, task_id: int, usv_id: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action: assign task to USV.
        
        Args:
            task_id: Task to execute
            usv_id: USV to assign
            
        Returns:
            next_state: New state dictionary
            reward: Step reward (negative makespan change)
            done: Whether all tasks are scheduled
            info: Additional information
        """
        # Validate action
        if self.task_states[task_id, 0] == 1:
            return self._get_state(), -10.0, True, {'error': 'task_already_scheduled'}
        
        if not self._can_usv_do_task(usv_id, task_id):
            return self._get_state(), -10.0, True, {'error': 'invalid_usv_for_task'}
        
        usv_pos = self.usv_states[usv_id, 0:2]
        task_pos = self.task_coords[task_id]
        
        # Calculate travel and execution times
        dist = np.linalg.norm(task_pos - usv_pos)
        travel_time = dist / self.config.usv_speed
        energy_move = dist * self.config.energy_cost_per_distance
        task_duration = self.task_durations[task_id]
        energy_task = task_duration * self.config.energy_cost_per_task_time
        
        task_start = self.current_time + travel_time
        task_end = task_start + task_duration
        
        # Record travel event
        if travel_time > 0:
            self.usv_history[usv_id].append({
                'type': 'move',
                'start': self.current_time,
                'end': task_start,
                'info': 'Travel'
            })
        
        # Record task event
        self.usv_history[usv_id].append({
            'type': 'task',
            'start': task_start,
            'end': task_end,
            'info': f'T{task_id}'
        })
        
        # Update states
        self.task_states[task_id] = [1, task_start, task_end, usv_id]
        self.usv_states[usv_id, 0:2] = task_pos
        self.usv_states[usv_id, 2] -= (energy_move + energy_task)
        self.usv_states[usv_id, 3] = task_end
        self.usv_task_counts[usv_id] += 1
        self.n_scheduled_tasks += 1
        
        done = self.n_scheduled_tasks == self.n_tasks
        
        # Handle post-action logic
        if done:
            self._final_return()
            self.current_time = self._compute_makespan()
        else:
            # Ensure environment is ready for next action
            # This handles auto-charging and time advancement
            has_valid_actions = self.ensure_ready_state()
            if not has_valid_actions:
                # No more valid actions possible - treat as done (failure case)
                done = True
        
        # Scale rewards by instance time scale to stabilize PPO/Critic.
        current_makespan = self._compute_makespan()
        reward = (self.last_makespan - current_makespan) / self.scale_time
        if done and self.n_scheduled_tasks == self.n_tasks:
            reward -= current_makespan / self.scale_time
        self.last_makespan = current_makespan
        
        return self._get_state(), reward, done, {'makespan': current_makespan}

    def _final_return(self):
        """Return all USVs to origin after all tasks are completed."""
        for u in range(self.n_usvs):
            curr_pos = self.usv_states[u, 0:2]
            dist_home = np.linalg.norm(curr_pos)
            
            if dist_home > 1e-3:
                travel_home = dist_home / self.config.usv_speed
                home_start = self.usv_states[u, 3]
                home_end = home_start + travel_home
                
                self.usv_history[u].append({
                    'type': 'move',
                    'start': home_start,
                    'end': home_end,
                    'info': 'Final Return'
                })
                
                self.usv_states[u, 3] = home_end
                self.usv_states[u, 0:2] = [0, 0]

    def _compute_makespan(self) -> float:
        """
        Compute current makespan estimate.
        Makespan = max(USV finish time + return time) for all USVs.
        """
        finish_times = []
        for i in range(self.n_usvs):
            busy_until = self.usv_states[i, 3]
            curr_pos = self.usv_states[i, 0:2]
            return_time = np.linalg.norm(curr_pos) / self.config.usv_speed
            finish_times.append(busy_until + return_time)
        
        return max(finish_times)
