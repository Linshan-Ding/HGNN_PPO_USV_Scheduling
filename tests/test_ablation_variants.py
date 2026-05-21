"""Smoke tests for PPO ablation configuration."""

import importlib.util
import unittest

import numpy as np

from config import InstanceConfig, get_config
from env import USVSchedulingEnv


class AblationVariantSmokeTest(unittest.TestCase):
    def test_no_reward_norm_config(self):
        cfg = get_config(ablation_variant='no_reward_norm')
        self.assertFalse(cfg.instance.reward_normalization)

    def test_reward_normalization_switch_changes_reward_scale(self):
        base_instance = {
            'instance_id': 'reward_smoke',
            'n_usvs': 1,
            'n_tasks': 1,
            'task_coords': np.array([[10.0, 0.0]]),
            'fuzzy_times': np.array([[4.0, 4.0, 4.0]]),
            'config': InstanceConfig(
                n_usvs=1,
                n_tasks=1,
                battery_capacity=1000.0,
                usv_speed=5.0,
                reward_normalization=True,
            ),
        }

        norm_env = USVSchedulingEnv(base_instance)
        _, norm_reward, _, _ = norm_env.step(0, 0)

        raw_instance = dict(base_instance)
        raw_instance['config'] = InstanceConfig(
            n_usvs=1,
            n_tasks=1,
            battery_capacity=1000.0,
            usv_speed=5.0,
            reward_normalization=False,
        )
        raw_env = USVSchedulingEnv(raw_instance)
        _, raw_reward, _, _ = raw_env.step(0, 0)

        self.assertLess(raw_reward, norm_reward)
        self.assertAlmostEqual(raw_reward, norm_reward * raw_env.scale_time)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed")
    def test_ppo_variant_encoder_selection(self):
        from ppo import PPOAgent

        no_hgnn = PPOAgent(get_config(ablation_variant='no_hgnn', hidden_dim=16), 2, 20)
        self.assertEqual(no_hgnn.actor_encoder.__class__.__name__, 'SimpleGraphEncoder')

        shared = PPOAgent(get_config(ablation_variant='shared_encoder', hidden_dim=16), 2, 20)
        self.assertIs(shared.actor_encoder, shared.critic_encoder)
        self.assertIsNone(shared.critic_optimizer)


if __name__ == "__main__":
    unittest.main()
