"""Smoke tests for batched HGNN and PairwiseActor paths."""

import importlib.util
import unittest

import numpy as np


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch not installed")
class VectorizedUpdateSmokeTest(unittest.TestCase):
    def test_batched_encoder_and_actor_shapes(self):
        import torch

        from config import get_config
        from ppo import PPOAgent

        cfg = get_config(
            hidden_dim=16,
            hgnn_layers=1,
            n_heads=4,
            dropout=0.0,
            vectorized_update=True,
        )
        agent = PPOAgent(cfg, n_usvs=2, n_tasks=20, device='cpu', verbose=False)
        state = {
            'usv_features': torch.rand(3, 2, 7),
            'task_features': torch.rand(3, 20, 8),
            'edge_features': torch.rand(3, 2, 20, 4),
        }
        encoded = agent.actor_encoder(state)
        self.assertEqual(tuple(encoded['usv_embed'].shape), (3, 2, 16))
        self.assertEqual(tuple(encoded['task_embed'].shape), (3, 20, 16))
        self.assertEqual(tuple(encoded['graph_embed'].shape), (3, 32))

        pair_mask = torch.ones(3, 20, 2, dtype=torch.bool)
        logits, probs = agent.actor(encoded, state['edge_features'], pair_mask)
        self.assertEqual(tuple(logits.shape), (3, 20, 2))
        self.assertEqual(tuple(probs.shape), (3, 20, 2))

        actions_flat = torch.tensor([0, 1, 2], dtype=torch.long)
        log_probs, entropy, _ = agent.actor.get_batch_action_log_prob(
            encoded,
            state['edge_features'],
            pair_mask,
            actions_flat,
        )
        self.assertEqual(tuple(log_probs.shape), (3,))
        self.assertEqual(tuple(entropy.shape), (3,))

    def test_vectorized_update_runs_and_clears_buffer(self):
        import torch

        from config import get_config
        from ppo import PPOAgent

        cfg = get_config(
            hidden_dim=16,
            hgnn_layers=1,
            n_heads=4,
            dropout=0.0,
            vectorized_update=True,
            update_batch_size=3,
            update_micro_batch_size=1,
            max_update_pairs=40,
            ppo_epochs=1,
        )
        agent = PPOAgent(cfg, n_usvs=2, n_tasks=20, device='cpu', verbose=False)
        for idx in range(3):
            state = {
                'usv_features': np.random.rand(2, 7).astype(np.float32),
                'task_features': np.random.rand(20, 8).astype(np.float32),
                'edge_features': np.random.rand(2, 20, 4).astype(np.float32),
            }
            agent.store_transition(
                state_dict=state,
                action=(idx % 20, idx % 2),
                log_prob=-3.0,
                reward=1.0,
                done=idx == 2,
                value=0.0,
                task_mask=torch.ones(20, dtype=torch.bool),
                usv_masks=torch.ones(20, 2, dtype=torch.bool),
            )

        loss_info = agent.update()
        self.assertIn('batch_prepare_time_sec', loss_info)
        self.assertIn('actor_update_time_sec', loss_info)
        self.assertIn('critic_update_time_sec', loss_info)
        self.assertEqual(loss_info['effective_update_batch_size'], 3)
        self.assertEqual(loss_info['effective_update_micro_batch_size'], 1)
        self.assertEqual(len(agent.states), 0)


if __name__ == "__main__":
    unittest.main()
