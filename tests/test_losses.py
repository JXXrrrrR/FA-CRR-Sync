from __future__ import annotations

import unittest

import torch

from fa_crr_sync.training import compute_crr_sync_losses, transition_targets


class LossTests(unittest.TestCase):
    def test_transition_targets(self) -> None:
        boundaries = torch.tensor([[10, 70], [20, 80]])
        targets = transition_targets(boundaries)
        self.assertEqual(tuple(targets.shape), (2, 96, 2))
        self.assertEqual(float(targets.sum()), 4.0)
        self.assertEqual(float(targets[0, 10, 0]), 1.0)
        self.assertEqual(float(targets[0, 70, 1]), 1.0)

    def test_joint_loss_is_finite_and_differentiable(self) -> None:
        score_forward = torch.zeros(2, 1, requires_grad=True)
        score_reverse = torch.zeros(2, 1, requires_grad=True)
        sync_forward = torch.zeros(2, 3, requires_grad=True)
        sync_reverse = torch.zeros(2, 3, requires_grad=True)
        transition_logits = torch.zeros(
            (4, 96, 2), requires_grad=True
        )
        output = {
            "score_delta_query_reference": score_forward,
            "score_delta_reference_query": score_reverse,
            "synchronisation_delta_query_reference": sync_forward,
            "synchronisation_delta_reference_query": sync_reverse,
            "transition_logits": transition_logits,
            "decoded_boundaries": torch.tensor(
                [[20, 70], [20, 70], [20, 70], [20, 70]]
            ),
            "used_boundaries": torch.tensor(
                [[20, 70], [20, 70], [20, 70], [20, 70]]
            ),
        }
        losses = compute_crr_sync_losses(
            output,
            query_score=torch.tensor([50.0, 60.0]),
            reference_score=torch.tensor([55.0, 58.0]),
            query_synchronisation=torch.tensor(
                [[8.0, 8.5, 9.0], [7.0, 7.5, 8.0]]
            ),
            reference_synchronisation=torch.tensor(
                [[7.0, 8.0, 8.5], [7.5, 7.0, 8.5]]
            ),
            target_boundaries=torch.tensor(
                [[20, 70], [25, 75], [22, 72], [26, 76]]
            ),
        )
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertIsNotNone(score_forward.grad)
        self.assertIsNotNone(transition_logits.grad)


if __name__ == "__main__":
    unittest.main()
