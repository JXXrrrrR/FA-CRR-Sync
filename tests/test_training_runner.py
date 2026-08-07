from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from fa_crr_sync.training.runner import predict_voters, train_one_epoch


class TinyCRRSync(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        live_query,
        live_reference,
        replay_query,
        replay_reference,
        stage_boundaries=None,
    ):
        batch = live_query.shape[0]
        scalar = self.value.expand(batch, 1)
        sync = self.value.expand(batch, 3)
        logits = self.value.expand(2 * batch, 96, 2)
        boundaries = torch.tensor(
            [[30, 60]], device=live_query.device
        ).expand(2 * batch, 2)
        return {
            "score_delta_query_reference": scalar,
            "score_delta_reference_query": scalar,
            "synchronisation_delta_query_reference": sync,
            "synchronisation_delta_reference_query": sync,
            "transition_logits": logits,
            "decoded_boundaries": boundaries,
            "used_boundaries": boundaries,
        }


class TinyBatchNormCRRSync(TinyCRRSync):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(1)

    def forward(
        self,
        live_query,
        live_reference,
        replay_query,
        replay_reference,
        stage_boundaries=None,
    ):
        batch = live_query.shape[0]
        normalised = self.batch_norm(live_query.reshape(batch, 1))
        scalar = (self.value + normalised[:, :1])
        sync = scalar.expand(batch, 3)
        logits = scalar.reshape(batch, 1, 1).expand(batch, 96, 2)
        logits = torch.cat([logits, logits], dim=0)
        boundaries = torch.tensor(
            [[30, 60]], device=live_query.device
        ).expand(2 * batch, 2)
        return {
            "score_delta_query_reference": scalar,
            "score_delta_reference_query": scalar,
            "synchronisation_delta_query_reference": sync,
            "synchronisation_delta_reference_query": sync,
            "transition_logits": logits,
            "decoded_boundaries": boundaries,
            "used_boundaries": boundaries,
        }


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def sample(live_value: float = 0.0) -> dict:
    role = {
        "live": torch.tensor([live_value]),
        "replay": torch.zeros(1),
        "score": torch.tensor(1.0),
        "synchronisation": torch.ones(3),
        "boundaries": torch.tensor([30, 60]),
        "sample_id": "sample",
        "action_code": "101b",
    }
    return {"query": role, "reference": dict(role)}


class GradientAccumulationTests(unittest.TestCase):
    def test_all_batch_norm_layers_remain_trainable(self) -> None:
        model = TinyBatchNormCRRSync()
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        config = {
            "optimization": {
                "mixed_precision": False,
                "gradient_accumulation_steps": 1,
            },
            "loss": {
                "score_weight": 1.0,
                "transition_weight": 1.0,
                "synchronisation_weight": 1.0,
            },
        }
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        train_one_epoch(
            model,
            DataLoader([sample(-1.0), sample(1.0)], batch_size=2),
            optimizer,
            scaler,
            torch.device("cpu"),
            config,
            use_ground_truth_stages=True,
        )
        self.assertTrue(model.batch_norm.training)
        self.assertEqual(int(model.batch_norm.num_batches_tracked), 1)
        self.assertFalse(
            torch.equal(
                model.batch_norm.weight,
                torch.ones_like(model.batch_norm.weight),
            )
        )

    def test_partial_final_window_steps_once(self) -> None:
        model = TinyCRRSync()
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        config = {
            "optimization": {
                "mixed_precision": False,
                "gradient_accumulation_steps": 2,
            },
            "loss": {
                "score_weight": 1.0,
                "transition_weight": 1.0,
                "synchronisation_weight": 1.0,
            },
        }
        metrics = train_one_epoch(
            model,
            DataLoader([sample(), sample(), sample()], batch_size=1),
            optimizer,
            scaler,
            torch.device("cpu"),
            config,
            use_ground_truth_stages=True,
        )
        self.assertEqual(optimizer.step_count, 2)
        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertEqual(metrics["gradient_accumulation_steps"], 2)
        self.assertTrue(torch.isfinite(model.value))

    def test_inference_uses_query_to_reference_direction_only(self) -> None:
        model = TinyCRRSync()
        rows = predict_voters(
            model,
            DataLoader([sample()], batch_size=1),
            torch.device("cpu"),
            mixed_precision=False,
        )
        self.assertAlmostEqual(rows[0]["reference_score"], 1.0)
        self.assertAlmostEqual(
            rows[0]["score_prediction_query_reference"], 1.1, places=6
        )
        self.assertAlmostEqual(rows[0]["score_prediction"], 1.1, places=6)


if __name__ == "__main__":
    unittest.main()
