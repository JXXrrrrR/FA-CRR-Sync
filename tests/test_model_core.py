from __future__ import annotations

import unittest

import torch

from fa_crr_sync.models import CRRSyncCore, ProcedureSegmentationNet
from fa_crr_sync.models.procedure import decode_transition_logits
from fa_crr_sync.models.stages import pool_three_stages_1d, pool_three_stages_3d


class ModelComponentTests(unittest.TestCase):
    def test_procedure_segmentation_shapes(self) -> None:
        model = ProcedureSegmentationNet()
        features, transitions = model(torch.randn(2, 9, 1024))
        self.assertEqual(tuple(features.shape), (2, 96, 64))
        self.assertEqual(tuple(transitions.shape), (2, 96, 2))

    def test_stage_pooling_shapes(self) -> None:
        boundaries = torch.tensor([[30, 70], [25, 65]])
        features = torch.randn(2, 64, 96)
        maps = torch.randn(2, 64, 96, 4, 4)
        self.assertEqual(
            tuple(pool_three_stages_1d(features, boundaries, 5).shape),
            (2, 64, 15),
        )
        self.assertEqual(
            tuple(pool_three_stages_3d(maps, boundaries, 5).shape),
            (2, 64, 15, 4, 4),
        )

    def test_decoded_boundaries_never_create_empty_outer_stages(self) -> None:
        logits = torch.zeros(1, 96, 2)
        logits[0, 0, 0] = 10
        logits[0, 95, 1] = 10
        self.assertEqual(
            decode_transition_logits(logits).tolist(),
            [[1, 94]],
        )

    def test_feature_core_forward_backward(self) -> None:
        torch.manual_seed(0)
        model = CRRSyncCore()
        feature = lambda: torch.randn(1, 9, 1024)
        feature_map = lambda: torch.randn(1, 9, 1024, 2, 4, 4)
        boundaries = torch.tensor([[30, 70], [32, 72]])
        output = model(
            feature(),
            feature(),
            feature_map(),
            feature_map(),
            feature(),
            feature(),
            stage_boundaries=boundaries,
        )
        self.assertEqual(
            tuple(output["score_delta_query_reference"].shape), (1, 1)
        )
        self.assertEqual(
            tuple(output["synchronisation_delta_query_reference"].shape), (1, 3)
        )
        self.assertEqual(
            tuple(output["transition_logits"].shape), (2, 96, 2)
        )
        loss = (
            output["score_delta_query_reference"].square().mean()
            + output["score_delta_reference_query"].square().mean()
            + output["synchronisation_delta_query_reference"].square().mean()
            + output["synchronisation_delta_reference_query"].square().mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
