from __future__ import annotations

import unittest

import torch
from torch import nn

from fa_crr_sync.models import CRRSyncCore, FACRRSync, I3DBackbone


class FakeBackbone(nn.Module):
    def forward(self, videos: torch.Tensor):
        batch = videos.shape[0]
        value = videos.mean(dim=(1, 2, 3, 4), keepdim=True)
        feature = value.reshape(batch, 1, 1).expand(batch, 9, 1024)
        feature_map = value.reshape(batch, 1, 1, 1, 1, 1).expand(
            batch, 9, 1024, 2, 4, 4
        )
        return feature.contiguous(), feature_map.contiguous()


class ForegroundAwareModelTests(unittest.TestCase):
    def test_mask_backbone_accepts_one_channel(self) -> None:
        model = I3DBackbone(modality="mask")
        self.assertEqual(
            model.backbone.conv3d_1a_7x7.conv3d.in_channels, 1
        )

    def test_sigmoid_gate_matches_manuscript_equation(self) -> None:
        rgb_feature = torch.full((1, 9, 1024), 2.0)
        mask_feature = torch.zeros_like(rgb_feature)
        rgb_map = torch.full((1, 9, 1024, 2, 4, 4), 4.0)
        mask_map = torch.zeros_like(rgb_map)
        gated_feature, gated_map = FACRRSync._gate(
            rgb_feature, rgb_map, mask_feature, mask_map
        )
        self.assertTrue(torch.equal(gated_feature, torch.ones_like(rgb_feature)))
        self.assertTrue(torch.equal(gated_map, torch.full_like(rgb_map, 2.0)))

    def test_full_feature_contract_with_both_view_masks(self) -> None:
        model = FACRRSync.__new__(FACRRSync)
        nn.Module.__init__(model)
        model.rgb_backbone = FakeBackbone()
        model.mask_backbone = FakeBackbone()
        model.core = CRRSyncCore()
        rgb = [torch.randn(1, 3, 96, 2, 2) for _ in range(4)]
        masks = [torch.rand(1, 1, 96, 2, 2) for _ in range(4)]
        output = model(*rgb, *masks, stage_boundaries=torch.tensor([[30, 70], [31, 71]]))
        self.assertEqual(tuple(output["score_delta_query_reference"].shape), (1, 1))
        self.assertEqual(
            tuple(output["synchronisation_delta_query_reference"].shape),
            (1, 3),
        )


if __name__ == "__main__":
    unittest.main()
