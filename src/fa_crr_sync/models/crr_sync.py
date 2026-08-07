"""End-to-end CRR-Sync model joining I3D and the maintained feature core."""

from __future__ import annotations

import torch
from torch import nn

from .crr_sync_core import CRRSyncCore, CRRSyncOutput
from .i3d import I3DBackbone


class CRRSync(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = I3DBackbone()
        self.core = CRRSyncCore()

    def forward(
        self,
        live_query: torch.Tensor,
        live_reference: torch.Tensor,
        replay_query: torch.Tensor,
        replay_reference: torch.Tensor,
        stage_boundaries: torch.Tensor | None = None,
    ) -> CRRSyncOutput:
        batch = live_query.shape[0]
        if not all(
            tensor.shape[0] == batch
            for tensor in (live_reference, replay_query, replay_reference)
        ):
            raise ValueError("All four video inputs must share a batch size")

        live_features, live_maps = self.backbone(
            torch.cat([live_query, live_reference], dim=0)
        )
        replay_features, _ = self.backbone(
            torch.cat([replay_query, replay_reference], dim=0)
        )
        live_query_feature, live_reference_feature = live_features.chunk(2, dim=0)
        live_query_map, live_reference_map = live_maps.chunk(2, dim=0)
        replay_query_feature, replay_reference_feature = replay_features.chunk(
            2, dim=0
        )
        return self.core(
            live_query_feature,
            live_reference_feature,
            live_query_map,
            live_reference_map,
            replay_query_feature,
            replay_reference_feature,
            stage_boundaries=stage_boundaries,
        )
