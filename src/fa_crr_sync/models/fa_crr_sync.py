"""Foreground-aware CRR-Sync model."""

from __future__ import annotations

import torch
from torch import nn

from .crr_sync_core import CRRSyncCore, CRRSyncOutput
from .i3d import I3DBackbone


class FACRRSync(nn.Module):
    """FA-CRR-Sync."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb_backbone = I3DBackbone(modality="rgb")
        self.mask_backbone = I3DBackbone(modality="mask")
        self.core = CRRSyncCore()

    @staticmethod
    def _validate_batch(tensors: tuple[torch.Tensor, ...]) -> int:
        batch = tensors[0].shape[0]
        if any(tensor.shape[0] != batch for tensor in tensors):
            raise ValueError("All RGB and mask inputs must share a batch size")
        for tensor in tensors[:4]:
            if tensor.ndim != 5 or tensor.shape[1] != 3:
                raise ValueError("RGB inputs must have shape [B,3,T,H,W]")
        for tensor in tensors[4:]:
            if tensor.ndim != 5 or tensor.shape[1] != 1:
                raise ValueError("Mask inputs must have shape [B,1,T,H,W]")
        return batch

    @staticmethod
    def _gate(
        rgb_feature: torch.Tensor,
        rgb_map: torch.Tensor,
        mask_feature: torch.Tensor,
        mask_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb_feature.shape != mask_feature.shape or rgb_map.shape != mask_map.shape:
            raise ValueError("RGB and mask I3D outputs must be dimensionally aligned")
        return (
            rgb_feature * torch.sigmoid(mask_feature),
            rgb_map * torch.sigmoid(mask_map),
        )

    def load_pretrained(self, path) -> None:
        """Initialize RGB I3D and the channel-averaged mask I3D together."""

        self.rgb_backbone.load_pretrained(path)
        self.mask_backbone.load_pretrained(path)

    def forward(
        self,
        live_query: torch.Tensor,
        live_reference: torch.Tensor,
        replay_query: torch.Tensor,
        replay_reference: torch.Tensor,
        live_mask_query: torch.Tensor,
        live_mask_reference: torch.Tensor,
        replay_mask_query: torch.Tensor,
        replay_mask_reference: torch.Tensor,
        stage_boundaries: torch.Tensor | None = None,
    ) -> CRRSyncOutput:
        inputs = (
            live_query,
            live_reference,
            replay_query,
            replay_reference,
            live_mask_query,
            live_mask_reference,
            replay_mask_query,
            replay_mask_reference,
        )
        batch = self._validate_batch(inputs)

        rgb_features, rgb_maps = self.rgb_backbone(torch.cat(inputs[:4], dim=0))
        mask_features, mask_maps = self.mask_backbone(torch.cat(inputs[4:], dim=0))
        gated_features, gated_maps = self._gate(
            rgb_features, rgb_maps, mask_features, mask_maps
        )
        live_q_feature, live_r_feature, replay_q_feature, replay_r_feature = (
            gated_features.chunk(4, dim=0)
        )
        live_q_map, live_r_map, _, _ = gated_maps.chunk(4, dim=0)
        if live_q_feature.shape[0] != batch:
            raise RuntimeError("Unexpected backbone batch decomposition")

        return self.core(
            live_q_feature,
            live_r_feature,
            live_q_map,
            live_r_map,
            replay_q_feature,
            replay_r_feature,
            stage_boundaries=stage_boundaries,
        )
