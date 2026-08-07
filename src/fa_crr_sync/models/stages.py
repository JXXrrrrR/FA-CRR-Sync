"""Three-stage pooling shared by CRR-Sync variants."""

from __future__ import annotations

import torch
from torch.nn import functional


def _validate_boundaries(boundaries: torch.Tensor, temporal_length: int) -> None:
    if boundaries.ndim != 2 or boundaries.shape[1] != 2:
        raise ValueError("Boundaries must have shape [B, 2]")
    if torch.any(boundaries[:, 0] <= 0):
        raise ValueError("First boundary must be greater than zero")
    if torch.any(boundaries[:, 1] <= boundaries[:, 0]):
        raise ValueError("Second boundary must follow the first")
    if torch.any(boundaries[:, 1] >= temporal_length):
        raise ValueError("Second boundary must precede the final temporal position")


def pool_three_stages_1d(
    features: torch.Tensor, boundaries: torch.Tensor, stage_size: int
) -> torch.Tensor:
    """Pool `[B,C,T]` features to `[B,C,3*stage_size]`."""

    if features.ndim != 3:
        raise ValueError("Expected features with shape [B, C, T]")
    _validate_boundaries(boundaries, features.shape[2])
    pooled = []
    for sample, sample_boundaries in zip(features, boundaries):
        first, second = (int(value) for value in sample_boundaries)
        segments = (
            sample[None, :, :first],
            sample[None, :, first:second],
            sample[None, :, second:],
        )
        pooled.append(
            torch.cat(
                [
                    functional.interpolate(
                        segment,
                        size=stage_size,
                        mode="linear",
                        align_corners=True,
                    )
                    for segment in segments
                ],
                dim=2,
            )
        )
    return torch.cat(pooled, dim=0)


def pool_three_stages_3d(
    feature_maps: torch.Tensor, boundaries: torch.Tensor, stage_size: int
) -> torch.Tensor:
    """Pool `[B,C,T,H,W]` maps to `[B,C,3*stage_size,H,W]`."""

    if feature_maps.ndim != 5:
        raise ValueError("Expected feature maps with shape [B, C, T, H, W]")
    _validate_boundaries(boundaries, feature_maps.shape[2])
    pooled = []
    height, width = feature_maps.shape[-2:]
    for sample, sample_boundaries in zip(feature_maps, boundaries):
        first, second = (int(value) for value in sample_boundaries)
        segments = (
            sample[None, :, :first],
            sample[None, :, first:second],
            sample[None, :, second:],
        )
        pooled.append(
            torch.cat(
                [
                    functional.interpolate(
                        segment,
                        size=(stage_size, height, width),
                        mode="trilinear",
                        align_corners=True,
                    )
                    for segment in segments
                ],
                dim=2,
            )
        )
    return torch.cat(pooled, dim=0)
