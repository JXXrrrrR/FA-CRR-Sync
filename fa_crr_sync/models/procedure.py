"""Procedure segmentation network used by historical CRR-Sync."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv1d(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransitionHead(nn.Module):
    def __init__(self, in_features: int = 64, outputs: int = 2) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ProcedureSegmentationNet(nn.Module):
    """Expand nine I3D snippet features to 96 positions and two boundaries."""

    def __init__(self, snippet_channels: int = 9, transition_outputs: int = 2) -> None:
        super().__init__()
        self.input_block = DoubleConv1d(snippet_channels, 12)
        self.down1 = DownBlock1d(12, 24)
        self.down2 = DownBlock1d(24, 48)
        self.down3 = DownBlock1d(48, 96)
        self.down4 = DownBlock1d(96, 96)
        self.transition_head = TransitionHead(64, transition_outputs)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError(f"PSNet expects [B, 9, 1024], got {inputs.shape}")
        features = self.input_block(inputs)
        features = self.down1(features)
        features = self.down2(features)
        features = self.down3(features)
        features = self.down4(features)
        transitions = self.transition_head(features)
        return features, transitions


def decode_transition_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode one boundary from each non-overlapping logit partition."""

    if logits.ndim != 3:
        raise ValueError("Transition logits must have shape [B, T, K]")
    batch, temporal_length, transition_count = logits.shape
    if temporal_length % transition_count != 0:
        raise ValueError("Temporal length must be divisible by transition count")
    partition = temporal_length // transition_count
    decoded = torch.empty(
        (batch, transition_count),
        device=logits.device,
        dtype=torch.long,
    )
    for transition_index in range(transition_count):
        start = transition_index * partition
        end = (transition_index + 1) * partition
        local = logits[:, start:end, transition_index].argmax(dim=1)
        boundary = local + start
        # All three pooled stages must contain at least one temporal position.
        # Partitioned decoding already preserves order; only the two outer
        # endpoints need projection into the valid open interval.
        boundary = boundary.clamp(min=1, max=temporal_length - 2)
        decoded[:, transition_index] = boundary
    return decoded
