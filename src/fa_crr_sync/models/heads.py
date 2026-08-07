"""Score-difference regression heads."""

from __future__ import annotations

import torch
from torch import nn


class ScoreRegressor(nn.Module):
    def __init__(self, input_dimension: int = 64, outputs: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, outputs),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
