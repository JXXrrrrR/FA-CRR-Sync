"""Procedure-aware cross-attention without unrelated text dependencies."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dimension: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dimension * ratio)
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int = 8,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension**-0.5
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dimension, dimension)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        return (
            tensor.reshape(batch, tokens, self.heads, self.head_dimension)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query_heads = self._heads(self.query(query))
        key_heads = self._heads(self.key(value))
        value_heads = self._heads(self.value(value))
        attention = (query_heads @ key_heads.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = attention @ value_heads
        output = output.permute(0, 2, 1, 3).flatten(2)
        return self.projection_dropout(self.projection(output))


class DecoderBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.value_norm = nn.LayerNorm(dimension)
        self.attention = CrossAttention(
            dimension,
            heads=heads,
            attention_dropout=dropout,
            projection_dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension, dropout=dropout)

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query = query + self.attention(
            self.query_norm(query), self.value_norm(value)
        )
        return query + self.feed_forward(self.output_norm(query))


class ProcedureAwareDecoder(nn.Module):
    def __init__(self, dimension: int = 64, heads: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(dimension, heads) for _ in range(layers)
        )

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, value)
        return query
