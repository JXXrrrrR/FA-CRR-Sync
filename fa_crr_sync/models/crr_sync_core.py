"""Feature-level CRR-Sync core, separated from the I3D backbone."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from .attention import ProcedureAwareDecoder
from .heads import ScoreRegressor
from .procedure import ProcedureSegmentationNet, decode_transition_logits
from .stages import pool_three_stages_1d, pool_three_stages_3d


class CRRSyncOutput(TypedDict):
    score_delta_query_reference: torch.Tensor
    score_delta_reference_query: torch.Tensor
    synchronisation_delta_query_reference: torch.Tensor
    synchronisation_delta_reference_query: torch.Tensor
    transition_logits: torch.Tensor
    decoded_boundaries: torch.Tensor
    used_boundaries: torch.Tensor


class CRRSyncCore(nn.Module):
    """Historical CRR-Sync logic operating on nine I3D snippet features."""

    def __init__(
        self,
        stage_size: int = 5,
        stages: int = 3,
        feature_dimension: int = 64,
    ) -> None:
        super().__init__()
        if stages != 3:
            raise ValueError("The maintained CRR-Sync contract uses exactly three stages")
        self.stage_size = stage_size
        self.stages = stages
        self.procedure = ProcedureSegmentationNet(
            snippet_channels=9, transition_outputs=2
        )
        self.decoder = ProcedureAwareDecoder(
            dimension=feature_dimension, heads=8, layers=3
        )
        self.score_regressor = ScoreRegressor(feature_dimension, outputs=1)
        self.synchronisation_regressor = ScoreRegressor(
            feature_dimension, outputs=3
        )

    @staticmethod
    def _validate_inputs(
        live_query_feature: torch.Tensor,
        live_reference_feature: torch.Tensor,
        live_query_map: torch.Tensor,
        live_reference_map: torch.Tensor,
        replay_query_feature: torch.Tensor,
        replay_reference_feature: torch.Tensor,
    ) -> None:
        expected_feature_tail = (9, 1024)
        for name, tensor in {
            "live_query_feature": live_query_feature,
            "live_reference_feature": live_reference_feature,
            "replay_query_feature": replay_query_feature,
            "replay_reference_feature": replay_reference_feature,
        }.items():
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected_feature_tail:
                raise ValueError(
                    f"{name} must have shape [B,9,1024], got {tuple(tensor.shape)}"
                )
        for name, tensor in {
            "live_query_map": live_query_map,
            "live_reference_map": live_reference_map,
        }.items():
            if tensor.ndim != 6 or tuple(tensor.shape[1:]) != (
                9,
                1024,
                2,
                4,
                4,
            ):
                raise ValueError(
                    f"{name} must have shape [B,9,1024,2,4,4], "
                    f"got {tuple(tensor.shape)}"
                )

    @staticmethod
    def _flatten_spatial_tokens(feature_maps: torch.Tensor) -> torch.Tensor:
        batch, channels, temporal, height, width = feature_maps.shape
        return (
            feature_maps.permute(0, 2, 3, 4, 1)
            .reshape(batch, temporal * height * width, channels)
            .contiguous()
        )

    @staticmethod
    def _fuse_live_replay(
        live_features: torch.Tensor, replay_features: torch.Tensor
    ) -> torch.Tensor:
        live_weight = live_features.softmax(dim=1)
        replay_weight = replay_features.softmax(dim=1)
        return live_weight * live_features + replay_weight * replay_features

    def forward(
        self,
        live_query_feature: torch.Tensor,
        live_reference_feature: torch.Tensor,
        live_query_map: torch.Tensor,
        live_reference_map: torch.Tensor,
        replay_query_feature: torch.Tensor,
        replay_reference_feature: torch.Tensor,
        stage_boundaries: torch.Tensor | None = None,
    ) -> CRRSyncOutput:
        self._validate_inputs(
            live_query_feature,
            live_reference_feature,
            live_query_map,
            live_reference_map,
            replay_query_feature,
            replay_reference_feature,
        )
        batch = live_query_feature.shape[0]

        live_features = torch.cat(
            [live_query_feature, live_reference_feature], dim=0
        )
        expanded_live, transition_logits = self.procedure(live_features)
        decoded_boundaries = decode_transition_logits(transition_logits)
        used_boundaries = (
            decoded_boundaries if stage_boundaries is None else stage_boundaries.long()
        )
        if tuple(used_boundaries.shape) != (2 * batch, 2):
            raise ValueError(
                f"stage_boundaries must have shape {(2 * batch, 2)}, "
                f"got {tuple(used_boundaries.shape)}"
            )

        replay_features = torch.cat(
            [replay_query_feature, replay_reference_feature], dim=0
        )
        expanded_replay, _ = self.procedure(replay_features)
        fused_features = self._fuse_live_replay(
            expanded_live, expanded_replay
        )

        live_maps = torch.cat([live_query_map, live_reference_map], dim=0)
        live_maps = live_maps.mean(dim=3)
        _, snippets, channels, height, width = live_maps.shape
        # Preserve the historical reshape contract used by the original code.
        flattened_maps = live_maps.reshape(-1, snippets, channels)
        expanded_maps, _ = self.procedure(flattened_maps)
        expanded_maps = expanded_maps.reshape(
            2 * batch, 96, 64, height, width
        )

        query_features, reference_features = fused_features.chunk(2, dim=0)
        query_maps, reference_maps = expanded_maps.chunk(2, dim=0)
        query_boundaries, reference_boundaries = used_boundaries.chunk(2, dim=0)

        query_stages = pool_three_stages_1d(
            query_features.transpose(1, 2), query_boundaries, self.stage_size
        ).transpose(1, 2)
        reference_stages = pool_three_stages_1d(
            reference_features.transpose(1, 2),
            reference_boundaries,
            self.stage_size,
        ).transpose(1, 2)
        query_stage_maps = pool_three_stages_3d(
            query_maps.transpose(1, 2), query_boundaries, self.stage_size
        )
        reference_stage_maps = pool_three_stages_3d(
            reference_maps.transpose(1, 2),
            reference_boundaries,
            self.stage_size,
        )

        query_map_tokens = self._flatten_spatial_tokens(query_stage_maps)
        reference_map_tokens = self._flatten_spatial_tokens(reference_stage_maps)

        query_to_reference = []
        reference_to_query = []
        spatial_tokens_per_stage = self.stage_size * height * width
        for stage in range(self.stages):
            feature_start = stage * self.stage_size
            feature_end = (stage + 1) * self.stage_size
            map_start = stage * spatial_tokens_per_stage
            map_end = (stage + 1) * spatial_tokens_per_stage
            query_to_reference.append(
                self.decoder(
                    query_stages[:, feature_start:feature_end],
                    reference_map_tokens[:, map_start:map_end],
                )
            )
            reference_to_query.append(
                self.decoder(
                    reference_stages[:, feature_start:feature_end],
                    query_map_tokens[:, map_start:map_end],
                )
            )

        directional_tokens = torch.cat(
            [
                torch.cat(query_to_reference, dim=1),
                torch.cat(reference_to_query, dim=1),
            ],
            dim=0,
        )
        score_delta = self.score_regressor(directional_tokens).mean(dim=1)
        synchronisation_delta = self.synchronisation_regressor(
            directional_tokens
        ).mean(dim=1)

        return {
            "score_delta_query_reference": score_delta[:batch],
            "score_delta_reference_query": score_delta[batch:],
            "synchronisation_delta_query_reference": synchronisation_delta[:batch],
            "synchronisation_delta_reference_query": synchronisation_delta[batch:],
            "transition_logits": transition_logits,
            "decoded_boundaries": decoded_boundaries,
            "used_boundaries": used_boundaries,
        }
