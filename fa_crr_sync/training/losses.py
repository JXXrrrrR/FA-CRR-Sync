"""Joint CRR-Sync training objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from fa_crr_sync.models.crr_sync_core import CRRSyncOutput


@dataclass(frozen=True)
class CRRSyncLosses:
    score: torch.Tensor
    transition: torch.Tensor
    synchronisation: torch.Tensor
    total: torch.Tensor


def transition_targets(
    boundaries: torch.Tensor,
    temporal_length: int = 96,
    transition_count: int = 2,
) -> torch.Tensor:
    """Create historical one-hot boundary targets with shape `[B,T,2]`."""

    if boundaries.ndim != 2 or boundaries.shape[1] != transition_count:
        raise ValueError(
            f"Expected boundaries [B,{transition_count}], got {boundaries.shape}"
        )
    if torch.any(boundaries < 0) or torch.any(boundaries >= temporal_length):
        raise ValueError("Boundary index outside temporal target range")
    targets = torch.zeros(
        (boundaries.shape[0], temporal_length, transition_count),
        device=boundaries.device,
        dtype=torch.float32,
    )
    batch_indices = torch.arange(boundaries.shape[0], device=boundaries.device)
    for transition_index in range(transition_count):
        targets[
            batch_indices,
            boundaries[:, transition_index].long(),
            transition_index,
        ] = 1.0
    return targets


def compute_crr_sync_losses(
    output: CRRSyncOutput,
    query_score: torch.Tensor,
    reference_score: torch.Tensor,
    query_synchronisation: torch.Tensor,
    reference_synchronisation: torch.Tensor,
    target_boundaries: torch.Tensor,
    score_weight: float = 1.0,
    transition_weight: float = 1.0,
    synchronisation_weight: float = 1.0,
) -> CRRSyncLosses:
    """Compute symmetric score/synchronisation deltas plus transition BCE."""

    query_score = query_score.reshape(-1, 1)
    reference_score = reference_score.reshape(-1, 1)
    if query_score.shape != reference_score.shape:
        raise ValueError("Query/reference score shapes differ")
    if query_synchronisation.shape != reference_synchronisation.shape:
        raise ValueError("Query/reference synchronisation shapes differ")

    score_difference = query_score - reference_score
    score_loss = functional.mse_loss(
        output["score_delta_query_reference"], score_difference
    ) + functional.mse_loss(
        output["score_delta_reference_query"], -score_difference
    )

    synchronisation_difference = (
        query_synchronisation - reference_synchronisation
    )
    synchronisation_loss = functional.mse_loss(
        output["synchronisation_delta_query_reference"],
        synchronisation_difference,
    ) + functional.mse_loss(
        output["synchronisation_delta_reference_query"],
        -synchronisation_difference,
    )

    logits = output["transition_logits"]
    targets = transition_targets(
        target_boundaries.to(logits.device),
        temporal_length=logits.shape[1],
        transition_count=logits.shape[2],
    )
    transition_loss = functional.binary_cross_entropy_with_logits(logits, targets)

    total = (
        score_weight * score_loss
        + transition_weight * transition_loss
        + synchronisation_weight * synchronisation_loss
    )
    return CRRSyncLosses(
        score=score_loss,
        transition=transition_loss,
        synchronisation=synchronisation_loss,
        total=total,
    )
