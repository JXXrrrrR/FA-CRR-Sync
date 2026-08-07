"""Dataset manifests and validated loading contracts."""

from .dataset import (
    ClipLoader,
    FineSynchroIndex,
    FineSynchroPairDataset,
    FixedPairDataset,
    EpochShuffleSampler,
    sampled_stage_boundaries,
    uniform_frame_indices,
)
from .manifest import build_finesynchro_manifest

__all__ = [
    "ClipLoader",
    "FineSynchroIndex",
    "FineSynchroPairDataset",
    "FixedPairDataset",
    "EpochShuffleSampler",
    "build_finesynchro_manifest",
    "sampled_stage_boundaries",
    "uniform_frame_indices",
]
