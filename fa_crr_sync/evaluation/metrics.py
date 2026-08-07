"""Canonical score and temporal-boundary metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ScoreMetrics:
    """FineSynchro score metrics in the historical reporting convention."""

    srcc: float
    mae: float
    mse: float
    relative_l2: float
    relative_l2_x100: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _paired_arrays(
    predictions: Iterable[float], targets: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(list(predictions), dtype=np.float64).reshape(-1)
    target_array = np.asarray(list(targets), dtype=np.float64).reshape(-1)
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{prediction_array.shape} vs {target_array.shape}"
        )
    if prediction_array.size < 2:
        raise ValueError("At least two paired samples are required")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("Metrics require finite predictions and targets")
    return prediction_array, target_array


def aggregate_score_metrics(
    predictions: Iterable[float], targets: Iterable[float]
) -> ScoreMetrics:
    """Compute SRCC, MAE, MSE, and the legacy range-normalized R-l2."""

    prediction_array, target_array = _paired_arrays(predictions, targets)
    score_range = float(target_array.max() - target_array.min())
    if score_range <= 0:
        raise ValueError("Relative L2 is undefined for a zero target-score range")
    residual = prediction_array - target_array
    relative_l2 = float(np.mean(np.square(residual / score_range)))
    srcc = float(stats.spearmanr(prediction_array, target_array).statistic)
    return ScoreMetrics(
        srcc=srcc,
        mae=float(np.mean(np.abs(residual))),
        mse=float(np.mean(np.square(residual))),
        relative_l2=relative_l2,
        relative_l2_x100=relative_l2 * 100.0,
    )


def temporal_iou(
    target_boundary: Iterable[float], predicted_boundary: Iterable[float]
) -> float:
    """Legacy interval IoU for the two macro-stage boundaries."""

    target = np.asarray(list(target_boundary), dtype=np.float64).reshape(-1)
    predicted = np.asarray(list(predicted_boundary), dtype=np.float64).reshape(-1)
    if target.shape != (2,) or predicted.shape != (2,):
        raise ValueError("Temporal IoU expects two boundaries per sample")
    if target[0] > target[1] or predicted[0] > predicted[1]:
        raise ValueError("Temporal boundaries must be ordered")
    intersection = max(
        0.0, min(target[1], predicted[1]) - max(target[0], predicted[0])
    )
    union = (
        (predicted[1] - predicted[0])
        + (target[1] - target[0])
        - intersection
    )
    return float(intersection / (union + np.finfo(np.float64).eps))


def temporal_iou_accuracy(
    target_boundaries: Iterable[Iterable[float]],
    predicted_boundaries: Iterable[Iterable[float]],
    threshold: float,
) -> float:
    """Fraction of samples whose temporal IoU reaches a threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    targets = list(target_boundaries)
    predictions = list(predicted_boundaries)
    if len(targets) != len(predictions):
        raise ValueError("Target/prediction boundary counts differ")
    if not targets:
        raise ValueError("At least one boundary pair is required")
    values = [
        temporal_iou(target, prediction)
        for target, prediction in zip(targets, predictions)
    ]
    return float(np.mean(np.asarray(values) >= threshold))
