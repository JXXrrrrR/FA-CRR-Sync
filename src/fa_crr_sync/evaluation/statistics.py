"""Bootstrap confidence intervals and paired prediction comparisons."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats

from .metrics import aggregate_score_metrics


def _arrays(
    predictions: Iterable[float], targets: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(list(predictions), dtype=np.float64).reshape(-1)
    target = np.asarray(list(targets), dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or prediction.size < 2:
        raise ValueError("Predictions and targets require at least two paired values")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Statistical analysis requires finite values")
    return prediction, target


def bootstrap_score_metrics(
    predictions: Iterable[float],
    targets: Iterable[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile CIs using paired resampling of the fixed test set."""

    prediction, target = _arrays(predictions, targets)
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    score_range = float(target.max() - target.min())
    if score_range <= 0:
        raise ValueError("Target score range must be positive")
    point = aggregate_score_metrics(prediction, target).to_dict()
    generator = np.random.default_rng(seed)
    values = {
        "srcc": [],
        "mae": [],
        "mse": [],
        "relative_l2": [],
        "relative_l2_x100": [],
    }
    for _ in range(resamples):
        indices = generator.integers(0, prediction.size, size=prediction.size)
        sampled_prediction = prediction[indices]
        sampled_target = target[indices]
        residual = sampled_prediction - sampled_target
        correlation = float(
            stats.spearmanr(sampled_prediction, sampled_target).statistic
        )
        if not np.isfinite(correlation):
            continue
        relative_l2 = float(np.mean(np.square(residual / score_range)))
        values["srcc"].append(correlation)
        values["mae"].append(float(np.mean(np.abs(residual))))
        values["mse"].append(float(np.mean(np.square(residual))))
        values["relative_l2"].append(relative_l2)
        values["relative_l2_x100"].append(relative_l2 * 100.0)
    minimum_valid = max(100, int(resamples * 0.95))
    if len(values["srcc"]) < minimum_valid:
        raise ValueError("Too many degenerate bootstrap resamples")
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "paired_percentile_bootstrap",
        "confidence": confidence,
        "requested_resamples": resamples,
        "valid_resamples": len(values["srcc"]),
        "seed": seed,
        "metrics": {
            name: {
                "estimate": float(point[name]),
                "lower": float(np.quantile(samples, alpha)),
                "upper": float(np.quantile(samples, 1.0 - alpha)),
            }
            for name, samples in values.items()
        },
    }


def paired_absolute_error_test(
    predictions_a: Iterable[float],
    predictions_b: Iterable[float],
    targets: Iterable[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Compare A against B using paired absolute-error differences.

    Negative differences favor method A.
    """

    prediction_a, target = _arrays(predictions_a, targets)
    prediction_b, second_target = _arrays(predictions_b, targets)
    if not np.array_equal(target, second_target):
        raise ValueError("Target arrays differ")
    differences = np.abs(prediction_a - target) - np.abs(prediction_b - target)
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(differences, alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        indices = generator.integers(0, differences.size, size=differences.size)
        bootstrap_means[index] = float(differences[indices].mean())
    alpha = (1.0 - confidence) / 2.0
    return {
        "comparison": "absolute_error_a_minus_b",
        "negative_favors": "method_a",
        "sample_count": int(differences.size),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "mean_difference_confidence_interval": {
            "confidence": confidence,
            "lower": float(np.quantile(bootstrap_means, alpha)),
            "upper": float(np.quantile(bootstrap_means, 1.0 - alpha)),
            "resamples": resamples,
            "seed": seed,
        },
        "wilcoxon_signed_rank": {
            "statistic": statistic,
            "two_sided_p_value": p_value,
        },
    }
