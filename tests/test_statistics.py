from __future__ import annotations

import unittest

import numpy as np

from fa_crr_sync.evaluation import (
    bootstrap_score_metrics,
    paired_absolute_error_test,
)


class StatisticalAnalysisTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_contains_point_estimate(self) -> None:
        targets = np.arange(1.0, 21.0)
        predictions = targets + np.linspace(-1.0, 1.0, 20)
        first = bootstrap_score_metrics(
            predictions, targets, resamples=500, seed=7
        )
        second = bootstrap_score_metrics(
            predictions, targets, resamples=500, seed=7
        )
        self.assertEqual(first, second)
        mae = first["metrics"]["mae"]
        self.assertLessEqual(mae["lower"], mae["estimate"])
        self.assertGreaterEqual(mae["upper"], mae["estimate"])

    def test_paired_test_direction_is_explicit(self) -> None:
        targets = np.arange(20.0)
        method_a = targets + 0.1
        method_b = targets + 2.0
        result = paired_absolute_error_test(
            method_a, method_b, targets, resamples=500, seed=0
        )
        self.assertLess(result["mean_difference"], 0.0)
        self.assertEqual(result["negative_favors"], "method_a")
        self.assertLess(
            result["wilcoxon_signed_rank"]["two_sided_p_value"], 0.01
        )

    def test_identical_methods_return_unit_p_value(self) -> None:
        targets = [1.0, 2.0, 3.0]
        result = paired_absolute_error_test(
            targets, targets, targets, resamples=100
        )
        self.assertEqual(
            result["wilcoxon_signed_rank"]["two_sided_p_value"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
