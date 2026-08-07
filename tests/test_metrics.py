from __future__ import annotations

import unittest

from fa_crr_sync.evaluation import (
    aggregate_score_metrics,
    temporal_iou,
    temporal_iou_accuracy,
)


class MetricTests(unittest.TestCase):
    def test_perfect_score_metrics(self) -> None:
        metrics = aggregate_score_metrics([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(metrics.srcc, 1.0)
        self.assertAlmostEqual(metrics.mae, 0.0)
        self.assertAlmostEqual(metrics.mse, 0.0)
        self.assertAlmostEqual(metrics.relative_l2_x100, 0.0)

    def test_relative_l2_matches_historical_formula(self) -> None:
        metrics = aggregate_score_metrics([2, 3, 4], [1, 3, 5])
        expected = (((1 / 4) ** 2) + 0 + ((-1 / 4) ** 2)) / 3
        self.assertAlmostEqual(metrics.relative_l2, expected)
        self.assertAlmostEqual(metrics.relative_l2_x100, expected * 100)

    def test_temporal_iou(self) -> None:
        self.assertAlmostEqual(temporal_iou([10, 30], [10, 30]), 1.0)
        self.assertAlmostEqual(temporal_iou([10, 30], [20, 40]), 1 / 3)

    def test_temporal_iou_accuracy(self) -> None:
        accuracy = temporal_iou_accuracy(
            [[10, 30], [10, 30]], [[10, 30], [20, 40]], threshold=0.5
        )
        self.assertAlmostEqual(accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
