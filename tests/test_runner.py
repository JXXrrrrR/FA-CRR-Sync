from __future__ import annotations

import unittest

from fa_crr_sync.training.runner import (
    aggregate_voter_predictions,
    compute_evaluation_metrics,
)


def voter_row(
    query_id: str,
    reference_id: str,
    target: float,
    prediction: float,
    boundaries: list[float],
) -> dict:
    return {
        "query_id": query_id,
        "reference_id": reference_id,
        "action_code": "101",
        "target_score": target,
        "reference_score": 80.0,
        "score_prediction": prediction,
        "target_synchronisation": [8.0, 8.0, 8.0],
        "predicted_synchronisation": [7.0, 8.0, 9.0],
        "target_boundaries": [30.0, 60.0],
        "predicted_boundaries": boundaries,
    }


class RunnerTests(unittest.TestCase):
    def test_voter_aggregation_is_query_ordered_and_auditable(self) -> None:
        rows = [
            voter_row("q2", "r1", 90.0, 88.0, [30.0, 60.0]),
            voter_row("q1", "r2", 70.0, 72.0, [30.0, 60.0]),
            voter_row("q1", "r3", 70.0, 68.0, [30.0, 60.0]),
            voter_row("q2", "r4", 90.0, 92.0, [30.0, 60.0]),
        ]
        aggregated = aggregate_voter_predictions(rows)
        self.assertEqual([row["query_id"] for row in aggregated], ["q1", "q2"])
        self.assertEqual([row["prediction"] for row in aggregated], [70.0, 90.0])
        self.assertEqual(aggregated[0]["reference_ids"], ["r2", "r3"])
        metrics = compute_evaluation_metrics(aggregated)
        self.assertAlmostEqual(metrics["srcc"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["aiou_0_75"], 1.0)

    def test_inconsistent_targets_are_rejected(self) -> None:
        rows = [
            voter_row("q1", "r1", 70.0, 70.0, [30.0, 60.0]),
            voter_row("q1", "r2", 71.0, 70.0, [30.0, 60.0]),
        ]
        with self.assertRaisesRegex(ValueError, "Inconsistent target"):
            aggregate_voter_predictions(rows)


if __name__ == "__main__":
    unittest.main()
