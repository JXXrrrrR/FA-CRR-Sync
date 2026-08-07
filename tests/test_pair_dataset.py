from __future__ import annotations

import unittest
from pathlib import Path

from fa_crr_sync.data import FineSynchroIndex, FineSynchroPairDataset


REPO_ROOT = Path(__file__).resolve().parents[1]


class PairDatasetTests(unittest.TestCase):
    def test_pair_ids_follow_action_contract(self) -> None:
        index = FineSynchroIndex(REPO_ROOT)
        dataset = FineSynchroPairDataset(index, split="train", seed=0)
        query_id = dataset.ids[0]
        reference_id = index.training_reference(query_id, seed=0, epoch=0)
        self.assertEqual(
            index.record(query_id)["action_code"],
            index.record(reference_id)["action_code"],
        )


if __name__ == "__main__":
    unittest.main()
