from __future__ import annotations

import json
import unittest
from pathlib import Path

from fa_crr_sync.data.manifest import SampleKey, _frame_sort_key


REPO_ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_sample_key_normalization(self) -> None:
        key = SampleKey.from_raw(("Synchro_001", 0))
        self.assertEqual(key.sample_id, "Synchro_001/0")

    def test_numeric_frame_sorting(self) -> None:
        names = ["10.jpg", "2.jpg", "1.jpg"]
        self.assertEqual(
            sorted(names, key=_frame_sort_key), ["1.jpg", "2.jpg", "10.jpg"]
        )

    def test_release_manifest_contract(self) -> None:
        path = REPO_ROOT / "data/manifests/finesynchro_450_50_165.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 665)
        self.assertTrue(all(row["schema_version"] == "2.0" for row in rows))
        self.assertTrue(all("replay1_mask_path" in row for row in rows))
        self.assertEqual(
            sum(not row["replay1_mask_aligned"] for row in rows), 6
        )

    def test_generated_split_is_disjoint_and_complete(self) -> None:
        split_path = REPO_ROOT / "data/splits/finesynchro_450_50_165.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train = set(split["train"])
        validation = set(split["validation"])
        test = set(split["test"])
        self.assertEqual(len(train), 450)
        self.assertEqual(len(validation), 50)
        self.assertEqual(len(test), 165)
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(validation.isdisjoint(test))
        self.assertEqual(len(train | validation | test), 665)


if __name__ == "__main__":
    unittest.main()
