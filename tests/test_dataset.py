from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from fa_crr_sync.data import (
    EpochShuffleSampler,
    FineSynchroIndex,
    sampled_stage_boundaries,
    uniform_frame_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = FineSynchroIndex(REPO_ROOT)

    def test_uniform_sampling_matches_legacy_flooring(self) -> None:
        sampled = uniform_frame_indices(frame_count=112, output_length=96)
        self.assertEqual(sampled.dtype, np.int64)
        self.assertEqual(int(sampled[0]), 0)
        self.assertEqual(int(sampled[-1]), 111)
        self.assertEqual(len(sampled), 96)

    def test_split_counts(self) -> None:
        self.assertEqual(len(self.index.sample_ids("train")), 450)
        self.assertEqual(len(self.index.sample_ids("validation")), 50)
        self.assertEqual(len(self.index.sample_ids("test")), 165)

    def test_sampled_stage_boundaries(self) -> None:
        record = self.index.record("Synchro_001/0")
        boundaries = sampled_stage_boundaries(record["phase_labels"])
        self.assertEqual(tuple(boundaries.shape), (2,))
        self.assertGreater(int(boundaries[0]), 0)
        self.assertGreater(int(boundaries[1]), int(boundaries[0]))
        self.assertLess(int(boundaries[1]), 96)

    def test_training_reference_is_deterministic(self) -> None:
        sample_id = "Synchro_001/0"
        first = self.index.training_reference(sample_id, seed=7, epoch=3)
        second = self.index.training_reference(sample_id, seed=7, epoch=3)
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            self.index.training_reference(sample_id, seed=7, epoch=99),
        )
        self.assertNotEqual(first, sample_id)
        self.assertEqual(
            self.index.record(first)["action_code"],
            self.index.record(sample_id)["action_code"],
        )

    def test_test_voters_are_training_samples_of_the_same_action(self) -> None:
        sample_id = self.index.sample_ids("test")[0]
        references = self.index.test_references(sample_id, voter_number=10, seed=0)
        action = self.index.record(sample_id)["action_code"]
        self.assertGreater(len(references), 0)
        self.assertLessEqual(len(references), 10)
        for reference in references:
            self.assertEqual(self.index.record(reference)["split"], "train")
            self.assertEqual(self.index.record(reference)["action_code"], action)

    def test_epoch_sampler_is_deterministic_and_epoch_aware(self) -> None:
        sampler = EpochShuffleSampler(dataset_length=10, seed=7)
        epoch_zero = list(sampler)
        self.assertEqual(epoch_zero, list(sampler))
        self.assertTrue(all(epoch == 0 for _, epoch in epoch_zero))
        sampler.set_epoch(1)
        epoch_one = list(sampler)
        self.assertNotEqual(
            [index for index, _ in epoch_zero],
            [index for index, _ in epoch_one],
        )
        self.assertTrue(all(epoch == 1 for _, epoch in epoch_one))


if __name__ == "__main__":
    unittest.main()
