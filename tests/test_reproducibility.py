from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from fa_crr_sync.utils import seed_everything


class ReproducibilityTests(unittest.TestCase):
    def test_seed_everything_repeats_all_rngs(self) -> None:
        seed_everything(17)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        seed_everything(17)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
