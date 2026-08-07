from __future__ import annotations

import unittest

from fa_crr_sync.models import I3DBackbone


class I3DContractTests(unittest.TestCase):
    def test_snippet_contract(self) -> None:
        backbone = I3DBackbone()
        self.assertEqual(
            backbone.snippet_starts, (0, 10, 20, 30, 40, 50, 60, 70, 80)
        )
        self.assertEqual(len(backbone.state_dict()), 344)


if __name__ == "__main__":
    unittest.main()
