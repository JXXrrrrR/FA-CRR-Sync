from __future__ import annotations

import unittest

from fa_crr_sync.models.compatibility import (
    map_decoder_key,
    map_procedure_key,
    map_regressor_key,
)


class CheckpointMappingTests(unittest.TestCase):
    def test_procedure_mapping(self) -> None:
        self.assertEqual(
            map_procedure_key("inc.conv.conv.0.weight"),
            "input_block.layers.0.weight",
        )
        self.assertEqual(
            map_procedure_key("down2.mpconv.1.conv.4.running_mean"),
            "down2.layers.1.layers.4.running_mean",
        )
        self.assertEqual(
            map_procedure_key("tas.layer3.bias"),
            "transition_head.layers.4.bias",
        )

    def test_decoder_mapping(self) -> None:
        self.assertEqual(
            map_decoder_key("model.1.attn.q_map.weight"),
            "layers.1.attention.query.weight",
        )
        self.assertEqual(
            map_decoder_key("model.2.mlp.fc2.bias"),
            "layers.2.feed_forward.layers.3.bias",
        )

    def test_regressor_mapping(self) -> None:
        self.assertEqual(
            map_regressor_key("layer2.weight"), "layers.2.weight"
        )


if __name__ == "__main__":
    unittest.main()
