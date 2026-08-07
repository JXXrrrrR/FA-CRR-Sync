"""Explicit mappings from historical checkpoint keys to maintained modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .crr_sync_core import CRRSyncCore


def _strip_parallel_prefix(key: str) -> str:
    return key.removeprefix("module.")


def map_procedure_key(key: str) -> str:
    key = _strip_parallel_prefix(key)
    if key.startswith("inc.conv.conv."):
        return key.replace("inc.conv.conv.", "input_block.layers.", 1)
    for index in range(1, 5):
        prefix = f"down{index}.mpconv.1.conv."
        if key.startswith(prefix):
            return key.replace(prefix, f"down{index}.layers.1.layers.", 1)
    head_mapping = {
        "tas.layer1.": "transition_head.layers.0.",
        "tas.layer2.": "transition_head.layers.2.",
        "tas.layer3.": "transition_head.layers.4.",
    }
    for source, target in head_mapping.items():
        if key.startswith(source):
            return key.replace(source, target, 1)
    return key


def map_decoder_key(key: str) -> str:
    key = _strip_parallel_prefix(key)
    replacements = (
        ("model.", "layers."),
        (".norm_q.", ".query_norm."),
        (".norm_v.", ".value_norm."),
        (".attn.q_map.", ".attention.query."),
        (".attn.k_map.", ".attention.key."),
        (".attn.v_map.", ".attention.value."),
        (".attn.proj.", ".attention.projection."),
        (".norm2.", ".output_norm."),
        (".mlp.fc1.", ".feed_forward.layers.0."),
        (".mlp.fc2.", ".feed_forward.layers.3."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def map_regressor_key(key: str) -> str:
    key = _strip_parallel_prefix(key)
    for source, target in {
        "layer1.": "layers.0.",
        "layer2.": "layers.2.",
        "layer3.": "layers.4.",
    }.items():
        if key.startswith(source):
            return key.replace(source, target, 1)
    return key


def _mapped_state(
    state: Mapping[str, torch.Tensor], mapper: Any
) -> dict[str, torch.Tensor]:
    mapped = {}
    for key, value in state.items():
        new_key = mapper(key)
        if new_key in mapped:
            raise ValueError(f"Historical keys collide at {new_key!r}")
        mapped[new_key] = value
    return mapped


def load_historical_core_state(
    core: CRRSyncCore, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Load compatible non-backbone CRR-Sync components strictly."""

    components = {
        "procedure": (
            core.procedure,
            checkpoint["psnet_model"],
            map_procedure_key,
        ),
        "decoder": (core.decoder, checkpoint["decoder"], map_decoder_key),
        "score_regressor": (
            core.score_regressor,
            checkpoint["regressor_delta"],
            map_regressor_key,
        ),
        "synchronisation_regressor": (
            core.synchronisation_regressor,
            checkpoint["regressor_synchro_delta"],
            map_regressor_key,
        ),
    }
    report = {}
    for name, (module, historical_state, mapper) in components.items():
        mapped = _mapped_state(historical_state, mapper)
        incompatible = module.load_state_dict(mapped, strict=True)
        report[name] = {
            "historical_tensors": len(historical_state),
            "mapped_tensors": len(mapped),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    report["unused_checkpoint_components"] = sorted(
        set(checkpoint)
        - {
            "psnet_model",
            "decoder",
            "regressor_delta",
            "regressor_synchro_delta",
        }
    )
    return report
