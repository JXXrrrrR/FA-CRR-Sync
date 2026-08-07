"""Load one complete FineSynchro sample through the maintained data path."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fa_crr_sync.data import ClipLoader, FineSynchroIndex  # noqa: E402


def main() -> int:
    index = FineSynchroIndex(REPO_ROOT)
    sample_id = index.sample_ids("train")[0]
    record = index.record(sample_id)
    loader = ClipLoader(output_length=96)
    spatial = loader.spatial_transform(training=False, seed=0)

    live, live_mask = loader.load_aligned_rgb_and_mask(
        index.resolve_sample_path(record, "live_path"),
        index.resolve_sample_path(record, "live_mask_path"),
        spatial,
    )
    replay, replay_mask = loader.load_aligned_rgb_and_mask(
        index.resolve_sample_path(record, "replay1_path"),
        index.resolve_sample_path(record, "replay1_mask_path"),
        spatial,
    )
    reference_id = index.training_reference(sample_id, seed=0, epoch=0)

    result = {
        "sample_id": sample_id,
        "reference_id": reference_id,
        "live_shape": list(live.shape),
        "replay_shape": list(replay.shape),
        "live_mask_shape": list(live_mask.shape),
        "replay_mask_shape": list(replay_mask.shape),
        "live_finite": bool(live.isfinite().all()),
        "replay_finite": bool(replay.isfinite().all()),
        "live_mask_range": [float(live_mask.min()), float(live_mask.max())],
        "replay_mask_range": [
            float(replay_mask.min()),
            float(replay_mask.max()),
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
