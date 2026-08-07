"""Audit a deterministic sample of existing Live foreground masks on CPU."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fa_crr_sync.data import FineSynchroIndex  # noqa: E402
from fa_crr_sync.data.dataset import (  # noqa: E402
    sorted_frame_paths,
    uniform_frame_indices,
)


FRAMES_PER_SAMPLE = 10
FOREGROUND_THRESHOLD = 127


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def main() -> int:
    index = FineSynchroIndex(REPO_ROOT)
    area_ratios: list[float] = []
    component_counts: list[int] = []
    heuristic_flags: Counter[str] = Counter()
    flagged_samples: dict[str, set[str]] = {}
    dimension_mismatches = []
    non_binary_frames = 0

    for sample_id in sorted(index.records):
        record = index.record(sample_id)
        mask_paths = sorted_frame_paths(
            index.resolve_sample_path(record, "live_mask_path")
        )
        live_paths = sorted_frame_paths(
            index.resolve_sample_path(record, "live_path")
        )
        positions = uniform_frame_indices(len(mask_paths), FRAMES_PER_SAMPLE)
        sample_flags: set[str] = set()
        for position in positions:
            mask_path = mask_paths[int(position)]
            live_path = live_paths[int(position)]
            with Image.open(mask_path) as image:
                mask_image = image.convert("L")
                mask = np.asarray(mask_image)
                mask_size = mask_image.size
            with Image.open(live_path) as live_image:
                live_size = live_image.size
            if mask_size != live_size:
                dimension_mismatches.append(
                    {
                        "sample_id": sample_id,
                        "frame": mask_path.name,
                        "mask_size": mask_size,
                        "live_size": live_size,
                    }
                )
            unique = np.unique(mask)
            if not set(unique.tolist()).issubset({0, 255}):
                non_binary_frames += 1
            foreground = mask > FOREGROUND_THRESHOLD
            area = float(foreground.mean())
            area_ratios.append(area)
            labels, count = ndimage.label(foreground)
            if count:
                sizes = np.bincount(labels.reshape(-1))[1:]
                count = int(
                    np.sum(sizes >= max(1, int(foreground.size * 0.0005)))
                )
            component_counts.append(count)
            touches_border = bool(
                foreground[0].any()
                or foreground[-1].any()
                or foreground[:, 0].any()
                or foreground[:, -1].any()
            )
            flags = []
            if area < 0.001:
                flags.append("near_empty")
            if area > 0.20:
                flags.append("oversized")
            if touches_border:
                flags.append("touches_border")
            if count == 0:
                flags.append("no_component")
            if count > 4:
                flags.append("fragmented_more_than_four_components")
            for flag in flags:
                heuristic_flags[flag] += 1
                sample_flags.add(flag)
        if sample_flags:
            flagged_samples[sample_id] = sample_flags

    report = {
        "schema_version": "1.0",
        "mask_source": "segmentation_output",
        "sampling": {
            "method": "uniform over every sample",
            "samples": len(index.records),
            "frames_per_sample": FRAMES_PER_SAMPLE,
            "audited_frames": len(area_ratios),
            "foreground_threshold": FOREGROUND_THRESHOLD,
        },
        "foreground_area_ratio": distribution(area_ratios),
        "large_component_count": {
            **distribution([float(value) for value in component_counts]),
            "histogram": {
                str(key): value
                for key, value in sorted(Counter(component_counts).items())
            },
        },
        "non_binary_frames": non_binary_frames,
        "dimension_mismatch_count": len(dimension_mismatches),
        "dimension_mismatches": dimension_mismatches,
        "heuristic_flag_frame_counts": dict(heuristic_flags),
        "heuristic_flag_sample_counts": dict(
            Counter(
                flag
                for flags in flagged_samples.values()
                for flag in flags
            )
        ),
        "samples_with_any_heuristic_flag": len(flagged_samples),
        "flagged_samples": {
            sample_id: sorted(flags)
            for sample_id, flags in sorted(flagged_samples.items())
        },
        "interpretation": "Automatic mask statistics.",
    }
    report_path = REPO_ROOT / "data" / "reports" / "mask_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
