"""Generate paper-ready FineSynchro split and annotation statistics."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fa_crr_sync.data import FineSynchroIndex  # noqa: E402


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def main() -> int:
    index = FineSynchroIndex(REPO_ROOT)
    records = list(index.records.values())
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "finesynchro_450_50_165_seed0_release",
        "sample_count": len(records),
        "split_counts": dict(Counter(str(row["split"]) for row in records)),
        "source_group_count": len({str(row["source_group"]) for row in records}),
        "action_code_count": len({str(row["action_code"]) for row in records}),
        "by_split": {},
    }
    for split in ("train", "validation", "test", "all"):
        selected = (
            records
            if split == "all"
            else [row for row in records if row["split"] == split]
        )
        report["by_split"][split] = {
            "samples": len(selected),
            "action_codes": len({str(row["action_code"]) for row in selected}),
            "source_groups": len({str(row["source_group"]) for row in selected}),
            "score": numeric_summary(
                [float(row["dive_score"]) for row in selected]
            ),
            "difficulty": numeric_summary(
                [float(row["difficulty"]) for row in selected]
            ),
            "live_frames": numeric_summary(
                [float(row["live_frames"]) for row in selected]
            ),
            "replay1_frames": numeric_summary(
                [float(row["replay1_frames"]) for row in selected]
            ),
        }

    action_split: dict[str, Counter[str]] = defaultdict(Counter)
    source_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        split = str(row["split"])
        action_split[str(row["action_code"])][split] += 1
        source_split[str(row["source_group"])][split] += 1

    train_only = sorted(
        action for action, counts in action_split.items() if counts["test"] == 0
    )
    test_only = sorted(
        action for action, counts in action_split.items() if counts["train"] == 0
    )
    singleton = sorted(
        action for action, counts in action_split.items()
        if counts["train"] + counts["test"] == 1
    )
    report["action_coverage"] = {
        "train_only": train_only,
        "test_only": test_only,
        "singletons": singleton,
        "all_test_actions_have_training_references": not test_only,
    }
    frequency_bins = Counter()
    for counts in action_split.values():
        total = counts["train"] + counts["test"]
        label = (
            "singleton"
            if total == 1
            else "rare_2_9"
            if total < 10
            else "medium_10_29"
            if total < 30
            else "common_30_plus"
        )
        frequency_bins[label] += 1
    report["action_frequency_bins"] = dict(frequency_bins)
    train_records = [row for row in records if row["split"] == "train"]
    test_records = [row for row in records if row["split"] == "test"]
    shared_groups = sorted(
        {str(row["source_group"]) for row in train_records}
        & {str(row["source_group"]) for row in test_records}
    )
    report["source_group_overlap"] = {
        "shared_count": len(shared_groups),
        "shared_groups": shared_groups,
        "interpretation": (
            "All source groups occur in both splits; this is a sample-random "
            "protocol and not a source-video-disjoint evaluation."
        ),
    }
    distribution_checks = {}
    for field in ("dive_score", "difficulty"):
        train_values = np.asarray(
            [float(row[field]) for row in train_records], dtype=np.float64
        )
        test_values = np.asarray(
            [float(row[field]) for row in test_records], dtype=np.float64
        )
        ks = stats.ks_2samp(train_values, test_values)
        pooled = np.sqrt(
            (
                (train_values.size - 1) * train_values.var(ddof=1)
                + (test_values.size - 1) * test_values.var(ddof=1)
            )
            / (train_values.size + test_values.size - 2)
        )
        distribution_checks[field] = {
            "ks_statistic": float(ks.statistic),
            "ks_two_sided_p_value": float(ks.pvalue),
            "cohens_d_train_minus_test": float(
                (train_values.mean() - test_values.mean()) / pooled
            ),
        }
    report["train_test_distribution_checks"] = distribution_checks

    report_path = REPO_ROOT / "data" / "reports" / "dataset_statistics.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    table_dir = REPO_ROOT / "data" / "tables"
    table_dir.mkdir(exist_ok=True)
    with (table_dir / "action_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["action_code", "train", "test", "total"])
        for action in sorted(action_split):
            counts = action_split[action]
            writer.writerow(
                [action, counts["train"], counts["test"], sum(counts.values())]
            )
    with (table_dir / "source_group_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_group", "train", "test", "total"])
        for group in sorted(source_split):
            counts = source_split[group]
            writer.writerow(
                [group, counts["train"], counts["test"], sum(counts.values())]
            )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
