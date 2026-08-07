"""Build the FineSynchro 450/50/165 manifest and split."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "data/manifests/finesynchro_original_500_165.jsonl"
OUTPUT_MANIFEST = REPO_ROOT / "data/manifests/finesynchro_450_50_165.jsonl"
OUTPUT_CSV = REPO_ROOT / "data/manifests/finesynchro_450_50_165.csv"
OUTPUT_SPLIT = REPO_ROOT / "data/splits/finesynchro_450_50_165.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    rows = [
        json.loads(line)
        for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    original_train = [row for row in rows if row["split"] == "train"]
    counts = Counter(str(row["action_code"]) for row in original_train)
    eligible = [
        str(row["sample_id"])
        for row in original_train
        if counts[str(row["action_code"])] > 1
    ]
    random.Random(0).shuffle(eligible)
    validation_ids = set(eligible[:50])
    if len(validation_ids) != 50:
        raise RuntimeError("Unable to construct a 50-sample validation subset")

    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in validation_ids:
            row["split"] = "validation"
        row["schema_version"] = "2.0"
        row["live_path"] = f"rgb/live/{sample_id}"
        row["replay1_path"] = f"rgb/replay1/{sample_id}"
        row["live_mask_path"] = f"masks/live/{sample_id}"
        row["replay1_mask_path"] = f"masks/replay1/{sample_id}"
        row["mask_role"] = "predicted_input"
        row["replay1_mask_aligned"] = sample_id not in {
            "Synchro_006/1",
            "Synchro_006/11",
            "Synchro_006/12",
            "Synchro_006/13",
            "Synchro_014/7",
            "Synchro_017/55",
        }

    counts_by_split = Counter(str(row["split"]) for row in rows)
    if counts_by_split != {"train": 450, "validation": 50, "test": 165}:
        raise RuntimeError(f"Unexpected split counts: {counts_by_split}")

    OUTPUT_MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    fieldnames = list(rows[0])
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
    split = {
        "schema_version": "2.0",
        "protocol": "finesynchro_450_50_165_seed0_release",
        "split_generation": "seed_0",
        "seed": 0,
        "train": sorted(
            str(row["sample_id"]) for row in rows if row["split"] == "train"
        ),
        "validation": sorted(validation_ids),
        "test": sorted(
            str(row["sample_id"]) for row in rows if row["split"] == "test"
        ),
    }
    OUTPUT_SPLIT.write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "counts": counts_by_split,
                "manifest_sha256": sha256(OUTPUT_MANIFEST),
                "split_sha256": sha256(OUTPUT_SPLIT),
            },
            default=dict,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
