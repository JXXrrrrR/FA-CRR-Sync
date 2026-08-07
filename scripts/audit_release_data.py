"""Audit the sibling FineSynchro release directory against its manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "FineSynchro_Final_Data"
MANIFEST = REPO_ROOT / "data/manifests/finesynchro_450_50_165.jsonl"
REPORT = REPO_ROOT / "data/reports/release_data_audit.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def frames(path: Path) -> list[Path]:
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_counts = Counter(str(row["split"]) for row in rows)
    totals = Counter()
    suffixes: dict[str, Counter[str]] = {
        name: Counter() for name in ("live", "replay1", "live_mask", "replay1_mask")
    }
    missing: list[str] = []
    alignment_failures: list[dict[str, str]] = []
    fields = {
        "live": "live_path",
        "replay1": "replay1_path",
        "live_mask": "live_mask_path",
        "replay1_mask": "replay1_mask_path",
    }
    for row in rows:
        sample_id = str(row["sample_id"])
        indexed: dict[str, list[Path]] = {}
        for name, field in fields.items():
            path = DATA_ROOT / str(row[field])
            if not path.is_dir():
                missing.append(f"{sample_id}:{field}")
                indexed[name] = []
                continue
            indexed[name] = frames(path)
            totals[name] += len(indexed[name])
            suffixes[name].update(item.suffix.lower() for item in indexed[name])
        for rgb_name, mask_name in (("live", "live_mask"), ("replay1", "replay1_mask")):
            rgb_names = [item.name for item in indexed[rgb_name]]
            mask_names = [item.name for item in indexed[mask_name]]
            if rgb_names != mask_names:
                alignment_failures.append(
                    {"sample_id": sample_id, "pair": f"{rgb_name}:{mask_name}"}
                )

    report = {
        "schema_version": "1.0",
        "manifest_sha256": sha256(MANIFEST),
        "samples": len(rows),
        "split_counts": dict(split_counts),
        "frame_counts": dict(totals),
        "suffix_counts": {name: dict(counts) for name, counts in suffixes.items()},
        "missing_paths": missing,
        "alignment_failures": alignment_failures,
        "release_ready": not missing and not alignment_failures,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metadata_report = DATA_ROOT / "metadata/release_data_audit.json"
    metadata_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["release_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
