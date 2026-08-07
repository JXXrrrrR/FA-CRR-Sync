"""Build a frozen, auditable FineSynchro manifest from the original annotations."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = "1.0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, order=True)
class SampleKey:
    """Canonical FineSynchro sample identifier."""

    source_group: str
    sequence_id: int

    @property
    def sample_id(self) -> str:
        return f"{self.source_group}/{self.sequence_id}"

    @classmethod
    def from_raw(cls, value: tuple[Any, Any]) -> "SampleKey":
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(f"Expected a two-item sample tuple, got {value!r}")
        return cls(str(value[0]), int(value[1]))


@dataclass(frozen=True)
class FrameIndex:
    """Sorted frame metadata for one sample directory."""

    relative_path: str
    names: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def first(self) -> str | None:
        return self.names[0] if self.names else None

    @property
    def last(self) -> str | None:
        return self.names[-1] if self.names else None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(Path(name).stem), name
    except ValueError:
        return 2**63 - 1, name


def _index_frames(root: Path, key: SampleKey, data_root: Path) -> FrameIndex:
    sample_dir = root / key.source_group / str(key.sequence_id)
    relative_path = sample_dir.relative_to(data_root).as_posix()
    if not sample_dir.is_dir():
        return FrameIndex(relative_path=relative_path, names=())

    names = tuple(
        sorted(
            (
                entry.name
                for entry in sample_dir.iterdir()
                if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=_frame_sort_key,
        )
    )
    return FrameIndex(relative_path=relative_path, names=names)


def _normalise_mapping(raw: Mapping[tuple[Any, Any], Any]) -> dict[SampleKey, Any]:
    result: dict[SampleKey, Any] = {}
    for raw_key, value in raw.items():
        key = SampleKey.from_raw(raw_key)
        if key in result:
            raise ValueError(f"Duplicate normalized sample key: {key.sample_id}")
        result[key] = value
    return result


def _normalise_split(raw: Iterable[tuple[Any, Any]]) -> list[SampleKey]:
    return [SampleKey.from_raw(item) for item in raw]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(dict(row)), ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty manifest")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else _jsonable(value)
                    for key, value in row.items()
                }
            )


def _resolve_paths(
    repo_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    data_root = (repo_root / dataset_config["root"]).resolve()
    paths = {
        "data_root": data_root,
        "annotation_root": data_root / dataset_config["annotation_root"],
        "live_root": data_root / dataset_config["live_root"],
        "replay1_root": data_root / dataset_config["replay1_root"],
        "replay1_alternate_root": data_root / dataset_config["replay1_alternate_root"],
        "live_mask_root": data_root / dataset_config["live_mask_root"],
        "replay_mask_root": data_root / dataset_config["replay_mask_root"],
        "replay2_root": data_root / dataset_config["replay2_root"],
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Configured data paths do not exist: {details}")
    return config, paths


def build_finesynchro_manifest(
    repo_root: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Generate manifests, hashes, frozen splits, and an integrity report."""

    repo_root = repo_root.resolve()
    config_path = (
        config_path.resolve()
        if config_path is not None
        else repo_root / "configs" / "data" / "finesynchro.yaml"
    )
    config, paths = _resolve_paths(repo_root, config_path)
    annotation_root = paths["annotation_root"]
    fine_root = annotation_root / "fine-grained_anno"

    source_files = {
        "coarse_annotation": annotation_root
        / "coarse_anno"
        / "coarse_annotation_Synchro.pkl",
        "fine_annotation": fine_root / "fine-grained_annotation_Synchro.pkl",
        "train_split": fine_root / "train_split_Synchro.pkl",
        "test_split": fine_root / "test_split_Synchro.pkl",
    }
    missing_source_files = [
        name for name, path in source_files.items() if not path.is_file()
    ]
    if missing_source_files:
        raise FileNotFoundError(f"Missing annotation files: {missing_source_files}")

    coarse = _normalise_mapping(_load_pickle(source_files["coarse_annotation"]))
    fine = _normalise_mapping(_load_pickle(source_files["fine_annotation"]))
    train = _normalise_split(_load_pickle(source_files["train_split"]))
    test = _normalise_split(_load_pickle(source_files["test_split"]))

    coarse_keys = set(coarse)
    fine_keys = set(fine)
    train_keys = set(train)
    test_keys = set(test)
    all_split_keys = train_keys | test_keys

    global_issues: list[str] = []
    if coarse_keys != fine_keys:
        global_issues.append("coarse_fine_key_mismatch")
    if train_keys & test_keys:
        global_issues.append("train_test_overlap")
    if all_split_keys != coarse_keys:
        global_issues.append("split_annotation_key_mismatch")
    if len(train) != len(train_keys):
        global_issues.append("duplicate_train_keys")
    if len(test) != len(test_keys):
        global_issues.append("duplicate_test_keys")

    split_by_key = {key: "train" for key in train}
    split_by_key.update({key: "test" for key in test})

    rows: list[dict[str, Any]] = []
    issue_counter: Counter[str] = Counter()
    action_counter: dict[str, Counter[str]] = defaultdict(Counter)
    source_counter: dict[str, Counter[str]] = defaultdict(Counter)
    replay_conflicts: list[str] = []
    replay2_samples = 0

    for key in sorted(coarse_keys):
        coarse_item = coarse[key]
        fine_item = fine[key]
        live = _index_frames(paths["live_root"], key, paths["data_root"])
        replay1 = _index_frames(paths["replay1_root"], key, paths["data_root"])
        replay1_alternate = _index_frames(
            paths["replay1_alternate_root"], key, paths["data_root"]
        )
        live_mask = _index_frames(paths["live_mask_root"], key, paths["data_root"])
        replay_mask = _index_frames(paths["replay_mask_root"], key, paths["data_root"])
        replay2 = _index_frames(paths["replay2_root"], key, paths["data_root"])

        issues: list[str] = []
        action_code = str(coarse_item["action_type"])
        fine_action = str(fine_item[0])
        score = float(coarse_item["dive_score"])
        fine_score = float(fine_item[1])
        difficulty = float(coarse_item["difficulty"])
        fine_difficulty = float(fine_item[2])
        fine_transitions = [int(value) for value in fine_item[3]]
        frame_labels = [int(value) for value in fine_item[4]]
        expected_live_count = int(coarse_item["end_frame"]) - int(
            coarse_item["start_frame"]
        ) + 1

        if fine_action != action_code:
            issues.append("action_mismatch")
        if not np.isclose(score, fine_score):
            issues.append("score_mismatch")
        if not np.isclose(difficulty, fine_difficulty):
            issues.append("difficulty_mismatch")
        if live.count == 0:
            issues.append("missing_live")
        if replay1.count == 0:
            issues.append("missing_replay1")
        if live_mask.count == 0:
            issues.append("missing_live_mask")
        if live.count != expected_live_count:
            issues.append("live_count_vs_coarse_boundary")
        if live.count != len(frame_labels):
            issues.append("live_count_vs_fine_labels")
        if live.names != live_mask.names:
            issues.append("live_mask_name_mismatch")
        if replay1.names != replay1_alternate.names:
            issues.append("alternate_replay1_conflict")
            replay_conflicts.append(key.sample_id)
        if replay1.names != replay_mask.names:
            issues.append("canonical_replay_mask_mismatch")
        if len(fine_transitions) < 2:
            issues.append("insufficient_transition_count")
        elif not (
            all(
                left < right
                for left, right in zip(fine_transitions, fine_transitions[1:])
            )
            and 0 <= fine_transitions[0]
            and fine_transitions[-1] < max(live.count, 1)
        ):
            issues.append("invalid_transition_order")

        replay2_available = replay2.count > 0
        replay2_samples += int(replay2_available)
        split = split_by_key.get(key, "unassigned")
        action_counter[split][action_code] += 1
        source_counter[split][key.source_group] += 1
        issue_counter.update(issues)

        row = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": key.sample_id,
            "source_group": key.source_group,
            "sequence_id": key.sequence_id,
            "split": split,
            "action_code": action_code,
            "dive_score": score,
            "difficulty": difficulty,
            "start_frame": int(coarse_item["start_frame"]),
            "end_frame": int(coarse_item["end_frame"]),
            # CRR-Sync collapses the fine semantic runs to three macro stages
            # by retaining the first and last annotated transition.
            "macro_transition_1": fine_transitions[0]
            if len(fine_transitions) >= 1
            else None,
            "macro_transition_2": fine_transitions[-1]
            if len(fine_transitions) >= 2
            else None,
            "fine_transition_count": len(fine_transitions),
            "fine_transitions": fine_transitions,
            "phase_label_count": len(frame_labels),
            "phase_labels": frame_labels,
            "judge_scores": [float(value) for value in coarse_item["judge_scores"]],
            "execution_scores": [
                float(value) for value in coarse_item["execution_score"]
            ],
            "synchronisation_scores": [
                float(value) for value in coarse_item["synchronisation_score"]
            ],
            "live_path": live.relative_path,
            "live_frames": live.count,
            "live_first_frame": live.first,
            "live_last_frame": live.last,
            "replay1_path": replay1.relative_path,
            "replay1_frames": replay1.count,
            "replay1_first_frame": replay1.first,
            "replay1_last_frame": replay1.last,
            "live_mask_path": live_mask.relative_path,
            "live_mask_frames": live_mask.count,
            "live_mask_aligned": live.names == live_mask.names,
            "replay2_path": replay2.relative_path if replay2_available else None,
            "replay2_frames": replay2.count,
            "replay2_available": replay2_available,
            "mask_provenance": config["dataset"]["mask_provenance"],
            "issues": issues,
        }
        rows.append(row)

    output_stem = "finesynchro_original_500_165"
    manifest_dir = repo_root / "data" / "manifests"
    _write_jsonl(manifest_dir / f"{output_stem}.jsonl", rows)
    _write_csv(manifest_dir / f"{output_stem}.csv", rows)

    frozen_split = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "original_500_165",
        "train": [key.sample_id for key in train],
        "test": [key.sample_id for key in test],
    }
    _write_json(repo_root / "data" / "splits" / "original_500_165.json", frozen_split)

    hashes = {
        name: {
            "path": path.relative_to(paths["data_root"]).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in source_files.items()
    }
    _write_json(repo_root / "data" / "hashes" / "annotation_sha256.json", hashes)

    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": config["protocol"],
        "counts": {
            "annotations": len(coarse),
            "train": len(train),
            "test": len(test),
            "actions": len({row["action_code"] for row in rows}),
            "source_groups": len({row["source_group"] for row in rows}),
            "live_frames": sum(row["live_frames"] for row in rows),
            "replay1_frames": sum(row["replay1_frames"] for row in rows),
            "live_mask_frames": sum(row["live_mask_frames"] for row in rows),
            "replay2_available_samples": replay2_samples,
            "replay2_frames": sum(row["replay2_frames"] for row in rows),
        },
        "global_issues": global_issues,
        "sample_issue_counts": dict(sorted(issue_counter.items())),
        "samples_with_any_issue": sum(bool(row["issues"]) for row in rows),
        "alternate_replay1_conflicts": replay_conflicts,
        "action_distribution": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(action_counter.items())
        },
        "source_group_distribution": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(source_counter.items())
        },
        "hashes": hashes,
    }
    _write_json(
        repo_root / "data" / "reports" / "finesynchro_integrity.json", report
    )
    return report
