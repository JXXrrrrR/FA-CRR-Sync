"""Deterministic FineSynchro indexing and video-frame loading."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional
from torch.utils.data import Dataset, Sampler


RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def uniform_frame_indices(frame_count: int, output_length: int) -> np.ndarray:
    """Match the legacy linspace sampling without assuming contiguous filenames."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    return np.linspace(0, frame_count - 1, output_length, dtype=np.int64)


def sampled_stage_boundaries(
    phase_labels: Iterable[int], output_length: int = 96
) -> np.ndarray:
    """Reproduce the legacy two macro-boundary targets after frame sampling."""

    labels = np.asarray(list(phase_labels), dtype=np.int64).reshape(-1)
    indices = uniform_frame_indices(len(labels), output_length)
    sampled = labels[indices]
    first_occurrences = []
    seen: set[int] = set()
    for index, label in enumerate(sampled):
        value = int(label)
        if value not in seen:
            seen.add(value)
            first_occurrences.append(index)
    if len(first_occurrences) < 3:
        raise ValueError(
            "At least three ordered semantic runs are required for macro stages"
        )
    return np.asarray(
        [first_occurrences[1], first_occurrences[-1]], dtype=np.int64
    )


def _frame_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**63 - 1, path.name


def sorted_frame_paths(sample_dir: Path) -> list[Path]:
    """Return numerically sorted image files from a sample directory."""

    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {sample_dir}")
    paths = sorted(
        (
            path
            for path in sample_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=_frame_sort_key,
    )
    if not paths:
        raise ValueError(f"No image frames found in: {sample_dir}")
    return paths


@dataclass(frozen=True)
class SpatialTransform:
    """One set of spatial parameters shared by aligned RGB and mask frames."""

    crop_left: int
    horizontal_flip: bool


class FineSynchroIndex:
    """Manifest-backed sample index and deterministic reference policy."""

    def __init__(
        self,
        repo_root: Path,
        manifest_path: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.manifest_path = (
            manifest_path.resolve()
            if manifest_path is not None
            else self.repo_root
            / "data"
            / "manifests"
            / "finesynchro_450_50_165.jsonl"
        )
        self.config_path = (
            config_path.resolve()
            if config_path is not None
            else self.repo_root / "configs" / "data" / "finesynchro.yaml"
        )

        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.data_root = (self.repo_root / config["dataset"]["root"]).resolve()
        self.records = self._load_records(self.manifest_path)
        self.by_split: dict[str, list[str]] = defaultdict(list)
        self.train_by_action: dict[str, list[str]] = defaultdict(list)

        for sample_id, record in self.records.items():
            split = str(record["split"])
            self.by_split[split].append(sample_id)
            if split == "train":
                self.train_by_action[str(record["action_code"])].append(sample_id)

        for values in self.by_split.values():
            values.sort()
        for values in self.train_by_action.values():
            values.sort()

    @staticmethod
    def _load_records(path: Path) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                if sample_id in records:
                    raise ValueError(
                        f"Duplicate sample ID {sample_id!r} at line {line_number}"
                    )
                records[sample_id] = record
        if len(records) != 665:
            raise ValueError(f"Expected 665 manifest records, found {len(records)}")
        return records

    def record(self, sample_id: str) -> dict[str, Any]:
        return self.records[sample_id]

    def sample_ids(self, split: str) -> tuple[str, ...]:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        return tuple(self.by_split[split])

    def training_reference(
        self, sample_id: str, seed: int, epoch: int | None = None
    ) -> str:
        """Choose one fixed same-action training reference reproducibly.

        ``epoch`` remains accepted for compatibility but deliberately does not
        affect the identity, matching the paper's frozen-reference protocol.
        """

        record = self.record(sample_id)
        action = str(record["action_code"])
        candidates = [
            candidate
            for candidate in self.train_by_action[action]
            if candidate != sample_id
        ]
        if not candidates:
            # Historical behavior permits self-reference for singleton classes.
            candidates = list(self.train_by_action[action])
        if not candidates:
            raise ValueError(f"No training reference for action {action!r}")
        generator = random.Random(f"{seed}:{sample_id}:train-reference")
        return candidates[generator.randrange(len(candidates))]

    def test_references(
        self, sample_id: str, voter_number: int, seed: int
    ) -> tuple[str, ...]:
        """Choose stable same-action training voters for a test sample."""

        if voter_number <= 0:
            raise ValueError("voter_number must be positive")
        record = self.record(sample_id)
        action = str(record["action_code"])
        candidates = list(self.train_by_action[action])
        if not candidates:
            raise ValueError(f"No training reference for action {action!r}")
        generator = random.Random(f"{seed}:{sample_id}:test-references")
        generator.shuffle(candidates)
        return tuple(candidates[:voter_number])

    def resolve_sample_path(self, record: dict[str, Any], field: str) -> Path:
        relative = record.get(field)
        if not relative:
            raise ValueError(f"Sample {record['sample_id']} has no {field}")
        return self.data_root / str(relative)


class ClipLoader:
    """Load 96-frame RGB clips and aligned single-channel predicted masks."""

    def __init__(
        self,
        output_length: int = 96,
        resize_height: int = 112,
        resize_width: int = 200,
        crop_size: int = 112,
    ) -> None:
        if crop_size > resize_height or crop_size > resize_width:
            raise ValueError("crop_size cannot exceed resized dimensions")
        self.output_length = output_length
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size

    def spatial_transform(self, training: bool, seed: int | str) -> SpatialTransform:
        generator = random.Random(str(seed))
        maximum_left = self.resize_width - self.crop_size
        crop_left = generator.randint(0, maximum_left) if training else maximum_left // 2
        horizontal_flip = generator.random() < 0.5 if training else False
        return SpatialTransform(crop_left=crop_left, horizontal_flip=horizontal_flip)

    def _selected_paths(self, sample_dir: Path) -> list[Path]:
        paths = sorted_frame_paths(sample_dir)
        indices = uniform_frame_indices(len(paths), self.output_length)
        return [paths[int(index)] for index in indices]

    def _prepare_image(
        self,
        image: Image.Image,
        spatial: SpatialTransform,
        *,
        mask: bool,
    ) -> torch.Tensor:
        image = image.convert("L" if mask else "RGB")
        image = vision_functional.resize(
            image,
            [self.resize_height, self.resize_width],
            interpolation=InterpolationMode.NEAREST
            if mask
            else InterpolationMode.BILINEAR,
        )
        image = vision_functional.crop(
            image,
            top=0,
            left=spatial.crop_left,
            height=self.crop_size,
            width=self.crop_size,
        )
        if spatial.horizontal_flip:
            image = vision_functional.hflip(image)
        tensor = vision_functional.pil_to_tensor(image).to(dtype=torch.float32) / 255.0
        return tensor

    def load_rgb(
        self, sample_dir: Path, spatial: SpatialTransform
    ) -> torch.Tensor:
        frames = []
        for path in self._selected_paths(sample_dir):
            with Image.open(path) as image:
                frames.append(self._prepare_image(image, spatial, mask=False))
        clip = torch.stack(frames, dim=1)
        mean = torch.tensor(RGB_MEAN, dtype=clip.dtype).view(3, 1, 1, 1)
        std = torch.tensor(RGB_STD, dtype=clip.dtype).view(3, 1, 1, 1)
        return (clip - mean) / std

    def load_mask(
        self, sample_dir: Path, spatial: SpatialTransform
    ) -> torch.Tensor:
        frames = [
            self._prepare_image(Image.open(path), spatial, mask=True)
            for path in self._selected_paths(sample_dir)
        ]
        return torch.stack(frames, dim=1)

    def load_aligned_rgb_and_mask(
        self,
        rgb_dir: Path,
        mask_dir: Path,
        spatial: SpatialTransform,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        live_paths = sorted_frame_paths(rgb_dir)
        mask_paths = sorted_frame_paths(mask_dir)
        if [path.name for path in live_paths] != [path.name for path in mask_paths]:
            raise ValueError(f"RGB/mask frame mismatch: {rgb_dir} vs {mask_dir}")
        indices = uniform_frame_indices(len(live_paths), self.output_length)
        live_frames = []
        mask_frames = []
        for index in indices:
            live_path = live_paths[int(index)]
            mask_path = mask_paths[int(index)]
            with Image.open(live_path) as live_image:
                live_frames.append(
                    self._prepare_image(live_image, spatial, mask=False)
                )
            with Image.open(mask_path) as mask_image:
                mask_frames.append(
                    self._prepare_image(mask_image, spatial, mask=True)
                )
        live_clip = torch.stack(live_frames, dim=1)
        mean = torch.tensor(RGB_MEAN, dtype=live_clip.dtype).view(3, 1, 1, 1)
        std = torch.tensor(RGB_STD, dtype=live_clip.dtype).view(3, 1, 1, 1)
        return (live_clip - mean) / std, torch.stack(mask_frames, dim=1)

    def load_aligned_live_and_mask(
        self,
        live_dir: Path,
        mask_dir: Path,
        spatial: SpatialTransform,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility alias for earlier callers."""

        return self.load_aligned_rgb_and_mask(live_dir, mask_dir, spatial)


class FineSynchroPairDataset(Dataset[dict[str, Any]]):
    """Load deterministic query/reference pairs for CRR-Sync training."""

    def __init__(
        self,
        index: FineSynchroIndex,
        split: str,
        clip_loader: ClipLoader | None = None,
        seed: int = 0,
        sample_ids: Iterable[str] | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "FineSynchroPairDataset supports train, validation, or test"
            )
        self.index = index
        self.split = split
        self.clip_loader = clip_loader or ClipLoader()
        self.seed = seed
        self.epoch = 0
        self.ids = (
            tuple(sample_ids)
            if sample_ids is not None
            else index.sample_ids(split)
        )
        for sample_id in self.ids:
            if str(index.record(sample_id)["split"]) != split:
                raise ValueError(
                    f"Sample {sample_id} does not belong to split {split}"
                )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)

    def _load_sample(
        self,
        sample_id: str,
        role: str,
        training: bool,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        effective_epoch = self.epoch if epoch is None else epoch
        record = self.index.record(sample_id)
        live_spatial = self.clip_loader.spatial_transform(
            training=training,
            seed=f"{self.seed}:{effective_epoch}:{sample_id}:{role}:live",
        )
        replay_spatial = self.clip_loader.spatial_transform(
            training=training,
            seed=f"{self.seed}:{effective_epoch}:{sample_id}:{role}:replay",
        )
        live, live_mask = self.clip_loader.load_aligned_rgb_and_mask(
            self.index.resolve_sample_path(record, "live_path"),
            self.index.resolve_sample_path(record, "live_mask_path"),
            live_spatial,
        )
        replay, replay_mask = self.clip_loader.load_aligned_rgb_and_mask(
            self.index.resolve_sample_path(record, "replay1_path"),
            self.index.resolve_sample_path(record, "replay1_mask_path"),
            replay_spatial,
        )
        return {
            "sample_id": sample_id,
            "live": live,
            "live_mask": live_mask,
            "replay": replay,
            "replay_mask": replay_mask,
            "score": torch.tensor(float(record["dive_score"]), dtype=torch.float32),
            "synchronisation": torch.tensor(
                record["synchronisation_scores"], dtype=torch.float32
            ),
            "boundaries": torch.from_numpy(
                sampled_stage_boundaries(
                    record["phase_labels"], self.clip_loader.output_length
                )
            ).long(),
            "action_code": record["action_code"],
        }

    def load_pair(
        self,
        query_id: str,
        reference_id: str,
        *,
        training: bool | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """Load an explicit pair, used by stable multi-reference evaluation."""

        if training is None:
            training = self.split == "train"
        return {
            "query": self._load_sample(
                query_id, "query", training, epoch=epoch
            ),
            "reference": self._load_sample(
                reference_id, "reference", training, epoch=epoch
            ),
        }

    def load_sample(
        self,
        sample_id: str,
        *,
        role: str = "sample",
        training: bool | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """Public single-sample loader for unique-video feature caching."""

        if training is None:
            training = self.split == "train"
        return self._load_sample(
            sample_id, role, training, epoch=epoch
        )

    def __getitem__(self, item: int | tuple[int, int]) -> dict[str, Any]:
        epoch = self.epoch
        if isinstance(item, tuple):
            item, epoch = item
        query_id = self.ids[item]
        if self.split == "train":
            reference_id = self.index.training_reference(
                query_id, seed=self.seed
            )
        else:
            reference_id = self.index.test_references(
                query_id, voter_number=1, seed=self.seed
            )[0]
        return self.load_pair(query_id, reference_id, epoch=epoch)


class FixedPairDataset(Dataset[dict[str, Any]]):
    """Load an explicit ordered list of query/reference pairs."""

    def __init__(
        self,
        base: FineSynchroPairDataset,
        pairs: Iterable[tuple[str, str]],
    ) -> None:
        self.base = base
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        query_id, reference_id = self.pairs[item]
        return self.base.load_pair(
            query_id, reference_id, training=self.base.split == "train"
        )


class EpochShuffleSampler(Sampler[tuple[int, int]]):
    """Send the epoch with each index so persistent workers stay epoch-aware."""

    def __init__(self, dataset_length: int, seed: int) -> None:
        if dataset_length <= 0:
            raise ValueError("dataset_length must be positive")
        self.dataset_length = dataset_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(
            self.dataset_length, generator=generator
        ).tolist()
        return iter((int(index), self.epoch) for index in order)

    def __len__(self) -> int:
        return self.dataset_length
