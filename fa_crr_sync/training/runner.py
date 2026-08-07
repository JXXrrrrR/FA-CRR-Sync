"""Reproducible training, evaluation, and run-artifact utilities."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fa_crr_sync.data import FineSynchroPairDataset
from fa_crr_sync.evaluation.metrics import (
    aggregate_score_metrics,
    temporal_iou,
    temporal_iou_accuracy,
)
from fa_crr_sync.models.crr_sync import CRRSync
from fa_crr_sync.models.fa_crr_sync import FACRRSync
from fa_crr_sync.training.losses import compute_crr_sync_losses


def load_experiment_config(repo_root: Path, config_path: Path) -> dict[str, Any]:
    """Load the experiment and its referenced data config into one snapshot."""

    repo_root = repo_root.resolve()
    config_path = (
        config_path if config_path.is_absolute() else repo_root / config_path
    ).resolve()
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = (repo_root / experiment["data"]["config"]).resolve()
    resolved = dict(experiment)
    resolved["data"] = dict(experiment["data"])
    resolved["data"]["dataset_config"] = yaml.safe_load(
        data_path.read_text(encoding="utf-8")
    )
    resolved["_resolved"] = {
        "repo_root": str(repo_root),
        "experiment_config": str(config_path),
        "data_config": str(data_path),
    }
    return resolved


def create_run_directory(
    repo_root: Path,
    config: dict[str, Any],
    run_name: str | None = None,
) -> Path:
    """Create a non-overwriting output directory and save resolved config."""

    base_name = run_name or str(config["experiment"]["name"])
    base = repo_root.resolve() / "outputs" / base_name
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{suffix:02d}")
        suffix += 1
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return run_dir


def build_optimizer(
    model: CRRSync | FACRRSync, config: dict[str, Any]
) -> torch.optim.Optimizer:
    settings = config["optimization"]
    base_lr = float(settings["base_learning_rate"])
    backbones = (
        [model.backbone]
        if isinstance(model, CRRSync)
        else [model.rgb_backbone, model.mask_backbone]
    )
    return torch.optim.Adam(
        [
            {
                "params": [
                    parameter
                    for backbone in backbones
                    for parameter in backbone.parameters()
                ],
                "lr": base_lr * float(settings["backbone_learning_rate_factor"]),
            },
            {"params": model.core.parameters(), "lr": base_lr},
        ],
        weight_decay=float(settings["weight_decay"]),
    )


def _forward_model(
    model: CRRSync | FACRRSync,
    batch: dict[str, dict[str, Any]],
    stage_boundaries: torch.Tensor | None,
):
    arguments = [
        batch["query"]["live"],
        batch["reference"]["live"],
        batch["query"]["replay"],
        batch["reference"]["replay"],
    ]
    if isinstance(model, FACRRSync):
        arguments.extend(
            [
                batch["query"]["live_mask"],
                batch["reference"]["live_mask"],
                batch["query"]["replay_mask"],
                batch["reference"]["replay_mask"],
            ]
        )
    return model(*arguments, stage_boundaries=stage_boundaries)


def _move_pair_batch(
    batch: dict[str, Any], device: torch.device
) -> dict[str, dict[str, Any]]:
    moved: dict[str, dict[str, Any]] = {"query": {}, "reference": {}}
    for role in ("query", "reference"):
        for key, value in batch[role].items():
            moved[role][key] = (
                value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
    return moved


def train_one_epoch(
    model: CRRSync | FACRRSync,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    *,
    use_ground_truth_stages: bool,
) -> dict[str, float | int]:
    """Train one epoch and return sample-weighted mean losses."""

    # Train the complete network, including every BatchNorm affine parameter
    # and its running statistics. Gradient accumulation cannot reproduce
    # BatchNorm statistics from a larger physical batch, so the experiment
    # configuration uses the manuscript's physical batch of four pairs.
    model.train()
    totals = defaultdict(float)
    sample_count = 0
    loss_config = config["loss"]
    accumulation_steps = int(
        config["optimization"].get("gradient_accumulation_steps", 1)
    )
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    batch_count = len(loader)
    optimizer_steps = 0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    for batch_index, raw_batch in enumerate(loader):
        batch = _move_pair_batch(raw_batch, device)
        size = int(batch["query"]["score"].shape[0])
        boundaries = torch.cat(
            [batch["query"]["boundaries"], batch["reference"]["boundaries"]],
            dim=0,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(config["optimization"]["mixed_precision"]),
        ):
            output = _forward_model(
                model,
                batch,
                boundaries if use_ground_truth_stages else None,
            )
            losses = compute_crr_sync_losses(
                output,
                query_score=batch["query"]["score"],
                reference_score=batch["reference"]["score"],
                query_synchronisation=batch["query"]["synchronisation"],
                reference_synchronisation=batch["reference"]["synchronisation"],
                target_boundaries=boundaries,
                score_weight=float(loss_config["score_weight"]),
                transition_weight=float(loss_config["transition_weight"]),
                synchronisation_weight=float(
                    loss_config["synchronisation_weight"]
                ),
            )
        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, batch_count - window_start)
        scaler.scale(losses.total / window_size).backward()
        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == batch_count
        )
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        for name in ("score", "transition", "synchronisation", "total"):
            totals[name] += float(getattr(losses, name).detach().cpu()) * size
        sample_count += size

    if sample_count == 0:
        raise ValueError("Training loader produced no samples")
    result: dict[str, float | int] = {
        **{name: value / sample_count for name, value in totals.items()},
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_mib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2
        )
    return result


def _as_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().float().cpu().reshape(-1)]


@torch.inference_mode()
def predict_voters_cached(
    model: CRRSync | FACRRSync,
    dataset: FineSynchroPairDataset,
    pairs: Iterable[tuple[str, str]],
    device: torch.device,
    *,
    mixed_precision: bool,
    pair_batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Encode every unique video once, then evaluate all voter pairs."""

    if dataset.split not in {"validation", "test"}:
        raise ValueError(
            "Cached voter prediction requires a validation or test dataset"
        )
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be positive")
    pair_list = tuple(pairs)
    unique_ids = sorted(
        {sample_id for pair in pair_list for sample_id in pair}
    )
    model.eval()
    cache: dict[str, dict[str, Any]] = {}
    for sample_id in unique_ids:
        sample = dataset.load_sample(
            sample_id, role="evaluation-cache", training=False, epoch=0
        )
        live = sample["live"].unsqueeze(0).to(device, non_blocking=True)
        replay = sample["replay"].unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision,
        ):
            if isinstance(model, FACRRSync):
                live_mask = sample["live_mask"].unsqueeze(0).to(
                    device, non_blocking=True
                )
                replay_mask = sample["replay_mask"].unsqueeze(0).to(
                    device, non_blocking=True
                )
                live_rgb_feature, live_rgb_map = model.rgb_backbone(live)
                replay_rgb_feature, replay_rgb_map = model.rgb_backbone(replay)
                live_mask_feature, live_mask_map = model.mask_backbone(live_mask)
                replay_mask_feature, replay_mask_map = model.mask_backbone(
                    replay_mask
                )
                live_feature, live_map = model._gate(
                    live_rgb_feature,
                    live_rgb_map,
                    live_mask_feature,
                    live_mask_map,
                )
                replay_feature, _ = model._gate(
                    replay_rgb_feature,
                    replay_rgb_map,
                    replay_mask_feature,
                    replay_mask_map,
                )
            else:
                live_feature, live_map = model.backbone(live)
                replay_feature, _ = model.backbone(replay)
        cache[sample_id] = {
            "live_feature": live_feature[0].cpu(),
            "live_map": live_map[0].cpu(),
            "replay_feature": replay_feature[0].cpu(),
            "score": float(sample["score"]),
            "synchronisation": sample["synchronisation"].float().tolist(),
            "boundaries": sample["boundaries"].float().tolist(),
            "action_code": str(sample["action_code"]),
        }

    rows: list[dict[str, Any]] = []
    for start in range(0, len(pair_list), pair_batch_size):
        batch_pairs = pair_list[start : start + pair_batch_size]
        queries = [cache[query_id] for query_id, _ in batch_pairs]
        references = [cache[reference_id] for _, reference_id in batch_pairs]

        def stacked(items: list[dict[str, Any]], key: str) -> torch.Tensor:
            return torch.stack(
                [item[key] for item in items], dim=0
            ).to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision,
        ):
            output = model.core(
                stacked(queries, "live_feature"),
                stacked(references, "live_feature"),
                stacked(queries, "live_map"),
                stacked(references, "live_map"),
                stacked(queries, "replay_feature"),
                stacked(references, "replay_feature"),
                stage_boundaries=None,
            )
        batch_size = len(batch_pairs)
        query_boundaries = output["decoded_boundaries"][:batch_size]
        delta_qr = output["score_delta_query_reference"].reshape(-1)
        delta_rq = output["score_delta_reference_query"].reshape(-1)
        sync_qr = output["synchronisation_delta_query_reference"]
        sync_rq = output["synchronisation_delta_reference_query"]
        for index, (query_id, reference_id) in enumerate(batch_pairs):
            query = queries[index]
            reference = references[index]
            reference_score = float(reference["score"])
            score_from_qr = reference_score + float(
                delta_qr[index].float().cpu()
            )
            score_from_rq = reference_score - float(
                delta_rq[index].float().cpu()
            )
            reference_sync = torch.tensor(
                reference["synchronisation"],
                device=device,
                dtype=sync_qr.dtype,
            )
            predicted_sync = reference_sync + sync_qr[index]
            target_boundary = list(query["boundaries"])
            predicted_boundary = _as_list(query_boundaries[index])
            rows.append(
                {
                    "query_id": query_id,
                    "reference_id": reference_id,
                    "action_code": query["action_code"],
                    "target_score": float(query["score"]),
                    "reference_score": reference_score,
                    "score_prediction_query_reference": score_from_qr,
                    "score_prediction_reference_query": score_from_rq,
                    "score_prediction": score_from_qr,
                    "target_synchronisation": list(
                        query["synchronisation"]
                    ),
                    "predicted_synchronisation": _as_list(predicted_sync),
                    "target_boundaries": target_boundary,
                    "predicted_boundaries": predicted_boundary,
                    "temporal_iou": temporal_iou(
                        target_boundary, predicted_boundary
                    ),
                }
            )
    return rows, {
        "pair_count": len(pair_list),
        "unique_encoded_videos": len(unique_ids),
        "avoided_video_encodings": 2 * len(pair_list) - len(unique_ids),
    }


@torch.inference_mode()
def predict_voters(
    model: CRRSync | FACRRSync,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool,
) -> list[dict[str, Any]]:
    """Return one auditable prediction row per query/reference pair."""

    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _move_pair_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision,
        ):
            output = _forward_model(model, batch, None)
        batch_size = int(batch["query"]["score"].shape[0])
        query_boundaries = output["decoded_boundaries"][:batch_size]
        delta_qr = output["score_delta_query_reference"].reshape(-1)
        delta_rq = output["score_delta_reference_query"].reshape(-1)
        sync_qr = output["synchronisation_delta_query_reference"]
        sync_rq = output["synchronisation_delta_reference_query"]
        for index in range(batch_size):
            reference_score = float(batch["reference"]["score"][index].cpu())
            score_from_qr = reference_score + float(delta_qr[index].float().cpu())
            score_from_rq = reference_score - float(delta_rq[index].float().cpu())
            reference_sync = batch["reference"]["synchronisation"][index]
            predicted_sync = reference_sync + sync_qr[index]
            target_boundary = _as_list(batch["query"]["boundaries"][index])
            predicted_boundary = _as_list(query_boundaries[index])
            rows.append(
                {
                    "query_id": str(batch["query"]["sample_id"][index]),
                    "reference_id": str(
                        batch["reference"]["sample_id"][index]
                    ),
                    "action_code": str(batch["query"]["action_code"][index]),
                    "target_score": float(
                        batch["query"]["score"][index].cpu()
                    ),
                    "reference_score": reference_score,
                    "score_prediction_query_reference": score_from_qr,
                    "score_prediction_reference_query": score_from_rq,
                    "score_prediction": score_from_qr,
                    "target_synchronisation": _as_list(
                        batch["query"]["synchronisation"][index]
                    ),
                    "predicted_synchronisation": _as_list(predicted_sync),
                    "target_boundaries": target_boundary,
                    "predicted_boundaries": predicted_boundary,
                    "temporal_iou": temporal_iou(
                        target_boundary, predicted_boundary
                    ),
                }
            )
    return rows


def aggregate_voter_predictions(
    voter_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate per-reference rows into one row per query."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in voter_rows:
        grouped[str(row["query_id"])].append(row)
    aggregated = []
    for query_id in sorted(grouped):
        rows = grouped[query_id]
        targets = {float(row["target_score"]) for row in rows}
        if len(targets) != 1:
            raise ValueError(f"Inconsistent target scores for {query_id}")
        prediction = float(np.mean([row["score_prediction"] for row in rows]))
        predicted_sync = np.mean(
            [row["predicted_synchronisation"] for row in rows], axis=0
        )
        predicted_boundaries = rows[0]["predicted_boundaries"]
        if any(
            row["predicted_boundaries"] != predicted_boundaries
            for row in rows[1:]
        ):
            predicted_boundaries = np.median(
                [row["predicted_boundaries"] for row in rows], axis=0
            ).tolist()
        target_boundaries = rows[0]["target_boundaries"]
        target_score = targets.pop()
        aggregated.append(
            {
                "query_id": query_id,
                "action_code": rows[0]["action_code"],
                "target_score": target_score,
                "prediction": prediction,
                "absolute_error": abs(prediction - target_score),
                "reference_ids": [row["reference_id"] for row in rows],
                "reference_scores": [row["reference_score"] for row in rows],
                "single_reference_predictions": [
                    row["score_prediction"] for row in rows
                ],
                "target_synchronisation": rows[0]["target_synchronisation"],
                "predicted_synchronisation": predicted_sync.tolist(),
                "target_boundaries": target_boundaries,
                "predicted_boundaries": predicted_boundaries,
                "temporal_iou": temporal_iou(
                    target_boundaries, predicted_boundaries
                ),
            }
        )
    return aggregated


def compute_evaluation_metrics(
    aggregated_rows: Iterable[dict[str, Any]],
) -> dict[str, float]:
    rows = list(aggregated_rows)
    score = aggregate_score_metrics(
        [row["prediction"] for row in rows],
        [row["target_score"] for row in rows],
    )
    targets = [row["target_boundaries"] for row in rows]
    predictions = [row["predicted_boundaries"] for row in rows]
    return {
        **asdict(score),
        "aiou_0_5": temporal_iou_accuracy(targets, predictions, 0.5),
        "aiou_0_75": temporal_iou_accuracy(targets, predictions, 0.75),
        "sample_count": len(rows),
    }


def save_evaluation_artifacts(
    run_dir: Path,
    epoch: int,
    voter_rows: list[dict[str, Any]],
    aggregated_rows: list[dict[str, Any]],
    metrics: dict[str, float],
) -> None:
    stem = f"epoch_{epoch:03d}"
    for path, rows in (
        (run_dir / "predictions" / f"{stem}_voters.jsonl", voter_rows),
        (run_dir / "predictions" / f"{stem}.jsonl", aggregated_rows),
    ):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    csv_path = run_dir / "predictions" / f"{stem}.csv"
    fieldnames = (
        "query_id",
        "action_code",
        "target_score",
        "prediction",
        "absolute_error",
        "temporal_iou",
        "reference_ids",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            output = {name: row[name] for name in fieldnames}
            output["reference_ids"] = "|".join(row["reference_ids"])
            writer.writerow(output)
    (run_dir / "metrics" / f"{stem}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def save_checkpoint(
    path: Path,
    model: CRRSync,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_metric: float,
) -> None:
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: CRRSync,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and state["cuda_random_state"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return int(state["epoch"]) + 1, float(state["best_metric"])
