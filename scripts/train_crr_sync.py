"""Train and evaluate CRR-Sync or the paper-aligned FA-CRR-Sync."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fa_crr_sync.data import (  # noqa: E402
    EpochShuffleSampler,
    FineSynchroIndex,
    FineSynchroPairDataset,
)
from fa_crr_sync.models import CRRSync, FACRRSync  # noqa: E402
from fa_crr_sync.training.runner import (  # noqa: E402
    aggregate_voter_predictions,
    build_optimizer,
    compute_evaluation_metrics,
    create_run_directory,
    load_checkpoint,
    load_experiment_config,
    predict_voters_cached,
    save_checkpoint,
    save_evaluation_artifacts,
    train_one_epoch,
)
from fa_crr_sync.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/fa_crr_sync.yaml"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--voters", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    return parser.parse_args()


def _bounded(values: tuple[str, ...], maximum: int | None) -> tuple[str, ...]:
    if maximum is None:
        return values
    if maximum <= 0:
        raise ValueError("Sample limits must be positive")
    return values[:maximum]


def main() -> int:
    args = parse_args()
    config = load_experiment_config(REPO_ROOT, args.config)
    if args.epochs is not None:
        config["optimization"]["epochs"] = args.epochs
    if args.workers is not None:
        config["data"]["workers"] = args.workers
    if args.train_batch_size is not None:
        config["data"]["train_batch_size"] = args.train_batch_size
    if args.gradient_accumulation_steps is not None:
        config["optimization"]["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    if args.voters is not None:
        config["data"]["voter_number"] = args.voters
    if args.evaluation_interval is not None:
        config["optimization"]["evaluation_interval"] = args.evaluation_interval

    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["optimization"]["deterministic"]))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for FA-CRR-Sync training")
    device = torch.device("cuda")
    index = FineSynchroIndex(REPO_ROOT)
    train_ids = _bounded(
        index.sample_ids("train"), args.max_train_samples
    )
    train_base = FineSynchroPairDataset(
        index,
        split="train",
        seed=seed,
        sample_ids=train_ids,
    )
    train_dataset = train_base
    workers = int(config["data"]["workers"])
    train_sampler = EpochShuffleSampler(len(train_base), seed)
    loader_options = {
        "batch_size": int(config["data"]["train_batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0
        and bool(config["data"]["persistent_workers"]),
    }
    if workers > 0:
        loader_options["prefetch_factor"] = int(
            config["data"]["prefetch_factor"]
        )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
        **loader_options,
    )

    validation_base = FineSynchroPairDataset(
        index, split="validation", seed=seed
    )
    validation_ids = validation_base.ids
    voters = int(config["data"]["voter_number"])
    validation_pairs = tuple(
        (query_id, reference_id)
        for query_id in validation_ids
        for reference_id in index.test_references(query_id, voters, seed)
    )

    test_base = FineSynchroPairDataset(index, split="test", seed=seed)
    test_ids = _bounded(test_base.ids, args.max_test_samples)
    test_pairs = tuple(
        (query_id, reference_id)
        for query_id in test_ids
        for reference_id in index.test_references(query_id, voters, seed)
    )
    model_name = str(config["experiment"]["model"])
    if model_name == "FACRRSync":
        model = FACRRSync().to(device)
    elif model_name == "CRRSync":
        model = CRRSync().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    pretrained = (
        REPO_ROOT / config["model"]["pretrained_backbone"]
    ).resolve()
    if isinstance(model, FACRRSync):
        model.load_pretrained(pretrained)
    else:
        model.backbone.load_pretrained(pretrained)
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(config["optimization"]["mixed_precision"]),
        init_scale=float(
            config["optimization"]["gradient_scaler_initial_scale"]
        ),
    )

    if args.resume:
        checkpoint = args.resume.resolve()
        run_dir = checkpoint.parents[1]
        start_epoch, best_metric = load_checkpoint(
            checkpoint, model, optimizer, scaler
        )
    else:
        run_dir = create_run_directory(REPO_ROOT, config, args.run_name)
        start_epoch, best_metric = 0, float("-inf")

    epochs = int(config["optimization"]["epochs"])
    interval = int(config["optimization"]["evaluation_interval"])
    warmup_epochs = max(
        1,
        int(
            epochs
            * float(config["loss"]["ground_truth_stage_warmup_fraction"])
        ),
    )
    history_path = run_dir / "history.jsonl"
    for epoch in range(start_epoch, epochs):
        train_base.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            use_ground_truth_stages=epoch < warmup_epochs,
        )
        record = {"epoch": epoch, "train": train_metrics}
        should_evaluate = (epoch + 1) % interval == 0 or epoch + 1 == epochs
        if should_evaluate:
            voter_rows, cache_telemetry = predict_voters_cached(
                model,
                validation_base,
                validation_pairs,
                device,
                mixed_precision=bool(config["optimization"]["mixed_precision"]),
                pair_batch_size=int(
                    config["data"]["cached_evaluation_pair_batch_size"]
                ),
            )
            rows = aggregate_voter_predictions(voter_rows)
            metrics = compute_evaluation_metrics(rows)
            metrics["cache"] = cache_telemetry
            record["validation"] = metrics
            save_evaluation_artifacts(
                run_dir, epoch, voter_rows, rows, metrics
            )
            if metrics["srcc"] > best_metric:
                best_metric = metrics["srcc"]
                best_path = run_dir / "checkpoints" / "best.pt"
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                )
        last_path = run_dir / "checkpoints" / "last.pt"
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scaler,
            epoch=epoch,
            best_metric=best_metric,
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    if (run_dir / "checkpoints" / "best.pt").exists():
        shutil.copy2(
            run_dir / "checkpoints" / "best.pt",
            run_dir / "checkpoints" / "selected.pt",
        )
        load_checkpoint(
            run_dir / "checkpoints" / "selected.pt",
            model,
            optimizer,
            scaler,
        )
        voter_rows, cache_telemetry = predict_voters_cached(
            model,
            test_base,
            test_pairs,
            device,
            mixed_precision=bool(config["optimization"]["mixed_precision"]),
            pair_batch_size=int(
                config["data"]["cached_evaluation_pair_batch_size"]
            ),
        )
        rows = aggregate_voter_predictions(voter_rows)
        metrics = compute_evaluation_metrics(rows)
        metrics["cache"] = cache_telemetry
        save_evaluation_artifacts(
            run_dir, epochs, voter_rows, rows, metrics
        )
        (run_dir / "metrics" / "final_test.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
