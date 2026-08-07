"""Run one real mixed-precision FA-CRR-Sync optimization step."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fa_crr_sync.data import FineSynchroIndex, FineSynchroPairDataset  # noqa: E402
from fa_crr_sync.models import FACRRSync  # noqa: E402
from fa_crr_sync.training import compute_crr_sync_losses  # noqa: E402
from fa_crr_sync.utils import seed_everything  # noqa: E402


PRETRAINED = REPO_ROOT / "checkpoints" / "i3d_kinetics400_rgb.pth"


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the training smoke test")
    seed_everything(0)
    device = torch.device("cuda")
    index = FineSynchroIndex(REPO_ROOT)
    dataset = FineSynchroPairDataset(index, split="train", seed=0)
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = FACRRSync().to(device)
    model.load_pretrained(PRETRAINED)
    model.train()
    optimizer = torch.optim.Adam(
        [
            {"params": model.rgb_backbone.parameters(), "lr": 1e-4},
            {"params": model.mask_backbone.parameters(), "lr": 1e-4},
            {"params": model.core.parameters(), "lr": 1e-3},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    optimizer.zero_grad(set_to_none=True)
    boundaries = torch.cat(
        [batch["query"]["boundaries"], batch["reference"]["boundaries"]], dim=0
    ).to(device)

    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            batch["query"]["live"].to(device),
            batch["reference"]["live"].to(device),
            batch["query"]["replay"].to(device),
            batch["reference"]["replay"].to(device),
            batch["query"]["live_mask"].to(device),
            batch["reference"]["live_mask"].to(device),
            batch["query"]["replay_mask"].to(device),
            batch["reference"]["replay_mask"].to(device),
            stage_boundaries=boundaries,
        )
        losses = compute_crr_sync_losses(
            output,
            query_score=batch["query"]["score"].to(device),
            reference_score=batch["reference"]["score"].to(device),
            query_synchronisation=batch["query"]["synchronisation"].to(device),
            reference_synchronisation=batch["reference"][
                "synchronisation"
            ].to(device),
            target_boundaries=boundaries,
        )
    scaler.scale(losses.total).backward()
    scaler.unscale_(optimizer)
    nonfinite_gradient_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    scaler.step(optimizer)
    scaler.update()
    elapsed = time.perf_counter() - start

    finite_gradients = not nonfinite_gradient_parameters
    report = {
        "query_id": batch["query"]["sample_id"][0],
        "reference_id": batch["reference"]["sample_id"][0],
        "batch_size": 1,
        "loss": {
            "score": float(losses.score.detach().cpu()),
            "transition": float(losses.transition.detach().cpu()),
            "synchronisation": float(losses.synchronisation.detach().cpu()),
            "total": float(losses.total.detach().cpu()),
        },
        "finite_gradients": bool(finite_gradients),
        "nonfinite_gradient_parameters": nonfinite_gradient_parameters,
        "final_grad_scale": float(scaler.get_scale()),
        "elapsed_seconds": round(elapsed, 3),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    report_path = REPO_ROOT / "data" / "reports" / "train_step_smoke.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if finite_gradients else 1


if __name__ == "__main__":
    raise SystemExit(main())
