"""Compute confidence intervals and paired tests from frozen prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fa_crr_sync.evaluation import (  # noqa: E402
    bootstrap_score_metrics,
    paired_absolute_error_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id in rows:
                raise ValueError(f"Duplicate {query_id} at line {line_number}")
            rows[query_id] = row
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    return rows


def main() -> int:
    args = parse_args()
    primary = load_rows(args.predictions.resolve())
    ids = sorted(primary)
    report: dict[str, Any] = {
        "primary_path": str(args.predictions.resolve()),
        "sample_count": len(ids),
        "bootstrap": bootstrap_score_metrics(
            [primary[key]["prediction"] for key in ids],
            [primary[key]["target_score"] for key in ids],
            resamples=args.resamples,
            seed=args.seed,
        ),
    }
    if args.compare:
        comparison = load_rows(args.compare.resolve())
        if set(comparison) != set(primary):
            missing = sorted(set(primary) - set(comparison))
            extra = sorted(set(comparison) - set(primary))
            raise ValueError(
                f"Prediction ID mismatch; missing={missing}, extra={extra}"
            )
        for key in ids:
            if float(primary[key]["target_score"]) != float(
                comparison[key]["target_score"]
            ):
                raise ValueError(f"Target mismatch for {key}")
        report["comparison_path"] = str(args.compare.resolve())
        report["paired_absolute_error_test"] = paired_absolute_error_test(
            [primary[key]["prediction"] for key in ids],
            [comparison[key]["prediction"] for key in ids],
            [primary[key]["target_score"] for key in ids],
            resamples=args.resamples,
            seed=args.seed,
        )
    output = (
        args.output.resolve()
        if args.output
        else args.predictions.resolve().with_name(
            f"{args.predictions.stem}_statistics.json"
        )
    )
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
