"""Focused, reproducible EDA for the training split only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.settings import NUMERIC_FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET


def run_eda(processed_dir: Path = PROCESSED_DIR, output: Path | None = None) -> dict:
    """Summarise target shape, missingness, and simple training-only associations."""
    train = pd.read_csv(processed_dir / "train.csv")
    target = train[TARGET]
    summary = {
        "rows": len(train),
        "target": target.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
        "share_perfect_score": float((target == 5).mean()),
        "share_below_4": float((target < 4).mean()),
        "missingness_percent": (train.isna().mean().sort_values(ascending=False) * 100).head(12).to_dict(),
        "absolute_numeric_correlations": (
            train[NUMERIC_FEATURES + [TARGET]]
            .corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        ),
        "target_by_room_type": train.groupby("room_type")[TARGET].agg(["count", "mean", "std"]).to_dict(),
    }
    output = output or RESULTS_DIR / "eda_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run focused EDA on the training split.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_eda(args.processed_dir, args.output)
