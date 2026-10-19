"""Inference for unlabeled CSV files using the selected fitted pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data import clean_listings
from src.settings import ARTIFACTS_DIR, FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Airbnb listing ratings.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input, low_memory=False)
    features = clean_listings(raw, require_target=False).drop(columns="review_scores_rating", errors="ignore")
    model = joblib.load(ARTIFACTS_DIR / "baseline_selected_model.joblib")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predicted_review_score": model.predict(features[FEATURES])}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
