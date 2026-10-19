"""Evaluate a fixed prospective model across cities without touching the test split."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import clean_listings
from src.features import fit_amenity_vocabulary, make_feature_set
from src.metrics import regression_metrics
from src.models import build_model
from src.settings import RESULTS_DIR, TARGET

XGBOOST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
}


def load_city(path: Path) -> pd.DataFrame:
    """Load one labelled city file using the same deterministic cleaning as training."""
    return clean_listings(pd.read_csv(path, low_memory=False))


def evaluate_direction(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float | int]:
    """Fit F4 without Superhost on one city and evaluate it on another."""
    vocabulary = fit_amenity_vocabulary(train)
    train_features = make_feature_set(train, "f4_missingness", vocabulary)
    validation_features = make_feature_set(validation, "f4_missingness", vocabulary)
    numeric = train_features.numeric_features
    categorical = [name for name in train_features.categorical_features if name != "host_is_superhost"]
    columns = numeric + categorical

    model = build_model("xgboost", XGBOOST_PARAMS, numeric, categorical)
    model.fit(train_features.data[columns], train[TARGET])
    metrics = regression_metrics(validation[TARGET], model.predict(validation_features.data[columns]))
    return {"train_rows": len(train), "validation_rows": len(validation), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LA↔NY city-held-out validation for F4.")
    parser.add_argument("--los-angeles", type=Path, default=Path("data/raw/airbnb_la_raw.csv"))
    parser.add_argument("--new-york", type=Path, default=Path("data/raw/airbnb_ny_raw.csv"))
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "city_holdout_metrics.csv")
    args = parser.parse_args()

    los_angeles = load_city(args.los_angeles)
    new_york = load_city(args.new_york)
    rows = []
    for train_name, train, validation_name, validation in [
        ("Los Angeles", los_angeles, "New York", new_york),
        ("New York", new_york, "Los Angeles", los_angeles),
    ]:
        metrics = evaluate_direction(train, validation)
        rows.append({"train_city": train_name, "validation_city": validation_name, **metrics})
        print(
            f"Train {train_name} → validate {validation_name}: "
            f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
