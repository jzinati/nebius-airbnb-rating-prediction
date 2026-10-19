"""Tune on validation only, then refit the selected model on train + validation."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import fit_amenity_vocabulary, make_feature_set
from src.metrics import regression_metrics
from src.models import build_model, candidate_models
from src.settings import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET


def _load_split(name: str, processed_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(processed_dir / f"{name}.csv")
    return data, data[TARGET]


def run_training(
    processed_dir: Path = PROCESSED_DIR,
    feature_set_name: str = "base",
    experiment_id: str = "baseline",
    exclude_features: tuple[str, ...] = (),
) -> pd.DataFrame:
    train_data, y_train = _load_split("train", processed_dir)
    validation_data, y_validation = _load_split("validation", processed_dir)
    vocabulary_required = feature_set_name in {"f3_top_amenities", "f4_missingness"}
    amenity_vocabulary = fit_amenity_vocabulary(train_data) if vocabulary_required else None
    train_features = make_feature_set(train_data, feature_set_name, amenity_vocabulary)
    validation_features = make_feature_set(validation_data, feature_set_name, amenity_vocabulary)
    excluded = set(exclude_features)
    available = set(train_features.numeric_features + train_features.categorical_features)
    unknown = excluded - available
    if unknown:
        raise ValueError(f"Cannot exclude features that are not present: {sorted(unknown)}")
    numeric = [column for column in train_features.numeric_features if column not in excluded]
    categorical = [column for column in train_features.categorical_features if column not in excluded]
    x_train = train_features.data[numeric + categorical]
    x_validation = validation_features.data[numeric + categorical]
    rows = []
    candidates = candidate_models(feature_set_name)

    for index, (name, params) in enumerate(candidates):
        model = build_model(name, params, numeric, categorical)
        model.fit(x_train, y_train)
        metrics = regression_metrics(y_validation, model.predict(x_validation))
        rows.append({
            "candidate_id": index,
            "model": name,
            "parameters": json.dumps(params, sort_keys=True),
            "feature_set": feature_set_name,
            "excluded_features": json.dumps(sorted(excluded)),
            **metrics,
        })
        print(f"{index}: {name} validation RMSE={metrics['rmse']:.4f}")

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"{experiment_id}_validation_metrics.csv", index=False)

    chosen = results.iloc[0]
    original_index = int(chosen["candidate_id"])
    name, params = candidates[original_index]
    final_model = build_model(name, params, numeric, categorical)
    x_refit = pd.concat([x_train, x_validation], ignore_index=True)
    y_refit = pd.concat([y_train, y_validation], ignore_index=True)
    final_model.fit(x_refit, y_refit)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, ARTIFACTS_DIR / f"{experiment_id}_selected_model.joblib")
    selection = {
        "experiment_id": experiment_id,
        "feature_set": feature_set_name,
        "excluded_features": sorted(excluded),
        "model": name,
        "parameters": params,
        "selection_metric": "validation_rmse",
        "validation_rmse": float(chosen["rmse"]),
    }
    (ARTIFACTS_DIR / f"{experiment_id}_model_selection.json").write_text(
        json.dumps(selection, indent=2)
    )
    if amenity_vocabulary is not None:
        (ARTIFACTS_DIR / f"{experiment_id}_amenity_vocabulary.json").write_text(
            json.dumps(amenity_vocabulary, indent=2)
        )
    print(f"Selected {name}; refit on train + validation. This does not evaluate the test set.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a model using validation data, then refit it on train + validation."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--feature-set",
        choices=("base", "f1_price_capacity", "f2_amenities", "f3_top_amenities", "f4_missingness"),
        default="base",
    )
    parser.add_argument("--experiment-id", default="baseline")
    parser.add_argument("--exclude-feature", action="append", default=[])
    args = parser.parse_args()
    run_training(
        processed_dir=args.processed_dir,
        feature_set_name=args.feature_set,
        experiment_id=args.experiment_id,
        exclude_features=tuple(args.exclude_feature),
    )
