"""Run a bounded Optuna search using the validation split only.

This module never reads ``test.csv``. Its saved trial table is an experiment
record, not a final-performance claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd

from src.features import fit_amenity_vocabulary, make_feature_set
from src.metrics import regression_metrics
from src.models import build_model
from src.settings import ARTIFACTS_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET


FEATURE_SET_CHOICES = ("base", "f1_price_capacity", "f2_amenities", "f3_top_amenities", "f4_missingness")


def suggest_xgboost_parameters(trial: optuna.Trial) -> dict[str, float | int]:
    """Use a compact, explainable search space for tree-ensemble complexity."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 16.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
    }


def run_tuning(
    processed_dir: Path = PROCESSED_DIR,
    feature_set_name: str = "f4_missingness",
    experiment_id: str = "optuna_xgboost_f4_without_superhost",
    trials: int = 20,
    exclude_features: tuple[str, ...] = ("host_is_superhost",),
) -> pd.DataFrame:
    """Fit each trial on train and score it once on validation."""
    train_data = pd.read_csv(processed_dir / "train.csv")
    validation_data = pd.read_csv(processed_dir / "validation.csv")
    vocabulary_required = feature_set_name in {"f3_top_amenities", "f4_missingness"}
    vocabulary = fit_amenity_vocabulary(train_data) if vocabulary_required else None
    train_features = make_feature_set(train_data, feature_set_name, vocabulary)
    validation_features = make_feature_set(validation_data, feature_set_name, vocabulary)

    excluded = set(exclude_features)
    available = set(train_features.numeric_features + train_features.categorical_features)
    unknown = excluded - available
    if unknown:
        raise ValueError(f"Cannot exclude features that are not present: {sorted(unknown)}")
    numeric = [column for column in train_features.numeric_features if column not in excluded]
    categorical = [column for column in train_features.categorical_features if column not in excluded]
    columns = numeric + categorical
    x_train = train_features.data[columns]
    x_validation = validation_features.data[columns]
    y_train = train_data[TARGET]
    y_validation = validation_data[TARGET]

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_xgboost_parameters(trial)
        model = build_model("xgboost", parameters, numeric, categorical)
        model.fit(x_train, y_train)
        metrics = regression_metrics(y_validation, model.predict(x_validation))
        trial.set_user_attr("mae", metrics["mae"])
        trial.set_user_attr("r2", metrics["r2"])
        print(f"trial={trial.number} validation RMSE={metrics['rmse']:.4f}")
        return metrics["rmse"]

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=experiment_id)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    records = []
    for trial in study.trials:
        if trial.value is None:
            continue
        records.append({
            "trial": trial.number,
            "model": "xgboost",
            "feature_set": feature_set_name,
            "excluded_features": json.dumps(sorted(excluded)),
            "parameters": json.dumps(trial.params, sort_keys=True),
            "rmse": trial.value,
            "mae": trial.user_attrs["mae"],
            "r2": trial.user_attrs["r2"],
        })
    results = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"{experiment_id}_trials.csv", index=False)

    selection = {
        "experiment_id": experiment_id,
        "model": "xgboost",
        "feature_set": feature_set_name,
        "excluded_features": sorted(excluded),
        "selection_metric": "validation_rmse",
        "trials": trials,
        "best_parameters": study.best_params,
        "validation_rmse": study.best_value,
        "validation_mae": study.best_trial.user_attrs["mae"],
        "validation_r2": study.best_trial.user_attrs["r2"],
        "test_set_used": False,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"{experiment_id}_selection.json").write_text(json.dumps(selection, indent=2))
    print(
        f"Best trial={study.best_trial.number}; validation RMSE={study.best_value:.4f}. "
        "The test set was not read."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost with Optuna on validation data only.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--feature-set", choices=FEATURE_SET_CHOICES, default="f4_missingness")
    parser.add_argument("--experiment-id", default="optuna_xgboost_f4_without_superhost")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--include-superhost", action="store_true")
    args = parser.parse_args()
    exclusions = () if args.include_superhost else ("host_is_superhost",)
    run_tuning(args.processed_dir, args.feature_set, args.experiment_id, args.trials, exclusions)
