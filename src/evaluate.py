"""Final evaluation on the untouched test set. This command does not tune models."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from src.metrics import regression_metrics
from src.settings import ARTIFACTS_DIR, FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET


def main() -> None:
    metrics_path = RESULTS_DIR / "final_test_metrics.csv"
    if metrics_path.exists():
        raise FileExistsError(
            "Final test metrics already exist. The test set has already been evaluated; "
            "do not rerun this command while making modelling decisions."
        )
    model = joblib.load(ARTIFACTS_DIR / "baseline_selected_model.joblib")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    predictions = model.predict(test[FEATURES])
    metrics = regression_metrics(test[TARGET], predictions)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame({"actual": test[TARGET], "prediction": predictions}).to_csv(
        RESULTS_DIR / "test_predictions.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
