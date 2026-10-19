"""Local Streamlit dashboard for the experiment record.

Run with: ``uv run streamlit run app.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
PUBLISHED_RESULTS = ROOT / "docs" / "published_validation_results.csv"


def _parameters_to_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Turn JSON parameter dictionaries into readable table columns."""
    if data.empty or "parameters" not in data:
        return data
    parameters = pd.json_normalize(data["parameters"].map(json.loads)).add_prefix("param_")
    return pd.concat([data.reset_index(drop=True), parameters], axis=1)


@st.cache_data
def load_experiments() -> pd.DataFrame:
    """Prefer generated local metrics; fall back to committed executed summaries."""
    files = sorted(RESULTS_DIR.glob("*_validation_metrics.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["experiment_id"] = path.name.removesuffix("_validation_metrics.csv")
        frames.append(frame)
    published = pd.read_csv(PUBLISHED_RESULTS)
    if not frames:
        return published
    local = pd.concat(frames, ignore_index=True)
    missing_summaries = published[~published["experiment_id"].isin(local["experiment_id"])]
    return pd.concat([local, missing_summaries], ignore_index=True)


@st.cache_data
def load_optuna_trials() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("*_trials.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["experiment_id"] = path.name.removesuffix("_trials.csv")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metric_cards(row: pd.Series) -> None:
    left, middle, right = st.columns(3)
    left.metric("Validation RMSE", f"{row['rmse']:.4f}")
    middle.metric("Validation MAE", f"{row['mae']:.4f}")
    right.metric("Validation R²", f"{row['r2']:.4f}")


st.set_page_config(page_title="Airbnb rating experiments", page_icon="🏠", layout="wide")
st.title("Airbnb rating prediction — experiment dashboard")
st.caption("Local, validation-led experiment record. The original test set is not used for tuning.")
st.info(
    "Read the labels carefully: results including `host_is_superhost` are historical, "
    "leakage-sensitive comparisons. The recommended prospective candidate excludes it."
)

experiments = load_experiments()
if experiments.empty:
    st.error("No experiment summaries are available.")
    st.stop()

experiments["rmse"] = pd.to_numeric(experiments["rmse"])
experiments["mae"] = pd.to_numeric(experiments["mae"])
experiments["r2"] = pd.to_numeric(experiments["r2"])

summary = experiments.sort_values("rmse").groupby("experiment_id", as_index=False).first()
prospective = summary[summary["excluded_features"].astype(str).str.contains("host_is_superhost")]
recommended = prospective.sort_values("rmse").iloc[0]
historical = summary.sort_values("rmse").iloc[0]

st.subheader("What the current evidence supports")
recommended_column, historical_column, test_column = st.columns(3)
with recommended_column:
    st.markdown("**Recommended prospective candidate**")
    st.caption(recommended["experiment_id"])
    st.metric("Validation RMSE", f"{recommended['rmse']:.4f}")
    st.caption("F4 without Superhost; validation only.")
with historical_column:
    st.markdown("**Best historical comparison**")
    st.caption(historical["experiment_id"])
    st.metric("Validation RMSE", f"{historical['rmse']:.4f}")
    st.caption("May include leakage-sensitive Superhost.")
with test_column:
    st.markdown("**One final test result**")
    st.metric("Test RMSE", "0.4224")
    st.caption("Run 0 XGBoost only; not reused after feature selection.")

st.subheader("Executed feature experiments")
chart_data = summary.set_index("experiment_id")[["rmse", "mae"]].sort_values("rmse")
st.bar_chart(chart_data)
st.caption("Each bar is the best recorded validation candidate for that experiment. Lower is better.")

selected_id = st.selectbox("Inspect an experiment", options=summary["experiment_id"].tolist(), index=0)
selected = experiments[experiments["experiment_id"] == selected_id].sort_values("rmse")
best = selected.iloc[0]
st.subheader(f"{selected_id}: best candidate")
metric_cards(best)
st.write(f"**Model:** `{best['model']}`  ")
st.write(f"**Feature set:** `{best['feature_set']}`  ")
st.write(f"**Excluded features:** `{best['excluded_features']}`")
st.json(json.loads(best["parameters"]))

with st.expander("All candidates and every searched hyperparameter", expanded=False):
    table = _parameters_to_columns(selected)
    visible = [column for column in table.columns if column != "candidate_id"]
    st.dataframe(table[visible], use_container_width=True, hide_index=True)

optuna_trials = load_optuna_trials()
st.subheader("Optuna tuning runs")
if optuna_trials.empty:
    st.caption("No Optuna run has been executed yet. Use the command below; the dashboard refreshes when the trial CSV is written.")
else:
    optuna_id = st.selectbox("Inspect an Optuna run", optuna_trials["experiment_id"].unique())
    optuna_selected = optuna_trials[optuna_trials["experiment_id"] == optuna_id].sort_values("rmse")
    st.line_chart(optuna_selected.sort_values("trial").set_index("trial")["rmse"])
    st.caption("Validation RMSE by trial. The search never reads the test split.")
    metric_cards(optuna_selected.iloc[0])
    with st.expander("Every Optuna trial and hyperparameter", expanded=False):
        st.dataframe(_parameters_to_columns(optuna_selected), use_container_width=True, hide_index=True)

with st.expander("How to reproduce or extend", expanded=False):
    st.code(
        "uv sync\n"
        "uv run python -m src.preprocess\n"
        "uv run python -m src.tune --trials 20\n"
        "uv run streamlit run app.py",
        language="bash",
    )
    st.write(
        "The tuning command fits candidates on `train.csv` and selects by `validation.csv`. "
        "Do not run `src.evaluate` for a tuned configuration until a fresh, untouched holdout exists."
    )
