"""Loading, light cleaning, and leakage-safe data splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.settings import AUXILIARY_FEATURES, FEATURES, RANDOM_SEED, TARGET


def _parse_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"
    )


def clean_listings(raw: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Select understandable listing features without fitting any dataset statistics.

    Imputation and categorical encoding deliberately happen later inside the fitted
    model pipeline, which prevents validation/test information leaking into training.
    """
    if require_target and TARGET not in raw:
        raise ValueError(f"Input data must contain the target column '{TARGET}'.")

    frame = raw.copy()
    for column in ("price", "host_response_rate", "host_acceptance_rate"):
        if column in frame:
            frame[column] = _parse_number(frame[column])
            if column != "price":
                frame[column] = frame[column] / 100

    if "bathrooms_text" in frame:
        extracted = frame["bathrooms_text"].astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0]
        frame["bathrooms"] = frame.get("bathrooms", pd.Series(index=frame.index)).fillna(
            pd.to_numeric(extracted, errors="coerce")
        )

    for column in ("host_is_superhost", "instant_bookable"):
        if column in frame:
            frame[column] = frame[column].replace({"t": "yes", "f": "no"}).astype("string")

    if "host_since" in frame:
        host_since = pd.to_datetime(frame["host_since"], errors="coerce")
        reference = pd.Timestamp("2025-01-01")
        frame["host_tenure_days"] = (reference - host_since).dt.days.clip(lower=0)

    for column in FEATURES:
        if column not in frame:
            frame[column] = pd.NA

    for column in AUXILIARY_FEATURES:
        if column not in frame:
            frame[column] = pd.NA

    columns = FEATURES + AUXILIARY_FEATURES + ([TARGET] if TARGET in frame else [])
    cleaned = frame[columns].copy()
    if TARGET in cleaned:
        cleaned[TARGET] = pd.to_numeric(cleaned[TARGET], errors="coerce")
        if require_target:
            # Keep the same row identity as the original basic-feature benchmark;
            # auxiliary raw text must not change which duplicate is retained.
            cleaned = cleaned.dropna(subset=[TARGET]).drop_duplicates(subset=FEATURES + [TARGET])
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def read_raw_csvs(raw_path: Path) -> pd.DataFrame:
    """Read one CSV or all CSVs in a directory; each must contain the target."""
    paths = [raw_path] if raw_path.is_file() else sorted(raw_path.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found at {raw_path}. See README.md for the expected input."
        )
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    return pd.concat(frames, ignore_index=True)


def split_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create fixed 70/15/15 train/validation/test splits.

    The test set is created first and is never passed to training or model selection.
    """
    train_validation, test = train_test_split(
        data, test_size=0.15, random_state=RANDOM_SEED
    )
    validation_share = 0.15 / 0.85
    train, validation = train_test_split(
        train_validation, test_size=validation_share, random_state=RANDOM_SEED
    )
    return {"train": train, "validation": validation, "test": test}
