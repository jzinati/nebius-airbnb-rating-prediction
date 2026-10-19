"""Feature-set definitions used during controlled experiments."""
from __future__ import annotations

from dataclasses import dataclass
import ast
from collections import Counter

import numpy as np
import pandas as pd

from src.settings import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class FeatureSet:
    """A feature matrix plus the column lists needed by the model pipeline."""

    data: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio, treating zero and missing denominators as missing values."""
    safe_denominator = denominator.where(denominator > 0)
    return numerator / safe_denominator


def _parse_amenities(value: object) -> list[str]:
    """Parse an Airbnb amenities list without executing untrusted input."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return [str(item).lower() for item in parsed] if isinstance(parsed, list) else []


def _contains(amenities: list[str], phrase: str) -> int:
    return int(any(phrase in item for item in amenities))


def fit_amenity_vocabulary(data: pd.DataFrame, limit: int = 50) -> list[str]:
    """Learn frequent amenities from training rows only (no target information)."""
    counts: Counter[str] = Counter()
    for values in data["amenities"].map(_parse_amenities):
        counts.update(values)
    return [amenity for amenity, _ in counts.most_common(limit)]


def make_feature_set(
    data: pd.DataFrame, name: str = "base", amenity_vocabulary: list[str] | None = None
) -> FeatureSet:
    """Build an explainable feature set without using the target column.

    ``f1_price_capacity`` adds deterministic transformations available from a
    listing itself. It intentionally does not use text, review counts, or target
    encodings.
    """
    frame = data[FEATURES].copy()
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)

    if name == "base":
        return FeatureSet(frame, numeric, categorical)
    if name not in {"f1_price_capacity", "f2_amenities", "f3_top_amenities", "f4_missingness"}:
        raise ValueError(f"Unknown feature set: {name}")

    additions = {
        "log_price": np.log1p(frame["price"].clip(lower=0)),
        "log_minimum_nights": np.log1p(frame["minimum_nights"].clip(lower=0)),
        "log_maximum_nights": np.log1p(frame["maximum_nights"].clip(lower=0)),
        "log_host_listings_count": np.log1p(frame["host_listings_count"].clip(lower=0)),
        "price_per_guest": _ratio(frame["price"], frame["accommodates"]),
        "beds_per_guest": _ratio(frame["beds"], frame["accommodates"]),
        "bedrooms_per_guest": _ratio(frame["bedrooms"], frame["accommodates"]),
        "bathrooms_per_guest": _ratio(frame["bathrooms"], frame["accommodates"]),
    }
    frame = frame.assign(**additions).replace([np.inf, -np.inf], np.nan)
    numeric.extend(additions)
    if name == "f1_price_capacity":
        return FeatureSet(frame, numeric, categorical)

    amenities = data["amenities"].map(_parse_amenities)
    amenity_additions = {
        "amenity_count": amenities.map(len),
        "has_wifi": amenities.map(lambda values: _contains(values, "wifi")),
        "has_kitchen": amenities.map(lambda values: _contains(values, "kitchen")),
        "has_air_conditioning": amenities.map(lambda values: _contains(values, "air conditioning")),
        "has_parking": amenities.map(lambda values: _contains(values, "parking")),
        "has_washer": amenities.map(lambda values: _contains(values, "washer")),
        "has_dryer": amenities.map(lambda values: _contains(values, "dryer")),
        "has_self_check_in": amenities.map(lambda values: _contains(values, "self check-in")),
        "has_workspace": amenities.map(lambda values: _contains(values, "workspace")),
        "has_smoke_alarm": amenities.map(lambda values: _contains(values, "smoke alarm")),
        "has_fire_extinguisher": amenities.map(lambda values: _contains(values, "fire extinguisher")),
        "has_elevator": amenities.map(lambda values: _contains(values, "elevator")),
    }
    frame = frame.assign(**amenity_additions)
    numeric.extend(amenity_additions)

    if name == "f2_amenities":
        return FeatureSet(frame, numeric, categorical)
    if amenity_vocabulary is None:
        raise ValueError("f3_top_amenities requires a vocabulary fitted on training data.")

    vocabulary_additions = {
        f"top_amenity_{index:02d}": amenities.map(lambda values, term=amenity: int(term in values))
        for index, amenity in enumerate(amenity_vocabulary)
    }
    frame = frame.assign(**vocabulary_additions)
    numeric.extend(vocabulary_additions)
    if name == "f3_top_amenities":
        return FeatureSet(frame, numeric, categorical)

    missingness_columns = [
        "price", "beds", "bedrooms", "host_response_time", "host_response_rate",
        "host_acceptance_rate", "bathrooms",
    ]
    missingness_additions = {
        f"{column}_was_missing": data[column].isna().astype(int)
        for column in missingness_columns
    }
    frame = frame.assign(**missingness_additions)
    numeric.extend(missingness_additions)
    return FeatureSet(frame, numeric, categorical)
