import pandas as pd

from src.data import clean_listings, split_data
from src.features import fit_amenity_vocabulary, make_feature_set
from src.settings import FEATURES, TARGET


def test_cleaning_and_split_keep_features_and_target():
    raw = pd.DataFrame({
        "review_scores_rating": [4.0, 4.5, 4.8, 5.0] * 10,
        "price": ["$100.00"] * 40,
        "bathrooms_text": ["1 bath"] * 40,
        "host_since": ["2020-01-01"] * 40,
        "property_type": ["Apartment"] * 40,
        "room_type": ["Entire home/apt"] * 40,
    })
    cleaned = clean_listings(raw)
    assert set(FEATURES + [TARGET]).issubset(cleaned.columns)
    splits = split_data(cleaned)
    assert sum(len(split) for split in splits.values()) == len(cleaned)


def test_f1_feature_set_creates_safe_capacity_ratios():
    data = pd.DataFrame({
        **{column: [1.0, 2.0] for column in FEATURES if column not in {"property_type", "room_type", "host_response_time", "host_is_superhost", "instant_bookable"}},
        "property_type": ["Apartment", "House"],
        "room_type": ["Entire home/apt", "Private room"],
        "host_response_time": ["within an hour", "within a day"],
        "host_is_superhost": ["yes", "no"],
        "instant_bookable": ["yes", "no"],
    })
    data.loc[0, "accommodates"] = 0
    feature_set = make_feature_set(data, "f1_price_capacity")
    assert "log_price" in feature_set.numeric_features
    assert pd.isna(feature_set.data.loc[0, "price_per_guest"])


def test_f2_feature_set_counts_amenities_and_flags():
    data = pd.DataFrame({
        **{column: [1.0] for column in FEATURES if column not in {"property_type", "room_type", "host_response_time", "host_is_superhost", "instant_bookable"}},
        "property_type": ["Apartment"],
        "room_type": ["Entire home/apt"],
        "host_response_time": ["within an hour"],
        "host_is_superhost": ["no"],
        "instant_bookable": ["yes"],
        "amenities": ["['Wifi', 'Kitchen', 'Free parking on premises']"],
    })
    feature_set = make_feature_set(data, "f2_amenities")
    assert feature_set.data.loc[0, "amenity_count"] == 3
    assert feature_set.data.loc[0, "has_wifi"] == 1
    assert feature_set.data.loc[0, "has_elevator"] == 0


def test_f3_top_amenities_uses_training_vocabulary():
    data = pd.DataFrame({
        **{column: [1.0, 1.0] for column in FEATURES if column not in {"property_type", "room_type", "host_response_time", "host_is_superhost", "instant_bookable"}},
        "property_type": ["Apartment", "Apartment"],
        "room_type": ["Entire home/apt", "Entire home/apt"],
        "host_response_time": ["within an hour", "within an hour"],
        "host_is_superhost": ["no", "no"],
        "instant_bookable": ["yes", "yes"],
        "amenities": ["['Wifi', 'Kitchen']", "['Wifi']"],
    })
    vocabulary = fit_amenity_vocabulary(data, limit=1)
    feature_set = make_feature_set(data, "f3_top_amenities", vocabulary)
    assert vocabulary == ["wifi"]
    assert list(feature_set.data["top_amenity_00"]) == [1, 1]


def test_f4_missingness_retains_the_observed_missingness_signal():
    data = pd.DataFrame({
        **{column: [1.0, 1.0] for column in FEATURES if column not in {"property_type", "room_type", "host_response_time", "host_is_superhost", "instant_bookable", "price"}},
        "price": [float("nan"), 100.0],
        "property_type": ["Apartment", "Apartment"],
        "room_type": ["Entire home/apt", "Entire home/apt"],
        "host_response_time": ["within an hour", "within an hour"],
        "host_is_superhost": ["no", "no"],
        "instant_bookable": ["yes", "yes"],
        "amenities": ["['Wifi']", "['Wifi']"],
    })
    feature_set = make_feature_set(data, "f4_missingness", ["wifi"])
    assert list(feature_set.data["price_was_missing"]) == [1, 0]
