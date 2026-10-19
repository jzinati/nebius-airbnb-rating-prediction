"""Project paths and constants kept in one easy-to-find module."""
from pathlib import Path

RANDOM_SEED = 42
TARGET = "review_scores_rating"
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("results")

NUMERIC_FEATURES = [
    "latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds",
    "price", "minimum_nights", "maximum_nights", "estimated_occupancy_l365d",
    "host_listings_count", "host_total_listings_count", "host_response_rate",
    "host_acceptance_rate", "host_tenure_days",
]
CATEGORICAL_FEATURES = [
    "property_type", "room_type", "host_response_time", "host_is_superhost",
    "instant_bookable",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Retained as raw listing-time inputs for controlled feature experiments.
AUXILIARY_FEATURES = ["amenities", "name", "description", "neighborhood_overview", "host_about"]
