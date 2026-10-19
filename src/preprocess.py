"""Create deterministic train, validation, and held-out test CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import clean_listings, read_raw_csvs, split_data
from src.settings import PROCESSED_DIR, RAW_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw listings and create 70/15/15 splits.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    cleaned = clean_listings(read_raw_csvs(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, split in split_data(cleaned).items():
        path = args.output / f"{name}.csv"
        split.to_csv(path, index=False)
        print(f"Wrote {path}: {len(split):,} rows")


if __name__ == "__main__":
    main()
