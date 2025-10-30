"""
ETL pipeline for building the merged training dataset.

Usage:
    uv run python -m energy_forecast.etl_build_dataset \
        --base-dir data/Load \
        --output-csv data/merged_training.csv \
        --holiday-csv data/Load/Holiday_List.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .feature_building import make_features, add_exogenous_lags_for_features_df


def clean_load_df(load_path: Path) -> pd.DataFrame:
    df = pd.read_csv(load_path)
    df.index = range(1, len(df) + 1)

    # remove commas, convert to float
    cols = [f"h{i}" for i in range(1, 25)]
    for c in cols:
        df[c] = df[c].str.replace(",", "").astype(float)

    # fill NaNs horizontally and vertically
    df[cols] = df[cols].interpolate(method="linear", axis=1, limit_direction="both")
    df[cols] = df[cols].interpolate(method="linear", axis=0, limit_direction="both")

    # reshape to long format
    long = df.melt(
        id_vars=["zone_id", "year", "month", "day"],
        value_vars=cols,
        var_name="hour",
        value_name="load",
    )
    long["hour"] = long["hour"].str.extract("h(\\d+)").astype(int)
    long["date_time"] = pd.to_datetime(
        long[["year", "month", "day"]]
    ) + pd.to_timedelta(long["hour"] - 1, unit="h")
    return (
        long[["zone_id", "date_time", "load"]]
        .sort_values(["zone_id", "date_time"])
        .reset_index(drop=True)
    )


def clean_temp_df(temp_path: Path) -> pd.DataFrame:
    df = pd.read_csv(temp_path)
    df.index = range(1, len(df) + 1)

    cols = [f"h{i}" for i in range(1, 25)]
    for col in [c for c in cols if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # vertical interpolation
    df[cols] = df[cols].interpolate(method="linear", axis=0, limit_direction="both")

    # reshape to long
    temp_long = df.melt(
        id_vars=["station_id", "year", "month", "day"],
        value_vars=cols,
        var_name="hour",
        value_name="temperature",
    )
    temp_long["hour"] = temp_long["hour"].str.extract("h(\\d+)").astype(int)
    temp_long["date_time"] = pd.to_datetime(
        temp_long[["year", "month", "day"]]
    ) + pd.to_timedelta(temp_long["hour"] - 1, unit="h")

    # pivot to wide (one column per station)
    temp_wide = temp_long.pivot(
        index="date_time", columns="station_id", values="temperature"
    )
    temp_wide.columns = [f"station_{c}" for c in temp_wide.columns]
    return temp_wide.reset_index()


def build_merged_dataset(load_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    merged = load_df.merge(temp_df, on="date_time", how="left")
    merged = merged.dropna().reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL for merged training dataset")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base directory containing Load and temperature data",
    )
    parser.add_argument(
        "--holiday-csv", type=Path, required=True, help="Path to Holiday_List.csv"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for merged dataset",
    )
    args = parser.parse_args()

    load_path = args.base_dir / "load_history.csv"
    temp_path = args.base_dir / "temperature_history.csv"
    holiday_path = args.holiday_csv

    print(f"[ETL] Loading data from {args.base_dir}")
    load_df = clean_load_df(load_path)
    temp_df = clean_temp_df(temp_path)

    print("[ETL] Merging...")
    merged = build_merged_dataset(load_df, temp_df)
    print(f"  Merged shape: {merged.shape}")

    print("[ETL] Feature engineering...")
    features = make_features(merged, holidays=holiday_path, temp_prefix="station_")
    features = add_exogenous_lags_for_features_df(features)
    features = features.dropna().reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)
    print(f"[ETL] Saved merged dataset → {args.output_csv}")
    print(f"[ETL] Final shape: {features.shape}")


if __name__ == "__main__":
    main()
