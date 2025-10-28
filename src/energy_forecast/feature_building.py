from __future__ import annotations

import os
import re
from math import pi
from typing import Iterable, Union

import numpy as np
import pandas as pd


def build_merged(load_long: pd.DataFrame, temp_wide: pd.DataFrame) -> pd.DataFrame:
    # My merge:
    merged_all = load_long.merge(temp_wide.reset_index(), on="date_time", how="left")
    merged_all_cleaned = merged_all.dropna().reset_index(drop=True)
    if "date_time" not in load_long.columns:
        raise KeyError("'load_long' must contain date_time")
    if "date_time" not in temp_wide.columns and temp_wide.index.name != "date_time":
        temp_wide = temp_wide.copy()
        temp_wide.index.name = "date_time"
    merged_all = load_long.merge(temp_wide.reset_index(), on="date_time", how="left")
    merged_all_cleaned = merged_all.dropna().reset_index(drop=True)
    return merged_all_cleaned


def _normalize_holiday_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles cases where the holiday name is in the index or in the first column,
    and drops any non-year columns like 'Unnamed: 0' before parsing.
    """
    if df.index.name is None or df.index.equals(pd.RangeIndex(len(df))):
        first_col = df.columns[0]

        # Explicit year detection - better for C++/Python interop
        year_count = 0
        for col_name in df.columns:
            try:
                year_val = float(str(col_name))  # Ensure string conversion
                if 1900 <= year_val <= 2100:
                    year_count += 1
            except (ValueError, TypeError):
                continue

        if year_count > 0 and not pd.api.types.is_object_dtype(df[first_col]):
            df2 = df.rename_axis("holiday").reset_index()
        else:
            df2 = df.copy()
            if first_col.lower() != "holiday":
                df2 = df2.rename(columns={first_col: "holiday"})
    else:
        df2 = df.rename_axis("holiday").reset_index()

    return df2

    # Melt to long format
    long = df2.melt(id_vars=["holiday"], var_name="year", value_name="raw")

    # Keep only rows where 'year' is numeric (drop 'Unnamed: 0' etc.)
    long["year_num"] = pd.to_numeric(long["year"], errors="coerce")
    long = long[long["year_num"].notna()].copy()

    # Clean raw strings; drop blanks
    long["raw"] = long["raw"].astype(str).str.strip()
    long = long[~long["raw"].isin(["", "nan", "NaN"])]

    # Parse each cell: if text already contains a 4-digit year, parse as-is; else append the numeric year
    def _parse_row(row):
        txt = row["raw"].strip().strip('"').strip("'")
        yr = int(row["year_num"])
        txt = re.sub(r"\s+,", ",", txt)  # fix stray spaces before commas
        txt = re.sub(r"\s{2,}", " ", txt)  # collapse multiple spaces
        if re.search(r"\b\d{4}\b", txt):
            dt = pd.to_datetime(txt, errors="coerce")
        else:
            dt = pd.to_datetime(f"{txt}, {yr}", errors="coerce")
        return dt

    long["date"] = long.apply(_parse_row, axis=1)
    dates = long["date"].dropna().dt.normalize().drop_duplicates().sort_values()
    return pd.DataFrame({"date": dates})


def make_features(
    df: pd.DataFrame,
    holidays: Union[str, os.PathLike[str], pd.DataFrame],
    *,
    date_col: str = "date_time",
    temp_prefix: str = "temp_station_",
    base_c: float = 18.0,  # base temperature for HDD/CDD calculations (°C)
    doy_period: int = 366,  # leap-year friendly
    sr_mean: float = 6.5,  # 06:30 average sunrise time
    sr_amp: float = 1.5,  # +/- 1.5 hours amplitude (earliest sunrise ~05:00, latest ~08:00)
    ss_mean: float = 18.5,  # 18:30 average sunset time
    ss_amp: float = 2.0,  # +/- 2.0 hours amplitude (earliest sunset ~17:30, latest ~20:30)
) -> pd.DataFrame:
    """
    Adds:
      • cyclical encodings (hour/dow/doy)
      • weekend + holiday flags
      • approximate sunrise/sunset + daylight
      • HDD/CDD from all temp_station_* columns (°C)

    Parameters:
    - df: Input dataframe with a datetime column and temperature columns
    - holidays: DataFrame containing US holidays with a 'date' column
    - date_col: Name of the datetime column in df
    - temp_prefix: Prefix for temperature columns to consider for HDD/CDD calculations
    - base_c: Base temperature in Celsius for HDD/CDD calculations
    - doy_period: Period for cyclical encoding of day of year (366 to include leap year)
    - sr_mean, sr_amp: Mean and amplitude for sunrise time approximation
    - ss_mean, ss_amp: Mean and amplitude for sunset time approximation
    """
    out = df.copy()

    # ---- timestamp
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        raise ValueError(f"{date_col} contains NaT after parsing.")
    dt = out[date_col]

    out["hour"] = dt.dt.hour
    out["dow"] = dt.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    doy = dt.dt.dayofyear.astype(int)

    # ---- cyclical encodings
    out["hour_sin"] = np.sin(2 * pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * pi * out["dow"] / 7)
    out["doy_sin"] = np.sin(2 * pi * doy / doy_period)
    out["doy_cos"] = np.cos(2 * pi * doy / doy_period)

    # ---- holidays (accept path or DataFrame)
    if isinstance(holidays, (str, os.PathLike)):
        holidays_df = pd.read_csv(holidays)
    elif isinstance(holidays, pd.DataFrame):
        holidays_df = holidays.copy()
    else:
        raise ValueError("holidays must be a CSV path or a DataFrame")

    # normalize the wide holiday matrix into a single 'date' column
    holidays_norm = _normalize_holiday_matrix(holidays_df)
    hol_days = set(holidays_norm["date"].unique())
    cal_day = dt.dt.normalize()

    out["is_holiday"] = cal_day.isin(hol_days).astype(int)
    out["is_day_before_holiday"] = cal_day.isin(
        {d - pd.Timedelta(days=1) for d in hol_days}
    ).astype(int)
    out["is_day_after_holiday"] = cal_day.isin({d + pd.Timedelta(days=1) for d in hol_days}).astype(
        int
    )

    # sunrise/sunset-like approximation
    sunrise_hour = sr_mean - sr_amp * np.cos(2 * pi * doy / doy_period)
    sunset_hour = ss_mean + ss_amp * np.cos(2 * pi * doy / doy_period)

    out["sunrise_hour_approx"] = sunrise_hour
    out["sunset_hour_approx"] = sunset_hour
    out["daylight_hours_approx"] = sunset_hour - sunrise_hour
    out["is_daylight_approx"] = (
        (out["hour"] >= sunrise_hour.astype(int)) & (out["hour"] < sunset_hour.astype(int))
    ).astype(int)
    out["daylight_proxy"] = np.sin(2 * pi * doy / doy_period)

    # ---- HDD/CDD from temperature columns (°C)
    temp_cols = [c for c in out.columns if c.startswith(temp_prefix)]
    if not temp_cols:
        # fallback: any column containing 'temp'
        temp_cols = [c for c in out.columns if re.search(r"temp", c, re.IGNORECASE)]
    if temp_cols:
        out["temp_mean"] = out[temp_cols].mean(axis=1)
        out["HDD"] = (base_c - out["temp_mean"]).clip(lower=0)
        out["CDD"] = (out["temp_mean"] - base_c).clip(lower=0)

    return out


def add_exogenous_lags_for_features_df(
    df: pd.DataFrame,
    *,
    group_col: str = "zone_id",
    date_col: str = "date_time",
    # temperature-like columns
    temp_prefix: str = "station_",  # station_1..station_11
    include_extra_temp_cols: Iterable[str] = ("temp_mean", "HDD", "CDD"),
    temp_lags: tuple[int, ...] = (1, 3, 6, 24, 48, 168),  # lags in hours
    temp_rolls: tuple[int, ...] = (6, 24, 168),  # rolling windows in hours
    # holiday columns present in features_df
    holiday_cols: Iterable[str] = ("is_day_before_holiday", "is_day_after_holiday"),
    holiday_lags: tuple[int, ...] = (),  # usually not needed
    holiday_leads: tuple[int, ...] = (24, 168),  # usually only 1-day lead
) -> pd.DataFrame:
    """
    Adds lagged and rolling-window features for exogenous variables (non-target, non-ID columns)
    to be used in an ARIMAX model (ARIMA with exogenous regressors).

    Parameters:
    - df: Input dataframe with a datetime column and temperature columns
    - group_col: Column to group by (e.g., zone_id) for separate lagging per group
    - date_col: Name of the datetime column in df
    - temp_prefix: Prefix for temperature columns to consider for lagging/rolling
    - include_extra_temp_cols: Additional temperature-like columns to include (e.g., temp_mean, HDD, CDD)
    - temp_lags: Tuple of integer lags (in hours) to create for temperature-like columns
    - temp_rolls: Tuple of integer rolling window sizes (in hours) to create for temperature-like columns
    - holidays_cols: List of holiday-related binary columns to consider for lagging/leading
    - holiday_lags: Tuple of integer lags (in hours) to create for holiday columns
    - holiday_leads: Tuple of integer leads (in hours) to create for holiday columns

    Returns:
    - DataFrame with original and new lagged/rolling features
    """
    out = df.copy()
    out = out.sort_values([group_col, date_col], kind="mergesort")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # ---- collect temperature columns
    temp_cols = [c for c in out.columns if c.startswith(temp_prefix)]
    for c in include_extra_temp_cols:
        if c in out.columns:
            temp_cols.append(c)
    temp_cols = list(dict.fromkeys(temp_cols))  # de-duplicate while keeping order

    if not temp_cols:
        raise ValueError("No temperature columns found (station_* / temp_mean / HDD / CDD).")

    # ---- group per zone
    g = out.groupby(group_col, group_keys=False)

    # ---- temperature lags
    for L in temp_lags:
        for c in temp_cols:
            out[f"{c}_lag{L}"] = g[c].shift(L)

    # ---- temperature rolling means
    for W in temp_rolls:
        for c in temp_cols:
            out[f"{c}_roll{W}"] = g[c].transform(lambda s: s.rolling(W, min_periods=1).mean())

    # ---- holiday lags/leads
    hol_cols_present = [h for h in holiday_cols if h in out.columns]

    for L in holiday_lags:
        for h in hol_cols_present:
            out[f"{h}_lag{L}"] = g[h].shift(L)

    for H in holiday_leads:
        for h in hol_cols_present:
            out[f"{h}_lead{H}"] = g[h].shift(-H)

    return out


def ensure_calendar_features(df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add hour/dow/doy sin/cos"""
    out = df.copy()
    # Only compute if missing
    need_hour = not {"hour_sin", "hour_cos"}.issubset(out.columns)
    need_dow = not {"dow_sin", "dow_cos"}.issubset(out.columns)
    need_doy = not {"doy_sin", "doy_cos"}.issubset(out.columns)

    if need_hour or need_dow or need_doy:
        hour = date_index.hour
        dow = date_index.dayofweek
        doy = date_index.dayofyear
        two_pi = 2 * np.pi

        if need_hour:
            out["hour_sin"] = np.sin(two_pi * hour / 24.0)
            out["hour_cos"] = np.cos(two_pi * hour / 24.0)
        if need_dow:
            out["dow_sin"] = np.sin(two_pi * dow / 7.0)
            out["dow_cos"] = np.cos(two_pi * dow / 7.0)
        if need_doy:
            # use 366 to be leap-year friendly
            out["doy_sin"] = np.sin(two_pi * doy / 366.0)
            out["doy_cos"] = np.cos(two_pi * doy / 366.0)
    return out


def pick_xgb_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Choose exogenous features:
      - base weather + holiday flags
      - any engineered lags/rolls/leads
      - station_* columns
      - calendar sin/cos terms if present
    """
    keep_exact = {
        "temp_mean",
        "HDD",
        "CDD",
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "is_weekend",
        "daylight_proxy",  # optional if present
        "sunrise_hour_approx",
        "sunset_hour_approx",
        "daylight_hours_approx",
        "is_daylight_approx",
    }
    cols = []
    for c in df.columns:
        if c in ("zone_id", "load"):  # exclude id + target
            continue
        if (
            c in keep_exact
            or c.startswith("station_")
            or ("_lag" in c)
            or ("_roll" in c)
            or ("_lead" in c)
        ):
            cols.append(c)
    # de-dup, preserve order
    return list(dict.fromkeys(cols))
