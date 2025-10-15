# import neccessary libarries

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from math import pi
from typing import Iterable, Dict, Tuple, List
import statsmodels.api as sm
#from __future__ import annotations
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import xgboost as xgb
from xgboost.callback import EarlyStopping
from xgboost import XGBRegressor
from lime.lime_tabular import LimeTabularExplainer

#load the data
base_dir = Path('/Users/madalinaciolan/dev/explainable_ai')

#load the energy demand data(load)
load_path = base_dir / 'data' / 'Load' / 'load_history.csv'
load_df = pd.read_csv(load_path)

#load the temperature data
temp_path = base_dir / 'data' / 'Load' / 'temperature_history.csv'
temp_df = pd.read_csv(temp_path)

#fix the index for both dataframes
load_df.index = range(1, len(load_df) + 1)
print(load_df.head())

temp_df.index = range(1, len(temp_df) + 1)
print(temp_df.head())

#check the info and missing values for both dataframes
print(load_df.info())
print(load_df.isna().sum())

print(temp_df.info())
print(temp_df.isna().sum())

# ---- Clean the load dataframe ----   
load_df_clean = load_df.copy()
cols = [f"h{i}" for i in range(1, 25)]
for col in cols:
    load_df_clean[col] = load_df_clean[col].str.replace(",", "").astype(float)

print(load_df_clean.dtypes)
print(load_df_clean.isna().sum())

# --- Fill NaNs in load_df_clean using interpolation before/after ---
# 1) Horizontal interpolation within each day across h1..h24
#    (fills gaps inside a day's 24 hours using surrounding hours)
load_df_clean[cols] = load_df_clean[cols].interpolate(method='linear', axis=1, limit_direction='both')

# 2) Vertical interpolation across days for each hour h1..h24
#    (fills gaps across days using surrounding days for each hour)
load_df_clean[cols] = load_df_clean[cols].interpolate(method='linear',  axis=0, limit_direction='both')

# check if all NaNs in the load data frame have been filled
print("\nRemaining NaNs per column after interpolation: ", load_df_clean[cols].isna().sum())

# ---- Clean the temperature dataframe ----  
print(load_df_clean.dtypes) #no need to change types, they are already integers and floats
print(temp_df.isna().sum()) #NaNs only in temperature colums h7 -> h24
temp_df_clean = temp_df.copy()

# --- Fill NaNs in load_df_clean using interpolation before/after ---
#  Vertical interpolation across days for each hour h7..h24
#    (fills gaps across days using surrounding days for each hour)
cols_temp = [f"h{i}" for i in range(7, 25)]
for col in cols_temp:
    temp_df_clean[col] = temp_df_clean[col].interpolate(method='linear', axis=0, limit_direction='both')    

#check if all NaNs in the temperature data frame have been filled
print("\nRemaining NaNs per column after interpolation: ", temp_df_clean.isna().sum()) 

print(load_df_clean.head())    

# ----Reshaping the load dataframe from wide to long format ----

# Melt the dataframe to have a long format with columns: year, month, day, hour, load
hours_cols = [f"h{i}" for i in range(1, 25)]
load_long = load_df_clean.melt(
    id_vars=["zone_id", "year", "month", "day"],
    value_vars=hours_cols,
    var_name="hour",
    value_name="load",
)

# Convert hour column to numeric hour
load_long['hour'] = load_long['hour'].str.extract('h(\d+)').astype(int)

# Create a datetime column
load_long['date_time'] = pd.to_datetime(load_long[['year', 'month', 'day']]) + pd.to_timedelta(load_long['hour'] - 1, unit='h')

#Keep only relevant columns and sort by date
load_long = load_long[['zone_id','date_time', 'load']].sort_values(['zone_id', 'date_time']).reset_index(drop=True)
print(load_long.head())

print(temp_df_clean.head())

# ----Reshaping the temperature dataframe from wide to long format ----

# Melt the dataframe to have a long format with columns: year, month, day, hour, temperature
temp_hour_cols = [f"h{i}" for i in range(1, 25)]
temp_long = temp_df_clean.melt(
    id_vars=["station_id", "year", "month", "day"],
    value_vars=temp_hour_cols,
    var_name="hour",
    value_name="temperature",
)

# Convert hour column to numeric hour
temp_long['hour'] = temp_long['hour'].str.extract('h(\d+)').astype(int)

# Create a datetime column
temp_long['date_time'] = pd.to_datetime(temp_long[['year', 'month', 'day']]) + pd.to_timedelta(temp_long['hour'] - 1, unit='h')

#Keep only relevant columns and sort by date
#temp_long = temp_long[['station_id','date_time', 'temperature']].sort_values(by='date_time').reset_index(drop=True)
#print(temp_long.head())

# Pivot so each station is one column
temp_wide = temp_long.pivot(index='date_time', columns='station_id', values='temperature')
temp_wide.columns = [f"station_{col}" for col in temp_wide.columns]

print(temp_long.head())

# --- Merge: attach all stations' temperatures to each load record ---
merged_all = load_long.merge(temp_wide.reset_index(), on='date_time', how='left')
print(merged_all)

merged_all.isnull().sum() #check for missing values after the merge

merged_all_cleaned = merged_all.dropna().reset_index(drop=True) #drop rows with any NaNs and reset index
print(merged_all_cleaned)

merged_all_cleaned.isnull().sum()

holidays = pd.read_csv(base_dir / 'data' / 'Load' / 'Holiday_List.csv')
print(type(holidays))
print(holidays.head())


# --- Robust: convert wide holiday matrix (rows=holiday names, cols=years) into tidy 'date' list
def _normalize_holiday_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles cases where the holiday name is in the index or in the first column,
    and drops any non-year columns like 'Unnamed: 0' before parsing.
    """
    # If holiday names are the index, promote to a column named 'holiday'
    if df.index.name is None or df.index.equals(pd.RangeIndex(len(df))):
        # Try to detect when the first column is the holiday name
        first_col = df.columns[0]
        # If the first column looks like year (numeric), then names must be in the index -> add from index
        if pd.to_numeric(pd.Index(df.columns), errors="coerce").notna().sum() > 0 and not pd.api.types.is_object_dtype(df[first_col]):
            df2 = df.rename_axis("holiday").reset_index()
        else:
            # Assume first column holds holiday names
            df2 = df.copy()
            if first_col.lower() != "holiday":
                df2 = df2.rename(columns={first_col: "holiday"})
    else:
        df2 = df.rename_axis("holiday").reset_index()

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
        yr  = int(row["year_num"])
        txt = re.sub(r"\s+,", ",", txt)          # fix stray spaces before commas
        txt = re.sub(r"\s{2,}", " ", txt)        # collapse multiple spaces
        if re.search(r"\b\d{4}\b", txt):
            dt = pd.to_datetime(txt, errors="coerce")
        else:
            dt = pd.to_datetime(f"{txt}, {yr}", errors="coerce")
        return dt

    long["date"] = long.apply(_parse_row, axis=1)
    dates = (long["date"]
             .dropna()
             .dt.normalize()
             .drop_duplicates()
             .sort_values())
    return pd.DataFrame({"date": dates})


def make_features(
    df: pd.DataFrame,
    holidays,                     # can be a path or a wide-format DataFrame
    *,
    date_col: str = "date_time",
    temp_prefix: str = "temp_station_",
    base_c: float = 18.0,       # base temperature for HDD/CDD calculations (°C)
    doy_period: int = 366,      # leap-year friendly
    sr_mean: float = 6.5,       # 06:30 average sunrise time
    sr_amp: float = 1.5,        # +/- 1.5 hours amplitude (earliest sunrise ~05:00, latest ~08:00) 
    ss_mean: float = 18.5,      # 18:30 average sunset time
    ss_amp: float = 2.0         # +/- 2.0 hours amplitude (earliest sunset ~17:30, latest ~20:30)
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
    out["dow_sin"]  = np.sin(2 * pi * out["dow"] / 7)
    out["dow_cos"]  = np.cos(2 * pi * out["dow"] / 7)
    out["doy_sin"]  = np.sin(2 * pi * doy / doy_period)
    out["doy_cos"]  = np.cos(2 * pi * doy / doy_period)

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

    out["is_holiday"]            = cal_day.isin(hol_days).astype(int)
    out["is_day_before_holiday"] = cal_day.isin({d - pd.Timedelta(days=1) for d in hol_days}).astype(int)
    out["is_day_after_holiday"]  = cal_day.isin({d + pd.Timedelta(days=1) for d in hol_days}).astype(int)

    # sunrise/sunset-like approximation
    sunrise_hour = sr_mean - sr_amp * np.cos(2 * pi * doy / doy_period)
    sunset_hour  = ss_mean + ss_amp * np.cos(2 * pi * doy / doy_period)

    out["sunrise_hour_approx"]   = sunrise_hour
    out["sunset_hour_approx"]    = sunset_hour
    out["daylight_hours_approx"] = sunset_hour - sunrise_hour
    out["is_daylight_approx"]    = ((out["hour"] >= sunrise_hour.astype(int)) &
                                    (out["hour"] <  sunset_hour.astype(int))).astype(int)
    out["daylight_proxy"]        = np.sin(2 * pi * doy / doy_period)

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

features_df = make_features(merged_all_cleaned, holidays, temp_prefix='station_')
print(features_df.head())     


def add_exogenous_lags_for_features_df(
    df: pd.DataFrame,
    *,
    group_col: str = "zone_id",
    date_col: str = "date_time",
    # temperature-like columns
    temp_prefix: str = "station_",                              # station_1..station_11
    include_extra_temp_cols: Iterable[str] = ("temp_mean", "HDD", "CDD"),
    temp_lags: tuple[int, ...] = (1, 3, 6, 24, 48, 168),        # lags in hours
    temp_rolls: tuple[int, ...] = (6, 24, 168),                 # rolling windows in hours
    # holiday columns present in features_df
    holiday_cols: Iterable[str] = ("is_day_before_holiday", "is_day_after_holiday"),
    holiday_lags: tuple[int, ...] = (),                         # usually not needed
    holiday_leads: tuple[int, ...] = (24, 168)                  # usually only 1-day lead
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

features_with_lags = add_exogenous_lags_for_features_df(
    features_df,                    # my DataFrame
    group_col="zone_id",
    date_col="date_time",
    temp_prefix="station_",         # station_1..station_11
    include_extra_temp_cols=("temp_mean", "HDD", "CDD"),
    temp_lags=(1, 3, 6, 24, 48, 168),
    temp_rolls=(6, 24, 168),
    holiday_cols=("is_day_before_holiday", "is_day_after_holiday"),
    holiday_lags=(),                # usually leave empty
    holiday_leads=(24, 168)         # holidays are known → safe leads
)

# Sort and check continuity per zone_id
features_with_lags = features_with_lags.sort_values(["zone_id", "date_time"]).reset_index(drop=True)

print(features_with_lags.head())

# Check time step gaps per zone_id
gaps = (
    features_with_lags.groupby("zone_id")["date_time"]          #handle multiple zones separately
    .diff()                                                     #computes the time difference between consecutive timestamps within each zone
    .dropna()                                                   #removes the first row in each group(since it has no previous timestamp to diff against)
    .value_counts()                                             #counts how many times each time gap occurs across the dataset
)
print(gaps.head())
print(gaps.isna().sum())

# Check for missing values 
features_with_lags.isna().sum().sort_values(ascending=False).head(20)

"""
Lagged and lead features introduced some NaNs at the start/end of each zone_id group:
- when shifting the time series backward by 168 hrs(7 days), the first 168 rows of each zone_id cannot be filled in and will have NaNs for those lagged features
- when shifting the time series forward by 168 hrs(7 days), the last 168 rows of each zone_id have "no future data" to shift into them.

So:
    - for station_1_lag168, station_2_lag168, ..., station_11_lag168, the first 168 timestamps for each zone_id will be NaN.
    - for is_day_before_holiday_lead168, is_day_after_holiday_lead168, the last 168 timestamps for each zone_id will be NaN.

"""

# Missing values percentage is 0.02% therefore we can drop the rows with NaN
features_with_lags = features_with_lags.dropna().reset_index(drop=True)
print((features_with_lags).isna().sum().sort_values(ascending=False).head(10))

# Check for infinite values
np.isinf(features_with_lags).sum()

print(features_with_lags.head())

# ---- SARIMAX model fitting and evaluation ----
# Running SARIMAX on all zones is too slow, therefore we will run it on a single zone)

# Quick test on a single zone
# 1) Filter one zone and enforce hourly frequency
zone = 1
df_zone = (features_with_lags
           .loc[features_with_lags["zone_id"] == zone]
           .sort_values("date_time")
           .set_index("date_time")
           .asfreq("h"))
# 2) Select target and exogenous columns
cols_exog = ["temp_mean", "HDD", "CDD", "is_holiday"]  
missing_cols = [c for c in cols_exog if c not in df_zone.columns]
if missing_cols:
    raise KeyError(f"Exogenous columns missing: {missing_cols}")

y = df_zone["load"]
X = df_zone[cols_exog].apply(pd.to_numeric, errors="coerce")
if "is_holiday" in X:
    X["is_holiday"] = X["is_holiday"].fillna(0).astype(int)

# 3) Clean rows: no NaN or inf values in y or X
mask = (y.notna() & np.isfinite(y) & X.notna().all(axis=1) & np.isfinite(X).all(axis=1))
y_clean = y[mask]
X_clean = X[mask]

# 4) Fit SARIMAX on the clean, aligned data
model = sm.tsa.SARIMAX(
    endog=y_clean,
    exog=X_clean,
    order=(1,1,1),
    seasonal_order=(1,0,1,24),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
res = model.fit(disp=False)
print(res.summary())


# 5) In-sample metrics (fitted vs actual)
# Note: fittedvalues are one-step-ahead predictions given the model/state.
fitted = res.fittedvalues
y_aligned, f_aligned = y_clean.align(fitted, join="inner")

mse_in  = mean_squared_error(y_aligned, f_aligned)
rmse_in = np.sqrt(mse_in)
mae_in  = mean_absolute_error(y_aligned, f_aligned)

print(f"\nIn-sample:  MSE={mse_in:.2f}  RMSE={rmse_in:.2f}  MAE={mae_in:.2f}")

# 6) Chronological holdout (e.g., last 7 days)
H = 24 * 7  # forecast horizon
if len(y_clean) > H + 1:
    cutoff = y_clean.index[-(H+1)]
    y_train, X_train = y_clean.loc[:cutoff], X_clean.loc[:cutoff]
    y_test,  X_test  = y_clean.loc[cutoff + pd.Timedelta(hours=1):], X_clean.loc[cutoff + pd.Timedelta(hours=1):]

    res_tr = sm.tsa.SARIMAX(
        endog=y_train,
        exog=X_train,
        order=(1,1,1),
        seasonal_order=(1,0,1,24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    fc = res_tr.get_forecast(steps=len(y_test), exog=X_test)
    y_pred = fc.predicted_mean

    # Test metrics 
    mse_te  = mean_squared_error(y_test, y_pred)
    rmse_te = np.sqrt(mse_te)
    mae_te  = mean_absolute_error(y_test, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-9))

    print(f"Holdout ({H}h): MSE={mse_te:.2f}  RMSE={rmse_te:.2f}  MAE={mae_te:.2f}  sMAPE={smape:.2f}%")
else:
    print("\n[Skip holdout metrics] Not enough data for a 7-day test split.")


# ----- Move to XGBoost for better accuracy and use SHAP for interpretability ----
# The below XGBoost model uses the engineered lag/rolling features to predict the load for a single zone. It includes calendar sin/cos terms (hour, dow, doy), holiday flags, 
# approximate sunrise/sunset features, and lagged/rolling temperature features.
# Reports MSE, RMSE, MAE, sMAPE both in-sample and on a chronological 7-day holdout period at the end of the data.


# helpers (XGBoost does not know about cyclical encodings, so we add them as calendar sin/cos terms which are very helpful to express seasonality. SARIMAX already has built-in seasonality handling.

def ensure_calendar_features(df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add hour/dow/doy sin/cos """
    out = df.copy()
    # Only compute if missing
    need_hour = not {"hour_sin","hour_cos"}.issubset(out.columns)
    need_dow  = not {"dow_sin","dow_cos"}.issubset(out.columns)
    need_doy  = not {"doy_sin","doy_cos"}.issubset(out.columns)

    if need_hour or need_dow or need_doy:
        hour = date_index.hour
        dow  = date_index.dayofweek
        doy  = date_index.dayofyear
        two_pi = 2*np.pi

        if need_hour:
            out["hour_sin"] = np.sin(two_pi * hour / 24.0)
            out["hour_cos"] = np.cos(two_pi * hour / 24.0)
        if need_dow:
            out["dow_sin"]  = np.sin(two_pi * dow  / 7.0)
            out["dow_cos"]  = np.cos(two_pi * dow  / 7.0)
        if need_doy:
            # use 366 to be leap-year friendly
            out["doy_sin"]  = np.sin(two_pi * doy / 366.0)
            out["doy_cos"]  = np.cos(two_pi * doy / 366.0)
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
        "temp_mean", "HDD", "CDD",
        "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "is_weekend", "daylight_proxy",  # optional if present
        "sunrise_hour_approx", "sunset_hour_approx", "daylight_hours_approx", "is_daylight_approx"
    }
    cols = []
    for c in df.columns:
        if c in ("zone_id", "load"):  # exclude id + target
            continue
        if (c in keep_exact or
            c.startswith("station_") or
            ("_lag" in c) or ("_roll" in c) or ("_lead" in c)):
            cols.append(c)
    # de-dup, preserve order
    return list(dict.fromkeys(cols))

def smape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.align(pd.Series(y_pred, index=y_true.index), join="inner")
    denom = np.abs(y_true.values) + np.abs(y_pred.values) + 1e-9
    return float(100.0 * np.mean(2.0 * np.abs(y_pred.values - y_true.values) / denom))

# 1) Slice zone & enforce hourly 
zone = 1
df_zone = (features_with_lags
           .loc[features_with_lags["zone_id"] == zone]
           .sort_values("date_time")
           .set_index("date_time")
           .asfreq("h"))

# 2) Build feature matrix 
# Add calendar sin/cos if missing (computed from index)
X_all = ensure_calendar_features(df_zone.drop(columns=["zone_id", "load"], errors="ignore"),
                                 df_zone.index)

# pick columns (lags/rolls/leads, station_*, base weather + holiday + sin/cos)
feature_cols = pick_xgb_feature_cols(X_all)

# keep only those columns
X_all = X_all[feature_cols].copy()

# ensure numeric; cast is_holiday to int if present
X_all = X_all.apply(pd.to_numeric, errors="coerce")
if "is_holiday" in X_all.columns:
    X_all["is_holiday"] = X_all["is_holiday"].fillna(0).astype(int)

y_all = df_zone["load"]

# 3) Clean rows 
mask = (y_all.notna() & np.isfinite(y_all) &
        X_all.notna().all(axis=1) & np.isfinite(X_all).all(axis=1))
y_clean = y_all[mask]
X_clean = X_all[mask]

#  4) Chronological split: last 7 days holdout
H = 24 * 7
if len(y_clean) <= H + 1:
    raise ValueError("Not enough clean rows for a 7-day holdout.")

cutoff = y_clean.index[-(H+1)]
y_train, X_train = y_clean.loc[:cutoff], X_clean.loc[:cutoff]
y_test,  X_test  = y_clean.loc[cutoff + pd.Timedelta(hours=1):], X_clean.loc[cutoff + pd.Timedelta(hours=1):]

# 5) Fit XGBoost model  
xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,                 # slightly deeper to use lag/roll richness
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",          # fast & robust
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train)

# 6) In-sample metrics 
pred_tr = xgb.predict(X_train)
mse_tr  = mean_squared_error(y_train, pred_tr)
rmse_tr = float(np.sqrt(mse_tr))
mae_tr  = mean_absolute_error(y_train, pred_tr)
print(f"[XGB] In-sample:  MSE={mse_tr:.2f}  RMSE={rmse_tr:.2f}  MAE={mae_tr:.2f}")

#  7) Holdout (last 7 days) metrics 
pred_te = xgb.predict(X_test)
mse_te  = mean_squared_error(y_test, pred_te)
rmse_te = float(np.sqrt(mse_te))
mae_te  = mean_absolute_error(y_test, pred_te)
smape_te = smape(y_test, pred_te)
print(f"[XGB] Holdout (168h): MSE={mse_te:.2f}  RMSE={rmse_te:.2f}  MAE={mae_te:.2f}  sMAPE={smape_te:.2f}%")

# (Optional) quick peek at top features
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features:\n", importances.head(15))

# === Apply SHAP for global interpretability ===

# 1) Choose the slice to explain (use test/holdout to avoid leakage)
X_shap = X_test.copy()

# (Optional) Sample for speed if the set is huge
MAX_EXPLAIN = 8000
if len(X_shap) > MAX_EXPLAIN:
    X_shap = X_shap.sample(MAX_EXPLAIN, random_state=42).sort_index()

# 2) Build the explainer (TreeExplainer is optimized for XGBoost)
explainer = shap.TreeExplainer(xgb)

# Handle SHAP API differences across versions
try:
    sv = explainer(X_shap)          # shap>=0.40 → returns Explanation
    shap_values = sv.values
except Exception:
    shap_values = explainer.shap_values(X_shap)   # older shap
    sv = None

# 3) Global feature importance (mean |SHAP|)
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=True)

# 4) Beeswarm: distribution of impacts per feature
shap.summary_plot(shap_values, X_shap, show=True)

# 5) Targeted dependence plots for key drivers
# pick a few likely-important features that exist in your frame
candidates = [
    # temperature/rolls/lags
    "temp_mean", "temp_mean_roll6", "temp_mean_lag48",
    "CDD", "HDD", "CDD_roll168",
    # station temps
    "station_2", "station_2_roll6", "station_2_lag1",
    "station_6_roll6", "station_10_roll6",
    # calendar cycles
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "is_daylight_approx", "is_holiday"
]
feat_list = [c for c in candidates if c in X_shap.columns]

# (nice interaction pairs for cyclic encodings)
def _pair(f):
    if f == "hour_sin" and "hour_cos" in X_shap.columns: return "hour_cos"
    if f == "hour_cos" and "hour_sin" in X_shap.columns: return "hour_sin"
    if f == "doy_sin"  and "doy_cos"  in X_shap.columns: return "doy_cos"
    if f == "doy_cos"  and "doy_sin"  in X_shap.columns: return "doy_sin"
    return "auto"

for f in feat_list[:10]:  # keep it concise
    try:
        shap.dependence_plot(
            f, shap_values, X_shap,
            interaction_index=_pair(f),
            show=True
        )
    except Exception as e:
        print(f"[warn] dependence plot for {f} skipped: {e}")


# 6) Save plots to files instead of interactive display

plt.figure(); shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False); plt.savefig("shap_importance_bar.png", dpi=160, bbox_inches="tight")
plt.figure(); shap.summary_plot(shap_values, X_shap, show=False); plt.savefig("shap_beeswarm.png", dpi=160, bbox_inches="tight")


# === Apply LIME for local interpretability ===
# LIME is slower than SHAP but can provide local explanations for individual predictions

# no need for sanity checks, as the X_train, X_test, y_test, and xgb previously created are constructed from zone 1 only

# Convert to numpy for LIME; keep feature names
feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]
Xtr_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
Xte_np = X_test.values  if hasattr(X_test, "values")  else np.asarray(X_test)

# 1) Build LIME explainer on the TRAIN distribution
explainer_lime = LimeTabularExplainer(
    training_data=Xtr_np,
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=False   # keep features continuous (good for tree models)
)

# 2) Explain a single prediction: by default, the last timestamp in the test set
idx = -1
instance_x = Xte_np[idx]
ts_label  = X_test.index[idx] if hasattr(X_test, "index") else f"row {len(X_test)+idx}"
true_y    = float(y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx])
pred_y    = float(xgb.predict(X_test.iloc[[idx]])[0] if hasattr(X_test, "iloc") else xgb.predict(X_test[[idx]])[0])

exp = explainer_lime.explain_instance(
    data_row=instance_x,
    predict_fn=xgb.predict,
    num_features=15,      # top-K local drivers
)

print(f"\n[LIME] Local explanation @ {ts_label}")
print(f"  True load: {true_y:.2f} | Pred load: {pred_y:.2f}")
for feat, weight in exp.as_list():
    print(f"  {feat:40s} {weight:+.3f}")

fig = exp.as_pyplot_figure()
plt.title(f"LIME — local explanation @ {ts_label}")
plt.tight_layout()
plt.show()

# 3) (Optional) Explain the worst absolute error in the test set
pred_test = xgb.predict(X_test)
if hasattr(y_test, "index"):
    pred_series = pd.Series(pred_test, index=y_test.index)
    abs_err = (y_test - pred_series).abs()
    worst_ts = abs_err.idxmax()
    ix_worst = X_test.index.get_loc(worst_ts)
    instance_worst = Xte_np[ix_worst]
    true_w = float(y_test.loc[worst_ts])
    pred_w = float(pred_series.loc[worst_ts])
    label_worst = worst_ts
else:
    abs_err = np.abs(y_test - pred_test)
    ix_worst = int(np.argmax(abs_err))
    instance_worst = Xte_np[ix_worst]
    true_w = float(y_test[ix_worst])
    pred_w = float(pred_test[ix_worst])
    label_worst = f"row {ix_worst}"

exp_worst = explainer_lime.explain_instance(
    data_row=instance_worst,
    predict_fn=xgb.predict,
    num_features=15
)

print(f"\n[LIME] WORST error explanation @ {label_worst}")
print(f"  True load: {true_w:.2f} | Pred load: {pred_w:.2f} | Abs err: {abs(pred_w-true_w):.2f}")
for feat, weight in exp_worst.as_list():
    print(f"  {feat:40s} {weight:+.3f}")

fig2 = exp_worst.as_pyplot_figure()
plt.title(f"LIME — worst error @ {label_worst}")
plt.tight_layout()
plt.show()

# 4) (Optional) Batch a few more timestamps to compare explanations
# pick a few representative indices (morning peak, evening, weekend, holiday, etc.)
def explain_at(i: int):
    row = Xte_np[i]
    ts  = X_test.index[i] if hasattr(X_test, "index") else f"row {i}"
    exp_i = explainer_lime.explain_instance(row, xgb.predict, num_features=10)
    print(f"\n[LIME] @ {ts}")
    for feat, weight in exp_i.as_list():
        print(f"  {feat:40s} {weight:+.3f}")
    _ = exp_i.as_pyplot_figure(); plt.title(f"LIME @ {ts}"); plt.tight_layout(); plt.show()

# Save the local explanation & worst error to PNG
fig.savefig(f"lime_explanation_zone1_{ts_label:%Y%m%d_%H%M}.png", dpi=200, bbox_inches="tight")
fig2.savefig(f"lime_worst_error_zone1_{label_worst:%Y%m%d_%H%M}.png" if isinstance(label_worst, pd.Timestamp) else f"lime_worst_error_zone1_{label_worst}.png", dpi=200, bbox_inches="tight")

# explain_at(0)          # first in test set
# explain_at(len(X_test)//2)  # middle of test set
# explain_at(H-1)        # last in test set (H=168)   





