# import neccessary libarries
from __future__ import annotations
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap


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
temp_long = temp_long[['station_id','date_time', 'temperature']].sort_values(by='date_time').reset_index(drop=True)
print(temp_long.head())

# Pivot so each station is one column
temp_wide = temp_long.pivot(index='date_time', columns='station_id', values='temperature')
temp_wide.columns = [f"station_{col}" for col in temp_wide.columns]

print(temp_long.head())

# ---- Merge: attach all stations' temperatures to each load record ----
merged_all = load_long.merge(temp_wide.reset_index(), on='date_time', how='left')
print(merged_all)

merged_all.isnull().sum() #check for missing values after the merge

merged_all_cleaned = merged_all.dropna().reset_index(drop=True) #drop rows with any NaNs and reset index
print(merged_all_cleaned)

merged_all_cleaned.isnull().sum()

# Build time features
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

# Create lag/rolling features


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

# Check again for data quality
# Sort and check continuity per zone_id
features_with_lags = features_with_lags.sort_values(["zone_id", "date_time"]).reset_index(drop=True)

print(features_with_lags.head())

# Check time step gaps per zone_id
gaps = (
    features_with_lags.groupby("zone_id")["date_time"]
    .diff()
    .dropna()
    .value_counts()
)
print(gaps.head())
print(gaps.isna().sum())

# Check for missing values 
features_with_lags.isna().sum().sort_values(ascending=False).head(10)

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


# Goal is forecast accuracy + feature interpretability, therefore the strategy is to start with SARIMAX model to get baseline interpretability and understand structure. 
# Move to XGBoost for better accuracy and use SHAP for interpretability

# ---- Quick test on a single zone ----

# 1) Filter a zone and enforce hourly frequency
zone = 1
df_zone = features_with_lags.loc[features_with_lags["zone_id"] == zone].copy()
df_zone = df_zone.sort_values("date_time").set_index("date_time").asfreq("h")  # may introduce NaNs

# 2) Build/ensure your exogenous columns exist
cols_exog = ["temp_mean", "HDD", "CDD", "is_holiday"]
missing_cols = [c for c in cols_exog if c not in df_zone.columns]
if missing_cols:
    raise KeyError(f"Exogenous columns missing: {missing_cols}")

y = df_zone["load"]
X = df_zone[cols_exog].copy()

# 3) Coerce to numeric and normalize types
X = X.apply(pd.to_numeric, errors="coerce")
if "is_holiday" in X.columns:
    # holidays should be deterministic 0/1; if you prefer, don’t interpolate them
    X["is_holiday"] = X["is_holiday"].fillna(0).astype(int)

# (Optional but helpful) Fill small holes in temperature-derived exog only
for c in ("temp_mean", "HDD", "CDD"):
    if c in X.columns:
        X[c] = X[c].interpolate(limit_direction="both")  # linear in time
        # if you’d rather never fill HDD/CDD, comment the line above and leave them NaN to be dropped

# 4) Hard guard: remove any remaining NaN / inf in y or X
mask = (
    y.notna() & np.isfinite(y) &
    X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
)
y_clean = y[mask]
X_clean = X[mask]

# Safety check: enough data left?
if len(y_clean) < 2 * 24:
    raise ValueError(f"Not enough clean hourly rows for zone {zone} after alignment: {len(y_clean)}")

# 5) Fit SARIMAX on clean data
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

# 6) Chronological split & evaluate
H = 24 * 7  # 1 week horizon
train_end = y_clean.index[-(H+1)]
y_train, X_train = y_clean.loc[:train_end], X_clean.loc[:train_end]
y_test,  X_test  = y_clean.loc[train_end + pd.Timedelta(hours=1):], X_clean.loc[train_end + pd.Timedelta(hours=1):]

model_tr = sm.tsa.SARIMAX(
    endog=y_train, exog=X_train,
    order=(1,1,1), seasonal_order=(1,0,1,24),
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False)

pred = model_tr.get_forecast(steps=len(y_test), exog=X_test)
y_pred = pred.predicted_mean
ci = pred.conf_int()


def smape(a: pd.Series, f: pd.Series) -> float:
    a, f = a.align(f, join="inner")
    return 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))

print(f"MAE={mean_absolute_error(y_test, y_pred):.2f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.2f}  "
      f"sMAPE={smape(y_test, y_pred):.2f}%")


# ---- Using the full dataset, but split chronologically into train/valid/test sets ----


# Set the chronological cutoffs (inclusive)
TRAIN_END = "2006-12-31 23:00"
VALID_END = "2007-12-31 23:00"
DATE_COL  = "date_time"
ZONE_COL  = "zone_id"
TARGET    = "load"
SEASONAL_PERIOD = 24   # hourly daily seasonality

# ====== METRICS ======
def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(mean_absolute_error(y_true, y_pred))

def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    denom = np.clip(np.abs(y_true.values) + np.abs(y_pred.values), 1e-9, None)
    return float(np.mean(2.0 * np.abs(y_pred.values - y_true.values) / denom) * 100.0)

# ====== EXOG COLUMN PICKER ======
def pick_exog_columns(df: pd.DataFrame) -> List[str]:
    """Auto-pick exogenous drivers from your schema."""
    keep_exact = {
        "temp_mean", "HDD", "CDD",
        "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
        "sunrise_hour_approx", "sunset_hour_approx",
        "daylight_hours_approx", "is_daylight_approx", "daylight_proxy",
    }
    cols: List[str] = []
    for c in df.columns:
        if c in (ZONE_COL, DATE_COL, TARGET):
            continue
        # station temps + any engineered lag/roll/lead and the exact extras
        if c.startswith("station_") or c in keep_exact or any(tok in c for tok in ("_lag", "_roll", "_lead")):
            cols.append(c)
    # de-dup preserve order
    return list(dict.fromkeys(cols))

# ====== DATA PREP PER ZONE ======
def prepare_splits_one_zone(df_zone: pd.DataFrame, exog_cols: List[str]
) -> Tuple[Tuple[pd.Series, pd.DataFrame], Tuple[pd.Series, pd.DataFrame], Tuple[pd.Series, pd.DataFrame]]:
    """
    Returns: (y_train, X_train), (y_valid, X_valid), (y_test, X_test)
    with an hourly DatetimeIndex and NA rows removed.
    """
    dfz = (df_zone
           .sort_values(DATE_COL)
           .set_index(DATE_COL)
           .asfreq("h"))  # enforce hourly frequency; gaps become NaN

    y = dfz[TARGET]
    X = dfz[exog_cols].copy()

    # Coerce exog to numeric and normalize holiday dtypes if present
    X = X.apply(pd.to_numeric, errors="coerce")
    if "is_holiday" in X.columns:
        X["is_holiday"] = X["is_holiday"].fillna(0).astype(int)

    # Drop any row with NA in y or exog (common after asfreq/lag windows)
    mask = y.notna() & X.notna().all(axis=1)
    y, X = y[mask], X[mask]

    # Chronological splits
    y_tr, X_tr = y.loc[:TRAIN_END], X.loc[:TRAIN_END]
    y_va, X_va = y.loc[TRAIN_END:VALID_END].iloc[1:], X.loc[TRAIN_END:VALID_END].iloc[1:]  # drop overlap row
    y_te, X_te = y.loc[VALID_END:].iloc[1:],           X.loc[VALID_END:].iloc[1:]

    return (y_tr, X_tr), (y_va, X_va), (y_te, X_te)

# ====== FIT + FORECAST HELPERS ======
def fit_sarimax(y_train: pd.Series,
                X_train: pd.DataFrame,
                order=(1,1,1),
                seasonal_order=(1,0,1,SEASONAL_PERIOD)):
    model = sm.tsa.SARIMAX(
        endog=y_train,
        exog=None if X_train.empty else X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)

def forecast_with_exog(res, X_future: pd.DataFrame) -> pd.Series:
    steps = len(X_future)
    if steps == 0:
        return pd.Series(dtype=float)
    fc = res.get_forecast(steps=steps, exog=None if X_future.empty else X_future).predicted_mean
    fc.index = X_future.index
    return fc

# ====== MAIN: SPLIT, FIT, EVAL PER ZONE ======
def run_sarimax_per_zone(features_with_lags: pd.DataFrame,
                         order=(1,1,1),
                         seasonal_order=(1,0,1,SEASONAL_PERIOD)
) -> Tuple[Dict[int, sm.tsa.statespace.sarimax.SARIMAXResults], pd.DataFrame]:
    exog_cols = pick_exog_columns(features_with_lags)
    zones = np.sort(features_with_lags[ZONE_COL].dropna().unique())

    models: Dict[int, sm.tsa.statespace.sarimax.SARIMAXResults] = {}
    rows = []

    for z in zones:
        df_zone = features_with_lags[features_with_lags[ZONE_COL] == z].copy()
        (y_tr, X_tr), (y_va, X_va), (y_te, X_te) = prepare_splits_one_zone(df_zone, exog_cols)

        # If not enough data remains after cleaning, skip
        if len(y_tr) == 0 or len(y_va) == 0:
            print(f"[Zone {z}] Skipped: insufficient clean data after split.")
            continue

        try:
            res = fit_sarimax(y_tr, X_tr, order=order, seasonal_order=seasonal_order)
        except Exception as e:
            print(f"[Zone {z}] SARIMAX fit failed: {e}")
            continue

        # Forecast and evaluate on validation
        fc_va = forecast_with_exog(res, X_va)
        val_mae  = mae(y_va, fc_va)
        val_rmse = rmse(y_va, fc_va)
        val_smape = smape(y_va, fc_va)

        # Optional: evaluate on test as well
        fc_te = forecast_with_exog(res, X_te)
        test_mae  = mae(y_te, fc_te)  if len(y_te) else np.nan
        test_rmse = rmse(y_te, fc_te) if len(y_te) else np.nan
        test_smape = smape(y_te, fc_te) if len(y_te) else np.nan

        rows.append({
            "zone_id": int(z),
            "AIC": float(res.aic),
            "VAL_MAE":  val_mae,   "VAL_RMSE":  val_rmse,   "VAL_sMAPE_%":  val_smape,
            "TEST_MAE": test_mae,  "TEST_RMSE": test_rmse,  "TEST_sMAPE_%": test_smape,
        })
        models[int(z)] = res

        print(f"[Zone {z}]  VAL  RMSE={val_rmse:.2f}  MAE={val_mae:.2f}  sMAPE={val_smape:.2f}%  "
              f"| TEST RMSE={test_rmse:.2f}  MAE={test_mae:.2f}  sMAPE={test_smape:.2f}%")

    results = pd.DataFrame(rows).sort_values("zone_id").reset_index(drop=True)
    return models, results

# ====== RUN ======
# Ensure the DataFrame is sorted before splitting
features_with_lags = features_with_lags.sort_values([ZONE_COL, DATE_COL]).reset_index(drop=True)

models_by_zone, metrics_table = run_sarimax_per_zone(
    features_with_lags,
    order=(1,1,1),            # start simple; tune later
    seasonal_order=(1,0,1,24) # daily (24h) seasonality
)

print("\n=== Metrics (head) ===")
print(metrics_table.head(10))




# ---- Train/validate/test a GLOBAL XGBoost on all zones


# ---------- CONFIG ----------
DATE_COL  = "date_time"
ZONE_COL  = "zone_id"
TARGET    = "load"

TRAIN_END = "2006-12-31 23:00"
VALID_END = "2007-12-31 23:00"

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    denom = np.clip(np.abs(y_true.values)+np.abs(y_pred.values), 1e-9, None)
    return float(np.mean(2*np.abs(y_pred.values-y_true.values)/denom)*100.0)

def pick_x_features(df: pd.DataFrame) -> List[str]:
    """Use the same idea as SARIMAX: keep exogenous + engineered lags/rolls/leads."""
    drop = {ZONE_COL, DATE_COL, TARGET}
    keep_exact = {
        "temp_mean","HDD","CDD",
        "is_holiday","is_day_before_holiday","is_day_after_holiday",
        "sunrise_hour_approx","sunset_hour_approx",
        "daylight_hours_approx","is_daylight_approx","daylight_proxy",
        "hour","dow","is_weekend","hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos",
    }
    cols: List[str] = []
    for c in df.columns:
        if c in drop: 
            continue
        if c in keep_exact or c.startswith("station_") or any(tok in c for tok in ("_lag","_roll","_lead")):
            cols.append(c)
    return list(dict.fromkeys(cols))  # de-dup, keep order

# Ensure sorted
features_with_lags = features_with_lags.sort_values([ZONE_COL, DATE_COL]).reset_index(drop=True)

# Build design matrix
X_cols = pick_x_features(features_with_lags)
df_all = features_with_lags.copy()
df_all[DATE_COL] = pd.to_datetime(df_all[DATE_COL], errors="coerce")

# Clean: drop rows with any NA in target or features
mask = df_all[TARGET].notna() & df_all[X_cols].notna().all(axis=1)
df_all = df_all.loc[mask, [DATE_COL, ZONE_COL, TARGET] + X_cols].copy()

# Chronological split across ALL zones (global model)
train_mask = df_all[DATE_COL] <= pd.Timestamp(TRAIN_END)
valid_mask = (df_all[DATE_COL] > pd.Timestamp(TRAIN_END)) & (df_all[DATE_COL] <= pd.Timestamp(VALID_END))
test_mask  = df_all[DATE_COL] > pd.Timestamp(VALID_END)

X_train, y_train = df_all.loc[train_mask, X_cols], df_all.loc[train_mask, TARGET]
X_valid, y_valid = df_all.loc[valid_mask, X_cols], df_all.loc[valid_mask, TARGET]
X_test,  y_test  = df_all.loc[test_mask,  X_cols], df_all.loc[test_mask,  TARGET]

print(f"Train: {len(y_train):,}  Valid: {len(y_valid):,}  Test: {len(y_test):,}")

# XGBoost params (good starting point; tune later)
xgb = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=2.0,
    tree_method="hist",         # fast + robust
    random_state=42,
)

# Early stopping on VALID
xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="rmse",
    verbose=False,
    early_stopping_rounds=100
)

# Evaluate
pred_va = xgb.predict(X_valid)
pred_te = xgb.predict(X_test)

print("VALID  MAE={:.2f}  RMSE={:.2f}  sMAPE={:.2f}%".format(
    mean_absolute_error(y_valid, pred_va), rmse(y_valid, pred_va), smape(y_valid, pred_va)))
print("TEST   MAE={:.2f}  RMSE={:.2f}  sMAPE={:.2f}%".format(
    mean_absolute_error(y_test, pred_te),  rmse(y_test, pred_te),  smape(y_test, pred_te)))