from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .feature_building import build_merged, make_features, pick_xgb_feature_cols

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def main(base_dir: Path = Path("."), zone_id: int = 1) -> None:
    # 1) Load the same raw files your script uses (adjust names if needed)
    load_long = pd.read_csv(base_dir / "data" / "Load" / "Load_history.csv")
    temp_wide = pd.read_csv(base_dir / "data" / "Load" / "temperature_history.csv")

    load_long["date_time"] = pd.to_datetime(load_long["date_time"], errors="coerce")
    if "date_time" in temp_wide.columns:
        temp_wide["date_time"] = pd.to_datetime(temp_wide["date_time"], errors="coerce")
        temp_wide = temp_wide.set_index("date_time")

    # 2) Your exact merge → cleaned frame
    merged_all_cleaned = build_merged(load_long, temp_wide)

    # 3) Pick one zone (or loop & save per-zone)
    df = merged_all_cleaned[merged_all_cleaned["zone_id"] == zone_id].copy()
    df = df.sort_values("date_time", ascending=True)

    # 4) Minimal features (you can add your lags/HDD/CDD here later)
    feats = make_features(df)

    # 5) Target/Features
    y = feats["load"]
    feature_cols = pick_xgb_feature_cols(feats)
    X = feats[feature_cols].apply(pd.to_numeric, errors="coerce")

    mask = y.notna() & np.isfinite(y) & X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    y_clean = y[mask]
    X_clean = X[mask]

    # 6) Chronological split with 7-day holdout (optional—train on the rest)
    H = 24 * 7
    cutoff = len(X_clean) - (H + 1)
    if cutoff < 100:
        raise RuntimeError("Not enough rows to hold out a week.")
    X_train, y_train = X_clean.iloc[:cutoff], y_clean.iloc[:cutoff]

    # 7) Train XGB (hyperparams from your script)
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    # 8) Save artifacts
    (ARTIFACTS / "xgb_zone1.json").write_text("")  # ensure directory exists for some editors
    xgb.save_model(ARTIFACTS / "xgb_zone1.json")
    (ARTIFACTS / "feature_cols_zone1.json").write_text(json.dumps(feature_cols, indent=2))

    print("Saved artifacts/xgb_zone1.json and artifacts/feature_cols_zone1.json")


if __name__ == "__main__":
    main()
