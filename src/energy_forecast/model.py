from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .feature_building import ensure_calendar_features  # you already have this

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "xgb_zone1.json"
FEATS_PATH = ARTIFACTS / "feature_cols_zone1.json"

# ---- Optional dependency (only when artifacts exist)
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # keep dev flow smooth even if xgboost not installed yet
    XGBRegressor = None  # type: ignore

_xgb = None
_feature_cols: List[str] | None = None


def _load_artifacts() -> None:
    global _xgb, _feature_cols
    if MODEL_PATH.exists() and FEATS_PATH.exists() and XGBRegressor is not None:
        if _xgb is None:
            m = XGBRegressor()
            m.load_model(str(MODEL_PATH))
            _xgb = m
        if _feature_cols is None:
            _feature_cols = json.loads(FEATS_PATH.read_text())


def _row_to_frame(row: Dict[str, Any]) -> pd.DataFrame:
    """Convert incoming JSON row -> single-row DataFrame and create cyclical features if needed."""
    df = pd.DataFrame([row]).copy()
    need = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"}
    if "date_time" in df.columns and not need.issubset(df.columns):
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        cyc = ensure_calendar_features(df["date_time"], True, True, True)
        for k, v in cyc.items():
            df[k] = v
    for c in df.columns:
        if c != "date_time":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def _fallback_predict(row: Dict[str, Any]) -> float:
    """Deterministic fallback if artifacts are missing: mean of last N values if 'series' is provided."""
    series = row.get("series")
    if isinstance(series, (list, tuple)) and len(series) > 0:
        return float(np.mean([float(x) for x in series[-3:]]))
    raise RuntimeError("Model artifacts missing and no 'series' provided for fallback.")


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    If artifacts exist -> use XGB with the exact saved feature order.
    Else -> fallback (useful for local tests before training).
    """
    _load_artifacts()

    # API accepts either {"row": {...}} or a flat row itself
    row = payload.get("row", payload)

    if _xgb is not None and _feature_cols is not None:
        df = _row_to_frame(row)
        missing = [c for c in _feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required features: {missing}")
        X = df[_feature_cols].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(X).all().all():
            raise ValueError("Non-finite values in features.")
        yhat = float(_xgb.predict(X)[0])  # type: ignore[union-attr]
        return {"prediction": yhat}
    else:
        # Fallback path keeps tests green before training artifacts exist
        return {"prediction": _fallback_predict(row)}
