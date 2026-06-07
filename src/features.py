"""
Leakage-aware feature engineering (separate, optional — NOT applied to the canonical
clean artifact). All temporal features are strictly past-only and grouped by station so
a row never sees another station's data or its own future.

Use on a dataframe that already carries `station` and `timestamp`. Call AFTER splitting,
or on the full frame as long as you only *fit* statistics (scalers/imputers) on train.
Building lag/rolling features themselves is deterministic from past observations and is
safe to compute on the full series; the guardrail enforced here is that each feature at
time t uses only information available at or before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import POLLUTANTS

# Bangladesh monsoon season ~ June–October (inclusive).
MONSOON_MONTHS = {6, 7, 8, 9, 10}
DEFAULT_LAGS = (1, 3, 24)
DEFAULT_WINDOWS = (3, 24)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + cyclical encodings + Bangladesh monsoon flag. No leakage (pure of t)."""
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["doy"] = ts.dt.dayofyear
    df["is_monsoon"] = df["month"].isin(MONSOON_MONTHS).astype("int8")
    # cyclical encodings (never feed raw periodic integers to a model)
    df["hour_sin"], df["hour_cos"] = _cyc(df["hour"], 24)
    df["doy_sin"], df["doy_cos"] = _cyc(df["doy"], 365.25)
    df["month_sin"], df["month_cos"] = _cyc(df["month"], 12)
    return df


def add_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode wind_dir (degrees) as sin/cos. Raw degrees are discontinuous at 0/360."""
    df = df.copy()
    if "wind_dir" in df.columns:
        rad = np.deg2rad(df["wind_dir"])
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: tuple[str, ...] | list[str] | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Per-station lag + rolling mean/std, strictly from the past (shifted by 1).

    Rolling stats are computed on the *shifted* series so the current row is excluded —
    no same-step leakage. Grouped by station, ordered by timestamp.
    """
    df = df.sort_values(["station", "timestamp"]).copy()
    cols = list(cols) if cols is not None else [c for c in POLLUTANTS if c in df.columns]
    g = df.groupby("station", sort=False)
    for c in cols:
        if c not in df.columns:
            continue
        past = g[c].shift(1)            # everything at <= t-1
        for lag in lags:
            df[f"{c}_lag{lag}"] = g[c].shift(lag)
        for w in windows:
            df[f"{c}_rollmean{w}"] = past.groupby(df["station"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{c}_rollstd{w}"] = past.groupby(df["station"]).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
    return df


def build_features(
    df: pd.DataFrame,
    target_cols: tuple[str, ...] | list[str] | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Full feature build: time + wind + lag/rolling. Returns a new frame."""
    out = add_time_features(df)
    out = add_wind_features(out)
    out = add_lag_features(out, cols=target_cols, lags=lags, windows=windows)
    return out


def _cyc(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    ang = 2 * np.pi * series.astype(float) / period
    return np.sin(ang), np.cos(ang)
