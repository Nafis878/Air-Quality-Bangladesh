"""
Leakage-safe temporal splitting and train-only scaler fitting.

The panel is a time series: NEVER random-shuffle. Splits are by timestamp using the
configurable boundaries in config.yaml. Any scaler/encoder/imputer must be fit on the
TRAIN slice only and then applied to val/test — `fit_scaler` enforces that contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import NUMERIC_COLS


def temporal_split(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Split by timestamp into train/val/test per config boundaries (inclusive ends)."""
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    ts = df["timestamp"]
    train = df[ts <= train_end]
    val = df[(ts > train_end) & (ts <= val_end)]
    test = df[ts > val_end]
    return {"train": train.copy(), "val": val.copy(), "test": test.copy()}


class StandardScalerFrame:
    """Minimal, leakage-safe standardizer. Fit on train only; optionally per station.

    Stores mean/std (population, ddof=0). Per-station scaling falls back to the global
    mean/std for stations or columns unseen at fit time.
    """

    def __init__(self, cols: list[str] | None = None, per_station: bool = False):
        self.cols = cols
        self.per_station = per_station
        self.global_: dict[str, tuple[float, float]] = {}
        self.by_station_: dict[str, dict[str, tuple[float, float]]] = {}

    def fit(self, train: pd.DataFrame) -> "StandardScalerFrame":
        if self.cols is None:
            self.cols = [c for c in NUMERIC_COLS if c in train.columns]
        for c in self.cols:
            mu, sd = float(train[c].mean()), float(train[c].std(ddof=0))
            self.global_[c] = (mu, sd if sd and not np.isnan(sd) else 1.0)
        if self.per_station:
            for st, g in train.groupby("station"):
                self.by_station_[st] = {}
                for c in self.cols:
                    mu, sd = float(g[c].mean()), float(g[c].std(ddof=0))
                    self.by_station_[st][c] = (mu, sd if sd and not np.isnan(sd) else 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.per_station:
            for c in self.cols:
                mu, sd = self.global_[c]
                out[c] = (out[c] - mu) / sd
            return out
        for st, g in out.groupby("station"):
            stats = self.by_station_.get(st, self.global_)
            for c in self.cols:
                mu, sd = stats.get(c, self.global_[c])
                out.loc[g.index, c] = (g[c] - mu) / sd
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)


def fit_scaler(train: pd.DataFrame, cols: list[str] | None = None, per_station: bool = False) -> StandardScalerFrame:
    """Convenience: fit a StandardScalerFrame on the TRAIN slice only."""
    return StandardScalerFrame(cols=cols, per_station=per_station).fit(train)
