"""Glue: cleaned parquet -> temporal splits -> train-only scaler -> dense panel arrays.

One call site so the leakage-safe contract is applied identically everywhere: the scaler is
fit on TRAIN only, the station set is resolved from TRAIN only, and arrays for val/test are
built with those train statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from .channels import CHANNELS, resolve_station_set
from .windows import PanelArrays, build_panel_arrays
from ..splits import temporal_split, fit_scaler
from ..schema import MODEL_CHANNELS


@dataclass
class Prepared:
    cfg: dict
    stations: list[str]
    excluded: dict[str, str]
    scaler: object
    arrays: dict[str, PanelArrays]      # 'train' / 'val' / 'test'
    splits: dict[str, pd.DataFrame]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prepare(model_cfg: dict, nrows: int | None = None) -> Prepared:
    data_cfg = load_config(model_cfg["data_config"])
    df = pd.read_parquet(model_cfg["clean_parquet"])
    df = df[["station", "timestamp", *MODEL_CHANNELS, "source_year_file", "is_gap"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if nrows:
        df = df.head(nrows)

    splits = temporal_split(df, data_cfg)
    stations, excluded = resolve_station_set(
        splits,
        min_train_obs_rate=model_cfg.get("station_min_train_obs_rate", 0.30),
        min_test_obs_rate=model_cfg.get("station_min_test_obs_rate", 0.05),
    )
    # train-only scaler (global per-channel standardization)
    scaler = fit_scaler(splits["train"][splits["train"]["station"].isin(stations)],
                        cols=CHANNELS, per_station=False)
    arrays = {
        name: build_panel_arrays(splits[name], stations, scaler, CHANNELS)
        for name in ("train", "val", "test")
    }
    return Prepared(model_cfg, stations, excluded, scaler, arrays, splits)


def inter_station_corr(train_arr: PanelArrays, channel: str = "pm25") -> np.ndarray:
    """TRAIN-only station-station Pearson correlation of one channel's series.

    Used to give GAMMA's spatial graph a physically-sensible, leakage-safe prior on which
    stations co-vary (no coordinates exist in this dataset). Pairwise-complete on the scaled
    series; NaNs (missing hours / no overlap) -> 0; diagonal -> 0 so the prior adds no self-bias.
    Returns [S, S] float32.
    """
    c = train_arr.channels.index(channel)
    series = pd.DataFrame(train_arr.val_scaled[:, :, c])        # [T, S], NaN preserved
    corr = series.corr().to_numpy()                            # pairwise Pearson, NaN where no overlap
    corr = np.nan_to_num(corr, nan=0.0).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    return corr
