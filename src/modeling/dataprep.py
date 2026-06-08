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
from . import geo as geomod
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
    geo: dict | None = None             # coords/dist/bearing for the resolved station order


def build_geo(stations: list[str], coords_path: str,
              length_scales=(50.0, 200.0)) -> dict:
    """Inductive geo tensors aligned to `stations`: coords[S,2], dist_feats[S,S,2] (two decay
    scales), bearing[S,S]. From EXTERNAL approximate coordinates (see data/external/README)."""
    lat, lon, missing = geomod.load_coords(coords_path, stations)
    if missing:
        print(f"[geo] WARNING: no coordinates for {missing} — those rows get NaN (graph falls back).")
    dist = geomod.haversine_km(lat, lon)
    brg = geomod.bearings_deg(lat, lon)
    dd = np.stack([geomod.distance_decay(dist, ls) for ls in length_scales], axis=-1)  # [S,S,2]
    coords = np.stack([lat, lon], axis=-1).astype(np.float32)
    return {"coords": coords, "dist_feats": dd.astype(np.float32),
            "bearing": brg.astype(np.float32), "dist_km": dist, "missing": missing}


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
    geo = build_geo(stations, model_cfg.get("coords_path", "data/external/station_coords.csv"))
    return Prepared(model_cfg, stations, excluded, scaler, arrays, splits, geo)


def build_cold_start(prep: Prepared, cold_station: str = "DoE") -> dict | None:
    """Build a TEST cross-section that appends an unseen `cold_station` to the trained stations.

    Returns {arr, geo, cold_index, station_list} or None if the station has no test rows. The model
    is inductive (coord-MLP embedding + coordinate/wind edges), so it can forecast `cold_station`
    despite never training on it. Per-station baselines have no weights for it and cannot.
    """
    test_df = prep.splits["test"]
    if cold_station not in set(test_df["station"]):
        return None
    stations = list(prep.stations) + [cold_station]
    arr = build_panel_arrays(test_df, stations, prep.scaler, CHANNELS)
    geo = build_geo(stations, prep.cfg.get("coords_path", "data/external/station_coords.csv"))
    return {"arr": arr, "geo": geo, "cold_index": len(prep.stations),
            "station_list": stations, "cold_station": cold_station}


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
