"""Leakage-safe windowing for the station panel.

Builds dense per-split arrays indexed [T(time), S(station), C(channel)] from the cleaned
hourly panel, plus GRU-D style observation masks and time-since-last-observation decay.
Two torch Datasets read the SAME arrays so every model is scored on identical target
points:

* `StationWindowDataset` — one sample per (anchor_time, station): a single station's
  past window. Used by all per-station baselines.
* `CrossSectionDataset` — one sample per anchor_time: the contemporaneous window of ALL
  stations. Used by GAMMA's spatial axis.

Hard leakage rules enforced here:
  - windows only look back (`[t-seq+1 .. t]`); targets are strictly future (`t+h`).
  - arrays are built per split, so no window ever straddles a split boundary.
  - inputs are scaled with a TRAIN-fit scaler; missing inputs are zero-filled (= channel
    mean) with the observation mask + decay telling the model what was synthesized.
  - targets carry their own observation mask; only observed (non-NaN, non-gap) targets are
    ever returned, so imputed/gap cells can never enter the loss or metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .channels import CHANNELS, HORIZONS
from ..splits import StandardScalerFrame


@dataclass
class PanelArrays:
    """Dense arrays for one split. Time axis is a regular hourly grid (the split's span)."""
    time_index: pd.DatetimeIndex
    stations: list[str]
    channels: list[str]
    x: np.ndarray        # [T,S,C] scaled inputs, missing -> 0 (channel mean in scaled space)
    m: np.ndarray        # [T,S,C] observation mask (1.0 = real observed value)
    d: np.ndarray        # [T,S,C] hours since last observation (GRU-D decay)
    present: np.ndarray  # [T,S]  station has a real (non-gap) row at this hour
    val_scaled: np.ndarray  # [T,S,C] scaled value with NaN preserved (for targets)

    @property
    def shape(self):
        return self.x.shape


def build_panel_arrays(
    df_split: pd.DataFrame,
    stations: list[str],
    scaler: StandardScalerFrame,
    channels: list[str] = CHANNELS,
) -> PanelArrays:
    """Construct dense [T,S,C] arrays for a split, scaling inputs with the train-fit scaler."""
    sub = df_split[df_split["station"].isin(stations)].copy()
    sub = scaler.transform(sub)  # scales `channels` in place (global, train-fit)

    time_index = pd.DatetimeIndex(sorted(sub["timestamp"].unique()))
    T, S, C = len(time_index), len(stations), len(channels)
    si = {s: i for i, s in enumerate(stations)}

    val = np.full((T, S, C), np.nan, dtype=np.float32)
    present = np.zeros((T, S), dtype=bool)

    t_idx = time_index.get_indexer(sub["timestamp"])
    s_idx = sub["station"].map(si).to_numpy()
    val[t_idx, s_idx, :] = sub[channels].to_numpy(dtype=np.float32)
    present[t_idx, s_idx] = sub["source_year_file"].notna().to_numpy()

    m = (~np.isnan(val)).astype(np.float32)
    x = np.where(m > 0, val, 0.0).astype(np.float32)

    # GRU-D decay: hours since last observation per (station, channel), along time.
    # Grid is hourly, so one grid step == one hour.
    d = np.zeros((T, S, C), dtype=np.float32)
    obs = m.reshape(T, S * C)
    d_prev = np.zeros(S * C, dtype=np.float32)
    flat = d.reshape(T, S * C)
    for t in range(T):
        d_prev = np.where(obs[t] > 0, 0.0, d_prev + 1.0)
        flat[t] = d_prev
    d = flat.reshape(T, S, C)

    return PanelArrays(time_index, list(stations), list(channels), x, m, d, present, val)


def valid_anchors(arr: PanelArrays, seq_len: int, horizons=HORIZONS, stride: int = 1) -> np.ndarray:
    """Anchor time indices t whose window [t-seq+1..t] and all targets t+h are in-range."""
    T = arr.shape[0]
    max_h = max(horizons)
    lo, hi = seq_len - 1, T - max_h - 1
    if hi < lo:
        return np.empty(0, dtype=np.int64)
    return np.arange(lo, hi + 1, stride, dtype=np.int64)


class StationWindowDataset(Dataset):
    """One sample per (anchor t, station s present at t). Single-station past window."""

    def __init__(self, arr: PanelArrays, anchors: np.ndarray, seq_len: int, horizons=HORIZONS):
        self.arr = arr
        self.seq_len = seq_len
        self.horizons = list(horizons)
        # enumerate (t, s) for stations actually present at the anchor hour
        ts, ss = [], []
        present_anchor = arr.present[anchors]  # [A,S]
        for ai, t in enumerate(anchors):
            for s in np.nonzero(present_anchor[ai])[0]:
                ts.append(int(t)); ss.append(int(s))
        self.t = np.asarray(ts, dtype=np.int64)
        self.s = np.asarray(ss, dtype=np.int64)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        t, s = int(self.t[i]), int(self.s[i])
        sl = slice(t - self.seq_len + 1, t + 1)
        x = torch.from_numpy(self.arr.x[sl, s, :])          # [seq,C]
        m = torch.from_numpy(self.arr.m[sl, s, :])
        d = torch.from_numpy(self.arr.d[sl, s, :])
        item = {"x": x, "mask": m, "decay": d, "station": torch.tensor(s, dtype=torch.long)}
        for h in self.horizons:
            yv = self.arr.val_scaled[t + h, s, :]
            ym = self.arr.m[t + h, s, :]
            item[f"y_t{h}"] = torch.from_numpy(np.nan_to_num(yv, nan=0.0).astype(np.float32))
            item[f"m_t{h}"] = torch.from_numpy(ym.astype(np.float32))
        return item


class CrossSectionDataset(Dataset):
    """One sample per anchor t: the contemporaneous window of ALL stations (for GAMMA)."""

    def __init__(self, arr: PanelArrays, anchors: np.ndarray, seq_len: int, horizons=HORIZONS):
        self.arr = arr
        self.anchors = anchors
        self.seq_len = seq_len
        self.horizons = list(horizons)
        self.S = arr.shape[1]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        t = int(self.anchors[i])
        sl = slice(t - self.seq_len + 1, t + 1)
        x = torch.from_numpy(self.arr.x[sl, :, :]).permute(1, 0, 2).contiguous()   # [S,seq,C]
        m = torch.from_numpy(self.arr.m[sl, :, :]).permute(1, 0, 2).contiguous()
        d = torch.from_numpy(self.arr.d[sl, :, :]).permute(1, 0, 2).contiguous()
        present = torch.from_numpy(self.arr.present[t].astype(np.float32))          # [S]
        item = {"x": x, "mask": m, "decay": d, "present": present,
                "station": torch.arange(self.S, dtype=torch.long)}
        for h in self.horizons:
            yv = self.arr.val_scaled[t + h, :, :]
            ym = self.arr.m[t + h, :, :]
            item[f"y_t{h}"] = torch.from_numpy(np.nan_to_num(yv, nan=0.0).astype(np.float32))  # [S,C]
            item[f"m_t{h}"] = torch.from_numpy(ym.astype(np.float32))
        return item
