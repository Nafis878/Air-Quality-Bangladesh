"""Naive forecasting floors. If learned models cannot beat these, that is the headline.

All three emit predictions in the SAME scaled space as the deep models, so they pass through
the identical de-scaling + masked-metric path (apples to apples).

  * persistence    : y_hat(t+h) = last OBSERVED value at/<=t (forward-filled).
  * seasonal_naive : y_hat(t+h) = value at the same clock hour 24h before the target
                     (t+h-24), forward-filled.
  * climatology    : y_hat(t+h) = TRAIN per-(station, month, hour) mean (fit on train only).

`predict(arr, t_idx, s_idx, horizon)` returns [N, C] scaled predictions aligned to the given
(anchor, station) pairs — the same ordering the evaluator uses for the learned models.
"""
from __future__ import annotations

import numpy as np

from ..windows import PanelArrays


def _ffill_time(val: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along the time axis (axis 0) of a [T,S,C] array."""
    T = val.shape[0]
    out = val.copy()
    last = np.full(val.shape[1:], np.nan, dtype=val.dtype)
    for t in range(T):
        cur = out[t]
        upd = ~np.isnan(cur)
        last = np.where(upd, cur, last)
        out[t] = last
    return out


class NaiveForecasters:
    def __init__(self, train_arr: PanelArrays):
        self.stations = train_arr.stations
        self.channels = train_arr.channels
        self._clim = self._fit_climatology(train_arr)

    # ---- climatology fit on TRAIN ----
    def _fit_climatology(self, arr: PanelArrays) -> np.ndarray:
        T, S, C = arr.val_scaled.shape
        months = arr.time_index.month.to_numpy() - 1     # 0..11
        hours = arr.time_index.hour.to_numpy()           # 0..23
        sums = np.zeros((S, 12, 24, C), dtype=np.float64)
        cnts = np.zeros((S, 12, 24, C), dtype=np.float64)
        v = arr.val_scaled
        obs = ~np.isnan(v)
        for t in range(T):
            mo, hr = months[t], hours[t]
            o = obs[t]
            sums[:, mo, hr, :] += np.where(o, v[t], 0.0)
            cnts[:, mo, hr, :] += o
        with np.errstate(invalid="ignore"):
            clim = np.where(cnts > 0, sums / cnts, 0.0)   # scaled-space mean; 0 == channel mean
        return clim.astype(np.float32)

    def predict(self, arr: PanelArrays, t_idx: np.ndarray, s_idx: np.ndarray,
                horizon: int, method: str) -> np.ndarray:
        if not hasattr(arr, "_ffill_cache"):
            arr._ffill_cache = _ffill_time(arr.val_scaled)   # cache per split
        ff = arr._ffill_cache
        N = len(t_idx)
        C = len(self.channels)
        out = np.zeros((N, C), dtype=np.float32)
        if method == "persistence":
            for i in range(N):
                out[i] = ff[t_idx[i], s_idx[i], :]
        elif method == "seasonal_naive":
            for i in range(N):
                src = t_idx[i] + horizon - 24
                src = src if src >= 0 else t_idx[i]
                out[i] = ff[src, s_idx[i], :]
        elif method == "climatology":
            months = arr.time_index.month.to_numpy() - 1
            hours = arr.time_index.hour.to_numpy()
            for i in range(N):
                tt = t_idx[i] + horizon
                out[i] = self._clim[s_idx[i], months[tt], hours[tt], :]
        else:
            raise ValueError(f"unknown naive method: {method}")
        return np.nan_to_num(out, nan=0.0)


NAIVE_METHODS = ["persistence", "seasonal_naive", "climatology"]
