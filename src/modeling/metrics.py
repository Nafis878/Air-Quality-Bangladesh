"""Point-forecast metrics in PHYSICAL units, computed on observed targets only.

Predictions and targets enter scaled; `descale` inverts the train-fit StandardScaler per
channel so MAE/RMSE are in native units (ug/m3 for PM, ppb/ppm for gases, etc.). R2 is the
standard coefficient of determination against the observed target mean.

Aggregations:
  - micro: row-weighted over all observed (sample, channel) pairs.
  - macro: station-averaged (each station weighted equally), to stop high-coverage stations
    from dominating.
  - per-station and per-channel breakdowns are returned for the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import CHANNELS
from ..splits import StandardScalerFrame


def descale(arr: np.ndarray, scaler: StandardScalerFrame, channels=CHANNELS) -> np.ndarray:
    """Invert global standardization: phys = scaled*sd + mu. arr last axis = channels."""
    mu = np.array([scaler.global_[c][0] for c in channels], dtype=np.float64)
    sd = np.array([scaler.global_[c][1] for c in channels], dtype=np.float64)
    return arr * sd + mu


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pointwise_frame(
    pred_phys: np.ndarray, true_phys: np.ndarray, mask: np.ndarray,
    station_ids: np.ndarray, channels=CHANNELS,
) -> pd.DataFrame:
    """Long-form observed (station, channel) errors for one horizon.

    pred_phys/true_phys/mask: [N, C]; station_ids: [N] station index per row.
    Returns columns: station, channel, y_true, y_pred, abs_err, sq_err.
    """
    N, C = true_phys.shape
    st = np.repeat(station_ids[:, None], C, axis=1)
    ch = np.tile(np.arange(C), (N, 1))
    keep = mask > 0
    yt = true_phys[keep]; yp = pred_phys[keep]
    return pd.DataFrame({
        "station": st[keep], "channel": ch[keep],
        "y_true": yt, "y_pred": yp,
        "abs_err": np.abs(yt - yp), "sq_err": (yt - yp) ** 2,
    })


def aggregate(frame: pd.DataFrame, channels=CHANNELS) -> dict:
    """Compute micro/macro MAE/RMSE/R2 and per-station/per-channel breakdowns from a frame."""
    out: dict = {}
    if len(frame) == 0:
        return {"micro": {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}}
    # micro (row-weighted over all observed cells)
    out["micro"] = {
        "mae": float(frame["abs_err"].mean()),
        "rmse": float(np.sqrt(frame["sq_err"].mean())),
        "r2": _safe_r2(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
        "n": int(len(frame)),
    }
    # macro over stations (equal station weight)
    per_st = []
    for st, g in frame.groupby("station"):
        per_st.append({
            "station": int(st), "mae": float(g["abs_err"].mean()),
            "rmse": float(np.sqrt(g["sq_err"].mean())),
            "r2": _safe_r2(g["y_true"].to_numpy(), g["y_pred"].to_numpy()), "n": int(len(g)),
        })
    per_st_df = pd.DataFrame(per_st)
    out["macro"] = {
        "mae": float(per_st_df["mae"].mean()),
        "rmse": float(per_st_df["rmse"].mean()),
        "r2": float(per_st_df["r2"].mean(skipna=True)),
        "n_stations": int(len(per_st_df)),
    }
    out["per_station"] = per_st_df
    # per channel (micro within channel)
    per_ch = []
    for c, g in frame.groupby("channel"):
        per_ch.append({
            "channel": channels[int(c)], "mae": float(g["abs_err"].mean()),
            "rmse": float(np.sqrt(g["sq_err"].mean())),
            "r2": _safe_r2(g["y_true"].to_numpy(), g["y_pred"].to_numpy()), "n": int(len(g)),
        })
    out["per_channel"] = pd.DataFrame(per_ch)
    return out
