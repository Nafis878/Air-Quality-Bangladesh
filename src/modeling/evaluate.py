"""Held-out evaluation: physical-unit predictions, metrics, and significance tests.

The test split is touched only here, for final reporting. Predictions are collected as a long
table keyed by (t, station, horizon, channel) so any two models can be aligned exactly for
paired Diebold-Mariano tests. Everything is de-scaled to physical units and restricted to
observed targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .channels import CHANNELS
from .losses import median_index
from .metrics import descale, pointwise_frame, aggregate
from .windows import valid_anchors, StationWindowDataset, CrossSectionDataset
from .train import is_gamma


def _descale_vecs(scaler):
    mu = np.array([scaler.global_[c][0] for c in CHANNELS], dtype=np.float64)
    sd = np.array([scaler.global_[c][1] for c in CHANNELS], dtype=np.float64)
    return mu, sd


@torch.no_grad()
def collect_predictions(name, model, arr, anchors, scaler, cfg, device="cpu") -> pd.DataFrame:
    """Return long df [t, station, horizon, channel, y_true, y_pred] (physical, observed only)."""
    model.eval()
    qmid = median_index(cfg["quantiles"])
    horizons = cfg["horizons"]
    mu, sd = _descale_vecs(scaler)
    rows = []

    if is_gamma(name):
        ds = CrossSectionDataset(arr, anchors, cfg["seq_len"], horizons)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=int(cfg.get("num_workers", 0)))
        i0 = 0
        for batch in loader:
            x = batch["x"].to(device); mask = batch["mask"].to(device)
            decay = batch["decay"].to(device); present = batch["present"].to(device)
            st = batch["station"].to(device)
            preds = model(x, mask, decay, present, st)[..., qmid]   # [B,S,H,C]
            B, S = preds.shape[0], preds.shape[1]
            t_anchor = ds.anchors[i0:i0 + B]; i0 += B
            for hi, h in enumerate(horizons):
                yp = preds[:, :, hi].cpu().numpy() * sd + mu          # [B,S,C]
                yt = batch[f"y_t{h}"].numpy() * sd + mu
                mm = batch[f"m_t{h}"].numpy()                         # [B,S,C]
                bi, si, ci = np.nonzero(mm > 0)
                rows.append(pd.DataFrame({
                    "t": t_anchor[bi], "station": si, "horizon": h, "channel": ci,
                    "y_true": yt[bi, si, ci], "y_pred": yp[bi, si, ci],
                }))
    else:
        ds = StationWindowDataset(arr, anchors, cfg["seq_len"], horizons)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=int(cfg.get("num_workers", 0)))
        i0 = 0
        for batch in loader:
            x = batch["x"].to(device)
            preds = model(x)[..., qmid]                              # [B,H,C]
            B = preds.shape[0]
            t_a = ds.t[i0:i0 + B]; s_a = ds.s[i0:i0 + B]; i0 += B
            for hi, h in enumerate(horizons):
                yp = preds[:, hi].cpu().numpy() * sd + mu            # [B,C]
                yt = batch[f"y_t{h}"].numpy() * sd + mu
                mm = batch[f"m_t{h}"].numpy()
                bi, ci = np.nonzero(mm > 0)
                rows.append(pd.DataFrame({
                    "t": t_a[bi], "station": s_a[bi], "horizon": h, "channel": ci,
                    "y_true": yt[bi, ci], "y_pred": yp[bi, ci],
                }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["t", "station", "horizon", "channel", "y_true", "y_pred"])


def metrics_by_horizon(pred_df: pd.DataFrame, horizons) -> dict:
    """{horizon: aggregate(...)} computed in physical units on observed points."""
    out = {}
    for h in horizons:
        sub = pred_df[pred_df["horizon"] == h]
        frame = pd.DataFrame({
            "station": sub["station"].to_numpy(), "channel": sub["channel"].to_numpy(),
            "y_true": sub["y_true"].to_numpy(), "y_pred": sub["y_pred"].to_numpy(),
            "abs_err": (sub["y_true"] - sub["y_pred"]).abs().to_numpy(),
            "sq_err": ((sub["y_true"] - sub["y_pred"]) ** 2).to_numpy(),
        })
        out[h] = aggregate(frame)
    return out


def pm25_frame(pred_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Observed PM2.5 points at a horizon, ordered by (station, t) for DM serial structure."""
    pm = CHANNELS.index("pm25")
    sub = pred_df[(pred_df["horizon"] == horizon) & (pred_df["channel"] == pm)]
    return sub.sort_values(["station", "t"]).reset_index(drop=True)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2):
    """Two-sided DM test with Newey-West HAC (lag h-1) + Harvey small-sample correction.

    e1, e2 are aligned forecast errors (model1, model2). Positive DM => model1 worse.
    Returns (DM_stat, p_value, n).
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    if n < 8 or np.allclose(d, 0):
        return float("nan"), float("nan"), n
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2.0 * cov
    var = var / n
    if var <= 0:
        return float("nan"), float("nan"), n
    dm = dbar / np.sqrt(var)
    # Harvey, Leybourne & Newbold (1997) small-sample correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p), n


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down. Returns {key: {p, p_adj, reject}}."""
    items = [(k, v) for k, v in pvals.items() if v == v]   # drop NaN
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        p_adj = min(max((m - i) * p, prev), 1.0)
        prev = p_adj
        out[k] = {"p": p, "p_adj": p_adj, "reject": p_adj < alpha}
    for k, v in pvals.items():
        if v != v:
            out[k] = {"p": float("nan"), "p_adj": float("nan"), "reject": False}
    return out
