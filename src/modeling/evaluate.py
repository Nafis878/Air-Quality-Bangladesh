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


def calibration(df: pd.DataFrame, nominal: float = 0.8) -> dict:
    """Interval coverage (PICP) and mean width (MPIW) for the [y_lo,y_hi] prediction interval."""
    if df is None or len(df) == 0 or "y_lo" not in df:
        return {"picp": float("nan"), "mpiw": float("nan"), "n": 0, "nominal": nominal}
    lo, hi, yt = df["y_lo"].to_numpy(), df["y_hi"].to_numpy(), df["y_true"].to_numpy()
    return {"picp": float(((yt >= lo) & (yt <= hi)).mean()), "mpiw": float((hi - lo).mean()),
            "n": int(len(df)), "nominal": nominal}


def calibration_by_model(pm25_by_model: dict, horizons, nominal: float = 0.8) -> dict:
    """{model: {horizon: calibration(...)}} on PM2.5; only models that carry interval columns."""
    out = {}
    for name, df in pm25_by_model.items():
        if df is None or len(df) == 0 or "y_lo" not in df:
            continue
        out[name] = {int(h): calibration(df[df["horizon"] == h], nominal) for h in horizons}
    return out


def outage_summary(outage_by_model: dict, normal_by_model: dict, horizons,
                   gamma_key="GAMMA", floor_key="persistence", baseline_names=None) -> dict:
    """Degradation under a simulated self-sensor outage vs the model's own fresh-point MAE, plus DM
    (GAMMA vs persistence / best baseline) computed on the OUTAGE points. The causal value of
    neighbours: GAMMA should degrade least; per-station baselines should collapse."""
    baseline_names = set(baseline_names or [])
    rows = []
    for name, odf in outage_by_model.items():
        ndf = normal_by_model.get(name)
        for h in horizons:
            o = odf[odf["horizon"] == h]
            mae_out = float((o["y_true"] - o["y_pred"]).abs().mean()) if len(o) else float("nan")
            mae_norm = float("nan")
            if ndf is not None and len(ndf):
                nf = ndf[(ndf["horizon"] == h) & (ndf["staleness"] <= 1)]
                mae_norm = float((nf["y_true"] - nf["y_pred"]).abs().mean()) if len(nf) else float("nan")
            rows.append({"model": name, "horizon": int(h), "mae_fresh": mae_norm,
                         "mae_outage": mae_out, "degradation": mae_out - mae_norm, "n": int(len(o))})
    # best baseline under outage (lowest outage MAE), per horizon
    best_base = {}
    for h in horizons:
        cands = [(r["model"], r["mae_outage"]) for r in rows
                 if r["model"] in baseline_names and r["horizon"] == h and r["mae_outage"] == r["mae_outage"]]
        best_base[h] = min(cands, key=lambda x: x[1])[0] if cands else None
    dm = {}
    if gamma_key in outage_by_model:
        g = outage_by_model[gamma_key]
        for h in horizons:
            comps = {}
            gh = g[g["horizon"] == h]
            for ckey, cname in (("vs_persistence", floor_key), ("vs_best_baseline", best_base.get(h))):
                if cname is None or cname not in outage_by_model:
                    comps[ckey] = {"competitor": cname, "dm": float("nan"), "p": float("nan"), "n": 0}
                    continue
                c = outage_by_model[cname]; c = c[c["horizon"] == h]
                merged = gh.merge(c, on=["t", "station", "horizon"], suffixes=("_g", "_c"))
                if len(merged) < 8:
                    comps[ckey] = {"competitor": cname, "dm": float("nan"), "p": float("nan"), "n": len(merged)}
                    continue
                e_g = merged["y_true_g"].to_numpy() - merged["y_pred_g"].to_numpy()
                e_c = merged["y_true_c"].to_numpy() - merged["y_pred_c"].to_numpy()
                dmv, p, n = diebold_mariano(e_g, e_c, h=h)
                comps[ckey] = {"competitor": cname, "dm": dmv, "p": p, "n": n}
            dm[int(h)] = comps
    return {"mae": rows, "best_baseline": best_base, "dm": dm}


def coldstart_summary(cold_df: pd.DataFrame, floor_df: pd.DataFrame, horizons, seq_len: int,
                      edges=(1, 6)) -> dict:
    """Zero-shot cold-start (GAMMA on an unseen station) vs its persistence floor: overall MAE per
    horizon, MAE by staleness bin, and DM vs persistence. Only GAMMA can forecast an unseen station."""
    bins = staleness_bin_labels(seq_len, edges)
    out = {"per_horizon": [], "by_bin": [], "dm": {}, "calibration": {}}
    if cold_df is None or len(cold_df) == 0:
        return out
    cd = cold_df.copy(); cd["bin"] = _assign_bin(cd["staleness"].to_numpy(), bins)
    fd = floor_df.copy() if floor_df is not None and len(floor_df) else None
    for h in horizons:
        c = cd[cd["horizon"] == h]
        gmae = float((c["y_true"] - c["y_pred"]).abs().mean()) if len(c) else float("nan")
        fmae = float("nan")
        if fd is not None:
            f = fd[fd["horizon"] == h]
            fmae = float((f["y_true"] - f["y_pred"]).abs().mean()) if len(f) else float("nan")
        out["per_horizon"].append({"horizon": int(h), "gamma_mae": gmae, "persistence_mae": fmae,
                                   "n": int(len(c))})
        out["calibration"][int(h)] = calibration(c)
        for b in [bb[0] for bb in bins]:
            cb = c[c["bin"] == b]
            out["by_bin"].append({"horizon": int(h), "bin": b,
                                  "gamma_mae": float((cb["y_true"] - cb["y_pred"]).abs().mean()) if len(cb) else float("nan"),
                                  "n": int(len(cb))})
        if fd is not None:
            merged = c.merge(fd[fd["horizon"] == h], on=["t", "station", "horizon"], suffixes=("_g", "_f"))
            if len(merged) >= 8:
                e_g = merged["y_true_g"].to_numpy() - merged["y_pred_g"].to_numpy()
                e_f = merged["y_true_f"].to_numpy() - merged["y_pred_f"].to_numpy()
                dmv, p, nn = diebold_mariano(e_g, e_f, h=h)
                out["dm"][int(h)] = {"dm": dmv, "p": p, "n": nn}
    return out


def staleness_bin_labels(seq_len: int, edges=(1, 6)) -> list[tuple[str, float, float]]:
    """Pre-registered bins by input-side PM2.5 staleness (hours since the station last reported).

    Returns ordered (label, lo, hi) with lo < staleness <= hi (the first bin includes 0). The final
    `self_blackout` bin is staleness >= seq_len: the station has NO observed PM2.5 in its whole input
    window, so per-station baselines and persistence are structurally blind and only GAMMA (neighbours)
    can forecast it.
    """
    e1, e2 = edges
    return [
        (f"fresh(<={e1}h)", -1.0, float(e1)),
        (f"moderate({e1+1}-{e2}h)", float(e1), float(e2)),
        (f"stale({e2+1}-{seq_len-1}h)", float(e2), float(seq_len - 1)),
        (f"self_blackout(>={seq_len}h)", float(seq_len - 1), float("inf")),
    ]


def _assign_bin(staleness: np.ndarray, bins) -> np.ndarray:
    out = np.empty(len(staleness), dtype=object)
    for label, lo, hi in bins:
        out[(staleness > lo) & (staleness <= hi)] = label
    return out


def staleness_strata(pm25_by_model: dict, horizons, seq_len: int, baseline_names,
                     edges=(1, 6), gamma_key="GAMMA", floor_key="persistence") -> dict:
    """Stratify PM2.5 test errors by input staleness; the cross-station capability lives here.

    Returns {"bins": [...], "mae": [rows], "dm": {bin: {h: {competitor: (dm,p,n)}}}}. DM (GAMMA vs
    persistence and vs the best baseline) is computed only in the stale + self_blackout bins, where
    the unique-to-GAMMA neighbour routing should pay off.
    """
    bins = staleness_bin_labels(seq_len, edges)
    bin_order = [b[0] for b in bins]
    # tag every model's rows with a bin label
    tagged = {}
    for name, df in pm25_by_model.items():
        if df is None or len(df) == 0 or "staleness" not in df:
            continue
        d = df.copy()
        d["bin"] = _assign_bin(d["staleness"].to_numpy(), bins)
        tagged[name] = d

    mae_rows = []
    for name, d in tagged.items():
        err = (d["y_true"] - d["y_pred"]).abs()
        g = d.assign(abs_err=err).groupby(["horizon", "bin"], as_index=False).agg(
            mae=("abs_err", "mean"), n=("abs_err", "size"))
        for _, r in g.iterrows():
            mae_rows.append({"model": name, "horizon": int(r["horizon"]),
                             "bin": r["bin"], "mae": float(r["mae"]), "n": int(r["n"])})

    # best baseline per horizon by OVERALL pm25 MAE (across all bins)
    best_baseline = {}
    for h in horizons:
        cands = [(name, ((d[d.horizon == h]["y_true"] - d[d.horizon == h]["y_pred"]).abs().mean()))
                 for name, d in tagged.items() if name in baseline_names and (d.horizon == h).any()]
        cands = [(n, m) for n, m in cands if m == m]
        best_baseline[h] = min(cands, key=lambda x: x[1])[0] if cands else None

    dm = {}
    if gamma_key in tagged:
        g_all = tagged[gamma_key]
        focus_bins = [b for b in bin_order if b.startswith(("stale", "self_blackout"))]
        for blabel in focus_bins:
            dm[blabel] = {}
            for h in horizons:
                comps = {}
                gb = g_all[(g_all.horizon == h) & (g_all.bin == blabel)]
                for ckey, cname in (("vs_persistence", floor_key),
                                    ("vs_best_baseline", best_baseline.get(h))):
                    if cname is None or cname not in tagged:
                        comps[ckey] = {"competitor": cname, "dm": float("nan"),
                                       "p": float("nan"), "n": 0}
                        continue
                    cb = tagged[cname]
                    cb = cb[(cb.horizon == h) & (cb.bin == blabel)]
                    merged = gb.merge(cb, on=["t", "station", "horizon"], suffixes=("_g", "_c"))
                    if len(merged) < 8:
                        comps[ckey] = {"competitor": cname, "dm": float("nan"),
                                       "p": float("nan"), "n": len(merged)}
                        continue
                    e_g = merged["y_true_g"].to_numpy() - merged["y_pred_g"].to_numpy()
                    e_c = merged["y_true_c"].to_numpy() - merged["y_pred_c"].to_numpy()
                    dmv, p, n = diebold_mariano(e_g, e_c, h=h)
                    comps[ckey] = {"competitor": cname, "dm": dmv, "p": p, "n": n}
                dm[blabel][h] = comps
    return {"bins": bin_order, "best_baseline": best_baseline, "mae": mae_rows, "dm": dm}


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
