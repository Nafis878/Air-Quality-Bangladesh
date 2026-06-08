"""GPU-resident training: keep the whole panel on-device and gather windows by indexing.

The DataLoader path (windows.py datasets) is fine on CPU but starves a GPU — assembling
thousands of small per-station windows in Python is the bottleneck on a 2-vCPU Colab T4.
Here the dense [T,S,C] arrays live on the device as tensors and each minibatch is built with
a single advanced-indexing gather, so an epoch is a few big GPU kernels instead of thousands
of CPU __getitem__ calls. Same model/loss/contract as `train.py`; results are numerically
equivalent (verified on CPU against the DataLoader path).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from .channels import CHANNELS, WIND_DIR_IDX, WIND_SPEED_IDX
from .losses import MaskedQuantileLoss, median_index
from .metrics import aggregate
from .windows import valid_anchors
from .models.naive import _ffill_time
from .train import build_model, count_params, is_gamma, set_seed

PM25 = CHANNELS.index("pm25")


def geo_to_device(geo: dict | None, device):
    """(coords[S,2], dist_feats[S,S,2], bearing[S,S]) tensors for GAMMA.forward, or None."""
    if geo is None:
        return None
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    return (t(geo["coords"]), t(geo["dist_feats"]), t(geo["bearing"]))


class _DeviceArrays:
    """PanelArrays as device tensors + cached anchor index lists."""

    def __init__(self, arr, seq_len, horizons, stride, device, scaler=None):
        self.seq_len, self.horizons, self.device = seq_len, horizons, device
        self.X = torch.from_numpy(arr.x).to(device)              # [T,S,C]
        self.M = torch.from_numpy(arr.m).to(device)
        self.D = torch.from_numpy(arr.d).to(device)
        self.V = torch.from_numpy(np.nan_to_num(arr.val_scaled, nan=0.0)).to(device)
        self.Vmask = torch.from_numpy((~np.isnan(arr.val_scaled)).astype(np.float32)).to(device)
        # persistence anchor: last OBSERVED value forward-filled (== the persistence floor),
        # scaled space; used as the shared residual base for every model. Leakage-safe (<= t).
        self.A = torch.from_numpy(np.nan_to_num(_ffill_time(arr.val_scaled), nan=0.0)).to(device)
        self.present = torch.from_numpy(arr.present.astype(np.float32)).to(device)  # [T,S]
        # raw wind at every hour for the transport graph (de-scaled; mask = both dir & speed observed)
        self.Wind = None
        if scaler is not None:
            vs = arr.val_scaled
            wd_mu, wd_sd = scaler.global_["wind_dir"]; ws_mu, ws_sd = scaler.global_["wind_speed"]
            wdir = vs[:, :, WIND_DIR_IDX] * wd_sd + wd_mu
            wspd = vs[:, :, WIND_SPEED_IDX] * ws_sd + ws_mu
            wmask = (~np.isnan(vs[:, :, WIND_DIR_IDX])) & (~np.isnan(vs[:, :, WIND_SPEED_IDX]))
            wind = np.stack([np.nan_to_num(wdir), np.nan_to_num(wspd), wmask.astype(np.float32)], axis=-1)
            self.Wind = torch.from_numpy(wind.astype(np.float32)).to(device)   # [T,S,3]
        self.T, self.S, self.C = arr.x.shape
        self.offsets = torch.arange(-seq_len + 1, 1, device=device)               # [seq]
        anc = valid_anchors(arr, seq_len, horizons, stride=stride)
        self.t_cs = torch.as_tensor(anc, dtype=torch.long, device=device)         # gamma anchors
        # baseline (t, s) where station present at anchor
        pres_anchor = arr.present[anc]                                            # [A,S]
        ts, ss = np.nonzero(pres_anchor)
        self.t_bl = torch.as_tensor(anc[ts], dtype=torch.long, device=device)
        self.s_bl = torch.as_tensor(ss, dtype=torch.long, device=device)

    def baseline_batch(self, idx):
        t, s = self.t_bl[idx], self.s_bl[idx]
        win = t[:, None] + self.offsets                                           # [B,seq]
        x = self.X[win, s[:, None]]                                               # [B,seq,C]
        d = self.D[win, s[:, None]]
        anchor = self.A[t, s]                                                     # [B,C] persistence
        ys = {h: (self.V[t + h, s], self.Vmask[t + h, s]) for h in self.horizons}  # [B,C]
        return x, d, t, s, anchor, ys

    def gamma_batch(self, idx):
        t = self.t_cs[idx]
        win = t[:, None] + self.offsets                                           # [B,seq]
        x = self.X[win].permute(0, 2, 1, 3).contiguous()                          # [B,S,seq,C]
        m = self.M[win].permute(0, 2, 1, 3).contiguous()
        d = self.D[win].permute(0, 2, 1, 3).contiguous()
        present = self.present[t]                                                  # [B,S]
        anchor = self.A[t]                                                        # [B,S,C] persistence
        wind = self.Wind[t] if self.Wind is not None else None                    # [B,S,3] raw wind
        ys = {h: (self.V[t + h], self.Vmask[t + h]) for h in self.horizons}        # [B,S,C]
        return x, m, d, present, t, anchor, wind, ys


def _forward_loss(name, model, dev: _DeviceArrays, idx, crit, horizons, use_anchor=True, geo=None):
    if is_gamma(name):
        x, m, d, present, t, anchor, wind, ys = dev.gamma_batch(idx)
        st = torch.arange(dev.S, device=dev.device).unsqueeze(0).expand(len(t), -1)
        preds = model(x, m, d, present, st, wind=wind, geo=geo)    # [B,S,H,C,Q]
        if use_anchor:
            preds = preds + anchor[:, :, None, :, None]           # residual on persistence
        loss = 0.0
        for hi, h in enumerate(horizons):
            y, ym = ys[h]
            loss = loss + crit(preds[:, :, hi], y, ym)
        return loss
    x, d, t, s, anchor, ys = dev.baseline_batch(idx)
    preds = model(x)                                              # [B,H,C,Q]
    if use_anchor:
        preds = preds + anchor[:, None, :, None]                  # residual on persistence
    loss = 0.0
    for hi, h in enumerate(horizons):
        y, ym = ys[h]
        loss = loss + crit(preds[:, hi], y, ym)
    return loss


def _n_anchors(name, dev):
    return len(dev.t_cs) if is_gamma(name) else len(dev.t_bl)


@torch.no_grad()
def _eval_loss(name, model, dev, crit, horizons, bs, use_anchor=True, geo=None):
    model.eval()
    n = _n_anchors(name, dev)
    tot, nb = 0.0, 0
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=dev.device)
        tot += float(_forward_loss(name, model, dev, idx, crit, horizons, use_anchor, geo)); nb += 1
    return tot / max(nb, 1)


def train_model(name, prep, cfg, seed, gamma_kwargs=None, device="cpu", verbose=False,
                use_anchor=True):
    set_seed(seed)
    horizons, bs = cfg["horizons"], cfg["batch_size"]
    # GAMMA may train on a denser anchor stride / more epochs (it is data-starved at stride 3).
    train_stride = cfg["stride"]["train"]
    max_epochs = cfg["max_epochs"]
    if is_gamma(name):
        train_stride = cfg.get("gamma_stride", train_stride)
        max_epochs = cfg.get("gamma_max_epochs", max_epochs)
    sc = prep.scaler if is_gamma(name) else None         # wind only needed by GAMMA
    dev = {"train": _DeviceArrays(prep.arrays["train"], cfg["seq_len"], horizons, train_stride, device, sc),
           "val": _DeviceArrays(prep.arrays["val"], cfg["seq_len"], horizons, cfg["stride"]["val"], device, sc)}
    geo = geo_to_device(prep.geo, device) if is_gamma(name) else None
    model = build_model(name, len(prep.stations), cfg, gamma_kwargs).to(device)
    crit = MaskedQuantileLoss(cfg["quantiles"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_train = _n_anchors(name, dev["train"])

    best_val, best_state, best_epoch, bad, history = float("inf"), None, -1, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = _forward_loss(name, model, dev["train"], idx, crit, horizons, use_anchor, geo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
        vl = _eval_loss(name, model, dev["val"], crit, horizons, bs, use_anchor, geo)
        history.append(vl)
        if verbose:
            print(f"    [{name} seed{seed}] epoch {epoch+1} val={vl:.6f}")
        if vl < best_val - 1e-7:
            best_val, best_state, best_epoch, bad = vl, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg.get("patience", 5):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "val_loss": best_val, "best_epoch": best_epoch,
            "history": history, "n_params": count_params(model), "seed": seed}


@torch.no_grad()
def collect_predictions(name, model, prep, cfg, device="cpu", use_anchor=True):
    """Long df [t, station, horizon, channel, y_true, y_pred, y_lo, y_hi, staleness] on TEST.

    Physical units, observed targets only. `y_lo`/`y_hi` are the outer quantiles (for calibration).
    `staleness` = hours since the station last observed PM2.5 at the anchor (== GRU-D decay at t);
    the input-side, leakage-safe stratifier for the cross-station capability analysis.
    """
    model.eval()
    horizons, bs = cfg["horizons"], cfg["batch_size"]
    qmid = median_index(cfg["quantiles"]); qlo, qhi = 0, len(cfg["quantiles"]) - 1
    sc = prep.scaler if is_gamma(name) else None
    dev = _DeviceArrays(prep.arrays["test"], cfg["seq_len"], horizons, cfg["stride"]["test"], device, sc)
    geo = geo_to_device(prep.geo, device) if is_gamma(name) else None
    mu = torch.tensor([prep.scaler.global_[c][0] for c in CHANNELS], device=device)
    sd = torch.tensor([prep.scaler.global_[c][1] for c in CHANNELS], device=device)
    stale_ts = dev.D[:, :, PM25]                                       # [T,S] hours-since-PM2.5-obs
    rows = []
    n = _n_anchors(name, dev)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=device)
        if is_gamma(name):
            x, m, d, present, t, anchor, wind, ys = dev.gamma_batch(idx)
            st = torch.arange(dev.S, device=device).unsqueeze(0).expand(len(t), -1)
            preds = model(x, m, d, present, st, wind=wind, geo=geo)   # [B,S,H,C,Q]
            if use_anchor:
                preds = preds + anchor[:, :, None, :, None]
            stale = stale_ts[t]                                       # [B,S]
            for hi, h in enumerate(horizons):
                y, ym = ys[h]
                yp = preds[:, :, hi, :, qmid] * sd + mu
                ylo = preds[:, :, hi, :, qlo] * sd + mu
                yhi = preds[:, :, hi, :, qhi] * sd + mu
                yt = y * sd + mu
                bi, si, ci = torch.nonzero(ym > 0, as_tuple=True)
                rows.append(pd.DataFrame({
                    "t": t[bi].cpu().numpy(), "station": si.cpu().numpy(), "horizon": h,
                    "channel": ci.cpu().numpy(),
                    "y_true": yt[bi, si, ci].cpu().numpy(), "y_pred": yp[bi, si, ci].cpu().numpy(),
                    "y_lo": ylo[bi, si, ci].cpu().numpy(), "y_hi": yhi[bi, si, ci].cpu().numpy(),
                    "staleness": stale[bi, si].cpu().numpy()}))
        else:
            x, d, t, s, anchor, ys = dev.baseline_batch(idx)
            preds = model(x)                                          # [B,H,C,Q]
            if use_anchor:
                preds = preds + anchor[:, None, :, None]
            stale = stale_ts[t, s]                                    # [B]
            for hi, h in enumerate(horizons):
                y, ym = ys[h]
                yp = preds[:, hi, :, qmid] * sd + mu
                ylo = preds[:, hi, :, qlo] * sd + mu
                yhi = preds[:, hi, :, qhi] * sd + mu
                yt = y * sd + mu
                bi, ci = torch.nonzero(ym > 0, as_tuple=True)
                rows.append(pd.DataFrame({
                    "t": t[bi].cpu().numpy(), "station": s[bi].cpu().numpy(), "horizon": h,
                    "channel": ci.cpu().numpy(),
                    "y_true": yt[bi, ci].cpu().numpy(), "y_pred": yp[bi, ci].cpu().numpy(),
                    "y_lo": ylo[bi, ci].cpu().numpy(), "y_hi": yhi[bi, ci].cpu().numpy(),
                    "staleness": stale[bi].cpu().numpy()}))
    cols = ["t", "station", "horizon", "channel", "y_true", "y_pred", "y_lo", "y_hi", "staleness"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


@torch.no_grad()
def outage_predictions(name, model, prep, cfg, device="cpu", use_anchor=True, outage_len=12):
    """PM2.5 predictions when the TARGET station's own last `outage_len` input hours are blacked out
    (sensor dead), neighbours intact. Restricted to originally-FRESH points (staleness<=1) so the
    target is known. GAMMA can recover from neighbours; per-station baselines see a dead window.

    Returns long df [t, station, horizon, y_true, y_pred] (physical, PM2.5).
    """
    model.eval()
    horizons, bs = cfg["horizons"], cfg["batch_size"]
    qmid = median_index(cfg["quantiles"]); L = cfg["seq_len"]
    sc = prep.scaler if is_gamma(name) else None
    dev = _DeviceArrays(prep.arrays["test"], cfg["seq_len"], horizons, cfg["stride"]["test"], device, sc)
    geo = geo_to_device(prep.geo, device) if is_gamma(name) else None
    mu = float(prep.scaler.global_["pm25"][0]); sd = float(prep.scaler.global_["pm25"][1])
    stale_ts = dev.D[:, :, PM25]
    rows, n = [], _n_anchors(name, dev)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=device)
        if is_gamma(name):
            x, m, d, present, t, anchor, wind, ys = dev.gamma_batch(idx)
            st = torch.arange(dev.S, device=device).unsqueeze(0).expand(len(t), -1)
            for star in range(dev.S):                                  # black out one station at a time
                xb, mb, db = x.clone(), m.clone(), d.clone()
                xb[:, star, L - outage_len:, :] = 0.0
                mb[:, star, L - outage_len:, :] = 0.0
                db[:, star, L - outage_len:, :] = float(L)             # reads as fully stale
                preds = model(xb, mb, db, present, st, wind=wind, geo=geo)
                if use_anchor:                                          # anchor becomes pre-outage value
                    pre = dev.A[(t - outage_len).clamp_min(0)]         # [B,S,C]
                    preds = preds + pre[:, :, None, :, None]
                for hi, h in enumerate(horizons):
                    y, ym = ys[h]
                    keep = (ym[:, star, PM25] > 0) & (present[:, star] > 0) & (stale_ts[t, star] <= 1)
                    if keep.sum() == 0:
                        continue
                    yp = preds[keep, star, hi, PM25, qmid] * sd + mu
                    yt = y[keep, star, PM25] * sd + mu
                    rows.append(pd.DataFrame({"t": t[keep].cpu().numpy(), "station": star, "horizon": h,
                                              "y_true": yt.cpu().numpy(), "y_pred": yp.cpu().numpy()}))
        else:
            x, d, t, s, anchor, ys = dev.baseline_batch(idx)
            xb = x.clone(); xb[:, L - outage_len:, :] = 0.0            # dead recent window
            preds = model(xb)
            if use_anchor:
                pre = dev.A[(t - outage_len).clamp_min(0), s]         # [B,C] pre-outage persistence
                preds = preds + pre[:, None, :, None]
            for hi, h in enumerate(horizons):
                y, ym = ys[h]
                keep = (ym[:, PM25] > 0) & (stale_ts[t, s] <= 1)
                if keep.sum() == 0:
                    continue
                yp = preds[keep, hi, PM25, qmid] * sd + mu
                yt = y[keep, PM25] * sd + mu
                rows.append(pd.DataFrame({"t": t[keep].cpu().numpy(), "station": s[keep].cpu().numpy(),
                                          "horizon": h, "y_true": yt.cpu().numpy(), "y_pred": yp.cpu().numpy()}))
    cols = ["t", "station", "horizon", "y_true", "y_pred"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


@torch.no_grad()
def cold_start_predictions(model, cold, cfg, device="cpu", use_anchor=True, scaler=None):
    """Zero-shot PM2.5 nowcast of the cold-start station (never trained on) from neighbours only.
    `cold` is from dataprep.build_cold_start. GAMMA only — it is inductive (coord embedding + coord/
    wind edges). Returns long df [t, station, horizon, y_true, y_pred, y_lo, y_hi, staleness]."""
    model.eval()
    horizons, bs = cfg["horizons"], cfg["batch_size"]
    qmid = median_index(cfg["quantiles"]); qlo, qhi = 0, len(cfg["quantiles"]) - 1
    arr, ci = cold["arr"], cold["cold_index"]
    dev = _DeviceArrays(arr, cfg["seq_len"], horizons, cfg["stride"]["test"], device, scaler)
    geo = geo_to_device(cold["geo"], device)
    mu = float(scaler.global_["pm25"][0]); sd = float(scaler.global_["pm25"][1])
    stale_ts = dev.D[:, :, PM25]
    rows, n = [], len(dev.t_cs)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=device)
        x, m, d, present, t, anchor, wind, ys = dev.gamma_batch(idx)
        st = torch.arange(dev.S, device=device).unsqueeze(0).expand(len(t), -1)
        preds = model(x, m, d, present, st, wind=wind, geo=geo)
        if use_anchor:
            preds = preds + anchor[:, :, None, :, None]
        for hi, h in enumerate(horizons):
            y, ym = ys[h]
            keep = ym[:, ci, PM25] > 0
            if keep.sum() == 0:
                continue
            rows.append(pd.DataFrame({
                "t": t[keep].cpu().numpy(), "station": cold["cold_station"], "horizon": h,
                "y_true": (y[keep, ci, PM25] * sd + mu).cpu().numpy(),
                "y_pred": (preds[keep, ci, hi, PM25, qmid] * sd + mu).cpu().numpy(),
                "y_lo": (preds[keep, ci, hi, PM25, qlo] * sd + mu).cpu().numpy(),
                "y_hi": (preds[keep, ci, hi, PM25, qhi] * sd + mu).cpu().numpy(),
                "staleness": stale_ts[t, ci][keep].cpu().numpy()}))
    cols = ["t", "station", "horizon", "y_true", "y_pred", "y_lo", "y_hi", "staleness"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
