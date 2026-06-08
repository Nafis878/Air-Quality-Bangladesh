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

from .channels import CHANNELS
from .losses import MaskedQuantileLoss, median_index
from .metrics import aggregate
from .windows import valid_anchors
from .train import build_model, count_params, is_gamma, set_seed


class _DeviceArrays:
    """PanelArrays as device tensors + cached anchor index lists."""

    def __init__(self, arr, seq_len, horizons, stride, device):
        self.seq_len, self.horizons, self.device = seq_len, horizons, device
        self.X = torch.from_numpy(arr.x).to(device)              # [T,S,C]
        self.M = torch.from_numpy(arr.m).to(device)
        self.D = torch.from_numpy(arr.d).to(device)
        self.V = torch.from_numpy(np.nan_to_num(arr.val_scaled, nan=0.0)).to(device)
        self.Vmask = torch.from_numpy((~np.isnan(arr.val_scaled)).astype(np.float32)).to(device)
        self.present = torch.from_numpy(arr.present.astype(np.float32)).to(device)  # [T,S]
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
        ys = {h: (self.V[t + h, s], self.Vmask[t + h, s]) for h in self.horizons}  # [B,C]
        return x, d, t, s, ys

    def gamma_batch(self, idx):
        t = self.t_cs[idx]
        win = t[:, None] + self.offsets                                           # [B,seq]
        x = self.X[win].permute(0, 2, 1, 3).contiguous()                          # [B,S,seq,C]
        m = self.M[win].permute(0, 2, 1, 3).contiguous()
        d = self.D[win].permute(0, 2, 1, 3).contiguous()
        present = self.present[t]                                                  # [B,S]
        ys = {h: (self.V[t + h], self.Vmask[t + h]) for h in self.horizons}        # [B,S,C]
        return x, m, d, present, t, ys


def _forward_loss(name, model, dev: _DeviceArrays, idx, crit, horizons):
    if is_gamma(name):
        x, m, d, present, t, ys = dev.gamma_batch(idx)
        st = torch.arange(dev.S, device=dev.device).unsqueeze(0).expand(len(t), -1)
        preds = model(x, m, d, present, st)                       # [B,S,H,C,Q]
        loss = 0.0
        for hi, h in enumerate(horizons):
            y, ym = ys[h]
            loss = loss + crit(preds[:, :, hi], y, ym)
        return loss
    x, d, t, s, ys = dev.baseline_batch(idx)
    preds = model(x)                                              # [B,H,C,Q]
    loss = 0.0
    for hi, h in enumerate(horizons):
        y, ym = ys[h]
        loss = loss + crit(preds[:, hi], y, ym)
    return loss


def _n_anchors(name, dev):
    return len(dev.t_cs) if is_gamma(name) else len(dev.t_bl)


@torch.no_grad()
def _eval_loss(name, model, dev, crit, horizons, bs):
    model.eval()
    n = _n_anchors(name, dev)
    tot, nb = 0.0, 0
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=dev.device)
        tot += float(_forward_loss(name, model, dev, idx, crit, horizons)); nb += 1
    return tot / max(nb, 1)


def train_model(name, prep, cfg, seed, gamma_kwargs=None, device="cpu", verbose=False):
    set_seed(seed)
    horizons, bs = cfg["horizons"], cfg["batch_size"]
    dev = {s: _DeviceArrays(prep.arrays[s], cfg["seq_len"], horizons, cfg["stride"][s], device)
           for s in ("train", "val")}
    model = build_model(name, len(prep.stations), cfg, gamma_kwargs).to(device)
    crit = MaskedQuantileLoss(cfg["quantiles"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n_train = _n_anchors(name, dev["train"])

    best_val, best_state, best_epoch, bad, history = float("inf"), None, -1, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = _forward_loss(name, model, dev["train"], idx, crit, horizons)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
        vl = _eval_loss(name, model, dev["val"], crit, horizons, bs)
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
def collect_predictions(name, model, prep, cfg, device="cpu"):
    """Long df [t, station, horizon, channel, y_true, y_pred] on TEST (physical, observed only)."""
    model.eval()
    horizons, bs = cfg["horizons"], cfg["batch_size"]
    qmid = median_index(cfg["quantiles"])
    dev = _DeviceArrays(prep.arrays["test"], cfg["seq_len"], horizons, cfg["stride"]["test"], device)
    mu = torch.tensor([prep.scaler.global_[c][0] for c in CHANNELS], device=device)
    sd = torch.tensor([prep.scaler.global_[c][1] for c in CHANNELS], device=device)
    rows = []
    n = _n_anchors(name, dev)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=device)
        if is_gamma(name):
            x, m, d, present, t, ys = dev.gamma_batch(idx)
            st = torch.arange(dev.S, device=device).unsqueeze(0).expand(len(t), -1)
            preds = model(x, m, d, present, st)[..., qmid]            # [B,S,H,C]
            for hi, h in enumerate(horizons):
                y, ym = ys[h]
                yp = preds[:, :, hi] * sd + mu
                yt = y * sd + mu
                bi, si, ci = torch.nonzero(ym > 0, as_tuple=True)
                rows.append(pd.DataFrame({
                    "t": t[bi].cpu().numpy(), "station": si.cpu().numpy(), "horizon": h,
                    "channel": ci.cpu().numpy(),
                    "y_true": yt[bi, si, ci].cpu().numpy(), "y_pred": yp[bi, si, ci].cpu().numpy()}))
        else:
            x, d, t, s, ys = dev.baseline_batch(idx)
            preds = model(x)[..., qmid]                                # [B,H,C]
            for hi, h in enumerate(horizons):
                y, ym = ys[h]
                yp = preds[:, hi] * sd + mu
                yt = y * sd + mu
                bi, ci = torch.nonzero(ym > 0, as_tuple=True)
                rows.append(pd.DataFrame({
                    "t": t[bi].cpu().numpy(), "station": s[bi].cpu().numpy(), "horizon": h,
                    "channel": ci.cpu().numpy(),
                    "y_true": yt[bi, ci].cpu().numpy(), "y_pred": yp[bi, ci].cpu().numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["t", "station", "horizon", "channel", "y_true", "y_pred"])
