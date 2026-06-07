"""Seeded training with val early-stopping. Same loss/optimiser contract for every model.

Baselines consume `StationWindowDataset` (single-station windows); GAMMA consumes
`CrossSectionDataset` (station cross-sections). Both are built from the SAME PanelArrays and
anchor set, so models see the same target points. The scaler/station set were already fit on
TRAIN only in `dataprep.prepare`. Early stopping monitors val masked-quantile loss; the test
split is never touched here.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from .channels import CHANNELS, HORIZONS
from .losses import MaskedQuantileLoss
from .windows import valid_anchors, StationWindowDataset, CrossSectionDataset
from .models.baselines import BASELINES
from .models.gamma import GAMMA


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def is_gamma(name: str) -> bool:
    return name == "GAMMA" or name.startswith("GAMMA")


def build_model(name: str, n_stations: int, cfg: dict, gamma_kwargs: dict | None = None):
    C = len(CHANNELS)
    H, Q = len(cfg["horizons"]), len(cfg["quantiles"])
    if is_gamma(name):
        return GAMMA(C, n_stations, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                     seq_len=cfg["seq_len"], n_horizons=H, n_quantiles=Q, **(gamma_kwargs or {}))
    cls = BASELINES[name]
    return cls(C, seq_len=cfg["seq_len"], hidden=cfg["d_model"], n_horizons=H, n_quantiles=Q)


def make_loaders(name: str, prep, cfg: dict):
    seq, hz = cfg["seq_len"], cfg["horizons"]
    stride = cfg["stride"]
    Dset = CrossSectionDataset if is_gamma(name) else StationWindowDataset
    loaders = {}
    for split in ("train", "val", "test"):
        arr = prep.arrays[split]
        anc = valid_anchors(arr, seq, hz, stride=stride[split])
        ds = Dset(arr, anc, seq, hz)
        loaders[split] = DataLoader(ds, batch_size=cfg["batch_size"],
                                    shuffle=(split == "train"), num_workers=0)
    return loaders


def _step_loss(name, model, batch, crit, horizons, device):
    if is_gamma(name):
        x = batch["x"].to(device); mask = batch["mask"].to(device)
        decay = batch["decay"].to(device); present = batch["present"].to(device)
        st = batch["station"].to(device)
        preds = model(x, mask, decay, present, st)            # [B,S,H,C,Q]
        loss = 0.0
        for hi, h in enumerate(horizons):
            loss = loss + crit(preds[:, :, hi], batch[f"y_t{h}"].to(device), batch[f"m_t{h}"].to(device))
        return loss
    x = batch["x"].to(device)
    preds = model(x)                                          # [B,H,C,Q]
    loss = 0.0
    for hi, h in enumerate(horizons):
        loss = loss + crit(preds[:, hi], batch[f"y_t{h}"].to(device), batch[f"m_t{h}"].to(device))
    return loss


@torch.no_grad()
def eval_loss(name, model, loader, crit, horizons, device):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        tot += float(_step_loss(name, model, batch, crit, horizons, device)); n += 1
    return tot / max(n, 1)


def train_model(name: str, prep, cfg: dict, seed: int, gamma_kwargs: dict | None = None,
                device: str = "cpu", verbose: bool = False) -> dict:
    set_seed(seed)
    horizons = cfg["horizons"]
    loaders = make_loaders(name, prep, cfg)
    model = build_model(name, len(prep.stations), cfg, gamma_kwargs).to(device)
    crit = MaskedQuantileLoss(cfg["quantiles"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    history = []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for batch in loaders["train"]:
            opt.zero_grad()
            loss = _step_loss(name, model, batch, crit, horizons, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
        vl = eval_loss(name, model, loaders["val"], crit, horizons, device)
        history.append(vl)
        if verbose:
            print(f"    [{name} seed{seed}] epoch {epoch+1}/{cfg['max_epochs']} val={vl:.6f}")
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
