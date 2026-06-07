"""End-to-end honest evaluation: train -> test -> tables/figures. One command, reproducible.

    python -m src.modeling.run --config configs/modeling.yaml            # full study
    python -m src.modeling.run --config configs/modeling.yaml --quick    # fast smoke
    python -m src.modeling.run --config configs/modeling.yaml --models GAMMA,GRU --ablations

The test split is used ONLY for final metrics here; selection/early-stopping used val.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .channels import CHANNELS, HORIZONS
from .dataprep import prepare, load_config
from .train import train_model, build_model, count_params, is_gamma
from .evaluate import (collect_predictions, metrics_by_horizon, pm25_frame,
                       diebold_mariano, holm_bonferroni)
from .models.baselines import BASELINES
from .models.naive import NaiveForecasters, NAIVE_METHODS
from .windows import valid_anchors, StationWindowDataset
from .ablations import ABLATIONS

ROSTER = ["GAMMA", *BASELINES.keys()]


def _ci95(vals: list[float]):
    a = np.array([v for v in vals if v == v], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    mean, sd = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    half = float(stats.t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return mean, sd, half


def summarize_seeds(per_seed_metrics: list[dict], horizons) -> list[dict]:
    """per_seed_metrics: list over seeds of {horizon: aggregate(...)}. -> long rows."""
    rows = []
    for h in horizons:
        for agg in ("micro", "macro"):
            for metric in ("mae", "rmse", "r2"):
                vals = [m[h][agg][metric] for m in per_seed_metrics if agg in m[h]]
                mean, sd, ci = _ci95(vals)
                rows.append({"horizon": h, "agg": agg, "metric": metric,
                             "mean": mean, "std": sd, "ci95": ci, "n_seeds": len(vals)})
    return rows


def train_and_eval_model(name, prep, cfg, seeds, device, gamma_kwargs=None, verbose=False):
    """Train across seeds; return (summary_rows, seed_mean_pm25_df, n_params, val_losses)."""
    test_arr = prep.arrays["test"]
    test_anchors = valid_anchors(test_arr, cfg["seq_len"], cfg["horizons"], stride=cfg["stride"]["test"])
    per_seed_metrics, pm25_per_seed, val_losses, n_params = [], [], [], None
    for seed in seeds:
        t0 = time.time()
        res = train_model(name, prep, cfg, seed, gamma_kwargs=gamma_kwargs, device=device, verbose=verbose)
        n_params = res["n_params"]; val_losses.append(res["val_loss"])
        pred = collect_predictions(name, res["model"], test_arr, test_anchors, prep.scaler, cfg, device)
        per_seed_metrics.append(metrics_by_horizon(pred, cfg["horizons"]))
        pm25_per_seed.append(pred[pred["channel"] == CHANNELS.index("pm25")]
                             [["t", "station", "horizon", "y_true", "y_pred"]])
        print(f"  [{name}] seed {seed}: val={res['val_loss']:.5f} "
              f"test_MAE_pm25_micro={_pm_mae(per_seed_metrics[-1]):.3f} ({time.time()-t0:.0f}s)")
    # average PM2.5 predictions across seeds for DM
    keyed = pd.concat(pm25_per_seed).groupby(["t", "station", "horizon"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    return summarize_seeds(per_seed_metrics, cfg["horizons"]), keyed, n_params, val_losses


def _pm_mae(metrics_h: dict) -> float:
    """Micro MAE on PM2.5 at t+24, for a quick progress print (best-effort)."""
    h = 24 if 24 in metrics_h else list(metrics_h)[0]
    pc = metrics_h[h].get("per_channel")
    if pc is None:
        return float("nan")
    row = pc[pc["channel"] == "pm25"]
    return float(row["mae"].iloc[0]) if len(row) else float("nan")


def naive_eval(prep, cfg):
    """Evaluate the three naive floors on test; return {method: summary_rows, pm25_df}."""
    test_arr = prep.arrays["test"]
    anchors = valid_anchors(test_arr, cfg["seq_len"], cfg["horizons"], stride=cfg["stride"]["test"])
    ds = StationWindowDataset(test_arr, anchors, cfg["seq_len"], cfg["horizons"])
    t_idx, s_idx = ds.t, ds.s
    nf = NaiveForecasters(prep.arrays["train"])
    from .metrics import descale, aggregate
    mu = np.array([prep.scaler.global_[c][0] for c in CHANNELS])
    sd = np.array([prep.scaler.global_[c][1] for c in CHANNELS])
    out = {}
    for method in NAIVE_METHODS:
        per_h, pm25_rows = {}, []
        for h in cfg["horizons"]:
            pred_scaled = nf.predict(test_arr, t_idx, s_idx, h, method)          # [N,C]
            yp = pred_scaled * sd + mu
            yt = test_arr.val_scaled[t_idx + h, s_idx, :] * sd + mu               # [N,C] (NaN where missing)
            m = (~np.isnan(yt)).astype(float)
            from .metrics import pointwise_frame
            frame = pointwise_frame(yp, np.nan_to_num(yt, nan=0.0), m, s_idx)
            per_h[h] = aggregate(frame)
            pm = CHANNELS.index("pm25")
            keep = m[:, pm] > 0
            pm25_rows.append(pd.DataFrame({"t": t_idx[keep], "station": s_idx[keep], "horizon": h,
                                           "y_true": yt[keep, pm], "y_pred": yp[keep, pm]}))
        out[method] = {"summary": summarize_seeds([per_h], cfg["horizons"]),
                       "pm25": pd.concat(pm25_rows, ignore_index=True)}
    return out


def run_dm(gamma_pm25: pd.DataFrame, others: dict[str, pd.DataFrame], horizons) -> dict:
    """DM GAMMA vs each competitor on PM2.5 per horizon, with Holm-Bonferroni per horizon."""
    results = {}
    for h in horizons:
        g = gamma_pm25[gamma_pm25["horizon"] == h].sort_values(["station", "t"])
        pvals, stats_ = {}, {}
        for name, df in others.items():
            o = df[df["horizon"] == h]
            merged = g.merge(o, on=["t", "station", "horizon"], suffixes=("_g", "_o"))
            if len(merged) < 8:
                pvals[name] = float("nan"); stats_[name] = float("nan"); continue
            e_g = merged["y_true_g"].to_numpy() - merged["y_pred_g"].to_numpy()
            e_o = merged["y_true_o"].to_numpy() - merged["y_pred_o"].to_numpy()
            dm, p, n = diebold_mariano(e_g, e_o, h=h)
            stats_[name] = dm; pvals[name] = p
        holm = holm_bonferroni(pvals)
        results[h] = {"dm": stats_, "holm": holm}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/modeling.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None, help="comma list; default all")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--stride-train", type=int, default=None, help="override train anchor stride")
    ap.add_argument("--seeds", default=None, help="override seeds, comma list e.g. 0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch size")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        q = cfg["quick"]; cfg["stride"] = q["stride"]; cfg["max_epochs"] = q["max_epochs"]; cfg["seeds"] = q["seeds"]
    if args.stride_train is not None:
        cfg["stride"] = {**cfg["stride"], "train": args.stride_train}
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.seeds is not None:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    seeds = cfg["seeds"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"== device: {device} ==")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    print(f"== preparing data =="); t0 = time.time()
    prep = prepare(cfg)
    print(f"stations({len(prep.stations)}): {prep.stations}")
    print(f"excluded: {prep.excluded}")
    print(f"prepare done in {time.time()-t0:.1f}s | seeds={seeds} | strides={cfg['stride']}")

    roster = args.models.split(",") if args.models else ROSTER
    all_summary, pm25_by_model, params, valloss = {}, {}, {}, {}
    for name in roster:
        print(f"\n== {name} ({len(seeds)} seeds) ==")
        summ, pm25, npar, vls = train_and_eval_model(name, prep, cfg, seeds, device)
        all_summary[name] = summ; pm25_by_model[name] = pm25; params[name] = npar; valloss[name] = vls

    print("\n== naive floors ==")
    naive = naive_eval(prep, cfg)
    for m in NAIVE_METHODS:
        all_summary[m] = naive[m]["summary"]; pm25_by_model[m] = naive[m]["pm25"]; params[m] = 0

    dm = {}
    if "GAMMA" in roster:
        competitors = {k: v for k, v in pm25_by_model.items() if k != "GAMMA"}
        dm = run_dm(pm25_by_model["GAMMA"], competitors, cfg["horizons"])

    ablation_summary = {}
    if args.ablations and "GAMMA" in roster:
        print("\n== GAMMA ablations ==")
        abl_seeds = seeds[:max(1, min(3, len(seeds)))]
        for vname, kw in ABLATIONS.items():
            print(f"-- {vname} ({len(abl_seeds)} seeds) --")
            summ, _, npar, _ = train_and_eval_model("GAMMA", prep, cfg, abl_seeds, device, gamma_kwargs=kw)
            ablation_summary[vname] = {"summary": summ, "n_params": npar}

    # ---- persist ----
    payload = {
        "stations": prep.stations, "excluded": prep.excluded, "seeds": seeds,
        "config": {k: cfg[k] for k in ("seq_len", "horizons", "quantiles", "stride", "max_epochs", "d_model")},
        "params": params, "val_loss": valloss,
        "summary": all_summary, "ablations": ablation_summary,
        "dm": {str(h): {"dm": v["dm"], "holm": v["holm"]} for h, v in dm.items()},
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    # flat table
    flat = []
    for name, rows in all_summary.items():
        for r in rows:
            flat.append({"model": name, "n_params": params.get(name, 0), **r})
    pd.DataFrame(flat).to_csv(os.path.join(args.out, "tables", "test_metrics.csv"), index=False)
    pm_concat = pd.concat([df.assign(model=k) for k, df in pm25_by_model.items()], ignore_index=True)
    pm_concat.to_parquet(os.path.join(args.out, "tables", "pm25_predictions.parquet"), index=False)
    print(f"\nwrote {args.out}/metrics.json and tables/. total {time.time()-t0:.0f}s")

    from .artifacts import generate_all
    generate_all(args.out)
    return payload


if __name__ == "__main__":
    main()
