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

from .channels import CHANNELS, HORIZONS, PM25_IDX
from .dataprep import prepare, load_config, inter_station_corr, build_cold_start
from .fasttrain import train_model, collect_predictions, outage_predictions, cold_start_predictions
from .train import build_model, count_params, is_gamma
from .evaluate import (metrics_by_horizon, pm25_frame, diebold_mariano, holm_bonferroni,
                       outage_summary, coldstart_summary)
from .models.baselines import BASELINES
from .models.naive import NaiveForecasters, NAIVE_METHODS, _ffill_time
from .windows import valid_anchors, StationWindowDataset
from .ablations import ABLATIONS, split_ablation

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


def train_and_eval_model(name, prep, cfg, seeds, device, gamma_kwargs=None, verbose=False,
                         use_anchor=True):
    """Train across seeds; return (summary_rows, seed_mean_pm25_df, n_params, val_losses).

    The seed-mean PM2.5 frame carries `staleness` (identical across seeds) for the stratified
    cross-station capability analysis.
    """
    per_seed_metrics, pm25_per_seed, val_losses, n_params, model0 = [], [], [], None, None
    for seed in seeds:
        t0 = time.time()
        res = train_model(name, prep, cfg, seed, gamma_kwargs=gamma_kwargs, device=device,
                          verbose=verbose, use_anchor=use_anchor)
        n_params = res["n_params"]; val_losses.append(res["val_loss"])
        if model0 is None:
            model0 = res["model"]                          # keep seed-0 model for outage/cold-start
        pred = collect_predictions(name, res["model"], prep, cfg, device, use_anchor=use_anchor)
        per_seed_metrics.append(metrics_by_horizon(pred, cfg["horizons"]))
        pm25_per_seed.append(pred[pred["channel"] == CHANNELS.index("pm25")]
                             [["t", "station", "horizon", "y_true", "y_pred", "y_lo", "y_hi", "staleness"]])
        print(f"  [{name}] seed {seed}: val={res['val_loss']:.5f} "
              f"test_MAE_pm25_micro={_pm_mae(per_seed_metrics[-1]):.3f} ({time.time()-t0:.0f}s)")
    # average PM2.5 predictions across seeds for DM (staleness/intervals seed-aggregated)
    keyed = pd.concat(pm25_per_seed).groupby(["t", "station", "horizon"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"), y_lo=("y_lo", "mean"),
        y_hi=("y_hi", "mean"), staleness=("staleness", "first"))
    return summarize_seeds(per_seed_metrics, cfg["horizons"]), keyed, n_params, val_losses, model0


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
    pm = CHANNELS.index("pm25")
    stale_pm25 = test_arr.d[t_idx, s_idx, pm]                                     # hours since PM2.5 obs
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
            keep = m[:, pm] > 0
            pm25_rows.append(pd.DataFrame({"t": t_idx[keep], "station": s_idx[keep], "horizon": h,
                                           "y_true": yt[keep, pm], "y_pred": yp[keep, pm],
                                           "staleness": stale_pm25[keep]}))
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


def _best_baseline(pm25_by_model: dict, horizons) -> str | None:
    """Baseline with the lowest overall PM2.5 MAE at the longest horizon."""
    h = max(horizons)
    best, bm = None, float("inf")
    for name in BASELINES:
        df = pm25_by_model.get(name)
        if df is None or len(df) == 0:
            continue
        sub = df[df["horizon"] == h]
        m = float((sub["y_true"] - sub["y_pred"]).abs().mean()) if len(sub) else float("inf")
        if m < bm:
            best, bm = name, m
    return best


def run_outage_and_coldstart(prep, cfg, device, use_anchor, seed0, pm25_by_model, params):
    """C3 outage-counterfactual (degradation vs simulated outage length) + C2 cold-start nowcast."""
    horizons = cfg["horizons"]
    best_base = _best_baseline(pm25_by_model, horizons)
    outage_models = {"GAMMA": seed0["GAMMA"]}
    if seed0.get("GAMMA_no_spatial") is not None:
        outage_models["GAMMA_no_spatial"] = seed0["GAMMA_no_spatial"]
    if best_base and seed0.get(best_base) is not None:
        outage_models[best_base] = seed0[best_base]

    # persistence under outage = the pre-outage last value (analytic, leakage-safe)
    test_arr = prep.arrays["test"]
    ff = _ffill_time(test_arr.val_scaled)
    pm_mu, pm_sd = prep.scaler.global_["pm25"]

    print("\n== C3 outage counterfactual ==")
    outage = {}
    for olen in cfg.get("outage_lengths", [6, 12, 24]):
        ob = {}
        for nm, mdl in outage_models.items():
            ob[nm] = outage_predictions(nm, mdl, prep, cfg, device, use_anchor, outage_len=olen)
        # persistence-outage aligned to GAMMA's points
        g = ob["GAMMA"]
        po = g[["t", "station", "horizon", "y_true"]].copy()
        src = (g["t"].to_numpy() - olen).clip(min=0)
        po["y_pred"] = ff[src, g["station"].to_numpy(), PM25_IDX] * pm_sd + pm_mu
        ob["persistence"] = po
        summ = outage_summary(ob, pm25_by_model, horizons, baseline_names=set(BASELINES.keys()))
        outage[str(olen)] = summ
        print(f"  outage_len={olen}: " + ", ".join(
            f"{r['model']} {r['mae_outage']:.1f}(+{r['degradation']:.1f})"
            for r in summ["mae"] if r["horizon"] == max(horizons)))

    print("\n== C2 cold-start ==")
    coldstart = {}
    cold = build_cold_start(prep, cfg.get("cold_start_station", "DoE"))
    if cold is not None:
        cdf = cold_start_predictions(seed0["GAMMA"], cold, cfg, device, use_anchor, scaler=prep.scaler)
        # persistence floor for the cold station (its own history works)
        cff = _ffill_time(cold["arr"].val_scaled); ci = cold["cold_index"]
        floor = cdf[["t", "station", "horizon", "y_true"]].copy()
        floor["y_pred"] = cff[floor["t"].to_numpy(), ci, PM25_IDX] * pm_sd + pm_mu
        coldstart = coldstart_summary(cdf, floor, horizons, cfg["seq_len"],
                                      edges=tuple(cfg.get("staleness_bins", [1, 6])))
        coldstart["station"] = cold["cold_station"]
        ph = {r["horizon"]: r for r in coldstart["per_horizon"]}
        print(f"  cold-start {cold['cold_station']}: " + ", ".join(
            f"t+{h} GAMMA {ph[h]['gamma_mae']:.1f} vs persist {ph[h]['persistence_mae']:.1f}"
            for h in horizons if h in ph))
    else:
        print("  (cold-start station has no test rows — skipped)")
    return {"outage": outage, "coldstart": coldstart}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/modeling.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default=None, help="comma list; default all")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--stride-train", type=int, default=None, help="override train anchor stride")
    ap.add_argument("--gamma-stride", type=int, default=None,
                    help="override GAMMA's train anchor stride (throttle its density on CPU)")
    ap.add_argument("--seeds", default=None, help="override seeds, comma list e.g. 0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch size")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (GPU: try 4)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        q = cfg["quick"]; cfg["stride"] = q["stride"]; cfg["max_epochs"] = q["max_epochs"]; cfg["seeds"] = q["seeds"]
        cfg["gamma_stride"] = q.get("gamma_stride", cfg.get("gamma_stride")); cfg["gamma_max_epochs"] = q.get("gamma_max_epochs", cfg.get("gamma_max_epochs"))
        cfg["outage_lengths"] = q.get("outage_lengths", cfg.get("outage_lengths"))
    if args.stride_train is not None:
        cfg["stride"] = {**cfg["stride"], "train": args.stride_train}
    if args.gamma_stride is not None:
        cfg["gamma_stride"] = args.gamma_stride
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.seeds is not None:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    seeds = cfg["seeds"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    # GPU is data-loading bound for the small baselines -> parallelize batch assembly.
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    elif device == "cuda":
        cfg["num_workers"] = 4
    cfg["pin_memory"] = (device == "cuda")
    print(f"== device: {device} | num_workers: {cfg.get('num_workers', 0)} ==")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    print(f"== preparing data =="); t0 = time.time()
    prep = prepare(cfg)
    print(f"stations({len(prep.stations)}): {prep.stations}")
    print(f"excluded: {prep.excluded}")
    print(f"prepare done in {time.time()-t0:.1f}s | seeds={seeds} | strides={cfg['stride']}")

    # TRAIN-only inter-station correlation prior for GAMMA's spatial graph (no coordinates exist).
    adj_prior = inter_station_corr(prep.arrays["train"])
    use_anchor = bool(cfg.get("use_anchor", True))
    print(f"anchor (persistence residual) for all models: {use_anchor}")

    roster = args.models.split(",") if args.models else ROSTER
    all_summary, pm25_by_model, params, valloss, seed0 = {}, {}, {}, {}, {}
    for name in roster:
        print(f"\n== {name} ({len(seeds)} seeds) ==")
        gk = {"adj_prior": adj_prior} if is_gamma(name) else None
        summ, pm25, npar, vls, m0 = train_and_eval_model(name, prep, cfg, seeds, device,
                                                         gamma_kwargs=gk, use_anchor=use_anchor)
        all_summary[name] = summ; pm25_by_model[name] = pm25; params[name] = npar
        valloss[name] = vls; seed0[name] = m0          # seed-0 model for outage / cold-start

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
            gkw, abl_anchor = split_ablation(kw)
            gkw = {**gkw, "adj_prior": adj_prior}
            print(f"-- {vname} ({len(abl_seeds)} seeds) --")
            summ, abl_pm25, npar, _, abl_m0 = train_and_eval_model(
                "GAMMA", prep, cfg, abl_seeds, device, gamma_kwargs=gkw, use_anchor=abl_anchor)
            ablation_summary[vname] = {"summary": summ, "n_params": npar}
            pm25_by_model[vname] = abl_pm25      # for the staleness-stratified capability analysis
            seed0[vname] = abl_m0

    # ---- staleness-stratified capability analysis (the uniquely-GAMMA story) ----
    from .evaluate import (staleness_strata, calibration_by_model, outage_summary, coldstart_summary)
    strata = {}
    if "GAMMA" in roster:
        strata = staleness_strata(pm25_by_model, cfg["horizons"], cfg["seq_len"],
                                  set(BASELINES.keys()),
                                  edges=tuple(cfg.get("staleness_bins", [1, 6])))

    # ---- C4 calibration (PICP/MPIW from the quantile heads) ----
    calib = calibration_by_model(pm25_by_model, cfg["horizons"],
                                 nominal=cfg.get("calibration_nominal", 0.8))

    # ---- C3 controlled outage counterfactual + C2 cold-start (need trained seed-0 models) ----
    outage, coldstart = {}, {}
    if "GAMMA" in roster and seed0.get("GAMMA") is not None:
        outage = run_outage_and_coldstart(prep, cfg, device, use_anchor, seed0, pm25_by_model, params)

    # ---- persist ----
    payload = {
        "stations": prep.stations, "excluded": prep.excluded, "seeds": seeds,
        "config": {k: cfg[k] for k in ("seq_len", "horizons", "quantiles", "stride", "max_epochs", "d_model")},
        "use_anchor": use_anchor,
        "params": params, "val_loss": valloss,
        "summary": all_summary, "ablations": ablation_summary,
        "dm": {str(h): {"dm": v["dm"], "holm": v["holm"]} for h, v in dm.items()},
        "stratified": strata, "calibration": calib,
        "outage": outage.get("outage", {}), "coldstart": outage.get("coldstart", {}),
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
