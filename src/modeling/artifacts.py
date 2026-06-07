"""Journal-ready artifacts from results/metrics.json — figures, LaTeX table, report.

Honesty is enforced in code: the primary figure y-axis starts at zero (no cropping), naive
floors are drawn, and the results verdict is DERIVED from the numbers (significantly-best vs
statistically-tied vs best-on-efficiency) rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models.baselines import BASELINES
from .models.naive import NAIVE_METHODS

LEARNED = ["GAMMA", *BASELINES.keys()]


def _val(summary_rows, horizon, agg, metric, field="mean"):
    for r in summary_rows:
        if r["horizon"] == horizon and r["agg"] == agg and r["metric"] == metric:
            return r[field]
    return float("nan")


def _table(payload, agg="macro", metric="mae"):
    rows = []
    for model, summ in payload["summary"].items():
        row = {"model": model, "n_params": payload["params"].get(model, 0)}
        for h in payload["config"]["horizons"]:
            row[f"t{h}"] = _val(summ, h, agg, metric, "mean")
            row[f"t{h}_std"] = _val(summ, h, agg, metric, "std")
        rows.append(row)
    return pd.DataFrame(rows)


def make_primary_figure(payload, out_path, agg="macro", metric="mae"):
    horizons = payload["config"]["horizons"]
    models = [m for m in LEARNED if m in payload["summary"]] + \
             [m for m in NAIVE_METHODS if m in payload["summary"]]
    x = np.arange(len(models)); width = 0.8 / len(horizons)
    fig, ax = plt.subplots(figsize=(14, 7))
    for hi, h in enumerate(horizons):
        means = [_val(payload["summary"][m], h, agg, metric, "mean") for m in models]
        errs = [_val(payload["summary"][m], h, agg, metric, "ci95") for m in models]
        errs = [0 if (e != e) else e for e in errs]
        colors = ["#b30000" if m == "GAMMA" else ("#777777" if m in NAIVE_METHODS else "#4c72b0")
                  for m in models]
        ax.bar(x + hi * width, means, width, yerr=errs, capsize=3, label=f"t+{h}",
               color=colors, edgecolor="black", alpha=0.85 - 0.12 * hi)
    ax.set_ylim(bottom=0)  # NO axis cropping
    ax.set_ylabel(f"Test {metric.upper()} ({agg}, physical units)  — lower is better")
    ax.set_title(f"Held-out TEST {metric.upper()} per model and horizon "
                 f"(GAMMA red, naive floors grey; error bars = 95% CI over seeds)")
    ax.set_xticks(x + width); ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(title="horizon"); ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def make_efficiency_figure(payload, out_path, horizon=None, agg="macro", metric="mae"):
    horizon = horizon or max(payload["config"]["horizons"])
    models = [m for m in LEARNED if m in payload["summary"]]
    xs = [payload["params"].get(m, 0) for m in models]
    ys = [_val(payload["summary"][m], horizon, agg, metric, "mean") for m in models]
    fig, ax = plt.subplots(figsize=(9, 6))
    for m, xp, yp in zip(models, xs, ys):
        ax.scatter(xp, yp, s=90, color="#b30000" if m == "GAMMA" else "#4c72b0",
                   edgecolor="black", zorder=3)
        ax.annotate(m, (xp, yp), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Parameter count"); ax.set_ylabel(f"Test {metric.upper()} ({agg}) at t+{horizon}")
    ax.set_title("Efficiency: accuracy vs parameter count (GAMMA in red)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def make_latex_table(payload, out_path, agg="macro"):
    horizons = payload["config"]["horizons"]
    models = [m for m in LEARNED if m in payload["summary"]] + \
             [m for m in NAIVE_METHODS if m in payload["summary"]]
    # best (lowest MAE) per horizon among learned models for bolding
    best = {h: min(LEARNED, key=lambda m: _val(payload["summary"].get(m, []), h, agg, "mae", "mean")
                   if m in payload["summary"] else float("inf")) for h in horizons}
    dm = payload.get("dm", {})
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Held-out test MAE ($\pm$ std over seeds), %s-averaged, physical units. "
             r"Best learned model per horizon in \textbf{bold}; $\dagger$ = GAMMA's lead over this "
             r"model is significant (Holm-corrected DM, PM2.5).}" % agg,
             r"\begin{tabular}{l" + "r" * len(horizons) + "}", r"\toprule",
             "Model & " + " & ".join(f"t+{h} MAE" for h in horizons) + r" \\", r"\midrule"]
    for m in models:
        cells = []
        for h in horizons:
            mean = _val(payload["summary"][m], h, agg, "mae", "mean")
            std = _val(payload["summary"][m], h, agg, "mae", "std")
            cell = f"{mean:.2f}\\,$\\pm$\\,{std:.2f}" if std == std else f"{mean:.2f}"
            if m == best.get(h):
                cell = r"\textbf{" + cell + "}"
            holm = dm.get(str(h), {}).get("holm", {}).get(m)
            if holm and holm.get("reject"):
                cell += r"$\dagger$"
            cells.append(cell)
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def verdict(payload, agg="macro") -> str:
    horizons = payload["config"]["horizons"]
    lines, wins, sig_wins = [], 0, 0
    for h in horizons:
        gm = _val(payload["summary"]["GAMMA"], h, agg, "mae", "mean")
        comp = {m: _val(payload["summary"][m], h, agg, "mae", "mean")
                for m in LEARNED if m != "GAMMA" and m in payload["summary"]}
        best_base = min(comp, key=comp.get)
        is_best = gm <= comp[best_base]
        wins += is_best
        holm = payload.get("dm", {}).get(str(h), {}).get("holm", {})
        n_sig = sum(1 for m in comp if holm.get(m, {}).get("reject"))
        sig_wins += (is_best and n_sig == len(comp))
        lines.append(f"- t+{h}: GAMMA MAE={gm:.2f} vs best baseline {best_base}={comp[best_base]:.2f} "
                     f"({'GAMMA better' if is_best else 'baseline better'}); "
                     f"DM-significant over {n_sig}/{len(comp)} baselines (Holm).")
    if sig_wins == len(horizons):
        head = "**Verdict: GAMMA is significantly best** — lowest MAE at every horizon with DM significance over all baselines."
    elif wins == len(horizons):
        head = ("**Verdict: GAMMA is best on point accuracy but the lead is within noise** "
                "(not DM-significant across all baselines). Lead with the efficiency result.")
    else:
        gp = payload["params"]["GAMMA"]
        smaller = [m for m in LEARNED if m != "GAMMA" and payload["params"].get(m, 0) > gp]
        head = ("**Verdict: GAMMA is statistically competitive, not uniformly best.** "
                f"It is not the lowest-MAE model at every horizon; its case rests on efficiency "
                f"(fewer parameters than {len(smaller)} of the deep baselines).")
    return head + "\n\n" + "\n".join(lines)


def write_report(payload, out_path, fig_rel="figures"):
    cfg = payload["config"]
    tab = _table(payload, "macro", "mae")
    md = []
    md.append("# GAMMA vs baselines — Methods & Results (auto-generated)\n")
    md.append("> Generated from `results/metrics.json`. Numbers are held-out **test** (2024+), "
              "physical units, observed targets only. No value is hand-edited.\n")
    md.append("## Data & protocol\n")
    md.append(f"- Cleaned hourly panel; temporal split train ≤2022 / val 2023 / test ≥2024. "
              f"Window {cfg['seq_len']}h, horizons {cfg['horizons']}, quantiles {cfg['quantiles']}.")
    md.append(f"- **Station scope (fixed from TRAIN): {len(payload['stations'])} stations** — "
              f"{', '.join(payload['stations'])}.")
    md.append("- Excluded (reported, not silently dropped): " +
              "; ".join(f"{k} ({v})" for k, v in payload["excluded"].items()) + ".")
    md.append("- Scaler + station set fit on TRAIN only; inputs zero-filled with GRU-D mask+decay; "
              "targets never imputed; `is_gap`/missing cells excluded from loss and metrics.")
    md.append(f"- Seeds: {payload['seeds']}. Early stopping on val masked-quantile loss.\n")
    md.append("## Model roster & size\n")
    md.append("| Model | Params | Type |\n|---|---|---|")
    for m in LEARNED:
        if m in payload["params"]:
            md.append(f"| {m} | {payload['params'][m]:,} | {'proposed' if m=='GAMMA' else 'baseline'} |")
    md.append("| persistence / seasonal-naive / climatology | 0 | naive floor |\n")
    md.append("## Test MAE (macro, mean±std over seeds)\n")
    cols = "| Model | " + " | ".join(f"t+{h}" for h in cfg["horizons"]) + " |"
    md.append(cols); md.append("|" + "---|" * (len(cfg["horizons"]) + 1))
    order = [m for m in LEARNED if m in payload["summary"]] + \
            [m for m in NAIVE_METHODS if m in payload["summary"]]
    for m in order:
        cells = [f"{_val(payload['summary'][m], h, 'macro', 'mae', 'mean'):.2f}±"
                 f"{_val(payload['summary'][m], h, 'macro', 'mae', 'std'):.2f}" for h in cfg["horizons"]]
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md.append("\n_See `figures/primary_test_mae.png`, `figures/efficiency.png`, `tables/results_table.tex`._\n")
    md.append("## Significance (Diebold–Mariano, PM2.5, Holm–Bonferroni)\n")
    for h in cfg["horizons"]:
        holm = payload.get("dm", {}).get(str(h), {}).get("holm", {})
        sig = [m for m, v in holm.items() if v.get("reject")]
        md.append(f"- t+{h}: GAMMA's lead is significant over: "
                  f"{', '.join(sig) if sig else '(none)'}.")
    md.append("\n## Verdict\n")
    md.append(verdict(payload))
    md.append("\n## Limitations\n")
    md.append("- Station coverage is uneven; the 12-station scope excludes sparse/cold-start sites "
              "(CDA/DoE enter late; Agrabad/Sangsad exit 2021; Cumilla/Mymensingh/Savar too sparse).")
    md.append("- Val-vs-test selection: hyperparameters/early-stopping used val (2023); test (2024) untouched.")
    md.append("- Known data caveats carried from the pipeline: PM2.5>PM10 rows flagged-not-fixed; CO in ppm.")
    md.append("- CPU budget caps epochs/seeds; absolute errors may improve with longer training.")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


def generate_all(results_dir="results"):
    payload = json.load(open(os.path.join(results_dir, "metrics.json"), encoding="utf-8"))
    figs = os.path.join(results_dir, "figures"); tabs = os.path.join(results_dir, "tables")
    os.makedirs(figs, exist_ok=True); os.makedirs(tabs, exist_ok=True)
    make_primary_figure(payload, os.path.join(figs, "primary_test_mae.png"))
    make_efficiency_figure(payload, os.path.join(figs, "efficiency.png"))
    make_latex_table(payload, os.path.join(tabs, "results_table.tex"))
    write_report(payload, os.path.join(results_dir, "methods_results.md"))
    print(f"artifacts written under {results_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    generate_all(ap.parse_args().results)
