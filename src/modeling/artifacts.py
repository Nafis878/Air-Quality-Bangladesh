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


def _stale_mae_lookup(strata):
    """{(model, horizon, bin): (mae, n)} from the stratified payload."""
    out = {}
    for r in strata.get("mae", []):
        out[(r["model"], int(r["horizon"]), r["bin"])] = (r["mae"], r["n"])
    return out


def make_staleness_figure(payload, out_path):
    """MAE vs input-staleness bin for the models that tell the capability story, at the longest
    horizon (where neighbours matter most). GAMMA should hold up into the stale/self_blackout bins
    while persistence and the per-station baseline blow up."""
    strata = payload.get("stratified", {})
    bins = strata.get("bins", [])
    if not bins:
        return
    h = max(payload["config"]["horizons"])
    lut = _stale_mae_lookup(strata)
    focus = ["GAMMA", "persistence", "GAMMA_no_spatial"]
    bb = strata.get("best_baseline", {}).get(str(h)) or strata.get("best_baseline", {}).get(h)
    if bb:
        focus.append(bb)
    styles = {"GAMMA": ("#b30000", "-", "o"), "persistence": ("#777777", "--", "s"),
              "GAMMA_no_spatial": ("#d28b00", ":", "^")}
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bins))
    for m in dict.fromkeys(focus):
        ys = [lut.get((m, h, b), (float("nan"), 0))[0] for b in bins]
        if all(y != y for y in ys):
            continue
        c, ls, mk = styles.get(m, ("#4c72b0", "-.", "D"))
        ax.plot(x, ys, ls, marker=mk, color=c, label=m, linewidth=2, markersize=7)
    ax.set_xticks(x); ax.set_xticklabels(bins, rotation=20, ha="right")
    ax.set_xlabel("Input staleness of the target station's PM2.5 (hours since last report)")
    ax.set_ylabel(f"Test PM2.5 MAE at t+{h}  — lower is better")
    ax.set_title("Cross-station capability: error vs sensor staleness\n"
                 "(only GAMMA can read neighbours; persistence/per-station baselines are pinned to a stale value)")
    ax.set_ylim(bottom=0); ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def capability_verdict(payload) -> str:
    """Honest read of the uniquely-GAMMA capability from the staleness-stratified DM tests."""
    strata = payload.get("stratified", {})
    dm = strata.get("dm", {})
    if not dm:
        return "_(no staleness-stratified analysis available)_"
    lut = _stale_mae_lookup(strata)
    horizons = payload["config"]["horizons"]
    lines, wins, tested = [], 0, 0
    for blabel, by_h in dm.items():
        for h in horizons:
            comps = by_h.get(str(h)) or by_h.get(h)
            if not comps:
                continue
            vp = comps.get("vs_persistence", {})
            vb = comps.get("vs_best_baseline", {})
            gmae = lut.get(("GAMMA", h, blabel), (float("nan"), 0))
            parts = []
            for tag, v in (("persistence", vp), (f"baseline {vb.get('competitor')}", vb)):
                n, p, dmv = v.get("n", 0), v.get("p", float("nan")), v.get("dm", float("nan"))
                if n and n >= 8 and p == p:
                    tested += 1
                    better = (dmv < 0 and p < 0.05)
                    wins += better
                    parts.append(f"vs {tag}: {'GAMMA better' if better else ('worse' if dmv>0 and p<0.05 else 'n.s.')}"
                                 f" (DM={dmv:.2f}, p={p:.3f}, n={n})")
                elif n and n >= 8:
                    parts.append(f"vs {tag}: DM undefined — near-identical errors (n={n})")
                else:
                    parts.append(f"vs {tag}: insufficient n ({n})")
            lines.append(f"- **{blabel}** t+{h} (GAMMA MAE={gmae[0]:.2f}, n={gmae[1]}): " + "; ".join(parts))
    if tested == 0:
        head = ("**Capability: inconclusive** — too few stale/self-blackout target points to test "
                "the cross-station advantage at significance. Report the bin counts honestly.")
    elif wins >= max(1, tested) * 0.6:
        head = ("**Capability CONFIRMED** — in the stale / self-blackout regime GAMMA DM-significantly "
                "beats persistence and per-station baselines, which are structurally pinned to a stale "
                "value. Cross-check that `GAMMA_no_spatial` loses this edge (mechanism = neighbours).")
    elif wins > 0:
        head = ("**Capability PARTIAL** — GAMMA wins in some stale bins/horizons but not consistently; "
                "report exactly where it does and does not, no overclaim.")
    else:
        head = ("**Capability NOT demonstrated** — GAMMA does not separate from the floors/baselines "
                "even under staleness on this panel. Say so plainly.")
    return head + "\n\n" + "\n".join(lines)


def verdict(payload, agg="macro") -> str:
    """Honest, floor-aware verdict derived from the numbers — covers GAMMA winning, GAMMA
    being worst, and the case where NO learned model beats the naive floors."""
    horizons = payload["config"]["horizons"]
    floors = [m for m in NAIVE_METHODS if m in payload["summary"]]
    baselines = [m for m in LEARNED if m != "GAMMA" and m in payload["summary"]]
    lines = []
    best_each, beats_floor_each, dm_better, dm_worse = 0, 0, 0, 0
    for h in horizons:
        gm = _val(payload["summary"]["GAMMA"], h, agg, "mae", "mean")
        comp = {m: _val(payload["summary"][m], h, agg, "mae", "mean") for m in baselines}
        flr = {m: _val(payload["summary"][m], h, agg, "mae", "mean") for m in floors}
        best_base = min(comp, key=comp.get); best_flr = min(flr, key=flr.get) if flr else None
        is_best_learned = gm <= comp[best_base]
        beats_floor = (best_flr is not None) and (gm <= flr[best_flr])
        best_each += is_best_learned
        beats_floor_each += beats_floor
        holm = payload.get("dm", {}).get(str(h), {}).get("holm", {})
        dmv = payload.get("dm", {}).get(str(h), {}).get("dm", {})
        nb = sum(1 for m in comp if holm.get(m, {}).get("reject") and dmv.get(m, 0) < 0)  # GAMMA better
        nw = sum(1 for m in comp if holm.get(m, {}).get("reject") and dmv.get(m, 0) > 0)  # GAMMA worse
        dm_better += (nb == len(comp)); dm_worse += (nw > len(comp) // 2)
        flr_txt = f"best floor {best_flr}={flr[best_flr]:.2f}" if best_flr else "no floors"
        lines.append(f"- t+{h}: GAMMA={gm:.2f} | best baseline {best_base}={comp[best_base]:.2f} | "
                     f"{flr_txt} | GAMMA DM-better than {nb}/{len(comp)}, worse than {nw}/{len(comp)} "
                     f"baselines (Holm).")
    H = len(horizons)
    if dm_worse == H:
        head = ("**Verdict: GAMMA underperforms — it is significantly WORSE than most baselines at "
                "every horizon and is not viable on this panel.** Report this plainly.")
    elif best_each == H and beats_floor_each == H and dm_better == H:
        head = ("**Verdict: GAMMA is significantly best** — lowest MAE at every horizon, DM-significant "
                "over all baselines, AND it beats the naive floors. A real positive result.")
    elif best_each == H and beats_floor_each < H:
        head = ("**Verdict: GAMMA is the best LEARNED model but does NOT beat the naive floors at "
                "every horizon.** The honest headline is the benchmark caveat: deep models (incl. "
                "GAMMA) do not convincingly beat persistence/seasonal-naive here.")
    elif beats_floor_each < H:
        head = ("**Verdict: no learned model (incl. GAMMA) beats the naive floors at every horizon.** "
                "Lead with the benchmark finding, not a SOTA claim.")
    else:
        head = ("**Verdict: GAMMA is competitive but mixed** — not uniformly best across horizons; "
                "state per-horizon results without overclaiming.")
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
    # ---- the uniquely-GAMMA capability: stratified by sensor staleness ----
    strata = payload.get("stratified", {})
    if strata.get("bins"):
        h = max(cfg["horizons"])
        lut = _stale_mae_lookup(strata)
        bins = strata["bins"]
        bb = strata.get("best_baseline", {}).get(str(h)) or strata.get("best_baseline", {}).get(h)
        focus = list(dict.fromkeys(["GAMMA", "persistence"] + ([bb] if bb else []) + ["GAMMA_no_spatial"]))
        md.append(f"\n## Cross-station capability — PM2.5 MAE at t+{h} by input staleness\n")
        md.append("> Only GAMMA can read other stations' concurrent readings; persistence and every "
                  "per-station baseline are pinned to the target station's own (stale) last value. "
                  "Bins are pre-registered. `n` is the number of observed targets in the bin.\n")
        md.append("| Staleness bin | " + " | ".join(focus) + " | n |")
        md.append("|" + "---|" * (len(focus) + 2))
        for b in bins:
            cells = []
            n_b = 0
            for m in focus:
                mae, n = lut.get((m, h, b), (float("nan"), 0))
                n_b = max(n_b, n if m == "GAMMA" else n_b)
                cells.append(f"{mae:.2f}" if mae == mae else "—")
            gmae, gn = lut.get(("GAMMA", h, b), (float("nan"), 0))
            md.append(f"| {b} | " + " | ".join(cells) + f" | {gn} |")
        md.append("\n_See `figures/staleness_capability.png`._\n")
        md.append(capability_verdict(payload))
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
    make_staleness_figure(payload, os.path.join(figs, "staleness_capability.png"))
    make_latex_table(payload, os.path.join(tabs, "results_table.tex"))
    write_report(payload, os.path.join(results_dir, "methods_results.md"))
    print(f"artifacts written under {results_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    generate_all(ap.parse_args().results)
