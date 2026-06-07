"""
Data-quality report generator.

Turns the build run-log + the cleaned parquet into `data_quality_report.md` (plus a
missingness heatmap and distribution figures). This report is the human sign-off
artifact: it surfaces the station map, the deliberately-unmerged ambiguous sites, every
cleaning decision with counts, and known inconsistencies (notably PM2.5 > PM10) WITHOUT
having silently fixed them.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .schema import AMBIGUOUS_STATIONS, NUMERIC_COLS, UNIT_NOTES


def _fmt_counts(d: dict) -> str:
    nz = {k: v for k, v in d.items() if v}
    if not nz:
        return "_none_"
    return ", ".join(f"`{k}`={v:,}" for k, v in sorted(nz.items(), key=lambda kv: -kv[1]))


def _missingness_heatmap(df: pd.DataFrame, cols: list[str], path: str) -> None:
    pivot = (
        df.groupby("station")[cols].apply(lambda g: g.isna().mean() * 100).round(1)
    )
    plt.figure(figsize=(max(8, len(cols) * 0.7), max(4, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="rocket_r", vmin=0, vmax=100,
                cbar_kws={"label": "% missing (on hourly grid)"})
    plt.title("Missingness % by station x variable")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def _distribution_fig(df: pd.DataFrame, cols: list[str], path: str) -> None:
    cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    n = len(cols)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    plt.figure(figsize=(ncol * 3.2, nrow * 2.4))
    for i, c in enumerate(cols, 1):
        ax = plt.subplot(nrow, ncol, i)
        df[c].dropna().plot(kind="hist", bins=60, ax=ax, color="#3b6ea5")
        ax.set_title(c, fontsize=9)
        ax.set_ylabel("")
    plt.suptitle("Value distributions (post range-nulling)", y=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def _pm_violation_table(df: pd.DataFrame) -> pd.DataFrame:
    both = df[["station", "timestamp", "pm25", "pm10"]].dropna(subset=["pm25", "pm10"]).copy()
    both["year"] = both["timestamp"].dt.year
    both["viol"] = both["pm25"] > both["pm10"]
    g = both.groupby(["station", "year"])["viol"].agg(["sum", "count"])
    g["pct"] = (g["sum"] / g["count"] * 100).round(1)
    g = g[g["sum"] > 0].sort_values("pct", ascending=False)
    return g.rename(columns={"sum": "violations", "count": "rows_with_both"})


def generate_report(cfg: dict, runlog: dict) -> str:
    paths = cfg["paths"]
    fig_dir = paths["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_parquet(paths["clean"])
    cols = [c for c in NUMERIC_COLS if c in df.columns]

    heatmap_rel = os.path.relpath(os.path.join(fig_dir, "missingness_heatmap.png"), paths["reports_dir"])
    dist_rel = os.path.relpath(os.path.join(fig_dir, "distributions.png"), paths["reports_dir"])
    _missingness_heatmap(df, cols, os.path.join(fig_dir, "missingness_heatmap.png"))
    _distribution_fig(df, cols, os.path.join(fig_dir, "distributions.png"))

    s = runlog["stages"]
    L: list[str] = []
    L.append("# Air-Quality Data-Quality Report\n")
    L.append(f"_Generated from `run_log.json`. Build elapsed: {runlog.get('elapsed_sec','?')}s. "
             f"Smoke config: {runlog.get('config')}._\n")
    L.append("> **This report is the sign-off artifact.** No row was deleted to fix a value; "
             "no two physically distinct stations were merged. Items marked ⚠️ need a human decision.\n")

    # --- Files ---
    L.append("## 1. Source files\n")
    L.append("| file | sheet | raw cols | units row dropped | unknown cols dropped | rows | bad timestamps |")
    L.append("|---|---|---|---|---|---|---|")
    for f in runlog["files"]:
        L.append(f"| `{f['path']}` | {f['sheet']} | {len(f['raw_columns'])} | "
                 f"{f['units_row_dropped']} | {f['unknown_dropped'] or '—'} | "
                 f"{f.get('rows',0):,} | {f.get('timestamp_unparseable',0)} |")
    L.append("")

    # --- Stations ---
    L.append("## 2. Station map & coverage\n")
    if s.get("unmapped_stations"):
        L.append(f"⚠️ **Unmapped station labels (kept, flagged):** {s['unmapped_stations']}\n")
    else:
        L.append("All raw station labels were recognized by `schema.STATION_MAP` (spelling-only merges).\n")
    L.append("⚠️ **Ambiguous sites deliberately kept separate — confirm before any merge:**\n")
    for k, v in AMBIGUOUS_STATIONS.items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    reindex = s.get("reindex", {})
    L.append("| station | observed rows | grid rows | % grid observed | start | end | source files |")
    L.append("|---|---|---|---|---|---|---|")
    for st, c in s["station_coverage"].items():
        ri = reindex.get(st, {})
        grid = ri.get("grid_rows", 0)
        obs = ri.get("observed_rows", c["rows"])
        pct = round(100 * obs / grid, 1) if grid else 0.0
        L.append(f"| {st} | {c['rows']:,} | {grid:,} | {pct} | {c['start'][:10]} | {c['end'][:10]} | "
                 f"{','.join(c['source_files'])} |")
    L.append("")
    L.append("_`% grid observed` = non-gap rows / hourly-grid rows over the station's span. Low values "
             "(esp. Rangpur/Narsingdi/Rajshahi/Sangsad/Khulna in 2012-2021) reflect the timestamp loss "
             "above. All stations are retained; downstream modeling can filter on coverage._\n")

    # --- Timestamp recovery & loss (the big 2012-2021 finding) ---
    L.append("## 2b. ⚠️ Timestamp recovery & data loss\n")
    L.append("The 2012-2021 sheet mixes encodings: Excel-serial **and** day-first-string dates, "
             "and `HH:MM` **and** bare-integer-hour times. The parser handles all of them. But many "
             "rows carry measurements with a **blank date and/or time** — they cannot be placed on a "
             "time axis. **Per your sign-off these are DROPPED (not fabricated) and audited in full** "
             "in `audit_dropped_untimestamped.parquet`.\n")
    L.append("| file | rows | timestamp ok | NaT total | NaT w/ data (dropped) | NaT empty |")
    L.append("|---|---|---|---|---|---|")
    for f in runlog["files"]:
        ta = f.get("timestamp_audit", {})
        L.append(f"| {f['label']} | {ta.get('rows',0):,} | {ta.get('timestamp_ok',0):,} | "
                 f"{ta.get('nat_total',0):,} | {ta.get('nat_with_data',0):,} | {ta.get('nat_empty',0):,} |")
    L.append("")
    du = s.get("dropped_untimestamped", {})
    if du.get("by_station"):
        L.append(f"**Total measured rows dropped for lack of a timestamp: {du.get('total',0):,}.** "
                 "By station (these stations therefore have thin 2012-2021 coverage — kept anyway, "
                 "flagged here):\n")
        L.append("| station | rows dropped (no timestamp) |")
        L.append("|---|---|")
        for st, n in sorted(du["by_station"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {st} | {n:,} |")
        L.append("")

    # --- Cleaning decisions ---
    L.append("## 3. Cleaning decisions (with counts)\n")
    d = s["dedup"]
    L.append(f"- **Concatenated rows:** {s['concat_rows']:,}")
    L.append(f"- **Deduplicate** (station, timestamp), newest source wins: "
             f"{d['rows_before']:,} → {d['rows_after']:,} (removed {d['rows_removed']:,}; "
             f"{d['duplicate_rows_involved']:,} rows involved in duplicates)")
    L.append(f"- **Coerced string→NaN:** {_fmt_counts(s['coerced_to_nan'])}")
    L.append(f"- **Range-nulled cells** (out of physical bounds → NaN, audited in "
             f"`audit_range_nulled.parquet`): {_fmt_counts(s['range_nulled'])}")
    L.append(f"- **IQR outliers flagged** (per-station, k={cfg['outlier']['iqr_k']}; values kept, "
             f"audited in `audit_outlier_flags.parquet`): {_fmt_counts(s['outlier_flagged'])}")
    L.append(f"- **Hourly grid:** {s['clean_rows']:,} rows, {s['grid_gap_rows']:,} inserted gap "
             f"rows (`is_gap=True`).\n")

    # --- PM assertion ---
    L.append("## 4. ⚠️ PM2.5 ≤ PM10 consistency check\n")
    both = df[["pm25", "pm10"]].dropna()
    viol = int((both["pm25"] > both["pm10"]).sum())
    pct = (viol / len(both) * 100) if len(both) else 0
    L.append(f"Of {len(both):,} rows with both PM values, **{viol:,} ({pct:.2f}%) have PM2.5 > PM10** — "
             "physically inconsistent. **Not auto-corrected.** Likely causes: sensor noise, or "
             "station-years where the two channels are swapped/miscalibrated. Per-station-year breakdown "
             "of offenders (top rows):\n")
    pmv = _pm_violation_table(df)
    if len(pmv):
        L.append("| station | year | violations | rows_with_both | % |")
        L.append("|---|---|---|---|---|")
        for (st, yr), row in pmv.head(25).iterrows():
            L.append(f"| {st} | {yr} | {int(row['violations']):,} | {int(row['rows_with_both']):,} | {row['pct']} |")
    L.append("")

    # --- Missingness ---
    L.append("## 5. Missingness (on the regular hourly grid)\n")
    L.append(f"![missingness]({heatmap_rel})\n")
    L.append("| station | grid rows | " + " | ".join(cols) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(cols)) + "|")
    for st, m in s["missingness"].items():
        row = " | ".join(f"{m.get(c, 0)}" for c in cols)
        L.append(f"| {st} | {m.get('_grid_rows', 0):,} | {row} |")
    L.append("")

    # --- Distributions ---
    L.append("## 6. Value distributions & units\n")
    L.append(f"![distributions]({dist_rel})\n")
    L.append("Units (documented, **not** auto-converted — note CO is ppm while other gases are ppb):\n")
    L.append("| variable | unit |")
    L.append("|---|---|")
    for c in cols:
        L.append(f"| {c} | {UNIT_NOTES.get(c, '?')} |")
    L.append("")
    desc = df[cols].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2)
    L.append("Summary statistics (post range-nulling):\n")
    L.append("| variable | count | mean | std | min | 1% | 50% | 99% | max |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in desc.index:
        r = desc.loc[c]
        L.append(f"| {c} | {int(r['count']):,} | {r['mean']} | {r['std']} | {r['min']} | "
                 f"{r['1%']} | {r['50%']} | {r['99%']} | {r['max']} |")
    L.append("")

    # --- Reproduce ---
    L.append("## 7. Reproduce\n")
    L.append("```\npython -m pip install -r requirements.txt\npython -m src.build --config config.yaml --report\n```\n")
    L.append("Artifacts: `data/interim/merged_raw.parquet`, `data/processed/air_quality_clean.parquet` "
             "(cleaned, **NOT imputed**), audit sidecars, and this report.\n")

    out_path = paths["report_md"]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return out_path
