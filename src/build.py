"""
Pipeline orchestrator: raw Excel -> interim/merged_raw.parquet -> processed/clean.parquet.

Deterministic and idempotent. Every stage logs before/after row counts into a run-log
that report.py turns into data_quality_report.md. Run:

    python -m src.build --config config.yaml
    python -m src.build --config config.yaml --nrows 5000        # fast smoke test
    python -m src.build --config config.yaml --only-small        # skip the 74 MB file

No imputation/scaling here — the canonical artifact is cleaned but NOT imputed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import yaml

from . import clean
from .io import read_raw_file
from .schema import CANONICAL_COLS, NUMERIC_COLS


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _ensure_dirs(cfg: dict) -> None:
    for key in ("interim_dir", "processed_dir", "reports_dir", "figures_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical column order; keep extras (timestamp, is_gap) in a stable place."""
    front = [c for c in ["station", "timestamp"] if c in df.columns]
    body = [c for c in CANONICAL_COLS if c in df.columns and c not in front]
    tail = [c for c in df.columns if c not in front + body]
    return df[front + body + tail]


def _log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def run_build(cfg: dict, nrows: int | None = None, only_small: bool = False) -> dict:
    t0 = time.time()
    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    _ensure_dirs(cfg)
    origin = cfg.get("excel_origin", "1899-12-30")

    runlog: dict = {"files": [], "stages": {}, "config": {"nrows": nrows, "only_small": only_small}}

    # --- Stage 1-2: read, drop units row, rename, build timestamp (per file) ---
    frames = []
    for spec in cfg["raw_files"]:
        if only_small and spec["label"] == "2012-2021":
            _log(f"skipping {spec['label']} (--only-small)")
            continue
        _log(f"reading {spec['path']} (sheet={spec['sheet']}) ...")
        df, meta = read_raw_file(spec["path"], spec["sheet"], spec["label"], nrows=nrows)
        df, n_bad_ts = clean.build_timestamp(df, origin)
        meta["timestamp_unparseable"] = n_bad_ts
        meta["rows"] = int(len(df))
        meta["timestamp_audit"] = _timestamp_audit(df)
        runlog["files"].append(meta)
        ta = meta["timestamp_audit"]
        _log(f"  {spec['label']}: {len(df):,} rows | timestamp NaT={n_bad_ts:,} "
             f"(with-data={ta['nat_with_data']:,}, empty={ta['nat_empty']:,})")
        frames.append(df)

    # --- Stage 3: concat by canonical name (missing cols -> NaN) ---
    merged = pd.concat(frames, ignore_index=True, sort=False)
    runlog["stages"]["concat_rows"] = int(len(merged))
    _log(f"concatenated: {len(merged):,} rows")

    # --- Stage 2 (station) + 4 (coerce) on the merged frame ---
    merged, unmapped = clean.canonicalize_stations(merged)
    runlog["stages"]["unmapped_stations"] = unmapped
    if unmapped:
        _log(f"  WARNING unmapped stations (kept, flagged): {unmapped}")
    merged, coerced = clean.coerce_numeric(merged)
    runlog["stages"]["coerced_to_nan"] = coerced

    merged = _order_columns(merged)
    merged.to_parquet(cfg["paths"]["merged_raw"], index=False)
    _log(f"wrote {cfg['paths']['merged_raw']} ({len(merged):,} rows)")

    # --- Policy (user sign-off): DROP measured-but-untimestamped rows, but log + audit
    #     them in full so nothing is silently lost. No timestamps are fabricated. ---
    val_cols = [c for c in NUMERIC_COLS if c in merged.columns]
    nat_mask = merged["timestamp"].isna() & merged[val_cols].notna().any(axis=1)
    dropped = merged.loc[nat_mask].drop(columns=["timestamp"])
    dropped.to_parquet(os.path.join(cfg["paths"]["processed_dir"], "audit_dropped_untimestamped.parquet"), index=False)
    runlog["stages"]["dropped_untimestamped"] = {
        "total": int(len(dropped)),
        "by_source": {str(k): int(v) for k, v in dropped["source_year_file"].value_counts().items()},
        "by_station": {str(k): int(v) for k, v in dropped["station"].value_counts().items()},
    }
    _log(f"dropped (measured, no timestamp; audited): {len(dropped):,} rows")

    # --- Stage 5: deduplicate ---
    dedup, dstats = clean.deduplicate(merged)
    runlog["stages"]["dedup"] = dstats
    _log(f"dedup: {dstats['rows_before']:,} -> {dstats['rows_after']:,} "
         f"(removed {dstats['rows_removed']:,}; {dstats['duplicate_rows_involved']:,} involved)")

    # --- Per-station coverage (from observed, pre-reindex data) for the report ---
    runlog["stages"]["station_coverage"] = _station_coverage(dedup)

    # --- Stage 6a: range nulling (audited) ---
    ranged, range_audit, range_counts = clean.apply_valid_ranges(dedup)
    runlog["stages"]["range_nulled"] = range_counts
    _log(f"range-nulled cells: {sum(range_counts.values()):,}")
    range_audit.to_parquet(os.path.join(cfg["paths"]["processed_dir"], "audit_range_nulled.parquet"), index=False)

    # --- Stage 6b: per-station IQR outlier flagging (no values changed) ---
    outlier_audit, outlier_counts = clean.flag_outliers(ranged, k=cfg["outlier"]["iqr_k"])
    runlog["stages"]["outlier_flagged"] = outlier_counts
    _log(f"outliers flagged (kept): {sum(outlier_counts.values()):,}")
    outlier_audit.to_parquet(os.path.join(cfg["paths"]["processed_dir"], "audit_outlier_flags.parquet"), index=False)

    # --- Stage 7: hourly reindex per station (gaps -> explicit NaN; is_gap) ---
    gridded, reindex_stats = clean.reindex_hourly(ranged, cfg["reindex"]["freq"])
    runlog["stages"]["reindex"] = reindex_stats
    gridded = _order_columns(gridded)
    total_grid = int(len(gridded))
    total_gap = int(gridded["is_gap"].sum())
    _log(f"hourly grid: {total_grid:,} rows ({total_gap:,} gap rows inserted)")

    # --- Stage 8: missingness matrix (station x variable, % missing on the grid) ---
    runlog["stages"]["missingness"] = _missingness_matrix(gridded)
    runlog["stages"]["clean_rows"] = total_grid
    runlog["stages"]["grid_gap_rows"] = total_gap

    gridded.to_parquet(cfg["paths"]["clean"], index=False)
    _log(f"wrote {cfg['paths']['clean']} ({total_grid:,} rows)")

    runlog["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(cfg["paths"]["interim_dir"], "run_log.json"), "w", encoding="utf-8") as fh:
        json.dump(runlog, fh, indent=2, default=str)

    _print_final_summary(gridded)
    _log(f"done in {runlog['elapsed_sec']}s")
    return runlog


def _timestamp_audit(df: pd.DataFrame) -> dict:
    """Quantify timestamp loss: rows that are NaT but still carry measurements (real loss)."""
    val_cols = [c for c in NUMERIC_COLS if c in df.columns]
    nat = df["timestamp"].isna()
    has_data = df[val_cols].notna().any(axis=1) if val_cols else pd.Series(False, index=df.index)
    nat_with_data = nat & has_data
    audit = {
        "rows": int(len(df)),
        "timestamp_ok": int((~nat).sum()),
        "nat_total": int(nat.sum()),
        "nat_with_data": int(nat_with_data.sum()),   # measured rows we cannot place in time
        "nat_empty": int((nat & ~has_data).sum()),
    }
    if nat_with_data.any():
        by_st = df.loc[nat_with_data, "station"].value_counts()
        audit["nat_with_data_by_station"] = {str(k): int(v) for k, v in by_st.items()}
    return audit


def _station_coverage(df: pd.DataFrame) -> dict:
    cov = {}
    for st, g in df.groupby("station", sort=True):
        ts = g["timestamp"]
        cov[st] = {
            "rows": int(len(g)),
            "start": str(ts.min()),
            "end": str(ts.max()),
            "by_year": {str(int(y)): int(n) for y, n in ts.dt.year.value_counts().sort_index().items()},
            "source_files": sorted(g["source_year_file"].dropna().unique().tolist()),
        }
    return cov


def _missingness_matrix(df: pd.DataFrame) -> dict:
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    out = {}
    for st, g in df.groupby("station", sort=True):
        n = len(g)
        out[st] = {c: round(float(g[c].isna().mean()) * 100, 2) for c in cols}
        out[st]["_grid_rows"] = int(n)
    return out


def _print_final_summary(df: pd.DataFrame) -> None:
    print("\n================= FINAL CLEAN ARTIFACT =================")
    print(f"rows: {len(df):,} | stations: {df['station'].nunique()} | "
          f"span: {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print("\nschema / dtypes:")
    print(df.dtypes.to_string())
    print("\n10-row sample:")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df.head(10).to_string())
    print("=======================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the cleaned air-quality panel.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--nrows", type=int, default=None, help="cap rows per file (smoke test)")
    ap.add_argument("--only-small", action="store_true", help="skip the 74 MB 2012-2021 file")
    ap.add_argument("--report", action="store_true", help="also generate the data-quality report")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    runlog = run_build(cfg, nrows=args.nrows, only_small=args.only_small)

    if args.report:
        from .report import generate_report
        path = generate_report(cfg, runlog)
        _log(f"report written: {path}")


if __name__ == "__main__":
    main()
