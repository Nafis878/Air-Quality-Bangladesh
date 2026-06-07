"""
Cleaning transforms. Each function is pure-ish: it returns the transformed frame and,
where relevant, an audit object recording exactly what changed. Orchestration and the
before/after logging live in build.py.

Guardrails enforced here:
- Rows are NEVER deleted to fix bad values; offending *cells* are set to NaN and audited.
- Two physically distinct stations are NEVER merged (canonicalization is spelling-only,
  driven by schema.STATION_MAP; unmapped names pass through and are reported).
- No imputation, scaling, or any statistic fit on data happens here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import NUMERIC_COLS, VALID_RANGES, canonical_station

# Source-file recency order for conflict resolution (higher = newer = preferred).
_SOURCE_ORDER = {"2012-2021": 0, "2022": 1, "2023": 2, "2024": 3}


# ---------------------------------------------------------------------------
# 1. Datetime unification (per file, before concat — encodings differ).
# ---------------------------------------------------------------------------
def _parse_dates(raw: pd.Series, origin: str) -> pd.Series:
    """Excel serial OR day-first string dates (the 2012-2021 file mixes both)."""
    num = pd.to_numeric(raw, errors="coerce")
    base = pd.to_datetime(num, unit="D", origin=origin, errors="coerce")
    miss = base.isna() & raw.notna()
    if miss.any():  # DD/MM/YYYY strings like "02/01/2016"
        base.loc[miss] = pd.to_datetime(raw[miss], dayfirst=True, errors="coerce")
    return base


def _parse_times(raw: pd.Series) -> pd.Series:
    """'HH:MM' strings OR bare integer hours (0-24); -> Timedelta. NaT where missing."""
    tstr = raw.astype(str).str.strip()
    td = pd.to_timedelta(tstr + ":00", errors="coerce")   # "01:00" -> "01:00:00"
    hi = pd.to_numeric(raw, errors="coerce")              # bare int hours, e.g. 8
    miss = td.isna() & hi.notna()
    if miss.any():
        hours = hi[miss].where(hi[miss].between(0, 24))   # 24 -> next midnight; >24 -> NaT
        td.loc[miss] = pd.to_timedelta(hours, unit="h")
    return td


def build_timestamp(df: pd.DataFrame, origin: str) -> tuple[pd.DataFrame, int]:
    """Build a single tz-naive `timestamp`; return (df, n_unparseable).

    Handles BOTH datetime encodings found in the data AND, within the 2012-2021 file,
    the heterogeneous mix of Excel-serial vs day-first-string dates and HH:MM-vs-bare-int
    times. Rows whose source date/time is genuinely blank remain NaT (reported, never faked).
    """
    df = df.copy()
    if "_datetime" in df.columns:
        s = df["_datetime"]
        num = pd.to_numeric(s, errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(s) and num.notna().mean() > 0.5:
            ts = pd.to_datetime(num, unit="D", origin=origin, errors="coerce")
        else:
            ts = pd.to_datetime(s, errors="coerce")
        bad = ts.isna() & s.notna()           # leftover string dates -> day-first
        if bad.any():
            ts.loc[bad] = pd.to_datetime(s[bad], dayfirst=True, errors="coerce")
        df = df.drop(columns=["_datetime"])
    elif "_date" in df.columns and "_time" in df.columns:
        ts = _parse_dates(df["_date"], origin) + _parse_times(df["_time"])
        df = df.drop(columns=["_date", "_time"])
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    df["timestamp"] = ts
    n_bad = int(ts.isna().sum())
    return df, n_bad


# ---------------------------------------------------------------------------
# 2. Station canonicalization (spelling-only; ambiguous sites kept separate).
# ---------------------------------------------------------------------------
def canonicalize_stations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply schema.canonical_station; return (df, {unmapped_raw: count})."""
    df = df.copy()
    raw = df["station"]
    mapped = raw.map(lambda v: canonical_station(v))
    df["station"] = mapped.map(lambda t: t[0])
    is_mapped = mapped.map(lambda t: t[1])
    unmapped = raw[~is_mapped].value_counts().to_dict()
    return df, {str(k): int(v) for k, v in unmapped.items()}


# ---------------------------------------------------------------------------
# 3. Numeric type coercion.
# ---------------------------------------------------------------------------
def coerce_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Coerce pollutant/met columns to numeric; return (df, {col: n_strings_to_NaN})."""
    df = df.copy()
    coerced_counts = {}
    for c in NUMERIC_COLS:
        if c not in df.columns:
            continue
        before_na = df[c].isna()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        newly_na = int((df[c].isna() & ~before_na).sum())
        coerced_counts[c] = newly_na
    return df, coerced_counts


# ---------------------------------------------------------------------------
# 4. Deduplicate on (station, timestamp); prefer newer source file.
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate (station, timestamp); keep newest source. Return (df, stats)."""
    df = df.copy()
    df = df.dropna(subset=["timestamp"])
    n_before = len(df)
    order = df["source_year_file"].map(_SOURCE_ORDER).fillna(-1)
    df = df.assign(_ord=order).sort_values(["station", "timestamp", "_ord"])
    dup_mask = df.duplicated(subset=["station", "timestamp"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    # keep='last' -> newest source wins (we sorted ascending by recency)
    df = df.drop_duplicates(subset=["station", "timestamp"], keep="last").drop(columns="_ord")
    stats = {
        "rows_before": n_before,
        "rows_after": int(len(df)),
        "duplicate_rows_involved": n_dup_rows,
        "rows_removed": n_before - int(len(df)),
    }
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# 5. Range / physical-validity nulling (cells -> NaN, rows kept; audited).
# ---------------------------------------------------------------------------
def apply_valid_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Null cells outside hard physical bounds. Return (df, audit_long, {col: n_nulled})."""
    df = df.copy()
    audit_rows = []
    counts = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        vals = df[col]
        bad = vals.notna() & ((vals < lo) | (vals > hi))
        n_bad = int(bad.sum())
        counts[col] = n_bad
        if n_bad:
            sub = df.loc[bad, ["station", "timestamp"]].copy()
            sub["column"] = col
            sub["original_value"] = vals[bad].values
            sub["reason"] = f"out_of_range[{lo},{hi}]"
            audit_rows.append(sub)
            df.loc[bad, col] = np.nan
    audit = (
        pd.concat(audit_rows, ignore_index=True)
        if audit_rows
        else pd.DataFrame(columns=["station", "timestamp", "column", "original_value", "reason"])
    )
    return df, audit, counts


# ---------------------------------------------------------------------------
# 6. Per-station IQR outlier FLAGGING (no values changed; sparse audit).
# ---------------------------------------------------------------------------
def flag_outliers(df: pd.DataFrame, k: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Flag values beyond per-station [Q1-k*IQR, Q3+k*IQR]. Return (audit_long, {col: n})."""
    audit_rows = []
    counts = {c: 0 for c in NUMERIC_COLS if c in df.columns}
    for st, g in df.groupby("station", sort=True):
        for col in counts:
            v = g[col]
            if v.notna().sum() < 20:  # too few points for a stable fence
                continue
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
            bad = v.notna() & ((v < lo) | (v > hi))
            n = int(bad.sum())
            if n:
                counts[col] += n
                sub = g.loc[bad, ["station", "timestamp", col]].copy()
                sub = sub.rename(columns={col: "value"})
                sub["column"] = col
                sub["fence_low"], sub["fence_high"] = lo, hi
                audit_rows.append(sub)
    audit = (
        pd.concat(audit_rows, ignore_index=True)
        if audit_rows
        else pd.DataFrame(columns=["station", "timestamp", "value", "column", "fence_low", "fence_high"])
    )
    return audit, counts


# ---------------------------------------------------------------------------
# 7. Reindex each station onto a regular hourly grid (gaps become explicit NaN).
# ---------------------------------------------------------------------------
def reindex_hourly(df: pd.DataFrame, freq: str = "1h") -> tuple[pd.DataFrame, dict]:
    """Reindex per station over its observed span; add `is_gap`. Return (df, stats)."""
    value_cols = [c for c in df.columns if c not in ("station", "timestamp")]
    parts = []
    stats = {}
    for st, g in df.groupby("station", sort=True):
        g = g.dropna(subset=["timestamp"]).sort_values("timestamp")
        if g.empty:
            continue
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=freq)
        g = g.set_index("timestamp").reindex(full)
        g["is_gap"] = g["source_year_file"].isna()   # inserted rows have no provenance
        g["station"] = st
        g.index.name = "timestamp"
        parts.append(g.reset_index()[["station", "timestamp"] + value_cols + ["is_gap"]])
        stats[st] = {
            "observed_rows": int(len(full) - g["is_gap"].sum()),
            "grid_rows": int(len(full)),
            "gap_rows": int(g["is_gap"].sum()),
        }
    out = pd.concat(parts, ignore_index=True) if parts else df.assign(is_gap=False)
    return out, stats
