"""Dedup, range-nulling, and the PM2.5<=PM10 policy (flag, never auto-fix)."""
import numpy as np
import pandas as pd

from src.clean import deduplicate, apply_valid_ranges


def _row(station, ts, src, **vals):
    base = dict(station=station, timestamp=pd.Timestamp(ts), source_year_file=src)
    base.update(vals)
    return base


def test_dedup_prefers_newer_source():
    df = pd.DataFrame([
        _row("BARC", "2022-01-01 01:00", "2022", pm25=10.0),
        _row("BARC", "2022-01-01 01:00", "2023", pm25=99.0),  # newer, should win
        _row("BARC", "2022-01-01 02:00", "2022", pm25=20.0),
    ])
    out, stats = deduplicate(df)
    assert stats["rows_before"] == 3 and stats["rows_after"] == 2
    assert stats["duplicate_rows_involved"] == 2
    kept = out[out["timestamp"] == pd.Timestamp("2022-01-01 01:00")]["pm25"].iloc[0]
    assert kept == 99.0  # newest source kept


def test_range_nulls_cells_keeps_rows():
    df = pd.DataFrame([
        _row("X", "2022-01-01 01:00", "2022", rh=150.0, pm25=-5.0, wind_dir=400.0, temp=25.0),
    ])
    out, audit, counts = apply_valid_ranges(df)
    assert len(out) == 1                       # row NOT deleted
    assert np.isnan(out["rh"].iloc[0])         # 150 > 100 -> NaN
    assert np.isnan(out["pm25"].iloc[0])       # negative -> NaN
    assert np.isnan(out["wind_dir"].iloc[0])   # 400 > 360 -> NaN
    assert out["temp"].iloc[0] == 25.0         # valid kept
    assert counts["rh"] == 1 and counts["pm25"] == 1
    assert len(audit) == 3                      # three cells audited


def test_pm25_gt_pm10_is_not_autocorrected():
    # Physically inconsistent but both within valid range -> must survive untouched.
    df = pd.DataFrame([
        _row("X", "2022-01-01 01:00", "2022", pm25=120.0, pm10=80.0),
    ])
    out, audit, counts = apply_valid_ranges(df)
    assert out["pm25"].iloc[0] == 120.0 and out["pm10"].iloc[0] == 80.0
    assert counts.get("pm25", 0) == 0 and counts.get("pm10", 0) == 0
