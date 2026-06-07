# Bangladesh Air-Quality Data Pipeline

A reproducible merge → clean → preprocess pipeline that turns four heterogeneous Excel
workbooks (an hourly, multi-station Bangladesh air-monitoring panel, 2012–2024) into a
single canonical, **cleaned-but-NOT-imputed** hourly Parquet panel, plus leakage-aware
feature/imputation/split helpers, a data-quality report, and a pytest suite.

Correctness, leakage-safety, and reproducibility are prioritized over speed. **No row is
deleted to "fix" a value** (offending cells are nulled and audited); **no two physically
distinct stations are merged** to tidy names (ambiguity is flagged for sign-off); **no
imputation/scaling is fit on validation/test data.**

## Quick start

```bash
python -m pip install -r requirements.txt
python -m src.build --config config.yaml --report        # full build + data-quality report
python -m pytest tests/ -q                                # 20 tests on the known gotchas
```

Useful flags: `--nrows 5000` (fast smoke test), `--only-small` (skip the 74 MB 2012-2021 file).

## Outputs

| Path | What |
|---|---|
| `data/interim/merged_raw.parquet` | All files merged by canonical name + unified timestamp (pre-clean). |
| `data/processed/air_quality_clean.parquet` | Canonical artifact: regular hourly grid per station, cleaned, **not imputed**, with `is_gap` mask. |
| `data/processed/air_quality_features.parquet` | Optional feature-built variant (see `src/features.py`). |
| `data/processed/audit_range_nulled.parquet` | Every cell nulled for being out of physical range. |
| `data/processed/audit_outlier_flags.parquet` | Per-station IQR outliers (flagged, **kept**, not changed). |
| `data/processed/audit_dropped_untimestamped.parquet` | Measured rows dropped because the source had no usable date/time. |
| `data/reports/data_quality_report.md` | Coverage, missingness heatmap, distributions, every decision with counts. **Sign-off artifact.** |
| `data/interim/run_log.json` | Machine-readable log of all stage counts. |

## Canonical schema

`station, timestamp` + pollutants `so2, no, no2, nox, co, o3, pm25, pm10` + meteorology
`wind_speed, wind_dir, temp, rh, bp, solar_rad, rain, v_wind_speed` + `ratio` (PM2.5/PM10,
often empty) + `source_year_file` + `is_gap`. Timestamps are tz-naive **Asia/Dhaka** local
(the network does not observe DST).

### Units (documented, never auto-converted)
Gases `so2/no/no2/nox/o3` = **ppb**; **`co` = ppm** (different!); `pm25/pm10` = µg/m³;
`wind_speed/v_wind_speed` = m/s; `wind_dir` = deg; `temp` = °C; `rh` = %; `bp` = hPa/mb;
`solar_rad` = W/m²; `rain` = mm.

## Known data inconsistencies handled (see `src/schema.py`, `src/clean.py`)

1. **Units row.** Row 2 of every sheet is a units row (`ppb`, `ug/m3`, …) — detected by
   content and dropped.
2. **Column names differ across files** (`WS`/`Wind Speed`, `Temp`/`Temperature`,
   `RS`/`Ratio`, …). Merged strictly **by name** via `RENAME_MAP`.
3. **Column order differs** — notably PM2.5/PM10 are swapped in some files (2022/2023 vs
   2012-2021/2024). Because we merge by name this is automatic; a `pm25 <= pm10` check is
   reported (violations **flagged, not auto-fixed**).
4. **Two datetime encodings** + heterogeneity within 2012-2021: Excel serial **and**
   day-first-string dates, `HH:MM` **and** bare-integer-hour times. All handled. Rows with a
   genuinely blank date/time are dropped and fully audited — **never fabricated**.
5. **Station name variants** canonicalized (spelling only). Ambiguous Chittagong/TV sites
   (`TV st-Chittagong` vs `TV center/Center/Sation`; `Agrabad Chittagong` vs `CDA`) are
   **kept separate pending sign-off** — see report.

### Station map
Spelling merges: `Mymensing`→`Mymensingh`; `Narayonganj/Narayangonj`→`Narayanganj`;
`TV center/Center/Sation`→`TV_Center`. Kept separate (flagged): `TV_st_Chittagong`,
`Agrabad_Chittagong`, `CDA`, `Sangsad`, `DoE`. The node set is **time-varying** (Agrabad &
Sangsad exit after 2021; CDA enters 2022; DoE enters 2023).

> **Major finding (2012-2021):** ~276k rows carry measurements but have no usable timestamp,
> concentrated in Rangpur/Narsingdi/Rajshahi/Sangsad/Khulna (66–92% of their decade). Per
> sign-off these are dropped+audited; all stations are retained with coverage flagged in the
> report so modeling code can decide what to use.

## Cleaning order (each step logged with before/after counts)
read → drop units row → rename → unify timestamp → canonicalize station → concat →
coerce numeric → **drop+audit untimestamped** → dedup (newest source wins) → range-null
(cells→NaN, audited) → IQR outlier flag (kept) → hourly reindex per station (`is_gap`) →
missingness matrix.

## Leakage-safe downstream helpers (not applied to the canonical artifact)
- `src/features.py` — time/cyclical/wind + per-station lag & rolling features, strictly past-only.
- `src/imputation.py` — fold-aware `GapImputer`: fit fallback medians on **train only**, time-
  interpolate up to a max gap, add `*_was_missing` indicators.
- `src/splits.py` — `temporal_split` (never shuffle) + `StandardScalerFrame` fit on **train only**.

## Data availability
The raw Excel workbooks and the generated Parquet artifacts are **not committed** to this
repository (size, and to keep the source data under the author's control). Place the four
raw files in the project root with the names/sheets listed in `config.yaml`, then run the
build to regenerate every artifact. The committed `data/reports/data_quality_report.md`
documents the contents and quality of the resulting dataset.

## Reproducibility
Everything derives from `config.yaml` + a fixed seed (`42`). Pinned `requirements.txt`.
The pipeline is deterministic and idempotent (re-running overwrites artifacts identically).
