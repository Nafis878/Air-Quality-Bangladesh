# Air-Quality Data-Quality Report

_Generated from `run_log.json`. Build elapsed: 139.1s. Smoke config: {'nrows': None, 'only_small': False}._

> **This report is the sign-off artifact.** No row was deleted to fix a value; no two physically distinct stations were merged. Items marked ⚠️ need a human decision.

## 1. Source files

| file | sheet | raw cols | units row dropped | unknown cols dropped | rows | bad timestamps |
|---|---|---|---|---|---|---|
| `2012-2021 Air.xlsx` | 2012-2021 | 20 | True | — | 966,650 | 372657 |
| `2022 Air.xlsx` | 2022 | 18 | True | — | 126,531 | 218 |
| `2023 Air.xlsx` | 2023 air | 19 | True | — | 140,219 | 59 |
| `2024 Air.xlsx` | 2024 | 18 | True | — | 140,586 | 42 |

## 2. Station map & coverage

All raw station labels were recognized by `schema.STATION_MAP` (spelling-only merges).

⚠️ **Ambiguous sites deliberately kept separate — confirm before any merge:**

- **TV_Center** — TV center/Center/Sation (2022-2024) — site unconfirmed vs TV_st_Chittagong
- **TV_st_Chittagong** — TV st-Chittagong (2012-2021) — labelled Chittagong; NOT merged with TV_Center
- **Agrabad_Chittagong** — Agrabad Chittagong (2012-2021) — NOT merged with CDA without sign-off
- **CDA** — Chittagong Development Authority (2022-2024) — NOT merged with Agrabad without sign-off

| station | observed rows | grid rows | % grid observed | start | end | source files |
|---|---|---|---|---|---|---|
| Agrabad_Chittagong | 63,935 | 71,614 | 82.9 | 2012-11-01 | 2021-01-01 | 2012-2021 |
| BARC | 90,299 | 106,656 | 80.3 | 2012-11-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| Barishal | 90,238 | 106,656 | 80.3 | 2012-11-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| CDA | 26,304 | 26,304 | 100.0 | 2022-01-01 | 2025-01-01 | 2022,2023,2024 |
| Cumilla | 30,778 | 43,847 | 3.1 | 2020-01-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| Darussalam | 90,238 | 106,656 | 80.3 | 2012-11-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| DoE | 17,544 | 17,544 | 100.0 | 2023-01-01 | 2025-01-01 | 2023,2024 |
| Gazipur | 90,239 | 106,656 | 80.3 | 2012-11-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| Khulna | 26,304 | 26,304 | 100.0 | 2022-01-01 | 2025-01-01 | 2022,2023,2024 |
| Mymensingh | 25,691 | 43,847 | 3.1 | 2020-01-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| Narayanganj | 88,102 | 106,656 | 79.6 | 2012-11-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| Narsingdi | 26,304 | 26,304 | 100.0 | 2022-01-01 | 2025-01-01 | 2022,2023,2024 |
| Rajshahi | 26,304 | 26,304 | 100.0 | 2022-01-01 | 2025-01-01 | 2022,2023,2024 |
| Rangpur | 26,304 | 26,304 | 100.0 | 2022-01-01 | 2025-01-01 | 2022,2023,2024 |
| Savar | 30,287 | 43,847 | 3.0 | 2020-01-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| Sylhet | 90,239 | 106,656 | 80.3 | 2012-11-01 | 2025-01-01 | 2012-2021,2022,2023,2024 |
| TV_Center | 26,304 | 26,304 | 100.0 | 2022-01-01 | 2025-01-01 | 2022,2023,2024 |
| TV_st_Chittagong | 63,932 | 71,608 | 82.9 | 2012-11-01 | 2021-01-01 | 2012-2021 |

_`% grid observed` = non-gap rows / hourly-grid rows over the station's span. Low values (esp. Rangpur/Narsingdi/Rajshahi/Sangsad/Khulna in 2012-2021) reflect the timestamp loss above. All stations are retained; downstream modeling can filter on coverage._

## 2b. ⚠️ Timestamp recovery & data loss

The 2012-2021 sheet mixes encodings: Excel-serial **and** day-first-string dates, and `HH:MM` **and** bare-integer-hour times. The parser handles all of them. But many rows carry measurements with a **blank date and/or time** — they cannot be placed on a time axis. **Per your sign-off these are DROPPED (not fabricated) and audited in full** in `audit_dropped_untimestamped.parquet`.

| file | rows | timestamp ok | NaT total | NaT w/ data (dropped) | NaT empty |
|---|---|---|---|---|---|
| 2012-2021 | 966,650 | 593,993 | 372,657 | 276,484 | 96,173 |
| 2022 | 126,531 | 126,313 | 218 | 15 | 203 |
| 2023 | 140,219 | 140,160 | 59 | 43 | 16 |
| 2024 | 140,586 | 140,544 | 42 | 31 | 11 |

**Total measured rows dropped for lack of a timestamp: 276,242.** By station (these stations therefore have thin 2012-2021 coverage — kept anyway, flagged here):

| station | rows dropped (no timestamp) |
|---|---|
| Rajshahi | 75,063 |
| Khulna | 53,186 |
| Sangsad | 41,294 |
| Rangpur | 16,238 |
| Narsingdi | 15,368 |
| Narayanganj | 10,720 |
| BARC | 8,619 |
| Gazipur | 8,187 |
| Barishal | 8,025 |
| Cumilla | 7,789 |
| Savar | 6,664 |
| Mymensingh | 6,274 |
| Agrabad_Chittagong | 6,192 |
| TV_st_Chittagong | 5,365 |
| Darussalam | 4,154 |
| Sylhet | 3,098 |
| CDA | 2 |
| DoE | 2 |
| TV_Center | 2 |

## 3. Cleaning decisions (with counts)

- **Concatenated rows:** 1,373,986
- **Deduplicate** (station, timestamp), newest source wins: 1,001,010 → 929,346 (removed 71,664; 143,328 rows involved in duplicates)
- **Coerced string→NaN:** `nox`=2,849, `pm10`=77, `so2`=76, `wind_speed`=76, `pm25`=75, `no`=74, `no2`=74, `co`=74, `o3`=74, `temp`=73, `rh`=73, `bp`=73, `wind_dir`=71, `solar_rad`=71, `rain`=71, `v_wind_speed`=61, `ratio`=29
- **Range-nulled cells** (out of physical bounds → NaN, audited in `audit_range_nulled.parquet`): `bp`=40,259, `ratio`=11,945, `temp`=11,318, `v_wind_speed`=9,602, `rain`=9,045, `no2`=3,834, `rh`=2,854, `solar_rad`=2,485, `no`=590, `nox`=518, `o3`=434, `wind_dir`=355, `pm10`=253, `co`=159, `pm25`=123, `so2`=54, `wind_speed`=49
- **IQR outliers flagged** (per-station, k=3.0; values kept, audited in `audit_outlier_flags.parquet`): `no`=36,331, `nox`=27,872, `so2`=21,777, `o3`=20,411, `wind_speed`=19,279, `rain`=18,761, `bp`=12,018, `co`=11,598, `no2`=10,377, `pm25`=8,376, `pm10`=7,137, `v_wind_speed`=4,126, `rh`=2,464, `solar_rad`=1,342, `temp`=686
- **Hourly grid:** 1,090,067 rows, 278,901 inserted gap rows (`is_gap=True`).

## 4. ⚠️ PM2.5 ≤ PM10 consistency check

Of 469,971 rows with both PM values, **23,962 (5.10%) have PM2.5 > PM10** — physically inconsistent. **Not auto-corrected.** Likely causes: sensor noise, or station-years where the two channels are swapped/miscalibrated. Per-station-year breakdown of offenders (top rows):

| station | year | violations | rows_with_both | % |
|---|---|---|---|---|
| BARC | 2021 | 1 | 1 | 100.0 |
| Gazipur | 2025 | 1 | 1 | 100.0 |
| BARC | 2014 | 1,172 | 1,509 | 77.7 |
| BARC | 2015 | 1,300 | 2,767 | 47.0 |
| TV_st_Chittagong | 2016 | 22 | 49 | 44.9 |
| TV_st_Chittagong | 2012 | 107 | 278 | 38.5 |
| TV_st_Chittagong | 2015 | 754 | 2,448 | 30.8 |
| Savar | 2020 | 277 | 910 | 30.4 |
| Narsingdi | 2024 | 760 | 2,905 | 26.2 |
| Rajshahi | 2023 | 1,296 | 5,098 | 25.4 |
| Gazipur | 2024 | 1,129 | 5,383 | 21.0 |
| BARC | 2023 | 540 | 3,144 | 17.2 |
| TV_st_Chittagong | 2013 | 634 | 3,968 | 16.0 |
| Sylhet | 2024 | 1,021 | 7,179 | 14.2 |
| Khulna | 2024 | 886 | 6,251 | 14.2 |
| DoE | 2024 | 1,012 | 7,159 | 14.1 |
| Sylhet | 2020 | 13 | 96 | 13.5 |
| BARC | 2016 | 106 | 875 | 12.1 |
| Rajshahi | 2022 | 788 | 6,800 | 11.6 |
| Khulna | 2023 | 695 | 6,339 | 11.0 |
| Rangpur | 2024 | 699 | 6,403 | 10.9 |
| Rajshahi | 2024 | 552 | 5,281 | 10.5 |
| Narsingdi | 2023 | 704 | 6,700 | 10.5 |
| Khulna | 2022 | 586 | 5,819 | 10.1 |
| Rangpur | 2023 | 537 | 5,339 | 10.1 |

## 5. Missingness (on the regular hourly grid)

![missingness](figures\missingness_heatmap.png)

| station | grid rows | so2 | no | no2 | nox | co | o3 | pm25 | pm10 | wind_speed | wind_dir | temp | rh | bp | solar_rad | rain | v_wind_speed | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agrabad_Chittagong | 71,614 | 50.78 | 48.2 | 46.79 | 41.29 | 45.55 | 42.33 | 48.98 | 42.75 | 30.08 | 30.14 | 30.06 | 30.14 | 29.99 | 29.98 | 52.2 | 88.5 | 80.64 |
| BARC | 106,656 | 46.85 | 53.58 | 51.4 | 51.7 | 40.55 | 33.86 | 43.66 | 54.29 | 43.78 | 42.17 | 46.26 | 44.2 | 73.76 | 59.45 | 61.33 | 91.04 | 93.01 |
| Barishal | 106,656 | 56.18 | 43.73 | 52.52 | 38.7 | 65.54 | 65.5 | 46.22 | 52.58 | 30.78 | 30.8 | 31.47 | 31.29 | 40.02 | 31.26 | 71.37 | 74.09 | 84.72 |
| CDA | 26,304 | 24.73 | 14.46 | 23.17 | 13.74 | 73.36 | 59.24 | 41.25 | 32.84 | 14.08 | 14.08 | 59.05 | 24.31 | 19.69 | 13.13 | 97.5 | 86.09 | 100.0 |
| Cumilla | 43,847 | 98.7 | 99.09 | 99.92 | 99.04 | 98.24 | 99.24 | 98.31 | 98.78 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Darussalam | 106,656 | 49.05 | 41.36 | 38.28 | 34.82 | 40.7 | 44.8 | 30.91 | 34.13 | 34.05 | 33.03 | 29.99 | 25.25 | 30.1 | 27.79 | 51.47 | 84.41 | 86.22 |
| DoE | 17,544 | 31.3 | 32.28 | 34.29 | 21.94 | 23.35 | 96.55 | 15.33 | 15.76 | 62.61 | 62.61 | 63.63 | 36.91 | 77.49 | 36.88 | 97.52 | 82.59 | 100.0 |
| Gazipur | 106,656 | 58.35 | 55.93 | 60.81 | 59.12 | 63.83 | 59.58 | 41.1 | 38.26 | 54.07 | 56.43 | 62.76 | 63.26 | 71.6 | 56.68 | 54.83 | 83.86 | 75.98 |
| Khulna | 26,304 | 24.89 | 32.66 | 32.23 | 32.02 | 35.38 | 13.88 | 19.84 | 24.21 | 9.76 | 9.76 | 9.3 | 7.47 | 7.38 | 25.48 | 99.43 | 65.79 | 100.0 |
| Mymensingh | 43,847 | 99.23 | 100.0 | 100.0 | 98.53 | 98.51 | 98.83 | 98.64 | 99.95 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Narayanganj | 106,656 | 69.98 | 48.17 | 52.62 | 44.67 | 56.12 | 55.68 | 44.7 | 40.32 | 64.61 | 64.64 | 84.0 | 79.8 | 75.18 | 87.07 | 73.68 | 93.11 | 77.58 |
| Narsingdi | 26,304 | 17.97 | 21.34 | 18.58 | 19.88 | 9.68 | 13.06 | 14.4 | 30.57 | 27.49 | 27.49 | 14.58 | 13.84 | 13.91 | 35.8 | 93.96 | 100.0 | 100.0 |
| Rajshahi | 26,304 | 24.6 | 16.27 | 16.21 | 15.27 | 11.38 | 10.96 | 9.74 | 30.42 | 16.83 | 11.96 | 4.29 | 3.92 | 60.56 | 43.63 | 72.43 | 95.85 | 100.0 |
| Rangpur | 26,304 | 38.58 | 39.53 | 38.37 | 36.99 | 14.72 | 24.41 | 17.02 | 29.63 | 24.9 | 24.9 | 13.51 | 11.22 | 11.1 | 36.83 | 96.63 | 96.41 | 100.0 |
| Savar | 43,847 | 98.1 | 97.55 | 97.55 | 97.55 | 98.78 | 97.57 | 97.87 | 97.74 | 99.57 | 99.57 | 98.92 | 98.92 | 100.0 | 98.92 | 98.92 | 100.0 | 100.0 |
| Sylhet | 106,656 | 63.07 | 38.87 | 44.12 | 37.06 | 51.98 | 59.93 | 44.37 | 42.04 | 50.78 | 51.83 | 51.16 | 44.96 | 55.79 | 44.97 | 73.01 | 92.76 | 75.89 |
| TV_Center | 26,304 | 36.34 | 34.38 | 33.31 | 32.81 | 45.84 | 23.08 | 33.53 | 55.43 | 57.65 | 37.94 | 23.86 | 25.76 | 47.97 | 79.78 | 85.93 | 97.02 | 100.0 |
| TV_st_Chittagong | 71,608 | 66.8 | 75.22 | 75.31 | 75.22 | 61.78 | 66.81 | 75.94 | 62.82 | 79.24 | 79.72 | 81.78 | 84.25 | 87.3 | 71.26 | 88.82 | 92.77 | 88.14 |

## 6. Value distributions & units

![distributions](figures\distributions.png)

Units (documented, **not** auto-converted — note CO is ppm while other gases are ppb):

| variable | unit |
|---|---|
| so2 | ppb |
| no | ppb |
| no2 | ppb |
| nox | ppb |
| co | ppm |
| o3 | ppb |
| pm25 | ug/m3 |
| pm10 | ug/m3 |
| wind_speed | m/s |
| wind_dir | deg |
| temp | degC |
| rh | % |
| bp | hPa/mb |
| solar_rad | W/m2 |
| rain | mm |
| v_wind_speed | m/s |
| ratio | fraction (PM2.5/PM10) |

Summary statistics (post range-nulling):

| variable | count | mean | std | min | 1% | 50% | 99% | max |
|---|---|---|---|---|---|---|---|---|
| so2 | 460,272 | 8.24 | 18.0 | 0.0 | 0.08 | 4.52 | 69.22 | 1481.32 |
| no | 523,833 | 19.89 | 39.86 | 0.0 | 0.08 | 6.2 | 214.17 | 719.51 |
| no2 | 503,925 | 12.68 | 18.95 | 0.0 | 0.07 | 6.71 | 99.05 | 386.52 |
| nox | 549,956 | 29.51 | 46.98 | 0.0 | 0.19 | 13.13 | 250.8 | 728.5 |
| co | 489,532 | 1.87 | 1.9 | 0.0 | 0.09 | 1.3 | 9.07 | 40.13 |
| o3 | 486,703 | 10.97 | 14.97 | 0.0 | 0.1 | 5.74 | 76.45 | 485.56 |
| pm25 | 565,273 | 88.19 | 107.72 | 0.01 | 5.5 | 55.31 | 455.11 | 1895.1 |
| pm10 | 549,259 | 142.51 | 133.18 | 0.0 | 11.94 | 101.74 | 653.47 | 2995.7 |
| wind_speed | 533,217 | 1.73 | 3.63 | 0.0 | 0.0 | 0.72 | 28.38 | 60.68 |
| wind_dir | 538,421 | 156.68 | 105.54 | 0.0 | 0.0 | 148.23 | 359.83 | 360.0 |
| temp | 508,981 | 26.13 | 5.86 | -9.82 | 9.27 | 26.81 | 38.29 | 53.38 |
| rh | 539,883 | 70.14 | 18.7 | 0.0 | 21.28 | 72.38 | 99.5 | 100.0 |
| bp | 449,186 | 1007.89 | 16.64 | 800.0 | 982.5 | 1007.79 | 1086.57 | 1100.0 |
| solar_rad | 490,640 | 208.53 | 257.42 | 0.0 | 0.0 | 67.96 | 920.94 | 1489.59 |
| rain | 285,965 | 1.74 | 17.39 | 0.0 | 0.0 | 0.08 | 16.5 | 499.9 |
| v_wind_speed | 118,050 | 3.59 | 10.37 | 0.0 | 0.01 | 1.01 | 52.29 | 71.83 |
| ratio | 136,058 | 0.53 | 0.22 | 0.0 | 0.0 | 0.55 | 0.95 | 1.0 |

## 7. Reproduce

```
python -m pip install -r requirements.txt
python -m src.build --config config.yaml --report
```

Artifacts: `data/interim/merged_raw.parquet`, `data/processed/air_quality_clean.parquet` (cleaned, **NOT imputed**), audit sidecars, and this report.
