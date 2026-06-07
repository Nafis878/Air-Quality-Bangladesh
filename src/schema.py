"""
Canonical schema, column/station harmonization maps, and physical valid ranges.

Single source of truth for *names* and *bounds*. Everything that decides "what is
this column / station called in the canonical artifact" lives here so that merges
happen strictly by NAME (never by column position) and station aliases are explicit
and auditable. Nothing here mutates data; clean.py applies these definitions.
"""
from __future__ import annotations


def norm(name: object) -> str:
    """Normalize a raw header/station string for lookup: strip + collapse ws + lower."""
    return " ".join(str(name).strip().split()).lower()


# ---------------------------------------------------------------------------
# Canonical column order of the cleaned artifact.
# ---------------------------------------------------------------------------
ID_COLS = ["station", "timestamp"]
POLLUTANTS = ["so2", "no", "no2", "nox", "co", "o3", "pm25", "pm10"]
MET = ["wind_speed", "wind_dir", "temp", "rh", "bp", "solar_rad", "rain", "v_wind_speed"]
DERIVED = ["ratio"]                      # PM2.5/PM10, often empty — kept but flagged
PROVENANCE = ["source_year_file"]

NUMERIC_COLS = POLLUTANTS + MET + DERIVED
CANONICAL_COLS = ID_COLS + POLLUTANTS + MET + DERIVED + PROVENANCE

# The 16 channels the forecasting models predict (pollutants + meteorology).
# `ratio` (DERIVED) is excluded — it is a near-empty derived quantity, not a forecast target.
MODEL_CHANNELS = POLLUTANTS + MET

# Columns whose units differ from the obvious default — documented, never auto-converted.
UNIT_NOTES = {
    "so2": "ppb", "no": "ppb", "no2": "ppb", "nox": "ppb", "o3": "ppb",
    "co": "ppm",                 # NOTE: CO is ppm, the others are ppb
    "pm25": "ug/m3", "pm10": "ug/m3",
    "wind_speed": "m/s", "v_wind_speed": "m/s", "wind_dir": "deg",
    "temp": "degC", "rh": "%", "bp": "hPa/mb", "solar_rad": "W/m2", "rain": "mm",
    "ratio": "fraction (PM2.5/PM10)",
}

# ---------------------------------------------------------------------------
# Column rename map: normalized raw header -> canonical name.
# Merge by NAME. Datetime source columns are handled separately (see DATE_*).
# ---------------------------------------------------------------------------
RENAME_MAP = {
    "location": "station",
    "so2": "so2",
    "no": "no",
    "no2": "no2",
    "nox": "nox",
    "co": "co",
    "o3": "o3",
    "pm2.5": "pm25",
    "pm25": "pm25",
    "pm10": "pm10",
    "wind speed": "wind_speed",
    "ws": "wind_speed",
    "wind dir": "wind_dir",
    "wd": "wind_dir",
    "temperature": "temp",
    "temp": "temp",
    "rh": "rh",
    "solar rad": "solar_rad",
    "sr": "solar_rad",
    "bp": "bp",
    "rain": "rain",
    "v wind speed": "v_wind_speed",
    "vws": "v_wind_speed",
    "ratio": "ratio",
    "rs": "ratio",
}

# Datetime source columns (normalized). Two encodings across files.
DATE_SINGLE = {"date & time", "date and time", "datetime"}   # one cell (2022-2024)
DATE_PART_DATE = {"date"}                                    # serial date (2012-2021)
DATE_PART_TIME = {"time"}                                    # "HH:MM" string (2012-2021)

# ---------------------------------------------------------------------------
# Station canonicalization: normalized raw Location -> canonical station id.
# Spelling-only merges. Ambiguous Chittagong/TV sites are KEPT SEPARATE pending
# human sign-off (see plan / data_quality_report). Unmapped values are passed
# through (title-cased) and reported as 'unmapped' rather than silently dropped.
# ---------------------------------------------------------------------------
STATION_MAP = {
    # spelling variants -> canonical
    "mymensingh": "Mymensingh",
    "mymensing": "Mymensingh",
    "narayanganj": "Narayanganj",
    "narayangonj": "Narayanganj",
    "narayonganj": "Narayanganj",
    "tv center": "TV_Center",
    "tv sation": "TV_Center",
    "tv st-chittagong": "TV_st_Chittagong",     # KEEP SEPARATE — flagged
    "agrabad chittagong": "Agrabad_Chittagong",  # KEEP SEPARATE from CDA — flagged
    "cda": "CDA",
    "sangsad": "Sangsad",
    "doe": "DoE",
    # stable core (identity, normalized)
    "barishal": "Barishal",
    "rajshahi": "Rajshahi",
    "barc": "BARC",
    "darussalam": "Darussalam",
    "gazipur": "Gazipur",
    "khulna": "Khulna",
    "sylhet": "Sylhet",
    "rangpur": "Rangpur",
    "savar": "Savar",
    "cumilla": "Cumilla",
    "narsingdi": "Narsingdi",
}

# Stations deliberately kept separate despite tidy-name temptation — surfaced in report.
AMBIGUOUS_STATIONS = {
    "TV_Center": "TV center/Center/Sation (2022-2024) — site unconfirmed vs TV_st_Chittagong",
    "TV_st_Chittagong": "TV st-Chittagong (2012-2021) — labelled Chittagong; NOT merged with TV_Center",
    "Agrabad_Chittagong": "Agrabad Chittagong (2012-2021) — NOT merged with CDA without sign-off",
    "CDA": "Chittagong Development Authority (2022-2024) — NOT merged with Agrabad without sign-off",
}

# ---------------------------------------------------------------------------
# Physical valid ranges (hard bounds). Values outside -> set to NaN (cell nulled,
# row kept). Generous upper caps: implausible physically, not merely extreme.
# Soft per-station IQR outlier *flagging* is separate (audit sidecar), never nulls.
# ---------------------------------------------------------------------------
VALID_RANGES = {
    "so2": (0.0, 2000.0),     # ppb
    "no": (0.0, 2000.0),      # ppb
    "no2": (0.0, 2000.0),     # ppb
    "nox": (0.0, 4000.0),     # ppb
    "co": (0.0, 50.0),        # ppm
    "o3": (0.0, 600.0),       # ppb
    "pm25": (0.0, 2000.0),    # ug/m3
    "pm10": (0.0, 3000.0),    # ug/m3
    "wind_speed": (0.0, 75.0),    # m/s
    "v_wind_speed": (0.0, 75.0),  # m/s
    "wind_dir": (0.0, 360.0),     # deg
    "temp": (-10.0, 55.0),        # degC
    "rh": (0.0, 100.0),           # %
    "bp": (800.0, 1100.0),        # hPa/mb
    "solar_rad": (0.0, 1500.0),   # W/m2
    "rain": (0.0, 500.0),         # mm (hourly)
    "ratio": (0.0, 1.0),          # PM2.5/PM10 fraction
}


def canonical_station(raw: object) -> tuple[str, bool]:
    """Map a raw Location to (canonical_id, is_mapped). Unmapped -> title-cased passthrough."""
    key = norm(raw)
    if key in STATION_MAP:
        return STATION_MAP[key], True
    # Passthrough: keep data, flag as unmapped for the report.
    return str(raw).strip(), False
