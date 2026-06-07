"""
Raw Excel ingestion.

`read_raw_file` opens one sheet, drops the units row (row 2 of every sheet), renames
columns to canonical names by NAME (never position), preserves the file's datetime
source columns under standardized helper keys for clean.py to unify, and stamps a
provenance column. It returns a tidy DataFrame plus a metadata dict recording every
decision (raw headers, dropped/unknown columns, units-row detection, row counts) so
the data-quality report can show exactly what happened to each file.

Nothing here parses timestamps or coerces value dtypes — that is clean.py's job.
"""
from __future__ import annotations

import pandas as pd

from .schema import (
    DATE_PART_DATE,
    DATE_PART_TIME,
    DATE_SINGLE,
    NUMERIC_COLS,
    RENAME_MAP,
    norm,
)

# Standardized helper column names for the two datetime encodings.
COL_DT = "_datetime"   # single Date & Time cell (2022-2024)
COL_DATE = "_date"     # Excel serial date part (2012-2021)
COL_TIME = "_time"     # "HH:MM" string part (2012-2021)


def _build_colmap(raw_columns: list[str]) -> tuple[dict, list[str]]:
    """Map each raw header to a target name; collect headers we don't recognize."""
    colmap: dict[str, str] = {}
    unknown: list[str] = []
    for raw in raw_columns:
        key = norm(raw)
        if key in DATE_SINGLE:
            colmap[raw] = COL_DT
        elif key in DATE_PART_DATE:
            colmap[raw] = COL_DATE
        elif key in DATE_PART_TIME:
            colmap[raw] = COL_TIME
        elif key in RENAME_MAP:
            colmap[raw] = RENAME_MAP[key]
        else:
            unknown.append(str(raw))
    return colmap, unknown


def _looks_like_units_row(row: pd.Series, numeric_cols: list[str]) -> bool:
    """True if the row is a units row: most numeric columns hold non-numeric text."""
    present = [c for c in numeric_cols if c in row.index]
    if not present:
        return False
    coerced = pd.to_numeric(pd.Series([row[c] for c in present]), errors="coerce")
    n_nonnumeric = int(coerced.isna().sum())
    return n_nonnumeric >= max(1, (len(present) + 1) // 2)  # majority non-numeric


def read_raw_file(path: str, sheet: str, label: str, nrows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Read one raw sheet → tidy canonical-named DataFrame + decision metadata.

    Parameters
    ----------
    path, sheet : the workbook path and sheet name (from config).
    label       : provenance tag stored in `source_year_file`.
    nrows       : optional cap for smoke-testing (reads header + first nrows data rows).
    """
    read_kwargs = dict(sheet_name=sheet, header=0, engine="openpyxl", dtype=object)
    if nrows is not None:
        read_kwargs["nrows"] = nrows + 1  # +1 so the units row is still included
    df = pd.read_excel(path, **read_kwargs)

    raw_columns = [str(c) for c in df.columns]
    colmap, unknown = _build_colmap(raw_columns)

    # Rename recognized columns; drop unknown ones (logged, not silent).
    df = df.rename(columns=colmap)
    if unknown:
        df = df.drop(columns=[c for c in unknown if c in df.columns])

    # Drop the units row (expected at position 0). Verify by content, don't assume.
    units_dropped = False
    if len(df) > 0 and _looks_like_units_row(df.iloc[0], NUMERIC_COLS):
        df = df.iloc[1:].reset_index(drop=True)
        units_dropped = True

    df["source_year_file"] = label

    meta = {
        "label": label,
        "path": path,
        "sheet": sheet,
        "raw_columns": raw_columns,
        "renamed": colmap,
        "unknown_dropped": unknown,
        "units_row_dropped": units_dropped,
        "n_rows_after_units": int(len(df)),
        "has_single_datetime": COL_DT in df.columns,
        "has_split_datetime": COL_DATE in df.columns and COL_TIME in df.columns,
    }
    return df, meta
