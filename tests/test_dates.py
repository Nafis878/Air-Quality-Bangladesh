"""Datetime unification: both encodings -> one tz-naive timestamp."""
import pandas as pd

from src.clean import build_timestamp

ORIGIN = "1899-12-30"


def test_split_date_serial_plus_time():
    # 2012-2021 encoding: Excel serial Date + "HH:MM" Time
    df = pd.DataFrame({"_date": [41214, 41214], "_time": ["01:00", "13:30"], "x": [1, 2]})
    out, n_bad = build_timestamp(df, ORIGIN)
    assert n_bad == 0
    base = pd.to_datetime(41214, unit="D", origin=ORIGIN)
    assert out["timestamp"].iloc[0] == base + pd.Timedelta(hours=1)
    assert out["timestamp"].iloc[1] == base + pd.Timedelta(hours=13, minutes=30)
    assert "_date" not in out.columns and "_time" not in out.columns


def test_single_datetime_passthrough():
    df = pd.DataFrame({"_datetime": pd.to_datetime(["2022-01-01 01:00", "2022-06-15 09:00"]), "x": [1, 2]})
    out, n_bad = build_timestamp(df, ORIGIN)
    assert n_bad == 0
    assert out["timestamp"].iloc[0] == pd.Timestamp("2022-01-01 01:00")


def test_single_datetime_as_excel_serial():
    # If a single Date&Time column arrives as numeric serials, convert by origin.
    df = pd.DataFrame({"_datetime": ["44562.041666667", "44562.5"], "x": [1, 2]})
    out, n_bad = build_timestamp(df, ORIGIN)
    assert n_bad == 0
    # 44562 -> 2022-01-01; .0416667 day ~ 01:00
    assert out["timestamp"].iloc[0].strftime("%Y-%m-%d") == "2022-01-01"
    assert out["timestamp"].iloc[0].hour == 1


def test_unparseable_counts_as_nat():
    df = pd.DataFrame({"_datetime": ["not a date", None], "x": [1, 2]})
    out, n_bad = build_timestamp(df, ORIGIN)
    assert n_bad == 2
    assert out["timestamp"].isna().all()
