"""Lag/rolling features must use only the past — no future leakage, no cross-station bleed."""
import numpy as np
import pandas as pd

from src.features import add_lag_features, add_time_features, add_wind_features


def _series(station, start, n):
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"station": station, "timestamp": ts, "pm25": np.arange(n, dtype=float)})


def test_lag1_equals_previous_value():
    df = _series("A", "2022-01-01", 5)
    out = add_lag_features(df, cols=["pm25"], lags=(1,), windows=())
    # pm25 = [0,1,2,3,4]; lag1 should be [NaN,0,1,2,3]
    assert np.isnan(out["pm25_lag1"].iloc[0])
    assert list(out["pm25_lag1"].iloc[1:]) == [0.0, 1.0, 2.0, 3.0]


def test_rolling_excludes_current_row():
    df = _series("A", "2022-01-01", 5)
    out = add_lag_features(df, cols=["pm25"], lags=(), windows=(2,))
    # rolling mean over the PAST 2 (shifted by 1): row t uses values t-2,t-1
    # values 0..4 -> rollmean2 = [NaN, 0, 0.5, 1.5, 2.5]
    rm = out["pm25_rollmean2"].tolist()
    assert np.isnan(rm[0])
    assert rm[1:] == [0.0, 0.5, 1.5, 2.5]
    # the current value is never included -> last row mean (2.5) < current value (4)
    assert rm[-1] < out["pm25"].iloc[-1]


def test_no_cross_station_bleed():
    a = _series("A", "2022-01-01", 3)
    b = _series("B", "2022-01-01", 3)
    out = add_lag_features(pd.concat([a, b], ignore_index=True), cols=["pm25"], lags=(1,), windows=())
    # first row of each station has no prior -> NaN lag (B must not borrow A's last value)
    first_b = out[out["station"] == "B"].sort_values("timestamp").iloc[0]
    assert np.isnan(first_b["pm25_lag1"])


def test_wind_dir_cyclical_continuity():
    df = pd.DataFrame({"station": "A", "timestamp": pd.date_range("2022-01-01", periods=2, freq="1h"),
                       "wind_dir": [0.0, 360.0]})
    out = add_wind_features(df)
    # 0 and 360 degrees must map to (nearly) the same sin/cos
    assert abs(out["wind_dir_sin"].iloc[0] - out["wind_dir_sin"].iloc[1]) < 1e-9
    assert abs(out["wind_dir_cos"].iloc[0] - out["wind_dir_cos"].iloc[1]) < 1e-9


def test_time_features_are_deterministic_functions_of_timestamp():
    df = pd.DataFrame({"station": "A", "timestamp": [pd.Timestamp("2022-07-15 13:00")], "pm25": [1.0]})
    out = add_time_features(df)
    assert out["hour"].iloc[0] == 13
    assert out["month"].iloc[0] == 7
    assert out["is_monsoon"].iloc[0] == 1   # July is monsoon
