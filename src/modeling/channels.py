"""Modeling constants and data-driven station-set resolution.

The spatial axis of GAMMA needs a FIXED set of stations, resolved from the TRAIN slice
only (never from val/test). A station qualifies if it has usable observed coverage in
train AND is also present in val and test (so it can actually be evaluated). Stations that
fail are returned in `excluded` with a reason, to be reported — never silently dropped.
"""
from __future__ import annotations

import pandas as pd

from ..schema import MODEL_CHANNELS

CHANNELS = list(MODEL_CHANNELS)          # 16 forecast channels (pollutants + meteorology)
N_CHANNELS = len(CHANNELS)
HORIZONS = [1, 6, 24]                    # hours ahead: t+1, t+6, t+24
PM25_IDX = CHANNELS.index("pm25")        # headline channel
WIND_DIR_IDX = CHANNELS.index("wind_dir")    # raw degrees (for the wind-transport graph)
WIND_SPEED_IDX = CHANNELS.index("wind_speed")  # raw m/s


def resolve_station_set(
    splits: dict[str, pd.DataFrame],
    min_train_obs_rate: float = 0.30,
    min_test_obs_rate: float = 0.05,
) -> tuple[list[str], dict[str, str]]:
    """Return (kept_stations_sorted, {excluded_station: reason}).

    `min_train_obs_rate` is the mean fraction of observed (non-NaN, non-gap) cells across
    the 16 channels required in TRAIN. A station must also appear in val and test with at
    least `min_test_obs_rate` observed PM2.5 to be evaluable.
    """
    train, val, test = splits["train"], splits["val"], splits["test"]

    def obs_rate(df: pd.DataFrame, st: str) -> float:
        sub = df[(df["station"] == st) & (~df["is_gap"])]
        if len(sub) == 0:
            return 0.0
        return float(sub[CHANNELS].notna().mean().mean())

    def pm25_obs(df: pd.DataFrame, st: str) -> int:
        sub = df[(df["station"] == st) & (~df["is_gap"])]
        return int(sub["pm25"].notna().sum())

    all_stations = sorted(set(train["station"]) | set(val["station"]) | set(test["station"]))
    kept: list[str] = []
    excluded: dict[str, str] = {}
    for st in all_stations:
        in_tr = st in set(train["station"])
        in_va = st in set(val["station"])
        in_te = st in set(test["station"])
        tr_rate = obs_rate(train, st)
        te_pm = pm25_obs(test, st)
        n_te = int((test["station"] == st).sum())
        te_rate = (te_pm / n_te) if n_te else 0.0
        if not in_tr:
            excluded[st] = "cold-start: absent from TRAIN (no learnable station embedding)"
        elif not (in_va and in_te):
            excluded[st] = "exits before val/test: present in train only, cannot be evaluated"
        elif tr_rate < min_train_obs_rate:
            excluded[st] = f"insufficient train coverage (obs-rate {tr_rate:.2f} < {min_train_obs_rate:.2f})"
        elif te_rate < min_test_obs_rate:
            excluded[st] = f"insufficient test coverage (PM2.5 obs-rate {te_rate:.2f} < {min_test_obs_rate:.2f})"
        else:
            kept.append(st)
    return sorted(kept), excluded
