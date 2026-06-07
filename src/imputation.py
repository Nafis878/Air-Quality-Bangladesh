"""
Fold-aware imputation — provided as a SEPARATE, callable transform. It is deliberately
NOT applied to the canonical clean artifact, because imputation strategy depends on the
model and must be fit inside CV folds to avoid leakage.

`GapImputer.fit` learns only train-slice fallback statistics (per-station, per-month
medians). `transform` does per-station time interpolation limited to `max_gap` hours,
adds a `<col>_was_missing` indicator for every imputed column, and fills any remaining
long-gap holes with the train-learned fallback. Indicators let a model know which values
were synthesized.
"""
from __future__ import annotations

import pandas as pd

from .schema import NUMERIC_COLS


class GapImputer:
    def __init__(self, cols: list[str] | None = None, max_gap: int = 6):
        self.cols = cols
        self.max_gap = max_gap
        self.fallback_: dict[tuple[str, int, str], float] = {}
        self.global_median_: dict[str, float] = {}

    def fit(self, train: pd.DataFrame) -> "GapImputer":
        """Learn fallback medians from TRAIN ONLY (per station x month, + global)."""
        if self.cols is None:
            self.cols = [c for c in NUMERIC_COLS if c in train.columns]
        for c in self.cols:
            self.global_median_[c] = float(train[c].median())
        tmp = train.copy()
        tmp["_m"] = tmp["timestamp"].dt.month
        for (st, m), g in tmp.groupby(["station", "_m"]):
            for c in self.cols:
                med = g[c].median()
                if pd.notna(med):
                    self.fallback_[(st, int(m), c)] = float(med)
        return self

    def transform(self, df: pd.DataFrame, add_indicator: bool = True) -> pd.DataFrame:
        """Per-station time-interpolate up to max_gap; add missing indicators; fill rest."""
        out = df.sort_values(["station", "timestamp"]).copy()
        for c in self.cols:
            if add_indicator:
                out[f"{c}_was_missing"] = out[c].isna().astype("int8")
        # limited-gap linear interpolation, per station, against the time index
        def _interp(g: pd.DataFrame) -> pd.DataFrame:
            gi = g.set_index("timestamp")
            for c in self.cols:
                gi[c] = gi[c].interpolate(method="time", limit=self.max_gap, limit_direction="both")
            return gi.reset_index()
        out = out.groupby("station", group_keys=False).apply(_interp)
        # remaining holes (gaps longer than max_gap) -> train-learned fallback
        month = out["timestamp"].dt.month
        for c in self.cols:
            mask = out[c].isna()
            if not mask.any():
                continue
            keys = list(zip(out.loc[mask, "station"], month[mask].astype(int)))
            fb = pd.Series(
                [self.fallback_.get((st, m, c), self.global_median_[c]) for st, m in keys],
                index=out.index[mask],
            )
            out.loc[mask, c] = fb
        return out

    def fit_transform(self, train: pd.DataFrame, add_indicator: bool = True) -> pd.DataFrame:
        return self.fit(train).transform(train, add_indicator=add_indicator)
