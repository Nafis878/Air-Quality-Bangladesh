# External station coordinates (documented, NOT part of the cleaned data artifact)

`station_coords.csv` holds **approximate** latitude/longitude for each monitoring station, used
only by the model's geographic/wind-aware spatial graph (`src/modeling/geo.py`). It is deliberately
kept here, separate from `data/processed/air_quality_clean.parquet`, and is **never merged** into the
cleaned artifact — the cleaning pipeline and the leakage-safe splits are unchanged.

- **Source:** public city/landmark centroids (Wikipedia / OpenStreetMap), transcribed by hand.
- **Precision:** city- or site-centroid level (≈ km). Every row is flagged `approximate=1`. Several
  Dhaka-area sites (BARC, Darussalam, TV_Center, DoE, plus nearby Gazipur/Narayanganj/Narsingdi)
  cluster within ~30 km of central Dhaka.
- **Caveats:** `TV_Center` is flagged UNCONFIRMED in `src/schema.py` (`AMBIGUOUS_STATIONS`); its
  coordinate is the Dhaka TV-centre (Rampura) best guess and should be treated as uncertain.
- **Role:** these coordinates are an *inductive prior* for the spatial graph (distance decay + wind
  transport direction). The model also keeps a learned-correlation graph variant
  (`GAMMA_learned_adj`) and a `GAMMA_no_geo` ablation, so results are reported with and without this
  external information and never depend on it silently.
