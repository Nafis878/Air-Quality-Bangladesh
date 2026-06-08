"""Geographic + wind-transport structure for GAMMA's spatial graph.

Station coordinates are EXTERNAL, approximate metadata (`data/external/station_coords.csv`, see its
README) — never merged into the cleaned data artifact. From them we derive an inductive spatial prior:

  * `haversine_km`   : great-circle distance between stations  -> distance-decay edges (near matters).
  * `bearings_deg`   : initial bearing a->b                     -> wind-transport direction.
  * `wind_transport_bias`: dynamic, per-hour edge strength — a source station j that is currently
    blowing TOWARD i (its wind vector points along bearing j->i) feeds pollution to i.

Meteorological convention: `wind_dir` is the direction the wind comes FROM, so the wind blows toward
`wind_dir + 180`. Edge j->i (key j attended by query i) is strong when (wind_dir_j + 180) aligns with
bearing(j->i). All angle math is in degrees; everything is inductive (works for any station set,
including a cold-start node not seen in training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def load_coords(path: str, stations: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (lat[S], lon[S], missing) aligned to `stations`. Missing stations get NaN coords."""
    df = pd.read_csv(path).set_index("station")
    lat = np.full(len(stations), np.nan, dtype=np.float64)
    lon = np.full(len(stations), np.nan, dtype=np.float64)
    missing = []
    for i, s in enumerate(stations):
        if s in df.index:
            lat[i] = float(df.loc[s, "lat"]); lon[i] = float(df.loc[s, "lon"])
        else:
            missing.append(s)
    return lat, lon, missing


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance [S,S] in km."""
    R = 6371.0
    la = np.radians(lat)[:, None]; lo = np.radians(lon)[:, None]
    la2 = np.radians(lat)[None, :]; lo2 = np.radians(lon)[None, :]
    dlat = la2 - la; dlon = lo2 - lo
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return (2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)


def bearings_deg(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Initial bearing [S,S] in degrees [0,360): element [a,b] is the bearing FROM a TO b."""
    la = np.radians(lat)[:, None]; la2 = np.radians(lat)[None, :]
    dlon = np.radians(lon)[None, :] - np.radians(lon)[:, None]
    x = np.sin(dlon) * np.cos(la2)
    y = np.cos(la) * np.sin(la2) - np.sin(la) * np.cos(la2) * np.cos(dlon)
    brg = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return brg.astype(np.float32)


def distance_decay(dist_km: np.ndarray, length_scale_km: float = 50.0) -> np.ndarray:
    """exp(-d/ell) in (0,1]; diagonal forced to 0 (self handled by the attention itself)."""
    out = np.exp(-dist_km / float(length_scale_km)).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def wind_transport_bias(wind_dir: torch.Tensor, wind_speed: torch.Tensor, wind_mask: torch.Tensor,
                        bearing: torch.Tensor, speed_scale: float = 5.0) -> torch.Tensor:
    """Dynamic edge strength [B,S,S]; out[b,i,j] = how strongly query i should pull from key j given
    j's current wind. Strong when j blows toward i: (wind_dir_j + 180) aligns with bearing(j->i).

    wind_dir/wind_speed/wind_mask : [B,S] raw (deg, m/s, 1=observed). bearing : [S,S] deg (a->b).
    """
    B, S = wind_dir.shape
    blow_to = (wind_dir + 180.0) % 360.0                      # direction wind blows TOWARD, per source j
    brg_ji = bearing.unsqueeze(0)                             # [1,S,S]; [.,j,i] = bearing(j->i)
    # align[b,j,i] in [0,1]: cos of angle between j's outflow and the j->i direction, relu'd
    diff = torch.deg2rad(blow_to.unsqueeze(-1) - brg_ji)      # [B,S(j),S(i)]
    align = torch.relu(torch.cos(diff))
    speed = torch.tanh(wind_speed.clamp_min(0.0) / speed_scale) * wind_mask   # [B,S(j)]
    bias_ji = align * speed.unsqueeze(-1)                     # [B,S(j),S(i)]
    return bias_ji.transpose(1, 2).contiguous()              # [B,S(i),S(j)]  -> query i, key j
