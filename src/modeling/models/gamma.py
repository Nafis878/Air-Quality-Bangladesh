"""GAMMA-final — inductive, wind-aware cross-station air-quality forecaster.

The point of the model: forecast a station THROUGH its own sensor outage by routing pollution
transport from neighbours — something no single-station baseline can do. Three things make that work
and make the model inductive (able to forecast a station never seen in training, e.g. cold-start DoE):

  * COORDINATE-CONDITIONED station embedding — an MLP on (lat,lon) instead of a per-station lookup,
    so any station (incl. an unseen one) gets an embedding from its location.
  * PHYSICS-INFORMED spatial graph — attention-logit bias from geography (distance decay) + dynamic
    WIND TRANSPORT (a source station currently blowing toward the target feeds it; see geo.py). An
    edge-MLP maps these per-edge features to per-head biases — inductive (no free per-pair params).
    Switches: use_geo (distance), use_wind (transport). use_learned_adj falls back to v3's free
    [n_heads,S,S] correlation graph (transductive — for the ablation only).
  * STALENESS-AWARE GATE + GRU-D decay — trust neighbours when the station's own data is stale.

The persistence prior is applied UNIFORMLY outside the net (the shared anchor in fasttrain), so the
heads emit a residual and the fight against the baselines is fair.

Inputs (cross-section):
  x, mask, decay : [B,S,seq,C]   present : [B,S]   station_idx : [B,S]
  wind           : [B,S,3] raw (dir_deg, speed_ms, mask) or None
  geo            : (coords[S,2], dist_feats[S,S,2], bearing[S,S]) tensors or None (-> v3 fallback)
Output           : preds [B,S,H,C,Q]  (residual; the harness adds the persistence anchor)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geo import wind_transport_bias

# Fixed coordinate normalisation (Bangladesh scale) so embeddings are consistent across station sets.
_LAT0, _LON0, _COORD_SCALE = 23.7, 90.4, 2.0


class _EncoderBlock(nn.Module):
    """Pre-norm residual block: x + MHA(LN(x)); x + FFN(LN(x)). Supports bias/padding masks."""

    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                nn.Linear(ff_mult * d_model, d_model))

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class _SpatialGraph(nn.Module):
    """Stacked pre-norm blocks (multi-hop). The per-edge attention bias is computed by GAMMA and
    passed in as `attn_mask`; this module only handles padding + the message-passing depth."""

    def __init__(self, d_model, n_heads, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([_EncoderBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, h, present, attn_mask):
        B, S, d = h.shape
        kp = torch.zeros(B, S, device=h.device, dtype=h.dtype).masked_fill(present == 0, float("-inf"))
        kp = kp.masked_fill(torch.isinf(kp).all(dim=1, keepdim=True), 0.0)   # all-absent -> no mask
        for blk in self.blocks:
            h = blk(h, key_padding_mask=kp, attn_mask=attn_mask)
        return h


class GAMMA(nn.Module):
    def __init__(
        self, num_channels: int, num_stations: int, d_model: int = 64, n_heads: int = 4,
        seq_len: int = 24, n_horizons: int = 3, n_quantiles: int = 3, spatial_layers: int = 2,
        use_variable: bool = True, use_temporal: bool = True, use_spatial: bool = True,
        use_gate: bool = True, use_decay: bool = True, use_staleness_gate: bool = True,
        use_geo: bool = True, use_wind: bool = True, use_learned_adj: bool = False,
        adj_prior=None,
    ):
        super().__init__()
        self.C, self.S, self.d = num_channels, num_stations, d_model
        self.H, self.Q, self.L, self.n_heads = n_horizons, n_quantiles, seq_len, n_heads
        self.use_variable, self.use_temporal, self.use_spatial = use_variable, use_temporal, use_spatial
        self.use_gate, self.use_decay, self.use_staleness_gate = use_gate, use_decay, use_staleness_gate
        self.use_geo, self.use_wind, self.use_learned_adj = use_geo, use_wind, use_learned_adj

        self.var_embedding = nn.Linear(2, d_model)
        self.temp_embedding = nn.Linear(num_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.coord_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.station_embedding = nn.Embedding(num_stations, d_model)   # fallback when geo is absent
        self.decay_rate = nn.Parameter(torch.zeros(d_model))

        self.var_block = _EncoderBlock(d_model, n_heads)
        self.temp_block = _EncoderBlock(d_model, n_heads)
        self.spatial_graph = _SpatialGraph(d_model, n_heads, spatial_layers)
        # inductive physics edges: per-edge features [dist_decay_short, dist_decay_long, wind] -> heads
        self.edge_mlp = nn.Sequential(nn.Linear(3, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, n_heads))
        # transductive fallback adjacency (v3) — used only for the GAMMA_learned_adj ablation / no-geo
        init = torch.zeros(n_heads, num_stations, num_stations)
        if adj_prior is not None:
            init = init + torch.as_tensor(adj_prior, dtype=torch.float32).unsqueeze(0)
        self.learned_adj = nn.Parameter(init)

        self.gate = nn.Sequential(nn.Linear(d_model * 3 + 1, d_model), nn.Tanh(), nn.Linear(d_model, 3))
        self.heads = nn.ModuleList([nn.Linear(d_model, num_channels * n_quantiles) for _ in range(n_horizons)])

    def _last_observed(self, x, mask):
        """Most-recent in-window observed value per (B,S,C); 0 if never observed. -> [B,S,C]."""
        B, S, L, C = x.shape
        pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
        seen = mask > 0
        last_idx = torch.where(seen, pos, torch.zeros_like(pos)).amax(dim=2)
        gathered = torch.gather(x, 2, last_idx.unsqueeze(2)).squeeze(2)
        ever = seen.any(dim=2)
        return torch.where(ever, gathered, torch.zeros_like(gathered))

    def _station_repr(self, station_idx, geo):
        """Inductive coord-MLP embedding when geo is present; else the v3 learned lookup."""
        if geo is not None:
            coords = geo[0]                                              # [S,2] (lat,lon)
            norm = torch.stack([(coords[:, 0] - _LAT0) / _COORD_SCALE,
                                (coords[:, 1] - _LON0) / _COORD_SCALE], dim=-1)
            emb = self.coord_mlp(norm)                                  # [S,d]
            return emb.unsqueeze(0).expand(station_idx.shape[0], -1, -1)
        return self.station_embedding(station_idx)

    def _spatial_bias(self, B, S, wind, geo, device, dtype):
        """Per-head attention-logit bias [B*n_heads, S, S]. Physics edges (inductive) or learned adj."""
        if self.use_learned_adj or geo is None:
            adj = self.learned_adj[:, :S, :S]                          # [nh,S,S]
            return adj.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.n_heads, S, S)
        dist_feats, bearing = geo[1], geo[2]                           # [S,S,2], [S,S]
        dd = dist_feats if self.use_geo else torch.zeros_like(dist_feats)
        dd = dd.unsqueeze(0).expand(B, -1, -1, -1)                     # [B,S,S,2]
        if self.use_wind and wind is not None:
            wt = wind_transport_bias(wind[..., 0], wind[..., 1], wind[..., 2], bearing)  # [B,S,S]
        else:
            wt = torch.zeros(B, S, S, device=device, dtype=dtype)
        feat = torch.cat([dd, wt.unsqueeze(-1)], dim=-1)              # [B,S,S,3]
        bias = self.edge_mlp(feat)                                    # [B,S,S,nh]
        return bias.permute(0, 3, 1, 2).reshape(B * self.n_heads, S, S)

    def forward(self, x, mask, decay, present, station_idx, wind=None, geo=None):
        B, S, L, C = x.shape
        d, BS = self.d, B * S
        stat = self._station_repr(station_idx, geo)               # [B,S,d]
        last_obs = self._last_observed(x, mask)                   # [B,S,C]

        # --- 1. variable axis ---
        if self.use_variable:
            win_mean = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            feat = torch.stack([last_obs, win_mean], dim=-1).reshape(BS, C, 2)
            hv = self.var_embedding(feat)
            kp = (mask[:, :, -1, :].reshape(BS, C) == 0)
            kp = kp & ~kp.all(dim=1, keepdim=True)
            h_var = self.var_block(hv, key_padding_mask=kp).mean(dim=1).reshape(B, S, d)
        else:
            h_var = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- 2. temporal axis (full window + bounded GRU-D decay gate) ---
        if self.use_temporal:
            ht = self.temp_embedding(x.reshape(BS, L, C)) + self.pos_embedding + stat.reshape(BS, 1, d)
            if self.use_decay:
                delta = (decay.mean(dim=-1).reshape(BS, L, 1)) / 24.0
                gamma = torch.exp(-F.softplus(self.decay_rate).view(1, 1, d) * delta)
                ht = ht * gamma
            h_temp = self.temp_block(ht)[:, -1, :].reshape(B, S, d)
        else:
            h_temp = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- 3. spatial axis: physics-informed cross-station graph ---
        if self.use_spatial:
            base = h_temp if self.use_temporal else stat
            attn_mask = self._spatial_bias(B, S, wind, geo, x.device, x.dtype)
            h_spatial = self.spatial_graph(base, present, attn_mask)
        else:
            h_spatial = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # self-staleness routes the gate toward neighbours when own data is stale
        self_stale = decay[:, :, -1, :].mean(dim=-1, keepdim=True) / 24.0
        if not self.use_staleness_gate:
            self_stale = torch.zeros_like(self_stale)

        # --- gated fusion ---
        active = [self.use_variable, self.use_temporal, self.use_spatial]
        streams = torch.stack([h_var, h_temp, h_spatial], dim=-2)
        if self.use_gate:
            logits = self.gate(torch.cat([h_var, h_temp, h_spatial, self_stale], dim=-1))
            keep = torch.tensor(active, device=x.device)
            logits = torch.where(keep, logits, torch.full_like(logits, float("-inf")))
            fused = (torch.softmax(logits, dim=-1).unsqueeze(-1) * streams).sum(dim=-2)
        else:
            keep = torch.tensor(active, device=x.device, dtype=x.dtype)
            fused = (streams * keep.view(1, 1, 3, 1)).sum(dim=-2) / keep.sum()

        # --- multi-horizon quantile heads (residual; harness adds the persistence anchor) ---
        preds = [head(fused).view(B, S, C, self.Q) for head in self.heads]
        return torch.stack(preds, dim=2)                              # [B,S,H,C,Q]
