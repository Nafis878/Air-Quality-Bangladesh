"""GAMMA v3 — staleness-aware cross-station spatial forecaster.

The whole point: every benchmark sees ONE station's window and is blind to other stations. GAMMA
ingests the full station cross-section, so when a station's own sensor is stale/silent it can pull a
fresh, correlated reading from a neighbour. Two pieces make that capability real and explicit:

  * a SPATIAL GRAPH (the reach) — multi-head learnable station->station adjacency added to the
    spatial-attention logits, stacked over `spatial_layers` pre-norm encoder blocks (multi-hop
    transport). The adjacency can be initialised from a TRAIN-only inter-station correlation prior
    (`adj_prior`) since this dataset has no coordinates. `present` key-padding stops absent stations
    leaking.
  * a STALENESS-AWARE GATE (the choice) — the fusion gate also sees each station's self-staleness
    (hours since its own last observation), so it can learn "when my own data is stale, trust the
    spatial stream (neighbours) over my own temporal stream." `use_staleness_gate=False` zeroes that
    signal (ablation), same architecture otherwise.

Carried over from v2: pre-norm residual encoder blocks on the variable + temporal axes, a bounded
multiplicative GRU-D decay gate gamma=exp(-softplus(rate)*delta/24) in (0,1], and full-window
embeddings. The persistence prior is now applied UNIFORMLY to every model OUTSIDE the net (the shared
anchor in fasttrain), so GAMMA no longer adds last_obs internally — the fight is fair.

Input is a station cross-section so the spatial axis has stations to attend over:
  x, mask, decay : [B, S, seq, C]    present : [B, S]    station_idx : [B, S]
Output           : preds [B, S, H, C, Q]  (residual; the harness adds the persistence anchor)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Cross-station message passing: per-head learned adjacency added to attention logits,
    stacked over `n_layers` blocks (multi-hop). Optional correlation prior initialises adjacency."""

    def __init__(self, d_model, n_heads, num_stations, n_layers=2, adj_prior=None):
        super().__init__()
        self.n_heads, self.S = n_heads, num_stations
        self.blocks = nn.ModuleList([_EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        init = torch.zeros(n_heads, num_stations, num_stations)
        if adj_prior is not None:
            init = init + torch.as_tensor(adj_prior, dtype=torch.float32).unsqueeze(0)
        self.adjacency = nn.Parameter(init)                      # [n_heads, S, S] learnable bias

    def forward(self, h, present):
        B, S, d = h.shape
        kp = torch.zeros(B, S, device=h.device, dtype=h.dtype).masked_fill(present == 0, float("-inf"))
        kp = kp.masked_fill(torch.isinf(kp).all(dim=1, keepdim=True), 0.0)   # all-absent -> no mask
        attn_mask = self.adjacency.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.n_heads, S, S)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=kp, attn_mask=attn_mask)
        return h


class GAMMA(nn.Module):
    def __init__(
        self, num_channels: int, num_stations: int, d_model: int = 64, n_heads: int = 4,
        seq_len: int = 24, n_horizons: int = 3, n_quantiles: int = 3, spatial_layers: int = 2,
        use_variable: bool = True, use_temporal: bool = True, use_spatial: bool = True,
        use_gate: bool = True, use_decay: bool = True, use_staleness_gate: bool = True,
        adj_prior=None,
    ):
        super().__init__()
        self.C, self.S, self.d = num_channels, num_stations, d_model
        self.H, self.Q, self.L = n_horizons, n_quantiles, seq_len
        self.use_variable, self.use_temporal, self.use_spatial = use_variable, use_temporal, use_spatial
        self.use_gate, self.use_decay, self.use_staleness_gate = use_gate, use_decay, use_staleness_gate

        # variable axis: per-channel token from (last value, window mean)
        self.var_embedding = nn.Linear(2, d_model)
        # temporal axis: per-step channel vector -> d, + positional + station embeddings
        self.temp_embedding = nn.Linear(num_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.station_embedding = nn.Embedding(num_stations, d_model)
        self.decay_rate = nn.Parameter(torch.zeros(d_model))     # softplus -> positive rate

        self.var_block = _EncoderBlock(d_model, n_heads)
        self.temp_block = _EncoderBlock(d_model, n_heads)
        self.spatial_graph = _SpatialGraph(d_model, n_heads, num_stations, spatial_layers, adj_prior)

        # fusion gate sees the three stream summaries + self-staleness scalar (-> spatial routing)
        self.gate = nn.Sequential(nn.Linear(d_model * 3 + 1, d_model), nn.Tanh(), nn.Linear(d_model, 3))
        self.heads = nn.ModuleList([nn.Linear(d_model, num_channels * n_quantiles) for _ in range(n_horizons)])

    def _last_observed(self, x, mask):
        """Most-recent in-window observed value per (B,S,C); 0 if never observed. -> [B,S,C]."""
        B, S, L, C = x.shape
        pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
        seen = mask > 0
        last_idx = torch.where(seen, pos, torch.zeros_like(pos)).amax(dim=2)        # [B,S,C]
        gathered = torch.gather(x, 2, last_idx.unsqueeze(2)).squeeze(2)             # [B,S,C]
        ever = seen.any(dim=2)
        return torch.where(ever, gathered, torch.zeros_like(gathered))

    def forward(self, x, mask, decay, present, station_idx):
        B, S, L, C = x.shape
        d, BS = self.d, B * S
        stat = self.station_embedding(station_idx)                # [B,S,d]
        last_obs = self._last_observed(x, mask)                   # [B,S,C] (variable-axis feature)

        # --- 1. variable axis (window summary per channel) ---
        if self.use_variable:
            win_mean = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)       # [B,S,C]
            feat = torch.stack([last_obs, win_mean], dim=-1).reshape(BS, C, 2)      # [BS,C,2]
            hv = self.var_embedding(feat)                                          # [BS,C,d]
            kp = (mask[:, :, -1, :].reshape(BS, C) == 0)
            kp = kp & ~kp.all(dim=1, keepdim=True)
            ov = self.var_block(hv, key_padding_mask=kp)
            h_var = ov.mean(dim=1).reshape(B, S, d)
        else:
            h_var = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- 2. temporal axis (full window + bounded GRU-D decay gate) ---
        if self.use_temporal:
            ht = self.temp_embedding(x.reshape(BS, L, C)) + self.pos_embedding + stat.reshape(BS, 1, d)
            if self.use_decay:
                delta = (decay.mean(dim=-1).reshape(BS, L, 1)) / 24.0               # [BS,L,1] days
                gamma = torch.exp(-F.softplus(self.decay_rate).view(1, 1, d) * delta)  # (0,1]
                ht = ht * gamma
            ot = self.temp_block(ht)
            h_temp = ot[:, -1, :].reshape(B, S, d)
        else:
            h_temp = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- 3. spatial axis: cross-station graph (the uniquely-GAMMA capability) ---
        if self.use_spatial:
            base = h_temp if self.use_temporal else stat
            h_spatial = self.spatial_graph(base, present)
        else:
            h_spatial = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # self-staleness per (B,S): hours since last obs (mean over channels at the anchor step) / 24
        self_stale = decay[:, :, -1, :].mean(dim=-1, keepdim=True) / 24.0           # [B,S,1]
        if not self.use_staleness_gate:
            self_stale = torch.zeros_like(self_stale)

        # --- gated fusion over active streams (gate routes by staleness) ---
        active = [self.use_variable, self.use_temporal, self.use_spatial]
        streams = torch.stack([h_var, h_temp, h_spatial], dim=-2)     # [B,S,3,d]
        if self.use_gate:
            logits = self.gate(torch.cat([h_var, h_temp, h_spatial, self_stale], dim=-1))
            keep = torch.tensor(active, device=x.device)
            logits = torch.where(keep, logits, torch.full_like(logits, float("-inf")))
            w = torch.softmax(logits, dim=-1).unsqueeze(-1)
            fused = (w * streams).sum(dim=-2)
        else:
            keep = torch.tensor(active, device=x.device, dtype=x.dtype)
            fused = (streams * keep.view(1, 1, 3, 1)).sum(dim=-2) / keep.sum()

        # --- multi-horizon quantile heads (residual; harness adds the persistence anchor) ---
        preds = [head(fused).view(B, S, C, self.Q) for head in self.heads]
        return torch.stack(preds, dim=2)                              # [B,S,H,C,Q]
