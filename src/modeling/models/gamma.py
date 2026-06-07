"""GAMMA — Gated tri-axis attention forecaster (honest reimplementation from intent).

Fixes every mock in the original notebook:
  * the spatial axis ACTUALLY uses `spatial_bias`: it is added into the station-attention
    logits via `attn_mask`, so it influences the output and receives gradient;
  * the temporal axis embeds the full channel vector per step (not `x.sum(-1)`) and injects
    a GRU-D decay penalty so stale steps are down-weighted;
  * the variable axis attends over real per-variable embeddings with an observation
    key-padding mask;
  * the gate is a real softmax over the three pooled streams.

Input is a station cross-section so the spatial axis has something to attend over:
  x, mask, decay : [B, S, seq, C]      present : [B, S]      station_idx : [B, S]
Output           : preds [B, S, H, C, Q]  (a quantile forecast per station/horizon/channel)

Ablation switches (`use_variable/use_temporal/use_spatial/use_gate/use_decay`) let
`ablations.py` remove one component at a time; a disabled stream is dropped from the gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return out


class GAMMA(nn.Module):
    def __init__(
        self, num_channels: int, num_stations: int, d_model: int = 64, n_heads: int = 4,
        seq_len: int = 24, n_horizons: int = 3, n_quantiles: int = 3,
        use_variable: bool = True, use_temporal: bool = True, use_spatial: bool = True,
        use_gate: bool = True, use_decay: bool = True,
    ):
        super().__init__()
        self.C, self.S, self.d = num_channels, num_stations, d_model
        self.H, self.Q = n_horizons, n_quantiles
        self.use_variable, self.use_temporal, self.use_spatial = use_variable, use_temporal, use_spatial
        self.use_gate, self.use_decay = use_gate, use_decay

        self.var_embedding = nn.Linear(1, d_model)            # per-variable scalar -> d
        self.temp_embedding = nn.Linear(num_channels, d_model)  # channel vector per step -> d
        self.station_embedding = nn.Embedding(num_stations, d_model)
        self.decay_proj = nn.Linear(1, d_model)

        self.var_attn = _MHA(d_model, n_heads)
        self.temp_attn = _MHA(d_model, n_heads)
        self.spatial_attn = _MHA(d_model, n_heads)
        self.spatial_bias = nn.Parameter(torch.zeros(num_stations, num_stations))

        self.gate = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.Tanh(), nn.Linear(d_model, 3))

        self.heads = nn.ModuleList([nn.Linear(d_model, num_channels * n_quantiles) for _ in range(n_horizons)])

    def forward(self, x, mask, decay, present, station_idx):
        B, S, L, C = x.shape
        d = self.d
        x_last = x[:, :, -1, :]                 # [B,S,C]
        m_last = mask[:, :, -1, :]              # [B,S,C]
        BS = B * S

        # --- station embedding (shared context) ---
        stat = self.station_embedding(station_idx)        # [B,S,d]

        # --- 1. variable axis ---
        if self.use_variable:
            hv = self.var_embedding(x_last.reshape(BS, C, 1))          # [BS,C,d]
            kp = (m_last.reshape(BS, C) == 0)                           # ignore unobserved vars
            kp = kp & ~kp.all(dim=1, keepdim=True)                     # guard all-masked rows
            ov = self.var_attn(hv, key_padding_mask=kp)               # [BS,C,d]
            h_var = ov.mean(dim=1).reshape(B, S, d)                    # [B,S,d]
        else:
            h_var = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- 2. temporal axis ---
        if self.use_temporal:
            ht = self.temp_embedding(x.reshape(BS, L, C))             # [BS,L,d]
            ht = ht + stat.reshape(BS, 1, d)
            if self.use_decay:
                dec = decay.mean(dim=-1).reshape(BS, L, 1)            # [BS,L,1] avg over channels
                ht = ht - self.decay_proj(dec)                       # penalize stale steps
            ot = self.temp_attn(ht)                                   # [BS,L,d]
            h_temp = ot[:, -1, :].reshape(B, S, d)
        else:
            h_temp = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- 3. spatial axis (spatial_bias injected into attention logits) ---
        if self.use_spatial:
            base = h_temp if self.use_temporal else stat              # per-station token [B,S,d]
            # float key-padding (-inf at absent stations) to match the float attn_mask dtype
            kp_s = torch.zeros_like(present)
            kp_s = kp_s.masked_fill(present == 0, float("-inf"))      # [B,S]
            all_absent = torch.isinf(kp_s).all(dim=1, keepdim=True)
            kp_s = kp_s.masked_fill(all_absent, 0.0)                  # guard all-masked rows
            os_ = self.spatial_attn(base, key_padding_mask=kp_s, attn_mask=self.spatial_bias)
            h_spatial = os_                                            # [B,S,d]
        else:
            h_spatial = torch.zeros(B, S, d, device=x.device, dtype=x.dtype)

        # --- gated fusion over the active streams ---
        active = [self.use_variable, self.use_temporal, self.use_spatial]
        streams = torch.stack([h_var, h_temp, h_spatial], dim=-2)     # [B,S,3,d]
        if self.use_gate:
            cat = torch.cat([h_var, h_temp, h_spatial], dim=-1)       # [B,S,3d]
            logits = self.gate(cat)                                   # [B,S,3]
            neg = torch.full_like(logits, float("-inf"))
            keep = torch.tensor(active, device=x.device)
            logits = torch.where(keep, logits, neg)
            w = torch.softmax(logits, dim=-1).unsqueeze(-1)           # [B,S,3,1]
            fused = (w * streams).sum(dim=-2)                         # [B,S,d]
        else:
            keep = torch.tensor(active, device=x.device, dtype=x.dtype)
            fused = (streams * keep.view(1, 1, 3, 1)).sum(dim=-2) / keep.sum()

        preds = [head(fused).view(B, S, C, self.Q) for head in self.heads]   # H x [B,S,C,Q]
        return torch.stack(preds, dim=2)                              # [B,S,H,C,Q]
