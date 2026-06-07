"""Masked multi-quantile (pinball) loss.

This is the REAL loss the prior notebook claimed but never used: it honours the configured
quantiles instead of hardcoding q=0.5, and it averages only over OBSERVED targets (the mask
zeroes out missing/gap cells so they never contribute or dilute the denominator).

Models emit, per horizon, a tensor whose last dim is `C*Q`; it is reshaped to (..., C, Q).
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


class MaskedQuantileLoss(nn.Module):
    def __init__(self, quantiles=DEFAULT_QUANTILES):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.register_buffer("_q", torch.tensor(self.quantiles, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """preds: (..., C, Q); target/mask: (..., C). Returns scalar mean pinball over observed."""
        q = self._q.to(preds.dtype)
        err = target.unsqueeze(-1) - preds                      # (..., C, Q)
        pin = torch.maximum((q - 1.0) * err, q * err)           # (..., C, Q)
        m = mask.unsqueeze(-1)                                   # (..., C, 1)
        denom = m.sum() * len(self.quantiles)
        if denom <= 0:
            return preds.sum() * 0.0                            # keeps graph, zero contribution
        return (pin * m).sum() / denom


def median_index(quantiles=DEFAULT_QUANTILES) -> int:
    """Index of the q=0.5 head (the point forecast). Falls back to the middle quantile."""
    qs = list(quantiles)
    if 0.5 in qs:
        return qs.index(0.5)
    return len(qs) // 2
