"""Deep baseline forecasters.

The ten architectures from the original notebook, kept as legitimate baselines but upgraded
to the real task: every model emits, per horizon, a (C, Q) quantile tensor (not a bare point),
so they train against the same masked multi-quantile loss as GAMMA.

Convention: `forward(x)` takes x [B, seq, C] (zero-filled scaled inputs) and returns
preds [B, H, C, Q] where H = number of horizons, Q = number of quantiles.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHorizonHead(nn.Module):
    """Map a [B, hidden] representation to [B, H, C, Q]."""

    def __init__(self, hidden: int, n_channels: int, n_horizons: int, n_quantiles: int):
        super().__init__()
        self.C, self.H, self.Q = n_channels, n_horizons, n_quantiles
        self.proj = nn.Linear(hidden, n_horizons * n_channels * n_quantiles)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).view(-1, self.H, self.C, self.Q)


class GRUModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.gru = nn.GRU(C, hidden, batch_first=True)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class BiLSTMModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.lstm = nn.LSTM(C, hidden, batch_first=True, bidirectional=True)
        self.head = MultiHorizonHead(hidden * 2, C, n_horizons, n_quantiles)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class FFNModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(C * seq_len, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        h = F.relu(self.fc2(F.relu(self.fc1(self.flatten(x)))))
        return self.head(h)


class CNN1DModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.conv1 = nn.Conv1d(C, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden * seq_len, hidden)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        h = F.relu(self.fc(self.flatten(x)))
        return self.head(h)


class _ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x):
        r = x
        x = F.relu(self.c1(x))
        return F.relu(self.c2(x) + r)


class ResNet1DModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.inp = nn.Conv1d(C, hidden, kernel_size=1)
        self.res1 = _ResBlock1D(hidden)
        self.res2 = _ResBlock1D(hidden)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden * seq_len, hidden)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.inp(x))
        x = self.res2(self.res1(x))
        h = F.relu(self.fc(self.flatten(x)))
        return self.head(h)


class TCNModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.conv1 = nn.Conv1d(C, hidden, 3, dilation=1, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, dilation=2, padding=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden * seq_len, hidden)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = F.relu(self.conv1(x))[:, :, :-2]      # causal trim
        x2 = F.relu(self.conv2(x1))[:, :, :-4]
        h = F.relu(self.fc(self.flatten(x2)))
        return self.head(h)


class _TSMixerBlock(nn.Module):
    def __init__(self, seq_len, C):
        super().__init__()
        self.time_mix = nn.Sequential(nn.Linear(seq_len, seq_len), nn.ReLU())
        self.feat_mix = nn.Sequential(nn.Linear(C, C), nn.ReLU())

    def forward(self, x):
        xt = self.time_mix(x.transpose(1, 2)).transpose(1, 2) + x
        return self.feat_mix(xt) + xt


class TSMixerModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.m1 = _TSMixerBlock(seq_len, C)
        self.m2 = _TSMixerBlock(seq_len, C)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(seq_len * C, hidden)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        x = self.m2(self.m1(x))
        h = F.relu(self.fc(self.flatten(x)))
        return self.head(h)


class VanillaTransformerModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, nhead=4, num_layers=2, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.embed = nn.Linear(C, hidden)
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        out = self.tr(self.embed(x))
        return self.head(out[:, -1, :])


class TimeSeriesTransformerModel(nn.Module):
    def __init__(self, C, seq_len=24, hidden=64, nhead=4, num_layers=2, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.embed = nn.Linear(C, hidden)
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        out = self.tr(self.embed(x) + self.pos)
        return self.head(out[:, -1, :])


class InformerStyleModel(nn.Module):
    """Standard attention with a 1D-conv distillation (stride-2 pooling) between layers."""

    def __init__(self, C, seq_len=24, hidden=64, nhead=4, n_horizons=3, n_quantiles=3):
        super().__init__()
        self.embed = nn.Linear(C, hidden)
        self.layer1 = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden, batch_first=True)
        self.pool = nn.Conv1d(hidden, hidden, 3, stride=2, padding=1)
        self.layer2 = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=4 * hidden, batch_first=True)
        self.head = MultiHorizonHead(hidden, C, n_horizons, n_quantiles)

    def forward(self, x):
        x = self.layer1(self.embed(x))
        x = F.relu(self.pool(x.transpose(1, 2))).transpose(1, 2)
        x = self.layer2(x)
        return self.head(x[:, -1, :])


BASELINES = {
    "GRU": GRUModel,
    "BiLSTM": BiLSTMModel,
    "FFN": FFNModel,
    "1D_CNN": CNN1DModel,
    "1D_ResNet": ResNet1DModel,
    "TCN": TCNModel,
    "TSMixer": TSMixerModel,
    "Vanilla_Transformer": VanillaTransformerModel,
    "Time-Series_Transformer": TimeSeriesTransformerModel,
    "Informer-style": InformerStyleModel,
}
