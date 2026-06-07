"""Invariants for the forecasting rebuild — especially the failures that sank the old notebook.

Covers: window/target leakage, observed-only masked loss, pinball correctness, the GAMMA
`spatial_bias` actually being LIVE (the old one was a dead parameter), DM sanity, Holm
monotonicity, naive-floor correctness, and de-scale round-trip.
"""
import numpy as np
import pandas as pd
import torch

from src.modeling.channels import CHANNELS
from src.modeling.windows import PanelArrays, StationWindowDataset
from src.modeling.losses import MaskedQuantileLoss, median_index
from src.modeling.models.gamma import GAMMA
from src.modeling.models.naive import NaiveForecasters
from src.modeling.metrics import descale
from src.modeling.evaluate import diebold_mariano, holm_bonferroni
from src.splits import StandardScalerFrame


def _toy_panel(T=60, S=3, C=len(CHANNELS), seed=0):
    rng = np.random.default_rng(seed)
    val = rng.standard_normal((T, S, C)).astype(np.float32)
    # punch some holes so masking is exercised
    holes = rng.random((T, S, C)) < 0.2
    val[holes] = np.nan
    m = (~np.isnan(val)).astype(np.float32)
    x = np.where(m > 0, val, 0.0).astype(np.float32)
    d = np.zeros((T, S, C), dtype=np.float32)
    present = (m.sum(axis=2) > 0)
    ti = pd.date_range("2024-01-01", periods=T, freq="h")
    return PanelArrays(ti, [f"S{i}" for i in range(S)], list(CHANNELS), x, m, d, present, val)


def test_window_targets_are_future_and_match_array():
    arr = _toy_panel()
    seq, hz = 8, [1, 6, 24]
    anc = np.array([30], dtype=np.int64)
    ds = StationWindowDataset(arr, anc, seq, hz)
    item = ds[0]
    t, s = int(ds.t[0]), int(ds.s[0])
    # the input window is strictly the past [t-seq+1 .. t]
    assert torch.allclose(item["x"], torch.from_numpy(arr.x[t - seq + 1:t + 1, s, :]))
    # every target is the FUTURE value at t+h (no leakage from inside the window)
    for h in hz:
        expected = np.nan_to_num(arr.val_scaled[t + h, s, :], nan=0.0)
        assert np.allclose(item[f"y_t{h}"].numpy(), expected)
        assert t + h > t


def test_masked_quantile_loss_ignores_unobserved():
    crit = MaskedQuantileLoss([0.5])
    preds = torch.randn(4, len(CHANNELS), 1)
    target = torch.randn(4, len(CHANNELS))
    mask = torch.ones(4, len(CHANNELS))
    full = crit(preds, target, mask)
    # corrupt only masked-out cells -> loss must not change
    mask2 = mask.clone(); mask2[:, 0] = 0
    t2 = target.clone(); t2[:, 0] += 999.0
    base = crit(preds, target, mask2)
    corrupt = crit(preds, t2, mask2)
    assert torch.allclose(base, corrupt), "masked cells leaked into the loss"
    assert not torch.allclose(full, base)


def test_pinball_at_median_equals_half_mae():
    crit = MaskedQuantileLoss([0.5])
    preds = torch.zeros(5, 3, 1)
    target = torch.tensor([[1.0, -2.0, 3.0]]).repeat(5, 1)
    mask = torch.ones(5, 3)
    loss = crit(preds, target, mask).item()
    assert abs(loss - 0.5 * target.abs().mean().item()) < 1e-6


def test_gamma_spatial_bias_is_live():
    """The old SpatialAttention never used spatial_bias. The new one must: it gets gradient."""
    torch.manual_seed(0)
    B, S, L, C = 2, 4, 8, len(CHANNELS)
    model = GAMMA(C, S, d_model=32, n_heads=4, seq_len=L, n_horizons=1, n_quantiles=1)
    x = torch.randn(B, S, L, C); mask = torch.ones(B, S, L, C); decay = torch.zeros(B, S, L, C)
    present = torch.ones(B, S); st = torch.arange(S).unsqueeze(0).repeat(B, 1)
    out = model(x, mask, decay, present, st)
    out.sum().backward()
    assert model.spatial_bias.grad is not None
    assert model.spatial_bias.grad.abs().sum().item() > 0.0


def test_gamma_spatial_ablation_changes_output():
    torch.manual_seed(0)
    B, S, L, C = 2, 4, 8, len(CHANNELS)
    kw = dict(d_model=32, n_heads=4, seq_len=L, n_horizons=1, n_quantiles=1)
    full = GAMMA(C, S, **kw)
    abl = GAMMA(C, S, use_spatial=False, **kw)
    abl.load_state_dict(full.state_dict(), strict=False)
    x = torch.randn(B, S, L, C); mask = torch.ones(B, S, L, C); decay = torch.zeros(B, S, L, C)
    present = torch.ones(B, S); st = torch.arange(S).unsqueeze(0).repeat(B, 1)
    of = full(x, mask, decay, present, st)
    oa = abl(x, mask, decay, present, st)
    assert not torch.allclose(of, oa), "removing the spatial axis changed nothing"


def test_dm_identical_models_is_insignificant():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(500)
    dm, p, n = diebold_mariano(e.copy(), e.copy(), h=1)
    assert np.isnan(dm) or abs(dm) < 1e-6
    # a clearly worse model 2 -> model1 better -> negative DM, small p
    e2 = e * 3.0
    dm2, p2, _ = diebold_mariano(e, e2, h=1)
    assert dm2 < 0 and p2 < 0.05


def test_holm_is_monotone_and_bounded():
    res = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.5})
    assert res["a"]["p_adj"] <= res["b"]["p_adj"] <= res["c"]["p_adj"]
    assert all(0.0 <= v["p_adj"] <= 1.0 for v in res.values())


def test_naive_persistence_equals_last_observed():
    arr = _toy_panel(seed=3)
    nf = NaiveForecasters(arr)
    t = np.array([40]); s = np.array([0])
    pred = nf.predict(arr, t, s, horizon=1, method="persistence")[0]
    # last observed value at/<=t for station 0
    ff = arr._ffill_cache
    assert np.allclose(pred, np.nan_to_num(ff[40, 0, :], nan=0.0))


def test_descale_roundtrip():
    sc = StandardScalerFrame(cols=list(CHANNELS))
    df = pd.DataFrame({c: np.random.randn(100) * (i + 1) + i for i, c in enumerate(CHANNELS)})
    df["station"] = "S0"
    sc.fit(df)
    scaled = ((df[CHANNELS] - [sc.global_[c][0] for c in CHANNELS])
              / [sc.global_[c][1] for c in CHANNELS]).to_numpy()
    back = descale(scaled, sc)
    assert np.allclose(back, df[CHANNELS].to_numpy(), atol=1e-6)
