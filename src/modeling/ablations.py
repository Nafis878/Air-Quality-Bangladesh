"""GAMMA component ablations: remove one axis/mechanism at a time and measure the test delta.

Each variant is a full GAMMA with one switch off, so the contribution of each piece is the change
in test metric relative to the full model. Most keys are GAMMA constructor switches; the special
`_use_anchor` key is a HARNESS flag (the persistence anchor is applied outside the net, uniformly to
all models) — `split_ablation` separates it so run.py can route it correctly.

The two ablations that carry the capability argument:
  * GAMMA_no_spatial        -> no neighbours at all; collapses to a per-station model. If the
                               stale-bin advantage disappears here, the win IS the cross-station graph.
  * GAMMA_no_staleness_gate -> graph still present, but the gate no longer sees self-staleness, so it
                               cannot deliberately route to neighbours when the station's own data is stale.
"""
from __future__ import annotations

ABLATIONS = {
    "GAMMA_no_variable":       {"use_variable": False},
    "GAMMA_no_spatial":        {"use_spatial": False},        # no cross-station reach
    "GAMMA_no_temporal":       {"use_temporal": False},
    "GAMMA_no_gate":           {"use_gate": False},            # mean fusion instead of learned gate
    "GAMMA_no_decay":          {"use_decay": False},           # GRU-D decay gate removed
    "GAMMA_no_staleness_gate": {"use_staleness_gate": False},  # gate blind to self-staleness
    "GAMMA_one_hop":           {"spatial_layers": 1},          # single-hop spatial message passing
    "GAMMA_no_anchor":         {"_use_anchor": False},         # drop the shared persistence residual
}


def split_ablation(kw: dict) -> tuple[dict, bool]:
    """Separate model constructor kwargs from the harness `_use_anchor` flag.

    Returns (gamma_kwargs, use_anchor).
    """
    gamma_kwargs = {k: v for k, v in kw.items() if not k.startswith("_")}
    use_anchor = bool(kw.get("_use_anchor", True))
    return gamma_kwargs, use_anchor
