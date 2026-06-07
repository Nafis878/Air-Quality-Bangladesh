"""GAMMA component ablations: remove one axis/mechanism at a time and measure the test delta.

Each variant is a full GAMMA with one switch off, so the contribution of each piece is the
change in test metric relative to the full model. Names match the constructor switches.
"""
from __future__ import annotations

ABLATIONS = {
    "GAMMA_no_variable": {"use_variable": False},
    "GAMMA_no_spatial":  {"use_spatial": False},   # spatial_bias path removed
    "GAMMA_no_temporal": {"use_temporal": False},
    "GAMMA_no_gate":     {"use_gate": False},       # mean fusion instead of learned gate
    "GAMMA_no_decay":    {"use_decay": False},       # GRU-D decay bias removed
}
