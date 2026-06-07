"""Forecasting models + leakage-safe training/evaluation for the air-quality panel.

This package is additive: it consumes the cleaned panel produced by the data pipeline
(`data/processed/air_quality_clean.parquet`) and never mutates it. Everything here obeys
the same leakage-safe contract as `src/splits.py`: temporal splits, train-only statistics,
observed-target-only scoring.
"""
