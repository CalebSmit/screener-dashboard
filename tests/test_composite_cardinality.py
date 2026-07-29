#!/usr/bin/env python3
"""Tests for Phase 13 F1: composite cardinality preservation.

The final Composite must be the CARDINAL weighted-average of category scores
(the ranking key), NOT a percentile rank. The percentile is exposed separately
as Composite_Pct for display.
"""
import numpy as np
import pandas as pd
import pytest

from factor_engine import compute_composite, rank_stocks


def _make_scored(n=50, seed=3):
    rng = np.random.default_rng(seed)
    cats = ["valuation", "quality", "growth", "momentum",
            "risk", "revisions", "size", "investment"]
    data = {"Ticker": [f"T{i:03d}" for i in range(n)],
            "Sector": ["Tech"] * n}
    for c in cats:
        data[f"{c}_score"] = rng.uniform(5, 95, n)
    return pd.DataFrame(data)


CFG = {
    "factor_weights": {
        "valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
        "risk": 10, "revisions": 10, "size": 5, "investment": 5,
    },
    "data_quality": {"coverage_discount": {"enabled": False}},
    "sector_neutral": {"sector_relative_composite": False},
}


def test_composite_is_cardinal_not_uniform_rank():
    df = compute_composite(_make_scored(), CFG)
    comp = df["Composite"].sort_values(ascending=False).to_numpy()
    # A percentile-rank ladder would have (nearly) constant spacing == 100/N.
    diffs = np.diff(comp[:10])
    # Cardinal spacing is NOT constant: its std should be clearly nonzero.
    assert np.std(diffs) > 1e-3, "Composite still looks like a uniform rank ladder"


def test_composite_pct_column_exists_and_is_percentile():
    df = compute_composite(_make_scored(), CFG)
    assert "Composite_Pct" in df.columns
    # Percentile column should span ~ (0, 100] with the top near 100.
    assert df["Composite_Pct"].max() <= 100.0 + 1e-6
    assert df["Composite_Pct"].max() > 90.0


def test_composite_within_category_score_range():
    # Cardinal composite is a weighted average of 0-100 category scores, so it
    # must itself lie within [0, 100].
    df = compute_composite(_make_scored(), CFG)
    assert df["Composite"].min() >= 0.0
    assert df["Composite"].max() <= 100.0


def test_ranking_matches_cardinal_order():
    df = compute_composite(_make_scored(), CFG)
    df = rank_stocks(df)
    # Rank 1 must be the max cardinal composite.
    top = df.sort_values("Rank").iloc[0]
    assert abs(top["Composite"] - df["Composite"].max()) < 1e-6


def test_pct_and_cardinal_are_rank_consistent():
    df = compute_composite(_make_scored(), CFG)
    # Higher cardinal composite <=> higher percentile (monotone).
    s = df.sort_values("Composite")
    assert s["Composite_Pct"].is_monotonic_increasing
