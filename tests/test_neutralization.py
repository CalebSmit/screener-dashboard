#!/usr/bin/env python3
"""Tests for Phase 13 F4: optional factor neutralization (OFF by default)."""
import numpy as np
import pandas as pd

from factor_engine import (
    compute_composite, neutralize_category_scores,
    compute_effective_dimensionality,
)

CATS = ["valuation", "quality", "growth", "momentum",
        "risk", "revisions", "size", "investment"]


def _scored(n=60, seed=7):
    rng = np.random.default_rng(seed)
    base = rng.uniform(5, 95, n)
    data = {"Ticker": [f"T{i}" for i in range(n)], "Sector": ["Tech"] * n}
    # Make valuation & quality strongly correlated to exercise neutralization.
    data["valuation_score"] = base
    data["quality_score"] = np.clip(base + rng.normal(0, 5, n), 0, 100)
    for c in CATS[2:]:
        data[f"{c}_score"] = rng.uniform(5, 95, n)
    return pd.DataFrame(data)


BASE_CFG = {
    "factor_weights": {c: w for c, w in zip(
        CATS, [22, 22, 13, 13, 10, 10, 5, 5])},
    "data_quality": {"coverage_discount": {"enabled": False}},
    "sector_neutral": {"sector_relative_composite": False},
}


def test_disabled_by_default_is_noop():
    df = _scored()
    cfg = {**BASE_CFG, "factor_neutralization": {"enabled": False}}
    before = df["quality_score"].copy()
    out = neutralize_category_scores(df.copy(), cfg)
    pd.testing.assert_series_equal(out["quality_score"], before)


def test_enabled_changes_correlated_scores():
    df = _scored()
    cfg = {**BASE_CFG, "factor_neutralization": {
        "enabled": True, "method": "gram_schmidt", "order": CATS}}
    out = neutralize_category_scores(df.copy(), cfg)
    # quality is residualized against valuation -> its values must change
    assert not np.allclose(out["quality_score"].to_numpy(),
                           df["quality_score"].to_numpy())
    # valuation is first in order -> unchanged
    assert np.allclose(out["valuation_score"].to_numpy(),
                       df["valuation_score"].to_numpy())


def test_effective_dimensionality_detects_redundancy():
    df = _scored()
    eff = compute_effective_dimensionality(df, BASE_CFG)
    # 8 categories but valuation~quality correlated -> effective < 8
    assert 1.0 <= eff <= 8.0
    assert eff < 8.0


def test_composite_unchanged_when_neutralization_off():
    df = _scored()
    cfg_off = {**BASE_CFG, "factor_neutralization": {"enabled": False}}
    c1 = compute_composite(df.copy(), cfg_off)["Composite"].to_numpy()
    c2 = compute_composite(df.copy(), {**BASE_CFG})["Composite"].to_numpy()  # no key = off
    assert np.allclose(c1, c2, equal_nan=True)
