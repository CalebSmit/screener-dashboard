"""Tests for Composite_Confidence column computation in compute_composite()."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factor_engine import compute_composite

# A representative subset of METRIC_COLS used to test coverage ratio
_SAMPLE_METRICS = [
    "ev_ebitda", "fcf_yield", "earnings_yield", "roic", "debt_equity",
    "revenue_growth", "return_12_1", "volatility", "beta", "size_log_mcap",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored_df(n=20, seed=42, include_metrics=False):
    """Build a minimal scored DataFrame with category score columns.

    Args:
        include_metrics: if True, add sample metric columns so that
                         Composite_Confidence coverage_ratio is non-trivial.
    """
    rng = np.random.default_rng(seed)
    categories = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
    data = {"Ticker": [f"T{i:03d}" for i in range(n)]}
    for cat in categories:
        data[f"{cat}_score"] = rng.uniform(0, 100, size=n)
    if include_metrics:
        for col in _SAMPLE_METRICS:
            data[col] = rng.uniform(0.5, 50.0, size=n)
    return pd.DataFrame(data)


def _default_weights():
    return {
        "factor_weights": {
            "valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
            "risk": 10, "revisions": 10, "size": 5, "investment": 5,
        },
        "data_quality": {},
        "sector_neutral": {},
    }


# ---------------------------------------------------------------------------
# Composite_Confidence presence and range
# ---------------------------------------------------------------------------

def test_composite_confidence_column_present():
    df = _make_scored_df()
    result = compute_composite(df, _default_weights())
    assert "Composite_Confidence" in result.columns


def test_composite_confidence_range():
    df = _make_scored_df()
    result = compute_composite(df, _default_weights())
    conf = result["Composite_Confidence"].dropna()
    assert (conf >= 0).all(), "Confidence values below 0 found"
    assert (conf <= 100).all(), "Confidence values above 100 found"


def test_composite_confidence_higher_with_full_coverage():
    """Tickers with all metrics populated should have higher confidence."""
    df = _make_scored_df(n=30, include_metrics=True)
    sparse_df = df.copy()
    # Rows 0-9: most metrics missing — drives coverage_ratio down
    for col in _SAMPLE_METRICS[1:]:
        sparse_df.loc[:9, col] = np.nan

    result = compute_composite(sparse_df, _default_weights())
    mean_sparse = result.loc[:9, "Composite_Confidence"].mean()
    mean_full = result.loc[10:, "Composite_Confidence"].mean()
    assert mean_full > mean_sparse, (
        f"Full-coverage rows should have higher confidence than sparse rows: "
        f"full={mean_full:.1f} sparse={mean_sparse:.1f}"
    )


def test_composite_confidence_type():
    df = _make_scored_df()
    result = compute_composite(df, _default_weights())
    assert result["Composite_Confidence"].dtype in (np.float64, np.float32, float)


# ---------------------------------------------------------------------------
# Composite column itself
# ---------------------------------------------------------------------------

def test_composite_column_present():
    df = _make_scored_df()
    result = compute_composite(df, _default_weights())
    assert "Composite" in result.columns


def test_composite_values_range():
    df = _make_scored_df()
    result = compute_composite(df, _default_weights())
    comp = result["Composite"].dropna()
    assert len(comp) > 0
    assert (comp >= 0).all()
    assert (comp <= 100).all()
