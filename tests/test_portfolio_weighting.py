"""Tests for portfolio construction and weighting schemes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from portfolio_constructor import construct_portfolio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_cfg(weighting="equal", num_stocks=10, max_sector=5):
    return {
        "portfolio": {
            "num_stocks": num_stocks,
            "max_sector_concentration": max_sector,
            "max_position_pct": 20.0,
            "min_avg_dollar_volume": 0,
            "weighting": weighting,
        },
        "value_trap_filters": {"flag_only": True},
        "growth_trap_filters": {"flag_only": True},
    }


def _make_universe(n=30, seed=42):
    """Generate a small scored universe DataFrame."""
    rng = np.random.default_rng(seed)
    sectors = ["Information Technology", "Financials", "Health Care",
               "Consumer Discretionary", "Industrials"]
    return pd.DataFrame({
        "Ticker": [f"T{i:03d}" for i in range(n)],
        "Company": [f"Company {i}" for i in range(n)],
        "Sector": [sectors[i % len(sectors)] for i in range(n)],
        "Composite": rng.uniform(40, 95, size=n),
        "Rank": list(range(1, n + 1)),
        "volatility": rng.uniform(0.10, 0.45, size=n),
        "avg_daily_dollar_volume": rng.uniform(50e6, 500e6, size=n),
        "Value_Trap_Flag": [False] * n,
        "Growth_Trap_Flag": [False] * n,
        "_metric_coverage": [0.9] * n,
        # Minimal metric cols so coverage filter passes
        "ev_ebitda": rng.uniform(5, 30, size=n),
        "roic": rng.uniform(0.05, 0.30, size=n),
        "revenue_growth": rng.uniform(-0.05, 0.25, size=n),
        "return_12_1": rng.uniform(-0.20, 0.40, size=n),
        "beta": rng.uniform(0.5, 1.8, size=n),
    })


# ---------------------------------------------------------------------------
# Portfolio construction basics
# ---------------------------------------------------------------------------

def test_portfolio_not_empty():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg())
    assert len(port) > 0


def test_portfolio_respects_num_stocks():
    df = _make_universe(n=30)
    port = construct_portfolio(df, _base_cfg(num_stocks=10))
    assert len(port) <= 10


def test_portfolio_has_port_rank():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg())
    assert "Port_Rank" in port.columns
    assert port["Port_Rank"].min() == 1


# ---------------------------------------------------------------------------
# Equal weighting
# ---------------------------------------------------------------------------

def test_equal_weights_sum_to_100():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg(weighting="equal"))
    total = port["Equal_Weight_Pct"].sum()
    assert abs(total - 100.0) < 0.05, f"Equal weights sum to {total:.4f}, expected 100"


def test_equal_weights_are_uniform():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg(weighting="equal", num_stocks=10))
    weights = port["Equal_Weight_Pct"]
    # All weights should be within 0.01% of the expected 1/n weight
    expected = 100.0 / len(port)
    assert (weights - expected).abs().max() < 0.02


# ---------------------------------------------------------------------------
# Inverse-volatility weighting
# ---------------------------------------------------------------------------

def test_inv_vol_weights_present():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg(weighting="inverse_vol"))
    assert "InvVol_Weight_Pct" in port.columns


def test_inv_vol_weights_sum_to_100():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg(weighting="inverse_vol"))
    total = port["InvVol_Weight_Pct"].sum()
    assert abs(total - 100.0) < 0.1, f"InvVol weights sum to {total:.4f}, expected 100"


def test_inv_vol_lower_vol_higher_weight():
    """Stocks with lower volatility should receive higher inv-vol weight."""
    df = _make_universe(n=30, seed=7)
    port = construct_portfolio(df, _base_cfg(weighting="inverse_vol", num_stocks=20))
    # Correlation between vol and inv-vol weight should be negative
    corr = port[["volatility", "InvVol_Weight_Pct"]].dropna().corr().iloc[0, 1]
    assert corr < 0, f"Expected negative correlation vol vs inv-vol weight, got {corr:.3f}"


def test_inv_vol_weights_positive():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg(weighting="inverse_vol"))
    assert (port["InvVol_Weight_Pct"] > 0).all()


# ---------------------------------------------------------------------------
# Score weighting
# ---------------------------------------------------------------------------

def test_score_weights_sum_to_100():
    df = _make_universe()
    port = construct_portfolio(df, _base_cfg(weighting="score"))
    total = port["Score_Weight_Pct"].sum()
    assert abs(total - 100.0) < 0.1, f"Score weights sum to {total:.4f}, expected 100"


def test_score_weights_higher_composite_higher_weight():
    """Higher composite score should correlate positively with score weight."""
    df = _make_universe(n=30, seed=99)
    port = construct_portfolio(df, _base_cfg(weighting="score", num_stocks=20))
    corr = port[["Composite", "Score_Weight_Pct"]].dropna().corr().iloc[0, 1]
    assert corr > 0, f"Expected positive correlation composite vs score weight, got {corr:.3f}"


# ---------------------------------------------------------------------------
# Position cap enforcement
# ---------------------------------------------------------------------------

def test_max_position_cap_respected():
    df = _make_universe(n=30)
    # Tight cap — use fewer stocks so equal weight stays below the cap.
    # With num_stocks=13 the natural equal weight is ~7.7% < 8%.
    cfg = _base_cfg(num_stocks=13)
    cfg["portfolio"]["max_position_pct"] = 8.0
    port = construct_portfolio(df, cfg)
    for col in ["Equal_Weight_Pct", "InvVol_Weight_Pct", "Score_Weight_Pct"]:
        if col not in port.columns:
            continue
        # Skip equal-weight check when cap < 100/n (mathematically impossible
        # to cap equal weights below the natural 1/n level).
        if col == "Equal_Weight_Pct" and len(port) > 0:
            natural_ew = 100.0 / len(port)
            if natural_ew > cfg["portfolio"]["max_position_pct"]:
                continue
        assert (port[col] <= 8.05).all(), f"{col} exceeds cap"


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------

def test_sector_concentration_respected():
    df = _make_universe(n=30)
    cfg = _base_cfg(num_stocks=25, max_sector=4)
    port = construct_portfolio(df, cfg)
    sector_counts = port["Sector"].value_counts()
    assert sector_counts.max() <= 4, f"Sector concentration exceeded: {sector_counts.to_dict()}"
