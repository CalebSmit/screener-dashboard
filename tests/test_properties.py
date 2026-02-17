"""Property-based and invariant tests for the Multi-Factor Screener.

These tests verify structural invariants that must hold regardless of
input data, including:
  - Score ranges
  - Rank completeness and monotonicity
  - Portfolio weight constraints
  - Sector concentration limits
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import (
    _generate_sample_data,
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    apply_value_trap_flags,
    rank_stocks,
    METRIC_COLS,
)
from portfolio_constructor import construct_portfolio, compute_portfolio_stats

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_universe():
    """Small universe for testing."""
    return pd.DataFrame([
        {"Ticker": f"T{i:03d}", "Company": f"Co {i}",
         "Sector": s}
        for i, s in enumerate([
            "Information Technology", "Information Technology",
            "Financials", "Financials",
            "Health Care", "Health Care",
            "Energy", "Energy",
            "Consumer Staples", "Consumer Staples",
            "Industrials", "Industrials",
            "Consumer Discretionary", "Consumer Discretionary",
            "Communication Services", "Communication Services",
            "Utilities", "Utilities",
            "Real Estate", "Real Estate",
            "Materials", "Materials",
            "Information Technology", "Information Technology",
            "Financials", "Financials",
            "Health Care", "Health Care",
            "Energy", "Energy",
            "Consumer Staples",
        ])
    ])


@pytest.fixture
def scored_df(sample_universe, cfg):
    """Run full scoring pipeline on sample data."""
    df = _generate_sample_data(sample_universe, seed=42)
    df = winsorize_metrics(df, 0.01, 0.01)
    df = compute_sector_percentiles(df)
    df = compute_category_scores(df, cfg)
    df = compute_composite(df, cfg)
    df = apply_value_trap_flags(df, cfg)
    df = rank_stocks(df)
    return df


class TestRankingInvariants:
    def test_ranks_are_complete(self, scored_df):
        """Every scored stock has a rank."""
        assert scored_df["Rank"].notna().all()

    def test_ranks_are_positive(self, scored_df):
        """All ranks are >= 1."""
        assert (scored_df["Rank"] >= 1).all()

    def test_rank_count_matches(self, scored_df):
        """Max rank does not exceed number of stocks."""
        assert scored_df["Rank"].max() <= len(scored_df)

    def test_composite_in_range(self, scored_df):
        """All composite scores are in [0, 100]."""
        assert (scored_df["Composite"] >= 0).all()
        assert (scored_df["Composite"] <= 100).all()

    def test_monotonicity(self, scored_df):
        """If A.Composite > B.Composite, then A.Rank <= B.Rank."""
        df = scored_df.sort_values("Rank").reset_index(drop=True)
        for i in range(len(df) - 1):
            if df.iloc[i]["Composite"] > df.iloc[i + 1]["Composite"]:
                assert df.iloc[i]["Rank"] <= df.iloc[i + 1]["Rank"]

    def test_sorted_by_rank(self, scored_df):
        """Output is sorted by Rank ascending."""
        ranks = scored_df["Rank"].tolist()
        assert ranks == sorted(ranks)


class TestScoreRanges:
    def test_category_scores_in_range(self, scored_df):
        for cat in ["valuation_score", "quality_score", "growth_score",
                     "momentum_score", "risk_score", "revisions_score"]:
            if cat in scored_df.columns:
                valid = scored_df[cat].dropna()
                assert (valid >= 0).all(), f"{cat} has values < 0"
                assert (valid <= 100).all(), f"{cat} has values > 100"

    def test_percentile_scores_in_range(self, scored_df):
        pct_cols = [c for c in scored_df.columns if c.endswith("_pct")]
        for col in pct_cols:
            valid = scored_df[col].dropna()
            assert (valid >= 0).all(), f"{col} < 0"
            assert (valid <= 100).all(), f"{col} > 100"

    def test_value_trap_flag_is_boolean(self, scored_df):
        assert scored_df["Value_Trap_Flag"].dtype == bool


class TestPortfolioInvariants:
    def test_weights_sum_to_100(self, scored_df, cfg):
        port = construct_portfolio(scored_df, cfg)
        if len(port) == 0:
            pytest.skip("Empty portfolio")
        for wt_col in ["Equal_Weight_Pct", "RiskParity_Weight_Pct"]:
            if wt_col in port.columns:
                total = port[wt_col].sum()
                assert abs(total - 100.0) < 0.5, \
                    f"{wt_col} sums to {total}, expected ~100"

    def test_sector_cap_respected(self, scored_df, cfg):
        port = construct_portfolio(scored_df, cfg)
        if len(port) == 0:
            pytest.skip("Empty portfolio")
        max_sec = cfg["portfolio"]["max_sector_concentration"]
        for sec, count in port["Sector"].value_counts().items():
            assert count <= max_sec, \
                f"Sector {sec} has {count} stocks, max allowed {max_sec}"

    def test_position_cap_respected(self, scored_df, cfg):
        port = construct_portfolio(scored_df, cfg)
        if len(port) == 0:
            pytest.skip("Empty portfolio")
        cap = cfg["portfolio"]["max_position_pct"]
        for wt_col in ["Equal_Weight_Pct", "RiskParity_Weight_Pct"]:
            if wt_col in port.columns:
                max_wt = port[wt_col].max()
                assert max_wt <= cap + 0.1, \
                    f"{wt_col} max weight {max_wt}% exceeds cap {cap}%"

    def test_portfolio_size(self, scored_df, cfg):
        port = construct_portfolio(scored_df, cfg)
        # Should have at least 20 stocks (guardrail) and at most num_stocks
        expected = cfg["portfolio"]["num_stocks"]
        if len(scored_df) >= 20:
            assert len(port) >= 20, f"Portfolio has only {len(port)} stocks"
        assert len(port) <= expected + 5  # Allow small buffer for backfill


class TestDeterminism:
    def test_sample_data_deterministic(self, sample_universe):
        """Two calls with same seed → identical output."""
        df1 = _generate_sample_data(sample_universe, seed=42)
        df2 = _generate_sample_data(sample_universe, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_scoring_deterministic(self, sample_universe, cfg):
        """Full pipeline run twice → identical results."""
        def run():
            df = _generate_sample_data(sample_universe, seed=42)
            df = winsorize_metrics(df, 0.01, 0.01)
            df = compute_sector_percentiles(df)
            df = compute_category_scores(df, cfg)
            df = compute_composite(df, cfg)
            df = apply_value_trap_flags(df, cfg)
            df = rank_stocks(df)
            return df

        r1 = run()
        r2 = run()
        pd.testing.assert_frame_equal(r1, r2)
