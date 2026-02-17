"""Tests for the scoring pipeline: winsorization, percentile ranking,
category scores, composite scoring, and ranking.

All tests use synthetic data — no network access required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import (
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    apply_value_trap_flags,
    rank_stocks,
    METRIC_COLS,
    METRIC_DIR,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _make_df(n=50, seed=42):
    """Create a synthetic DataFrame with all metric columns and sectors."""
    rng = np.random.default_rng(seed)
    sectors = ["Information Technology", "Financials", "Health Care",
               "Energy", "Consumer Staples"]
    records = []
    for i in range(n):
        rec = {
            "Ticker": f"T{i:03d}",
            "Company": f"Company {i}",
            "Sector": sectors[i % len(sectors)],
        }
        for col in METRIC_COLS:
            if col in ("eps_revision_ratio", "eps_estimate_change"):
                rec[col] = np.nan  # Always NaN (placeholder)
            elif col == "analyst_surprise":
                rec[col] = rng.normal(0.03, 0.05) if rng.random() > 0.5 else np.nan
            elif col == "debt_equity":
                rec[col] = max(0, rng.normal(1.0, 0.5))
            elif col == "piotroski_f_score":
                rec[col] = int(rng.integers(2, 9))
            elif col == "beta":
                rec[col] = rng.normal(1.0, 0.3)
            elif col == "volatility":
                rec[col] = max(0.05, rng.normal(0.25, 0.08))
            elif METRIC_DIR.get(col, True):
                # "Higher is better" → positive values
                rec[col] = rng.normal(0.1, 0.05)
            else:
                # "Lower is better" → positive values (e.g., EV/EBITDA)
                rec[col] = max(1, rng.normal(15, 5))
            # Introduce some NaN randomly
            if rng.random() < 0.05:
                rec[col] = np.nan
        records.append(rec)
    return pd.DataFrame(records)


# =====================================================================
# WINSORIZATION
# =====================================================================

class TestWinsorization:
    def test_extreme_values_clipped(self):
        """Create a controlled dataset where one value is clearly extreme."""
        n = 100
        # All values tightly clustered around 15, with one extreme outlier
        vals = [15.0 + i * 0.1 for i in range(n)]
        vals[0] = 9999.0  # Plant extreme outlier
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(n)],
            "ev_ebitda": vals,
        })
        # Add other metric cols as NaN so winsorize_metrics doesn't fail
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = winsorize_metrics(df, 0.01, 0.01)
        # After winsorization, the extreme value should be reduced
        assert df.loc[0, "ev_ebitda"] < 9999.0

    def test_small_sample_no_winsorize(self):
        """With < 10 non-NaN values, winsorization should be skipped."""
        df = pd.DataFrame({
            "Ticker": [f"T{i}" for i in range(5)],
            "ev_ebitda": [5.0, 10.0, 15.0, 20.0, 100.0],
        })
        df_result = winsorize_metrics(df.copy(), 0.01, 0.01)
        # Should be unchanged (< 10 values)
        assert df_result.loc[4, "ev_ebitda"] == 100.0

    def test_nan_preserved(self):
        df = _make_df(50)
        nan_count_before = df["ev_ebitda"].isna().sum()
        df = winsorize_metrics(df, 0.01, 0.01)
        nan_count_after = df["ev_ebitda"].isna().sum()
        assert nan_count_before == nan_count_after


# =====================================================================
# SECTOR PERCENTILES
# =====================================================================

class TestSectorPercentiles:
    def test_output_range(self):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        for col in METRIC_COLS:
            pc = f"{col}_pct"
            if pc in df.columns:
                valid = df[pc].dropna()
                assert (valid >= 0).all(), f"{pc} has values < 0"
                assert (valid <= 100).all(), f"{pc} has values > 100"

    def test_higher_is_better_direction(self):
        """For 'higher is better' metrics, the highest value should get ~100."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D", "E"],
            "Sector": ["Tech"] * 5,
            "roic": [0.01, 0.05, 0.10, 0.20, 0.30],
        })
        # Add other required columns as NaN
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        # ROIC is "higher is better" → highest value should get highest pct
        assert df.loc[4, "roic_pct"] > df.loc[0, "roic_pct"]

    def test_lower_is_better_direction(self):
        """For 'lower is better' metrics, the lowest value should get ~100."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D", "E"],
            "Sector": ["Tech"] * 5,
            "ev_ebitda": [5.0, 10.0, 15.0, 20.0, 30.0],
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        # EV/EBITDA is "lower is better" → lowest value should get highest pct
        assert df.loc[0, "ev_ebitda_pct"] > df.loc[4, "ev_ebitda_pct"]

    def test_nan_gets_50(self):
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D", "E"],
            "Sector": ["Tech"] * 5,
            "roic": [0.10, 0.20, np.nan, 0.15, 0.25],
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        assert df.loc[2, "roic_pct"] == 50.0

    def test_small_sector_all_50(self):
        """Sector with < 3 valid values → all get 50."""
        df = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Sector": ["Tiny"] * 2,
            "roic": [0.10, 0.20],
        })
        for col in METRIC_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = compute_sector_percentiles(df)
        assert df.loc[0, "roic_pct"] == 50.0
        assert df.loc[1, "roic_pct"] == 50.0


# =====================================================================
# CATEGORY SCORES
# =====================================================================

class TestCategoryScores:
    def test_scores_computed(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        for cat in ["valuation_score", "quality_score", "growth_score",
                     "momentum_score", "risk_score", "revisions_score"]:
            assert cat in df.columns
            assert df[cat].notna().any()

    def test_scores_in_range(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        for cat in ["valuation_score", "quality_score", "growth_score",
                     "momentum_score", "risk_score"]:
            valid = df[cat].dropna()
            assert (valid >= 0).all(), f"{cat} has values < 0"
            assert (valid <= 100).all(), f"{cat} has values > 100"


# =====================================================================
# COMPOSITE SCORE
# =====================================================================

class TestComposite:
    def test_min_max_scaling(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        assert df["Composite"].min() >= 0
        assert df["Composite"].max() <= 100
        # After min-max, min should be ~0 and max should be ~100
        assert df["Composite"].min() < 1.0
        assert df["Composite"].max() > 99.0

    def test_all_equal(self, cfg):
        """All same category scores → all composite = 50.0."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Sector": ["Tech"] * 3,
            "valuation_score": [50.0, 50.0, 50.0],
            "quality_score": [50.0, 50.0, 50.0],
            "growth_score": [50.0, 50.0, 50.0],
            "momentum_score": [50.0, 50.0, 50.0],
            "risk_score": [50.0, 50.0, 50.0],
            "revisions_score": [50.0, 50.0, 50.0],
        })
        df = compute_composite(df, cfg)
        assert (df["Composite"] == 50.0).all()

    def test_empty_df(self, cfg):
        df = pd.DataFrame()
        result = compute_composite(df, cfg)
        assert "Composite" in result.columns
        assert len(result) == 0


# =====================================================================
# VALUE TRAP FLAGS
# =====================================================================

class TestValueTrapFlags:
    def test_flags_applied(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = apply_value_trap_flags(df, cfg)
        assert "Value_Trap_Flag" in df.columns
        # Should have some flags and some non-flags
        assert df["Value_Trap_Flag"].any()
        assert not df["Value_Trap_Flag"].all()

    def test_disabled(self, cfg):
        cfg_copy = cfg.copy()
        cfg_copy["value_trap_filters"] = {"enabled": False}
        df = _make_df(20)
        df["quality_score"] = 10.0  # Very low
        df["momentum_score"] = 10.0
        df["revisions_score"] = 10.0
        df = apply_value_trap_flags(df, cfg_copy)
        assert not df["Value_Trap_Flag"].any()


# =====================================================================
# RANKING
# =====================================================================

class TestRanking:
    def test_descending_order(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = rank_stocks(df)
        # Rank 1 should have the highest composite
        assert df.iloc[0]["Rank"] == 1
        assert df.iloc[0]["Composite"] >= df.iloc[1]["Composite"]

    def test_ranks_complete(self, cfg):
        df = _make_df(50)
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = rank_stocks(df)
        assert df["Rank"].notna().all()
        assert (df["Rank"] >= 1).all()
        assert df["Rank"].max() <= len(df)

    def test_tie_handling(self):
        """Equal composites → same rank, next rank skipped."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Composite": [85.0, 85.0, 83.0],
        })
        df = rank_stocks(df)
        assert list(df["Rank"]) == [1, 1, 3]
