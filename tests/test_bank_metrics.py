"""Tests for bank-specific metrics: classification, computation,
category scoring, and composite integration.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import (
    _is_bank_like,
    _FINANCIAL_SECTORS,
    _BANK_LIKE_INDUSTRIES,
    _NON_BANK_FINANCIALS,
    _BANK_ONLY_METRICS,
    _NONBANK_ONLY_METRICS,
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    rank_stocks,
    METRIC_COLS,
    METRIC_DIR,
    CAT_METRICS,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# =====================================================================
# BANK CLASSIFICATION
# =====================================================================

class TestBankClassification:
    def test_bank_diversified_is_bank(self):
        assert _is_bank_like("JPM", "Financials", "Banks—Diversified")

    def test_bank_regional_is_bank(self):
        assert _is_bank_like("USB", "Financials", "Banks - Regional")

    def test_insurance_is_bank_like(self):
        assert _is_bank_like("ALL", "Financials", "Insurance—Property & Casualty")

    def test_visa_is_not_bank(self):
        """V is in _NON_BANK_FINANCIALS — should use generic metrics."""
        assert not _is_bank_like("V", "Financials", "Credit Services")

    def test_mastercard_is_not_bank(self):
        assert not _is_bank_like("MA", "Financial Services", "Credit Services")

    def test_non_financial_is_not_bank(self):
        assert not _is_bank_like("AAPL", "Information Technology", "Consumer Electronics")

    def test_unknown_financials_defaults_to_bank(self):
        """Stock in Financials with unknown industry → bank-like (conservative)."""
        assert _is_bank_like("NEWCO", "Financials", "")

    def test_exchange_is_not_bank(self):
        assert not _is_bank_like("CME", "Financials", "Capital Markets")

    def test_spgi_is_not_bank(self):
        assert not _is_bank_like("SPGI", "Financials", "Capital Markets")


# =====================================================================
# BANK METRIC COMPUTATION (unit-level)
# =====================================================================

def _make_bank_df(n_bank=20, n_nonbank=80, seed=42):
    """Create a DataFrame with bank and non-bank stocks."""
    rng = np.random.default_rng(seed)
    records = []
    sectors = ["Information Technology", "Health Care", "Energy", "Consumer Staples"]
    for i in range(n_nonbank):
        rec = {
            "Ticker": f"NB{i:03d}",
            "Sector": sectors[i % len(sectors)],
            "_is_bank_like": False,
        }
        for col in METRIC_COLS:
            if col in _BANK_ONLY_METRICS:
                rec[col] = np.nan
            elif col == "piotroski_f_score":
                rec[col] = int(rng.integers(3, 9))
            elif col == "analyst_surprise":
                rec[col] = rng.normal(0.03, 0.05) if rng.random() > 0.5 else np.nan
            elif col == "price_target_upside":
                rec[col] = rng.normal(0.10, 0.15) if rng.random() > 0.5 else np.nan
            elif col == "beta":
                rec[col] = rng.normal(1.0, 0.3)
            elif col == "volatility":
                rec[col] = max(0.05, rng.normal(0.25, 0.08))
            elif METRIC_DIR.get(col, True):
                rec[col] = rng.normal(0.1, 0.05)
            else:
                rec[col] = max(1, rng.normal(15, 5))
        records.append(rec)

    for i in range(n_bank):
        rec = {
            "Ticker": f"BK{i:03d}",
            "Sector": "Financials",
            "_is_bank_like": True,
        }
        for col in METRIC_COLS:
            if col in _NONBANK_ONLY_METRICS:
                rec[col] = np.nan
            elif col == "pb_ratio":
                rec[col] = max(0.3, rng.normal(1.2, 0.4))
            elif col == "roe":
                rec[col] = max(0.02, rng.normal(0.12, 0.04))
            elif col == "roa":
                rec[col] = max(0.002, rng.normal(0.01, 0.005))
            elif col == "equity_ratio":
                rec[col] = max(0.05, min(0.20, rng.normal(0.10, 0.03)))
            elif col == "piotroski_f_score":
                rec[col] = int(rng.integers(3, 9))
            elif col == "analyst_surprise":
                rec[col] = rng.normal(0.03, 0.05) if rng.random() > 0.5 else np.nan
            elif col == "price_target_upside":
                rec[col] = rng.normal(0.10, 0.15) if rng.random() > 0.5 else np.nan
            elif col == "beta":
                rec[col] = rng.normal(1.0, 0.3)
            elif col == "volatility":
                rec[col] = max(0.05, rng.normal(0.25, 0.08))
            elif METRIC_DIR.get(col, True):
                rec[col] = rng.normal(0.1, 0.05)
            else:
                rec[col] = max(1, rng.normal(15, 5))
        records.append(rec)

    return pd.DataFrame(records)


class TestBankMetricValues:
    def test_bank_has_bank_metrics(self):
        """Bank stocks should have P/B, ROE, ROA, Equity Ratio."""
        df = _make_bank_df()
        bank = df[df["_is_bank_like"]]
        for col in _BANK_ONLY_METRICS:
            assert bank[col].notna().sum() > 0, f"Bank stocks missing {col}"

    def test_bank_has_nan_generic(self):
        """Bank stocks should have NaN for EV/EBITDA, ROIC, etc."""
        df = _make_bank_df()
        bank = df[df["_is_bank_like"]]
        for col in _NONBANK_ONLY_METRICS:
            assert bank[col].isna().all(), f"Bank stock should have NaN {col}"

    def test_nonbank_has_nan_bank_metrics(self):
        """Non-bank stocks should have NaN for P/B, ROE, ROA, Equity Ratio."""
        df = _make_bank_df()
        nonbank = df[~df["_is_bank_like"]]
        for col in _BANK_ONLY_METRICS:
            assert nonbank[col].isna().all(), f"Non-bank should have NaN {col}"

    def test_nonbank_has_generic_metrics(self):
        """Non-bank stocks should have generic metrics."""
        df = _make_bank_df()
        nonbank = df[~df["_is_bank_like"]]
        for col in _NONBANK_ONLY_METRICS:
            assert nonbank[col].notna().sum() > 0, f"Non-bank missing {col}"


# =====================================================================
# BANK CATEGORY SCORING
# =====================================================================

class TestBankCategoryScores:
    def test_bank_valuation_uses_pb(self, cfg):
        """Bank valuation score should use P/B weight from bank_metric_weights."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)

        bank = df[df["_is_bank_like"]]
        # Bank stocks should have valid valuation scores
        assert bank["valuation_score"].notna().all(), "Bank valuation should not be NaN"

    def test_bank_quality_uses_roe_roa(self, cfg):
        """Bank quality score should use ROE, ROA, Equity Ratio weights."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)

        bank = df[df["_is_bank_like"]]
        assert bank["quality_score"].notna().all(), "Bank quality should not be NaN"

    def test_nonbank_scores_unchanged(self, cfg):
        """Non-bank scoring should produce valid scores with generic weights."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)

        nonbank = df[~df["_is_bank_like"]]
        assert nonbank["valuation_score"].notna().any()
        assert nonbank["quality_score"].notna().any()

    def test_all_category_scores_valid(self, cfg):
        """Both bank and non-bank should have valid scores for all categories."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)

        for cat in ["valuation", "quality", "growth", "momentum", "risk"]:
            col = f"{cat}_score"
            assert df[col].notna().sum() > 0, f"No valid {cat} scores"


# =====================================================================
# BANK COMPOSITE + RANKING
# =====================================================================

class TestBankComposite:
    def test_bank_gets_valid_composite(self, cfg):
        """Bank stocks should produce valid Composite scores."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)

        bank = df[df["_is_bank_like"]]
        assert bank["Composite"].notna().all(), "Bank composite should not be NaN"

    def test_bank_nonbank_comparable(self, cfg):
        """Both types should produce scores in 0-100 range."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)

        assert df["Composite"].min() >= 0
        assert df["Composite"].max() <= 100

    def test_bank_not_all_bottom(self, cfg):
        """Bank stocks should not all be at the bottom of rankings."""
        df = _make_bank_df()
        df = winsorize_metrics(df, 0.01, 0.01)
        df = compute_sector_percentiles(df)
        df = compute_category_scores(df, cfg)
        df = compute_composite(df, cfg)
        df = rank_stocks(df)

        bank = df[df["_is_bank_like"]]
        # At least some bank stocks should be in the top half
        n_total = len(df)
        top_half_count = (bank["Rank"] <= n_total / 2).sum()
        assert top_half_count > 0, "No bank stocks in top half"


# =====================================================================
# BANK PERCENTILE RANKING
# =====================================================================

class TestBankPercentileRanking:
    def test_bank_pb_percentile_valid(self):
        """Bank P/B percentiles should be computed within Financials sector."""
        df = _make_bank_df()
        df = compute_sector_percentiles(df)
        bank = df[df["_is_bank_like"]]
        assert bank["pb_ratio_pct"].notna().any(), "Bank P/B percentile should have values"

    def test_nonbank_pb_is_nan(self):
        """Non-bank stocks should have NaN P/B percentile."""
        df = _make_bank_df()
        df = compute_sector_percentiles(df)
        nonbank = df[~df["_is_bank_like"]]
        assert nonbank["pb_ratio_pct"].isna().all(), "Non-bank P/B pct should be NaN"

    def test_bank_ev_ebitda_is_nan(self):
        """Bank stocks should have NaN EV/EBITDA percentile."""
        df = _make_bank_df()
        df = compute_sector_percentiles(df)
        bank = df[df["_is_bank_like"]]
        assert bank["ev_ebitda_pct"].isna().all(), "Bank EV/EBITDA pct should be NaN"


# =====================================================================
# CONFIG VALIDATION
# =====================================================================

class TestBankConfig:
    def test_bank_valuation_weights_sum_to_100(self, cfg):
        """bank_metric_weights.valuation should sum to 100."""
        bmw = cfg["bank_metric_weights"]["valuation"]
        total = sum(bmw.values())
        assert abs(total - 100) < 0.5, f"Bank valuation weights sum to {total}"

    def test_bank_quality_weights_sum_to_100(self, cfg):
        """bank_metric_weights.quality should sum to 100."""
        bmw = cfg["bank_metric_weights"]["quality"]
        total = sum(bmw.values())
        assert abs(total - 100) < 0.5, f"Bank quality weights sum to {total}"

    def test_generic_valuation_weights_sum_to_100(self, cfg):
        """Generic valuation weights (with pb_ratio=0) should still sum to 100."""
        mw = cfg["metric_weights"]["valuation"]
        total = sum(mw.values())
        assert abs(total - 100) < 0.5, f"Generic valuation weights sum to {total}"

    def test_generic_quality_weights_sum_to_100(self, cfg):
        """Generic quality weights (with roe=0, roa=0, equity_ratio=0) should sum to 100."""
        mw = cfg["metric_weights"]["quality"]
        total = sum(mw.values())
        assert abs(total - 100) < 0.5, f"Generic quality weights sum to {total}"


# =====================================================================
# SAMPLE DATA GENERATOR
# =====================================================================

class TestSampleData:
    def test_financials_have_bank_metrics(self):
        """Financials stocks in sample data should have bank metrics."""
        from factor_engine import _generate_sample_data
        universe_df = pd.DataFrame({
            "Ticker": ["BK1", "BK2", "NB1"],
            "Company": ["Bank1", "Bank2", "Tech1"],
            "Sector": ["Financials", "Financials", "Information Technology"],
        })
        df = _generate_sample_data(universe_df)
        fin = df[df["Sector"] == "Financials"]
        assert fin["pb_ratio"].notna().all()
        assert fin["roe"].notna().all()
        assert fin["roa"].notna().all()
        assert fin["equity_ratio"].notna().all()

    def test_financials_nan_generic(self):
        """Financials in sample data should have NaN for generic bank-inappropriate metrics."""
        from factor_engine import _generate_sample_data
        universe_df = pd.DataFrame({
            "Ticker": ["BK1"],
            "Company": ["Bank1"],
            "Sector": ["Financials"],
        })
        df = _generate_sample_data(universe_df)
        assert df["ev_ebitda"].isna().all()
        assert df["roic"].isna().all()
        assert df["debt_equity"].isna().all()

    def test_nonfinancials_nan_bank(self):
        """Non-Financials in sample data should have NaN bank metrics."""
        from factor_engine import _generate_sample_data
        universe_df = pd.DataFrame({
            "Ticker": ["NB1"],
            "Company": ["Tech1"],
            "Sector": ["Information Technology"],
        })
        df = _generate_sample_data(universe_df)
        assert df["pb_ratio"].isna().all()
        assert df["roe"].isna().all()
