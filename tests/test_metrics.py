"""Unit tests for all 17 factor metrics.

Each metric is tested with:
  - Normal case (known inputs → known output)
  - Edge cases (negative denominators, zero values, missing data)
  - NaN propagation (missing input → NaN output)

These tests call compute_metrics() with a single-ticker raw data dict
and verify the resulting metric value.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from factor_engine import compute_metrics


def _make_rec(**overrides) -> dict:
    """Create a complete raw-data record with sensible defaults."""
    base = {
        "Ticker": "TEST",
        "shortName": "Test Corp",
        "sector": "Information Technology",
        "marketCap": 100e9,
        "enterpriseValue": 110e9,
        "currentPrice": 100.0,
        "trailingEps": 5.0,
        "forwardEps": 5.5,
        "totalDebt": 20e9,
        "totalCash": 10e9,
        "sharesOutstanding": 1e9,
        "earningsGrowth": 0.10,
        "dividendRate": 2.0,
        "totalRevenue": 50e9,
        "totalRevenue_prior": 45e9,
        "grossProfit": 25e9,
        "grossProfit_prior": 22e9,
        "ebit": 15e9,
        "ebitda": 18e9,
        "netIncome": 10e9,
        "netIncome_prior": 9e9,
        "incomeTaxExpense": 3e9,
        "pretaxIncome": 13e9,
        "costOfRevenue": 25e9,
        "totalAssets": 80e9,
        "totalAssets_prior": 75e9,
        "totalEquity": 40e9,
        "totalDebt_bs": 20e9,
        "longTermDebt": 15e9,
        "longTermDebt_prior": 16e9,
        "currentLiabilities": 15e9,
        "currentLiabilities_prior": 14e9,
        "currentAssets": 25e9,
        "currentAssets_prior": 23e9,
        "cash_bs": 10e9,
        "sharesBS": 1e9,
        "sharesBS_prior": 1.02e9,
        "operatingCashFlow": 14e9,
        "capex": -3e9,
        "dividendsPaid": -2e9,
        "price_latest": 100.0,
        "price_1m_ago": 98.0,
        "price_6m_ago": 85.0,
        "price_12m_ago": 80.0,
        "volatility_1y": 0.25,
    }
    base.update(overrides)
    return base


def _compute_one(rec, market_returns=None):
    """Compute metrics for a single-ticker record, return the row as dict."""
    if market_returns is None:
        market_returns = pd.Series(dtype=float)
    raw = [rec]
    df = compute_metrics(raw, market_returns)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    return df.iloc[0].to_dict()


# =====================================================================
# VALUATION METRICS
# =====================================================================

class TestEVEBITDA:
    def test_normal(self):
        row = _compute_one(_make_rec(enterpriseValue=110e9, ebitda=18e9))
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_negative_ebitda(self):
        row = _compute_one(_make_rec(ebitda=-5e9))
        assert np.isnan(row["ev_ebitda"])

    def test_zero_ebitda(self):
        row = _compute_one(_make_rec(ebitda=0))
        assert np.isnan(row["ev_ebitda"])

    def test_missing_ev_with_fallback(self):
        """When EV is NaN but MC/Debt/Cash available, EV is computed."""
        row = _compute_one(_make_rec(
            enterpriseValue=np.nan,
            marketCap=100e9, totalDebt=20e9, totalCash=10e9,
            ebitda=18e9,
        ))
        # Fallback EV = 100 + 20 - 10 = 110B
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_negative_ev(self):
        row = _compute_one(_make_rec(enterpriseValue=-10e9, ebitda=5e9))
        assert np.isnan(row["ev_ebitda"])

    def test_missing_ebitda(self):
        row = _compute_one(_make_rec(ebitda=np.nan))
        assert np.isnan(row["ev_ebitda"])


class TestFCFYield:
    def test_normal_negative_capex(self):
        """CapEx reported as negative number (yfinance convention)."""
        row = _compute_one(_make_rec(
            operatingCashFlow=14e9, capex=-3e9, enterpriseValue=110e9))
        # FCF = 14B - abs(-3B) = 11B; yield = 11/110 = 0.1
        assert abs(row["fcf_yield"] - 0.1) < 0.001

    def test_positive_capex(self):
        """CapEx reported as positive (alternative convention)."""
        row = _compute_one(_make_rec(
            operatingCashFlow=14e9, capex=3e9, enterpriseValue=110e9))
        # capex > 0 path: FCF = 14 - 3 = 11B
        assert abs(row["fcf_yield"] - 0.1) < 0.001

    def test_missing_capex(self):
        """When capex is NaN, FCF = OCF (documented limitation)."""
        row = _compute_one(_make_rec(
            operatingCashFlow=14e9, capex=np.nan, enterpriseValue=110e9))
        expected = 14e9 / 110e9
        assert abs(row["fcf_yield"] - expected) < 0.001

    def test_zero_ev_with_no_fallback(self):
        """EV=0 triggers fallback; with no MC/Debt/Cash → NaN."""
        row = _compute_one(_make_rec(
            operatingCashFlow=14e9, capex=-3e9, enterpriseValue=0,
            marketCap=np.nan, totalDebt=np.nan, totalCash=np.nan))
        assert np.isnan(row["fcf_yield"])


class TestEarningsYield:
    def test_normal(self):
        row = _compute_one(_make_rec(trailingEps=5.0, currentPrice=100.0))
        assert abs(row["earnings_yield"] - 0.05) < 0.001

    def test_negative_eps(self):
        """Negative EPS → negative earnings yield (not NaN).

        Value investors need to see that a company is unprofitable.
        Masking negative yields as NaN (→ 50th percentile) would
        silently boost loss-making companies' valuation scores.
        """
        row = _compute_one(_make_rec(trailingEps=-3.0, currentPrice=100.0))
        assert abs(row["earnings_yield"] - (-0.03)) < 0.001

    def test_zero_price(self):
        row = _compute_one(_make_rec(currentPrice=0))
        assert np.isnan(row["earnings_yield"])

    def test_missing_eps(self):
        row = _compute_one(_make_rec(trailingEps=np.nan))
        assert np.isnan(row["earnings_yield"])


class TestEVSales:
    def test_normal(self):
        row = _compute_one(_make_rec(enterpriseValue=110e9, totalRevenue=50e9))
        assert abs(row["ev_sales"] - 2.2) < 0.01

    def test_zero_revenue(self):
        row = _compute_one(_make_rec(totalRevenue=0))
        assert np.isnan(row["ev_sales"])

    def test_negative_ev(self):
        row = _compute_one(_make_rec(enterpriseValue=-5e9, totalRevenue=50e9))
        assert np.isnan(row["ev_sales"])


# =====================================================================
# QUALITY METRICS
# =====================================================================

class TestROIC:
    def test_normal(self):
        row = _compute_one(_make_rec(
            ebit=15e9, incomeTaxExpense=3e9, pretaxIncome=13e9,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9))
        # tax_rate = 3/13 ≈ 0.2308, clamped to [0, 0.5]
        tax_rate = 3e9 / 13e9
        nopat = 15e9 * (1 - tax_rate)
        ic = 40e9 + 20e9 - 10e9  # 50B
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_negative_pretax_defaults_to_21(self):
        row = _compute_one(_make_rec(
            ebit=5e9, pretaxIncome=-1e9,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9))
        # Should use default 21% tax rate
        nopat = 5e9 * (1 - 0.21)
        ic = 40e9 + 20e9 - 10e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_negative_ic(self):
        row = _compute_one(_make_rec(
            ebit=5e9, totalEquity=-10e9, totalDebt=5e9, totalCash=10e9))
        # IC = -10 + 5 - 10 = -15B → NaN
        assert np.isnan(row["roic"])

    def test_tax_rate_clamping(self):
        """Tax rate > 50% should be clamped to 50%."""
        row = _compute_one(_make_rec(
            ebit=10e9, incomeTaxExpense=8e9, pretaxIncome=10e9,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9))
        # Raw tax_rate = 8/10 = 0.8, clamped to 0.5
        nopat = 10e9 * (1 - 0.5)
        ic = 40e9 + 20e9 - 10e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_missing_ebit(self):
        row = _compute_one(_make_rec(ebit=np.nan))
        assert np.isnan(row["roic"])


class TestGrossProfitAssets:
    def test_normal(self):
        row = _compute_one(_make_rec(grossProfit=25e9, totalAssets=80e9))
        assert abs(row["gross_profit_assets"] - 0.3125) < 0.001

    def test_zero_assets(self):
        row = _compute_one(_make_rec(totalAssets=0))
        assert np.isnan(row["gross_profit_assets"])


class TestDebtEquity:
    def test_normal(self):
        row = _compute_one(_make_rec(totalDebt=20e9, totalEquity=40e9))
        assert abs(row["debt_equity"] - 0.5) < 0.001

    def test_negative_equity(self):
        """Negative equity → sentinel 999.0."""
        row = _compute_one(_make_rec(totalEquity=-5e9, totalDebt=20e9))
        assert row["debt_equity"] == 999.0

    def test_zero_equity(self):
        row = _compute_one(_make_rec(totalEquity=0, totalDebt=20e9))
        assert row["debt_equity"] == 999.0


class TestPiotroskiFScore:
    def test_all_passing(self):
        """All 9 signals testable and passing → raw score 9."""
        row = _compute_one(_make_rec(
            netIncome=10e9, netIncome_prior=9e9,
            operatingCashFlow=14e9,
            totalAssets=80e9, totalAssets_prior=75e9,
            longTermDebt=15e9, longTermDebt_prior=16e9,
            currentAssets=25e9, currentAssets_prior=23e9,
            currentLiabilities=15e9, currentLiabilities_prior=14e9,
            sharesBS=1e9, sharesBS_prior=1.02e9,
            grossProfit=25e9, grossProfit_prior=22e9,
            totalRevenue=50e9, totalRevenue_prior=45e9,
        ))
        assert row["piotroski_f_score"] == 9

    def test_partial_data_not_inflated(self):
        """5 of 5 testable signals pass → raw score 5 (NOT 9).

        Old proportional normalization would have returned (5/5)*9 = 9.0,
        falsely equating partial data with perfect quality.
        """
        row = _compute_one(_make_rec(
            netIncome=10e9,
            operatingCashFlow=14e9,
            # These 3 are testable: NI>0 ✓, OCF>0 ✓, OCF>NI ✓
            # Need at least 4 testable; add shares test
            sharesBS=1e9, sharesBS_prior=1.02e9,
            # And revenue/assets for ATO
            totalRevenue=50e9, totalRevenue_prior=45e9,
            totalAssets=80e9, totalAssets_prior=75e9,
            # Disable remaining signals
            netIncome_prior=np.nan,  # ROA change untestable (need both NI periods)
            longTermDebt=np.nan, longTermDebt_prior=np.nan,
            currentAssets=np.nan, currentAssets_prior=np.nan,
            currentLiabilities=np.nan, currentLiabilities_prior=np.nan,
            grossProfit=np.nan, grossProfit_prior=np.nan,
        ))
        # Testable: NI>0 ✓, OCF>0 ✓, OCF>NI ✓, Shares↓ ✓, ATO↑ ✓ = 5 of 5
        # Raw score = 5 (not 9)
        assert row["piotroski_f_score"] == 5

    def test_insufficient_data(self):
        """Fewer than 4 testable signals → NaN."""
        row = _compute_one(_make_rec(
            netIncome=10e9,
            operatingCashFlow=np.nan,
            totalAssets=np.nan, totalAssets_prior=np.nan,
            longTermDebt=np.nan, longTermDebt_prior=np.nan,
            currentAssets=np.nan, currentAssets_prior=np.nan,
            currentLiabilities=np.nan, currentLiabilities_prior=np.nan,
            sharesBS=np.nan, sharesBS_prior=np.nan,
            grossProfit=np.nan, grossProfit_prior=np.nan,
        ))
        assert np.isnan(row["piotroski_f_score"])

    def test_is_integer(self):
        """F-Score should be an integer (not a float like 6.43)."""
        row = _compute_one(_make_rec())
        if not np.isnan(row["piotroski_f_score"]):
            assert row["piotroski_f_score"] == int(row["piotroski_f_score"])


class TestAccruals:
    def test_normal(self):
        row = _compute_one(_make_rec(
            netIncome=10e9, operatingCashFlow=14e9, totalAssets=80e9))
        # (10 - 14) / 80 = -0.05
        assert abs(row["accruals"] - (-0.05)) < 0.001

    def test_positive_accruals(self):
        """NI > OCF → positive accruals (lower quality)."""
        row = _compute_one(_make_rec(
            netIncome=14e9, operatingCashFlow=10e9, totalAssets=80e9))
        assert row["accruals"] > 0

    def test_zero_assets(self):
        row = _compute_one(_make_rec(totalAssets=0))
        assert np.isnan(row["accruals"])


# =====================================================================
# GROWTH METRICS
# =====================================================================

class TestForwardEPSGrowth:
    def test_normal(self):
        row = _compute_one(_make_rec(forwardEps=5.5, trailingEps=5.0))
        # (5.5 - 5.0) / |5.0| = 0.10
        assert abs(row["forward_eps_growth"] - 0.10) < 0.001

    def test_negative_trailing(self):
        """Negative trailing EPS: growth uses absolute value in denominator."""
        row = _compute_one(_make_rec(forwardEps=2.0, trailingEps=-3.0))
        # (2.0 - (-3.0)) / |-3.0| = 5.0/3.0 ≈ 1.667
        expected = (2.0 - (-3.0)) / abs(-3.0)
        assert abs(row["forward_eps_growth"] - expected) < 0.001

    def test_near_zero_trailing(self):
        row = _compute_one(_make_rec(trailingEps=0.005))
        # |0.005| < 0.01 → NaN
        assert np.isnan(row["forward_eps_growth"])

    def test_missing_forward(self):
        row = _compute_one(_make_rec(forwardEps=np.nan))
        assert np.isnan(row["forward_eps_growth"])


class TestPEGRatio:
    def test_normal(self):
        row = _compute_one(_make_rec(
            currentPrice=100.0, trailingEps=5.0, earningsGrowth=0.10))
        # P/E = 100/5 = 20; PEG = 20 / (0.10 * 100) = 20/10 = 2.0
        assert abs(row["peg_ratio"] - 2.0) < 0.01

    def test_negative_growth(self):
        row = _compute_one(_make_rec(earningsGrowth=-0.05))
        assert np.isnan(row["peg_ratio"])

    def test_zero_growth(self):
        row = _compute_one(_make_rec(earningsGrowth=0.005))
        assert np.isnan(row["peg_ratio"])


class TestRevenueGrowth:
    def test_normal(self):
        row = _compute_one(_make_rec(
            totalRevenue=50e9, totalRevenue_prior=45e9))
        expected = (50 - 45) / 45
        assert abs(row["revenue_growth"] - expected) < 0.001

    def test_zero_prior(self):
        row = _compute_one(_make_rec(totalRevenue_prior=0))
        assert np.isnan(row["revenue_growth"])

    def test_declining_revenue(self):
        row = _compute_one(_make_rec(
            totalRevenue=40e9, totalRevenue_prior=50e9))
        assert row["revenue_growth"] < 0


class TestSustainableGrowth:
    def test_normal(self):
        row = _compute_one(_make_rec(
            netIncome=10e9, totalEquity=40e9,
            dividendsPaid=-2e9, sharesOutstanding=1e9))
        # ROE = 10/40 = 0.25; retention = 1 - 2/10 = 0.8
        # Sustainable growth = 0.25 * 0.8 = 0.20
        expected = 0.25 * 0.80
        assert abs(row["sustainable_growth"] - expected) < 0.01

    def test_negative_ni(self):
        row = _compute_one(_make_rec(netIncome=-5e9))
        assert np.isnan(row["sustainable_growth"])

    def test_negative_equity(self):
        row = _compute_one(_make_rec(totalEquity=-10e9))
        assert np.isnan(row["sustainable_growth"])

    def test_unknown_dividends_is_nan(self):
        """When dividend data is entirely unavailable, sustainable_growth = NaN.

        Previously assumed full retention (divs=0), which biased growth
        estimates upward for companies that do pay dividends.
        """
        row = _compute_one(_make_rec(
            dividendsPaid=np.nan, dividendRate=np.nan,
            sharesOutstanding=np.nan))
        assert np.isnan(row["sustainable_growth"])


# =====================================================================
# MOMENTUM METRICS
# =====================================================================

class TestReturn12_1:
    def test_normal(self):
        row = _compute_one(_make_rec(price_1m_ago=98.0, price_12m_ago=80.0))
        expected = (98 - 80) / 80  # 0.225
        assert abs(row["return_12_1"] - expected) < 0.001

    def test_price_decline(self):
        row = _compute_one(_make_rec(price_1m_ago=70.0, price_12m_ago=80.0))
        assert row["return_12_1"] < 0

    def test_missing_12m(self):
        row = _compute_one(_make_rec(price_12m_ago=np.nan))
        assert np.isnan(row["return_12_1"])

    def test_zero_12m_price(self):
        row = _compute_one(_make_rec(price_12m_ago=0))
        assert np.isnan(row["return_12_1"])


class TestReturn6M:
    def test_normal(self):
        """6-1M return: (price_1m_ago - price_6m_ago) / price_6m_ago."""
        row = _compute_one(_make_rec(price_1m_ago=98.0, price_6m_ago=85.0))
        expected = (98 - 85) / 85
        assert abs(row["return_6m"] - expected) < 0.001

    def test_excludes_recent_month(self):
        """6-1M convention excludes the most recent month (uses p1m, not pnow)."""
        row = _compute_one(_make_rec(price_latest=110.0, price_1m_ago=98.0, price_6m_ago=85.0))
        expected = (98 - 85) / 85  # Should use p1m=98, NOT pnow=110
        assert abs(row["return_6m"] - expected) < 0.001

    def test_missing_6m(self):
        row = _compute_one(_make_rec(price_6m_ago=np.nan))
        assert np.isnan(row["return_6m"])

    def test_missing_1m(self):
        """Missing price_1m_ago → NaN (needed for 6-1M calculation)."""
        row = _compute_one(_make_rec(price_1m_ago=np.nan, price_6m_ago=85.0))
        assert np.isnan(row["return_6m"])


# =====================================================================
# RISK METRICS
# =====================================================================

class TestVolatility:
    def test_normal(self):
        row = _compute_one(_make_rec(volatility_1y=0.25))
        assert abs(row["volatility"] - 0.25) < 0.001

    def test_missing(self):
        row = _compute_one(_make_rec(volatility_1y=np.nan))
        assert np.isnan(row["volatility"])


class TestBeta:
    def test_no_market_returns(self):
        """No market returns → NaN beta."""
        row = _compute_one(_make_rec(), market_returns=pd.Series(dtype=float))
        assert np.isnan(row["beta"])

    def test_with_market_returns(self):
        """With aligned returns, beta should be computable."""
        # Create 250 aligned daily returns
        np.random.seed(42)
        dates = pd.bdate_range("2025-01-01", periods=250)
        market_ret = pd.Series(np.random.normal(0.0004, 0.01, 250), index=dates)
        # Stock returns correlated with market
        stock_ret = market_ret * 1.2 + np.random.normal(0, 0.005, 250)

        rec = _make_rec()
        rec["_daily_returns"] = {
            dt.strftime("%Y-%m-%d"): float(v)
            for dt, v in zip(dates, stock_ret)
        }
        row = _compute_one(rec, market_returns=market_ret)
        # Beta should be roughly 1.2 (with noise)
        assert not np.isnan(row["beta"])
        assert 0.5 < row["beta"] < 2.0


# =====================================================================
# REVISIONS METRICS
# =====================================================================

class TestAnalystSurprise:
    def test_normal(self):
        row = _compute_one(_make_rec(analyst_surprise=0.05))
        assert abs(row["analyst_surprise"] - 0.05) < 0.001

    def test_missing(self):
        row = _compute_one(_make_rec())  # No analyst_surprise in defaults
        # analyst_surprise not in base defaults → NaN
        assert np.isnan(row.get("analyst_surprise", np.nan))

    def test_negative_surprise(self):
        row = _compute_one(_make_rec(analyst_surprise=-0.10))
        assert abs(row["analyst_surprise"] - (-0.10)) < 0.001


class TestPlaceholderRevisions:
    def test_revision_ratio_always_nan(self):
        row = _compute_one(_make_rec())
        assert np.isnan(row["eps_revision_ratio"])

    def test_estimate_change_always_nan(self):
        row = _compute_one(_make_rec())
        assert np.isnan(row["eps_estimate_change"])
