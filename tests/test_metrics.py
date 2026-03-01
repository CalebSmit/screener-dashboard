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
        "da_cf": 3e9,
        "totalEquity_prior": 38e9,
        "payoutRatio": np.nan,
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
        row = _compute_one(_make_rec(ebitda=-5e9, ebit=-8e9, da_cf=3e9))
        # EBIT(-8) + D&A(3) = -5 → negative EBITDA → NaN
        assert np.isnan(row["ev_ebitda"])

    def test_zero_ebitda(self):
        row = _compute_one(_make_rec(ebitda=0, ebit=-3e9, da_cf=3e9))
        # EBIT(-3) + D&A(3) = 0 → zero EBITDA → NaN
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
        row = _compute_one(_make_rec(ebitda=np.nan, ebit=np.nan, da_cf=np.nan))
        assert np.isnan(row["ev_ebitda"])

    def test_ebitda_computed_from_ebit_plus_da(self):
        """When D&A is available, EBITDA = EBIT + D&A (not reported EBITDA)."""
        row = _compute_one(_make_rec(
            enterpriseValue=110e9, ebit=15e9, da_cf=3e9, ebitda=20e9))
        # Should use EBIT(15) + D&A(3) = 18, NOT reported ebitda(20)
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_ebitda_fallback_when_da_missing(self):
        """When D&A is NaN, fall back to reported EBITDA."""
        row = _compute_one(_make_rec(
            enterpriseValue=110e9, ebit=15e9, da_cf=np.nan, ebitda=18e9))
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_ebitda_fallback_when_ebit_missing(self):
        """When EBIT is NaN, fall back to reported EBITDA."""
        row = _compute_one(_make_rec(
            enterpriseValue=110e9, ebit=np.nan, da_cf=3e9, ebitda=18e9))
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01


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
        """When capex is NaN, FCF is NaN (not OCF — zero capex assumption
        would overstate FCF for capital-intensive companies)."""
        row = _compute_one(_make_rec(
            operatingCashFlow=14e9, capex=np.nan, enterpriseValue=110e9))
        assert np.isnan(row["fcf_yield"])

    def test_zero_ev_with_no_fallback(self):
        """EV=0 triggers fallback; with no MC/Debt/Cash → NaN."""
        row = _compute_one(_make_rec(
            operatingCashFlow=14e9, capex=-3e9, enterpriseValue=0,
            marketCap=np.nan, totalDebt=np.nan, totalCash=np.nan))
        assert np.isnan(row["fcf_yield"])


class TestEarningsYield:
    def test_normal_ltm(self):
        """Primary path: LTM NI / MC."""
        row = _compute_one(_make_rec(netIncome=10e9, marketCap=100e9))
        assert abs(row["earnings_yield"] - 0.10) < 0.001

    def test_negative_ni(self):
        """Negative NI → negative earnings yield (not NaN).

        Value investors need to see that a company is unprofitable.
        Masking negative yields as NaN (→ 50th percentile) would
        silently boost loss-making companies' valuation scores.
        """
        row = _compute_one(_make_rec(netIncome=-5e9, marketCap=100e9))
        assert abs(row["earnings_yield"] - (-0.05)) < 0.001

    def test_fallback_to_trailing_eps(self):
        """When NI or MC is missing, fall back to trailingEps / price."""
        row = _compute_one(_make_rec(netIncome=np.nan, trailingEps=5.0, currentPrice=100.0))
        assert abs(row["earnings_yield"] - 0.05) < 0.001

    def test_zero_mc(self):
        """MC=0 → falls back to trailingEps path."""
        row = _compute_one(_make_rec(marketCap=0, trailingEps=5.0, currentPrice=100.0))
        # MC=0 triggers NaN from primary path, fallback uses EPS/price
        assert abs(row["earnings_yield"] - 0.05) < 0.001

    def test_all_missing(self):
        """Both paths fail → NaN."""
        row = _compute_one(_make_rec(netIncome=np.nan, marketCap=np.nan,
                                      trailingEps=np.nan, currentPrice=0))
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


class TestEVCrossValidation:
    """EV cross-validation: API EV vs computed MC+Debt-Cash (H4)."""

    def test_normal_ev_not_flagged(self):
        """EV within 10% of computed → no flag, API EV used."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=110B → match
        row = _compute_one(_make_rec(
            marketCap=100e9, enterpriseValue=110e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" not in row or pd.isna(row.get("_ev_flag"))
        # EV/EBITDA uses 110B as expected
        expected = 110e9 / 18e9  # default ebitda=18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_ev_discrepancy_uses_computed(self):
        """API EV > 10% off from computed → flag set, computed EV used."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=440B (4x off)
        row = _compute_one(_make_rec(
            marketCap=100e9, enterpriseValue=440e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" in row and pd.notna(row.get("_ev_flag"))
        # Should use computed EV=110B, not API EV=440B
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_ev_slightly_off_not_flagged(self):
        """API EV within 10% tolerance → no flag."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=115B (4.5% off)
        row = _compute_one(_make_rec(
            marketCap=100e9, enterpriseValue=115e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" not in row or pd.isna(row.get("_ev_flag"))

    def test_ev_low_discrepancy_uses_computed(self):
        """API EV significantly below computed → flag set, computed EV used."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=50B (under by 55%)
        row = _compute_one(_make_rec(
            marketCap=100e9, enterpriseValue=50e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" in row and pd.notna(row.get("_ev_flag"))
        expected = 110e9 / 18e9
        assert abs(row["ev_ebitda"] - expected) < 0.01

    def test_financials_wider_threshold_not_flagged(self):
        """Financials within 25% tolerance → no flag (wider than 10%)."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=130B (18% off)
        # Would flag at 10% but NOT at 25% for Financials
        row = _compute_one(_make_rec(
            sector="Financials",
            marketCap=100e9, enterpriseValue=130e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" not in row or pd.isna(row.get("_ev_flag"))

    def test_financials_extreme_still_flagged(self):
        """Financials beyond 25% tolerance → still flagged and corrected."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=440B (4x off)
        row = _compute_one(_make_rec(
            sector="Financials",
            marketCap=100e9, enterpriseValue=440e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" in row and pd.notna(row.get("_ev_flag"))
        # Verify the flag message contains the correct ratio
        assert "4.00" in row["_ev_flag"] or "ratio=" in row["_ev_flag"]

    def test_financial_services_sector_uses_25pct(self):
        """yfinance returns 'Financial Services' — must also get 25% tolerance."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=130B (18% off)
        row = _compute_one(_make_rec(
            sector="Financial Services",
            marketCap=100e9, enterpriseValue=130e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" not in row or pd.isna(row.get("_ev_flag"))

    def test_non_financial_still_uses_10pct(self):
        """Non-Financials at 18% off → flagged (10% threshold)."""
        # MC=100B, Debt=20B, Cash=10B → computed=110B; API=130B (18% off)
        row = _compute_one(_make_rec(
            sector="Information Technology",
            marketCap=100e9, enterpriseValue=130e9,
            totalDebt=20e9, totalCash=10e9))
        assert "_ev_flag" in row and pd.notna(row.get("_ev_flag"))


# =====================================================================
# QUALITY METRICS
# =====================================================================

class TestROIC:
    def test_normal(self):
        """IC = Equity + Debt - Excess Cash (capped at 50% of cash).
        Excess Cash = min(max(0, cash - 2% of revenue), 50% of cash).
        IC floored at 10% of Total Assets.
        """
        row = _compute_one(_make_rec(
            ebit=15e9, incomeTaxExpense=3e9, pretaxIncome=13e9,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9,
            totalRevenue=50e9, cash_bs=10e9, totalDebt_bs=20e9,
            totalAssets=80e9))
        # tax_rate = 3/13 ≈ 0.2308
        tax_rate = 3e9 / 13e9
        nopat = 15e9 * (1 - tax_rate)
        # excess_cash = max(0, 10B - 1B) = 9B, capped at min(9B, 5B) = 5B
        # IC = 40 + 20 - 5 = 55B; floor = 0.10 * 80B = 8B → IC = 55B
        ic = 40e9 + 20e9 - 5e9  # 55B
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_negative_pretax_uses_zero_tax(self):
        """Negative pretax income = tax-loss position → 0% tax rate (H3)."""
        row = _compute_one(_make_rec(
            ebit=5e9, pretaxIncome=-1e9,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9,
            totalRevenue=50e9, cash_bs=10e9, totalDebt_bs=20e9,
            totalAssets=80e9))
        # Tax-loss position: tax_rate = 0%, so NOPAT = EBIT
        nopat = 5e9 * (1 - 0.0)
        # excess_cash = 9B capped at 5B; IC = 40+20-5 = 55B
        ic = 40e9 + 20e9 - 5e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_zero_pretax_uses_zero_tax(self):
        """Zero pretax income → 0% tax rate (boundary of H3 fix)."""
        row = _compute_one(_make_rec(
            ebit=5e9, pretaxIncome=0,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9,
            totalRevenue=50e9, cash_bs=10e9, totalDebt_bs=20e9,
            totalAssets=80e9))
        nopat = 5e9  # 0% tax
        ic = 40e9 + 20e9 - 5e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_missing_pretax_defaults_to_21(self):
        """Missing pretax (NaN) → fall through to 21% default."""
        row = _compute_one(_make_rec(
            ebit=5e9, pretaxIncome=np.nan, incomeTaxExpense=np.nan,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9,
            totalRevenue=50e9, cash_bs=10e9, totalDebt_bs=20e9,
            totalAssets=80e9))
        nopat = 5e9 * (1 - 0.21)
        ic = 40e9 + 20e9 - 5e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_negative_ic_floored(self):
        """Negative IC is floored at 10% of Total Assets."""
        row = _compute_one(_make_rec(
            ebit=5e9, totalEquity=-10e9, totalDebt_bs=5e9, cash_bs=0,
            totalRevenue=50e9, totalAssets=80e9))
        # IC = -10 + 5 - 0 = -5B, floored at 0.10*80B = 8B
        tax_rate = 3e9 / 13e9  # default from _make_rec
        nopat = 5e9 * (1 - tax_rate)
        expected = nopat / 8e9  # IC floor
        assert abs(row["roic"] - expected) < 0.0001

    def test_tax_rate_clamping(self):
        """Tax rate > 50% should be clamped to 50%."""
        row = _compute_one(_make_rec(
            ebit=10e9, incomeTaxExpense=8e9, pretaxIncome=10e9,
            totalEquity=40e9, totalDebt=20e9, totalCash=10e9,
            totalRevenue=50e9, cash_bs=10e9, totalDebt_bs=20e9,
            totalAssets=80e9))
        # Raw tax_rate = 8/10 = 0.8, clamped to 0.5
        nopat = 10e9 * (1 - 0.5)
        # excess_cash = 9B capped at 5B; IC = 55B
        ic = 40e9 + 20e9 - 5e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_uses_bs_debt_for_ic(self):
        """ROIC invested capital uses balance-sheet debt, not info debt."""
        row = _compute_one(_make_rec(
            ebit=15e9, incomeTaxExpense=3e9, pretaxIncome=13e9,
            totalEquity=40e9, totalDebt=30e9, totalDebt_bs=20e9,
            totalCash=10e9, cash_bs=10e9, totalRevenue=50e9,
            totalAssets=80e9))
        # IC should use totalDebt_bs=20B, not totalDebt=30B
        tax_rate = 3e9 / 13e9
        nopat = 15e9 * (1 - tax_rate)
        # excess_cash = 9B capped at 5B; IC = 40+20-5 = 55B
        ic = 40e9 + 20e9 - 5e9
        expected = nopat / ic
        assert abs(row["roic"] - expected) < 0.0001

    def test_excess_cash_capped_at_50pct(self):
        """Excess cash is capped at 50% of total cash."""
        row = _compute_one(_make_rec(
            ebit=10e9, incomeTaxExpense=2e9, pretaxIncome=10e9,
            totalEquity=40e9, totalDebt_bs=10e9, cash_bs=50e9,
            totalRevenue=20e9, totalAssets=100e9))
        # excess_cash = max(0, 50B - 0.4B) = 49.6B
        # capped at 50% of 50B = 25B
        # IC = 40 + 10 - 25 = 25B; floor = 10B → IC = 25B
        tax_rate = 2e9 / 10e9
        nopat = 10e9 * (1 - tax_rate)
        expected = nopat / 25e9
        assert abs(row["roic"] - expected) < 0.01

    def test_ic_floored_at_10pct_total_assets(self):
        """IC cannot go below 10% of total assets."""
        row = _compute_one(_make_rec(
            ebit=5e9, incomeTaxExpense=1e9, pretaxIncome=5e9,
            totalEquity=5e9, totalDebt_bs=2e9, cash_bs=10e9,
            totalRevenue=10e9, totalAssets=200e9))
        # excess_cash = max(0, 10B - 0.2B) = 9.8B
        # capped at 50% of 10B = 5B
        # IC = 5 + 2 - 5 = 2B; floor = 10% * 200B = 20B → IC = 20B
        tax_rate = 1e9 / 5e9
        nopat = 5e9 * (1 - tax_rate)
        expected = nopat / 20e9
        assert abs(row["roic"] - expected) < 0.01

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
        """D/E uses balance-sheet debt for temporal consistency."""
        row = _compute_one(_make_rec(totalDebt=20e9, totalDebt_bs=20e9, totalEquity=40e9))
        assert abs(row["debt_equity"] - 0.5) < 0.001

    def test_uses_bs_debt_not_info_debt(self):
        """D/E should use balance sheet debt, not .info totalDebt."""
        row = _compute_one(_make_rec(totalDebt=50e9, totalDebt_bs=20e9, totalEquity=40e9))
        # D/E = 20B / 40B = 0.5 (not 50B / 40B = 1.25)
        assert abs(row["debt_equity"] - 0.5) < 0.001

    def test_negative_equity(self):
        """Negative equity -> NaN (not a sentinel value)."""
        row = _compute_one(_make_rec(totalEquity=-5e9, totalDebt=20e9))
        assert np.isnan(row["debt_equity"])

    def test_zero_equity(self):
        row = _compute_one(_make_rec(totalEquity=0, totalDebt=20e9))
        assert np.isnan(row["debt_equity"])


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
        """6 of 6 testable signals pass → raw score 6 (NOT 9).

        Old proportional normalization would have returned (6/6)*9 = 9.0,
        falsely equating partial data with perfect quality.
        Minimum 6 testable signals required for a meaningful score.
        """
        row = _compute_one(_make_rec(
            netIncome=10e9,
            netIncome_prior=9e9,
            operatingCashFlow=14e9,
            # Testable: NI>0 ✓, OCF>0 ✓, ROA change ✓, OCF>NI ✓,
            #           Shares↓ ✓, ATO↑ ✓ = 6 of 6
            sharesBS=1e9, sharesBS_prior=1.02e9,
            totalRevenue=50e9, totalRevenue_prior=45e9,
            totalAssets=80e9, totalAssets_prior=75e9,
            # Disable remaining signals
            longTermDebt=np.nan, longTermDebt_prior=np.nan,
            currentAssets=np.nan, currentAssets_prior=np.nan,
            currentLiabilities=np.nan, currentLiabilities_prior=np.nan,
            grossProfit=np.nan, grossProfit_prior=np.nan,
        ))
        # Testable: NI>0 ✓, OCF>0 ✓, ROA↑ ✓, OCF>NI ✓, Shares↓ ✓, ATO↑ ✓ = 6
        # Raw score = 6 (not 9)
        assert row["piotroski_f_score"] == 6

    def test_insufficient_data(self):
        """Fewer than 6 testable signals → NaN."""
        row = _compute_one(_make_rec(
            netIncome=10e9,
            operatingCashFlow=14e9,
            # Only 3 testable: NI>0, OCF>0, OCF>NI — below threshold of 6
            totalAssets=np.nan, totalAssets_prior=np.nan,
            longTermDebt=np.nan, longTermDebt_prior=np.nan,
            currentAssets=np.nan, currentAssets_prior=np.nan,
            currentLiabilities=np.nan, currentLiabilities_prior=np.nan,
            sharesBS=np.nan, sharesBS_prior=np.nan,
            grossProfit=np.nan, grossProfit_prior=np.nan,
        ))
        assert np.isnan(row["piotroski_f_score"])

    def test_five_testable_is_insufficient(self):
        """Exactly 5 testable signals → NaN (minimum is 6)."""
        row = _compute_one(_make_rec(
            netIncome=10e9,
            operatingCashFlow=14e9,
            sharesBS=1e9, sharesBS_prior=1.02e9,
            totalRevenue=50e9, totalRevenue_prior=45e9,
            totalAssets=80e9, totalAssets_prior=75e9,
            # 5 testable: NI>0, OCF>0, OCF>NI, Shares↓, ATO↑
            netIncome_prior=np.nan,
            longTermDebt=np.nan, longTermDebt_prior=np.nan,
            currentAssets=np.nan, currentAssets_prior=np.nan,
            currentLiabilities=np.nan, currentLiabilities_prior=np.nan,
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
        """Negative trailing EPS: growth uses absolute value in denominator,
        floored at $1.00, and clamped to [-75%, +150%]."""
        row = _compute_one(_make_rec(forwardEps=2.0, trailingEps=-3.0))
        # (2.0 - (-3.0)) / max(|-3.0|, 1.0) = 5.0/3.0 ≈ 1.667
        # Clamped to 1.50 (new upper bound)
        assert abs(row["forward_eps_growth"] - 1.50) < 0.001

    def test_clamp_at_150_pct(self):
        """Growth above 150% is clamped to 1.50."""
        row = _compute_one(_make_rec(forwardEps=20.0, trailingEps=5.0))
        # Raw = (20-5)/5 = 3.0, clamped to 1.50
        assert abs(row["forward_eps_growth"] - 1.50) < 0.001

    def test_near_zero_trailing(self):
        row = _compute_one(_make_rec(trailingEps=0.005))
        # |0.005| < 0.01 → NaN
        assert np.isnan(row["forward_eps_growth"])

    def test_missing_forward(self):
        row = _compute_one(_make_rec(forwardEps=np.nan))
        assert np.isnan(row["forward_eps_growth"])


class TestPEGRatio:
    def test_normal(self):
        """PEG uses computed forward_eps_growth = (fwd - trail) / |trail|."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, trailingEps=5.0, forwardEps=5.5))
        # forward_eps_growth = (5.5-5.0)/5.0 = 0.10
        # P/E = 100/5 = 20; PEG = 20 / (0.10 * 100) = 20/10 = 2.0
        assert abs(row["peg_ratio"] - 2.0) < 0.01

    def test_negative_growth(self):
        """Negative forward EPS growth → PEG is NaN."""
        row = _compute_one(_make_rec(
            trailingEps=5.0, forwardEps=4.5))
        # forward_eps_growth = (4.5-5.0)/5.0 = -0.10 → PEG NaN
        assert np.isnan(row["peg_ratio"])

    def test_very_low_growth(self):
        """Near-zero forward EPS growth (0.8%) → PEG is high but valid (capped at 50)."""
        row = _compute_one(_make_rec(
            trailingEps=5.0, forwardEps=5.04))
        # forward_eps_growth = (5.04-5.0)/5.0 = 0.008 = 0.8%
        # P/E = 100/5 = 20; PEG = 20 / (0.008 * 100) = 25.0 (under cap)
        assert abs(row["peg_ratio"] - 25.0) < 0.5

    def test_zero_growth_exact(self):
        """Exactly zero growth → PEG is NaN."""
        row = _compute_one(_make_rec(
            trailingEps=5.0, forwardEps=5.0))
        # forward_eps_growth = (5.0-5.0)/5.0 = 0.0 → PEG NaN
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
    def test_normal_with_avg_equity(self):
        """SGR uses average equity when prior year available."""
        row = _compute_one(_make_rec(
            netIncome=10e9, totalEquity=40e9, totalEquity_prior=36e9,
            dividendsPaid=-2e9, sharesOutstanding=1e9, payoutRatio=np.nan))
        # avg_eq = (40+36)/2 = 38B; ROE = 10/38 ≈ 0.2632
        # retention = 1 - 2/10 = 0.8; SGR = 0.2632 * 0.8 ≈ 0.2105
        avg_eq = (40e9 + 36e9) / 2
        roe = 10e9 / avg_eq
        expected = roe * 0.8
        assert abs(row["sustainable_growth"] - expected) < 0.01

    def test_uses_payout_ratio_from_info(self):
        """payoutRatio from .info takes priority over dividendsPaid computation."""
        row = _compute_one(_make_rec(
            netIncome=10e9, totalEquity=40e9, totalEquity_prior=40e9,
            payoutRatio=0.30,
            dividendsPaid=-5e9))  # would give 50% payout, but payoutRatio wins
        # avg_eq = (40+40)/2 = 40B; ROE = 10/40 = 0.25
        # retention = 1 - 0.30 = 0.70; SGR = 0.25 * 0.70 = 0.175
        expected = 0.25 * 0.70
        assert abs(row["sustainable_growth"] - expected) < 0.01

    def test_sgr_clamped_to_0_1(self):
        """SGR is clamped to [0%, 100%]."""
        row = _compute_one(_make_rec(
            netIncome=50e9, totalEquity=10e9, totalEquity_prior=10e9,
            dividendsPaid=0, payoutRatio=0.0))
        # ROE = 50/10 = 5.0; retention = 1.0; SGR = 5.0 → clamped to 1.0
        assert abs(row["sustainable_growth"] - 1.0) < 0.001

    def test_fallback_to_single_year_equity(self):
        """When prior equity is unavailable, uses current year only."""
        row = _compute_one(_make_rec(
            netIncome=10e9, totalEquity=40e9, totalEquity_prior=np.nan,
            dividendsPaid=-2e9, payoutRatio=np.nan))
        # ROE = 10/40 = 0.25; retention = 0.8; SGR = 0.20
        assert abs(row["sustainable_growth"] - 0.20) < 0.01

    def test_negative_ni(self):
        row = _compute_one(_make_rec(netIncome=-5e9))
        assert np.isnan(row["sustainable_growth"])

    def test_negative_equity(self):
        row = _compute_one(_make_rec(totalEquity=-10e9))
        assert np.isnan(row["sustainable_growth"])

    def test_unknown_dividends_is_nan(self):
        """When dividend data is entirely unavailable and payoutRatio missing,
        sustainable_growth = NaN."""
        row = _compute_one(_make_rec(
            dividendsPaid=np.nan, dividendRate=np.nan,
            sharesOutstanding=np.nan, payoutRatio=np.nan))
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


class TestRevisionsCategory:
    def test_revisions_metrics(self):
        """Revisions category has analyst_surprise and price_target_upside."""
        row = _compute_one(_make_rec(
            analyst_surprise=0.05,
            currentPrice=100.0, targetMeanPrice=120.0,
            numberOfAnalystOpinions=10))
        assert abs(row["analyst_surprise"] - 0.05) < 0.001
        assert abs(row["price_target_upside"] - 0.20) < 0.001
        assert "eps_revision_ratio" not in row
        assert "eps_estimate_change" not in row


class TestPriceTargetUpside:
    def test_normal_upside(self):
        """Standard case: target above current price."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, targetMeanPrice=120.0,
            numberOfAnalystOpinions=10))
        assert abs(row["price_target_upside"] - 0.20) < 0.001

    def test_downside(self):
        """Target below current price -> negative upside."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, targetMeanPrice=85.0,
            numberOfAnalystOpinions=10))
        assert abs(row["price_target_upside"] - (-0.15)) < 0.001

    def test_clamp_high(self):
        """Extreme upside clamped to +100%."""
        row = _compute_one(_make_rec(
            currentPrice=10.0, targetMeanPrice=50.0,
            numberOfAnalystOpinions=5))
        assert abs(row["price_target_upside"] - 1.00) < 0.001

    def test_clamp_low(self):
        """Extreme downside clamped to -50%."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, targetMeanPrice=30.0,
            numberOfAnalystOpinions=5))
        assert abs(row["price_target_upside"] - (-0.50)) < 0.001

    def test_insufficient_analysts(self):
        """Fewer than 3 analysts -> NaN."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, targetMeanPrice=120.0,
            numberOfAnalystOpinions=2))
        assert np.isnan(row["price_target_upside"])

    def test_missing_target(self):
        """Missing targetMeanPrice -> NaN."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, targetMeanPrice=np.nan,
            numberOfAnalystOpinions=10))
        assert np.isnan(row["price_target_upside"])

    def test_missing_analyst_count(self):
        """Missing numberOfAnalystOpinions -> NaN."""
        row = _compute_one(_make_rec(
            currentPrice=100.0, targetMeanPrice=120.0,
            numberOfAnalystOpinions=np.nan))
        assert np.isnan(row["price_target_upside"])

    def test_zero_price(self):
        """Zero current price -> NaN (can't divide)."""
        row = _compute_one(_make_rec(
            currentPrice=0.0, targetMeanPrice=120.0,
            numberOfAnalystOpinions=10))
        assert np.isnan(row["price_target_upside"])


# =====================================================================
# SIZE METRIC
# =====================================================================

class TestSizeLogMcap:
    def test_normal(self):
        """size_log_mcap = -log(marketCap)."""
        row = _compute_one(_make_rec(marketCap=100e9))
        expected = -math.log(100e9)
        assert abs(row["size_log_mcap"] - expected) < 0.001

    def test_small_cap_higher_score(self):
        """Smaller companies should have higher (less negative) size_log_mcap."""
        row_small = _compute_one(_make_rec(marketCap=2e9))
        row_big = _compute_one(_make_rec(marketCap=500e9))
        assert row_small["size_log_mcap"] > row_big["size_log_mcap"]

    def test_zero_mcap(self):
        row = _compute_one(_make_rec(marketCap=0))
        assert np.isnan(row["size_log_mcap"])

    def test_missing_mcap(self):
        row = _compute_one(_make_rec(marketCap=np.nan))
        assert np.isnan(row["size_log_mcap"])


# =====================================================================
# INVESTMENT / ASSET GROWTH METRIC
# =====================================================================

class TestAssetGrowth:
    def test_normal(self):
        """asset_growth = (totalAssets - totalAssets_prior) / totalAssets_prior."""
        row = _compute_one(_make_rec(totalAssets=80e9, totalAssets_prior=75e9))
        expected = (80e9 - 75e9) / 75e9
        assert abs(row["asset_growth"] - expected) < 0.001

    def test_shrinking_assets(self):
        """Declining assets → negative asset growth."""
        row = _compute_one(_make_rec(totalAssets=70e9, totalAssets_prior=80e9))
        assert row["asset_growth"] < 0

    def test_zero_prior_assets(self):
        row = _compute_one(_make_rec(totalAssets_prior=0))
        assert np.isnan(row["asset_growth"])

    def test_missing_prior(self):
        row = _compute_one(_make_rec(totalAssets_prior=np.nan))
        assert np.isnan(row["asset_growth"])

    def test_missing_current(self):
        row = _compute_one(_make_rec(totalAssets=np.nan))
        assert np.isnan(row["asset_growth"])
