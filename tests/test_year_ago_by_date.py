"""The year-ago period is chosen by its DATE, never by its column position (2026-10-19).

The fetch used to take "a year earlier" as column four of a newest-first quarterly statement: the
fifth quarter with revenue for revenue growth, column four of the balance sheet for asset growth,
average equity and Piotroski 5-7, and quarters four to seven for the prior twelve months. That is a
year earlier only when the source carries every quarter in between. Berkshire Hathaway's quarterly
statements omit September 2025, so column four was March 2025: the balance-sheet and revenue checks
rejected it (and the June 2025 period in column three went unused), while the prior twelve months
had no date check at all and would have summed the wrong four quarters.

Measured on the 2026-10-19 run (``research/measurements/2026-10-19-year-ago-by-date.py``): the old
rule reproduces the run's raw fetch for 501 of 501 stocks; choosing by date changes the inputs of
three (BRK-B, FERG, SKYD) and eight metric values, top 25 unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402

T = pd.Timestamp
# Berkshire's quarters on 2026-10-19: 2025-09-30 is absent.
BRK_Q = [T(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-06-30", "2025-03-31")]
FULL_Q = [T(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30",
                         "2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30")]


def _stmt(cols, rows: dict):
    return pd.DataFrame({c: [rows[r][i] for r in rows] for i, c in enumerate(cols)}, index=list(rows))


# ---------------------------------------------------------------- the rule

def test_a_missing_quarter_does_not_shift_the_year_ago_period():
    assert fe._year_ago_position(BRK_Q) == 3                       # 2025-06-30, not column four
    assert fe._year_ago_position(BRK_Q, tol=fe.YEAR_AGO_TOL_BALANCE_DAYS) == 3


def test_a_complete_statement_still_uses_column_four():
    """Where nothing is missing the date rule and the old position agree - no change for them."""
    assert fe._year_ago_position(FULL_Q) == 4


def test_no_period_a_year_earlier_is_none_not_a_guess():
    cols = [T("2026-06-30"), T("2026-03-31"), T("2025-12-31"), T("2025-03-31")]
    assert fe._year_ago_position(cols) is None
    assert fe._year_ago_position(cols, tol=fe.YEAR_AGO_TOL_BALANCE_DAYS) is None


def test_a_52_53_week_calendar_is_a_year():
    cols = [T("2026-06-27"), T("2026-03-28"), T("2025-12-27"), T("2025-09-27"), T("2025-06-21")]
    assert 371 == (cols[0] - cols[4]).days
    assert fe._year_ago_position(cols) == 4


def test_the_tolerances_are_the_windows_each_comparison_already_used():
    """Flows 350-380 days (the revenue check since 2026-10-09), balance sheets 330-400. AutoZone's
    third quarter is labelled 2026-04-30 on its balance sheet and 2025-05-31 a year before: 334
    days, which the run uses today and the balance-sheet window must keep."""
    assert (fe.YEAR_AGO_TOL_FLOW_DAYS, fe.YEAR_AGO_TOL_BALANCE_DAYS) == (15, 35)
    azo = [T("2026-04-30"), T("2026-02-28"), T("2025-11-30"), T("2025-08-31"), T("2025-05-31")]
    assert fe._year_ago_position(azo, tol=fe.YEAR_AGO_TOL_BALANCE_DAYS) == 4
    assert fe._year_ago_position(azo) is None                      # too far for a seasonal flow
    ferg = [T("2026-06-30"), T("2026-03-31"), T("2025-12-31"), T("2025-10-31"), T("2025-07-31")]
    assert fe._year_ago_position(ferg) is None                     # Apr-Jun against May-Jul
    assert fe._year_ago_position(ferg, tol=fe.YEAR_AGO_TOL_BALANCE_DAYS) == 4


def test_undated_or_empty_columns_give_none():
    assert fe._year_ago_position(["a", "b", "c", "d", "e"]) is None
    assert fe._year_ago_position([]) is None


# ---------------------------------------------------------------- the three call sites

def test_revenue_growth_pairs_the_same_quarter_a_year_earlier():
    q = _stmt(BRK_Q, {"Total Revenue": [117.9e9, 90e9, 95e9, 98.9e9, 89e9]})
    v0, v4, d0, d4 = fe._year_ago_quarter(q, "Total Revenue")
    assert (v0, v4, d0, d4) == (117.9e9, 98.9e9, T("2026-06-30"), T("2025-06-30"))


def test_revenue_pairs_by_date_even_when_a_blank_column_is_listed():
    """The source lists 2025-09-30 with almost nothing in it; revenue reads the quarters that
    carry a value, and the year-ago one is still the June quarter."""
    cols = FULL_Q[:6]
    q = _stmt(cols, {"Total Revenue": [117.9e9, 90e9, 95e9, np.nan, 98.9e9, 89e9]})
    assert fe._year_ago_quarter(q, "Total Revenue")[3] == T("2025-06-30")


def test_no_year_ago_quarter_leaves_revenue_growth_to_the_annual_basis():
    q = _stmt(BRK_Q[:3] + [BRK_Q[4]], {"Total Revenue": [1.0, 2.0, 3.0, 4.0]})
    assert fe._year_ago_quarter(q, "Total Revenue") is None
    assert fe._year_ago_quarter(None) is None


def test_the_prior_twelve_months_start_a_year_before_the_latest_quarter():
    """Positions four to seven are four ADJACENT quarters here - the old rule summed them without
    complaint - but they end in March 2025, fifteen months before the latest quarter."""
    cols = [T(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-06-30", "2025-03-31",
                           "2024-12-31", "2024-09-30", "2024-06-30")]
    vals = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
    q = _stmt(cols, {"Net Income": vals})
    assert fe._stmt_val_ltm(q, "Net Income", offset=4) == 6.0 + 5.0 + 4.0 + 3.0   # the old reading
    assert fe._stmt_val_ltm_year_ago(q, "Net Income") == 7.0 + 6.0 + 5.0 + 4.0    # Jul 2024-Jun 2025


def test_the_prior_twelve_months_equal_column_four_on_a_complete_statement():
    q = _stmt(FULL_Q[:8], {"Net Income": [float(i) for i in range(8, 0, -1)]})
    assert fe._stmt_val_ltm_year_ago(q, "Net Income") == fe._stmt_val_ltm(q, "Net Income", offset=4)


def test_the_prior_twelve_months_are_never_annualised():
    q = _stmt(BRK_Q, {"Net Income": [1.0, 2.0, 3.0, 4.0, 5.0]})       # June 2025 + one quarter
    assert np.isnan(fe._stmt_val_ltm_year_ago(q, "Net Income"))
    assert np.isnan(fe._stmt_val_ltm_year_ago(None, "Net Income"))
    assert np.isnan(fe._stmt_val_ltm_year_ago(q, "Not A Line"))


# ---------------------------------------------------------------- through the fetch

class _FakeTicker:
    """Enough of yfinance for the statement blocks of the fetch, with Berkshire's hole in both the
    income statement and the balance sheet."""

    def __init__(self):
        self.info = {"symbol": "BRK-B", "marketCap": 1.1e12, "currentPrice": 500.0}
        rows = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
        self.quarterly_financials = pd.DataFrame(
            {BRK_Q[0]: [117.9e9, 30e9, 20e9, 24e9], BRK_Q[1]: [100e9, 28e9, 18e9, 16e9],
             BRK_Q[2]: [105e9, 29e9, 19e9, 20e9], BRK_Q[3]: [98.9e9, 27e9, 17e9, 13e9],
             BRK_Q[4]: [90e9, 26e9, 16e9, 5e9]}, index=rows)
        self.financials = pd.DataFrame({T("2025-12-31"): [400e9, 110e9, 70e9, 67e9],
                                        T("2024-12-31"): [380e9, 105e9, 68e9, 89e9]}, index=rows)
        bs_rows = ["Total Assets", "Stockholders Equity", "Long Term Debt", "Ordinary Shares Number"]
        self.quarterly_balance_sheet = pd.DataFrame(
            {BRK_Q[0]: [1263e9, 720e9, 120e9, 1.43e6], BRK_Q[1]: [1250e9, 710e9, 121e9, 1.43e6],
             BRK_Q[2]: [1240e9, 700e9, 122e9, 1.43e6], BRK_Q[3]: [1164e9, 668e9, 124e9, 1.44e6],
             BRK_Q[4]: [1150e9, 650e9, 125e9, 1.44e6]}, index=bs_rows)

    def __getattr__(self, name):
        return None


@pytest.fixture
def brk(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: _FakeTicker())
    return fe._fetch_single_ticker_inner("BRK-B")


def test_the_fetch_reads_the_balance_sheet_a_year_earlier_by_date(brk):
    assert brk["totalAssets_prior"] == 1164e9                    # June 2025, not March (1150e9)
    assert brk["totalEquity_prior"] == 668e9
    assert brk["longTermDebt_prior"] == 124e9
    assert brk["sharesBS_prior"] == 1.44e6


def test_the_fetch_pairs_revenue_with_the_june_quarter(brk):
    assert (brk["_rev_q0"], brk["_rev_q4"]) == (117.9e9, 98.9e9)
    assert (brk["_rev_q0_date"], brk["_rev_q4_date"]) == ("2026-06-30", "2025-06-30")


def test_the_fetch_falls_back_to_the_annual_prior_when_no_year_ago_twelve_months(brk):
    assert brk["netIncome_prior"] == 89e9                        # annual column one, as before
    assert brk["totalRevenue_prior"] == 380e9


def test_the_fetch_takes_no_year_ago_column_by_position():
    """A tripwire on the three forms the positional rule took."""
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def _fetch_single_ticker_inner"):src.index("def fetch_all_tickers")]
    assert "offset=4" not in body
    assert "_qc[4]" not in body
    assert "_bs_prior_col = 4" not in body
    assert "_year_ago_position(_bs_cols, tol=YEAR_AGO_TOL_BALANCE_DAYS)" in body
