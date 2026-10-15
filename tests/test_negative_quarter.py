"""A quarterly statement derived across a restatement is not summed (``factor_engine.quarterly_statement_defect``).

Found 2026-10-15 on DuPont: the source derived Q4-2025 as the restated year less the un-restated
nine months, so Q4 revenue read -$1.871bn and every trailing-twelve-month flow inherited it
(revenue $4.70bn against ~$6.99bn filed). The fetch now rejects such a statement - income and
cash flow alike - uses the annual statements, and records why on the stock.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402

Q = [pd.Timestamp(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30")]


def _dd_quarterly():
    """DuPont's quarterly income statement as the source served it on 2026-10-15 ($bn)."""
    return pd.DataFrame({q: v for q, v in zip(Q, [
        [1.819e9, 0.639e9, 0.260e9, 0.143e9],
        [1.681e9, 0.602e9, 0.232e9, 0.161e9],
        [-1.871e9, -1.055e9, -0.652e9, -0.126e9],
        [3.072e9, 1.195e9, 0.547e9, -0.123e9],
        [1.749e9, 0.606e9, 0.217e9, 0.059e9]])},
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"])


def test_a_negative_revenue_quarter_in_the_window_is_a_defect():
    why = fe.quarterly_statement_defect(_dd_quarterly())
    assert why is not None and "-1.871bn" in why and "2025-12-31" in why


def test_a_clean_statement_and_an_old_negative_quarter_are_not():
    clean = _dd_quarterly().copy()
    clean[Q[2]] = [1.693e9, 0.6e9, 0.25e9, 0.1e9]
    clean[Q[3]] = [1.795e9, 0.6e9, 0.25e9, 0.1e9]
    assert fe.quarterly_statement_defect(clean) is None
    old = clean.copy()
    old[Q[4]] = [-1.0e9, 0, 0, 0]                  # fifth quarter back: outside the summed window
    assert fe.quarterly_statement_defect(old) is None
    assert fe.quarterly_statement_defect(None) is None
    assert fe.quarterly_statement_defect(pd.DataFrame()) is None
    assert fe.quarterly_statement_defect(pd.DataFrame({Q[0]: [1.0]}, index=["Net Income"])) is None


class _FakeTicker:
    """Enough of yfinance for the statement blocks of the fetch."""

    def __init__(self):
        self.info = {"symbol": "DD", "marketCap": 3.0e10, "currentPrice": 70.0}
        self.quarterly_financials = _dd_quarterly()
        self.quarterly_cashflow = pd.DataFrame({q: [9.9e9] for q in Q[:4]}, index=["Operating Cash Flow"])
        a = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
        self.financials = pd.DataFrame({a[0]: [6.849e9, 2.5e9, 0.95e9, 0.4e9],
                                        a[1]: [6.719e9, 2.4e9, 0.90e9, 0.3e9]},
                                       index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"])
        self.cashflow = pd.DataFrame({a[0]: [1.2e9], a[1]: [1.1e9]}, index=["Operating Cash Flow"])

    def __getattr__(self, name):
        return None


def test_the_fetch_uses_the_annual_statements_and_says_why(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", lambda *a, **k: _FakeTicker())
    rec = fe._fetch_single_ticker_inner("DD")
    assert rec["_quarterly_rejected"].startswith("quarterly revenue of -1.871bn")
    assert rec["totalRevenue"] == 6.849e9                  # the restated year, not 4.701bn
    assert rec["ebit"] == 0.95e9
    assert rec["_data_source"] == "annual"
    assert rec["operatingCashFlow"] == 1.2e9               # the cash-flow quarters are rejected too


def test_the_rejection_happens_before_any_quarter_is_summed():
    src = (ROOT / "factor_engine.py").read_text(encoding="utf-8")
    body = src[src.index("def _fetch_single_ticker_inner"):]
    i = body.index("_q_defect = quarterly_statement_defect(q_fins)")
    assert i < body.index('rec["totalRevenue"]           = _stmt_val_ltm(q_fins')
    assert i < body.index('rec["operatingCashFlow"]      = _stmt_val_ltm(q_cf')
    assert "q_cf = None" in body[i:i + 600]
