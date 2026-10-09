"""Fundamentals as they were knowable on a date - the data layer backtest v2 needs.

WHY THIS EXISTS - ``plan/backtest-v2.md`` step 3. ``backtest.py`` v1 scores every month since 2020
on *today's* fundamentals: a negative reporting lag of up to 80 months on 49.0 points of
composite weight. SEC XBRL facts carry the date each figure was **filed**, so "what was the latest
figure filed by date d" has an exact answer from free data. Measured 2026-10-09
(``research/2026-10-09-pit-fundamentals-census.md``): ~95-97% of name-months for net income,
assets, equity, revenue and operating cash flow; 68-85% for operating income, D&A, debt, cash and
capex; 39% for a gross-profit tag.

**Three questions, all answered only from facts filed on or before ``as_of``:**

* ``instant(ticker, input, as_of)`` - the latest balance-sheet value (assets, equity, debt...):
  the most recent period end, and of the filings reporting it, the latest one filed by then (a
  restatement filed before ``as_of`` is what an investor would have read; one filed after is not).
* ``annual(ticker, input, as_of)`` - the latest fiscal-year flow (a ~12-month duration).
* ``quarter(ticker, input, as_of)`` - the latest single-quarter flow (a ~3-month duration).

Every answer comes back with the period end and the filing date it came from, so a v2 can show
its own reporting lag.

**Not wired into anything.** ``backtest.py`` v1 must not import it - a half-fixed backtest is what
the plan forbids (``tests/test_pit_fundamentals.py`` checks, as ``test_lookahead.py`` does for
``lookahead.py``). It is for a v2 that rebuilds every fundamentals metric from it in one piece.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FACTS_PATH = ROOT / "data" / "sec" / "pit" / "facts.parquet"

# Input -> XBRL us-gaap concepts; any of them counts (filers switch tags over time). Kept in step
# with research/measurements/2026-10-09-xbrl-point-in-time-census.py, which builds the cache.
INPUTS = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "total_assets": ("Assets",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "long_term_debt": ("LongTermDebt", "LongTermDebtNoncurrent"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "d_and_a": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
}
ANNUAL_DAYS = (340, 380)
QUARTER_DAYS = (80, 100)


@dataclass(frozen=True)
class Fact:
    value: float
    period_end: pd.Timestamp
    filed: pd.Timestamp


class PointInTime:
    """Answers from a facts table (columns: ticker, concept, start, end, filed, form, val)."""

    def __init__(self, facts: pd.DataFrame):
        f = facts.copy()
        for c in ("start", "end", "filed"):
            f[c] = pd.to_datetime(f[c], errors="coerce")
        f = f.dropna(subset=["end", "filed", "val"])
        f["days"] = (f["end"] - f["start"]).dt.days
        self._by = {k: g.sort_values(["end", "filed"]) for k, g in f.groupby("ticker")}

    @classmethod
    def from_cache(cls, path: Path | None = None) -> "PointInTime":
        return cls(pd.read_parquet(path or FACTS_PATH))

    def _rows(self, ticker: str, input_name: str, as_of) -> pd.DataFrame:
        g = self._by.get(ticker)
        if g is None:
            return g
        as_of = pd.Timestamp(as_of)
        return g[g["concept"].isin(INPUTS[input_name]) & (g["filed"] <= as_of)]

    @staticmethod
    def _latest(rows: pd.DataFrame | None) -> Fact | None:
        if rows is None or rows.empty:
            return None
        end = rows["end"].max()
        r = rows[rows["end"] == end].sort_values("filed").iloc[-1]   # latest version known by then
        return Fact(float(r["val"]), r["end"], r["filed"])

    def instant(self, ticker: str, input_name: str, as_of) -> Fact | None:
        rows = self._rows(ticker, input_name, as_of)
        return self._latest(None if rows is None else rows[rows["start"].isna()])

    def annual(self, ticker: str, input_name: str, as_of) -> Fact | None:
        rows = self._rows(ticker, input_name, as_of)
        return self._latest(None if rows is None else rows[rows["days"].between(*ANNUAL_DAYS)])

    def quarter(self, ticker: str, input_name: str, as_of) -> Fact | None:
        rows = self._rows(ticker, input_name, as_of)
        return self._latest(None if rows is None else rows[rows["days"].between(*QUARTER_DAYS)])
