"""A stock's raw record as it could have been built on a past date, from SEC filings alone.

WHY THIS EXISTS - ``plan/backtest-v2.md`` step 3. ``backtest.py`` v1 scores every month since 2020
on *today's* fundamentals, and the 2026-10-01 measurement put that look-ahead at >= 63.2% of the
name-month panel. ``pit_fundamentals.py`` (2026-10-09) answers single questions - "the latest
total assets filed by date d" - but the engine does not score inputs, it scores a *raw record*:
the ~100 fields ``factor_engine._fetch_single_ticker_inner`` builds from Yahoo's statements and
``compute_metrics`` turns into metrics. A v2 that re-derived each metric from filings would be a
second implementation of 29 formulas, which is the failure row 0.8 of CLAUDE.md exists to stop
(one computation, published - never a second one).

So this module rebuilds the **record**, field for field, from facts filed on or before ``as_of``,
and v2 hands it to the engine's own ``compute_metrics``. One set of formulas scores the live run
and every historical month.

* ``statement_fields(facts, ticker, as_of)`` - every statement field the engine reads, as the
  filings stood on ``as_of``: trailing-twelve-month flows (a fiscal year where one ends at the
  latest quarter, else year-to-date + prior year - prior year-to-date), the same a year earlier,
  balance-sheet instants now and a year ago, the fiscal-year figures Piotroski, the revenue CAGR,
  operating leverage and Beneish read. Of several versions of one figure, the **latest filed by
  then** wins: a restatement filed before ``as_of`` is what an investor would have read.
* ``market_fields(stmt, facts, ticker, as_of, price)`` - the fields Yahoo's ``.info`` serves that
  are fundamentals divided by a price (market value, enterprise value, book value, P/B, trailing
  EPS, ROE, ROA, payout), rebuilt from the same filings at a price the caller supplies.
* ``FIELD_CLASSES`` - every raw field ``compute_metrics`` reads, by what a backtest must do with
  it. ``tests/test_pit_record.py`` parses the engine and fails if it reads a field nobody has
  classified, so a new input cannot leak today's value into a historical month unnoticed.

**Validated against today, not assumed.** Rebuilt as of the newest run's date and scored by the
engine, each weighted metric is compared with the live run's value -
``research/measurements/2026-10-15-pit-record-vs-live.py``,
``research/2026-10-15-pit-record-validation.md``.

**Not wired into anything.** No production module and not ``backtest.py`` v1 may import it (a
half-fixed backtest is what the plan forbids); the tests enforce it, as for ``lookahead.py`` and
``pit_fundamentals.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")
TOL = pd.Timedelta(days=10)          # period ends drift by days (52/53-week years); quarters are ~91 apart
YEAR = pd.Timedelta(days=365)

# Concepts per input, in order of preference (filers switch tags; ASC 606 moved most revenue).
REVENUE = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
           "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense")
C = {
    "revenue": REVENUE,
    "gross_profit": ("GrossProfit",),
    "cogs": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "tax": ("IncomeTaxExpenseBenefit",),
    "pretax": ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
    "interest": ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"),
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "d_and_a": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"),
    "depreciation": ("Depreciation",),
    "amortization": ("AmortizationOfIntangibleAssets",),
    # charges inside operating income that the source's "Operating Income" leaves out: a combined
    # restructuring-and-impairment tag where a filer uses one, otherwise its parts (never both)
    "special_combined": ("RestructuringCostsAndAssetImpairmentCharges",
                         "RestructuringSettlementAndImpairmentProvisions"),
    "restructuring": ("RestructuringCharges",),
    "impairment_combined": ("GoodwillAndIntangibleAssetImpairment",),
    "goodwill_impairment": ("GoodwillImpairmentLoss",),
    "intangible_impairment": ("ImpairmentOfIntangibleAssetsExcludingGoodwill",
                              "ImpairmentOfIntangibleAssetsIndefinitelivedExcludingGoodwill"),
    "asset_impairment": ("AssetImpairmentCharges", "ImpairmentOfLongLivedAssetsHeldForUse"),
    "other_impairment": ("OtherAssetImpairmentCharges",),
    "dividends": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    "sga": ("SellingGeneralAndAdministrativeExpense",),
    "assets": ("Assets",),
    "equity": ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "lt_debt": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"),
    "current_debt": ("DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    "current_debt_parts": ("LongTermDebtCurrent", "ShortTermBorrowings", "CommercialPaper"),
    "leases": ("OperatingLeaseLiabilityNoncurrent", "OperatingLeaseLiabilityCurrent",
               "FinanceLeaseLiabilityNoncurrent", "FinanceLeaseLiabilityCurrent"),
    "lease_total": ("OperatingLeaseLiability",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "sti": ("ShortTermInvestments", "MarketableSecuritiesCurrent",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
    "cash_sti": ("CashCashEquivalentsAndShortTermInvestments",),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "receivables": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
    "ppe": ("PropertyPlantAndEquipmentNet",),
    "shares_out": ("CommonStockSharesOutstanding",),
    "cover_shares": ("EntityCommonStockSharesOutstanding",),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",
                       "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                       "WeightedAverageNumberOfSharesOutstandingBasic"),
}

BENEISH = ("net_receivables", "revenue", "cogs", "current_assets", "ppe", "total_assets",
           "depreciation", "sga", "lt_debt", "current_liab")

# Every raw field compute_metrics reads, by what a point-in-time backtest must do with it.
FIELD_CLASSES = {
    # rebuilt from filings by statement_fields()
    "statement": (
        "totalRevenue", "totalRevenue_prior", "grossProfit", "grossProfit_prior", "ebit", "ebitda",
        "netIncome", "netIncome_prior", "incomeTaxExpense", "pretaxIncome", "interestExpense",
        "_rev_q0", "_rev_q4", "totalRevenue_3yr_ago", "totalRevenue_annual",
        "totalRevenue_annual_prior", "ebit_annual", "ebit_prior", "_ni_a0", "_ni_a1", "_gp_a0",
        "_gp_a1", "_ta_a1", "_ta_a2", "totalAssets", "totalAssets_prior", "totalEquity",
        "totalEquity_prior", "commonEquity", "totalDebt_bs", "longTermDebt", "longTermDebt_prior",
        "currentAssets", "currentAssets_prior", "currentLiabilities", "currentLiabilities_prior",
        "cash_bs", "cash_sti_bs", "sharesBS", "sharesBS_prior", "operatingCashFlow", "capex",
        "dividendsPaid", "da_cf", "_data_source", "_ltm_annualized", "_stmt_date_financials",
        "_quarterly_rejected", "_ltm_rolled", "_ltm_rolled_labels",
    ) + tuple(f"_beneish_{n}{s}" for n in BENEISH for s in ("", "_p")) + (
        "_beneish_net_income", "_beneish_ocf"),
    # fundamentals over a price: market_fields() at the caller's price
    "market": ("marketCap", "enterpriseValue", "totalDebt", "totalCash", "sharesOutstanding",
               "bookValue", "priceToBook", "trailingEps", "payoutRatio"),
    # a price history the backtest already holds, restated per month
    "price_history": ("currentPrice", "price_latest", "price_1m_ago", "price_6m_ago", "price_12m_ago",
                      "volatility_1y", "_vol_daily_sd", "_daily_returns", "_price_series_rejected",
                      "fiftyTwoWeekHigh", "avg_daily_dollar_volume",
                      # the session of the published price (2026-10-22): display and gating only
                      "regularMarketTime", "_price_bar_date"),
    # no free retrospective source: analyst estimates and history, short interest, holdings
    "no_free_history": ("forwardEps", "targetMeanPrice", "targetHighPrice", "targetLowPrice",
                        "numberOfAnalystOpinions", "recommendationMean", "analyst_surprise",
                        "earnings_acceleration", "consecutive_beat_streak", "_eps_quarters",
                        "_eps_quarters_stale", "_fy1_eps_current", "_fy1_eps_90d_ago",
                        "_fy2_eps_current", "_next_fy_end", "shortRatio", "shortPercentOfFloat",
                        "heldPercentInsiders", "dividendRate", "_evol", "_roe5"),
    # who the company is, not what it reported
    "static": ("Ticker", "longName", "shortName", "sector", "industry", "_gics_sector", "_gics_sub"),
}


class Filed:
    """One company's facts as they stood on ``as_of``: of each (concept, start, end), the version
    filed last on or before that day. Nothing filed later is visible."""

    def __init__(self, facts: pd.DataFrame, ticker: str, as_of):
        self.as_of = pd.Timestamp(as_of)
        f = facts[(facts["ticker"] == ticker)]
        f = f[f["form"].isin(FORMS) & (pd.to_datetime(f["filed"]) <= self.as_of)].copy()
        for c in ("start", "end", "filed"):
            f[c] = pd.to_datetime(f[c], errors="coerce")
        f = f.dropna(subset=["end", "val"])
        f["days"] = (f["end"] - f["start"]).dt.days
        # A fiscal year is reported in a 10-K. Of several versions of a fiscal-year context, the
        # latest **10-K** wins over any 10-Q filed after it; a 10-Q's value is used only where no
        # annual report carries that year. Found 2026-10-19 (research/2026-10-19-top-of-ranking-
        # data-audit.md): Comfort Systems' Q1 2026 10-Q tagged its first-quarter figures with a
        # calendar-2025 context, so "latest filed" read FY2025 revenue as $1.83bn against the
        # 10-K's $9.10bn, and its trailing revenue, gross profit, operating income and D&A with it.
        # In the 2026-10-15 cache 29 such 10-Q values for 13 companies disagreed with their 10-K
        # and were never superseded by a later one (FIX, CBOE, HPE, MNST and ZBRA since 2024);
        # restatements still flow, because a restated year reaches the filings as a later 10-K's
        # comparative column. research/measurements/2026-10-19-top-of-ranking-data-audit.py.
        f["_annual_report"] = (f["days"].between(340, 380) & f["form"].str.startswith("10-K")).astype(int)
        # Stable sorts throughout, and an amendment after its original on the same day, so a tie
        # is never settled by an unstable sort's order (2026-10-19, see ``_one``).
        f["_amendment"] = f["form"].str.endswith("/A").astype(int)
        f = f.sort_values(["_annual_report", "filed", "_amendment"], kind="mergesort")
        f = f.drop_duplicates(["concept", "start", "end"], keep="last").sort_values("filed", kind="mergesort")
        self._by = {c: g for c, g in f.groupby("concept")}
        self._fam: dict = {}

    # ---------------------------------------------------------------- one concept
    def _rows(self, concept: str) -> pd.DataFrame | None:
        """One concept's facts - or, for ``"@key"``, the whole family of ``C[key]`` as one series.

        Filers switch tags between filings (NVIDIA's revenue moved from
        ``RevenueFromContractWithCustomerExcludingAssessedTax`` to ``Revenues`` in 2020), so a
        trailing twelve months can need its year-to-date from one tag and the prior year from
        another. The family keeps one fact per period: the largest for revenue (several revenue
        tags can describe one period at different scopes), else the most preferred tag."""
        if not concept.startswith("@"):
            return self._by.get(concept)
        if concept not in self._fam:
            key = concept[1:]
            parts = [self._by[c].assign(_p=i) for i, c in enumerate(C[key]) if c in self._by]
            g = None
            if parts:
                g = pd.concat(parts)
                if key == "revenue":
                    g = g.assign(_a=g["val"].abs()).sort_values("_a")
                else:
                    g = g.sort_values("_p", ascending=False)
                g = g.drop_duplicates(["start", "end"], keep="last")
            self._fam[concept] = g
        return self._fam[concept]

    @staticmethod
    def _one(rows: pd.DataFrame, end=None) -> float | None:
        """The value of the latest-filed row. Rows filed the same day are told apart by how near
        their end is to ``end`` (when given), then by end and value - never by row order.

        Found 2026-10-19, rebuilding the v2 months after the fiscal-year fix: the tie was broken by
        an unstable sort's order, so it moved when unrelated code changed the order. It matters
        where one filing carries two instants within ``TOL`` of each other - an accounting
        change's opening balance beside the period end (911 such pairs in the cache, mostly 1
        January: ASC 842's adoption date beside 31 December). Air Products' FY2018 10-K carries
        both 2018-09-30 and 2018-10-01; its current assets were read from one date and its current
        liabilities from the other, and which one moved with the code. The period asked for is
        the period end, so the nearest end wins."""
        if rows is None or rows.empty:
            return None
        near = -(rows["end"] - end).abs() if end is not None else pd.Series(pd.Timedelta(0), index=rows.index)
        r = rows.assign(_near=near).sort_values(["filed", "_near", "end", "val"], kind="mergesort")
        return float(r.iloc[-1]["val"])

    def _instant1(self, concept, end):
        g = self._rows(concept)
        if g is None:
            return None
        return self._one(g[g["start"].isna() & ((g["end"] - end).abs() <= TOL)], end)

    def _duration1(self, concept, end, lo, hi, start=None):
        g = self._rows(concept)
        if g is None:
            return None
        r = g[g["days"].between(lo, hi) & ((g["end"] - end).abs() <= TOL)]
        if start is not None:
            r = r[(r["start"] - start).abs() <= TOL]
        return r

    def _ttm1(self, concept, end):
        fy = self._duration1(concept, end, 340, 380)
        if fy is None:
            return None
        if not fy.empty:
            return self._one(fy, end)
        ytd = self._duration1(concept, end, 80, 339)
        if ytd.empty:
            return None
        ytd = ytd[ytd["days"] == ytd["days"].max()]
        start = ytd["start"].iloc[0]
        prior_fy = self._duration1(concept, start - pd.Timedelta(days=1), 340, 380)
        prior_ytd = self._duration1(concept, end - YEAR, 80, 339, start=start - YEAR)
        if prior_fy.empty or prior_ytd.empty:
            return None
        return self._one(ytd) + self._one(prior_fy) - self._one(prior_ytd)

    def _quarter1(self, concept, end):
        q = self._duration1(concept, end, 80, 100)
        if q is None:
            return None
        if not q.empty:
            return self._one(q, end)
        # a year-to-date (or a fiscal year, for Q4) less the year-to-date one quarter earlier
        ytd = self._duration1(concept, end, 160, 380)
        if ytd.empty:
            return None
        for _, r in ytd.sort_values("days", ascending=False).iterrows():
            prev = self._duration1(concept, end - pd.Timedelta(days=91), 70, 300, start=r["start"])
            if not prev.empty:
                return float(r["val"]) - self._one(prev)
        return None

    # ---------------------------------------------------------------- an input
    def _pick(self, key, fn, *args, largest=False):
        got = []
        for c in C[key]:
            v = fn(c, *args)
            if v is not None and np.isfinite(v):
                if not largest:
                    return v
                got.append(v)
        if got:
            return max(got, key=abs)
        v = fn("@" + key, *args)                        # pieces from more than one tag
        return v if v is not None and np.isfinite(v) else np.nan

    def instant(self, key, end):
        return self._pick(key, self._instant1, end)

    def ttm(self, key, end):
        return self._pick(key, self._ttm1, end, largest=(key == "revenue"))

    def fy(self, key, end):
        return self._pick(key, lambda c, e: self._one(self._duration1(c, e, 340, 380), e), end,
                          largest=(key == "revenue"))

    def quarter(self, key, end):
        return self._pick(key, self._quarter1, end, largest=(key == "revenue"))

    # ---------------------------------------------------------------- the reporting calendar
    def latest_end(self, keys, lo, hi):
        """The latest period end of a duration between ``lo`` and ``hi`` days among ``keys``."""
        ends = [g.loc[g["days"].between(lo, hi), "end"].max()
                for k in keys for c in C[k] if (g := self._rows(c)) is not None]
        ends = [e for e in ends if pd.notna(e)]
        return max(ends) if ends else None

    def cover_shares_after(self, end, within_days: int = 100):
        """The cover-page share count of the filing that reported ``end``: the earliest count
        dated after the period end and within ``within_days`` of it."""
        best = np.nan
        for c in C["cover_shares"]:
            g = self._rows(c)
            if g is None:
                continue
            g = g[(g["end"] > end) & (g["end"] <= end + pd.Timedelta(days=within_days))]
            if not g.empty:
                return float(g.sort_values(["end", "filed"]).iloc[0]["val"])
        return best

    def latest_instant(self, key):
        """(value, end) of the most recent instant among ``key``'s concepts."""
        best = None
        for c in C[key]:
            g = self._rows(c)
            if g is None:
                continue
            g = g[g["start"].isna()]
            if g.empty:
                continue
            r = g.sort_values(["end", "filed"]).iloc[-1]
            if best is None or r["end"] > best[1]:
                best = (float(r["val"]), r["end"])
        return best


def _nan_sum(*vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(sum(vals)) if vals else np.nan


def _neg(v):
    return -abs(v) if np.isfinite(v) else np.nan


def statement_fields(facts: pd.DataFrame, ticker: str, as_of, filed: Filed | None = None) -> dict:
    """Every ``FIELD_CLASSES['statement']`` field as the filings stood on ``as_of``.

    Empty dict when the company had filed nothing usable by then."""
    F = filed or Filed(facts, ticker, as_of)
    q0 = F.latest_end(("net_income", "revenue", "ocf"), 80, 380)
    fy0 = F.latest_end(("net_income", "revenue"), 340, 380)
    if q0 is None:
        return {}
    q4 = q0 - YEAR
    r: dict = {"_data_source": "quarterly", "_ltm_annualized": False, "_ltm_rolled": False,
               "_ltm_rolled_labels": None, "_stmt_date_financials": str(q0.date())}

    def gp_ttm(end):
        v = F.ttm("gross_profit", end)
        return v if np.isfinite(v) else F.ttm("revenue", end) - F.ttm("cogs", end)

    def special_charges(get, end):
        """Restructuring and impairment charges for the period, as filed - the items the live
        source's "Operating Income" leaves out (research/2026-10-15-earnings-definition.md). A
        combined tag stands for its parts; parts are summed only when no combined tag exists."""
        comb = get("special_combined", end)
        if np.isfinite(comb):
            return abs(comb)
        imp = get("impairment_combined", end)
        if not np.isfinite(imp):
            imp = _nan_sum(get("goodwill_impairment", end), get("intangible_impairment", end),
                           get("asset_impairment", end), get("other_impairment", end))
        total = _nan_sum(abs(imp) if np.isfinite(imp) else np.nan,
                         abs(get("restructuring", end)) if np.isfinite(get("restructuring", end)) else np.nan)
        return total if np.isfinite(total) else 0.0

    def ebit(get, end):
        # Operating income before restructuring and impairment charges, which is what the live
        # source's "Operating Income" row is (as reported less its "Special Income Charges";
        # 2026-10-15). Where operating income is not tagged: EBIT = pretax income + interest, as
        # the fetch falls back from that row to the source's "EBIT" row.
        v = get("operating_income", end)
        if np.isfinite(v):
            return v + special_charges(get, end)
        pt, ie = get("pretax", end), get("interest", end)
        return pt + abs(ie) if np.isfinite(pt) and np.isfinite(ie) else np.nan

    def gp_fy(end):
        v = F.fy("gross_profit", end)
        return v if np.isfinite(v) else F.fy("revenue", end) - F.fy("cogs", end)

    # trailing twelve months, now and a year earlier
    # The ``*_prior`` flows are the fiscal year before the latest one, not the trailing twelve
    # months a year earlier: that is what the engine's fetch fills them with (measured 2026-10-15:
    # 503 of 503 live records), and no weighted metric reads them since 2026-10-09.
    fy1 = fy0 - YEAR if fy0 is not None else None
    r["totalRevenue"] = F.ttm("revenue", q0)
    r["totalRevenue_prior"] = F.fy("revenue", fy1) if fy1 is not None else np.nan
    r["grossProfit"] = gp_ttm(q0)
    r["grossProfit_prior"] = gp_fy(fy1) if fy1 is not None else np.nan
    r["ebit"] = ebit(F.ttm, q0)
    r["da_cf"] = F.ttm("d_and_a", q0)
    if not np.isfinite(r["da_cf"]):
        # filers without the combined tag: depreciation plus amortisation of intangibles
        r["da_cf"] = _nan_sum(F.ttm("depreciation", q0), F.ttm("amortization", q0))
    if not np.isfinite(r["da_cf"]) and fy0 is not None:
        # no quarterly D&A: the last fiscal year's, as the engine's fetch does (``_da_annual``)
        r["da_cf"] = F.fy("d_and_a", fy0)
    r["ebitda"] = r["ebit"] + r["da_cf"] if np.isfinite(r["ebit"]) and np.isfinite(r["da_cf"]) else np.nan
    r["netIncome"] = F.ttm("net_income", q0)
    r["netIncome_prior"] = F.fy("net_income", fy1) if fy1 is not None else np.nan
    r["incomeTaxExpense"] = F.ttm("tax", q0)
    r["pretaxIncome"] = F.ttm("pretax", q0)
    r["costOfRevenue"] = F.ttm("cogs", q0)
    r["interestExpense"] = abs(F.ttm("interest", q0)) if np.isfinite(F.ttm("interest", q0)) else np.nan
    r["operatingCashFlow"] = F.ttm("ocf", q0)
    r["capex"] = _neg(F.ttm("capex", q0))
    r["dividendsPaid"] = _neg(F.ttm("dividends", q0))

    # the latest quarter against the same quarter a year earlier (revenue_growth)
    rq0, rq4 = F.quarter("revenue", q0), F.quarter("revenue", q4)
    if np.isfinite(rq0) and np.isfinite(rq4) and rq4 > 0:
        r["_rev_q0"], r["_rev_q4"] = rq0, rq4
        r["_rev_q0_date"], r["_rev_q4_date"] = str(q0.date()), str(q4.date())

    # fiscal years
    if fy0 is not None:
        r["totalRevenue_annual"] = F.fy("revenue", fy0)
        r["totalRevenue_annual_prior"] = F.fy("revenue", fy0 - YEAR)
        r["totalRevenue_3yr_ago"] = F.fy("revenue", fy0 - 3 * YEAR)
        r["ebit_annual"] = ebit(F.fy, fy0)
        r["ebit_prior"] = ebit(F.fy, fy0 - YEAR)
        r["_ni_a0"], r["_ni_a1"] = F.fy("net_income", fy0), F.fy("net_income", fy0 - YEAR)
        r["_gp_a0"], r["_gp_a1"] = gp_fy(fy0), gp_fy(fy0 - YEAR)
        r["_ta_a1"] = F.instant("assets", fy0 - YEAR)
        r["_ta_a2"] = F.instant("assets", fy0 - 2 * YEAR)
        # Beneish: annual statements, this fiscal year and the one before
        for name, key, kind in (("net_receivables", "receivables", "i"), ("revenue", "revenue", "f"),
                                ("cogs", "cogs", "f"), ("current_assets", "current_assets", "i"),
                                ("ppe", "ppe", "i"), ("total_assets", "assets", "i"),
                                ("depreciation", "d_and_a", "f"), ("sga", "sga", "f"),
                                ("lt_debt", "lt_debt", "i"), ("current_liab", "current_liabilities", "i")):
            get = F.instant if kind == "i" else F.fy
            r[f"_beneish_{name}"] = get(key, fy0)
            r[f"_beneish_{name}_p"] = get(key, fy0 - YEAR)
        r["_beneish_net_income"] = F.fy("net_income", fy0)
        r["_beneish_ocf"] = F.fy("ocf", fy0)

    # balance sheet at the latest quarter, and a year earlier
    for field, key in (("totalAssets", "assets"), ("totalEquity", "equity"),
                       ("longTermDebt", "lt_debt"), ("currentAssets", "current_assets"),
                       ("currentLiabilities", "current_liabilities"), ("sharesBS", "shares_out")):
        r[field] = F.instant(key, q0)
        r[f"{field}_prior"] = F.instant(key, q4)
    for field, end in (("sharesBS", q0), ("sharesBS_prior", q4)):
        if not np.isfinite(r[field]):
            r[field] = F.cover_shares_after(end)
    # Common shareholders' equity, which the bank P/B divides market value by (2026-10-20). The
    # facts cache carries ``StockholdersEquity`` and no common-equity concept, and not the
    # preferred-stock items the deduction would need either, so v2's bank P/B is market value over
    # *stockholders'* equity and this line is where that is stated. It is the same number
    # ``market_fields`` already built as ``priceToBook`` (price x shares / equity = market value /
    # equity), so the scored construction changing left every v2 figure unmoved.
    r["commonEquity"] = r["totalEquity"]
    r["cash_bs"] = F.instant("cash", q0)
    cs = F.instant("cash_sti", q0)
    r["cash_sti_bs"] = cs if np.isfinite(cs) else _nan_sum(r["cash_bs"], F.instant("sti", q0))
    cur = F.instant("current_debt", q0)
    if not np.isfinite(cur):
        cur = _nan_sum(*(F._instant1(c, q0) for c in C["current_debt_parts"]))
    # Yahoo's "Total Debt" includes lease liabilities (ASC 842): borrowings plus leases
    leases = _nan_sum(*(F._instant1(c, q0) for c in C["leases"]))
    if not np.isfinite(leases):
        leases = F.instant("lease_total", q0)
    r["totalDebt_bs"] = _nan_sum(r["longTermDebt"], cur, leases)
    return r


def split_after(splits: pd.DataFrame | None, ticker: str, as_of) -> float:
    """How many of today's shares one share on ``as_of`` became: the product of the split ratios
    dated after it. A split-adjusted price panel is in today's shares; a filing is in the shares
    of its day, so per-share figures from it must be put on the price's basis (NVDA's 2022
    filings are in pre-10:1 shares)."""
    if splits is None or splits.empty:
        return 1.0
    s = splits[(splits["ticker"] == ticker) & (pd.to_datetime(splits["date"]) > pd.Timestamp(as_of))]
    return float(np.prod(s["ratio"])) if not s.empty else 1.0


def market_fields(stmt: dict, facts: pd.DataFrame, ticker: str, as_of, price: float,
                  filed: Filed | None = None, split: float = 1.0) -> dict:
    """The ``FIELD_CLASSES['market']`` fields at ``price``, from the same filings.

    ``split`` (``split_after``) converts the filings' share counts and EPS to the basis of a
    split-adjusted ``price``; 1.0 when the price is the day's own."""
    F = filed or Filed(facts, ticker, as_of)
    out = {k: np.nan for k in FIELD_CLASSES["market"]}
    if not stmt or not (price and np.isfinite(price) and price > 0):
        return out
    got = [x for x in (F.latest_instant("shares_out"), F.latest_instant("cover_shares")) if x]
    shares = max(got, key=lambda x: x[1])[0] if got else stmt.get("sharesBS", np.nan)
    shares = shares * split
    debt, cash = stmt.get("totalDebt_bs", np.nan), stmt.get("cash_sti_bs", np.nan)
    if not np.isfinite(cash):
        cash = stmt.get("cash_bs", np.nan)
    ni, eq, ta = stmt.get("netIncome", np.nan), stmt.get("totalEquity", np.nan), stmt.get("totalAssets", np.nan)
    out["sharesOutstanding"] = shares
    out["totalDebt"], out["totalCash"] = debt, cash
    if np.isfinite(shares) and shares > 0:
        out["marketCap"] = price * shares
        out["enterpriseValue"] = out["marketCap"] + (debt if np.isfinite(debt) else 0.0) - \
            (cash if np.isfinite(cash) else 0.0)
        if np.isfinite(eq) and eq > 0:
            out["bookValue"] = eq / shares
            out["priceToBook"] = price / out["bookValue"]
    q0 = pd.Timestamp(stmt["_stmt_date_financials"])
    eps = F.ttm("eps", q0)
    if np.isfinite(eps):
        out["trailingEps"] = eps / split
    else:
        dil = F._pick("diluted_shares", lambda c, e: F._one(F._duration1(c, e, 80, 380)), q0)
        if np.isfinite(ni) and np.isfinite(dil) and dil > 0:
            out["trailingEps"] = ni / (dil * split)
    # No returnOnEquity / returnOnAssets since 2026-10-20: compute_metrics no longer reads
    # them. The bank ROE and ROA are now built from netIncome and the two balance sheets,
    # which are statement fields and so are restated by statement_fields() as of the month
    # being scored - a stronger point-in-time basis than this function could give them, and
    # one fewer construction (this built them on ENDING balances, a third definition).
    div = stmt.get("dividendsPaid", np.nan)
    if np.isfinite(ni) and ni > 0 and np.isfinite(div):
        out["payoutRatio"] = abs(div) / ni
    return out


def record_as_of(base: dict, facts: pd.DataFrame, as_of, price: float, split: float = 1.0) -> dict:
    """``base`` (a raw record: static, price-history and estimate fields) with every statement
    and market field replaced by what the filings said on ``as_of``. A field the filings cannot
    supply becomes NaN - never the base record's (today's) value."""
    t = base["Ticker"]
    F = Filed(facts, t, as_of)
    stmt = statement_fields(facts, t, as_of, filed=F)
    rec = dict(base)
    for k in FIELD_CLASSES["statement"] + FIELD_CLASSES["market"]:
        rec[k] = np.nan
    for k in [k for k in rec if k.startswith("_beneish_") or k.startswith("_rev_q")]:
        rec[k] = np.nan
    rec["_ltm_annualized"], rec["_data_source"] = False, "quarterly"
    # rebuilt from filings, so neither the 4/3 fabrication nor the roll-forward applies
    rec["_ltm_rolled"], rec["_ltm_rolled_labels"] = False, None
    rec.update(stmt)
    rec.update(market_fields(stmt, facts, t, as_of, price, filed=F, split=split))
    rec["currentPrice"] = price
    return rec
