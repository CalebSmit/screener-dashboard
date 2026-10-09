"""Five fiscal years of return on equity from the SEC's own 10-K figures, for earnings variability.

WHY THIS EXISTS - 2026-10-09, ``research/2026-10-09-operating-leverage.md`` section 6. Both
published practitioner definitions of quality measure *durability* as how variable earnings have
been: MSCI's Quality Indexes use the five-year standard deviation of EPS growth, and AQR's
Quality Minus Junk uses the standard deviation of ROE (60 quarters, or **five fiscal years of
annual ROE** when quarterly data is unavailable - Asness, Frazzini & Pedersen 2019, *Review of
Accounting Studies* 24, p. 74). Yahoo's statements carry four annual years, so no faithful
version could be computed. The SEC carries a decade.

**Source.** Each company's ``companyfacts`` (one request per company, cached weekly in
``data/sec/pit/facts.parquet`` - the same cache the backtest's point-in-time layer reads). For
each of its last five fiscal years: net income for the year from its 10-K, and shareholders'
equity at that same fiscal-year end. (The XBRL *frames* API was used first, for one morning; it
mis-scaled a Con Edison figure by 1,000x and paired fiscal-year income with calendar-year-end
equity - see ``refresh_companyfacts``.)

**Metric.** ``earnings_variability`` = sample standard deviation of the five ROEs, **all five
required and consecutive** (AQR's rule); missing when equity is zero or negative in any of them.
Lower = steadier.

**A candidate (weight 0).** Computed, published and shown with its five years; it moves no score
(``research/2026-10-09-earnings-variability-candidate.md`` explains why it stays unweighted).

Needs the SEC contact identity (``insider_activity.user_agent``); without one it returns nothing
and the metric is missing for every stock, which the scorer skips.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EQ_CONCEPTS = ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
YEARS = 5


def last_complete_year(today: date) -> int:  # used by the measurement scripts
    """The latest calendar year whose annual reports are all in (10-Ks land by ~March)."""
    return today.year - 1 if today.month >= 4 else today.year - 2


def years_for(today: date, n: int = YEARS) -> list[int]:
    y = last_complete_year(today)
    return list(range(y - n + 1, y + 1))


FACTS_PATH = ROOT / "data" / "sec" / "pit" / "facts.parquet"
FACTS_MAX_AGE_DAYS = 7
# Concepts kept from each company's companyfacts: everything the point-in-time layer
# (pit_fundamentals.INPUTS) reads, so one weekly download serves both.
FACT_CONCEPTS = (
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
    "RevenuesNetOfInterestExpense", "GrossProfit", "CostOfRevenue", "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold", "OperatingIncomeLoss", "NetIncomeLoss", "ProfitLoss", "Assets",
    "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "LongTermDebt", "LongTermDebtNoncurrent", "CashAndCashEquivalentsAtCarryingValue",
    "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
    "DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "AssetsCurrent",
    "LiabilitiesCurrent",
)
# Concepts reported in shares rather than dollars (valuation_history.py's market value).
SHARE_CONCEPTS = ("WeightedAverageNumberOfDilutedSharesOutstanding",)


def refresh_companyfacts(tickers: list[str], edgar=None, log=print, force: bool = False,
                         allow_write: bool = True):
    """Every 10-K / 10-Q fact for ``FACT_CONCEPTS``, one ``companyfacts`` request per company,
    cached in ``FACTS_PATH`` and refreshed at most weekly (~500 requests, ~3 minutes).

    Why not the frames API used first: on 2026-10-09 its NetIncomeLoss CY2024 frame gave Con
    Edison 1,820,000 where the 10-K says 1,820,000,000 - a mis-scaled fact the frame chose -
    and frames align a fiscal year to a calendar year, pairing a June-year-end company's income
    with its December equity. A company's own 10-K figures avoid both."""
    import pandas as pd
    # A subset run (``--tickers``) never rewrites the universe-wide cache: it would leave a
    # 32-company file stamped fresh for a week (review, 2026-10-09).
    if not allow_write or (not force and FACTS_PATH.exists()
                           and time.time() - FACTS_PATH.stat().st_mtime < FACTS_MAX_AGE_DAYS * 86400):
        return pd.read_parquet(FACTS_PATH) if FACTS_PATH.exists() else None
    from insider_activity import Edgar, ticker_map, user_agent
    if edgar is None:
        ua = user_agent()
        if not ua:
            log("  SEC companyfacts: no SEC identity configured - skipped")
            return pd.read_parquet(FACTS_PATH) if FACTS_PATH.exists() else None
        edgar = Edgar(ua)
    cmap = ticker_map(edgar)
    rows, failed = [], 0
    for t in tickers:
        cik = cmap.get(str(t).upper())
        if cik is None:
            continue
        try:
            j = edgar.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json").json()
        except Exception:  # noqa: BLE001 - one company failing must not lose the rest
            failed += 1
            continue
        gaap = (j.get("facts") or {}).get("us-gaap") or {}
        for c, unit in [(c, "USD") for c in FACT_CONCEPTS] + [(c, "shares") for c in SHARE_CONCEPTS]:
            for f in ((gaap.get(c) or {}).get("units") or {}).get(unit, []):
                if f.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                    rows.append((t, c, f.get("start"), f.get("end"), f.get("filed"), f.get("form"),
                                 f.get("fp"), f.get("val")))
    got = len({r[0] for r in rows})
    asked = sum(1 for t in tickers if cmap.get(str(t).upper()) is not None)
    # A refresh that reached under 90% of the companies it asked for (a network outage, a rate
    # limit) must not replace a complete cache with a partial one for a week.
    if not rows or (FACTS_PATH.exists() and asked and got < 0.9 * asked):
        log(f"  SEC companyfacts: refresh reached {got} of {asked} companies - keeping the existing cache")
        return pd.read_parquet(FACTS_PATH) if FACTS_PATH.exists() else None
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "form", "fp", "val"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FACTS_PATH)
    log(f"  SEC companyfacts: {len(df):,} facts for {df['ticker'].nunique()} companies ({failed} failed)")
    return df


NI_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")


MAX_STALENESS_DAYS = 550    # latest fiscal year must end within ~18 months of today


def roe_history(tickers: list[str], today: date | None = None, facts=None, log=print,
                refresh: bool = True) -> dict:
    """``{ticker: [[fiscal_year_end, net_income, equity, roe_or_None], ...]}``, oldest first.

    The last ``YEARS`` fiscal years from the company's own 10-K figures: net income for each
    fiscal year (a 340-380 day duration; the latest version filed) and shareholders' equity at
    that same fiscal-year end - so income and equity always describe the same date."""
    import pandas as pd
    today = today or date.today()
    if facts is None:
        facts = refresh_companyfacts(tickers, log=log, allow_write=refresh)
    if facts is None or len(facts) == 0:
        return {}
    f = facts[facts["form"].str.startswith("10-K")].copy()
    f = f[f["end"] <= pd.Timestamp(today)]
    f["days"] = (f["end"] - f["start"]).dt.days
    out: dict = {}
    for t, g in f.groupby("ticker"):
        if t not in set(tickers):
            continue
        # Per fiscal year, the first concept that reports it: filers move between
        # NetIncomeLoss and ProfitLoss over the years, so one tag per company loses years.
        yearly = g[g["concept"].isin(NI_CONCEPTS) & g["days"].between(340, 380)].copy()
        if yearly.empty:
            continue
        yearly["rank"] = yearly["concept"].map({c: i for i, c in enumerate(NI_CONCEPTS)})
        ni = (yearly.sort_values(["end", "rank", "filed"], ascending=[True, True, False])
                    .drop_duplicates("end", keep="first").sort_values("end"))
        eq_all = g[g["concept"].isin(EQ_CONCEPTS) & g["start"].isna()]
        rows = []
        for _, r in ni.tail(YEARS).iterrows():
            e = eq_all[eq_all["end"] == r["end"]]
            e_val = None
            for c in EQ_CONCEPTS:
                ec = e[e["concept"] == c]
                if len(ec):
                    e_val = float(ec.sort_values("filed").iloc[-1]["val"])
                    break
            nv = float(r["val"])
            rows.append([r["end"].date().isoformat(), nv, e_val,
                         (nv / e_val) if (e_val is not None and e_val > 0) else None])
        # The five years must be the RECENT five: BKNG stopped tagging net income after FY2015,
        # and its 2011-2015 table was being published as current (review, 2026-10-09).
        if rows and (pd.Timestamp(today) - pd.Timestamp(rows[-1][0])).days > MAX_STALENESS_DAYS:
            rows = rows[-1:]
        # five consecutive fiscal years, no gaps (a missing year would stretch the window)
        if len(rows) == YEARS:
            ends = [pd.Timestamp(x[0]) for x in rows]
            if any((b - a).days > 400 for a, b in zip(ends, ends[1:])):
                rows = rows[-1:]  # not five consecutive years: earnings_variability() will refuse
        out[t] = rows
    return out


def earnings_variability(rows: list) -> float | None:
    """Sample standard deviation of the annual ROEs; all ``YEARS`` must be present."""
    vals = [r[3] for r in rows if r[3] is not None]
    if len(rows) < YEARS or len(vals) < YEARS:
        return None
    return float(statistics.stdev(vals))
