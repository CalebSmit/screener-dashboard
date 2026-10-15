"""The inputs the run scored with, checked against the companies' own SEC filings.

WHY THIS EXISTS - 2026-10-14, ``research/2026-10-14-yahoo-vs-sec.md``. Every fundamental the
screener scores comes from Yahoo through ``yfinance``. That week Yahoo silently stopped serving one
field (``dividendRate``), and the availability check added the same day catches a field that
*vanishes*. It cannot catch one that changes scale or meaning while still being served. The SEC's
own XBRL figures can: rebuilt for the same quarter, total assets agreed with Yahoo to within 0.5%
for 99.8% of companies on 2026-10-14, so a field that suddenly disagrees for dozens of companies is
a source event, the morning it appears.

**Read-only and offline.** It reads the weekly ``companyfacts`` cache the run already maintains
(``sec_fundamentals.FACTS_PATH``) and the payload's ``inp`` / ``asof``. No network, no write, and
nothing here can reach a score.

How a figure is rebuilt for the quarter ending E:

* a trailing-twelve-month **flow** is the fiscal year if E ends one, otherwise
  year-to-date(E) + prior fiscal year - year-to-date(E a year earlier);
* a **balance** is the instant value at E;
* each period's value is the one **first filed** - later filings' comparative columns are where
  mis-tagged contexts appear (FIX's Q1 2026 10-Q tags a quarter's revenue as fiscal 2025's);
* **revenue** takes the largest revenue concept reported at E, because the others are components.
"""
from __future__ import annotations

from datetime import timedelta

REVENUE = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
           "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
           "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense")
# (name, payload `inp` key, SEC concepts in priority order, "flow" or "instant", `asof` statement)
FIELDS = [
    ("revenue", "totalRevenue", REVENUE, "flow", "is"),
    ("net_income", "netIncome", ("NetIncomeLoss", "ProfitLoss"), "flow", "is"),
    ("gross_profit", "grossProfit", ("GrossProfit",), "flow", "is"),
    ("operating_cash_flow", "operatingCashFlow", ("NetCashProvidedByUsedInOperatingActivities",), "flow", "cf"),
    ("capex", "capex", ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"), "flow", "cf"),
    ("total_assets", "totalAssets", ("Assets",), "instant", "bs"),
    ("equity", "totalEquity", ("StockholdersEquity",
                               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "instant", "bs"),
]
FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")
FAR = 0.10        # a disagreement "over 10%" - of the larger of the two figures


def first_filed(rows):
    """The value as first filed, or None."""
    if rows.empty:
        return None
    return float(rows.sort_values("filed").iloc[0]["val"])


def _near(series, when, days: int):
    return (series - when).abs() <= timedelta(days=days)


def flow_ttm(f, end):
    """Trailing twelve months ending ``end`` from one concept's facts: (value or None, how)."""
    dur = (f["end"] - f["start"]).dt.days
    at = f[_near(f["end"], end, 3) & f["start"].notna()]
    d_at = (at["end"] - at["start"]).dt.days
    fy = at[(d_at >= 350) & (d_at <= 380)]
    if not fy.empty:
        return first_filed(fy), "fiscal year"
    ytd = at[(d_at >= 80) & (d_at < 350)]
    if ytd.empty:
        return None, "no period ending at that quarter"
    span = (ytd["end"] - ytd["start"]).dt.days
    ytd = ytd[span == span.max()]
    start = ytd["start"].iloc[0]
    prior_fy = f[_near(f["end"], start - timedelta(days=1), 7) & (dur >= 350) & (dur <= 380)]
    prior_ytd = f[_near(f["end"], end - timedelta(days=365), 7) & _near(f["start"], start - timedelta(days=365), 7)]
    if prior_fy.empty or prior_ytd.empty:
        return None, "prior-year columns missing"
    return first_filed(ytd) + first_filed(prior_fy) - first_filed(prior_ytd), "ytd + prior fy - prior ytd"


def instant(f, end):
    at = f[_near(f["end"], end, 3) & f["start"].isna()]
    return (first_filed(at), "instant") if not at.empty else (None, "no balance at that date")


def sec_value(facts, concepts, kind: str, end, largest: bool = False):
    """(value, concept, how) for one company's facts, or (None, None, why)."""
    rebuild = flow_ttm if kind == "flow" else instant
    got = []
    for c in concepts:
        f = facts[facts["concept"] == c]
        if f.empty:
            continue
        v, how = rebuild(f, end)
        if v is not None:
            if not largest:
                return v, c, how
            got.append((v, c, how))
    if got:
        return max(got, key=lambda g: abs(g[0]))
    return None, None, "no concept with a value at that date"


def compare(payload: dict, facts, fields=FIELDS):
    """One row per (stock, field): the run's figure, the filing's, and how far apart."""
    import pandas as pd
    facts = facts[facts["form"].isin(FORMS)]
    by_ticker = dict(tuple(facts.groupby("ticker")))
    rows = []
    for t, det in (payload.get("stock_detail") or {}).items():
        inp, asof = det.get("inp") or {}, det.get("asof") or {}
        f_t = by_ticker.get(t)
        for name, key, concepts, kind, stmt in fields:
            y, when = inp.get(key), asof.get(stmt)
            rec = {"ticker": t, "field": name, "yahoo": y, "asof": when, "sector": det.get("sector")}
            if y is None or when is None:
                rec["status"] = "no yahoo value"
            elif f_t is None:
                rec["status"] = "not in SEC cache"
            else:
                v, concept, how = sec_value(f_t, concepts, kind, pd.Timestamp(when), largest=(name == "revenue"))
                rec.update(sec=v, concept=concept, how=how)
                if v is None:
                    rec["status"] = "no SEC value: " + how
                else:
                    if name == "capex":
                        y = abs(y)                  # Yahoo signs capex negative; the SEC tag is a payment
                        rec["yahoo"] = y
                    rec["rel_diff"] = (y - v) / max(abs(v), abs(y), 1.0)
                    rec["status"] = "compared"
            rows.append(rec)
    return pd.DataFrame(rows)


def summary(df) -> dict:
    """Per field: how many were compared, how many agree within 0.5%, how many differ by over 10%."""
    out = {}
    for name in df["field"].unique():
        c = df[(df["field"] == name) & (df["status"] == "compared")]
        a = c["rel_diff"].abs()
        out[name] = {"compared": int(len(c)), "within_half_pct": int((a <= 0.005).sum()), "over_10pct": int((a > FAR).sum())}
    return out
