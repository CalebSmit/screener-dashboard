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
* **revenue** is compared with the company's filed revenue line **nearest** the run's figure
  (2026-10-15). Companies file several "revenue" lines - gross and net, contract revenue and total -
  and which one Yahoo reports varies: Williams' is the filing's ``Revenues`` to the dollar, Republic
  Services' and Hasbro's the contract-revenue line, the ``Revenues`` tag being their gross figure.
  The "largest line" rule used until then made all three look like disagreements. A corrupted
  figure (DuPont's -$1.871bn derived quarter) still matches no line the company filed, so the check
  keeps its point: is the run's number one the company actually reported? Camden's component tag
  ($12.6M of non-rental income) is never nearest a $1.57bn figure.
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


def sec_value(facts, concepts, kind: str, end, largest: bool = False, near: float | None = None):
    """(value, concept, how) for one company's facts, or (None, None, why).

    The first concept in order with a value; with ``largest``, the largest; with ``near``, the one
    closest to that figure (revenue - see the module docstring)."""
    rebuild = flow_ttm if kind == "flow" else instant
    got = []
    for c in concepts:
        f = facts[facts["concept"] == c]
        if f.empty:
            continue
        v, how = rebuild(f, end)
        if v is not None:
            if not largest and near is None:
                return v, c, how
            got.append((v, c, how))
    if got:
        if near is not None:
            return min(got, key=lambda g: abs(g[0] - near))
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
                v, concept, how = sec_value(f_t, concepts, kind, pd.Timestamp(when),
                                            near=float(y) if name == "revenue" else None)
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
    import pandas as pd
    out = {}
    if df.empty:
        return out
    for name in df["field"].unique():
        c = df[(df["field"] == name) & (df["status"] == "compared")]
        # a field with nothing compared has no rel_diff column at all, not an empty one
        a = c["rel_diff"].abs() if "rel_diff" in c else pd.Series(dtype=float)
        out[name] = {"compared": int(len(c)), "within_half_pct": int((a <= 0.005).sum()), "over_10pct": int((a > FAR).sum())}
    return out


# --------------------------------------------------------------- what the capex figure contains
# 2026-10-15, research/2026-10-14-capex-definition.md section 5. Yahoo's `Capital Expenditure`
# row is a sum, and for a company with a captive finance arm most of it can be the vehicles or
# equipment that arm buys to lease to customers - 61% of GM's. The companies themselves report
# free cash flow for the industrial business only, from that segment's own operating cash flow,
# which this screener's consolidated ratio does not separate. Refusing to publish the figure, or
# restating it, were both weighed and refused (section 5); saying what it contains is what is
# left, and every number in the note below is the company's own filed figure.
#
# Display-only, like everything else in this module: a test fails if a scoring module imports it.
LEASE_PURCHASE = ("PaymentsToAcquireLeasesHeldForInvestment", "PaymentsToAcquireEquipmentOnLease",
                  "PaymentsToAcquireLoansAndLeasesHeldForInvestment")
LEASE_PROCEEDS = ("ProceedsFromLeasesHeldForInvestment", "ProceedsFromSaleOfEquipmentOnLease")
PPE_PURCHASE = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
# A share of capex below this is not worth a sentence on the row. 10% is a tenth of the metric's
# input, pre-registered rather than fitted: on the 2026-10-15 cache it selects GM (61%), DE (69%),
# CAT (37%) and PCAR (53%) - the four the proposal named - and nothing else.
LEASE_SHARE_MIN = 0.10
# The note tells the reader the lease purchases are *part of* the capex above, so that has to be
# true of this company and not merely of the four that were checked by hand. It is asserted:
# filed PP&E + filed lease purchases must come within 5% of the capex the run used. Ford is why -
# it files $55.7B of loan and lease purchases, 5.8x its capex, because Yahoo's Ford capex already
# excludes Ford Credit (section 4 of the note). Without this check the row would tell a reader
# that 577% of Ford's capex was vehicles for lease.
LEASE_RECONCILE_TOL = 0.05


def lease_book(payload: dict, facts):
    """``{ticker: {...}}`` for every stock whose own filings report purchases of assets to lease
    out worth ``LEASE_SHARE_MIN`` or more of the capex the run used **and** account for the
    difference between that capex and the company's filed property, plant and equipment.

    Each value holds the run's ``capex``, the filed ``ppe`` and ``purchases`` it splits into, the
    purchases' ``share`` of it, the ``proceeds`` leg where the company files one (only GM of the
    four, 2026-10-15), the concepts the company tags, and the cash-flow quarter the twelve months
    end at. Figures are rebuilt for the same period as the cash flow the run scored, by the same
    code as the filing check above."""
    import pandas as pd
    facts = facts[facts["form"].isin(FORMS)]
    by_ticker = dict(tuple(facts.groupby("ticker")))
    out = {}
    for t, det in (payload.get("stock_detail") or {}).items():
        inp, asof = det.get("inp") or {}, det.get("asof") or {}
        capex, when, f_t = inp.get("capex"), asof.get("cf"), by_ticker.get(t)
        if capex is None or not when or f_t is None:
            continue
        capex = abs(float(capex))
        if capex <= 0:
            continue
        end = pd.Timestamp(when)
        purchases, concept, _ = sec_value(f_t, LEASE_PURCHASE, "flow", end)
        if purchases is None or purchases <= 0:
            continue
        share = purchases / capex
        if share < LEASE_SHARE_MIN or share > 1:
            continue
        ppe, ppe_concept, _ = sec_value(f_t, PPE_PURCHASE, "flow", end)
        if ppe is None or abs(ppe + purchases - capex) > LEASE_RECONCILE_TOL * capex:
            continue
        proceeds, p_concept, _ = sec_value(f_t, LEASE_PROCEEDS, "flow", end)
        out[t] = {"capex": capex, "ppe": float(ppe), "purchases": float(purchases),
                  "share": round(share, 4),
                  "proceeds": None if proceeds is None else float(proceeds),
                  "concept": concept, "ppe_concept": ppe_concept, "proceeds_concept": p_concept,
                  "asof": str(when)[:10]}
    return out


def _usd(v) -> str:
    """$1.2B / $340M, the page's own scale for a cash-flow figure."""
    a = abs(float(v))
    return f"${a / 1e9:,.1f}B" if a >= 1e9 else f"${a / 1e6:,.0f}M"


def lease_capex_note(entry: dict) -> str:
    """The sentence the FCF-yield row carries for one stock, from `lease_book`'s filed figures.

    Says three things and nothing else: how much of the capex above is assets bought to lease to
    customers, the proceeds leg where the company files it, and that the company's own free cash
    flow covers the industrial business only. It never says what to do about any of that - the
    note is provenance, not a verdict (claim `drilldown.capex_lease_note`)."""
    s = (f"The {_usd(entry['capex'])} of capital expenditure above is {_usd(entry['ppe'])} of "
         f"property, plant and equipment plus {_usd(entry['purchases'])} "
         f"({entry['share'] * 100:.0f}% of it) that this company's own filings report as equipment "
         f"or vehicles bought to lease to customers - the asset side of its finance arm, which it "
         f"tags {entry['concept']}.")
    if entry.get("proceeds"):
        s += (f" The {_usd(entry['proceeds'])} received when those assets are sold at the end of a "
              f"lease is an investing receipt ({entry['proceeds_concept']}) and is in neither "
              f"figure above.")
    return s + (" A company with a finance arm reports free cash flow for its industrial business "
                "only, starting from that segment's own operating cash flow; this ratio uses the "
                "consolidated figure on both sides, so it is not the company's own measure.")


SHARE_CONCEPTS = ("WeightedAverageNumberOfDilutedSharesOutstanding",
                  "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                  "WeightedAverageNumberOfSharesOutstandingBasic")
SHARES_RECENT_DAYS = 270


def market_value_check(payload: dict, facts, today=None):
    """Market value / price - the share count scoring implicitly uses - against the latest filed
    weighted-average share count (a quarter ending within 9 months). One row per stock with
    ``ratio`` = implied / filed. Added 2026-10-14: within 5% for 91.8% of stocks; the gaps over
    10% traced to splits after the filed quarter (KLAC, APH, MNST), partnership-unit structures
    (IBKR, TKO, BX, CVNA) and two filers' own scale errors (MCD, WAT)."""
    import pandas as pd
    if today is None:                       # the run's own date, so a fixture never ages (2026-10-15)
        stamp = (payload.get("kpis") or {}).get("run_timestamp")
        today = pd.Timestamp(str(stamp)[:10]) if stamp else pd.Timestamp.now()
    today = pd.Timestamp(today)
    f = facts[facts["concept"].isin(SHARE_CONCEPTS) & facts["start"].notna()]
    f = f[(f["end"] - f["start"]).dt.days.between(80, 100) & (f["end"] >= today - pd.Timedelta(days=SHARES_RECENT_DAYS))]
    by_ticker = dict(tuple(f.groupby("ticker")))
    rows = []
    for t, det in (payload.get("stock_detail") or {}).items():
        mc, px = (det.get("financials") or {}).get("market_cap"), det.get("price")
        g = by_ticker.get(t)
        if not mc or not px or px <= 0 or g is None:
            continue
        for c in SHARE_CONCEPTS:
            h = g[g["concept"] == c]
            if h.empty:
                continue
            last = h[h["end"] == h["end"].max()].sort_values("filed").iloc[0]
            if last["val"] > 0:
                rows.append({"ticker": t, "implied": mc / px, "filed": float(last["val"]),
                             "ratio": (mc / px) / float(last["val"])})
            break
    return pd.DataFrame(rows, columns=["ticker", "implied", "filed", "ratio"])
