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


NEWER_FILING_DAYS = 45      # a period end this much later than the run's is a newer quarter, not a date wobble


def newer_filings(payload: dict, facts, today=None) -> dict:
    """``{ticker: {"end", "filed", "form"}}`` for every stock whose income statement in the run ends
    a quarter or more before the newest period the company had filed with the SEC by the run date.

    2026-10-15: 32 of 501 stocks were scored a quarter behind a filed 10-Q, 16 of them by more than
    70 days (Amazon, Bank of America, Boeing - the source lists the newer quarter as a column with 4
    or 5 rows and nothing in it). The engine is right not to read an empty column; the reader should
    still know the figures are the quarter before, and why. Display and reporting only.
    research/measurements/2026-10-15-statement-lag.py"""
    import pandas as pd
    run = pd.Timestamp(today) if today is not None else pd.Timestamp(
        str((payload.get("kpis") or {}).get("run_timestamp") or pd.Timestamp.today())[:10])
    f = facts[facts["form"].isin(("10-K", "10-Q")) & facts["start"].notna() & (facts["filed"] <= run)]
    newest = {t: g.loc[g["end"].idxmax()] for t, g in f.groupby("ticker")}
    out = {}
    for t, det in (payload.get("stock_detail") or {}).items():
        used = (det.get("asof") or {}).get("is")
        r = newest.get(t)
        if not used or r is None:
            continue
        if (r["end"] - pd.Timestamp(used)).days > NEWER_FILING_DAYS:
            out[t] = {"end": str(r["end"].date()), "filed": str(pd.Timestamp(r["filed"]).date()),
                      "form": str(r["form"])}
    return out


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


# --- one-off charges inside the earnings the screener scored (2026-10-16) ---
#
# WHY: `research/2026-10-15-earnings-definition.md`, section 8. The proposal to score earnings
# yield on the source's "Normalized Income" - net income without its unusual items - was refused:
# at the four-quarter window the engine sums, 23 companies have no normalized figure at all
# (Amazon, Bank of America, Berkshire, Boeing among them), so the metric would rank them on
# earnings *including* their one-off items beside 265 companies ranked excluding them, which is
# the defect `factor_engine.py:2094` records Phase 13 fixing in this exact metric. The evidence
# for removing one-off items is good (Rouen, So & Wang 2021: next-year persistence 0.810 for core
# earnings against 0.637 for net income); the input that would do it is not available on one basis.
#
# So the figure stays as reported and the row says what is inside it - the same remedy the refused
# capex proposal got above. The complementary half of `ranking_audit`'s earnings-yield note, which
# fires only when net income runs *above* after-tax operating income and so catches the gain
# direction; it is correctly one-sided, because interest expense routinely puts net income below
# after-tax operating income for sound reasons. The charge direction had nothing on the page.
#
# Display-only, like everything else here: a test fails if a scoring module imports this module.
#
# The company's own tags, in priority order. `sec_value(..., largest=True)` returns the **single
# largest** line, never a sum: these tags overlap (a restructuring-and-impairment roll-up contains
# what the separate impairment tags report), and summing them would publish a figure larger than
# the company reported. One line, named by its tag, is checkable in the filing and cannot double
# count - so the note says "at least", which understates rather than overstates.
SPECIAL_ITEM_CONCEPTS = (
    "RestructuringCostsAndAssetImpairmentCharges", "RestructuringSettlementAndImpairmentProvisions",
    "GoodwillAndIntangibleAssetImpairment", "GoodwillImpairmentLoss",
    "AssetImpairmentCharges", "OtherAssetImpairmentCharges",
    "ImpairmentOfIntangibleAssetsExcludingGoodwill",
    "ImpairmentOfIntangibleAssetsIndefinitelivedExcludingGoodwill",
    "ImpairmentOfLongLivedAssetsHeldForUse", "RestructuringCharges",
)
# Plain English for each tag, so the sentence does not make a reader decode a tag name to read it.
SPECIAL_ITEM_WORDS = {
    "RestructuringCostsAndAssetImpairmentCharges": "restructuring and asset impairment charges",
    "RestructuringSettlementAndImpairmentProvisions": "restructuring, settlement and impairment provisions",
    "GoodwillAndIntangibleAssetImpairment": "goodwill and intangible asset impairment",
    "GoodwillImpairmentLoss": "goodwill impairment",
    "AssetImpairmentCharges": "asset impairment charges",
    "OtherAssetImpairmentCharges": "other asset impairment charges",
    "ImpairmentOfIntangibleAssetsExcludingGoodwill": "impairment of intangible assets other than goodwill",
    "ImpairmentOfIntangibleAssetsIndefinitelivedExcludingGoodwill":
        "impairment of indefinite-lived intangible assets",
    "ImpairmentOfLongLivedAssetsHeldForUse": "impairment of long-lived assets",
    "RestructuringCharges": "restructuring charges",
}
# A charge below this share of the net income the run scored is not worth a sentence on the row.
# A tenth of the metric's numerator, pre-registered rather than fitted - the same reasoning and the
# same number as LEASE_SHARE_MIN above, written before the selection was run. What it picks on the
# 2026-10-16 payload is recorded in `tests/test_special_item_note.py` and in the research note.
SPECIAL_ITEM_SHARE_MIN = 0.10


# Concepts every filer reports annually, used only to enumerate which fiscal years a company has
# filed. The window must NOT be derived from the special-item tags themselves: a year without a
# charge files no such tag, so counting only years that carry one makes "N of M" report N of N.
# That was the first draft's defect, found 2026-10-16 by its own output - 37 of 47 selected stocks
# read "4 of 4", which is the bug and not the companies.
YEAR_ANCHORS = ("NetIncomeLoss", "IncomeTaxExpenseBenefit", "Revenues", "Assets")


def _charge_years(f_t, end, back: int = 4) -> tuple:
    """(fiscal years carrying a tagged charge, fiscal years filed) over the last `back` years.

    The note's point turns on this. Measured 2026-10-16
    (`research/2026-10-16-unusual-items-vs-filings.py`): of 493 companies with four fiscal years
    filed, **120 tagged a restructuring or impairment charge in every one of them**. So "one-off"
    is not a safe word, and the honest thing to put beside the charge is how often the company has
    reported one. Counted from the companies' own tags, annual periods only."""
    import pandas as pd
    fy = f_t[f_t["fp"] == "FY"].copy()
    if fy.empty:
        return 0, 0
    fy["_end"] = pd.to_datetime(fy["end"])
    fy["_start"] = pd.to_datetime(fy["start"])
    span = (fy["_end"] - fy["_start"]).dt.days
    fy = fy[span.between(330, 400)]
    if fy.empty:
        return 0, 0
    filed_years = sorted({d.year for d in fy[fy["concept"].isin(YEAR_ANCHORS)]["_end"]})
    window = [y for y in filed_years if y <= pd.Timestamp(end).year][-back:]
    charged = fy[fy["concept"].isin(SPECIAL_ITEM_CONCEPTS) & (fy["val"] > 0)]
    charge_years = {d.year for d in charged["_end"]}
    return sum(1 for y in window if y in charge_years), len(window)


def special_items(payload: dict, facts):
    """``{ticker: {...}}`` for every stock whose own filings report a one-off charge worth
    ``SPECIAL_ITEM_SHARE_MIN`` or more of the trailing net income the run scored.

    The charge is rebuilt for the **same twelve months as the income statement the run used**
    (``asof['is']``), by the same code as the filing check above, so the two figures describe one
    period. Each value holds the ``charge``, the ``net_income`` it sits inside, the ``share``, the
    ``concept`` the company tagged, how many of its last four fiscal years carried a charge of the
    same kind, and the period end."""
    import pandas as pd
    facts = facts[facts["form"].isin(FORMS)]
    by_ticker = dict(tuple(facts.groupby("ticker")))
    out = {}
    for t, det in (payload.get("stock_detail") or {}).items():
        inp, asof = det.get("inp") or {}, det.get("asof") or {}
        ni, when, f_t = inp.get("netIncome"), asof.get("is"), by_ticker.get(t)
        if ni is None or not when or f_t is None:
            continue
        ni = float(ni)
        if abs(ni) < 1e6:            # too small a bottom line to state a share against
            continue
        end = pd.Timestamp(when)
        charge, concept, _ = sec_value(f_t, SPECIAL_ITEM_CONCEPTS, "flow", end, largest=True)
        # A charge is filed positive. A negative value is a reversal, a different story, not claimed.
        if charge is None or charge <= 0:
            continue
        share = charge / abs(ni)
        if share < SPECIAL_ITEM_SHARE_MIN:
            continue
        years, of_years = _charge_years(f_t, end)
        out[t] = {"charge": float(charge), "net_income": ni, "share": round(share, 4),
                  "concept": concept, "charge_years": years, "of_years": of_years,
                  "asof": str(when)[:10]}
    return out


def special_item_note(entry: dict) -> str:
    """The sentence the earnings-yield row carries for one stock, from `special_items`' figures.

    Says what is inside the published figure and nothing about what to do with it - the note is
    provenance, not a verdict (claim ``drilldown.special_item_note``). Four things it is careful
    about, three of them found by reading the first draft's own output on 2026-10-16:

    * a **loss** is called a loss. ``_usd`` takes an absolute value, so the first draft told readers
      of Gilead, Kraft Heinz, General Mills and Intel that a loss "includes" a charge, which reads
      as though the bottom line were a profit;
    * the **share is printed only when the bottom line is a profit**. "7,887% of the net income"
      against Air Products' $47M loss is arithmetic, not information;
    * the charge is **before tax** while net income is after it, so the share is labelled as such
      rather than presented as a decomposition;
    * it is the largest **single** line the company tagged, never a sum of overlapping tags, so it
      says "at least"."""
    ni, charge = entry["net_income"], entry["charge"]
    words = SPECIAL_ITEM_WORDS.get(entry["concept"], "one-off charges")
    bottom = (f"The {_usd(ni)} of net income above" if ni > 0
              else f"The {_usd(ni)} loss above")
    # "is after", not "includes": net income is struck after the charge is deducted, and that
    # reads correctly when the charge is larger than the bottom line - which it is for 9 of the
    # 47 stocks selected on 2026-10-16 (General Mills' $3.0B against an $88M loss).
    s = (f"{bottom} is after at least {_usd(charge)} of {words} that this company's own filings "
         f"report for the same twelve months, tagged {entry['concept']}")
    if ni > 0:
        share = entry["share"]
        s += (f" - equal to {share * 100:.0f}% of that net income" if share < 1
              else f" - equal to {share:.1f} times that net income")
        s += ", before any tax effect."
    else:
        s += ", before any tax effect."
    s += " This yield is computed from earnings as reported, so the charge is in it."
    years, of_years = entry.get("charge_years") or 0, entry.get("of_years") or 0
    if of_years:
        s += (f" Its filings report a charge of this kind in {years} of its last "
              f"{of_years} fiscal years.")
    return s


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
