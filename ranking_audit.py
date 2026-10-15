"""What to check before trusting a high rank: the questions a careful reader asks of the top names.

WHY THIS EXISTS - 2026-10-15, owner: *"look into the top 5 stocks in the screener and make sure they
should actually be there."* Done by hand that day (`research/2026-10-15-top-five-audit.md`), the
audit found every top-five input agreeing with the companies' own filings, and two things a number
cannot show by itself:

* **Host Hotels (#3)** - net income of $1.03bn against $0.89bn of operating income: about $0.2bn
  of gains from selling hotels lifted its earnings yield, Piotroski's profit signals and its
  sustainable growth. 46 of 501 stocks had net income above 1.25x their after-tax operating income
  that day; Western Digital (#28) was at 2.56x.
* **Expedia (#1)** - equity of 4.6% of assets after years of buybacks, so return on equity read
  ~170% and sustainable growth sat at its 100% cap: a measure of the small equity base, not of
  growth capacity. Four stocks had equity under 10% of assets with sustainable growth at the cap.

Neither is an error in the data, and the screener scores them as defined; changing how earnings
are measured is a methodology question with its own proposal (the note's section 4). What this
module does is say it - on the metric it affects in each stock's view, and in the morning health
note for the top 25 - so a reader sees *why* a number is high before acting on the rank.

**Display and reporting only.** It reads the payload's published inputs and figures and writes
prose; no scoring module may import it (`tests/test_ranking_audit.py`).
"""
from __future__ import annotations

TAX = 0.21                    # a statutory rate for "after-tax operating income"; the ratio only flags
NONOPERATING_RATIO = 1.25     # net income above 1.25x after-tax operating income
THIN_EQUITY = 0.10            # equity under 10% of total assets
SGR_CAP = 0.999               # sustainable growth is clipped to [0, 1]
FIELD_NAMES = {"ta": "total assets", "eq": "equity", "rev": "revenue", "ni": "net income",
               "ocf": "operating cash flow"}
TOP_LINE = "top 25, worth a look: "


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _bn(v: float) -> str:
    return f"${v / 1e9:,.2f}bn" if abs(v) >= 1e9 else f"${v / 1e6:,.0f}m"


def nonoperating_ratio(detail: dict) -> float | None:
    """Net income over after-tax operating income, when both are positive."""
    inp = detail.get("inp") or {}
    ni, ebit = _f(inp.get("netIncome")), _f(inp.get("ebit"))
    if ni is None or ebit is None or ni <= 0 or ebit <= 0:
        return None
    return ni / (ebit * (1 - TAX))


def equity_share(detail: dict) -> float | None:
    inp = detail.get("inp") or {}
    eq, ta = _f(inp.get("totalEquity")), _f(inp.get("totalAssets"))
    if eq is None or ta is None or ta <= 0:
        return None
    return eq / ta


def stock_notes(detail: dict) -> dict:
    """``{metric: sentence}`` for the metrics a stock's own figures make worth a second look."""
    out = {}
    inp = detail.get("inp") or {}
    r = nonoperating_ratio(detail)
    if r is not None and r > NONOPERATING_RATIO:
        ni, ebit = _f(inp.get("netIncome")), _f(inp.get("ebit"))
        out["earnings_yield"] = (
            f"Net income over the last twelve months ({_bn(ni)}) is {r:.2f} times operating income "
            f"after a {TAX:.0%} tax ({_bn(ebit * (1 - TAX))}). The difference is income from outside "
            f"the business's operations - a gain on selling assets or investments, interest, a tax "
            f"item - which may not recur. This yield uses net income as reported, so it reads "
            f"higher than the operations alone would support.")
    share = equity_share(detail)
    sgr = _f((detail.get("raw") or {}).get("sustainable_growth"))
    if share is not None and 0 < share < THIN_EQUITY and sgr is not None and sgr >= SGR_CAP:
        out["sustainable_growth"] = (
            f"Shareholders' equity is only {share:.1%} of total assets - usually the result of years "
            f"of share buybacks - so return on equity is very high and this measure sits at its 100% "
            f"cap. It reflects the small equity base more than the company's capacity to grow.")
    return out


def attach(stock_detail: dict) -> int:
    """Put each stock's notes in ``stock_detail[t]["fign"]`` - its own field, shown under its own
    heading ("In this company's reported figures"), because ``cavs`` is the filed-figure note and
    is headed "In this company's filings": these sentences come from the figures the source
    reports, and the heading must say which. Returns how many stocks got one."""
    n = 0
    for d in stock_detail.values():
        notes = stock_notes(d)
        if notes:
            d["fign"] = notes
            n += 1
    return n


def top_flags(payload: dict, n: int = 25) -> list[dict]:
    """For the top ``n`` of the ranking: every reason to look twice, in plain words."""
    rows = sorted(payload.get("table_data") or [], key=lambda r: (r.get("Rank") or 10**6, r.get("Ticker")))
    detail = payload.get("stock_detail") or {}
    out = []
    for row in rows:
        if not isinstance(row.get("Rank"), int) or row["Rank"] > n:      # ranks 1..n; ties share one
            continue
        t = row.get("Ticker")
        d = detail.get(t) or {}
        why = []
        r = nonoperating_ratio(d)
        if r is not None and r > NONOPERATING_RATIO:
            why.append(f"net income {r:.2f}x after-tax operating income")
        share = equity_share(d)
        sgr = _f((d.get("raw") or {}).get("sustainable_growth"))
        if share is not None and 0 < share < THIN_EQUITY and sgr is not None and sgr >= SGR_CAP:
            why.append(f"equity {share:.1%} of assets")
        for k, v in (d.get("xchk") or {}).items():
            if v is not None and abs(v) > 0.10:
                why.append(f"{FIELD_NAMES.get(k, k)} {abs(v):.0%} off the filing")
        if d.get("newer_filing"):
            why.append("a period behind its filings")
        fl = d.get("flags") or {}
        if fl.get("q_rejected"):
            why.append("annual figures used")
        if why:
            out.append({"ticker": t, "rank": row.get("Rank"), "why": why})
    return out
