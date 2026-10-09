"""Five years of annual return on equity from the SEC's XBRL "frames", for earnings variability.

WHY THIS EXISTS - 2026-10-09, ``research/2026-10-09-operating-leverage.md`` section 6. Both
published practitioner definitions of quality measure *durability* as how variable earnings have
been: MSCI's Quality Indexes use the five-year standard deviation of EPS growth, and AQR's
Quality Minus Junk uses the standard deviation of ROE (60 quarters, or **five fiscal years of
annual ROE** when quarterly data is unavailable - Asness, Frazzini & Pedersen 2019, *Review of
Accounting Studies* 24, p. 74). The screener had neither: Yahoo's statements carry four annual
years, so no faithful version could be computed. The SEC carries a decade.

**Source.** ``data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json`` returns, in one
request, the value every filer reported for a concept in a calendar period - so five years of net
income and equity for the whole universe is ~25 requests, not 2,500. The frames API assigns each
company's fiscal year to the calendar year it most closely fits; equity is the instant nearest the
end of that calendar year (``CY{y}Q4I``). For a company whose fiscal year does not end in
December the two can be up to six months apart, which this records and does not correct.

**Metric.** ``earnings_variability`` = sample standard deviation of annual ROE (net income /
year-end shareholders' equity) over the last five complete calendar years, **all five required**
(AQR's rule); missing when equity is zero or negative in any of them. Lower = steadier.

**A candidate (weight 0).** It is computed, published and shown with its five years, and the
improvement engine records it; it moves no score until it has its own note and changelog entry
(CLAUDE.md rule 4).

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
FRAMES_DIR = ROOT / "data" / "sec" / "frames"

# First concept with a value wins; the fallbacks are the tags large filers use instead
# (measured 2026-10-09: NetIncomeLoss alone covers 399 of 503).
NI_CONCEPTS = ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
EQ_CONCEPTS = ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
YEARS = 5
RECENT_MAX_AGE_DAYS = 7      # frames for the last two years are still being filled in


def last_complete_year(today: date) -> int:
    """The latest calendar year whose annual reports are all in (10-Ks land by ~March)."""
    return today.year - 1 if today.month >= 4 else today.year - 2


def years_for(today: date, n: int = YEARS) -> list[int]:
    y = last_complete_year(today)
    return list(range(y - n + 1, y + 1))


def _frame(edgar, concept: str, period: str, today: date) -> dict | None:
    """``{cik: [value, period_end]}`` for one concept and period, cached on disk."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAMES_DIR / f"{concept}_{period}.json"
    settled = today.year - int(period[2:6]) >= 2
    if path.exists() and (settled or time.time() - path.stat().st_mtime < RECENT_MAX_AGE_DAYS * 86400):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    try:
        r = edgar.s.get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/{period}.json", timeout=60)
        edgar.requests += 1
        time.sleep(0.15)
        if r.status_code == 404:                     # no filer used this concept that period
            data = {}
        else:
            r.raise_for_status()
            data = {str(d["cik"]): [d["val"], d.get("end")] for d in r.json().get("data", [])}
    except Exception:  # noqa: BLE001 - keep a stale frame rather than lose the metric
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                return None
        return None
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


def _first(frames: list[dict | None], cik: str):
    for f in frames:
        if f and cik in f:
            return f[cik]
    return None


def roe_history(tickers: list[str], today: date | None = None, edgar=None, log=print) -> dict:
    """``{ticker: [[year, net_income, equity, roe_or_None], ...]}``, oldest year first."""
    from insider_activity import Edgar, ticker_map, user_agent
    today = today or date.today()
    if edgar is None:
        ua = user_agent()
        if not ua:
            log("  Earnings variability: no SEC identity configured - skipped")
            return {}
        edgar = Edgar(ua)
    cmap = ticker_map(edgar)
    years = years_for(today)
    ni = {y: [_frame(edgar, c, f"CY{y}", today) for c in NI_CONCEPTS] for y in years}
    eq = {y: [_frame(edgar, c, f"CY{y}Q4I", today) for c in EQ_CONCEPTS] for y in years}
    out: dict = {}
    for t in tickers:
        cik = cmap.get(str(t).upper())
        if cik is None:
            continue
        cik = str(cik)
        rows = []
        for y in years:
            n, e = _first(ni[y], cik), _first(eq[y], cik)
            nv = n[0] if n else None
            ev = e[0] if e else None
            roe = (nv / ev) if (nv is not None and ev is not None and ev > 0) else None
            rows.append([y, nv, ev, roe])
        out[t] = rows
    return out


def earnings_variability(rows: list) -> float | None:
    """Sample standard deviation of the annual ROEs; all ``YEARS`` must be present."""
    vals = [r[3] for r in rows if r[3] is not None]
    if len(rows) < YEARS or len(vals) < YEARS:
        return None
    return float(statistics.stdev(vals))
