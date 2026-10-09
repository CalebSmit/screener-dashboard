"""Insider buying and selling, from the SEC's own Form 4 filings.

WHY THIS EXISTS - 2026-10-08 (owner, ``plan/context-layer.md``). Of the free signals a long-term
investor could look at, open-market *purchases* by a company's own officers and directors have
some of the strongest published support: Lakonishok & Lee (2001, *Review of Financial Studies*)
find insider purchases - not sales - predict returns; Jeng, Metrick & Zeckhauser (2003, *Review of
Economics and Statistics*) estimate purchase portfolios earn abnormal returns of roughly 6% a
year; Cohen, Malloy & Pomorski (2012, *Journal of Finance*) show the information is in
*opportunistic* trades, not routine ones. Sales are mostly diversification and planned
(Rule 10b5-1) selling, so they are shown but not read as a view.

**Context only.** Nothing here enters a score. It is recorded every run so that, once there is
history, its value as a candidate factor can be measured the same way every other candidate is.

**Source, from 2026-10-08 (late):** the SEC's own EDGAR archive - every Form 4 filed in the last
180 days, parsed here (``refresh`` / ``parse_form4`` / ``rows_from_sec``). It is the primary source
and it carries two things Yahoo's compiled feed does not: the **Rule 10b5-1 checkbox** (a sale
made on a pre-arranged trading plan, which says little about the insider's view) and a link to
each filing. ``www.sec.gov`` refuses any User-Agent without a contact email; the owner gave one
on 2026-10-08 and it lives **outside the public repo** - ``SEC_USER_AGENT`` in the environment or
``data/sec/user_agent.txt`` (gitignored). With neither, ``user_agent()`` returns None and the run
uses Yahoo's insider-transactions feed (compiled from Form 4) for every stock, as it did before.
A stock whose SEC record was not refreshed in the last ``SEC_FRESH_DAYS`` falls back to Yahoo
individually. Both sources feed the same ``summarise_rows``.

Amendments (Form 4/A) are **not** read: an amendment restates an earlier filing, and adding its
trades would count them twice. The few corrections they carry are the price of not double
counting.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "insider"
CACHE_PATH = DATA_DIR / "filings.json"
TICKERS_PATH = ROOT / "data" / "sec" / "company_tickers.json"
SUMMARY_PATH = ROOT / "data" / "insider_summary.json"

USER_AGENT = "screener-dashboard personal research"   # data.sec.gov accepts; www.sec.gov needs an email
USER_AGENT_FILE = ROOT / "data" / "sec" / "user_agent.txt"
SEC_FRESH_DAYS = 3             # older than this, a stock's SEC record yields to Yahoo for the run
MIN_INTERVAL = 0.13            # seconds between requests (< 8/s, under the SEC's 10/s)
LOOKBACK_DAYS = 180
KEEP_DAYS = 400
CLUSTER_MIN_BUYERS = 3         # distinct insiders buying in the window -> "cluster"

CODES = {"P": "open-market purchase", "S": "open-market sale", "A": "grant or award",
         "M": "option exercise", "F": "shares withheld for tax", "G": "gift",
         "D": "returned to company", "C": "conversion", "X": "option exercise"}


def user_agent() -> str | None:
    """The SEC contact identity, from the environment or the gitignored file - never the repo.

    Returns None when no identity with an email is configured: the archive refuses requests
    without one, so the caller should not try."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        try:
            ua = USER_AGENT_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            ua = ""
    return ua if "@" in ua else None


class Edgar:
    """A polite EDGAR client: one session, a declared identity, a request-rate floor."""

    def __init__(self, user_agent: str = USER_AGENT):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0
        self.requests = 0

    def get(self, url: str):
        wait = MIN_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        self.requests += 1
        for attempt in range(3):
            r = self.s.get(url, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r


# --------------------------------------------------------------------------- ticker -> CIK
def ticker_map(edgar: Edgar | None = None, max_age_days: int = 7) -> dict:
    fresh = TICKERS_PATH.exists() and (time.time() - TICKERS_PATH.stat().st_mtime) < max_age_days * 86400
    if not fresh and edgar is not None:
        try:
            r = edgar.get("https://www.sec.gov/files/company_tickers.json")
            TICKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
            TICKERS_PATH.write_text(r.text, encoding="utf-8")
        except Exception:  # noqa: BLE001 - fall back to the cached map
            if not TICKERS_PATH.exists():
                raise
    raw = json.loads(TICKERS_PATH.read_text(encoding="utf-8"))
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}


# --------------------------------------------------------------------------- Form 4 parsing
def _t(el, path):
    x = el.find(path)
    return x.text.strip() if x is not None and x.text else None


def _f(el, path):
    v = _t(el, path)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def parse_form4(xml_text: str) -> dict:
    """The parts of a Form 4 a reader needs: who, their role, and each non-derivative trade."""
    xml_text = re.sub(r"^\s*<\?xml[^>]*\?>", "", xml_text)
    root = ET.fromstring(xml_text)
    owners = []
    for ro in root.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        roles = []
        if rel is not None:
            if _t(rel, "isDirector") in ("1", "true"):
                roles.append("Director")
            if _t(rel, "isOfficer") in ("1", "true"):
                roles.append(_t(rel, "officerTitle") or "Officer")
            if _t(rel, "isTenPercentOwner") in ("1", "true"):
                roles.append("10% owner")
        owners.append({"name": _t(ro, "reportingOwnerId/rptOwnerName") or "?", "role": ", ".join(roles) or "Insider"})
    plan = _t(root, "aff10b5One") in ("1", "true")
    trades = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _t(tx, "transactionCoding/transactionCode")
        shares = _f(tx, "transactionAmounts/transactionShares/value")
        price = _f(tx, "transactionAmounts/transactionPricePerShare/value")
        trades.append({
            "date": _t(tx, "transactionDate/value"),
            "code": code,
            "shares": shares,
            "price": price,
            "ad": _t(tx, "transactionAmounts/transactionAcquiredDisposedCode/value"),
            "value": round(shares * price, 2) if shares is not None and price is not None else None,
        })
    return {"owners": owners, "plan": plan, "trades": trades}


# --------------------------------------------------------------------------- refresh
def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")


def refresh(tickers: list[str], today: date | None = None, edgar: Edgar | None = None,
            budget_seconds: float | None = None, log=print) -> dict:
    """Fetch any Form 4 filed in the last ``LOOKBACK_DAYS`` that is not cached yet.

    ``budget_seconds`` stops the refresh early and keeps what it has - a first run (thousands of
    filings) can be completed over several nights without ever holding up the data loop.
    """
    today = today or date.today()
    edgar = edgar or Edgar(user_agent() or USER_AGENT)
    started = time.time()
    cmap = ticker_map(edgar)
    cache = _load_cache()
    since = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    keep = (today - timedelta(days=KEEP_DAYS)).isoformat()
    new_filings = failures = 0
    complete = True
    for i, tk in enumerate(tickers):
        if budget_seconds and time.time() - started > budget_seconds:
            complete = False
            log(f"  insider refresh: time budget reached after {i} of {len(tickers)} tickers; the rest next run")
            break
        cik = cmap.get(tk.upper())
        if cik is None:
            continue
        entry = cache.setdefault(tk, {"cik": cik, "filings": {}})
        entry["cik"] = cik
        try:
            sub = edgar.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        except Exception as e:  # noqa: BLE001
            failures += 1
            log(f"  insider: {tk} submissions failed: {type(e).__name__}")
            continue
        rec = sub.get("filings", {}).get("recent", {})
        for j, form in enumerate(rec.get("form", [])):
            if form != "4":            # 4/A restates an earlier filing - see the module note
                continue
            fdate = rec["filingDate"][j]
            if fdate < since:
                continue
            acc = rec["accessionNumber"][j]
            if acc in entry["filings"]:
                continue
            doc = rec["primaryDocument"][j].split("/")[-1]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
            try:
                parsed = parse_form4(edgar.get(url).text)
            except Exception as e:  # noqa: BLE001 - one malformed filing must not stop the rest
                failures += 1
                parsed = {"error": type(e).__name__}
            parsed["filed"] = fdate
            parsed["form"] = form
            entry["filings"][acc] = parsed
            new_filings += 1
        # forget filings past the retention window
        entry["filings"] = {a: f for a, f in entry["filings"].items() if f.get("filed", "9999") >= keep}
        entry["checked"] = today.isoformat()
        if i % 25 == 0:
            _save_cache(cache)
    cache["_meta"] = {"updated": today.isoformat(), "complete": complete,
                      "requests": edgar.requests, "new_filings": new_filings, "failures": failures}
    _save_cache(cache)
    log(f"  insider refresh: {new_filings} new filings, {failures} failures, {edgar.requests} requests, "
        f"{time.time() - started:.0f}s, {'complete' if complete else 'partial'}")
    return cache


# --------------------------------------------------------------------------- summarise
def summarise_ticker(entry: dict, today: date, window_days: int = 90) -> dict:
    """Open-market buying and selling by insiders over ``window_days``, plus the latest trades."""
    since = (today - timedelta(days=window_days)).isoformat()
    since_long = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    cik = entry.get("cik")
    buys, sells, recent = [], [], []
    for acc, f in (entry.get("filings") or {}).items():
        if "trades" not in f:
            continue
        who = (f.get("owners") or [{}])[0]
        for t in f["trades"]:
            if t.get("code") not in ("P", "S") or not t.get("date"):
                continue
            row = {"date": t["date"], "code": t["code"], "name": who.get("name"), "role": who.get("role"),
                   "shares": t.get("shares"), "price": t.get("price"), "value": t.get("value"),
                   "plan": bool(f.get("plan")),
                   "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm"}
            if t["date"] >= since_long:
                recent.append(row)
            if t["date"] >= since:
                (buys if t["code"] == "P" else sells).append(row)
    recent.sort(key=lambda r: r["date"], reverse=True)

    def total(rows):
        return round(sum(r["value"] or 0 for r in rows), 2)

    buyers = sorted({r["name"] for r in buys if r["name"]})
    officer_buy = any(re.search(r"chief|ceo|cfo|president", (r["role"] or ""), re.I) for r in buys)
    out = {
        "window": window_days,
        "buy_n": len(buys), "buy_people": len(buyers), "buy_value": total(buys),
        "sell_n": len(sells), "sell_people": len({r["name"] for r in sells if r["name"]}),
        "sell_value": total(sells), "sell_planned_value": total([r for r in sells if r["plan"]]),
        "cluster": len(buyers) >= CLUSTER_MIN_BUYERS,
        "officer_buy": officer_buy,
        "recent": recent[:8],
    }
    return out


def _person(name: str | None) -> str | None:
    """EDGAR files names surname first and often in capitals ("COOK TIMOTHY D"); keep the order
    (it is the filing's) but not the shouting."""
    if not name:
        return None
    return name.title() if name.isupper() else name


def rows_from_sec(entry: dict, today: date | None = None) -> list[dict]:
    """Open-market purchases (P) and sales (S) from one stock's parsed Form 4 filings.

    Same row shape as ``rows_from_yahoo``, plus ``plan`` (the filing's Rule 10b5-1 checkbox) and
    ``url`` (the filing's index page). Grants, exercises, tax withholding and gifts are other
    codes and are left out, as they are from Yahoo's feed."""
    today = today or date.today()
    since = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    cik = entry.get("cik")
    out = []
    for acc, f in (entry.get("filings") or {}).items():
        if "trades" not in f:
            continue
        who = (f.get("owners") or [{}])[0]
        for t in f["trades"]:
            if t.get("code") not in ("P", "S") or not t.get("date") or t["date"] < since:
                continue
            out.append({"date": t["date"][:10], "code": t["code"], "name": _person(who.get("name")),
                        "role": who.get("role"), "shares": t.get("shares"), "value": t.get("value"),
                        "plan": bool(f.get("plan")),
                        "url": (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
                                f"{acc}-index.htm") if cik else None})
    return out


def sec_rows_for(cache: dict, ticker: str, today: date) -> list[dict] | None:
    """A stock's SEC rows if its record was refreshed recently enough to trust, else None."""
    entry = cache.get(ticker)
    if not entry or not entry.get("checked"):
        return None
    if entry["checked"] < (today - timedelta(days=SEC_FRESH_DAYS)).isoformat():
        return None
    return rows_from_sec(entry, today)


def rows_from_yahoo(df, today: date | None = None) -> list[dict]:
    """Open-market purchases and sales from yfinance ``Ticker.insider_transactions``.

    Yahoo labels each row in ``Text``: "Purchase at price ..." and "Sale at price ..." are the
    open-market trades (Form 4 codes P and S). Grants, gifts, option exercises and tax withholding
    are other labels and are left out - the reason the provider's own "Purchases" summary is not
    used: it counts grants and exercises as purchases.
    """
    today = today or date.today()
    since = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    out = []
    if df is None or len(df) == 0:
        return out
    for _, r in df.iterrows():
        text = str(r.get("Text") or "")
        code = "P" if text.startswith("Purchase") else ("S" if text.startswith("Sale") else None)
        if code is None:
            continue
        d = r.get("Start Date")
        try:
            d = (d.date() if hasattr(d, "date") else datetime.strptime(str(d)[:10], "%Y-%m-%d").date()).isoformat()
        except (TypeError, ValueError):
            continue
        if d < since:
            continue
        sh, val = r.get("Shares"), r.get("Value")
        out.append({"date": d, "code": code, "name": str(r.get("Insider") or "").title() or None,
                    "role": str(r.get("Position") or "") or None,
                    "shares": float(sh) if sh is not None and sh == sh else None,
                    "value": float(val) if val is not None and val == val else None,
                    "plan": None})
    return out


def summarise_rows(rows: list[dict], today: date, window_days: int = 90, link: str | None = None) -> dict:
    """Open-market buying and selling by insiders over ``window_days`` (from any source)."""
    since = (today - timedelta(days=window_days)).isoformat()
    buys = [r for r in rows if r["code"] == "P" and r["date"] >= since]
    sells = [r for r in rows if r["code"] == "S" and r["date"] >= since]

    def total(rs):
        return round(sum(r.get("value") or 0 for r in rs), 2)

    buyers = sorted({r["name"] for r in buys if r.get("name")})
    recent = sorted(rows, key=lambda r: r["date"], reverse=True)[:8]
    # The plan flag exists only where the source carries it (SEC); None means "not known".
    known = any(r.get("plan") is not None for r in rows)
    planned = [r for r in sells if r.get("plan")]
    return {
        "window": window_days,
        "buy_n": len(buys), "buy_people": len(buyers), "buy_value": total(buys),
        "sell_n": len(sells), "sell_people": len({r["name"] for r in sells if r.get("name")}),
        "sell_value": total(sells),
        "sell_planned_n": len(planned) if known else None,
        "sell_planned_value": total(planned) if known else None,
        "cluster": len(buyers) >= CLUSTER_MIN_BUYERS,
        "officer_buy": any(re.search(r"chief|ceo|cfo|president|officer", (r.get("role") or ""), re.I) for r in buys),
        "recent": [{k: v for k, v in r.items() if v is not None} for r in recent],
        "link": link,
    }


def edgar_link(ticker: str) -> str:
    return ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
            f"{ticker}&type=4&dateb=&owner=include&count=40")


def summarise(cache: dict, today: date | None = None, write: bool = True) -> dict:
    today = today or date.today()
    meta = cache.get("_meta", {})
    out = {"as_of": today.isoformat(), "source": "SEC EDGAR Form 4 filings",
           "complete": meta.get("complete", False), "stocks": {}}
    for tk, entry in cache.items():
        if tk.startswith("_"):
            continue
        out["stocks"][tk] = summarise_ticker(entry, today)
    if write:
        SUMMARY_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


def load() -> dict | None:
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    import sys
    tks = json.loads((ROOT / "sp500_tickers.json").read_text())
    tks = [t["Ticker"] if isinstance(t, dict) else t for t in tks]
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else None
    c = refresh(tks, budget_seconds=budget)
    s = summarise(c)
    flagged = [(t, v["buy_people"], v["buy_value"]) for t, v in s["stocks"].items() if v["buy_n"]]
    print(f"{len(s['stocks'])} stocks summarised; {len(flagged)} with insider buying in 90 days")
    print(sorted(flagged, key=lambda x: -x[2])[:15])
