"""The stock a browser test opens - one the published payload is guaranteed to carry.

Added 2026-11-04. Six browser modules opened named stocks - EXPE, APA, HST, BBY - as "a stock
with a full sheet", and four of those modules run in the 02:00 publish gate
(`scripts/check_published_claims.py`), where a failure **discards the night's data**. A name that
leaves the S&P 500 (a quarterly rebalance, an acquisition) would therefore have stopped the data loop
publishing on every run until a session noticed - CLAUDE.md settled row "data->red": a test that reads
the published payload asserts its own premise, never which stock today's data happened to carry.

So a test names the stock it would *like* and gets it back unchanged while the payload carries it -
every test reads exactly as it did before - and otherwise the best-ranked stock that meets the same
premise: ranked, priced, an analyst target, financials and a free-cash-flow yield (a non-bank sheet
with every section). Never random; the same payload gives the same answer.

JPM and MSFT are not routed through here: their tests assert things only those companies are (the
bank metric set, JPMorgan's name) and both are among the largest members of the index.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = ROOT / "dashboard_data.js"
DETAIL = ROOT / "dashboard_detail.js"


@functools.lru_cache(maxsize=1)
def _payload() -> dict:
    txt = PAYLOAD.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


@functools.lru_cache(maxsize=1)
def _detail() -> dict:
    """``dashboard_detail.js``'s per-stock part, or ``{}`` if the file is not there.

    The sheet-only fields have shipped in a second file since 2026-11-08
    (`plan/universe-expansion.md` phase 1), and `financials` joined them the same day. A premise
    about what the *sheet* has must therefore read the union, exactly as the page does when the
    loader merges the file into ``D.stock_detail`` - otherwise the premise below goes false for
    every stock at once and the four browser modules in the 02:00 publish gate fail, which
    discards the night's data (CLAUDE.md settled row "data->red")."""
    if not DETAIL.exists():
        return {}
    txt = DETAIL.read_text(encoding="utf-8")
    return (json.loads(txt[txt.index("{"):txt.rindex("}") + 1]) or {}).get("detail") or {}


def sheet(ticker: str) -> dict:
    """Everything the open sheet has for ``ticker``: first-paint payload plus the detail file."""
    return {**(_payload()["stock_detail"].get(ticker) or {}), **(_detail().get(ticker) or {})}


def _full_sheet(s: dict) -> bool:
    return (isinstance(s.get("composite"), (int, float)) and bool(s.get("price"))
            and bool(s.get("pt_mean")) and bool(s.get("financials"))
            and (s.get("raw") or {}).get("fcf_yield") is not None)


def stock(preferred: str, exclude: tuple[str, ...] = ()) -> str:
    """``preferred`` if the payload carries it, else the best-ranked full-sheet stock not excluded."""
    if not PAYLOAD.exists():  # the modules that call this skip without a payload; never fail collection
        return preferred
    sd = _payload()["stock_detail"]
    if preferred in sd and preferred not in exclude:
        return preferred
    ranked = sorted((r for r in _payload()["table_data"] if r.get("Rank")), key=lambda r: (r["Rank"], r["Ticker"]))
    for r in ranked:
        t = r["Ticker"]
        if t not in exclude and _full_sheet(sheet(t)):
            return t
    raise LookupError("the payload carries no ranked stock with a full sheet")


def search_text(ticker: str) -> str:
    """What a reader types to find ``ticker`` by name: the company's first word, lower-cased, when no
    other company starts with it (``expedia``), else the whole name."""
    sd = _payload()["stock_detail"]
    name = (sd[ticker].get("company") or ticker).strip()
    first = name.split()[0].lower()
    others = [t for t, s in sd.items() if t != ticker and (s.get("company") or "").lower().startswith(first)]
    return first if not others else name.lower()
