"""Valuation against the stock's own history - display-only context (north-star gap 3).

The screener's valuation percentiles are cross-sectional: they say a stock's earnings yield is
higher than most of its sector's, never whether it is higher than *the stock's own* usual. This
module answers the second question for two yields, from each company's own SEC filings:

* **earnings yield** = trailing-12-month net income / market value
* **free-cash-flow yield** = trailing-12-month (operating cash flow - capital expenditure) / market value

at each of the past 60 month-ends and today, and reports where today sits in that range.

**Context only** (CLAUDE.md settled row "ctx"): the result rides the payload's ``ctx`` and never
reaches ``raw``, ``pct``, a category score or the composite. ``tests/test_valuation_history.py``.

Three construction rules, each the answer to a way this goes wrong:

1. **As first filed.** Each month-end uses only figures filed with the SEC by that date, at the
   value first reported - a later restatement does not rewrite what the history showed then.
2. **Trailing twelve months the standard way**: the latest fiscal year plus the year-to-date of the
   current year minus the same year-to-date a year earlier, so a quarterly filer's TTM moves every
   quarter rather than once a year. Cash-flow statements report only year-to-date figures, which is
   why the construction works from year-to-date periods throughout.
3. **Splits.** Prices are split-adjusted; share counts are as reported. A count filed before a split
   is multiplied by that split's ratio, or NVDA's 2023 market value reads ten times too small
   (plan/valuation-vs-own-history.md, "The trap"). Market value = split-adjusted price x diluted
   weighted-average shares, split-adjusted.

**The self-check that decides whether a stock is shown at all:** today's market value built this
way must agree with the data provider's own market capitalisation within ``MCAP_TOLERANCE``. A
company whose filed share count does not reproduce its market value today (several share classes
reported separately, an unusual tag) cannot be trusted for the past either, so it gets nothing.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRICE_PATH = ROOT / "data" / "valhist" / "monthly.parquet"
SPLIT_PATH = ROOT / "data" / "valhist" / "splits.json"

NI = ("NetIncomeLoss", "ProfitLoss")                      # preference order, per period
OCF = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
SHARES = ("WeightedAverageNumberOfDilutedSharesOutstanding",       # preference order
          "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
          "WeightedAverageNumberOfSharesOutstandingBasic")

MONTHS = 60                 # five years of month-ends
MIN_MONTHS = 36             # fewer and the range is not shown
MCAP_TOLERANCE = 0.15       # |our market value / provider's - 1| above this -> not shown
MAX_TTM_AGE_DAYS = 460      # a TTM whose period ended longer ago than this is stale
# "Today's" yield must rest on a recent period: VTRS's filings carry only annual figures, so the
# 460-day bound let a fiscal-year-old TTM stand as today's (-17.1% against Yahoo's -2.0%).
TODAY_MAX_AGE_DAYS = 200
ANNUAL = (340, 380)
YTD = (80, 290)


def _days(s: pd.Series) -> np.ndarray:
    """Dates as integer days since 1970 (NaT -> a very small number)."""
    return pd.to_datetime(s).values.astype("datetime64[D]").astype(np.int64)


def _day(ts) -> int:
    return int(np.datetime64(pd.Timestamp(ts).date(), "D").astype(np.int64))


def _first_filed(f: pd.DataFrame, concepts: tuple) -> pd.DataFrame:
    """One row per (start, end): the first concept in preference order that reports it, at its
    first-filed value."""
    g = f[f["concept"].isin(concepts)]
    if g.empty:
        return g
    g = g.assign(_pref=g["concept"].map({c: i for i, c in enumerate(concepts)}))
    g = g.sort_values(["start", "end", "_pref", "filed"]).drop_duplicates(["start", "end"], keep="first")
    return g.assign(days=(g["end"] - g["start"]).dt.days)[["start", "end", "filed", "val", "days"]]


def ttm_series(f: pd.DataFrame, concepts: tuple) -> dict:
    """Every trailing-12-month value the filings allow, as arrays ``end``, ``avail`` (the day the
    last of its components was filed) and ``val``, in integer days, sorted by ``end``."""
    g = _first_filed(f, concepts)
    g = g.dropna(subset=["start"]) if not g.empty else g
    if g.empty:
        return {"end": np.array([], dtype=np.int64), "avail": np.array([], dtype=np.int64), "val": np.array([])}
    s, e, fl = _days(g["start"]), _days(g["end"]), _days(g["filed"])
    v, d = g["val"].to_numpy(float), g["days"].to_numpy()
    ann = (d >= ANNUAL[0]) & (d <= ANNUAL[1])
    ytd = (d >= YTD[0]) & (d <= YTD[1])
    ends, avail, vals = list(e[ann]), list(fl[ann]), list(v[ann])
    ai, yi = np.flatnonzero(ann), np.flatnonzero(ytd)
    for i in yi:
        # the fiscal year ending the day before this year-to-date began ...
        fy = ai[np.abs(e[ai] - (s[i] - 1)) <= 10]
        # ... and the same year-to-date a year earlier
        pv = yi[(np.abs(e[yi] - (e[i] - 365)) <= 10) & (np.abs(d[yi] - d[i]) <= 10)]
        if len(fy) == 0 or len(pv) == 0:
            continue
        a, p = fy[-1], pv[-1]
        ends.append(e[i]); avail.append(max(fl[i], fl[a], fl[p])); vals.append(v[a] + v[i] - v[p])
    ends, avail, vals = np.array(ends), np.array(avail), np.array(vals)
    # one value per period end: the earliest-available construction
    o = np.lexsort((avail, ends))
    ends, avail, vals = ends[o], avail[o], vals[o]
    keep = np.r_[True, ends[1:] != ends[:-1]]
    return {"end": ends[keep], "avail": avail[keep], "val": vals[keep]}


def _as_of(series: dict, when: int, max_age: int = MAX_TTM_AGE_DAYS):
    """The latest-ending value known by day ``when``, or None if none or stale."""
    k = series["avail"] <= when
    if not k.any():
        return None
    i = np.flatnonzero(k)[np.argmax(series["end"][k])]
    if when - series["end"][i] > max_age:
        return None
    return float(series["val"][i])


def _fcf_as_of(ocf: dict, cx: dict, when: int, max_age: int = MAX_TTM_AGE_DAYS):
    """Operating cash flow minus capex for the latest period BOTH report by ``when`` - one
    trailing twelve months, never two (VLO paired cash flow to 2026-06 with capex to 2025-09)."""
    ko, kc = ocf["avail"] <= when, cx["avail"] <= when
    common = np.intersect1d(ocf["end"][ko], cx["end"][kc])
    if not len(common):
        return None
    e = common.max()
    if when - e > max_age:
        return None
    o = ocf["val"][ko][ocf["end"][ko] == e][0]
    c = cx["val"][kc][cx["end"][kc] == e][0]
    return float(o - c)


def share_series(f: pd.DataFrame) -> dict:
    """Weighted-average shares per period end (diluted, else the fallbacks in ``SHARES``), as
    arrays ``end``, ``avail`` (filed), ``val``; the three-month figure where a filing gives several
    for the same period end."""
    g = _first_filed(f, SHARES)
    if g.empty:
        return {"end": np.array([], dtype=np.int64), "avail": np.array([], dtype=np.int64), "val": np.array([])}
    g = g.assign(_q=(~g["days"].between(80, 100)).astype(int)).sort_values(["end", "_q", "filed"])
    g = g.drop_duplicates("end", keep="first")
    return {"end": _days(g["end"]), "avail": _days(g["filed"]), "val": g["val"].to_numpy(float)}


def split_factor(splits: list, after: int, until: int | None = None) -> float:
    """Product of split ratios dated after day ``after`` (and up to ``until``)."""
    k = 1.0
    for d, ratio in splits:
        dd = _day(d)
        if dd > after and (until is None or dd <= until) and ratio and ratio > 0:
            k *= float(ratio)
    return k


def _shares_as_of(sh: dict, when: int, splits: list):
    k = sh["avail"] <= when
    if not k.any():
        return None
    i = np.flatnonzero(k)[np.argmax(sh["end"][k])]
    if when - sh["end"][i] > MAX_TTM_AGE_DAYS:
        return None
    # In today's split-adjusted units, to match the split-adjusted price.
    return float(sh["val"][i]) * split_factor(splits, int(sh["avail"][i]))


def _summary(now: float, hist: list) -> dict | None:
    h = [x for x in hist if x is not None and np.isfinite(x)]
    if now is None or not np.isfinite(now) or len(h) < MIN_MONTHS:
        return None
    arr = np.array(h)
    return {
        "now": round(now, 5),
        "pct": round(float((arr < now).mean()) * 100, 1),     # month-ends with a lower yield
        "lo": round(float(arr.min()), 5), "med": round(float(np.median(arr)), 5),
        "hi": round(float(arr.max()), 5), "n": len(h),
        # the monthly series for the chart, in basis points (whole numbers keep the file small)
        "s": [None if x is None or not np.isfinite(x) else int(round(x * 10000)) for x in hist],
    }


def for_ticker(f: pd.DataFrame, closes: pd.Series, splits: list, price_now: float,
               mcap_now: float, today: pd.Timestamp, fcf: bool = True) -> dict | None:
    """The block for one stock, or None when it cannot be built or fails the self-check.

    ``closes``: split-adjusted month-end closes indexed by month-end date, oldest first, the
    current (incomplete) month excluded."""
    sh = share_series(f)
    td = _day(today)
    shares_now = _shares_as_of(sh, td, splits)
    if not shares_now or not price_now or not mcap_now or mcap_now <= 0:
        return None
    check = price_now * shares_now / mcap_now - 1
    if abs(check) > MCAP_TOLERANCE:
        return None
    ni = ttm_series(f, NI)
    ocf, cx = (ttm_series(f, OCF), ttm_series(f, CAPEX)) if fcf else (None, None)
    months = closes.index[-MONTHS:]
    ey, fy = [], []
    for m in months:
        # Yahoo labels a monthly bar with the month's first day; its close is the month's last
        # trading day, so "filed by then" is judged at the month's end.
        px, md = closes.get(m), _day(m + pd.offsets.MonthEnd(0))
        s = _shares_as_of(sh, md, splits)
        mv = px * s if (px is not None and s and np.isfinite(px)) else None
        n = _as_of(ni, md)
        ey.append(n / mv if (mv and n is not None) else None)
        if fcf:
            v = _fcf_as_of(ocf, cx, md)
            fy.append(v / mv if (mv and v is not None) else None)
    mv_now = price_now * shares_now
    out = {"asof": str(months[-1].date()) if len(months) else None,
           "m0": str(months[0].date()) if len(months) else None,
           "chk": round(check, 4)}
    n_now = _as_of(ni, td, TODAY_MAX_AGE_DAYS)
    e = _summary(n_now / mv_now if n_now is not None else None, ey)
    if e:
        out["ey"] = e
    if fcf:
        v = _fcf_as_of(ocf, cx, td, TODAY_MAX_AGE_DAYS)
        x = _summary(v / mv_now if v is not None else None, fy)
        if x:
            out["fy"] = x
    return out if ("ey" in out or "fy" in out) else None


def build(rows: list[dict], facts: pd.DataFrame, monthly: pd.DataFrame, splits: dict,
          today: date | None = None, no_fcf: set | None = None) -> dict:
    """``rows``: dicts with Ticker, price and market cap (``currentPrice``/``marketCap``)."""
    today = pd.Timestamp(today or date.today())
    no_fcf = no_fcf or set()
    by = {t: g for t, g in facts.groupby("ticker")}
    # Complete months only: a bar for the current month is a partial month.
    cur = today.to_period("M").to_timestamp()
    monthly = monthly[monthly.index < cur]
    out = {}
    for r in rows:
        t = r.get("Ticker")
        if t not in by or t not in monthly.columns:
            continue
        try:
            b = for_ticker(by[t], monthly[t].dropna(), splits.get(t, []),
                           r.get("currentPrice") or r.get("price_latest"), r.get("marketCap"),
                           today, fcf=t not in no_fcf)
        except Exception:  # noqa: BLE001 - one company's odd filings must not cost the rest
            b = None
        if b:
            out[t] = b
    return out


def _yahoo(t: str) -> str:
    return t.replace(".", "-")


def monthly_prices(tickers: list[str], today: date | None = None, log=print,
                   allow_write: bool = True) -> tuple[pd.DataFrame, dict]:
    """Split-adjusted (not dividend-adjusted) month-end closes for six years, and each ticker's
    split history [[date, ratio], ...]. Cached; refetched when the cache lacks the last complete
    month. One batched download, plus one request per ticker that split in the window."""
    today = pd.Timestamp(today or date.today())
    last_complete = (today.to_period("M") - 1).to_timestamp()
    if PRICE_PATH.exists() and SPLIT_PATH.exists():
        p = pd.read_parquet(PRICE_PATH)
        if len(p.index) and p.index.max() >= last_complete and set(tickers) <= set(p.columns):
            return p, json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if not allow_write:
        if PRICE_PATH.exists() and SPLIT_PATH.exists():
            return pd.read_parquet(PRICE_PATH), json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        return pd.DataFrame(), {}
    import yfinance as yf
    ymap = {_yahoo(t): t for t in tickers}
    d = yf.download(list(ymap), start=(today - pd.DateOffset(years=6)).strftime("%Y-%m-%d"),
                    interval="1mo", auto_adjust=False, actions=True, progress=False, threads=True)
    close = d["Close"].rename(columns=ymap)
    sp = d["Stock Splits"].rename(columns=ymap) if "Stock Splits" in d.columns.get_level_values(0) else None
    close.index = pd.to_datetime(close.index)
    # Complete months only: the current month's bar holds a part-month price, and cached it would
    # later stand as that month's close (2026-10-09 review).
    close = close[close.index < today.to_period("M").to_timestamp()]
    splits: dict = {}
    if sp is not None:
        for t in [c for c in sp.columns if (sp[c].fillna(0) > 0).any()]:
            # Exact dates: the monthly bar only says which month a split fell in, and the split
            # rule compares the split date with a filing date.
            try:
                s = yf.Ticker(_yahoo(t)).splits
                s.index = pd.to_datetime(s.index).tz_localize(None)
                s = s[s.index >= close.index.min()]
                splits[t] = [[str(i.date()), float(v)] for i, v in s.items() if v and v > 0]
            except Exception:  # noqa: BLE001
                continue
            time.sleep(0.2)
    got = close.notna().any().sum()
    if got < 0.9 * len(tickers):
        log(f"  Valuation history: prices for only {got} of {len(tickers)} - not caching")
        return close, splits
    PRICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(PRICE_PATH)
    SPLIT_PATH.write_text(json.dumps(splits, indent=0), encoding="utf-8")
    log(f"  Valuation history: monthly prices for {got} stocks, {len(splits)} with splits")
    return close, splits


def attach(raw: list[dict], refresh: bool = True, log=print) -> int:
    """Compute the block for every fetched stock and set it on the raw records as context:
    ``_ctx_valhist`` (the block, JSON, for the page) and ``_ctx_vh_ey_pct`` / ``_ctx_vh_fy_pct``
    (today's place in its own range, scalars the context log records for the signal's
    out-of-sample test). Returns how many stocks got a block. ``refresh=False`` (a --tickers run)
    reads the caches but never rewrites them."""
    from factor_engine import _is_bank_like
    from sec_fundamentals import refresh_companyfacts
    live = [r for r in raw if r.get("Ticker") and "_error" not in r]
    tickers = [r["Ticker"] for r in live]
    facts = refresh_companyfacts(tickers, log=log, allow_write=refresh)
    if facts is None or facts.empty:
        log("  Valuation history: no SEC facts cache - skipped")
        return 0
    monthly, splits = monthly_prices(tickers, log=log, allow_write=refresh)
    if monthly.empty:
        log("  Valuation history: no monthly prices - skipped")
        return 0
    # Free cash flow means little for a bank or an insurer, whose operating cash flow moves
    # with loans, deposits and claims - the same reason the screener scores them differently.
    no_fcf = {r["Ticker"] for r in live
              if _is_bank_like(r["Ticker"], r.get("_gics_sector") or r.get("sector") or "",
                               r.get("industry") or "", r.get("_gics_sub"))}
    out = build(live, facts, monthly, splits, no_fcf=no_fcf)
    for r in live:
        b = out.get(r["Ticker"])
        if not b:
            continue
        r["_ctx_valhist"] = json.dumps(b, separators=(",", ":"))
        if "ey" in b:
            r["_ctx_vh_ey_pct"] = b["ey"]["pct"]
        if "fy" in b:
            r["_ctx_vh_fy_pct"] = b["fy"]["pct"]
    return len(out)
