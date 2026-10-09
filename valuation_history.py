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
CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment",)
SHARES = ("WeightedAverageNumberOfDilutedSharesOutstanding",)

MONTHS = 60                 # five years of month-ends
MIN_MONTHS = 36             # fewer and the range is not shown
MCAP_TOLERANCE = 0.15       # |our market value / provider's - 1| above this -> not shown
MAX_TTM_AGE_DAYS = 460      # a TTM whose period ended longer ago than this is stale
ANNUAL = (340, 380)
YTD = (80, 290)


def _first_filed(f: pd.DataFrame, concepts: tuple) -> pd.DataFrame:
    """One row per (start, end): the first concept in preference order that reports it, at its
    first-filed value."""
    g = f[f["concept"].isin(concepts)].copy()
    if g.empty:
        return g
    g["_pref"] = g["concept"].map({c: i for i, c in enumerate(concepts)})
    g = g.sort_values(["start", "end", "_pref", "filed"])
    g = g.drop_duplicates(["start", "end"], keep="first")
    g["days"] = (g["end"] - g["start"]).dt.days
    return g[["start", "end", "filed", "val", "days"]]


def ttm_series(f: pd.DataFrame, concepts: tuple) -> pd.DataFrame:
    """Every trailing-12-month value the filings allow: columns ``end``, ``avail`` (the date the
    last of its components was filed), ``val``. Sorted by ``end``."""
    g = _first_filed(f, concepts)
    if g.empty:
        return pd.DataFrame(columns=["end", "avail", "val"])
    g = g.dropna(subset=["start"])
    ann = g[g["days"].between(*ANNUAL)]
    ytd = g[g["days"].between(*YTD)]
    out = [(r.end, r.filed, float(r.val)) for r in ann.itertuples()]
    day = pd.Timedelta(days=1)
    for r in ytd.itertuples():
        fy = ann[(ann["end"] - (r.start - day)).abs() <= pd.Timedelta(days=10)]
        prev = ytd[((ytd["end"] - (r.end - pd.Timedelta(days=365))).abs() <= pd.Timedelta(days=10))
                   & ((ytd["days"] - r.days).abs() <= 10)]
        if fy.empty or prev.empty:
            continue
        a, p = fy.iloc[-1], prev.iloc[-1]
        out.append((r.end, max(r.filed, a["filed"], p["filed"]), float(a["val"] + r.val - p["val"])))
    s = pd.DataFrame(out, columns=["end", "avail", "val"]).sort_values(["end", "avail"])
    return s.drop_duplicates("end", keep="first").reset_index(drop=True)


def _as_of(series: pd.DataFrame, when: pd.Timestamp):
    """The latest-ending value known by ``when``, or None if none or stale."""
    if series.empty:
        return None
    k = series[series["avail"] <= when]
    if k.empty:
        return None
    r = k.loc[k["end"].idxmax()]
    if (when - r["end"]).days > MAX_TTM_AGE_DAYS:
        return None
    return float(r["val"])


def share_series(f: pd.DataFrame) -> pd.DataFrame:
    """Diluted weighted-average shares per period: ``end``, ``filed``, ``val``, preferring the
    three-month figure where a filing gives several for the same period end."""
    g = _first_filed(f, SHARES)
    if g.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    g = g.assign(_q=(~g["days"].between(80, 100)).astype(int)).sort_values(["end", "_q", "filed"])
    return g.drop_duplicates("end", keep="first")[["end", "filed", "val"]].reset_index(drop=True)


def split_factor(splits: list, after: pd.Timestamp, until: pd.Timestamp | None = None) -> float:
    """Product of split ratios dated after ``after`` (and up to ``until``)."""
    k = 1.0
    for d, ratio in splits:
        d = pd.Timestamp(d)
        if d > after and (until is None or d <= until) and ratio and ratio > 0:
            k *= float(ratio)
    return k


def _shares_as_of(sh: pd.DataFrame, when: pd.Timestamp, splits: list):
    if sh.empty:
        return None
    k = sh[sh["filed"] <= when]
    if k.empty:
        return None
    r = k.loc[k["end"].idxmax()]
    if (when - r["end"]).days > MAX_TTM_AGE_DAYS:
        return None
    # In today's split-adjusted units, to match the split-adjusted price.
    return float(r["val"]) * split_factor(splits, r["filed"])


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
        "s": [None if x is None or not np.isfinite(x) else round(x, 4) for x in hist],
    }


def for_ticker(f: pd.DataFrame, closes: pd.Series, splits: list, price_now: float,
               mcap_now: float, today: pd.Timestamp, fcf: bool = True) -> dict | None:
    """The block for one stock, or None when it cannot be built or fails the self-check.

    ``closes``: split-adjusted month-end closes indexed by month-end date, oldest first, the
    current (incomplete) month excluded."""
    sh = share_series(f)
    shares_now = _shares_as_of(sh, today, splits)
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
        px = closes.get(m)
        s = _shares_as_of(sh, m, splits)
        mv = px * s if (px is not None and s and np.isfinite(px)) else None
        n = _as_of(ni, m)
        ey.append(n / mv if (mv and n is not None) else None)
        if fcf:
            o, c = _as_of(ocf, m), _as_of(cx, m)
            fy.append((o - c) / mv if (mv and o is not None and c is not None) else None)
    mv_now = price_now * shares_now
    out = {"asof": str(months[-1].date()) if len(months) else None,
           "m0": str(months[0].date()) if len(months) else None,
           "chk": round(check, 4)}
    n_now = _as_of(ni, today)
    e = _summary(n_now / mv_now if n_now is not None else None, ey)
    if e:
        out["ey"] = e
    if fcf:
        o, c = _as_of(ocf, today), _as_of(cx, today)
        x = _summary((o - c) / mv_now if (o is not None and c is not None) else None, fy)
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
