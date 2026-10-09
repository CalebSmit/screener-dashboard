"""The ranking's live track record: how its top names have actually done since each run.

WHY THIS EXISTS - 2026-10-08 (owner, ``plan/context-layer.md``). The most persuasive thing a
screener can show is not another metric; it is what happened next. Every comparable run since
2026-02-20 left a snapshot of every stock's rank, written on the day, so this is an
**out-of-sample** record - no backfill, no look-ahead, no survivorship in the selection.

What it measures, and what it is not:

* It measures **the ranking**: the top 25 by composite, and the top fifth against the bottom
  fifth, equal-weighted, rebalanced at the first comparable run of each calendar month and held
  through any month with no run (as someone following it would have done).
* Entry is the close of the **first trading day on or after** the run date (the 02:00 run uses the
  previous close, so a reader could not have traded at that price).
* It is **not** a portfolio to follow. The Model Portfolio panel was removed on 2026-08-26 because
  a fixed buy list on a public page is the closest this tool came to a recommendation; this page
  lists no current holdings and no weights to copy.
* It is **not** evidence for changing the methodology (CLAUDE.md rule 4): a few months of one
  market is far too short, and the methodology itself changed during the window - the page marks
  each change from ``METHODOLOGY_CHANGELOG.md``.
* No trading costs or taxes. Prices are dividend-adjusted closes (total return).
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "track_record.json"
PRICE_CACHE = ROOT / "data" / "track" / "prices.parquet"
TOP_N = 25
QUINTILE = 0.20
BENCHMARKS = {"RSP": "S&P 500 equal-weight (RSP)", "SPY": "S&P 500 (SPY)"}


# --------------------------------------------------------------------------- schedule
def rebalance_dates(run_dates: list[str]) -> list[str]:
    """The first run of each calendar month, in order."""
    seen, out = set(), []
    for d in sorted(run_dates):
        m = d[:7]
        if m not in seen:
            seen.add(m)
            out.append(d)
    return out


def portfolio_from_snapshot(ranks: dict[str, int]) -> dict[str, list[str]]:
    ordered = [t for t, _ in sorted(ranks.items(), key=lambda kv: kv[1])]
    n = len(ordered)
    q = max(1, int(round(n * QUINTILE)))
    return {"top": ordered[:TOP_N], "q1": ordered[:q], "q5": ordered[-q:]}


# --------------------------------------------------------------------------- prices
def load_prices(tickers: list[str], start: date, end: date | None = None, download=None) -> pd.DataFrame:
    """Dividend-adjusted daily closes, cached; only the missing tail is downloaded."""
    end = end or date.today()
    cached = None
    if PRICE_CACHE.exists():
        try:
            cached = pd.read_parquet(PRICE_CACHE)
        except (OSError, ValueError):
            cached = None
    need_all = cached is None or not set(tickers).issubset(cached.columns) or cached.index.min().date() > start
    dl_start = start if need_all else (cached.index.max().date() - timedelta(days=7))
    if download is None:
        import yfinance as yf

        def download(tks, s, e):
            df = yf.download(tks, start=s.isoformat(), end=(e + timedelta(days=1)).isoformat(),
                             auto_adjust=True, progress=False, threads=True)
            return df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tks[0]})
    fresh = download(sorted(set(tickers)), dl_start, end)
    fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
    if cached is not None and not need_all:
        prices = pd.concat([cached.loc[cached.index < fresh.index.min()], fresh]).sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]
        # A column that exists is not a column that is complete (2026-10-09: PSKY had 9
        # closes in the cache over a 160-day window while trading every day, so it dropped
        # out of three bottom-fifth baskets). Availability is tested by coverage of the
        # window the record reads, never by the column's presence: any ticker missing more
        # than a fifth of the window's trading days is downloaded again from the start.
        win = prices.loc[prices.index >= pd.Timestamp(start)]
        if len(win) >= 20:
            frac = win.reindex(columns=sorted(set(tickers))).notna().mean()
            thin = sorted(t for t, f in frac.items() if f < 0.8)
            if thin:
                try:
                    redo = download(thin, start, end)
                    redo.index = pd.to_datetime(redo.index).tz_localize(None)
                    for t in thin:
                        if t in redo.columns and (redo[t].notna().sum() > (win[t].notna().sum() if t in win.columns else 0)):
                            prices[t] = redo[t].reindex(prices.index).combine_first(prices[t]) if t in prices.columns else redo[t].reindex(prices.index)
                except Exception:  # noqa: BLE001 - a failed repair keeps what we had
                    pass
    else:
        prices = fresh.sort_index()
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICE_CACHE)
    return prices


# --------------------------------------------------------------------------- arithmetic
def run_basket(prices: pd.DataFrame, names: list[str], start: pd.Timestamp, end: pd.Timestamp | None,
               start_value: float) -> tuple[pd.Series, dict]:
    """Equal-weight buy-and-hold of ``names`` from the first close on/after ``start`` through
    ``end`` (exclusive of the next period's entry day). A name whose prices stop (acquired,
    removed) is held at its last price - cash, in effect - and reported."""
    window = prices.loc[(prices.index >= start) & ((prices.index <= end) if end is not None else True)]
    if window.empty:
        return pd.Series(dtype=float), {"held": 0, "stale": []}
    entry = window.iloc[0]
    have = [t for t in names if t in window.columns and pd.notna(entry.get(t)) and entry.get(t) > 0]
    if not have:
        return pd.Series(dtype=float), {"held": 0, "stale": []}
    px = window[have].ffill()
    cutoff = window.index[-1] - pd.Timedelta(days=7)   # a missing last day or two is a late print, not a stop
    stale = [t for t in have if window[t].last_valid_index() is not None and window[t].last_valid_index() < cutoff]
    units = (start_value / len(have)) / entry[have]
    value = (px * units).sum(axis=1)
    return value, {"held": len(have), "missing": sorted(set(names) - set(have)), "stale": stale}


def max_drawdown(series: pd.Series) -> float | None:
    if series is None or len(series) < 2:
        return None
    peak = series.cummax()
    return float((series / peak - 1).min())


def methodology_dates(path: Path | None = None) -> list[dict]:
    """Dated headings in METHODOLOGY_CHANGELOG.md - the moments the ranking's rules changed."""
    path = path or (ROOT / "METHODOLOGY_CHANGELOG.md")
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^##\s+(\d{4}-\d{2}-\d{2})\s*[-—:]?\s*(.*)", line)
            if m:
                out.append({"date": m.group(1), "title": m.group(2).strip()[:120]})
    except OSError:
        pass
    seen, uniq = set(), []
    for e in out:
        if e["date"] not in seen:
            seen.add(e["date"])
            uniq.append(e)
    return sorted(uniq, key=lambda e: e["date"])


def build(snapshots: list, prices: pd.DataFrame, today: date | None = None, changelog: list[dict] | None = None) -> dict:
    """``snapshots`` are history.RunSnapshot objects (date, ranks). Returns the payload block."""
    today = today or date.today()
    by_date = {s.date: s for s in snapshots}
    rebal = rebalance_dates(list(by_date))
    if len(rebal) < 1 or prices.empty:
        return {"available": False}
    baskets = {"top": [], "q1": [], "q5": []}
    bench = {b: [] for b in BENCHMARKS}
    values = {k: 100.0 for k in list(baskets) + list(bench)}
    periods = []
    prev_top = None
    unpriced = {k: set() for k in baskets}     # picks with no price at entry, per basket
    picks = {k: 0 for k in baskets}
    for i, d in enumerate(rebal):
        start = pd.Timestamp(d)
        nxt = pd.Timestamp(rebal[i + 1]) if i + 1 < len(rebal) else None
        # hold until the day before the next entry day
        end = None
        if nxt is not None:
            after = prices.index[prices.index >= nxt]
            end = after[0] - pd.Timedelta(days=1) if len(after) else None
        port = portfolio_from_snapshot(by_date[d].ranks)
        row = {"date": d, "entry": None}
        for k in baskets:
            v, info = run_basket(prices, port[k], start, end, values[k])
            picks[k] += len(port[k])
            unpriced[k] |= set(info.get("missing") or [])
            if v.empty:
                continue
            baskets[k].append(v)
            row[k] = round(float(v.iloc[-1] / values[k] - 1), 5)
            values[k] = float(v.iloc[-1])
            if k == "top":
                row["entry"] = v.index[0].strftime("%Y-%m-%d")
                row["exit"] = v.index[-1].strftime("%Y-%m-%d")
                row["held"] = info["held"]
                row["stale"] = info["stale"]
                row["changed"] = (len(set(port["top"]) - set(prev_top)) if prev_top is not None else None)
        for b in bench:
            v, _ = run_basket(prices, [b], start, end, values[b])
            if v.empty:
                continue
            bench[b].append(v)
            row[b] = round(float(v.iloc[-1] / values[b] - 1), 5)
            values[b] = float(v.iloc[-1])
        prev_top = port["top"]
        periods.append(row)

    def join(parts):
        if not parts:
            return pd.Series(dtype=float)
        s = pd.concat(parts)
        return s[~s.index.duplicated(keep="last")]

    curves = {k: join(v) for k, v in {**baskets, **bench}.items()}
    base = curves["top"]
    if base.empty:
        return {"available": False}
    # weekly samples for the chart, every curve on the same dates, starting at 100
    idx = base.index
    wk = idx[pd.Series(idx).groupby(pd.Series(idx).dt.to_period("W-FRI")).transform("max").eq(pd.Series(idx)).values]
    if wk[0] != idx[0]:
        wk = idx[:1].append(wk)
    series = {"d": [d.strftime("%Y-%m-%d") for d in wk]}
    for k, s in curves.items():
        s = s.reindex(idx).ffill()
        series[k] = [round(float(x), 3) if pd.notna(x) else None for x in s.loc[wk].values]
    total = {k: (round(float(curves[k].iloc[-1] / 100 - 1), 5) if not curves[k].empty else None) for k in curves}
    first, last = idx[0], idx[-1]
    beat = [p for p in periods if p.get("top") is not None and p.get("RSP") is not None]
    marks = [m for m in (changelog if changelog is not None else methodology_dates())
             if first.strftime("%Y-%m-%d") <= m["date"] <= last.strftime("%Y-%m-%d")]
    return {
        "available": True,
        "as_of": today.isoformat(),
        "start": first.strftime("%Y-%m-%d"), "end": last.strftime("%Y-%m-%d"),
        "days": int((last - first).days),
        "rule": {"top_n": TOP_N, "quintile": QUINTILE, "rebalance": "first comparable run of each month",
                 "entry": "close of the first trading day on or after the run"},
        "benchmarks": BENCHMARKS,
        "total": total,
        "spread": (round(total["q1"] - total["q5"], 5) if total.get("q1") is not None and total.get("q5") is not None else None),
        "max_dd": {k: (round(max_drawdown(curves[k]), 5) if not curves[k].empty else None) for k in curves},
        "periods": periods,
        "beat_rsp": sum(1 for p in beat if p["top"] > p["RSP"]),
        "periods_compared": len(beat),
        "series": series,
        "methodology_changes": marks,
        # Names the ranking picked that the free price source no longer serves (taken
        # private, acquired, renamed) - left out of their basket, which is a survivorship
        # gap the page states rather than hides (plan/context-layer.md item 6).
        "unpriced": {k: sorted(v) for k, v in unpriced.items()},
        "picks": picks,
        "gaps": [p["date"] for i, p in enumerate(periods[1:], 1)
                 if (pd.Timestamp(p["date"]) - pd.Timestamp(periods[i - 1]["date"])).days > 45],
    }


def build_from_disk(today: date | None = None, write: bool = True, live=None) -> dict:
    """Load the comparable snapshots and prices, build, and write ``data/track_record.json``.

    ``live`` is ``(date, {ticker: rank})`` for the run being published, which is not in the
    snapshot directory yet (the snapshot is written after the dashboard)."""
    import history
    today = today or date.today()
    kept, _ = history.select_comparable_runs()
    if live and live[1] and not any(s.date == live[0] for s in kept):
        kept = list(kept) + [history.RunSnapshot(date=live[0], ranks={t: int(r) for t, r in live[1].items() if r == r},
                                                 composites={}, cat_scores={})]
    if not kept:
        return {"available": False}
    tickers = sorted({t for s in kept for t in s.ranks}) + list(BENCHMARKS)
    start = date.fromisoformat(min(s.date for s in kept))
    prices = load_prices(tickers, start, today)
    out = build(kept, prices, today)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


def load() -> dict | None:
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    res = build_from_disk()
    print(json.dumps({k: v for k, v in res.items() if k not in ("series",)}, indent=1)[:4000])
