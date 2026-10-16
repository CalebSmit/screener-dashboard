"""Build backtest v2's inputs for 2012-2019 - the period registered as an untouched test on
2026-10-15 (``research/experiments/registry.yaml``, ``presample``) before any of it was built.

Three steps, each skipped when its output is already there (``--force`` redoes them):

1. **Membership, 2012-01 .. 2019-12**, from the Wikipedia revision current at each month-end
   (``universe_history``'s own parser and validation), into the tracked
   ``data/universe_history/sp500_membership.json``. The same revision's CIK column is kept **per
   month** in ``data/universe_history/ciks_by_month.json``: a symbol can belong to another company
   in another year, and only the registrant number tells them apart. The flat
   ``ticker_ciks.json`` gains only symbols it does not have - later revisions keep winning, as
   its builder (2026-09-30) intended.
2. **Daily prices from 2010-11-01** for every symbol the latest run scored (the companies v2 can
   reconstruct), close and adjusted close, into ``cache/v2_daily.parquet`` - one download, so
   the whole history carries one dividend adjustment.
3. **The index and the T-bill rate from 2008-06-01** (``^SP500TR``, ``^GSPC``, ``^IRX``) into
   ``cache/v2_index.parquet`` - the momentum rule's panic state reads 24 months of the index and
   126 days of its volatility before the first month.

Survivorship is the known limit: a member that has since left the index has no current record
and no price here. ``backtest_v2.py``'s census counts it, month by month.

    python research/measurements/2026-10-15-presample-inputs.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import universe_history as uh  # noqa: E402

START, END = "2012-01-31", "2019-12-31"
PRICES_FROM, INDEX_FROM = "2010-11-01", "2008-06-01"
BY_MONTH = ROOT / "data" / "universe_history" / "ciks_by_month.json"
FLAT = ROOT / "data" / "universe_history" / "ticker_ciks.json"
DAILY = ROOT / "cache" / "v2_daily.parquet"
INDEX = ROOT / "cache" / "v2_index.parquet"


def membership(force: bool) -> None:
    cache = uh.load_cache()
    by_month = json.loads(BY_MONTH.read_text(encoding="utf-8")) if BY_MONTH.exists() else {}
    flat = json.loads(FLAT.read_text(encoding="utf-8"))
    session = uh._session()
    todo = [d for d in uh.month_ends(START, END) if force or d not in cache or d not in by_month]
    print(f"membership: {len(todo)} month-ends to fetch", flush=True)
    for i, as_of in enumerate(todo, 1):
        for attempt in range(3):
            try:
                rev = uh.find_revision(as_of, session=session)
                html = uh.fetch_revision_html(rev["revid"], session=session)
                break
            except Exception as exc:  # noqa: BLE001 - a network hiccup, retried
                print(f"  retry {as_of}: {exc}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"could not fetch {as_of}")
        tickers = uh.parse_constituents(html)
        uh.validate_membership(tickers, as_of=as_of)
        cache[as_of] = uh.UniverseSnapshot(as_of=as_of, tickers=tuple(sorted(tickers)), revid=rev["revid"],
                                           revision_timestamp=rev["timestamp"],
                                           staleness_days=uh._staleness_days(as_of, rev["timestamp"]))
        ciks = uh.parse_ticker_ciks(html)
        by_month[as_of] = {"revid": rev["revid"], "ciks": dict(sorted(ciks.items()))}
        for tk, c in ciks.items():
            flat["ciks"].setdefault(tk, int(c))            # later revisions keep winning
        print(f"  [{i}/{len(todo)}] {as_of}: {len(tickers)} members, {len(ciks)} CIKs, "
              f"{cache[as_of].staleness_days}d old", flush=True)
        if i % 6 == 0 or i == len(todo):
            uh.save_cache(cache)
            BY_MONTH.write_text(json.dumps(dict(sorted(by_month.items())), indent=1), encoding="utf-8")
            FLAT.write_text(json.dumps(flat, indent=1), encoding="utf-8")
        time.sleep(0.5)


def month_ciks(force: bool) -> None:
    """Every cached month's own CIK column (added the same night for 2020-2026, whose membership was
    cached before CIKs were kept per month): the cached revision is fetched again by its id, so the
    membership itself is untouched."""
    cache = uh.load_cache()
    by_month = json.loads(BY_MONTH.read_text(encoding="utf-8")) if BY_MONTH.exists() else {}
    session = uh._session()
    todo = [d for d in sorted(cache) if force or d not in by_month]
    print(f"per-month CIKs: {len(todo)} months to fetch", flush=True)
    for i, as_of in enumerate(todo, 1):
        revid = cache[as_of].revid
        for attempt in range(3):
            try:
                html = uh.fetch_revision_html(revid, session=session)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  retry {as_of}: {exc}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"could not fetch {as_of}")
        ciks = uh.parse_ticker_ciks(html)
        by_month[as_of] = {"revid": revid, "ciks": dict(sorted(ciks.items()))}
        print(f"  [{i}/{len(todo)}] {as_of}: {len(ciks)} CIKs", flush=True)
        if i % 6 == 0 or i == len(todo):
            BY_MONTH.write_text(json.dumps(dict(sorted(by_month.items())), indent=1), encoding="utf-8")
        time.sleep(0.5)


def _download(symbols: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=start, auto_adjust=False, actions=False,
                                group_by="ticker", threads=True, progress=False)
                break
            except Exception as exc:  # noqa: BLE001
                print("  retry", i, exc, flush=True)
                time.sleep(5)
        for s in chunk:
            if s not in d.columns.get_level_values(0):
                continue
            x = d[s][["Close", "Adj Close", "Volume"]].dropna(how="all").copy()
            x.columns = ["close", "adj_close", "volume"]
            x["ticker"] = s if s.startswith("^") else uh.normalize_ticker(s)
            frames.append(x.reset_index().rename(columns={"Date": "date"}))
        print(f"  {i + len(chunk)} of {len(symbols)}", flush=True)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def prices(force: bool) -> None:
    if not force and DAILY.exists() and pd.read_parquet(DAILY, columns=["date"])["date"].min() <= pd.Timestamp(PRICES_FROM) + pd.Timedelta(days=7):
        print("daily prices: already from", PRICES_FROM)
    else:
        runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
        tickers = sorted(pd.read_parquet(runs[-1], columns=["Ticker"])["Ticker"])
        df = _download([t.replace(".", "-") for t in tickers] + ["^GSPC", "^SP500TR", "^IRX"], PRICES_FROM)
        df.to_parquet(DAILY)
        print("daily prices:", df.shape, df["ticker"].nunique(), "symbols, from", df["date"].min(), flush=True)
    if not force and INDEX.exists() and pd.read_parquet(INDEX, columns=["date"])["date"].min() <= pd.Timestamp(INDEX_FROM) + pd.Timedelta(days=7):
        print("index: already from", INDEX_FROM)
    else:
        idx = _download(["^SP500TR", "^GSPC", "^IRX"], INDEX_FROM)[["ticker", "date", "close"]]
        idx.to_parquet(INDEX)
        print("index:", idx.groupby("ticker")["date"].agg(["min", "max"]), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["membership", "prices", "ciks"], default=None)
    a = ap.parse_args()
    if a.only in (None, "membership"):
        membership(a.force)
    if a.only in (None, "prices"):
        prices(a.force)
    if a.only in (None, "ciks"):
        month_ciks(a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
