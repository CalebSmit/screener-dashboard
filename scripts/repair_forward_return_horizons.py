#!/usr/bin/env python3
"""One-off repair: re-measure every forward return to a horizon end that exists (2026-10-16).

The defect (METHODOLOGY_CHANGELOG.md 2026-10-16). The data loop runs at 02:00,
before that day's close. `compute_forward_returns` treated a horizon as complete
once `span` calendar days had elapsed, so on the morning of D+30 it measured the
1-month return when the newest close was D+29 (or the Friday before);
`_nearest_price` looked in either direction and silently used that earlier
close; `_fetch_prices_for_returns` cached the short frame under the (D, D+30)
key and served it on every later run; and when the 3-month horizon came due its
pass re-measured 1w and 1m with complete data, rewriting IC rows that had
already been reported. The engine is fixed; this script corrects what is
already on disk.

Rule applied, the engine's own: a horizon ends at the first close on or after
D + span (`improvement_engine._horizon_end_close`), and is filled only if that
close exists. The base price is unchanged (`_nearest_price` at D).

Price sources, per (run date, horizon), in order:
  1. the engine's own cached frames (improvement/price_cache/prices_<D>_*.parquet)
     whose last complete close reaches the horizon end - the data the engine
     originally used, so nothing but the end date moves;
  2. the main checkout's cache/v2_daily.parquet (`adj_close`, the same
     split-and-dividend adjusted close yfinance's auto_adjust returns);
  3. one batched yfinance download for any ticker neither covers.
Sources 1 and 2 are checked against each other on every overlapping window and
the largest disagreement is printed.

Writes improvement/performance_history.csv and regenerates
improvement/live_ic_history.csv through `compute_live_ic`. Price caches are
not touched: the fixed engine refuses the short ones and refetches.

Usage:
  python scripts/repair_forward_return_horizons.py --price-root <main checkout> [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import improvement_engine as ie  # noqa: E402


def engine_frames(run_date: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """(last trusted close, frame) for each cached frame starting at run_date.

    A frame's final bar is not trusted. Several were fetched during a trading
    session (the February backfill ran on 2026-03-09), so their last bar is an
    intraday price, not a close: on 2026-03-09 that bar disagrees with the
    published close by up to 10% (CF, the cruise lines, the airlines). Only
    closes strictly before the last complete bar are used.
    """
    out = []
    for p in sorted(ie.PRICE_CACHE_DIR.glob(f"prices_{run_date}_*.parquet")):
        f = pd.read_parquet(p)
        f.index = pd.DatetimeIndex(f.index)
        last = ie._last_complete_close(f)
        if last is None:
            continue
        f = f[f.index < last]
        trusted = ie._last_complete_close(f)
        if trusted is not None:
            out.append((trusted, f))
    # Longest first: the most recent fetch of a date.
    return sorted(out, key=lambda x: x[0], reverse=True)


def load_v2_daily(price_root: Path, today: pd.Timestamp) -> pd.DataFrame | None:
    path = price_root / "cache" / "v2_daily.parquet"
    if not path.exists():
        print(f"  (no {path}; skipping that source)")
        return None
    d = pd.read_parquet(path, columns=["date", "ticker", "adj_close"])
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= "2026-01-01") & (d["date"] < today)]
    return d.pivot_table(index="date", columns="ticker", values="adj_close")


def fetch_yf(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp,
             today: pd.Timestamp) -> pd.DataFrame | None:
    if not tickers:
        return None
    import yfinance as yf
    print(f"  yfinance: {len(tickers)} ticker(s), {start.date()} .. {end.date()}")
    data = yf.download(sorted(tickers), start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), auto_adjust=True,
                       progress=False, threads=True)
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"]
    else:
        px = data[["Close"]]
        px.columns = sorted(tickers)[:1]
    px = px.copy()
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px[px.index < today]


def fwd_return(series: pd.Series | None, d: pd.Timestamp, span: int) -> float:
    if series is None:
        return np.nan
    s = series.dropna()
    if s.empty:
        return np.nan
    base = ie._nearest_price(s, d)
    if base is None or base <= 0:
        return np.nan
    return ie._compute_fwd_return(s, d, span, base)


def column(frame: pd.DataFrame | None, ticker: str) -> pd.Series | None:
    if frame is None or ticker not in frame.columns:
        return None
    s = frame[ticker].dropna()
    return s if not s.empty else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-root", type=Path, default=ROOT,
                    help="checkout whose cache/v2_daily.parquet to read (read-only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-network", action="store_true")
    args = ap.parse_args()

    today = pd.Timestamp(datetime.now().date())
    perf = ie._normalize_performance_history(pd.read_csv(ie.PERFORMANCE_HISTORY_PATH))
    ic_before = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
    v2 = load_v2_daily(args.price_root, today)
    v2_last = ie._last_complete_close(v2) if v2 is not None else None
    print(f"performance_history.csv: {len(perf):,} rows, {perf['run_date'].nunique()} run dates")
    print(f"v2_daily last complete close: {v2_last.date() if v2_last is not None else None}")

    # Plan every (date, horizon): which source, and which tickers need yfinance.
    plan: dict[tuple[str, str], tuple[str, pd.DataFrame | None]] = {}
    yf_need: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    agreement = []
    for run_date, grp in perf.groupby("run_date"):
        d = pd.Timestamp(run_date)
        frames = engine_frames(run_date)
        for h, span in ie.HORIZON_DAYS.items():
            target = d + timedelta(days=span)
            if target >= today:
                plan[(run_date, h)] = ("not yet due", None)
                continue
            filled = grp.loc[grp[f"fwd_return_{h}"].notna(), "ticker"]
            complete = [f for last, f in frames if last >= target]
            if complete:
                plan[(run_date, h)] = ("engine cache", complete[0])
                missing = [t for t in filled if column(complete[0], t) is None]
            elif v2_last is not None and v2_last >= target:
                plan[(run_date, h)] = ("v2_daily", v2)
                missing = [t for t in filled if column(v2, t) is None]
            else:
                plan[(run_date, h)] = ("incomplete", None)
                missing = list(filled)
            for t in missing:
                lo, hi = yf_need.get(t, (d, target))
                yf_need[t] = (min(lo, d), max(hi, target))

        # Source agreement: engine cache vs v2_daily on the window they share.
        if v2 is not None:
            for last, f in frames:
                for t in f.columns:
                    a, b = column(f, t), column(v2, t)
                    if a is None or b is None:
                        continue
                    shared = a.index.intersection(b.index)
                    shared = shared[(shared >= d) & (shared <= last)]
                    if len(shared) < 2:
                        continue
                    ra = a[shared[-1]] / a[shared[0]] - 1
                    rb = b[shared[-1]] / b[shared[0]] - 1
                    agreement.append(abs(ra - rb))

    if agreement:
        agr = np.array(agreement)
        print(f"engine cache vs v2_daily, same window: {len(agr):,} ticker-windows, "
              f"median |diff| {np.median(agr):.2e}, 99th pct {np.quantile(agr, 0.99):.2e}, "
              f"max {agr.max():.2e}")

    yf_px = None
    if yf_need and not args.no_network:
        start = min(lo for lo, _ in yf_need.values()) - timedelta(days=7)
        end = max(hi for _, hi in yf_need.values()) + timedelta(days=ie.PRICE_FETCH_END_PAD_DAYS)
        yf_px = fetch_yf(list(yf_need), start, end, today)

    # Re-measure.
    old = perf[[f"fwd_return_{h}" for h in ie.HORIZON_DAYS]].copy()
    sources: dict[str, dict[str, int]] = {}
    for idx, row in perf.iterrows():
        d = pd.Timestamp(row["run_date"])
        t = row["ticker"]
        for h, span in ie.HORIZON_DAYS.items():
            # Only cells the engine filled are re-measured. A cell it left empty
            # (no price for that ticker) stays empty: filling it would be a
            # coverage change, not this correction, and would blur the
            # before/after this script exists to report.
            if pd.isna(old.at[idx, f"fwd_return_{h}"]):
                continue
            label, frame = plan[(row["run_date"], h)]
            val, used = np.nan, label
            if label in ("engine cache", "v2_daily"):
                val = fwd_return(column(frame, t), d, span)
            if np.isnan(val) and yf_px is not None and label != "not yet due":
                yv = fwd_return(column(yf_px, t), d, span)
                if not np.isnan(yv):
                    val, used = yv, "yfinance"
            if np.isnan(val):
                used = "unfilled" if label != "not yet due" else label
            perf.at[idx, f"fwd_return_{h}"] = val
            sources.setdefault(h, {}).setdefault(used, 0)
            sources[h][used] += 1

    print("\nForward returns re-measured, by source:")
    for h, counts in sources.items():
        print(f"  {h}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    for h in ie.HORIZON_DAYS:
        c = f"fwd_return_{h}"
        both = old[c].notna() & perf[c].notna()
        moved = both & ~np.isclose(old[c], perf[c], rtol=0, atol=1e-12)
        # An unchanged value keeps its stored text, so the diff shows only
        # the cells this repair corrects.
        perf.loc[both & ~moved, c] = old.loc[both & ~moved, c]
        print(f"  {h}: {int(old[c].notna().sum()):,} filled before, "
              f"{int(perf[c].notna().sum()):,} after; {int(moved.sum()):,} values changed, "
              f"{int((old[c].notna() & perf[c].isna()).sum()):,} now empty (end close not yet published, or no price for the ticker)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    perf.to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)
    # live_ic_history.csv is derived: replace every horizon slice from the
    # repaired history. A horizon with no rows left must lose its old rows too.
    for h in ie.HORIZON_DAYS:
        if ie.compute_live_ic(horizon=h) is None:
            ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
            ic[ic["horizon"].astype(str) != h].to_csv(ie.LIVE_IC_HISTORY_PATH, index=False)
    ic_after = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)

    # Before / after, every IC row that changed.
    key = ["run_date", "horizon"]
    m = ic_before[key + ["composite_ic"]].merge(
        ic_after[key + ["composite_ic"]], on=key, how="outer", suffixes=("_old", "_new"))
    m = m.sort_values(["horizon", "run_date"])
    changed = m[~np.isclose(m["composite_ic_old"].fillna(9), m["composite_ic_new"].fillna(9),
                            rtol=0, atol=5e-7)]
    print(f"\nComposite IC rows changed: {len(changed)} of {len(m)}")
    print("| run_date | horizon | old IC | new IC | change |")
    print("|---|---|---|---|---|")
    for _, r in changed.iterrows():
        o, n = r["composite_ic_old"], r["composite_ic_new"]
        delta = "" if (pd.isna(o) or pd.isna(n)) else f"{n - o:+.4f}"
        fo = "-" if pd.isna(o) else f"{o:.4f}"
        fn = "removed (end not yet published)" if pd.isna(n) else f"{n:.4f}"
        print(f"| {r['run_date']} | {r['horizon']} | {fo} | {fn} | {delta} |")

    print("\nEffective (non-overlapping) observations, before -> after:")
    for h in ie.HORIZON_DAYS:
        b = ic_before.loc[ic_before["horizon"].astype(str) == h, "run_date"]
        a = ic_after.loc[ic_after["horizon"].astype(str) == h, "run_date"]
        print(f"  {h}: {len(b)} rows / {ie._effective_observations(b, h)} effective -> "
              f"{len(a)} rows / {ie._effective_observations(a, h)} effective; "
              f"newest {b.max() if len(b) else None} -> {a.max() if len(a) else None}")
    for h in ie.HORIZON_DAYS:
        cat_cols = [c for c in ic_after.columns if c.endswith("_ic")]
        mb = ic_before[ic_before["horizon"] == h].set_index("run_date")[cat_cols]
        ma = ic_after[ic_after["horizon"] == h].set_index("run_date")[cat_cols]
        common = mb.index.intersection(ma.index)
        if len(common):
            diff = (ma.loc[common] - mb.loc[common]).abs()
            print(f"  {h}: largest |change| in any IC column {np.nanmax(diff.values):.4f}; "
                  f"mean composite IC {mb.loc[common, 'composite_ic'].mean():.4f} -> "
                  f"{ma.loc[common, 'composite_ic'].mean():.4f} over {len(common)} shared rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
