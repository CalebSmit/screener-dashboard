"""What would residual momentum cost the core fetch? (2026-11-21, deciding the 2026-11-20 proposal)

research/2026-11-20-jensens-alpha-in-momentum.md section 7 item 2 asks two things before the
proposal can be decided:

  1. Is a 37-month history the same request as today's 13-month one, and what does it cost in
     time and bytes?  (Settled row "ctx": an extra per-ticker request inside the core fetch
     tripped the rate limiter - a longer period on the same call is a different matter, but
     its cost has to be measured, not assumed.)
  2. Coverage: how many of today's universe have the 37 month-end closes the construction needs
     (36 monthly returns ending at the last month-end before the run)?

Part 1 times both periods on the same sample of tickers, alternating the order so neither
period always gets a warm connection.  Part 2 reads each ticker's first daily close from
backtest v2's daily cache (cache/v2_daily.parquet) and probes live only the tickers the cache
cannot answer.  No forward return, IC or backtest figure is computed (rules 4 and 5).

Run: python research/measurements/2026-11-21-residual-momentum-fetch-cost.py
Writes the .json beside this file.
"""
from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")

# The run date this decision was taken on; the last month-end before it and the close
# 36 months before that are what the construction needs.
RUN_DATE = pd.Timestamp("2026-11-21")
LAST_ME = (RUN_DATE - pd.offsets.MonthEnd(1)).normalize()          # 2026-10-31
NEEDED_FROM = (LAST_ME - pd.offsets.MonthEnd(36)).normalize()       # 2023-10-31


def timed_history(sym: str, period: str) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    h = yf.Ticker(sym).history(period=period, auto_adjust=True)
    dt = time.perf_counter() - t0
    nbytes = int(h.memory_usage(deep=True).sum()) if h is not None else 0
    return dt, (0 if h is None else len(h)), nbytes


def main() -> None:
    universe = json.loads((ROOT / "sp500_tickers.json").read_text())
    universe = [str(t["Ticker"]) if isinstance(t, dict) else str(t) for t in universe]

    # ---- part 2: coverage --------------------------------------------------------------
    d = pd.read_parquet(ROOT / "cache" / "v2_daily.parquet", columns=["date", "ticker", "adj_close"])
    d = d.dropna(subset=["adj_close"])
    d["date"] = pd.to_datetime(d["date"])
    first = d.groupby("ticker")["date"].min()
    yf_sym = {t: t.replace(".", "-") for t in universe}
    covered, short, unknown = [], [], []
    for t in universe:
        f = first.get(t, first.get(yf_sym[t]))
        if f is None or pd.isna(f):
            unknown.append(t)
        elif f <= NEEDED_FROM:
            covered.append(t)
        else:
            short.append(t)

    probed = {}
    for t in short + unknown:
        h = yf.Ticker(yf_sym[t]).history(period="5y", auto_adjust=True)
        if h is None or h.empty:
            probed[t] = None
            continue
        idx = h.index.tz_localize(None) if h.index.tz is not None else h.index
        probed[t] = str(idx.min().date())
    still_short = sorted(t for t, f in probed.items() if f is None or pd.Timestamp(f) > NEEDED_FROM)

    # ---- part 1: cost of the longer period on the same call -----------------------------
    random.seed(20261121)
    sample = random.sample([yf_sym[t] for t in covered], 30)
    rows = []
    for i, sym in enumerate(sample):
        order = ("13mo", "37mo") if i % 2 == 0 else ("37mo", "13mo")
        r = {"ticker": sym}
        for p in order:
            dt, n, b = timed_history(sym, p)
            r[p] = {"seconds": round(dt, 3), "bars": n, "frame_bytes": b}
        rows.append(r)

    def med(p, k):
        return statistics.median(r[p][k] for r in rows)

    result = {
        "run_date": str(RUN_DATE.date()),
        "last_month_end": str(LAST_ME.date()),
        "needed_from": str(NEEDED_FROM.date()),
        "coverage": {
            "universe": len(universe),
            "covered_by_v2_cache": len(covered),
            "checked_live": len(probed),
            "first_close_live": probed,
            "short_of_37_month_ends": still_short,
            "n_short": len(still_short),
            "share_covered": round(1 - len(still_short) / len(universe), 4),
        },
        "cost": {
            "sample": len(rows),
            "median_seconds_13mo": med("13mo", "seconds"),
            "median_seconds_37mo": med("37mo", "seconds"),
            "median_bars_13mo": med("13mo", "bars"),
            "median_bars_37mo": med("37mo", "bars"),
            "median_frame_bytes_13mo": med("13mo", "frame_bytes"),
            "median_frame_bytes_37mo": med("37mo", "frame_bytes"),
            "rows": rows,
        },
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "cost"}, indent=1)[:3000])
    print({k: v for k, v in result["cost"].items() if k != "rows"})


if __name__ == "__main__":
    main()
