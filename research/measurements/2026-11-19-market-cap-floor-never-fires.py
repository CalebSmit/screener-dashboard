"""Can ``universe.min_market_cap`` exclude a stock? Measured 2026-11-19.

``config.yaml`` has said ``min_market_cap: 2e9  # $2B minimum`` since February.
``factor_engine.apply_universe_filters`` applies it "if min_mc > 0 and 'marketCap' in df.columns",
and both of its callers - ``run_screener.py`` (the live run) and ``backtest_v2.rank_month`` (the
research engine) - hand it the frame ``compute_metrics`` returns. This script asks three things:

1. Does that frame carry a ``marketCap`` column? (It carries ``size_log_mcap``, the scored metric.)
2. What does the filter do to the newest run's metrics frame at floors no S&P 500 member could
   pass - $1 trillion, $1 quadrillion?
3. How many name-months would a floor that *did* work have removed from the research engine's
   point-in-time history, 2012-2026, at $2B? (``size_log_mcap`` is -ln(market cap), so the cap is
   recoverable from the cached months.) This is the size of what the published engine numbers
   would have moved by had the control ever been connected - a description, not evidence of
   anything about returns (CLAUDE.md rule 5).

Read-only. Run from the repo root: ``python research/measurements/2026-11-19-market-cap-floor-never-fires.py``
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402
import run_context  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    run = run_context.latest_run_dir()
    metrics = pd.read_parquet(run / "01_raw_metrics.parquet")
    raw = pd.read_parquet(run / "00_raw_fetch.parquet")
    print(f"run {run.name}: metrics frame {metrics.shape}, raw fetch {raw.shape}")
    print(f"1. 'marketCap' in the frame the filter is handed: {'marketCap' in metrics.columns}"
          f"  (in the raw fetch: {'marketCap' in raw.columns})")

    print("2. the filter as shipped, on the newest run's metrics frame:")
    for floor in (2e9, 1e12, 1e15):
        c = dict(cfg)
        c["universe"] = dict(cfg["universe"], min_market_cap=floor)
        kept = len(fe.apply_universe_filters(metrics.copy(), c))
        print(f"   floor ${floor:,.0f}: {len(metrics)} in -> {kept} out")

    mc = pd.to_numeric(raw["marketCap"], errors="coerce")
    print(f"   today's members below $2B by the raw fetch: {int((mc < 2e9).sum())} of {int(mc.notna().sum())}"
          f"; smallest ${mc.min() / 1e9:.3f}B ({raw.loc[mc.idxmin(), 'Ticker']})")

    print("3. the research engine's cached months (cache/v2_metrics), at a $2B floor that worked:")
    months = sorted((ROOT / "cache" / "v2_metrics").glob("????-??-??.parquet"))
    rows = []
    absurd: dict[str, int] = {}      # under $100M: not a size an S&P 500 member has ever been
    low: dict[str, int] = {}         # $100M-$2B: some real (a fallen member), some not
    for p in months:
        d = pd.read_parquet(p, columns=["Ticker", "size_log_mcap"])
        s = pd.to_numeric(d["size_log_mcap"], errors="coerce")
        rows.append({"month": p.stem, "n": int(s.notna().sum()), "below": int((-s < math.log(2e9)).sum())})
        for tkr in d.loc[-s < math.log(1e8), "Ticker"]:
            absurd[tkr] = absurd.get(tkr, 0) + 1
        for tkr in d.loc[(-s >= math.log(1e8)) & (-s < math.log(2e9)), "Ticker"]:
            low[tkr] = low.get(tkr, 0) + 1
    t = pd.DataFrame(rows)
    if t.empty:
        print("   no cached months on this machine")
        return
    t["year"] = t["month"].str[:4]
    by = t.groupby("year").agg(months=("month", "count"), names=("n", "sum"), below=("below", "sum"))
    by["share"] = (by["below"] / by["names"] * 100).round(2)
    print(by.to_string())
    print(f"   all {len(t)} months: {int(t['below'].sum())} of {int(t['n'].sum())} name-months below $2B "
          f"({t['below'].sum() / t['n'].sum() * 100:.2f}%); months with at least one: {int((t['below'] > 0).sum())}; "
          f"newest month with one: {t.loc[t['below'] > 0, 'month'].max() if (t['below'] > 0).any() else 'none'}")
    # 4. A side finding, recorded because a *working* $2B floor would have acted on it: some of
    # those market caps are not small companies, they are wrong share counts in the point-in-time
    # record (price x filed shares). Berkshire Hathaway at under $1B is the plainest.
    print(f"4. of those, under $100M (no index member is): {sum(absurd.values())} name-months, "
          f"{len(absurd)} tickers: {sorted(absurd.items(), key=lambda kv: -kv[1])}")
    print(f"   $100M-$2B: {sum(low.values())} name-months, {len(low)} tickers; the ten most frequent: "
          f"{sorted(low.items(), key=lambda kv: -kv[1])[:10]}")


if __name__ == "__main__":
    main()
