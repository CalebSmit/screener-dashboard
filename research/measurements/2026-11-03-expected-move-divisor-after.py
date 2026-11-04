"""The first refresh under the divisor rule (2026-11-03 20:00), measured against the rule.

Reads the gitignored option cache (``data/options/quotes.json``) the 20:00 task wrote and the newest
full run's prices, and reports: how many readings carry their divisor and from which source; that
every at-the-money strike sits within 5% of its own divisor (the bar ``options_context`` applies);
that every move is its straddle over that divisor; and how far the new divisor sits from the run
close the old rule would have used - the size of what the fix changed. Run from the repo root:

    python research/measurements/2026-11-03-expected-move-divisor-after.py [ROOT]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import options_cache  # noqa: E402

cache = options_cache.load(ROOT)
R = cache.get("readings") or {}
print(f"cache quote_date {cache.get('quote_date')}, {len(R)} readings")
today = cache.get("quote_date")
fresh = {t: r for t, r in R.items() if r.get("_ctx_opt_quote_date") == today}
print(f"readings from {today}: {len(fresh)}")
print("divisor source:", dict(Counter(r.get("_ctx_opt_spot_src") for r in fresh.values())))

recs = {r["Ticker"]: r for r in options_cache._records_from_latest_run(ROOT)}
bad_strike, bad_move, gaps, big = [], [], [], []
for t, r in fresh.items():
    sp, k = r.get("_ctx_opt_spot"), r.get("_ctx_opt_atm_strike")
    if sp and k and abs(k / sp - 1) > 0.05 + 1e-9:
        bad_strike.append(t)
    mv, st = r.get("_ctx_opt_move"), r.get("_ctx_opt_straddle")
    if mv is not None and st is not None and sp and abs(mv - st / sp) > 1e-9:
        bad_move.append(t)
    old = (recs.get(t) or {}).get("price_latest")
    if sp and old:
        g = abs(sp / old - 1)
        gaps.append(g)
        if g > 0.05:
            big.append((t, round(old, 2), round(sp, 2), f"{g:.1%}"))
print(f"strike more than 5% from its own divisor: {len(bad_strike)} {bad_strike[:10]}")
print(f"move not equal to straddle / divisor: {len(bad_move)} {bad_move[:10]}")
if gaps:
    print(f"divisor vs the run close the old rule used: median {statistics.median(gaps):.2%}, "
          f"over 1% for {sum(g > 0.01 for g in gaps)} of {len(gaps)}, over 5% for {len(big)}")
    print("over 5%:", sorted(big, key=lambda x: -float(x[3][:-1]))[:10])
for t in ("NOW", "AAPL", "JPM", "VLO"):
    r = R.get(t) or {}
    print(t, {k: r.get(k) for k in ("_ctx_opt_quote_date", "_ctx_opt_spot", "_ctx_opt_spot_src",
                                      "_ctx_opt_atm_strike", "_ctx_opt_move")})
