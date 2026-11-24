"""Coverage of our own implied volatilities against the source field, through the production code.

The decision (research/2026-11-03-open-source-quant-stack.md section 1d) left one cost unsized:
ours needs a usable bid/ask mid where the source field needs nothing. This runs the pipeline's own
option path (``options_cache._fetch_one`` with ``implied_vol``) on a sample and counts, per reading,
which constructions produced an at-the-money IV and a skew. Read-only: writes nothing but its
printout (the Treasury curve cache under the gitignored data/options/ aside).

Dividend fields: the newest run predates 2026-11-24's fetch change, so they are taken from each
stock's ``.info`` here - one extra request per sampled stock, a measurement cost the pipeline never
pays (its fetch keeps the fields from the ``.info`` it already pulls).

    python research/measurements/2026-11-24-own-iv-coverage.py [--every 10] [--offset 0]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import implied_vol  # noqa: E402
import options_cache  # noqa: E402


def main() -> int:
    import yfinance as yf

    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    today = options_cache.market_session_date(None)
    curve = implied_vol.treasury_curve(today)
    print(f"session {today}; QuantLib available: {implied_vol.available()}; curve: {curve}")
    recs = sorted(options_cache._records_from_latest_run(ROOT), key=lambda r: r["Ticker"])
    sample = recs[a.offset::a.every]
    n = {"readings": 0, "src_iv": 0, "own_iv": 0, "src_skew": 0, "own_skew": 0, "no_divs": 0,
         "src_skew_neg": 0, "own_skew_neg": 0}
    diffs, skew_diffs = [], []
    for r in sample:
        tk = r["Ticker"]
        try:
            info = yf.Ticker(tk).info
            rec = dict(r)
            for k in options_cache.DIVIDEND_FIELDS:
                rec[k] = info.get(k)
            out = options_cache._fetch_one(tk, r.get("price_latest"), r.get("earningsTimestampStart"),
                                           today, rec=rec, curve=curve)
        except Exception as e:  # noqa: BLE001 - reported, not dropped silently
            print(f"  {tk}: fetch failed {type(e).__name__}")
            continue
        if not out.get("_ctx_opt_status"):
            print(f"  {tk}: no reading")
            continue
        n["readings"] += 1
        has = {k: k in out for k in ("_ctx_opt_src_iv", "_ctx_opt_own_iv", "_ctx_opt_src_skew", "_ctx_opt_own_skew")}
        n["src_iv"] += has["_ctx_opt_src_iv"]
        n["own_iv"] += has["_ctx_opt_own_iv"]
        n["src_skew"] += has["_ctx_opt_src_skew"]
        n["own_skew"] += has["_ctx_opt_own_skew"]
        if implied_vol.dividend_schedule(rec, today, today) is None:
            n["no_divs"] += 1
        if has["_ctx_opt_src_skew"]:
            n["src_skew_neg"] += out["_ctx_opt_src_skew"] < 0
        if has["_ctx_opt_own_skew"]:
            n["own_skew_neg"] += out["_ctx_opt_own_skew"] < 0
        if has["_ctx_opt_src_iv"] and has["_ctx_opt_own_iv"]:
            diffs.append(100 * (out["_ctx_opt_own_iv"] - out["_ctx_opt_src_iv"]))
        if has["_ctx_opt_src_skew"] and has["_ctx_opt_own_skew"]:
            skew_diffs.append(100 * (out["_ctx_opt_own_skew"] - out["_ctx_opt_src_skew"]))
        print(f"  {tk}: src iv {out.get('_ctx_opt_src_iv')}, own iv {out.get('_ctx_opt_own_iv')}, "
              f"src skew {out.get('_ctx_opt_src_skew')}, own skew {out.get('_ctx_opt_own_skew')}")
    print(f"\nsample {len(sample)} (every {a.every}th by ticker, offset {a.offset}); counts: {n}")
    if diffs:
        print(f"ATM IV, ours minus source, vol points: median {statistics.median(diffs):+.2f}, "
              f"median |d| {statistics.median(abs(d) for d in diffs):.2f}, n {len(diffs)}")
    if skew_diffs:
        print(f"skew, ours minus source, vol points: median {statistics.median(skew_diffs):+.2f}, n {len(skew_diffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
