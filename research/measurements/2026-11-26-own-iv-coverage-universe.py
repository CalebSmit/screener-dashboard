"""Our own implied volatilities against the source field, over the whole universe from one refresh.

The 2026-11-04 decision (research/2026-11-03-open-source-quant-stack.md section 1d) asked for
coverage "on the whole universe from one 20:00 refresh before the switch"; the 2026-11-24 build
recorded both constructions on every reading and measured 51 stocks. This counts the cache the
2026-11-25 20:00 refresh wrote - the first refresh to read a run that kept the dividend fields -
and the raw fetch that refresh read. Read-only: prints, writes nothing.

    python research/measurements/2026-11-26-own-iv-coverage-universe.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import options_cache  # noqa: E402


def main() -> int:
    cache = json.loads((ROOT / "data" / "options" / "quotes.json").read_text(encoding="utf-8"))
    rd = cache["readings"]
    print(f"cache quote_date {cache.get('quote_date')}, written {cache.get('written')}, readings {len(rd)}")

    # The run the refresh read (newest raw fetch by mtime, the refresh's own rule).
    files = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    raw = files[0]
    meta = json.loads((raw.parent / "meta.json").read_text(encoding="utf-8")) if (raw.parent / "meta.json").exists() else {}
    recs = options_cache._records_from_latest_run(ROOT)
    df = pd.DataFrame(recs)
    print(f"newest raw fetch: {raw.parent.name} start {meta.get('start_time')}; columns kept: "
          f"{[c for c in options_cache.DIVIDEND_FIELDS if c in df.columns]}")
    rate = df.get("dividendRate")
    payers = df[(rate.fillna(0) > 0)] if rate is not None else df.iloc[0:0]
    exd = payers["exDividendDate"].notna().sum() if "exDividendDate" in payers else 0
    print(f"stocks {len(df)}; payers (dividendRate > 0) {len(payers)}; payers with exDividendDate {exd}")

    def has(r, k):
        v = r.get(k)
        return v is not None and v == v

    n = len(rd)
    src_iv = sum(has(r, "_ctx_opt_src_iv") for r in rd.values())
    own_iv = sum(has(r, "_ctx_opt_own_iv") for r in rd.values())
    src_sk = sum(has(r, "_ctx_opt_src_skew") for r in rd.values())
    own_sk = sum(has(r, "_ctx_opt_own_skew") for r in rd.values())
    print(f"ATM IV: source {src_iv}/{n}, ours {own_iv}/{n}")
    print(f"skew:   source {src_sk}/{n}, ours {own_sk}/{n}")
    both_iv = [(r["_ctx_opt_own_iv"], r["_ctx_opt_src_iv"]) for r in rd.values()
               if has(r, "_ctx_opt_own_iv") and has(r, "_ctx_opt_src_iv")]
    both_sk = [(r["_ctx_opt_own_skew"], r["_ctx_opt_src_skew"]) for r in rd.values()
               if has(r, "_ctx_opt_own_skew") and has(r, "_ctx_opt_src_skew")]
    only_src_iv = [t for t, r in rd.items() if has(r, "_ctx_opt_src_iv") and not has(r, "_ctx_opt_own_iv")]
    only_own_iv = [t for t, r in rd.items() if has(r, "_ctx_opt_own_iv") and not has(r, "_ctx_opt_src_iv")]
    only_src_sk = [t for t, r in rd.items() if has(r, "_ctx_opt_src_skew") and not has(r, "_ctx_opt_own_skew")]
    only_own_sk = [t for t, r in rd.items() if has(r, "_ctx_opt_own_skew") and not has(r, "_ctx_opt_src_skew")]
    print(f"ATM IV: only source {len(only_src_iv)} {sorted(only_src_iv)}; only ours {len(only_own_iv)}")
    print(f"skew:   only source {len(only_src_sk)} {sorted(only_src_sk)}; only ours {len(only_own_sk)}")
    if both_iv:
        d = [100 * (a - b) for a, b in both_iv]
        print(f"ATM IV, ours minus source (vol points), n={len(d)}: median {statistics.median(d):+.2f}, "
              f"p10 {sorted(d)[len(d)//10]:+.2f}, p90 {sorted(d)[9*len(d)//10]:+.2f}")
    if both_sk:
        d = [100 * (a - b) for a, b in both_sk]
        print(f"skew, ours minus source (vol points), n={len(d)}: median {statistics.median(d):+.2f}")
        sign_dis = sum((a < 0) != (b < 0) for a, b in both_sk)
        print(f"skew sign disagrees on {sign_dis} of {len(both_sk)}")
    own_neg = sum(r["_ctx_opt_own_skew"] < 0 for r in rd.values() if has(r, "_ctx_opt_own_skew"))
    src_neg = sum(r["_ctx_opt_src_skew"] < 0 for r in rd.values() if has(r, "_ctx_opt_src_skew"))
    print(f"skew negative: ours {own_neg} of {own_sk} ({own_neg / max(own_sk, 1):.1%}); "
          f"source {src_neg} of {src_sk} ({src_neg / max(src_sk, 1):.1%}); XZZ 2010 Table 1: about 1 in 16")
    # Payers whose reading has no IV of ours: dividend schedule unknown, or the mid failed.
    pay = set(payers["Ticker"])
    lost_pay = sorted(t for t in only_src_iv if t in pay)
    print(f"of the ATM IVs only the source has, payers {len(lost_pay)}: {lost_pay}")
    # Why ours is missing, from what the cache holds (it keeps no bid/ask): ours needs a mid inside
    # MAX_REL_SPREAD, which a cheap contract on a calm stock is the likeliest to miss. Compare the
    # source's own volatility and the straddle's price for the stocks ours lost against the rest.
    def med(xs):
        xs = [x for x in xs if x is not None and x == x]
        return statistics.median(xs) if xs else float("nan")

    for label, lost in (("ATM IV", set(only_src_iv)), ("skew", set(only_src_sk))):
        key = "_ctx_opt_src_iv"
        kept_set = [t for t, r in rd.items() if has(r, key) and t not in lost]
        print(f"{label}: source ATM IV median, lost {med([rd[t].get(key) for t in lost]):.3f} "
              f"vs kept {med([rd[t].get(key) for t in kept_set]):.3f}; straddle $ median, lost "
              f"{med([rd[t].get('_ctx_opt_straddle') for t in lost]):.2f} vs kept "
              f"{med([rd[t].get('_ctx_opt_straddle') for t in kept_set]):.2f}")
    eq = sum(abs(a - b) < 1e-9 for a, b in both_iv)
    print(f"ATM IV readings where ours equals the source exactly: {eq} of {len(both_iv)}")
    # The ATM leg alone: which side was missing.
    one_leg = {"call only": 0, "put only": 0, "neither": 0}
    for t in only_src_iv:
        r = rd[t]
        c, p = has(r, "_ctx_opt_own_call_iv"), has(r, "_ctx_opt_own_put_iv")
        one_leg["call only" if c and not p else "put only" if p and not c else "neither"] += 1
    print(f"ATM IV lost, our legs present: {one_leg}")
    # A skew is stored only with both legs, so read the lost ones through the ATM call: where ours
    # priced the ATM call, the skew's call leg (nearest moneyness 1) had a usable quote too, and the
    # missing leg is the out-of-the-money put.
    with_call = sum(has(rd[t], "_ctx_opt_own_call_iv") for t in only_src_sk)
    print(f"skew lost: {with_call} of {len(only_src_sk)} have our ATM call - the OTM put leg is the gap there")
    stamps = {}
    for r in rd.values():
        stamps[r.get("_ctx_opt_iv_src")] = stamps.get(r.get("_ctx_opt_iv_src"), 0) + 1
    print(f"main-column stamps today: {stamps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
