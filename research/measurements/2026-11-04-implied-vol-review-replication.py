"""Review of the QuantLib-vs-Yahoo implied volatility proposal: a disjoint replication.

Pre-registered in research/2026-11-03-open-source-quant-stack.md section 1c, committed (fb27f1c)
before this script computed anything. The 2026-11-04 13:00 session is the independent reader; it
did not write the proposal or the 11-03 measurement.

Read-only: pulls option chains and 1-minute bars for ~50 stocks and writes nothing but its own
printout. Requires QuantLib - a measurement dependency only, never imported by the pipeline.

Four things, all fixed in advance:

1. The 11-03 measurement repeated on ``recs[5::10]`` - disjoint from its own ``recs[::10]``.
2. What Yahoo's ``impliedVolatility`` field actually inverts, with the candidate the 11-03
   diagnostic could not test after the close: the last trade against the underlying price **at the
   moment of that trade**, from the day's 1-minute bars.
3. Whether Yahoo's skew is internally consistent - the gap between its two legs' last trades and
   the underlying's move over that gap.
4. The dividend gap in the author's cross-check: how many checked contracts carry a dividend, plus
   ten that do.

    python research/measurements/2026-11-04-implied-vol-review-replication.py
"""
from __future__ import annotations

import importlib.util
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import context_signals as cs  # noqa: E402
import options_cache  # noqa: E402

# The 11-03 script's own helpers, by path (its name starts with a digit). A replication should run
# the same instrument; the independent arm is its CRR tree, which is separate code from QuantLib.
_spec = importlib.util.spec_from_file_location(
    "iv1103", ROOT / "research" / "measurements" / "2026-11-03-implied-vol-quantlib-vs-yahoo.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

NORM = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(kind, S, K, sigma, r, T):
    """European Black-Scholes, no dividends - candidate 4's model."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    if kind == "call":
        return S * NORM(d1) - K * math.exp(-r * T) * NORM(d2)
    return K * math.exp(-r * T) * NORM(-d2) - S * NORM(-d1)


def bs_iv(kind, S, K, price, r, T):
    lo, hi = 1e-3, 5.0
    f = lambda v: bs_price(kind, S, K, v, r, T) - price
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def spot_at(bars, ts):
    """The underlying's price at the minute of ``ts``, from today's 1-minute bars. None if that
    timestamp is not inside the bars we hold (a last trade from an earlier day)."""
    if bars is None or bars.empty or ts is None or pd.isna(ts):
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert(bars.index.tz)
    lo, hi = bars.index[0], bars.index[-1]
    if t < lo - pd.Timedelta(minutes=1) or t > hi + pd.Timedelta(minutes=1):
        return None
    i = bars.index.get_indexer([t], method="nearest")[0]
    return float(bars["Close"].iloc[i])


def leg_rows(ch, S):
    """The four contracts the panel reads: ATM call, ATM put, XZZ put leg, XZZ call leg."""
    out = []
    common = sorted(set(ch.calls["strike"]) & set(ch.puts["strike"]))
    if common and S:
        K = min(common, key=lambda k: abs(k - S))
        for kind, frame in (("call", ch.calls), ("put", ch.puts)):
            r = frame.loc[frame["strike"] == K]
            if not r.empty:
                out.append(("atm_" + kind, kind, float(K), r.iloc[0].to_dict()))
    put_leg = cs._xzz_leg(ch.puts, S, cs.XZZ_PUT_LO, cs.XZZ_PUT_HI, cs.XZZ_PUT_TARGET, inclusive=False)
    call_leg = cs._xzz_leg(ch.calls, S, cs.XZZ_CALL_LO, cs.XZZ_CALL_HI, cs.XZZ_CALL_TARGET, inclusive=True)
    for name, leg, frame, kind in (("skew_put", put_leg, ch.puts, "put"), ("skew_call", call_leg, ch.calls, "call")):
        if leg is not None:
            r = frame.loc[frame["strike"] == leg[0]]
            if not r.empty:
                out.append((name, kind, float(leg[0]), r.iloc[0].to_dict()))
    return out


def main() -> int:
    import yfinance as yf

    M.self_test()
    today = options_cache.market_session_date(None)
    now = datetime.now(timezone.utc)
    curve = M.treasury_curve()
    print("curve (days, cc rate):", [(d, round(r, 4)) for d, r in curve])
    recs = sorted(options_cache._records_from_latest_run(ROOT), key=lambda r: r["Ticker"])
    raw = pd.read_parquet(max((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime),
                          columns=["Ticker", "dividendRate"]).set_index("Ticker")["dividendRate"]
    sample = recs[5::10]
    print(f"session {today}; pulled {now.isoformat(timespec='seconds')}; "
          f"sample {len(sample)} of {len(recs)} (every 10th by ticker, offset 5)")

    rows, contracts, crr_nodiv, crr_div, skewpairs = [], [], [], [], []
    for rec in sample:
        tk = rec["Ticker"]
        t = yf.Ticker(tk)
        try:
            ed = (datetime.fromtimestamp(int(rec["earningsTimestampStart"]), tz=timezone.utc).date()
                  if rec.get("earningsTimestampStart") else None)
        except (TypeError, ValueError, OverflowError):
            ed = None
        try:
            exp, _ = cs.choose_expiry(t.options, today, ed)
        except Exception as e:                                    # a source failure, reported
            rows.append({"t": tk, "why": f"expiry list failed: {type(e).__name__}"})
            continue
        if not exp:
            rows.append({"t": tk, "why": "no expiry"})
            continue
        try:
            ch = t.option_chain(exp)
            bars = t.history(period="1d", interval="1m")
        except Exception as e:
            rows.append({"t": tk, "why": f"chain failed: {type(e).__name__}"})
            continue
        S, _src = cs.chain_spot(getattr(ch, "underlying", None), rec.get("price_latest"))
        expiry = datetime.strptime(exp, "%Y-%m-%d").date()
        days = (expiry - today).days
        r = M.rate_for(curve, days)
        divs = M.projected_dividends(t, raw.get(tk), today, expiry)
        if not S:
            rows.append({"t": tk, "why": "no spot"})
            continue
        legs = leg_rows(ch, S)
        got = {}
        for name, kind, K, row in legs:
            mid, _ = cs._mid(row)
            yiv = pd.to_numeric(pd.Series([row.get("impliedVolatility")]), errors="coerce").iloc[0]
            last = pd.to_numeric(pd.Series([row.get("lastPrice")]), errors="coerce").iloc[0]
            lts = row.get("lastTradeDate")
            S_then = spot_at(bars, lts)
            age_h = None
            if lts is not None and not pd.isna(lts):
                ts = pd.Timestamp(lts)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
                age_h = (now - ts.to_pydatetime()).total_seconds() / 3600
            T = days / 365
            c = {"t": tk, "leg": name, "K": K, "kind": kind, "y": yiv, "age_h": age_h,
                 "ndiv": len(divs), "S": S, "S_then": S_then, "mid": mid, "last": last}
            # candidate 1 - ours: American, bid/ask mid, current spot
            c["c1"] = M.ql_iv(kind, S, K, mid, r, divs, today, expiry) if mid and mid > 0 else None
            # candidate 2 - American, last trade, current spot
            c["c2"] = M.ql_iv(kind, S, K, last, r, divs, today, expiry) if last and last > 0 else None
            # candidate 3 - American, last trade, spot at the moment of that trade
            c["c3"] = (M.ql_iv(kind, S_then, K, last, r, divs, today, expiry)
                       if (last and last > 0 and S_then) else None)
            # candidate 4 - European, last trade, spot at the moment of that trade
            c["c4"] = bs_iv(kind, S_then, K, last, r, T) if (last and last > 0 and S_then) else None
            contracts.append(c)
            got[name] = c
            if (name == "atm_call" and c["c1"] is not None and mid):
                tgt = crr_div if divs else crr_nodiv
                if len(tgt) < 10:
                    tgt.append((tk, len(divs), c["c1"], M.crr_iv(kind, S, K, mid, r, divs, today, expiry)))
        # the replication's headline numbers, on the panel's own construction
        out = {"t": tk, "S": S, "days": days, "ndiv": len(divs)}
        if all(k in got and got[k]["c1"] is not None and got[k]["y"] is not None and np.isfinite(got[k]["y"])
               for k in ("atm_call", "atm_put")):
            out["ql_atm"] = (got["atm_call"]["c1"] + got["atm_put"]["c1"]) / 2
            out["y_atm"] = (got["atm_call"]["y"] + got["atm_put"]["y"]) / 2
        if all(k in got and got[k]["c1"] is not None for k in ("skew_put", "skew_call")):
            out["ql_skew"] = got["skew_put"]["c1"] - got["skew_call"]["c1"]
            out["y_skew"] = got["skew_put"]["y"] - got["skew_call"]["y"]
            skewpairs.append((tk, got["skew_put"], got["skew_call"]))
        rows.append(out)

    # ----------------------------------------------------------------- 1. the replication
    ok = [x for x in rows if "ql_atm" in x]
    print(f"\n[1] priced at the money: {len(ok)} of {len(sample)}; not: "
          f"{[(x['t'], x.get('why', 'a leg with no usable quote or no IV')) for x in rows if 'ql_atm' not in x]}")
    d = [100 * (x["ql_atm"] - x["y_atm"]) for x in ok]
    if d:
        ad = sorted(abs(v) for v in d)
        print(f"    ATM IV, QuantLib minus Yahoo, vol points: median {statistics.median(d):+.2f}, "
              f"median |diff| {statistics.median(ad):.2f}, 90th pct {ad[int(0.9 * (len(ad) - 1))]:.2f}, "
              f"max {ad[-1]:.2f}")
        print(f"    (a) median |ATM diff| > 1 point: {'YES' if statistics.median(ad) > 1.0 else 'NO'}")
    sk = [x for x in rows if "ql_skew" in x]
    if sk:
        yneg = [x["t"] for x in sk if x["y_skew"] < 0]
        qneg = [x["t"] for x in sk if x["ql_skew"] < 0]
        flips = [x["t"] for x in sk if (x["ql_skew"] > 0) != (x["y_skew"] > 0)]
        sd = [100 * (x["ql_skew"] - x["y_skew"]) for x in sk]
        print(f"    skew: {len(sk)} stocks, median diff {statistics.median(sd):+.2f} pts, "
              f"sign differs for {len(flips)}")
        print(f"    skew negative - Yahoo's field {len(yneg)} of {len(sk)}; ours {len(qneg)} of {len(sk)}")
        print(f"      Yahoo-negative: {yneg}")
        print(f"      ours-negative:  {qneg}")

    # ----------------------------------------------------------------- 2. what Yahoo inverts
    print("\n[2] what Yahoo's impliedVolatility reproduces (median |difference| in vol points)")
    have_y = [c for c in contracts if c["y"] is not None and np.isfinite(c["y"])]
    for key, label in (("c1", "ours:     American, bid/ask mid,  spot now"),
                       ("c2", "cand 2:   American, last trade,   spot now"),
                       ("c3", "cand 3:   American, last trade,   spot AT that trade"),
                       ("c4", "cand 4:   European, last trade,   spot AT that trade")):
        v = [abs(100 * (c[key] - c["y"])) for c in have_y if c[key] is not None]
        if v:
            within = sum(1 for x in v if x <= 0.5)
            print(f"    {label}: n {len(v):4d}  median {statistics.median(v):5.2f}  "
                  f"within 0.5 pt {within:3d} ({100 * within / len(v):.0f}%)")
    ages = [c["age_h"] for c in have_y if c["age_h"] is not None]
    if ages:
        print(f"    last-trade age, hours: median {statistics.median(ages):.2f}, "
              f"over 1h for {sum(1 for a in ages if a > 1)} of {len(ages)}, "
              f"not today's session for {sum(1 for c in have_y if c['S_then'] is None)}")

    # ----------------------------------------------------------------- 3. is Yahoo's skew one moment?
    print("\n[3] Yahoo's two skew legs: are they inverted at the same moment?")
    gaps, moves = [], []
    for tk, p, c in skewpairs:
        if p["age_h"] is None or c["age_h"] is None:
            continue
        gaps.append(abs(p["age_h"] - c["age_h"]) * 60)
        if p["S_then"] and c["S_then"]:
            moves.append(100 * (p["S_then"] - c["S_then"]) / c["S_then"])
    if gaps:
        gs = sorted(gaps)
        print(f"    gap between the two legs' last trades, minutes: median {statistics.median(gs):.1f}, "
              f"90th pct {gs[int(0.9 * (len(gs) - 1))]:.1f}, max {gs[-1]:.1f}  (n {len(gs)})")
        print(f"    over 5 minutes apart: {sum(1 for g in gaps if g > 5)} of {len(gaps)}; "
              f"over 60: {sum(1 for g in gaps if g > 60)}")
    if moves:
        am = sorted(abs(m) for m in moves)
        print(f"    underlying between those two moments, %: median |move| {statistics.median(am):.3f}, "
              f"max {am[-1]:.3f}  (n {len(moves)})")

    # ----------------------------------------------------------------- 4. the dividend gap
    print("\n[4] independent CRR check (QuantLib vs a CRR tree, ATM calls on the bid/ask mid)")
    for label, checks in (("no dividend in the option's life", crr_nodiv),
                          ("WITH a projected dividend", crr_div)):
        good = 0
        print(f"    {label}: {len(checks)} contracts")
        for tk, nd, q, c in checks:
            same = c is not None and abs(q - c) <= 0.005
            good += same
            print(f"       {tk:6s} divs {nd}  QuantLib {q:.2%}  CRR "
                  f"{('%.2f%%' % (100 * c)) if c is not None else 'none':>7s}  {'ok' if same else 'DIFFERS'}")
        print(f"       reproduces within 0.5 vol points: {good} of {len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
