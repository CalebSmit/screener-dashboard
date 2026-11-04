"""What does Yahoo's `impliedVolatility` invert? A diagnostic after the pre-registered measurement.

The pre-registered comparison (2026-11-03-implied-vol-quantlib-vs-yahoo.py) found our American IVs
from bid/ask mids a median 3.2 vol points below Yahoo's at the money and the put skew's sign
different for 17 of 30 stocks. That says the two disagree, not why. This asks which simple
construction Yahoo's number matches best, on the same sample (every 10th stock), contract by
contract (ATM call, ATM put, the XZZ OTM put):

  A. American (QuantLib, discrete dividends) on the bid/ask mid  - ours
  B. American on the LAST TRADE price
  C. European Black-Scholes, r from the curve, no dividends, on the mid
  D. European Black-Scholes, r = 0, no dividends, on the last trade

and reports median |Yahoo - X| per construction, plus how old the last trades were. Diagnostic only;
it decides nothing (the decision on section 1a's proposal stays OPEN for another session).

    python research/measurements/2026-11-03-yahoo-iv-what-does-it-invert.py
"""
from __future__ import annotations

import importlib.util
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "ivm", ROOT / "research" / "measurements" / "2026-11-03-implied-vol-quantlib-vs-yahoo.py")
ivm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ivm)

import context_signals as cs  # noqa: E402
import options_cache  # noqa: E402


def bs_price(kind, S, K, T, r, v):
    if v <= 0 or T <= 0:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    d1 = (math.log(S / K) + (r + v * v / 2) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    if kind == "call":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def bs_iv(kind, S, K, T, r, price):
    lo, hi = 1e-4, 5.0
    if not price or price <= 0:
        return None
    if (bs_price(kind, S, K, T, r, lo) - price) * (bs_price(kind, S, K, T, r, hi) - price) > 0:
        return None
    for _ in range(100):
        m = (lo + hi) / 2
        if (bs_price(kind, S, K, T, r, m) - price) * (bs_price(kind, S, K, T, r, lo) - price) <= 0:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def main() -> int:
    import yfinance as yf

    today = options_cache.market_session_date(None)
    curve = ivm.treasury_curve()
    recs = sorted(options_cache._records_from_latest_run(ROOT), key=lambda r: r["Ticker"])[::10]
    raw = pd.read_parquet(max((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime),
                          columns=["Ticker", "dividendRate"]).set_index("Ticker")["dividendRate"]
    err = {k: [] for k in "ABCD"}
    ages, yneg, qneg, nsk = [], 0, 0, 0
    now = datetime.now(timezone.utc)
    for rec in recs:
        t = yf.Ticker(rec["Ticker"])
        try:
            ed = (datetime.fromtimestamp(int(rec["earningsTimestampStart"]), tz=timezone.utc).date()
                  if rec.get("earningsTimestampStart") else None)
        except (TypeError, ValueError, OverflowError, OSError):
            ed = None
        exp, _ = cs.choose_expiry(t.options, today, ed)
        if not exp:
            continue
        ch = t.option_chain(exp)
        S, _ = cs.chain_spot(getattr(ch, "underlying", None), rec.get("price_latest"))
        expiry = datetime.strptime(exp, "%Y-%m-%d").date()
        days = (expiry - today).days
        T, r = days / 365, ivm.rate_for(curve, days)
        divs = ivm.projected_dividends(t, raw.get(rec["Ticker"]), today, expiry)
        common = sorted(set(ch.calls["strike"]) & set(ch.puts["strike"]))
        if not common or not S:
            continue
        K = min(common, key=lambda k: abs(k - S))
        put_leg = cs._xzz_leg(ch.puts, S, cs.XZZ_PUT_LO, cs.XZZ_PUT_HI, cs.XZZ_PUT_TARGET, inclusive=False)
        call_leg = cs._xzz_leg(ch.calls, S, cs.XZZ_CALL_LO, cs.XZZ_CALL_HI, cs.XZZ_CALL_TARGET, inclusive=True)
        contracts = [("call", ch.calls, K), ("put", ch.puts, K)] + ([("put", ch.puts, put_leg[0])] if put_leg else [])
        for kind, frame, strike in contracts:
            row = frame.loc[frame["strike"] == strike].iloc[0].to_dict()
            y = row.get("impliedVolatility")
            mid, _ = cs._mid(row)
            last = row.get("lastPrice")
            lt = row.get("lastTradeDate")
            if lt is not None and not pd.isna(lt):
                ages.append((now - pd.Timestamp(lt).tz_convert("UTC").to_pydatetime()).total_seconds() / 3600)
            if not (y and y > 0):
                continue
            cand = {"A": ivm.ql_iv(kind, S, strike, mid, r, divs, today, expiry) if mid else None,
                    "B": ivm.ql_iv(kind, S, strike, last, r, divs, today, expiry) if last else None,
                    "C": bs_iv(kind, S, strike, T, r, mid) if mid else None,
                    "D": bs_iv(kind, S, strike, T, 0.0, last) if last else None}
            for k, v in cand.items():
                if v is not None:
                    err[k].append(abs(v - y) * 100)
        if put_leg and call_leg:
            nsk += 1
            yneg += (put_leg[1] - call_leg[1]) < 0
            pr = ch.puts.loc[ch.puts["strike"] == put_leg[0]].iloc[0].to_dict()
            cr = ch.calls.loc[ch.calls["strike"] == call_leg[0]].iloc[0].to_dict()
            pm, _ = cs._mid(pr)
            cm, _ = cs._mid(cr)
            if pm and cm:
                qp = ivm.ql_iv("put", S, put_leg[0], pm, r, divs, today, expiry)
                qc = ivm.ql_iv("call", S, call_leg[0], cm, r, divs, today, expiry)
                if qp is not None and qc is not None:
                    qneg += (qp - qc) < 0
    names = {"A": "American, mid (ours)", "B": "American, last trade", "C": "European, mid",
             "D": "European r=0, last trade"}
    print("median |Yahoo - X| in vol points, over the same contracts:")
    for k in "ABCD":
        e = sorted(err[k])
        if e:
            print(f"   {k}. {names[k]:28s} n={len(e):3d}  median {statistics.median(e):5.2f}  "
                  f"share within 0.5: {sum(x <= 0.5 for x in e) / len(e):.0%}")
    if ages:
        print(f"last trade age at the time of the pull, hours: median {statistics.median(ages):.1f}, "
              f"over 6h for {sum(a > 6 for a in ages)} of {len(ages)}")
    print(f"put skew negative (OTM put IV below ATM call IV): Yahoo {yneg} of {nsk}, ours {qneg} of {nsk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
