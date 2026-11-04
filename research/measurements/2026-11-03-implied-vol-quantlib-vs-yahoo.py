"""Our own implied volatilities (QuantLib, American, discrete dividends) against Yahoo's field.

Pre-registered in research/2026-11-03-open-source-quant-stack.md section 1a, committed (57958d5)
before this script computed anything. Read-only: pulls ~50 option chains after the close and
writes nothing but its own printout. Requires QuantLib (``pip install QuantLib``) - a measurement
dependency only, never imported by the pipeline.

    python research/measurements/2026-11-03-implied-vol-quantlib-vs-yahoo.py
"""
from __future__ import annotations

import io
import math
import statistics
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import context_signals as cs  # noqa: E402
import options_cache  # noqa: E402

# --------------------------------------------------------------------------- inputs
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
CURVE = [("DGS1MO", 30), ("DGS3MO", 91), ("DGS6MO", 182), ("DGS1", 365)]


def treasury_curve() -> list[tuple[int, float]]:
    """(days, continuously compounded rate) from FRED's latest constant-maturity yields."""
    out = []
    for sid, days in CURVE:
        with urllib.request.urlopen(FRED.format(sid=sid), timeout=30) as r:
            df = pd.read_csv(io.StringIO(r.read().decode()))
        v = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna()
        out.append((days, math.log(1 + float(v.iloc[-1]) / 100)))
    return out


def rate_for(curve, days: int) -> float:
    xs, ys = [d for d, _ in curve], [r for _, r in curve]
    return float(np.interp(days, xs, ys))


def projected_dividends(ticker, rate: float, today: date, expiry: date) -> list[tuple[date, float]]:
    """Equal instalments of the annual rate, on dates projected from the last ex-date at the median
    past interval - the pre-registered rule."""
    if not rate or not np.isfinite(rate) or rate <= 0:
        return []
    hist = ticker.dividends
    if hist is None or len(hist) < 2:
        return []
    days = [d.date() for d in pd.to_datetime(hist.index)]
    gaps = [(b - a).days for a, b in zip(days[-9:], days[-8:])]
    step = int(statistics.median(gaps)) if gaps else 91
    if step <= 0:
        return []
    per_year = max(1, round(365 / step))
    amt = rate / per_year
    out, d = [], days[-1]
    while True:
        d = d + timedelta(days=step)
        if d > expiry:
            break
        if d > today:
            out.append((d, amt))
    return out


# --------------------------------------------------------------------------- QuantLib
def ql_iv(kind: str, S: float, K: float, price: float, r: float, divs, today: date, expiry: date):
    import QuantLib as ql
    ev = ql.Date(today.day, today.month, today.year)
    ql.Settings.instance().evaluationDate = ev
    ex = ql.Date(expiry.day, expiry.month, expiry.year)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K)
    opt = ql.VanillaOption(payoff, ql.AmericanExercise(ev, ex))
    dc = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(ev, 0.0, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(ev, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ev, ql.NullCalendar(), 0.3, dc)))
    dv = ql.DividendVector([ql.Date(d.day, d.month, d.year) for d, _ in divs], [a for _, a in divs])
    try:
        return float(opt.impliedVolatility(price, process, dv, 1e-6, 500, 1e-4, 5.0))
    except RuntimeError:
        return None


def ql_price(kind, S, K, sigma, r, divs, today, expiry) -> float:
    """Round-trip helper for the self-test: price at a known volatility."""
    import QuantLib as ql
    ev = ql.Date(today.day, today.month, today.year)
    ql.Settings.instance().evaluationDate = ev
    ex = ql.Date(expiry.day, expiry.month, expiry.year)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K),
                           ql.AmericanExercise(ev, ex))
    dc = ql.Actual365Fixed()
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)), ql.YieldTermStructureHandle(ql.FlatForward(ev, 0.0, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(ev, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ev, ql.NullCalendar(), sigma, dc)))
    dv = ql.DividendVector([ql.Date(d.day, d.month, d.year) for d, _ in divs], [a for _, a in divs])
    opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, dv, 400, 400))
    return float(opt.NPV())


# --------------------------------------------------------------------------- the independent check
def crr_price(kind, S, K, sigma, r, divs, today, expiry, n=600) -> float:
    """Cox-Ross-Rubinstein, American, escrowed discrete dividends (Hull ch. 13 and 21)."""
    T = (expiry - today).days / 365
    dt = T / n
    pv = lambda t0: sum(a * math.exp(-r * ((d - today).days / 365 - t0)) for d, a in divs
                        if (d - today).days / 365 > t0)
    s_star = S - pv(0.0)
    u = math.exp(sigma * math.sqrt(dt))
    d_ = 1 / u
    p = (math.exp(r * dt) - d_) / (u - d_)
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    st = s_star * u ** (n - j) * d_ ** j
    val = np.maximum(st - K, 0) if kind == "call" else np.maximum(K - st, 0)
    for i in range(n - 1, -1, -1):
        jj = np.arange(i + 1)
        st = s_star * u ** (i - jj) * d_ ** jj + pv(i * dt)
        cont = disc * (p * val[:-1] + (1 - p) * val[1:])
        ex = (st - K) if kind == "call" else (K - st)
        val = np.maximum(cont, ex)
    return float(val[0])


def crr_iv(kind, S, K, price, r, divs, today, expiry):
    lo, hi = 1e-3, 4.0
    f = lambda v: crr_price(kind, S, K, v, r, divs, today, expiry) - price
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < 1e-6 or hi - lo < 1e-6:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def self_test():
    """QuantLib must recover a volatility it priced with, with and without dividends."""
    t0, t1 = date(2026, 11, 3), date(2026, 12, 18)
    for divs in ([], [(date(2026, 11, 20), 1.0)]):
        for kind in ("call", "put"):
            p = ql_price(kind, 100.0, 100.0, 0.25, 0.04, divs, t0, t1)
            iv = ql_iv(kind, 100.0, 100.0, p, 0.04, divs, t0, t1)
            assert iv is not None and abs(iv - 0.25) < 1e-3, (kind, divs, iv)
            c = crr_iv(kind, 100.0, 100.0, p, 0.04, divs, t0, t1)
            assert c is not None and abs(c - 0.25) < 5e-3, ("crr", kind, divs, c)
    print("self-test: QuantLib and the CRR tree each recover 25.0% from QuantLib's own price, with and without a dividend")


# --------------------------------------------------------------------------- the measurement
def main() -> int:
    import yfinance as yf

    self_test()
    today = options_cache.market_session_date(None)
    curve = treasury_curve()
    print("curve (days, cc rate):", [(d, round(r, 4)) for d, r in curve])
    recs = sorted(options_cache._records_from_latest_run(ROOT), key=lambda r: r["Ticker"])
    raw = pd.read_parquet(max((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime),
                          columns=["Ticker", "dividendRate"]).set_index("Ticker")["dividendRate"]
    sample = recs[::10]
    print(f"session {today}; sample {len(sample)} of {len(recs)} (every 10th by ticker)")
    rows, crr_checks = [], []
    for rec in sample:
        tk = rec["Ticker"]
        t = yf.Ticker(tk)
        try:
            ed = (datetime.fromtimestamp(int(rec["earningsTimestampStart"]), tz=__import__("datetime").timezone.utc).date()
                  if rec.get("earningsTimestampStart") else None)
        except (TypeError, ValueError, OverflowError):
            ed = None
        exp, _ = cs.choose_expiry(t.options, today, ed)
        if not exp:
            rows.append({"t": tk, "why": "no expiry"})
            continue
        ch = t.option_chain(exp)
        S, src = cs.chain_spot(getattr(ch, "underlying", None), rec.get("price_latest"))
        expiry = datetime.strptime(exp, "%Y-%m-%d").date()
        r = rate_for(curve, (expiry - today).days)
        divs = projected_dividends(t, raw.get(tk), today, expiry)
        common = sorted(set(ch.calls["strike"]) & set(ch.puts["strike"]))
        if not common or not S:
            rows.append({"t": tk, "why": "no common strike"})
            continue
        K = min(common, key=lambda k: abs(k - S))
        legs = {}
        for kind, frame, strike in (("call", ch.calls, K), ("put", ch.puts, K)):
            row = frame.loc[frame["strike"] == strike].iloc[0].to_dict()
            mid, _ = cs._mid(row)
            legs[kind] = (strike, mid, row.get("impliedVolatility"))
        put_leg = cs._xzz_leg(ch.puts, S, cs.XZZ_PUT_LO, cs.XZZ_PUT_HI, cs.XZZ_PUT_TARGET, inclusive=False)
        call_leg = cs._xzz_leg(ch.calls, S, cs.XZZ_CALL_LO, cs.XZZ_CALL_HI, cs.XZZ_CALL_TARGET, inclusive=True)
        out = {"t": tk, "S": S, "K": K, "days": (expiry - today).days, "r": r, "ndiv": len(divs)}
        ivs = {}
        for kind in ("call", "put"):
            strike, mid, yiv = legs[kind]
            ivs[kind] = (ql_iv(kind, S, strike, mid, r, divs, today, expiry) if mid else None, yiv)
        if all(v[0] is not None and v[1] is not None for v in ivs.values()):
            out["ql_atm"] = (ivs["call"][0] + ivs["put"][0]) / 2
            out["y_atm"] = (ivs["call"][1] + ivs["put"][1]) / 2
        if len(crr_checks) < 10 and ivs["call"][0] is not None and legs["call"][1]:
            c = crr_iv("call", S, K, legs["call"][1], r, divs, today, expiry)
            crr_checks.append((tk, ivs["call"][0], c))
        skew = {}
        for name, leg, frame, kind in (("put", put_leg, ch.puts, "put"), ("call", call_leg, ch.calls, "call")):
            if leg is None:
                continue
            row = frame.loc[frame["strike"] == leg[0]].iloc[0].to_dict()
            mid, _ = cs._mid(row)
            skew[name] = (ql_iv(kind, S, leg[0], mid, r, divs, today, expiry) if mid else None, leg[1])
        if len(skew) == 2 and all(v[0] is not None for v in skew.values()):
            out["ql_skew"] = skew["put"][0] - skew["call"][0]
            out["y_skew"] = skew["put"][1] - skew["call"][1]
        rows.append(out)

    ok = [x for x in rows if "ql_atm" in x]
    print(f"\npriced at the money: {len(ok)} of {len(sample)}; not: "
          f"{[(x['t'], x.get('why', 'a leg with no usable quote or no IV')) for x in rows if 'ql_atm' not in x]}")
    d = [100 * (x["ql_atm"] - x["y_atm"]) for x in ok]
    if d:
        ad = sorted(abs(v) for v in d)
        print(f"ATM IV, QuantLib minus Yahoo, vol points: median {statistics.median(d):+.2f}, median |diff| "
              f"{statistics.median(ad):.2f}, 90th pct |diff| {ad[int(0.9 * (len(ad) - 1))]:.2f}, max |diff| {ad[-1]:.2f}")
        for x in sorted(ok, key=lambda x: -abs(x["ql_atm"] - x["y_atm"]))[:6]:
            print(f"   {x['t']:6s} S {x['S']:.2f} K {x['K']:.1f} {x['days']}d divs {x['ndiv']}: "
                  f"QuantLib {x['ql_atm']:.1%} Yahoo {x['y_atm']:.1%}")
    sk = [x for x in rows if "ql_skew" in x]
    flips = [x["t"] for x in sk if (x["ql_skew"] > 0) != (x["y_skew"] > 0)]
    sd = [100 * (x["ql_skew"] - x["y_skew"]) for x in sk]
    if sk:
        print(f"skew (put leg - call leg): {len(sk)} stocks, median diff {statistics.median(sd):+.2f} pts, "
              f"sign changes {len(flips)} {flips}")
    bounds = [x["t"] for x in ok if not (cs.IV_MIN <= x["ql_atm"] <= cs.IV_MAX)]
    print(f"QuantLib ATM outside the panel's bounds [{cs.IV_MIN}, {cs.IV_MAX}]: {bounds}")
    print("\nindependent CRR check (first 10 ATM calls), QuantLib vs CRR:")
    good = 0
    for tk, q, c in crr_checks:
        same = c is not None and abs(q - c) <= 0.005
        good += same
        print(f"   {tk:6s} QuantLib {q:.2%}  CRR {('%.2f%%' % (100 * c)) if c is not None else 'none'}  {'ok' if same else 'DIFFERS'}")
    print(f"reproduces within 0.5 vol points: {good} of {len(crr_checks)}")
    if d:
        verdict = statistics.median(ad) > 1.0 and good == len(crr_checks) == 10
        print(f"\npre-registered bar (median |ATM diff| > 1 pt AND 10 of 10 reproduce): {'MET' if verdict else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
