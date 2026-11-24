"""Implied volatility worked out here, from our own bid/ask mids. Display only, never scored.

Until 2026-11-24 every implied volatility on the options panel was the quote response's own
``impliedVolatility`` field, whose method - model, rate, dividends, which price it inverts - is
documented nowhere, and which eight constructions tried across two sessions could not rebuild
(research/2026-11-03-open-source-quant-stack.md sections 1b-1d). This module is the construction
that decision ADOPTED, and the one the two measurements validated:

* **price** - the mid of the bid and ask the reading already keeps (``context_signals._mid``, with
  its ``MAX_REL_SPREAD`` bar - never loosened to win coverage, CLAUDE.md 0.11);
* **model** - American exercise, QuantLib's finite-difference engine with a discrete dividend
  schedule (``VanillaOption.impliedVolatility``). Checked against an independently written
  Cox-Ross-Rubinstein tree on 19 of 19 contracts, 9 of them carrying a dividend, all within half a
  volatility point (section 1d (b));
* **rate** - FRED's constant-maturity Treasury yields at 1, 3, 6 and 12 months, converted to
  continuous compounding and interpolated linearly in time to the expiry (Cboe interpolates the
  same curve with a cubic spline; over 1-12 months the difference is far below a volatility point);
* **dividends** - the stock's annual rate in equal instalments, the first on its ex-dividend date
  and the rest one payment interval apart (:func:`dividend_schedule`).

Nothing here makes a network request per stock. The dividend fields ride the ``.info`` dict the
core fetch already pulls (``exDividendDate``, ``lastDividendValue``, 2026-11-24) - the measurement
read ``Ticker.dividends`` instead, a request per stock inside the pass that once tripped the rate
limiter (settled row "ctx"). The curve is four small FRED requests a day, cached under
``data/options/`` (the 20:00 refresh writes nowhere else).

Where an input is missing - QuantLib not installed, no curve, a dividend payer whose ex-date the
fetch did not carry - :func:`inverter` returns ``None`` and the caller keeps the source field,
stamped as such. A signal that can only fail closed blanks a panel; one that says which
construction it used never misleads.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CURVE_PATH = ROOT / "data" / "options" / "treasury_curve.json"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
# (FRED series, days to maturity) - the constant-maturity yields the measurement used.
CURVE_SERIES = (("DGS1MO", 30), ("DGS3MO", 91), ("DGS6MO", 182), ("DGS1", 365))
CURVE_MAX_AGE_HOURS = 20          # refetch after this; the series are daily
CURVE_STALE_DAYS = 10             # a cached curve older than this is not used at all

# Payments a year the dividend rule may assume. The ratio of the annual rate to the last payment
# picks the nearest; without a last payment, quarterly - what almost every S&P 500 payer pays.
PAYMENT_FREQUENCIES = (1, 2, 4, 12)
DEFAULT_FREQUENCY = 4

# QuantLib's solver: accuracy, evaluations, and the volatility bracket.
_SOLVER = (1e-6, 500, 1e-4, 5.0)
_LOCK = threading.Lock()          # QuantLib's evaluation date is process-global


def available() -> bool:
    """Whether QuantLib can be imported. Where it cannot, every reading keeps the source field."""
    try:
        import QuantLib  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available", never a crash
        return False
    return True


# --------------------------------------------------------------------------- the rate
def _parse_fred(text: str) -> tuple[str, float] | None:
    """(date, percent) of the newest observation in a FRED CSV, or None."""
    rows = [ln.split(",") for ln in text.strip().splitlines()[1:]]
    for d, v in reversed([r for r in rows if len(r) >= 2]):
        try:
            return d.strip(), float(v)
        except ValueError:
            continue
    return None


def _fetch_curve(today: date) -> dict:
    import requests
    start = (today - timedelta(days=30)).isoformat()
    points = []
    for sid, days in CURVE_SERIES:
        r = requests.get(FRED_CSV.format(sid=sid, start=start), timeout=30)
        r.raise_for_status()
        if not r.text.startswith("observation_date"):
            raise ValueError(f"unexpected FRED response for {sid}")
        got = _parse_fred(r.text)
        if got is None:
            raise ValueError(f"FRED {sid}: no observation")
        points.append({"sid": sid, "days": days, "date": got[0], "pct": got[1]})
    return {"fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"), "points": points}


def treasury_curve(today: date | None = None, path: Path | None = None, fetch=None,
                   now: float | None = None) -> list[tuple[int, float]] | None:
    """[(days, continuously compounded rate)] from FRED, cached daily; None if unavailable.

    A failed fetch falls back to the cached curve while its newest observation is within
    ``CURVE_STALE_DAYS``; beyond that the curve is not used and the readings keep the source field.
    """
    today = today or date.today()
    path = path or CURVE_PATH
    now = time.time() if now is None else now
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            cached = None
    fresh = cached is not None and path.exists() and (now - path.stat().st_mtime) < CURVE_MAX_AGE_HOURS * 3600
    data = cached if fresh else None
    if data is None:
        try:
            data = (fetch or _fetch_curve)(today)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001 - a stale curve within its bound beats no inversion
            data = cached
    return curve_from(data, today)


def curve_from(data: dict | None, today: date) -> list[tuple[int, float]] | None:
    """The cached curve as (days, cc rate), or None if any point is missing or too old."""
    pts = (data or {}).get("points") or []
    if len(pts) != len(CURVE_SERIES):
        return None
    out = []
    for p in pts:
        try:
            obs = datetime.strptime(str(p["date"]), "%Y-%m-%d").date()
            pct = float(p["pct"])
        except (KeyError, TypeError, ValueError):
            return None
        if (today - obs).days > CURVE_STALE_DAYS or not np.isfinite(pct):
            return None
        out.append((int(p["days"]), math.log(1 + pct / 100)))
    return sorted(out)


def rate_for(curve, days: int) -> float:
    """Linear interpolation in time; flat beyond the curve's ends."""
    xs, ys = [d for d, _ in curve], [r for _, r in curve]
    return float(np.interp(days, xs, ys))


# --------------------------------------------------------------------------- the dividends
def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _day(ts) -> date | None:
    """An epoch-seconds field (as ``.info`` serves ``exDividendDate``) as a UTC date."""
    f = _num(ts)
    if f is None or f <= 0:
        return None
    try:
        return datetime.fromtimestamp(int(f), tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


def annual_dividend(rec: dict) -> float | None:
    """The annual dividend per share: ``dividendRate``, else ``trailingAnnualDividendRate`` - the
    order ``factor_engine.DIVIDEND_RATE_FIELDS`` uses. 0.0 for a stock that pays none; None when
    neither field is there to say."""
    seen = False
    for k in ("dividendRate", "trailingAnnualDividendRate"):
        v = _num(rec.get(k))
        if v is None:
            continue
        seen = True
        if v > 0:
            return v
    return 0.0 if seen else None


def dividend_schedule(rec: dict, today: date, expiry: date) -> list[tuple[date, float]] | None:
    """The cash dividends expected to go ex after ``today`` and on or before ``expiry``.

    Equal instalments of the annual rate (:func:`annual_dividend`); payments a year from the rate
    over ``lastDividendValue``, rounded to the nearest of ``PAYMENT_FREQUENCIES`` (quarterly
    without one); the first on ``exDividendDate`` - the next ex-date when the source has announced
    it, the last one otherwise - and the rest one interval apart. The measurement's rule placed
    them from the dividend history instead, which costs a request per stock.

    ``[]`` for a stock that pays nothing; ``None`` when it pays and the fetch did not say when, so
    the caller keeps the source field rather than price an option as if no dividend fell in its life.
    """
    rate = annual_dividend(rec)
    if rate is None:
        return None
    if rate == 0:
        return []
    anchor = _day(rec.get("exDividendDate"))
    if anchor is None:
        return None
    last = _num(rec.get("lastDividendValue"))
    per_year = DEFAULT_FREQUENCY
    if last and last > 0:
        per_year = min(PAYMENT_FREQUENCIES, key=lambda n: abs(n - rate / last))
    step = timedelta(days=round(365 / per_year))
    amount = rate / per_year
    d = anchor
    while d <= today:
        d += step
    out = []
    while d <= expiry:
        out.append((d, amount))
        d += step
    return out


# --------------------------------------------------------------------------- the inversion
def _ql_date(d: date):
    import QuantLib as ql
    return ql.Date(d.day, d.month, d.year)


def _process(S: float, r: float, sigma: float, ev):
    import QuantLib as ql
    dc = ql.Actual365Fixed()
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)),
        ql.YieldTermStructureHandle(ql.FlatForward(ev, 0.0, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(ev, r, dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ev, ql.NullCalendar(), sigma, dc)))


def _option(kind: str, K: float, ev, ex):
    import QuantLib as ql
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, K)
    return ql.VanillaOption(payoff, ql.AmericanExercise(ev, ex))


def american_iv(kind: str, S: float, K: float, price: float, r: float, divs, today: date,
                expiry: date) -> float | None:
    """The volatility at which an American option with these dividends is worth ``price``."""
    import QuantLib as ql
    if not (S > 0 and K > 0 and price > 0) or expiry <= today:
        return None
    with _LOCK:
        ev = _ql_date(today)
        ql.Settings.instance().evaluationDate = ev
        opt = _option(kind, K, ev, _ql_date(expiry))
        dv = ql.DividendVector([_ql_date(d) for d, _ in divs], [a for _, a in divs])
        try:
            return float(opt.impliedVolatility(price, _process(S, r, 0.3, ev), dv, *_SOLVER))
        except RuntimeError:          # no volatility in the bracket prices it: below intrinsic, say
            return None


def american_price(kind: str, S: float, K: float, sigma: float, r: float, divs, today: date,
                   expiry: date) -> float:
    """The price at a known volatility - the round trip the tests check the inversion with."""
    import QuantLib as ql
    with _LOCK:
        ev = _ql_date(today)
        ql.Settings.instance().evaluationDate = ev
        opt = _option(kind, K, ev, _ql_date(expiry))
        dv = ql.DividendVector([_ql_date(d) for d, _ in divs], [a for _, a in divs])
        opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(_process(S, r, sigma, ev), dv, 400, 400))
        return float(opt.NPV())


def inverter(rec: dict, curve, today: date, expiry: date):
    """A function ``(kind, S, K, price) -> iv | None`` for one stock's chosen expiry, or None when
    QuantLib, the curve or the dividend schedule is unavailable (the caller keeps the source field).
    """
    if curve is None or not available():
        return None
    divs = dividend_schedule(rec or {}, today, expiry)
    if divs is None:
        return None
    r = rate_for(curve, (expiry - today).days)

    def iv(kind: str, S: float, K: float, price: float) -> float | None:
        return american_iv(kind, S, K, price, r, divs, today, expiry)

    return iv
