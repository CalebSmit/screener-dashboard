"""The market backdrop: rates, credit, volatility, inflation and jobs, from FRED.

WHY THIS EXISTS - 2026-10-08 (owner, ``plan/context-layer.md``): "maybe ... macro data?" The
screener ranks stocks against each other; nothing on the page said what kind of market they were
being ranked in. This module fetches a handful of public series from FRED (Federal Reserve Bank
of St. Louis - free, no key), caches them under ``data/market/``, and turns each into a number,
a change, where it sits in its own 10-year history, and one plain sentence.

**Context only.** Nothing here enters a score. The one place the screener already reacts to the
market - momentum's weight moving with the volatility regime - is computed in ``factor_engine``
from the universe itself, not from these series, and this module does not touch it.

The readings describe; they do not forecast and never say what to do. Each threshold is either a
published rule (Sahm 2019; the 10-year minus 3-month spread used by Estrella & Mishkin 1998 and
the New York Fed) or stated as this series' own 10-year percentile.
"""

from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "market"
OUT_PATH = ROOT / "data" / "market_context.json"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
HISTORY_YEARS = 15
MAX_AGE_HOURS = 20

# id -> (label, unit, how to show it, frequency)
SERIES = {
    "DGS10":    ("10-year Treasury yield", "%", "level", "daily"),
    "DGS2":     ("2-year Treasury yield", "%", "level", "daily"),
    "T10Y3M":   ("10-year minus 3-month yield", "pp", "level", "daily"),
    "T10Y2Y":   ("10-year minus 2-year yield", "pp", "level", "daily"),
    "BAA10Y":   ("Corporate credit spread (Baa over 10-year)", "pp", "level", "daily"),
    "VIXCLS":   ("VIX (expected S&P 500 volatility)", "", "level", "daily"),
    "DFF":      ("Federal funds rate", "%", "level", "daily"),
    "UNRATE":   ("Unemployment rate", "%", "level", "monthly"),
    "CPIAUCSL": ("Inflation (CPI, year over year)", "%", "yoy", "monthly"),
    "ICSA":     ("Initial jobless claims", "k", "thousands", "weekly"),
}


# --------------------------------------------------------------------------- fetch
def _cache_path(sid: str) -> Path:
    return DATA_DIR / f"{sid}.csv"


def fetch_series(sid: str, session=None, today: date | None = None) -> pd.Series:
    """One FRED series as a float Series indexed by date. Uses the cache when it is fresh;
    falls back to a stale cache rather than failing when FRED cannot be reached."""
    today = today or date.today()
    path = _cache_path(sid)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < MAX_AGE_HOURS * 3600
    if not fresh:
        try:
            import requests
            start = (today - timedelta(days=365 * HISTORY_YEARS)).isoformat()
            r = (session or requests).get(FRED_CSV.format(sid=sid, start=start), timeout=60)
            r.raise_for_status()
            if not r.text.startswith("observation_date"):
                raise ValueError("unexpected FRED response")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text, encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - a stale cache is better than no backdrop
            if not path.exists():
                raise RuntimeError(f"FRED {sid} unavailable and no cache: {e}") from e
    return read_series_csv(path.read_text(encoding="utf-8"))


def read_series_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


# --------------------------------------------------------------------------- summarise
def _value_on_or_before(s: pd.Series, when: pd.Timestamp):
    prior = s.loc[s.index <= when]
    return float(prior.iloc[-1]) if len(prior) else None


def summarise(sid: str, raw: pd.Series) -> dict:
    """Latest value, changes, 10-year percentile and a weekly spark line for one series."""
    label, unit, how, freq = SERIES[sid]
    s = raw.copy()
    if how == "yoy":
        s = (s / s.shift(12) - 1) * 100       # monthly index -> % change on a year earlier
        s = s.dropna()
    elif how == "thousands":
        s = s / 1000.0
    if s.empty:
        return {}
    last_d = s.index[-1]
    last = float(s.iloc[-1])
    out = {"id": sid, "label": label, "unit": unit, "freq": freq,
           "last": round(last, 3), "date": last_d.strftime("%Y-%m-%d")}
    for tag, days in (("chg_1m", 30), ("chg_1y", 365)):
        prev = _value_on_or_before(s, last_d - pd.Timedelta(days=days))
        if prev is not None:
            out[tag] = round(last - prev, 3)
    ten = s.loc[s.index >= last_d - pd.Timedelta(days=3653)]
    if len(ten) >= 24:
        out["pct_10y"] = round(float((ten < last).mean() * 100 + (ten == last).mean() * 50), 1)
        out["min_10y"], out["max_10y"] = round(float(ten.min()), 3), round(float(ten.max()), 3)
        out["median_10y"] = round(float(ten.median()), 3)
    three = s.loc[s.index >= last_d - pd.Timedelta(days=3 * 365)]
    wk = three.groupby(three.index.to_period("W-FRI")).tail(1) if freq == "daily" else three
    out["spark"] = [round(float(v), 3) for v in wk.values]
    out["spark_d"] = [d.strftime("%Y-%m-%d") for d in wk.index]
    return out


def sahm_indicator(unrate: pd.Series) -> float | None:
    """Sahm (2019): 3-month average unemployment minus its low over the prior 12 months.
    A reading of 0.50pp or more has marked the start of every US recession since 1970."""
    u = unrate.dropna()
    if len(u) < 15:
        return None
    avg3 = u.rolling(3).mean()
    low12 = avg3.shift(1).rolling(12).min()
    v = avg3.iloc[-1] - low12.iloc[-1]
    return round(float(v), 2) if np.isfinite(v) else None


def readings(summ: dict, sahm: float | None, sahm_basis: str | None = "revised") -> list[dict]:
    """One descriptive sentence per theme, with the rule or source behind the word it uses."""
    out = []
    c = summ.get("T10Y3M")
    if c:
        state = "inverted" if c["last"] < 0 else ("flat" if c["last"] < 0.5 else "positive")
        out.append({"k": "curve", "state": state, "title": "Yield curve",
                    "text": f"The 10-year yield is {abs(c['last']):.2f}pp {'below' if c['last'] < 0 else 'above'} the 3-month. "
                            "An inverted curve (10-year below 3-month) has preceded every US recession since the late 1960s, "
                            "with lags of roughly 6 to 24 months and at least one false alarm - it describes the cycle, it does not time it "
                            "(Estrella & Mishkin 1998; the New York Fed's recession model uses this spread)."})
    b = summ.get("BAA10Y")
    if b and "pct_10y" in b:
        p = b["pct_10y"]
        state = "tight" if p < 33 else ("wide" if p > 67 else "normal")
        out.append({"k": "credit", "state": state, "title": "Credit spreads",
                    "text": f"Lower-grade corporate bonds yield {b['last']:.2f}pp over Treasuries, higher than {p:.0f}% of the last ten years. "
                            "Spreads widen when lenders want more for default risk; wide spreads have coincided with stress for indebted companies."})
    v = summ.get("VIXCLS")
    if v:
        state = "calm" if v["last"] < 15 else ("elevated" if v["last"] > 25 else "normal")
        out.append({"k": "vol", "state": state, "title": "Volatility",
                    "text": f"The VIX is {v['last']:.1f}, against a ten-year median of {v.get('median_10y', float('nan')):.1f}. "
                            "It is the S&P 500 options market's price for the next 30 days' swings. "
                            "Momentum strategies have historically suffered their sharpest losses in rebounds after high-volatility "
                            "declines (Daniel & Moskowitz 2016), which is why this screener already cuts momentum's weight when the "
                            "universe is turbulent and raises it when calm."})
    i = summ.get("CPIAUCSL")
    if i:
        state = "above target" if i["last"] > 2.5 else ("near target" if i["last"] >= 1.5 else "below target")
        out.append({"k": "inflation", "state": state, "title": "Inflation",
                    "text": f"Consumer prices are {i['last']:.1f}% higher than a year earlier, against the Federal Reserve's 2% goal "
                            f"(measured there on a different index, PCE). Latest reading: {i['date'][:7]}."})
    if sahm is not None:
        state = "triggered" if sahm >= 0.5 else "not triggered"
        u = summ.get("UNRATE", {})
        out.append({"k": "jobs", "state": state, "title": "Labor market",
                    "text": f"Unemployment is {u.get('last', float('nan')):.1f}%. The Sahm indicator - the 3-month average against "
                            f"its 12-month low - reads {sahm:+.2f}pp; at +0.50pp or more it has marked the start of every US recession "
                            "since 1970 (Sahm 2019). "
                            + ("Computed on unemployment as first published (FRED's real-time series), as the rule is defined."
                               if sahm_basis == "real-time" else
                               "Computed here on today's revised unemployment history; the rule is defined on the figures as first published.")})
    return out


FACTOR_NOTES = [
    {"title": "Quality in downturns",
     "text": "Profitable, conservatively financed companies have tended to hold up better when markets fall - the "
             "'flight to quality' Asness, Frazzini & Pedersen (2019) document across decades and countries. "
             "Quality carries 22% of this screener's composite."},
    {"title": "Momentum and rebounds",
     "text": "Momentum's worst months cluster in sharp rebounds after a high-volatility decline (Daniel & Moskowitz 2016). "
             "The screener's volatility rule scales momentum's weight for exactly this reason; the stat strip shows this run's setting."},
    {"title": "Rates and long-duration stocks",
     "text": "Companies whose value rests on profits far in the future are more sensitive to interest rates, the way a "
             "long bond is. Each stock's drilldown shows how its price has actually moved with the 10-year yield."},
]


def build(today: date | None = None, session=None, write: bool = True) -> dict:
    """Fetch (or reuse) every series, summarise, and write ``data/market_context.json``."""
    today = today or date.today()
    raw, errors = {}, {}
    for sid in SERIES:
        try:
            raw[sid] = fetch_series(sid, session=session, today=today)
        except Exception as e:  # noqa: BLE001 - one series failing must not lose the rest
            errors[sid] = str(e)[:200]
    summ = {sid: summarise(sid, s) for sid, s in raw.items()}
    summ = {k: v for k, v in summ.items() if v}
    # Sahm's rule is defined on unemployment as first published; FRED serves the latest
    # revised history, so computing it from UNRATE reads a number nobody saw at the time.
    # FRED publishes the real-time version (SAHMREALTIME); use it, and fall back to the
    # computed figure only when it cannot be fetched (plan/context-layer.md item 7).
    sahm, sahm_basis = None, None
    try:
        rt = fetch_series("SAHMREALTIME", session=session, today=today).dropna()
        if len(rt):
            sahm, sahm_basis = round(float(rt.iloc[-1]), 2), "real-time"
    except Exception as e:  # noqa: BLE001
        errors["SAHMREALTIME"] = str(e)[:200]
    if sahm is None and "UNRATE" in raw:
        sahm, sahm_basis = sahm_indicator(raw["UNRATE"]), "revised"
    out = {
        "as_of": today.isoformat(),
        "source": "FRED, Federal Reserve Bank of St. Louis (fred.stlouisfed.org)",
        "series": summ,
        "sahm": sahm,
        "sahm_basis": sahm_basis,
        "readings": readings(summ, sahm, sahm_basis),
        "factor_notes": FACTOR_NOTES,
        "errors": errors,
    }
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def yield_changes(today: date | None = None, session=None) -> pd.Series:
    """Daily change in the 10-year yield (pp), for each stock's rate sensitivity."""
    s = fetch_series("DGS10", session=session, today=today)
    return s.diff().dropna()


def load() -> dict | None:
    try:
        return json.loads(OUT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    res = build()
    print(json.dumps({k: (v if k != "series" else list(v)) for k, v in res.items() if k != "factor_notes"}, indent=1)[:3000])
