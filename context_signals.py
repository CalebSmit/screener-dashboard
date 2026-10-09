"""Context signals: what the market, the options market and the price trend say about a stock.

WHY THIS EXISTS - 2026-10-08 (owner, ``plan/context-layer.md``).

The owner asked for technicals, macro and options data "to look at when deciding what to buy",
with the goal still long-term. The screener's credibility rests on a score built from published
research, and short-horizon timing signals do not have that backing for a long-term investor - so
**nothing in this module enters a score.** These are context: shown beside the score, labelled as
context, and recorded every run so that each one builds an out-of-sample track record. A signal
that earns its way in later does so through research and ``METHODOLOGY_CHANGELOG.md``, like any
other metric.

Pure functions only - no network here. ``factor_engine`` calls ``price_context`` with the price
history it already fetched (no extra API call) and ``options_context`` with the yfinance Ticker it
already holds; ``market_context``, ``insider_activity`` and ``track_record`` own their own fetches.
Every field this module returns starts ``_ctx_`` and is display-only (a test asserts none reaches
``raw`` or ``pct``).
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

# Trading-day windows. 252 ~ one year, 21 ~ one month.
SMA_SHORT, SMA_LONG = 50, 200
SLOPE_LAG = 20          # sma200 now vs 20 trading days ago -> rising or falling
VOL_SHORT, VOL_LONG = 20, 63
WEEKS = 53

# Options quality bars. A 2 AM fetch reads the previous session's quotes; a chain that fails
# these is reported as unusable rather than turned into a confident-looking number.
OPT_MIN_DAYS, OPT_TARGET_DAYS, OPT_MAX_DAYS = 7, 30, 75
OPT_EARNINGS_WINDOW_DAYS = 60
IV_MIN, IV_MAX = 0.03, 3.0
MAX_REL_SPREAD = 0.60   # (ask - bid) / mid above this and the quote is not used


def _sig(x: float, digits: int = 5):
    """Round to significant digits, for compact payload series."""
    if x is None or not np.isfinite(x) or x == 0:
        return None if x is None or not np.isfinite(x) else 0.0
    return float(f"{x:.{digits}g}")


# --------------------------------------------------------------------------- price
def price_context(closes: pd.Series, volume: pd.Series | None = None) -> dict:
    """Trend, range, recent move and volume context from a daily close series.

    ``closes`` must be split- and dividend-adjusted and date-indexed (the fetch's
    ``history(auto_adjust=True)``). Returns ``_ctx_*`` fields; a field is absent when the
    history is too short for it rather than estimated from less.
    """
    out: dict = {}
    c = pd.to_numeric(closes, errors="coerce").dropna()
    c = c[c > 0].sort_index()
    if getattr(c.index, "tz", None) is not None:
        c.index = c.index.tz_localize(None)   # exchange-local dates; drops a pandas warning
    if len(c) < 30:
        return out
    last = float(c.iloc[-1])
    out["_ctx_last_close"] = last
    out["_ctx_last_date"] = c.index[-1].strftime("%Y-%m-%d")

    if len(c) >= SMA_SHORT:
        out["_ctx_sma50"] = float(c.tail(SMA_SHORT).mean())
        # 21-day average: with the 200-day it is the "moving average distance" Avramov,
        # Kaplanski & Subrahmanyam (2021) find predicts returns across stocks. Recorded so
        # context_eval.py can keep its record here; not scored.
        out["_ctx_sma21"] = float(c.tail(21).mean())
    if len(c) >= SMA_LONG:
        sma200 = c.rolling(SMA_LONG).mean()
        out["_ctx_sma200"] = float(sma200.iloc[-1])
        if len(c) >= SMA_LONG + SLOPE_LAG:
            out["_ctx_sma200_prev"] = float(sma200.iloc[-1 - SLOPE_LAG])

    year = c.tail(252)
    out["_ctx_high_52w"] = float(year.max())
    out["_ctx_low_52w"] = float(year.min())
    out["_ctx_high_52w_date"] = year.idxmax().strftime("%Y-%m-%d")

    def _ret(days: int):
        target = c.index[-1] - pd.Timedelta(days=days)
        prior = c.loc[c.index <= target]
        return float(last / prior.iloc[-1] - 1) if len(prior) else None

    for label, days in (("5d", 7), ("1m", 30), ("3m", 91)):
        r = _ret(days)
        if r is not None:
            out[f"_ctx_ret_{label}"] = r

    if volume is not None:
        v = pd.to_numeric(volume, errors="coerce")
        if getattr(v.index, "tz", None) is not None:
            v.index = v.index.tz_localize(None)
        v = v.reindex(c.index).dropna()
        v = v[v > 0]
        if len(v) >= VOL_LONG:
            out["_ctx_vol_ratio"] = float(v.tail(VOL_SHORT).mean() / v.tail(VOL_LONG).mean())

    # Weekly series for the chart: the close and both averages, sampled at each week's last
    # trading day. The averages are the daily ones, sampled - not recomputed from weekly data.
    sma50_d = c.rolling(SMA_SHORT).mean()
    sma200_d = c.rolling(SMA_LONG).mean()
    wk = c.groupby(c.index.to_period("W-FRI")).tail(1).tail(WEEKS)
    # Stored as a JSON string so the raw-fetch parquet keeps a flat schema.
    out["_ctx_weekly"] = json.dumps({
        "d": [d.strftime("%Y-%m-%d") for d in wk.index],
        "c": [_sig(float(x), 4) for x in wk.values],
        "s50": [_sig(float(sma50_d.get(d)), 4) if pd.notna(sma50_d.get(d)) else None for d in wk.index],
        "s200": [_sig(float(sma200_d.get(d)), 4) if pd.notna(sma200_d.get(d)) else None for d in wk.index],
    }, separators=(",", ":"))
    return out


# --------------------------------------------------------------------------- options
def _mid(row) -> tuple[float | None, str]:
    bid, ask = row.get("bid"), row.get("ask")
    if bid is not None and ask is not None and np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
        if (ask - bid) / mid <= MAX_REL_SPREAD:
            return float(mid), "mid"
    return None, "no-quote"


def choose_expiry(expiries, today: date, earnings_date: date | None):
    """The expiry that answers the question a buyer asks.

    If a report falls within ``OPT_EARNINGS_WINDOW_DAYS``, the first expiry on or after it (so
    the straddle prices the report); otherwise the expiry closest to ``OPT_TARGET_DAYS`` out.
    Returns ``(expiry_str, spans_earnings)`` or ``(None, False)``.
    """
    parsed = []
    for e in expiries or ():
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        days = (d - today).days
        if OPT_MIN_DAYS <= days <= OPT_MAX_DAYS:
            parsed.append((d, e))
    if not parsed:
        return None, False
    if earnings_date is not None and 0 <= (earnings_date - today).days <= OPT_EARNINGS_WINDOW_DAYS:
        after = [p for p in parsed if p[0] >= earnings_date]
        if after:
            return after[0][1], True
    best = min(parsed, key=lambda p: abs((p[0] - today).days - OPT_TARGET_DAYS))
    return best[1], False


def options_context(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, expiry: str,
                    today: date, spans_earnings: bool) -> dict:
    """What one expiry's chain says: the move it prices, implied vol, put skew, positioning.

    - **Expected move** = (ATM call mid + ATM put mid) / spot. An at-the-money straddle costs
      roughly the expected absolute move to expiry (for a normal return, E|move| = sigma*sqrt(T)
      *sqrt(2/pi), and the straddle is ~0.8*sigma*sqrt(T)*S - the same quantity), so this is the
      average move the options market is paying for, not a forecast of direction.
    - **ATM implied volatility** = mean of the ATM call and put IVs (annualised).
    - **Put skew** = IV of the put nearest 90% of spot minus ATM IV (Xing, Zhang & Zhao 2010
      document that steeper skew predicts lower subsequent returns - recorded here, not scored).
    - **Put/call open interest** = total put OI / total call OI on that expiry.
    """
    out: dict = {"_ctx_opt_expiry": expiry, "_ctx_opt_spans_earnings": bool(spans_earnings)}
    try:
        exp_d = datetime.strptime(expiry, "%Y-%m-%d").date()
        out["_ctx_opt_days"] = (exp_d - today).days
    except (TypeError, ValueError):
        return {}
    if calls is None or puts is None or calls.empty or puts.empty or not spot or spot <= 0:
        out["_ctx_opt_status"] = "no-chain"
        return out
    common = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
    if not common:
        out["_ctx_opt_status"] = "no-chain"
        return out
    atm = min(common, key=lambda k: abs(k - spot))
    if abs(atm - spot) / spot > 0.05:
        out["_ctx_opt_status"] = "no-atm"
        return out
    c_row = calls.loc[calls["strike"] == atm].iloc[0].to_dict()
    p_row = puts.loc[puts["strike"] == atm].iloc[0].to_dict()
    c_mid, _ = _mid(c_row)
    p_mid, _ = _mid(p_row)
    out["_ctx_opt_atm_strike"] = float(atm)
    if c_mid is not None and p_mid is not None:
        out["_ctx_opt_move"] = float((c_mid + p_mid) / spot)
        out["_ctx_opt_straddle"] = float(c_mid + p_mid)
    ivs = [r.get("impliedVolatility") for r in (c_row, p_row)]
    ivs = [float(v) for v in ivs if v is not None and np.isfinite(v) and IV_MIN <= v <= IV_MAX]
    if len(ivs) == 2:
        out["_ctx_opt_iv"] = float(sum(ivs) / 2)
    # put skew at ~90% moneyness
    p90 = puts.iloc[(puts["strike"] - 0.9 * spot).abs().argsort()[:1]]
    if len(p90) and "_ctx_opt_iv" in out:
        k = float(p90["strike"].iloc[0])
        iv90 = p90["impliedVolatility"].iloc[0]
        if abs(k / spot - 0.9) <= 0.04 and iv90 is not None and np.isfinite(iv90) and IV_MIN <= iv90 <= IV_MAX:
            out["_ctx_opt_skew"] = float(iv90 - out["_ctx_opt_iv"])
            out["_ctx_opt_skew_strike"] = k
    coi, poi = calls.get("openInterest"), puts.get("openInterest")
    if coi is not None and poi is not None:
        c_sum, p_sum = float(np.nansum(coi.values)), float(np.nansum(poi.values))
        if c_sum >= 100:
            out["_ctx_opt_pc_oi"] = p_sum / c_sum
    usable = "_ctx_opt_move" in out and "_ctx_opt_iv" in out
    out["_ctx_opt_status"] = "ok" if usable else ("partial" if ("_ctx_opt_move" in out or "_ctx_opt_iv" in out) else "stale-quotes")
    return out


def earnings_date_from_info(info: dict) -> date | None:
    # Only the *Start* field: plain `earningsTimestamp` is the last report for some tickers and
    # the next for others (see generate_dashboard._earnings_block), so it cannot pick an expiry.
    ts = (info or {}).get("earningsTimestampStart")
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).date() if ts else None
    except (TypeError, ValueError, OSError, OverflowError):
        return None


# --------------------------------------------------------------------------- rate sensitivity
def rate_sensitivity(daily_returns: dict, yield_changes: pd.Series, min_obs: int = 120) -> dict:
    """Slope of a stock's daily log return on the daily change in the 10-year yield.

    ``daily_returns`` is the fetch's ``{date: log return}``; ``yield_changes`` is the daily change
    in DGS10 in percentage points, date-indexed. Returns ``{beta, r2, n}`` where beta is the
    return per +1.00pp move in the 10-year yield (so -0.05 = -5% per point), or ``{}`` when there
    are too few common days. A description of the past ~13 months, not a forecast; r2 says how
    much of the stock's daily movement it explains (usually little).
    """
    if not daily_returns or yield_changes is None or yield_changes.empty:
        return {}
    r = pd.Series(daily_returns, dtype=float)
    r.index = pd.to_datetime(r.index)
    y = yield_changes.copy()
    y.index = pd.to_datetime(y.index)
    df = pd.concat([r.rename("r"), y.rename("y")], axis=1, join="inner").dropna()
    if len(df) < min_obs or df["y"].var() == 0:
        return {}
    cov = float(np.cov(df["r"], df["y"], ddof=1)[0, 1])
    var = float(df["y"].var(ddof=1))
    beta = cov / var
    corr = float(df["r"].corr(df["y"]))
    return {"beta": beta, "r2": corr * corr, "n": int(len(df))}


# --------------------------------------------------------------------------- reading the numbers
def trend_state(ctx: dict) -> str | None:
    """A plain description of where the price sits against its averages - never advice."""
    last, s50, s200 = ctx.get("_ctx_last_close"), ctx.get("_ctx_sma50"), ctx.get("_ctx_sma200")
    if not last or not s200:
        return None
    above200 = last >= s200
    if s50:
        if above200 and s50 >= s200:
            return "uptrend"        # above the 200-day, 50-day above 200-day
        if not above200 and s50 < s200:
            return "downtrend"
    return "mixed"


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# --------------------------------------------------------------------------- the record
CONTEXT_LOG_DIR = None  # set lazily so tests can point it elsewhere


def context_log_dir():
    from pathlib import Path
    return CONTEXT_LOG_DIR or (Path(__file__).resolve().parent / "data" / "context_log")


def write_context_log(run_dir, run_day: str) -> int:
    """Write one file per run date with every scalar context signal per stock.

    This is how a context signal earns (or fails to earn) a place in the score later: joined to
    the forward returns the improvement engine already computes, each column becomes an
    out-of-sample test. One file per date - a second run the same day replaces it, the rule
    every other observation file follows since 2026-10-08. Returns the number of rows written.
    """
    from pathlib import Path
    raw_path = Path(run_dir) / "00_raw_fetch.parquet"
    if not raw_path.exists():
        return 0                      # a run that did not fetch has nothing new to record
    raw = pd.read_parquet(raw_path)
    # Sector rides along so the evaluation can form sector-relative signals (2026-10-09).
    keep = ["Ticker"] + (["Sector"] if "Sector" in raw.columns else []) + [c for c in raw.columns if c.startswith("_ctx_")
                         and c not in ("_ctx_weekly", "_ctx_insider")]
    log = raw[keep].copy()
    if "_ctx_insider" in raw.columns:
        from insider_activity import summarise_rows
        today = datetime.strptime(run_day, "%Y-%m-%d").date()
        summ = []
        for v in raw["_ctx_insider"]:
            try:
                s = summarise_rows(json.loads(v) if isinstance(v, str) else [], today)
            except (TypeError, ValueError):
                s = {}
            summ.append({"_ctx_ins_buy_n": s.get("buy_n"), "_ctx_ins_buy_people": s.get("buy_people"),
                         "_ctx_ins_buy_value": s.get("buy_value"), "_ctx_ins_sell_value": s.get("sell_value"),
                         "_ctx_ins_sell_planned_value": s.get("sell_planned_value"),
                         "_ctx_ins_cluster": s.get("cluster")})
        log = pd.concat([log.reset_index(drop=True), pd.DataFrame(summ)], axis=1)
    out = context_log_dir()
    out.mkdir(parents=True, exist_ok=True)
    log.insert(1, "date", run_day)
    log.to_parquet(out / f"{run_day}.parquet", index=False)
    return len(log)
