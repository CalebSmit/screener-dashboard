"""Does any context signal predict the next month? The record each one is building.

WHY THIS EXISTS - ``plan/context-layer.md`` queue item 5. The context layer (trend, recent move,
options, insider trades, rate sensitivity) is shown beside the score, never in it, and the rule
for ever moving a signal into the score is a research note **and** a measured record at the
``1m`` horizon meeting the improvement engine's own observation gate (CLAUDE.md settled row
"ctx"). This module keeps that record.

**How.** Every run writes ``data/context_log/YYYY-MM-DD.parquet``: each stock's signals and its
last close on that date. A later log carries the later close, so the forward return needs no
extra download: for a log dated d, the forward return is the close in the first log dated 30-40
days later over the close at d (price return - dividends excluded, which shifts every stock's
return by roughly its yield and barely moves a rank). For each signal and each start date, the
**information coefficient** is the Spearman rank correlation between the signal and that forward
return across stocks (at least ``MIN_STOCKS``).

**Counting what matters.** Daily logs give daily ICs whose one-month windows overlap almost
entirely. The count reported against the gate is the improvement engine's own effective
(non-overlapping) observation count, ``improvement_engine._effective_observations`` - about
one a month. A t-statistic is reported only from ``MIN_EFFECTIVE_FOR_T`` effective observations,
computed on those non-overlapping dates only.

**Reporting only.** Nothing here can reach a score; ``tests/test_context_layer.py`` already fails
if a scoring module imports a context module, and this one is added to that list.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "data" / "context_eval.json"
HORIZON_MIN_DAYS, HORIZON_MAX_DAYS = 30, 40
MIN_STOCKS = 30
MIN_EFFECTIVE_FOR_T = 3

# Signal name -> (how it is read from a log row, what a positive IC would mean in words).
SIGNALS = {
    "distance_from_200d": ("close / 200-day average - 1", "stocks further above their 200-day average did better"),
    "distance_from_50d": ("close / 50-day average - 1", "stocks further above their 50-day average did better"),
    "ma_distance_21_200": ("21-day average / 200-day average - 1 (Avramov, Kaplanski & Subrahmanyam 2021)",
                           "stocks whose short average sat further above the long one did better"),
    "return_1m": ("last month's price return", "last month's winners kept winning (negative = reversal)"),
    "return_5d": ("last week's price return", "last week's winners kept winning (negative = reversal)"),
    "return_1m_vs_sector": ("last month's return minus its sector's median (Da, Liu & Schaumburg 2014)",
                            "sector-relative winners kept winning (negative = industry-relative reversal)"),
    "return_3m": ("last three months' price return", "three-month winners kept winning"),
    "position_in_52w_range": ("(close - 52-week low) / (52-week high - low)", "stocks nearer their 52-week high did better"),
    "volume_ratio": ("20-day / 3-month average volume", "heavier recent volume preceded better returns"),
    "options_expected_move": ("at-the-money straddle / price", "a bigger implied move preceded better returns"),
    "options_put_skew": ("90% put IV - at-the-money IV", "steeper put skew preceded better returns (literature: worse)"),
    "options_put_call_oi": ("put / call open interest", "more put open interest preceded better returns"),
    "insider_buyers": ("officers and directors buying in 90 days", "more insider buyers preceded better returns"),
    "insider_buy_value": ("dollar value of insider purchases, 90 days", "larger insider purchases preceded better returns"),
    "rate_sensitivity": ("return per +1pp in the 10-year yield", "rate-sensitive stocks did better"),
    "earnings_yield_vs_own_history": ("today's earnings yield as a percentile of its own past 60 month-ends",
                                      "stocks whose earnings yield sat higher in their own range did better"),
    "fcf_yield_vs_own_history": ("today's free-cash-flow yield as a percentile of its own past 60 month-ends",
                                 "stocks whose FCF yield sat higher in their own range did better"),
}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def signals_frame(log: pd.DataFrame) -> pd.DataFrame:
    """One row per stock, one column per signal, from a context log."""
    px = _num(log.get("_ctx_last_close"))
    out = pd.DataFrame({"Ticker": log["Ticker"].values})
    out["close"] = px.values
    out["distance_from_200d"] = (px / _num(log.get("_ctx_sma200")) - 1).values
    out["distance_from_50d"] = (px / _num(log.get("_ctx_sma50")) - 1).values
    s21 = _num(log["_ctx_sma21"]) if "_ctx_sma21" in log.columns else pd.Series(np.nan, index=log.index)
    out["ma_distance_21_200"] = (s21 / _num(log.get("_ctx_sma200")) - 1).values
    out["return_1m"] = _num(log.get("_ctx_ret_1m")).values
    out["return_5d"] = _num(log.get("_ctx_ret_5d")).values
    r1 = _num(log.get("_ctx_ret_1m"))
    if "Sector" in log.columns:
        out["return_1m_vs_sector"] = (r1 - r1.groupby(log["Sector"]).transform("median")).values
    else:
        out["return_1m_vs_sector"] = np.nan
    out["return_3m"] = _num(log.get("_ctx_ret_3m")).values
    hi, lo = _num(log.get("_ctx_high_52w")), _num(log.get("_ctx_low_52w"))
    rng = (hi - lo).where((hi - lo) > 0)
    out["position_in_52w_range"] = ((px - lo) / rng).values
    out["volume_ratio"] = _num(log.get("_ctx_vol_ratio")).values
    for col, key in (("options_expected_move", "_ctx_opt_move"), ("options_put_skew", "_ctx_opt_skew"),
                     ("options_put_call_oi", "_ctx_opt_pc_oi")):
        v = _num(log[key]) if key in log.columns else pd.Series(np.nan, index=log.index)
        # Only readings the page would show: a usable chain.
        if "_ctx_opt_status" in log.columns:
            v = v.where(log["_ctx_opt_status"].isin(["ok", "partial"]))
        out[col] = v.values
    out["insider_buyers"] = (_num(log["_ctx_ins_buy_people"]) if "_ctx_ins_buy_people" in log.columns
                             else pd.Series(np.nan, index=log.index)).values
    out["insider_buy_value"] = (_num(log["_ctx_ins_buy_value"]) if "_ctx_ins_buy_value" in log.columns
                                else pd.Series(np.nan, index=log.index)).values
    out["rate_sensitivity"] = (_num(log["_ctx_rate_beta"]) if "_ctx_rate_beta" in log.columns
                               else pd.Series(np.nan, index=log.index)).values
    for col, key in (("earnings_yield_vs_own_history", "_ctx_vh_ey_pct"), ("fcf_yield_vs_own_history", "_ctx_vh_fy_pct")):
        out[col] = (_num(log[key]) if key in log.columns else pd.Series(np.nan, index=log.index)).values
    return out


def load_logs(log_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    from context_signals import context_log_dir
    d = Path(log_dir) if log_dir else context_log_dir()
    out = {}
    for f in sorted(d.glob("*.parquet")):
        try:
            out[f.stem] = signals_frame(pd.read_parquet(f))
        except Exception:  # noqa: BLE001 - one unreadable day must not hide the rest
            continue
    return out


def _forward_date(dates: list[str], d: str) -> str | None:
    t0 = pd.Timestamp(d)
    for x in dates:
        gap = (pd.Timestamp(x) - t0).days
        if HORIZON_MIN_DAYS <= gap <= HORIZON_MAX_DAYS:
            return x
    return None


def daily_ics(logs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (start date, signal): the IC and how many stocks it used."""
    dates = sorted(logs)
    rows = []
    for d in dates:
        f = _forward_date(dates, d)
        if f is None:
            continue
        a = logs[d].set_index("Ticker")
        b = logs[f].set_index("Ticker")["close"]
        fwd = (b / a["close"] - 1).replace([np.inf, -np.inf], np.nan)
        for s in SIGNALS:
            pair = pd.DataFrame({"x": a[s], "y": fwd}).dropna()
            if len(pair) < MIN_STOCKS or pair["x"].nunique() < 3:
                continue
            ic = pair["x"].rank().corr(pair["y"].rank())
            if ic == ic:
                rows.append({"date": d, "forward_date": f, "signal": s, "ic": float(ic), "n": int(len(pair))})
    return pd.DataFrame(rows, columns=["date", "forward_date", "signal", "ic", "n"])


def _non_overlapping(dates: list[str], span: int = 30) -> list[str]:
    keep, end = [], None
    for d in sorted(set(dates)):
        t = pd.Timestamp(d)
        if end is None or t >= end:
            keep.append(d)
            end = t + pd.Timedelta(days=span)
    return keep


def evaluate(log_dir: Path | None = None, today: date | None = None, write: bool = True) -> dict:
    """The record for every signal; written to ``data/context_eval.json``."""
    from improvement_engine import _effective_observations
    try:
        import yaml
        gate = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["improvement"]["min_observations_for_proposal"]
    except Exception:  # noqa: BLE001
        gate = 8
    logs = load_logs(log_dir)
    ics = daily_ics(logs)
    out = {"as_of": (today or date.today()).isoformat(), "horizon": "1m",
           "log_dates": len(logs), "first_log": min(logs) if logs else None,
           "gate_effective_observations": gate,
           "method": ("Spearman IC between each context signal and the price return to the first log "
                      f"{HORIZON_MIN_DAYS}-{HORIZON_MAX_DAYS} days later; effective observations are "
                      "non-overlapping one-month windows. Reporting only - nothing here moves a score."),
           "signals": {}}
    for s, (definition, meaning) in SIGNALS.items():
        g = ics[ics["signal"] == s] if len(ics) else ics
        dates = list(g["date"]) if len(g) else []
        eff = _effective_observations(dates, "1m") if dates else 0
        rec = {"definition": definition, "positive_ic_means": meaning,
               "daily_observations": len(dates), "effective_observations": eff,
               "mean_ic_all_days": round(float(g["ic"].mean()), 4) if len(g) else None}
        if eff >= MIN_EFFECTIVE_FOR_T:
            ind = g[g["date"].isin(_non_overlapping(dates))]["ic"]
            m, sd = float(ind.mean()), float(ind.std(ddof=1))
            rec["mean_ic_independent"] = round(m, 4)
            rec["t_stat_independent"] = round(m / (sd / math.sqrt(len(ind))), 2) if sd > 0 else None
        rec["meets_gate"] = eff >= gate
        out["signals"][s] = rec
    if logs:
        first = pd.Timestamp(min(logs))
        out["first_forward_window_closes"] = (first + pd.Timedelta(days=HORIZON_MIN_DAYS)).date().isoformat()
        out["earliest_gate_date_if_logged_daily"] = (first + pd.Timedelta(days=30 * gate)).date().isoformat()
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def summary_line(ev: dict) -> str:
    """One sentence for the morning brief."""
    sig = ev.get("signals") or {}
    best = max((v.get("effective_observations", 0) for v in sig.values()), default=0)
    if best == 0:
        return (f"Context signals: {ev.get('log_dates', 0)} daily logs since {ev.get('first_log')}; the first "
                f"one-month return window closes {ev.get('first_forward_window_closes')}, so no signal has a "
                f"record yet (gate: {ev.get('gate_effective_observations')} effective observations).")
    lead = sorted(((k, v) for k, v in sig.items() if v.get("mean_ic_independent") is not None),
                  key=lambda kv: -abs(kv[1]["mean_ic_independent"]))[:3]
    parts = [f"{k} IC {v['mean_ic_independent']:+.3f} (t {v.get('t_stat_independent')})" for k, v in lead]
    return (f"Context signals: up to {best} effective observations of {ev.get('gate_effective_observations')} "
            "needed. " + ("; ".join(parts) + "." if parts else "Not enough for a t-statistic yet."))


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=1))
