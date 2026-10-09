"""The context signals' record (``context_eval.py``; plan/context-layer.md item 5).

Synthetic logs with a planted relationship: the harness must find it, count overlapping days
as one observation, and never let a scoring module import it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import context_eval as ce  # noqa: E402


def _write_logs(tmp_path, dates, n=60, seed=0):
    """Logs where a stock's distance above its 200-day average predicts its next-month return."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n)]
    dist = rng.normal(0, 0.1, n)                     # fixed per stock
    px = np.full(n, 100.0)
    for d in dates:
        log = pd.DataFrame({
            "Ticker": tickers, "date": d,
            "_ctx_last_close": px, "_ctx_sma200": px / (1 + dist), "_ctx_sma50": px,
            "_ctx_ret_1m": rng.normal(0, 0.05, n), "_ctx_ret_5d": rng.normal(0, 0.02, n),
            "_ctx_ret_3m": rng.normal(0, 0.08, n), "_ctx_high_52w": px * 1.2, "_ctx_low_52w": px * 0.8,
            "_ctx_vol_ratio": rng.uniform(0.5, 2, n), "_ctx_opt_status": "stale-quotes",
            "_ctx_ins_buy_people": rng.integers(0, 3, n), "_ctx_ins_buy_value": rng.uniform(0, 1e6, n),
            "_ctx_rate_beta": rng.normal(0, 1, n),
        })
        log.to_parquet(tmp_path / f"{d}.parquet")
        px = px * (1 + 0.5 * dist + rng.normal(0, 0.01, n))   # next period: dist pays off
    return tmp_path


def _monthly(n):
    return [d.date().isoformat() for d in pd.date_range("2026-01-02", periods=n, freq="35D")]


def test_a_planted_signal_is_found(tmp_path):
    ev = ce.evaluate(_write_logs(tmp_path, _monthly(5)), write=False)
    s = ev["signals"]["distance_from_200d"]
    assert s["daily_observations"] == 4                 # the last log has no forward window
    assert s["effective_observations"] == 4
    assert s["mean_ic_independent"] > 0.8 and s["t_stat_independent"] > 3
    assert abs(ev["signals"]["return_5d"]["mean_ic_all_days"]) < 0.5


def test_overlapping_days_count_once(tmp_path):
    days = [d.date().isoformat() for d in pd.date_range("2026-01-02", periods=45, freq="D")]
    ev = ce.evaluate(_write_logs(tmp_path, days), write=False)
    s = ev["signals"]["distance_from_200d"]
    assert s["daily_observations"] >= 6               # days 1..15 each have a log 30-40 days on
    assert s["effective_observations"] == 1           # but they are one month of evidence
    assert "t_stat_independent" not in s              # too few for a t-statistic


def test_stale_option_readings_are_not_evaluated(tmp_path):
    ev = ce.evaluate(_write_logs(tmp_path, _monthly(4)), write=False)
    assert ev["signals"]["options_expected_move"]["daily_observations"] == 0


def test_no_record_yet_is_said_plainly(tmp_path):
    ev = ce.evaluate(_write_logs(tmp_path, ["2026-10-08", "2026-10-09"]), write=False)
    line = ce.summary_line(ev)
    assert "no signal has a record yet" in line and "2026-11-07" in line
    assert not any(v["meets_gate"] for v in ev["signals"].values())


def test_nothing_that_scores_imports_it():
    for name in ("factor_engine.py", "calc_trace.py", "portfolio_constructor.py", "improvement_engine.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+context_eval\b", src, re.M), name
