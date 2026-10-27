"""The forward-return pass skips a download no horizon can use - and changes nothing else.

research/2026-10-26-data-run-timing.md: the 2026-10-26 02:00 run spent about three minutes in
``record_run_snapshot``, and four of the six 500-ticker price downloads in
``compute_forward_returns`` were for a horizon whose end close could not exist yet - one ending
that day, two ending on the weekend before a Monday run. Each such download was followed by the
``will retry`` branch and was not cached. ``_close_possible_since`` now skips it.

The claim is that the skip is output-identical, so the test that matters is the last one: the
same scenarios run with the skip disabled and enabled, against a fetch that behaves like the real
one (never returns a bar dated today), produce byte-identical performance history and IC files.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import improvement_engine as ie

N = 40
TICKERS = [f"T{i:02d}" for i in range(N)]


def _ts(s):
    return pd.Timestamp(s)


class TestClosePossibleSince:
    # 2026-10-26 is a Monday.
    def test_an_end_dated_today_cannot_have_closed(self):
        assert not ie._close_possible_since(_ts("2026-10-26"), today=_ts("2026-10-26"))

    def test_a_weekend_end_before_a_monday_run_cannot_have_closed(self):
        assert not ie._close_possible_since(_ts("2026-10-24"), today=_ts("2026-10-26"))
        assert not ie._close_possible_since(_ts("2026-10-25"), today=_ts("2026-10-26"))

    def test_a_weekday_before_today_can(self):
        assert ie._close_possible_since(_ts("2026-10-23"), today=_ts("2026-10-26"))
        assert ie._close_possible_since(_ts("2026-10-20"), today=_ts("2026-10-21"))

    def test_a_holiday_is_treated_as_a_possible_close(self):
        # 2026-11-26 is Thanksgiving (a Thursday). Not excluded: the rule errs towards downloading.
        assert ie._close_possible_since(_ts("2026-11-26"), today=_ts("2026-11-27"))

    def test_an_end_after_today(self):
        assert not ie._close_possible_since(_ts("2026-10-30"), today=_ts("2026-10-26"))

    def test_defaults_to_the_local_date(self):
        today = pd.Timestamp(datetime.now().date())
        assert not ie._close_possible_since(today)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    d = tmp_path / "improvement"
    (d / "snapshots").mkdir(parents=True)
    (d / "price_cache").mkdir()
    monkeypatch.setattr(ie, "IMPROVEMENT_DIR", d)
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", d / "snapshots")
    monkeypatch.setattr(ie, "PRICE_CACHE_DIR", d / "price_cache")
    monkeypatch.setattr(ie, "PERFORMANCE_HISTORY_PATH", d / "performance_history.csv")
    monkeypatch.setattr(ie, "LIVE_IC_HISTORY_PATH", d / "live_ic_history.csv")
    monkeypatch.setattr(ie, "METRIC_IC_HISTORY_PATH", d / "metric_ic_history.csv")
    monkeypatch.setattr(ie, "DISPERSION_HISTORY_PATH", d / "dispersion_history.csv")
    monkeypatch.setattr(ie, "CHANGE_LOG_PATH", d / "change_log.csv")
    return d


def _write_snapshots(dates):
    rng = np.random.default_rng(3)
    for k, day in enumerate(dates):
        df = pd.DataFrame({"Ticker": TICKERS, "Sector": "Technology",
                           "Composite": rng.uniform(20, 100, N), "Rank": np.arange(1, N + 1),
                           "in_portfolio": False, "run_date": day, "run_id": f"r{k}"})
        for col in ie.CATEGORY_SCORES:
            df[col] = rng.uniform(10, 90, N)
        df.to_parquet(ie.SNAPSHOTS_DIR / f"{day}_r{k}.parquet", index=False)


def _realistic_fetch(calls):
    """Weekday closes up to (never including) today - the real function drops today's bar."""
    rng = np.random.default_rng(11)
    today = pd.Timestamp(datetime.now().date())
    idx = pd.bdate_range(today - timedelta(days=200), today - timedelta(days=1))
    prices = pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
                           for t in TICKERS}, index=idx)

    def fetch(tickers, start, end):
        calls.append((start, end))
        lo = pd.Timestamp(start) - timedelta(days=5)
        return prices[prices.index >= lo]
    return fetch


def _scenario_dates():
    """Snapshot dates relative to today that put a horizon end on today, on each of the last
    few days, and on a weekend - plus dates whose horizons are long complete."""
    today = pd.Timestamp(datetime.now().date())
    days = set()
    for span in (7, 30, 90):
        for back in range(0, 5):
            d = today - timedelta(days=span + back)
            if d.dayofweek < 5:
                days.add(d.strftime("%Y-%m-%d"))
    for back in (10, 45, 120):
        d = today - timedelta(days=back)
        while d.dayofweek >= 5:
            d -= timedelta(days=1)
        days.add(d.strftime("%Y-%m-%d"))
    return sorted(days)


def _run(sandbox, monkeypatch, skip_enabled):
    for f in sandbox.rglob("*.csv"):
        f.unlink()
    calls = []
    monkeypatch.setattr(ie, "_fetch_prices_for_returns", _realistic_fetch(calls))
    if not skip_enabled:
        monkeypatch.setattr(ie, "_close_possible_since", lambda end, today=None: True)
    today = datetime.now().strftime("%Y-%m-%d")
    ie.compute_forward_returns(today)
    for h in ie.HORIZON_DAYS:
        ie.compute_live_ic(horizon=h)
    out = {f.name: f.read_bytes() for f in sorted(sandbox.glob("*.csv"))}
    monkeypatch.undo()
    return out, calls


def test_skipping_changes_no_output_only_the_downloads(sandbox, monkeypatch):
    _write_snapshots(_scenario_dates())
    # monkeypatch.undo() inside _run also undoes the sandbox paths - re-apply them each time.
    paths = {k: getattr(ie, k) for k in ("IMPROVEMENT_DIR", "SNAPSHOTS_DIR", "PRICE_CACHE_DIR",
                                         "PERFORMANCE_HISTORY_PATH", "LIVE_IC_HISTORY_PATH",
                                         "METRIC_IC_HISTORY_PATH", "DISPERSION_HISTORY_PATH",
                                         "CHANGE_LOG_PATH")}

    def apply():
        for k, v in paths.items():
            monkeypatch.setattr(ie, k, v)

    apply()
    old, old_calls = _run(sandbox, monkeypatch, skip_enabled=False)
    apply()
    new, new_calls = _run(sandbox, monkeypatch, skip_enabled=True)
    apply()

    assert old, "the scenario produced no output - it tests nothing"
    assert "performance_history.csv" in old
    assert old == new                                   # byte-identical files
    assert set(new_calls) < set(old_calls)              # strictly fewer downloads
    assert len(new_calls) < len(old_calls)
