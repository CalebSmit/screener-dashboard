"""The forward-return pass skips a download no horizon can use - and changes nothing else.

research/2026-10-26-data-run-timing.md: the 2026-10-26 02:00 run spent about three minutes in
``record_run_snapshot``, and four of the six 500-ticker price downloads in
``compute_forward_returns`` were for a horizon whose end close could not exist yet - one ending
that day, two ending on the weekend before a Monday run. Each such download was followed by the
``will retry`` branch and was not cached. ``_close_possible_since`` now skips it.

The claim is that the skip is output-identical, so the test that matters is the last one: the
same scenarios run with the skip disabled and enabled, against a fetch that behaves like the real
one (never returns a bar dated today), produce byte-identical performance history and IC files.

**The scenario is a fixed week, not the clock** - corrected 2026-10-31, after the first weekend
since this file shipped turned ``main`` red with no commit touching any code it covers. Built
from ``datetime.now()`` it could not fire the skip on a Saturday or a Sunday, so it failed on
both: with an empty performance history the only skippable download is the one whose 1-week end
is today, that needs a *market-day* snapshot exactly seven days earlier, and seven days before a
weekend is a weekend. The scenario now runs for each day of one fixed week, Monday to Sunday, and
every date in it - the snapshots, the prices, and ``_close_possible_since``'s notion of today -
comes from the frozen date rather than the clock.

It also warms the history up first, which is both what makes the skip reachable on all seven days
and the shape the 02:00 loop actually meets: by the time a horizon comes due, the *shorter*
horizons of that run date were recorded weeks ago, so the only horizon still wanted is the one
whose end is at hand. With an empty history every snapshot still wants its 1-week return, whose
end is long past, and the download is never skippable - which is why the version built on the
clock exercised the skip through exactly one of its sixteen snapshot dates.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import improvement_engine as ie

N = 40
TICKERS = [f"T{i:02d}" for i in range(N)]

# Captured before any test can monkeypatch it: the arm with the skip enabled needs the real rule,
# bound to the frozen date rather than to the clock.
_REAL_CLOSE_POSSIBLE = ie._close_possible_since

# Monday .. Sunday of one fixed week. The claim is that the skip is output-identical on any day
# the pass can run, so every day of the week is a case - including the two the data loop does not
# run on, because the suite does.
WEEK = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31",
        "2026-11-01"]
WEEKDAY_NAME = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

# How far back the warm-up pass stands. Long enough that every horizon ending more than a week
# ago is already recorded, short enough that the ones ending in the last few days are not.
WARM_DAYS = 11


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


def _price_panel():
    """One deterministic panel of weekday closes, dated absolutely rather than relative to a
    run date, so the same horizon measured at the warm-up and at the target date agrees."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2026-01-01", "2026-12-31")
    return pd.DataFrame({t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
                         for t in TICKERS}, index=idx)


_PANEL = _price_panel()


def _realistic_fetch(calls, as_of):
    """Weekday closes up to (never including) ``as_of`` - the real function drops today's bar."""
    def fetch(tickers, start, end):
        calls.append((start, end))
        lo = pd.Timestamp(start) - timedelta(days=5)
        return _PANEL[(_PANEL.index >= lo) & (_PANEL.index < as_of)]
    return fetch


def _scenario_dates(today):
    """Snapshot dates relative to ``today`` that put a horizon end on today, on each of the last
    few days, and on a weekend - plus dates whose horizons are long complete."""
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


def _pass(mp, as_of, calls, skip_enabled):
    """One forward-return + IC pass with every date taken from ``as_of``, not from the clock."""
    mp.setattr(ie, "_fetch_prices_for_returns", _realistic_fetch(calls, as_of))
    if skip_enabled:
        mp.setattr(ie, "_close_possible_since",
                   lambda end, today=None: _REAL_CLOSE_POSSIBLE(end, today=as_of))
    else:
        mp.setattr(ie, "_close_possible_since", lambda end, today=None: True)
    rows = ie.compute_forward_returns(as_of.strftime("%Y-%m-%d"))
    for h in ie.HORIZON_DAYS:
        ie.compute_live_ic(horizon=h)
    return 0 if rows is None else len(rows)


def _arm(sandbox, today, skip_enabled):
    """Warm the history up the way the live loop's is warm, then run the target day's pass.

    The warm-up is identical in both arms - the skip is disabled for it - so any difference in
    the files afterwards comes from the target pass and nothing else.
    """
    for f in sandbox.rglob("*.csv"):
        f.unlink()
    warm_calls, calls = [], []
    with pytest.MonkeyPatch.context() as mp:
        warm_rows = _pass(mp, today - timedelta(days=WARM_DAYS), warm_calls, skip_enabled=False)
    with pytest.MonkeyPatch.context() as mp:
        rows = _pass(mp, today, calls, skip_enabled)
    files = {f.name: f.read_bytes() for f in sorted(sandbox.glob("*.csv"))}
    return files, calls, rows, warm_rows


@pytest.mark.parametrize("day", WEEK, ids=[f"{d}-{WEEKDAY_NAME[_ts(d).dayofweek]}" for d in WEEK])
def test_skipping_changes_no_output_only_the_downloads(sandbox, day):
    today = _ts(day)
    _write_snapshots(_scenario_dates(today))

    old, old_calls, old_rows, warm_rows = _arm(sandbox, today, skip_enabled=False)
    new, new_calls, new_rows, _ = _arm(sandbox, today, skip_enabled=True)

    # The premise, asserted rather than assumed: the warm-up recorded something, the target pass
    # had both a download it had to make and one it could skip, and it wrote rows either way.
    assert warm_rows, "the warm-up recorded nothing - the history is not in the live loop's shape"
    assert old, "the scenario produced no output - it tests nothing"
    assert "performance_history.csv" in old
    assert old_rows, "the target pass measured no horizon - it tests nothing"
    assert new_calls, "every download was skipped - the scenario has no download to preserve"
    assert len(old_calls) - len(new_calls) >= 1, "no download was skipped on this day"

    assert old == new                                   # byte-identical files
    assert new_rows == old_rows                         # and the same horizons measured
    assert set(new_calls) < set(old_calls)              # strictly fewer downloads
    assert len(new_calls) < len(old_calls)


# (run date, the day the pass runs). Two horizons are due at once and only the *later* one's end
# is out of reach, so the download must still be made for the earlier one - which is why the skip
# reads ``min`` over the due ends and not ``max``. A run date reaches this state whenever it is
# first processed late: a backfilled snapshot (``_backfill_snapshots``), or a date whose shorter
# horizons the price source could not measure when they came due.
TWO_HORIZON_CASES = [
    ("2026-07-29", "2026-10-27"),   # the 3-month end is the run day itself (a Tuesday)
    ("2026-07-27", "2026-10-26"),   # the 3-month end is the Sunday before a Monday run
]


@pytest.mark.parametrize("run_date,day", TWO_HORIZON_CASES)
def test_the_skip_reads_the_earliest_due_end_not_the_latest(sandbox, run_date, day):
    """Added 2026-10-31. The output-identity test above could not see ``min`` become ``max``:
    after its warm-up every run date has exactly one horizon still due, so the two agree. This is
    the case that separates them, and with ``max`` the skip loses two measured horizons."""
    today = _ts(day)
    assert (today - _ts(run_date)).days >= 90, "the scenario needs all three horizons due"
    _write_snapshots([run_date])

    off_calls, on_calls = [], []
    with pytest.MonkeyPatch.context() as mp:
        off_rows = _pass(mp, today, off_calls, skip_enabled=False)
    off = {f.name: f.read_bytes() for f in sorted(sandbox.glob("*.csv"))}
    history = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
    for f in sandbox.rglob("*.csv"):
        f.unlink()
    with pytest.MonkeyPatch.context() as mp:
        on_rows = _pass(mp, today, on_calls, skip_enabled=True)
    on = {f.name: f.read_bytes() for f in sorted(sandbox.glob("*.csv"))}

    # The premise: with the skip disabled this date measures its 1-week and 1-month returns and
    # not its 3-month one, whose end has not closed. So there is something for the skip to lose.
    assert off_rows == N
    assert history["fwd_return_1w"].notna().all()
    assert history["fwd_return_1m"].notna().all()
    assert history["fwd_return_3m"].isna().all()

    assert on_calls == off_calls, "the download was skipped although the 1-week end is long past"
    assert on == off
    assert on_rows == off_rows
