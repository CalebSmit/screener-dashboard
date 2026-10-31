#!/usr/bin/env python3
"""Regression tests for the five evidence-integrity defects in the improvement
engine (CLAUDE.md priority 0).

Every test here fails against the code as it stood on 2026-08-24. The defects
all had the same shape: the engine's confidence is gated on an observation
*count*, and four separate bugs corrupted that count while one stopped it
growing at all.

  1. A snapshot was processed once, at 7 days old, when only the 1-week return
     existed. Its date then joined `existing_dates` and was never revisited, so
     `fwd_return_1m` stayed NaN forever - and `optimization_horizon` is '1m'.
  2. Every snapshot file for a date was processed, so a day with four runs
     appended the same ticker-date rows four times. The live file reached ~60%
     duplicates and reported 6,539 "tickers" for one day of an S&P 500 screener.
  3. Overlapping return windows were counted as independent observations.
  4. Weekend run dates, which have no market close, were counted as separate
     observations from the adjacent Friday.
  5. Nothing ever called `compute_live_ic()`, so live_ic_history.csv held the
     same 3 rows from February 2026 for 183 days.
  6. (2026-10-16) A horizon was measured on the 02:00 run the day it came due,
     before its end close existed, against the previous close; the truncated
     price frame was cached; and a later horizon's pass rewrote the IC rows.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import improvement_engine as ie


@pytest.fixture(autouse=True)
def temp_improvement_dirs(tmp_path, monkeypatch):
    """Redirect improvement_engine's on-disk state into tmp_path."""
    improvement_dir = tmp_path / "improvement"
    snapshots_dir = improvement_dir / "snapshots"
    price_cache_dir = improvement_dir / "price_cache"
    proposals_dir = improvement_dir / "proposals"
    for d in (improvement_dir, snapshots_dir, price_cache_dir, proposals_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ie, "IMPROVEMENT_DIR", improvement_dir)
    monkeypatch.setattr(ie, "SNAPSHOTS_DIR", snapshots_dir)
    monkeypatch.setattr(ie, "PRICE_CACHE_DIR", price_cache_dir)
    monkeypatch.setattr(ie, "PROPOSALS_DIR", proposals_dir)
    monkeypatch.setattr(ie, "PERFORMANCE_HISTORY_PATH",
                        improvement_dir / "performance_history.csv")
    monkeypatch.setattr(ie, "LIVE_IC_HISTORY_PATH",
                        improvement_dir / "live_ic_history.csv")
    monkeypatch.setattr(ie, "DISPERSION_HISTORY_PATH",
                        improvement_dir / "dispersion_history.csv")
    monkeypatch.setattr(ie, "METRIC_IC_HISTORY_PATH",
                        improvement_dir / "metric_ic_history.csv")
    monkeypatch.setattr(ie, "CHANGE_LOG_PATH", improvement_dir / "change_log.csv")
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_TICKERS = 60
TICKERS = [f"T{i:03d}" for i in range(N_TICKERS)]


def _snapshot_frame(run_date, run_id, seed=0):
    """A minimal but realistic snapshot: scores that correlate with nothing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Ticker": TICKERS,
        "Sector": ["Technology"] * N_TICKERS,
        "Composite": rng.uniform(20, 100, N_TICKERS),
        "Rank": np.arange(1, N_TICKERS + 1),
        "in_portfolio": False,
        "run_date": run_date,
        "run_id": run_id,
    })
    for col in ie.CATEGORY_SCORES:
        df[col] = rng.uniform(10, 90, N_TICKERS)
    return df


def _write_snapshot(run_date, run_id, seed=0):
    path = ie.SNAPSHOTS_DIR / f"{run_date}_{run_id}.parquet"
    _snapshot_frame(run_date, run_id, seed).to_parquet(path, index=False)
    return path


def _fake_prices(start="2026-01-01", days=200, seed=7):
    """Deterministic daily closes for every ticker, weekdays only."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=days)
    data = {
        t: 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx)))
        for t in TICKERS
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def stub_price_fetch(monkeypatch):
    """Replace the yfinance call. These tests must never touch the network."""
    prices = _fake_prices()
    calls = []

    def _fetch(tickers, start_date, end_date):
        calls.append((tuple(tickers), start_date, end_date))
        return prices

    monkeypatch.setattr(ie, "_fetch_prices_for_returns", _fetch)
    return calls


# ---------------------------------------------------------------------------
# Defect 4: weekend run dates
# ---------------------------------------------------------------------------

class TestMarketDay:
    def test_weekdays_are_market_days(self):
        # 2026-08-24 is a Monday, 2026-08-28 a Friday
        for d in ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"]:
            assert ie._is_market_day(d), d

    def test_weekends_are_not(self):
        # 2026-02-21 Sat, 2026-02-22 Sun - both are real run dates in the
        # live performance history, and both produced live IC rows.
        for d in ["2026-02-21", "2026-02-22", "2026-08-29", "2026-08-30"]:
            assert not ie._is_market_day(d), d

    def test_garbage_is_not_a_market_day(self):
        assert not ie._is_market_day("not-a-date")
        assert not ie._is_market_day(None)


# ---------------------------------------------------------------------------
# Defect 3: overlapping windows counted as independent
# ---------------------------------------------------------------------------

class TestEffectiveObservations:
    def test_single_date_is_one_observation(self):
        assert ie._effective_observations(["2026-03-02"], "1m") == 1

    def test_empty_is_zero(self):
        assert ie._effective_observations([], "1m") == 0
        assert ie._effective_observations(None, "1m") == 0

    def test_dates_inside_one_window_collapse(self):
        """Six run dates in a 9-day stretch are one 1-month observation.

        This is the exact cluster research/2026-08-10-ic-evidence-independence.md
        found in the live snapshot set.
        """
        dates = ["2026-02-20", "2026-02-21", "2026-02-22",
                 "2026-02-23", "2026-02-24", "2026-02-28"]
        assert ie._effective_observations(dates, "1m") == 1

    def test_the_backfill_is_two_observations_not_eleven(self):
        """The documented 2.35x inflation, reproduced.

        All eleven backfillable dates fall inside a 53-day window, so at most
        two non-overlapping 30-day return windows fit among them.
        """
        dates = ["2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23",
                 "2026-02-24", "2026-02-28", "2026-03-01", "2026-03-09",
                 "2026-03-15", "2026-03-16", "2026-04-14"]
        assert len(dates) == 11
        assert ie._effective_observations(dates, "1m") == 2

    def test_horizon_changes_the_count(self):
        """The same dates are more independent at a shorter horizon."""
        dates = ["2026-03-02", "2026-03-10", "2026-03-18", "2026-03-26"]
        assert ie._effective_observations(dates, "1w") == 4   # 8 days apart
        assert ie._effective_observations(dates, "1m") == 1   # 24-day span
        assert ie._effective_observations(dates, "3m") == 1

    def test_never_exceeds_raw_count(self):
        dates = pd.bdate_range("2026-01-01", periods=40).strftime("%Y-%m-%d").tolist()
        for h in ("1w", "1m", "3m"):
            assert ie._effective_observations(dates, h) <= len(dates)

    def test_duplicate_dates_count_once(self):
        assert ie._effective_observations(
            ["2026-03-02", "2026-03-02", "2026-03-02"], "1m") == 1

    def test_inflation_moves_a_borderline_result_through_the_gate(self):
        """Why this matters: t = IR * sqrt(n).

        At the documented borderline IC-IR of 0.5, counting 11 overlapping
        windows as independent turns a clearly-insignificant result into one
        that passes a p<0.05 gate. The effective count does not.
        """
        p_raw = ie._ir_to_one_sided_pvalue(0.5, 11)
        p_effective = ie._ir_to_one_sided_pvalue(0.5, 2)
        assert p_raw < 0.05, "raw count clears the significance gate"
        assert p_effective > 0.20, "effective count correctly does not"


# ---------------------------------------------------------------------------
# Defects 2 and 4: normalization of performance history
# ---------------------------------------------------------------------------

class TestNormalizePerformanceHistory:
    def test_duplicate_ticker_dates_collapse(self):
        df = pd.DataFrame({
            "run_date": ["2026-03-02"] * 4,
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "fwd_return_1w": [0.01, 0.02, 0.01, 0.02],
        })
        out = ie._normalize_performance_history(df)
        assert len(out) == 2
        assert set(out["ticker"]) == {"AAA", "BBB"}

    def test_weekend_run_dates_are_dropped(self):
        df = pd.DataFrame({
            "run_date": ["2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23"],
            "ticker": ["AAA"] * 4,
            "fwd_return_1w": [0.01, 0.02, 0.03, 0.04],
        })
        out = ie._normalize_performance_history(df)
        assert sorted(out["run_date"]) == ["2026-02-20", "2026-02-23"]

    def test_later_values_win_but_do_not_erase_earlier_ones(self):
        """The upsert that makes horizon backfill possible.

        Reprocessing a date to fill in its 1-month return must not blank the
        1-week return recorded when the date was first processed.
        """
        df = pd.DataFrame({
            "run_date": ["2026-03-02", "2026-03-02"],
            "ticker": ["AAA", "AAA"],
            "fwd_return_1w": [0.05, np.nan],
            "fwd_return_1m": [np.nan, 0.09],
        })
        out = ie._normalize_performance_history(df)
        assert len(out) == 1
        assert out["fwd_return_1w"].iloc[0] == pytest.approx(0.05)
        assert out["fwd_return_1m"].iloc[0] == pytest.approx(0.09)

    def test_column_order_is_preserved(self):
        df = pd.DataFrame({
            "run_date": ["2026-03-02"], "ticker": ["AAA"],
            "composite_score": [50.0], "fwd_return_1w": [0.01],
        })
        out = ie._normalize_performance_history(df)
        assert list(out.columns) == list(df.columns)

    def test_empty_and_malformed_pass_through(self):
        assert ie._normalize_performance_history(None) is None
        empty = pd.DataFrame()
        assert len(ie._normalize_performance_history(empty)) == 0
        no_key = pd.DataFrame({"foo": [1, 2]})
        assert len(ie._normalize_performance_history(no_key)) == 2


# ---------------------------------------------------------------------------
# Defect 1: a date is never revisited as it ages
# ---------------------------------------------------------------------------

class TestHorizonAwareReprocessing:
    def test_one_month_return_is_filled_in_later(self, stub_price_fetch):
        """The priority-0 bug, end to end.

        2026-03-02 is a Monday. Process it at 7 days old: only the 1-week
        return is computable. Process it again at 40 days old: the 1-month
        return must appear. Before the fix the date was skipped forever and
        fwd_return_1m stayed NaN, which is why the engine - which optimizes on
        the '1m' horizon - could never propose anything.
        """
        _write_snapshot("2026-03-02", "aaaaaaaaaaaa")

        ie.compute_forward_returns("2026-03-09")   # 7 days old
        first = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
        assert first["fwd_return_1w"].notna().any()
        assert not first["fwd_return_1m"].notna().any(), "1m cannot exist yet"

        ie.compute_forward_returns("2026-04-11")   # 40 days old
        second = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
        assert second["fwd_return_1m"].notna().any(), (
            "the date was never revisited once it aged into the 1m horizon"
        )
        assert second["fwd_return_1w"].notna().any(), (
            "revisiting must not erase the return recorded the first time"
        )

    def test_revisiting_does_not_duplicate_rows(self, stub_price_fetch):
        _write_snapshot("2026-03-02", "aaaaaaaaaaaa")
        ie.compute_forward_returns("2026-03-09")
        n_first = len(pd.read_csv(ie.PERFORMANCE_HISTORY_PATH))
        ie.compute_forward_returns("2026-04-11")
        after = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
        assert len(after) == n_first
        assert not after.duplicated(["run_date", "ticker"]).any()

    def test_a_fully_processed_date_is_not_refetched(self, stub_price_fetch):
        """Once every eligible horizon has a value there is nothing to redo."""
        _write_snapshot("2026-03-02", "aaaaaaaaaaaa")
        ie.compute_forward_returns("2026-06-15")   # >90 days: all horizons
        n_calls = len(stub_price_fetch)
        ie.compute_forward_returns("2026-06-16")
        assert len(stub_price_fetch) == n_calls, "re-fetched an exhausted date"

    def test_price_window_is_bounded_by_the_horizon(self, stub_price_fetch):
        """The fetch window must not stretch to today.

        _fetch_prices_for_returns caches on (start, end). If end is the current
        date the key changes daily, so every revisited snapshot becomes a fresh
        full-universe yfinance download - and rate limiting already costs the
        data loop 10-25% of its tickers per run.
        """
        _write_snapshot("2026-03-02", "aaaaaaaaaaaa")
        ie.compute_forward_returns("2026-12-01")   # long after all horizons

        _tickers, start, end = stub_price_fetch[-1]
        assert start == "2026-03-02"
        # 90-day horizon from 2026-03-02, not the 2026-12-01 run date
        assert end == "2026-05-31", f"fetch window ran to {end}"


# ---------------------------------------------------------------------------
# Defect 2: one snapshot per date, not one per file
# ---------------------------------------------------------------------------

class TestSnapshotDeduplication:
    def test_four_runs_on_one_day_produce_one_observation(self, stub_price_fetch):
        """2026-02-21 had thirteen snapshot files. Each was processed, so the
        same 503 tickers were appended thirteen times and the resulting IC row
        claimed 6,539 tickers."""
        for i, run_id in enumerate(["aaa1", "bbb2", "ccc3", "ddd4"]):
            _write_snapshot("2026-03-02", run_id, seed=i)

        ie.compute_forward_returns("2026-03-10")
        perf = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)

        assert not perf.duplicated(["run_date", "ticker"]).any()
        assert len(perf) <= N_TICKERS

    def test_weekend_snapshots_are_not_processed(self, stub_price_fetch):
        _write_snapshot("2026-02-20", "friday000000")   # Friday
        _write_snapshot("2026-02-21", "saturday0000")   # Saturday
        _write_snapshot("2026-02-22", "sunday000000")   # Sunday

        ie.compute_forward_returns("2026-03-05")
        perf = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
        assert set(perf["run_date"]) == {"2026-02-20"}


# ---------------------------------------------------------------------------
# Defect 5: the IC series never advanced
# ---------------------------------------------------------------------------

class TestLiveICAccrual:
    def _seed_history(self, dates):
        rng = np.random.default_rng(3)
        rows = []
        for d in dates:
            for t in TICKERS:
                row = {
                    "run_date": d, "ticker": t,
                    "composite_score": rng.uniform(20, 100),
                    "rank": 1, "in_portfolio": False,
                    "fwd_return_1w": rng.normal(0, 0.03),
                    "fwd_return_1m": np.nan, "fwd_return_3m": np.nan,
                }
                for c in ie.CATEGORY_SCORES:
                    row[c] = rng.uniform(10, 90)
                rows.append(row)
        pd.DataFrame(rows).to_csv(ie.PERFORMANCE_HISTORY_PATH, index=False)

    def test_recording_a_snapshot_advances_the_ic_series(self, stub_price_fetch, monkeypatch):
        """record_run_snapshot() must compute live IC.

        It previously called only compute_dispersion() and
        compute_forward_returns(), so the data loop ran every weekday for 183
        days without live_ic_history.csv gaining a single row.
        """
        self._seed_history(["2026-02-20", "2026-02-23", "2026-02-24"])
        assert not ie.LIVE_IC_HISTORY_PATH.exists()

        scored = _snapshot_frame("2026-03-02", "zzzzzzzzzzzz")
        ie.record_run_snapshot("zzzzzzzzzzzz", "2026-03-02", scored, None, {})

        assert ie.LIVE_IC_HISTORY_PATH.exists(), "IC series did not advance"
        ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
        assert len(ic) == 3
        assert set(ic["run_date"]) == {"2026-02-20", "2026-02-23", "2026-02-24"}

    def test_stale_rows_are_regenerated_not_preserved(self):
        """live_ic_history.csv is derived, so it self-heals.

        The three rows on the live file were computed from duplicated data (one
        reported 2,012 tickers for 503 real ones) and two of them are weekend
        dates. Appending only unseen keys would have kept them forever.
        """
        self._seed_history(["2026-02-20", "2026-02-23"])
        pd.DataFrame([{
            "run_date": "2026-02-20", "horizon": "1w", "n_tickers": 2012,
            "composite_ic": 0.045,
            **{f"{c}_ic": 0.0 for c in ie.CATEGORY_NAMES},
        }, {
            "run_date": "2026-02-21", "horizon": "1w", "n_tickers": 6539,
            "composite_ic": 0.061,
            **{f"{c}_ic": 0.0 for c in ie.CATEGORY_NAMES},
        }]).to_csv(ie.LIVE_IC_HISTORY_PATH, index=False)

        ie.compute_live_ic(horizon="1w")
        ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)

        assert "2026-02-21" not in set(ic["run_date"]), "weekend row survived"
        row = ic[ic["run_date"] == "2026-02-20"].iloc[0]
        assert row["n_tickers"] == N_TICKERS, "inflated ticker count survived"

    def test_other_horizons_are_untouched(self):
        self._seed_history(["2026-02-20", "2026-02-23"])
        pd.DataFrame([{
            "run_date": "2026-01-05", "horizon": "3m", "n_tickers": 500,
            "composite_ic": 0.02,
            **{f"{c}_ic": 0.0 for c in ie.CATEGORY_NAMES},
        }]).to_csv(ie.LIVE_IC_HISTORY_PATH, index=False)

        ie.compute_live_ic(horizon="1w")
        ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
        assert (ic["horizon"] == "3m").sum() == 1

    def test_duplicate_and_weekend_rows_never_reach_the_ic(self):
        """A stale file on disk cannot reintroduce the defects."""
        self._seed_history(["2026-02-20"])
        perf = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
        weekend = perf.copy()
        weekend["run_date"] = "2026-02-21"
        pd.concat([perf, perf, weekend], ignore_index=True).to_csv(
            ie.PERFORMANCE_HISTORY_PATH, index=False)

        ie.compute_live_ic(horizon="1w")
        ic = pd.read_csv(ie.LIVE_IC_HISTORY_PATH)
        assert len(ic) == 1
        assert ic["run_date"].iloc[0] == "2026-02-20"
        assert ic["n_tickers"].iloc[0] == N_TICKERS


# ---------------------------------------------------------------------------
# Defect 3, wired through: the gates must read the effective count
# ---------------------------------------------------------------------------

class TestTrendsUseEffectiveCount:
    def _write_ic(self, dates, horizon="1m", ic=0.05):
        rng = np.random.default_rng(11)
        rows = []
        for d in dates:
            rows.append({
                "run_date": d, "horizon": horizon, "n_tickers": 500,
                "composite_ic": ic,
                **{f"{c}_ic": ic + rng.normal(0, 0.005) for c in ie.CATEGORY_NAMES},
            })
        pd.DataFrame(rows).to_csv(ie.LIVE_IC_HISTORY_PATH, index=False)

    def test_clustered_dates_do_not_clear_the_trend_gate(self):
        """Eleven IC rows inside a 53-day window are two 1-month observations,
        which is below the 6 the trend analysis requires."""
        self._write_ic(["2026-02-20", "2026-02-23", "2026-02-24", "2026-02-25",
                        "2026-02-26", "2026-02-27", "2026-03-02", "2026-03-03",
                        "2026-03-04", "2026-03-05", "2026-04-14"])
        trends = ie.analyze_ic_trends(horizon="1m")
        assert "_warning" in trends, "raw row count cleared a gate it should not"
        assert trends["_n_observations"] == 2
        assert trends["_n_raw_observations"] == 11

    def test_genuinely_spaced_dates_do_clear_it(self):
        """The same eleven-row count, spread a month apart, is real evidence."""
        dates = pd.date_range("2026-01-05", periods=11, freq="35D").strftime("%Y-%m-%d")
        self._write_ic(list(dates))
        trends = ie.analyze_ic_trends(horizon="1m")
        assert "_warning" not in trends
        assert trends["_n_observations"] == 11

    def test_per_category_counts_are_effective_too(self):
        """This value reaches _ir_to_one_sided_pvalue() as t = IR * sqrt(n)."""
        dates = pd.date_range("2026-01-05", periods=11, freq="35D").strftime("%Y-%m-%d")
        self._write_ic(list(dates))
        trends = ie.analyze_ic_trends(horizon="1m")
        for cat in ie.CATEGORY_NAMES:
            if isinstance(trends.get(cat), dict):
                assert trends[cat]["n_observations"] <= trends[cat]["n_raw_observations"]

    def test_proposal_refuses_on_clustered_evidence(self):
        """The end the whole package exists to protect: no weight proposal off
        eleven overlapping windows."""
        self._write_ic(["2026-02-20", "2026-02-23", "2026-02-24", "2026-02-25",
                        "2026-02-26", "2026-02-27", "2026-03-02", "2026-03-03",
                        "2026-03-04", "2026-03-05", "2026-04-14"])
        result = ie.propose_weight_changes(min_observations=8)
        assert result["status"] == "insufficient_data"
        assert result["n_observations"] == 2


# ---------------------------------------------------------------------------
# Defect 6 (2026-10-16): a horizon measured before its end close existed
#
# The data loop runs at 02:00, before that day's close. On the morning of
# D+30 the newest close is D+29, and `_nearest_price` (either direction, within
# 7 days) used it as the 1-month end; the truncated frame was cached under the
# (D, D+30) key and served on every later run; the 3-month pass then
# re-measured 1w/1m with complete data and rewrote their IC rows. Measured: 25
# of 31 `1m` IC rows were on the short window.
# ---------------------------------------------------------------------------

def _yf_frame(prices):
    """What yfinance.download returns for several tickers."""
    return pd.concat({"Close": prices}, axis=1)


@pytest.fixture
def fake_yfinance(monkeypatch):
    """A stand-in yfinance module whose download serves a settable frame."""
    state = {"frame": None, "calls": []}

    def download(tickers, start=None, end=None, **_kw):
        state["calls"].append((start, end))
        f = state["frame"]
        return _yf_frame(f.loc[(f.index >= start) & (f.index < end)])

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    return state


@pytest.fixture
def settable_price_fetch(monkeypatch):
    """Like stub_price_fetch, but the test chooses the frame each call returns."""
    state = {"frame": _fake_prices()}

    def _fetch(tickers, start_date, end_date):
        return state["frame"]

    monkeypatch.setattr(ie, "_fetch_prices_for_returns", _fetch)
    return state


class TestHorizonEndClose:
    def test_a_frame_ending_the_day_before_the_end_leaves_the_horizon_unfilled(
            self, settable_price_fetch):
        """The bug itself. 2026-03-02 + 30 days is Wednesday 2026-04-01; at
        02:00 that morning the newest close is 03-31. The old rule took it."""
        _write_snapshot("2026-03-02", "aaaaaaaaaaaa")
        full = _fake_prices()
        ie.compute_forward_returns("2026-03-10")                 # 1w recorded

        settable_price_fetch["frame"] = full[full.index <= "2026-03-31"]
        ie.compute_forward_returns("2026-04-01")
        perf = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH)
        assert perf["fwd_return_1m"].isna().all(), "measured to the day before the end"
        assert ie.compute_live_ic(horizon="1m") is None

        settable_price_fetch["frame"] = full
        ie.compute_forward_returns("2026-04-02")                 # the retry
        perf = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH).set_index("ticker")
        t = TICKERS[0]
        expected = full.loc["2026-04-01", t] / full.loc["2026-03-02", t] - 1
        assert perf.loc[t, "fwd_return_1m"] == pytest.approx(expected)

    def test_the_end_close_is_never_an_earlier_one(self):
        s = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-03-30", "2026-03-31"]))
        assert ie._horizon_end_close(s, pd.Timestamp("2026-04-01")) is None
        assert np.isnan(ie._compute_fwd_return(s, pd.Timestamp("2026-03-02"), 30, 100.0))

    def test_a_weekend_end_resolves_to_monday_not_friday(self):
        # 2026-04-04 is a Saturday
        s = pd.Series([110.0, 120.0], index=pd.to_datetime(["2026-04-03", "2026-04-06"]))
        assert ie._horizon_end_close(s, pd.Timestamp("2026-04-04")) == 120.0

    def test_a_holiday_end_resolves_to_the_next_trading_day(self):
        # 2026-09-07 is Labor Day: no close
        s = pd.Series([50.0, 51.0, 52.0],
                      index=pd.to_datetime(["2026-09-04", "2026-09-08", "2026-09-09"]))
        assert ie._horizon_end_close(s, pd.Timestamp("2026-09-07")) == 51.0

    def test_the_forward_search_is_bounded(self):
        s = pd.Series([50.0, 60.0], index=pd.to_datetime(["2026-09-01", "2026-09-20"]))
        assert ie._horizon_end_close(s, pd.Timestamp("2026-09-07")) is None

    def test_a_recorded_horizon_is_not_rewritten_by_a_later_pass(self, settable_price_fetch):
        """The 3-month pass used to re-measure 1w and 1m and rewrite their IC rows."""
        _write_snapshot("2026-03-02", "aaaaaaaaaaaa")
        ie.compute_forward_returns("2026-04-03")                 # 1w + 1m
        before = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH).set_index("ticker")
        assert before["fwd_return_1m"].notna().all()

        settable_price_fetch["frame"] = _fake_prices(seed=99)    # different prices
        ie.compute_forward_returns("2026-06-15")                 # 3m comes due
        after = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH).set_index("ticker")

        assert after["fwd_return_3m"].notna().any()
        pd.testing.assert_series_equal(before["fwd_return_1w"], after["fwd_return_1w"])
        pd.testing.assert_series_equal(before["fwd_return_1m"], after["fwd_return_1m"])


class TestEntryCloseOnAHoliday:
    """A run made on a market holiday enters at the next session's close (2026-10-27).

    The data loop runs Monday to Friday, holidays included, from the previous session's
    close. The entry price was the nearest close in either direction, ties to the earlier,
    so Thanksgiving's run entered at Wednesday's close - the one its ranking was built from,
    before the ranking existed. ``research/2026-10-27-tradingagents-ideas.md`` section 3.
    """

    def _prices_without(self, holiday):
        full = _fake_prices(start="2026-09-01", days=120)
        return full.drop(pd.Timestamp(holiday))

    def test_good_friday_enters_at_mondays_close_through_the_whole_pass(self, settable_price_fetch):
        """2026-04-03 (Good Friday) ran at 02:00 from Thursday's close. The old rule entered at
        Thursday 04-02 (one day back, against Monday three days forward)."""
        prices = _fake_prices(start="2026-01-01", days=200).drop(pd.Timestamp("2026-04-03"))
        settable_price_fetch["frame"] = prices
        _write_snapshot("2026-04-03", "bbbbbbbbbbbb")
        ie.compute_forward_returns("2026-04-17")
        perf = pd.read_csv(ie.PERFORMANCE_HISTORY_PATH).set_index("ticker")
        t = TICKERS[0]
        end = prices.loc[prices.index >= "2026-04-10", t].iloc[0]
        assert perf.loc[t, "fwd_return_1w"] == pytest.approx(end / prices.loc["2026-04-06", t] - 1)
        assert perf.loc[t, "fwd_return_1w"] != pytest.approx(end / prices.loc["2026-04-02", t] - 1)

    def test_thanksgiving_enters_on_friday_not_wednesday(self):
        prices = self._prices_without("2026-11-26")[TICKERS[0]]
        assert ie._entry_close(prices, pd.Timestamp("2026-11-26")) == prices.loc["2026-11-27"]
        assert ie._nearest_price(prices, pd.Timestamp("2026-11-26")) == prices.loc["2026-11-25"]

    def test_a_friday_holiday_enters_on_monday_not_thursday(self):
        prices = self._prices_without("2026-12-25")[TICKERS[0]]
        assert ie._entry_close(prices, pd.Timestamp("2026-12-25")) == prices.loc["2026-12-28"]
        assert ie._nearest_price(prices, pd.Timestamp("2026-12-25")) == prices.loc["2026-12-24"]

    def test_an_ordinary_weekday_is_unchanged(self):
        prices = _fake_prices(start="2026-09-01", days=120)[TICKERS[0]]
        for d in ("2026-10-27", "2026-09-08", "2026-11-02"):
            assert ie._entry_close(prices, pd.Timestamp(d)) == ie._nearest_price(prices, pd.Timestamp(d))


class TestPriceCacheCompleteness:
    def test_an_incomplete_window_is_not_cached(self, fake_yfinance):
        full = _fake_prices()
        fake_yfinance["frame"] = full[full.index <= "2026-03-31"]
        got = ie._fetch_prices_for_returns(TICKERS, "2026-03-02", "2026-04-01")

        assert got is not None and got.index.max() == pd.Timestamp("2026-03-31")
        assert not (ie.PRICE_CACHE_DIR / "prices_2026-03-02_2026-04-01.parquet").exists()

    def test_a_complete_window_is_cached(self, fake_yfinance):
        fake_yfinance["frame"] = _fake_prices()
        ie._fetch_prices_for_returns(TICKERS, "2026-03-02", "2026-04-01")
        assert (ie.PRICE_CACHE_DIR / "prices_2026-03-02_2026-04-01.parquet").exists()

    def test_a_truncated_cache_file_is_refetched(self, fake_yfinance):
        """The frames cached before the fix are short; they must not be served."""
        full = _fake_prices()
        path = ie.PRICE_CACHE_DIR / "prices_2026-03-02_2026-04-01.parquet"
        full[(full.index >= "2026-02-25") & (full.index <= "2026-03-31")].to_parquet(path)

        fake_yfinance["frame"] = full
        got = ie._fetch_prices_for_returns(TICKERS, "2026-03-02", "2026-04-01")
        assert len(fake_yfinance["calls"]) == 1, "served the truncated cache"
        assert got.index.max() >= pd.Timestamp("2026-04-01")

    def test_a_complete_cache_file_is_served(self, fake_yfinance):
        full = _fake_prices()
        full[full.index <= "2026-04-08"].to_parquet(
            ie.PRICE_CACHE_DIR / "prices_2026-03-02_2026-04-01.parquet")
        fake_yfinance["frame"] = full
        ie._fetch_prices_for_returns(TICKERS, "2026-03-02", "2026-04-01")
        assert fake_yfinance["calls"] == []

    def test_a_bar_dated_today_is_dropped(self, fake_yfinance):
        # clock: the claim is only that a bar dated today is dropped, which holds for any today -
        # bdate_range just yields the last weekday on or before it, and nothing asserts which.
        today = pd.Timestamp(pd.Timestamp.now().date())
        idx = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=30).append(
            pd.DatetimeIndex([today]))
        fake_yfinance["frame"] = pd.DataFrame(
            {t: np.linspace(100, 130, len(idx)) for t in TICKERS}, index=idx)
        start = idx[0].strftime("%Y-%m-%d")
        got = ie._fetch_prices_for_returns(TICKERS, start, today.strftime("%Y-%m-%d"))
        assert today not in got.index
        assert not any(ie.PRICE_CACHE_DIR.iterdir()), "cached a window ending today"

    def test_a_mostly_empty_last_bar_is_not_a_close(self):
        f = _fake_prices(days=10)
        f.iloc[-1, 1:] = np.nan                     # one ticker of 60 has a value
        assert ie._last_complete_close(f) == f.index[-2]
        assert ie._last_complete_close(None) is None
