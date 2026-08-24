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
"""

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
