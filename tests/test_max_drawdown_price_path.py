"""`max_drawdown_1y` is the largest fall of the price path, and the page shows which two
closes it was measured between.

WHY THIS EXISTS - 2026-10-08.

`fetch_fundamentals` stores **log** returns (`log(close / close.shift(1))`). Step 16d of
`compute_metrics` used to build the path it measured on with `cumprod(1 + log r)`, which is
neither the price path nor the log path: because ln(1+r) <= r it drifts below the real path,
and the drift is path-dependent, so the peak-to-trough ratio taken on it was not the stock's
actual largest fall. Measured over 50 real 13-month histories it overstated the fall for 50
of 50 tickers - median 1.09pp, max 4.70pp (AMD -32.46% against -27.76%) - and because the
bias grows with volatility it fell hardest on exactly the stocks a tail-risk metric exists to
separate. See `research/measurements/2026-10-08-max-drawdown-log-return-compounding.py` and
`METHODOLOGY_CHANGELOG.md` 2026-10-08.

These tests work from a price series whose drawdown is known by construction, so they pin the
definition rather than a recorded number.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import metric_lineage as ml  # noqa: E402
from factor_engine import compute_metrics  # noqa: E402

TRADING_DAYS = pd.bdate_range("2025-09-01", periods=260)


def _record(closes, ticker="TEST"):
    """A raw-fetch record carrying a price history, the way the fetch writes it."""
    s = pd.Series(closes, index=TRADING_DAYS[: len(closes)], dtype=float)
    log_ret = np.log(s / s.shift(1)).dropna()
    return {
        "Ticker": ticker,
        "Sector": "Information Technology",
        "price_latest": float(s.iloc[-1]),
        "volatility_1y": float(log_ret.std() * math.sqrt(252)),
        "_daily_returns": {d.strftime("%Y-%m-%d"): float(v)
                           for d, v in zip(log_ret.index, log_ret.values)},
    }


def _metrics(closes):
    df = compute_metrics([_record(closes)], pd.Series(dtype=float))
    return df.iloc[0]


def _sawtooth(n, peak_at, trough_at, start=100.0, peak=150.0, trough=90.0, end=120.0):
    """A path that rises to `peak`, falls to `trough`, then recovers to `end`."""
    legs = [np.linspace(start, peak, peak_at + 1),
            np.linspace(peak, trough, trough_at - peak_at + 1)[1:],
            np.linspace(trough, end, n - trough_at)[1:]]
    return np.concatenate(legs)


# --------------------------------------------------------------------------- the definition

def test_the_drawdown_is_the_fall_of_the_price_path():
    """150 -> 90 is a 40% fall, whatever the path on either side of it."""
    closes = _sawtooth(240, peak_at=60, trough_at=150)
    row = _metrics(closes)
    assert row["max_drawdown_1y"] == pytest.approx(-0.40, abs=5e-4)


def test_a_monotonically_rising_series_has_no_drawdown():
    row = _metrics(np.linspace(100.0, 200.0, 240))
    assert row["max_drawdown_1y"] == pytest.approx(0.0, abs=1e-9)


def test_the_deepest_fall_wins_not_the_latest():
    """Peak 150 -> 90 (-40%) then 130 -> 110 (-15.4%): the metric reports -40%."""
    closes = np.concatenate([
        np.linspace(100.0, 150.0, 60),
        np.linspace(150.0, 90.0, 60)[1:],
        np.linspace(90.0, 130.0, 60)[1:],
        np.linspace(130.0, 110.0, 64)[1:],
    ])
    row = _metrics(closes)
    assert row["max_drawdown_1y"] == pytest.approx(-0.40, abs=5e-4)


def test_the_drawdown_matches_an_independent_calculation_on_the_closes():
    """Against the textbook definition applied straight to the closes - no returns involved.

    Magdon-Ismail & Atiya (2004) and Chekhlov, Uryasev & Zabarankin (2005) both define the
    maximum drawdown on the price/equity path: min_t (P_t - max_{s<=t} P_s) / max_{s<=t} P_s.
    """
    rng = np.random.default_rng(20261008)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, 240)))
    row = _metrics(closes)
    # The engine only sees returns, so its window starts one close later.
    path = closes[1:]
    running_peak = np.maximum.accumulate(path)
    expected = float(np.min((path - running_peak) / running_peak))
    assert row["max_drawdown_1y"] == pytest.approx(expected, abs=1e-9)


def test_compounding_log_returns_as_simple_ones_would_overstate_the_fall():
    """The old formula, kept here as the thing that must not come back.

    On a volatile path `cumprod(1 + log r)` reports a strictly deeper fall than the price
    path does, which is why the fix was one-directional for all 50 tickers measured."""
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 240)))
    row = _metrics(closes)

    log_ret = np.diff(np.log(closes))
    old_cum = np.cumprod(1 + log_ret)
    old_peak = np.maximum.accumulate(old_cum)
    old = float(np.min((old_cum - old_peak) / old_peak))

    assert old < row["max_drawdown_1y"] - 1e-4, (
        f"the old formula gave {old:.4%} and the price path gives "
        f"{row['max_drawdown_1y']:.4%}; this series no longer distinguishes them")


def test_the_series_is_read_in_date_order_not_insertion_order():
    """The drawdown is order-dependent. A fetch dict that arrived shuffled must still give
    the chronological answer, because the metric is sorted by date before it is measured."""
    closes = _sawtooth(240, peak_at=60, trough_at=150)
    rec = _record(closes)
    items = list(rec["_daily_returns"].items())
    rng = np.random.default_rng(3)
    rng.shuffle(items)
    rec["_daily_returns"] = dict(items)
    row = compute_metrics([rec], pd.Series(dtype=float)).iloc[0]
    assert row["max_drawdown_1y"] == pytest.approx(-0.40, abs=5e-4)


# --------------------------------------------------------------------------- what is published

def test_the_engine_publishes_the_two_closes_it_measured_between():
    closes = _sawtooth(240, peak_at=60, trough_at=150)
    row = _metrics(closes)
    assert row["_mdd_peak"] == pytest.approx(150.0, abs=0.3)
    assert row["_mdd_trough"] == pytest.approx(90.0, abs=0.3)
    # Rebased to real prices, so the pair is readable as money and not as index levels.
    assert row["_mdd_peak"] > row["_mdd_trough"] > 0


def test_the_published_dates_bracket_the_fall():
    closes = _sawtooth(240, peak_at=60, trough_at=150)
    row = _metrics(closes)
    assert row["_mdd_peak_date"] < row["_mdd_trough_date"]
    dates = sorted(_record(closes)["_daily_returns"])
    assert row["_mdd_peak_date"] in dates and row["_mdd_trough_date"] in dates


def test_the_published_pair_rebuilds_the_published_value():
    """The equation on the row is the engine's arithmetic, for a path of known shape."""
    closes = _sawtooth(240, peak_at=60, trough_at=150)
    row = _metrics(closes)
    inp = {"mdd_peak": row["_mdd_peak"], "mdd_trough": row["_mdd_trough"]}
    assert ml.RECOMPUTE["max_drawdown_1y"](inp) == pytest.approx(row["max_drawdown_1y"], abs=1e-9)
    tpl = ml.choose_template("max_drawdown_1y", inp)
    assert tpl is not None
    assert ml.evaluate_template(tpl, inp) == pytest.approx(row["max_drawdown_1y"], abs=1e-9)


def test_a_thin_history_gets_no_drawdown_and_no_pair():
    """Below the 200-return gate the metric is withheld - and so is the equation, rather
    than a pair of closes with nothing to explain."""
    row = _metrics(np.linspace(100.0, 80.0, 150))
    assert pd.isna(row["max_drawdown_1y"])
    assert "_mdd_peak" not in row or pd.isna(row["_mdd_peak"])
    assert ml.choose_template("max_drawdown_1y", {}) is None


def test_the_rebasing_cannot_move_the_metric():
    """The pair is rebased onto `price_latest`; the ratio is scale-invariant, so a record
    with a missing latest price scores the same drawdown."""
    closes = _sawtooth(240, peak_at=60, trough_at=150)
    with_price = _metrics(closes)["max_drawdown_1y"]
    rec = _record(closes)
    del rec["price_latest"]
    without = compute_metrics([rec], pd.Series(dtype=float)).iloc[0]["max_drawdown_1y"]
    assert with_price == pytest.approx(without, abs=1e-12)
