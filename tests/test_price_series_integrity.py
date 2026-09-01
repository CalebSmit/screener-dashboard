"""Regression tests for the price-series split-scale integrity check.

Why this file exists
--------------------
Nine of the 44 metrics are derived from a single ``Ticker.history()`` call:
the whole ``risk`` category (volatility, beta, sharpe_ratio, sortino_ratio,
max_drawdown_1y) and three quarters of ``momentum`` (return_12_1, return_6m,
jensens_alpha - only proximity_52w_high comes from elsewhere).  Nothing
checked that the series was internally consistent.

On 2026-08-26 it was not.  Yahoo's 13-month series for MNST alternated
between pre-split and post-split prices across its 2026-08-11 2:1 split:

    2026-08-05    94.46      <- unadjusted
    2026-08-06    47.08      <- adjusted
    2026-08-07    90.36      <- unadjusted
    2026-08-11    45.53      <- split date

so ``return_12_1`` was computed as (93.49 - 62.30)/62.30 = **+0.5006**,
mixing an unadjusted July price with an adjusted 2025 price.  That put MNST
in the **97th percentile** of 12-1 momentum when its true split-adjusted
12-1 return was about **-25%**, the 3rd percentile - a swing of roughly 110
composite ranks.  ``auto_adjust=False`` returned byte-identical numbers, so
no adjustment was ever being applied.

Note for anyone reading NIGHTLY_LOG.md 2026-08-25, which recorded this the
other way round: the 97.1 reading is the artifact and the 2.9 reading was
correct.  See METHODOLOGY_CHANGELOG.md 2026-08-26.

The rule these tests pin down
-----------------------------
The check is *exact*, not heuristic: it uses the split ratio Yahoo itself
reports.  A correctly back-adjusted series contains no day whose
close-to-close price ratio is near 1/k or k for a declared split of ratio k.

Two calibration facts, both measured 2026-08-26 and both load-bearing:

* Arming floor (``_SPLIT_MIN_JUMP`` = 0.25).  Over 137,313 ticker-days
  (503 S&P 500 names, 13 months) p99.9 of |daily return| is 17.2% and only
  21 days exceed 30%.  Ratios implying a jump smaller than 25% cannot be
  told apart from ordinary trading, so they are left alone.  This is what
  keeps the small spin-off "ratios" Yahoo also reports as splits (SPGI
  1.057, HON 1.061, CMCSA 1.067, FDX 1.241, BDX 1.272) from flagging every
  ordinary down day.
* Verified against all 17 real S&P 500 split events of the prior 13 months:
  11 were large enough to arm the check and it fired on exactly one - MNST -
  with no false positives, including volatile controls (MRNA's genuine
  +177% day, SMCI, ORCL, DELL).
"""

import numpy as np
import pandas as pd
import pytest

from factor_engine import (
    PRICE_SERIES_DERIVED_FIELDS,
    _SPLIT_MIN_JUMP,
    check_price_series_integrity,
)


# ---------------------------------------------------------------- helpers

def _series(values, start="2025-08-01"):
    idx = pd.bdate_range(start=start, periods=len(values), tz="America/New_York")
    return pd.Series([float(v) for v in values], index=idx)


def _splits(closes, events=None):
    """A "Stock Splits" column: zero everywhere except the given events."""
    s = pd.Series(0.0, index=closes.index)
    for pos, ratio in (events or {}).items():
        s.iloc[pos] = float(ratio)
    return s


def _clean_walk(n=260, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.012, size=n)
    return _series(start_price * np.exp(np.cumsum(steps)))


# ------------------------------------------------- the check: acceptance

def test_clean_series_with_no_splits_is_accepted():
    c = _clean_walk()
    assert check_price_series_integrity(c, _splits(c)) is None


def test_correctly_back_adjusted_split_is_accepted():
    """A back-adjusted series shows an ordinary return on the split day."""
    c = _clean_walk()
    sp = _splits(c, {150: 2.0})
    assert check_price_series_integrity(c, sp) is None


def test_no_declared_split_means_no_opinion_even_on_a_huge_move():
    """MRNA moved +177% in one day on 2026-08-19 with no split.

    The check must stay silent: it only ever speaks about days that match a
    ratio the exchange actually declared.
    """
    vals = list(_clean_walk(200).values)
    vals[120] = vals[119] * 2.77          # a genuine +177% day
    c = _series(vals)
    assert check_price_series_integrity(c, _splits(c)) is None


def test_small_spinoff_ratios_do_not_arm_the_check():
    """SPGI 1.057, HON 1.061, CMCSA 1.067 are reported as splits.

    1/1.057 - 1 = -5.4%, well inside the ordinary daily return
    distribution, so arming on them would flag routine down days.
    """
    c = _clean_walk(seed=3)
    for ratio in (1.057, 1.061, 1.067, 1.241, 1.272):
        assert check_price_series_integrity(c, _splits(c, {150: ratio})) is None, ratio


def test_arming_floor_is_where_the_measurement_put_it():
    assert _SPLIT_MIN_JUMP == 0.25


# ------------------------------------------------- the check: rejection

def test_unadjusted_two_for_one_split_is_rejected():
    """The textbook failure: history never back-adjusted, so the split day
    shows -50%."""
    vals = list(_clean_walk(200).values)
    for i in range(120, 200):
        vals[i] /= 2.0
    c = _series(vals)
    reason = check_price_series_integrity(c, _splits(c, {120: 2.0}))
    assert reason is not None
    assert "2:1" in reason


def test_mnst_shaped_alternating_scale_is_rejected():
    """MNST's actual failure: the series flips scale repeatedly.

    This is why the fix withholds the metrics instead of repairing them -
    there is no single factor that puts an alternating series right.
    """
    vals = list(_clean_walk(200).values)
    for i in (118, 122, 126):             # isolated adjusted days...
        vals[i] /= 2.0
    for i in range(130, 200):             # ...then the split itself
        vals[i] /= 2.0
    c = _series(vals)
    reason = check_price_series_integrity(c, _splits(c, {130: 2.0}))
    assert reason is not None
    assert "mixes pre- and post-split prices" in reason


def test_reverse_split_is_also_checked():
    """AMCR's 2026-01-15 event is reported as ratio 0.2, i.e. 1-for-5
    reverse: an unadjusted series jumps +400%, not -80%."""
    vals = list(_clean_walk(200).values)
    for i in range(120, 200):
        vals[i] *= 5.0
    c = _series(vals)
    assert check_price_series_integrity(c, _splits(c, {120: 0.2})) is not None


def test_reason_names_the_offending_dates():
    vals = list(_clean_walk(200).values)
    for i in range(120, 200):
        vals[i] /= 2.0
    c = _series(vals)
    reason = check_price_series_integrity(c, _splits(c, {120: 2.0}))
    assert str(c.index[120].date()) in reason


# ------------------------------------------------- degenerate inputs

@pytest.mark.parametrize("closes,splits", [
    (None, None),
    (_clean_walk(50), None),
])
def test_missing_inputs_are_not_an_error(closes, splits):
    assert check_price_series_integrity(closes, splits) is None


def test_series_too_short_is_not_an_error():
    c = _series([100.0, 50.0])
    assert check_price_series_integrity(c, _splits(c, {1: 2.0})) is None


def test_non_positive_prices_do_not_raise():
    vals = list(_clean_walk(200).values)
    vals[50] = 0.0
    c = _series(vals)
    # Must not raise on log(0).
    check_price_series_integrity(c, _splits(c, {120: 2.0}))


# ------------------------------------------------- what gets withheld

def test_price_latest_is_deliberately_kept():
    """The defect is in relationships *between* dates.

    The most recent close is a single point, `info["currentPrice"]` takes
    precedence over it everywhere it is used, and withholding it would
    silently disable valuation metrics that have nothing to do with the bug.
    """
    assert "price_latest" not in PRICE_SERIES_DERIVED_FIELDS


def test_every_withheld_field_is_actually_history_derived():
    assert set(PRICE_SERIES_DERIVED_FIELDS) == {
        "price_1m_ago", "price_6m_ago", "price_12m_ago",
        "volatility_1y", "_daily_returns", "avg_daily_dollar_volume",
    }


def test_withholding_those_fields_blanks_momentum_and_risk():
    """The eight metrics that must go missing, and the one that must not."""
    from factor_engine import compute_metrics

    raw = {
        "Ticker": "TEST", "Company": "Test Co", "Sector": "Consumer Staples",
        "currentPrice": 48.73, "fiftyTwoWeekHigh": 50.17,
        "_price_series_rejected": "series mixes pre- and post-split prices",
    }
    for f in PRICE_SERIES_DERIVED_FIELDS:
        raw[f] = np.nan

    out = compute_metrics([raw], pd.Series(dtype=float), risk_free_rate=0.045)
    row = out.iloc[0]

    for m in ("return_12_1", "return_6m", "jensens_alpha",
              "volatility", "beta", "sharpe_ratio", "sortino_ratio",
              "max_drawdown_1y"):
        assert pd.isna(row[m]), f"{m} should be withheld, got {row[m]}"

    # proximity_52w_high comes from info["fiftyTwoWeekHigh"], a different
    # field that was correct throughout the MNST incident (50.17 against a
    # 48.73 price).  It must survive.
    assert not pd.isna(row["proximity_52w_high"])
    assert row["_price_series_rejected"]


# --------------------------------------- what it does to the composite

WITHHELD_METRICS = ("return_12_1", "return_6m", "jensens_alpha",
                    "volatility", "beta", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown_1y")


def _score_universe(cfg, blank_victim):
    """Score the same synthetic universe twice - once intact, once with the
    victim's eight price-derived metrics withheld.  Comparing one stock
    against itself is the only controlled comparison; comparing it to a
    different stock confounds the effect with that stock's own profile.
    """
    from tests.test_scoring import _make_df
    from factor_engine import (
        compute_category_scores,
        compute_composite,
        compute_sector_percentiles,
    )

    df = _make_df(100)
    victim = df.index[0]
    if blank_victim:
        for m in WITHHELD_METRICS:
            df.loc[victim, m] = np.nan

    df = compute_sector_percentiles(df)
    df = compute_category_scores(df, cfg)
    df = compute_composite(df, cfg)
    return df.loc[victim], df


@pytest.fixture
def cfg():
    import yaml
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        return yaml.safe_load(f)


def test_losing_those_metrics_does_not_crash_or_zero_the_composite(cfg):
    """Withholding must not be worse than the bug it replaces.

    `compute_category_scores` renormalises over the metrics a stock does
    have, so a missing category is neutral - the stock neither gains nor
    loses from it.  The failure mode this guards against is scoring the
    absence as a zero, which would be a fresh way of publishing fiction.
    """
    victim, df = _score_universe(cfg, blank_victim=True)

    assert pd.isna(victim["risk_score"]), "all 5 risk metrics gone -> no risk score"
    assert not pd.isna(victim["Composite"])
    assert victim["Composite"] > 0
    # Still ranked among its peers rather than dumped to the bottom.
    assert victim["Composite"] > df["Composite"].min()


def test_momentum_is_lost_too_because_proximity_carries_zero_weight(cfg):
    """A coherence finding, recorded here so it is not rediscovered.

    `proximity_52w_high` is the one momentum metric NOT derived from the
    13-month price series, so on paper momentum survives a rejected series
    at 1/4 strength.  In practice it does not: proximity is a Phase 11
    candidate carrying `metric_weights.momentum.proximity_52w_high: 0`, so
    the renormalised weight sum is zero and the category goes NaN.

    So a rejected price series costs a stock **two entire categories**, not
    one and a fraction.  If proximity is ever given a non-zero weight this
    test will fail, which is the right moment to revisit the claim.
    """
    assert cfg["metric_weights"]["momentum"]["proximity_52w_high"] == 0
    victim, _df = _score_universe(cfg, blank_victim=True)
    assert pd.isna(victim["momentum_score"])


def test_confidence_falls_for_the_affected_stock(cfg):
    """The dashboard already shows Composite_Confidence, and it is coverage
    driven - so the eight missing metrics surface to the user without any
    new UI.

    Measured on the synthetic universe: 80.0 intact -> 61.8 rejected.
    """
    intact, _ = _score_universe(cfg, blank_victim=False)
    rejected, _ = _score_universe(cfg, blank_victim=True)
    assert rejected["Composite_Confidence"] < intact["Composite_Confidence"]


def test_the_composite_shifts_only_slightly(cfg):
    """Renormalisation is meant to be neutral, not a penalty or a bonus.

    The two lost categories are replaced by the stock's average over the
    six it keeps, so the composite moves by whatever those two were worth
    relative to that average - a couple of points, not tens.
    """
    intact, _ = _score_universe(cfg, blank_victim=False)
    rejected, _ = _score_universe(cfg, blank_victim=True)
    assert abs(rejected["Composite"] - intact["Composite"]) < 5.0


# ------------------------------------------------- the incident itself

def test_the_mnst_arithmetic_that_motivated_this():
    """Numbers from `runs/83c9e2e2dd48/00_raw_fetch.parquet`, 2026-08-26.

    Pins the size of the error so nobody has to re-derive it.
    """
    p1m, p12 = 93.49, 62.30           # July close (unadjusted), 2025 close (adjusted)
    published = (p1m - p12) / p12
    corrected = (p1m / 2 - p12) / p12  # July close on the post-split scale

    assert published == pytest.approx(0.5006, abs=5e-4)   # 97th percentile
    assert corrected == pytest.approx(-0.2497, abs=5e-4)  # 3rd percentile
    assert published - corrected > 0.7
