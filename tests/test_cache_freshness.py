"""Regression tests for the factor_scores cache freshness decision.

Why this file exists
--------------------
The data loop runs at 02:00 every weekday and is the *only* source of
evidence the improvement engine ever gets. On 2026-08-13 the 02:00 run
finished in 4.5 seconds, never touched the network, and was correctly
discarded by ``scripts/check_run_health.py`` for having no fetch evidence.
That was the third documented instance (2026-08-07, 2026-08-10, 2026-08-13).

Root cause, in two parts:

1. ``run_factor_engine`` bounded the *whole* ``factor_scores`` cache by
   ``fundamental_data_refresh_days`` (7), even though that cache holds the
   fully-scored dataset - momentum, volatility, 52-week proximity and
   analyst price-target upside are all price-derived and go stale daily.
   ``price_data_refresh_days`` (1) existed in config and was never consulted
   on this path.

2. The warm-start path returns *before* ``write_scores_parquet``, so a
   warm-started run does not lay down a new cache file. The cache date stays
   pinned to the last real fetch, so a single fetch suppressed the next
   seven days of fetches - roughly one real observation per eight days.

The rule these tests pin down: ``<tier>_refresh_days: N`` means the cache is
reusable for N calendar days *starting with the day it was written*. So
``price_data_refresh_days: 1`` means "refetch unless the cache is from
today", which is what a daily loop needs.
"""

from datetime import datetime

import pytest

from factor_engine import (
    cache_age_days,
    cache_is_usable,
    factor_scores_cache_max_age_days,
)


class TestFactorScoresCacheTier:
    """The factor_scores cache must be bounded by its fastest-moving contents."""

    def test_uses_price_tier_not_fundamental_tier(self):
        cfg = {"price_data_refresh_days": 1, "fundamental_data_refresh_days": 7}
        assert factor_scores_cache_max_age_days(cfg) == 1

    def test_tracks_the_price_tier_if_it_is_retuned(self):
        cfg = {"price_data_refresh_days": 3, "fundamental_data_refresh_days": 7}
        assert factor_scores_cache_max_age_days(cfg) == 3

    def test_defaults_to_daily_when_unset(self):
        assert factor_scores_cache_max_age_days({}) == 1

    def test_never_exceeds_the_fundamental_tier(self):
        """A cache is only as fresh as its fastest-moving contents."""
        cfg = {"price_data_refresh_days": 30, "fundamental_data_refresh_days": 7}
        assert factor_scores_cache_max_age_days(cfg) == 7


class TestCacheAgeDays:
    """Cache dates come from the filename, so they are midnight-anchored."""

    def test_same_calendar_day_is_age_zero(self):
        written = datetime(2026, 8, 13)  # filename 20260813
        assert cache_age_days(written, now=datetime(2026, 8, 13, 2, 0, 1)) == 0
        assert cache_age_days(written, now=datetime(2026, 8, 13, 23, 59)) == 0

    def test_previous_calendar_day_is_age_one(self):
        written = datetime(2026, 8, 12)
        # 02:00 the next morning is only 26 wall-clock hours later, but it is
        # a different trading day and a new close exists.
        assert cache_age_days(written, now=datetime(2026, 8, 13, 2, 0, 1)) == 1

    def test_counts_calendar_days_not_elapsed_hours(self):
        written = datetime(2026, 8, 6)
        assert cache_age_days(written, now=datetime(2026, 8, 13, 2, 0)) == 7


class TestCacheIsUsable:
    """The rule: reusable for N calendar days starting the day it was written."""

    def test_todays_cache_is_reusable_at_daily_refresh(self):
        written = datetime(2026, 8, 13)
        assert cache_is_usable(written, 1, now=datetime(2026, 8, 13, 2, 0)) is True

    def test_yesterdays_cache_is_stale_at_daily_refresh(self):
        """THE 2026-08-13 BUG. This is the assertion that was false before.

        A cache dated 2026-08-12 was reused by the 02:00 run on 2026-08-13,
        which then had no fetch evidence and was discarded.
        """
        written = datetime(2026, 8, 12)
        assert cache_is_usable(written, 1, now=datetime(2026, 8, 13, 2, 0, 1)) is False

    def test_the_old_fundamental_bound_would_have_reused_it(self):
        """Documents the old behaviour so the regression is unmistakable.

        Under the old rule (fundamental tier, 7 days, ``<=``) a cache up to
        eight calendar days old was reused.
        """
        written = datetime(2026, 8, 12)
        old_age = cache_age_days(written, now=datetime(2026, 8, 13, 2, 0, 1))
        assert old_age <= 7, "the old bound reused this cache"
        assert cache_is_usable(written, 1, now=datetime(2026, 8, 13, 2, 0, 1)) is False

    @pytest.mark.parametrize("age_days,max_age,expected", [
        (0, 1, True),
        (1, 1, False),
        (0, 7, True),
        (6, 7, True),
        (7, 7, False),
        (8, 7, False),
    ])
    def test_boundary_is_exclusive(self, age_days, max_age, expected):
        now = datetime(2026, 8, 13, 2, 0)
        written = datetime(2026, 8, 13 - age_days)
        assert cache_is_usable(written, max_age, now=now) is expected

    def test_zero_max_age_never_reuses(self):
        written = datetime(2026, 8, 13)
        assert cache_is_usable(written, 0, now=datetime(2026, 8, 13, 2, 0)) is False

    def test_missing_cache_date_is_not_usable(self):
        assert cache_is_usable(None, 7, now=datetime(2026, 8, 13)) is False


class TestDailyLoopActuallyFetches:
    """End-to-end property: consecutive daily runs must each fetch.

    A warm-started run returns before ``write_scores_parquet``, so it never
    lays down a new cache file. If day 2 warm-starts off day 1's cache, the
    cache date never advances and every subsequent run warm-starts too - the
    failure that produced one real observation per eight days.
    """

    def test_eight_consecutive_daily_runs_all_fetch(self):
        cfg = {"price_data_refresh_days": 1, "fundamental_data_refresh_days": 7}
        max_age = factor_scores_cache_max_age_days(cfg)

        cache_date = None  # cold start
        fetches = 0
        for day in range(6, 14):
            now = datetime(2026, 8, day, 2, 0, 1)
            if cache_date is not None and cache_is_usable(cache_date, max_age, now=now):
                continue  # warm start: no fetch, and cache_date does NOT advance
            fetches += 1
            cache_date = datetime(2026, 8, day)

        assert fetches == 8, (
            f"expected a real fetch on each of 8 consecutive daily runs, got {fetches}"
        )

    def test_old_bound_would_have_fetched_only_once(self):
        """The measured old behaviour: 1 fetch per 8 daily runs."""
        cache_date = None
        fetches = 0
        for day in range(6, 14):
            now = datetime(2026, 8, day, 2, 0, 1)
            if cache_date is not None and cache_age_days(cache_date, now=now) <= 7:
                continue
            fetches += 1
            cache_date = datetime(2026, 8, day)

        assert fetches == 1, "old rule fetched once and then coasted for a week"

    def test_price_driven_metrics_claim_is_still_true(self):
        """Pins the evidence in the 2026-08-13 METHODOLOGY_CHANGELOG entry.

        That entry justifies bounding the factor_scores cache by the price
        tier on the grounds that 18 of the 44 scored metrics move with the
        daily close, across five of the eight categories. If someone adds or
        renames one of these, the changelog claim silently becomes false -
        so assert it here rather than trusting prose.
        """
        from factor_engine import METRIC_COLS

        price_driven = {
            # Valuation - every one has price or market cap in it
            "ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales",
            "pb_ratio", "peg_ratio", "dividend_yield",
            # Momentum
            "return_12_1", "return_6m", "proximity_52w_high",
            # Risk
            "volatility", "beta", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_1y", "jensens_alpha",
            # Revisions
            "price_target_upside",
            # Size
            "size_log_mcap",
        }

        missing = sorted(price_driven - set(METRIC_COLS))
        assert not missing, (
            f"METRIC_COLS no longer contains {missing}; the 2026-08-13 "
            "changelog entry needs updating"
        )
        assert len(price_driven) == 18
        assert len(METRIC_COLS) == 44

    def test_same_day_rerun_still_warm_starts(self):
        """A manual re-run on the same day has no new close to fetch.

        This is deliberate and unchanged: the 2026-08-11 07:52 re-run warm
        started and the health gate rejected it, which was the right call.
        """
        cfg = {"price_data_refresh_days": 1}
        max_age = factor_scores_cache_max_age_days(cfg)
        written = datetime(2026, 8, 13)
        assert cache_is_usable(written, max_age, now=datetime(2026, 8, 13, 7, 52)) is True
