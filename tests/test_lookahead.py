"""Tests for the look-ahead diagnostic.

No test here touches the network. ``lookahead`` is deliberately a pure module
over ``config.yaml`` and a snapshot frame, so everything with judgement in it is
testable from literals.

The tests that matter most are the ones pinning a mistake a reader would not
notice:

* **The bucket classification can go stale silently.** It is a statement about
  ``backtest.simulate_monthly_scores``'s ``dynamic_cols`` list and about
  ``factor_engine.METRIC_COLS``. If either moves and the buckets do not, the
  weight accounting is wrong and still sums to 100. Both are pinned against the
  real source rather than assumed.
* **Inverting a clamped metric invents data.** ``price_target_upside`` is
  clamped to ``metric_clamps``, so restating it by dividing the metric by the
  price ratio fabricates an analyst target for every name sitting on the bound.
  It is rebuilt from ``pt_mean`` instead, and a test drives the clamped case.
* **A net-cash company's enterprise value can go negative** when the share price
  is low enough, and ``factor_engine`` withholds every EV-based metric when EV is
  not positive. A restatement that quietly returns a negative EV/EBITDA would
  rank that name as the cheapest in its sector.
* **The module must not become a fix.** ``plan/backtest-v2.md`` forbids a
  half-fixed backtest, so a test asserts neither ``backtest.py`` nor
  ``run_screener.py`` imports it.
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lookahead as la
from factor_engine import METRIC_COLS, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# Classification: exhaustive, exclusive, and true to its two sources
# ---------------------------------------------------------------------------
def test_every_registry_metric_is_classified():
    """A metric added to METRIC_COLS must land in a bucket or the suite fails."""
    missing = sorted(set(METRIC_COLS) - la.ALL_CLASSIFIED)
    assert missing == [], (
        f"{missing} are in factor_engine.METRIC_COLS but in no lookahead bucket, "
        "so they are invisible to weight_buckets()"
    )


def test_no_bucket_invents_a_metric():
    extra = sorted(la.ALL_CLASSIFIED - set(METRIC_COLS))
    assert extra == [], f"{extra} are classified but not scored metrics"


def test_buckets_are_disjoint():
    seen = {}
    for name, metrics in la.BUCKETS.items():
        for m in metrics:
            assert m not in seen, f"{m} is in both {seen[m]} and {name}"
            seen[m] = name


def test_recomputed_matches_backtests_dynamic_cols():
    """The one bucket that is a claim about another module's source.

    ``simulate_monthly_scores`` keeps ``dynamic_cols`` as a local literal. If a
    future session adds ``max_drawdown_1y`` to it, that metric stops being
    held constant and this classification silently overstates look-ahead.
    """
    src = (ROOT / "backtest.py").read_text(encoding="utf-8")
    m = re.search(r"dynamic_cols\s*=\s*\[([^\]]*)\]", src)
    assert m, "could not find dynamic_cols in backtest.py"
    found = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert found == la.RECOMPUTED, (
        f"backtest.py recomputes {found} but lookahead.RECOMPUTED says "
        f"{la.RECOMPUTED}"
    )


def test_restated_here_is_a_subset_of_price_restatable():
    assert set(la.RESTATED_HERE) <= set(la.PRICE_RESTATABLE)


# ---------------------------------------------------------------------------
# Weight accounting
# ---------------------------------------------------------------------------
def test_weight_shares_account_for_the_whole_composite(cfg):
    b = la.weight_buckets(cfg)
    assert b["unclassified"] == 0
    # weight_buckets rounds each share to 4dp, so four of them can miss 100 by
    # up to 2e-4. The tolerance is that rounding and nothing else.
    assert b["recomputed"] + b["held_constant"] == pytest.approx(100.0, abs=1e-3)
    assert b["held_constant"] == pytest.approx(
        b["price_restatable"] + b["price_derived_held"] + b["needs_point_in_time"],
        abs=1e-3,
    )


def test_live_config_shares_are_the_ones_the_research_note_quotes(cfg):
    """Pinned so a reweight cannot leave the 2026-10-01 note stale.

    If this fails because weights legitimately changed, update
    ``research/2026-10-01-lookahead-price-component.md`` and
    ``plan/backtest-v2.md`` with the new figures, then update this test.
    """
    b = la.weight_buckets(cfg)
    # 2026-10-09: earnings_acceleration to 0 and its points to fy1_revision_3m /
    # price_target_upside / short_interest_ratio moved 0.35pp from "needs point-in-time" to
    # "price-restatable" (28.0 -> 28.35, 49.0 -> 48.65); plan/backtest-v2.md carries the update.
    assert b["recomputed"] == pytest.approx(16.893, abs=0.002)
    assert b["price_restatable"] == pytest.approx(28.35, abs=0.002)
    assert b["price_derived_held"] == pytest.approx(6.107, abs=0.002)
    assert b["needs_point_in_time"] == pytest.approx(48.65, abs=0.002)
    assert b["held_constant"] == pytest.approx(83.107, abs=0.002)


def test_shares_are_derived_from_config_not_written_down(cfg):
    """Halving momentum's weight must move the recomputed share, not a constant."""
    import copy

    base = la.weight_buckets(cfg)
    alt = copy.deepcopy(cfg)
    alt["factor_weights"]["momentum"] = 0
    moved = la.weight_buckets(alt)
    assert moved["recomputed"] < base["recomputed"]
    # Momentum's three weighted metrics split 75/25 between recomputed and held,
    # so dropping the category removes weight from both buckets.
    assert moved["price_derived_held"] < base["price_derived_held"]
    assert moved["recomputed"] + moved["held_constant"] == pytest.approx(100.0, abs=1e-3)


def test_zero_weight_metric_contributes_nothing(cfg):
    per = la.composite_metric_weights(cfg)
    assert per["pb_ratio"] == 0.0
    assert per["proximity_52w_high"] == 0.0
    assert per["fcf_yield"] > 0


def test_empty_config_does_not_explode():
    assert la.composite_metric_weights({}) == {}
    assert la.composite_metric_weights({"factor_weights": {"valuation": 0}}) == {}


def test_restatable_weight_gap_is_zero_today_and_warns_when_it_is_not(cfg):
    import copy

    assert la.restatable_weight_gap(cfg) == 0.0
    alt = copy.deepcopy(cfg)
    # Turning on the bank book-value metric makes a price-restatable metric
    # carry weight that this module cannot restate.
    alt["metric_weights"]["valuation"]["pb_ratio"] = 20
    assert la.restatable_weight_gap(alt) > 0


# ---------------------------------------------------------------------------
# Restatement fixtures
# ---------------------------------------------------------------------------
def _snapshot():
    """Three hand-built names with the properties that matter.

    * ``LEV`` carries net debt, so its EV exceeds its market cap.
    * ``CASH`` is net cash, so EV is below market cap and can go negative.
    * ``GAP`` has NaN fundamentals and no analyst target.
    """
    return pd.DataFrame(
        {
            "market_cap": [1000.0, 1000.0, 1000.0],
            "enterprise_value": [1500.0, 400.0, 1000.0],
            "price": [100.0, 50.0, 20.0],
            "pt_mean": [120.0, 40.0, np.nan],
            "earnings_yield": [0.05, 0.08, np.nan],
            "fcf_yield": [0.04, 0.10, np.nan],
            "ev_ebitda": [15.0, 4.0, np.nan],
            "ev_sales": [3.0, 0.8, np.nan],
            "size_log_mcap": [-math.log(1000.0), -math.log(1000.0), -math.log(1000.0)],
            "price_target_upside": [0.20, -0.20, np.nan],
            "roic": [0.18, 0.09, 0.11],
        },
        index=["LEV", "CASH", "GAP"],
    )


CLAMPS = {"price_target_upside": [-0.50, 1.0]}


# ---------------------------------------------------------------------------
# Restatement algebra
# ---------------------------------------------------------------------------
def test_ratio_of_one_is_the_identity():
    """The property the whole measurement rests on."""
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(1.0, index=snap.index), CLAMPS)
    for col in ("earnings_yield", "fcf_yield", "ev_ebitda", "ev_sales",
                "size_log_mcap", "market_cap", "enterprise_value", "price"):
        pd.testing.assert_series_equal(out[col], snap[col], check_names=False,
                                       rtol=1e-12, atol=1e-12)
    assert out.loc["LEV", "price_target_upside"] == pytest.approx(0.20, abs=1e-12)


def test_earnings_yield_scales_inversely_with_price():
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    # Half the price, same net income, so twice the earnings yield.
    assert out.loc["LEV", "earnings_yield"] == pytest.approx(0.10)
    assert out.loc["CASH", "earnings_yield"] == pytest.approx(0.16)


def test_enterprise_value_holds_its_non_equity_part():
    """EV is not scaled wholesale: debt and cash do not move with the price."""
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    # LEV: mc 1000 -> 500, non-equity part 500 held, so EV 1500 -> 1000.
    assert out.loc["LEV", "market_cap"] == pytest.approx(500.0)
    assert out.loc["LEV", "enterprise_value"] == pytest.approx(1000.0)
    assert out.loc["LEV", "ev_ebitda"] == pytest.approx(15.0 * 1000.0 / 1500.0)
    assert out.loc["LEV", "ev_sales"] == pytest.approx(3.0 * 1000.0 / 1500.0)
    assert out.loc["LEV", "fcf_yield"] == pytest.approx(0.04 * 1500.0 / 1000.0)


def test_scaling_ev_wholesale_would_give_a_different_answer():
    """Guards the design choice, not just the arithmetic.

    If a future edit replaced ``ev + mc*(r-1)`` with ``ev*r``, LEV's restated
    EV/EBITDA would be 7.5 rather than 10.0 — a third cheaper, from an
    assumption that its debt halved along with its share price.
    """
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    assert out.loc["LEV", "ev_ebitda"] == pytest.approx(10.0)
    assert out.loc["LEV", "ev_ebitda"] != pytest.approx(7.5)


def test_size_metric_is_a_log_shift():
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.25, index=snap.index), CLAMPS)
    assert out.loc["LEV", "size_log_mcap"] == pytest.approx(-math.log(250.0))


def test_net_cash_name_loses_ev_metrics_when_ev_turns_negative():
    """CASH has 600 of net cash; at a tenth of the price its EV is -500."""
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.1, index=snap.index), CLAMPS)
    assert out.loc["CASH", "enterprise_value"] == pytest.approx(-500.0)
    for col in ("ev_ebitda", "ev_sales", "fcf_yield"):
        assert pd.isna(out.loc["CASH", col]), f"{col} survived a negative EV"
    # Market-cap-based metrics are still defined — only EV is unusable.
    assert out.loc["CASH", "earnings_yield"] == pytest.approx(0.8)
    assert not pd.isna(out.loc["CASH", "size_log_mcap"])


def test_leveraged_name_keeps_ev_metrics_at_the_same_ratio():
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.1, index=snap.index), CLAMPS)
    assert out.loc["LEV", "enterprise_value"] == pytest.approx(600.0)
    assert out.loc["LEV", "ev_ebitda"] == pytest.approx(15.0 * 600.0 / 1500.0)


def test_missing_snapshot_metric_is_never_fabricated():
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    for col in ("earnings_yield", "fcf_yield", "ev_ebitda", "ev_sales",
                "price_target_upside"):
        assert pd.isna(out.loc["GAP", col]), f"GAP gained a {col} it never had"


def test_non_positive_or_missing_ratio_withholds_everything():
    snap = _snapshot()
    out = la.restate_at_price(
        snap, pd.Series([0.0, -1.0, np.nan], index=snap.index), CLAMPS)
    for t in snap.index:
        for col in ("earnings_yield", "fcf_yield", "ev_ebitda", "ev_sales",
                    "size_log_mcap", "price_target_upside", "market_cap",
                    "enterprise_value", "price"):
            assert pd.isna(out.loc[t, col]), f"{t}.{col} survived a bad ratio"


def test_round_trip_recovers_the_snapshot():
    """Restate at r, then at 1/r, and the algebra must return where it started."""
    snap = _snapshot()
    r = pd.Series([0.4, 2.5, 1.3], index=snap.index)
    once = la.restate_at_price(snap, r, CLAMPS)
    back = la.restate_at_price(once, 1.0 / r, CLAMPS)
    for col in ("earnings_yield", "fcf_yield", "ev_ebitda", "ev_sales",
                "size_log_mcap", "market_cap", "enterprise_value", "price"):
        pd.testing.assert_series_equal(back[col], snap[col], check_names=False,
                                       rtol=1e-9, atol=1e-9)


def test_cheaper_price_means_cheaper_valuation_metrics():
    """Direction check: the metrics must move the way a reader expects."""
    snap = _snapshot()
    low = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    high = la.restate_at_price(snap, pd.Series(2.0, index=snap.index), CLAMPS)
    assert low.loc["LEV", "earnings_yield"] > snap.loc["LEV", "earnings_yield"]
    assert high.loc["LEV", "earnings_yield"] < snap.loc["LEV", "earnings_yield"]
    assert low.loc["LEV", "ev_ebitda"] < snap.loc["LEV", "ev_ebitda"]
    assert high.loc["LEV", "ev_ebitda"] > snap.loc["LEV", "ev_ebitda"]
    # size_log_mcap is -log(mc): a smaller company scores higher.
    assert low.loc["LEV", "size_log_mcap"] > snap.loc["LEV", "size_log_mcap"]


def test_input_frame_is_not_mutated():
    snap = _snapshot()
    before = snap.copy(deep=True)
    la.restate_at_price(snap, pd.Series(0.3, index=snap.index), CLAMPS)
    pd.testing.assert_frame_equal(snap, before)


@pytest.mark.parametrize("dropped", ["earnings_yield", "fcf_yield", "ev_ebitda",
                                     "ev_sales", "size_log_mcap",
                                     "price_target_upside", "pt_mean"])
def test_an_absent_optional_column_is_skipped_not_an_error(dropped):
    """A snapshot that does not carry a metric cannot restate it.

    That is a missing column, not a failure — and the three columns in
    ``REQUIRED_SNAPSHOT_COLS`` are the only ones whose absence is refused.
    """
    snap = _snapshot().drop(columns=[dropped])
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    assert dropped not in out.columns
    # Everything else still restates.
    if dropped != "earnings_yield":
        assert out.loc["LEV", "earnings_yield"] == pytest.approx(0.10)
    if dropped == "pt_mean":
        assert pd.isna(out.loc["LEV", "price_target_upside"])


def test_missing_required_column_is_refused_loudly():
    snap = _snapshot().drop(columns=["enterprise_value"])
    with pytest.raises(ValueError) as exc:
        la.restate_at_price(snap, pd.Series(1.0, index=snap.index), CLAMPS)
    assert "enterprise_value" in str(exc.value)


# ---------------------------------------------------------------------------
# price_target_upside: the clamped metric
# ---------------------------------------------------------------------------
def test_upside_is_rebuilt_from_the_target_not_inverted():
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    # LEV: target 120 against a price of 50 is +140%, clamped to +100%.
    assert out.loc["LEV", "price_target_upside"] == pytest.approx(1.0)
    # CASH: target 40 against a price of 25 is +60%.
    assert out.loc["CASH", "price_target_upside"] == pytest.approx(0.6)


def test_inverting_the_clamped_metric_would_have_been_wrong():
    """The concrete reason the implementation reads ``pt_mean``.

    A target of 125 against a price of 100 is a snapshot upside of +25%. At twice
    the price the honest upside is ``125/200 - 1 = -37.5%``. Dividing the metric by
    the ratio instead gives ``0.25/2 = +12.5%`` — the wrong sign, and the error is
    unbounded for any name whose snapshot upside sat on a clamp bound, because
    there the metric no longer identifies the target at all.
    """
    snap = _snapshot().loc[["LEV"]].copy()
    snap.loc["LEV", "pt_mean"] = 125.0
    snap.loc["LEV", "price"] = 100.0
    snap.loc["LEV", "price_target_upside"] = 0.25
    out = la.restate_at_price(snap, pd.Series(1.0, index=snap.index), CLAMPS)
    assert out.loc["LEV", "price_target_upside"] == pytest.approx(0.25)
    # Now at twice the price the honest upside is -37.5%, not 0.25/2.
    out2 = la.restate_at_price(snap, pd.Series(2.0, index=snap.index), CLAMPS)
    assert out2.loc["LEV", "price_target_upside"] == pytest.approx(-0.375)
    assert out2.loc["LEV", "price_target_upside"] != pytest.approx(0.125)


def test_upside_respects_the_configured_clamp(cfg):
    snap = _snapshot()
    lo, hi = cfg["metric_clamps"]["price_target_upside"]
    out = la.restate_at_price(snap, pd.Series(0.01, index=snap.index),
                             cfg["metric_clamps"])
    assert out.loc["LEV", "price_target_upside"] == pytest.approx(float(hi))
    out2 = la.restate_at_price(snap, pd.Series(100.0, index=snap.index),
                              cfg["metric_clamps"])
    assert out2.loc["LEV", "price_target_upside"] == pytest.approx(float(lo))


def test_upside_defaults_to_the_documented_clamp_when_none_given():
    snap = _snapshot()
    out = la.restate_at_price(snap, pd.Series(0.01, index=snap.index), None)
    assert out.loc["LEV", "price_target_upside"] == pytest.approx(1.0)


def test_upside_is_nan_without_a_target():
    snap = _snapshot()
    snap.loc["LEV", "pt_mean"] = np.nan
    out = la.restate_at_price(snap, pd.Series(0.5, index=snap.index), CLAMPS)
    assert pd.isna(out.loc["LEV", "price_target_upside"])


# ---------------------------------------------------------------------------
# The module must stay a diagnostic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", ["backtest.py", "run_screener.py",
                                    "factor_engine.py", "generate_dashboard.py"])
def test_production_modules_do_not_import_the_diagnostic(module):
    """``plan/backtest-v2.md`` forbids shipping a half-fixed backtest.

    Restating a valuation ratio at a historical price while leaving its
    fundamental at today's value removes some look-ahead and leaves the rest,
    which is exactly the false confidence the bench period exists to prevent.
    If a session genuinely wants to wire this in, it needs a changelog entry and
    a reason, not a passing test suite.
    """
    src = (ROOT / module).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import lookahead|from lookahead)", src,
                         re.MULTILINE), f"{module} imports lookahead"


def test_diagnostic_does_not_depend_on_the_scoring_engine():
    """One-way dependency: a change to factor_engine cannot alter the buckets."""
    src = (ROOT / "lookahead.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import factor_engine|from factor_engine)", src,
                         re.MULTILINE)


def test_measurement_does_not_append_to_the_tracked_vol_history():
    """A documented contamination: the first version of the script did.

    ``factor_engine.adjust_momentum_weight`` appends a row to
    ``<root>/factor_vol_history.csv`` on every call, and that file is tracked and
    feeds the live momentum-weight regime decision. The measurement calls it once
    to show its reconstruction is faithful, and the first version passed the repo
    root — duplicating the day's row twice, which is the same evidence-inflation
    shape as ``CLAUDE.md`` priority 0.6 arriving from a measurement instead of a
    run. It must use a scratch directory.
    """
    script = (ROOT / "research" / "measurements"
              / "2026-10-01-lookahead-price-component.py").read_text(encoding="utf-8")
    assert "adjust_momentum_weight(df, cfg, str(ROOT))" not in script, (
        "the measurement would append a duplicate row to the tracked "
        "factor_vol_history.csv"
    )
    assert "mkdtemp" in script and "adjust_momentum_weight(df, cfg, str(scratch))" in script


def test_measurement_script_exists_and_is_named_for_its_date():
    """The published numbers must stay reproducible from the repo."""
    script = ROOT / "research" / "measurements" / \
        "2026-10-01-lookahead-price-component.py"
    assert script.exists()
    out = script.with_suffix(".json")
    assert out.exists(), "the measurement's output JSON is the reproducibility path"
