"""The research engine's registry cannot hold a test it cannot run.

Three sessions have now been asked to register an experiment the engine is unable to measure, and
each refusal rested on the same fact rediscovered by hand:

* **2026-10-23** refused three registrations from
  `research/2026-10-22-free-data-that-adds-evidence.md`. The decisive one put weight *on*
  ``short_interest_ratio``: ``v2_config`` zeroes the metric and the record builder NaNs its inputs,
  so the entry "would have run clean on an all-NaN column and published a null caused by the
  missing panel."
* **2026-11-05** refused `research/2026-11-03-short-interest-days-to-cover.md` section 5, which
  asked for the opposite arm - *drop* ``short_interest_ratio`` from Revisions. Measured over all
  176 months on file (`research/measurements/2026-11-05-short-interest-registration.py`): the arm
  ranks **0 of 67,113 stock-months** differently from the baseline, because v2's baseline already
  carries no Revisions category at all. A row like that reports a difference of exactly zero *by
  construction*, which reads as "removing it costs nothing" and is in fact "this test was not run".

Both are refusals a reader had to derive. This module derives them instead, so a fourth proposal
of the same shape fails the suite at registration rather than after a ~45-minute run - and so that
a *reported* null is always a measurement rather than an identity.

The three conditions, all read from the configuration every registered experiment actually starts
from (``X.base_config()`` = ``backtest_v2.v2_config(config.yaml)``):

1. the transform changes that configuration at all (``baseline`` and ``whole_market`` excepted, by
   name and with their reasons);
2. it puts no weight on a metric in ``backtest_v2.UNMEASURABLE_METRICS``, whose column the record
   builder fills with NaN;
3. it leaves at least one metric v2 *can* measure carrying weight, so the arm ranks on something.

Each guard is checked against the entries it exists to refuse, because a guard that passes on the
thing it was built to catch is decoration (`OWNER_FOCUS.md` item 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_v2 as v2      # noqa: E402
import v2_experiments as X    # noqa: E402

#: The two kinds whose configuration is *meant* to equal the baseline's, with the reason each is
#: not a vacuous arm. Anything else that leaves the configuration untouched is one.
CONFIG_IDENTICAL_BY_DESIGN = {
    "baseline": "it is the baseline",
    # `rank_all(..., whole_market=True)` scores every stock against the whole market instead of
    # its sector; the weighting is deliberately unchanged (v2_experiments.apply_transform).
    "whole_market": "it changes the peer group in the ranking, not the weighting",
}


def _measurable_weight(cfg: dict) -> dict[str, float]:
    """Each metric's share of the composite under ``cfg``, counting only metrics v2 can measure.

    A metric counts for nothing if its category carries no weight, and nothing if the category's
    weights sum to zero - which is how ``v2_config`` leaves Revisions.
    """
    fw = cfg["factor_weights"]
    total = sum(float(w) for w in fw.values())
    out: dict[str, float] = {}
    if total <= 0:
        return out
    for cat, metrics in (cfg.get("metric_weights") or {}).items():
        cat_w = float(fw.get(cat, 0.0))
        within = sum(float(w) for w in metrics.values())
        if cat_w <= 0 or within <= 0:
            continue
        for m, w in metrics.items():
            if float(w) > 0:
                out[m] = out.get(m, 0.0) + cat_w / total * float(w) / within
    return out


def unmeasurable_weight(cfg: dict) -> dict[str, float]:
    """The share of the composite ``cfg`` puts on columns the v2 record leaves NaN."""
    return {m: w for m, w in _measurable_weight(cfg).items() if m in v2.UNMEASURABLE_METRICS}


def measurable_weight(cfg: dict) -> dict[str, float]:
    """The share ``cfg`` puts on columns the v2 record actually carries."""
    return {m: w for m, w in _measurable_weight(cfg).items() if m not in v2.UNMEASURABLE_METRICS}


# The arms three sessions were asked to register, written the way the registry's own precedents
# write them. Each is named by the guard that must refuse it.
REFUSED = {
    # 2026-11-05, research/2026-11-03-short-interest-days-to-cover.md section 5. The faithful
    # expression, copied from `drop-revenue-growth`, the registry's own metric-level precedent.
    "drop-short-interest-as-config-set": (
        {"kind": "config", "set": {"metric_weights.revisions.short_interest_ratio": 0}},
        "unchanged"),
    # the same proposal read as "the category without its one non-analyst leg"
    "drop-revisions": ({"kind": "drop_category", "category": "revisions"}, "unchanged"),
    # 2026-10-23, research/2026-10-22-free-data-that-adds-evidence.md: weight *on* the metric
    "score-short-interest": (
        {"kind": "metrics", "metrics": {"revisions": {"short_interest_ratio": 100}}},
        "unmeasurable"),
    # a category v2 cannot measure, scored alone: the configuration differs from the baseline and
    # still ranks on nothing. The registry has an `only-X` for each of the seven categories v2
    # weights and deliberately none for Revisions; this is the entry that would break that.
    "only-revisions": ({"kind": "only_category", "category": "revisions"}, "nothing-measurable"),
}


@pytest.fixture(scope="module")
def base() -> dict:
    return X.base_config()


def test_the_baseline_v2_scores_cannot_see_revisions_at_all(base):
    """The fact both refusals rest on, stated once where a session will find it."""
    assert float(base["factor_weights"]["revisions"]) == 0.0
    assert [c for c, w in base["factor_weights"].items() if not float(w)] == ["revisions"]
    assert all(float(w) == 0.0 for w in base["metric_weights"]["revisions"].values())
    assert "short_interest_ratio" in v2.UNMEASURABLE_METRICS
    # and no registered entry names the category, which is what the registry already knew
    assert not [e["id"] for e in X.load_registry()
                if e["transform"].get("category") == "revisions"]


def test_no_registered_transform_leaves_the_baseline_configuration_unchanged(base):
    for e in X.load_registry():
        kind = e["transform"]["kind"]
        identical = X.apply_transform(base, e["transform"]) == base
        if kind in CONFIG_IDENTICAL_BY_DESIGN:
            continue
        assert not identical, (
            f"{e['id']}: its transform leaves v2's configuration exactly as the baseline, so the "
            "row would report a difference of zero by construction. Either it targets something "
            "v2 already scores at 0 (see UNMEASURABLE_METRICS) or the transform is a no-op.")


def test_the_two_kinds_whose_configuration_equals_the_baseline_are_the_two_named_here(base):
    """So a third no-op kind cannot hide behind the exemption above."""
    identical = {e["transform"]["kind"] for e in X.load_registry()
                 if X.apply_transform(base, e["transform"]) == base}
    assert identical <= set(CONFIG_IDENTICAL_BY_DESIGN), identical
    for kind, reason in CONFIG_IDENTICAL_BY_DESIGN.items():
        assert reason


def test_no_registered_transform_weights_a_metric_the_record_leaves_nan(base):
    for e in X.load_registry():
        cfg = X.apply_transform(base, e["transform"])
        on_nan = unmeasurable_weight(cfg)
        assert not on_nan, (
            f"{e['id']}: weight on {sorted(on_nan)}, whose columns pit_record fills with NaN "
            "(no_free_history). The arm would run clean and publish a null caused by the missing "
            "panel, not by the weighting.")


def test_every_registered_transform_leaves_something_v2_can_measure(base):
    for e in X.load_registry():
        cfg = X.apply_transform(base, e["transform"])
        assert measurable_weight(cfg), f"{e['id']}: ranks on nothing v2 carries"


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_each_refused_registration_is_refused_by_a_guard_above(base, name):
    """The guards fire on the entries they exist for - checked, not assumed."""
    transform, why = REFUSED[name]
    cfg = X.apply_transform(base, transform)
    if why == "unchanged":
        assert cfg == base and transform["kind"] not in CONFIG_IDENTICAL_BY_DESIGN
    elif why == "unmeasurable":
        assert unmeasurable_weight(cfg)
    elif why == "nothing-measurable":
        assert cfg != base and not measurable_weight(cfg)
    else:                                                    # pragma: no cover - typo guard
        raise AssertionError(why)


def test_a_real_metric_level_drop_still_passes_every_guard(base):
    """The guards refuse what cannot be measured, not metric-level experiments as such.

    ``drop-revenue-growth`` is registered and is exactly the shape today's proposal asked for -
    one metric's weight set to 0 - and it measures something, because v2 carries revenue growth.
    """
    tr = {"kind": "config", "set": {"metric_weights.growth.revenue_growth": 0}}
    cfg = X.apply_transform(base, tr)
    assert cfg != base
    assert not unmeasurable_weight(cfg)
    assert measurable_weight(cfg)
    assert "revenue_growth" not in measurable_weight(cfg)
    assert tr in [e["transform"] for e in X.load_registry()]


def test_the_measurable_share_is_the_composite_and_sums_to_one(base):
    """``_measurable_weight`` is the arithmetic the guards read; pin it rather than trust it."""
    shares = _measurable_weight(base)
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    # Revisions contributes nothing, and the seven categories v2 weights contribute everything
    assert not set(shares) & set(base["metric_weights"]["revisions"])
    # a category at zero weight contributes nothing even though its metrics carry within-weights
    only_value = X.apply_transform(base, {"kind": "only_category", "category": "valuation"})
    assert set(_measurable_weight(only_value)) == {
        m for m, w in base["metric_weights"]["valuation"].items() if float(w) > 0}
