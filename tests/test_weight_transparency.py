"""The weights the dashboard shows must be the weights it actually used.

The stock drilldown prints its own arithmetic to the reader:

    Score: 65.3/100   [Average]   x 13% = 9.76 pts

That line is the tool's central teaching claim - it is how a student is meant
to learn what a multi-factor composite *is*. On 2026-08-28 it was checked
against the live published payload for the first time and it did not add up:
65.3 x 13% is 8.49, not 9.76.

**The cause.** ``factor_engine.adjust_momentum_weight()`` scales the momentum
weight with the market's volatility regime and returns a *deep copy* of the
config. ``run_factor_engine`` rebinds its local ``cfg`` to that copy, so the
adjustment never reaches ``main()``, and ``ctx.save_effective_weights(cfg)`` -
called from ``main()`` - recorded the configured weights instead of the ones
the composite was built from. The revisions/investment auto-disables happen to
mutate the shared dict, which is why only momentum and valuation were wrong.

**The measured damage**, taken from the payload that was live on the public
site that morning: the run scored with momentum at 14.95% and valuation at
20.05% (a LOW VOL regime: 13 x 1.15, funded out of valuation) while publishing
13% and 22%. **Momentum arithmetic was wrong for 498 of 502 stocks and
valuation for 501 of 502.**

A second, smaller gap was found alongside it. When a category cannot be scored
for a stock, ``compute_factor_contributions`` drops it and renormalises the
survivors, so 11 more stocks disagreed on the *other* categories too - MNST,
whose price series the split-integrity check rejects, showed
"22% weight -> 20.64 pts" against a quality score of 70.43.

These tests pin both, and pin the reconciliation guard that stops a third
variant shipping silently. See ``METHODOLOGY_CHANGELOG.md`` 2026-08-28.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402
from run_context import RunContext  # noqa: E402

CATEGORIES = ["valuation", "quality", "growth", "momentum",
              "risk", "revisions", "size", "investment"]

# The configured defaults, as published in SCREENER_OVERVIEW.md.
BASE = {"valuation": 22, "quality": 22, "growth": 13, "momentum": 13,
        "risk": 10, "revisions": 10, "size": 5, "investment": 5}

# What a LOW VOL regime actually scored with on 2026-08-28: momentum x 1.15,
# the 1.95pp funded out of valuation.
REGIME = dict(BASE, momentum=14.95, valuation=20.05)


def _frame(n=60, withhold=None, weights=None):
    """A scored universe whose contributions are built the way the engine does.

    ``withhold`` names categories to blank out for the first ticker, which is
    how a real run represents "this stock could not be scored on momentum".
    """
    weights = weights or REGIME
    withhold = withhold or []
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "Ticker": [f"T{i:03d}" for i in range(n)],
        "Company": [f"Company {i}" for i in range(n)],
        "Sector": ["Information Technology"] * n,
        "Rank": list(range(1, n + 1)),
        "Value_Trap_Flag": [False] * n,
        "Growth_Trap_Flag": [False] * n,
    })
    for cat in CATEGORIES:
        df[cat + "_score"] = rng.uniform(5, 95, n).round(4)
    for cat in withhold:
        df.loc[0, cat + "_score"] = np.nan

    # Mirror factor_engine.compute_factor_contributions exactly.
    for cat in CATEGORIES:
        col, out = cat + "_score", cat + "_contrib"
        vals = []
        for _, row in df.iterrows():
            live = [c for c in CATEGORIES
                    if weights.get(c, 0) > 0 and pd.notna(row[c + "_score"])]
            total = sum(weights[c] for c in live)
            if cat not in live or total <= 0:
                vals.append(0.0)
            else:
                vals.append(round(row[col] * weights[cat] / total, 2))
        df[out] = vals
    df["Composite"] = sum(df[c + "_contrib"] for c in CATEGORIES).round(2)
    return df


def _payload(df, weights):
    return json.loads(g.prepare_dashboard_data({
        "df": df, "meta": {"run_date": "2026-08-28"},
        "weights": weights, "sens_df": None, "corr_df": None, "cfg": {},
    }))


# =========================================================================
# 1. The reconciliation guard - the thing that would have caught this
# =========================================================================
class TestReconciliation:
    """`_reconcile_factor_weights` refuses to publish weights that don't add up."""

    def test_the_exact_live_bug_is_caught_and_corrected(self):
        """Scores built at 14.95%, weights recorded as 13% - the 2026-08-28 state."""
        df = _frame(weights=REGIME)
        out = g._reconcile_factor_weights(df, {"factor_weights": dict(BASE)})

        assert out["factor_weights"]["momentum"] == pytest.approx(14.95, abs=0.02)
        assert out["factor_weights"]["valuation"] == pytest.approx(20.05, abs=0.02)
        assert out["factor_weights_derived"] is True
        # The configured weights are kept so the page can explain the gap.
        assert out["base_factor_weights"]["momentum"] == 13

    def test_correct_weights_pass_through_untouched(self):
        """No false positives: a truthful run must not be second-guessed."""
        df = _frame(weights=REGIME)
        out = g._reconcile_factor_weights(df, {"factor_weights": dict(REGIME)})
        assert out["factor_weights"] == REGIME
        assert "factor_weights_derived" not in out

    def test_a_normal_regime_run_reconciles(self):
        """When the regime is NORMAL the config weights are already correct."""
        df = _frame(weights=BASE)
        out = g._reconcile_factor_weights(df, {"factor_weights": dict(BASE)})
        assert out["factor_weights"] == BASE
        assert "factor_weights_derived" not in out

    def test_derivation_recovers_weights_from_scores_alone(self):
        derived = g._derive_factor_weights(_frame(weights=REGIME))
        for cat in CATEGORIES:
            assert derived[cat] == pytest.approx(REGIME[cat], abs=0.02)
        assert sum(derived.values()) == pytest.approx(100.0, abs=0.05)

    def test_derivation_declines_rather_than_guesses_on_a_tiny_universe(self):
        """Better to publish the recorded weights than a number from 3 rows."""
        assert g._derive_factor_weights(_frame(n=5)) is None

    def test_derivation_declines_when_contrib_columns_are_absent(self):
        df = _frame(n=40).drop(columns=["momentum_contrib"])
        assert g._derive_factor_weights(df) is None

    def test_reconcile_is_a_no_op_when_it_cannot_verify(self):
        """An unverifiable run keeps its recorded weights - it does not blank them."""
        weights = {"factor_weights": dict(BASE)}
        out = g._reconcile_factor_weights(_frame(n=5), weights)
        assert out["factor_weights"] == BASE

    def test_reconcile_survives_an_empty_frame(self):
        out = g._reconcile_factor_weights(pd.DataFrame(), {"factor_weights": dict(BASE)})
        assert out["factor_weights"] == BASE

    def test_reconcile_survives_missing_weights(self):
        assert g._reconcile_factor_weights(_frame(), {}) == {}


# =========================================================================
# 2. The published payload must reconcile, end to end
# =========================================================================
class TestPublishedPayloadAddsUp:
    """score x published weight = published contribution, for every stock."""

    def _check(self, payload, tol=0.011):
        fw = payload["weights"]["factor_weights"]
        bad = []
        for ticker, s in payload["stock_detail"].items():
            present = [c for c in CATEGORIES if s["cat_scores"].get(c) is not None]
            eff = g._effective_weight_row(fw, present)
            for cat in CATEGORIES:
                score = s["cat_scores"].get(cat)
                if score is None:
                    assert (s["contrib"].get(cat) or 0) == 0, (
                        f"{ticker}/{cat}: no score but non-zero contribution")
                    continue
                predicted = score * eff[cat] / 100.0
                actual = s["contrib"].get(cat) or 0
                if abs(predicted - actual) > tol:
                    bad.append((ticker, cat, round(predicted, 3), actual))
        return bad

    def test_it_adds_up_when_the_recorded_weights_were_wrong(self):
        """The live 2026-08-28 case: reconciliation makes the page truthful."""
        payload = _payload(_frame(weights=REGIME), {"factor_weights": dict(BASE)})
        assert self._check(payload) == []

    def test_it_adds_up_when_the_recorded_weights_were_right(self):
        payload = _payload(_frame(weights=REGIME), {"factor_weights": dict(REGIME)})
        assert self._check(payload) == []

    def test_it_adds_up_for_a_stock_with_categories_withheld(self):
        """MNST's case: a rejected price series removes momentum and risk."""
        df = _frame(withhold=["momentum", "risk"], weights=REGIME)
        payload = _payload(df, {"factor_weights": dict(REGIME)})
        first = payload["stock_detail"]["T000"]
        assert first["cat_scores"]["momentum"] is None
        assert first["cat_scores"]["risk"] is None
        assert self._check(payload) == []

    def test_the_withheld_stocks_surviving_weights_are_scaled_up_not_left_alone(self):
        """The renormalisation is real, not cosmetic - quality must exceed 22%."""
        fw = dict(REGIME)
        eff = g._effective_weight_row(fw, [c for c in CATEGORIES
                                           if c not in ("momentum", "risk")])
        assert eff["quality"] > 22.0
        assert eff["momentum"] == 0.0
        assert sum(eff.values()) == pytest.approx(100.0, abs=1e-6)

    def test_a_fully_scored_stock_is_not_renormalised(self):
        eff = g._effective_weight_row(dict(REGIME), CATEGORIES)
        for cat in CATEGORIES:
            assert eff[cat] == pytest.approx(REGIME[cat], abs=1e-6)

    def test_contributions_sum_to_the_composite(self):
        payload = _payload(_frame(weights=REGIME), {"factor_weights": dict(REGIME)})
        for ticker, s in payload["stock_detail"].items():
            total = sum(v for v in s["contrib"].values() if v)
            assert total == pytest.approx(s["composite"], abs=0.05), ticker

    def test_effective_weight_row_handles_a_stock_with_no_categories(self):
        eff = g._effective_weight_row(dict(REGIME), [])
        assert all(v == 0.0 for v in eff.values())


# =========================================================================
# 3. The pipeline records what it actually used
# =========================================================================
class TestEffectiveWeightsArePropagated:
    """`save_effective_weights` writes the weights the composite was built from.

    Called on a stub rather than a real ``RunContext``: constructing one
    creates a directory under ``runs/`` and attaches a file logger, neither of
    which this method needs and neither of which a test should leave behind.
    """

    @staticmethod
    def _save(tmp_path, cfg, factor_weights=None):
        stub = SimpleNamespace(run_dir=tmp_path)
        return RunContext.save_effective_weights(stub, cfg,
                                                 factor_weights=factor_weights)

    def test_the_regime_adjustment_is_recorded_when_passed(self, tmp_path):
        path = self._save(tmp_path,
                          {"factor_weights": dict(BASE), "metric_weights": {}},
                          factor_weights=dict(REGIME))
        data = json.loads(Path(path).read_text())

        assert data["factor_weights"]["momentum"] == 14.95
        assert data["base_factor_weights"]["momentum"] == 13
        assert data["factor_weights_adjusted"] is True

    def test_without_an_override_the_config_weights_are_recorded(self, tmp_path):
        """The warm-cache path: no scoring ran, so there is nothing to adjust."""
        path = self._save(tmp_path,
                          {"factor_weights": dict(BASE), "metric_weights": {}})
        data = json.loads(Path(path).read_text())

        assert data["factor_weights"] == BASE
        assert data["base_factor_weights"] == BASE
        assert data["factor_weights_adjusted"] is False

    def test_an_unadjusted_run_is_not_reported_as_adjusted(self, tmp_path):
        path = self._save(tmp_path,
                          {"factor_weights": dict(BASE), "metric_weights": {}},
                          factor_weights=dict(BASE))
        assert json.loads(Path(path).read_text())["factor_weights_adjusted"] is False

    def test_the_recorded_weights_are_a_copy_not_a_live_reference(self, tmp_path):
        """Writing must not alias the config the rest of the pipeline holds."""
        cfg = {"factor_weights": dict(BASE), "metric_weights": {}}
        path = self._save(tmp_path, cfg, factor_weights=dict(REGIME))
        cfg["factor_weights"]["momentum"] = 999
        assert json.loads(Path(path).read_text())["base_factor_weights"]["momentum"] == 13

    def test_run_screener_hands_the_adjusted_weights_back(self):
        """The plumbing itself: the fix is a returned value, so pin it statically."""
        src = (Path(__file__).resolve().parent.parent / "run_screener.py").read_text(
            encoding="utf-8")
        assert 'stats["_effective_factor_weights"]' in src, (
            "run_factor_engine must hand the regime-adjusted weights back - "
            "rebinding its local cfg does not reach main()")
        assert 'fe_stats.pop("_effective_factor_weights"' in src
        assert "save_effective_weights(cfg, factor_weights=effective_fw)" in src

    def test_the_adjustment_is_captured_after_it_is_applied_not_before(self):
        """Ordering matters: capture before adjust_momentum_weight records nothing."""
        src = (Path(__file__).resolve().parent.parent / "run_screener.py").read_text(
            encoding="utf-8")
        adjust = src.index("cfg = adjust_momentum_weight(")
        capture = src.index('stats["_effective_factor_weights"]')
        assert capture > adjust


# =========================================================================
# 4. What the reader is told
# =========================================================================
@pytest.fixture(scope="module")
def html():
    return g.generate_html(data_json="{}", methodology_html="")


class TestTheExplanationIsRendered:
    """A weight that differs from the docs has to say why, on the page."""

    def test_the_per_stock_weight_helper_exists(self, html):
        assert "function effWeights(" in html

    def test_the_arithmetic_line_uses_the_effective_weight(self, html):
        """The 'x N% = M pts' line must read effWeights, not the raw config."""
        assert "× ${fmtWeight(weight)} = <strong>" in html

    def test_the_regime_adjustment_is_explained(self, html):
        assert "function weightNote(" in html
        assert "volatility regime" in html

    def test_the_renormalisation_is_explained(self, html):
        assert "shares its weight across the rest" in html

    def test_a_withheld_category_is_shown_rather_than_hidden(self, html):
        """Silently dropping the row would hide why the others add to >100%."""
        assert "contrib-row-na" in html
        assert "Not scored for this stock" in html

    def test_weights_render_to_one_decimal_when_not_round(self, html):
        assert "function fmtWeight(" in html

    def test_the_panel_description_warns_the_weights_can_differ(self, html):
        assert "actually multiplied by" in html

    def test_the_category_badge_no_longer_hardcodes_the_config_weight(self, html):
        assert "const badge = scored" in html
        assert "${factorWt}% weight" not in html


class TestTheMethodologyDocumentSaysSo:
    """SCREENER_OVERVIEW.md is the public explainer and is generated from config.

    It printed the composite formula with the default weights and left the
    reader to assume those were the weights used. They were not.
    """

    @staticmethod
    def _overview():
        import yaml
        import run_screener
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text())
        # Render into a temp location is not supported; the generator writes the
        # tracked file, so assert against the committed output instead.
        assert cfg["factor_weights"], "config must carry factor weights"
        return (Path(__file__).resolve().parent.parent
                / "SCREENER_OVERVIEW.md").read_text(encoding="utf-8")

    def test_it_warns_the_printed_weights_are_defaults(self):
        assert "configured defaults" in self._overview()

    def test_it_names_both_rules_that_move_a_weight(self):
        text = self._overview()
        assert "volatility-regime adjustment" in text.lower()
        assert "Missing-data redistribution" in text

    def test_it_points_the_reader_at_where_the_real_weights_are(self):
        text = self._overview()
        assert "effective_weights.json" in text
        assert "actually multiplied by" in text

    def test_the_generator_is_the_source_not_the_file(self):
        """Rule 10: the doc is generated, so the wording lives in the generator."""
        src = (Path(__file__).resolve().parent.parent / "run_screener.py").read_text(
            encoding="utf-8")
        assert "configured defaults" in src
