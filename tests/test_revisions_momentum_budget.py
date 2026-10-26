"""The Revisions category's independence budget, as a tripwire instead of a comment.

`config.yaml`'s Revisions block has carried this rule since the 2026-09-10 reweight:

    The revisions category has now SPENT its independence budget: it goes from ~91% unspanned
    by the other seven categories to 85.2%, and revisions~momentum becomes the 4th largest of
    the 28 category pairs (+0.171 -> +0.317). Adding any further momentum-adjacent metric here
    requires re-running the SS8.4 spanning regression first, not just a citation.

The bar `research/2026-09-07-revisions-category-has-no-revisions.md` SS8.4 set **before seeing its
own number** is ~0.35, and it used it to *admit* `fy1_revision_3m` at +0.317. On 2026-10-26 it
did the other job: the proposal in `research/2026-10-26-price-target-upside.md` SS6.2 - replace
the scored level of implied upside with a 90-day change in the consensus target - was measured
at **+0.3778** and refused (that note's SS8). Nothing enforced the rule; it was prose a session
had to remember to read, which is how row "ctx"'s lesson reads one level up.

So it is a test now. Calibrated before shipping, the way the bot's input gates were
(`DECISIONS.md`, settled row 0.5/0.7): replayed across **all 57 committed payload versions**,
min -0.0835, median +0.2498, max **+0.3206** - that maximum on 2026-09-11..09-18, immediately
after the 09-10 reweight, which is SS8.4's own predicted +0.317 appearing in published data.
**No ordinary day fires; the refused change does.** A guard that fires in normal use gets
ignored (the 2026-10-18 lesson), and one that cannot fire is decoration (rule 8's).

Two deliberate choices, both from the claims-mutation audit (`plan/claims-mutation-audit.md`):

* **The bar is a literal here and nowhere else.** A check may not import the number it is
  checking - the capex tolerance's only assertion read `rel=X.LEASE_RECONCILE_TOL`, so widening
  the constant widened the assertion with it and the test passed at any value. Widening this one
  means editing a number that `test_the_bar_is_the_one_registered_in_advance` also reads out of
  the 09-07 note, so the research record has to be edited too.
* **The boundary is probed *at* the bar**, not either side of it:
  `test_the_comparison_is_at_the_bar_not_past_it` is the one point that separates `>` from `>=`.

This reads the published run's own category scores, which is the right artifact: they are
produced by the scorer, not written by the dashboard generator, so rule 10's "assert against the
generator's output" does not apply (`tests/test_growth_category.py` reads the same file for the
same reason). No forward return, IC or backtest figure is read (rules 4 and 5).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTE_0907 = ROOT / "research" / "2026-09-07-revisions-category-has-no-revisions.md"
NOTE_1026 = ROOT / "research" / "2026-10-26-price-target-upside.md"
SPAN_JSON = ROOT / "research" / "measurements" / "2026-10-26-target-change-span.json"

# The bar SS8.4 pre-registered. A literal, on purpose - see the module docstring.
BAR = 0.35


def within_budget(rho: float) -> bool:
    """True while the pair is inside the pre-registered budget. Exactly at the bar passes."""
    return rho <= BAR


@pytest.fixture(scope="module")
def published() -> pd.DataFrame:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    return pd.DataFrame(data["table_data"])


def _rho(table: pd.DataFrame, a: str, b: str) -> float:
    return float(pd.to_numeric(table[a], errors="coerce").corr(
        pd.to_numeric(table[b], errors="coerce"), method="spearman"))


# --------------------------------------------------------------------------- the assertion
def test_revisions_has_not_spent_more_of_its_budget_against_momentum(published):
    """The assertion, not the plumbing. If a run ever publishes a Revisions category that is
    more than `BAR` correlated with Momentum, a momentum-adjacent metric has entered it without
    the SS8.4 regression the config rule requires - or a weight moved without one."""
    rho = _rho(published, "revisions_score", "momentum_score")
    assert within_budget(rho), (
        f"revisions ~ momentum is +{rho:.4f}, through the ~{BAR} bar pre-registered in "
        "research/2026-09-07-revisions-category-has-no-revisions.md SS8.4. Over 57 published "
        "payloads this pair has never exceeded +0.3206. Re-run "
        "research/measurements/2026-10-26-target-change-span.py and either reverse the change "
        "or argue the bar in METHODOLOGY_CHANGELOG.md - do not widen this constant to pass."
    )


def test_the_pair_is_actually_measurable_so_this_guard_can_fire(published):
    """A tripwire wired to a number that cannot move is decoration (rule 8). Both category
    scores must be present and varying, or the assertion above is vacuously true."""
    for col in ("revisions_score", "momentum_score"):
        assert col in published.columns, col
        v = pd.to_numeric(published[col], errors="coerce")
        assert v.notna().sum() >= 400, (col, int(v.notna().sum()))
        assert v.std() > 1.0, (col, float(v.std()))
    rho = _rho(published, "revisions_score", "momentum_score")
    assert rho == rho and -1.0 < rho < 1.0


def test_the_comparison_is_at_the_bar_not_past_it():
    """The one point that separates `>` from `>=`. Probing 0.001 either side would pass whichever
    way the comparison is written - the defect `test_the_threshold_is_strict_and_configurable`
    had (plan/claims-mutation-audit.md, 2026-10-25)."""
    assert within_budget(BAR) is True
    assert within_budget(BAR + 1e-12) is False
    assert within_budget(0.3206) is True      # the historical maximum, 2026-09-11..09-18
    assert within_budget(0.3778) is False     # the change refused on 2026-10-26


# --------------------------------------------------------------------------- the written rule
def test_the_config_still_carries_the_rule_this_guard_enforces():
    """The comment is the rule's statement of itself. If it goes, a reader has a test with no
    stated reason, which is how the Revisions block got here in the first place."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "SPENT its independence budget" in text
    assert "requires re-running" in text and "spanning regression" in text.replace("SS8.4", "8.4")
    # and it must point at the note that holds the bar
    assert "research/2026-09-07-revisions-category-has-no-revisions.md" in text


def test_the_bar_is_the_one_registered_in_advance():
    """`BAR` is not free. 0.35 is the number SS8.4 states it set before seeing its result, and
    0.317 is what it admitted at. Both are read out of the note, so moving `BAR` means editing
    the research record."""
    note = NOTE_0907.read_text(encoding="utf-8")
    assert "0.35 bar Monday set before seeing the number" in note
    assert f"{BAR} bar" in note
    assert "+0.317" in note


def test_the_refusal_and_its_evidence_are_on_the_record():
    """The 2026-10-26 decision, and the measurement behind it. A note whose status line and
    whose measurement disagree is worse than neither."""
    note = NOTE_1026.read_text(encoding="utf-8")
    assert "**Proposal status: REFUSED 2026-10-26" in note
    assert SPAN_JSON.exists(), SPAN_JSON
    span = json.loads(SPAN_JSON.read_text(encoding="utf-8"))
    after = span["revisions_category_pairs"]["momentum"]["after_swap"]
    assert not within_budget(after), after
    assert f"+{after:.4f}" in note, (
        "the note must quote the measured post-swap figure its refusal rests on")
    # the rebuild the counterfactual rests on reproduced the published scores
    assert span["fidelity"]["revisions_spearman_vs_published"] == 1.0
    assert span["fidelity"]["revisions_mean_abs_diff_points"] == 0.0


def test_no_weight_moved_with_the_refusal():
    """The refusal left the weights alone. A later session may still change them - it has to
    write a changelog entry citing the note, which is what makes this skip itself."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    changelog = (ROOT / "METHODOLOGY_CHANGELOG.md").read_text(encoding="utf-8")
    if "research/2026-10-26-price-target-upside.md" in changelog:
        pytest.skip("a changelog entry has taken the question up; this guard has done its job")
    rev = cfg["metric_weights"]["revisions"]
    assert rev["price_target_upside"] == 13.5
    assert "target_change_90d" not in rev, (
        "the 90-day target-change metric was refused on 2026-10-26 (SS8 of the note). "
        "Building it needs a changelog entry and a fresh SS8.4 regression.")
