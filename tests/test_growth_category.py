"""The Growth category: what the methodology says stands behind it.

`research/2026-10-12-growth-category.md` found no published return premium for growth as this
category measures it, and that the category's real job in the composite is to offset Valuation,
Investment and Size. The methodology page now says both. These tests read the generator's
output (CLAUDE.md rule 10) and hold its one structural sentence to the published run.

No return of the screener's ranking is read here or in the note (rule 4).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "research" / "2026-10-12-growth-category.md"
SCRIPT = ROOT / "research" / "measurements" / "2026-10-12-growth-category-structure.py"


@pytest.fixture(scope="module")
def growth_section() -> str:
    import run_screener
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    text = run_screener.build_screener_overview(cfg)
    start = text.index("### 3. Growth")
    return text[start: text.index("### 4.", start)]


def test_the_methodology_says_growth_has_no_premium_behind_it(growth_section):
    assert "no published return premium behind it" in growth_section
    assert "not as a factor that has been rewarded" in growth_section
    for study in ("Lakonishok, Shleifer & Vishny 1994", "Chan, Karceski & Lakonishok 2003", "Da & Warachka 2011"):
        assert study in growth_section, study
    assert "research/2026-10-12-growth-category.md" in growth_section
    # it states the category's weight from the config, whatever that weight is
    weight = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]["growth"]
    assert f"Growth's {weight}% offsets" in growth_section


def test_the_methodology_does_not_promise_a_return_for_growth(growth_section):
    """The old rationale stays (why forward growth is the heaviest metric) but the section must
    not read as a claim that growth is rewarded."""
    assert "not as evidence about its return" in growth_section
    for phrase in ("growth premium", "growth stocks earn more", "rewarded for growth"):
        assert phrase not in growth_section.lower()


def test_every_study_the_page_names_is_in_the_note_with_what_was_read():
    note = NOTE.read_text(encoding="utf-8")
    for name in ("Lakonishok, J., Shleifer, A. & Vishny, R. (1994)", "Chan, L., Karceski, J. & Lakonishok, J. (2003)",
                 "Da, Z. & Warachka, M. (2011)", "Bordalo, P., Gennaioli, N., La Porta, R. & Shleifer, A. (2019)",
                 "Asness, C., Frazzini, A. & Pedersen, L. (2019)", "Mohanram, P. (2005)"):
        assert name in note, name
    # the two papers that could not be obtained are marked as such, not cited as read
    assert "Second-hand only, not read: La Porta, R. (1996)" in note
    # the note argues the other side of its own proposal, and says no weight moved
    assert "The case against" in note and "No weight, metric or threshold changed" in note


def test_growth_runs_against_the_categories_the_page_says_it_offsets():
    """ "fast-growing companies tend to score lower on Valuation, Investment and Size" - held to
    the committed run. If a run ever stops showing it, the sentence needs rewriting."""
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    table = pd.DataFrame(data["table_data"])
    growth = pd.to_numeric(table["growth_score"], errors="coerce")
    for other in ("valuation", "investment", "size"):
        rho = growth.corr(pd.to_numeric(table[other + "_score"], errors="coerce"), method="spearman")
        assert rho < -0.1, (other, rho)


def test_no_weight_moved_with_the_note():
    """The note proposed Growth 13 -> 10 for a second session to review. **It was reviewed and
    refused on 2026-10-13** (section 10 of the note), so the config is unchanged and this guard
    now holds a decision rather than an open question. A later session may still move the weight
    - the refusal is not a lock - but it has to write a changelog entry citing the note, which is
    what makes this skip itself."""
    weights = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    changelog = (ROOT / "METHODOLOGY_CHANGELOG.md").read_text(encoding="utf-8")
    if "research/2026-10-12-growth-category.md" in changelog:
        pytest.skip("a changelog entry has taken the proposal up; this guard has done its job")
    assert (weights["growth"], weights["quality"]) == (13, 22)


# ---------------------------------------------------------------------------
# The 2026-10-13 review. The proposal was refused because the measured thing it
# would change is the ranking's *net* growth exposure, which at 13 points is
# about zero - and the evidence the note assembled does not establish the sign
# of a growth tilt, so there is nothing to tilt toward. The page now says so,
# and these tests hold it to the run being published.
# ---------------------------------------------------------------------------

SWEEP = ROOT / "research" / "measurements" / "2026-10-13-growth-weight-sweep.py"
NEUTRAL_TOLERANCE = 0.05


def _published_scores() -> pd.DataFrame:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    return pd.DataFrame(data["table_data"])


def test_the_page_says_why_the_weight_was_kept(growth_section):
    weight = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]["growth"]
    assert f"**Why the weight stays at {weight}%.**" in growth_section
    assert "Reviewed on 2026-10-13" in growth_section
    assert "was refused" in growth_section
    # it must say what cutting the weight would do, not merely that it was not done
    assert "would not remove a position on growth; it would take one" in growth_section
    # and point at the measurement, so the claim is checkable from the page
    assert "research/measurements/2026-10-13-growth-weight-sweep.py" in growth_section
    assert SWEEP.exists()


def test_the_paragraph_is_rewritten_if_the_weight_ever_moves(growth_section):
    """The paragraph says the weight *stays* and names the proposal that was refused. If a later
    session moves it - which it may, with a changelog entry - that sentence stops being true."""
    weights = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    if (weights["growth"], weights["quality"]) != (13, 22):
        assert "Reviewed on 2026-10-13" not in growth_section, (
            "Growth's weight has moved, so the methodology page may no longer say the "
            "2026-10-13 review kept it. Rewrite the 'Why the weight stays' paragraph in "
            "run_screener.build_screener_overview to describe the change that was made."
        )


def test_the_ranking_really_is_close_to_growth_neutral():
    """The claim on the page - "the ranking taken whole sits close to neutral, leaning neither
    way" - held to the published run. This is the assertion, not the plumbing: if a run ever
    produces a ranking that leans on growth, the sentence is false and this fails."""
    table = _published_scores()
    composite = pd.to_numeric(table["Composite"], errors="coerce")
    growth = pd.to_numeric(table["growth_score"], errors="coerce")
    rho = composite.corr(growth, method="spearman")
    assert abs(rho) <= NEUTRAL_TOLERANCE, (
        f"the published ranking correlates {rho:+.3f} with the Growth score; the methodology "
        f"page says it is close to neutral (within {NEUTRAL_TOLERANCE}). Either the run has "
        f"changed character or the paragraph needs rewriting."
    )


def test_growth_is_the_category_that_runs_against_the_others():
    """The page calls Growth "the only one of the eight categories whose score runs against the
    others". Measured as each category against the mean of the other seven: on 2026-10-13 Growth
    was -0.39 and every other category between -0.06 and +0.16."""
    table = _published_scores()
    cats = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
    scores = table[[c + "_score" for c in cats]].apply(pd.to_numeric, errors="coerce")
    scores.columns = cats
    against_rest = {
        c: scores[c].corr(scores[[x for x in cats if x != c]].mean(axis=1), method="spearman")
        for c in cats
    }
    assert against_rest["growth"] < -0.2, against_rest
    others = {c: v for c, v in against_rest.items() if c != "growth"}
    assert min(others.values()) > against_rest["growth"] + 0.15, against_rest


def test_the_note_records_a_decision_rather_than_an_open_proposal():
    """An open proposal is listed in the morning brief with its age until someone decides it
    (write_brief.open_proposals). This one was decided, so the line must say so."""
    note = NOTE.read_text(encoding="utf-8")
    assert "**Proposal status: OPEN**" not in note
    assert "**Proposal status: DECIDED 2026-10-13** - refused" in note
    # the reasoning lives in the note, not only in the log
    assert "## 10. The review, 2026-10-13" in note


def test_the_sweep_script_runs_from_a_clean_checkout():
    res = subprocess.run([sys.executable, "-X", "utf8", str(SWEEP)], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", timeout=180)
    assert res.returncode == 0, res.stderr[-800:]
    assert "growth exposure is zero at weight" in res.stdout


def test_the_measurement_script_runs_from_a_clean_checkout():
    res = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT)], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", timeout=120)
    assert res.returncode == 0, res.stderr[-800:]
    assert "Spearman correlation of Growth's score with each other category's" in res.stdout
    assert "top 25 in common" in res.stdout
