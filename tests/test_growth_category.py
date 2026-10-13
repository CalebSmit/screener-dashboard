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
    """The note proposes Growth 13 -> 10 for a second session to review. Until a changelog entry
    records a decision, the config is unchanged - and when one does, delete this test with it."""
    weights = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    changelog = (ROOT / "METHODOLOGY_CHANGELOG.md").read_text(encoding="utf-8")
    if "research/2026-10-12-growth-category.md" in changelog:
        pytest.skip("a changelog entry has taken the proposal up; this guard has done its job")
    assert (weights["growth"], weights["quality"]) == (13, 22)


def test_the_measurement_script_runs_from_a_clean_checkout():
    res = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT)], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", timeout=120)
    assert res.returncode == 0, res.stderr[-800:]
    assert "Spearman correlation of Growth's score with each other category's" in res.stdout
    assert "top 25 in common" in res.stdout
