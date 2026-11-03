"""The universe-expansion phase-0 decisions, pinned so they cannot be re-fitted.

`research/2026-11-02-universe-expansion-phase0.md` pre-registered three choices and a
2026-11-03 session (which did not write the note) adopted all three. They govern universes
that do not exist yet, which is exactly the condition under which a number drifts unnoticed:
nothing in the pipeline reads them today, so nothing today would fail if one were quietly
changed to whatever the first shadow run made convenient.

This is the same reasoning as `tests/test_momentum_panic_rule.py` - "the three numbers are set
in advance and are never re-fitted" - applied to a decision that is prose rather than code.
A floor chosen after seeing what it admits is not a floor.

What is pinned:

1. **The phase-4 floor**, $1,224.13M, with its source, its June 2026 provenance and the fact
   that it is held twelve months. Re-derivable from
   `research/measurements/2026-11-03-nyse-breakpoints.py`.
2. **The reason it was adopted** - the replication papers' shared breakpoint - and the two
   reasons the review struck, so a later session does not reinstate an argument that was
   measured and refused.
3. **The band split and its measured exit test** (0 of 23,092 percentile values).
4. **That `SECTOR_MIN_PEERS` keeps its value**, since only its *scope* changes. A session that
   lowers it to make a thin sector-band group work has changed the rule, not implemented it.
5. **That the decisions live in both files** - the plan and the note - so neither can be
   updated alone.

These tests read two tracked markdown files and nothing else: no network, no run directory,
no published artifact, and no clock.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_engine as fe  # noqa: E402

PLAN = ROOT / "plan" / "universe-expansion.md"
NOTE = ROOT / "research" / "2026-11-02-universe-expansion-phase0.md"

# The pre-registered floor. Verified against Ken French's ME_Breakpoints (CRSP vintage 202609,
# June 2026 row, n = 1,079 NYSE stocks) on 2026-11-03.
FLOOR_M = 1224.13
FLOOR_N_NYSE = 1079


@pytest.fixture(scope="module")
def plan() -> str:
    return PLAN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def note() -> str:
    return NOTE.read_text(encoding="utf-8")


# --- 1. the floor ----------------------------------------------------------------

def test_the_plan_states_the_pre_registered_floor_to_the_cent(plan):
    """$1,224.13M, written out, not rounded and not a range."""
    assert "$1,224.13M" in plan, (
        "the phase-4 floor must be stated exactly in plan/universe-expansion.md; "
        "re-derive it with research/measurements/2026-11-03-nyse-breakpoints.py "
        "before changing this"
    )


def test_the_floor_names_its_source_and_its_june_provenance(plan):
    low = plan.lower()
    assert "me_breakpoints" in low, "the floor must name the file it comes from"
    assert "nyse 20th-percentile" in low or "nyse 20th percentile" in low
    assert "june 2026" in low, "the floor must say which June row it is"
    # The prose writes thousands with a comma; compare on digits so either form passes.
    assert str(FLOOR_N_NYSE) in re.sub(r"(?<=\d),(?=\d)", "", plan), (
        f"the floor must carry its NYSE stock count ({FLOOR_N_NYSE}), which is what makes the "
        "row identifiable in a later CRSP vintage"
    )


def test_the_floor_is_held_twelve_months_not_recomputed_per_run(plan):
    """Fama-French timing: read in June, applied from July, fixed for a year."""
    assert re.search(r"held\s+twelve\s+months", plan, re.I), (
        "a floor recomputed every run is a moving universe boundary; the decision is a June "
        "reading held for twelve months"
    )


def test_the_adopted_reason_is_the_replication_papers_not_the_sp_alignment(plan):
    """The review struck two of the note's five reasons. Keep them struck."""
    assert re.search(r"adopted for one reason only", plan, re.I), (
        "the floor rests on the replication papers' shared breakpoint alone"
    )
    for name in ("Fama & French", "Hou", "Jensen"):
        assert name in plan, f"the adopted reason must name {name}"
    # The struck arguments must stay recorded as struck, with the numbers that struck them.
    assert "coincidence" in plan.lower(), (
        "the agreement with S&P's $1.2B addition floor was measured to be this June's "
        "coincidence (June 2025: $870M; June 2020: $484M) - that finding must not be lost"
    )
    assert "+112.1%" in plan and "-34.4%" in plan and "+40.7%" in plan, (
        "the floor's year-to-year moves are why 'it never needs re-tuning' was struck; "
        "keep the three figures that show it"
    )


def test_the_july_churn_consequence_is_carried_into_phase_4(plan):
    """The floor's annual reset is the reason the input gates need a July replay."""
    assert re.search(r"july replay", plan, re.I), (
        "an average-month continuity threshold will read the annual floor reset as a broken "
        "run; the plan must say the replay spans a July"
    )


# --- 2. the band split -----------------------------------------------------------

def test_the_plan_states_the_band_split_and_the_band_wide_fallback(plan):
    assert "(Sector, size_band)" in plan or "`(Sector, size_band)`" in plan
    assert re.search(r"band-?\s*wide", plan, re.I), (
        "the fallback below ten peers becomes band-wide, not universe-wide; falling back "
        "across bands reintroduces the size mixing the split removes"
    )


def test_the_exit_test_is_recorded_as_a_measurement_with_its_floor(plan):
    """0 of 23,092 means nothing without the number that shows the check can fail."""
    assert "23,092" in plan, "the exit test's measured size must be stated"
    assert "20,151" in plan, (
        "the floor test (a genuine two-band split moves 20,151 values) is what makes the zero "
        "meaningful; an identity check that cannot fail proves nothing"
    )


def test_sector_min_peers_keeps_its_value_because_only_its_scope_changes(plan):
    assert fe.SECTOR_MIN_PEERS == 10, (
        "the decision changes the scope of the sub-ten-peer fallback, not the threshold. "
        "Lowering it to accommodate a thin sector-band group changes the rule - argue that "
        "case in the note and the changelog first"
    )
    assert re.search(r"SECTOR_MIN_PEERS` stays\s*\n?\s*at 10|SECTOR_MIN_PEERS` stays at 10", plan) \
        or "`SECTOR_MIN_PEERS` stays" in plan, \
        "the plan must say the threshold stays and only its scope moves"


def test_the_one_thin_group_is_named_with_its_count(plan):
    """S&P 400 Communication Services, 6 members - the only group under ten in the S&P 1500."""
    assert "Communication Services" in plan
    assert re.search(r"Communication\s*\n?\s*Services \(6 members\)", plan) \
        or "Communication Services (6 members)" in plan, \
        "the single guaranteed-fallback group must be named with its member count"
    assert re.search(r"middle band", plan, re.I), (
        "the thin group is in the S&P 400, so phasing the 600 in last does not defer it - "
        "that is the part a later session would otherwise have to rediscover"
    )


# --- 3. the min_market_cap resolution --------------------------------------------

def test_the_min_market_cap_contradiction_stays_resolved_with_its_measurement(plan):
    assert "min_market_cap: 0" in plan, (
        "the note gave two answers for the S&P 500 band; the measurement (0 of 503 members "
        "below $2B) resolves it to one rule for every S&P index band"
    )
    assert "$6.050B" in plan or "6.050B" in plan, (
        "keep the measured headroom (smallest member MOS at $6.050B against a $2B floor); "
        "it is what makes the single rule safe, and what a later session needs to re-check"
    )


def test_the_reopening_condition_for_the_floor_is_written_down(plan):
    assert re.search(r"within 2x of the floor", plan, re.I), (
        "a decision with no reopening condition is a decision nobody can revisit honestly"
    )


# --- 4. the two files move together ----------------------------------------------

def test_the_note_records_the_decision_rather_than_still_asking(note):
    assert "**Proposal status: DECIDED" in note, (
        "the proposal's status line must be rewritten by the deciding session "
        "(prompts/nightly.md section 3)"
    )
    assert "Proposal status: OPEN" not in note, (
        "an OPEN line left behind puts the proposal back on the morning brief's "
        "'Waiting for a decision' list for ever"
    )
    assert re.search(r"^## 9\. Review and decision", note, re.M), (
        "the review belongs in the note, where the next reader of the proposal will find it"
    )


def test_the_floor_agrees_between_the_note_and_the_plan(note, plan):
    """Two copies of one number are free to drift - the 2026-10-31 share-class lesson."""
    assert "$1,224.13M" in note and "$1,224.13M" in plan, (
        "the floor is stated in both files; if you change one, change both"
    )
    in_note = set(re.findall(r"\$1,2\d\d\.\d\dM", note))
    in_plan = set(re.findall(r"\$1,2\d\d\.\d\dM", plan))
    assert in_note == in_plan == {"$1,224.13M"}, (
        f"the two files must name exactly one floor and the same one: "
        f"note {sorted(in_note)}, plan {sorted(in_plan)}"
    )


def test_the_measurement_scripts_behind_the_decision_are_committed(note):
    """A decision whose numbers cannot be re-derived is an assertion."""
    scripts = [
        "research/measurements/2026-11-03-nyse-breakpoints.py",
        "research/measurements/2026-11-03-sector-band-peer-counts.py",
        "research/measurements/2026-11-03-band-percentiles-identity.py",
    ]
    for rel in scripts:
        assert rel in note, f"the note must name {rel}"
        assert (ROOT / rel).is_file(), f"{rel} is named by the note but not committed"


def test_the_open_coverage_question_is_not_silently_folded_into_the_decision(plan, note):
    """Section 4's (a)/(b)/(c) choice is a present-day methodology change, left open."""
    assert "Loews" in plan, (
        "the one S&P 500 stock whose Revisions category rests on days-to-cover alone must stay "
        "named in the plan: it is what makes the coverage-rule choice a change of its own"
    )
    assert re.search(r"left open on purpose, not deferred", plan, re.I), (
        "an open question recorded as a deferral reads as something nobody did; this one was "
        "deliberately not decided here"
    )
