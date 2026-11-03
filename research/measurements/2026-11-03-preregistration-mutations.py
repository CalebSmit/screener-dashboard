"""Do the phase-0 pre-registration tests have teeth? Mutate, run, restore.

`tests/test_universe_expansion_preregistration.py` guards prose. A test that reads a markdown
file is the easiest kind to write and the easiest kind to be vacuous, so this applies the
audit `plan/claims-mutation-audit.md` applies to published claims: break the thing each test
claims to see, and confirm that test fails.

The project's own correction from 2026-10-31 is applied: a mutation that fails only a *drift*
tripwire (or only some other test) is NOT caught for this purpose. Each mutation is scored
against the named test alone.

Both files are restored with `git checkout --` in a finally block, and the script refuses to
run if either file has uncommitted changes (so a restore cannot lose work).

Run:  python research/measurements/2026-11-03-preregistration-mutations.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = "plan/universe-expansion.md"
NOTE = "research/2026-11-02-universe-expansion-phase0.md"
TESTS = "tests/test_universe_expansion_preregistration.py"

# (label, file, find, replace, the test that must fail)
MUTATIONS = [
    ("floor re-fitted to a round number", PLAN, "$1,224.13M", "$1,200M",
     "test_the_plan_states_the_pre_registered_floor_to_the_cent"),
    ("floor's NYSE count dropped", PLAN, "n = 1,079 NYSE stocks", "n = NYSE stocks",
     "test_the_floor_names_its_source_and_its_june_provenance"),
    ("floor recomputed per run", PLAN, "held twelve months", "recomputed each run",
     "test_the_floor_is_held_twelve_months_not_recomputed_per_run"),
    ("the struck S&P-alignment argument reinstated", PLAN, "this June's coincidence",
     "a durable alignment",
     "test_the_adopted_reason_is_the_replication_papers_not_the_sp_alignment"),
    ("the floor's volatility figures removed", PLAN, "+112.1%", "a little",
     "test_the_adopted_reason_is_the_replication_papers_not_the_sp_alignment"),
    ("July replay requirement softened", PLAN, "July replay", "representative replay",
     "test_the_july_churn_consequence_is_carried_into_phase_4"),
    ("fallback flipped in the decision", PLAN, "to **band**-wide", "to **universe**-wide",
     "test_the_plan_states_the_band_split_and_the_band_wide_fallback"),
    ("fallback flipped in phase 1's work order", PLAN, "fallback becomes band-wide",
     "fallback becomes universe-wide",
     "test_the_plan_states_the_band_split_and_the_band_wide_fallback"),
    ("exit test's floor figure removed", PLAN, "**20,151 differing (up to 68.93 points)**",
     "a lot differing", "test_the_exit_test_is_recorded_as_a_measurement_with_its_floor"),
    ("the thin group loses its count", PLAN, "Communication Services (6 members)",
     "Communication Services", "test_the_one_thin_group_is_named_with_its_count"),
    ("the middle-band finding dropped", PLAN, "**The thin group is in the\n   middle band**",
     "**It is a small group**", "test_the_one_thin_group_is_named_with_its_count"),
    ("min_market_cap resolution reversed", PLAN, "`min_market_cap: 0` for every S&P index band",
     "`min_market_cap` per band as convenient",
     "test_the_min_market_cap_contradiction_stays_resolved_with_its_measurement"),
    ("the measured headroom removed", PLAN, "MOS at $6.050B", "MOS",
     "test_the_min_market_cap_contradiction_stays_resolved_with_its_measurement"),
    ("reopening condition removed", PLAN, "within 2x of the floor", "clearly too close",
     "test_the_reopening_condition_for_the_floor_is_written_down"),
    ("Loews dropped, folding the open question away", PLAN, "Loews", "one stock",
     "test_the_open_coverage_question_is_not_silently_folded_into_the_decision"),
    ("open question recorded as a deferral", PLAN, "**Left open on purpose, not deferred:**",
     "**Deferred:**",
     "test_the_open_coverage_question_is_not_silently_folded_into_the_decision"),
    ("status line left OPEN", NOTE, "**Proposal status: DECIDED 2026-11-03",
     "**Proposal status: OPEN** - 2026-11-03",
     "test_the_note_records_the_decision_rather_than_still_asking"),
    ("review section removed from the note", NOTE, "## 9. Review and decision",
     "## 9. Some thoughts",
     "test_the_note_records_the_decision_rather_than_still_asking"),
    ("the two files' floors drift apart", NOTE, "$1,224.13M", "$1,225.00M",
     "test_the_floor_agrees_between_the_note_and_the_plan"),
    ("a measurement script unnamed", NOTE,
     "research/measurements/2026-11-03-band-percentiles-identity.py", "a script I ran",
     "test_the_measurement_scripts_behind_the_decision_are_committed"),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def mutate_last(text: str, find: str, repl: str) -> str:
    """Replace the LAST occurrence. Half the mutations below target the second copy of a
    duplicated statement, which is the half the first version of the tests could not see."""
    i = text.rfind(find)
    return text[:i] + repl + text[i + len(find):]


def run_one(test: str) -> bool:
    """True if the named test FAILS (i.e. the mutation was caught by that test)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", f"{TESTS}::{test}", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return r.returncode != 0


def main() -> int:
    dirty = [ln for ln in git("status", "--porcelain").splitlines()
             if ln[3:].strip() in (PLAN, NOTE)]
    if dirty:
        print("REFUSING: commit the plan and the note first; this script restores with "
              "`git checkout --` and would discard uncommitted work.")
        print("\n".join(dirty))
        return 2

    print(f"{len(MUTATIONS)} mutation targets against {TESTS}")
    print("(a duplicated statement is mutated twice: first copy and last)\n")
    caught = 0
    applied = 0
    equivalent = []
    try:
        for i, (label, rel, find, repl, test) in enumerate(MUTATIONS, 1):
            path = ROOT / rel
            text = path.read_text(encoding="utf-8")
            if find not in text:
                print(f"{i:>3}. SKIP (target absent)  {label}")
                print(f"       looked for: {find!r}")
                equivalent.append(label)
                continue
            # Every duplicated statement is mutated in BOTH directions: first copy and last.
            # A test that reads `needle in text` passes on either mutation alone.
            for which, mutated in (("first", text.replace(find, repl, 1)),
                                   ("last", mutate_last(text, find, repl))):
                if mutated == text:
                    continue
                path.write_text(mutated, encoding="utf-8")
                failed = run_one(test)
                git("checkout", "--", rel)
                applied += 1
                caught += failed
                copies = text.count(find)
                tag = f" [{which} of {copies}]" if copies > 1 else ""
                print(f"{i:>3}. {'CAUGHT' if failed else 'GAP   '}  {label}{tag}")
                if not failed:
                    print(f"       {test} still passes -- it is not seeing what it claims to")
                if copies == 1:
                    break
    finally:
        git("checkout", "--", PLAN)
        git("checkout", "--", NOTE)
        after = [ln for ln in git("status", "--porcelain").splitlines()
                 if ln[3:].strip() in (PLAN, NOTE)]
        print(f"\nrestored; both files clean: {not after}")
        if after:
            print("!! NOT CLEAN:\n" + "\n".join(after))

    print(f"\n{caught} caught of {applied} applied ({len(equivalent)} targets skipped)")
    if equivalent:
        print("skipped (the mutation's target text was not found - fix the script, not the test):")
        for label in equivalent:
            print(f"  - {label}")
    return 0 if caught == applied else 1


if __name__ == "__main__":
    sys.exit(main())
