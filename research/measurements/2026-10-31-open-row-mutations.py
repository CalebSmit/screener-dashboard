"""Claims-mutation audit, 2026-10-31: the two `open:` rows the plan called the sharpest.

`plan/claims-mutation-audit.md`'s next list names `clamp-bullet-second-printing` and
`share-class-switching-margin` as the two most likely to be wrong without anyone noticing. This
mutates the generator four ways and records, for each, whether the new check fails and whether
anything that existed *before* today would have failed.

The second question is the one that matters, and it is asked the way the rest of that audit asks
it: apply the mutation, then run the whole suite with only the check that closes the row
deselected. The row's own check is excluded because it is the thing being credited; everything
else in the project gets its chance.

    python research/measurements/2026-10-31-open-row-mutations.py
    python research/measurements/2026-10-31-open-row-mutations.py --rest-of-suite A C

Both modes restore `run_screener.py` in a ``finally``.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("run_screener.py")

CLAMP_TEST = ("tests/test_methodology_claims_audit.py::"
              "test_the_clamps_and_floors_on_the_page_are_the_ones_applied")
MARGIN_CLASS = "tests/test_share_classes.py::TestThePagePrintsTheMargin"

# (id, what it breaks, find, replace, the check that closes the row)
MUTATIONS = [
    ("A", "the clamp bullet prints the forward-EPS ceiling 10 points high",
     "+{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100)}%]",
     "+{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100) + 10}%]",
     CLAMP_TEST),
    ("B", "the clamp bullet prints the price-target floor as the ceiling",
     "price target upside is clamped to [{int(clamps.get('price_target_upside', [-0.50, 1.0])[0] * 100)}%",
     "price target upside is clamped to [{-int(clamps.get('price_target_upside', [-0.50, 1.0])[1] * 100)}%",
     CLAMP_TEST),
    ("C", "the switching margin is typed into the page instead of read from config",
     'f"{_sc_margin:.0%} (S&P DJI\'s own switching margin): measured over 89 "',
     '"20% (S&P DJI\'s own switching margin): measured over 89 "',
     MARGIN_CLASS),
    ("D", "the page's fallback margin drifts from share_classes.CHALLENGE_MARGIN",
     '_sc_margin = float(_sc_cfg.get("challenge_margin", _sc_mod.CHALLENGE_MARGIN))',
     '_sc_margin = float(_sc_cfg.get("challenge_margin", 0.25))',
     MARGIN_CLASS),
]


def run(target: str, rest_of_suite: bool) -> str:
    if rest_of_suite:
        cmd = [sys.executable, "scripts/run_tests.py", "tests/", "test_screener.py", "-q",
               "--deselect", target, "--tb=no", "-rf"]
    else:
        cmd = [sys.executable, "-m", "pytest", target.split("::")[0], "-q", "--no-header",
               "--tb=no"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    out = lines[-1].strip() if lines else f"(no summary, exit {r.returncode})"
    # Which tests failed matters as much as how many: a *drift* tripwire ("the committed page is
    # what the generator builds") fires for any generator edit and is green on the day the change
    # really ships, because the 02:00 run rewrites the artifact. That is the audit's own
    # 2026-10-30 correction, and it has to be read out by name rather than counted.
    failed = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("FAILED ")]
    if failed:
        out += "\n           failed: " + ", ".join(sorted(set(failed)))
    return out


def main(argv: list[str]) -> int:
    rest = "--rest-of-suite" in argv
    only = [a for a in argv if a in {m[0] for m in MUTATIONS}]
    original = SRC.read_text(encoding="utf-8")
    worst = 0
    try:
        print("scope:", "whole suite, the row's own check deselected" if rest
              else "the module holding the row's check")
        for mid, what, find, replace, target in MUTATIONS:
            if only and mid not in only:
                continue
            if original.count(find) != 1:
                print(f"SKIPPED {mid}: anchor appears {original.count(find)} times - {what}")
                worst = 1
                continue
            SRC.write_text(original.replace(find, replace), encoding="utf-8")
            summary = run(target, rest)
            caught = "failed" in summary
            verdict = ("CAUGHT " if caught else "BLIND  ") if rest else \
                      ("CAUGHT " if caught else "MISSED ")
            print(f"{verdict} {mid}: {what}\n           {summary}")
            if not caught and not rest:
                worst = 1
            SRC.write_text(original, encoding="utf-8")
    finally:
        SRC.write_text(original, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == original, "failed to restore run_screener.py"
        print("run_screener.py restored byte-for-byte")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
