"""Does `test_a_red_baseline_leaves_its_whole_output_too` actually see the thing it claims to?

CLAUDE.md's standing lesson from the claims audit: a check that still passes when the code it
describes is broken is checking plumbing, not the claim. This applies each way the 2026-11-05 fix
could be undone to `scripts/nightly-screener.ps1`, re-runs `tests/test_gate_ordering.py`, and
restores the file. Every mutation must fail the suite.

    python research/measurements/2026-11-05-baseline-output-mutations.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PS1 = ROOT / "scripts" / "nightly-screener.ps1"
SAVE = "Save-GateOutput $b 'baseline' 'WARN'"

# Each mutation is a list of (old, new) replacements, every `old` required to appear exactly once.
MUTATIONS = [
    # the fix removed outright
    ("the save removed", [(SAVE, "# (removed)")]),
    # saved, but at ERROR - the baseline reading like a gate failure
    ("saved at ERROR", [(SAVE, "Save-GateOutput $b 'baseline' 'ERROR'")]),
    # *moved* inside the named-failures branch, so "could not be taken" saves nothing - the state
    # a session can do least with. The outer call has to go, or both branches still save.
    ("only the named-failures branch", [
        (SAVE, "# (moved up)"),
        ("""            Write-Log "BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green." 'WARN'""",
         """            Write-Log "BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green." 'WARN'
            Save-GateOutput $b 'baseline' 'WARN'"""),
    ]),
    # the level made mandatory, so gates 1 and 5 would have to pass one
    ("level has no default", [("[string]$Level = 'ERROR'", "[string]$Level")]),
    # the gates made to pass a level explicitly - two literals for one behaviour, free to drift
    ("gate 1 passes its own level",
     [("Save-GateOutput $t 'gate1'", "Save-GateOutput $t 'gate1' 'ERROR'")]),
]


def run_tests() -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_gate_ordering.py", "-q"],
                       cwd=ROOT, capture_output=True, text=True, errors="replace")
    tail = [l for l in (r.stdout or "").splitlines() if " passed" in l or " failed" in l]
    return r.returncode, (tail[-1].strip() if tail else "(no count line)")


def main() -> int:
    original = PS1.read_text(encoding="utf-8")
    code, line = run_tests()
    print(f"unmutated: exit {code}  {line}")
    if code != 0:
        print("the suite is not green before the mutations - stopping")
        return 1
    bad = []
    try:
        for name, pairs in MUTATIONS:
            counts = [original.count(old) for old, _ in pairs]
            if any(c != 1 for c in counts):
                print(f"{name:32s} SKIPPED - anchors appear {counts} times, each must appear once")
                bad.append(name)
                continue
            text = original
            for old, new in pairs:
                text = text.replace(old, new)
            PS1.write_text(text, encoding="utf-8")
            code, line = run_tests()
            verdict = "CAUGHT" if code != 0 else "GOT THROUGH"
            print(f"{name:32s} {verdict:12s} exit {code}  {line}")
            if code == 0:
                bad.append(name)
    finally:
        PS1.write_text(original, encoding="utf-8")
    code, line = run_tests()
    print(f"restored:  exit {code}  {line}")
    print("\n" + ("every mutation was caught" if not bad and code == 0 else f"PROBLEM: {bad}"))
    return 0 if not bad and code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
