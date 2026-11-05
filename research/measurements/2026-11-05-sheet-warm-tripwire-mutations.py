"""Does `tests/test_sheet_warm_is_settled.py` see the fix being undone?

Each mutation removes one part of the 2026-11-05 harness fix, runs the tripwire module, and
restores the files. Every one must fail the module - and the last two check the sweep's own teeth,
since a sweep that passes vacuously reads as coverage.

    python research/measurements/2026-11-05-sheet-warm-tripwire-mutations.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOD = "tests/test_sheet_warm_is_settled.py"
HELPER = ROOT / "tests" / "sheet_warm.py"
FIXED = ["tests/test_context_layer_claims.py", "tests/test_context_cards_printed.py",
         "tests/test_dashboard_printed_figures.py"]

MUTATIONS = [
    # the settle dropped from each fixed module in turn - the regression that would reopen the race
    *[(f"settle dropped from {Path(f).name}", [(ROOT / f, "sheet_warm.settle(pg)", "pass")])
      for f in FIXED],
    # the import dropped too, so the module looks untouched to the sweep
    ("settle and import both dropped from test_context_cards_printed.py",
     [(ROOT / "tests/test_context_cards_printed.py", "sheet_warm.settle(pg)", "pass"),
      (ROOT / "tests/test_context_cards_printed.py",
       "from tests import sheet_warm  # noqa: E402 - settle the sheet pre-warm (2026-11-05)\n", "")]),
    # the helper made permissive - it would accept a pre-warm that is still pending
    ("helper accepts a pending pre-warm",
     [(HELPER, 'assert state in ("done", "absent")', 'assert state in ("done", "absent", "wait", "run")')]),
    # the helper made a no-op, which is how a "fix" quietly stops fixing
    ("helper does nothing", [(HELPER, "if (SHEET_WARM.stop) SHEET_WARM.stop(); else SHEET_WARM.state = 'done';",
                              "")]),
]


def run() -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", MOD, "-q"], cwd=ROOT,
                       capture_output=True, text=True, errors="replace")
    tail = [l for l in (r.stdout or "").splitlines() if " passed" in l or " failed" in l or " error" in l]
    return r.returncode, (tail[-1].strip() if tail else "(no count line)")


def main() -> int:
    paths = {HELPER, *(ROOT / f for f in FIXED)}
    original = {p: p.read_text(encoding="utf-8") for p in paths}
    code, line = run()
    print(f"unmutated: exit {code}  {line}")
    if code != 0:
        print("the module is not green before the mutations - stopping")
        return 1
    bad = []
    try:
        for name, edits in MUTATIONS:
            counts = [original[p].count(old) for p, old, _ in edits]
            if any(c < 1 for c in counts):
                print(f"{name:58s} SKIPPED - anchors appear {counts} times")
                bad.append(name)
                continue
            for p in paths:
                p.write_text(original[p], encoding="utf-8")
            for p, old, new in edits:
                p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
            code, line = run()
            print(f"{name:58s} {'CAUGHT' if code else 'GOT THROUGH':12s} exit {code}  {line}")
            if code == 0:
                bad.append(name)
    finally:
        for p in paths:
            p.write_text(original[p], encoding="utf-8")
    code, line = run()
    print(f"restored:  exit {code}  {line}")
    print("\n" + ("every mutation was caught" if not bad and code == 0 else f"PROBLEM: {bad}"))
    return 0 if not bad and code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
