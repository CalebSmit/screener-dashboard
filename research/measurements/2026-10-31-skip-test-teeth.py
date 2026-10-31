"""Does the rewritten download-skip test still have teeth, on every day of its week?

The 2026-10-31 session rewrote ``tests/test_forward_return_download_skip.py`` to run on a frozen
week instead of ``datetime.now()`` (it failed every Saturday and Sunday, and turned ``main`` red
on the first weekend after it shipped). A test made deterministic can easily be made vacuous at
the same time, so this mutates the production rule it covers four ways and records what the test
does. Run from the repo root; it restores ``improvement_engine.py`` in a ``finally``.

    python research/measurements/2026-10-31-skip-test-teeth.py
    python research/measurements/2026-10-31-skip-test-teeth.py --rest-of-suite M3   # see below

``--rest-of-suite`` runs the whole suite with this module deselected instead, which answers the
other half of the question: not "does the new test fail?" but "was anything in the project
already seeing this?". Name one or more mutations (``M1``..``M5``) to run only those.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

TEST = "tests/test_forward_return_download_skip.py"
SRC = pathlib.Path("improvement_engine.py")

TODAY_LINE = ("    today = pd.Timestamp(today if today is not None "
              "else datetime.now().date()).normalize()")
DAY_LINE = "    day = pd.Timestamp(end).normalize()"
SKIP_CALL = ("if not _close_possible_since(min(snap_dt + timedelta(days=HORIZON_DAYS[h]) "
             "for h in needed)):")

MUTATIONS = [
    # (name, what it breaks, find, replace)
    ("M1 the rule always says a close is possible - nothing is ever skipped",
     TODAY_LINE, "    return True\n" + TODAY_LINE),
    ("M2 the rule always says no close is possible - every download is skipped",
     TODAY_LINE, "    return False\n" + TODAY_LINE),
    ("M3 the skip reads the LATEST due horizon end instead of the earliest",
     SKIP_CALL, SKIP_CALL.replace("min(", "max(")),
    # The first draft of M4 inserted its line *above* the real ``day = ...`` assignment, which
    # overwrote it: a no-op dressed as a mutation, and it read as a gap in the test. Mutate the
    # line itself.
    ("M4 the scan starts a day after the horizon end - off by one, late",
     DAY_LINE, DAY_LINE + " + timedelta(days=1)"),
    ("M5 the scan counts the weekend as a session",
     "        if day.dayofweek < 5:", "        if day.dayofweek < 7:"),
]


def run_test(rest_of_suite: bool) -> str:
    """This module alone, or the whole suite with this module deselected - "was anything else
    seeing it?", the question `plan/claims-mutation-audit.md` asks of every mutation."""
    if rest_of_suite:
        cmd = [sys.executable, "scripts/run_tests.py", "tests/", "test_screener.py", "-q",
               "--ignore=" + TEST]
    else:
        cmd = [sys.executable, "-m", "pytest", TEST, "-q", "--no-header", "--tb=no"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    return lines[-1].strip() if lines else f"(no summary, exit {r.returncode})"


def main(argv: list[str]) -> int:
    rest = "--rest-of-suite" in argv
    only = [a for a in argv if a.startswith("M")]
    original = SRC.read_text(encoding="utf-8")
    worst = 0
    try:
        print(f"scope: {'whole suite, this module deselected' if rest else 'this module'}")
        print(f"unmutated: {run_test(rest)}")
        for name, find, replace in MUTATIONS:
            if only and name.split()[0] not in only:
                continue
            if original.count(find) != 1:
                print(f"SKIPPED  {name}: anchor not found exactly once")
                worst = 1
                continue
            SRC.write_text(original.replace(find, replace), encoding="utf-8")
            summary = run_test(rest)
            caught = "failed" in summary
            print(f"{'CAUGHT ' if caught else 'MISSED '} {name}\n           {summary}")
            if not caught and not rest:
                worst = 1
            SRC.write_text(original, encoding="utf-8")
    finally:
        SRC.write_text(original, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == original, "failed to restore the source"
        print("source restored byte-for-byte")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
