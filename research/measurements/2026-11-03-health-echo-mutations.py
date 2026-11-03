"""Would the new tests have caught the health-line echo defect? Revert the fix, run, restore.

`scripts/health_numbers.py`'s `last_code_session()` matched the bare phrase
"SESSION DID NOT RUN", which the runner echoes into every log via this module's own output
(2026-11-03). The fix matches the runner's line shape instead. Four tests were added with it,
and tests written the same day as a fix pass by construction - so the question this script
asks is the one `plan/claims-mutation-audit.md` asks: with the fix undone, does each test fail,
and would anything *else* have noticed?

Restores with `git checkout --` in a finally block; refuses to run on a dirty target.

Run:  python research/measurements/2026-11-03-health-echo-mutations.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = "scripts/health_numbers.py"
TESTS = "tests/test_health_numbers.py"

# The pre-fix logic, verbatim from git history (commit 9be2cdc's parent).
FIXED = """        marker = next((m for m in (DID_NOT_RUN.search(ln) for ln in lines) if m), None)
        if marker:
            why = marker.group(1).strip()
            return f"**no** - `logs/{path.name}`: SESSION DID NOT RUN ({why[:140]})\""""

BROKEN = """        if "SESSION DID NOT RUN" in text:
            why = next((ln.split("SESSION DID NOT RUN:", 1)[1].strip() for ln in lines
                        if "SESSION DID NOT RUN:" in ln), "")
            return f"**no** - `logs/{path.name}`: SESSION DID NOT RUN ({why[:140]})\""""

# Classified by what the first run of this script measured, not by what they are named.
# DETECT: fails with the fix reverted. GUARD: passes either way - it pins behaviour the fix
# must not break, which is a real job but is not coverage of this defect.
NEW_TESTS = [
    ("DETECT",
     "test_a_healthy_session_is_not_called_dead_by_the_previous_verdict_in_its_own_log"),
    ("GUARD", "test_a_real_marker_still_fires_when_the_echo_sits_in_the_same_log"),
    ("GUARD", "test_the_reason_is_never_empty_when_the_verdict_is_no"),
    ("DETECT", "test_the_real_logs_do_not_report_a_shipped_session_as_dead"),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def pytest(*args: str) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    if any(ln[3:].strip() == TARGET for ln in git("status", "--porcelain").splitlines()):
        print(f"REFUSING: {TARGET} has uncommitted changes; this script restores it with "
              "`git checkout --`.")
        return 2

    path = ROOT / TARGET
    text = path.read_text(encoding="utf-8")
    if FIXED not in text:
        print(f"REFUSING: could not find the fixed block in {TARGET}. Update this script.")
        return 2

    try:
        path.write_text(text.replace(FIXED, BROKEN, 1), encoding="utf-8")
        print("fix reverted (the pre-2026-11-03 substring match is back)\n")

        print("=== 1. Does each new test fail with the fix reverted? ===")
        detect = [n for role, n in NEW_TESTS if role == "DETECT"]
        caught = 0
        for role, name in NEW_TESTS:
            rc, _ = pytest(f"{TESTS}::{name}")
            ok = rc != 0
            caught += ok
            flag = "CAUGHT" if ok else "passes"
            mismatch = "" if (ok == (role == "DETECT")) else "   <-- MISCLASSIFIED"
            print(f"  [{role}] {flag}  {name}{mismatch}")
        print(f"  {caught} of {len(NEW_TESTS)} fail; {len(detect)} are claimed as detectors")
        if caught != len(detect):
            print("  !! the classification above no longer matches the measurement")

        print("\n=== 2. Would anything else have noticed? ===")
        print("  (the whole suite with only these four deselected - the 2026-10-29 method)")
        deselect = [a for _, n in NEW_TESTS for a in ("--deselect", f"{TESTS}::{n}")]
        rc, out = pytest("tests/", "test_screener.py", *deselect)
        last = [ln for ln in out.strip().splitlines() if ln.strip()][-1]
        print(f"  exit {rc}: {last}")
        print("  => " + ("something else sees it" if rc != 0 else
                         "NOTHING else sees it - the defect was invisible to the suite"))
    finally:
        git("checkout", "--", TARGET)
        dirty = [ln for ln in git("status", "--porcelain").splitlines()
                 if ln[3:].strip() == TARGET]
        print(f"\nrestored; {TARGET} clean: {not dirty}")
        if dirty:
            print("!! NOT CLEAN:\n" + "\n".join(dirty))
    return 0


if __name__ == "__main__":
    sys.exit(main())
