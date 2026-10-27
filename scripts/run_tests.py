#!/usr/bin/env python3
"""Run pytest in parallel when pytest-xdist is importable, and exactly as before when it is not.

Why this exists (2026-10-26, owner-run night)
---------------------------------------------
Gate 1 - ``python -m pytest tests/ test_screener.py -q`` - took 6:10-7:20 serially on this
6-core machine (~3,160 tests), and every scheduled code session runs it about three times:
twice in-session, under the Bash tool's 600 s per-call ceiling, and once as the runner's own
gate 1 (``scripts/nightly-screener.ps1``). Fourteen code sessions a week. Spread over the
cores with pytest-xdist it takes about a third of that.

The rule that shapes this file: **a gate must never fail because a plugin is missing.** So:

* xdist importable -> ``python -m pytest <args> -n auto``;
* xdist not importable -> ``python -m pytest <args>``, the exact serial command, byte for byte;
* xdist importable but the parallel run could not run the suite at all (pytest exit 3,
  internal error, or 4, usage error - neither is a verdict on any test) -> the serial command
  is run, and its exit code is the answer. A test that *fails* (exit 1) is never retried: a
  retry would turn a real failure into a coin toss.

The caller's arguments pass through untouched. If they already choose a worker count
(``-n``/``--numprocesses``/``--dist``) or switch xdist off (``-p no:xdist``) nothing is added.

The published-artifact tripwire in the root ``conftest.py`` works on both paths: serially as it
always has, and under xdist the controller snapshots before any worker starts and every worker
checks against that snapshot (``tests/test_parallel_suite.py``).

Usage:
    python scripts/run_tests.py tests/ test_screener.py -q
    python scripts/run_tests.py --serial tests/ test_screener.py -q   # force the serial path

Exit code: pytest's own, from whichever run decided.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORKERS = "auto"

# pytest exit codes that mean the run itself did not happen, not that a test failed.
NOT_A_VERDICT = (3, 4)


def xdist_available() -> bool:
    """True when the pytest-xdist plugin can be imported by this interpreter."""
    try:
        return importlib.util.find_spec("xdist") is not None
    except (ImportError, ValueError):
        return False


def chooses_its_own_workers(args: list[str]) -> bool:
    """The caller already decided how (or whether) to distribute - leave that alone."""
    for i, a in enumerate(args):
        if a == "-n" or re.fullmatch(r"-n=?(\d+|auto|logical)", a):
            return True
        if a.startswith("--numprocesses") or a.startswith("--dist"):
            return True
        if a == "no:xdist" and i > 0 and args[i - 1] == "-p":
            return True
        if a == "-pno:xdist":
            return True
    return False


def serial_command(args: list[str]) -> list[str]:
    return [sys.executable, "-m", "pytest", *args]


def parallel_command(args: list[str]) -> list[str]:
    return [*serial_command(args), "-n", WORKERS]


def plan(args: list[str], force_serial: bool = False) -> tuple[str, list[str]]:
    """Which command to run first, and a one-line description of why."""
    if force_serial:
        return "serial (--serial)", serial_command(args)
    if chooses_its_own_workers(args):
        return "as given (the arguments choose the distribution)", serial_command(args)
    if not xdist_available():
        return "serial (pytest-xdist is not importable)", serial_command(args)
    return f"parallel (pytest-xdist, -n {WORKERS})", parallel_command(args)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    force_serial = "--serial" in args
    args = [a for a in args if a != "--serial"]

    mode, cmd = plan(args, force_serial)
    print(f"TEST RUNNER: {mode}", flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)

    if cmd != serial_command(args) and rc in NOT_A_VERDICT:
        print(f"TEST RUNNER: the parallel run did not run the suite (pytest exit {rc}) - "
              "rerunning serially; that run's exit code decides.", flush=True)
        rc = subprocess.call(serial_command(args), cwd=ROOT)
    return rc


if __name__ == "__main__":
    sys.exit(main())
