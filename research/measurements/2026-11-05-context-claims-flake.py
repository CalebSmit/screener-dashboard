"""Why did `tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken`
fail in the 13:00 session's gate-1 baseline on 2026-11-05 and pass every time since?

The runner logged exactly one pre-existing failure at 13:03:29 with `main` at 0081d38, and the same
command on the same commit reported 3,976 passed seven minutes later. The baseline, unlike gates 1
and 5, does not save its output, so there is no record of the assertion that failed.

This script reproduces the conditions a parallel run puts the module under, one condition at a time,
and prints how many of N repetitions fail:

  serial          the module on its own, declaration order
  reverse         the module on its own, reversed (module-scoped `page`, so order can leak state)
  xdist-4/8       the module under -n 4 / -n 8 (default --dist load scatters a module's tests
                  across workers, each launching its own Chromium for it)
  with-browsers   the module together with every other Chromium module under -n auto, which is
                  what gate 1 actually runs

    python research/measurements/2026-11-05-context-claims-flake.py [reps]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = "tests/test_context_layer_claims.py"
TEST = f"{TARGET}::test_the_page_says_when_each_kind_of_quote_was_taken"

# Every collected module that drives Chromium, found by importing playwright.
def browser_modules() -> list[str]:
    out = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        t = p.read_text(encoding="utf-8", errors="replace")
        if "playwright" in t:
            out.append(f"tests/{p.name}")
    return out


SCENARIOS = {
    "serial":        [TARGET, "-q", "-p", "no:xdist"],
    "reverse":       [TARGET, "-q", "-p", "no:randomly", "--reverse"],   # needs pytest-reverse; skipped if absent
    "xdist-4":       [TARGET, "-q", "-n", "4"],
    "xdist-8":       [TARGET, "-q", "-n", "8"],
    "with-browsers": None,      # filled in below, needs browser_modules()
}


def run(args: list[str]) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *args], cwd=ROOT,
                       capture_output=True, text=True, errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mods = browser_modules()
    SCENARIOS["with-browsers"] = [*mods, "-q", "-n", "auto"]
    print(f"repo {ROOT}")
    print(f"chromium modules ({len(mods)}): {', '.join(m.split('/')[-1] for m in mods)}")
    print(f"{reps} repetition(s) per scenario; the test of interest is\n  {TEST}\n")
    for name, args in SCENARIOS.items():
        fails = 0
        detail = []
        for i in range(reps):
            code, out = run(args)
            if "ERROR: usage" in out or "unrecognized arguments" in out:
                print(f"{name:14s} SKIPPED - {out.strip().splitlines()[-1][:90]}")
                detail = None
                break
            tail = [l for l in out.splitlines() if " passed" in l or " failed" in l or " error" in l]
            line = tail[-1] if tail else out.strip().splitlines()[-1][:120]
            ours = TEST.split("::")[-1] in out and ("FAILED" in out or "failed" in line)
            fails += bool(ours)
            detail.append(f"    rep {i + 1}: exit {code}  {line.strip()}"
                          + ("   <-- our test is named in the output" if ours else ""))
        if detail is None:
            continue
        print(f"{name:14s} our test failed in {fails} of {reps}")
        print("\n".join(detail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
