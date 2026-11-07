"""Does `tests/test_per_band_gates.py` have teeth? Six mutations, measured.

WHY. `plan/claims-mutation-audit.md` has found the same defect nine times: a check that reads
as coverage and is not. A test module written the same day as its feature is the easiest place
for that to happen, because every assertion was written looking at the code it is asserting on.

So each mutation below breaks the production rule in a way a reader could plausibly ship, and
the script reports **which tests fail** - and, for the two that matter most, whether anything
*outside* this module would have noticed. A mutation that nothing catches is a gap to close.

It patches tracked files in place and restores them with `git checkout --` in a `finally`, so
an interrupted run leaves the tree clean. Nothing published is touched.

Run: `python research/measurements/2026-11-07-per-band-gate-mutations.py`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GATE = ROOT / "scripts" / "check_run_health.py"
SCREENER = ROOT / "run_screener.py"
MODULE = "tests/test_per_band_gates.py"

# (name, file, old, new, what a reader would think they were doing)
MUTATIONS = [
    ("the feature is a no-op", GATE,
     '    detail = payload.get("stock_detail") or {}\n    if not detail:\n        return\n'
     '    if not bands:',
     '    detail = payload.get("stock_detail") or {}\n    if True:\n        return\n'
     '    if not bands:',
     "an early return that looks like a guard"),

    ("the floor becomes exclusive", GATE,
     "        if cov < MIN_TARGET_COVERAGE:",
     "        if cov <= MIN_TARGET_COVERAGE:",
     "a band exactly at 50% starts failing"),

    ("every run takes the one-band door", GATE,
     "    if len(by_band) == 1:",
     "    if len(by_band) >= 1:",
     "the check reports and never judges"),

    ("blank bands are kept as bands", GATE,
     '    ok = col.notna() & ~col.isin(("", "nan", "None"))',
     "    ok = col.notna()",
     "a NaN band becomes the string 'nan', its own band"),

    ("the per-band figure is the universe figure", SCREENER,
     "    return {str(band): weighted_coverage_pct(g, cfg, cat)\n"
     "            for band, g in df.groupby(universe.SIZE_BAND_COL, dropna=False)}",
     "    return {str(band): weighted_coverage_pct(df, cfg, cat)\n"
     "            for band, g in df.groupby(universe.SIZE_BAND_COL, dropna=False)}",
     "`df` instead of `g` - one character, and the report is the mean in every band"),

    ("the auto-disable quietly goes per band", SCREENER,
     "def auto_disable_sparse_category(df: pd.DataFrame, cfg: dict, cat: str,\n"
     "                                 threshold: float = CATEGORY_AUTO_DISABLE_PCT)",
     "def auto_disable_sparse_category(df: pd.DataFrame, cfg: dict, cat: str,\n"
     "                                 threshold: float = CATEGORY_AUTO_DISABLE_PCT,\n"
     "                                 by_band: bool = False)",
     "a methodology change arriving as a keyword argument"),
]


def pytest(*target: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", *target, "-q", "--no-header",
                        "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def restore() -> None:
    subprocess.run(["git", "checkout", "--",
                    str(GATE.relative_to(ROOT)), str(SCREENER.relative_to(ROOT))],
                   cwd=ROOT, check=False)


def failing(out: str) -> list[str]:
    return sorted({ln.split(" ")[1].split("::")[-1].split("[")[0]
                   for ln in out.splitlines() if ln.startswith("FAILED ")})


def main() -> int:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(GATE), str(SCREENER)],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("REFUSING: the two files are already modified; commit first so restore is exact")
        print(dirty)
        return 2

    code, out = pytest(MODULE)
    print(f"unmutated: exit {code} - {out.strip().splitlines()[-1]}")
    if code != 0:
        print("REFUSING: the module is red before any mutation")
        return 2

    caught = 0
    try:
        for name, path, old, new, why in MUTATIONS:
            text = path.read_text(encoding="utf-8")
            n = text.count(old)
            if n != 1:
                print(f"\n!! {name}: anchor found {n} times, not 1 - mutation not applied")
                continue
            path.write_text(text.replace(old, new), encoding="utf-8")
            code, out = pytest(MODULE)
            names = failing(out)
            verdict = "CAUGHT" if code != 0 else "**GAP**"
            caught += code != 0
            print(f"\n{verdict}  {name}")
            print(f"        ({why})")
            print(f"        {out.strip().splitlines()[-1]}")
            for t in names[:6]:
                print(f"        - {t}")
            restore()
    finally:
        restore()

    print(f"\n{caught} of {len(MUTATIONS)} caught by {MODULE}")

    # The question this project asks of every new check: before today, would *anything* have
    # noticed? The whole suite, with the new module deselected, against the mutation that
    # matters most - the feature being a no-op. "Invisible" is the expected answer for a
    # feature nothing else reads, and it is the number worth writing down rather than assuming.
    name, path, old, new, _why = MUTATIONS[0]
    print(f"\nWould anything else have caught {name!r}? Whole suite, "
          f"{MODULE} deselected (~4 min):")
    try:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace(old, new), encoding="utf-8")
        code, out = pytest("tests/", "test_screener.py", f"--ignore={MODULE}")
        print(f"  exit {code} - {out.strip().splitlines()[-1]}")
        for t in failing(out)[:10]:
            print(f"    - {t}")
    finally:
        restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
