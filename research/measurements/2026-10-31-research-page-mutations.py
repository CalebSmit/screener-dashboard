"""Claims-mutation audit, 2026-10-31: the trading rule's monthly figure and its coverage years.

`bot-monthly-excess-figure` and `bot-coverage-by-year`, open since 2026-10-29. Both are figures
on `research.html` whose *words* were pinned and whose numbers were not - the monthly excess was
matched as ``[0-9.]+%``, a wildcard, and the two coverage percentages were read by nothing.

Four mutations of `research_page.py`. The new checks read the **generated** page, so no
regeneration step is needed here: that is the point of reading the generator rather than the
committed file (2026-10-29's structural fix), and it is why a mutation that would have shipped is
visible immediately.

    python research/measurements/2026-10-31-research-page-mutations.py
    python research/measurements/2026-10-31-research-page-mutations.py --rest-of-suite
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path("research_page.py")
MODULE = "tests/test_research_page.py"
NEW = ["tests/test_research_page.py::test_the_trading_rules_monthly_figure_is_the_ledgers_own_number",
       "tests/test_research_page.py::test_the_survivorship_coverage_percentages_are_the_two_years_means"]

MUTATIONS = [
    ("E", "the monthly excess is printed ten times too large",
     'In {lab} it returned <span class="num">{abs(st["mean"]) * 100:.2f}%</span>',
     'In {lab} it returned <span class="num">{abs(st["mean"]) * 1000:.2f}%</span>'),
    ("F", "the monthly excess keeps its sign, so the figure and the word disagree",
     '{abs(st["mean"]) * 100:.2f}%</span> a month {word} that ',
     '{st["mean"] * 100:.2f}%</span> a month {word} that '),
    ("G", "the first year's coverage is the last year's figure",
     "{yc(ys[0]):.0f}", "{yc(ys[-1]):.0f}"),
    ("H", "the coverage percentages are printed as fractions",
     'yc = lambda y: np.mean([v["coverage"] for m, v in mc.items() if m.startswith(y)]) * 100',
     'yc = lambda y: np.mean([v["coverage"] for m, v in mc.items() if m.startswith(y)])'),
]


def run(rest_of_suite: bool) -> str:
    if rest_of_suite:
        cmd = [sys.executable, "scripts/run_tests.py", "tests/", "test_screener.py", "-q",
               "--tb=no", "-rf"]
        for t in NEW:
            cmd += ["--deselect", t]
        # The committed page drifts from any generator edit, so its drift tripwire is not
        # evidence that anything saw the *number* (the audit's 2026-10-30 correction).
        cmd += ["--deselect",
                MODULE + "::test_the_committed_page_is_what_the_generator_makes_from_the_ledger"]
    else:
        cmd = [sys.executable, "-m", "pytest", MODULE, "-q", "--no-header", "--tb=no", "-rf",
               "--deselect",
               MODULE + "::test_the_committed_page_is_what_the_generator_makes_from_the_ledger"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    out = lines[-1].strip() if lines else f"(no summary, exit {r.returncode})"
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
        print("scope:", "whole suite, the two new checks deselected" if rest
              else "tests/test_research_page.py, the drift tripwire deselected")
        print(f"unmutated: {run(rest)}")
        for mid, what, find, replace in MUTATIONS:
            if only and mid not in only:
                continue
            if original.count(find) != 1:
                print(f"SKIPPED {mid}: anchor appears {original.count(find)} times - {what}")
                worst = 1
                continue
            SRC.write_text(original.replace(find, replace), encoding="utf-8")
            summary = run(rest)
            caught = "failed" in summary
            print(f"{('CAUGHT ' if caught else ('BLIND  ' if rest else 'MISSED '))} {mid}: {what}"
                  f"\n           {summary}")
            if not caught and not rest:
                worst = 1
            SRC.write_text(original, encoding="utf-8")
    finally:
        SRC.write_text(original, encoding="utf-8")
        assert SRC.read_text(encoding="utf-8") == original, "failed to restore research_page.py"
        print("research_page.py restored byte-for-byte")
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
