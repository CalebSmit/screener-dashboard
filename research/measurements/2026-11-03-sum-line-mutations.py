"""Mutation check for the workings' sum-of-parts lines (2026-11-03, owner-run night).

Each mutation breaks one of the lines that show how a Piotroski score, the three earnings-surprise
metrics and earnings variability are built from the inputs printed beside them;
``test_every_sum_of_parts_line_redoes_its_own_arithmetic`` must fail on every one. Run from the repo
root: ``python research/measurements/2026-11-03-sum-line-mutations.py``. Restores the file byte for
byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
TESTS = ["tests/test_dashboard_printed_figures.py", "-k", "sum_of_parts"]

MUTATIONS = [
    ("F-score line counts the fails", "`</ol><div class=\"wk-sum\">${sig.filter(c => c === '1').length} passed",
     "`</ol><div class=\"wk-sum\">${sig.filter(c => c === '0').length} passed"),
    ("F-score row line counts every signal as testable",
     "sig.filter(c => c !== '-').length + '</span> <span class=\"wk-l\">testable signals passed",
     "sig.length + '</span> <span class=\"wk-l\">testable signals passed"),
    ("the median counts every quarter", "sum = `Median of the ${vs.length} valid surprises",
     "sum = `Median of the ${q.length} valid surprises"),
    ("the change subtracts the wrong way round",
     "sum = `Latest surprise ${sp(q[L - 1][3])} &minus; the quarter before ${sp(q[L - 2][3])}",
     "sum = `Latest surprise ${sp(q[L - 2][3])} &minus; the quarter before ${sp(q[L - 1][3])}"),
    ("beat weights counted from the oldest", "const pos = q.map((r, i) => [4 - (L - 1 - i), r[3]])",
     "const pos = q.map((r, i) => [i + 1 + (4 - L) * 0 + (L - 1 - i) * 0 + 0, r[3]]).reverse()"),
    ("the quarter table header miscounts", "h += `<div class=\"wk-k\">The last ${q.length} quarters, oldest first</div>",
     "h += `<div class=\"wk-k\">The last ${q.length + 1} quarters, oldest first</div>"),
    ("the ROE line prints another metric", "'Sample standard deviation of the five ROEs = ' + fmtMetric(s.raw[m], meta.fmt)",
     "'Sample standard deviation of the five ROEs = ' + fmtMetric(s.raw.roic, meta.fmt)"),
]


def main() -> int:
    orig = GEN.read_bytes()
    text = orig.decode("utf-8")
    caught = 0
    try:
        for name, old, new in MUTATIONS:
            assert text.count(old) == 1, f"{name}: anchor found {text.count(old)} times"
            GEN.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                               cwd=ROOT, capture_output=True, text=True)
            hit = r.returncode != 0
            caught += hit
            print(f"{'CAUGHT' if hit else 'MISSED'}  {name}")
    finally:
        GEN.write_bytes(orig)
    print(f"{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
