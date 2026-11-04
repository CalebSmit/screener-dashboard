"""Mutation check for the workings' weight lines and points equation (2026-11-03, owner-run night).

The first draft of the test rebuilt the weight line from its own copy of the template, and the
sixth mutation below survived it - a check of itself, not of the page. It now reads the title the
page rendered on each weight cell.

Each mutation breaks one step of ``categoryWorkings`` or the lines printed from it; the two workings
tests in ``tests/test_dashboard_printed_figures.py`` must fail on every one. Run from the repo root:
``python research/measurements/2026-11-03-workings-line-mutations.py``. Restores the file byte for
byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
TESTS = ["tests/test_dashboard_printed_figures.py", "-k", "workings_row"]

MUTATIONS = [
    ("share over every weighted metric, not those with data",
     "if (p !== null && p !== undefined) { total += table[m]; withData++; }",
     "total += table[m]; if (p !== null && p !== undefined) { withData++; }"),
    ("points from the configured weight", "points: has && share !== null ? p * share / 100 : null",
     "points: has && share !== null ? p * w / 100 : null"),
    ("the equation prints the configured weight",
     "<span class=\"wk-n\">${fmtWeight(r.share)}</span>", "<span class=\"wk-n\">${fmtWeight(r.configured)}</span>"),
    ("weights round to whole numbers", "return (Math.abs(w - Math.round(w)) < 0.05 ? Math.round(w) : w.toFixed(1)) + '%';",
     "return Math.round(w) + '%';"),
    ("the stock's own profile ignored", "const pid = (s.wp && s.wp[cat]) || 'generic';", "const pid = 'generic';"),
    ("the rescaled line names the configured weight twice",
     "${fmtWeight(r.share)} of this score after rescaling", "${fmtWeight(r.configured)} of this score after rescaling"),
    ("the no-data line prints the share", ": `Configured ${fmtWeight(r.configured)}; no data for this stock",
     ": `Configured ${fmtWeight(r.share)}; no data for this stock"),
]


def main() -> int:
    orig = GEN.read_bytes()
    text = orig.decode("utf-8")
    caught = 0
    try:
        for name, old, new in MUTATIONS:
            n = text.count(old)
            assert n >= 1, f"{name}: anchor not found"
            GEN.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                               cwd=ROOT, capture_output=True, text=True)
            hit = r.returncode != 0
            caught += hit
            print(f"{'CAUGHT' if hit else 'MISSED'}  {name}" + (f" ({n} sites)" if n > 1 else ""))
    finally:
        GEN.write_bytes(orig)
    print(f"{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
