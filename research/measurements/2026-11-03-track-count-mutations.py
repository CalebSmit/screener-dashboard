"""Mutation check for the track record panel's three counts (2026-11-03, owner-run evening).

Each mutation breaks one count's printing or its source; the three tests added to
``tests/test_dashboard_printed_figures.py`` must fail on every one. Run from the repo root:
``python research/measurements/2026-11-03-track-count-mutations.py``. Restores each file byte for
byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ["tests/test_dashboard_printed_figures.py", "-k",
         "track_panel_prints_its_own or hand_worked_counts or one_per_dated"]

MUTATIONS = [
    ("periods counted from the series", "generate_dashboard.py",
     "Every holding period (' + T.periods.length + ')", "Every holding period (' + (T.series || []).length + ')"),
    ("ticks counted as entries plus one", "generate_dashboard.py",
     "entries on ' + (T.methodology_changes || []).length + ' dates", "entries on ' + ((T.methodology_changes || []).length + 1) + ' dates"),
    ("basket names swapped", "generate_dashboard.py",
     "({ top: 'top-25', q1: 'top-fifth', q5: 'bottom-fifth' })[k] + ' baskets'",
     "({ top: 'top-25', q1: 'bottom-fifth', q5: 'top-fifth' })[k] + ' baskets'"),
    ("left-out count printed as the basket size", "generate_dashboard.py",
     ".map(k => (T.unpriced[k].length) + ' of the ' + picks[k]", ".map(k => picks[k] + ' of the ' + picks[k]"),
    ("an empty basket kept in the clause", "generate_dashboard.py",
     "['top', 'q1', 'q5'].filter(k => T.unpriced && (T.unpriced[k] || []).length && picks[k])",
     "['top', 'q1', 'q5'].filter(k => T.unpriced && picks[k])"),
    ("two entries on one day are two ticks", "track_record.py",
     "        if e[\"date\"] not in seen:", "        if True:"),
]


def main() -> int:
    caught = 0
    for name, rel, old, new in MUTATIONS:
        p = ROOT / rel
        orig = p.read_bytes()
        text = orig.decode("utf-8")
        assert text.count(old) == 1, f"{name}: anchor found {text.count(old)} times in {rel}"
        try:
            p.write_bytes(text.replace(old, new).encode("utf-8"))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                               cwd=ROOT, capture_output=True, text=True)
        finally:
            p.write_bytes(orig)
        hit = r.returncode != 0
        caught += hit
        print(f"{'CAUGHT' if hit else 'MISSED'}  {name}")
    print(f"{caught} of {len(MUTATIONS)} caught")
    return 0 if caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
