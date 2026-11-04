"""Mutation check for the history and count lines (2026-11-03, owner-run night).

Two mutations survived the first draft on today's data (every current stock ranked today; no report
exactly 8 days out); hand-worked cases for both edges now catch them.

Each mutation breaks one count's printing; the three history tests in
``tests/test_dashboard_printed_figures.py`` must fail on every one. Run from the repo root:
``python research/measurements/2026-11-03-history-count-mutations.py``. Restores the file byte for
byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
TESTS = ["tests/test_dashboard_printed_figures.py", "-k",
         "rank_history_line or caption_and_footnote or reporting_soon or stopped_before_today or window_edges"]

MUTATIONS = [
    ("the run count counts every date", "across ${{pts.length}} comparable runs</span>", "across ${{ser.r.length}} comparable runs</span>"),
    ("the closing rank is the opening rank", "&rarr; <strong>${{last[1]}}</strong>", "&rarr; <strong>${{first[1]}}</strong>"),
    ("the closing date is the last published date", "on ${{escapeHtml(prettyDate(H.dates[last[0]]))}}, across",
     "on ${{escapeHtml(prettyDate(H.dates[H.dates.length - 1]))}}, across"),
    ("'Based on' counts one fewer", "Based on ${{H.dates.length}} comparable runs.", "Based on ${{H.dates.length - 1}} comparable runs."),
    ("the footnote starts at the second date", "comparable runs, ${{escapeHtml(prettyDate(H.dates[0]))}} to",
     "comparable runs, ${{escapeHtml(prettyDate(H.dates[1]))}} to"),
    ("the caption prints observations as pairs", "measured across ${{n.n_pairs}} paired runs",
     "measured across ${{n.n_observations}} paired runs"),
    ("a week becomes eight days", "const a = reportingRows(7, 'all').length", "const a = reportingRows(8, 'all').length"),
    ("reports already past are counted", "if (d === null || d < 0 || d > days) return;", "if (d === null || d > days) return;"),
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
