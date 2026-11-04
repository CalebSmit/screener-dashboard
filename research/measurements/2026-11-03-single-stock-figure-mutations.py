"""Mutation check for the price-target strip, the share-class gap and the compare sentence
(2026-11-03, owner-run night). Each mutation breaks one figure's printing; the three tests in
``tests/test_dashboard_printed_figures.py`` must fail on every one. Run from the repo root:
``python research/measurements/2026-11-03-single-stock-figure-mutations.py``. Restores the file byte
for byte, whatever happens.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "generate_dashboard.py"
TESTS = ["tests/test_dashboard_printed_figures.py", "-k", "price_target_strip or share_class_line or compare_sentence"]

MUTATIONS = [
    ("distance measured from the target", "const pct = ((target - cur) / cur * 100);", "const pct = ((target - cur) / target * 100);"),
    ("low and high targets swapped", "${{fmtDollar(ptLow)}} ${{fmtPct(ptLow, price)}}", "${{fmtDollar(ptHigh)}} ${{fmtPct(ptHigh, price)}}"),
    ("prices to the dime", "const fmtDollar = v => v !== null && v !== undefined ? '$' + v.toFixed(2) : '—';",
     "const fmtDollar = v => v !== null && v !== undefined ? '$' + v.toFixed(1) + '0' : '—';"),
    ("share-class gap side reversed", "(sc.price_gap_pct < 0 ? 'below ' : 'above ')", "(sc.price_gap_pct > 0 ? 'below ' : 'above ')"),
    ("compare gap direction reversed", "' composite points ' + (gap >= 0 ? 'above' : 'below')", "' composite points ' + (gap < 0 ? 'above' : 'below')"),
    ("compare names the smallest differences", "const lead = parts.filter(p => Math.abs(p[1]) >= 0.05).slice(0, 2)",
     "const lead = parts.filter(p => Math.abs(p[1]) >= 0.05).slice(-2)"),
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
