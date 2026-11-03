"""The lines of the generated page that build a percentage without the near-zero rule.

The companion to `tests/test_percent_near_zero.py`: it reads the page as published, so a
formatter is judged where it ends up rather than in whichever f-string wrote it. Run it to build
or re-check that module's EXEMPT list.

    python -X utf8 research/measurements/2026-11-03-unexplained-percents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import generate_dashboard as gd  # noqa: E402
from test_percent_near_zero import EXEMPT, PCT_SHAPE  # noqa: E402

html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
lines = html.splitlines()

n_rule = n_exempt = 0
for line in lines:
    if not PCT_SHAPE.search(line):
        continue
    if "pctNearZero" in line or "pctPlain" in line:
        n_rule += 1
        continue
    if any(frag in line for frag in EXEMPT):
        n_exempt += 1
        continue
    print(line.strip()[:200])
    print()

print(f"[{n_rule} lines through the rule, {n_exempt} exempt]")
dead = [f for f in EXEMPT if not any(f in l for l in lines)]
if dead:
    print(f"\nEXEMPT fragments matching no line ({len(dead)}):")
    for f in dead:
        print(f"  {f!r}")
