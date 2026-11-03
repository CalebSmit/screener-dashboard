"""Every place the dashboard's JavaScript prints a percentage, and whether it can print a
zero for a value that is not zero.

Two defects of one family have reached the live site: an **unsigned** "0.0%" for BXP's close
0.03% below its 200-day average (2026-10-28, the data loop refused the run) and a **signed**
"-0.0%" for AMD's and INTC's five-year earnings-yield range (2026-10-29, fixed in `cPlain`
only) and for NVR's average analyst target (2026-11-03, gate 1 red on the published payload).

Run: python research/measurements/2026-11-03-signed-zero-scan.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = (ROOT / "generate_dashboard.py").read_text(encoding="utf-8").splitlines()

# a toFixed(...) whose result is followed, within a few characters, by a percent sign
NEAR_PCT = re.compile(r"toFixed\([^)]*\)[^\n]{0,12}%")

for i, line in enumerate(SRC, 1):
    if NEAR_PCT.search(line):
        print(f"{i:6d}  {line.strip()[:160]}")
