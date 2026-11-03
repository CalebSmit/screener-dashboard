"""How many percentage lines on today's page print a zero, split by sign.

A signed zero ("-0.0%") is a value the page could not format - the defect
`tests/test_every_panel_reads_clean.py` fails on. An unsigned zero ("0.0%") is only a defect
where the underlying value is not zero, which this count cannot tell from the text; it is
reported as an upper bound.

    python -X utf8 research/measurements/2026-11-03-count-zero-percents.py /tmp/pct-before.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

SIGNED = re.compile(r"[-−]0\.0+%")
UNSIGNED = re.compile(r"(?<![-−\d.])0\.0+%")

lines = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
signed: Counter[str] = Counter()
unsigned = 0
sig_examples: list[tuple[str, str]] = []
for t, ls in lines.items():
    for l in ls:
        if SIGNED.search(l):
            signed[t] += 1
            if len(sig_examples) < 10:
                sig_examples.append((t, l))
        unsigned += len(UNSIGNED.findall(l))

print(f"stocks: {len(lines)}  percentage lines: {sum(len(v) for v in lines.values())}")
print(f"signed zeros: {sum(signed.values())} on {len(signed)} stocks")
for t, l in sig_examples:
    print(f"   {t}: {l[:120]}")
print(f"unsigned '0.0%' printings (upper bound on the BXP direction): {unsigned}")
