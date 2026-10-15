"""What the captives tag about their lease books - the concept names, from the saved facts.

A helper for the capex-proposal review: lists every us-gaap concept whose name mentions leases,
with its most recent annual value, so the review's arithmetic uses the tags these filers actually
use rather than guessed names. Offline: reads the frame saved by
2026-10-15-capex-proposal-review.py --out.

    python research/measurements/2026-10-15-captive-concepts.py data/tmp_capex_review/captive_facts.parquet [PATTERN]
"""
from __future__ import annotations

import sys

import pandas as pd

path = sys.argv[1]
pattern = sys.argv[2] if len(sys.argv) > 2 else "Lease|Depreciation|OperatingActivities"
facts = pd.read_parquet(path)
facts = facts[facts["start"].notna()]
dur = (facts["end"] - facts["start"]).dt.days
facts = facts[(dur >= 350) & (dur <= 380)]          # annual durations only
for t, g in facts.groupby("ticker"):
    hits = g[g["concept"].str.contains(pattern, case=False, regex=True)]
    if hits.empty:
        continue
    print(f"=== {t}")
    latest = hits["end"].max()
    for c, h in hits[hits["end"] >= latest - pd.Timedelta(days=400)].groupby("concept"):
        row = h.sort_values(["end", "filed"]).iloc[-1]
        print(f"  {c:<80} {row['end'].date()}  ${row['val']/1e9:>9,.2f}B")
    print()
