"""How wrong is "its composite of X is a percentile: it scores above X% of the
universe" (stock_summary._sentence_rank)?

Measured 2026-10-06 on the live payload (plan/calculation-transparency.md, Defect 3):
median gap 19.6 percentage points between the claimed share and the share of the
universe the stock actually ranks ahead of; >10 points for 75.1% of stocks; 31 points
at worst; and the stock ranked 1st of 502 is told it beats 74%.
Cause: Composite has been the cardinal weighted average since Phase 13; the percentile
is the separate Composite_Pct.
Run:  python research/measurements/2026-10-06-rank-sentence-claim.py
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
txt = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
D = json.loads(txt[txt.index("{"):].rstrip().rstrip(";"))
sd = D["stock_detail"]
n = len(sd)
diffs = []
for t, s in sd.items():
    comp, rank = s["composite"], s["rank"]
    if comp is None or rank is None:
        continue
    claimed = round(comp)  # "scores above {composite:.0f}% of the universe"
    actual = 100.0 * (n - rank) / (n - 1)  # share of the universe it actually beats
    diffs.append((abs(claimed - actual), t, rank, comp, claimed, round(actual, 1)))
diffs.sort()
vals = [d[0] for d in diffs]
print("stocks:", len(diffs))
print("median gap (pct points):", round(vals[len(vals) // 2], 1))
print("share off by more than 5 points:", round(100 * sum(v > 5 for v in vals) / len(vals), 1), "%")
print("share off by more than 10 points:", round(100 * sum(v > 10 for v in vals) / len(vals), 1), "%")
print("largest 3:", diffs[-3:])
print("smallest 3:", diffs[:3])
for t in ("JPM", "EXPE"):
    s = sd[t]
    m = [x["t"] for x in s["summary"] if x["k"] == "rank"]
    print(t, "rank", s["rank"], "composite", s["composite"], "|", m)
