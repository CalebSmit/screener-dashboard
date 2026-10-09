"""How far the 2026-10-09 owner-run changes moved the ranking: the evening run against the 02:00 run.

Both data and method changed between the two runs (a fresh fetch after the close, and every
methodology change of the day), so this measures the combined effect; the per-change figures are in
each changelog entry. Usage: python 2026-10-09-day-effect.py <evening_run_id> [<morning_run_id>]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
evening = sys.argv[1]
morning = sys.argv[2] if len(sys.argv) > 2 else "a2d76219dc0a"


def load(run):
    d = pd.read_parquet(ROOT / "runs" / run / "05_final_scored.parquet")
    return d.set_index("Ticker")


a, b = load(morning), load(evening)
common = a.index.intersection(b.index)
a, b = a.loc[common], b.loc[common]
print(f"stocks in both: {len(common)}")
print(f"Spearman of composite ranks: {a['Rank'].corr(b['Rank'], method='spearman'):.3f}")
for n in (5, 25, 50):
    ta, tb = set(a.nsmallest(n, 'Rank').index), set(b.nsmallest(n, 'Rank').index)
    print(f"top {n}: {len(ta & tb)} in both")
mv = (b["Rank"] - a["Rank"]).abs()
print(f"rank moves: median {mv.median():.0f}, 90th pct {mv.quantile(.9):.0f}, >=50 places: {(mv >= 50).sum()}")
cats = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
for c in cats:
    col = f"{c}_score"
    if col in a.columns and col in b.columns:
        print(f"  {c:<11} score Spearman {a[col].corr(b[col], method='spearman'):.3f}")
for flag in ("Value_Trap_Flag", "Growth_Trap_Flag"):
    if flag in a.columns and flag in b.columns:
        print(f"{flag}: {int(a[flag].sum())} -> {int(b[flag].sum())}")
d = (b["Rank"] - a["Rank"]).sort_values()
print("largest rises:", ", ".join(f"{t} {int(a.loc[t, 'Rank'])}->{int(b.loc[t, 'Rank'])}" for t in d.index[:8]))
print("largest falls:", ", ".join(f"{t} {int(a.loc[t, 'Rank'])}->{int(b.loc[t, 'Rank'])}" for t in d.index[-8:][::-1]))
