"""Read-out of ``2026-11-19-engine-share-counts.py --evaluate``'s CSV: the checks on the
evaluation itself, and what the candidate rule leaves. Read-only.

1. Is the emulation of the shipped rule exact? The CSV's price is cached market value / emulated
   count; compared here with the engine's own price (``cache/v2_daily.parquet`` close at the
   month-end). If the emulation were off, every candidate market value would be too.
2. The name-months the candidate still puts over $4T - real, or a slip?
3. The jumps of more than 5x the candidate leaves.
4. Where the candidate says "unknown" because its sources disagree.
5. The named cases from the morning's list, shipped against candidate, newest month.
6. How far the scored size and the market-value ratios move: name-months whose market value
   changes by more than 10%, 2x, 10x.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

VERSION = 3 if "--version3" in sys.argv else 2 if "--version2" in sys.argv else 1   # which candidate
d = pd.read_csv(ROOT / "logs" / ("2026-11-19-engine-share-counts.evaluate-v3.csv" if VERSION == 3
                                 else "2026-11-19-engine-share-counts.evaluate.csv"))
if VERSION >= 2:
    d = d.drop(columns=["cand", "why", "cap_cand"]).rename(
        columns={"cand2": "cand", "why2": "why", "cap_cand2": "cap_cand"})
print(f"candidate rule version {VERSION}")

daily = pd.read_parquet(ROOT / "cache" / "v2_daily.parquet", columns=["ticker", "date", "close"])
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.dropna(subset=["close"]).sort_values(["ticker", "date"])
d["ts"] = pd.to_datetime(d["month"])
m = pd.merge_asof(d.sort_values("ts"), daily.rename(columns={"ticker": "tk"}).sort_values("date"),
                  left_on="ts", right_on="date", left_by="ticker", right_by="tk", direction="backward")
ok = m["price"].notna() & m["close"].notna()
rel = (m.loc[ok, "price"] / m.loc[ok, "close"] - 1).abs()
print(f"1. emulation: {int(ok.sum()):,} name-months with both prices; implied price within 0.1% of the "
      f"engine's close: {int((rel <= 0.001).sum()):,} ({(rel <= 0.001).mean() * 100:.2f}%); "
      f"off by more than 1%: {int((rel > 0.01).sum()):,}")
bad = m.loc[ok][rel > 0.01]
print("   tickers behind the mismatches:", bad["ticker"].value_counts().head(12).to_dict())

hi = d[d["cap_cand"] > 4e12]
print("\n2. candidate over $4T:")
print(hi.groupby("ticker").agg(n=("month", "count"), first=("month", "min"), last=("month", "max"),
                               max_T=("cap_cand", lambda s: round(s.max() / 1e12, 2))).to_string())

d = d.sort_values(["ticker", "month"])
j = d.groupby("ticker")["cand"].transform(lambda s: s / s.shift(1))
print("\n3. candidate jumps over 5x:")
print(d.assign(jump=j)[(j > 5) | (j < 0.2)][["ticker", "month", "cur", "cand", "why", "jump"]].to_string())
j15 = d.assign(jump=j)[((j > 1.5) | (j < 1 / 1.5))]
print("   jumps over 1.5x, by ticker:", j15["ticker"].value_counts().head(45).to_dict())

dis = d[d["why"] == "disagree"]
print(f"\n4. sources disagree -> unknown: {len(dis)} name-months, {dis['ticker'].nunique()} tickers")
print("  ", dis.groupby("ticker")["month"].agg(["count", "min", "max"]).sort_values("count", ascending=False).head(30).to_string().replace("\n", "\n   "))
fl = d[d["cand"].isna() & (d["why"] != "disagree") & (d["why"] != "none")]
print(f"   under the $50M backstop: {len(fl)} name-months: {fl['ticker'].value_counts().to_dict()}")

print("\n5. named cases, newest month each has (market value $B, shipped -> candidate):")
for t in ("BRK-B", "SPG", "FOX", "FOXA", "BKR", "SYK", "PSA", "ROP", "LH", "VTRS", "ETN", "ICE", "AMCR",
          "CMG", "RMD", "SW", "CME", "MA", "V", "CMCSA", "NVDA", "GRMN", "PKG"):
    g = d[d["ticker"] == t]
    if g.empty:
        print(f"   {t}: not in the months"); continue
    r = g.iloc[-1]
    w = g[(g["cap_cached"] < 1e8) | (g["cap_cached"] > 4e12)]
    worst = w.iloc[len(w) // 2] if len(w) else None
    print(f"   {t:6s} {r['month']}: {r['cap_cached'] / 1e9:12.3f} -> {r['cap_cand'] / 1e9:10.1f} ({r['why']})"
          + (f" | worst month {worst['month']}: {worst['cap_cached'] / 1e9:.6f} -> {worst['cap_cand'] / 1e9:.1f} ({worst['why']})" if worst is not None else ""))
b = d[d["ticker"] == "BRK-B"]
print("   BRK-B by 'why':", b["why"].value_counts().to_dict(), "| median candidate $B:", round(b["cap_cand"].median() / 1e9, 1))

both = d[d["cap_cached"].notna() & d["cap_cand"].notna()]
q = both["cap_cand"] / both["cap_cached"]
print(f"\n6. of {len(both):,} name-months with a market value under both rules: moved by more than 10%: "
      f"{int(((q - 1).abs() > 0.10).sum()):,} ({((q - 1).abs() > 0.10).mean() * 100:.2f}%), "
      f"{both.loc[(q - 1).abs() > 0.10, 'ticker'].nunique()} tickers; by more than 2x: "
      f"{int(((q > 2) | (q < 0.5)).sum()):,}; by more than 10x: {int(((q > 10) | (q < 0.1)).sum()):,}")
print(f"   gains a market value it did not have: {int((d['cap_cached'].isna() & d['cand'].notna()).sum()):,} (no price in the CSV for these, so no value shown); "
      f"loses one: {int((d['cap_cached'].notna() & d['cap_cand'].isna()).sum()):,}")
lost = d[d["cap_cached"].notna() & d["cap_cand"].isna()]
print("   the name-months that lose one, by ticker and by reason:",
      lost["ticker"].value_counts().to_dict(), lost["why"].value_counts().to_dict())
yr = both.assign(year=both["month"].str[:4], moved=(q - 1).abs() > 0.10).groupby("year")["moved"].agg(["sum", "count"])
yr["pct"] = (yr["sum"] / yr["count"] * 100).round(2)
print(yr.T.to_string())
