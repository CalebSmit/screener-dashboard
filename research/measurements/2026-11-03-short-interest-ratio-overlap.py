"""What `short_interest_ratio` (days to cover, lower = better) overlaps with, and what rides on it.

For the 2026-11-03 research note on the Revisions category's last unresearched leg. From the
published payload only (sector percentiles, direction-adjusted, so 100 = fewest days to cover):

1. coverage, and the raw distribution;
2. Spearman correlation of its sector percentile with every other weighted metric's, and with each
   category score - what else the 1.35 composite points are a bet on;
3. what removing it would move: the Revisions score recomputed without it (the engine's renormalised
   weighted mean over metrics with data, using the stock's own weight table), the composite
   recomputed with the published category weights and coverage discount, then re-ranked the engine's
   way (round to 2 dp, ties share the lower rank).

    python research/measurements/2026-11-03-short-interest-ratio-overlap.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
M = "short_interest_ratio"


def main() -> int:
    s = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    D = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    sd, W = D["stock_detail"], D["weights"]
    raw = pd.Series({t: (v.get("raw") or {}).get(M) for t, v in sd.items()}, dtype="float64")
    pct = pd.DataFrame({t: v.get("pct") or {} for t, v in sd.items()}).T
    print(f"coverage: {raw.notna().sum()} of {len(sd)}; days to cover median {raw.median():.2f}, "
          f"10th-90th {raw.quantile(0.1):.2f}-{raw.quantile(0.9):.2f}")

    weighted = sorted({m for cat in W["metric_weights"].values() for m, w in cat.items() if w and w > 0})
    corr = {}
    for m in weighted:
        if m == M or m not in pct.columns:
            continue
        both = pct[[M, m]].dropna()
        if len(both) > 100:
            corr[m] = both[M].rank().corr(both[m].rank())
    cats = pd.DataFrame({t: v.get("cat_scores") or {} for t, v in sd.items()}).T
    for c in CATS:
        both = pd.concat([pct[M], cats[c]], axis=1).dropna()
        corr[f"[{c} score]"] = both.iloc[:, 0].rank().corr(both.iloc[:, 1].rank())
    print("Spearman with its sector percentile (largest first):")
    for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:12]:
        print(f"   {k:28s} {v:+.3f}")

    # what rides on it
    fw = W["factor_weights"]
    rev_prof = W["profiles"]["revisions"]
    old, new = {}, {}
    for t, v in sd.items():
        p = v.get("pct") or {}
        table = rev_prof.get((v.get("wp") or {}).get("revisions") or "generic") or rev_prof["generic"]

        def score(drop):
            have = {m: w for m, w in table.items() if w and w > 0 and p.get(m) is not None and m != drop}
            return sum(p[m] * w for m, w in have.items()) / sum(have.values()) if have else None

        sc = dict(v.get("cat_scores") or {})
        for store, drop in ((old, None), (new, M)):
            cs = dict(sc)
            cs["revisions"] = score(drop)
            have = {c: cs[c] for c in CATS if cs.get(c) is not None and fw.get(c, 0) > 0}
            tot = sum(fw[c] for c in have)
            disc = (v.get("cov") or {}).get("disc") or 0.0
            store[t] = round(sum(have[c] * fw[c] for c in have) / tot * (1 - disc), 2) if tot else None

    def rank(d):
        sr = pd.Series({k: x for k, x in d.items() if x is not None})
        return sr.rank(ascending=False, method="min").astype(int)

    r0, r1 = rank(old), rank(new)
    pub = {r["Ticker"]: r["Rank"] for r in D["table_data"]}
    assert all(r0[t] == pub[t] for t in pub), "the recomputation does not reproduce the published ranks"
    moves = (r1 - r0).abs()
    top0, top1 = set(r0[r0 <= 25].index), set(r1[r1 <= 25].index)
    print(f"without it: the published ranks reproduce first (checked); median rank move {moves.median():.0f}, "
          f"90th pct {moves.quantile(0.9):.0f}, max {moves.max()}; top 25 changes by {len(top0 - top1)} name(s): "
          f"{sorted(top0 - top1)} out, {sorted(top1 - top0)} in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
