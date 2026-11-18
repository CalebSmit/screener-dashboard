"""How far are the last fiscal year's flows from the twelve months the companies have filed?

For every stock the published run read from its annual statement (`flags.annual_instead`, or a
refused quarterly statement), compare the net income and operating cash flow the run scored
with the company's own filed twelve months to the newest period end in the SEC cache that is
no later than the balance sheet the same run read. Offline and read-only: the published payload
(HEAD) and `data/sec/pit/facts.parquet`.

This sizes a question; it changes nothing and decides nothing. Scoring may not read a filing
(CLAUDE.md 0.14), and `research/2026-10-29-ttm-fills-the-empty-quarter.md` section 5.5 holds the
pre-registered alternative.

    python research/measurements/2026-11-18-what-the-annual-fallback-leaves-out.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import check_run_health as H  # noqa: E402
import sec_crosscheck as X  # noqa: E402
from sec_fundamentals import FACTS_PATH  # noqa: E402

FLOWS = (("net income", "netIncome", ("NetIncomeLoss", "ProfitLoss")),
         ("operating cash flow", "operatingCashFlow", ("NetCashProvidedByUsedInOperatingActivities",)))
CATS = ("valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment")


def filed_ttm(f_t, concepts, when):
    for c in concepts:
        f = f_t[f_t["concept"] == c]
        if f.empty:
            continue
        ttm, _ = X.flow_ttm(f, when)
        if ttm is not None:
            return ttm
    return None


if __name__ == "__main__":
    D = H.published_payload_whole("HEAD")
    sd = D["stock_detail"]
    rank = {r["Ticker"]: r["Rank"] for r in D["table_data"]}
    facts = pd.read_parquet(FACTS_PATH)
    facts = facts[facts["form"].isin(X.FORMS)]
    by = dict(tuple(facts.groupby("ticker")))

    pool = sorted((t for t, s in sd.items()
                   if (s.get("flags") or {}).get("annual_instead") or (s.get("flags") or {}).get("q_rejected")),
                  key=lambda t: rank.get(t, 999))
    print(f"run {D['kpis']['run_timestamp'][:10]}: {len(pool)} stocks read from the annual statement\n")
    print("ticker rank  statement   filed to    net income: scored / filed 12m      op. cash flow: scored / filed 12m")
    gaps = {name: [] for name, _, _ in FLOWS}
    for t in pool:
        s, f_t = sd[t], by.get(t)
        asof = s.get("asof") or {}
        line = f"{t:6} {rank.get(t, 0):4}  {asof.get('is', '-'):10}  "
        if f_t is None or not asof.get("bs"):
            print(line + "no filings in the cache")
            continue
        when = pd.Timestamp(asof["bs"])
        line += f"{asof['bs']:10}  "
        for name, key, concepts in FLOWS:
            y = (s.get("inp") or {}).get(key)
            ttm = filed_ttm(f_t, concepts, when)
            if y is None or ttm is None or abs(ttm) < 1.0:
                line += f"{'-':>34}  "
                continue
            gap = (y - ttm) / abs(ttm)
            gaps[name].append((t, gap))
            line += f"{y / 1e9:9.3f}bn /{ttm / 1e9:9.3f}bn ({gap:+7.1%})  "
        print(line)

    print()
    for name, rows in gaps.items():
        g = [x for _, x in rows]
        a = [abs(x) for x in g]
        print(f"{name}: {len(g)} comparable. scored below the filed twelve months for "
              f"{sum(1 for x in g if x < -0.005)}, above for {sum(1 for x in g if x > 0.005)}, "
              f"within 0.5% for {sum(1 for x in a if x <= 0.005)}; median gap {statistics.median(g):+.1%}, "
              f"median |gap| {statistics.median(a):.1%}; over 10%: {sum(1 for x in a if x > 0.10)}, "
              f"over 25%: {sum(1 for x in a if x > 0.25)}")

    # Where these stocks sit, category by category, against everyone else (mean score).
    rest = [t for t in sd if t not in set(pool)]
    print(f"\nmean category score: the {len(pool)} / the other {len(rest)}")
    for c in CATS:
        a = [sd[t]["cat_scores"][c] for t in pool if isinstance(sd[t]["cat_scores"].get(c), (int, float))]
        b = [sd[t]["cat_scores"][c] for t in rest if isinstance(sd[t]["cat_scores"].get(c), (int, float))]
        se = (statistics.pvariance(a) / len(a) + statistics.pvariance(b) / len(b)) ** 0.5
        print(f"  {c:11} {statistics.mean(a):5.1f} / {statistics.mean(b):5.1f}   "
              f"difference {statistics.mean(a) - statistics.mean(b):+5.1f} ({(statistics.mean(a) - statistics.mean(b)) / se:+.1f} se)")
    ranks = [rank[t] for t in pool]
    print(f"  rank: median {statistics.median(ranks):.0f}, mean {statistics.mean(ranks):.0f} "
          f"(a random 27 of 499 would average 250, se about {144 / len(ranks) ** 0.5:.0f})")
