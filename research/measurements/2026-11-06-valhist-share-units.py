"""How far does a past month's filed share count sit from today's, across the published universe?

2026-11-06 (owner-run night). The public "Against its own five years" panel printed HST's five-year
earnings-yield range as -84764.2% to 5455967.6%, and nine more stocks' ranges were impossible
(EG, COP, CCL, GRMN, WRB, EXE, TER, CHD, BRO). `valuation_history.for_ticker` checks today's
share count against the provider's market value (15%) but uses every past month's count unchecked,
so a filing that tags shares in thousands makes that month's market value 1,000x too small.

This measures the ratio  month's split-adjusted shares / today's shares  for every stock and
month-end the panel uses, to find where genuine change (buybacks, issuance, mergers) ends and
unit errors begin. Read-only; uses the caches the data run already wrote.

    python research/measurements/2026-11-06-valhist-share-units.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import valuation_history as vh  # noqa: E402
from sec_fundamentals import FACTS_PATH  # noqa: E402


def main() -> None:
    facts = pd.read_parquet(FACTS_PATH)
    monthly = pd.read_parquet(vh.PRICE_PATH)
    splits = json.loads(vh.SPLIT_PATH.read_text())
    s = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    D = json.loads(s[s.index("{"): s.rindex("}") + 1])
    today = pd.Timestamp(D["kpis"]["run_timestamp"][:10])
    td = vh._day(today)
    cur = today.to_period("M").to_timestamp()
    monthly = monthly[monthly.index < cur]
    by = {t: g for t, g in facts.groupby("ticker")}
    ratios, worst = [], []
    for t in D["stock_detail"]:
        if t not in by or t not in monthly.columns:
            continue
        sh = vh.share_series(by[t])
        sp = splits.get(t, [])
        now = vh._shares_as_of(sh, td, sp)
        if not now:
            continue
        closes = monthly[t].dropna()
        for m in closes.index[-vh.MONTHS:]:
            x = vh._shares_as_of(sh, vh._day(m + pd.offsets.MonthEnd(0)), sp)
            if x:
                r = x / now
                ratios.append(r)
                worst.append((abs(np.log10(r)), t, str(m.date()), r))
    a = np.array(ratios)
    lg = np.abs(np.log10(a))
    print(f"{len(a)} stock-months over {len({w[1] for w in worst})} stocks")
    for f in (1.5, 2, 3, 4, 5, 10, 100, 1000):
        print(f"  ratio outside [1/{f}, {f}]: {int((lg > np.log10(f)).sum())} stock-months, "
              f"{len({w[1] for w in worst if w[0] > np.log10(f)})} stocks")
    worst.sort(reverse=True)
    print("largest genuine-looking departures (below 10x):")
    for w in [w for w in worst if w[0] < 1][:12]:
        print(f"  {w[1]:6s} {w[2]}  ratio {w[3]:.3f}")
    print("every departure of 10x or more:")
    seen = {}
    for w in worst:
        if w[0] >= 1:
            seen.setdefault(w[1], []).append((w[2], round(w[3], 6)))
    for t, ms in seen.items():
        print(f"  {t}: {len(ms)} months, e.g. {ms[:2]}")


if __name__ == "__main__":
    main()


def effect() -> None:
    """With the bound in place: rebuild every block from the same caches and compare."""
    facts = pd.read_parquet(FACTS_PATH)
    monthly = pd.read_parquet(vh.PRICE_PATH)
    splits = json.loads(vh.SPLIT_PATH.read_text())
    s = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    D = json.loads(s[s.index("{"): s.rindex("}") + 1])
    c = (ROOT / "dashboard_context.js").read_text(encoding="utf-8")
    C = json.loads(c[c.index("{"): c.rindex("}") + 1])["ctx"]
    rows = [{"Ticker": t, "currentPrice": d.get("price"), "marketCap": (d.get("inp") or {}).get("marketCap")}
            for t, d in D["stock_detail"].items()]
    rows = [r for r in rows if r["marketCap"]]
    out = vh.build(rows, facts, monthly, splits, today=pd.Timestamp(D["kpis"]["run_timestamp"][:10]).date())
    pub = {t: v["vh"] for t, v in C.items() if "vh" in v}
    absurd_before = [t for t, b in pub.items() if any(abs(b.get(k, {}).get(x, 0)) > 1 for k in ("ey", "fy") for x in ("lo", "hi"))]
    absurd_after = [t for t, b in out.items() if any(abs(b.get(k, {}).get(x, 0)) > 1 for k in ("ey", "fy") for x in ("lo", "hi"))]
    print(f"blocks published {len(pub)}, rebuilt {len(out)}; absurd ranges before {len(absurd_before)} "
          f"{sorted(absurd_before)}, after {len(absurd_after)} {sorted(absurd_after)}")
    print("lost a block:", sorted(set(pub) - set(out)), " gained:", sorted(set(out) - set(pub)))
    moved = [(t, k, pub[t][k]["pct"], out[t][k]["pct"]) for t in set(pub) & set(out) for k in ("ey", "fy")
             if k in pub[t] and k in out[t] and abs(pub[t][k]["pct"] - out[t][k]["pct"]) > 0.05]
    print(f"blocks whose own-history percentile moved: {len(moved)}", sorted(moved)[:20])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "effect":
    effect()
