"""How many stocks carry last fiscal year's flows under the newest quarter's date?

`2026-11-17-ocf-off-the-filing.py` showed the mechanism on ten stocks: the source's quarterly
cash-flow statement lacks one of the four most recent quarters, nothing can be summed or rolled
forward, the fetch falls back to the annual statement - and `_stmt_date_cashflow` is still read
off the quarterly statement's newest column. This counts it across the published run, offline,
from the payload and the SEC cache:

  a stock is "annual under a quarterly date" for a flow when the run's figure is within 0.5% of
  the company's filed figure for its last *fiscal year*, that fiscal year ended more than 45
  days before the date the page gives the statement, and the run's figure is more than 0.5%
  from the filed twelve months to that date.

    python research/measurements/2026-11-17-annual-figures-under-a-quarterly-date.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import check_run_health as H  # noqa: E402
import run_context  # noqa: E402
import sec_crosscheck as X  # noqa: E402
from sec_fundamentals import FACTS_PATH  # noqa: E402

FLOWS = (("operating cash flow", "operatingCashFlow", ("NetCashProvidedByUsedInOperatingActivities",), "cf"),
         ("net income", "netIncome", ("NetIncomeLoss", "ProfitLoss"), "is"))


def last_fiscal_year(f, before):
    """(value, end) of the newest 350-380 day period ending over 45 days before ``before`` that
    is a fiscal year - i.e. reported in a 10-K."""
    k = f[(f["form"].isin(("10-K", "10-K/A"))) & f["start"].notna()]
    d = (k["end"] - k["start"]).dt.days
    k = k[(d >= 350) & (d <= 380) & ((before - k["end"]).dt.days > 45)]
    if k.empty:
        return None, None
    end = k["end"].max()
    return X.first_filed(k[k["end"] == end]), end


if __name__ == "__main__":
    run_dir = Path(run_context.latest_run_dir())
    payload = H.load_payload(run_dir)
    facts = pd.read_parquet(FACTS_PATH)
    facts = facts[facts["form"].isin(X.FORMS)]
    by = dict(tuple(facts.groupby("ticker")))
    detail = payload["stock_detail"]
    rank = {r["Ticker"]: r["Rank"] for r in payload["table_data"]}
    print(f"run {run_dir.name}, {len(detail)} stocks")
    for name, key, concepts, stmt in FLOWS:
        hits, checked = [], 0
        for t, det in detail.items():
            y, when = (det.get("inp") or {}).get(key), (det.get("asof") or {}).get(stmt)
            f_t = by.get(t)
            if y is None or when is None or f_t is None:
                continue
            when = pd.Timestamp(when)
            for c in concepts:
                f = f_t[f_t["concept"] == c]
                if f.empty:
                    continue
                ttm, _ = X.flow_ttm(f, when)
                fy, fy_end = last_fiscal_year(f, when)
                if ttm is None or fy is None:
                    continue
                checked += 1
                rel = lambda a, b: abs(a - b) / max(abs(a), abs(b), 1.0)  # noqa: E731
                if rel(y, fy) <= 0.005 and rel(y, ttm) > 0.005:
                    hits.append((rank.get(t, 9999), t, y, ttm, fy_end.date(), when.date(),
                                 det.get("data_source"), (det.get("flags") or {}).get("q_rejected")))
                break
        hits.sort()
        print(f"\n{name}: {len(hits)} of {checked} checked carry the last fiscal year's figure "
              f"under a later statement date")
        print("  data_source:", dict(Counter(h[6] for h in hits)),
              " q_rejected (already explained on the page):", sum(1 for h in hits if h[7]))
        print("  in the top 25:", [h[1] for h in hits if h[0] <= 25], " top 100:", sum(1 for h in hits if h[0] <= 100))
        for h in hits[:40]:
            print(f"  #{h[0]:<4}{h[1]:<6} run {h[2] / 1e9:>9.3f}bn  filed 12m to {h[5]} {h[3] / 1e9:>9.3f}bn  "
                  f"({(h[2] - h[3]) / max(abs(h[2]), abs(h[3])):+.1%})  = fiscal year to {h[4]}  source={h[6]}")
