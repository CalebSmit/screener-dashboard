"""Operating income is not in the filing check, and three weighted metrics are built on it.

`sec_crosscheck.FIELDS` checks revenue, net income, gross profit, operating cash flow, capex,
total assets and equity. It does not check `ebit` - the run's operating income - which feeds
ROIC, EV/EBITDA and net debt / EBITDA.

Found while measuring the three-quarter annualisation (2026-10-15): General Mills' scored
operating income is $2.807bn where its own 10-K filed `OperatingIncomeLoss` of $0.886bn for
the same twelve months, a gap of 68%, while its revenue, net income and operating cash flow
agree with the filing to the dollar. This sizes that gap over the whole universe before
anything is built on it.

Offline: the weekly `companyfacts` cache plus the published payload. Reads nothing else.

Usage:  python research/measurements/2026-10-15-operating-income-vs-filings.py
"""
from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, ".")

import sec_crosscheck as xc  # noqa: E402
import sec_fundamentals as sf  # noqa: E402

FIELDS = [
    ("operating_income", "ebit", ("OperatingIncomeLoss",), "flow", "is"),
    # the lines already in the check, for a like-for-like baseline on the same run
    ("revenue", "totalRevenue", xc.REVENUE, "flow", "is"),
    ("net_income", "netIncome", ("NetIncomeLoss", "ProfitLoss"), "flow", "is"),
    ("operating_cash_flow", "operatingCashFlow",
     ("NetCashProvidedByUsedInOperatingActivities",), "flow", "cf"),
]


def main():
    with open("dashboard_data.js", encoding="utf-8") as fh:
        payload = json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))
    facts = pd.read_parquet(sf.FACTS_PATH)
    df = xc.compare(payload, facts, fields=FIELDS)

    print(f"{len(payload['stock_detail'])} stocks in the payload; "
          f"{df['ticker'].nunique()} with facts in the cache\n")
    for name in [f[0] for f in FIELDS]:
        s = df[(df["field"] == name) & df["rel_diff"].notna()]
        if s.empty:
            print(f"{name:<22} no comparable rows")
            continue
        g = s["rel_diff"].abs()
        print(f"{name:<22} {len(s):>4} compared | within 0.5%: {(g <= 0.005).sum():>4} "
              f"({(g <= 0.005).mean():5.1%}) | within 2%: {(g <= 0.02).sum():>4} "
              f"| over 10%: {(g > 0.10).sum():>4} | median |gap| {g.median():.3%}")

    oi = df[(df["field"] == "operating_income") & df["rel_diff"].notna()].copy()
    oi["absgap"] = oi["rel_diff"].abs()
    sector = {t: d.get("sector") for t, d in payload["stock_detail"].items()}
    rank = {t: d.get("rank") for t, d in payload["stock_detail"].items()}
    worst = oi.sort_values("absgap", ascending=False).head(25)
    print("\nthe 25 largest operating-income disagreements")
    print(f"{'ticker':<8}{'rank':>5}  {'run $bn':>10}{'filed $bn':>11}{'gap':>9}  sector")
    for _, r in worst.iterrows():
        print(f"{r['ticker']:<8}{str(rank.get(r['ticker'])):>5}  {r['yahoo'] / 1e9:>10.3f}"
              f"{r['sec'] / 1e9:>11.3f}{r['rel_diff']:>+9.1%}  {sector.get(r['ticker'])}")

    over = oi[oi["absgap"] > 0.10]
    print(f"\nover 10%: {len(over)} stocks")
    print("by sector:")
    for sec, n in over["ticker"].map(sector).value_counts().items():
        tot = oi["ticker"].map(sector).value_counts().get(sec, 0)
        print(f"  {str(sec):<26}{n:>4} of {tot}")


if __name__ == "__main__":
    main()
