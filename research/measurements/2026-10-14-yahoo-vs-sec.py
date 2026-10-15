"""How often do the fundamentals the screener scores with agree with the companies' own SEC filings?

For research/2026-10-14-yahoo-vs-sec.md. Every input below is read from the published payload
(`stock_detail[t].inp`, the Yahoo figures the run scored with, and `asof`, the quarter each
statement ends) and compared with the same quantity rebuilt from the SEC `companyfacts` cache
(`data/sec/pit/facts.parquet`, refreshed weekly by the run; see sec_fundamentals.py). No
network, no return, nothing written outside --out.

Trailing-twelve-month flows for a quarter ending E are rebuilt from the company's own tags:
the fiscal year if E is a fiscal year-end, otherwise year-to-date(E) + prior fiscal year -
year-to-date(E a year earlier), which is how a 10-Q's cumulative columns combine. Balance-sheet
items are the instant value at E. Each period's figure is the one from the **first** filing that
reported it, not the latest: a later filing's comparative columns are where mis-tagged contexts
turn up (FIX's Q1 2026 10-Q tags a quarter's revenue as fiscal 2025's). Revenue takes the largest
of the revenue concepts a company reports at E, because the others are components of it (a REIT's
`RevenueFromContractWithCustomer...` is its non-rental income). Capex is compared with Yahoo's two
sides of it separately - see the note.

Run from the repo root:
    python research/measurements/2026-10-14-yahoo-vs-sec.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data" / "sec" / "pit" / "facts.parquet"

# The rebuild lives in sec_crosscheck.py (the data loop's health check uses the same code, so the
# note and the loop cannot drift apart); this script prints the full picture from it.
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
import sec_crosscheck as X  # noqa: E402

FIELDS = X.FIELDS
sec_value = X.sec_value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    facts = pd.read_parquet(FACTS)
    facts = facts[facts["form"].isin(X.FORMS)]
    print(f"run {data['kpis'].get('run_timestamp')}; SEC cache {pd.Timestamp(FACTS.stat().st_mtime, unit='s'):%Y-%m-%d}, "
          f"{facts['ticker'].nunique()} companies")

    df = X.compare(data, facts)

    print("\nPer field: compared, and how close (|Yahoo - SEC| / larger of the two)")
    print(f"{'field':22}{'compared':>9}{'<=0.5%':>8}{'<=2%':>8}{'<=10%':>8}{'>10%':>7}   not compared (why)")
    for name, *_ in FIELDS:
        sub = df[df["field"] == name]
        c = sub[sub["status"] == "compared"]
        a = c["rel_diff"].abs()
        why = sub[sub["status"] != "compared"]["status"].str.split(":").str[0].value_counts().to_dict()
        print(f"{name:22}{len(c):>9}{(a <= .005).mean():>8.1%}{(a <= .02).mean():>8.1%}{(a <= .10).mean():>8.1%}"
              f"{(a > .10).sum():>7}   {why}")

    c = df[df["status"] == "compared"].copy()
    c["abs"] = c["rel_diff"].abs()
    print("\nThe 25 largest disagreements:")
    show = c.sort_values("abs", ascending=False).head(25)
    for _, r in show.iterrows():
        print(f"  {r['ticker']:6} {r['field']:20} asof {r['asof']}  Yahoo {r['yahoo']:>18,.0f}  SEC {r['sec']:>18,.0f}"
              f"  {r['rel_diff']:+7.1%}  ({r['concept']}, {r['how']}; {r['sector']})")
    stocks = c[c["abs"] > .10].groupby("ticker")["field"].apply(list)
    print(f"\nStocks with any field more than 10% from its filing: {len(stocks)} of {df['ticker'].nunique()}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out / "yahoo_vs_sec.csv", index=False)


if __name__ == "__main__":
    main()
