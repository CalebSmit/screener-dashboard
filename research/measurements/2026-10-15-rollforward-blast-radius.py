"""What the roll-forward changes, over the whole universe, against the filings.

Re-fetches every stock with the working tree's `factor_engine` and compares each one's
trailing flows with the figures the published run scored, then scores both against the
company's own filing. The question the change has to answer is not "does it move numbers" but
"does it move them toward the companies' own reported figures".

Usage:  python research/measurements/2026-10-15-rollforward-blast-radius.py [--tickers A,B]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import factor_engine as fe  # noqa: E402
import sec_crosscheck as xc  # noqa: E402
import sec_fundamentals as sf  # noqa: E402

FIELDS = [
    ("revenue", "totalRevenue", xc.REVENUE, "flow", "is"),
    ("net_income", "netIncome", ("NetIncomeLoss", "ProfitLoss"), "flow", "is"),
    ("operating_cash_flow", "operatingCashFlow",
     ("NetCashProvidedByUsedInOperatingActivities",), "flow", "cf"),
    ("capex", "capex", ("PaymentsToAcquirePropertyPlantAndEquipment",
                        "PaymentsToAcquireProductiveAssets"), "flow", "cf"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    with open("dashboard_data.js", encoding="utf-8") as fh:
        payload = json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))
    detail = payload["stock_detail"]
    tickers = args.tickers.split(",") if args.tickers else sorted(detail)

    recs = [r for r in (fe.fetch_single_ticker(t) for t in tickers) if r]
    raw = pd.DataFrame(recs).set_index("Ticker")
    print(f"fetched {len(raw)} of {len(tickers)}")

    rolled = [t for t in raw.index if bool(raw.at[t, "_ltm_rolled"])]
    still_43 = [t for t in raw.index if bool(raw.at[t, "_ltm_annualized"])]
    print(f"\nroll-forward used: {len(rolled)} stocks -> {rolled}")
    print(f"still 3-of-4 x 4/3: {len(still_43)} stocks -> {still_43}")

    # the same stocks' figures, before and after, each against the filing
    facts_all = pd.read_parquet(sf.FACTS_PATH)
    facts_all = facts_all[facts_all["form"].isin(xc.FORMS)]
    print(f"\n{'ticker':<8}{'field':<21}{'published':>13}{'now':>13}{'filed':>13}"
          f"{'was':>9}{'now':>9}")
    better = worse = same = 0
    for t in rolled:
        f_t = facts_all[facts_all["ticker"] == t]
        asof = detail.get(t, {}).get("asof") or {}
        inp = detail.get(t, {}).get("inp") or {}
        for name, key, concepts, kind, stmt in FIELDS:
            when, old = asof.get(stmt), inp.get(key)
            new = raw.at[t, key] if key in raw.columns else None
            if when is None or old is None or new is None or pd.isna(new):
                continue
            v, _c, _h = xc.sec_value(f_t, concepts, kind, pd.Timestamp(when),
                                     near=float(old) if name == "revenue" else None)
            if v is None:
                continue
            o, n = (abs(old), abs(new)) if name == "capex" else (old, new)
            v = abs(v) if name == "capex" else v
            g_o = (o - v) / max(abs(v), abs(o), 1.0)
            g_n = (n - v) / max(abs(v), abs(n), 1.0)
            if abs(g_n) < abs(g_o) - 1e-6:
                better += 1
                mark = "  <- closer"
            elif abs(g_n) > abs(g_o) + 1e-6:
                worse += 1
                mark = "  <- further"
            else:
                same += 1
                mark = ""
            print(f"{t:<8}{name:<21}{o / 1e9:>13.3f}{n / 1e9:>13.3f}{v / 1e9:>13.3f}"
                  f"{g_o:>+9.2%}{g_n:>+9.2%}{mark}")
    print(f"\nagainst the filings: {better} closer, {worse} further, {same} unchanged")

    # nothing else may move
    moved = []
    for t in raw.index:
        if t in rolled:
            continue
        inp = detail.get(t, {}).get("inp") or {}
        for _n, key, *_ in FIELDS:
            a, b = inp.get(key), raw.at[t, key] if key in raw.columns else None
            if a is None or b is None or pd.isna(b):
                continue
            if abs(float(b) - float(a)) > max(1.0, abs(float(a)) * 1e-9):
                moved.append((t, key, a, float(b)))
    print(f"\nstocks that did not use the roll-forward and whose flows moved anyway: "
          f"{len(moved)}")
    for m in moved[:40]:
        print("   ", m)


if __name__ == "__main__":
    main()
