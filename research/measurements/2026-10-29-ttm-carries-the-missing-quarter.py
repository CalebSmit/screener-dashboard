"""Does the source's trailing-twelve-month income series carry the quarter its quarterly
statement leaves empty? (2026-10-29, owner-run)

The data loop's health check reports stocks "scored a period behind a filing over 30 days old":
15 on 2026-10-29 (Amazon, Bank of America, ...). The 2026-10-15 measurement
(`2026-10-15-statement-lag.py`) found the cause - the source lists the newer quarter as a column
with 4 or 5 rows and nothing else in it - and the engine rightly does not read an empty column.

This asks the next question for every flagged stock, read-only: for the newest filed period,
does `get_income_stmt(freq="trailing")` have that period end, with net income, revenue and
operating income; does the quarterly balance sheet or cash-flow statement have it at all; and
do the implied quarter's figures (TTM minus the three quarters before) match the company's own
filing in the SEC companyfacts cache.

Run:  python research/measurements/2026-10-29-ttm-carries-the-missing-quarter.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import sec_crosscheck as X  # noqa: E402


def payload() -> dict:
    t = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(t[t.find("{"):t.rfind("}") + 1])


def main() -> None:
    import yfinance as yf
    d = payload()
    run = pd.Timestamp(str(d["kpis"]["run_timestamp"])[:10])
    from sec_fundamentals import FACTS_PATH          # the weekly companyfacts cache the health check reads
    facts = pd.read_parquet(FACTS_PATH)
    late = {t: nf for t, nf in X.newer_filings(d, facts).items()
            if (run - pd.Timestamp(nf["filed"])).days > 30}
    print(f"run {run.date()}: {len(late)} stocks a period behind a filing over 30 days old\n")
    rows = []
    for t, nf in sorted(late.items()):
        tk = yf.Ticker(t)
        end = pd.Timestamp(nf["end"])
        ttm = tk.get_income_stmt(freq="trailing")
        q = tk.quarterly_income_stmt
        bs, cf = tk.quarterly_balance_sheet, tk.quarterly_cashflow
        has_ttm = ttm is not None and not ttm.empty and any(pd.Timestamp(c) == end for c in ttm.columns)
        ni_ttm = float(ttm.loc["NetIncome", [c for c in ttm.columns if pd.Timestamp(c) == end][0]]) \
            if has_ttm and "NetIncome" in ttm.index else None
        q_cells = int(q[[c for c in q.columns if pd.Timestamp(c) == end]].notna().sum().sum()) \
            if q is not None and any(pd.Timestamp(c) == end for c in q.columns) else 0
        # the implied quarter: TTM at `end` less the three quarters before it in the quarterly table
        implied = None
        if ni_ttm is not None and "Net Income" in q.index:
            prior = [c for c in q.columns if pd.Timestamp(c) < end][:3]
            if len(prior) == 3 and q.loc["Net Income", prior].notna().all():
                implied = ni_ttm - float(q.loc["Net Income", prior].sum())
        filed = facts[(facts["ticker"] == t) & (facts["end"] == end)]
        # the company's own figure for that quarter: NetIncomeLoss over a ~3-month span ending `end`
        ni = filed[(filed["concept"].isin(("NetIncomeLoss", "ProfitLoss")))
                   & ((filed["end"] - filed["start"]).dt.days.between(80, 100))]
        sec_q = float(ni.sort_values("filed")["val"].iloc[0]) if len(ni) else None
        rows.append({
            "ticker": t, "scored": d["stock_detail"][t]["asof"].get("is"), "filed_end": nf["end"],
            "filed": nf["filed"], "quarterly_cells": q_cells, "ttm_has_end": has_ttm,
            "bs_has_end": any(pd.Timestamp(c) == end for c in bs.columns),
            "cf_has_end": any(pd.Timestamp(c) == end for c in cf.columns),
            "implied_q_net_income_bn": None if implied is None else round(implied / 1e9, 2),
            "sec_q_net_income_bn": None if sec_q is None else round(sec_q / 1e9, 2),
        })
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))
    print(f"\nTTM carries the filed period for {int(out['ttm_has_end'].sum())} of {len(out)}; "
          f"balance sheet for {int(out['bs_has_end'].sum())}; cash flow for {int(out['cf_has_end'].sum())}.")


if __name__ == "__main__":
    main()
