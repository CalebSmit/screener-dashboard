"""Is multiplying three quarters by 4/3 better or worse than the annual statement?

Re-measures `research/2026-10-14-yahoo-vs-sec.md` section 4b. The engine's `_stmt_val_ltm`
falls back, when only 3 of a company's last 4 quarters carry a value, to `sum(3) * 4/3`
(`factor_engine.py` line 623). On the run of 2026-10-15 that fired for four stocks; for
BRK-B it fired on nine labels including revenue, net income, EBIT and operating cash flow.

For each affected stock this prints three candidates for the same trailing flow and the
company's own filed figure for the quarter the run says it scored:

  A  sum(3 quarters) * 4/3      - what ships today
  B  the annual statement        - the last complete fiscal year, one source, no fabrication
  C  the filing                  - SEC XBRL, the yardstick (never an input; see CLAUDE.md 0.14)

Usage:  python research/measurements/2026-10-15-three-quarter-annualisation.py
Network: yfinance for A and B; the SEC side is the local weekly cache.
"""
from __future__ import annotations

import json
import sys
import warnings

import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import factor_engine as fe  # noqa: E402
import sec_crosscheck as xc  # noqa: E402
import sec_fundamentals as sf  # noqa: E402

TICKERS = ["BRK-B", "COST", "GIS", "V"]
# (engine label, SEC concepts, which yfinance annual statement)
ROWS = [
    ("Total Revenue", xc.REVENUE, "is"),
    ("Net Income", ("NetIncomeLoss", "ProfitLoss"), "is"),
    ("Operating Income", ("OperatingIncomeLoss",), "is"),
    ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",), "cf"),
]


def pct(a, b):
    """Signed relative gap of a against b, on the larger magnitude."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    d = max(abs(a), abs(b))
    return None if d == 0 else (a - b) / d


def fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "        -"
    return f"{v / 1e9:9.3f}"


def main():
    import yfinance as yf

    facts_all = pd.read_parquet(sf.FACTS_PATH)
    payload = json.loads(
        open("dashboard_data.js", encoding="utf-8").read().split("=", 1)[1].strip().rstrip(";")
    )
    detail = payload["stock_detail"]

    for t in TICKERS:
        tk = yf.Ticker(t)
        q_is, a_is = tk.quarterly_financials, tk.financials
        q_cf, a_cf = tk.quarterly_cashflow, tk.cashflow
        asof = (detail.get(t, {}).get("asof") or {}).get("is")
        end = pd.Timestamp(asof) if asof else None
        facts = facts_all[facts_all["ticker"] == t]

        print(f"\n=== {t}   statement quarter {asof}   rank {detail.get(t, {}).get('rank')}")
        cols = [c for c in fe._populated_columns(q_is)]
        print(f"    quarterly income-statement columns: {[str(c.date()) for c in cols]}")
        print("    line                  A: 3q x 4/3   B: annual    C: filed   "
              "A vs C      B vs C")
        for label, concepts, which in ROWS:
            q, a = (q_is, a_is) if which == "is" else (q_cf, a_cf)
            partial: list[str] = []
            va = fe._stmt_val_ltm(q, label, partial_labels=partial)
            vb = fe._stmt_val(a, label)
            vc, concept, how = xc.sec_value(
                facts, concepts, "flow", end,
                near=va if concepts is xc.REVENUE else None,
            ) if end is not None else (None, None, "no asof")
            ga, gb = pct(va, vc), pct(vb, vc)
            flag = "  <- 4/3 used" if label in partial else ""
            print(f"    {label:<21}{fmt(va)}  {fmt(vb)}  {fmt(vc)}  "
                  f"{'' if ga is None else f'{ga:+7.2%}'}     "
                  f"{'' if gb is None else f'{gb:+7.2%}'}{flag}")
            if vc is not None:
                print(f"        filed: {concept} ({how})")


if __name__ == "__main__":
    main()
