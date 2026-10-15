"""A third candidate for the stocks with a hole in their quarterly statement: the roll-forward.

`2026-10-15-three-quarter-annualisation.py` measured the two candidates the code already
knows about and neither wins: for BRK-B, three quarters times 4/3 is closer to the filing on
net income (-14.5% against -21.9%) and further away on revenue (+10.2% against +6.3%) and
operating cash flow (-6.1% against -1.4%).

But BRK-B's hole is not a blank column - the quarter is absent:

    2026-06-30, 2026-03-31, 2025-12-31, 2025-06-30, 2025-03-31   (no 2025-09-30)

which is why `_quarters_adjacent` refuses to sum the first four. The textbook trailing-twelve-
month construction does not need that quarter:

    TTM(E) = last fiscal year + year-to-date(E) - year-to-date(E one year earlier)

and every term is in the source the run already fetches: the annual statement for the fiscal
year, and the quarters either side of it for the two year-to-date stubs. It is the same
identity `sec_crosscheck.flow_ttm` uses on the filings ("ytd + prior fy - prior ytd").

This prints candidate D beside A and B for every affected stock and line, against the filing.

Usage:  python research/measurements/2026-10-15-ttm-rollforward.py
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
ROWS = [
    ("Total Revenue", xc.REVENUE, "is"),
    ("Net Income", ("NetIncomeLoss", "ProfitLoss"), "is"),
    ("Operating Income", ("OperatingIncomeLoss",), "is"),
    ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",), "cf"),
    ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",), "cf"),
]


def rollforward(q_stmt, a_stmt, label):
    """TTM = last fiscal year + ytd(latest) - ytd(same point a year earlier), or None.

    The year-to-date stubs are summed from consecutive quarters after the annual statement's
    own period end, and the same consecutive quarters a year before them. Returns
    (value, how) so the caller can print which quarters were used.
    """
    try:
        q_row = fe._period_row(q_stmt, fe._find_stmt_label(q_stmt, label))
        a_cols = [c for c in fe._populated_columns(a_stmt)]
        fy_end = a_cols[0]
        fy = fe._stmt_val(a_stmt, label)
        if pd.isna(fy):
            return None, "no annual figure"
        q_cols = [c for c in q_row.index if pd.notna(q_row[c])]
        stub = [c for c in q_cols if c > fy_end]
        if not stub:
            return None, "no quarter after the fiscal year end"
        prior = [c for c in q_cols if any(abs((s - c).days - 365) <= 20 for s in stub)]
        if len(prior) != len(stub):
            return None, f"{len(stub)} stub quarter(s), {len(prior)} a year earlier"
        v = fy + sum(float(q_row[c]) for c in stub) - sum(float(q_row[c]) for c in prior)
        how = (f"FY to {fy_end.date()} + {len(stub)}q to {max(stub).date()}"
               f" - {len(prior)}q to {max(prior).date()}")
        return v, how
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
        return None, f"{type(e).__name__}"


def pct(a, b):
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
    with open("dashboard_data.js", encoding="utf-8") as fh:
        payload = json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))
    detail = payload["stock_detail"]
    wins = {"A": 0, "B": 0, "D": 0}

    for t in TICKERS:
        tk = yf.Ticker(t)
        stmts = {"is": (tk.quarterly_financials, tk.financials),
                 "cf": (tk.quarterly_cashflow, tk.cashflow)}
        asof = (detail.get(t, {}).get("asof") or {}).get("is")
        end = pd.Timestamp(asof) if asof else None
        facts = facts_all[facts_all["ticker"] == t]
        print(f"\n=== {t}   statement quarter {asof}")
        print("    line                   A: 3q x4/3   B: annual   D: rolled    C: filed"
              "     A vs C   B vs C   D vs C")
        for label, concepts, which in ROWS:
            q, a = stmts[which]
            partial: list[str] = []
            va = fe._stmt_val_ltm(q, label, partial_labels=partial)
            vb = fe._stmt_val(a, label)
            vd, how = rollforward(q, a, label)
            vc, concept, _h = xc.sec_value(
                facts, concepts, "flow", end,
                near=va if concepts is xc.REVENUE else None,
            ) if end is not None else (None, None, "")
            ga, gb, gd = pct(va, vc), pct(vb, vc), pct(vd, vc)
            if None not in (ga, gb, gd):
                best = min((abs(ga), "A"), (abs(gb), "B"), (abs(gd), "D"))[1]
                wins[best] += 1
            else:
                best = ""
            print(f"    {label:<22}{fmt(va)} {fmt(vb)} {fmt(vd)} {fmt(vc)}  "
                  f"{'' if ga is None else f'{ga:+7.2%}'} "
                  f"{'' if gb is None else f'{gb:+7.2%}'} "
                  f"{'' if gd is None else f'{gd:+7.2%}'}"
                  f"{'   <- ' + best if best else ''}"
                  f"{'   [4/3 used]' if label in partial else ''}")
            if vd is None:
                print(f"        D unavailable: {how}")
            else:
                print(f"        D: {how}")
    print(f"\nclosest to the filing, by line: {wins}")


if __name__ == "__main__":
    main()
