"""Review of the 2026-10-29 proposal: can the source's trailing series actually replace the
quarterly income statement for the stocks whose newest quarter is empty?  (2026-10-30)

The proposal (`research/2026-10-29-ttm-fills-the-empty-quarter.md`) asks a later session to
answer its section 4 before building.  Items 2 and 3 of that section are measurable here, and
item 1's arithmetic half is too.  Three questions, read-only, about 60 requests:

A. **Row coverage.**  `factor_engine` reads eleven income-statement labels through
   ``_stmt_val_ltm``.  Which of them does ``get_income_stmt(freq="trailing")`` carry, for the
   stocks that would be filled from it?  A label it does not carry stays at the older period,
   so the stock would be scored on a *mixture* of two period ends - and the ratios built from
   two labels (gross margin, operating margin, effective tax rate, interest cover) would mix
   them.

B. **Definitional equivalence.**  On a control set whose quarterly statement is complete, is the
   trailing series' value for each label equal to the engine's own four-quarter sum of the
   quarterly row?  If it is not, filling 15 stocks from the trailing series puts them on a
   different definition from the other ~483 inside a percentile screen.  Operating income
   matters most: CLAUDE.md 0.14 keeps the source's pre-special-items "Operating Income" row on
   purpose, and a trailing series derived post-special-items would silently reverse that.

C. **Prior-year availability.**  ``revenue_growth`` and the other ``*_prior`` inputs compare the
   twelve months with the twelve months a year earlier.  How many period ends does the trailing
   series return, and is one of them a year before the newest?  If not, a June numerator would
   be compared with a March denominator - a fifteen-month growth rate.

Run:  python research/measurements/2026-10-30-trailing-series-review.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import factor_engine as FE  # noqa: E402
import sec_crosscheck as X  # noqa: E402

OUT = Path(__file__).with_suffix(".json")

# The income-statement labels factor_engine reads through _stmt_val_ltm (factor_engine.py
# ~1359-1382), with the fallback label each one tries second.
ENGINE_LABELS = [
    ("Total Revenue", None),
    ("Gross Profit", None),
    ("Operating Income", "EBIT"),
    ("EBITDA", None),
    ("Net Income", None),
    ("Tax Provision", "Income Tax"),
    ("Pretax Income", None),
    ("Cost Of Revenue", None),
    ("Interest Expense", "Interest Expense Non Operating"),
]
# The same labels as the trailing series names them (CamelCase, no spaces).
def camel(label: str) -> str:
    return "".join(w for w in label.split())


def payload() -> dict:
    t = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(t[t.find("{"):t.rfind("}") + 1])


def trailing_value(ttm: pd.DataFrame, label: str, col) -> float:
    """The trailing series' value for one engine label at one period end, or nan."""
    if ttm is None or ttm.empty:
        return np.nan
    for name in (camel(label), label):
        idx = FE._find_stmt_label(ttm, name)
        if idx is not None:
            row = FE._period_row(ttm, idx)
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
    return np.nan


def main() -> None:
    import yfinance as yf

    d = payload()
    run = pd.Timestamp(str(d["kpis"]["run_timestamp"])[:10])
    from sec_fundamentals import FACTS_PATH
    facts = pd.read_parquet(FACTS_PATH)
    late = {t: nf for t, nf in X.newer_filings(d, facts).items()
            if (run - pd.Timestamp(nf["filed"])).days > 30}
    flagged = sorted(late)
    # Control: the twelve highest-ranked stocks that are NOT flagged, so their quarterly
    # statement is the one the engine actually sums.
    ranked = sorted(d["table_data"], key=lambda r: r["Rank"])
    control = [r["Ticker"] for r in ranked if r["Ticker"] not in late][:12]

    print(f"run {run.date()}: {len(flagged)} flagged, {len(control)} control\n")

    # ---------------------------------------------------------------- A + C, flagged stocks
    cov_rows, period_rows = [], []
    for t in flagged:
        tk = yf.Ticker(t)
        end = pd.Timestamp(late[t]["end"])
        ttm, q = tk.get_income_stmt(freq="trailing"), tk.quarterly_income_stmt
        cols = [] if ttm is None or ttm.empty else sorted(pd.Timestamp(c) for c in ttm.columns)
        has_end = any(c == end for c in cols)
        col = [c for c in (ttm.columns if ttm is not None and not ttm.empty else []) if pd.Timestamp(c) == end]
        col = col[0] if col else None
        row = {"ticker": t, "filed_end": str(end.date()), "ttm_has_end": has_end,
               "ttm_periods": len(cols),
               "ttm_oldest": str(min(cols).date()) if cols else None,
               "year_before_end": any(abs((c - (end - pd.Timedelta(days=365))).days) <= 20 for c in cols)}
        # which of the engine's labels the trailing series carries at that period end
        got = {}
        for label, alt in ENGINE_LABELS:
            v = trailing_value(ttm, label, col) if col is not None else np.nan
            if np.isnan(v) and alt:
                v = trailing_value(ttm, alt, col) if col is not None else np.nan
            got[label] = v
        row["labels_present"] = int(sum(not np.isnan(v) for v in got.values()))
        row["labels_missing"] = [k for k, v in got.items() if np.isnan(v)]
        # which cells the sparse quarterly column actually holds, by engine label
        qcol = [c for c in (q.columns if q is not None and not q.empty else []) if pd.Timestamp(c) == end]
        qgot = []
        if qcol:
            for label, alt in ENGINE_LABELS:
                for name in (label, alt):
                    if not name:
                        continue
                    idx = FE._find_stmt_label(q, name)
                    if idx is not None and pd.notna(FE._period_row(q, idx).get(qcol[0], np.nan)):
                        qgot.append(label)
                        break
        row["quarterly_engine_labels_present"] = qgot
        cov_rows.append(row)
        period_rows.append({"ticker": t, **{k: (None if np.isnan(v) else round(v / 1e9, 3))
                                            for k, v in got.items()}})
        print(f"  {t:6s} ttm_has_end={has_end!s:5s} periods={len(cols):2d} "
              f"labels={row['labels_present']}/9 missing={row['labels_missing']}")

    cov = pd.DataFrame(cov_rows)

    # -------------------------------------------------------------- B, definitional equivalence
    print("\nB. trailing series vs the engine's own four-quarter sum, control set")
    diff_rows = []
    for t in control:
        tk = yf.Ticker(t)
        q = tk.quarterly_income_stmt
        ttm = tk.get_income_stmt(freq="trailing")
        if q is None or q.empty or ttm is None or ttm.empty:
            continue
        # the engine's own twelve months, exactly as factor_engine computes it
        newest = max(pd.Timestamp(c) for c in q.columns)
        tcols = [c for c in ttm.columns if abs((pd.Timestamp(c) - newest).days) <= 20]
        if not tcols:
            diff_rows.append({"ticker": t, "aligned": False})
            continue
        col = tcols[0]
        rec = {"ticker": t, "aligned": True, "period": str(newest.date())}
        for label, alt in ENGINE_LABELS:
            eng = FE._stmt_val_ltm(q, label)
            if np.isnan(eng) and alt:
                eng = FE._stmt_val_ltm(q, alt)
            tv = trailing_value(ttm, label, col)
            if np.isnan(tv) and alt:
                tv = trailing_value(ttm, alt, col)
            if np.isnan(eng) or np.isnan(tv):
                rec[label] = None
            else:
                rec[label] = None if eng == 0 else round((tv - eng) / abs(eng), 6)
        diff_rows.append(rec)
        shown = {k: v for k, v in rec.items() if k not in ("ticker", "aligned", "period")}
        print(f"  {t:6s} " + "  ".join(f"{k.split()[0][:6]}={v}" for k, v in shown.items()))

    diffs = pd.DataFrame(diff_rows)

    # ---------------------------------------------------------------------------- summary
    print("\n--- SUMMARY ---")
    print(f"A. trailing series reaches the filed period: {int(cov['ttm_has_end'].sum())} of {len(cov)}")
    if int(cov["ttm_has_end"].sum()):
        sub = cov[cov["ttm_has_end"]]
        print(f"   of those, all nine engine labels present: "
              f"{int((sub['labels_present'] == 9).sum())} of {len(sub)}; "
              f"median present {sub['labels_present'].median():.0f}/9")
        miss = {}
        for ls in sub["labels_missing"]:
            for k in ls:
                miss[k] = miss.get(k, 0) + 1
        print(f"   missing by label: {miss}")
    print(f"C. a period end a year before the filed one: "
          f"{int(cov['year_before_end'].sum())} of {len(cov)}; "
          f"period counts {sorted(cov['ttm_periods'].unique().tolist())}")
    if len(diffs) and "aligned" in diffs:
        al = diffs[diffs["aligned"] == True]  # noqa: E712
        print(f"B. control stocks aligned on a period end: {len(al)} of {len(control)}")
        for label, _ in ENGINE_LABELS:
            if label not in al:
                continue
            v = pd.to_numeric(al[label], errors="coerce").dropna()
            if len(v):
                print(f"   {label:24s} n={len(v):2d}  median |rel diff| {v.abs().median():.6f}  "
                      f"max |rel diff| {v.abs().max():.6f}  within 0.5%: {int((v.abs() <= 0.005).sum())}/{len(v)}")

    OUT.write_text(json.dumps({
        "run": str(run.date()),
        "flagged": flagged, "control": control,
        "coverage": json.loads(cov.to_json(orient="records")),
        "trailing_values_bn": period_rows,
        "control_rel_diffs": json.loads(diffs.to_json(orient="records")),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
