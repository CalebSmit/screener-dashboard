"""The source's trailing series returns one period end, and it runs ahead of the quarterly
table by one or two quarters (2026-10-30).

The load-bearing fact in the review of `research/2026-10-29-ttm-fills-the-empty-quarter.md`.
That note's rule is "fill where the trailing series reaches the newer quarter", with the newer
quarter taken from the newest filing in the weekly SEC companyfacts cache.  This prints, for
each of the fifteen stocks the note named, the three period ends that actually decide the
question:

  * the newest **column** of the quarterly income statement (which may be sparse);
  * the newest column that carries **Total Revenue** - what ``_stmt_val_ltm`` actually sums,
    and therefore what the stock is scored on;
  * every period end ``get_income_stmt(freq="trailing")`` returns.

The gap between the second and the third, in quarters, is the size of the opportunity and the
basis for a guard: a one-quarter gap leaves the implied newest quarter computable from the
quarterly table (TTM less the three populated quarters before it), which is how the 2026-10-29
note checked eight of them against the companies' own 10-Qs.  A two-quarter gap leaves nothing
to check it against.

Note on the SEC columns below: ``facts["end"].max()`` is the newest fact end of **any** kind in
the cache, including instant facts such as the cover-page share count, so it is **not** a
period end and must not be compared with the columns above.  It is printed only to show how
stale the cache is.  The note's own period comparison comes from ``sec_crosscheck.newer_filings``
and is in `2026-10-30-trailing-series-review.py`.

Run:  python research/measurements/2026-10-30-trailing-column-moves.py
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

OUT = Path(__file__).with_suffix(".json")

# the fifteen stocks the note's section 2 table names
NOTE = ["AMZN", "APH", "CHD", "CHRW", "CPT", "DD", "AEE", "ES", "FDXF",
        "BAC", "BNY", "TFC", "EXR", "FLEX", "ILMN"]


def quarter_gap(a: str | None, b: str | None) -> int | None:
    """How many quarters period end ``b`` is after period end ``a``, rounded from days."""
    if not a or not b:
        return None
    days = (pd.Timestamp(b) - pd.Timestamp(a)).days
    return int(round(days / 91.25))


def main() -> None:
    import yfinance as yf
    from sec_fundamentals import FACTS_PATH

    facts = pd.read_parquet(FACTS_PATH)
    rows = []
    for t in NOTE:
        tk = yf.Ticker(t)
        ttm, q = tk.get_income_stmt(freq="trailing"), tk.quarterly_income_stmt
        ttm_cols = sorted(str(pd.Timestamp(c).date()) for c in ttm.columns) \
            if ttm is not None and not ttm.empty else []
        q_cols = sorted(str(pd.Timestamp(c).date()) for c in q.columns) \
            if q is not None and not q.empty else []
        q_newest_pop = None
        if q is not None and not q.empty and "Total Revenue" in q.index:
            row = q.loc["Total Revenue"]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pop = [pd.Timestamp(c) for c in row.index if pd.notna(row[c])]
            q_newest_pop = str(max(pop).date()) if pop else None
        f = facts[facts["ticker"] == t]
        rows.append({
            "ticker": t,
            "quarterly_columns": len(q_cols),
            "quarterly_newest_column": q_cols[-1] if q_cols else None,
            "scored_on": q_newest_pop,
            "trailing_columns": ttm_cols,
            "gap_quarters": quarter_gap(q_newest_pop, ttm_cols[-1] if ttm_cols else None),
            "sec_cache_newest_fact_end_any_kind": str(pd.Timestamp(f["end"].max()).date()) if len(f) else None,
        })
        r = rows[-1]
        print(f"  {t:6s} newest_col={r['quarterly_newest_column']}  scored_on={r['scored_on']}  "
              f"trailing={ttm_cols}  gap={r['gap_quarters']}q")

    df = pd.DataFrame(rows)
    n1 = int((df["trailing_columns"].apply(len) == 1).sum())
    g = df["gap_quarters"]
    print("\n--- SUMMARY ---")
    print(f"trailing series returns exactly one period end: {n1} of {len(df)}")
    print(f"gap between what the stock is scored on and the trailing column:")
    for k in sorted(x for x in g.dropna().unique()):
        who = sorted(df.loc[g == k, "ticker"])
        print(f"  {int(k)} quarter(s): {len(who):2d}  {who}")
    one = sorted(df.loc[g == 1, "ticker"])
    two_plus = sorted(df.loc[g >= 2, "ticker"])
    print(f"\nnewer at all (gap >= 1): {int((g >= 1).sum())} of {len(df)}")
    print(f"  gap == 1, so the implied quarter is computable and checkable: {len(one)}  {one}")
    print(f"  gap >= 2, so nothing in the quarterly table can check it:     {len(two_plus)}  {two_plus}")
    OUT.write_text(json.dumps({
        "rows": rows, "one_period_only": n1,
        "gap_one_checkable": one, "gap_two_or_more_uncheckable": two_plus,
        "newer_at_all": int((g >= 1).sum()),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
