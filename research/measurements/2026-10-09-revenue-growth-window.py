"""How many months does "year-over-year" revenue growth actually span? (CLAUDE.md 0.9(b))

`revenue_growth` = TTM revenue / `totalRevenue_prior` - 1, where `totalRevenue_prior` is the
TTM a year earlier when Yahoo supplies 8 quarters, and otherwise falls back to the annual
statement's column 1 - the fiscal year *before* the latest completed one. Yahoo supplies 5
quarters for 19 of 20 sampled stocks (2026-10-09), so the fallback is the rule, not the exception.

Reads the newest run's raw fetch (no network) and reports:
1. how often the prior figure is the annual column-1 figure (the fallback);
2. the span between the midpoints of the two 12-month windows, by fiscal year-end month, using
   the annual statement date and the run date (TTM assumed to end at the latest quarter end on
   or before the run, 0-3 months back - the span is reported as a range for that reason);
3. the current figure against the two consistent 12-month alternatives that need no new data:
   annual vs annual (latest fiscal year over the one before), and how far the sector ranks move.

    python research/measurements/2026-10-09-revenue-growth-window.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def newest_run() -> Path:
    runs = [d for d in (ROOT / "runs").iterdir() if (d / "05_final_scored.parquet").exists()]
    return max(runs, key=lambda d: (d / "05_final_scored.parquet").stat().st_mtime)


def main() -> dict:
    run = newest_run()
    raw = pd.read_parquet(run / "00_raw_fetch.parquet").set_index("Ticker")
    sc = pd.read_parquet(run / "05_final_scored.parquet").set_index("Ticker")
    run_day = pd.Timestamp(json.loads((run / "meta.json").read_text()).get("run_date", "2026-10-09")[:10])
    out: dict = {"run": run.name, "run_date": str(run_day.date())}

    ttm, prior = raw["totalRevenue"], raw["totalRevenue_prior"]
    a0, a1 = raw["totalRevenue_annual"], raw["totalRevenue_annual_prior"]
    ok = ttm.notna() & prior.notna() & (prior > 0)
    fallback = ok & a1.notna() & np.isclose(prior, a1, rtol=1e-9)
    out["with_revenue_growth"] = int(ok.sum())
    out["prior_is_annual_col1"] = int(fallback.sum())

    # Window span between the END of the TTM (the latest quarter, `_stmt_date_financials`) and
    # the END of the fiscal year it is compared with (column 1 = the year before the latest
    # completed one). Each company's fiscal year-end month comes from its own SEC filing (the
    # NetIncomeLoss CY2025 frame's period end, cached by sec_fundamentals); both windows are
    # 12 months long, so the gap between their ends is the span the "growth" covers.
    sys.path.insert(0, str(ROOT))
    import insider_activity as ia
    frame = json.loads((ROOT / "data" / "sec" / "frames" / "NetIncomeLoss_CY2025.json").read_text())
    cmap = ia.ticker_map(None)
    q0 = pd.to_datetime(raw["_stmt_date_financials"], errors="coerce")
    spans = {}
    fye_m = {}
    for t in raw.index[fallback]:
        cik = cmap.get(t)
        f = frame.get(str(cik)) if cik else None
        if not f or pd.isna(q0.get(t)):
            continue
        m = pd.Timestamp(f[1]).month
        fy0 = pd.Timestamp(year=q0[t].year, month=m, day=1) + pd.offsets.MonthEnd(0)
        if fy0 > q0[t] + pd.Timedelta(days=7):
            fy0 = pd.Timestamp(year=q0[t].year - 1, month=m, day=1) + pd.offsets.MonthEnd(0)
        col1_end = fy0 - pd.DateOffset(years=1)
        spans[t] = round((q0[t] - col1_end).days / 30.44)
        fye_m[t] = m
    sp = pd.Series(spans)
    out["span_months_when_fallback"] = {int(k): int(v) for k, v in sp.value_counts().sort_index().items()}
    out["span_median_months"] = float(sp.median())
    out["span_by_fiscal_year_end_month"] = {int(m): float(sp[[t for t in sp.index if fye_m[t] == m]].median())
                                           for m in sorted(set(fye_m.values()))}
    out["stocks_with_known_span"] = int(len(sp))

    cur = (ttm / prior - 1)[ok]
    ann = (a0 / a1 - 1)[a0.notna() & a1.notna() & (a1 > 0)]
    both = cur.index.intersection(ann.index)
    out["median_growth_current_definition"] = round(float(cur.median()), 4)
    out["median_growth_annual_vs_annual"] = round(float(ann.median()), 4)
    out["spearman_current_vs_annual"] = round(float(cur[both].rank().corr(ann[both].rank())), 3)
    # Within-sector rank movement (the score is a sector percentile).
    sec = sc["Sector"].reindex(both)
    p_cur = cur[both].groupby(sec).rank(pct=True)
    p_ann = ann[both].groupby(sec).rank(pct=True)
    mv = (p_cur - p_ann).abs() * 100
    out["sector_percentile_move_median_pts"] = round(float(mv.median()), 1)
    out["sector_percentile_move_p90_pts"] = round(float(mv.quantile(0.9)), 1)
    out["share_moving_over_20pts"] = round(float((mv > 20).mean()), 3)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
