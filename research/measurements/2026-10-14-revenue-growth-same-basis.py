"""Is revenue growth comparing the same company with itself?

For research/2026-10-14-yahoo-vs-sec.md, section 5. `revenue_growth` is the latest quarter's
revenue against the same quarter a year earlier (`inp._rev_q0` / `inp._rev_q4`, both from Yahoo's
quarterly statements). After a spin-off or a disposal the company re-presents last year's quarter
on the new basis in its next 10-Q - the comparative column - but a quarterly series stitched from
each period's own report keeps the old, larger figure. DuPont: Q2 2026 revenue $1,819M against a
Q2 2025 first reported at $3,257M, before Qnity left.

For each stock this rebuilds the year-ago quarter **as re-presented in the latest filing** (the
same-basis comparative) and as first filed, from the SEC cache, and compares both growth rates
with the one the run scored. Structure only - no return is read (rule 4).

Run from the repo root:  python research/measurements/2026-10-14-revenue-growth-same-basis.py
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import sec_crosscheck as X  # noqa: E402


def quarter(f: pd.DataFrame, end: pd.Timestamp, which: str):
    """The three-month figure ending `end`: as first filed, or as most recently re-presented."""
    d = (f["end"] - f["start"]).dt.days
    q = f[((f["end"] - end).abs() <= timedelta(days=7)) & (d >= 80) & (d <= 100)]
    if q.empty:
        return None
    q = q.sort_values("filed")
    return float(q.iloc[0 if which == "first" else -1]["val"])


def main() -> None:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    facts = facts[facts["form"].isin(X.FORMS) & facts["concept"].isin(X.REVENUE)]
    by_ticker = dict(tuple(facts.groupby("ticker")))
    rows = []
    for t, det in data["stock_detail"].items():
        inp, raw = det.get("inp") or {}, det.get("raw") or {}
        g = raw.get("revenue_growth")
        q0d, q4d = inp.get("_rev_q0_date"), inp.get("_rev_q4_date")
        if g is None or not q0d or not q4d or t not in by_ticker:
            continue
        best = None
        for c in X.REVENUE:                      # the largest concept with both quarters, as in the crosscheck
            f = by_ticker[t][by_ticker[t]["concept"] == c]
            now, then_latest = quarter(f, pd.Timestamp(q0d), "first"), quarter(f, pd.Timestamp(q4d), "latest")
            then_first = quarter(f, pd.Timestamp(q4d), "first")
            if None in (now, then_latest, then_first) or then_latest <= 0 or then_first <= 0:
                continue
            if best is None or now > best[1]:
                best = (c, now, then_latest, then_first)
        if best is None:
            continue
        c, now, then_latest, then_first = best
        rows.append({"ticker": t, "sector": det.get("sector"), "scored": g, "same_basis": now / then_latest - 1,
                     "as_first_filed": now / then_first - 1, "pct": (det.get("pct") or {}).get("revenue_growth"),
                     "growth_score": (det.get("cat_scores") or {}).get("growth"), "rank": det.get("rank"),
                     "q4_restated": then_latest / then_first - 1})
    df = pd.DataFrame(rows)
    df["gap_pp"] = (df["scored"] - df["same_basis"]) * 100
    print(f"run {data['kpis'].get('run_timestamp')}: {len(df)} stocks with both quarters in the filings")
    print(f"the year-ago quarter was re-presented by more than 5% in a later filing for "
          f"{(df['q4_restated'].abs() > .05).sum()} stocks")
    print(f"scored revenue growth differs from the same-basis growth by more than 5pp for {(df['gap_pp'].abs() > 5).sum()}, "
          f"10pp for {(df['gap_pp'].abs() > 10).sum()}, 25pp for {(df['gap_pp'].abs() > 25).sum()}")
    re = df[df["q4_restated"].abs() > .05].sort_values("gap_pp", key=abs, ascending=False)
    print("\nRe-presented year-ago quarters (a spin-off, disposal or restatement), largest gaps first:")
    for _, r in re.head(20).iterrows():
        print(f"  {r['ticker']:6} {str(r['sector'])[:22]:22} scored {r['scored']:+7.1%}  same-basis {r['same_basis']:+7.1%}"
              f"  (year-ago re-presented {r['q4_restated']:+.0%})  growth pct {r['pct']}  Growth score {r['growth_score']}  rank {r['rank']}")
    other = df[(df["q4_restated"].abs() <= .05) & (df["gap_pp"].abs() > 10)]
    print(f"\nGaps over 10pp with no re-presentation ({len(other)}): " + ", ".join(
        f"{r['ticker']} {r['scored']:+.0%} vs {r['same_basis']:+.0%}" for _, r in other.iterrows()))


if __name__ == "__main__":
    main()
