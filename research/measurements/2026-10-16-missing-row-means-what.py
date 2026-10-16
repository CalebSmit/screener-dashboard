"""Does a missing quarterly "unusual items" row mean there were none? - 2026-10-16

This decides whether the open proposal in ``research/2026-10-15-earnings-definition.md`` can be
shipped on **one definition for the whole universe**, which is the condition
``factor_engine.py`` records for this exact metric (Phase 13 / F19: earnings yield previously fell
back to a different, EPS-based definition when the trailing figure was missing, "so the fallback
cohort was ranked on an incomparable basis inside one percentile rank", and the fix was to score
NaN instead).

The proposal reads a four-quarter sum. Measured in
``2026-10-16-normalized-income-reconstructible.py``, at that window 119 companies have "Total
Unusual Items" populated in **0 of 4 quarters**. There are two possibilities, and they lead to
opposite decisions:

* the row is absent **because the company had no unusual items** - then treating absent as zero is
  true, every stock is scored on the same definition, and the proposal is shippable;
* the row is absent **because the source does not serve it quarterly** - then treating absent as
  zero scores those companies on earnings *including* their one-off items while their peers are
  scored on earnings excluding them, inside one percentile rank, which is the defect Phase 13 fixed.

The latest fiscal year settles it: if a company with no quarterly row reports non-zero unusual
items annually, the row's absence is the source's, not the company's.

Offline; reads the two committed scans.

Usage: python research/measurements/2026-10-16-missing-row-means-what.py
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCAN = HERE / "2026-10-16-normalized-income-same-column.json"
OUT = Path(__file__).with_suffix(".json")

UI, NO, NI = "Total Unusual Items", "Normalized Income", "Net Income"
MATERIAL = 0.05        # unusual items worth 5% of net income - enough to move a percentile


def main():
    rows = json.loads(SCAN.read_text(encoding="utf-8"))["rows"]
    verdict, detail = Counter(), {"absent_quarterly_but_annual_says_otherwise": [],
                                  "no_normalized_income_at_ttm": []}
    for t, r in rows.items():
        ttm, ann = r.get("ttm"), r.get("annual")
        if not ttm:
            verdict["no trailing window could be formed"] += 1
            continue
        q = ttm["quarters_populated"].get(UI, 0)
        av = (ann or {}).get("vals", {}).get(UI)
        ani = (ann or {}).get("vals", {}).get(NI)
        if ttm["vals"].get(NO) is None:
            verdict["no Normalized Income at the trailing window - would fall back to net income"] += 1
            detail["no_normalized_income_at_ttm"].append(
                {"ticker": t, "annual_unusual": av, "annual_net": ani,
                 "quarters_with_unusual": q})
            continue
        if q == 4:
            verdict["unusual items served for all four quarters"] += 1
            continue
        # the row is short. What does the company's own fiscal year say?
        if av is None:
            verdict["row short, and no annual row either - genuinely unknown"] += 1
        elif abs(av) < 1.0:
            verdict["row short, and the fiscal year says zero - absent means none"] += 1
        else:
            material = ani and abs(ani) > 1e6 and abs(av) / abs(ani) >= MATERIAL
            verdict["row short, but the fiscal year reports unusual items - absent means not served"] += 1
            if material:
                verdict["  ... of those, worth >= 5% of net income"] += 1
            detail["absent_quarterly_but_annual_says_otherwise"].append(
                {"ticker": t, "quarters_with_unusual": q, "annual_unusual": av,
                 "annual_net": ani,
                 "share_of_net": round(abs(av) / abs(ani), 3) if ani and abs(ani) > 1e6 else None})
    detail["absent_quarterly_but_annual_says_otherwise"].sort(
        key=lambda d: -(d["share_of_net"] or 0))
    OUT.write_text(json.dumps({"verdict": dict(verdict.most_common()), "detail": detail},
                              indent=1), encoding="utf-8")
    for k, n in verdict.most_common():
        print(f"  {n:4d}  {k}")
    print("\n  worst by share of net income (the row is absent quarterly, the fiscal year is not):")
    for d in detail["absent_quarterly_but_annual_says_otherwise"][:10]:
        print(f"     {d['ticker']:6s} {d['quarters_with_unusual']}/4 quarters; annual unusual "
              f"${d['annual_unusual']/1e6:,.0f}M on ${(d['annual_net'] or 0)/1e6:,.0f}M net "
              f"({d['share_of_net']})")
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
