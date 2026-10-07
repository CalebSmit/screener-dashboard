"""Recompute every stock's category scores and composite from the published
payload alone, and count the ones the page's own numbers cannot reproduce.

Measured 2026-10-06 on the live payload (see plan/calculation-transparency.md):
  - 334 of 4,012 stock-category pairs do not reproduce (59 valuation, 275 quality),
    touching 276 of 502 stocks - the page prints the generic metric weight where
    the engine used bank / Piotroski-conditional / renormalised weights.
  - 2 of 502 composites differ from the sum of contributions (coverage discount).
T0 promotes this into tests/test_calculation_reproducibility.py.

Run:  python research/measurements/2026-10-06-calculation-reproducibility.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]

TOL_SCORE = 0.06  # display rounds to 1dp; T0's test should tighten this to 0.01


def load_payload() -> dict:
    txt = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    txt = txt[txt.index("{"):].rstrip().rstrip(";")
    return json.loads(txt)


def category_score_from_generic_weights(s: dict, cat: str, mw: dict):
    num = den = 0.0
    for m, w in mw.get(cat, {}).items():
        p = s["pct"].get(m)
        if w > 0 and p is not None:
            num += p * w
            den += w
    return (num / den) if den else None


def main() -> None:
    D = load_payload()
    sd = D["stock_detail"]
    mw = D["weights"]["metric_weights"]
    cats = list(D["weights"]["factor_weights"].keys())

    pairs = bad = 0
    by_cat: dict = {}
    stocks_bad = set()
    for t, s in sd.items():
        for c in cats:
            cs = s["cat_scores"].get(c)
            if cs is None:
                continue
            rep = category_score_from_generic_weights(s, c, mw)
            if rep is None:
                continue
            pairs += 1
            if abs(rep - cs) > TOL_SCORE:
                bad += 1
                by_cat[c] = by_cat.get(c, 0) + 1
                stocks_bad.add(t)
    print(f"category scores reproduced from the weights the page prints: "
          f"{pairs - bad}/{pairs}  ({bad} do not)")
    print("  not reproducing, by category:", by_cat)
    print(f"  distinct stocks affected: {len(stocks_bad)} of {len(sd)}")

    n = off = 0
    gaps = []
    for t, s in sd.items():
        comp = s.get("composite")
        if comp is None:
            continue
        n += 1
        total = sum((s["contrib"].get(c) or 0) for c in cats)
        if abs(comp - total) > TOL_SCORE:
            off += 1
            gaps.append((round(comp - total, 2), t, comp, round(total, 2)))
    print(f"composite == sum of published contributions: {n - off}/{n}  ({off} differ)")
    for g in sorted(gaps):
        print("  gap, ticker, composite, sum-of-points:", g)


if __name__ == "__main__":
    main()
