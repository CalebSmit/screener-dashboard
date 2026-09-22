"""Is the sector-relative volatility percentile safe to use as a cross-holding
risk comparison on the My Holdings panel?

Context: research/2026-09-21-position-sizing-and-how-much.md section 8.4 item 2
proposes showing "the volatility percentile of each holding" so a reader can see
that "a position twice as volatile as another contributes twice the risk at
equal dollars".

This script tests whether the percentile the payload already carries can carry
that claim. It cannot, and the numbers below are why the panel shows RAW
annualised volatility as the cross-holding comparison and labels the percentile
as sector-relative.

Run:  python research/measurements/2026-09-22-holdings-risk-comparability.py
"""

import json
import os
import sys
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_payload(path=None):
    path = path or os.path.join(ROOT, "dashboard_data.js")
    src = open(path, encoding="utf-8").read()
    return json.loads(src[src.index("=") + 1:].strip().rstrip(";"))


def main():
    payload = load_payload()
    sd = payload["stock_detail"]

    rows = []
    for t, s in sd.items():
        raw = (s.get("raw") or {}).get("volatility")
        pct = (s.get("pct") or {}).get("volatility")
        if raw is None or pct is None:
            continue
        rows.append((t, s.get("sector") or "Unclassified", float(raw), float(pct)))

    print(f"Universe with both raw and pct volatility: {len(rows)} of {len(sd)}")
    vols = sorted(r[2] for r in rows)
    print(
        f"Raw annualised volatility: min {vols[0]:.3f}  median {median(vols):.3f}  "
        f"max {vols[-1]:.3f}"
    )

    # ---- 1. Direction check -------------------------------------------------
    # METRIC_DIR['volatility'] is False, so the percentile is inverted:
    # a HIGH percentile means LOW volatility. Confirm empirically.
    hi = [r for r in rows if r[3] >= 90]
    lo = [r for r in rows if r[3] <= 10]
    print(
        f"\nDirection: pct>=90 median raw vol {median([r[2] for r in hi]):.3f} "
        f"vs pct<=10 median raw vol {median([r[2] for r in lo]):.3f}"
    )
    print("  -> a HIGH percentile is a LOW-volatility stock. Labelling matters.")

    # ---- 2. Cross-sector comparability -------------------------------------
    # The decision-relevant question is 'which of MY holdings is riskier'.
    # Find pairs in different sectors where the percentile ranks them one way
    # and the raw volatility ranks them the other.
    inversions = []
    worst = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a[1] == b[1]:
                continue
            # a looks SAFER than b on the percentile (higher pct = lower vol)
            # but is actually MORE volatile in raw terms.
            if a[3] > b[3] and a[2] > b[2]:
                gap = a[2] - b[2]
                inversions.append(gap)
                if worst is None or gap > worst[0]:
                    worst = (gap, a, b)
            elif b[3] > a[3] and b[2] > a[2]:
                gap = b[2] - a[2]
                inversions.append(gap)
                if worst is None or gap > worst[0]:
                    worst = (gap, b, a)

    total_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][1] != rows[j][1]:
                total_pairs += 1

    share = len(inversions) / total_pairs * 100 if total_pairs else 0
    print(f"\nCross-sector pairs: {total_pairs:,}")
    print(
        f"Pairs where the sector percentile ranks risk BACKWARDS vs raw vol: "
        f"{len(inversions):,} ({share:.1f}%)"
    )
    if worst:
        gap, a, b = worst
        print(
            f"  Worst case: {a[0]} ({a[1]}) pct {a[3]:.1f} raw {a[2]:.3f}  "
            f"vs {b[0]} ({b[1]}) pct {b[3]:.1f} raw {b[2]:.3f}"
        )
        print(
            f"  -> {a[0]} reads as the safer holding and is {gap:.3f} "
            f"({a[2] / b[2]:.2f}x) more volatile."
        )

    # ---- 3. Does raw volatility actually span enough to matter? -------------
    ratio = vols[-1] / vols[0]
    p10 = vols[int(len(vols) * 0.10)]
    p90 = vols[int(len(vols) * 0.90)]
    print(
        f"\nRaw vol p10 {p10:.3f} -> p90 {p90:.3f} = {p90 / p10:.2f}x spread; "
        f"full range {ratio:.2f}x"
    )
    print(
        "  -> at equal dollars the p90 name contributes "
        f"{p90 / p10:.2f}x the risk of the p10 name. That is the arithmetic "
        "the panel states."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
