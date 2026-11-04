"""Re-measure the review-cadence turnover the page quotes, on the history it now has.

The holdings and What Changed panels say "on this screener's own history, acting on the top 25 at
every run implies 122% monthly one-sided turnover against 24% reviewing the same rule monthly"
(generate_dashboard.CADENCE_TURNOVER). Those two numbers were measured on 2026-09-14 over 23 runs
(research/2026-09-14-sell-discipline-and-hold-bands.md section 8.2). This recomputes them from the
published comparable-run history (``history.series``, one rank per stock per run), with the same
definitions:

- one-sided turnover of a review = names entering the top 25 / 25;
- every run: the sum over consecutive comparable runs, divided by the months the window spans;
- monthly: the mean over disjoint pairs of runs 28-35 days apart, walked forward through the dense
  weekday stretch (the earlier runs are weeks apart, so pairs there are not a month's review).

    python research/measurements/2026-11-03-cadence-turnover-remeasured.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = 25


def main() -> int:
    s = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    H = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))["history"]
    dates = [date.fromisoformat(d) for d in H["dates"]]
    tops = []
    for i in range(len(dates)):
        ranked = sorted(((ser["r"][i], t) for t, ser in H["series"].items() if ser["r"][i] is not None))
        tops.append({t for _, t in ranked[:TOP]})

    def entering(a, b):
        return len(tops[b] - tops[a]) / TOP

    every = [entering(i, i + 1) for i in range(len(dates) - 1)]
    span_months = (dates[-1] - dates[0]).days / (365.25 / 12)
    print(f"{len(dates)} comparable runs, {dates[0]} to {dates[-1]} ({span_months:.1f} months)")
    print(f"every run: {100 * sum(every) / span_months:.1f}% monthly one-sided turnover")
    # the dense weekday stretch only, as section 8.2 measured (gaps of 4 days or less)
    start = max(i for i in range(len(dates)) if i == 0 or (dates[i] - dates[i - 1]).days > 4)
    dense = every[start:]
    dm = (dates[-1] - dates[start]).days / (365.25 / 12)
    print(f"  dense stretch {dates[start]} to {dates[-1]} ({len(dense)} reviews): "
          f"{100 * sum(dense) / dm:.1f}% monthly")
    pairs, i = [], start
    while True:
        j = next((k for k in range(i + 1, len(dates)) if 28 <= (dates[k] - dates[i]).days <= 35), None)
        if j is None:
            break
        pairs.append(entering(i, j))
        i = j
    if pairs:
        print(f"monthly: {100 * sum(pairs) / len(pairs):.1f}% over {len(pairs)} disjoint pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
