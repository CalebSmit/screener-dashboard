"""Value and momentum among big companies, period by period, from the Factor Lab's published data.

For research/2026-10-12-value-and-momentum-among-large-companies.md. Reads only the committed
lab_data.js (derived from Kenneth R. French's Data Library), so it runs from a clean checkout.
These are the library's value-weighted portfolios of the larger half of NYSE companies - not the
screener's ranking, and nothing here is a return of that ranking (CLAUDE.md rule 4).

Run from the repo root:  python research/measurements/2026-10-12-large-company-value-momentum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import factor_lab as fl  # noqa: E402

PERIODS = [("196307", "196912"), ("197001", "197912"), ("198001", "198912"), ("199001", "199912"),
           ("200001", "200912"), ("201001", "201912"), ("202001", "209912")]
PARTS = [("big", "factor"), ("long", "held side"), ("short", "shorted side")]


def cell(s: dict | None) -> str:
    return "     n/a     " if s is None else f"{s['mean']:+.2f} ({s['t']:+.2f})"


def main() -> None:
    pub = fl.load_published()
    split, market = pub["split"], pub["market"]
    months = split["months"]
    print(f"months {months[0]}..{months[-1]} ({len(months)}); average month in %, t-statistic in brackets")

    for key in ("value", "momentum", "profitability", "investment"):
        print(f"\n{key}, among big companies")
        print(f"  {'period':16}" + "".join(f"{label:>16}" for _, label in PARTS) + f"{'among small':>16}")
        for a, b in PERIODS + [("196307", "209912"), ("200001", "209912")]:
            r = fl.split_run(split, a, b)["factors"][key]
            n = len(fl.split_run(split, a, b)["months"])
            print(f"  {a}-{min(b, months[-1])} " + "".join(f"{cell(r[p]):>16}" for p, _ in PARTS)
                  + f"{cell(r['small']):>16}   {n} months")

    # How unusual is the stretch since 2000? Every window of the same length inside the record.
    n_late = len([m for m in months if m >= "200001"])
    for key in ("momentum", "value"):
        series = split[key]["big"]
        means = [sum(series[i:i + n_late]) / n_late for i in range(len(series) - n_late + 1)]
        latest = means[-1]
        print(f"\n{key}, big companies: {len(means)} windows of {n_late} months; the latest averages {latest:+.2f}%, "
              f"the lowest of all is {min(means):+.2f}% and the highest {max(means):+.2f}%; "
              f"{sum(m <= latest for m in means)} window(s) at or below the latest")

    # Momentum among big companies inside and outside the screener's panic state, as the state
    # was known before each month (experiment 2's series: 24-month market return, 126-day volatility).
    state_months = pub["momentum"]["months"]
    trail, vol = market["trail"]["24"], market["vol"]["126"]
    at = {m: i for i, m in enumerate(state_months)}
    for a, b in (("196307", "209912"), ("200001", "209912")):
        inside, outside = [], []
        for i, m in enumerate(months):
            if not (a <= m <= b) or m not in at:
                continue
            j = at[m]
            if trail[j] is None or vol[j] is None:
                continue
            panic = trail[j] < 0 and vol[j] > 0.206
            (inside if panic else outside).append(split["momentum"]["big"][i])
        print(f"\nmomentum among big companies, {a}-{min(b, months[-1])}: in the panic state "
              f"{cell(fl.mean_t(inside))} over {len(inside)} months; outside it {cell(fl.mean_t(outside))} "
              f"over {len(outside)} months")


if __name__ == "__main__":
    main()
