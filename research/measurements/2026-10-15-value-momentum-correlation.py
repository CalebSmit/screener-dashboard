"""Value and momentum together: is the negative correlation the momentum regime rule leans on real,
and does it hold among large companies? (research/2026-10-15-value-momentum-correlation.md)

The 2026-10-12 changelog entry sends weight freed from Momentum in a panic half to Valuation, partly
because value and momentum hedge each other. Asness, Moskowitz & Pedersen (2013), Table I Panel A,
US stocks 1972-01..2011-07 (their universe is the large, liquid stocks): value P3-P1 Sharpe 0.29,
momentum 0.33, the 50/50 combination 0.63; correlation -0.53 (P3-P1), -0.65 (rank-weighted factors).

Rebuilt from Kenneth French's 2x3 portfolios (already in data/lab/french/): value = high minus low
book-to-market, momentum = high minus low prior 12-2 return, each the average of the small and big
halves ("all") or the big half alone ("big"). Sharpe ratios are annualised from monthly returns;
correlations are of the raw spreads and of their residuals on the market (AMP report the latter).
Not decision-grade for anything (rule 5); it checks a premise, it sets nothing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import factor_lab as fl  # noqa: E402

RAW = ROOT / "data" / "lab" / "french"


def table(key: str) -> dict:
    fname, mark = fl.FRENCH_FILES[key]
    dates, rows, _ = fl.read_table(RAW / fname, mark)
    return {d: r for d, r in zip(dates, rows)}


def main() -> None:
    val, mom, ff = table("split_value"), table("split_momentum"), table("factors_monthly")
    months = sorted(set(val) & set(mom) & set(ff))
    # columns: small lo, small mid, small hi, big lo, big mid, big hi
    series = {
        "value_all": [(val[m][2] + val[m][5]) / 2 - (val[m][0] + val[m][3]) / 2 for m in months],
        "mom_all": [(mom[m][2] + mom[m][5]) / 2 - (mom[m][0] + mom[m][3]) / 2 for m in months],
        "value_big": [val[m][5] - val[m][3] for m in months],
        "mom_big": [mom[m][5] - mom[m][3] for m in months],
        "mkt": [ff[m][0] for m in months],
    }
    idx = {m: i for i, m in enumerate(months)}

    def window(a, b):
        return [idx[m] for m in months if a <= m <= b]

    def sharpe(x):
        x = np.asarray(x)
        return float(x.mean() / x.std(ddof=1) * math.sqrt(12))

    def resid(y, x):
        y, x = np.asarray(y), np.asarray(x)
        b = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        return y - b * x

    last = months[-1]
    print(f"{'window':24} {'set':4} {'SR val':>7} {'SR mom':>7} {'SR 50/50':>9} {'corr':>6} {'corr resid':>11}  months")
    for label, a, b in (("AMP 1972-01..2011-07", "197201", "201107"), ("after 2011-08", "201108", last),
                        ("since 2007", "200701", last), ("all 1927-", months[0], last)):
        w = window(a, b)
        for s in ("all", "big"):
            v = [series[f"value_{s}"][i] for i in w]
            mo = [series[f"mom_{s}"][i] for i in w]
            mk = [series["mkt"][i] for i in w]
            combo = [(x + y) / 2 for x, y in zip(v, mo)]
            c = float(np.corrcoef(v, mo)[0, 1])
            cr = float(np.corrcoef(resid(v, mk), resid(mo, mk))[0, 1])
            print(f"{label:24} {s:4} {sharpe(v):7.2f} {sharpe(mo):7.2f} {sharpe(combo):9.2f} {c:6.2f} {cr:11.2f}  {len(w)}")


if __name__ == "__main__":
    main()
