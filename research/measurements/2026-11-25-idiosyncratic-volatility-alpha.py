"""Factor Lab experiment 20: the three-factor alpha of the library's idiosyncratic-volatility gap.

Ang, Hodrick, Xing & Zhang (2006) report the gap between their most and least idiosyncratically
volatile fifths both raw (Table VI Panel B: -1.06% a month) and after the Fama-French three
factors (-1.31%, Newey-West t -7.00). The lab's comparison scores the raw figures only. This prints
the alpha of the library's value-weighted gap (most minus least) on the three factors, regressed
with an intercept, with a Newey-West t (6 lags), two ways:

  * on the three-factor file (F-F_Research_Data_Factors), the factors the paper used;
  * on the published lab series (the 2x3 five-factor file's Mkt-RF, SMB, HML), which is what
    tests/test_factor_lab.py checks the page's sentence against.

Run: python research/measurements/2026-11-25-idiosyncratic-volatility-alpha.py
Reads data/lab/french (python factor_lab.py --fetch). Descriptive only; decides nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import factor_lab as fl  # noqa: E402


def main() -> None:
    d = fl.sort_data("idiosyncratic-volatility")
    fname, mark = fl.FRENCH_FILES["factors_monthly"]
    fd, fv, _ = fl.read_table(fl.RAW_DIR / fname, mark)
    three = {m: r[:3] for m, r in zip(fd, fv)}
    five = fl.factor_data()
    lab = {m: [five["mktrf"][i], five["smb"][i], five["hml"][i]] for i, m in enumerate(five["months"])}
    for label, fac in (("three-factor file", three), ("lab's 2x3 file", lab)):
        for a, b in (("196307", "200012"), ("200101", "209912")):
            r = fl.ff3_alpha(d, fac, a, b)
            print(f"{label:18s} {a}-{b}: n {r['n']}, alpha {r['alpha']:+.3f}% a month, NW t {r['t']:+.2f}, "
                  f"loadings mkt {r['loadings'][0]:+.2f} smb {r['loadings'][1]:+.2f} hml {r['loadings'][2]:+.2f}")


if __name__ == "__main__":
    main()
