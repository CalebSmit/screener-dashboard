"""Do the two skew constructions produce values in the range the paper reports?

Xing, Zhang & Zhao (2010) Table 1 (p. 647, read from the paper) reports SKEW across 1996-2005,
averaging ~840 firms a week:

    mean 6.40%   5% -0.99   25% 2.40   50% 4.76   75% 8.43   95% 19.92

That is a plausibility benchmark, not a target: a 2026 single-expiry snapshot of 500 large
companies is a different sample from a decade of weekly cross-sections over every optioned US
stock, so the levels need not agree. What it *can* catch is a construction producing values no
version of the measure has ever taken.

Input is the paired sample printed by ``2026-10-14-skew-construction-paired.py``, pasted in
below so this is reproducible without re-reading the network (option quotes are not
retrospectively available).

Usage: python -I -X utf8 research/measurements/2026-10-14-skew-distribution-vs-paper.py
"""
from __future__ import annotations

import pandas as pd

# Ticker, old construction, new construction - from the 2026-10-14 paired run, session date
# 2026-10-14. CDNS and LHX had no usable chain; EBAY had no old value.
PAIRED = [
    ("MMM", -0.009033, 0.008118), ("GOOG", 0.007172, -0.015930), ("APP", -0.033355, -0.037414),
    ("BRK-B", 0.050048, 0.025421), ("CMG", -0.028076, -0.001953), ("COP", 0.145812, 0.041137),
    ("DECK", -0.026428, 0.006348), ("EBAY", None, -0.088988), ("EXC", 0.110839, -0.005859),
    ("F", -0.035156, -0.042968), ("GL", 0.465633, 0.061767), ("HUM", -0.003494, -0.012878),
    ("IQV", -0.030761, -0.071471), ("MLM", 0.113875, -0.014587), ("MNST", -0.051879, -0.083495),
    ("NRG", -0.030029, 0.004639), ("PNR", 0.015625, 0.031494), ("QCOM", 0.024566, 0.000549),
    ("SPGI", 0.038238, -0.001099), ("STT", 0.042663, 0.032226), ("TXT", -0.010620, -0.054931),
    ("URI", -0.014313, -0.019119), ("GWW", 0.060951, 0.021683),
]

PAPER = {"mean": 6.40, "p5": -0.99, "p25": 2.40, "p50": 4.76, "p75": 8.43, "p95": 19.92}

# Ticker, OTM put strike, ATM call strike, spot - the same run. The paper's two legs sit a
# nominal 0.05 of moneyness apart (0.95 and 1.00); a real strike grid puts them wherever it
# puts them, and the realised gap is what the measure is actually the slope over.
LEGS = [
    ("MMM", 155, 160, 164.460007), ("GOOG", 325, 345, 343.649994), ("APP", 245, 260, 262.190002),
    ("BRK-B", 500, 525, 526.349976), ("CMG", 28, 31, 30.520000), ("COP", 135, 140, 142.339996),
    ("DECK", 78, 83, 82.820000), ("EBAY", 111, 117, 116.879997), ("EXC", 40, 43, 42.599998),
    ("F", 11, 12, 12.050000), ("GL", 155, 165, 167.410004), ("HUM", 405, 430, 427.399994),
    ("IQV", 240, 260, 262.010010), ("MLM", 440, 480, 471.100006), ("MNST", 42, 44, 44.220001),
    ("NRG", 95, 103, 100.849998), ("PNR", 47.5, 52.5, 52.110001), ("QCOM", 160, 175, 172.619995),
    ("SPGI", 385, 415, 408.570007), ("STT", 155, 165, 166.889999), ("TXT", 70, 75, 73.800003),
    ("URI", 980, 1020, 1034.810059), ("GWW", 1260, 1330, 1330.619995),
]


def main() -> int:
    df = pd.DataFrame(PAIRED, columns=["Ticker", "old", "new"])
    print("Xing, Zhang & Zhao (2010) Table 1, SKEW in IV points, 1996-2005, ~840 firms/week:")
    print("   " + "  ".join(f"{k} {v:+.2f}" for k, v in PAPER.items()))
    print()
    for name in ("old", "new"):
        s = (df[name].dropna()) * 100
        print(f"{name:>4}  n {len(s):2d}  mean {s.mean():+6.2f}  "
              f"5% {s.quantile(.05):+6.2f}  25% {s.quantile(.25):+6.2f}  "
              f"50% {s.median():+6.2f}  75% {s.quantile(.75):+6.2f}  "
              f"95% {s.quantile(.95):+6.2f}  min {s.min():+7.2f}  max {s.max():+7.2f}")
    print()
    old = df["old"].dropna() * 100
    new = df["new"].dropna() * 100
    beyond = PAPER["p95"]
    print(f"beyond the paper's 95th percentile ({beyond:+.2f} pts):")
    print(f"   old {int((old.abs() > beyond).sum())} of {len(old)}"
          f"   -> {sorted(round(v, 1) for v in old[old.abs() > beyond])}")
    print(f"   new {int((new.abs() > beyond).sum())} of {len(new)}"
          f"   -> {sorted(round(v, 1) for v in new[new.abs() > beyond])}")
    print()
    print("spread of the measure (95th - 5th percentile), in IV points:")
    print(f"   paper {PAPER['p95'] - PAPER['p5']:.2f}   "
          f"old {old.quantile(.95) - old.quantile(.05):.2f}   "
          f"new {new.quantile(.95) - new.quantile(.05):.2f}")
    print()

    legs = pd.DataFrame(LEGS, columns=["Ticker", "put_k", "call_k", "spot"])
    legs["put_m"] = legs["put_k"] / legs["spot"]
    legs["call_m"] = legs["call_k"] / legs["spot"]
    legs["gap"] = legs["call_m"] - legs["put_m"]
    print("the two legs' moneyness, which the strike grid decides (the paper's nominal gap is "
          "0.95 -> 1.00, i.e. 0.050):")
    print(f"   OTM put  mean {legs['put_m'].mean():.3f}  min {legs['put_m'].min():.3f}  "
          f"max {legs['put_m'].max():.3f}   (eligible band: 0.80-0.95)")
    print(f"   ATM call mean {legs['call_m'].mean():.3f}  min {legs['call_m'].min():.3f}  "
          f"max {legs['call_m'].max():.3f}   (eligible band: 0.95-1.05)")
    print(f"   gap      mean {legs['gap'].mean():.3f}  min {legs['gap'].min():.3f}  "
          f"max {legs['gap'].max():.3f}  median {legs['gap'].median():.3f}")
    narrow = legs[legs["gap"] < 0.05]
    print(f"   narrower than the paper's nominal 0.050: {len(narrow)} of {len(legs)}  "
          f"-> {list(narrow['Ticker'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
