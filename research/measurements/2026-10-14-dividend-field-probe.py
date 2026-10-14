"""Which dividend fields does the source still serve, and what are their units?

`dividend_yield` fell from 402 of 501 stocks to **6** between the 2026-10-13 and
2026-10-14 02:00 runs (`cache/factor_scores_56e2a679e777_*.parquet`), which took
`tests/test_input_churn.py` red: one metric percentile lost for 99% of the universe.
`factor_engine` builds it as `.info["dividendRate"] / currentPrice`, so the question
is whether the source stopped serving `dividendRate` and which sibling field, if any,
is a sound replacement - including its units, which differ between fields and have
changed between yfinance releases.

Usage: python research/measurements/2026-10-14-dividend-field-probe.py [N]
Network. Writes nothing.
"""
from __future__ import annotations

import sys

import yfinance as yf

# Known dividend payers across sectors, plus two non-payers as controls.
PAYERS = ["KO", "JNJ", "PG", "XOM", "JPM", "T", "VZ", "MMM", "CVX", "PEP",
          "IBM", "MO", "ABBV", "O", "MCD", "HD", "WMT", "CAT", "LMT", "SO"]
NON_PAYERS = ["GOOGL", "BRK-B"]

FIELDS = ["dividendRate", "dividendYield", "trailingAnnualDividendRate",
          "trailingAnnualDividendYield", "fiveYearAvgDividendYield",
          "lastDividendValue", "payoutRatio"]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tickers = (PAYERS + NON_PAYERS)[:n] if n < len(PAYERS) else PAYERS + NON_PAYERS
    print(f"yfinance {yf.__version__}; probing {len(tickers)} tickers\n")
    hdr = "ticker".ljust(8) + "price".rjust(9) + "".join(f[:13].rjust(15) for f in FIELDS)
    print(hdr)
    present = {f: 0 for f in FIELDS}
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception as e:  # noqa: BLE001 - a probe reports, it does not raise
            print(f"{t.ljust(8)}  ERROR {type(e).__name__}: {e}")
            continue
        px = info.get("currentPrice") or info.get("regularMarketPrice")
        row = t.ljust(8) + (f"{px:.2f}" if px else "-").rjust(9)
        for f in FIELDS:
            v = info.get(f)
            if v is not None:
                present[f] += 1
            row += (f"{v:.4f}" if isinstance(v, (int, float)) else str(v)).rjust(15)
        print(row)
    print("\nserved for how many of the probed tickers:")
    for f in FIELDS:
        print(f"  {f:<32} {present[f]}/{len(tickers)}")
    print("\nUnits to read off the table above, not from memory:")
    print("  dividendRate / trailingAnnualDividendRate are per-share dollar amounts;")
    print("  a *Yield field near 0.02 is a fraction, near 2.0 is already a percent.")


if __name__ == "__main__":
    main()
