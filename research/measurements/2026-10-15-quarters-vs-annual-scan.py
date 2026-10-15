"""Do the source's quarterly revenues for the last fiscal year add up to its annual revenue?

A clean statement does by construction (the source derives Q4 as the year less nine months). When a
company restates for a separation, the source overwrites the year-ago quarter with the restated
comparative while the other quarters keep the old basis - Aptiv on 2026-10-15: quarters 4.825 + 3.199
+ 5.212 + 5.153 = 18.389bn against an annual 20.398bn - and the trailing-twelve-month sum mixes the
two bases. This scans every constituent live (one request each) and records, per company, the
fiscal year's quarters, their sum and the annual figure.

Usage: python research/measurements/2026-10-15-quarters-vs-annual-scan.py   (JSON beside it)
"""
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
tickers = [r["Ticker"] for r in json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))]
rows, failed = [], []
for t in tickers:
    try:
        tk = yf.Ticker(t)
        q, a = tk.quarterly_financials, tk.financials
        if q is None or a is None or q.empty or a.empty or "Total Revenue" not in q.index or "Total Revenue" not in a.index:
            continue
        ar = a.loc["Total Revenue"].dropna()
        if ar.empty:
            continue
        fy_end = max(ar.index)
        annual = float(ar[fy_end])
        qr = q.loc["Total Revenue"].dropna()
        inside = {c: float(v) for c, v in qr.items()
                  if fy_end - pd.Timedelta(days=350) < c <= fy_end + pd.Timedelta(days=5)}
        rows.append({"ticker": t, "fy_end": str(pd.Timestamp(fy_end).date()), "annual": annual,
                     "quarters": {str(pd.Timestamp(c).date()): v for c, v in sorted(inside.items())},
                     "n_quarters": len(inside), "sum": sum(inside.values()) if len(inside) == 4 else None,
                     "latest_quarter": str(pd.Timestamp(max(qr.index)).date())})
    except Exception as e:  # noqa: BLE001
        failed.append((t, str(e)[:80]))
    time.sleep(0.2)
full = [r for r in rows if r["sum"] is not None and r["annual"]]
for r in full:
    r["gap"] = (r["sum"] - r["annual"]) / abs(r["annual"])
off = sorted([r for r in full if abs(r["gap"]) > 0.02], key=lambda r: -abs(r["gap"]))
OUT.write_text(json.dumps({"scanned": len(tickers), "with_four_quarters": len(full),
                           "over_2pct": [{k: r[k] for k in ("ticker", "fy_end", "annual", "sum", "gap", "quarters")}
                                         for r in off],
                           "failed": failed}, indent=1), encoding="utf-8")
print(f"{len(full)} companies with four quarters in the last fiscal year; {len(off)} differ from the annual by >2%")
for r in off:
    print(f"  {r['ticker']:6} fy {r['fy_end']}  quarters sum {r['sum']/1e9:9.3f}bn  annual {r['annual']/1e9:9.3f}bn  gap {r['gap']:+.1%}")
