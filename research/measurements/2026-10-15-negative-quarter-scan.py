"""Which constituents does Yahoo serve a quarter of negative revenue for, inside the last four?

Found 2026-10-15 on DuPont: Yahoo derives a fiscal Q4 as the year less the nine months to Q3, and
across the Qnity spin-off the year is restated to continuing operations while the nine months are
not - Q4-2025 revenue reads -$1.871bn and every trailing-twelve-month flow the engine sums from the
quarterly statement inherits it. Negative revenue is not a thing a company reports; it is the
signature of that derivation. This counts it across the universe, live, one request per company.

Usage: python research/measurements/2026-10-15-negative-quarter-scan.py   (JSON beside it)
"""
import json
import time
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
tickers = [r["Ticker"] for r in json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))]
hits, failed = [], []
for i, t in enumerate(tickers):
    try:
        q = yf.Ticker(t).quarterly_financials
        if q is None or q.empty or "Total Revenue" not in q.index:
            continue
        row = q.loc["Total Revenue"]
        cols = sorted([c for c in row.index if row[c] == row[c]], reverse=True)[:4]
        neg = [(str(c.date()), float(row[c])) for c in cols if float(row[c]) < 0]
        if neg:
            hits.append({"ticker": t, "negative_quarters": neg,
                         "last4": [(str(c.date()), float(row[c])) for c in cols]})
            print(t, neg, flush=True)
    except Exception as e:  # noqa: BLE001
        failed.append((t, str(e)[:80]))
    time.sleep(0.2)
OUT.write_text(json.dumps({"scanned": len(tickers), "hits": hits, "failed": failed}, indent=1), encoding="utf-8")
print("done", len(hits), "hits;", len(failed), "failed")
