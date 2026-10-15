"""Yahoo's adjusted and as-reported earnings rows, trailing twelve months, for every constituent.

Found 2026-10-15: the source's quarterly income statement carries both "Operating Income" (which
the screener reads; it leaves out "Special Income Charges") and "Total Operating Income As
Reported" (which equals each company's filed `OperatingIncomeLoss` - GIS 0.886, KHC -4.669,
CVS 4.66, CNC -7.623, BA 4.281, all to the dollar), and both "Net Income" and "Normalized Income"
(net income without "Total Unusual Items"). This records, per company, the four rows' trailing
twelve months (the last four populated quarters, summed only when they are four consecutive
quarters) and the latest fiscal year's, so the choice between them can be measured.

Usage: python research/measurements/2026-10-15-earnings-rows-scan.py   (JSON beside it; network)
"""
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
ROWS = ["Operating Income", "Total Operating Income As Reported", "Net Income", "Normalized Income",
        "Total Unusual Items", "Special Income Charges", "Net Income Common Stockholders"]


def ttm(q: pd.DataFrame, row: str):
    if q is None or q.empty or row not in q.index:
        return None
    s = q.loc[row]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    s = s.dropna()
    cols = sorted(s.index, reverse=True)[:4]
    if len(cols) < 4:
        return None
    gaps = [(cols[i] - cols[i + 1]).days for i in range(3)]
    if not all(80 <= g <= 100 for g in gaps):
        return None
    return float(sum(s[c] for c in cols))


def annual(a: pd.DataFrame, row: str):
    if a is None or a.empty or row not in a.index:
        return None
    s = a.loc[row]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    s = s.dropna()
    if s.empty:
        return None
    c = max(s.index)
    return {"end": str(pd.Timestamp(c).date()), "val": float(s[c])}


tickers = [r["Ticker"] for r in json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))]
out, failed = {}, []
for t in tickers:
    try:
        tk = yf.Ticker(t)
        q, a = tk.quarterly_financials, tk.income_stmt
        out[t] = {"ttm": {r: ttm(q, r) for r in ROWS}, "annual": {r: annual(a, r) for r in ROWS}}
    except Exception as e:  # noqa: BLE001
        failed.append((t, str(e)[:80]))
    time.sleep(0.15)
OUT.write_text(json.dumps({"scanned": len(tickers), "rows": out, "failed": failed}, indent=1), encoding="utf-8")
have = sum(1 for v in out.values() if v["ttm"]["Total Operating Income As Reported"] is not None)
print(f"{len(out)} companies read, {len(failed)} failed; as-reported TTM for {have}")
