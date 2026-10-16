"""Is "Normalized Income" reconstructible from rows the source already serves? - 2026-10-16

Reviewing the open proposal in ``research/2026-10-15-earnings-definition.md`` (score earnings yield
on "Normalized Income" instead of net income as reported), the first question is not whether the
research supports removing one-off items - it does - but whether the resulting input satisfies
CLAUDE.md rows 0.8 and 0.8c: the page prints each metric's own figures through its formula, and an
equation is shown **only if it reproduces**.

The 2026-10-15 scan recorded, per row, the latest *populated* fiscal year. That is not the same
year for every row - UPS's "Normalized Income" is blank for fiscal 2025 while its "Net Income" is
not, and BAC's "Total Unusual Items" is blank for 2025 - so a cross-row comparison off that file
can compare one year's net income with another year's normalisation. This scan reads every row
**from the same column**, so the comparison is within one period, and adds the two rows the source
serves for the adjustment itself:

    Normalized Income  ?=  Net Income - (Total Unusual Items - Tax Effect Of Unusual Items)

If that holds from served rows, the page can print a reproducing equation and the proposal clears
0.8c. It also records, for the trailing-twelve-month window the screener actually scores, how many
of the four quarters populate each row - a row the screener cannot sum is a row the page cannot
show.

Usage: python research/measurements/2026-10-16-normalized-income-same-column.py   (network; JSON beside it)
"""
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
ROWS = ["Net Income", "Net Income Common Stockholders", "Normalized Income", "Total Unusual Items",
        "Total Unusual Items Excluding Goodwill", "Tax Effect Of Unusual Items",
        "Tax Rate For Calcs", "Tax Provision", "Pretax Income", "Operating Income",
        "Total Operating Income As Reported", "Special Income Charges"]


def cell(df, row, col):
    if df is None or df.empty or row not in df.index or col not in df.columns:
        return None
    v = df.loc[row, col]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def latest_annual(a):
    """Every row read from the newest annual column."""
    if a is None or a.empty:
        return None
    col = max(a.columns)
    return {"end": str(pd.Timestamp(col).date()),
            "vals": {r: cell(a, r, col) for r in ROWS}}


def ttm(q):
    """Every row summed over the same last four consecutive quarters, with a populated count."""
    if q is None or q.empty:
        return None
    cols = sorted(q.columns, reverse=True)[:4]
    if len(cols) < 4:
        return None
    gaps = [(pd.Timestamp(cols[i]) - pd.Timestamp(cols[i + 1])).days for i in range(3)]
    if not all(80 <= g <= 100 for g in gaps):
        return None
    out = {"end": str(pd.Timestamp(cols[0]).date()), "vals": {}, "quarters_populated": {}}
    for r in ROWS:
        vs = [cell(q, r, c) for c in cols]
        have = [v for v in vs if v is not None]
        out["quarters_populated"][r] = len(have)
        out["vals"][r] = float(sum(have)) if len(have) == 4 else None
    return out


def main():
    tickers = [r["Ticker"] for r in
               json.loads((ROOT / "sp500_tickers.json").read_text(encoding="utf-8"))]
    out, failed = {}, []
    for i, t in enumerate(tickers, 1):
        try:
            tk = yf.Ticker(t)
            out[t] = {"annual": latest_annual(tk.income_stmt),
                      "ttm": ttm(tk.quarterly_income_stmt)}
        except Exception as e:  # noqa: BLE001
            failed.append((t, str(e)[:80]))
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}", flush=True)
        time.sleep(0.15)
    OUT.write_text(json.dumps({"scanned": len(tickers), "rows": out, "failed": failed}, indent=1),
                   encoding="utf-8")
    print(f"{len(out)} read, {len(failed)} failed -> {OUT.name}")


if __name__ == "__main__":
    main()
