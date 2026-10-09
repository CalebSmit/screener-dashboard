"""Can the fundamentals half of the backtest be made point-in-time from free SEC data? (2026-10-09)

`plan/backtest-v2.md` step 1's one unmeasured piece: the 49.0 points of composite weight whose
inputs come from filings. SEC XBRL `companyfacts` returns every fact a filer has reported, each
with the date it was **filed** - so for any past month we can ask what was knowable then.

For the same panel as the 2026-10-01 look-ahead measurement (rebalance month-ends 2020-01 ..
2026-09, the current S&P 500 names), and for each input concept the fundamentals metrics need:

1. **coverage** - share of name-months with a value that had been *filed* by that month-end and
   whose period ended within the previous 15 months (i.e. a usable latest figure existed);
2. **reporting lag** - filed date minus period end, for 10-Q and 10-K facts separately;
3. **what v1 assumes** - v1 holds today's value for every month, so its implied "lag" is negative.

Downloads one `companyfacts` JSON per company (~500 requests, SEC identity required), keeps only
the concepts below, and caches a compact parquet in `data/sec/pit/facts.parquet` (gitignored).
Writes its summary beside this script as JSON.

    python research/measurements/2026-10-09-xbrl-point-in-time-census.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "data" / "sec" / "pit"
FACTS = OUT_DIR / "facts.parquet"
SUMMARY = Path(__file__).with_suffix(".json")

# Input -> XBRL us-gaap concepts, first with data wins (per company).
INPUTS = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "total_assets": ("Assets",),
    "equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "long_term_debt": ("LongTermDebt", "LongTermDebtNoncurrent"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "d_and_a": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"),
    "current_assets": ("AssetsCurrent",),
    "current_liabilities": ("LiabilitiesCurrent",),
}
CONCEPTS = sorted({c for v in INPUTS.values() for c in v})
WINDOW = pd.date_range("2020-01-31", "2026-09-30", freq="ME")


def download(tickers: list[str], log=print) -> pd.DataFrame:
    import insider_activity as ia
    ua = ia.user_agent()
    if not ua:
        raise SystemExit("SEC identity needed (data/sec/user_agent.txt)")
    e = ia.Edgar(ua)
    cmap = ia.ticker_map(e)
    rows = []
    t0 = time.time()
    for i, t in enumerate(tickers):
        cik = cmap.get(t)
        if cik is None:
            continue
        try:
            j = e.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
        except Exception as ex:  # noqa: BLE001
            log(f"  {t}: {type(ex).__name__}")
            continue
        gaap = (j.get("facts") or {}).get("us-gaap") or {}
        for c in CONCEPTS:
            for unit, facts in ((gaap.get(c) or {}).get("units") or {}).items():
                if unit != "USD":
                    continue
                for f in facts:
                    if f.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                        continue
                    rows.append((t, c, f.get("start"), f.get("end"), f.get("filed"), f.get("form"), f.get("fp"), f.get("val")))
        if i % 50 == 0:
            log(f"  {i}/{len(tickers)} companies, {len(rows)} facts, {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "form", "fp", "val"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FACTS)
    return df


def census(df: pd.DataFrame, tickers: list[str]) -> dict:
    out = {"panel": {"months": len(WINDOW), "names": len(tickers), "name_months": len(WINDOW) * len(tickers),
                     "window": [str(WINDOW[0].date()), str(WINDOW[-1].date())]}, "inputs": {}}
    for name, concepts in INPUTS.items():
        sub = df[df["concept"].isin(concepts)].dropna(subset=["end", "filed"])
        # per company, the first concept (in priority order) that has any facts
        pick = {}
        for t, g in sub.groupby("ticker"):
            for c in concepts:
                if (g["concept"] == c).any():
                    pick[t] = g[g["concept"] == c]
                    break
        covered = 0
        for t in tickers:
            g = pick.get(t)
            if g is None:
                continue
            filed = g["filed"].values
            ends = g["end"].values
            for m in WINDOW:
                mv = m.to_datetime64()
                ok = (filed <= mv) & (ends >= (m - pd.DateOffset(months=15)).to_datetime64())
                covered += bool(ok.any())
        lag = (sub["filed"] - sub["end"]).dt.days
        q = lag[sub["form"].str.startswith("10-Q")]
        k = lag[sub["form"].str.startswith("10-K")]
        out["inputs"][name] = {
            "concepts": list(concepts),
            "companies_with_facts": len(pick),
            "name_month_coverage": round(covered / (len(WINDOW) * len(tickers)), 4),
            "lag_days_10q": {"p50": float(q.median()) if len(q) else None, "p90": float(q.quantile(0.9)) if len(q) else None},
            "lag_days_10k": {"p50": float(k.median()) if len(k) else None, "p90": float(k.quantile(0.9)) if len(k) else None},
        }
    return out


def main() -> dict:
    tickers = [x["Ticker"] for x in json.loads((ROOT / "sp500_tickers.json").read_text())]
    df = pd.read_parquet(FACTS) if FACTS.exists() else download(tickers)
    out = census(df, tickers)
    out["facts"] = int(len(df))
    SUMMARY.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
