"""Do Yahoo's share counts - and the market value every valuation ratio rests on - agree with the filings?

For research/2026-10-14-yahoo-vs-sec.md, section 2b. Market value = price x shares; enterprise
value, earnings yield, free-cash-flow yield and the size tilt all rest on it. The SEC cache holds
each company's weighted-average diluted shares per reported quarter; this compares the latest
quarter's figure with (1) `financials.shares_outstanding` in the published payload, which is a
display figure, and (2) the share count Yahoo's market value implies (market value / price),
which is what scoring actually reads.

Weighted-average diluted shares are not point-in-time shares outstanding: buybacks, issuance and
dilution move them a few percent, so small gaps are normal. What this looks for is the large gap
- a share class left out, a scale error, a split counted twice. No returns are read.

Run from the repo root:  python research/measurements/2026-10-14-shares-vs-sec.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONCEPTS = ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
            "WeightedAverageNumberOfSharesOutstandingBasic")


def main() -> None:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    facts = facts[facts["concept"].isin(CONCEPTS) & facts["start"].notna()]
    facts = facts[(facts["end"] - facts["start"]).dt.days.between(80, 100)]       # quarterly figures
    by_ticker = dict(tuple(facts.groupby("ticker")))
    rows = []
    for t, det in data["stock_detail"].items():
        fin = det.get("financials") or {}
        if t not in by_ticker:
            continue
        f = by_ticker[t]
        for c in CONCEPTS:
            g = f[f["concept"] == c]
            if g.empty:
                continue
            last = g[g["end"] == g["end"].max()].sort_values("filed").iloc[0]
            rows.append({"ticker": t, "sector": det.get("sector"), "sec": float(last["val"]), "quarter": last["end"],
                         "yahoo_shares": fin.get("shares_outstanding"), "mcap": fin.get("market_cap"),
                         "price": det.get("price")})
            break
    df = pd.DataFrame(rows)
    recent = df[df["quarter"] >= pd.Timestamp.now() - pd.Timedelta(days=270)].copy()
    print(f"run {data['kpis'].get('run_timestamp')}: {len(df)} stocks with a filed count, "
          f"{len(recent)} with one from the last 9 months (only those are compared)")

    s = recent[recent["yahoo_shares"].notna()].copy()
    s["ratio"] = s["yahoo_shares"] / s["sec"]
    a = (s["ratio"] - 1).abs()
    print(f"\n(1) Yahoo's displayed share count vs filed: within 5% {(a <= .05).mean():.1%}; "
          f"over 10% apart {(a > .10).sum()}; over 25% {(a > .25).sum()}")

    m = recent[recent["mcap"].notna() & recent["price"].notna() & (recent["price"] > 0)].copy()
    m["implied"] = m["mcap"] / m["price"]
    m["ratio"] = m["implied"] / m["sec"]
    b = (m["ratio"] - 1).abs()
    print(f"(2) Market value / price vs filed (what scoring reads): within 5% {(b <= .05).mean():.1%}; "
          f"over 10% apart {(b > .10).sum()}; over 25% {(b > .25).sum()}")
    print("\nLargest gaps in (2):")
    for _, r in m.reindex(b.sort_values(ascending=False).index).head(20).iterrows():
        ys = f"{r['yahoo_shares']/1e6:,.1f}M" if pd.notna(r["yahoo_shares"]) else "-"
        print(f"  {r['ticker']:6} {str(r['sector'])[:22]:22} implied {r['implied']/1e6:>10,.1f}M  filed {r['sec']/1e6:>10,.1f}M"
              f"  ratio {r['ratio']:7.3f}  (displayed count {ys}; filed quarter {r['quarter'].date()})")


if __name__ == "__main__":
    main()
