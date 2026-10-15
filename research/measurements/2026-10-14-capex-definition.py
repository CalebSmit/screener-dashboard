"""How much does the capex definition move free-cash-flow yield, and the ranking?

For research/2026-10-14-yahoo-vs-sec.md, section 3. The screener's FCF is operating cash flow
minus Yahoo's "Capital Expenditure" row. That row is not one line of the filing: on the probe of
2026-10-14 it was purchases of property, plant and equipment **plus** purchases of intangibles
(capitalised software, but also bought-in IP - BSX $1.8B of $2.7B), and for companies with a
captive finance arm, the vehicles or equipment bought to lease out (GM $13.8B of $22.6B).

This rebuilds each stock's FCF yield with capex = the PP&E purchases it tags in its own filings
(the SEC cache, same period as the cash flow the run used) and reports how far the yield, its
sector percentile and the approximated composite move. Structure only: no return is read (rule 4).

Run from the repo root:  python research/measurements/2026-10-14-capex-definition.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("xchk", ROOT / "research" / "measurements" / "2026-10-14-yahoo-vs-sec.py")
xchk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xchk)

PPE = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")


def main() -> None:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    facts = pd.read_parquet(xchk.FACTS)
    facts = facts[facts["form"].isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])]
    by_ticker = dict(tuple(facts.groupby("ticker")))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    rows = []
    for t, det in data["stock_detail"].items():
        inp, asof = det.get("inp") or {}, det.get("asof") or {}
        ocf, capex, ev = inp.get("operatingCashFlow"), inp.get("capex"), inp.get("ev_used")
        fcf_used = inp.get("fcf_used")
        if None in (ocf, capex, ev, fcf_used, asof.get("cf")) or not ev or ev <= 0 or t not in by_ticker:
            continue
        ppe, concept, _ = xchk.sec_value(by_ticker[t], PPE, "flow", pd.Timestamp(asof["cf"]))
        if ppe is None:
            continue
        rows.append({"ticker": t, "sector": det.get("sector"), "ev": ev, "yahoo_capex": abs(capex), "sec_ppe": ppe,
                     "fy_used": fcf_used / ev, "fy_ppe": (ocf - ppe) / ev,
                     "pct_used": (det.get("pct") or {}).get("fcf_yield"), "composite": det.get("composite"),
                     "rank": det.get("rank")})
    df = pd.DataFrame(rows)
    df["extra_share"] = (df["yahoo_capex"] - df["sec_ppe"]) / df["yahoo_capex"].where(df["yahoo_capex"] > 0)
    df["dy_pp"] = (df["fy_ppe"] - df["fy_used"]) * 100
    print(f"run {data['kpis'].get('run_timestamp')}: {len(df)} stocks with both definitions available")
    print(f"Yahoo's capex exceeds the PP&E purchases a company files by more than 10% for "
          f"{(df['extra_share'] > .10).sum()} stocks, by more than a third for {(df['extra_share'] > 1/3).sum()}")
    print(f"FCF yield moves by more than 0.5 percentage points for {(df['dy_pp'].abs() > .5).sum()}, "
          f"more than 1pp for {(df['dy_pp'].abs() > 1).sum()}, more than 2pp for {(df['dy_pp'].abs() > 2).sum()}")

    # Sector percentiles on each definition, among the stocks compared (an approximation of the
    # engine's sector percentile, which ranks within sector over every stock with the metric).
    df["pct_a"] = df.groupby("sector")["fy_used"].rank(pct=True) * 100
    df["pct_b"] = df.groupby("sector")["fy_ppe"].rank(pct=True) * 100
    df["dpct"] = df["pct_b"] - df["pct_a"]
    w = cfg["factor_weights"]["valuation"] / 100 * cfg["metric_weights"]["valuation"]["fcf_yield"] / 100
    df["dcomp"] = df["dpct"] * w
    print(f"Sector percentile moves >10 points for {(df['dpct'].abs() > 10).sum()}, >25 for {(df['dpct'].abs() > 25).sum()}; "
          f"at FCF yield's weight in the composite ({w:.3f}) the largest composite move is {df['dcomp'].abs().max():.1f} points")
    top = df.reindex(df["dpct"].abs().sort_values(ascending=False).index).head(15)
    print("\nLargest percentile moves:")
    for _, r in top.iterrows():
        print(f"  {r['ticker']:6} {r['sector'][:22]:22} capex Yahoo {r['yahoo_capex']/1e6:>9,.0f}M  filed PP&E {r['sec_ppe']/1e6:>8,.0f}M"
              f"  FCF yield {r['fy_used']:+.1%} -> {r['fy_ppe']:+.1%}  pct {r['pct_a']:.0f} -> {r['pct_b']:.0f}  rank {r['rank']}")
    in25 = df[df["rank"] <= 25]
    print(f"\nOf the published top 25, {len(in25)} were compared; their composite would move by at most "
          f"{in25['dcomp'].abs().max():.1f} points")


if __name__ == "__main__":
    main()
