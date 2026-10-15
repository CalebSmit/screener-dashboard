"""Is the operating-income gap the source's definition, or my trailing-twelve-month rebuild?

`2026-10-15-operating-income-vs-filings.py` found the run's `ebit` disagreeing with the
companies' filed `OperatingIncomeLoss` by over 10% for 70 of 322 stocks, against 8 of 410 for
revenue and 1 of 411 for operating cash flow. Two explanations fit that:

  (a) the source's "Operating Income" row excludes charges the company puts inside operating
      income (impairments, restructuring, litigation) - a definitional difference, and the
      interesting one, because `ebit` feeds ROIC, EV/EBITDA and net debt / EBITDA;
  (b) my trailing rebuild (`sec_crosscheck.flow_ttm`: ytd + prior fy - prior ytd) going wrong
      on this concept in particular.

This separates them. The run also carries `ebit_annual` - the source's own **annual** operating
income - which can be compared with the company's filed **fiscal year** directly: one period,
one concept, no reconstruction at all. If the gap is still there, (b) is ruled out.

Offline. Usage:  python research/measurements/2026-10-15-operating-income-annual-check.py
"""
from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, ".")

import sec_crosscheck as xc  # noqa: E402
import sec_fundamentals as sf  # noqa: E402

FAR = 0.10


def main():
    with open("dashboard_data.js", encoding="utf-8") as fh:
        payload = json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))
    facts = pd.read_parquet(sf.FACTS_PATH)
    facts = facts[facts["form"].isin(xc.FORMS) & (facts["concept"] == "OperatingIncomeLoss")
                  & facts["start"].notna()]
    facts = facts.assign(days=(facts["end"] - facts["start"]).dt.days)
    fy = facts[(facts["days"] >= 350) & (facts["days"] <= 380)]
    by_ticker = dict(tuple(fy.groupby("ticker")))

    rows = []
    for t, det in payload["stock_detail"].items():
        inp = det.get("inp") or {}
        ttm, ann = inp.get("ebit"), inp.get("ebit_annual")
        g = by_ticker.get(t)
        if ann is None or g is None or g.empty:
            continue
        # the newest fiscal year the company has filed, as first filed - but only if it is the
        # one the run's annual statement covers. Several companies stopped tagging
        # `OperatingIncomeLoss` years ago (Devon 2016, Mid-America 2012), and comparing a 2026
        # figure with a 2012 fiscal year measures nothing.
        when = (det.get("asof") or {}).get("is")
        if when is None:
            continue
        newest = g["end"].max()
        if (pd.Timestamp(when) - newest).days > 400 or newest > pd.Timestamp(when):
            rows.append({"ticker": t, "rank": det.get("rank"), "sector": det.get("sector"),
                         "fy_end": newest.date(), "yahoo_annual": ann, "filed_annual": None,
                         "annual_gap": None, "ttm": ttm, "behind": False, "stale_tag": True})
            continue
        filed = xc.first_filed(g[g["end"] == newest])
        if filed is None:
            continue
        rows.append({
            "ticker": t, "rank": det.get("rank"), "sector": det.get("sector"),
            "fy_end": newest.date(), "yahoo_annual": ann, "filed_annual": filed,
            "annual_gap": (ann - filed) / max(abs(ann), abs(filed), 1.0),
            "ttm": ttm,
            "behind": bool(det.get("newer_filing")),
            "stale_tag": False,
        })
    allrows = pd.DataFrame(rows)
    stale = allrows[allrows["stale_tag"]]
    df = allrows[~allrows["stale_tag"]].copy()
    df["abs_annual"] = df["annual_gap"].abs()
    print(f"{len(allrows)} stocks carry an annual operating income and have filed the concept; "
          f"{len(stale)} last tagged `OperatingIncomeLoss` for a fiscal year the run's statement "
          f"does not cover (oldest {stale['fy_end'].min() if len(stale) else '-'}) and are "
          f"excluded as not comparable\n")
    print(f"{len(df)} comparable\n")
    print(f"annual operating income, one period and one concept, no rebuild:")
    print(f"  within 0.5%: {(df['abs_annual'] <= 0.005).sum():>4} ({(df['abs_annual'] <= 0.005).mean():.1%})")
    print(f"  within   2%: {(df['abs_annual'] <= 0.02).sum():>4}")
    print(f"  over    10%: {(df['abs_annual'] > FAR).sum():>4}")
    print(f"  median |gap|: {df['abs_annual'].median():.3%}")

    far = df[df["abs_annual"] > FAR].sort_values("abs_annual", ascending=False)
    print(f"\nof the {len(far)} over 10% on the annual figure, {far['behind'].sum()} are also "
          f"a period behind their filings")
    print(f"\n{'ticker':<8}{'rank':>5}{'fy end':>13}{'yahoo $bn':>12}{'filed $bn':>12}{'gap':>9}  sector")
    for _, r in far.head(30).iterrows():
        print(f"{r['ticker']:<8}{str(r['rank']):>5}{str(r['fy_end']):>13}"
              f"{r['yahoo_annual'] / 1e9:>12.3f}{r['filed_annual'] / 1e9:>12.3f}"
              f"{r['annual_gap']:>+9.1%}  {r['sector']}")

    same_sign = ((df["yahoo_annual"] > 0) != (df["filed_annual"] > 0)).sum()
    print(f"\nopposite signs on the annual figure: {same_sign} stocks")
    bigger = (df.loc[df['abs_annual'] > FAR, 'yahoo_annual'].abs()
              > df.loc[df['abs_annual'] > FAR, 'filed_annual'].abs()).sum()
    print(f"of the {len(far)} over 10%, the source's figure is the larger for {bigger} "
          f"({bigger / max(len(far), 1):.0%}) - the direction a charge excluded would give")


if __name__ == "__main__":
    main()
