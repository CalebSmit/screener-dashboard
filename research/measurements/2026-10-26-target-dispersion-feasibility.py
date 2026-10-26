"""Is Palley, Steffen & Zhang's moderator measurable from what this screener already fetches?

Verified in Management Science (read from PDF bytes, 2026-10-26): consensus implied return
(their PRET, which is this screener's `price_target_upside`) sorts future 12-month returns the
WRONG way unconditionally - deciles 1-8 flat near 12%, then 10.84% and 7.28%, long-top /
short-bottom -4.60% (t -3.08), 537,519 firm-months, July 1999 - December 2020. But it sorts the
RIGHT way inside the lowest decile of target-price dispersion: +2.96% (t 2.54), against -11.44%
(t -5.34) in the highest. Their mechanism is "incentive-driven staleness ... as a stock performs
poorly and some analysts are slow to update their target prices, dispersion increases, and the
consensus target price becomes too high."

That is an independent, peer-reviewed statement of the staleness this project measured in
research/2026-10-26-price-target-upside.md SS1.2 - with a moderator attached. So the question for
a later session is whether the moderator can be built here at all. Their DISP is the standard
deviation of the individual target prices over price; the fetch carries `targetHighPrice` and
`targetLowPrice` (published as `pt_high` / `pt_low`) and an analyst count, but no standard
deviation and no individual targets.

Measured here, on the published run only:

  1. Coverage of `pt_high` / `pt_low`, and whether a range-based proxy can be formed at all.
  2. The proxy's distribution, and how it relates to the analyst count (a range widens
     mechanically with the number of estimates, which an sd does not - the proxy's main defect).
  3. **The decision-relevant one:** are the stocks this screener scores BEST on implied upside
     concentrated in high dispersion - i.e. in exactly the region where Palley et al. find the
     relation inverts?

This is a feasibility measurement, not a proposal. Nothing here reads a forward return, an IC
or a backtest figure. Imports no scoring module.

Run from the repo root:
    python research/measurements/2026-10-26-target-dispersion-feasibility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")


def main() -> None:
    blob = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(blob[blob.index("{"): blob.rindex("}") + 1])
    rows = {}
    for t, sd in (data.get("stock_detail") or {}).items():
        if not isinstance(sd, dict):
            continue
        rows[t] = {
            "price": sd.get("price"),
            "pt_mean": sd.get("pt_mean"),
            "pt_high": sd.get("pt_high"),
            "pt_low": sd.get("pt_low"),
            "n_analysts": sd.get("num_analysts"),
            "sector": sd.get("sector"),
            "upside": (sd.get("raw") or {}).get("price_target_upside"),
            "rank": sd.get("rank"),
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    for c in df.columns:
        if c != "sector":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n_univ = len(df)
    have = df[["pt_high", "pt_low", "price"]].notna().all(axis=1) & (df["price"] > 0)
    scored = df["upside"].notna()

    # The only dispersion proxy the payload can support: the high-low range over price.
    df["disp_range"] = ((df["pt_high"] - df["pt_low"]) / df["price"]).where(have)
    # Palley's DISP is an sd over price. For a normal sample the range is about
    # sd * d2(n), which grows with n - so the proxy is contaminated by coverage in a way
    # their measure is not. Report the contamination rather than assuming it away.
    both = scored & df["disp_range"].notna()

    res: dict = {
        "run_timestamp": (data.get("kpis") or {}).get("run_timestamp"),
        "n_universe": int(n_univ),
        "coverage": {
            "n_with_price_target_upside": int(scored.sum()),
            "n_with_pt_high_and_pt_low": int(have.sum()),
            "n_with_both": int(both.sum()),
            "pct_with_both": round(100.0 * float(both.sum()) / n_univ, 1),
            "n_individual_targets_available": 0,
            "n_target_sd_available": 0,
        },
    }

    d = df[both]
    res["proxy_distribution"] = {
        "median": round(float(d["disp_range"].median()), 4),
        "p10": round(float(d["disp_range"].quantile(0.10)), 4),
        "p90": round(float(d["disp_range"].quantile(0.90)), 4),
        "min": round(float(d["disp_range"].min()), 4),
        "max": round(float(d["disp_range"].max()), 4),
        "n_zero_range": int((d["disp_range"] <= 0).sum()),
    }
    # the proxy's main defect, measured
    res["proxy_contamination"] = {
        "spearman_disp_vs_analyst_count": round(
            float(d["disp_range"].corr(d["n_analysts"], method="spearman")), 4),
        "median_range_by_analyst_bucket": {
            lab: (None if k.sum() < 5 else round(float(d.loc[k, "disp_range"].median()), 4))
            for lab, k in {
                "3-9": (d["n_analysts"] <= 9),
                "10-16": (d["n_analysts"].between(10, 16)),
                "17-24": (d["n_analysts"].between(17, 24)),
                "25+": (d["n_analysts"] >= 25),
            }.items()
        },
    }

    # 3. where do the best-scored names sit in the dispersion distribution?
    # Sector-relative, because that is what the engine scores.
    parts = []
    for _, g in d.groupby("sector"):
        if len(g) < 10:
            continue
        parts.append(pd.DataFrame({
            "up_pct": g["upside"].rank(pct=True) * 100,
            "disp_pct": g["disp_range"].rank(pct=True) * 100,
            "upside": g["upside"], "disp": g["disp_range"], "rank": g["rank"],
        }, index=g.index))
    w = pd.concat(parts)
    res["upside_vs_dispersion"] = {
        "n": int(len(w)),
        "spearman_within_sector": round(float(w["up_pct"].corr(w["disp_pct"], method="spearman")), 4),
        "median_dispersion_percentile_of_top_upside_quintile": round(
            float(w.loc[w["up_pct"] >= 80, "disp_pct"].median()), 1),
        "median_dispersion_percentile_of_bottom_upside_quintile": round(
            float(w.loc[w["up_pct"] <= 20, "disp_pct"].median()), 1),
        "share_of_top_upside_quintile_in_top_dispersion_quintile_pct": round(
            100.0 * float((w.loc[w["up_pct"] >= 80, "disp_pct"] >= 80).mean()), 1),
        "share_of_top_upside_quintile_in_bottom_dispersion_quintile_pct": round(
            100.0 * float((w.loc[w["up_pct"] >= 80, "disp_pct"] <= 20).mean()), 1),
    }

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"published run {res['run_timestamp']}, {n_univ} stocks\n")
    c = res["coverage"]
    print("1. CAN THE MODERATOR BE BUILT AT ALL?")
    print(f"   price_target_upside present            {c['n_with_price_target_upside']}")
    print(f"   pt_high and pt_low present             {c['n_with_pt_high_and_pt_low']}")
    print(f"   both (a range proxy is possible)       {c['n_with_both']} ({c['pct_with_both']}%)")
    print(f"   individual analyst targets in payload  {c['n_individual_targets_available']}")
    print(f"   a target standard deviation anywhere   {c['n_target_sd_available']}")
    print("   -> Palley's DISP (sd / price) is NOT available; only a high-low range is.\n")
    p = res["proxy_distribution"]
    print("2. THE RANGE PROXY, (pt_high - pt_low) / price")
    print(f"   median {p['median']:.3f}  p10 {p['p10']:.3f}  p90 {p['p90']:.3f}  "
          f"min {p['min']:.3f}  max {p['max']:.3f}  zero-range {p['n_zero_range']}")
    k = res["proxy_contamination"]
    print(f"   Spearman against the analyst count: {k['spearman_disp_vs_analyst_count']:+.4f}  "
          "(an sd would not do this; a range does)")
    print(f"   median range by analyst count: {k['median_range_by_analyst_bucket']}\n")
    u = res["upside_vs_dispersion"]
    print("3. DO THE BEST-SCORED NAMES SIT IN THE REGION WHERE THE RELATION INVERTS?")
    print(f"   within sector, Spearman(implied upside, dispersion proxy) = "
          f"{u['spearman_within_sector']:+.4f}  (n={u['n']})")
    print(f"   median dispersion percentile of the TOP upside quintile:    "
          f"{u['median_dispersion_percentile_of_top_upside_quintile']}")
    print(f"   median dispersion percentile of the BOTTOM upside quintile: "
          f"{u['median_dispersion_percentile_of_bottom_upside_quintile']}")
    print(f"   top-upside names also in the top dispersion quintile:    "
          f"{u['share_of_top_upside_quintile_in_top_dispersion_quintile_pct']}%")
    print(f"   top-upside names in the BOTTOM dispersion quintile:      "
          f"{u['share_of_top_upside_quintile_in_bottom_dispersion_quintile_pct']}%")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
