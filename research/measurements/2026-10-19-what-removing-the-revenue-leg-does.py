"""If `revenue_growth` left the score, what position would the screener then hold?

Written for research/2026-10-19-revenue-growth-vs-asset-growth.md section 11, to decide the
note's recommendation by measuring the counterfactual rather than asserting it. This is the same
test the 2026-10-13 Growth refusal applied to the category weight
(research/measurements/2026-10-13-growth-weight-sweep.py), applied here one level down, to a
metric.

The argument the note has to settle: section 3 measured that the pair's points correlate +0.045
with raw revenue growth - the screener pays 3.25 composite points for a position on revenue growth
that it does not, on net, hold. Two readings follow, and they recommend opposite things:

  (a) "the points buy nothing, so remove them." The literature in sections 7 and 9 does not
      support a POSITIVE sign on one-year revenue growth, so a leg that contributes nothing and
      is pointed the wrong way should go.
  (b) "the near-zero net exposure IS the position." Remove the revenue leg and the asset-growth
      leg stands unopposed, so a ranking that is neutral on revenue growth becomes one that is
      short it - and the evidence for a short position on ONE-YEAR revenue growth is Robeco's -2%
      a year plus two five-year-rank studies, which is suggestive and not settled.

What separates them is the size of the exposure that removing the leg would create. If it stays
near zero, (a) is a free simplification. If it swings to a material short, (b) is right and the
leg is load-bearing - which is the same shape of argument that refused the Growth weight cut.

Reported for the composite as a whole, not only for the pair, because the composite is what gets
published. Growth's remaining metrics are renormalised the way the engine renormalises any absent
metric, so the category keeps its 13 points.

Structure only. No forward return, no IC, no backtest number (CLAUDE.md rules 4 and 5). Reads the
committed dashboard_data.js and config.yaml.

Run from the repo root:
    python research/measurements/2026-10-19-what-removing-the-revenue-leg-does.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def load():
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    rows = []
    for ticker, s in data["stock_detail"].items():
        raw, pct = s.get("raw") or {}, s.get("pct") or {}
        rows.append({"Ticker": ticker, "sector": s.get("sector"),
                     "composite": s.get("composite"), "rank": s.get("rank"),
                     "rg_raw": raw.get("revenue_growth"), "ag_raw": raw.get("asset_growth"),
                     **{f"p_{m}": pct.get(m) for m in
                        ("revenue_growth", "asset_growth", "revenue_cagr_3yr",
                         "sustainable_growth", "forward_eps_growth")},
                     **{f"c_{c}": (s.get("cat_scores") or {}).get(c) for c in CATS}})
    df = pd.DataFrame(rows).set_index("Ticker")
    for c in df.columns:
        if c != "sector":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return data, df


def growth_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Rebuild the Growth category score from its metric percentiles, renormalising over the
    metrics each stock actually has - which is what the engine does for an absent metric."""
    cols = {m: f"p_{m}" for m in weights}
    num = sum(df[cols[m]].fillna(0) * w for m, w in weights.items())
    den = sum(df[cols[m]].notna() * w for m, w in weights.items())
    return (num / den.replace(0, pd.NA)).astype(float)


def composite(df: pd.DataFrame, gscore: pd.Series, fw: dict) -> pd.Series:
    """The same approximation of the composite the 10-13 sweep used: weighted mean over the
    categories a stock has, renormalised. main() reports how well it reproduces the published
    ranking before anything is read off it."""
    parts = {c: (gscore if c == "growth" else df[f"c_{c}"]) for c in CATS}
    w = {c: float(fw[c]) for c in CATS}
    have = pd.DataFrame({c: parts[c].notna() for c in CATS})
    num = sum(parts[c].fillna(0) * w[c] for c in CATS)
    den = sum(have[c] * w[c] for c in CATS)
    return num / den.replace(0, pd.NA)


def main() -> None:
    data, df = load()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    fw, mwg = cfg["factor_weights"], cfg["metric_weights"]["growth"]
    base_w = {m: w for m, w in mwg.items() if float(w) > 0}
    cut_w = {m: w for m, w in base_w.items() if m != "revenue_growth"}

    res = {"run_timestamp": data["kpis"].get("run_timestamp"),
           "growth_weights_now": base_w, "growth_weights_without_revenue_growth": cut_w}
    print(f"run: {res['run_timestamp']}  |  stocks: {len(df)}")
    print(f"Growth now                      : {base_w}")
    print(f"Growth without revenue_growth   : {cut_w} (renormalised by the engine)")

    g_now, g_cut = growth_score(df, base_w), growth_score(df, cut_w)
    c_now, c_cut = composite(df, g_now, fw), composite(df, g_cut, fw)

    # Does the approximation reproduce the published ranking well enough to read off?
    ok = df["composite"].notna() & c_now.notna()
    rho = float(c_now[ok].corr(df["composite"][ok], method="spearman"))
    top_pub = set(df.sort_values("rank").index[:25])
    top_app = set(c_now.sort_values(ascending=False).index[:25])
    res["approximation"] = {"spearman_vs_published": round(rho, 4),
                            "top25_in_common": len(top_pub & top_app)}
    print(f"\napproximation against the published composite: Spearman {rho:.4f}; "
          f"top 25 in common {len(top_pub & top_app)} of 25")
    if rho < 0.99:
        print("  NOTE: below 0.99 - read the swings below as indicative, not exact")

    # --- the counterfactual ---------------------------------------------------------------
    pair = df.dropna(subset=["rg_raw", "ag_raw"]).copy()
    pair["spread_rank"] = (pair.groupby("sector")["rg_raw"].rank(pct=True)
                           - pair.groupby("sector")["ag_raw"].rank(pct=True))
    out = {}
    for label, comp, gs in (("now", c_now, g_now), ("without revenue_growth", c_cut, g_cut)):
        c = comp.reindex(pair.index)
        out[label] = {
            "composite_vs_revenue_growth": round(
                float(c.corr(pair["rg_raw"], method="spearman")), 4),
            "composite_vs_asset_growth": round(
                float(c.corr(pair["ag_raw"], method="spearman")), 4),
            "composite_vs_spread": round(
                float(c.corr(pair["spread_rank"], method="spearman")), 4),
            "growth_score_vs_revenue_growth": round(
                float(gs.reindex(pair.index).corr(pair["rg_raw"], method="spearman")), 4),
        }
    res["exposure"] = out
    print("\n--- the composite's exposure, now and with the revenue leg removed ---")
    hdr = f"{'':34s} {'now':>10s} {'without':>10s} {'change':>9s}"
    print(hdr); print("-" * len(hdr))
    for k, name in (("composite_vs_revenue_growth", "vs raw revenue growth"),
                    ("composite_vs_asset_growth", "vs raw asset growth"),
                    ("composite_vs_spread", "vs the within-sector spread"),
                    ("growth_score_vs_revenue_growth", "Growth score vs revenue growth")):
        a, b = out["now"][k], out["without revenue_growth"][k]
        print(f"{name:34s} {a:>+10.4f} {b:>+10.4f} {b - a:>+9.4f}")

    # How much of the ranking would actually move?
    r_now = c_now.rank(ascending=False)
    r_cut = c_cut.rank(ascending=False)
    moved = (r_cut - r_now).dropna()
    res["ranking_effect"] = {
        "spearman_now_vs_cut": round(float(c_now.corr(c_cut, method="spearman")), 4),
        "median_abs_rank_move": float(moved.abs().median()),
        "p90_abs_rank_move": round(float(moved.abs().quantile(0.90)), 1),
        "max_abs_rank_move": float(moved.abs().max()),
        "top25_retained": len(set(c_now.sort_values(ascending=False).index[:25])
                              & set(c_cut.sort_values(ascending=False).index[:25])),
    }
    re_ = res["ranking_effect"]
    print(f"\n--- how much the ranking moves ---")
    print(f"Spearman between the two rankings : {re_['spearman_now_vs_cut']:.4f}")
    print(f"median |rank move| {re_['median_abs_rank_move']:.0f}, "
          f"90th percentile {re_['p90_abs_rank_move']:.0f}, "
          f"largest {re_['max_abs_rank_move']:.0f}")
    print(f"top 25 retained                   : {re_['top25_retained']} of 25")

    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
