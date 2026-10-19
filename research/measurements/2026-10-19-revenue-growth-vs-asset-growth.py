"""Are `revenue_growth` and `asset_growth` one fact scored twice, in opposite directions?

Written for the Monday note research/2026-10-19-revenue-growth-vs-asset-growth.md, which answers
the first of the open Growth composition questions left by the 2026-10-13 refusal
(research/2026-10-12-growth-category.md section 11): "the sign of `revenue_growth` against
`asset_growth` (two categories, one fact, opposite directions)".

The screener's position today:

  * `revenue_growth` is 25% of Growth's 13 points  -> 3.25 composite points, HIGHER is better.
  * `asset_growth`   is 100% of Investment's 5 pts -> 5.00 composite points, LOWER is better.

The 10-12 note measured one number - Spearman +0.51 between the two raw values - and concluded
"for most companies two categories argue about the same fact". A reviewer needs more than that,
because +0.51 is equally consistent with three quite different situations, which imply different
actions:

  (a) one fact, two names. Then the screener holds a net SHORT position on company growth worth
      5.00 - 3.25 = 1.75 points, and nobody decided that; it is the residue of two independently
      chosen weights.
  (b) two correlated but separable facts - growth in demand, and growth in the balance sheet -
      in which case opposite signs are a deliberate and defensible construction ("growth that
      does not consume capital"), and the thing to check is whether the screener actually
      expresses it rather than cancelling both out.
  (c) mostly one fact plus a thin, noisy residual, in which case the construction in (b) is
      being claimed but not delivered.

What separates them is measurable without any return: how much of each metric survives the
other (within sector, which is the only frame the engine ever scores in), what the NET points
exposure to each raw fact is once both legs are counted, and how large and how populated the
off-diagonal cells are - companies that grew revenue without growing assets, and the reverse.

Structure only. No forward return, no IC, no backtest number is read anywhere (CLAUDE.md rules 4
and 5). Nothing here writes a file except the JSON summary beside it. Reads the committed
dashboard_data.js, so the figures belong to whichever run that is - the script prints the run
timestamp so the note can name it.

Run from the repo root:
    python research/measurements/2026-10-19-revenue-growth-vs-asset-growth.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")

# The two metrics under examination, and the third growth metric that shares their basis.
RG, AG = "revenue_growth", "asset_growth"
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def load() -> tuple[dict, pd.DataFrame]:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    rows = []
    for ticker, s in data["stock_detail"].items():
        raw, pct = s.get("raw") or {}, s.get("pct") or {}
        rows.append({
            "Ticker": ticker,
            "sector": s.get("sector"),
            "composite": s.get("composite"),
            "rank": s.get("rank"),
            "rg_raw": raw.get(RG), "ag_raw": raw.get(AG),
            "rg_pct": pct.get(RG), "ag_pct": pct.get(AG),
            "cagr_raw": raw.get("revenue_cagr_3yr"),
            "growth_score": (s.get("cat_scores") or {}).get("growth"),
            "investment_score": (s.get("cat_scores") or {}).get("investment"),
        })
    df = pd.DataFrame(rows).set_index("Ticker")
    for c in df.columns:
        if c != "sector":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return data, df


def within_sector_spearman(df: pd.DataFrame, a: str, b: str, min_n: int = 8) -> tuple[float, int, int]:
    """Spearman of a against b, pooled over sectors after ranking inside each sector.

    The engine percentiles every metric inside the stock's GICS sector, so a universe-wide
    correlation is not the number that governs whether the two scores cancel. Ranking within
    sector first and then pooling is the sector-demeaned equivalent.
    """
    parts, sectors = [], 0
    for sector, g in df.groupby("sector"):
        pair = g[[a, b]].dropna()
        if len(pair) < min_n:
            continue
        sectors += 1
        parts.append(pair.rank(pct=True))
    if not parts:
        return float("nan"), 0, 0
    pooled = pd.concat(parts)
    return float(pooled[a].corr(pooled[b], method="spearman")), len(pooled), sectors


def main() -> None:
    data, df = load()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    fw, mw = cfg["factor_weights"], cfg["metric_weights"]

    pts_rg = fw["growth"] * mw["growth"][RG] / 100.0
    pts_ag = fw["investment"] * mw["investment"][AG] / 100.0

    res: dict = {
        "run_timestamp": data["kpis"].get("run_timestamp"),
        "stocks_in_payload": int(len(df)),
        "points": {
            "revenue_growth": round(pts_rg, 4),
            "asset_growth": round(pts_ag, 4),
            "net_if_identical": round(pts_rg - pts_ag, 4),
        },
    }
    print(f"run: {res['run_timestamp']}  |  stocks in payload: {len(df)}")
    print(f"\ncomposite points: {RG} {pts_rg:.2f} (higher better)  "
          f"{AG} {pts_ag:.2f} (lower better)  "
          f"-> net on 'the company grew', if the two were the same fact: {pts_rg - pts_ag:+.2f}")

    # ---- 1. coverage ----------------------------------------------------------------------
    both = df[["rg_raw", "ag_raw"]].dropna()
    res["coverage"] = {
        "revenue_growth": int(df["rg_raw"].notna().sum()),
        "asset_growth": int(df["ag_raw"].notna().sum()),
        "both": int(len(both)),
    }
    print(f"\ncoverage: {RG} {res['coverage']['revenue_growth']}, "
          f"{AG} {res['coverage']['asset_growth']}, both {res['coverage']['both']}")

    # ---- 2. the overlap, three ways -------------------------------------------------------
    uni_raw = float(both["rg_raw"].corr(both["ag_raw"], method="spearman"))
    pair_pct = df[["rg_pct", "ag_pct"]].dropna()
    uni_pct = float(pair_pct["rg_pct"].corr(pair_pct["ag_pct"], method="spearman"))
    ws_raw, n_ws, n_sec = within_sector_spearman(df, "rg_raw", "ag_raw")
    res["overlap"] = {
        "universe_spearman_raw": round(uni_raw, 4),
        "universe_spearman_scored_percentiles": round(uni_pct, 4),
        "within_sector_spearman_raw": round(ws_raw, 4),
        "within_sector_n": n_ws,
        "sectors_used": n_sec,
        "shared_variance_within_sector_pct": round(100 * ws_raw ** 2, 1),
    }
    print("\n--- 2. how much the two measures overlap ---")
    print(f"universe Spearman, raw values                : {uni_raw:+.4f}   "
          f"(the 10-12 note's +0.51, re-measured)")
    print(f"universe Spearman, the SCORED percentiles    : {uni_pct:+.4f}   "
          f"(negative because Investment inverts its metric)")
    print(f"WITHIN SECTOR Spearman, raw values           : {ws_raw:+.4f}   "
          f"({n_ws} stocks over {n_sec} sectors) <- the frame the engine scores in")
    print(f"  -> shared variance within sector           : {100 * ws_raw ** 2:.1f}% "
          f"(so {100 - 100 * ws_raw ** 2:.1f}% of each is its own)")

    # ---- 3. net points exposure -----------------------------------------------------------
    # The question (a)-vs-(b): once BOTH legs are counted, does the screener end up rewarding
    # or penalising a company for having grown? Computed on the engine's own percentiles.
    p = df.dropna(subset=["rg_pct", "ag_pct", "rg_raw", "ag_raw"]).copy()
    p["pts_rg"] = p["rg_pct"] * pts_rg / 100.0
    p["pts_ag"] = p["ag_pct"] * pts_ag / 100.0
    p["pts_pair"] = p["pts_rg"] + p["pts_ag"]
    res["net_exposure"] = {
        "n": int(len(p)),
        "pair_points_range": [round(float(p["pts_pair"].min()), 2),
                              round(float(p["pts_pair"].max()), 2)],
        "pair_points_sd": round(float(p["pts_pair"].std()), 3),
        "spearman_pair_points_vs_revenue_growth": round(
            float(p["pts_pair"].corr(p["rg_raw"], method="spearman")), 4),
        "spearman_pair_points_vs_asset_growth": round(
            float(p["pts_pair"].corr(p["ag_raw"], method="spearman")), 4),
        "spearman_pts_rg_vs_pts_ag": round(
            float(p["pts_rg"].corr(p["pts_ag"], method="spearman")), 4),
    }
    ne = res["net_exposure"]
    print("\n--- 3. what the PAIR, both legs counted, is a position on ---")
    print(f"the pair contributes {ne['pair_points_range'][0]:.2f} to "
          f"{ne['pair_points_range'][1]:.2f} composite points "
          f"(sd {ne['pair_points_sd']:.2f}) across {ne['n']} stocks")
    print(f"pair points vs raw revenue growth : {ne['spearman_pair_points_vs_revenue_growth']:+.4f}")
    print(f"pair points vs raw asset growth   : {ne['spearman_pair_points_vs_asset_growth']:+.4f}")
    print(f"the two legs' points against each other: {ne['spearman_pts_rg_vs_pts_ag']:+.4f}")

    # Counterfactual: if the screener held ONLY the Investment leg, how different is its
    # position on asset growth? This prices how much of the Investment bet revenue_growth eats.
    only_ag = float(p["pts_ag"].corr(p["ag_raw"], method="spearman"))
    only_rg = float(p["pts_rg"].corr(p["rg_raw"], method="spearman"))
    res["net_exposure"]["spearman_investment_leg_alone_vs_asset_growth"] = round(only_ag, 4)
    res["net_exposure"]["spearman_growth_leg_alone_vs_revenue_growth"] = round(only_rg, 4)
    print(f"  for comparison, the Investment leg alone vs asset growth : {only_ag:+.4f}")
    print(f"                  the revenue leg alone vs revenue growth  : {only_rg:+.4f}")

    # ---- 4. the off-diagonal cells --------------------------------------------------------
    # Case (b) lives or dies here. If "grew revenue without growing assets" is a large, populated
    # cell, the two metrics are separable and opposite signs express something real. If the
    # population sits on the diagonal, they are one fact.
    q = p.copy()
    for col, name in (("rg_raw", "rg_q"), ("ag_raw", "ag_q")):
        q[name] = q.groupby("sector")[col].transform(
            lambda s: pd.qcut(s, 2, labels=["low", "high"], duplicates="drop")
            if s.notna().sum() >= 8 else pd.Series(index=s.index, dtype="object"))
    cells = q.dropna(subset=["rg_q", "ag_q"])
    tab = pd.crosstab(cells["rg_q"], cells["ag_q"])
    res["cells"] = {
        "n_classified": int(len(cells)),
        "table": {str(i): {str(c): int(tab.loc[i, c]) for c in tab.columns} for i in tab.index},
    }
    print("\n--- 4. the four cells, split at each sector's median ---")
    print("(rows: revenue growth, columns: asset growth)")
    print(tab.to_string())
    offdiag = int(tab.loc["high", "low"]) + int(tab.loc["low", "high"]) \
        if {"high", "low"} <= set(tab.index) and {"high", "low"} <= set(tab.columns) else 0
    res["cells"]["off_diagonal"] = offdiag
    res["cells"]["off_diagonal_pct"] = round(100 * offdiag / max(len(cells), 1), 1)
    print(f"\noff-diagonal (grew one without the other): {offdiag} of {len(cells)} "
          f"= {res['cells']['off_diagonal_pct']:.1f}%")
    if {"high", "low"} <= set(tab.index) and {"high", "low"} <= set(tab.columns):
        cap_light = cells[(cells["rg_q"] == "high") & (cells["ag_q"] == "low")]
        cap_heavy = cells[(cells["rg_q"] == "low") & (cells["ag_q"] == "high")]
        for label, cell in (("revenue up, assets flat (capital-light growth)", cap_light),
                            ("assets up, revenue flat (capital without demand)", cap_heavy)):
            res["cells"][label] = {
                "n": int(len(cell)),
                "median_composite": round(float(cell["composite"].median()), 2),
                "median_pair_points": round(float(cell["pts_pair"].median()), 2),
                "median_growth_score": round(float(cell["growth_score"].median()), 1),
                "median_investment_score": round(float(cell["investment_score"].median()), 1),
            }
            c = res["cells"][label]
            print(f"  {label}: n={c['n']}, median composite {c['median_composite']}, "
                  f"median pair points {c['median_pair_points']}, "
                  f"Growth {c['median_growth_score']} / Investment {c['median_investment_score']}")

    # ---- 5. does either leg survive the other? --------------------------------------------
    # Sector-demeaned OLS, each percentile on the other. The residual share is what the second
    # metric adds that the first does not already carry.
    print("\n--- 5. how much of each scored percentile the other explains (within sector) ---")
    res["spanning"] = {}
    for target, other in (("rg_pct", "ag_pct"), ("ag_pct", "rg_pct")):
        parts = []
        for _, g in df.groupby("sector"):
            pair = g[[target, other]].dropna()
            if len(pair) < 8:
                continue
            parts.append(pair - pair.mean())
        if not parts:
            continue
        pooled = pd.concat(parts)
        x, y = pooled[other].to_numpy(), pooled[target].to_numpy()
        beta = float(np.dot(x, y) / np.dot(x, x))
        r2 = float(1 - np.var(y - beta * x) / np.var(y))
        res["spanning"][f"{target}_on_{other}"] = {
            "beta": round(beta, 4), "r2": round(r2, 4), "n": int(len(pooled)),
            "residual_share": round(1 - r2, 4),
        }
        print(f"  {target} on {other}: beta {beta:+.3f}, R2 {r2:.3f} "
              f"-> {100 * (1 - r2):.1f}% of {target} is not explained by {other}")

    # ---- 6. and the third metric that shares the basis ------------------------------------
    ws_cagr, _, _ = within_sector_spearman(df, "cagr_raw", "ag_raw")
    ws_rg_cagr, _, _ = within_sector_spearman(df, "rg_raw", "cagr_raw")
    res["three_year_cagr"] = {
        "within_sector_spearman_vs_asset_growth": round(ws_cagr, 4),
        "within_sector_spearman_vs_revenue_growth": round(ws_rg_cagr, 4),
        "points": round(fw["growth"] * mw["growth"]["revenue_cagr_3yr"] / 100.0, 4),
    }
    print("\n--- 6. revenue_cagr_3yr, the category's other top-line leg ---")
    print(f"  within sector vs asset growth   : {ws_cagr:+.4f}")
    print(f"  within sector vs revenue growth : {ws_rg_cagr:+.4f}")
    print(f"  it carries {res['three_year_cagr']['points']:.2f} composite points, "
          f"so the top-line total is "
          f"{pts_rg + res['three_year_cagr']['points']:.2f} against asset growth's {pts_ag:.2f}")

    # ---- 7. the published top 25 ----------------------------------------------------------
    top = df.sort_values("rank").head(25)
    res["top25"] = {
        "median_revenue_growth": round(float(top["rg_raw"].median()), 4),
        "median_asset_growth": round(float(top["ag_raw"].median()), 4),
        "universe_median_revenue_growth": round(float(df["rg_raw"].median()), 4),
        "universe_median_asset_growth": round(float(df["ag_raw"].median()), 4),
        "median_revenue_growth_pct": round(float(top["rg_pct"].median()), 1),
        "median_asset_growth_pct": round(float(top["ag_pct"].median()), 1),
    }
    t = res["top25"]
    print("\n--- 7. the published top 25 on the two measures ---")
    print(f"  revenue growth: top25 median {t['median_revenue_growth']:+.1%} vs "
          f"universe {t['universe_median_revenue_growth']:+.1%} "
          f"(scored percentile {t['median_revenue_growth_pct']})")
    print(f"  asset growth  : top25 median {t['median_asset_growth']:+.1%} vs "
          f"universe {t['universe_median_asset_growth']:+.1%} "
          f"(scored percentile {t['median_asset_growth_pct']})")

    # ---- 8. is the pair a position on the LEVEL of either, or on the SPREAD? --------------
    # Section 3 found the pair's exposure to revenue growth is ~0 while its exposure to asset
    # growth survives. That is consistent with two different descriptions, and this separates
    # them: if the pair is really "growth that does not consume capital", its points should
    # track the within-sector SPREAD between the two - how much faster revenue grew than the
    # balance sheet - better than they track either level.
    sp = p.copy()
    sp["rg_rank"] = sp.groupby("sector")["rg_raw"].rank(pct=True)
    sp["ag_rank"] = sp.groupby("sector")["ag_raw"].rank(pct=True)
    sp["spread_rank"] = sp["rg_rank"] - sp["ag_rank"]
    # The economic version of the same thing: revenue growth minus asset growth, in points.
    sp["spread_raw"] = sp["rg_raw"] - sp["ag_raw"]
    res["spread"] = {
        "spearman_pair_points_vs_rank_spread": round(
            float(sp["pts_pair"].corr(sp["spread_rank"], method="spearman")), 4),
        "spearman_pair_points_vs_raw_spread": round(
            float(sp["pts_pair"].corr(sp["spread_raw"], method="spearman")), 4),
        "median_raw_spread": round(float(sp["spread_raw"].median()), 4),
        "share_revenue_outgrew_assets": round(float((sp["spread_raw"] > 0).mean()), 4),
    }
    s8 = res["spread"]
    print("\n--- 8. level or spread? ---")
    print(f"pair points vs the within-sector rank spread (rev rank - asset rank): "
          f"{s8['spearman_pair_points_vs_rank_spread']:+.4f}")
    print(f"pair points vs raw revenue growth minus raw asset growth          : "
          f"{s8['spearman_pair_points_vs_raw_spread']:+.4f}")
    print(f"  compare section 3: vs revenue growth alone "
          f"{ne['spearman_pair_points_vs_revenue_growth']:+.4f}, "
          f"vs asset growth alone {ne['spearman_pair_points_vs_asset_growth']:+.4f}")
    print(f"revenue outgrew assets for {100 * s8['share_revenue_outgrew_assets']:.1f}% of stocks; "
          f"median spread {s8['median_raw_spread']:+.1%}")

    # ---- 9. the same arithmetic counting BOTH of Growth's top-line legs -------------------
    # revenue_cagr_3yr is also a top-line growth measure and also scored higher-is-better. The
    # net position on "the company grew" is not 3.25 against 5.00 but 5.20 against 5.00, and
    # section 3's near-zero exposure has to be re-read with the third leg in.
    pts_cagr = fw["growth"] * mw["growth"]["revenue_cagr_3yr"] / 100.0
    tri = df.dropna(subset=["rg_pct", "ag_pct", "rg_raw", "ag_raw", "cagr_raw"]).copy()
    cagr_pct = {t: (data["stock_detail"][t].get("pct") or {}).get("revenue_cagr_3yr")
                for t in tri.index}
    tri["cagr_pct"] = pd.to_numeric(pd.Series(cagr_pct), errors="coerce")
    tri = tri.dropna(subset=["cagr_pct"])
    tri["pts_tri"] = (tri["rg_pct"] * pts_rg + tri["cagr_pct"] * pts_cagr
                      + tri["ag_pct"] * pts_ag) / 100.0
    res["three_legs"] = {
        "n": int(len(tri)),
        "points_top_line": round(pts_rg + pts_cagr, 4),
        "points_asset_growth": round(pts_ag, 4),
        "spearman_vs_revenue_growth": round(
            float(tri["pts_tri"].corr(tri["rg_raw"], method="spearman")), 4),
        "spearman_vs_three_year_cagr": round(
            float(tri["pts_tri"].corr(tri["cagr_raw"], method="spearman")), 4),
        "spearman_vs_asset_growth": round(
            float(tri["pts_tri"].corr(tri["ag_raw"], method="spearman")), 4),
    }
    t3 = res["three_legs"]
    print("\n--- 9. all three top-line/balance-sheet legs together ---")
    print(f"{t3['points_top_line']:.2f} points of top-line growth "
          f"(revenue_growth {pts_rg:.2f} + revenue_cagr_3yr {pts_cagr:.2f}) "
          f"against {t3['points_asset_growth']:.2f} on low asset growth, n={t3['n']}")
    print(f"  the three legs' points vs raw revenue growth  : "
          f"{t3['spearman_vs_revenue_growth']:+.4f}")
    print(f"  the three legs' points vs raw 3-year CAGR     : "
          f"{t3['spearman_vs_three_year_cagr']:+.4f}")
    print(f"  the three legs' points vs raw asset growth    : "
          f"{t3['spearman_vs_asset_growth']:+.4f}")

    # ---- 10. does the WHOLE composite hold the position the pair holds? -------------------
    # Sections 3 and 8 are about 8.25 of the composite's 100 points. The coherence question
    # (CLAUDE.md, "Build a coherent screener") is whether the finished ranking inherits that
    # position or whether the other 91.75 points wash it out - and whether the other categories
    # are already leaning the same way, in which case the pair is not the only place the
    # screener bets on asset turnover.
    cw = sp.dropna(subset=["composite"]).copy()
    res["composite_exposure"] = {
        "vs_revenue_growth": round(
            float(cw["composite"].corr(cw["rg_raw"], method="spearman")), 4),
        "vs_asset_growth": round(
            float(cw["composite"].corr(cw["ag_raw"], method="spearman")), 4),
        "vs_rank_spread": round(
            float(cw["composite"].corr(cw["spread_rank"], method="spearman")), 4),
        "vs_raw_spread": round(
            float(cw["composite"].corr(cw["spread_raw"], method="spearman")), 4),
        "n": int(len(cw)),
    }
    ce = res["composite_exposure"]
    print("\n--- 10. the finished composite's own exposure (all 100 points) ---")
    print(f"composite vs raw revenue growth : {ce['vs_revenue_growth']:+.4f}")
    print(f"composite vs raw asset growth   : {ce['vs_asset_growth']:+.4f}")
    print(f"composite vs the rank spread    : {ce['vs_rank_spread']:+.4f}  "
          f"(the pair alone: {s8['spearman_pair_points_vs_rank_spread']:+.4f})")

    # Where else in the composite does that exposure come from? Each category score against the
    # spread, so the note can say whether the pair is the screener's only bet on asset turnover.
    print("\ncategory scores against the within-sector rank spread:")
    res["composite_exposure"]["by_category"] = {}
    cat_cols = {c: c + "_score" for c in CATS}
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    table = pd.DataFrame(json.loads(
        text[text.index("{"): text.rindex("}") + 1])["table_data"]).set_index("Ticker")
    for c, col in cat_cols.items():
        if col not in table.columns:
            continue
        s = pd.to_numeric(table[col], errors="coerce").reindex(cw.index)
        rho = float(s.corr(cw["spread_rank"], method="spearman"))
        res["composite_exposure"]["by_category"][c] = round(rho, 4)
        print(f"  {c:11s} (weight {fw[c]:>2}): {rho:+.4f}")

    OUT.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
