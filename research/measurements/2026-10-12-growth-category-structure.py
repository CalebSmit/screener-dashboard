"""How the Growth category sits among the other seven in the published run.

For research/2026-10-12-growth-category.md. Structure only: correlations between category
scores and between metrics, and how far the ranking moves if Growth's weight is changed. No
return is read anywhere (CLAUDE.md rule 4), and nothing here writes a file.

Run from the repo root:  python research/measurements/2026-10-12-growth-category-structure.py
It reads the committed dashboard_data.js, so the figures belong to whichever run that is; the
note quotes the run of 2026-10-12 02:00.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def load() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    table = pd.DataFrame(data["table_data"]).set_index("Ticker")
    scores = table[[c + "_score" for c in CATS]].apply(pd.to_numeric, errors="coerce")
    scores.columns = CATS
    raw = pd.DataFrame({t: d.get("raw") or {} for t, d in data["stock_detail"].items()}).T
    return data, scores.assign(Composite=pd.to_numeric(table["Composite"]), Rank=pd.to_numeric(table["Rank"])), \
        raw.apply(pd.to_numeric, errors="coerce")


def blend(scores: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted mean of the category scores a stock has, renormalised over those present. An
    approximation of the engine's composite (no coverage discount, no bank or regime handling);
    `main` reports how closely it reproduces the published ranking before using it."""
    w = pd.Series(weights, dtype=float)
    have = scores[CATS].notna()
    return (scores[CATS].fillna(0) * w).sum(axis=1) / (have * w).sum(axis=1)


def main() -> None:
    data, scores, raw = load()
    weights = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    print("run:", data["kpis"].get("run_timestamp"), "| stocks:", len(scores))

    print("\nSpearman correlation of Growth's score with each other category's:")
    print(scores[CATS].corr(method="spearman")["growth"].drop("growth").round(2).to_string())

    growth = ["forward_eps_growth", "revenue_growth", "revenue_cagr_3yr", "sustainable_growth"]
    others = ["asset_growth", "roic", "gross_profit_assets", "earnings_yield", "fcf_yield", "fy1_revision_3m",
              "return_12_1"]
    print("\nSpearman correlation of each Growth metric (columns) with other raw metrics (rows):")
    print(raw[growth + others].corr(method="spearman").loc[growth + others, growth].round(2).to_string())

    base = blend(scores, weights)
    rho = base.corr(scores["Composite"], method="spearman")
    top = lambda s, n=25: set(s.sort_values(ascending=False).index[:n])
    published = set(scores.sort_values("Rank").index[:25])
    print(f"\nThe approximation against the published composite: Spearman {rho:.4f}; "
          f"top 25 in common {len(top(base) & published)} of 25")

    def moved(label: str, w: dict) -> None:
        s = blend(scores, w)
        print(f"  {label:58} rank correlation with today's {s.corr(base, method='spearman'):.3f}; "
              f"top 25 kept {len(top(s) & top(base))}; top 50 kept {len(top(s, 50) & top(base, 50))}")

    print("\nHow far the (approximated) ranking moves:")
    moved("Growth 13 -> 10, 3 points to Quality", {**weights, "growth": 10, "quality": weights["quality"] + 3})
    moved("Growth 13 -> 10, 3 points spread pro rata", {**weights, "growth": 10})
    moved("Growth 13 -> 0, spread pro rata", {**weights, "growth": 0})
    moved("Growth 13 -> 0, 13 points to Quality", {**weights, "growth": 0, "quality": weights["quality"] + 13})

    proposed = {**weights, "growth": 10, "quality": weights["quality"] + 3}
    tilt = pd.DataFrame({"today": [base.corr(scores[c], method="spearman") for c in CATS],
                         "Growth 10 / Quality 25": [blend(scores, proposed).corr(scores[c], method="spearman") for c in CATS]},
                        index=CATS)
    print()
    print("Spearman correlation of the composite with each category's score (what the ranking leans on):")
    print(tilt.round(3).to_string())
    was, now = base.sort_values(ascending=False).index[:25], blend(scores, proposed).sort_values(ascending=False).index[:25]
    print("  top 25 leaves:", sorted(set(was) - set(now)), "| enters:", sorted(set(now) - set(was)))

    q = scores["growth"].rank(pct=True)
    hi, lo = scores[q >= 0.8], scores[q <= 0.2]
    print("\nAverage score in each other category, top fifth on Growth vs bottom fifth:")
    print(pd.DataFrame({"top fifth": hi[CATS].mean(), "bottom fifth": lo[CATS].mean()}).round(1).to_string())
    in25 = scores.sort_values("Rank").head(25)
    print(f"\nPublished top 25: average Growth score {in25['growth'].mean():.1f} "
          f"(universe {scores['growth'].mean():.1f}); average Valuation {in25['valuation'].mean():.1f}, "
          f"Quality {in25['quality'].mean():.1f}")


if __name__ == "__main__":
    main()
