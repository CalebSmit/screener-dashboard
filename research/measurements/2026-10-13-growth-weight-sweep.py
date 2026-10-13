"""At what Growth weight is the ranking neutral to growth, and how flat is that neutrality?

Written for the review of the proposal in section 7 of research/2026-10-12-growth-category.md
(Growth 13 -> 10, Quality 22 -> 25). That note measured one point - the composite's Spearman
with the Growth score goes from -0.01 at weight 13 to -0.06 at weight 10 - and argued the cut
from the evidence that growth as the category measures it earns nothing of its own. A reviewer
needs the curve, not the point, because the two readings of the same evidence differ in what
they predict:

  * "weight belongs where evidence is" -> cutting Growth is an improvement whatever the curve;
  * "a trait that earns nothing should be neither rewarded nor penalised" -> the right weight is
    whichever one leaves the ranking neutral, and moving away from it is a disimprovement.

The test that separates them: if the composite's growth exposure is flat over a wide band of
weights, neutrality is not a real constraint and the first reading wins on evidence. If
exposure moves steeply through zero near the current weight, then 13 points is buying
neutrality and the second reading is a live argument.

Structure only. No return is read anywhere (CLAUDE.md rule 4), no backtest number (rule 5), and
nothing here writes a file. Reads the committed dashboard_data.js, so the figures belong to
whichever run that is.

Run from the repo root:  python research/measurements/2026-10-13-growth-weight-sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


def load() -> tuple[dict, pd.DataFrame]:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    table = pd.DataFrame(data["table_data"]).set_index("Ticker")
    scores = table[[c + "_score" for c in CATS]].apply(pd.to_numeric, errors="coerce")
    scores.columns = CATS
    return data, scores.assign(Composite=pd.to_numeric(table["Composite"]),
                               Rank=pd.to_numeric(table["Rank"]))


def blend(scores: pd.DataFrame, weights: dict) -> pd.Series:
    """The same approximation of the composite the 10-12 script used: weighted mean over the
    categories a stock has, renormalised. No coverage discount, no bank or regime handling;
    main() reports how closely it reproduces the published ranking before anything is read off
    it."""
    w = pd.Series(weights, dtype=float)
    have = scores[CATS].notna()
    return (scores[CATS].fillna(0) * w).sum(axis=1) / (have * w).sum(axis=1)


def main() -> None:
    data, scores = load()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["factor_weights"]
    base = blend(scores, cfg)
    rho = base.corr(scores["Composite"], method="spearman")
    top = lambda s, n=25: set(s.sort_values(ascending=False).index[:n])
    published = set(scores.sort_values("Rank").index[:25])
    print("run:", data["kpis"].get("run_timestamp"), "| stocks:", len(scores))
    print(f"the approximation against the published composite: Spearman {rho:.4f}; "
          f"top 25 in common {len(top(base) & published)} of 25")
    print(f"config weights: {cfg}")

    # --- the sweep -------------------------------------------------------------------------
    # Growth from 0 to 26 in steps of 1. The freed (or borrowed) points go to Quality, which is
    # what the proposal does, and pro rata across the other seven, which is the neutral
    # alternative. Both are reported because the destination changes the other exposures even
    # where it barely changes growth exposure.
    print("\nComposite's Spearman with the Growth score, as Growth's weight moves")
    print("(the other 87 points held as configured; 'to Quality' moves the difference to "
          "Quality, 'pro rata' spreads it over the other seven)")
    print(f"{'growth wt':>9} {'to Quality':>11} {'pro rata':>9}   "
          f"{'top25 kept (to Quality)':>23}")
    rows = []
    for g in range(0, 27):
        to_q = {**cfg, "growth": g, "quality": cfg["quality"] + (cfg["growth"] - g)}
        others = [c for c in CATS if c != "growth"]
        scale = (sum(cfg[c] for c in others) + (cfg["growth"] - g)) / sum(cfg[c] for c in others)
        pro = {**{c: cfg[c] * scale for c in others}, "growth": g}
        s_q, s_p = blend(scores, to_q), blend(scores, pro)
        e_q = s_q.corr(scores["growth"], method="spearman")
        e_p = s_p.corr(scores["growth"], method="spearman")
        kept = len(top(s_q) & top(base))
        rows.append((g, e_q, e_p, kept))
        mark = "  <- configured" if g == cfg["growth"] else ("  <- proposed" if g == 10 else "")
        print(f"{g:>9} {e_q:>+11.3f} {e_p:>+9.3f}   {kept:>23}{mark}")

    frame = pd.DataFrame(rows, columns=["growth", "to_quality", "pro_rata", "top25_kept"])

    def crossing(col: str) -> float | None:
        """Linear interpolation of the weight at which growth exposure is exactly zero."""
        s = frame[col]
        for i in range(len(s) - 1):
            a, b = s.iloc[i], s.iloc[i + 1]
            if (a <= 0 <= b) or (b <= 0 <= a):
                if a == b:
                    return float(frame["growth"].iloc[i])
                t = -a / (b - a)
                return float(frame["growth"].iloc[i] + t)
        return None

    print(f"\ngrowth exposure is zero at weight {crossing('to_quality'):.1f} "
          f"(freed points to Quality) and {crossing('pro_rata'):.1f} (pro rata)")
    span = frame["to_quality"]
    print(f"exposure over the whole 0-26 sweep: {span.min():+.3f} to {span.max():+.3f} "
          f"(range {span.max() - span.min():.3f})")
    slope = (frame.loc[frame.growth == 16, 'to_quality'].iloc[0]
             - frame.loc[frame.growth == 10, 'to_quality'].iloc[0]) / 6
    print(f"slope near the configured weight (10 -> 16): {slope:+.4f} of correlation per point")
    band = frame[frame["to_quality"].abs() <= 0.05]
    print(f"weights at which |growth exposure| <= 0.05: {band['growth'].min()}-{band['growth'].max()}")

    # --- what the proposal does to every exposure ------------------------------------------
    proposed = {**cfg, "growth": 10, "quality": cfg["quality"] + 3}
    tilt = pd.DataFrame({
        "configured": [base.corr(scores[c], method="spearman") for c in CATS],
        "proposed": [blend(scores, proposed).corr(scores[c], method="spearman") for c in CATS],
    }, index=CATS)
    tilt["change"] = tilt["proposed"] - tilt["configured"]
    print("\nComposite's Spearman with each category score, configured vs proposed:")
    print(tilt.round(3).to_string())

    # --- is Quality the right destination? -------------------------------------------------
    # If Quality's score is already the thing the composite leans on most, extra weight there
    # buys less new information than the same weight somewhere less correlated with the rest.
    print("\nEach category's score against the mean of the other seven (how much of it is "
          "already in the composite):")
    for c in CATS:
        rest = scores[[x for x in CATS if x != c]].mean(axis=1)
        print(f"  {c:<11} {scores[c].corr(rest, method='spearman'):+.3f}")


if __name__ == "__main__":
    main()
