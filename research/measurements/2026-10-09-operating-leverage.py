"""What `operating_leverage` (one-year DOL = %change EBIT / %change revenue) actually measures here.

Companion to research/2026-10-09-operating-leverage.md. Reads the newest run's scored table
(no network) and reports:

1. the distribution: how many values are negative, how many are extreme, and how the sign
   lines up with what happened to margins;
2. how the score treats them (sector percentile by sign);
3. how much of the value is explained by the *size of the revenue change* - the denominator -
   rather than by cost structure;
4. what removing it from the Quality category would do to the published ranking, re-scored
   with the engine's own functions from the run's own percentiles.

    python research/measurements/2026-10-09-operating-leverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def newest_run() -> Path:
    runs = [d for d in (ROOT / "runs").iterdir() if (d / "05_final_scored.parquet").exists()]
    return max(runs, key=lambda d: (d / "05_final_scored.parquet").stat().st_mtime)


def main() -> dict:
    run = newest_run()
    df = pd.read_parquet(run / "05_final_scored.parquet")
    raw = pd.read_parquet(run / "00_raw_fetch.parquet").set_index("Ticker")
    out: dict = {"run": run.name}

    dol = df.set_index("Ticker")["operating_leverage"].dropna()
    out["n"] = int(len(dol))
    out["negative"] = int((dol < 0).sum())
    out["abs_gt_5"] = int((dol.abs() > 5).sum())
    out["abs_gt_10"] = int((dol.abs() > 10).sum())
    out["quantiles"] = {str(q): round(float(dol.quantile(q)), 2) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}

    # What a negative value means: EBIT and revenue moved in opposite directions.
    ebit_c = raw["ebit_annual"].reindex(dol.index)
    ebit_p = raw["ebit_prior"].reindex(dol.index)
    rev_c = raw["totalRevenue_annual"].reindex(dol.index)
    rev_p = raw["totalRevenue_annual_prior"].reindex(dol.index)
    rg = (rev_c - rev_p) / rev_p.abs()
    eg = (ebit_c - ebit_p) / ebit_p.abs()
    neg = dol < 0
    out["negative_breakdown"] = {
        "revenue_up_ebit_down (margin squeeze)": int((neg & (rg > 0) & (eg < 0)).sum()),
        "revenue_down_ebit_up (margin expansion on falling sales)": int((neg & (rg < 0) & (eg > 0)).sum()),
    }
    # Denominator dominance: how much of |DOL| is just 1 / |revenue change|.
    ok = dol.index[rg.notna() & eg.notna()]
    out["spearman_absDOL_vs_inverse_abs_rev_change"] = round(float(
        pd.Series(dol.loc[ok].abs()).rank().corr(pd.Series(1 / rg.loc[ok].abs()).rank())), 3)
    out["share_with_rev_change_under_5pct"] = round(float((rg.loc[ok].abs() < 0.05).mean()), 3)
    out["median_abs_DOL_rev_change_under_5pct"] = round(float(dol.loc[ok][rg.loc[ok].abs() < 0.05].abs().median()), 2)
    out["median_abs_DOL_rev_change_over_10pct"] = round(float(dol.loc[ok][rg.loc[ok].abs() > 0.10].abs().median()), 2)

    # How the score reads them (sector percentile, direction-adjusted: higher = better).
    pct_col = "operating_leverage_pct" if "operating_leverage_pct" in df.columns else None
    if pct_col:
        p = df.set_index("Ticker")[pct_col].reindex(dol.index)
        out["mean_pct_negative"] = round(float(p[neg].mean()), 1)
        out["mean_pct_positive"] = round(float(p[~neg].mean()), 1)

    # Overlap with the rest of Quality (rank correlation of raw values, sign as scored).
    others = ["roic", "gross_profit_assets", "net_debt_to_ebitda", "piotroski_f_score", "accruals", "beneish_m_score"]
    sc = df.set_index("Ticker")
    out["spearman_with_other_quality"] = {m: round(float(sc.loc[dol.index, "operating_leverage"].rank().corr(sc.loc[dol.index, m].rank())), 3)
                                          for m in others if m in sc.columns}
    out["effect_of_weight_zero"] = effect_of_weight_zero(run, df)
    print(json.dumps(out, indent=1))
    return out


def effect_of_weight_zero(run: Path, df: pd.DataFrame) -> dict:
    """Re-score the run's own table with operating_leverage at weight 0 (its 8 points spread
    over the other Quality metrics in proportion, which is what the weighted average does),
    using the engine's own category and composite functions and the run's effective config."""
    import copy

    import yaml

    import factor_engine as fe
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    eff = json.loads((run / "effective_weights.json").read_text(encoding="utf-8"))
    cfg["factor_weights"] = eff["factor_weights"]
    cfg["metric_weights"] = eff["metric_weights"]
    base = fe.compute_composite(fe.compute_category_scores(df.copy(), copy.deepcopy(cfg)), copy.deepcopy(cfg))
    alt_cfg = copy.deepcopy(cfg)
    alt_cfg["metric_weights"]["quality"]["operating_leverage"] = 0
    alt = fe.compute_composite(fe.compute_category_scores(df.copy(), alt_cfg), copy.deepcopy(alt_cfg))
    b = base.set_index("Ticker")["Composite"].rank(ascending=False, method="min")
    a = alt.set_index("Ticker")["Composite"].rank(ascending=False, method="min")
    pub = df.set_index("Ticker")["Composite"]
    top = lambda r, n: set(r[r <= n].index)
    moved = (a - b).abs()
    return {
        # Re-running the category step on a finished table reproduces all but a handful: the
        # growth-trap Piotroski profile reads quality-score quantiles, which in the pipeline
        # come from the pass before. Baseline and alternative share the procedure, so the
        # comparison between them is like for like.
        "baseline_matches_published": f"{int(((base.set_index('Ticker')['Composite'] - pub).abs() < 1e-6).sum())} of {len(pub)}",
        "baseline_max_abs_diff": round(float((base.set_index("Ticker")["Composite"] - pub).abs().max()), 2),
        "spearman_rank": round(float(a.corr(b, method="spearman")), 4),
        "top25_changes": len(top(b, 25) - top(a, 25)),
        "top50_changes": len(top(b, 50) - top(a, 50)),
        "median_abs_rank_move": float(moved.median()),
        "max_abs_rank_move": float(moved.max()),
        "quality_score_mean_abs_change": round(float((alt["quality_score"] - base["quality_score"]).abs().mean()), 2),
    }


if __name__ == "__main__":
    main()
