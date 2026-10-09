"""The effect of each 2026-10-09 methodology change on the ranking, measured on one run's own data.

Comparing tonight's ranks with this morning's would mix the changes with a day of price moves. This
re-scores the SAME run (its raw fetch and metrics) with each change undone, one at a time and all
together, using the engine's own percentile, category and composite functions, and reports the
rank movement each change causes:

1. `operating_leverage` weight 8 -> 0 (Quality reweighted 27/20/18/15/5/8/7 -> 29/22/20/16/5/0/8)
2. `revenue_growth`: TTM over the fiscal year before last -> latest quarter over the same quarter
   a year earlier (else fiscal year over fiscal year)
3. Piotroski signals 3/8/9: TTM vs the fiscal year before last -> two fiscal years, beginning-of-year assets
4. coverage discount: all registered metrics -> weighted metrics only

    python research/measurements/2026-10-09-methodology-effects.py [run_id]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import factor_engine as fe  # noqa: E402

OLD_QUALITY = {"roic": 27, "gross_profit_assets": 20, "net_debt_to_ebitda": 18, "piotroski_f_score": 15,
               "accruals": 5, "operating_leverage": 8, "beneish_m_score": 7}


def newest_run() -> Path:
    runs = [d for d in (ROOT / "runs").iterdir() if (d / "05_final_scored.parquet").exists()
            and (d / "00_raw_fetch.parquet").exists()]
    return max(runs, key=lambda d: (d / "05_final_scored.parquet").stat().st_mtime)


def old_revenue_growth(raw: pd.DataFrame) -> pd.Series:
    r, p = raw["totalRevenue"], raw["totalRevenue_prior"]
    return ((r - p) / p).where((p > 0) & r.notna() & p.notna())


def old_piotroski(raw: pd.DataFrame, scored: pd.DataFrame) -> pd.Series:
    """Rebuild the F-score with signals 3/8/9 on the old basis, from the published signal string."""
    out = {}
    for t, row in raw.iterrows():
        sig = scored.at[t, "_pio_signals"] if t in scored.index and "_pio_signals" in scored.columns else None
        if not isinstance(sig, str) or len(sig) != 9:
            out[t] = np.nan
            continue
        s = [None if c == "-" else int(c) for c in sig]
        ni, nip = row.get("netIncome"), row.get("netIncome_prior")
        ta, tap = row.get("totalAssets"), row.get("totalAssets_prior")
        gp, gpp = row.get("grossProfit"), row.get("grossProfit_prior")
        rv, rvp = row.get("totalRevenue"), row.get("totalRevenue_prior")
        ok = lambda *x: all(pd.notna(v) for v in x)
        s[2] = int(ni / ta > nip / tap) if ok(ni, nip, ta, tap) and ta > 0 and tap > 0 else None
        s[7] = int(gp / rv > gpp / rvp) if ok(gp, gpp, rv, rvp) and rv > 0 and rvp > 0 else None
        s[8] = int(rv / ta > rvp / tap) if ok(rv, rvp, ta, tap) and ta > 0 and tap > 0 else None
        n = sum(x is not None for x in s)
        out[t] = float(sum(x for x in s if x is not None)) if n >= 6 else np.nan
    return pd.Series(out)


def score(metrics: pd.DataFrame, cfg: dict, coverage_cfg: bool = True) -> pd.Series:
    """Percentiles -> categories -> composite with the engine's functions; returns rank."""
    df = fe.compute_sector_percentiles(metrics.copy())
    df = fe.compute_category_scores(df, copy.deepcopy(cfg))
    c = copy.deepcopy(cfg)
    if not coverage_cfg:
        # old coverage rule: applicable_coverage(df) without cfg (all registered metrics)
        orig = fe.applicable_coverage
        fe.applicable_coverage = lambda d, cfg=None: orig(d, None)
        try:
            df = fe.compute_composite(df, c)
        finally:
            fe.applicable_coverage = orig
    else:
        df = fe.compute_composite(df, c)
    return df.set_index("Ticker")["Composite"].rank(ascending=False, method="min")


def compare(a: pd.Series, b: pd.Series) -> dict:
    both = a.index.intersection(b.index)
    a, b = a[both], b[both]
    top = lambda r, n: set(r[r <= n].index)
    mv = (a - b).abs()
    return {"spearman": round(float(a.corr(b, method="spearman")), 4),
            "top25_changes": len(top(a, 25) - top(b, 25)), "top50_changes": len(top(a, 50) - top(b, 50)),
            "median_rank_move": float(mv.median()), "p90_rank_move": float(mv.quantile(0.9)),
            "max_rank_move": float(mv.max())}


def main(run_id: str | None = None) -> dict:
    run = (ROOT / "runs" / run_id) if run_id else newest_run()
    scored = pd.read_parquet(run / "05_final_scored.parquet")
    raw = pd.read_parquet(run / "00_raw_fetch.parquet").set_index("Ticker")
    # Post-outlier metrics: the table the pipeline computes percentiles from.
    metrics = pd.read_parquet(run / "02_outliers_flagged.parquet")
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    eff = json.loads((run / "effective_weights.json").read_text(encoding="utf-8"))
    cfg["factor_weights"] = eff["factor_weights"]
    sc = scored.set_index("Ticker")
    base = metrics[[c for c in metrics.columns if not c.endswith("_pct") and not c.endswith("_score")]].copy()

    new = score(base, cfg)
    variants = {}
    # 1. operating leverage back at 8
    c1 = copy.deepcopy(cfg)
    c1["metric_weights"]["quality"].update(OLD_QUALITY)
    variants["1_operating_leverage_weight"] = score(base, c1)
    # 2. old revenue growth
    m2 = base.copy()
    m2["revenue_growth"] = m2["Ticker"].map(old_revenue_growth(raw))
    variants["2_revenue_growth_window"] = score(m2, cfg)
    # 3. old Piotroski signals
    m3 = base.copy()
    m3["piotroski_f_score"] = m3["Ticker"].map(old_piotroski(raw, sc))
    variants["3_piotroski_annual"] = score(m3, cfg)
    # 4. old coverage rule
    variants["4_coverage_weighted_only"] = score(base, cfg, coverage_cfg=False)
    # all four undone
    m_all = base.copy()
    m_all["revenue_growth"] = m_all["Ticker"].map(old_revenue_growth(raw))
    m_all["piotroski_f_score"] = m_all["Ticker"].map(old_piotroski(raw, sc))
    variants["all_four"] = score(m_all, c1, coverage_cfg=False)

    out = {"run": run.name,
           "new_reproduces_published": compare(new, sc["Composite"].rank(ascending=False, method="min")),
           "effects": {k: compare(new, v) for k, v in variants.items()}}
    (Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
