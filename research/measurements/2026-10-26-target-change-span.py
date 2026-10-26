"""The SS8.4 spanning test for the OPEN proposal in research/2026-10-26-price-target-upside.md.

`config.yaml:128` carries a written rule for the Revisions category:

    The revisions category has now SPENT its independence budget: it goes from ~91% unspanned
    by the other seven categories to 85.2% ... Adding any further momentum-adjacent metric
    here requires re-running the SS8.4 spanning regression first, not just a citation.

The proposal substitutes a 90-day change in the consensus analyst target for the scored level
of implied upside, at the same 13.5 points. The companion script
`2026-10-26-target-change-candidate.py` establishes that the candidate is **momentum-adjacent**
(within sector, median +0.533 with the Momentum category score at the 90-day window and +0.626
with `return_6m`, over 18 end dates), so the rule binds and this is that regression.

Measured here, all through this screener's own arithmetic:

  1. Fidelity. The Revisions score and composite rebuilt from the published percentiles with
     the incumbent weights, against the published numbers. Reported before anything else so
     the counterfactuals can be discounted appropriately.
  2. The eight category-pair correlations for Revisions, published against post-swap - the
     unit the pre-registered ~0.35 bar in SS8.4 was set in.
  3. The independence table: every category's unspanned share (1 - R2 of its score on the
     other seven), published against post-swap.
  4. What the swap does to the ranking.

The candidate's sector percentile is computed with the engine's own rule, re-implemented here
rather than imported (this is a measurement, not production): the midpoint rank
(rank - 0.5) / n * 100 within Sector, falling back to a universe-wide rank when a sector has
fewer than SECTOR_MIN_PEERS = 10 valid values.

Nothing here reads a forward return, an IC or a backtest figure. Imports no scoring module.

Run from the repo root:
    python research/measurements/2026-10-26-target-change-span.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".json")

CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
SECTOR_MIN_PEERS = 10          # factor_engine.SECTOR_MIN_PEERS
WINDOW_DAYS = 90               # the proposal's own window
WINDOW_TOL = 22


# --------------------------------------------------------------------- run + payload inputs
def latest_run_dir() -> Path:
    best, best_t = None, ""
    for d in (ROOT / "runs").iterdir():
        meta = d / "meta.json"
        if not meta.is_file():
            continue
        t = json.loads(meta.read_text(encoding="utf-8")).get("start_time", "")
        if t > best_t:
            best, best_t = d, t
    return best


def payload_dates() -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "log", "--format=%H %ad", "--date=short", "--", "dashboard_data.js"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    seen, rows = set(), []
    for line in out.splitlines():
        sha, date = line.split()
        if date in seen:
            continue
        seen.add(date)
        rows.append((date, sha))
    return sorted(rows)


def targets_at(sha: str) -> pd.Series | None:
    blob = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"],
                          cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace")
    try:
        data = json.loads(blob[blob.index("{"): blob.rindex("}") + 1])
    except ValueError:
        return None
    detail = data.get("stock_detail") or {}
    vals = {t: sd.get("pt_mean") for t, sd in detail.items()
            if isinstance(sd, dict) and isinstance(sd.get("pt_mean"), (int, float))
            and sd["pt_mean"] > 0}
    if len(vals) < 100:
        return None
    return pd.Series(vals, dtype=float)


# --------------------------------------------------------------------- engine-rule percentile
def directed_pct(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    r = s.rank(method="average", na_option="keep")
    p = (r - 0.5) / s.notna().sum() * 100
    return p if higher_is_better else 100 - p


def sector_pct(values: pd.Series, sectors: pd.Series) -> pd.Series:
    """compute_sector_percentiles' rule for one metric, higher-is-better."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    universe = directed_pct(values)
    for _, idx in sectors.groupby(sectors).groups.items():
        grp = values.loc[idx]
        if grp.notna().sum() < SECTOR_MIN_PEERS:
            out.loc[idx] = universe.loc[idx]
        else:
            out.loc[idx] = directed_pct(grp)
    return out


# --------------------------------------------------------------------- blending
def blend(pct: pd.DataFrame, weights: dict[str, float], idx: pd.Index) -> pd.Series:
    """Weighted mean of the percentiles a stock has, renormalised over the weights present."""
    num = pd.Series(0.0, index=idx)
    den = pd.Series(0.0, index=idx)
    for m, w in weights.items():
        if w <= 0 or m not in pct.columns:
            continue
        v = pd.to_numeric(pct[m].reindex(idx), errors="coerce")
        ok = v.notna()
        num = num.add((v.fillna(0.0) * w).where(ok, 0.0))
        den = den.add(pd.Series(float(w), index=idx).where(ok, 0.0))
    return num / den.replace(0.0, np.nan)


def ols_r2(y: pd.Series, X: pd.DataFrame) -> float:
    ok = y.notna() & X.notna().all(axis=1)
    if ok.sum() < 50:
        return float("nan")
    yy = y[ok].rank().values.astype(float)
    xx = X[ok].rank().values.astype(float)
    xx = np.column_stack([np.ones(len(xx)), xx])
    beta, *_ = np.linalg.lstsq(xx, yy, rcond=None)
    resid = yy - xx @ beta
    ss = float(((yy - yy.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan")


def sp(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(a[ok].corr(b[ok], method="spearman")) if ok.sum() >= 30 else float("nan")


# --------------------------------------------------------------------- main
def main() -> None:
    run = latest_run_dir()
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    raw = pd.read_parquet(run / "01_raw_metrics.parquet").set_index("Ticker")
    pctq = pd.read_parquet(run / "03_percentiles.parquet").set_index("Ticker")
    final = pd.read_parquet(run / "05_final_scored.parquet").set_index("Ticker")
    idx = final.index
    sectors = raw["Sector"].reindex(idx)

    # ---- the candidate, over the proposal's 90-day window ending at the newest payload
    dates = payload_dates()
    end_date, end_sha = dates[-1]
    want = pd.Timestamp(end_date) - pd.Timedelta(days=WINDOW_DAYS)
    cand_dates = [(d, s) for d, s in dates[:-1]
                  if abs((pd.Timestamp(d) - want).days) <= WINDOW_TOL]
    start_date, start_sha = min(cand_dates, key=lambda ds: abs((pd.Timestamp(ds[0]) - want).days))
    t_now, t_then = targets_at(end_sha), targets_at(start_sha)
    gap = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    # MSCI PTGF_C's scaling. Rank-identical to S&P Global MI's TP_t/TP_{t-W} - 1 for positive
    # targets (verified on all 98 end dates by the companion script), so the choice between
    # the two documented scalings cannot matter to a percentile screener.
    common = t_now.index.intersection(t_then.index)
    cand = ((t_now[common] - t_then[common]) / ((t_now[common].abs() + t_then[common].abs()) / 2))
    cand = cand.replace([np.inf, -np.inf], np.nan).reindex(idx)

    # ---- percentile frame: published columns, plus the candidate's own
    mw = dict(cfg["metric_weights"]["revisions"])
    pct = pd.DataFrame(index=idx)
    for m in mw:
        col = m + "_pct"
        if col in pctq.columns:
            pct[m] = pd.to_numeric(pctq[col].reindex(idx), errors="coerce")
    pct["target_change_90d"] = sector_pct(cand, sectors)

    # ---- 1. fidelity
    weighted = {m: w for m, w in mw.items() if w > 0}
    rev_pub_rebuild = blend(pct, weighted, idx)
    rev_published = pd.to_numeric(final["revisions_score"], errors="coerce")
    fw = cfg["factor_weights"]
    cat_pub = {c: pd.to_numeric(final[c + "_score"], errors="coerce") for c in CATS}

    def composite(rev: pd.Series) -> pd.Series:
        num = pd.Series(0.0, index=idx)
        den = pd.Series(0.0, index=idx)
        for c in CATS:
            v = rev if c == "revisions" else cat_pub[c]
            w = float(fw[c])
            ok = v.notna()
            num = num.add((v.fillna(0.0) * w).where(ok, 0.0))
            den = den.add(pd.Series(w, index=idx).where(ok, 0.0))
        return num / den.replace(0.0, np.nan)

    comp_pub_rebuild = composite(rev_pub_rebuild)
    comp_published = pd.to_numeric(final["Composite"], errors="coerce")
    fidelity = {
        "revisions_spearman_vs_published": round(sp(rev_pub_rebuild, rev_published), 5),
        "revisions_mean_abs_diff_points": round(
            float((rev_pub_rebuild - rev_published).abs().mean()), 4),
        "composite_spearman_vs_published": round(sp(comp_pub_rebuild, comp_published), 5),
    }

    # ---- the swap: the incumbent's 13.5 points move to the candidate
    w_swap = {m: w for m, w in weighted.items() if m != "price_target_upside"}
    w_swap["target_change_90d"] = float(weighted["price_target_upside"])
    rev_swap = blend(pct, w_swap, idx)
    comp_swap = composite(rev_swap)

    # ---- 2. the eight category-pair correlations for Revisions
    pairs = {}
    for c in CATS:
        if c == "revisions":
            continue
        pairs[c] = {"published": round(sp(rev_published, cat_pub[c]), 4),
                    "rebuilt": round(sp(rev_pub_rebuild, cat_pub[c]), 4),
                    "after_swap": round(sp(rev_swap, cat_pub[c]), 4)}
    pairs["composite"] = {"published": round(sp(rev_published, comp_published), 4),
                          "rebuilt": round(sp(rev_pub_rebuild, comp_pub_rebuild), 4),
                          "after_swap": round(sp(rev_swap, comp_swap), 4)}

    # ---- 3. the independence table
    indep_pub, indep_swap = {}, {}
    for c in CATS:
        others = [o for o in CATS if o != c]
        X = pd.DataFrame({o: cat_pub[o] for o in others})
        y = cat_pub[c]
        indep_pub[c] = round(100.0 * (1.0 - ols_r2(y, X)), 1)
        if c == "revisions":
            indep_swap[c] = round(100.0 * (1.0 - ols_r2(rev_swap, X)), 1)
        else:
            Xs = X.copy()
            Xs["revisions"] = rev_swap
            indep_swap[c] = round(100.0 * (1.0 - ols_r2(y, Xs)), 1)

    # the candidate metric's own spanning, against the incumbent's
    X8 = pd.DataFrame({c: cat_pub[c] for c in CATS})
    metric_span = {
        "candidate_r2_on_all_8": round(ols_r2(cand, X8), 4),
        "candidate_r2_on_momentum_alone": round(ols_r2(cand, X8[["momentum"]]), 4),
        "incumbent_r2_on_all_8": round(ols_r2(
            pd.to_numeric(raw["price_target_upside"].reindex(idx), errors="coerce"), X8), 4),
        "incumbent_r2_on_momentum_alone": round(ols_r2(
            pd.to_numeric(raw["price_target_upside"].reindex(idx), errors="coerce"),
            X8[["momentum"]]), 4),
    }

    # ---- 4. ranking effect
    r_now = comp_pub_rebuild.rank(ascending=False, method="min")
    r_swap = comp_swap.rank(ascending=False, method="min")
    moved = (r_swap - r_now).abs()
    t25n, t25s = set(r_now.nsmallest(25).index), set(r_swap.nsmallest(25).index)
    t50n, t50s = set(r_now.nsmallest(50).index), set(r_swap.nsmallest(50).index)
    ranking = {
        "spearman": round(float(r_now.corr(r_swap, method="spearman")), 5),
        "median_abs_rank_move": float(moved.median()),
        "p90_abs_rank_move": float(moved.quantile(0.90)),
        "max_abs_rank_move": float(moved.max()),
        "top25_changed": len(t25n - t25s),
        "top25_leaving": sorted(t25n - t25s),
        "top25_entering": sorted(t25s - t25n),
        "top50_turnover": len(t50n - t50s),
        "names_moving_over_10": int((moved > 10).sum()),
    }

    # ---- composite-level exposures
    exposure = {
        "composite_momentum_now": round(sp(comp_pub_rebuild, cat_pub["momentum"]), 4),
        "composite_momentum_after": round(sp(comp_swap, cat_pub["momentum"]), 4),
        "composite_valuation_now": round(sp(comp_pub_rebuild, cat_pub["valuation"]), 4),
        "composite_valuation_after": round(sp(comp_swap, cat_pub["valuation"]), 4),
    }

    result = {
        "run_dir": run.name,
        "run_start": json.loads((run / "meta.json").read_text(encoding="utf-8")).get("start_time"),
        "window": {"start_date": start_date, "end_date": end_date, "gap_days": gap,
                   "n_with_candidate": int(cand.notna().sum()), "n_universe": int(len(idx))},
        "weights_before": weighted,
        "weights_after": w_swap,
        "fidelity": fidelity,
        "revisions_category_pairs": pairs,
        "independence_pct_unspanned": {"published": indep_pub, "after_swap": indep_swap},
        "metric_spanning": metric_span,
        "ranking": ranking,
        "exposure": exposure,
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ report
    print(f"run {run.name} ({result['run_start']})")
    print(f"candidate window {start_date} -> {end_date} ({gap} days), "
          f"{result['window']['n_with_candidate']}/{result['window']['n_universe']} stocks\n")
    print("1. FIDELITY of the rebuild against the published numbers")
    for k, v in fidelity.items():
        print(f"   {k:42s} {v}")
    print()
    print("2. REVISIONS CATEGORY PAIRS (Spearman)   [SS8.4's pre-registered bar: ~0.35]")
    print(f"   {'pair':24s} {'published':>10s} {'rebuilt':>9s} {'after swap':>11s}   delta")
    for c, v in pairs.items():
        print(f"   revisions ~ {c:12s} {v['published']:+10.4f} {v['rebuilt']:+9.4f} "
              f"{v['after_swap']:+11.4f}   {v['after_swap'] - v['rebuilt']:+.4f}")
    print()
    print("3. INDEPENDENCE (% unspanned by the other seven)")
    print(f"   {'category':14s} {'published':>10s} {'after swap':>11s}")
    for c in CATS:
        print(f"   {c:14s} {indep_pub[c]:10.1f} {indep_swap[c]:11.1f}")
    print("   the metric itself:")
    for k, v in metric_span.items():
        print(f"     {k:38s} {v}")
    print()
    print("4. RANKING EFFECT of the swap")
    for k, v in ranking.items():
        if k.endswith(("leaving", "entering")):
            print(f"   {k:24s} {v}")
        else:
            print(f"   {k:24s} {v}")
    print()
    print("5. COMPOSITE EXPOSURE")
    for k, v in exposure.items():
        print(f"   {k:30s} {v:+.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
