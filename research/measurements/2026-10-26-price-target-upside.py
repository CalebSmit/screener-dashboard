"""What is `price_target_upside` actually measuring inside this screener?

Written for research/2026-10-26-price-target-upside.md, the Monday note on the one weighted
metric in the composite that has never had a research note of its own. It carries 13.5 of the
Revisions category's 100 points, i.e. 1.35 of the 100 composite points, and it is the only
metric in that category that is a *level* rather than a change:

    price_target_upside = clip(targetMeanPrice / currentPrice - 1, -0.50, +1.00)

Price is in the denominator. So if consensus targets move more slowly than prices do, the
cross-sectional variation in this metric is mostly variation in *recent price*, with the sign
reversed - and the screener would be paying 1.35 points for a slow negative-momentum signal
inside a category whose own config comment warns that it has already spent its independence
budget against momentum.

That is a structural question about this screener's inputs, and it is answerable from data
already on disk without any return, IC or backtest figure (CLAUDE.md rules 4 and 5). Nothing
here reads a forward return. Four measurements:

  A. Coverage and distribution in the published run, including how much of the range is clamp.
  B. **Stickiness.** From the committed history of dashboard_data.js (one version per published
     run date, each carrying pt_mean and price per stock), how far does the consensus target
     move between runs compared with the price? And how much of the change in implied upside is
     the target moving versus the price moving? This is the crux and it is a fact about the
     data, not about returns.
  C. **Overlap.** Spearman of the metric against the momentum inputs and the category scores of
     the same published run, pooled and within sector.
  D. **What it buys.** The Revisions score recomputed with this metric's weight redistributed to
     the category's other weighted metrics, and what that does to the ranking and to the
     composite's momentum exposure. Reported only after showing how closely the reconstruction
     reproduces the published numbers, so the reader can discount it appropriately.

Writes research/measurements/2026-10-26-price-target-upside.json and prints a report. Touches
no tracked artifact and imports no scoring module.

Run from the repo root:  python research/measurements/2026-10-26-price-target-upside.py
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

MOMENTUM_INPUTS = ["return_12_1", "return_6m", "return_12m", "jensens_alpha"]
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
# How far back through the committed payload history to walk. 60 published run dates is about
# three months of weekdays, which is as far as pt_mean has been in the payload.
MAX_DATES = 60


# --------------------------------------------------------------------------- helpers
def latest_run_dir() -> Path:
    """The newest run by meta.json start_time - run_context.latest_run_dir's rule, not mtime."""
    best, best_t = None, ""
    for d in (ROOT / "runs").iterdir():
        meta = d / "meta.json"
        if not meta.is_file():
            continue
        t = json.loads(meta.read_text(encoding="utf-8")).get("start_time", "")
        if t > best_t:
            best, best_t = d, t
    return best


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna()
    if ok.sum() < 20:
        return float("nan"), int(ok.sum())
    return float(a[ok].corr(b[ok], method="spearman")), int(ok.sum())


def within_sector_spearman(df: pd.DataFrame, x: str, y: str, sector: str = "Sector") -> tuple[float, int]:
    """Pool the sector-demeaned ranks, which is what a sector-relative screener actually scores."""
    parts = []
    for _, g in df.groupby(sector):
        ok = g[x].notna() & g[y].notna()
        if ok.sum() < 5:
            continue
        sub = g[ok]
        parts.append(pd.DataFrame({
            "x": sub[x].rank(pct=True) - sub[x].rank(pct=True).mean(),
            "y": sub[y].rank(pct=True) - sub[y].rank(pct=True).mean(),
        }))
    if not parts:
        return float("nan"), 0
    p = pd.concat(parts)
    return float(p["x"].corr(p["y"])), len(p)


# --------------------------------------------------------------------------- A. distribution
def measure_distribution(raw: pd.DataFrame, cfg: dict) -> dict:
    lo, hi = cfg.get("metric_clamps", {}).get("price_target_upside", [-0.50, 1.0])
    u = raw["price_target_upside"].dropna()
    return {
        "clamp": [lo, hi],
        "n_universe": int(len(raw)),
        "n_with_value": int(len(u)),
        "coverage_pct": round(100.0 * len(u) / len(raw), 1),
        "mean": round(float(u.mean()), 4),
        "median": round(float(u.median()), 4),
        "p10": round(float(u.quantile(0.10)), 4),
        "p90": round(float(u.quantile(0.90)), 4),
        "share_negative_pct": round(100.0 * float((u < 0).mean()), 1),
        "n_at_lower_clamp": int((u <= lo + 1e-9).sum()),
        "n_at_upper_clamp": int((u >= hi - 1e-9).sum()),
    }


# --------------------------------------------------------------------------- B. stickiness
def payload_versions() -> list[tuple[str, str]]:
    """(date, sha) for the newest committed dashboard_data.js on each date, newest first."""
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
    return rows[:MAX_DATES]


def read_version(sha: str) -> pd.DataFrame | None:
    """{ticker -> price, pt_mean, num_analysts} from one committed payload, or None."""
    blob = subprocess.run(["git", "show", f"{sha}:dashboard_data.js"],
                          cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8", "replace")
    try:
        data = json.loads(blob[blob.index("{"): blob.rindex("}") + 1])
    except ValueError:
        return None
    detail = data.get("stock_detail") or {}
    rows = {}
    for t, sd in detail.items():
        if not isinstance(sd, dict):
            continue
        price, pt = sd.get("price"), sd.get("pt_mean")
        if isinstance(price, (int, float)) and isinstance(pt, (int, float)) and price > 0 and pt > 0:
            rows[t] = {"price": float(price), "pt_mean": float(pt),
                       "n": sd.get("num_analysts"), "price_date": sd.get("price_date")}
    if len(rows) < 100:
        return None
    return pd.DataFrame.from_dict(rows, orient="index")


def measure_stickiness() -> dict:
    versions = payload_versions()
    frames: dict[str, pd.DataFrame] = {}
    for date, sha in versions:
        f = read_version(sha)
        if f is not None:
            frames[date] = f
    dates = sorted(frames)
    steps = []
    for a, b in zip(dates, dates[1:]):
        fa, fb = frames[a], frames[b]
        common = fa.index.intersection(fb.index)
        if len(common) < 100:
            continue
        pa, pb = fa.loc[common, "price"], fb.loc[common, "price"]
        ta, tb = fa.loc[common, "pt_mean"], fb.loc[common, "pt_mean"]
        dlp = np.log(pb / pa)
        dlt = np.log(tb / ta)
        # implied upside, in logs, is exactly log(target) - log(price); so the change in it
        # decomposes additively and the two shares below are a real decomposition.
        dlu = dlt - dlp
        steps.append({
            "from": a, "to": b, "n": int(len(common)),
            "median_abs_dlog_price": round(float(dlp.abs().median()), 5),
            "median_abs_dlog_target": round(float(dlt.abs().median()), 5),
            "pct_target_exactly_unchanged": round(100.0 * float((tb == ta).mean()), 1),
            "pct_price_exactly_unchanged": round(100.0 * float((pb == pa).mean()), 1),
            "corr_dlogupside_dlogprice": round(float(dlu.corr(dlp)), 4),
            "corr_dlogupside_dlogtarget": round(float(dlu.corr(dlt)), 4),
            "var_share_price_pct": round(100.0 * float(dlp.var() / dlu.var()), 1) if dlu.var() > 0 else None,
            "var_share_target_pct": round(100.0 * float(dlt.var() / dlu.var()), 1) if dlu.var() > 0 else None,
        })

    # The same decomposition over the whole window, first date to last, which is the horizon a
    # monthly screener actually re-reads the metric over.
    span = None
    if len(dates) >= 2:
        fa, fb = frames[dates[0]], frames[dates[-1]]
        common = fa.index.intersection(fb.index)
        dlp = np.log(fb.loc[common, "price"] / fa.loc[common, "price"])
        dlt = np.log(fb.loc[common, "pt_mean"] / fa.loc[common, "pt_mean"])
        dlu = dlt - dlp
        span = {
            "from": dates[0], "to": dates[-1], "n": int(len(common)),
            "sd_dlog_price": round(float(dlp.std()), 4),
            "sd_dlog_target": round(float(dlt.std()), 4),
            "sd_dlog_upside": round(float(dlu.std()), 4),
            "corr_dlogupside_dlogprice": round(float(dlu.corr(dlp)), 4),
            "corr_dlogupside_dlogtarget": round(float(dlt.corr(dlu)), 4),
            "corr_dlogtarget_dlogprice": round(float(dlt.corr(dlp)), 4),
            "beta_dlogtarget_on_dlogprice": round(float(np.polyfit(dlp, dlt, 1)[0]), 4),
            "pct_target_exactly_unchanged": round(
                100.0 * float((fb.loc[common, "pt_mean"] == fa.loc[common, "pt_mean"]).mean()), 1),
        }
    # The term structure of staleness. A one-day gap says what the metric is on the day; the
    # horizon that matters is the one the screener is re-read over (the review cadence is
    # monthly, and the bot rebalances monthly), so measure the decomposition at several gaps.
    horizons = []
    for gap in (7, 30, 91, 182):
        pairs, used = [], set()
        for i, a in enumerate(dates):
            want = pd.Timestamp(a) + pd.Timedelta(days=gap)
            cand = [d for d in dates[i + 1:]
                    if abs((pd.Timestamp(d) - want).days) <= max(3, gap * 0.25)]
            if not cand:
                continue
            b = min(cand, key=lambda d: abs((pd.Timestamp(d) - want).days))
            if (a, b) in used:
                continue
            used.add((a, b))
            pairs.append((a, b))
        dlps, dlts = [], []
        for a, b in pairs:
            fa, fb = frames[a], frames[b]
            common = fa.index.intersection(fb.index)
            if len(common) < 100:
                continue
            dlps.append(np.log(fb.loc[common, "price"] / fa.loc[common, "price"]))
            dlts.append(np.log(fb.loc[common, "pt_mean"] / fa.loc[common, "pt_mean"]))
        if not dlps:
            continue
        dlp, dlt = pd.concat(dlps).values, pd.concat(dlts).values
        ok = np.isfinite(dlp) & np.isfinite(dlt)
        dlp, dlt = dlp[ok], dlt[ok]
        dlu = dlt - dlp
        horizons.append({
            "gap_days": gap, "n_pairs": len(pairs), "n_obs": int(len(dlp)),
            # beta 0 = the target never moves (upside is pure inverted price);
            # beta 1 = the target fully tracks the price (upside carries no price information).
            "beta_dlogtarget_on_dlogprice": round(float(np.polyfit(dlp, dlt, 1)[0]), 4),
            "corr_dlogupside_dlogprice": round(float(np.corrcoef(dlu, dlp)[0, 1]), 4),
            "corr_dlogupside_dlogtarget": round(float(np.corrcoef(dlu, dlt)[0, 1]), 4),
            "var_share_explained_by_price_pct": round(
                100.0 * float(np.corrcoef(dlu, dlp)[0, 1] ** 2), 1),
            "median_abs_dlog_price": round(float(np.median(np.abs(dlp))), 4),
            "median_abs_dlog_target": round(float(np.median(np.abs(dlt))), 4),
            "pct_target_exactly_unchanged": round(100.0 * float((dlt == 0).mean()), 1),
        })
    return {"dates_read": dates, "n_dates": len(dates), "steps": steps,
            "span": span, "horizons": horizons}


# --------------------------------------------------------------------------- C. overlap
def measure_overlap(pct: pd.DataFrame, final: pd.DataFrame, raw: pd.DataFrame) -> dict:
    add = [c for c in ["Composite", "Rank"] + [c + "_score" for c in CATS] if c not in raw.columns]
    df = raw.join(final[add], how="inner")
    res = {"pooled": {}, "within_sector": {}}
    targets = (MOMENTUM_INPUTS + ["fy1_revision_3m", "analyst_surprise", "short_interest_ratio"]
               + [c + "_score" for c in CATS] + ["Composite"])
    for col in targets:
        if col not in df.columns:
            continue
        r, n = spearman(df["price_target_upside"], df[col])
        res["pooled"][col] = {"spearman": round(r, 4), "n": n}
        rw, nw = within_sector_spearman(df, "price_target_upside", col)
        res["within_sector"][col] = {"corr": round(rw, 4), "n": nw}

    # The sector-relative percentile the engine actually scores, against the momentum percentile.
    if "price_target_upside_pct" in pct.columns:
        p = pct.copy()
        for c in ["return_12_1_pct", "return_6m_pct", "jensens_alpha_pct"]:
            if c in p.columns:
                r, n = spearman(p["price_target_upside_pct"], p[c])
                res["pooled"][f"PCT::{c}"] = {"spearman": round(r, 4), "n": n}
    return res


# --------------------------------------------------------------------------- D. what it buys
def measure_counterfactual(pct: pd.DataFrame, final: pd.DataFrame, cfg: dict) -> dict:
    """Rebuild the Revisions score with and without the metric, then the composite.

    Deliberately simple: a weighted mean of the sector percentiles a stock has, renormalised
    over the weights present. Fidelity against the published numbers is reported first.
    """
    mw = cfg["metric_weights"]["revisions"]
    weighted = {m: w for m, w in mw.items() if w > 0}
    cols = {m: m + "_pct" for m in weighted if m + "_pct" in pct.columns}
    missing = [m for m in weighted if m not in cols]

    idx = final.index
    p = pct.reindex(idx)

    def blend(weights: dict[str, float]) -> pd.Series:
        num = pd.Series(0.0, index=idx)
        den = pd.Series(0.0, index=idx)
        for m, w in weights.items():
            if m not in cols or w <= 0:
                continue
            v = pd.to_numeric(p[cols[m]], errors="coerce")
            ok = v.notna()
            num = num.add((v.fillna(0.0) * w).where(ok, 0.0))
            den = den.add(pd.Series(float(w), index=idx).where(ok, 0.0))
        # The percentile columns are already on 0-100, the same scale as the published scores.
        return num / den.replace(0.0, np.nan)

    rev_now = blend(weighted)
    published = pd.to_numeric(final["revisions_score"], errors="coerce")
    fidelity = {
        "spearman_vs_published_revisions": round(float(rev_now.corr(published, method="spearman")), 4),
        "mean_abs_diff_points": round(float((rev_now - published).abs().mean()), 3),
        "metrics_weighted": sorted(weighted),
        "metrics_without_a_percentile_column": missing,
    }

    # Counterfactual: price_target_upside's 13.5 points spread over the rest in proportion.
    w0 = {m: w for m, w in weighted.items() if m != "price_target_upside"}
    total = sum(w0.values())
    w_cf = {m: 100.0 * w / total for m, w in w0.items()}
    rev_cf = blend(w_cf)

    fw = cfg["factor_weights"]
    scores_now = {c: pd.to_numeric(final[c + "_score"], errors="coerce") for c in CATS}

    def composite(rev: pd.Series) -> pd.Series:
        num = pd.Series(0.0, index=idx)
        den = pd.Series(0.0, index=idx)
        for c in CATS:
            v = rev if c == "revisions" else scores_now[c]
            w = float(fw[c])
            ok = v.notna()
            num = num.add((v.fillna(0.0) * w).where(ok, 0.0))
            den = den.add(pd.Series(w, index=idx).where(ok, 0.0))
        return num / den.replace(0.0, np.nan)

    comp_now, comp_cf = composite(rev_now), composite(rev_cf)
    pub_comp = pd.to_numeric(final["Composite"], errors="coerce")
    fidelity["spearman_vs_published_composite"] = round(
        float(comp_now.corr(pub_comp, method="spearman")), 4)

    rank_now = comp_now.rank(ascending=False, method="min")
    rank_cf = comp_cf.rank(ascending=False, method="min")
    moved = (rank_cf - rank_now).abs()
    top25_now = set(rank_now.nsmallest(25).index)
    top25_cf = set(rank_cf.nsmallest(25).index)

    mom = scores_now["momentum"]
    val = scores_now["valuation"]
    exposure = {
        "momentum_exposure_now": round(float(comp_now.corr(mom, method="spearman")), 4),
        "momentum_exposure_without": round(float(comp_cf.corr(mom, method="spearman")), 4),
        "valuation_exposure_now": round(float(comp_now.corr(val, method="spearman")), 4),
        "valuation_exposure_without": round(float(comp_cf.corr(val, method="spearman")), 4),
    }
    return {
        "fidelity": fidelity,
        "rank_change": {
            "spearman_now_vs_without": round(float(rank_now.corr(rank_cf, method="spearman")), 5),
            "median_abs_rank_move": round(float(moved.median()), 1),
            "p90_abs_rank_move": round(float(moved.quantile(0.90)), 1),
            "max_abs_rank_move": round(float(moved.max()), 1),
            "top25_names_changed": len(top25_now - top25_cf),
            "top25_leaving": sorted(top25_now - top25_cf),
            "top25_entering": sorted(top25_cf - top25_now),
        },
        "exposure": exposure,
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    run = latest_run_dir()
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    raw, pct, final = (pd.read_parquet(run / f).set_index("Ticker") for f in
                       ("01_raw_metrics.parquet", "03_percentiles.parquet", "05_final_scored.parquet"))

    result = {
        "run_dir": run.name,
        "run_start": json.loads((run / "meta.json").read_text(encoding="utf-8")).get("start_time"),
        "category_weight_revisions": cfg["factor_weights"]["revisions"],
        "metric_weight_price_target_upside": cfg["metric_weights"]["revisions"]["price_target_upside"],
        "composite_points": round(
            cfg["factor_weights"]["revisions"]
            * cfg["metric_weights"]["revisions"]["price_target_upside"] / 100.0, 3),
        "A_distribution": measure_distribution(raw, cfg),
        "B_stickiness": measure_stickiness(),
        "C_overlap": measure_overlap(pct, final, raw),
        "D_counterfactual": measure_counterfactual(pct, final, cfg),
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    a, b, c, d = (result["A_distribution"], result["B_stickiness"],
                  result["C_overlap"], result["D_counterfactual"])
    print(f"run {result['run_dir']} ({result['run_start']})")
    print(f"weight: {result['metric_weight_price_target_upside']} of the revisions category's 100, "
          f"category {result['category_weight_revisions']}% -> {result['composite_points']} composite points\n")

    print("A. DISTRIBUTION")
    print(f"   coverage {a['n_with_value']}/{a['n_universe']} ({a['coverage_pct']}%); "
          f"median {a['median']:+.3f}, mean {a['mean']:+.3f}, p10 {a['p10']:+.3f}, p90 {a['p90']:+.3f}")
    print(f"   negative upside: {a['share_negative_pct']}% of stocks; "
          f"at the clamps: {a['n_at_lower_clamp']} low, {a['n_at_upper_clamp']} high\n")

    print(f"B. STICKINESS  ({b['n_dates']} published run dates read: "
          f"{b['dates_read'][0] if b['dates_read'] else '-'} .. {b['dates_read'][-1] if b['dates_read'] else '-'})")
    for s in b["steps"][-6:]:
        print(f"   {s['from']} -> {s['to']}  n={s['n']:3d}  "
              f"|dlog price| {s['median_abs_dlog_price']:.4f}  |dlog target| {s['median_abs_dlog_target']:.4f}  "
              f"target unchanged {s['pct_target_exactly_unchanged']:5.1f}%  "
              f"corr(dupside,dprice) {s['corr_dlogupside_dlogprice']:+.3f}")
    if b["span"]:
        s = b["span"]
        print(f"   WHOLE WINDOW {s['from']} -> {s['to']}  n={s['n']}")
        print(f"     sd dlog price {s['sd_dlog_price']:.4f} | sd dlog target {s['sd_dlog_target']:.4f} "
              f"| sd dlog upside {s['sd_dlog_upside']:.4f}")
        print(f"     corr(d upside, d price) {s['corr_dlogupside_dlogprice']:+.4f}   "
              f"corr(d upside, d target) {s['corr_dlogupside_dlogtarget']:+.4f}")
        print(f"     beta of d log target on d log price {s['beta_dlogtarget_on_dlogprice']:+.4f}   "
              f"target exactly unchanged over the window {s['pct_target_exactly_unchanged']}%")
    print("   TERM STRUCTURE (beta 0 = target never moves, upside is pure inverted price; "
          "beta 1 = target fully tracks price)")
    for h in b.get("horizons", []):
        print(f"     ~{h['gap_days']:3d}d  pairs {h['n_pairs']:2d}  obs {h['n_obs']:6d}  "
              f"beta {h['beta_dlogtarget_on_dlogprice']:+.3f}  "
              f"corr(dupside,dprice) {h['corr_dlogupside_dlogprice']:+.3f}  "
              f"price explains {h['var_share_explained_by_price_pct']:4.1f}% of d upside  "
              f"target unchanged {h['pct_target_exactly_unchanged']:5.1f}%")
    print()

    print("C. OVERLAP (Spearman with price_target_upside, published run)")
    for k, v in c["pooled"].items():
        w = c["within_sector"].get(k, {})
        ws = f"  within-sector {w['corr']:+.3f}" if w.get("corr") == w.get("corr") and w else ""
        print(f"   {k:28s} {v['spearman']:+.3f}  (n={v['n']}){ws}")
    print()

    print("D. WHAT THE 13.5 POINTS BUY")
    f = d["fidelity"]
    print(f"   reconstruction fidelity: revisions Spearman {f['spearman_vs_published_revisions']}, "
          f"mean abs diff {f['mean_abs_diff_points']} pts; "
          f"composite Spearman {f['spearman_vs_published_composite']}")
    if f["metrics_without_a_percentile_column"]:
        print(f"   NOTE weighted metrics with no percentile column: {f['metrics_without_a_percentile_column']}")
    r = d["rank_change"]
    print(f"   removing it: rank Spearman {r['spearman_now_vs_without']}, median move "
          f"{r['median_abs_rank_move']}, p90 {r['p90_abs_rank_move']}, max {r['max_abs_rank_move']}")
    print(f"   top 25 changes {r['top25_names_changed']}: out {r['top25_leaving']} in {r['top25_entering']}")
    e = d["exposure"]
    print(f"   composite momentum exposure {e['momentum_exposure_now']:+.4f} -> "
          f"{e['momentum_exposure_without']:+.4f}")
    print(f"   composite valuation exposure {e['valuation_exposure_now']:+.4f} -> "
          f"{e['valuation_exposure_without']:+.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
