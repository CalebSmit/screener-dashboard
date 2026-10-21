"""What the Size category does inside the published composite (factor-lab question 4).

Written for research/2026-10-20-size-weight.md. Structure only: no return, IC, backtest or
performance figure is read anywhere (CLAUDE.md rules 4 and 5), and nothing in the screener is
changed. The composite is recomputed with the engine's own `factor_engine.compute_composite` from
the run's own scored frame and config, and the script first checks that it reproduces the
published composite exactly before reading anything off a variant.

Reads the newest full run (runs/<id>/05_final_scored.parquet, config.yaml, effective_weights.json)
from the main checkout (`runs/` is gitignored), and checks that its run timestamp is the one the
committed dashboard_data.js carries.

  1. How Size is built on this run: its coverage, and how far within-sector ranking leaves it a
     universe-wide size bet (Spearman with -ln market cap).
  2. Its correlation with the other seven category scores, and how much of it they span (R^2),
     with Valuation and Investment singled out.
  3. The composite's own exposure to size, as the Size weight moves from 0 to 15 (the other
     categories pro rata): at what weight is the ranking neutral to size? (The Growth note's
     test, research/2026-10-12-growth-category.md section 10.)
  4. The top 25 with Size's 5 points held neutral (every stock's Size score set to 50 - the
     category keeps its weight and says nothing) against the published top 25: who leaves, who
     enters, and what the two lists look like on every other category and on market cap.
  5. The stocks Size promotes and demotes most, by category (the "junk" check of the 2026-08-31
     note, re-measured on today's run).

Run from the repo root:  python research/measurements/2026-10-20-size-composite.py
Writes research/measurements/2026-10-20-size-composite.json.
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

MAIN = Path(r"C:\Users\smitc\OneDrive\Documents\Screener")
RUNS = (ROOT / "runs") if (ROOT / "runs").exists() else MAIN / "runs"
OUT = Path(__file__).with_suffix(".json")
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
SC = [c + "_score" for c in CATS]


def payload() -> dict:
    text = (ROOT / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def find_run(ts: str) -> Path:
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = d / "meta.json"
        if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("start_time") == ts \
                and (d / "05_final_scored.parquet").exists():
            return d
    raise SystemExit(f"no run directory with start_time {ts}")


def composite(df: pd.DataFrame, cfg: dict, weights: dict) -> pd.Series:
    c = copy.deepcopy(cfg)
    c["factor_weights"] = dict(weights)
    out = fe.compute_composite(df.drop(columns=["Composite"], errors="ignore").copy(), c)
    return out["Composite"]


def rank(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="first")


def top(s: pd.Series, n: int = 25) -> list[str]:
    return list(s.sort_values(ascending=False).index[:n])


def r2(y: pd.Series, X: pd.DataFrame) -> float:
    m = pd.concat([y, X], axis=1).dropna()
    A = np.column_stack([np.ones(len(m)), m.iloc[:, 1:].to_numpy()])
    b, *_ = np.linalg.lstsq(A, m.iloc[:, 0].to_numpy(), rcond=None)
    e = m.iloc[:, 0].to_numpy() - A @ b
    v = m.iloc[:, 0].to_numpy()
    return float(1 - e @ e / ((v - v.mean()) @ (v - v.mean())))


def profile(df: pd.DataFrame, names: list[str]) -> dict:
    sub = df.loc[names]
    out = {c: round(float(sub[c + "_score"].mean()), 1) for c in CATS}
    out["median_mcap_bn"] = round(float(sub["mcap_bn"].median()), 1)
    out["n_below_50bn"] = int((sub["mcap_bn"] < 50).sum())
    out["sectors"] = sub["Sector"].value_counts().to_dict()
    return out


def main() -> None:
    data = payload()
    ts = data["kpis"]["run_timestamp"]
    run = find_run(ts)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    eff = json.loads((run / "effective_weights.json").read_text(encoding="utf-8"))
    fw = eff["factor_weights"]
    df = pd.read_parquet(run / "05_final_scored.parquet").set_index("Ticker")
    df["mcap_bn"] = np.exp(-df["size_log_mcap"]) / 1e9
    published = pd.Series({r["Ticker"]: r["Composite"] for r in data["table_data"]}, dtype=float)

    base = composite(df, cfg, fw)
    gap = (base - published.reindex(base.index)).abs()
    repro = {"run_dir": run.name, "run_timestamp": ts, "stocks": int(len(df)),
             "max_abs_diff_vs_published": float(gap.max()),
             "top25_identical": top(base) == top(published)}
    print("reproduction:", repro)
    assert gap.max() < 0.011, "the recomputed composite does not reproduce the published one"

    out: dict = {"reproduction": repro, "weights": fw}

    # 1. how Size is built
    s = df["size_score"]
    out["size_build"] = {
        "coverage": int(s.notna().sum()),
        "spearman_with_minus_ln_mcap_universe": round(float(s.corr(df["size_log_mcap"], method="spearman")), 3),
        "mcap_bn_quantiles": {q: round(float(df["mcap_bn"].quantile(q)), 1) for q in (0, .1, .25, .5, .75, .9, 1)},
        "size_score_by_mcap_bucket": {
            lab: round(float(s[(df["mcap_bn"] >= lo) & (df["mcap_bn"] < hi)].mean()), 1)
            for lab, lo, hi in (("<25bn", 0, 25), ("25-50bn", 25, 50), ("50-100bn", 50, 100),
                                ("100-500bn", 100, 500), (">=500bn", 500, 1e9))},
    }

    # 2. correlations
    sc = df[SC].rename(columns=lambda c: c[:-6])
    corr = sc.corr(method="spearman")["size"].drop("size").round(3).to_dict()
    pear = sc.corr()["size"].drop("size").round(3).to_dict()
    others = [c for c in CATS if c != "size"]
    out["size_vs_other_categories"] = {
        "spearman": corr, "pearson": pear,
        "r2_on_other_seven": round(r2(sc["size"], sc[others]), 3),
        "r2_on_valuation_and_investment": round(r2(sc["size"], sc[["valuation", "investment"]]), 3),
        "r2_on_valuation_alone": round(r2(sc["size"], sc[["valuation"]]), 3),
        "valuation_vs_investment_spearman": round(float(sc["valuation"].corr(sc["investment"], method="spearman")), 3),
        "rest_mean_vs_size_spearman": round(float(sc[others].mean(axis=1).corr(sc["size"], method="spearman")), 3),
    }
    # the same correlations within sector (each sector's Spearman, stock-weighted average)
    within = {}
    for c in others:
        vals = []
        for _, g in df.groupby("Sector"):
            if len(g) >= 8:
                vals.append((len(g), g["size_score"].corr(g[c + "_score"], method="spearman")))
        within[c] = round(float(sum(n * v for n, v in vals) / sum(n for n, _ in vals)), 3)
    out["size_vs_other_categories"]["spearman_within_sector_avg"] = within

    # 3. the composite's exposure to size, by Size weight (others pro rata)
    sweep = []
    other_total = sum(fw[c] for c in others)
    for w in range(0, 16):
        wts = {**{c: fw[c] * (100 - w) / other_total for c in others}, "size": w}
        comp = composite(df, cfg, wts)
        sweep.append({"size_weight": w,
                      "spearman_with_size_score": round(float(comp.corr(s, method="spearman")), 3),
                      "spearman_with_minus_ln_mcap": round(float(comp.corr(df["size_log_mcap"], method="spearman")), 3),
                      "top25_kept_vs_published": len(set(top(comp)) & set(top(base))),
                      "top25_median_mcap_bn": round(float(df.loc[top(comp), "mcap_bn"].median()), 1)})
    out["size_weight_sweep"] = sweep

    def crossing(key: str) -> float | None:
        for a, b in zip(sweep, sweep[1:]):
            x, y = a[key], b[key]
            if (x <= 0 <= y) or (y <= 0 <= x):
                return round(a["size_weight"] + (0 if x == y else -x / (y - x)), 2)
        return None
    out["neutral_weight"] = {"within_sector_size_score": crossing("spearman_with_size_score"),
                             "universe_minus_ln_mcap": crossing("spearman_with_minus_ln_mcap")}
    # and the composite's exposure to size with the category's own contribution removed
    neutral_df = df.copy()
    neutral_df["size_score"] = 50.0
    neutral = composite(neutral_df, cfg, fw)
    out["exposure_without_size_signal"] = {
        "spearman_with_size_score": round(float(neutral.corr(s, method="spearman")), 3),
        "spearman_with_minus_ln_mcap": round(float(neutral.corr(df["size_log_mcap"], method="spearman")), 3)}
    out["exposure_published"] = {
        "spearman_with_size_score": round(float(base.corr(s, method="spearman")), 3),
        "spearman_with_minus_ln_mcap": round(float(base.corr(df["size_log_mcap"], method="spearman")), 3)}

    # 4. the top 25, Size held neutral
    variants = {
        "published": base,
        "size_neutral_50": neutral,
        "size_dropped_renormalised": composite(df, cfg, {**fw, "size": 0}),
        "size_points_to_quality": composite(df, cfg, {**fw, "size": 0, "quality": fw["quality"] + fw["size"]}),
        "size_points_to_valuation": composite(df, cfg, {**fw, "size": 0, "valuation": fw["valuation"] + fw["size"]}),
    }
    pr = rank(base)
    vt = {}
    for k, comp in variants.items():
        t = top(comp)
        rk = rank(comp)
        moved = (rk - pr).abs()
        vt[k] = {"top25": t, "kept_of_25": len(set(t) & set(top(base))),
                 "kept_of_50": len(set(top(comp, 50)) & set(top(base, 50))),
                 "spearman_vs_published": round(float(comp.corr(base, method="spearman")), 4),
                 "median_abs_rank_change": float(moved.median()), "p90_abs_rank_change": float(moved.quantile(.9)),
                 "max_abs_rank_change": float(moved.max()), "profile_top25": profile(df, t),
                 "composite_exposure": {c: round(float(comp.corr(df[c + "_score"], method="spearman")), 3) for c in CATS}}
    out["variants"] = vt
    pub_t, neu_t = top(base), top(neutral)
    rn = rank(neutral)
    out["top25_changes_size_neutral"] = {
        "leave": [{"ticker": t, "sector": df.at[t, "Sector"], "mcap_bn": round(float(df.at[t, "mcap_bn"]), 1),
                   "size_score": round(float(df.at[t, "size_score"]), 1), "rank_published": int(pr[t]),
                   "rank_size_neutral": int(rn[t]),
                   **{c: round(float(df.at[t, c + "_score"]), 1) for c in ("valuation", "quality", "investment", "risk")}}
                  for t in pub_t if t not in neu_t],
        "enter": [{"ticker": t, "sector": df.at[t, "Sector"], "mcap_bn": round(float(df.at[t, "mcap_bn"]), 1),
                   "size_score": round(float(df.at[t, "size_score"]), 1), "rank_published": int(pr[t]),
                   "rank_size_neutral": int(rn[t]),
                   **{c: round(float(df.at[t, c + "_score"]), 1) for c in ("valuation", "quality", "investment", "risk")}}
                  for t in neu_t if t not in pub_t],
        "top25_size_score_mean_published": round(float(df.loc[pub_t, "size_score"].mean()), 1),
        "universe_size_score_mean": round(float(s.mean()), 1),
    }

    # 5. promoted and demoted most by Size (rank with Size neutral minus published rank)
    delta = rn - pr                     # positive = Size moved the stock up
    up, down = list(delta.sort_values(ascending=False).index[:50]), list(delta.sort_values().index[:50])
    prof = lambda names: {**{c: round(float(df.loc[names, c + "_score"].mean()), 1) for c in CATS},   # noqa: E731
                          "median_mcap_bn": round(float(df.loc[names, "mcap_bn"].median()), 1),
                          "volatility": round(float(df.loc[names, "volatility"].mean()), 3) if "volatility" in df else None,
                          "beta": round(float(df.loc[names, "beta"].mean()), 3) if "beta" in df else None}
    out["promoted_vs_demoted_50"] = {"promoted": prof(up), "demoted": prof(down),
                                     "max_places_up": int(delta.max()), "max_places_down": int(-delta.min())}

    # 6. the candidate changes the lab's reading rules allow (at most 3 points per step,
    # plan/factor-lab.md), with each destination for the freed points: what each does to the
    # composite's exposures - Size, and Growth, whose near-neutrality is pinned by
    # tests/test_growth_category.py at 0.05 - and to the top 25.
    grid = []
    for w in (5, 4, 3, 2):
        freed = fw["size"] - w
        for dest in ("pro_rata", "quality", "valuation", "investment", "risk", "momentum"):
            if freed == 0 and dest != "pro_rata":
                continue
            if dest == "pro_rata":
                wts = {**{c: fw[c] * (100 - w) / other_total for c in others}, "size": w}
            else:
                wts = {**fw, "size": w, dest: fw[dest] + freed}
            comp = composite(df, cfg, wts)
            t = top(comp)
            grid.append({"size": w, "freed_to": dest,
                         "size_exposure": round(float(comp.corr(s, method="spearman")), 3),
                         "growth_exposure": round(float(comp.corr(df["growth_score"], method="spearman")), 3),
                         "valuation_exposure": round(float(comp.corr(df["valuation_score"], method="spearman")), 3),
                         "quality_exposure": round(float(comp.corr(df["quality_score"], method="spearman")), 3),
                         "top25_kept": len(set(t) & set(top(base))),
                         "leave": [x for x in top(base) if x not in t], "enter": [x for x in t if x not in top(base)],
                         "spearman_vs_published": round(float(comp.corr(base, method="spearman")), 4)})
    out["candidate_grid"] = grid
    for g in grid:
        print("grid", g)

    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("size_build", "size_vs_other_categories", "neutral_weight",
                                          "exposure_published", "exposure_without_size_signal",
                                          "promoted_vs_demoted_50")}, indent=1, default=float))
    for row in sweep:
        print(row)
    for k, v in vt.items():
        print(k, {x: v[x] for x in ("kept_of_25", "kept_of_50", "spearman_vs_published", "median_abs_rank_change",
                                    "p90_abs_rank_change", "max_abs_rank_change")})
        print("   ", v["profile_top25"])
        print("   exposure", v["composite_exposure"])
    print(json.dumps(out["top25_changes_size_neutral"], indent=1, default=float))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
