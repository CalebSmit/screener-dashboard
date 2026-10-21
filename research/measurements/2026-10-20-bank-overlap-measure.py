"""How much of bank Quality is bank Valuation, measured on the published run.

The question (NIGHTLY_LOG 2026-10-20, 13:00 entry, Next): bank ROE now equals earnings yield x
P/B on a near-common book, and both of those changed definition on 2026-10-20, so is 35 points
of bank Quality a double count of two Valuation inputs?

This script reads the run the live site publishes (the newest full run in the MAIN checkout;
`runs/` is gitignored) and measures, for the 46 bank-like stocks:

  1. the identities - how exactly ROE = EY x P/B and ROA = ROE x equity ratio hold on the
     engine's own constructions (average vs ending balances, common vs total equity);
  2. cross-sectional correlations between ROE, ROA, earnings yield, P/B, equity ratio and the
     bank Valuation / Quality category scores, raw (Spearman) and as scored (the percentiles);
  3. how much of bank Quality's cross-sectional variance the Valuation inputs explain;
  4. what the bank Valuation + Quality block is net-exposed to, once the identities are used
     to put every ratio on three independent log coordinates (E/P, P/B, B/A);
  5. the same Valuation-Quality correlation for the non-bank universe, for contrast.

Read-only. Nothing here is a return, an IC or a backtest number (CLAUDE.md rules 4-5).

    python research/measurements/2026-10-20-bank-overlap-measure.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().with_suffix(".json")

from scipy.stats import spearmanr  # noqa: E402

import run_context  # noqa: E402

# runs/ is gitignored, so a worktree has none: read the main checkout's, then the local one.
CANDIDATES = [Path(r"C:/Users/smitc/OneDrive/Documents/Screener/runs"), ROOT / "runs"]
RUNS = next(p for p in CANDIDATES if p.is_dir())
RUN = Path(run_context.latest_run_dir(RUNS))


def rnd(v, n=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


raw = pd.read_parquet(RUN / "00_raw_fetch.parquet").set_index("Ticker")
f02 = pd.read_parquet(RUN / "02_outliers_flagged.parquet").set_index("Ticker")
sc = pd.read_parquet(RUN / "05_final_scored.parquet").set_index("Ticker")
bank = sc["_is_bank_like"].fillna(False).astype(bool)
B = sorted(sc.index[bank])
NB = sorted(sc.index[~bank])

out = {
    "run": RUN.name,
    "run_started": json.loads((RUN / "meta.json").read_text())["start_time"],
    "bank_like": len(B),
    "note": "Cross-sectional, one run. Descriptive only - no return, IC or backtest figure.",
}

# The payload the site serves must be this run, or the numbers describe something else.
payload = Path(r"C:/Users/smitc/OneDrive/Documents/Screener/dashboard_data.js")
if payload.is_file():
    head = payload.read_text(encoding="utf-8")[:400]
    out["published_payload_is_this_run"] = out["run_started"].replace(" ", "T") in head

M = ["roe", "roa", "earnings_yield", "pb_ratio", "equity_ratio", "piotroski_f_score", "accruals"]
d = sc.loc[B, M + [f"{m}_pct" for m in M]
           + ["valuation_score", "quality_score", "Composite", "Rank"]].astype(float)
out["coverage"] = {m: f"{int(d[m].notna().sum())} of {len(B)}" for m in M}

# ---- 1. the identities --------------------------------------------------------------------
r = raw.loc[B]
ey_x_pb = d["earnings_yield"] * d["pb_ratio"]          # = NI / ending COMMON equity
gap = d["roe"] / ey_x_pb                               # = ending common / average total equity
er_x_roe = d["roe"] * d["equity_ratio"]                # = NI / avg TE * ending TE / ending TA
gap_a = d["roa"] / er_x_roe
pref_share = 1 - r["commonEquity"].astype(float) / r["totalEquity"].astype(float)
out["1_identities"] = {
    "ROE = EY x P/B": {
        "spearman(roe, ey*pb)": rnd(spearmanr(d["roe"], ey_x_pb).statistic, 4),
        "roe / (ey*pb), median": rnd(gap.median()),
        "roe / (ey*pb), 10th-90th pct": [rnd(gap.quantile(.1)), rnd(gap.quantile(.9))],
        "within 5%": f"{int((gap.sub(1).abs() <= .05).sum())} of {len(B)}",
        "within 10%": f"{int((gap.sub(1).abs() <= .10).sum())} of {len(B)}",
        "largest gaps": {t: rnd(gap[t]) for t in gap.sub(1).abs().sort_values(
            ascending=False).head(5).index},
        "why it is not exact": "ROE's denominator is AVERAGE total equity (incl. preferred); "
                               "P/B's is ENDING common equity. Ratio = ending common / average "
                               "total.",
        "preferred share of equity, median / max": [rnd(pref_share.median()),
                                                    rnd(pref_share.max())],
    },
    "ROA = ROE x equity ratio": {
        "spearman(roa, roe*er)": rnd(spearmanr(d["roa"], er_x_roe).statistic, 4),
        "roa / (roe*er), median": rnd(gap_a.median()),
        "within 5%": f"{int((gap_a.sub(1).abs() <= .05).sum())} of {len(B)}",
    },
}

# ---- 2. correlations ----------------------------------------------------------------------
# Raw Spearman: the economics. Percentile Pearson: what the scorer adds up (P/B's percentile is
# direction-flipped - higher = cheaper - so its sign is the opposite of the raw one).
raw_cols = ["roe", "roa", "earnings_yield", "pb_ratio", "equity_ratio",
            "valuation_score", "quality_score"]
sp = d[raw_cols].corr(method="spearman")
pct_cols = ["roe_pct", "roa_pct", "earnings_yield_pct", "pb_ratio_pct", "equity_ratio_pct",
            "piotroski_f_score_pct", "accruals_pct", "valuation_score", "quality_score"]
pe = d[pct_cols].corr(method="pearson")
out["2_correlations"] = {
    "raw_spearman": {a: {b: rnd(sp.loc[a, b], 3) for b in raw_cols} for a in raw_cols},
    "scored_percentiles_pearson (pb_ratio_pct: higher = cheaper)":
        {a: {b: rnd(pe.loc[a, b], 3) for b in pct_cols} for a in pct_cols},
    "headline": {
        "spearman(roe, pb)": rnd(sp.loc["roe", "pb_ratio"], 3),
        "spearman(roe, earnings_yield)": rnd(sp.loc["roe", "earnings_yield"], 3),
        "spearman(earnings_yield, pb)": rnd(sp.loc["earnings_yield", "pb_ratio"], 3),
        "spearman(valuation_score, quality_score), banks":
            rnd(sp.loc["valuation_score", "quality_score"], 3),
    },
}


# ---- 3. variance of bank Quality explained by the Valuation inputs -----------------------
def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - resid.var() / y.var()), beta


q = d["quality_score"]
r2_val, _ = r2(q, [d["valuation_score"]])
r2_inputs, b_inputs = r2(q, [d["earnings_yield_pct"], d["pb_ratio_pct"]])
r2_roe, b_roe = r2(d["roe_pct"], [d["earnings_yield_pct"], d["pb_ratio_pct"]])
r2_roa, _ = r2(d["roa_pct"], [d["earnings_yield_pct"], d["pb_ratio_pct"]])
lr2, lb = r2(np.log(d["roe"]), [np.log(d["earnings_yield"]), np.log(d["pb_ratio"])])

# The share of each stock's Quality score that the two identity-linked metrics supply
# (all 46 have all five metrics, so the bank table's weights are the effective ones).
w = {"roe": .35, "roa": .25, "equity_ratio": .15, "piotroski_f_score": .15, "accruals": .10}
contrib = pd.DataFrame({m: d[f"{m}_pct"] * wt for m, wt in w.items()})
assert np.allclose(contrib.sum(axis=1), q, atol=1e-6), "bank Quality does not rebuild"
cov_share = {m: rnd(np.cov(contrib[m], q)[0, 1] / q.var(), 3) for m in w}
out["3_variance_explained"] = {
    "R2 quality_score ~ valuation_score": rnd(r2_val, 3),
    "R2 quality_score ~ ey_pct + pb_pct": rnd(r2_inputs, 3),
    "  coefficients (const, ey_pct, pb_pct[cheap=high])": [rnd(x, 3) for x in b_inputs],
    "R2 roe_pct ~ ey_pct + pb_pct": rnd(r2_roe, 3),
    "  coefficients (const, ey_pct, pb_pct[cheap=high])": [rnd(x, 3) for x in b_roe],
    "R2 roa_pct ~ ey_pct + pb_pct": rnd(r2_roa, 3),
    "R2 log(roe) ~ log(ey) + log(pb)": rnd(lr2, 4),
    "  coefficients (const, log ey, log pb)": [rnd(x, 3) for x in lb],
    "bank Quality rebuilds from its five weighted percentiles": "46 of 46",
    "share of bank Quality's variance supplied by each metric (cov(w*pct, Q)/var(Q))": cov_share,
    "note": "ey_pct is ranked among all 76 Financials stocks; roe/roa/pb/er among the 46 "
            "bank-like stocks only (the only ones that carry them).",
}

# ---- 3b. the ranking pools differ: earnings yield is ranked among all 76 Financials -------
# A bank's P/B, ROE, ROA and equity ratio are ranked against the 46 bank-like stocks (no other
# stock carries them); its earnings yield against all 76 Financials, 30 of which are payment
# networks, exchanges, data vendors and asset managers. That changes how far apart the banks'
# EY percentiles sit, which is part of what a weight on them means.
fin = sc[sc["Sector"] == "Financials"]
fin_nb_ey = fin.loc[~fin["_is_bank_like"].fillna(False).astype(bool), "earnings_yield"].astype(float)
out["3b_ranking_pools"] = {
    "sd of percentile among the bank-like (EY pct, P/B pct, ROE pct)":
        [rnd(d["earnings_yield_pct"].std(), 1), rnd(d["pb_ratio_pct"].std(), 1),
         rnd(d["roe_pct"].std(), 1)],
    "bank-like EY pct min / median / max": [rnd(d["earnings_yield_pct"].min(), 1),
                                            rnd(d["earnings_yield_pct"].median(), 1),
                                            rnd(d["earnings_yield_pct"].max(), 1)],
    "median EY, bank-like vs non-bank Financials": [rnd(d["earnings_yield"].median()),
                                                     rnd(fin_nb_ey.median())],
    "reading": "a uniform rank over 46 has sd 29.0; the EY percentiles of the 46 have less "
               "spread because the non-bank Financials sit mostly below them, so EY's 40% "
               "moves a bank's Valuation score less than its weight suggests.",
}

# ---- 4. what the Valuation + Quality block is net-exposed to -----------------------------
# Every bank ratio here is a combination of four company-level amounts: earnings E, market
# value P, book B, assets A. Dividing out scale leaves three independent log coordinates:
#   x1 = log E/P (earnings yield), x2 = log P/B, x3 = log B/A (equity ratio)
# with log ROE ~ x1 + x2 and log ROA ~ x1 + x2 + x3 (exact on ending balances).
# The arithmetic below is the first-order reading of the weights; the regression after it is
# the measured one on the scores the engine actually produced.
fw = {"valuation": 22, "quality": 22}
pts = {"earnings_yield": fw["valuation"] * .40, "pb_ratio": -fw["valuation"] * .60,
       "roe": fw["quality"] * .35, "roa": fw["quality"] * .25,
       "equity_ratio": fw["quality"] * .15}
load = {
    "x1 log E/P": pts["earnings_yield"] + pts["roe"] + pts["roa"],
    "x2 log P/B": pts["pb_ratio"] + pts["roe"] + pts["roa"],
    "x3 log B/A": pts["roa"] + pts["equity_ratio"],
}
block = (fw["valuation"] * d["valuation_score"] + fw["quality"] * d["quality_score"]) / 44.0
X = pd.DataFrame({"x1": np.log(d["earnings_yield"]), "x2": np.log(d["pb_ratio"]),
                  "x3": np.log(d["equity_ratio"])})
Z = (X - X.mean()) / X.std()
R2b, bb = r2(block, [Z.x1, Z.x2, Z.x3])
R2v, bv = r2(d["valuation_score"], [Z.x1, Z.x2, Z.x3])
R2q, bq = r2(d["quality_score"], [Z.x1, Z.x2, Z.x3])
# rank-space version: the percentiles the engine adds are ranks, so regress on ranks too
Rk = X.rank(pct=True) * 100
R2br, bbr = r2(block, [Rk.x1, Rk.x2, Rk.x3])
out["4_net_exposure"] = {
    "first-order composite points on each log coordinate (weights x signs)":
        {k: rnd(v, 2) for k, v in load.items()},
    "reading": "P/B's 13.2 composite points are offset by ROE's 7.7 and ROA's 5.5, which "
               "both rise with P/B; earnings yield's 8.8 are reinforced by them.",
    "measured: block = (22*Val + 22*Qual)/44 on standardised x1, x2, x3": {
        "R2": rnd(R2b, 3),
        "coef per 1 sd (x1 E/P, x2 P/B, x3 B/A)": [rnd(x, 2) for x in bb[1:]],
    },
    "measured: valuation_score alone": {"R2": rnd(R2v, 3),
                                        "coef per 1 sd": [rnd(x, 2) for x in bv[1:]]},
    "measured: quality_score alone": {"R2": rnd(R2q, 3),
                                      "coef per 1 sd": [rnd(x, 2) for x in bq[1:]]},
    "measured: block on percentile ranks of x1, x2, x3": {
        "R2": rnd(R2br, 3), "coef per 1 pct-pt": [rnd(x, 3) for x in bbr[1:]]},
    "spearman(block, P/B)": rnd(spearmanr(block, d["pb_ratio"]).statistic, 3),
    "spearman(block, earnings yield)": rnd(spearmanr(block, d["earnings_yield"]).statistic, 3),
    "spearman(valuation_score, P/B)": rnd(spearmanr(d["valuation_score"],
                                                    d["pb_ratio"]).statistic, 3),
    "sd of x (E/P, P/B, B/A) in logs": [rnd(X[c].std(), 3) for c in X],
    "corr of x (x1x2, x1x3, x2x3)": [rnd(X.x1.corr(X.x2), 3), rnd(X.x1.corr(X.x3), 3),
                                     rnd(X.x2.corr(X.x3), 3)],
}

# ---- 5. contrast: Valuation against Quality outside the bank set -------------------------
nb = sc.loc[NB, ["valuation_score", "quality_score"]].astype(float).dropna()
fin_nb = [t for t in NB if sc.loc[t, "Sector"] == "Financials"]
per_sector = {}
for sec, g in sc.loc[NB].groupby("Sector"):
    g = g[["valuation_score", "quality_score"]].astype(float).dropna()
    if len(g) >= 15:
        per_sector[sec] = [len(g), rnd(spearmanr(g.valuation_score, g.quality_score).statistic, 3)]
out["5_contrast_non_banks"] = {
    "spearman(valuation_score, quality_score), non-bank universe":
        [len(nb), rnd(spearmanr(nb.valuation_score, nb.quality_score).statistic, 3)],
    "same, banks": [len(B), rnd(spearmanr(d.valuation_score, d.quality_score).statistic, 3)],
    "by sector (n, rho), sectors with 15+ non-bank stocks": per_sector,
    "non-bank Financials (n)": len(fin_nb),
}

# ---- 6. the bank set by sub-type (insurers carry most of the top) -------------------------
ind = raw.loc[B, "industry"] if "industry" in raw.columns else pd.Series("", index=B)
out["6_bank_like_by_industry"] = {k: int(v) for k, v in ind.fillna("?").value_counts().items()}
top = d.sort_values("Rank").head(10)
out["6_best_ranked_bank_like"] = {
    t: {"rank": int(top.loc[t, "Rank"]), "industry": str(ind.get(t, "")),
        "roe": rnd(top.loc[t, "roe"]), "ey": rnd(top.loc[t, "earnings_yield"]),
        "pb": rnd(top.loc[t, "pb_ratio"], 2), "val": rnd(top.loc[t, "valuation_score"], 1),
        "qual": rnd(top.loc[t, "quality_score"], 1)}
    for t in top.index}

# ---- 7. the third coordinate (B/A) is mostly WHICH KIND of financial --------------------
# The 46 are ranked as one peer group, but an insurer carries several times a bank's equity per
# dollar of assets. How much of each coordinate, and of the Valuation + Quality block, is just
# the sub-type?
def group(s: str) -> str:
    s = (s or "").lower()
    if "insurance" in s:
        return "insurer"
    if s.startswith("banks"):
        return "bank"
    return "broker / asset manager / credit"


g = ind.reindex(B).fillna("").map(group)
dummies = pd.get_dummies(g, drop_first=True).astype(float)
by_group = {}
for name, idx in g.groupby(g).groups.items():
    by_group[name] = {
        "n": len(idx),
        "median equity ratio": rnd(d.loc[idx, "equity_ratio"].median(), 3),
        "median ROA": rnd(d.loc[idx, "roa"].median(), 4),
        "median ROE": rnd(d.loc[idx, "roe"].median(), 3),
        "median P/B": rnd(d.loc[idx, "pb_ratio"].median(), 2),
        "median EY": rnd(d.loc[idx, "earnings_yield"].median(), 3),
        "mean equity_ratio pct": rnd(d.loc[idx, "equity_ratio_pct"].mean(), 1),
        "mean roa pct": rnd(d.loc[idx, "roa_pct"].mean(), 1),
        "mean quality_score": rnd(d.loc[idx, "quality_score"].mean(), 1),
        "mean valuation_score": rnd(d.loc[idx, "valuation_score"].mean(), 1),
        "in the best 10 bank-like": int(sum(t in set(top.index) for t in idx)),
    }
out["7_sub_type"] = {
    "groups": by_group,
    "R2 on sub-type dummies": {
        "log B/A (equity ratio)": rnd(r2(X.x3, [dummies[c] for c in dummies])[0], 3),
        "log E/P": rnd(r2(X.x1, [dummies[c] for c in dummies])[0], 3),
        "log P/B": rnd(r2(X.x2, [dummies[c] for c in dummies])[0], 3),
        "quality_score": rnd(r2(d["quality_score"], [dummies[c] for c in dummies])[0], 3),
        "Val+Qual block": rnd(r2(block, [dummies[c] for c in dummies])[0], 3),
    },
}

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
print("\nwrote", OUT)
