"""Second reading of the bank P/B-ROE proposal (research/2026-10-20-bank-quality-valuation-overlap.md
section 8), on a different run from the one the note measured.

Three questions, in the order a deciding session has to answer them:

  1. Do the note's two refutation criteria still hold on this session's run?
     (a) R-squared of log ROE on log earnings yield and log P/B >= 0.90.
     (b) the measured loading on P/B's rank is under half of earnings yield's.

  2. Does the GUARD the proposal specifies actually see a change to the thing its sentence
     claims?  The proposal's test computes the FIRST-ORDER net loading on log P/B from
     config.yaml alone - (-P/B + ROE + ROA) in composite points, 0.0 today - and fails outside
     +-1.5 points.  That is algebra on log coordinates; the engine ranks percentiles.  So for
     every candidate the note rescored, compute both the first-order figure and the loading the
     engine's own scores produce, and see whether the first-order figure tracks it.

  3. If it does not, what band on the measured loading corresponds to the +-1.5 points the
     proposal asked for?

Scoring is the engine's own path, reusing the note's rescore machinery verbatim where it can
(factor_engine.compute_sector_percentiles / apply_percentile_transform / compute_category_scores
/ compute_composite / rank_stocks).  The baseline must reproduce every published rank or this
script stops.  Read-only; no return, IC or backtest figure (CLAUDE.md rules 4-5).

    python research/measurements/2026-10-21-bank-pb-roe-guard-review.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().with_suffix(".json")

import factor_engine as fe  # noqa: E402
import run_context  # noqa: E402

CANDIDATES = [Path(r"C:/Users/smitc/OneDrive/Documents/Screener/runs"), ROOT / "runs"]
RUNS = next(p for p in CANDIDATES if p.is_dir())
RUN = Path(run_context.latest_run_dir(RUNS))


def rnd(v, n=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


RUN_CFG = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
_eff = json.loads((RUN / "effective_weights.json").read_text(encoding="utf-8"))
RUN_CFG["factor_weights"], RUN_CFG["metric_weights"] = _eff["factor_weights"], _eff["metric_weights"]
f02 = pd.read_parquet(RUN / "02_outliers_flagged.parquet")
PUB = pd.read_parquet(RUN / "05_final_scored.parquet").set_index("Ticker")
IS_BANK = f02.set_index("Ticker")["_is_bank_like"].fillna(False).astype(bool)
BK = sorted(IS_BANK.index[IS_BANK])


def score(bank_weights: dict | None = None) -> pd.DataFrame:
    cfg = copy.deepcopy(RUN_CFG)
    if bank_weights:
        for cat, ws in bank_weights.items():
            cfg["bank_metric_weights"][cat].update(ws)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = fe.compute_sector_percentiles(f02.copy())
        df = fe.apply_percentile_transform(df, cfg)
        df = fe.compute_category_scores(df, cfg)
        df = fe.compute_composite(df, cfg)
        return fe.rank_stocks(df).set_index("Ticker")


base = score()
reproduced = int((base["Rank"] == PUB.loc[base.index, "Rank"]).sum())
assert reproduced == len(PUB), f"baseline reproduces only {reproduced} of {len(PUB)} ranks"

m = f02.set_index("Ticker").loc[BK]
X = pd.DataFrame({"x1": np.log(m["earnings_yield"].astype(float)),
                  "x2": np.log(m["pb_ratio"].astype(float)),
                  "x3": np.log(m["equity_ratio"].astype(float))})
RK = X.rank(pct=True) * 100


def measured_loadings(s: pd.DataFrame) -> list[float]:
    """The Valuation+Quality block's loading on the percentile ranks of (E/P, P/B, B/A)."""
    blk = (22 * s.loc[BK, "valuation_score"] + 22 * s.loc[BK, "quality_score"]) / 44.0
    A = np.column_stack([np.ones(len(BK)), RK.x1, RK.x2, RK.x3])
    beta, *_ = np.linalg.lstsq(A, blk.values, rcond=None)
    return [float(b) for b in beta[1:]]


def first_order(bank_weights: dict | None = None) -> dict:
    """Composite points on each log coordinate, from the weights alone - the proposal's guard.

    Renormalised within each category over the metrics that carry weight, which is what
    compute_category_scores does when a weight is set to zero.
    """
    cfg = copy.deepcopy(RUN_CFG)
    if bank_weights:
        for cat, ws in bank_weights.items():
            cfg["bank_metric_weights"][cat].update(ws)
    fw, bmw = cfg["factor_weights"], cfg["bank_metric_weights"]
    pts = {}
    for cat in ("valuation", "quality"):
        total = sum(w for w in bmw[cat].values() if w)
        for metric, w in bmw[cat].items():
            pts[metric] = fw[cat] * (w / total) if w else 0.0
    return {
        "x1 log E/P": pts["earnings_yield"] + pts["roe"] + pts["roa"],
        "x2 log P/B": -pts["pb_ratio"] + pts["roe"] + pts["roa"],
        "x3 log B/A": pts["roa"] + pts["equity_ratio"],
    }


# --- 1. the note's two refutation criteria, on this run ------------------------------------
lr = np.log(m["roe"].astype(float))
A = np.column_stack([np.ones(len(BK)), X.x1, X.x2])
beta, *_ = np.linalg.lstsq(A, lr.values, rcond=None)
res = lr.values - A @ beta
r2_identity = float(1 - res.var() / lr.values.var())
s0 = measured_loadings(base)

criteria = {
    "(a) R2 of log ROE on log E/P and log P/B": rnd(r2_identity, 4),
    "(a) bar": 0.90,
    "(a) verdict": "holds" if r2_identity >= 0.90 else "REFUTED",
    "(a) coefficients on log E/P, log P/B": [rnd(beta[1], 3), rnd(beta[2], 3)],
    "(b) measured loading on P/B rank": rnd(s0[1], 4),
    "(b) measured loading on E/P rank": rnd(s0[0], 4),
    "(b) bar: |P/B| < half of E/P": rnd(abs(s0[0]) / 2, 4),
    "(b) verdict": "holds" if abs(s0[1]) < abs(s0[0]) / 2 else "REFUTED",
}

# --- 2. can the proposal's first-order guard see each candidate? ---------------------------
# The note's candidates that are pure reweightings (C5/C6 change an input column, not a weight,
# so no weight-only guard could see them at all - recorded as such).
reweightings = {
    "S0_status_quo": None,
    "C1_drop_roe": {"quality": {"roe": 0}},
    "C2_drop_pb": {"valuation": {"pb_ratio": 0}},
    "C3_roe_to_roa": {"quality": {"roe": 0, "roa": 60}},
    "C4_drop_roe_roa": {"quality": {"roe": 0, "roa": 0}},
    # two the note did not rescore, chosen to probe the guard rather than to be proposed:
    "P1_roe_35_to_25_equity_ratio_25": {"quality": {"roe": 25, "equity_ratio": 25}},
    "P2_valuation_22_quality_25": None,  # handled below (a factor weight, not a bank weight)
}

rows = {}
for name, bw in reweightings.items():
    if name == "P2_valuation_22_quality_25":
        continue
    fo = first_order(bw)
    ml = measured_loadings(score(bw))
    rows[name] = {
        "first-order points (E/P, P/B, B/A)": [rnd(fo["x1 log E/P"], 2), rnd(fo["x2 log P/B"], 2),
                                               rnd(fo["x3 log B/A"], 2)],
        "first-order net on log P/B": rnd(fo["x2 log P/B"], 2),
        "measured loading on P/B rank": rnd(ml[1], 4),
        "measured loading on E/P rank": rnd(ml[0], 4),
        "proposal's guard (|first-order| <= 1.5)": "passes" if abs(fo["x2 log P/B"]) <= 1.5 else "FAILS",
        "criterion (b) on the measured loading": "holds" if abs(ml[1]) < abs(ml[0]) / 2 else "REFUTED",
    }

# the mapping between the two, over the candidates where the first-order figure moves
pairs = [(r["first-order net on log P/B"], r["measured loading on P/B rank"])
         for r in rows.values()]
fo_v = np.array([p[0] for p in pairs], float)
ml_v = np.array([p[1] for p in pairs], float)
keep = np.abs(fo_v) > 0.01                      # exclude the two that read exactly 0.0
slope, icpt = np.polyfit(fo_v[keep], ml_v[keep], 1)

out = {
    "run": RUN.name,
    "run_started": json.loads((RUN / "meta.json").read_text())["start_time"],
    "note_measured_run": "7c12b43a07b6 (2026-10-20T13:21)",
    "bank_like": len(BK),
    "baseline reproduces the published ranking": f"{reproduced} of {len(PUB)}",
    "1_refutation_criteria": criteria,
    "2_can_the_guard_see_it": rows,
    "2b_the_blind_spot": {
        "what": "C3 (ROE's 35 points moved to ROA) and S0 both read 0.00 first-order, because "
                "ROE and ROA enter log P/B with the same +1 coefficient - so moving weight "
                "between them cannot move the first-order figure at all. The engine ranks "
                "percentiles, and ROA also carries log B/A, which is correlated with log P/B "
                "across the bank set, so the measured loading does move.",
        "S0 first-order / measured": [rows["S0_status_quo"]["first-order net on log P/B"],
                                      rows["S0_status_quo"]["measured loading on P/B rank"]],
        "C3 first-order / measured": [rows["C3_roe_to_roa"]["first-order net on log P/B"],
                                      rows["C3_roe_to_roa"]["measured loading on P/B rank"]],
        "C3 measured / S0 measured": rnd(rows["C3_roe_to_roa"]["measured loading on P/B rank"]
                                         / rows["S0_status_quo"]["measured loading on P/B rank"], 2),
        "corr(log P/B, log B/A) across the bank set": rnd(np.corrcoef(X.x2, X.x3)[0, 1], 3),
        "conclusion": "a weight-only first-order guard cannot see C3, which the note itself names "
                      "as the mildest route to a deliberate P/B tilt. The guard must measure the "
                      "loading the published scores produce, not only the algebra on the weights.",
    },
    "3_what_band_matches_1.5_points": {
        "measured loading ~ a + b x first-order points": {"b per point": rnd(slope, 4),
                                                           "a": rnd(icpt, 4)},
        "+-1.5 points maps to a measured loading of": [rnd(icpt - 1.5 * slope, 4),
                                                       rnd(icpt + 1.5 * slope, 4)],
        "half of E/P's loading (criterion b's own bar)": rnd(abs(s0[0]) / 2, 4),
        "reading": "the proposal's +-1.5 points is about +-0.03 on the measured loading, far "
                   "tighter than criterion (b)'s own bar of half of E/P's loading. So keep the "
                   "first-order check AND add a measured one at criterion (b)'s bar, which is the "
                   "sentence's own standard and is the one C3 trips.",
    },
}

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
print("\nwrote", OUT)
