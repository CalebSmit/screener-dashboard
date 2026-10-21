"""The bank ranking with the ROE / P/B overlap removed, under each candidate construction.

Companion to `2026-10-20-bank-overlap-measure.py`. Takes the published run's own metric table
(`02_outliers_flagged.parquet`), its own config and effective weights, and scores it with the
engine's functions - `factor_engine.compute_sector_percentiles`, `apply_percentile_transform`,
`compute_category_scores`, `compute_composite`, `rank_stocks` - changing only the bank table's
weights (and, for one candidate, the input in the pb_ratio column). Never a re-implementation
of the scorer. The baseline must reproduce all 501 published ranks or the script stops.

Candidates (bank-like stocks only; every non-bank composite must come out unchanged):
  S0  status quo: Valuation EY 40 / P/B 60; Quality ROE 35 / ROA 25 / ER 15 / F 15 / acc 10
  C1  drop ROE (its 35 points renormalised over the rest of bank Quality)
  C2  drop P/B (Valuation = earnings yield alone)
  C3  ROE's 35 points moved to ROA (ROA 60)
  C4  drop ROE and ROA (Quality = equity ratio, Piotroski, accruals)
  C5  "P/B relative to ROE": P/B replaced by the residual of log P/B on log ROE across the
      bank set (the Wilcox 1984 P/B-ROE line, lower = cheaper than its ROE justifies);
      ROE stays in Quality
  C6  C5 with ROE dropped from Quality (the combined signal stated once, in Valuation)

For each: the bank ranking's movement, the top 25, and what the Valuation + Quality block is
net-exposed to on the three independent log coordinates (E/P, P/B, B/A).

Read-only. No return, IC or backtest figure (CLAUDE.md rules 4-5).

    python research/measurements/2026-10-20-bank-overlap-rescore.py
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

from scipy.stats import spearmanr  # noqa: E402

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
WATCH = ["C", "JPM", "PGR", "AIG", "AMP", "EG", "ALL"]
NB = sorted(IS_BANK.index[~IS_BANK])


def score(bank_weights: dict | None = None, pb_override: dict | None = None) -> pd.DataFrame:
    """The engine's own scoring path on this run's metric table."""
    cfg = copy.deepcopy(RUN_CFG)
    if bank_weights:
        for cat, ws in bank_weights.items():
            cfg["bank_metric_weights"][cat].update(ws)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = f02.copy()
        if pb_override is not None:
            bank = df["_is_bank_like"].fillna(False).astype(bool)
            df.loc[bank, "pb_ratio"] = df.loc[bank, "Ticker"].map(pb_override)
        df = fe.compute_sector_percentiles(df)
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

# The Wilcox line: log P/B on log ROE across the bank set, this run.
lr, lp = np.log(m["roe"].astype(float)), np.log(m["pb_ratio"].astype(float))
slope, icpt = np.polyfit(lr, lp, 1)
resid = lp - (icpt + slope * lr)
wilcox = {t: float(np.exp(resid[t])) for t in BK}         # >0, lower = cheaper than ROE implies
fit_r2 = float(np.corrcoef(lr, lp)[0, 1] ** 2)


def exposure(s: pd.DataFrame) -> dict:
    blk = (22 * s.loc[BK, "valuation_score"] + 22 * s.loc[BK, "quality_score"]) / 44.0
    A = np.column_stack([np.ones(len(BK)), RK.x1, RK.x2, RK.x3])
    beta, *_ = np.linalg.lstsq(A, blk.values, rcond=None)
    res = blk.values - A @ beta
    return {
        "block on ranks: coef per pct-pt (E/P, P/B, B/A)": [rnd(b, 3) for b in beta[1:]],
        "R2": rnd(1 - res.var() / blk.var(), 3),
        "spearman(block, P/B)": rnd(spearmanr(blk, m["pb_ratio"]).statistic, 3),
        "spearman(block, earnings yield)": rnd(spearmanr(blk, m["earnings_yield"]).statistic, 3),
        "spearman(block, Wilcox residual)": rnd(spearmanr(blk, resid).statistic, 3),
        "spearman(valuation, quality), banks":
            rnd(spearmanr(s.loc[BK, "valuation_score"], s.loc[BK, "quality_score"]).statistic, 3),
    }


def compare(name: str, new: pd.DataFrame, what: str) -> dict:
    bank_rank_old = base.loc[BK, "Composite"].rank(ascending=False)
    bank_rank_new = new.loc[BK, "Composite"].rank(ascending=False)
    mv = (new.loc[BK, "Rank"] - base.loc[BK, "Rank"]).astype(int)
    t_old, t_new = set(base.nsmallest(25, "Rank").index), set(new.nsmallest(25, "Rank").index)
    bk_top = lambda s: [t for t in s.sort_values("Rank").index if t in BK][:10]  # noqa: E731
    return {
        "what": what,
        "non-bank composites changed": int((new.loc[NB, "Composite"].round(9)
                                            != base.loc[NB, "Composite"].round(9)).sum()),
        "universe composite spearman": rnd(spearmanr(base["Composite"],
                                                     new.loc[base.index, "Composite"]).statistic, 5),
        "bank-set rank spearman (within the 46)": rnd(spearmanr(bank_rank_old,
                                                                bank_rank_new).statistic, 4),
        "bank-like median abs universe-rank move": float(mv.abs().median()),
        "moved 10+": int((mv.abs() >= 10).sum()), "moved 25+": int((mv.abs() >= 25).sum()),
        "top 25 lost": sorted(t_old - t_new), "top 25 gained": sorted(t_new - t_old),
        "bank-like in top 25": int(sum(t in BK for t in t_new)),
        "bank-like in top 100": int(sum(t in BK for t in new.nsmallest(100, "Rank").index)),
        "best 10 bank-like": {t: int(new.loc[t, "Rank"]) for t in bk_top(new)},
        "largest rank moves": {t: [int(base.loc[t, "Rank"]), int(new.loc[t, "Rank"])]
                               for t in mv.abs().sort_values(ascending=False).head(8).index},
        "net exposure": exposure(new),
        "teaching examples (rank before, after)": {t: [int(base.loc[t, "Rank"]),
                                                        int(new.loc[t, "Rank"])] for t in WATCH},
    }


cands = {
    "C1_drop_roe": ({"quality": {"roe": 0}}, None,
                    "ROE 0; ROA 25 / ER 15 / F 15 / acc 10 renormalised over 65"),
    "C2_drop_pb": ({"valuation": {"pb_ratio": 0}}, None, "Valuation = earnings yield alone"),
    "C3_roe_to_roa": ({"quality": {"roe": 0, "roa": 60}}, None, "ROE 0, ROA 60"),
    "C4_drop_roe_roa": ({"quality": {"roe": 0, "roa": 0}}, None,
                        "Quality = ER 15 / F 15 / acc 10 renormalised over 40"),
    "C5_pb_relative_to_roe": (None, wilcox,
                              "pb_ratio column = exp(residual of log P/B on log ROE); ROE kept"),
    "C6_pb_relative_to_roe_drop_roe": ({"quality": {"roe": 0}}, wilcox,
                                       "C5 with ROE 0 in Quality"),
}

out = {
    "run": RUN.name,
    "run_started": json.loads((RUN / "meta.json").read_text())["start_time"],
    "baseline reproduces the published ranking": f"{reproduced} of {len(PUB)}",
    "bank_like": len(BK),
    "wilcox line (log P/B = a + b log ROE), this run": {
        "slope b": rnd(slope, 3), "intercept a": rnd(icpt, 3), "R2": rnd(fit_r2, 3),
        "spearman(residual, earnings yield)": rnd(spearmanr(resid, m["earnings_yield"]).statistic, 3),
    },
    "S0_status_quo": {
        "bank-like in top 25": int(sum(t in BK for t in base.nsmallest(25, "Rank").index)),
        "bank-like in top 100": int(sum(t in BK for t in base.nsmallest(100, "Rank").index)),
        "best 10 bank-like": {t: int(base.loc[t, "Rank"]) for t in
                              [t for t in base.sort_values("Rank").index if t in BK][:10]},
        "net exposure": exposure(base),
    },
}
for name, (bw, pbo, what) in cands.items():
    out[name] = compare(name, score(bw, pbo), what)

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
print("\nwrote", OUT)
