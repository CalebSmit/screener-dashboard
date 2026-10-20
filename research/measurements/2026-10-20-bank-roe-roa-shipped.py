"""What the bank ROE/ROA change did, on the run that published it.

The review (`2026-10-20-bank-roe-roa-review.py`) measured the stakes by rescoring the 06:00
run's metric table. This measures the two runs as published - 06:00 (the source's ratios)
against 13:40 (net income over average balances) - which is the figure the changelog quotes.

CAREFUL - the two runs are NOT a clean comparison, and the first version of this script
implied they were. They are 7.5 hours apart and the 13:40 run refetched every price, so
momentum, risk and the valuation ratios moved for the whole universe: 434 of 455 non-bank
composites changed and the top 25 gained two names and lost two. **None of that is this
change.** Section 3 is therefore reported as "the two published runs, drift included", and
section 5 is the attribution the changelog quotes: the 13:40 run's own metric table rescored
with the OLD construction, so the only thing that differs is the ROE/ROA definition.

    python research/measurements/2026-10-20-bank-roe-roa-shipped.py
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

BEFORE, AFTER = "a35b687ccf19", None          # AFTER = the newest run


def rnd(v, n=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


import run_context  # noqa: E402

after = Path(run_context.latest_run_dir())
before = ROOT / "runs" / BEFORE
assert after.name != BEFORE, "the newest run is still the one that used the source's ratios"

A = pd.read_parquet(after / "05_final_scored.parquet").set_index("Ticker")
B = pd.read_parquet(before / "05_final_scored.parquet").set_index("Ticker")
rawA = pd.read_parquet(after / "00_raw_fetch.parquet").set_index("Ticker")
f02A = pd.read_parquet(after / "02_outliers_flagged.parquet").set_index("Ticker")
f02B = pd.read_parquet(before / "02_outliers_flagged.parquet").set_index("Ticker")

banks = sorted(f02A.index[f02A["_is_bank_like"].fillna(False).astype(bool)])
both = [t for t in A.index if t in B.index]
bk = [t for t in banks if t in both]
nb = [t for t in both if t not in bk]

out = {
    "before_run": BEFORE, "after_run": after.name,
    "before_started": json.loads((before / "meta.json").read_text())["start_time"],
    "after_started": json.loads((after / "meta.json").read_text())["start_time"],
    "universe": [len(B), len(A)], "compared": len(both), "bank_like": len(bk),
}

# ---- 1. every bank-like stock keeps both metrics -------------------------------------------
out["1_coverage"] = {
    "roe scored, after": f'{int(f02A.loc[bk, "roe"].notna().sum())} of {len(bk)}',
    "roa scored, after": f'{int(f02A.loc[bk, "roa"].notna().sum())} of {len(bk)}',
    "roe scored, before": f'{int(f02B.loc[bk, "roe"].notna().sum())} of {len(bk)}',
    "roa scored, before": f'{int(f02B.loc[bk, "roa"].notna().sum())} of {len(bk)}',
    "lost roe": sorted(t for t in bk if pd.isna(f02A.loc[t, "roe"])
                       and pd.notna(f02B.loc[t, "roe"])),
    "lost roa": sorted(t for t in bk if pd.isna(f02A.loc[t, "roa"])
                       and pd.notna(f02B.loc[t, "roa"])),
    "gained roe": sorted(t for t in bk if pd.notna(f02A.loc[t, "roe"])
                         and pd.isna(f02B.loc[t, "roe"])),
}

# ---- 2. each stock's figures rebuild from its two published inputs --------------------------
rebuilt = {"roe": 0, "roa": 0}
worst = {"roe": (None, 0.0), "roa": (None, 0.0)}
for t in bk:
    for m, (now, prior) in (("roe", ("totalEquity", "totalEquity_prior")),
                            ("roa", ("totalAssets", "totalAssets_prior"))):
        ni = float(rawA.loc[t, "netIncome"])
        a, p = float(rawA.loc[t, now]), float(rawA.loc[t, prior])
        want, got = ni / ((a + p) / 2), float(f02A.loc[t, m])
        err = abs(got / want - 1) if want else None
        if err is not None and err <= 1e-9:
            rebuilt[m] += 1
        if err is not None and err > worst[m][1]:
            worst[m] = (t, err)
out["2_rebuilds_from_its_published_inputs"] = {
    "roe": f'{rebuilt["roe"]} of {len(bk)}', "roa": f'{rebuilt["roa"]} of {len(bk)}',
    "worst relative error": {m: [w[0], rnd(w[1], 12)] for m, w in worst.items()},
}

# ---- 3. the two published runs, DRIFT INCLUDED (not the attribution - see section 5) -------
mv = (A.loc[bk, "Rank"] - B.loc[bk, "Rank"]).astype(int)
ta, tb = set(B.nsmallest(25, "Rank").index), set(A.nsmallest(25, "Rank").index)
out["3_two_published_runs_drift_included"] = {
    "WARNING": "7.5 hours and a full refetch apart; the non-bank and top-25 movement below is "
               "market drift, NOT this change. Section 5 isolates the change.",
    "universe composite spearman": rnd(spearmanr(B.loc[both, "Composite"],
                                                 A.loc[both, "Composite"]).statistic, 5),
    "bank_like median abs rank move": float(mv.abs().median()),
    "moved 10+": int((mv.abs() >= 10).sum()), "moved 25+": int((mv.abs() >= 25).sum()),
    "top 25 lost": sorted(ta - tb), "top 25 gained": sorted(tb - ta),
    "non-bank composites changed": int((A.loc[nb, "Composite"].round(6)
                                        != B.loc[nb, "Composite"].round(6)).sum()),
    "largest rank moves": {t: [int(B.loc[t, "Rank"]), int(A.loc[t, "Rank"])]
                           for t in mv.abs().sort_values(ascending=False).head(10).index},
    "largest ratio changes": {
        t: {"roe": [rnd(f02B.loc[t, "roe"]), rnd(f02A.loc[t, "roe"])],
            "roa": [rnd(f02B.loc[t, "roa"], 5), rnd(f02A.loc[t, "roa"], 5)],
            "rank": [int(B.loc[t, "Rank"]), int(A.loc[t, "Rank"])]}
        for t in (f02A.loc[bk, "roe"].fillna(0) - f02B.loc[bk, "roe"].fillna(0)).abs()
                  .sort_values(ascending=False).head(6).index},
    # the two that were on the ending-balance fallback before
    "the mixed-basis pair": {
        t: {"roe": [rnd(f02B.loc[t, "roe"]), rnd(f02A.loc[t, "roe"])],
            "roa": [rnd(f02B.loc[t, "roa"], 5), rnd(f02A.loc[t, "roa"], 5)],
            "rank": [int(B.loc[t, "Rank"]), int(A.loc[t, "Rank"])]}
        for t in ("MS", "WRB") if t in both},
    "IBKR": {"roa": [rnd(f02B.loc["IBKR", "roa"], 5), rnd(f02A.loc["IBKR", "roa"], 5)],
             "roe": [rnd(f02B.loc["IBKR", "roe"]), rnd(f02A.loc["IBKR", "roe"])],
             "rank": [int(B.loc["IBKR", "Rank"]), int(A.loc["IBKR", "Rank"])]}
    if "IBKR" in both else None,
}

# ---- 4. one ROE, not two -------------------------------------------------------------------
# sustainable_growth's ROE and the bank metric are now the same arithmetic; on a bank-like
# stock with positive net income and a payout, the two must agree.
agree = []
for t in bk:
    ni = float(rawA.loc[t, "netIncome"])
    eq, p = float(rawA.loc[t, "totalEquity"]), float(rawA.loc[t, "totalEquity_prior"])
    if not (np.isfinite(ni) and ni > 0 and eq > 0 and p > 0):
        continue
    sg_roe = ni / ((eq + p) / 2)
    agree.append(abs(float(f02A.loc[t, "roe"]) - sg_roe) <= 1e-12)
out["4_one_roe_not_two"] = {
    "bank-like stocks where the scored ROE equals sustainable growth's ROE":
        f"{sum(agree)} of {len(agree)}",
}

# ---- 5. THE ATTRIBUTION: the published run rescored with the OLD construction ---------------
# Same metric table, same prices, same everything - only roe and roa differ. This is what the
# changelog quotes as the effect of the change.
import contextlib  # noqa: E402
import io  # noqa: E402

import yaml  # noqa: E402

import factor_engine as fe  # noqa: E402

RUN_CFG = yaml.safe_load((after / "config.yaml").read_text(encoding="utf-8"))
_eff = json.loads((after / "effective_weights.json").read_text(encoding="utf-8"))
RUN_CFG["factor_weights"], RUN_CFG["metric_weights"] = _eff["factor_weights"], _eff["metric_weights"]
f02_flat = pd.read_parquet(after / "02_outliers_flagged.parquet")


def score(roe=None, roa=None):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = f02_flat.copy()
        bank = df["_is_bank_like"].fillna(False).astype(bool)
        for col, mapping in (("roe", roe), ("roa", roa)):
            if mapping is not None:
                v = df["Ticker"].map(mapping)
                df.loc[bank, col] = v[bank]
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, RUN_CFG)
        df = fe.compute_category_scores(df, RUN_CFG)
        df = fe.compute_composite(df, RUN_CFG)
        return fe.rank_stocks(df).set_index("Ticker")


new = score()
# the OLD construction, rebuilt from the fetch this run actually made: the source's ratio when
# it serves one, otherwise net income over the ENDING balance (the fallback MS and WRB hit).
old_roe, old_roa = {}, {}
for t in bk:
    ni = float(rawA.loc[t, "netIncome"])
    y_e, y_a = rawA.loc[t, "returnOnEquity"], rawA.loc[t, "returnOnAssets"]
    eq, ta = float(rawA.loc[t, "totalEquity"]), float(rawA.loc[t, "totalAssets"])
    old_roe[t] = float(y_e) if pd.notna(y_e) else (ni / eq if eq > 0 else np.nan)
    old_roa[t] = float(y_a) if pd.notna(y_a) else (ni / ta if ta > 0 else np.nan)
old = score(old_roe, old_roa)

mv5 = (new.loc[bk, "Rank"] - old.loc[bk, "Rank"]).astype(int)
t_old, t_new = set(old.nsmallest(25, "Rank").index), set(new.nsmallest(25, "Rank").index)
out["5_attribution_same_run_old_vs_new_construction"] = {
    "note": "the published 13:40 table, scored both ways - the only difference is roe/roa",
    "new reproduces the published ranking":
        f'{int((new["Rank"] == A.loc[new.index, "Rank"]).sum())} of {len(new)}',
    "universe composite spearman": rnd(spearmanr(old["Composite"],
                                                 new.loc[old.index, "Composite"]).statistic, 5),
    "bank_like median abs rank move": float(mv5.abs().median()),
    "moved 10+": int((mv5.abs() >= 10).sum()), "moved 25+": int((mv5.abs() >= 25).sum()),
    "top 25 lost": sorted(t_old - t_new), "top 25 gained": sorted(t_new - t_old),
    "non-bank composites changed": int((new.loc[nb, "Composite"].round(9)
                                        != old.loc[nb, "Composite"].round(9)).sum()),
    "largest rank moves": {t: [int(old.loc[t, "Rank"]), int(new.loc[t, "Rank"])]
                           for t in mv5.abs().sort_values(ascending=False).head(10).index},
    "largest composite moves": {
        t: [rnd(old.loc[t, "Composite"], 2), rnd(new.loc[t, "Composite"], 2)]
        for t in (new.loc[bk, "Composite"] - old.loc[bk, "Composite"]).abs()
                  .sort_values(ascending=False).head(6).index},
}

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
print("\nwrote", OUT)
