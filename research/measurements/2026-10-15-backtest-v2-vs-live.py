"""End to end: v2's inputs at the newest date against the live run's, under the same weights.

backtest_v2.py builds every month from SEC filings (pit_record), the cached daily history
(factor_engine.price_history_fields) and the cached index series. At the date of the newest run
those three can be checked against what the live fetch actually used: score the current
constituents both ways with v2's weights (backtest_v2.v2_config - analyst inputs at weight 0, so
the comparison is like for like) and compare.

Usage: python research/measurements/2026-10-15-backtest-v2-vs-live.py
Writes the JSON beside this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import backtest_v2 as v2  # noqa: E402  (a measurement script, not production)
import pit_record as pr  # noqa: E402

OUT = Path(__file__).with_suffix(".json")


def main():
    cfg = v2.v2_config(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))
    inputs = v2.Inputs.load()
    runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
    raw = pd.read_parquet(runs[-1])
    raw = raw[raw["marketCap"].notna()]
    as_of = pd.Timestamp(str(raw["_ctx_last_date"].dropna().max()))
    mr, rf, state = v2.market_inputs(inputs, as_of, cfg.get("momentum_regime"))

    live = []
    for r in raw.to_dict("records"):
        r = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        for k in pr.FIELD_CLASSES["no_free_history"]:
            r[k] = np.nan                                  # as v2 has them
        live.append(r)
    rebuilt, missing = [], {}
    for t in raw["Ticker"]:
        rec, why = v2.record_at(t, inputs, as_of)
        if rec is None:
            missing.setdefault(why, []).append(t)
        else:
            rebuilt.append(rec)
    # The run artifact does not keep _daily_returns (beta, drawdown and alpha read it), so the live
    # arm takes it from the cached window - whose other fields reproduce the live fetch exactly
    # (tests/test_backtest_v2.py). Without this the live arm silently loses three metrics.
    dr = {r["Ticker"]: r.get("_daily_returns") for r in rebuilt}
    for r in live:
        if not isinstance(r.get("_daily_returns"), dict) and dr.get(r["Ticker"]):
            r["_daily_returns"] = dr[r["Ticker"]]

    a = v2.score_month(live, cfg, mr, rf, state).set_index("Ticker")
    b = v2.score_month(rebuilt, cfg, mr, rf, state).set_index("Ticker")
    common = a.index.intersection(b.index)
    ra, rb = a.loc[common, "Rank"], b.loc[common, "Rank"]
    top = lambda s, n: set(s.nsmallest(n).index)  # noqa: E731
    cats = {}
    for c in cfg["factor_weights"]:
        col = f"{c}_score"
        if col in a and col in b and cfg["factor_weights"][c]:
            x, y = a.loc[common, col], b.loc[common, col]
            ok = x.notna() & y.notna()
            cats[c] = round(float(x[ok].rank().corr(y[ok].rank())), 4)
    out = {"as_of": str(as_of.date()), "run": runs[-1].parent.name, "live_scored": int(len(a)),
           "v2_scored": int(len(b)), "common": int(len(common)), "not_built": {k: v for k, v in missing.items()},
           "composite_spearman": round(float(a.loc[common, "Composite"].rank().corr(b.loc[common, "Composite"].rank())), 4),
           "top25_overlap": len(top(ra, 25) & top(rb, 25)),
           "top_decile_overlap": len(top(ra, len(common) // 10) & top(rb, len(common) // 10)),
           "top_decile_size": len(common) // 10,
           "median_abs_rank_change": float((ra - rb).abs().median()),
           "category_spearman": cats, "regime": state.get("state"), "risk_free": rf}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
