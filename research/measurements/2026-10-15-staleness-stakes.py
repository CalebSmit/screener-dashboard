"""What a quarter of staleness costs the stocks the source is behind on.

32 stocks on the run of 2026-10-15 were scored on statements a quarter older than the company's
newest filing (2026-10-15-statement-lag.py). How much would a fresh quarter move them? To keep the
source constant, each lagging stock's statement and market fields are rebuilt from its SEC filings
twice (pit_record): as they stood just after the quarter the run used was filed ("stale") and as of
the run date ("fresh"). Everyone else keeps the run's own record. The universe is scored by the
engine both ways; the difference is the quarter's.

Usage: python research/measurements/2026-10-15-staleness-stakes.py   (JSON beside it)
"""
import contextlib
import glob
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import factor_engine as fe  # noqa: E402
import pit_record as pr  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
lag = json.loads((ROOT / "research" / "measurements" / "2026-10-15-statement-lag.json").read_text(encoding="utf-8"))
facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
raw = pd.read_parquet(sorted(glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet")), key=os.path.getmtime)[-1])
base = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in raw.to_dict("records")]
run = pd.Timestamp(lag["run_date"])
stale_at = {}
f10 = facts[facts["form"].isin(("10-K", "10-Q"))]
for s in lag["stocks"]:
    t, used = s["ticker"], pd.Timestamp(s["used"])
    # the day after the used quarter's own filing - what was known while the source still served it
    g = f10[(f10["ticker"] == t) & ((f10["end"] - used).abs() <= pd.Timedelta(days=10))]
    if not g.empty:
        stale_at[t] = g["filed"].min() + pd.Timedelta(days=1)


def arm(when_for):
    out = []
    for r in base:
        t = r["Ticker"]
        if t in when_for and r.get("currentPrice"):
            out.append(pr.record_as_of(r, facts[facts["ticker"] == t], when_for[t], r["currentPrice"]))
        else:
            out.append(r)
    return out


def score(rs):
    with contextlib.redirect_stdout(io.StringIO()):
        df = fe.compute_metrics(rs, pd.Series(dtype=float), cfg, risk_free_rate=0.04)
        df = df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, cfg)
        df = fe.compute_category_scores(df, cfg)
        df = fe.compute_composite(df, cfg)
        return fe.rank_stocks(df).set_index("Ticker")


a = score(arm(stale_at))
b = score(arm({t: run for t in stale_at}))
rows = []
for t in stale_at:
    if t in a.index and t in b.index:
        rows.append({"ticker": t, "rank_stale": int(a.loc[t, "Rank"]), "rank_fresh": int(b.loc[t, "Rank"]),
                     "move": int(a.loc[t, "Rank"] - b.loc[t, "Rank"]),
                     "composite_change": round(float(b.loc[t, "Composite"] - a.loc[t, "Composite"]), 2)})
df = pd.DataFrame(rows).sort_values("move", key=abs, ascending=False)
top_a, top_b = set(a.nsmallest(25, "Rank").index), set(b.nsmallest(25, "Rank").index)
out = {"run_date": str(run.date()), "stocks": len(df),
       "median_abs_rank_move": float(df["move"].abs().median()),
       "share_moving_25_or_more": round(float((df["move"].abs() >= 25).mean()), 3),
       "max_abs_rank_move": int(df["move"].abs().max()),
       "median_abs_composite_change": float(df["composite_change"].abs().median()),
       "top25_changed": sorted(top_a ^ top_b), "per_stock": df.to_dict("records")}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "per_stock"}, indent=1))
print(df.head(15).to_string(index=False))

# The other side of the trade: switching these stocks from the source to the filings changes the
# definitions too. Same quarter, two sources: the run's own record against the filings as of the
# stale quarter.
y = score(base)
cost = []
for t in stale_at:
    if t in a.index and t in y.index:
        cost.append(abs(int(a.loc[t, "Rank"]) - int(y.loc[t, "Rank"])))
switch = []
for t in stale_at:
    if t in b.index and t in y.index:
        switch.append(abs(int(b.loc[t, "Rank"]) - int(y.loc[t, "Rank"])))
out["definitions_median_abs_rank_move"] = float(np.median(cost))
out["source_to_fresh_filings_median_abs_rank_move"] = float(np.median(switch))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("same quarter, source vs filings (definitions only): median |rank move|", np.median(cost))
print("source (stale) vs filings (fresh): median |rank move|", np.median(switch))
