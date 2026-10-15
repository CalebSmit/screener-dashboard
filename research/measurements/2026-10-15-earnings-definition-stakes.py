"""Which earnings rows should the screener score, and what would each choice move?

Reads the scan written by `2026-10-15-earnings-rows-scan.py` (the source's adjusted and
as-reported rows for every constituent), the SEC companyfacts cache and the newest run's raw
records. Three measurements:

1. Coverage: for how many companies each row gives a trailing twelve months.
2. Agreement with the filing, fiscal year against fiscal year: the source's "Operating Income"
   (what the screener reads today) and its "Total Operating Income As Reported", each against the
   company's own `OperatingIncomeLoss`.
3. The stakes, with everyone else's record unchanged - the run's raw records rescored by the
   engine at today's configuration:
     OP  operating income as reported (moves ROIC, EV/EBITDA, net debt / EBITDA)
     EY  earnings yield on "Normalized Income" (net income without unusual items), nothing else
     BOTH
   Where a company has no as-reported or normalized figure, its record is left as it is.

Usage: python research/measurements/2026-10-15-earnings-definition-stakes.py   (JSON beside it)
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

scan = json.loads((ROOT / "research" / "measurements" / "2026-10-15-earnings-rows-scan.json").read_text(encoding="utf-8"))["rows"]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
raw = pd.read_parquet(sorted(glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet")), key=os.path.getmtime)[-1])
base = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in raw.to_dict("records")]
out = {}

# 1. coverage
rows = ["Operating Income", "Total Operating Income As Reported", "Net Income", "Normalized Income"]
out["coverage_ttm"] = {r: sum(1 for v in scan.values() if v["ttm"].get(r) is not None) for r in rows}

# 2. agreement with the filing, fiscal year
oi = facts[(facts["concept"] == "OperatingIncomeLoss") & (facts["form"] == "10-K") & facts["start"].notna()]
oi = oi[((oi["end"] - oi["start"]).dt.days.between(340, 380))]
agree = {}
for row in ("Operating Income", "Total Operating Income As Reported"):
    diffs = []
    for t, v in scan.items():
        a = v["annual"].get(row)
        if not a:
            continue
        end = pd.Timestamp(a["end"])
        g = oi[(oi["ticker"] == t) & ((oi["end"] - end).abs() <= pd.Timedelta(days=10))]
        if g.empty:
            continue
        filed = float(g.sort_values("filed").iloc[-1]["val"])
        diffs.append(abs(a["val"] - filed) / max(abs(a["val"]), abs(filed), 1.0))
    d = np.array(diffs)
    agree[row] = {"compared": int(len(d)), "within_0.5pct": int((d <= 0.005).sum()),
                  "share_within_0.5pct": round(float((d <= 0.005).mean()), 4),
                  "over_10pct": int((d > 0.10).sum()), "median_abs_gap": round(float(np.median(d)), 5)}
out["agreement_with_filed_operating_income"] = agree

# how much the two operating-income rows differ, trailing twelve months
gap = []
for t, v in scan.items():
    a, r = v["ttm"].get("Operating Income"), v["ttm"].get("Total Operating Income As Reported")
    if a is not None and r is not None:
        gap.append((t, a, r, abs(a - r) / max(abs(a), abs(r), 1.0)))
g = np.array([x[3] for x in gap])
out["adjusted_vs_as_reported_ttm"] = {"both": int(len(g)), "identical_within_0.5pct": int((g <= 0.005).sum()),
                                      "differ_over_10pct": int((g > 0.10).sum())}
ni_gap = []
for t, v in scan.items():
    a, r = v["ttm"].get("Net Income"), v["ttm"].get("Normalized Income")
    if a is not None and r is not None:
        ni_gap.append(abs(a - r) / max(abs(a), abs(r), 1.0))
ng = np.array(ni_gap)
out["net_vs_normalized_ttm"] = {"both": int(len(ng)), "identical_within_0.5pct": int((ng <= 0.005).sum()),
                                "differ_over_10pct": int((ng > 0.10).sum())}


# 3. the stakes
def score(records, ey_override=None):
    with contextlib.redirect_stdout(io.StringIO()):
        df = fe.compute_metrics(records, pd.Series(dtype=float), cfg, risk_free_rate=0.04)
        df = df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()
        if ey_override is not None:
            m = df["Ticker"].map(ey_override)
            df["earnings_yield"] = np.where(m.notna(), m, df["earnings_yield"])
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, cfg)
        df = fe.compute_category_scores(df, cfg)
        df = fe.compute_composite(df, cfg)
        return fe.rank_stocks(df).set_index("Ticker")


def op_records():
    recs = []
    for r in base:
        r = dict(r)
        v = scan.get(r["Ticker"], {}).get("ttm", {}).get("Total Operating Income As Reported")
        if v is not None and r.get("_data_source") != "annual":
            r["ebit"] = v
        recs.append(r)
    return recs


ey = {}
for r in base:
    v = scan.get(r["Ticker"], {}).get("ttm", {}).get("Normalized Income")
    mc = r.get("marketCap")
    if v is not None and mc and mc > 0:
        ey[r["Ticker"]] = v / mc

a = score(base)
arms = {"OP": score(op_records()), "EY": score(base, ey), "BOTH": score(op_records(), ey)}
res = {}
top_a = set(a.nsmallest(25, "Rank").index)
for k, b in arms.items():
    common = a.index.intersection(b.index)
    mv = (a.loc[common, "Rank"] - b.loc[common, "Rank"])
    top_b = set(b.nsmallest(25, "Rank").index)
    big = mv.abs().sort_values(ascending=False).head(12)
    res[k] = {"composite_spearman": round(float(a.loc[common, "Composite"].rank().corr(b.loc[common, "Composite"].rank())), 4),
              "median_abs_rank_move": float(mv.abs().median()), "moved_25_or_more": int((mv.abs() >= 25).sum()),
              "top25_out": sorted(top_a - top_b), "top25_in": sorted(top_b - top_a),
              "biggest_moves": {t: {"from": int(a.loc[t, "Rank"]), "to": int(b.loc[t, "Rank"])} for t in big.index},
              "top10_after": [f"{t}#{int(b.loc[t, 'Rank'])}" for t in b.nsmallest(10, "Rank").index]}
out["stakes"] = res
out["named"] = {t: {"rank_now": int(a.loc[t, "Rank"]), **{k: int(arms[k].loc[t, "Rank"]) for k in arms}}
                for t in ("EXPE", "APA", "HST", "INCY", "BMY", "WDC", "KHC", "GIS", "CVS", "BA", "CNC") if t in a.index}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
