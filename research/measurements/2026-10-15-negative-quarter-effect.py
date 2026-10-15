"""What rejecting DuPont's derived quarter does to its inputs and its place in today's ranking.

Rescores the newest run's raw records twice with the engine at today's configuration: as fetched,
and with DD's record re-fetched live through the fixed fetch (annual statements). Everyone else's
record is identical in both, so the difference is DD's.
Usage: python research/measurements/2026-10-15-negative-quarter-effect.py  (JSON beside it)
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

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
raw = pd.read_parquet(sorted(glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet")), key=os.path.getmtime)[-1])
recs = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in raw.to_dict("records")]
old = next(r for r in recs if r["Ticker"] == "DD")
new = fe.fetch_single_ticker("DD")
fixed = [dict(r) for r in recs]
i = next(j for j, r in enumerate(fixed) if r["Ticker"] == "DD")
# keep the run's price-derived fields so only the statements differ
for k in ("price_latest", "price_1m_ago", "price_6m_ago", "price_12m_ago", "volatility_1y", "_vol_daily_sd",
          "currentPrice", "marketCap", "enterpriseValue"):
    if k in old:
        new[k] = old[k]
fixed[i] = {**old, **{k: v for k, v in new.items() if not k.startswith("_ctx_")}}


def score(rs):
    with contextlib.redirect_stdout(io.StringIO()):
        df = fe.compute_metrics(rs, pd.Series(dtype=float), cfg, risk_free_rate=0.04)
        df = df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, cfg)
        df = fe.compute_category_scores(df, cfg)
        df = fe.compute_composite(df, cfg)
        return fe.rank_stocks(df).set_index("Ticker")


a, b = score(recs), score(fixed)
inputs = {k: (old.get(k), new.get(k)) for k in ("totalRevenue", "grossProfit", "ebit", "netIncome", "operatingCashFlow", "_data_source")}
metrics = {m: (float(a.loc["DD", m]) if pd.notna(a.loc["DD", m]) else None,
               float(b.loc["DD", m]) if pd.notna(b.loc["DD", m]) else None)
           for m in ("ev_sales", "ev_ebitda", "roic", "gross_profit_assets", "fcf_yield", "earnings_yield", "accruals")}
out = {"rejected": new.get("_quarterly_rejected"), "inputs_before_after": inputs, "metrics_before_after": metrics,
       "composite": [float(a.loc["DD", "Composite"]), float(b.loc["DD", "Composite"])],
       "rank": [int(a.loc["DD", "Rank"]), int(b.loc["DD", "Rank"])], "of": int(len(a)),
       "others_unchanged_spearman": float(a.drop("DD")["Composite"].rank().corr(b.drop("DD")["Composite"].rank()))}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
