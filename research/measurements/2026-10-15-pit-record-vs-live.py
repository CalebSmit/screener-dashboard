"""Can the engine score a stock from its SEC filings alone, as it would from Yahoo's statements?

Backtest v2 needs every historical month's fundamentals as they were filed by then
(plan/backtest-v2.md step 3). pit_record.py rebuilds the engine's raw record from SEC facts, so
v2 can score history with the engine's own formulas. Before any history is scored, this checks the
rebuild where the answer is known: today.

Arms, all scored by the same factor_engine code at the same config:
  A  the live run's raw records, as fetched (Yahoo)
  S  A with every *statement* field rebuilt from filings as of the run date
  B  A with every statement and *market* field rebuilt (market value, EV, book value, EPS, ROE..
     from filings at the live price) - what v2 would actually score
Price-history and analyst fields stay the live ones in every arm: this measures the fundamentals
rebuild only.

Usage: python research/measurements/2026-10-15-pit-record-vs-live.py [RUN_DIR]
Writes the JSON beside this file.
"""
from __future__ import annotations

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

import factor_engine as fe  # noqa: E402
import lookahead  # noqa: E402  (diagnostic module; fine in a measurement script)
import pit_record as pr  # noqa: E402

OUT = Path(__file__).with_suffix(".json")


def newest_run_dir() -> Path:
    runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
    return runs[-1].parent


def to_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for r in df.to_dict("records"):
        for k, v in list(r.items()):
            if isinstance(v, np.ndarray):
                r[k] = v.tolist()
        recs.append(r)
    return recs


def score(raw: list[dict], cfg: dict) -> pd.DataFrame:
    df = fe.compute_metrics(raw, pd.Series(dtype=float), cfg, risk_free_rate=0.04)
    df = df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()
    metrics = df.copy()
    df = fe.compute_sector_percentiles(df)
    df = fe.apply_percentile_transform(df, cfg)
    df = fe.compute_category_scores(df, cfg)
    df = fe.compute_composite(df, cfg)
    df = fe.rank_stocks(df)
    return metrics.set_index("Ticker"), df.set_index("Ticker")


def rel_close(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    den = np.maximum(a.abs(), b.abs())
    return ((a - b).abs() <= tol * den) | ((a - b).abs() < 1e-9)


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_run_dir()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw_df = pd.read_parquet(run_dir / "00_raw_fetch.parquet")
    raw_df = raw_df[raw_df["marketCap"].notna()]
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    run_date = pd.Timestamp(raw_df.get("_ctx_last_date").dropna().max() if "_ctx_last_date" in raw_df
                            else pd.Timestamp.today()).normalize() + pd.Timedelta(days=1)
    base = to_records(raw_df)

    arms = {"A": base, "S": [], "B": []}
    for r in base:
        price = r.get("currentPrice")
        full = pr.record_as_of(r, facts, run_date, price)
        stmt_only = dict(r)
        for k in pr.FIELD_CLASSES["statement"]:
            stmt_only[k] = full.get(k, np.nan)
        for k in [k for k in full if k.startswith("_beneish_") or k.startswith("_rev_q")]:
            stmt_only[k] = full[k]
        arms["S"].append(stmt_only)
        arms["B"].append(full)

    # field by field: the rebuilt record against Yahoo's, where Yahoo has a value
    fields = {}
    for k in [k for k in pr.FIELD_CLASSES["statement"] + pr.FIELD_CLASSES["market"]
              if k not in ("_ltm_annualized", "_data_source", "_stmt_date_financials")]:
        a = pd.Series([r.get(k) for r in base], dtype="object")
        b = pd.Series([r.get(k) for r in arms["B"]], dtype="object")
        a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        live = a.notna()
        both = live & b.notna()
        if live.sum() < 10:
            continue
        fields[k] = {"live": int(live.sum()), "coverage": round(both.sum() / live.sum(), 4),
                     "within_2pct": round(float(rel_close(a[both], b[both], 0.02).mean()), 4)
                     if both.sum() else None,
                     "within_10pct": round(float(rel_close(a[both], b[both], 0.10).mean()), 4)
                     if both.sum() else None}

    weights = lookahead.composite_metric_weights(cfg)
    weighted = sorted([m for m, w in weights.items() if w > 0], key=lambda m: -weights[m])
    scored = {k: score(v, cfg) for k, v in arms.items()}
    mA, sA = scored["A"]

    out = {"run_dir": run_dir.name, "as_of": str(run_date.date()), "n_live": int(len(sA)),
           "fields": fields, "metrics": {}, "arms": {}}
    for arm in ("S", "B"):
        mX, sX = scored[arm]
        per = {}
        for m in weighted:
            if m not in mA or m not in mX:
                continue
            a, b = mA[m], mX[m].reindex(mA.index)
            live = a.notna()
            both = live & b.notna()
            row = {"weight": round(weights[m], 3), "bucket": next(
                (k for k, v in lookahead.BUCKETS.items() if m in v), "?"),
                "live": int(live.sum()), "rebuilt": int(both.sum()),
                "coverage": round(both.sum() / live.sum(), 4) if live.sum() else None}
            if both.sum() >= 10:
                row["spearman"] = round(float(a[both].rank().corr(b[both].rank())), 4)
                row["within_2pct"] = round(float(rel_close(a[both], b[both], 0.02).mean()), 4)
                row["within_10pct"] = round(float(rel_close(a[both], b[both], 0.10).mean()), 4)
            per[m] = row
        out["metrics"][arm] = per

        common = sA.index.intersection(sX.index)
        ca, cb = sA.loc[common, "Composite"], sX.loc[common, "Composite"]
        ra, rb = sA.loc[common, "Rank"], sX.loc[common, "Rank"]
        topA = set(sA.nsmallest(25, "Rank").index)
        topB = set(sX.nsmallest(25, "Rank").index)
        decA = set(sA.nsmallest(len(sA) // 10, "Rank").index)
        decB = set(sX.nsmallest(len(sX) // 10, "Rank").index)
        cats = {}
        for c in cfg["factor_weights"]:
            col = f"{c}_score"
            if col in sA and col in sX:
                x, y = sA.loc[common, col], sX.loc[common, col]
                ok = x.notna() & y.notna()
                cats[c] = round(float(x[ok].rank().corr(y[ok].rank())), 4)
        out["arms"][arm] = {
            "scored": int(len(sX)), "common": int(len(common)),
            "composite_spearman": round(float(ca.rank().corr(cb.rank())), 4),
            "top25_overlap": len(topA & topB), "top_decile_overlap": len(decA & decB),
            "top_decile_size": len(decA),
            "median_abs_rank_change": float((ra - rb).abs().median()),
            "p90_abs_rank_change": float((ra - rb).abs().quantile(0.9)),
            "category_spearman": cats,
        }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, row in fields.items():
        print(f"   field {k:28} live {row['live']:4}  cov {row['coverage']}  2% {row['within_2pct']}  10% {row['within_10pct']}")
    for arm in ("S", "B"):
        print(f"== arm {arm}:", json.dumps(out["arms"][arm]))
        for m, row in out["metrics"][arm].items():
            print(f"   {m:24} w{row['weight']:6.2f} {row['bucket']:20} cov {row['coverage']}  "
                  f"rho {row.get('spearman')}  2% {row.get('within_2pct')}  10% {row.get('within_10pct')}")


if __name__ == "__main__":
    main()
