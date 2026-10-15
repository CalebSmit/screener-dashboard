"""The fundamentals half of backtest v1's look-ahead, measured month by month.

plan/backtest-v2.md step 1 measured the *price* half on 2026-10-01: restating the price-dependent
metrics at each month's own price moves 63.2% of name-months to a different decile. It called that
a lower bound because 49.0 points of composite weight - everything built from filings - were
frozen at today's values in both arms. pit_record.py can now rebuild those fields as they were
filed by any date (validated against today: research/measurements/2026-10-15-pit-record-vs-live.py),
so the other half can be measured.

For each rebalance month 2020-01..2026-09 (the 2026-10-01 panel), three versions of the same
current-constituent universe are scored by the engine's own compute_metrics and composite:

  V0   today's filings, market value at today's price          - v1's frozen fundamentals
  V1   today's filings, market value at the month's price      - v1 with the price half fixed
  PIT  the filings as they stood at the month's end, at its price - both halves fixed

In all three the price-history fields (12-1 and 6-1 returns) are the month's own and the analyst
fields are today's (no free history - the 9.0pp that stays unmeasurable), so they cancel.
V0 -> V1 is the price half; V1 -> PIT is the fundamentals half; V0 -> PIT is what is fixed in all.
All arms rebuild statements from SEC filings (not Yahoo), so a difference between arms is the
date the filings are read at, never a difference of source.

Usage: python research/measurements/2026-10-15-lookahead-fundamentals-half.py [--max-months N]
Writes the JSON beside this file. Needs cache/lookahead_prices.parquet (the 2026-10-01 panel),
cache/splits.parquet and data/sec/pit/facts.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

OUT = Path(__file__).with_suffix(".json")
TODAY = pd.Timestamp("2026-10-15")


def newest_raw() -> pd.DataFrame:
    runs = sorted((ROOT / "runs").glob("*/00_raw_fetch.parquet"), key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(runs[-1])
    return df[df["marketCap"].notna()]


def to_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})
    return out


def composite(raw: list[dict], cfg: dict) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            df = fe.compute_metrics(raw, pd.Series(dtype=float), cfg, risk_free_rate=0.04)
            df = df[~df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)].copy()
            df = fe.compute_sector_percentiles(df)
            df = fe.apply_percentile_transform(df, cfg)
            df = fe.compute_category_scores(df, cfg)
            df = fe.compute_composite(df, cfg)
    return df.set_index("Ticker")["Composite"]


def deciles(c: pd.Series, n: int = 10) -> pd.Series:
    """backtest._assign_deciles, as the 2026-10-01 measurement applied it."""
    return np.ceil(c.rank(pct=True, method="first") * n).clip(1, n).astype(int)


def compare(a: pd.Series, b: pd.Series) -> dict:
    common = a.dropna().index.intersection(b.dropna().index)
    da, db = deciles(a.loc[common]), deciles(b.loc[common])
    top_a = set(da[da == 10].index)
    return {"n": int(len(common)),
            "decile_changed": int((da != db).sum()),
            "decile_2plus": int(((da - db).abs() >= 2).sum()),
            "top_a": len(top_a), "top_kept": len(top_a & set(db[db == 10].index)),
            "spearman": float(a.loc[common].rank().corr(b.loc[common].rank()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-months", type=int, default=0)
    ap.add_argument("--shard", default="", help="k/K: score months[k::K] only, into a shard file")
    ap.add_argument("--merge", type=int, default=0, help="merge K shard files into the JSON")
    args = ap.parse_args()
    if args.merge:
        return merge(args.merge)

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    prices = pd.read_parquet(ROOT / "cache" / "lookahead_prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    splits = pd.read_parquet(ROOT / "cache" / "splits.parquet")
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    by_t = {t: g for t, g in facts.groupby("ticker")}
    base = {r["Ticker"]: r for r in to_records(newest_raw()) if r["Ticker"] in prices.columns}
    print(f"{len(base)} priced constituents; facts for {len(by_t)}")

    # today's filings, once
    filed_today = {t: pr.Filed(by_t.get(t, facts.iloc[:0]), t, TODAY) for t in base}
    stmt_today = {t: pr.statement_fields(None, t, TODAY, filed=filed_today[t]) for t in base}

    def with_fields(r, stmt, market, hist):
        rec = dict(r)
        for k in pr.FIELD_CLASSES["statement"] + pr.FIELD_CLASSES["market"]:
            rec[k] = np.nan
        for k in [k for k in rec if k.startswith("_beneish_") or k.startswith("_rev_q")]:
            rec[k] = np.nan
        rec.update(stmt)
        rec.update(market)
        rec.update(hist)
        rec["_ltm_annualized"], rec["_data_source"] = False, "quarterly"
        return rec

    v0 = {}
    for t, r in base.items():
        v0[t] = pr.market_fields(stmt_today[t], None, t, TODAY, r.get("currentPrice"), filed=filed_today[t])

    months = [m for m in sorted(prices.index) if m >= prices.index[13]]
    if args.max_months:
        months = months[-args.max_months:]
    out = OUT
    if args.shard:
        k, K = (int(x) for x in args.shard.split("/"))
        months = months[k::K]
        out = OUT.with_suffix(f".shard{k}.json")
    per_month = []
    t0 = time.time()
    for m in months:
        i = prices.index.get_loc(m)
        as_of = m + pd.offsets.MonthEnd(0)
        arms = {"V0": [], "V1": [], "PIT": []}
        for t, r in base.items():
            pm = prices.at[m, t]
            if not np.isfinite(pm):
                continue
            hist = {"price_latest": pm, "currentPrice": pm,
                    "price_1m_ago": prices.iloc[i - 1][t], "price_6m_ago": prices.iloc[i - 6][t],
                    "price_12m_ago": prices.iloc[i - 12][t]}
            F = pr.Filed(by_t.get(t, facts.iloc[:0]), t, as_of)
            st = pr.statement_fields(None, t, as_of, filed=F)
            if not st:
                continue                      # nothing filed yet: not scoreable point-in-time
            arms["V0"].append(with_fields(r, stmt_today[t], v0[t], hist))
            arms["V1"].append(with_fields(r, stmt_today[t], pr.market_fields(
                stmt_today[t], None, t, TODAY, pm, filed=filed_today[t]), hist))
            arms["PIT"].append(with_fields(r, st, pr.market_fields(
                st, None, t, as_of, pm, filed=F, split=pr.split_after(splits, t, as_of)), hist))
        comp = {k: composite(v, cfg) for k, v in arms.items()}
        row = {"month": str(m.date()),
               "price_half": compare(comp["V0"], comp["V1"]),
               "fundamentals_half": compare(comp["V1"], comp["PIT"]),
               "total": compare(comp["V0"], comp["PIT"])}
        per_month.append(row)
        print(f"{m:%Y-%m}  n={row['total']['n']}  price {row['price_half']['decile_changed'] / row['price_half']['n']:.1%}"
              f"  fundamentals {row['fundamentals_half']['decile_changed'] / row['fundamentals_half']['n']:.1%}"
              f"  total {row['total']['decile_changed'] / row['total']['n']:.1%}"
              f"  rho(V1,PIT) {row['fundamentals_half']['spearman']:.3f}  [{time.time() - t0:.0f}s]", flush=True)

    out.write_text(json.dumps(per_month, indent=1), encoding="utf-8")
    if out == OUT:
        OUT.write_text(json.dumps(summarise(per_month), indent=1), encoding="utf-8")


def merge(K: int):
    per_month = []
    for k in range(K):
        f = OUT.with_suffix(f".shard{k}.json")
        per_month += json.loads(f.read_text(encoding="utf-8"))
    per_month.sort(key=lambda r: r["month"])
    OUT.write_text(json.dumps(summarise(per_month), indent=1), encoding="utf-8")
    for k in range(K):
        OUT.with_suffix(f".shard{k}.json").unlink()
    print(json.dumps(summarise(per_month)["summary"], indent=1))


def summarise(per_month: list) -> dict:
    tot = {}
    for row in per_month:
        for k in ("price_half", "fundamentals_half", "total"):
            acc = tot.setdefault(k, {"n": 0, "decile_changed": 0, "decile_2plus": 0, "top_a": 0, "top_kept": 0})
            for f in acc:
                acc[f] += row[k][f]
    summary = {k: {"name_months": v["n"],
                   "decile_changed": round(v["decile_changed"] / v["n"], 4),
                   "decile_2plus": round(v["decile_2plus"] / v["n"], 4),
                   "top_decile_kept": round(v["top_kept"] / v["top_a"], 4) if v["top_a"] else None,
                   "median_monthly_spearman": round(float(np.median([r[k]["spearman"] for r in per_month])), 4)}
               for k, v in tot.items()}
    by_year = {}
    for r in per_month:
        y = r["month"][:4]
        e = by_year.setdefault(y, {"n": 0, "chg": 0})
        e["n"] += r["fundamentals_half"]["n"]
        e["chg"] += r["fundamentals_half"]["decile_changed"]
    summary["fundamentals_half_by_year"] = {y: round(e["chg"] / e["n"], 4) for y, e in by_year.items()}
    return {"months": len(per_month), "summary": summary, "per_month": per_month}


if __name__ == "__main__":
    main()
