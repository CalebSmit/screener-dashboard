"""The second reader's measurements for the share-class proposal (2026-10-21).

`research/2026-10-19-share-classes.md` proposed scoring each company once on its designated
listing, picked from each run's own `avg_daily_dollar_volume`. This script measures the four
things the review needed and the note did not have, and it is what the accepted change rests on.

    python research/measurements/2026-10-21-share-class-stability.py            # A, C, D (offline)
    python research/measurements/2026-10-21-share-class-stability.py --network   # add B

A. **Availability** - the test that sank the earnings-definition proposal and cleared the bank
   P/B one. Does every universe line resolve to an SEC CIK, and does CIK grouping ever merge two
   lines that are not share classes of one company?
B. **Stability** (needs the network) - 89 month-ends of 63-day average daily dollar volume for
   the three pairs. How often are the two lines within S&P DJI's 20% switching margin, and how
   often does the higher-volume line change? This is the measurement that changed the proposal:
   the designated line is now *declared* in `config.yaml`, not re-picked every run.
C. **The flip cost** - the run rescored with Alphabet on GOOG instead of GOOGL.
D. **Re-measurement of the note's own headline figures** on the newest run, independently.

Offline except for B. Reads only the newest full run, the SEC ticker cache and config.yaml;
writes `2026-10-21-share-class-stability.json` beside itself. Nothing here is imported by
production code.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import factor_engine as fe            # noqa: E402
import share_classes as sc            # noqa: E402

OUT = Path(__file__).with_suffix(".json")
PAIRS = {"Alphabet": ("GOOGL", "GOOG"), "Fox": ("FOXA", "FOX"), "News Corp": ("NWSA", "NWS")}


def newest_run() -> Path:
    runs = glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet"))
    if not runs:
        raise SystemExit("no run artifacts under runs/ - nothing to measure")
    return Path(sorted(runs, key=os.path.getmtime)[-1]).parent


# ------------------------------------------------------------------- A. availability
def part_a(raw: pd.DataFrame) -> dict:
    tickers = [str(t).upper() for t in raw.index]
    ciks = sc.cik_map()
    exact = [t for t in tickers if t in ciks]
    missing = [t for t in tickers if t not in ciks]
    groups_cik, basis = sc.group_companies(tickers, ciks=ciks)
    names = raw["longName"].to_dict() if "longName" in raw.columns else {}
    groups_name, _ = sc.group_companies(tickers, names={str(k).upper(): v for k, v in names.items()},
                                        ciks={})
    key = lambda gs: sorted(tuple(sorted(g)) for g in gs)   # noqa: E731
    return {
        "universe_lines": len(tickers),
        "resolved_to_a_cik": len(exact),
        "no_cik": missing,
        "grouping_basis": basis,
        "multi_line_companies_by_cik": key(groups_cik),
        "multi_line_companies_by_longname": key(groups_name),
        "the_two_bases_agree": key(groups_cik) == key(groups_name),
    }


# ------------------------------------------------------------------- B. stability
def part_b() -> dict:
    import yfinance as yf
    tickers = [t for p in PAIRS.values() for t in p]
    h = yf.download(tickers, period="10y", interval="1d", auto_adjust=False,
                    progress=False, group_by="ticker", threads=True)
    dv = pd.concat({t: (h[t]["Close"] * h[t]["Volume"]).dropna() for t in tickers}, axis=1)
    monthly = dv.rolling(63).mean().resample("ME").last().dropna()
    out = {"window": "63 trading-day average daily dollar volume, read at each month end",
           "month_ends": len(monthly),
           "from": str(monthly.index.min().date()), "to": str(monthly.index.max().date())}
    for a, b in PAIRS.values():
        m = monthly[[a, b]].dropna()
        win = m.apply(lambda r: a if r[a] >= r[b] else b, axis=1)
        ratio = m.max(axis=1) / m.min(axis=1)
        changed = (win != win.shift()).iloc[1:]
        out[f"{a}/{b}"] = {
            "months": len(m),
            "months_won_by_the_declared_line": int((win == a).sum()),
            "times_the_higher_volume_line_changed": int(changed.sum()),
            "change_months": [str(d.date()) for d in win.index[1:][changed]],
            "months_within_the_20pct_margin": int((ratio < 1.20).sum()),
            "share_within_the_margin": round(float((ratio < 1.20).mean()), 3),
            "ratio_min": round(float(ratio.min()), 3),
            "ratio_median": round(float(ratio.median()), 3),
        }
    return out


# ------------------------------------------------------------------- C/D. rescoring
def _scorer(run: Path):
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    eff = json.loads((run / "effective_weights.json").read_text(encoding="utf-8"))
    cfg["factor_weights"] = eff["factor_weights"]
    cfg["metric_weights"] = eff["metric_weights"]

    def score(df):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = fe.compute_sector_percentiles(df.copy())
            d = fe.apply_percentile_transform(d, cfg)
            d = fe.compute_category_scores(d, cfg)
            d = fe.compute_composite(d, cfg)
            return fe.rank_stocks(d).set_index("Ticker")
    return score


def part_cd(run: Path, raw: pd.DataFrame) -> dict:
    flagged = pd.read_parquet(run / "02_outliers_flagged.parquet")
    published = pd.read_parquet(run / "05_final_scored.parquet").set_index("Ticker")
    score = _scorer(run)
    base = score(flagged.copy())
    reproduced = int((base["Rank"] == published.loc[base.index, "Rank"]).sum())

    def once_drop(keep: set[str]):
        drop = [t for p in PAIRS.values() for t in p if t not in keep]
        return score(flagged[~flagged["Ticker"].isin(drop)].copy())

    out = {"run": run.name, "baseline_reproduces_published_ranks": f"{reproduced} of {len(base)}"}
    arms = {"declared_GOOGL": {"GOOGL", "FOXA", "NWSA"},
            "after_a_flip_GOOG": {"GOOG", "FOXA", "NWSA"}}
    scored = {}
    for label, keep in arms.items():
        s = scored[label] = once_drop(keep)
        common = [t for t in base.index if t in s.index]
        mv = (base.loc[common, "Rank"] - s.loc[common, "Rank"]).astype(int)
        out[label] = {
            "ranked_rows": len(s),
            "kept_lines": {t: f'{int(base.loc[t, "Rank"])} -> {int(s.loc[t, "Rank"])}'
                           for t in sorted(keep) if t in s.index},
            "top25_unchanged": sorted(s.nsmallest(25, "Rank").index)
                               == sorted(published.nsmallest(25, "Rank").index),
            "other_rows_moved": int((mv != 0).sum()),
            "largest_move": int(mv.abs().max()),
        }
    a, b = scored["declared_GOOGL"], scored["after_a_flip_GOOG"]
    shared = [t for t in a.index if t in b.index]
    d = (a.loc[shared, "Rank"] - b.loc[shared, "Rank"]).astype(int)
    out["flip_cost"] = {
        "alphabet_rank_on_GOOGL": int(a.loc["GOOGL", "Rank"]),
        "alphabet_rank_on_GOOG": int(b.loc["GOOG", "Rank"]),
        "places": abs(int(a.loc["GOOGL", "Rank"]) - int(b.loc["GOOG", "Rank"])),
        "the_ticker_in_the_ranking_changes": True,
        "other_rows_moved_between_the_arms": int((d != 0).sum()),
        "largest_other_move": int(d.abs().max()),
    }

    # D. the note's peer-group figure, re-measured
    sector = raw["_gics_sector"] if "_gics_sector" in raw.columns else raw["sector"]
    lines = {t for p in PAIRS.values() for t in p}
    cs = [t for t in base.index if sector.get(t) == "Communication Services"]
    others = [t for t in cs if t not in lines]
    delta = a.loc[others, "Composite"] - base.loc[others, "Composite"]
    non_cs = [t for t in base.index if t not in cs and t in a.index]
    out["communication_services"] = {
        "rows_published": len(cs),
        "rows_after": len([t for t in cs if t not in lines]) + len(PAIRS),
        "other_stocks": len(others),
        "mean_composite_change": round(float(delta.mean()), 3),
        "min": round(float(delta.min()), 3), "max": round(float(delta.max()), 3),
        "any_went_down": bool((delta < 0).any()),
    }
    out["largest_composite_change_outside_communication_services"] = round(
        float((a.loc[non_cs, "Composite"] - base.loc[non_cs, "Composite"]).abs().max()), 6)

    # the price-ratio identity the page's sentence rests on
    ident = {}
    for a_, b_ in PAIRS.values():
        if a_ in flagged["Ticker"].values and b_ in flagged["Ticker"].values:
            f = flagged.set_index("Ticker")
            ident[f"{a_}/{b_}"] = round(
                float((f.loc[a_, "earnings_yield"] / f.loc[b_, "earnings_yield"])
                      / (raw.loc[b_, "currentPrice"] / raw.loc[a_, "currentPrice"])), 6)
    out["earnings_yield_ratio_over_price_ratio"] = ident
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true",
                    help="also run part B, which downloads ten years of daily volume")
    args = ap.parse_args()

    run = newest_run()
    raw = pd.read_parquet(run / "00_raw_fetch.parquet").set_index("Ticker")
    result = {"run": run.name, "A_availability": part_a(raw)}
    if args.network:
        result["B_stability"] = part_b()
    result.update({"C_D_rescored": part_cd(run, raw)})
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
