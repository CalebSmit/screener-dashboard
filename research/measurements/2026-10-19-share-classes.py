"""One company, two rows: share classes in the screener's universe (2026-10-19).

research/2026-10-19-share-classes.md is the write-up. This script, offline, on the newest full run:

  A. inventory - every company in the universe with more than one class of common stock:
       * two lines in the universe (same SEC CIK in data/sec/company_tickers.json);
       * one line in the universe and another listed common class outside it (named below);
       * one line whose own share count is under 90% of the company's balance-sheet count (an
         unlisted or unindexed class), with a check that the source's market value is the
         company's (all classes at this line's price) rather than the line's;
       * a per-share unit check: equity / bookValue and net income / trailingEps against the
         company share count implied by market value - a ratio far from 1 is a per-share figure
         on another class's unit (Berkshire's book value is per Class A share).
  B. for each pair in the universe: every weighted metric, both rows, whether they differ, the
     class of input that makes them differ (company statements / company value at the line's
     price / the line's own return series / the line's own market data / analyst inputs), a
     check that the price-driven ones differ by exactly the price ratio, and the composite gap
     decomposed by category and by input class.
  C. the stakes - the run's own metric table (02_outliers_flagged) rescored with the run's own
     config and effective weights (the baseline reproduces all published ranks; asserted), under:
       BASE        as published
       FILL        a line missing an analyst figure takes its sibling's (gaps only)
       CONSENSUS   both lines take the designated line's analyst figures (designated = higher
                   median dollar volume, S&P DJI's and FTSE Russell's rule; here also the line
                   with more analysts)
       CONS_GOOG   CONSENSUS, but Alphabet on GOOG's figures (the line whose EPS history is on
                   the consensus basis, section 3 of the note)
       ONCE        score once and copy: the secondary line takes every metric of the designated
                   line, so both rows carry one composite
       ONCE_DROP   the company is represented once, by the designated line (S&P DJI approach B);
                   the secondary lines leave the ranking
     each with rank moves, the top 25, and how many companies hold two of its places.
  D. history - every snapshot in improvement/snapshots: how often both lines of one company stood
     inside the paper-trading bot's buy list (top 25) and hold band (rank 40).

Usage:  python research/measurements/2026-10-19-share-classes.py   (JSON beside this file)
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

OUT = Path(__file__).with_suffix(".json")


def newest_full_run():
    runs = glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet"))
    return Path(sorted(runs, key=os.path.getmtime)[-1]).parent


RUN = newest_full_run()
raw = pd.read_parquet(RUN / "00_raw_fetch.parquet").set_index("Ticker")
FLAGGED = pd.read_parquet(RUN / "02_outliers_flagged.parquet")
published = pd.read_parquet(RUN / "05_final_scored.parquet").set_index("Ticker")
RUN_CFG = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
_eff = json.loads((RUN / "effective_weights.json").read_text(encoding="utf-8"))
RUN_CFG["factor_weights"] = _eff["factor_weights"]
RUN_CFG["metric_weights"] = _eff["metric_weights"]
FW = _eff["factor_weights"]
MW = {c: {k: v for k, v in w.items() if v} for c, w in _eff["metric_weights"].items()}
WEIGHTED = [m for c in MW.values() for m in c]
CAT_OF = {m: c for c, w in MW.items() for m in w}
out = {"run": RUN.name}


def rnd(v, n=4):
    try:
        return None if v is None or not np.isfinite(float(v)) else round(float(v), n)
    except (TypeError, ValueError):
        return v


# ------------------------------------------------------------------------- A. inventory
ct = pd.DataFrame(json.loads((ROOT / "data" / "sec" / "company_tickers.json").read_text()).values())
ct["t"] = ct["ticker"].str.replace(".", "-", regex=False)
U = set(raw.index)
cik = ct[ct["t"].isin(U)]
pairs = {}
for c, g in cik.groupby("cik_str"):
    if len(g) > 1:
        pairs[int(c)] = sorted(g["t"])
# Other listed classes of *common* stock (the CIK file also lists preferreds, notes and warrants,
# which are not share classes of the equity being scored); checked by hand against the file.
OTHER_LISTED_COMMON = {"BRK-B": "BRK-A", "BF-B": "BF-A", "LEN": "LEN-B", "MKC": "MKC-V"}
for t, o in OTHER_LISTED_COMMON.items():
    assert t in U and o in set(ct["t"]) and o not in U, (t, o)

company_shares = raw["marketCap"] / raw["currentPrice"]      # all classes at this line's price
line_ratio = raw["sharesOutstanding"] / raw["sharesBS"]
mc_is_company = company_shares / raw["sharesBS"]
in_pairs = {t for v in pairs.values() for t in v}
unlisted = raw.index[(line_ratio < 0.90) & ~raw.index.isin(in_pairs)]
A = {
    "pairs_in_universe": {", ".join(v): raw.loc[v[0], "longName"] for v in pairs.values()},
    "rows_for_companies": f"{len(U)} rows, {len(U) - sum(len(v) - 1 for v in pairs.values())} companies",
    "other_listed_common_class_outside_universe": OTHER_LISTED_COMMON,
    "line_shares_under_90pct_of_company_shares": {
        t: {"line_over_company_shares": rnd(line_ratio[t], 3),
            "marketCap_over_price_over_company_shares": rnd(mc_is_company[t], 3)}
        for t in sorted(unlisted, key=lambda x: line_ratio[x])},
    "marketCap_is_company_level_within_2pct": int(((mc_is_company - 1).abs() <= 0.02).sum()),
    "marketCap_checked": int(mc_is_company.notna().sum()),
}
unit_book = (raw["totalEquity"] / raw["bookValue"]) / company_shares
unit_eps = (raw["netIncome"] / raw["trailingEps"]) / company_shares
A["per_share_unit_check"] = {
    "book_value_off_by_10x_or_more": {t: rnd(v, 5) for t, v in unit_book.items()
                                      if np.isfinite(v) and (v < 0.1 or v > 10)},
    "trailing_eps_off_by_10x_or_more": {t: rnd(v, 5) for t, v in unit_eps.items()
                                        if np.isfinite(v) and (v < 0.1 or v > 10)},
    "BRK-B": {"bookValue": rnd(raw.loc["BRK-B", "bookValue"], 1),
              "priceToBook": rnd(raw.loc["BRK-B", "priceToBook"], 6),
              "price": rnd(raw.loc["BRK-B", "currentPrice"], 2),
              "sharesBS (Class A equivalents)": rnd(raw.loc["BRK-B", "sharesBS"], 0),
              "book_per_B_share = equity / (sharesBS x 1500)":
                  rnd(raw.loc["BRK-B", "totalEquity"] / (raw.loc["BRK-B", "sharesBS"] * 1500), 2),
              "trailingEps (per B share, check)":
                  rnd(raw.loc["BRK-B", "netIncome"] / (raw.loc["BRK-B", "sharesBS"] * 1500), 2)},
}
out["A_inventory"] = A

# ------------------------------------------------------------- B. metric by metric, per pair
CLASS = {}
for m in WEIGHTED:
    CLASS[m] = "company statements"
for m in ("ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales", "size_log_mcap", "pb_ratio"):
    CLASS[m] = "company figure / company value at this line's price"
for m in ("return_12_1", "return_6m", "jensens_alpha", "volatility", "beta", "max_drawdown_1y"):
    CLASS[m] = "this line's own return series"
CLASS["short_interest_ratio"] = "this line's own market data (short interest / volume)"
for m in ("forward_eps_growth", "analyst_surprise", "consecutive_beat_streak"):
    CLASS[m] = "analyst inputs per share (the source's record for this line)"
CLASS["fy1_revision_3m"] = "analyst revision per share (the source's record for this line), over this line's price"
CLASS["price_target_upside"] = "analyst target in this line's price"
ANALYST = ["forward_eps_growth", "fy1_revision_3m", "analyst_surprise", "consecutive_beat_streak",
           "price_target_upside"]

F = FLAGGED.set_index("Ticker")
DESIGNATED = {}
B = {}
for v in pairs.values():
    a, b = sorted(v, key=lambda t: -raw.loc[t, "avg_daily_dollar_volume"])
    DESIGNATED[b] = a                     # secondary -> designated
    rows = {}
    for m in WEIGHTED:
        x, y = F.loc[a, m], F.loc[b, m]
        same = (pd.isna(x) and pd.isna(y)) or (pd.notna(x) and pd.notna(y) and abs(x - y) <= 1e-9 * max(1, abs(x)))
        rows[m] = {"category": CAT_OF[m], a: rnd(x), b: rnd(y), "differ": not same,
                   "input_class": CLASS[m]}
    pa, pb = raw.loc[a, "currentPrice"], raw.loc[b, "currentPrice"]
    mca, mcb = raw.loc[a, "marketCap"], raw.loc[b, "marketCap"]
    checks = {
        "price_ratio": rnd(pa / pb),
        "marketCap_ratio": rnd(mca / mcb),
        "earnings_yield_ratio_x_price_ratio (1 = price explains it)": rnd(F.loc[a, "earnings_yield"] / F.loc[b, "earnings_yield"] * pa / pb),
        "fcf_yield_ratio_x_price_ratio": rnd(F.loc[a, "fcf_yield"] / F.loc[b, "fcf_yield"] * pa / pb),
        "EV_gap_minus_marketCap_gap ($bn, 0 = price explains it)": rnd((raw.loc[a, "enterpriseValue"] - raw.loc[b, "enterpriseValue"] - (mca - mcb)) / 1e9, 3),
    }
    same_inputs = {f: {a: rnd(raw.loc[a, f]), b: rnd(raw.loc[b, f])}
                   for f in ("trailingEps", "forwardEps", "_fy1_eps_current", "_fy2_eps_current",
                             "_fy1_eps_90d_ago", "numberOfAnalystOpinions", "targetMeanPrice",
                             "bookValue", "dividendRate", "netIncome", "totalEquity",
                             "avg_daily_dollar_volume", "heldPercentInsiders")}
    eps_hist = {t: (json.loads(raw.loc[t, "_eps_quarters"]) if isinstance(raw.loc[t, "_eps_quarters"], str) else None)
                for t in (a, b)}
    # composite gap, by category and by input class (category score gap x factor weight / 100)
    cat_gap = {c: rnd((published.loc[a, f"{c}_score"] - published.loc[b, f"{c}_score"]) * FW[c] / 100, 2)
               for c in FW}
    B[f"{a} / {b}"] = {
        "designated (higher dollar volume)": a,
        "ranks": {a: int(published.loc[a, "Rank"]), b: int(published.loc[b, "Rank"])},
        "composites": {a: rnd(published.loc[a, "Composite"], 2), b: rnd(published.loc[b, "Composite"], 2)},
        "composite_gap_by_category (points)": cat_gap,
        "metrics_that_differ": {m: r for m, r in rows.items() if r["differ"]},
        "metrics_identical": sorted(m for m, r in rows.items() if not r["differ"]),
        "price_checks": checks,
        "inputs": same_inputs,
        "eps_history [quarter, actual, estimate, surprise]": eps_hist,
        "feg_basis": {t: F.loc[t, "_feg_basis"] for t in (a, b)},
    }
out["B_pairs"] = B

# Alphabet: the two lines' EPS histories against the filings (EarningsPerShareDiluted)
facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
def filed_quarterly_eps(t):
    """Quarterly diluted EPS as filed; Q4 = fiscal year less the first nine months."""
    g = facts[(facts["ticker"] == t) & (facts["concept"] == "EarningsPerShareDiluted")
              & (facts["end"] >= "2025-01-01")].copy()
    g["days"] = (g["end"] - g["start"]).dt.days
    q = {str(e.date()): v for e, v in zip(*g[g["days"] < 100].drop_duplicates(["end"])[["end", "val"]].T.values)}
    fy = g[g["days"] > 350].drop_duplicates(["end"])
    for _, r in fy.iterrows():
        nine = g[(g["start"] == r["start"]) & (g["days"].between(260, 280))]
        if len(nine):
            q.setdefault(str(r["end"].date()), round(r["val"] - nine["val"].iloc[0], 2))
    return dict(sorted(q.items()))


out["filed_quarterly_diluted_eps"] = {t: filed_quarterly_eps(t) for t in ("GOOGL", "AMZN")}
out["source_eps_history_amzn"] = json.loads(raw.loc["AMZN", "_eps_quarters"])

# How common is the GOOGL signature (a quarter's |surprise| > 50%) across the universe?
sig = []
for t, q in raw["_eps_quarters"].items():
    if isinstance(q, str):
        s = [x[3] for x in json.loads(q)[-4:] if x[3] is not None]
        if any(abs(v) > 0.5 for v in s):
            sig.append(t)
out["stocks_with_a_quarter_surprise_over_50pct"] = {"count": len(sig), "tickers": sig}


# ---------------------------------------------------------------------------- C. stakes
def score(df):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = fe.compute_sector_percentiles(df.copy())
        df = fe.apply_percentile_transform(df, RUN_CFG)
        df = fe.compute_category_scores(df, RUN_CFG)
        df = fe.compute_composite(df, RUN_CFG)
        return fe.rank_stocks(df).set_index("Ticker")


def with_copied(source_of, cols, only_missing=False):
    """source_of: {target_ticker: source_ticker}. Copies the 02-table values of `cols`."""
    df = FLAGGED.copy().set_index("Ticker")
    for tgt, src in source_of.items():
        for m in cols:
            if only_missing and pd.notna(df.loc[tgt, m]):
                continue
            if m == "fy1_revision_3m" and cols is ANALYST:
                # the per-share revision is the company's; the engine scales it by the line's own
                # price (factor_engine, "Scaled by price"), so the target keeps its own price
                df.loc[tgt, m] = revision_per_share(src) / raw.loc[tgt, "currentPrice"]
            else:
                df.loc[tgt, m] = df.loc[src, m]
    return df.reset_index()


def revision_per_share(t):
    return raw.loc[t, "_fy1_eps_current"] - raw.loc[t, "_fy1_eps_90d_ago"]


for _t in ("GOOGL", "GOOG", "FOXA", "NWSA", "NWS"):   # the formula reproduces the run's value
    assert abs(revision_per_share(_t) / raw.loc[_t, "currentPrice"] - F.loc[_t, "fy1_revision_3m"]) < 1e-9, _t


METRIC_COLS = [m for m in fe.METRIC_COLS if m in FLAGGED.columns]
both_ways = {}
for sec_, des in DESIGNATED.items():
    both_ways[sec_] = des
    both_ways[des] = sec_
ARMS = {
    "BASE": FLAGGED.copy(),
    "FILL": with_copied(both_ways, ANALYST, only_missing=True),
    "CONSENSUS": with_copied(DESIGNATED, ANALYST),
    "CONS_GOOG": with_copied({**{k: v for k, v in DESIGNATED.items() if k != "GOOG"}, "GOOGL": "GOOG"}, ANALYST),
    "ONCE": with_copied(DESIGNATED, METRIC_COLS),
    "ONCE_DROP": FLAGGED[~FLAGGED["Ticker"].isin(list(DESIGNATED))].copy(),
}
base = score(ARMS["BASE"])
out["baseline_reproduces_published_ranks"] = f'{int((base["Rank"] == published.loc[base.index, "Rank"]).sum())} of {len(base)}'
company_of = {t: t for t in base.index}
for s_, d_ in DESIGNATED.items():
    company_of[s_] = d_


def top(df, n):
    return list(df.nsmallest(n, "Rank").index)


def doubles(names):
    seen = {}
    for t in names:
        seen.setdefault(company_of.get(t, t), []).append(t)
    return {k: v for k, v in seen.items() if len(v) > 1}


C = {}
LINES = sorted(in_pairs)
for k, df in ARMS.items():
    s = score(df)
    common = [t for t in base.index if t in s.index]
    mv = (base.loc[common, "Rank"] - s.loc[common, "Rank"]).astype(int)
    t25, t40 = top(s, 25), top(s, 40)
    C[k] = {
        "lines": {t: {"rank": f'{int(base.loc[t, "Rank"])} -> {int(s.loc[t, "Rank"])}' if t in s.index else f'{int(base.loc[t, "Rank"])} -> (not ranked)',
                      "composite": f'{rnd(base.loc[t, "Composite"], 2)} -> {rnd(s.loc[t, "Composite"], 2) if t in s.index else None}',
                      "growth": rnd(s.loc[t, "growth_score"], 1) if t in s.index else None,
                      "revisions": rnd(s.loc[t, "revisions_score"], 1) if t in s.index else None}
                  for t in LINES},
        "others_moved_any": int((mv[~mv.index.isin(LINES)] != 0).sum()),
        "others_max_abs_move": int(mv[~mv.index.isin(LINES)].abs().max()),
        "biggest_other_moves": {t: {"sector": base.loc[t, "Sector"],
                                    "rank": f'{int(base.loc[t, "Rank"])} -> {int(s.loc[t, "Rank"])}'}
                                for t in mv[~mv.index.isin(LINES)].abs().sort_values(ascending=False).head(5).index},
        "same_sector_others_mean_composite_change": rnd(
            (s.loc[[t for t in common if t not in LINES and base.loc[t, "Sector"] == base.loc["GOOGL", "Sector"]], "Composite"]
             - base.loc[[t for t in common if t not in LINES and base.loc[t, "Sector"] == base.loc["GOOGL", "Sector"]], "Composite"]).mean(), 2),
        "others_moved_10_or_more_by_sector": base.loc[[t for t in mv.index if t not in LINES and abs(mv[t]) >= 10],
                                                      "Sector"].value_counts().to_dict(),
        "top25_out": sorted(set(top(base, 25)) - set(t25)),
        "top25_in": sorted(set(t25) - set(top(base, 25))),
        "companies_twice_in_top25": doubles(t25),
        "companies_twice_in_top40": doubles(t40),
    }
out["C_stakes"] = C
out["designated_lines"] = {s_: d_ for s_, d_ in DESIGNATED.items()}
_sec = base.loc[LINES[0], "Sector"]
out["sector_peer_group"] = {"sector": _sec, "rows": int((base["Sector"] == _sec).sum()),
                            "rows_that_are_second_lines": sum(1 for t in DESIGNATED if base.loc[t, "Sector"] == _sec),
                            "all_six_lines_in_it": bool(all(base.loc[t, "Sector"] == _sec for t in LINES))}

# ------------------------------------------------------------------------------ D. history
snaps = sorted(glob.glob(str(ROOT / "improvement" / "snapshots" / "*.parquet")))
H = {f"{d_}/{s_}": {"runs": 0, "both_top25": [], "both_top40_one_top25": [], "gap_ranks": []}
     for s_, d_ in DESIGNATED.items()}
brk = []
for p in snaps:
    sn = pd.read_parquet(p).set_index("Ticker")
    date = Path(p).stem.split("_")[0]
    for s_, d_ in DESIGNATED.items():
        if s_ in sn.index and d_ in sn.index:
            h = H[f"{d_}/{s_}"]
            r1, r2 = int(sn.loc[d_, "Rank"]), int(sn.loc[s_, "Rank"])
            h["runs"] += 1
            h["gap_ranks"].append(abs(r1 - r2))
            if r1 <= 25 and r2 <= 25:
                h["both_top25"].append(date)
            if max(r1, r2) <= 40 and min(r1, r2) <= 25:
                h["both_top40_one_top25"].append(date)
for k, h in H.items():
    g_ = h.pop("gap_ranks")
    h["median_rank_gap"] = float(np.median(g_)) if g_ else None
    h["max_rank_gap"] = int(max(g_)) if g_ else None
out["D_history"] = {"snapshots": len(snaps), "first": Path(snaps[0]).stem, "last": Path(snaps[-1]).stem, "pairs": H}

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
