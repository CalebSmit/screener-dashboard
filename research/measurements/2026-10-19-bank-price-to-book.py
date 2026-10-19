"""Bank price-to-book: what the source's ratio is, which quarter its book value is from, and what a
P/B computed from the run's own figures would move (2026-10-19).

research/2026-10-19-bank-price-to-book.md is the write-up. P/B is 60% of bank Valuation (bank
Valuation 22% of the composite: 13.2 composite points per bank-like stock) and is read from the
source's ``.info`` ``priceToBook`` (factor_engine.py, "Bank Valuation: Price-to-Book").

  sec     (network) for every bank-like stock, the SEC ``companyconcept`` series for Goodwill,
          IntangibleAssetsNetExcludingGoodwill and PreferredStockValue - the three items between
          stockholders' equity and tangible common equity, which the weekly facts cache does not
          carry. Written to ``2026-10-19-bank-price-to-book-sec.json`` beside this file.
  (none)  offline, on the newest full run, the bank ROE/ROA probe of the same morning
          (``2026-10-19-bank-roe-roa-probe.json``: the source's ``.info`` and quarterly balance
          sheets for the 46 bank-like stocks) and the SEC facts cache:
     A. the source's definition: priceToBook against currentPrice / bookValue, universe-wide;
     B. what bookValue is and which quarter: bookValue against equity / shares at the run's
        balance-sheet date on three equity bases (stockholders', common, total incl. minority
        interests), split by whether the source's mostRecentQuarter is the run's statement quarter;
        and a per-share unit check (equity / bookValue against the company's share count);
     C. P/B computed from the run's own fields (market value / stockholders' equity, and price /
        (equity / balance-sheet shares)) and from the filings (market value / filed
        StockholdersEquity at the same date; tangible common equity = filed equity - preferred -
        goodwill - other intangibles), against the source's;
     D. the stakes: the run's metric table (02_outliers_flagged) rescored with the run's own
        config and effective weights (baseline = published ranks, asserted), bank-like pb_ratio
        replaced by:
          BRK_ONLY  the source's ratio, Berkshire's unit error alone corrected
          MC_SE     market value / stockholders' equity, run's own balance sheet   (no new field)
          MC_CE     market value / common equity (excl. preferred), same date      (probe row)
          MC_TBV    market value / tangible book value, same date                  (probe row)
     E. history: Berkshire's P/B percentile in every improvement snapshot.

Usage:
    python research/measurements/2026-10-19-bank-price-to-book.py sec   # network, ~140 requests
    python research/measurements/2026-10-19-bank-price-to-book.py       # offline
"""
import contextlib
import glob
import io
import json
import os
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
HERE = Path(__file__).resolve()
SEC_JSON = HERE.with_name(HERE.stem + "-sec.json")
PROBE = HERE.with_name("2026-10-19-bank-roe-roa-probe.json")
OUT = HERE.with_suffix(".json")
SEC_CONCEPTS = ("Goodwill", "IntangibleAssetsNetExcludingGoodwill", "PreferredStockValue",
                "PreferredStockValueOutstanding")


def newest_full_run():
    runs = glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet"))
    return Path(sorted(runs, key=os.path.getmtime)[-1]).parent


RUN = newest_full_run()
F02 = pd.read_parquet(RUN / "02_outliers_flagged.parquet")
BANKS = sorted(F02.loc[F02["_is_bank_like"].fillna(False).astype(bool), "Ticker"])


# ------------------------------------------------------------------------------- sec probe
def sec_probe():
    from insider_activity import Edgar, user_agent
    ua = user_agent()
    assert ua, "no SEC identity configured (data/sec/user_agent.txt)"
    ed = Edgar(ua)
    cmap = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in
            json.loads((ROOT / "data" / "sec" / "company_tickers.json").read_text()).values()}
    data = {"fetched": time.strftime("%Y-%m-%d %H:%M"), "tickers": {}}
    for t in BANKS:
        cik = cmap.get(t)
        rows = {}
        for c in SEC_CONCEPTS:
            try:
                j = ed.get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{c}.json").json()
                rows[c] = [{"end": f["end"], "filed": f["filed"], "form": f["form"], "val": f["val"]}
                           for f in j.get("units", {}).get("USD", [])
                           if f.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A") and f.get("end", "") >= "2024-01-01"]
            except Exception as e:  # noqa: BLE001 - a 404 means the company does not tag it
                rows[c] = f"{type(e).__name__}"
        data["tickers"][t] = rows
        print(t, {c: (len(v) if isinstance(v, list) else v) for c, v in rows.items()})
    SEC_JSON.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", SEC_JSON)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "sec":
    sec_probe()
    sys.exit(0)

# --------------------------------------------------------------------------------- offline
from scipy.stats import spearmanr  # noqa: E402

import factor_engine as fe  # noqa: E402
import sec_crosscheck as sx  # noqa: E402

raw = pd.read_parquet(RUN / "00_raw_fetch.parquet").set_index("Ticker")
P = json.loads(PROBE.read_text(encoding="utf-8"))
S = json.loads(SEC_JSON.read_text(encoding="utf-8")) if SEC_JSON.exists() else {"tickers": {}}
published = pd.read_parquet(RUN / "05_final_scored.parquet").set_index("Ticker")
cfg = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
out = {"run": RUN.name, "probe_taken": P.get("probed_at"), "sec_fetched": S.get("fetched"),
       "bank_like": len(BANKS),
       "composite_points_on_pb": round(cfg["bank_metric_weights"]["valuation"]["pb_ratio"] / 100
                                       * cfg["factor_weights"]["valuation"], 2)}


def rnd(v, n=4):
    try:
        return None if v is None or not np.isfinite(float(v)) else round(float(v), n)
    except (TypeError, ValueError):
        return None


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b)) if np.isfinite(a) and np.isfinite(b) and max(abs(a), abs(b)) > 0 else np.nan


def ser(block, row):
    s = (block or {}).get(row, {})
    return pd.Series({pd.Timestamp(k): v for k, v in s.items() if v is not None}, dtype=float)


# ---- A. the source's definition --------------------------------------------------------
implied = raw["currentPrice"] / raw["bookValue"]
both = raw[["priceToBook"]].assign(implied=implied).dropna()
out["A_definition"] = {
    "compared": len(both),
    "priceToBook_equals_currentPrice_over_bookValue_within_1e-5": int(((both["priceToBook"] / both["implied"] - 1).abs() < 1e-5).sum()),
}
# the probe of 08:31 describes the run of 02:00
same = [(t, raw.loc[t, "bookValue"], P["tickers"][t]["info"]["bookValue"]) for t in BANKS
        if "info" in P["tickers"].get(t, {}) and P["tickers"][t]["info"].get("bookValue") is not None]
out["A_probe_bookValue_equals_run"] = f"{sum(1 for _, a, b in same if abs(a - b) < 1e-9)} of {len(same)}"

# ---- B. what bookValue is, and which quarter ---------------------------------------------
rows = {}
for t in BANKS:
    d = P["tickers"].get(t, {})
    if "info" not in d:
        continue
    r = raw.loc[t]
    stmt = pd.Timestamp(r["_stmt_date_balance_sheet"])
    mrq = pd.Timestamp(d["info"]["mostRecentQuarter"], unit="s").normalize() if d["info"].get("mostRecentQuarter") else None
    sh = r["sharesBS"] * (1500 if t == "BRK-B" else 1)          # Berkshire's count is in Class A units
    bv = r["bookValue"]
    cand = {}
    for k, row in (("SE", "Stockholders Equity"), ("CE", "Common Stock Equity"),
                   ("TE", "Total Equity Gross Minority Interest"), ("TBV", "Tangible Book Value")):
        e = ser(d.get("q_bs"), row)
        cand[k] = e.get(stmt, np.nan)
    rows[t] = {
        "stmt_date": str(stmt.date()), "info_mrq": str(mrq.date()) if mrq is not None else None,
        "info_quarter_is_statement_quarter": bool(mrq is not None and mrq == stmt),
        "price": r["currentPrice"], "bookValue": bv, "priceToBook": r["priceToBook"],
        "marketCap": r["marketCap"], "run_totalEquity": r["totalEquity"], "sharesBS": sh,
        "SE": cand["SE"], "CE": cand["CE"], "TE": cand["TE"], "TBV": cand["TBV"],
        "bv_over_SE_per_share": bv / (cand["SE"] / sh) if sh else np.nan,
        "bv_over_CE_per_share": bv / (cand["CE"] / sh) if sh else np.nan,
        "bv_over_TE_per_share": bv / (cand["TE"] / sh) if sh else np.nan,
        "unit_check_equity_over_bookValue_over_company_shares": (r["totalEquity"] / bv) / (r["marketCap"] / r["currentPrice"]),
    }
Bd = pd.DataFrame(rows).T
for c in [c for c in Bd.columns if c not in ("stmt_date", "info_mrq", "info_quarter_is_statement_quarter")]:
    Bd[c] = pd.to_numeric(Bd[c], errors="coerce")
aligned = Bd["info_quarter_is_statement_quarter"].astype(bool)


def fit(sub, col, tol=0.005):
    v = (sub[col] - 1).abs()
    return {"n": int(v.notna().sum()), "within_0.5pct": int((v <= tol).sum()), "within_2pct": int((v <= 0.02).sum())}


B = {"info_quarter_newer_than_statements": sorted(Bd.index[~aligned]),
     "aligned": int(aligned.sum())}
for grp, m in (("aligned", aligned), ("info_newer", ~aligned)):
    B[grp] = {k: fit(Bd[m], f"bv_over_{k}_per_share") for k in ("SE", "CE", "TE")}
B["aligned_misses_on_common_basis"] = {t: rnd(v, 4) for t, v in Bd.loc[aligned, "bv_over_CE_per_share"].items()
                                       if not abs(v - 1) <= 0.005}
B["info_newer_bv_over_june_common_bvps"] = {t: rnd(v, 4) for t, v in Bd.loc[~aligned, "bv_over_CE_per_share"].items()}
B["info_newer_median_bv_over_statement_common_bvps"] = rnd(Bd.loc[~aligned, "bv_over_CE_per_share"].median(), 4)
B["unit_check_off_by_more_than_25pct"] = {t: rnd(v, 5) for t, v in Bd["unit_check_equity_over_bookValue_over_company_shares"].items()
                                          if not 0.8 <= v <= 1.25}
out["B_book_value"] = B

# ---- C. computed P/B, from the run and from the filings ----------------------------------
facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")


def sec_instant(rows_, end):
    if not isinstance(rows_, list):
        return None
    m = [r for r in rows_ if pd.Timestamp(r["end"]) == end]
    if not m:
        return None
    return float(sorted(m, key=lambda r: r["filed"])[-1]["val"])


C = {}
for t in BANKS:
    r = raw.loc[t]
    E = pd.Timestamp(r["_stmt_date_balance_sheet"])
    f_t = facts[facts["ticker"] == t]
    se_sec, _, _ = sx.sec_value(f_t, ("StockholdersEquity",), "instant", E) if not f_t.empty else (None, None, None)
    sh_sec, _, _ = sx.sec_value(f_t, ("CommonStockSharesOutstanding",), "instant", E) if not f_t.empty else (None, None, None)
    st = S["tickers"].get(t, {})
    gw = sec_instant(st.get("Goodwill"), E)
    ia = sec_instant(st.get("IntangibleAssetsNetExcludingGoodwill"), E)
    pf = sec_instant(st.get("PreferredStockValue"), E)
    if pf is None:
        pf = sec_instant(st.get("PreferredStockValueOutstanding"), E)
    tce = (se_sec - (pf or 0) - (gw or 0) - (ia or 0)) if se_sec is not None else None
    sh = r["sharesBS"] * (1500 if t == "BRK-B" else 1)
    C[t] = {
        "yahoo_pb": r["priceToBook"],
        "run_mc_over_SE": r["marketCap"] / r["totalEquity"],
        "run_price_over_SE_per_bs_share": r["currentPrice"] / (r["totalEquity"] / sh),
        "probe_mc_over_CE": r["marketCap"] / Bd.loc[t, "CE"] if t in Bd.index else np.nan,
        "probe_mc_over_TBV": r["marketCap"] / Bd.loc[t, "TBV"] if t in Bd.index else np.nan,
        "sec_mc_over_SE": r["marketCap"] / se_sec if se_sec else np.nan,
        "sec_price_over_SE_per_filed_share": r["currentPrice"] / (se_sec / sh_sec) if se_sec and sh_sec else np.nan,
        "sec_mc_over_TCE": r["marketCap"] / tce if tce else np.nan,
        "sec_goodwill_tagged": gw is not None, "sec_intangibles_tagged": ia is not None,
        "sec_preferred_tagged": pf is not None,
        "run_SE_vs_sec_SE": rel(r["totalEquity"], se_sec) if se_sec else np.nan,
        "probe_CE_vs_sec_SE_less_preferred": rel(Bd.loc[t, "CE"], se_sec - (pf or 0)) if se_sec and t in Bd.index else np.nan,
        "marketCap_over_price_over_bs_shares": r["marketCap"] / r["currentPrice"] / sh,
        "probe_TBV_vs_sec_TCE": rel(Bd.loc[t, "TBV"], tce) if tce and t in Bd.index else np.nan,
    }
Cd = pd.DataFrame(C).T
for c in Cd.columns:
    if not c.startswith("sec_") or c.endswith(("_SE", "_share", "_TCE")):
        Cd[c] = pd.to_numeric(Cd[c], errors="coerce")
ex = [t for t in Cd.index if t != "BRK-B"]


def agree(a, b, idx=None):
    sub = Cd.loc[idx or Cd.index, [a, b]].astype(float).dropna()
    g = np.array([rel(x, y) for x, y in sub.values])
    return {"n": len(sub), "within_2pct": int((g <= 0.02).sum()), "within_5pct": int((g <= 0.05).sum()),
            "over_10pct": {t: rnd(v, 3) for t, v in zip(sub.index, g) if v > 0.10},
            "spearman": rnd(spearmanr(sub[a], sub[b]).statistic)}


C_out = {
    "run_equity_vs_filed_equity_within_2pct": f'{int((Cd["run_SE_vs_sec_SE"].astype(float) <= 0.02).sum())} of {int(Cd["run_SE_vs_sec_SE"].notna().sum())}',
    "probe_common_equity_vs_filed_equity_less_preferred": {
        "within_2pct": f'{int((Cd["probe_CE_vs_sec_SE_less_preferred"].astype(float) <= 0.02).sum())} of {int(Cd["probe_CE_vs_sec_SE_less_preferred"].notna().sum())}',
        "over_2pct": {t: rnd(v, 3) for t, v in Cd["probe_CE_vs_sec_SE_less_preferred"].astype(float).items() if v > 0.02}},
    "marketCap_not_price_x_balance_sheet_shares (outside 0.95-1.05)": {
        t: rnd(v, 3) for t, v in Cd["marketCap_over_price_over_bs_shares"].astype(float).items() if not 0.95 <= v <= 1.05},
    "earnings_yield_of_these (net income / marketCap, as scored)": {
        t: rnd(raw.loc[t, "netIncome"] / raw.loc[t, "marketCap"], 4) for t in ("IBKR", "BRK-B")},
    "yahoo vs run mc/SE (excl. BRK-B)": agree("yahoo_pb", "run_mc_over_SE", ex),
    "run mc/SE vs run price/(SE/bs shares)": agree("run_mc_over_SE", "run_price_over_SE_per_bs_share"),
    "run mc/SE vs filings mc/SE": agree("run_mc_over_SE", "sec_mc_over_SE"),
    "filings mc/SE vs filings price/(SE/filed shares)": agree("sec_mc_over_SE", "sec_price_over_SE_per_filed_share"),
    "yahoo vs probe mc/CE (excl. BRK-B)": agree("yahoo_pb", "probe_mc_over_CE", ex),
    "probe mc/CE vs run mc/SE": agree("probe_mc_over_CE", "run_mc_over_SE"),
    "probe mc/TBV vs filings mc/TCE": agree("probe_mc_over_TBV", "sec_mc_over_TCE"),
    "probe mc/TBV vs probe mc/CE": agree("probe_mc_over_TBV", "probe_mc_over_CE"),
    "tagged_in_filings": {k: int(Cd[f"sec_{k}_tagged"].astype(bool).sum()) for k in ("goodwill", "intangibles", "preferred")},
    "per_stock": {t: {k: (rnd(v, 4) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in row.items()}
                  for t, row in Cd.iterrows()},
}
out["C_computed"] = C_out

# ---- D. the stakes ---------------------------------------------------------------------
RUN_CFG = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
_eff = json.loads((RUN / "effective_weights.json").read_text(encoding="utf-8"))
RUN_CFG["factor_weights"] = _eff["factor_weights"]
RUN_CFG["metric_weights"] = _eff["metric_weights"]


def score(pb=None):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = F02.copy()
        if pb is not None:
            v = df["Ticker"].map(pb)
            bank = df["_is_bank_like"].fillna(False).astype(bool)
            df.loc[bank, "pb_ratio"] = v[bank]
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, RUN_CFG)
        df = fe.compute_category_scores(df, RUN_CFG)
        df = fe.compute_composite(df, RUN_CFG)
        return fe.rank_stocks(df).set_index("Ticker")


def positive(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s > 0)


brk = Cd["yahoo_pb"].copy()
brk["BRK-B"] = Cd.loc["BRK-B", "run_price_over_SE_per_bs_share"]
ARMS = {"BRK_ONLY": positive(brk), "MC_SE": positive(Cd["run_mc_over_SE"]),
        "MC_CE": positive(Cd["probe_mc_over_CE"]), "MC_TBV": positive(Cd["probe_mc_over_TBV"])}
a = score()
out["baseline_reproduces_published_ranks"] = f'{int((a["Rank"] == published.loc[a.index, "Rank"]).sum())} of {len(a)}'
top_a = set(a.nsmallest(25, "Rank").index)
D = {}
for k, pb in ARMS.items():
    b = score(pb.to_dict())
    bk = [t for t in BANKS if t in a.index]
    mv = (a.loc[bk, "Rank"] - b.loc[bk, "Rank"]).astype(int)
    comp = b.loc[bk, "Composite"] - a.loc[bk, "Composite"]
    pct = (b.loc[bk, "pb_ratio_pct"] - a.loc[bk, "pb_ratio_pct"]).abs()
    top_b = set(b.nsmallest(25, "Rank").index)
    D[k] = {
        "missing": sorted(t for t in BANKS if not np.isfinite(pb.get(t, np.nan))),
        "pb_spearman_vs_yahoo": rnd(spearmanr(a.loc[bk, "pb_ratio"], b.loc[bk, "pb_ratio"], nan_policy="omit").statistic),
        "pb_spearman_vs_yahoo_excl_BRK": rnd(spearmanr(a.loc[[t for t in bk if t != "BRK-B"], "pb_ratio"],
                                                       b.loc[[t for t in bk if t != "BRK-B"], "pb_ratio"], nan_policy="omit").statistic),
        "pb_pct_moves_over_10": {t: rnd(v, 1) for t, v in pct[pct > 10].sort_values(ascending=False).items()},
        "universe_composite_spearman": rnd(spearmanr(a["Composite"], b.loc[a.index, "Composite"]).statistic, 5),
        "bank_like_median_abs_rank_move": float(mv.abs().median()),
        "bank_like_moved_10_or_more": int((mv.abs() >= 10).sum()),
        "bank_like_moved_25_or_more": int((mv.abs() >= 25).sum()),
        "bank_like_max_composite_points": rnd(comp.abs().max(), 2),
        "biggest_moves": {t: {"from": int(a.loc[t, "Rank"]), "to": int(b.loc[t, "Rank"]), "composite": rnd(comp[t], 2),
                              "pb": f'{rnd(a.loc[t, "pb_ratio"], 3)} -> {rnd(b.loc[t, "pb_ratio"], 3)}'}
                          for t in mv.abs().sort_values(ascending=False).head(8).index},
        "top25_out": sorted(top_a - top_b), "top25_in": sorted(top_b - top_a),
        "best_bank_like_before": [f"{t}#{int(a.loc[t, 'Rank'])}" for t in a.loc[bk].nsmallest(6, "Rank").index],
        "best_bank_like_after": [f"{t}#{int(b.loc[t, 'Rank'])}" for t in b.loc[bk].nsmallest(6, "Rank").index],
        "BRK-B": {"rank": f'{int(a.loc["BRK-B", "Rank"])} -> {int(b.loc["BRK-B", "Rank"])}',
                  "pb_pct": f'{rnd(a.loc["BRK-B", "pb_ratio_pct"], 1)} -> {rnd(b.loc["BRK-B", "pb_ratio_pct"], 1)}',
                  "valuation": f'{rnd(a.loc["BRK-B", "valuation_score"], 1)} -> {rnd(b.loc["BRK-B", "valuation_score"], 1)}',
                  "composite": f'{rnd(a.loc["BRK-B", "Composite"], 2)} -> {rnd(b.loc["BRK-B", "Composite"], 2)}'},
    }
out["D_stakes"] = D

# ---- E. Berkshire's P/B percentile across the snapshots -----------------------------------
hist = []
for p in sorted(glob.glob(str(ROOT / "improvement" / "snapshots" / "*.parquet"))):
    sn = pd.read_parquet(p).set_index("Ticker")
    if "BRK-B" in sn.index and "pb_ratio_pct" in sn.columns:
        hist.append((Path(p).stem.split("_")[0], rnd(sn.loc["BRK-B", "pb_ratio_pct"], 1), int(sn.loc["BRK-B", "Rank"])))
pcts = [h[1] for h in hist if h[1] is not None]
out["E_brk_history"] = {"snapshots": len(hist), "pb_pct_min": min(pcts) if pcts else None,
                        "pb_pct_median": float(np.median(pcts)) if pcts else None,
                        "first": hist[0] if hist else None, "last": hist[-1] if hist else None,
                        "best_rank": min(h[2] for h in hist) if hist else None}

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "C_computed"}, indent=1, default=str))
print(json.dumps({k: v for k, v in C_out.items() if k != "per_stock"}, indent=1, default=str))
