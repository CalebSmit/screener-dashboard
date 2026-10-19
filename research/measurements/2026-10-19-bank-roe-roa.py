"""Bank ROE and ROA: what Yahoo's ratios are, and what scoring them on the run's own figures would move.

Question (research/2026-10-19-bank-roe-roa.md): 60% of bank Quality - ROE 35 + ROA 25 of the bank
Quality table, about 13.2 composite points for each bank-like stock - is read from Yahoo's
``returnOnEquity`` / ``returnOnAssets`` (``.info``), two ratios with no published definition. The
engine dropped ``earningsGrowth`` for exactly that reason. This script:

  probe   (network) for every bank-like stock in the newest full run, the source's ``.info`` ratios
          and every quarterly / annual statement row an ROE or ROA could be built from; written to
          ``2026-10-19-bank-roe-roa-probe.json`` beside this file, so everything below re-runs
          offline.
  (none)  offline:
     A. the audit's comparison, reproduced on the newest run: Yahoo's ratios against net income
        over the average of the two balance sheets the run already fetches;
     B. which definition Yahoo's ratios follow - every candidate numerator x denominator x
        averaging rule, scored by how many stocks it reproduces within 0.5% (probe data);
     C. the same ratios rebuilt from the companies' own SEC filings (``data/sec/pit/facts.parquet``),
        for the quarter the run's balance sheet ends;
     D. the stakes: the run's own metric table (02_outliers_flagged) rescored with the run's own
        configuration and effective weights - the baseline reproduces all 501 published ranks - with
        only the bank-like stocks' roe / roa replaced:
          AVG2    net income / average of equity (assets) now and a year earlier  (the proposal)
          END     net income / ending equity (assets)  (today's fallback, for everyone)
          AVG5    net income / average of the five quarter-ends (probe balance sheets)
          COMMON  net income to common / average common equity (probe rows); ROA as AVG2

Usage:
    python research/measurements/2026-10-19-bank-roe-roa.py probe   # network, ~46 requests x 4
    python research/measurements/2026-10-19-bank-roe-roa.py         # offline; JSON beside this file
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
PROBE = HERE.with_name(HERE.stem + "-probe.json")
OUT = HERE.with_suffix(".json")

BS_ROWS = ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest",
           "Preferred Stock Equity", "Minority Interest", "Total Assets", "Tangible Book Value"]
IS_ROWS = ["Net Income", "Net Income Common Stockholders", "Preferred Stock Dividends",
           "Net Income Including Noncontrolling Interests", "Minority Interests",
           "Net Income From Continuing Operation Net Minority Interest", "Pretax Income", "EBIT",
           "Operating Income", "Tax Provision", "Interest Expense", "Total Revenue",
           "Normalized Income"]
INFO = ["returnOnEquity", "returnOnAssets", "mostRecentQuarter", "netIncomeToCommon", "bookValue",
        "sharesOutstanding", "lastFiscalYearEnd", "priceToBook"]


def newest_full_run():
    runs = [p for p in glob.glob(str(ROOT / "runs" / "*" / "00_raw_fetch.parquet"))]
    return Path(sorted(runs, key=os.path.getmtime)[-1]).parent


def bank_like(run):
    m = pd.read_parquet(run / "01_raw_metrics.parquet")
    return sorted(m.loc[m["_is_bank_like"] == True, "Ticker"].tolist())  # noqa: E712


# ---------------------------------------------------------------------------------------- probe
def _frame(df, rows):
    if df is None or df.empty:
        return {}
    out = {}
    for r in rows:
        if r not in df.index:
            continue
        s = df.loc[r]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        out[r] = {str(pd.Timestamp(c).date()): (None if pd.isna(v) else float(v)) for c, v in s.items()}
    return out


def probe():
    import yfinance as yf
    run = newest_full_run()
    tickers = bank_like(run)
    data = {"run": run.name, "probed_at": time.strftime("%Y-%m-%d %H:%M"), "tickers": {}}
    for t in tickers:
        for attempt in range(3):
            try:
                k = yf.Ticker(t)
                info = k.info or {}
                data["tickers"][t] = {
                    "info": {f: info.get(f) for f in INFO},
                    "q_bs": _frame(k.quarterly_balance_sheet, BS_ROWS),
                    "q_is": _frame(k.quarterly_income_stmt, IS_ROWS),
                    "a_bs": _frame(k.balance_sheet, BS_ROWS),
                    "a_is": _frame(k.income_stmt, IS_ROWS),
                }
                break
            except Exception as e:  # network: retry, then record the failure
                if attempt == 2:
                    data["tickers"][t] = {"error": f"{type(e).__name__}: {e}"}
                time.sleep(3)
        time.sleep(0.5)
        print(t, "ok" if "error" not in data["tickers"][t] else data["tickers"][t]["error"])
    PROBE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", PROBE)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "probe":
    probe()
    sys.exit(0)

# ---------------------------------------------------------------------------------- offline
from scipy.stats import spearmanr  # noqa: E402

import factor_engine as fe  # noqa: E402
import sec_crosscheck as sx  # noqa: E402

RUN = newest_full_run()
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
raw = pd.read_parquet(RUN / "00_raw_fetch.parquet")
rawi = raw.set_index("Ticker")
BANKS = bank_like(RUN)
P = json.loads(PROBE.read_text(encoding="utf-8")) if PROBE.exists() else {"tickers": {}}
out = {"run": RUN.name, "probe_taken": P.get("probed_at"), "bank_like": len(BANKS)}
W_ROE = cfg["bank_metric_weights"]["quality"]["roe"]
W_ROA = cfg["bank_metric_weights"]["quality"]["roa"]
W_Q = cfg["factor_weights"]["quality"]
out["composite_points_on_yahoo_ratios"] = round((W_ROE + W_ROA) / 100 * W_Q, 2)


def rnd(v, n=4):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b)) if np.isfinite(a) and np.isfinite(b) and max(abs(a), abs(b)) > 0 else np.nan


# ---- A. the audit's comparison, on the run's own fields --------------------------------
x = rawi.loc[BANKS]
built = pd.DataFrame({
    "yahoo_roe": x["returnOnEquity"], "yahoo_roa": x["returnOnAssets"],
    "roe_avg": x["netIncome"] / ((x["totalEquity"] + x["totalEquity_prior"]) / 2),
    "roa_avg": x["netIncome"] / ((x["totalAssets"] + x["totalAssets_prior"]) / 2),
    "roe_end": x["netIncome"] / x["totalEquity"], "roa_end": x["netIncome"] / x["totalAssets"],
})
A = {}
for m in ("roe", "roa"):
    both = built[[f"yahoo_{m}", f"{m}_avg"]].dropna()
    A[m] = {"both": len(both),
            "spearman_yahoo_vs_avg": rnd(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic),
            "within_0.5pct": int(sum(rel(a, b) <= 0.005 for a, b in both.values)),
            "within_2pct": int(sum(rel(a, b) <= 0.02 for a, b in both.values)),
            "yahoo_missing": sorted(built.index[built[f"yahoo_{m}"].isna()]),
            "avg_missing": sorted(built.index[built[f"{m}_avg"].isna()])}
A["named"] = {t: {k: rnd(v) for k, v in built.loc[t].items()} for t in ("ALL", "KKR", "IBKR", "APO", "MS", "BRK-B", "JPM")}
out["A_run_fields"] = A


# ---- B. which definition the source's ratios follow (probe) ----------------------------
def ser(block, row):
    s = (block or {}).get(row, {})
    return pd.Series({pd.Timestamp(k): v for k, v in s.items() if v is not None}, dtype=float).sort_index(ascending=False)


def year_ago(idx, s):
    c = [d for d in idx if 330 <= (s - d).days <= 400]
    return c[0] if c else None


NUMS = {"NI": "Net Income", "NIC": "Net Income Common Stockholders",
        "NINCI": "Net Income Including Noncontrolling Interests"}
DENS = {"SE": "Stockholders Equity", "CE": "Common Stock Equity",
        "TE": "Total Equity Gross Minority Interest", "TA": "Total Assets"}
rows, aligned_flag, fit = {}, {}, {}
for t, d in P["tickers"].items():
    if "error" in d:
        continue
    info = d["info"]
    ni = ser(d["q_is"], "Net Income")
    if len(ni) < 4:
        continue
    S = ni.index[0]
    mrq = pd.Timestamp(info["mostRecentQuarter"], unit="s").normalize() if info.get("mostRecentQuarter") else None
    aligned_flag[t] = bool(mrq is not None and mrq == S)
    r = {}
    for nk, nrow in NUMS.items():
        n = ser(d["q_is"], nrow)
        if len(n) >= 4 and n.index[0] == S:
            num = n.iloc[:4].sum()
        else:
            continue
        for dk, drow in DENS.items():
            e = ser(d["q_bs"], drow)
            if S not in e.index:
                continue
            ya = year_ago(e.index, S)
            target = info.get("returnOnAssets") if dk == "TA" else info.get("returnOnEquity")
            if target is None:
                continue
            ms = "roa" if dk == "TA" else "roe"
            cands = {"end": e[S]}
            if ya is not None:
                cands["avg2"] = (e[S] + e[ya]) / 2
                cands["begin"] = e[ya]
                win = e[(e.index >= ya) & (e.index <= S)]
                if len(win) >= 5:
                    cands["avg5"] = win.mean()
            for ck, den in cands.items():
                if den and den > 0:
                    r[f"{ms}:{nk}/{dk}_{ck}"] = rel(num / den, target)
    # the latest fiscal year, for the cases where the trailing window fits nothing
    ani, ata = ser(d["a_is"], "Net Income"), ser(d["a_bs"], "Total Assets")
    ase = ser(d["a_bs"], "Stockholders Equity")
    if len(ani) and len(ata) >= 2 and info.get("returnOnAssets"):
        r["roa:FY NI/TA_avg2(FY)"] = rel(ani.iloc[0] / ((ata.iloc[0] + ata.iloc[1]) / 2), info["returnOnAssets"])
    if len(ani) and len(ase) >= 2 and info.get("returnOnEquity"):
        r["roe:FY NI/SE_avg2(FY)"] = rel(ani.iloc[0] / ((ase.iloc[0] + ase.iloc[1]) / 2), info["returnOnEquity"])
    rows[t] = r
cand = pd.DataFrame(rows).T
B = {"probed": len(rows),
     "info_quarter_newer_than_statements": sorted(t for t, a in aligned_flag.items() if not a),
     "aligned": sum(aligned_flag.values())}
for ms in ("roe", "roa"):
    cols = [c for c in cand.columns if c.startswith(ms + ":")]
    tab = {}
    for grp, mask in (("aligned", [aligned_flag[t] for t in cand.index]),
                      ("info_newer", [not aligned_flag[t] for t in cand.index])):
        sub = cand.loc[mask, cols]
        tab[grp] = sorted(([c.split(":", 1)[1], int(sub[c].notna().sum()), int((sub[c] <= 0.005).sum()),
                            int((sub[c] <= 0.02).sum())] for c in cols), key=lambda z: -z[2])[:6]
    B[ms + "_top_candidates [definition, compared, within 0.5%, within 2%]"] = tab
    al = cand.loc[[aligned_flag[t] for t in cand.index], cols]
    B[ms + "_aligned_best_fit_per_stock"] = {t: [al.loc[t].idxmin().split(":", 1)[1], rnd(al.loc[t].min())]
                                              for t in al.index if al.loc[t].notna().any()}
_ins = rawi["industry"].fillna("").str.startswith("Insurance") | rawi.index.isin(["AMP", "PFG"])
for ms, c in (("roe", "roe:NINCI/TE_avg2"), ("roa", "roa:NINCI/TA_avg2")):
    al = cand.loc[[aligned_flag[t] for t in cand.index], c]
    B[f"{ms}_aligned_fit_of_{c.split(':')[1]}"] = {
        grp: {"n": int(len(s)), "within_0.5pct": int((s <= 0.005).sum()), "within_2pct": int((s <= 0.02).sum()),
              "misses": {t: rnd(v) for t, v in s[s > 0.005].sort_values(ascending=False).items()}}
        for grp, s in (("insurers (incl. AMP, PFG)", al[_ins.reindex(al.index).values]),
                       ("everyone else", al[~_ins.reindex(al.index).values]))}
out["B_source_definition"] = B


# preferred stock and minority interests: how far 'total' and 'common' bases differ
pref = {}
for t, d in P["tickers"].items():
    if "error" in d:
        continue
    se, ps = ser(d["q_bs"], "Stockholders Equity"), ser(d["q_bs"], "Preferred Stock Equity")
    ni, nic = ser(d["q_is"], "Net Income"), ser(d["q_is"], "Net Income Common Stockholders")
    if len(se) and len(ps) and se.index[0] in ps.index:
        pref[t] = {"preferred_share_of_equity": rnd(ps[se.index[0]] / se.iloc[0]),
                   "common_share_of_net_income": rnd(nic.iloc[:4].sum() / ni.iloc[:4].sum()) if len(nic) >= 4 and len(ni) >= 4 else None}
out["preferred"] = {"with_preferred": len(pref),
                    "preferred_over_5pct_of_equity": sorted(t for t, v in pref.items() if (v["preferred_share_of_equity"] or 0) > 0.05),
                    "detail": pref}


# ---- C. the same ratios from the companies' own filings --------------------------------
facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
C = {}
for t in BANKS:
    f_t = facts[facts["ticker"] == t]
    E = rawi.loc[t, "_stmt_date_balance_sheet"]
    if f_t.empty or not isinstance(E, str):
        continue
    E = pd.Timestamp(E)
    ni_s, _, _ = sx.sec_value(f_t, ("NetIncomeLoss", "ProfitLoss"), "flow", E)
    eq0, _, _ = sx.sec_value(f_t, fe_eq := ("StockholdersEquity",), "instant", E)
    eq1, _, _ = sx.sec_value(f_t, fe_eq, "instant", E - pd.Timedelta(days=365))
    ta0, _, _ = sx.sec_value(f_t, ("Assets",), "instant", E)
    ta1, _, _ = sx.sec_value(f_t, ("Assets",), "instant", E - pd.Timedelta(days=365))
    if eq1 is None:   # year-end a year ago is sometimes dated a day or two away
        eq1, _, _ = sx.sec_value(f_t, fe_eq, "instant", E - pd.Timedelta(days=366))
        ta1, _, _ = sx.sec_value(f_t, ("Assets",), "instant", E - pd.Timedelta(days=366))
    if None in (ni_s, eq0, eq1, ta0, ta1):
        continue
    C[t] = {"sec_roe_avg": ni_s / ((eq0 + eq1) / 2), "sec_roa_avg": ni_s / ((ta0 + ta1) / 2),
            "run_roe_avg": built.loc[t, "roe_avg"], "run_roa_avg": built.loc[t, "roa_avg"],
            "yahoo_roe": built.loc[t, "yahoo_roe"], "yahoo_roa": built.loc[t, "yahoo_roa"]}
Cd = pd.DataFrame(C).T.astype(float)
sec = {"compared": len(Cd)}
for m in ("roe", "roa"):
    for other in (f"run_{m}_avg", f"yahoo_{m}"):
        pair = Cd[[f"sec_{m}_avg", other]].dropna()
        g = np.array([rel(a, b) for a, b in pair.values])
        sec[f"{m}: filings vs {other}"] = {"n": len(pair), "within_2pct": int((g <= 0.02).sum()),
                                           "over_10pct": sorted(pair.index[g > 0.10]),
                                           "spearman": rnd(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)}
out["C_filings"] = sec


# ---- D. the stakes ---------------------------------------------------------------------
published = pd.read_parquet(RUN / "05_final_scored.parquet").set_index("Ticker")
# the stocks the run scored (the fetch failures are dropped before percentiles, as in the run)
base = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        for r in raw.to_dict("records") if r["Ticker"] in published.index]


def five_quarter(t, row):
    d = P["tickers"].get(t, {})
    e = ser(d.get("q_bs"), row)
    S = rawi.loc[t, "_stmt_date_balance_sheet"]
    if not isinstance(S, str) or pd.Timestamp(S) not in e.index:
        return np.nan
    S = pd.Timestamp(S)
    ya = year_ago(e.index, S)
    win = e[(e.index >= ya) & (e.index <= S)] if ya is not None else e.iloc[:0]
    return win.mean() if len(win) >= 5 else np.nan


def common_basis(t):
    """TTM net income to common / average common equity (now and a year earlier), from the probe,
    only where the probe's statements end on the run's balance-sheet date."""
    d = P["tickers"].get(t, {})
    S = rawi.loc[t, "_stmt_date_balance_sheet"]
    nic, ce = ser(d.get("q_is"), "Net Income Common Stockholders"), ser(d.get("q_bs"), "Common Stock Equity")
    if not isinstance(S, str) or len(nic) < 4 or nic.index[0] != pd.Timestamp(S) or pd.Timestamp(S) not in ce.index:
        return np.nan
    ya = year_ago(ce.index, pd.Timestamp(S))
    return nic.iloc[:4].sum() / ((ce[pd.Timestamp(S)] + ce[ya]) / 2) if ya is not None else np.nan


ARMS = {
    "AVG2": lambda r: (r["netIncome"] / ((r["totalEquity"] + r["totalEquity_prior"]) / 2),
                      r["netIncome"] / ((r["totalAssets"] + r["totalAssets_prior"]) / 2)),
    "END": lambda r: (r["netIncome"] / r["totalEquity"], r["netIncome"] / r["totalAssets"]),
    "AVG5": lambda r: (r["netIncome"] / five_quarter(r["Ticker"], "Stockholders Equity"),
                      r["netIncome"] / five_quarter(r["Ticker"], "Total Assets")),
    # ROE to common shareholders (JPM's and MSCI's basis); ROA as AVG2
    "COMMON": lambda r: (common_basis(r["Ticker"]),
                         r["netIncome"] / ((r["totalAssets"] + r["totalAssets_prior"]) / 2)),
}


# Scored from the run's own metric table (02_outliers_flagged: after compute_metrics and the
# run's filters, before percentiles), with the run's own configuration and effective weights,
# so the baseline reproduces the published ranking and only roe / roa differ between arms.
RUN_CFG = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
_eff = json.loads((RUN / "effective_weights.json").read_text(encoding="utf-8"))
RUN_CFG["factor_weights"] = _eff["factor_weights"]
RUN_CFG["metric_weights"] = _eff["metric_weights"]
FLAGGED = pd.read_parquet(RUN / "02_outliers_flagged.parquet")


def score(override=None):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = RUN_CFG
        df = FLAGGED.copy()
        if override is not None:
            for m in ("roe", "roa"):
                v = df["Ticker"].map(override[m])
                bank = df["_is_bank_like"].fillna(False).astype(bool)
                df.loc[bank, m] = v[bank]
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, cfg)
        df = fe.compute_category_scores(df, cfg)
        df = fe.compute_composite(df, cfg)
        return fe.rank_stocks(df).set_index("Ticker")


def arm_values(fn):
    ov = {"roe": {}, "roa": {}}
    for r in base:
        if r["Ticker"] not in BANKS:
            continue
        e, a = fn(r)
        ov["roe"][r["Ticker"]] = e if np.isfinite(e) and r["totalEquity"] > 0 else np.nan
        ov["roa"][r["Ticker"]] = a if np.isfinite(a) else np.nan
    return ov


a = score()
out["baseline_reproduces_published_ranks"] = f'{int((a["Rank"] == published.loc[a.index, "Rank"]).sum())} of {len(a)}'
top_a = set(a.nsmallest(25, "Rank").index)
D = {}
for k, fn in ARMS.items():
    ov = arm_values(fn)
    b = score(ov)
    bk = [t for t in BANKS if t in a.index and t in b.index]
    mv = (a.loc[bk, "Rank"] - b.loc[bk, "Rank"]).astype(int)
    comp = (b.loc[bk, "Composite"] - a.loc[bk, "Composite"])
    roe_pct = (b.loc[bk, "roe_pct"] - a.loc[bk, "roe_pct"]).abs()
    roa_pct = (b.loc[bk, "roa_pct"] - a.loc[bk, "roa_pct"]).abs()
    top_b = set(b.nsmallest(25, "Rank").index)
    D[k] = {
        "missing_roe": sorted(t for t, v in ov["roe"].items() if not np.isfinite(v)),
        "missing_roa": sorted(t for t, v in ov["roa"].items() if not np.isfinite(v)),
        "roe_spearman_vs_yahoo": rnd(spearmanr(a.loc[bk, "roe"], b.loc[bk, "roe"], nan_policy="omit").statistic),
        "roa_spearman_vs_yahoo": rnd(spearmanr(a.loc[bk, "roa"], b.loc[bk, "roa"], nan_policy="omit").statistic),
        "roe_pct_moves_over_10": {t: rnd(v, 1) for t, v in roe_pct[roe_pct > 10].sort_values(ascending=False).items()},
        "roa_pct_moves_over_10": {t: rnd(v, 1) for t, v in roa_pct[roa_pct > 10].sort_values(ascending=False).items()},
        "universe_composite_spearman": rnd(spearmanr(a["Composite"], b.loc[a.index, "Composite"]).statistic, 5),
        "bank_like_median_abs_rank_move": float(mv.abs().median()),
        "bank_like_moved_10_or_more": int((mv.abs() >= 10).sum()),
        "bank_like_moved_25_or_more": int((mv.abs() >= 25).sum()),
        "bank_like_max_composite_points": rnd(comp.abs().max(), 2),
        "biggest_moves": {t: {"from": int(a.loc[t, "Rank"]), "to": int(b.loc[t, "Rank"]),
                              "composite": rnd(comp[t], 2)} for t in mv.abs().sort_values(ascending=False).head(8).index},
        "named": {t: {"rank": f"{int(a.loc[t, 'Rank'])} -> {int(b.loc[t, 'Rank'])}",
                      "roe": f"{rnd(a.loc[t, 'roe'])} -> {rnd(b.loc[t, 'roe'])}",
                      "roa": f"{rnd(a.loc[t, 'roa'])} -> {rnd(b.loc[t, 'roa'])}",
                      "quality": f"{rnd(a.loc[t, 'quality_score'], 1)} -> {rnd(b.loc[t, 'quality_score'], 1)}"}
                  for t in ("BRK-B", "KKR", "IBKR", "APO", "ALL", "MS", "JPM")},
        "top25_out": sorted(top_a - top_b), "top25_in": sorted(top_b - top_a),
        "best_bank_like_before": [f"{t}#{int(a.loc[t, 'Rank'])}" for t in a.loc[bk].nsmallest(5, "Rank").index],
        "best_bank_like_after": [f"{t}#{int(b.loc[t, 'Rank'])}" for t in b.loc[bk].nsmallest(5, "Rank").index],
    }
out["D_stakes"] = D
_a2, _a5 = arm_values(ARMS["AVG2"]), arm_values(ARMS["AVG5"])
out["avg2_vs_avg5_spearman"] = {
    m: rnd(spearmanr(*pd.DataFrame({"a2": _a2[m], "a5": _a5[m]}).dropna().T.values).statistic) for m in ("roe", "roa")}
# the probe's .info ratios against the run's: the probe describes what the run scored
_same = [(t, rawi.loc[t, f], P["tickers"][t]["info"][f]) for t in BANKS for f in ("returnOnEquity", "returnOnAssets")
         if t in P["tickers"] and "info" in P["tickers"][t]]
out["probe_info_equals_run"] = {
    "compared": sum(1 for _, r, p in _same if pd.notna(r) and p is not None),
    "equal_to_1e-6": sum(1 for _, r, p in _same if pd.notna(r) and p is not None and abs(r - p) < 1e-6),
    "missing_both": sorted({t for t, r, p in _same if pd.isna(r) and p is None})}
_t, _c = _a2["roe"], arm_values(ARMS["COMMON"])["roe"]
_pair = pd.DataFrame({"total": _t, "common": _c}).dropna()
out["roe_total_vs_common_basis"] = {
    "both": len(_pair), "spearman": rnd(spearmanr(_pair["total"], _pair["common"]).statistic),
    "median_ratio_common_to_total": rnd(float((_pair["common"] / _pair["total"]).median())),
    "max_ratio": {t: rnd(v) for t, v in (_pair["common"] / _pair["total"]).sort_values(ascending=False).head(4).items()},
    "common_missing": sorted(t for t, v in _c.items() if not np.isfinite(v))}

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
