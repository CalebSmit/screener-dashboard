"""Review of the bank ROE/ROA proposal (research/2026-10-19-bank-roe-roa.md section 7).

The proposal: for bank-like stocks, score

    roe = netIncome / ((totalEquity   + totalEquity_prior)   / 2)
    roa = netIncome / ((totalAssets   + totalAssets_prior)   / 2)

both > 0 required else NaN, with NO fallback to the source's `.info` returnOnEquity /
returnOnAssets and none to an ending balance.

This script is the REVIEWER's measurement, not the proposer's. It re-measures the one thing
that can sink the change - is the input actually there, for every stock, on the same basis -
on a SECOND day's run and a FRESH probe, because that is exactly the test the
earnings-definition proposal failed on 2026-10-16 (23 companies with no figure at all) and
the test the price-to-book proposal passed on 2026-10-20 (46 of 46).

Four things it measures that the note could not:

  A. Availability on the 2026-10-20 run, which is the first that carries
     `factor_engine._year_ago_position` (made late on 2026-10-19, after the note). The note
     says Berkshire "loses both ratios under any averaged construction" because its year-ago
     balance sheet was taken by column position. If that is still true for any stock, the
     change creates a no-ROE cohort inside bank Quality and should be refused or amended.
  B. The gap in days between the two balance sheets actually used, from a fresh probe - the
     averaging claim is only true if the second sheet is a year earlier.
  C. The proposed ratios against the companies' own SEC filings (the note: 38 of 39 within
     2%; Yahoo's 23 and 15 of 39).
  D. The stakes on TODAY's run, whose baseline already contains this morning's bank P/B
     change - so the note's section 6 numbers cannot simply be carried over - plus the one
     judgement call the note asks the reviewer to settle: ROA's minority-interest basis.

    python research/measurements/2026-10-20-bank-roe-roa-review.py probe   # network
    python research/measurements/2026-10-20-bank-roe-roa-review.py         # offline
"""
from __future__ import annotations

import contextlib
import io
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
HERE = Path(__file__).resolve()
PROBE = HERE.with_name(HERE.stem + "-probe.json")
OUT = HERE.with_suffix(".json")

BS_ROWS = ("Stockholders Equity", "Total Stockholder", "Total Assets",
           "Common Stock Equity", "Minority Interest",
           "Total Equity Gross Minority Interest")
INFO = ("returnOnEquity", "returnOnAssets", "mostRecentQuarter")


def newest_full_run() -> Path:
    import run_context
    return Path(run_context.latest_run_dir())


def bank_like(run: Path) -> list:
    f02 = pd.read_parquet(run / "02_outliers_flagged.parquet")
    return sorted(f02.loc[f02["_is_bank_like"].fillna(False).astype(bool), "Ticker"])


# ------------------------------------------------------------------------------------- probe
def probe():
    """Record, for each bank-like stock, the two balance-sheet columns the engine would use.

    Uses the engine's own helpers (`_populated_columns`, `_year_ago_position`, `_stmt_val`) so
    the probe describes what `compute_metrics` would read, not a second interpretation of it.
    """
    import yfinance as yf

    import factor_engine as fe
    run = newest_full_run()
    tickers = bank_like(run)
    data = {"run": run.name, "probed_at": time.strftime("%Y-%m-%d %H:%M"), "tickers": {}}
    for t in tickers:
        rec = None
        for attempt in range(3):
            try:
                k = yf.Ticker(t)
                info = k.info or {}
                q_bs = k.quarterly_balance_sheet
                cols = fe._populated_columns(q_bs) if (q_bs is not None and not q_bs.empty) else []
                j = (fe._year_ago_position(cols, tol=fe.YEAR_AGO_TOL_BALANCE_DAYS)
                     if cols else None)
                def val(lbl, col=0):
                    # col is an index, not optional: _stmt_val compares len(vals) > col, so
                    # passing None raises rather than meaning "the newest column".
                    v = fe._stmt_val(q_bs, lbl, col)
                    return None if not np.isfinite(v) else float(v)
                rec = {
                    "info": {f: info.get(f) for f in INFO},
                    # _find_stmt_label is fuzzy: asked for "Minority Interest" a company that
                    # has no such row answers with "Total Equity Gross Minority Interest", which
                    # for a company with no minority holders equals stockholders' equity (it
                    # read 1.000 for 25 of 46 on the first pass). Record which labels the
                    # statement actually carries so section E can require an exact one.
                    "labels": [str(i) for i in q_bs.index],
                    "bs_cols": [str(pd.Timestamp(c).date()) for c in cols],
                    "bs_date": str(pd.Timestamp(cols[0]).date()) if cols else None,
                    "prior_pos": j,
                    "prior_date": str(pd.Timestamp(cols[j]).date()) if j is not None else None,
                    "gap_days": (int((pd.Timestamp(cols[0]) - pd.Timestamp(cols[j])).days)
                                 if j is not None else None),
                    "now": {lbl: val(lbl) for lbl in BS_ROWS},
                    "prior": ({lbl: val(lbl, j) for lbl in BS_ROWS} if j is not None else {}),
                }
                break
            except Exception as e:                                 # network: retry, then record
                if attempt == 2:
                    rec = {"error": f"{type(e).__name__}: {e}"}
                time.sleep(3)
        data["tickers"][t] = rec
        time.sleep(0.4)
        print(t, "ok" if "error" not in rec else rec["error"], flush=True)
    PROBE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", PROBE)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "probe":
    probe()
    sys.exit(0)

# ------------------------------------------------------------------------------------ offline
from scipy.stats import spearmanr                                           # noqa: E402

import factor_engine as fe                                                  # noqa: E402
import sec_crosscheck as sx                                                 # noqa: E402

RUN = newest_full_run()
BANKS = bank_like(RUN)
raw = pd.read_parquet(RUN / "00_raw_fetch.parquet").set_index("Ticker")
F02 = pd.read_parquet(RUN / "02_outliers_flagged.parquet")
published = pd.read_parquet(RUN / "05_final_scored.parquet").set_index("Ticker")
P = json.loads(PROBE.read_text(encoding="utf-8")) if PROBE.exists() else {"tickers": {}}
out = {"run": RUN.name,
       "run_started": json.loads((RUN / "meta.json").read_text())["start_time"],
       "probe": {"probed_at": P.get("probed_at"), "run": P.get("run")},
       "bank_like": len(BANKS)}


def rnd(v, n=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


def num(t, col):
    try:
        v = float(raw.loc[t, col])
    except (KeyError, TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


# ---- A. availability, on the run's own extraction path --------------------------------------
# The refusal test. Every input the proposal needs, read exactly as compute_metrics reads it.
rows = {}
for t in BANKS:
    ni = num(t, "netIncome")
    eq, eqp = num(t, "totalEquity"), num(t, "totalEquity_prior")
    ta, tap = num(t, "totalAssets"), num(t, "totalAssets_prior")
    avg_eq = (eq + eqp) / 2 if (np.isfinite(eq) and np.isfinite(eqp)) else np.nan
    avg_ta = (ta + tap) / 2 if (np.isfinite(ta) and np.isfinite(tap)) else np.nan
    p = P["tickers"].get(t) or {}
    rows[t] = {
        "netIncome": rnd(ni, 0), "totalEquity": rnd(eq, 0), "totalEquity_prior": rnd(eqp, 0),
        "totalAssets": rnd(ta, 0), "totalAssets_prior": rnd(tap, 0),
        "run_bs_date": str(raw.loc[t, "_stmt_date_balance_sheet"])[:10],
        # the proposal, computed here
        "roe_proposed": rnd(ni / avg_eq) if (np.isfinite(ni) and avg_eq and avg_eq > 0) else None,
        "roa_proposed": rnd(ni / avg_ta) if (np.isfinite(ni) and avg_ta and avg_ta > 0) else None,
        # what the run published
        "roe_published": rnd(F02.set_index("Ticker").loc[t, "roe"]),
        "roa_published": rnd(F02.set_index("Ticker").loc[t, "roa"]),
        "yahoo_roe": rnd(num(t, "returnOnEquity")),
        "yahoo_roa": rnd(num(t, "returnOnAssets")),
        # ending-balance variants, for the "average or ending" trade-off
        "roe_ending": rnd(ni / eq) if (np.isfinite(ni) and eq > 0) else None,
        "roa_ending": rnd(ni / ta) if (np.isfinite(ni) and ta > 0) else None,
        # probe
        "probe_error": p.get("error"),
        "probe_bs_date": p.get("bs_date"), "probe_prior_date": p.get("prior_date"),
        "probe_gap_days": p.get("gap_days"), "probe_prior_pos": p.get("prior_pos"),
        # Only when the statement carries the row under that exact name (see the probe's note).
        "probe_minority_now": ((p.get("now") or {}).get("Minority Interest")
                               if "Minority Interest" in (p.get("labels") or []) else None),
        "probe_gross_equity": (p.get("now") or {}).get("Total Equity Gross Minority Interest"),
        "probe_common_equity": (p.get("now") or {}).get("Common Stock Equity"),
    }
A = pd.DataFrame(rows).T
have = lambda c: [t for t in BANKS if rows[t][c] is not None]                      # noqa: E731
out["A_availability"] = {
    "note": "the refusal test: inputs read through the engine's own extraction, on a second "
            "day's run (2026-10-20), which is the first carrying _year_ago_position",
    "netIncome present and positive":
        f'{sum(1 for t in BANKS if (rows[t]["netIncome"] or 0) > 0)} of {len(BANKS)}',
    "totalEquity_prior present and positive":
        f'{sum(1 for t in BANKS if (rows[t]["totalEquity_prior"] or 0) > 0)} of {len(BANKS)}',
    "totalAssets_prior present and positive":
        f'{sum(1 for t in BANKS if (rows[t]["totalAssets_prior"] or 0) > 0)} of {len(BANKS)}',
    "roe_proposed computable": f'{len(have("roe_proposed"))} of {len(BANKS)}',
    "roa_proposed computable": f'{len(have("roa_proposed"))} of {len(BANKS)}',
    "roe_proposed missing": sorted(set(BANKS) - set(have("roe_proposed"))),
    "roa_proposed missing": sorted(set(BANKS) - set(have("roa_proposed"))),
    # what the run scores TODAY, for the mixed-basis comparison
    "today: scored on yahoo .info":
        f'{sum(1 for t in BANKS if rows[t]["yahoo_roe"] is not None)} of {len(BANKS)} roe, '
        f'{sum(1 for t in BANKS if rows[t]["yahoo_roa"] is not None)} of {len(BANKS)} roa',
    "today: scored on the ENDING-balance fallback (a second definition inside one ranking)":
        sorted(t for t in BANKS if rows[t]["yahoo_roe"] is None
               and rows[t]["roe_published"] is not None),
}

# ---- B. the gap between the two balance sheets, from the fresh probe -------------------------
gaps = [rows[t]["probe_gap_days"] for t in BANKS if rows[t]["probe_gap_days"] is not None]
out["B_averaging_is_a_year_apart"] = {
    "note": "the average is only an average over a year if the second sheet is a year earlier; "
            "_year_ago_position picks it by date within 330-400 days",
    "probe_errors": sorted(t for t in BANKS if rows[t]["probe_error"]),
    "probe found a year-ago column": f'{len(gaps)} of {len(BANKS)}',
    "gap_days": {"min": min(gaps) if gaps else None, "max": max(gaps) if gaps else None,
                 "median": float(np.median(gaps)) if gaps else None},
    "outside 350-380 days": {t: rows[t]["probe_gap_days"] for t in BANKS
                             if rows[t]["probe_gap_days"] is not None
                             and not 350 <= rows[t]["probe_gap_days"] <= 380},
    "prior NOT in column four (where taking it by position would have been wrong)":
        {t: rows[t]["probe_prior_pos"] for t in BANKS
         if rows[t]["probe_prior_pos"] not in (None, 4)},
    "probe_bs_date equals the run's": f'''{sum(
        1 for t in BANKS if rows[t]["probe_bs_date"] == rows[t]["run_bs_date"])} of {len(BANKS)}''',
}

# ---- C. against the companies' own SEC filings ----------------------------------------------
C = {"note": "trailing-twelve-month NetIncomeLoss over the average of StockholdersEquity / "
             "Assets at the run's balance-sheet date and a year earlier, from the weekly "
             "companyfacts cache - the same comparison the note made (38 of 39 within 2%)"}
try:
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    per = {}
    for t in BANKS:
        f_t = facts[facts["ticker"] == t]
        if f_t.empty:
            continue
        d0 = pd.Timestamp(rows[t]["run_bs_date"])
        d1 = d0 - pd.Timedelta(days=365)
        eq0, _, _ = sx.sec_value(f_t, ("StockholdersEquity",), "instant", d0)
        eq1, _, _ = sx.sec_value(f_t, ("StockholdersEquity",), "instant", d1)
        ta0, _, _ = sx.sec_value(f_t, ("Assets",), "instant", d0)
        ta1, _, _ = sx.sec_value(f_t, ("Assets",), "instant", d1)
        # "flow" is the trailing-twelve-month rebuild. sec_value's `kind` is tested as
        # `flow_ttm if kind == "flow" else instant`, so any other string - "duration",
        # say - silently asks for an instant and returns nothing for a flow concept.
        ni, _, _ = sx.sec_value(f_t, ("NetIncomeLoss",), "flow", d0)
        vals = (eq0, eq1, ta0, ta1, ni)
        if any(v is None for v in vals) or not all(np.isfinite(float(v)) for v in vals):
            continue
        sec_roe, sec_roa = ni / ((eq0 + eq1) / 2), ni / ((ta0 + ta1) / 2)
        rp, ap = rows[t]["roe_proposed"], rows[t]["roa_proposed"]
        yr, ya = rows[t]["yahoo_roe"], rows[t]["yahoo_roa"]
        per[t] = {
            "sec_roe": rnd(sec_roe), "sec_roa": rnd(sec_roa),
            "proposed_roe": rp, "proposed_roa": ap, "yahoo_roe": yr, "yahoo_roa": ya,
            "roe_proposed_err": rnd(abs(rp / sec_roe - 1), 4) if (rp and sec_roe) else None,
            "roa_proposed_err": rnd(abs(ap / sec_roa - 1), 4) if (ap and sec_roa) else None,
            "roe_yahoo_err": rnd(abs(yr / sec_roe - 1), 4) if (yr and sec_roe) else None,
            "roa_yahoo_err": rnd(abs(ya / sec_roa - 1), 4) if (ya and sec_roa) else None,
        }
    def within(key, tol):
        v = [x[key] for x in per.values() if x[key] is not None]
        return f'{sum(1 for e in v if e <= tol)} of {len(v)}'
    C.update({
        "comparable_stocks": len(per),
        "proposed ROE within 2% of the filings": within("roe_proposed_err", 0.02),
        "proposed ROA within 2% of the filings": within("roa_proposed_err", 0.02),
        "yahoo ROE within 2% of the filings": within("roe_yahoo_err", 0.02),
        "yahoo ROA within 2% of the filings": within("roa_yahoo_err", 0.02),
        "proposed ROE within 0.5%": within("roe_proposed_err", 0.005),
        "proposed ROA within 0.5%": within("roa_proposed_err", 0.005),
        "worst proposed ROE": dict(sorted(
            ((t, v["roe_proposed_err"]) for t, v in per.items() if v["roe_proposed_err"]),
            key=lambda kv: -kv[1])[:6]),
        "worst proposed ROA": dict(sorted(
            ((t, v["roa_proposed_err"]) for t, v in per.items() if v["roa_proposed_err"]),
            key=lambda kv: -kv[1])[:6]),
        "per_stock": per,
    })
except Exception as e:                                               # pragma: no cover - report
    C["error"] = f"{type(e).__name__}: {e}"
out["C_against_the_filings"] = C

# ---- D. the stakes, on TODAY's run ----------------------------------------------------------
RUN_CFG = yaml.safe_load((RUN / "config.yaml").read_text(encoding="utf-8"))
_eff = json.loads((RUN / "effective_weights.json").read_text(encoding="utf-8"))
RUN_CFG["factor_weights"] = _eff["factor_weights"]
RUN_CFG["metric_weights"] = _eff["metric_weights"]


def score(roe=None, roa=None):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = F02.copy()
        bank = df["_is_bank_like"].fillna(False).astype(bool)
        for col, mapping in (("roe", roe), ("roa", roa)):
            if mapping is not None:
                v = df["Ticker"].map(mapping)
                df.loc[bank, col] = v[bank]
        df = fe.compute_sector_percentiles(df)
        df = fe.apply_percentile_transform(df, RUN_CFG)
        df = fe.compute_category_scores(df, RUN_CFG)
        df = fe.compute_composite(df, RUN_CFG)
        return fe.rank_stocks(df).set_index("Ticker")


base = score()
out["baseline_reproduces_published_ranks"] = \
    f'{int((base["Rank"] == published.loc[base.index, "Rank"]).sum())} of {len(base)}'

bk = [t for t in BANKS if t in base.index]
nonbank = [t for t in base.index if t not in bk]


def stakes(label, roe, roa):
    b = score(roe, roa)
    mv = (b.loc[bk, "Rank"] - base.loc[bk, "Rank"]).astype(int)
    ta, tb = set(base.nsmallest(25, "Rank").index), set(b.nsmallest(25, "Rank").index)
    return {
        "label": label,
        "universe_composite_spearman":
            rnd(spearmanr(base["Composite"], b.loc[base.index, "Composite"]).statistic, 5),
        "bank_like_median_abs_rank_move": float(mv.abs().median()),
        "moved_10_or_more": int((mv.abs() >= 10).sum()),
        "moved_25_or_more": int((mv.abs() >= 25).sum()),
        "top_25_in_only_baseline": sorted(ta - tb),
        "top_25_in_only_proposed": sorted(tb - ta),
        "non_bank_composite_changed":
            int((b.loc[nonbank, "Composite"] != base.loc[nonbank, "Composite"]).sum()),
        "largest_rank_moves": {t: [int(base.loc[t, "Rank"]), int(b.loc[t, "Rank"])]
                               for t in mv.abs().sort_values(ascending=False).head(10).index},
        "largest_composite_moves": {
            t: [rnd(base.loc[t, "Composite"], 2), rnd(b.loc[t, "Composite"], 2)]
            for t in (b.loc[bk, "Composite"] - base.loc[bk, "Composite"]).abs()
                      .sort_values(ascending=False).head(6).index},
    }


roe_avg = {t: rows[t]["roe_proposed"] for t in BANKS if rows[t]["roe_proposed"] is not None}
roa_avg = {t: rows[t]["roa_proposed"] for t in BANKS if rows[t]["roa_proposed"] is not None}
roe_end = {t: rows[t]["roe_ending"] for t in BANKS if rows[t]["roe_ending"] is not None}
roa_end = {t: rows[t]["roa_ending"] for t in BANKS if rows[t]["roa_ending"] is not None}
out["D_stakes"] = {
    "proposed (average balances)": stakes("average", roe_avg, roa_avg),
    "ending balances, for contrast": stakes("ending", roe_end, roa_end),
    "ROE only (ROA left on the source)": stakes("roe only", roe_avg, None),
    "ROA only (ROE left on the source)": stakes("roa only", None, roa_avg),
}

# ---- E. the judgement call the note asks the reviewer to settle: ROA and minority interests --
# For an Up-C or fund-consolidating company, shareholders' net income over CONSOLIDATED assets
# mixes a numerator that excludes the minority holders with a denominator that includes their
# assets. The note proposes the regulators' convention anyway. Measure how many stocks it is
# material for, rather than reasoning about three names.
E = {"note": "netIncome is attributable to the company; totalAssets is consolidated. Minority "
             "interest as a share of total equity measures how far apart the two claims are."}
mi, cross = {}, {}
for t in BANKS:
    m, eq = rows[t]["probe_minority_now"], rows[t]["totalEquity"]
    if m is not None and eq:
        mi[t] = rnd(float(m) / float(eq), 4)
    # The same quantity from arithmetic rather than from a label: gross equity (including the
    # minority holders) less the stockholders' equity the run scores. Agreement between the two
    # is the check that the label was read correctly.
    g = rows[t]["probe_gross_equity"]
    if g is not None and eq:
        cross[t] = rnd((float(g) - float(eq)) / float(eq), 4)
E.update({
    "stocks carrying an exact 'Minority Interest' row": len(mi),
    "minority interest / stockholders equity (from the row)": {
        "median": rnd(float(np.median(list(mi.values()))), 4) if mi else None,
        "over 5%": {t: v for t, v in sorted(mi.items(), key=lambda kv: -kv[1]) if v > 0.05},
        "over 1%": sum(1 for v in mi.values() if v > 0.01),
        "all": dict(sorted(mi.items(), key=lambda kv: -kv[1])),
    },
    "(gross equity - stockholders equity) / stockholders equity, as a cross-check": {
        "stocks": len(cross),
        "over 5%": {t: v for t, v in sorted(cross.items(), key=lambda kv: -kv[1]) if v > 0.05},
        "over 1%": sum(1 for v in cross.values() if v > 0.01),
        "disagrees with the row by more than 0.001":
            {t: [mi[t], cross[t]] for t in mi if t in cross and abs(mi[t] - cross[t]) > 0.001},
    },
    "IBKR": {k: rows["IBKR"].get(k) for k in
             ("netIncome", "totalAssets", "totalAssets_prior", "totalEquity",
              "roa_proposed", "yahoo_roa", "probe_minority_now")} if "IBKR" in rows else None,
})
out["E_minority_interest_for_roa"] = E

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
printable = {k: v for k, v in out.items() if k != "C_against_the_filings"}
printable["C_against_the_filings"] = {k: v for k, v in out["C_against_the_filings"].items()
                                      if k != "per_stock"}
print(json.dumps(printable, indent=1, default=str))
print("\nwrote", OUT)
