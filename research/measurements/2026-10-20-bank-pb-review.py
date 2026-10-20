"""Review of the open bank price-to-book proposal (research/2026-10-19-bank-price-to-book.md).

The note proposes, for bank-like stocks, ``pb_ratio = marketCap / commonEquity`` where
``commonEquity`` is a new raw field read from the quarterly balance sheet's "Common Stock Equity"
row at the run's balance-sheet date. This script is the reviewing session's own check, on a
*different* day's run and a *fresh* probe, of the two things the decision turns on:

  probe   (network) for every bank-like stock in the newest full run, read the quarterly balance
          sheet **through the engine's own extraction path** (``factor_engine._populated_columns``
          and ``factor_engine._stmt_val``) for "Stockholders Equity", "Common Stock Equity",
          "Total Equity Gross Minority Interest" and "Tangible Book Value", at the same column the
          fetch reads, plus the column's date and the source's ``.info`` figures. Written to
          ``2026-10-20-bank-pb-review-probe.json``.

  (none)  offline, on the newest full run:
     A. AVAILABILITY - the failure mode that refused the earnings-definition proposal
        (research/2026-10-15-earnings-definition.md section 8): for how many of the bank-like
        stocks is "Common Stock Equity" present and positive at the run's own balance-sheet date,
        and does the probe's statement date equal the run's ``_stmt_date_balance_sheet``?
     B. THE STAKES, on this run - the run's metric table rescored with the run's own config and
        effective weights (baseline asserted to reproduce the published ranks), bank-like
        ``pb_ratio`` replaced by market value / common equity.
     C. BERKSHIRE from the filings - equity and Class A-equivalent shares from the SEC facts
        cache, so the corrected P/B is checked against the companies' own report rather than
        against another of the source's rows.

Usage:
    python research/measurements/2026-10-20-bank-pb-review.py probe   # network, ~46 requests
    python research/measurements/2026-10-20-bank-pb-review.py         # offline
"""
import contextlib
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

EQUITY_ROWS = ("Stockholders Equity", "Common Stock Equity",
               "Total Equity Gross Minority Interest", "Tangible Book Value",
               "Ordinary Shares Number", "Total Assets")
INFO = ("priceToBook", "bookValue", "marketCap", "currentPrice", "mostRecentQuarter")


def newest_full_run() -> Path:
    import run_context
    return Path(run_context.latest_run_dir())


def bank_like(run: Path) -> list:
    f02 = pd.read_parquet(run / "02_outliers_flagged.parquet")
    return sorted(f02.loc[f02["_is_bank_like"].fillna(False).astype(bool), "Ticker"])


# ------------------------------------------------------------------------------------- probe
def probe():
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
                # The engine's own choice of column: _populated_columns(...)[0].
                cols = fe._populated_columns(q_bs) if (q_bs is not None and not q_bs.empty) else []
                rec = {
                    "info": {f: info.get(f) for f in INFO},
                    "bs_date": str(pd.Timestamp(cols[0]).date()) if cols else None,
                    "bs_cols": [str(pd.Timestamp(c).date()) for c in cols],
                    # exactly what rec[...] = _stmt_val(_bs_src, label) would read
                    "rows": {lbl: (None if not np.isfinite(v := fe._stmt_val(q_bs, lbl)) else float(v))
                             for lbl in EQUITY_ROWS},
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
P = json.loads(PROBE.read_text(encoding="utf-8"))
out = {"run": RUN.name, "run_started": json.loads((RUN / "meta.json").read_text())["start_time"],
       "probe": {"probed_at": P["probed_at"], "run": P["run"]}, "bank_like": len(BANKS)}


def rnd(v, n=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, n)


# ---- A. availability ------------------------------------------------------------------------
rows = {}
for t in BANKS:
    p = P["tickers"].get(t) or {}
    r = (p.get("rows") or {})
    run_date = str(raw.loc[t, "_stmt_date_balance_sheet"])[:10]
    ce = r.get("Common Stock Equity")
    se = r.get("Stockholders Equity")
    rows[t] = {
        "probe_error": p.get("error"),
        "run_bs_date": run_date,
        "probe_bs_date": p.get("bs_date"),
        "same_date": p.get("bs_date") == run_date,
        "common_equity": ce,
        "stockholders_equity": se,
        "run_totalEquity": rnd(raw.loc[t, "totalEquity"], 0),
        "ce_positive": bool(ce is not None and ce > 0),
        "ce_over_se": rnd(ce / se, 4) if (ce and se) else None,
        "marketCap": rnd(raw.loc[t, "marketCap"], 0),
        "yahoo_pb": rnd(raw.loc[t, "priceToBook"]),
        "mc_over_ce": rnd(raw.loc[t, "marketCap"] / ce, 4) if (ce and ce > 0) else None,
    }
A = pd.DataFrame(rows).T
out["A_availability"] = {
    "probe_errors": sorted(t for t in BANKS if rows[t]["probe_error"]),
    "common_equity_present_and_positive": f'{int(A["ce_positive"].astype(bool).sum())} of {len(A)}',
    "missing": sorted(t for t in BANKS if not rows[t]["ce_positive"]),
    "probe_bs_date_equals_run_bs_date": f'{int(A["same_date"].astype(bool).sum())} of {len(A)}',
    "dates_differing": {t: [rows[t]["run_bs_date"], rows[t]["probe_bs_date"]]
                        for t in BANKS if not rows[t]["same_date"]},
    "run_totalEquity_vs_probe_stockholders_equity_within_0.5pct":
        f'{int(sum(1 for t in BANKS if rows[t]["stockholders_equity"] and abs(rows[t]["stockholders_equity"] / float(rows[t]["run_totalEquity"]) - 1) <= 0.005))}'
        f' of {len(A)}',
    "common_over_stockholders_equity": {
        "median": rnd(pd.to_numeric(A["ce_over_se"], errors="coerce").median()),
        "under_0.95 (preferred-heavy)": sorted(t for t in BANKS
                                               if (rows[t]["ce_over_se"] or 1) < 0.95),
    },
    "mc_over_ce_vs_yahoo_pb_within_2pct (excl. BRK-B)": f'''{int(sum(
        1 for t in BANKS if t != "BRK-B" and rows[t]["mc_over_ce"] and rows[t]["yahoo_pb"]
        and abs(rows[t]["mc_over_ce"] / rows[t]["yahoo_pb"] - 1) <= 0.02))} of {
        sum(1 for t in BANKS if t != "BRK-B" and rows[t]["mc_over_ce"] and rows[t]["yahoo_pb"])}''',
}

# ---- B. the stakes, on this run -------------------------------------------------------------
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


a = score()
out["baseline_reproduces_published_ranks"] = \
    f'{int((a["Rank"] == published.loc[a.index, "Rank"]).sum())} of {len(a)}'

pb_new = {t: rows[t]["mc_over_ce"] for t in BANKS if rows[t]["mc_over_ce"]}
b = score(pb_new)
bk = [t for t in BANKS if t in a.index]
mv = (b.loc[bk, "Rank"] - a.loc[bk, "Rank"]).astype(int)
top_a, top_b = set(a.nsmallest(25, "Rank").index), set(b.nsmallest(25, "Rank").index)
out["B_stakes_mc_over_common_equity"] = {
    "replaced": len(pb_new),
    "universe_composite_spearman": rnd(spearmanr(a["Composite"], b.loc[a.index, "Composite"]).statistic, 5),
    "bank_like_median_abs_rank_move": float(mv.abs().median()),
    "bank_like_moved_10_or_more": int((mv.abs() >= 10).sum()),
    "top_25_in_only_baseline": sorted(top_a - top_b),
    "top_25_in_only_proposed": sorted(top_b - top_a),
    "largest_moves": {t: [int(a.loc[t, "Rank"]), int(b.loc[t, "Rank"])]
                      for t in mv.abs().sort_values(ascending=False).head(8).index},
    "BRK-B": {"rank": [int(a.loc["BRK-B", "Rank"]), int(b.loc["BRK-B", "Rank"])],
              "pb_ratio": [rnd(a.loc["BRK-B", "pb_ratio"]), rnd(b.loc["BRK-B", "pb_ratio"])],
              "pb_percentile": [rnd(a.loc["BRK-B", "pb_ratio_pct"], 1), rnd(b.loc["BRK-B", "pb_ratio_pct"], 1)],
              "valuation": [rnd(a.loc["BRK-B", "valuation_score"], 1), rnd(b.loc["BRK-B", "valuation_score"], 1)],
              "composite": [rnd(a.loc["BRK-B", "Composite"], 2), rnd(b.loc["BRK-B", "Composite"], 2)]},
    "non_bank_composite_changed": int((b.loc[[t for t in a.index if t not in bk], "Composite"]
                                      != a.loc[[t for t in a.index if t not in bk], "Composite"]).sum()),
}

# ---- C. Berkshire from the filings ----------------------------------------------------------
C = {}
try:
    facts = pd.read_parquet(ROOT / "data" / "sec" / "pit" / "facts.parquet")
    f_t = facts[facts["ticker"] == "BRK-B"]
    bs_date = pd.Timestamp(raw.loc["BRK-B", "_stmt_date_balance_sheet"])
    eq, _, _ = sx.sec_value(f_t, ("StockholdersEquity",), "instant", bs_date)
    sh, _, _ = sx.sec_value(f_t, ("CommonStockSharesOutstanding",), "instant", bs_date)
    mc = float(raw.loc["BRK-B", "marketCap"])
    bs_date = str(bs_date.date())
    C = {"bs_date": bs_date, "filed_stockholders_equity": rnd(eq, 0),
         "filed_common_shares": rnd(sh, 0),
         "probe_common_equity": rows["BRK-B"]["common_equity"],
         "run_totalEquity": rows["BRK-B"]["run_totalEquity"],
         "marketCap": rnd(mc, 0),
         "pb_from_filed_equity": rnd(mc / eq, 4) if eq else None,
         "pb_from_probe_common_equity": rows["BRK-B"]["mc_over_ce"],
         "yahoo_pb_published": rnd(raw.loc["BRK-B", "priceToBook"], 6),
         "yahoo_bookValue_per_share": rnd(raw.loc["BRK-B", "bookValue"], 2),
         "run_sharesBS": rnd(raw.loc["BRK-B", "sharesBS"], 0),
         "price": rnd(raw.loc["BRK-B", "_current_price"], 2) if "_current_price" in raw.columns else None}
except Exception as e:                                               # pragma: no cover - report it
    C = {"error": f"{type(e).__name__}: {e}"}
out["C_berkshire_from_filings"] = C

OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
print("\nwrote", OUT)
