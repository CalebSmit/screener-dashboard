"""What choosing the year-ago column by DATE instead of by POSITION changes on today's universe.

2026-10-19. The fetch took the year-ago quarter as "column four" (the fifth quarter with revenue for
revenue growth; column four of the quarterly balance sheet; quarters four to seven for the prior
twelve months). When the source omits a quarter - BRK-B's balance sheet has no September 2025 -
column four is not a year earlier. ``factor_engine._year_ago_position`` now picks it by date.

This script re-reads the four statements for every stock in the newest full run, computes every
year-ago input both ways (the old rule is replicated here verbatim; the new one is the engine's
own functions), checks the old rule reproduces the run's raw fetch, and then rescores the run with
the changed inputs written in - the same rescoring the 2026-10-19 data audit uses, asserted to
reproduce the published composites first. It also checks the annual statements' positional
"col 1 = a year earlier" assumption, which this change does not touch.

Usage:
    python research/measurements/2026-10-19-year-ago-by-date.py --stmts <dir of TICKER.pkl>
        (each pickle a dict of yfinance DataFrames: quarterly_financials, quarterly_balance_sheet,
         quarterly_cashflow, financials, balance_sheet, cashflow; ``--fetch`` fills the folder)
Writes the JSON beside this file.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
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
from run_context import latest_run_dir  # noqa: E402

PROPS = ["quarterly_financials", "quarterly_balance_sheet", "quarterly_cashflow",
         "financials", "balance_sheet", "cashflow"]
BS_FIELDS = {  # field: labels tried in order (the fetch's own order)
    "totalAssets_prior": ["Total Assets"],
    "totalEquity_prior": ["Stockholders Equity", "Total Stockholder"],
    "longTermDebt_prior": ["Long Term Debt"],
    "currentAssets_prior": ["Current Assets"],
    "currentLiabilities_prior": ["Current Liabilities"],
}
LTM_FIELDS = {"totalRevenue_prior": "Total Revenue", "grossProfit_prior": "Gross Profit",
              "netIncome_prior": "Net Income"}
REV_FIELDS = ["_rev_q0", "_rev_q4", "_rev_q0_date", "_rev_q4_date"]


def fetch(tickers, out: Path):
    import yfinance as yf
    out.mkdir(parents=True, exist_ok=True)
    for tk in tickers:
        p = out / f"{tk}.pkl"
        if p.exists():
            continue
        t = yf.Ticker(tk)
        with open(p, "wb") as f:
            pickle.dump({k: getattr(t, k) for k in PROPS}, f)


# ----------------------------------------------------------------------------- the old rule
def old_rev(q_fins):
    if q_fins is None or "Total Revenue" not in q_fins.index:
        return None
    _qr = q_fins.loc["Total Revenue"]
    _qc = sorted([c for c in _qr.index if pd.notna(_qr[c])], reverse=True)
    if len(_qc) >= 5:
        _gap = (pd.Timestamp(_qc[0]) - pd.Timestamp(_qc[4])).days
        if 350 <= _gap <= 380 and float(_qr[_qc[4]]) > 0:
            return float(_qr[_qc[0]]), float(_qr[_qc[4]]), str(pd.Timestamp(_qc[0]).date()), str(pd.Timestamp(_qc[4]).date())
    return None


def new_rev(q_fins):
    yq = fe._year_ago_quarter(q_fins, "Total Revenue")
    if yq is None or not yq[1] > 0:
        return None
    return yq[0], yq[1], str(yq[2].date()), str(yq[3].date())


def old_bs_col(src, is_quarterly):
    col = 4 if (is_quarterly and src is not None and len(src.columns) >= 5) else 1
    cols = fe._populated_columns(src) if src is not None else []
    if len(cols) <= col:
        return None, cols
    gap = (pd.Timestamp(cols[0]) - pd.Timestamp(cols[col])).days
    return (col if 330 <= gap <= 400 else None), cols


def bs_values(src, col):
    out = {}
    c = 10_000 if col is None else col
    for f, labels in BS_FIELDS.items():
        v = np.nan
        for lab in labels:
            v = fe._stmt_val(src, lab, c)
            if not np.isnan(v):
                break
        out[f] = v
    sh = fe._stmt_val(src, "Ordinary Shares Number")
    out["sharesBS_prior"] = fe._stmt_val(src, "Ordinary Shares Number", c)
    if np.isnan(sh):
        out["sharesBS_prior"] = fe._stmt_val(src, "Share Issued", c)
    return out


def ltm_values(q_fins, fins, year_ago):
    cur = [fe._stmt_val_ltm(q_fins, lab, annual=fins) for lab in ("Total Revenue", "Net Income")]
    ebit = fe._stmt_val_ltm(q_fins, "Operating Income", annual=fins)
    if np.isnan(ebit):
        ebit = fe._stmt_val_ltm(q_fins, "EBIT", annual=fins)
    out = {}
    if all(np.isnan(x) for x in cur + [ebit]):           # the fetch's annual branch: same both ways
        for f, lab in LTM_FIELDS.items():
            out[f] = fe._stmt_val(fins, lab, 1)
        return out
    for f, lab in LTM_FIELDS.items():
        v = (fe._stmt_val_ltm_year_ago(q_fins, lab) if year_ago
             else fe._stmt_val_ltm(q_fins, lab, offset=4))
        out[f] = v if not np.isnan(v) else fe._stmt_val(fins, lab, 1)
    return out


def norm(v):
    """None for a missing value of any kind, else the value."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    return v


def same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    a = np.nan if a is None else a
    b = np.nan if b is None else b
    return (pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(a - b) <= 1e-9 * max(abs(a), 1))


def annual_positions(d):
    """Annual statements whose col 1/2/3 is not 1/2/3 years before col 0 (populated periods)."""
    bad = []
    for k in ("financials", "balance_sheet", "cashflow"):
        s = d.get(k)
        if s is None or s.empty:
            continue
        cols = fe._populated_columns(s)
        for i in (1, 2, 3):
            if len(cols) > i:
                gap = (pd.Timestamp(cols[0]) - pd.Timestamp(cols[i])).days
                if abs(gap - 365 * i) > 20 * i:
                    bad.append({"statement": k, "col": i, "dates": [str(pd.Timestamp(c).date()) for c in cols]})
                    break
    return bad


# ----------------------------------------------------------------------------- the rescoring
class Scorer:
    """Rescore a run with some stocks' raw records changed (as research/measurements/
    2026-10-19-top-of-ranking-data-audit.py does): each changed stock's metrics recomputed from its
    record, written into the run's own outlier-flagged frame, then percentiles to rank."""

    def __init__(self, run_dir):
        self.cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        self.base = pd.read_parquet(run_dir / "02_outliers_flagged.parquet").set_index("Ticker", drop=False)
        raw = pd.read_parquet(run_dir / "00_raw_fetch.parquet")
        self.raw = {r["Ticker"]: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                    for r in raw.to_dict("records")}
        self.metric_cols = [c for c in fe.METRIC_COLS if c in self.base.columns]

    def metrics(self, rec):
        with contextlib.redirect_stdout(io.StringIO()):
            return fe.compute_metrics([rec], pd.Series(dtype=float), self.cfg).iloc[0]

    def changed(self, a, b):
        x, y = self.metrics(a), self.metrics(b)
        out = {}
        for c in self.metric_cols:
            if not same(x.get(c), y.get(c)):
                out[c] = (None if pd.isna(x.get(c)) else float(x.get(c)), None if pd.isna(y.get(c)) else float(y.get(c)))
        return out

    def score(self, metric_changes: dict):
        df = self.base.copy()
        for t, cols in metric_changes.items():
            for c, (_, v) in cols.items():
                df.loc[t, c] = np.nan if v is None else v
        with contextlib.redirect_stdout(io.StringIO()):
            d = fe.compute_sector_percentiles(df.reset_index(drop=True).copy())
            d = fe.apply_percentile_transform(d, self.cfg)
            d = fe.compute_category_scores(d, self.cfg)
            d = fe.compute_composite(d, self.cfg)
            return fe.rank_stocks(d).set_index("Ticker")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stmts", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    run_dir = a.run or latest_run_dir(ROOT / "runs")
    S = Scorer(run_dir)
    if a.fetch:
        fetch(sorted(S.raw), a.stmts)

    per, drift, annual_bad, no_stmts = {}, [], {}, []
    for t, raw in sorted(S.raw.items()):
        p = a.stmts / f"{t}.pkl"
        if not p.exists() or t not in S.base.index:          # not scored by the run (fetch failed)
            no_stmts.append(t)
            continue
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        q_fins, q_bs, fins, bs = (d.get("quarterly_financials"), d.get("quarterly_balance_sheet"),
                                  d.get("financials"), d.get("balance_sheet"))
        if fe.quarterly_statement_defect(q_fins) is not None:
            q_fins = None
        src = q_bs if (q_bs is not None and not q_bs.empty) else bs
        old_col, cols = old_bs_col(src, src is q_bs)
        new_col = fe._year_ago_position(cols, tol=fe.YEAR_AGO_TOL_BALANCE_DAYS) if cols else None

        old = {**bs_values(src, old_col), **ltm_values(q_fins, fins, False)}
        new = {**bs_values(src, new_col), **ltm_values(q_fins, fins, True)}
        orv, nrv = old_rev(q_fins), new_rev(q_fins)
        for i, f in enumerate(REV_FIELDS):
            old[f] = orv[i] if orv else None
            new[f] = nrv[i] if nrv else None

        # does the old rule, on today's statements, reproduce what the run fetched?
        mism = [f for f in old if not same(norm(old[f]), norm(raw.get(f)))]
        if mism:
            drift.append({"ticker": t, "fields": mism})
        diff = {f: (old[f], new[f]) for f in old if not same(norm(old[f]), norm(new[f]))}
        bad = annual_positions(d)
        if bad:
            annual_bad[t] = bad
        if diff or (old_col != new_col):
            def when(c):
                return None if c is None or c >= len(cols) else str(pd.Timestamp(cols[c]).date())
            per[t] = {"bs_prior_old": when(old_col), "bs_prior_new": when(new_col),
                      "bs_latest": str(pd.Timestamp(cols[0]).date()) if cols else None,
                      "rev_q4_old": orv[3] if orv else None, "rev_q4_new": nrv[3] if nrv else None,
                      "fields": {f: [None if (v is None or (not isinstance(v, str) and pd.isna(v))) else v for v in ab]
                                 for f, ab in diff.items()},
                      "reproduces_run": not mism}

    # rescoring: only stocks whose old-rule inputs reproduce the run, so the change is the rule's
    pub = S.score({})
    ran = pd.read_parquet(run_dir / "05_final_scored.parquet").set_index("Ticker")
    assert (pub["Composite"] - ran.loc[pub.index, "Composite"]).abs().max() < 1e-9, "rescoring does not reproduce the run"
    metric_changes = {}
    for t, info in per.items():
        if not info["reproduces_run"] or not info["fields"]:
            continue
        rec = dict(S.raw[t])
        for f, (_, v) in info["fields"].items():
            rec[f] = (np.nan if v is None and not f.endswith("_date") else v)
        ch = S.changed(S.raw[t], rec)
        if ch:
            metric_changes[t] = ch
            info["metrics"] = {m: list(v) for m, v in ch.items()}
    after = S.score(metric_changes)
    common = pub.index.intersection(after.index)
    moves = (after.loc[common, "Rank"] - pub.loc[common, "Rank"]).astype(int)
    comp = (after.loc[common, "Composite"] - pub.loc[common, "Composite"])
    top_b = set(pub.index[pub["Rank"] <= 25]); top_a = set(after.index[after["Rank"] <= 25])
    for t in metric_changes:
        per[t]["composite"] = [round(float(pub.loc[t, "Composite"]), 3), round(float(after.loc[t, "Composite"]), 3)]
        per[t]["rank"] = [int(pub.loc[t, "Rank"]), int(after.loc[t, "Rank"])]

    out = {
        "run": run_dir.name,
        "stocks_read": len(S.raw) - len(no_stmts), "not_read": no_stmts,
        "old_rule_does_not_reproduce_run": drift,
        "stocks_whose_year_ago_column_changes": sorted(t for t, i in per.items()
                                                       if i["bs_prior_old"] != i["bs_prior_new"] or i["rev_q4_old"] != i["rev_q4_new"]
                                                       or any(f in LTM_FIELDS for f in i["fields"])),
        "stocks_with_an_input_change": sorted(t for t, i in per.items() if i["fields"]),
        "input_changes_by_field": {f: sum(f in i["fields"] for i in per.values())
                                   for f in list(BS_FIELDS) + ["sharesBS_prior"] + list(LTM_FIELDS) + REV_FIELDS},
        "stocks_with_a_metric_change": sorted(metric_changes),
        "metric_values_changed": int(sum(len(v) for v in metric_changes.values())),
        "metric_changes_by_metric": pd.Series([m for v in metric_changes.values() for m in v]).value_counts().to_dict()
        if metric_changes else {},
        "composite_moved": int((comp.abs() > 1e-9).sum()),
        "largest_composite_move": (lambda s: {"ticker": s.idxmax(), "points": round(float(comp[s.idxmax()]), 3)})(comp.abs())
        if len(comp) else None,
        "rank_moved": int((moves != 0).sum()), "largest_rank_move": int(moves.abs().max()) if len(moves) else 0,
        "spearman_composite": float(pub.loc[common, "Composite"].rank().corr(after.loc[common, "Composite"].rank())),
        "top25_changed": sorted(top_a ^ top_b),
        "annual_col_not_a_year_apart": annual_bad,
        "per_stock": per,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_stock"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
