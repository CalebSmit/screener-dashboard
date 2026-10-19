"""Is the top of the ranking there on the companies' real figures? (2026-10-19)

For each of the run's top N stocks (and every stock the paper-trading bot holds), every statement
input a weighted metric reads is compared with the company's own SEC filings **for the same period
the run scored** - the trailing twelve months ending at the run's own statement date, the balance
sheet at its own date, the quarter its revenue growth used, the fiscal years its CAGR used.

Each input gets two comparisons:

* **nearest** - the closest figure the company filed under any tag (or construction) that can carry
  that line. Over 10% from every one of them means the run's number is not one the company
  reported: it is then tested for a period mismatch (it matches the filing for a neighbouring
  quarter or year) and for a scale error (a power of ten, or a split ratio).
* **primary** - the standard tag (``Revenues``, ``OperatingIncomeLoss``, ``NetIncomeLoss`` ...). A
  run figure near a company number but far from the standard one is a definitional choice of the
  source; the script measures how far substituting the standard figure moves the stock.

Then the stakes, scored by the engine's own functions (never a second implementation):
``compute_metrics`` recomputes the affected stock's metrics from its raw record with one input
replaced, only the metric columns that change are written into the run's own
``02_outliers_flagged`` frame, and ``compute_sector_percentiles`` -> ``compute_category_scores`` ->
``compute_composite`` -> ``rank_stocks`` re-rank all 501 stocks with the run's own config. With no
substitution that pipeline reproduces the published composites exactly (asserted below).

Also checked: statements a period behind a filing the company had made (scored again on the newer
quarter, as the change the newer filing makes to each statement-driven metric), share counts behind
market value, stock splits inside the momentum window, and dual share classes of one company.

Usage: python research/measurements/2026-10-19-top-of-ranking-data-audit.py [--run DIR] [--n 40]
Reads runs/<newest>/, data/sec/pit/facts.parquet and cache/splits.parquet; writes the JSON beside
this file. research/2026-10-19-top-of-ranking-data-audit.md is the write-up.
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
import pit_record as pr  # noqa: E402
from run_context import latest_run_dir  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
# The bot's 25 holdings on 2026-10-15 (CalebSmit/screener-bot journal/REPORT.md, read 2026-10-19).
BOT_HOLDINGS = ["BBY", "CF", "CAH", "EXPE", "AIZ", "HST", "DLTR", "MPC", "SWK", "DECK", "ALL", "DXCM",
                "TRV", "ED", "EG", "TPR", "APA", "EXPD", "BMY", "VLO", "INCY", "ES", "NTAP", "WDC", "SNDK"]
AGREE, FAR = 0.02, 0.10          # within 2% agrees; over 10% from every filed figure is "far"
NEWER_DAYS = 45                  # a filed period end this much later than the run's is a newer quarter
Q = pd.Timedelta(days=91)
Y = pd.Timedelta(days=365)
REV = pr.REVENUE


def rel(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    return (a - b) / max(abs(a), abs(b), 1.0)


def _v(x):
    return float(x) if x is not None and np.isfinite(x) else None


# ----------------------------------------------------------------------------- candidates
# Each returns [(label, value)] for one input at period end ``e``; the first entry is the
# standard ("primary") figure where the company files one.
def _ttm(F, c, e):
    return _v(F._ttm1(c, e))


def _fy(F, c, e):
    return _v(F._one(F._duration1(c, e, 340, 380)))


def _q(F, c, e):
    return _v(F._quarter1(c, e))


def _inst(F, c, e):
    return _v(F._instant1(c, e))


def special(F, get, e):
    """Restructuring + impairment charges, as pit_record's EBIT adds them back."""
    def g(key):
        return F._pick(key, get, e)
    comb = g("special_combined")
    if np.isfinite(comb):
        return abs(comb)
    imp = g("impairment_combined")
    if not np.isfinite(imp):
        imp = pr._nan_sum(g("goodwill_impairment"), g("intangible_impairment"), g("asset_impairment"),
                          g("other_impairment"))
    rs = g("restructuring")
    tot = pr._nan_sum(abs(imp) if np.isfinite(imp) else np.nan, abs(rs) if np.isfinite(rs) else np.nan)
    return tot if np.isfinite(tot) else 0.0


def cands(F, field, e, rec):
    out = []
    if field in ("totalRevenue", "_rev_q0", "_rev_q4", "totalRevenue_annual", "totalRevenue_3yr_ago"):
        fn = {"totalRevenue": _ttm, "_rev_q0": _q, "_rev_q4": _q}.get(field, _fy)
        out = [(c, fn(F, c, e)) for c in REV]
    elif field == "grossProfit":
        out = [("GrossProfit", _ttm(F, "GrossProfit", e))]
        for r in REV:
            for c in pr.C["cogs"]:
                a, b = _ttm(F, r, e), _ttm(F, c, e)
                if a is not None and b is not None:
                    out.append((f"{r}-{c}", a - b))
    elif field == "ebit":
        oi = _ttm(F, "OperatingIncomeLoss", e)
        out = [("OperatingIncomeLoss", oi)]
        if oi is not None:
            out.append(("OperatingIncomeLoss+special", oi + special(F, F._ttm1, e)))
        for p in pr.C["pretax"]:
            for i in pr.C["interest"]:
                a, b = _ttm(F, p, e), _ttm(F, i, e)
                if a is not None and b is not None:
                    out.append((f"{p}+{i}", a + abs(b)))
    elif field == "netIncome":
        out = [(c, _ttm(F, c, e)) for c in ("NetIncomeLoss", "ProfitLoss")]
    elif field == "pretaxIncome":
        out = [(c, _ttm(F, c, e)) for c in pr.C["pretax"]]
    elif field == "incomeTaxExpense":
        out = [("IncomeTaxExpenseBenefit", _ttm(F, "IncomeTaxExpenseBenefit", e))]
    elif field == "operatingCashFlow":
        out = [("NetCashProvidedByUsedInOperatingActivities", _ttm(F, "NetCashProvidedByUsedInOperatingActivities", e))]
    elif field == "capex":
        out = [(c, _ttm(F, c, e)) for c in pr.C["capex"]]
        a = _ttm(F, "PaymentsToAcquirePropertyPlantAndEquipment", e)
        for c in ("PaymentsToAcquireLeasesHeldForInvestment", "PaymentsToAcquireEquipmentOnLease"):
            b = _ttm(F, c, e)
            if a is not None and b is not None:
                out.append((f"PPE+{c}", a + b))
    elif field == "da_cf":
        out = [(c, _ttm(F, c, e)) for c in pr.C["d_and_a"]]
        a, b = _ttm(F, "Depreciation", e), _ttm(F, "AmortizationOfIntangibleAssets", e)
        if a is not None:
            out.append(("Depreciation+Amortization", a + (b or 0.0)))
    elif field == "dividendsPaid":
        out = [(c, _ttm(F, c, e)) for c in pr.C["dividends"]]
    elif field in ("totalAssets", "totalAssets_prior"):
        out = [("Assets", _inst(F, "Assets", e))]
    elif field in ("totalEquity", "totalEquity_prior"):
        out = [(c, _inst(F, c, e)) for c in pr.C["equity"]]
    elif field == "cash_bs":
        out = [(c, _inst(F, c, e)) for c in pr.C["cash"]]
    elif field == "cash_sti_bs":
        cs = _inst(F, "CashCashEquivalentsAndShortTermInvestments", e)
        out = [("CashCashEquivalentsAndShortTermInvestments", cs)]
        cash = _inst(F, "CashAndCashEquivalentsAtCarryingValue", e)
        for s in pr.C["sti"]:
            v = _inst(F, s, e)
            if cash is not None and v is not None:
                out.append((f"Cash+{s}", cash + v))
        out.append(("Cash", cash))
    elif field == "totalDebt_bs":
        lt = F.instant("lt_debt", e)
        cur = F.instant("current_debt", e)
        if not np.isfinite(cur):
            cur = pr._nan_sum(*(F._instant1(c, e) for c in pr.C["current_debt_parts"]))
        leases = pr._nan_sum(*(F._instant1(c, e) for c in pr.C["leases"]))
        if not np.isfinite(leases):
            leases = F.instant("lease_total", e)
        out = [("borrowings+leases", _v(pr._nan_sum(lt, cur, leases))), ("borrowings", _v(pr._nan_sum(lt, cur))),
               ("LongTermDebt", _v(lt))]
    return [(k, v) for k, v in out if v is not None]


# (field, date key, shift applied to the date, weighted metrics that read it)
FIELDS = [
    ("totalRevenue", "_stmt_date_financials", None, "ev_sales, roic (operating cash)"),
    ("grossProfit", "_stmt_date_financials", None, "gross_profit_assets"),
    ("ebit", "_stmt_date_financials", None, "ev_ebitda, roic, net_debt_to_ebitda"),
    ("netIncome", "_stmt_date_financials", None, "earnings_yield, accruals, sustainable_growth, piotroski"),
    ("pretaxIncome", "_stmt_date_financials", None, "roic (tax rate)"),
    ("incomeTaxExpense", "_stmt_date_financials", None, "roic (tax rate)"),
    ("_rev_q0", "_rev_q0_date", None, "revenue_growth"),
    ("_rev_q4", "_rev_q4_date", None, "revenue_growth"),
    ("operatingCashFlow", "_stmt_date_cashflow", None, "fcf_yield, accruals, piotroski"),
    ("capex", "_stmt_date_cashflow", None, "fcf_yield"),
    ("da_cf", "_stmt_date_cashflow", None, "ev_ebitda, net_debt_to_ebitda"),
    ("dividendsPaid", "_stmt_date_cashflow", None, "sustainable_growth"),
    ("totalAssets", "_stmt_date_balance_sheet", None, "gross_profit_assets, accruals, asset_growth, roic floor"),
    ("totalAssets_prior", "_stmt_date_balance_sheet", -Y, "asset_growth"),
    ("totalEquity", "_stmt_date_balance_sheet", None, "roic, sustainable_growth, P/B, ROE"),
    ("totalEquity_prior", "_stmt_date_balance_sheet", -Y, "sustainable_growth"),
    ("cash_bs", "_stmt_date_balance_sheet", None, "roic"),
    ("cash_sti_bs", "_stmt_date_balance_sheet", None, "net_debt_to_ebitda"),
    ("totalDebt_bs", "_stmt_date_balance_sheet", None, "roic, net_debt_to_ebitda"),
]
ANNUAL = [("totalRevenue_annual", 0, "revenue_cagr_3yr"), ("totalRevenue_3yr_ago", 3, "revenue_cagr_3yr")]
SIGNED_ABS = {"capex", "dividendsPaid", "da_cf"}      # the source signs outflows negative


def fy_ends(F, before):
    """The company's fiscal-year ends (from its annual revenue / net income facts), newest first."""
    ends = set()
    for c in REV + ("NetIncomeLoss",):
        g = F._rows(c)
        if g is not None:
            ends |= set(g.loc[g["days"].between(340, 380), "end"])
    ends = sorted({e for e in ends if e <= before + pd.Timedelta(days=10)}, reverse=True)
    out = []
    for e in ends:                                   # collapse 52/53-week wobbles
        if not out or (out[-1] - e).days > 200:
            out.append(e)
    return out


def compare_stock(F, rec, splits):
    rows = []
    for field, dkey, shift, feeds in FIELDS:
        run = rec.get(field)
        if run is None or not np.isfinite(run) or not rec.get(dkey):
            continue
        run_v = abs(run) if field in SIGNED_ABS else float(run)
        e = pd.Timestamp(rec[dkey]) + (shift if shift is not None else pd.Timedelta(0))
        cs = cands(F, field, e, rec)
        if field in SIGNED_ABS:
            cs = [(k, abs(v)) for k, v in cs]
        row = {"field": field, "feeds": feeds, "period_end": str(e.date()), "run": run_v}
        if not cs:
            row["status"] = "no filed figure"
            rows.append(row)
            continue
        k_n, v_n = min(cs, key=lambda kv: abs(kv[1] - run_v))
        k_p, v_p = cs[0]
        row.update(nearest_tag=k_n, nearest=v_n, d_nearest=rel(run_v, v_n), primary_tag=k_p, primary=v_p,
                   d_primary=rel(run_v, v_p))
        dn = abs(row["d_nearest"])
        row["status"] = "agrees" if dn <= AGREE else ("far" if dn > FAR else "near")
        if row["status"] == "far":
            row["diagnosis"] = diagnose(F, field, e, run_v, splits)
        rows.append(row)
    # the annual endpoints of the 3-year CAGR: which fiscal year did the run's figure come from?
    ends = fy_ends(F, pd.Timestamp(rec["_stmt_date_financials"]))
    for field, back, feeds in ANNUAL:
        run = rec.get(field)
        if run is None or not np.isfinite(run) or len(ends) <= back:
            continue
        found = []
        for i, e in enumerate(ends[:back + 3]):
            for c in REV:
                v = _fy(F, c, e)
                if v is not None:
                    found.append((i, e, c, v))
        if not found:
            continue
        i, e, c, v = min(found, key=lambda x: abs(x[3] - run))
        expect = [x for x in found if x[0] == back]
        k_p = max(expect, key=lambda x: abs(x[3])) if expect else None
        rows.append({"field": field, "feeds": feeds, "period_end": str(ends[back].date()), "run": float(run),
                     "nearest_tag": f"{c} FY{e.year}", "nearest": v, "d_nearest": rel(run, v),
                     "matched_years_back": i, "expected_years_back": back,
                     "primary_tag": k_p[2] if k_p else None, "primary": k_p[3] if k_p else None,
                     "d_primary": rel(run, k_p[3]) if k_p else None,
                     "status": ("agrees" if abs(rel(run, v)) <= AGREE and i == back else
                                "wrong fiscal year" if abs(rel(run, v)) <= AGREE else
                                "far" if abs(rel(run, v)) > FAR else "near")})
    return rows


def diagnose(F, field, e, run_v, splits):
    """Why a run figure matches nothing filed for its own period."""
    for lab, d in (("one quarter earlier", -Q), ("one quarter later", Q), ("a year earlier", -Y)):
        cs = cands(F, field, e + d, {})
        if field in SIGNED_ABS:
            cs = [(k, abs(v)) for k, v in cs]
        hit = [k for k, v in cs if abs(rel(run_v, v)) <= AGREE]
        if hit:
            return f"matches the filing for the period {lab} ({hit[0]})"
    cs = cands(F, field, e, {})
    for k, v in cs:
        if v:
            r = run_v / v
            for s in (1e3, 1e6, 1e-3, 1e-6) + tuple(splits):
                if abs(r / s - 1) <= 0.03:
                    return f"off by a factor of {s:g} against {k}"
    return "unexplained"


def newer_period(F, rec, run_date):
    """The newest quarter or year the company had filed by the run date, if later than the run's."""
    used = pd.Timestamp(rec.get("_stmt_date_financials"))
    newest = F.latest_end(("net_income", "revenue", "ocf"), 80, 380)
    if newest is None or (newest - used).days <= NEWER_DAYS:
        return None
    g = pd.concat([x for c in ("NetIncomeLoss", "Revenues", "NetCashProvidedByUsedInOperatingActivities")
                   if (x := F._rows(c)) is not None])
    filed = g.loc[(g["end"] - newest).abs() <= pd.Timedelta(days=5), "filed"].min()
    used_filed = g.loc[(g["end"] - used).abs() <= pd.Timedelta(days=5), "filed"].min()
    return {"used": str(used.date()), "newest": str(newest.date()), "filed": str(filed.date()),
            "used_filed": str(used_filed.date()) if pd.notna(used_filed) else None,
            "days_since_filed": int((run_date - filed).days)}


def shares_check(F, rec):
    px, mc = rec.get("currentPrice"), rec.get("marketCap")
    out = {"implied_shares": (mc / px) if px and mc else None, "yahoo_shares": rec.get("sharesOutstanding")}
    li = F.latest_instant("cover_shares")
    lo = F.latest_instant("shares_out")
    out["cover_shares"], out["cover_date"] = (li[0], str(li[1].date())) if li else (None, None)
    out["bs_shares"], out["bs_date"] = (lo[0], str(lo[1].date())) if lo else (None, None)
    ref = out["cover_shares"] or out["bs_shares"]
    out["ratio"] = (out["implied_shares"] / ref) if out["implied_shares"] and ref else None
    return out


# ----------------------------------------------------------------------------- the stakes
class Scorer:
    def __init__(self, run_dir):
        self.cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        self.base = pd.read_parquet(run_dir / "02_outliers_flagged.parquet").set_index("Ticker", drop=False)
        raw = pd.read_parquet(run_dir / "00_raw_fetch.parquet")
        self.raw = {r["Ticker"]: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                    for r in raw.to_dict("records")}
        self.metric_cols = [c for c in fe.METRIC_COLS if c in self.base.columns]
        self.published = self.score({})

    def metrics(self, rec):
        with contextlib.redirect_stdout(io.StringIO()):
            return fe.compute_metrics([rec], pd.Series(dtype=float), self.cfg).iloc[0]

    def frame(self, changes: dict, overrides: dict | None = None):
        """The run's metric frame with each changed stock's recomputed metric columns written in,
        then any direct ``{ticker: {metric: value}}`` overrides."""
        df = self.base.copy()
        for t, rec in changes.items():
            for c, (_, y) in self.changed(self.raw[t], rec).items():
                df.loc[t, c] = y
        for t, cols in (overrides or {}).items():
            for c, v in cols.items():
                df.loc[t, c] = v
        return df

    def changed(self, rec_a, rec_b):
        """{metric: (value under rec_a, value under rec_b)} for the metric columns that differ."""
        a, b = self.metrics(rec_a), self.metrics(rec_b)
        out = {}
        for c in self.metric_cols:
            x, y = a.get(c), b.get(c)
            same = (pd.isna(x) and pd.isna(y)) or (pd.notna(x) and pd.notna(y) and abs(x - y) <= 1e-12 * max(abs(x), 1))
            if not same:
                out[c] = (x, y)
        return out

    def score(self, changes: dict, frame=None, overrides=None):
        df = frame if frame is not None else self.frame(changes, overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            d = fe.compute_sector_percentiles(df.reset_index(drop=True).copy())
            d = fe.apply_percentile_transform(d, self.cfg)
            d = fe.compute_category_scores(d, self.cfg)
            d = fe.compute_composite(d, self.cfg)
            return fe.rank_stocks(d).set_index("Ticker")


def weighted(payload, t):
    """{metric: weight within its category} for the metrics that carry weight for stock t."""
    W = payload["weights"]
    wp = (payload["stock_detail"][t].get("wp") or {})
    out = {}
    for cat, prof in W["profiles"].items():
        for m, w in prof[wp.get(cat, "generic")].items():
            if w and w > 0:
                out[m] = (cat, w)
    return out


def impact(S, payload, t, rec, base_scored):
    """Composite, rank and weighted-metric percentile moves when stock t's record is ``rec``."""
    s = S.score({t: rec})
    pct = [c for c in s.columns if c.endswith("_pct")]
    moves = {}
    for m in weighted(payload, t):
        c = f"{m}_pct"
        if c in s.columns:
            a, b = base_scored.loc[t, c], s.loc[t, c]
            if pd.notna(a) or pd.notna(b):
                if pd.isna(a) or pd.isna(b) or abs(b - a) >= 0.5:
                    moves[m] = [None if pd.isna(a) else round(float(a), 1), None if pd.isna(b) else round(float(b), 1)]
    return {"composite": [float(base_scored.loc[t, "Composite"]), float(s.loc[t, "Composite"])],
            "rank": [int(base_scored.loc[t, "Rank"]), int(s.loc[t, "Rank"])], "pct_moves": moves}, s


TAX = 0.21                  # ranking_audit's statutory rate for "after-tax operating income"


def run_scenarios(S, pub, stocks, facts, run_date, ranked):
    """Re-rank the universe with each class of finding corrected, one class at a time and together."""
    base_top = set(pub.index[pub["Rank"] <= 25])
    raw = S.raw

    # 1. Statements a period behind a filing: the run's own metric moved by what the newer quarter
    #    changes, both quarters rebuilt from the filings the same way (so the definitions cancel).
    stale = {}
    for t, s in stocks.items():
        nf = s.get("newer_filing")
        if not nf or not nf.get("used_filed"):
            continue
        at = pd.Timestamp(nf["used_filed"]) + pd.Timedelta(days=1)
        old = dict(raw[t]); old.update(pr.statement_fields(facts, t, at))
        new = dict(raw[t]); new.update(pr.statement_fields(facts, t, run_date))
        cols = {}
        for c, (x, y) in S.changed(old, new).items():
            b = S.base.loc[t, c]
            if pd.notna(b) and pd.notna(x) and pd.notna(y):
                v = b + (y - x)
                if c == "sustainable_growth":
                    v = float(np.clip(v, 0.0, 1.0))
                if c == "piotroski_f_score":
                    v = float(np.clip(round(v), 0, 9))
                cols[c] = v
        stale[t] = cols

    # 2. A share class scored on the fallback forward-EPS growth while another class of the same
    #    company has the consensus basis: the other class's figure (one company, one EPS per share).
    feg = {}
    names_of = {t: (raw[t].get("longName") or raw[t].get("shortName")) for t in raw}
    for t in stocks:
        if S.base.loc[t].get("_feg_basis") != "fy2_over_trailing":
            continue
        sib = [u for u, n in names_of.items() if u != t and n == names_of[t] and u in S.base.index
               and S.base.loc[u].get("_feg_basis") == "msci_12m"]
        if sib:
            feg[t] = {"forward_eps_growth": float(S.base.loc[sib[0], "forward_eps_growth"]), "_from": sib[0]}

    # 3. Operating income as filed (OperatingIncomeLoss) where the source's figure is not it -
    #    the settled 2026-10-15 choice (keep the source's row), measured as a sensitivity.
    oi = {}
    for t, s in stocks.items():
        f = next((f for f in s["fields"] if f["field"] == "ebit"), None)
        if f and f.get("primary_tag") == "OperatingIncomeLoss" and abs(f.get("d_primary") or 0) > AGREE:
            rec = dict(raw[t]); rec["ebit"] = f["primary"]
            oi[t] = rec

    # 4. Earnings yield on after-tax operating income where net income runs above 1.25x it
    #    (ranking_audit's rule) - the refused 2026-10-16 proposal's question, as a sensitivity.
    ey = {}
    for t in stocks:
        r = raw[t]
        ni, eb, mc = r.get("netIncome"), r.get("ebit"), r.get("marketCap")
        if all(v is not None and np.isfinite(v) for v in (ni, eb, mc)) and ni > 0 and eb > 0 and mc > 0 \
                and ni > 1.25 * eb * (1 - TAX):
            ey[t] = {"earnings_yield": eb * (1 - TAX) / mc}

    def run(why, changes=None, overrides=None):
        changes, overrides = changes or {}, overrides or {}
        ov = {t: {k: v for k, v in c.items() if not k.startswith("_")} for t, c in overrides.items()}
        s = S.score(changes, overrides=ov)
        top = set(s.index[s["Rank"] <= 25])
        touched = sorted(set(changes) | set(overrides))
        return {"why": why, "stocks_changed": touched,
                "moves": {t: {"rank": [int(pub.loc[t, "Rank"]), int(s.loc[t, "Rank"])],
                              "composite": [round(float(pub.loc[t, "Composite"]), 2), round(float(s.loc[t, "Composite"]), 2)]}
                          for t in touched},
                "leave_top25": sorted(base_top - top), "enter_top25": sorted(top - base_top),
                "top40_in_top25": int(sum(1 for t in stocks if s.loc[t, "Rank"] <= 25))}

    out = {
        "1_stale_statements_refreshed": run("statements a period behind a filed quarter, moved by that quarter", overrides=stale),
        "2_share_class_consensus": run("fallback forward-EPS growth replaced by the sibling share class's consensus figure", overrides=feg),
        "3_operating_income_as_filed": run("EBIT = filed OperatingIncomeLoss (sensitivity on a settled choice)", changes=oi),
        "4_earnings_yield_on_operating_income": run("earnings yield on after-tax EBIT where NI > 1.25x it (sensitivity)", overrides=ey),
    }
    out["defects_only_1_2"] = run("1 + 2: the corrections that are not a methodology choice",
                                  overrides={**stale, **feg})
    out["everything_1_to_4"] = run("1 + 2 + 3 + 4", changes=oi, overrides={**stale, **feg, **ey})
    out["_inputs"] = {"stale_overrides": stale, "feg_overrides": feg, "oi_substituted": sorted(oi),
                      "ey_overrides": ey}
    return out


def tenq_census(facts):
    """Fiscal-year-length contexts filed in a 10-Q that disagree (>2%) with the 10-K's value for the
    same context and were never superseded by a later 10-K - what pit_record's "latest filed" rule
    read until 2026-10-19."""
    f = facts[facts["start"].notna()].copy()
    f["days"] = (f["end"] - f["start"]).dt.days
    fy = f[f["days"].between(340, 380)]
    k = fy[fy["form"].str.startswith("10-K")]
    q = fy[fy["form"].str.startswith("10-Q")]
    key = ["ticker", "concept", "start", "end"]
    m = q.merge(k[key + ["val", "filed"]], on=key, suffixes=("", "_k"))
    m["far"] = (m["val"] - m["val_k"]).abs() > AGREE * np.maximum(np.maximum(m["val"].abs(), m["val_k"].abs()), 1)
    g = m.groupby(key + ["val", "filed"]).agg(all_far=("far", "all"), last_k=("filed_k", "max")).reset_index()
    live = g[g["all_far"] & (g["last_k"] < g["filed"])]
    return {"fy_contexts_in_10q": int(len(q)), "disagreeing_with_10k": int(g["all_far"].sum()),
            "never_superseded": int(len(live)), "companies": sorted(live["ticker"].unique().tolist()),
            "rows": live.assign(end=live["end"].dt.date.astype(str), filed=live["filed"].dt.date.astype(str))
                       [["ticker", "concept", "end", "filed", "val"]].to_dict("records")}


def load_payload(run_dir):
    text = (run_dir / "dashboard_data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("{"):].rstrip().rstrip(";"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--facts", type=Path, default=ROOT / "data" / "sec" / "pit" / "facts.parquet")
    a = ap.parse_args()
    run_dir = a.run or latest_run_dir(ROOT / "runs")
    payload = load_payload(run_dir)
    run_date = pd.Timestamp(str(payload["kpis"]["run_timestamp"])[:10])
    facts = pd.read_parquet(a.facts)
    sp_path = ROOT / "cache" / "splits.parquet"
    splits = pd.read_parquet(sp_path) if sp_path.exists() else pd.DataFrame(columns=["ticker", "date", "ratio"])
    splits["date"] = pd.to_datetime(splits["date"])

    S = Scorer(run_dir)
    pub = S.published
    ranked = pd.read_parquet(run_dir / "05_final_scored.parquet").set_index("Ticker")
    assert (pub["Composite"] - ranked.loc[pub.index, "Composite"]).abs().max() < 1e-9, "rescoring does not reproduce the run"
    top = list(ranked.sort_values("Rank").index[:a.n])
    names = top + [t for t in BOT_HOLDINGS if t not in top]

    stocks = {}
    for t in names:
        rec = S.raw[t]
        F = pr.Filed(facts, t, run_date)
        sp = splits[(splits["ticker"] == t) & (splits["date"] > run_date - pd.Timedelta(days=800))]
        row = {"rank": int(ranked.loc[t, "Rank"]), "composite": float(ranked.loc[t, "Composite"]),
               "sector": ranked.loc[t, "Sector"], "bot": t in BOT_HOLDINGS,
               "dates": {k: rec.get(k) for k in ("_stmt_date_financials", "_stmt_date_balance_sheet",
                                                  "_stmt_date_cashflow", "_rev_q0_date", "_rev_q4_date")},
               "in_sec_cache": F._by != {},
               "splits_recent": [[str(d.date()), float(r)] for d, r in zip(sp["date"], sp["ratio"])],
               "shares": shares_check(F, rec), "newer_filing": newer_period(F, rec, run_date),
               "fields": compare_stock(F, rec, list(sp["ratio"])) if F._by else []}
        stocks[t] = row

    # per-field stakes: the standard (primary) figure in place of the run's, one field at a time
    for t, row in stocks.items():
        for f in row["fields"]:
            dp = f.get("d_primary")
            use = f.get("primary") if f["status"] != "far" else f.get("nearest")
            if f["field"] in ("totalRevenue_annual", "totalRevenue_3yr_ago"):
                use = f.get("primary")
            if use is None or dp is None or abs(rel(f["run"], use)) <= AGREE:
                continue
            rec = dict(S.raw[t])
            sign = -1.0 if (f["field"] in SIGNED_ABS and (S.raw[t].get(f["field"]) or 0) < 0) else 1.0
            rec[f["field"]] = sign * use
            if f["field"] == "ebit":
                rec["ebit"] = sign * use
            imp, _ = impact(S, payload, t, rec, pub)
            f["stakes"] = imp
            f["moves_10pts"] = any(v[0] is None or v[1] is None or abs(v[1] - v[0]) >= 10 for v in imp["pct_moves"].values())

    scenarios = run_scenarios(S, pub, stocks, facts, run_date, ranked)
    out = {"run": run_dir.name, "run_date": str(run_date.date()), "facts_filed_through": str(facts["filed"].max().date()),
           "top_n": a.n, "names": names, "scenarios": scenarios, "tenq_fiscal_year_census": tenq_census(facts),
           "stocks": stocks}
    for k, v in ((k, v) for k, v in scenarios.items() if not k.startswith("_")):
        print(f"{k}: leave top 25 {v['leave_top25']}; enter {v['enter_top25']}")
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(names)} stocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
