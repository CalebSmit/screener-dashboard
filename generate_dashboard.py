#!/usr/bin/env python3
"""
Interactive HTML Dashboard Generator for the Multi-Factor Screener.
===================================================================
Reads run artifacts (parquet + meta.json) and the Excel output to produce
a single self-contained HTML dashboard with Chart.js visualisations,
sortable/filterable tables, and factor analytics.

Usage:
    python generate_dashboard.py                        # latest run
    python generate_dashboard.py --run-dir runs/abc123  # specific run
"""

import argparse
import hashlib
import json
import math
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from calc_trace import verify_payload
from metric_lineage import (
    ENGINE_KEYS,
    INPUT_KEYS,
    RECOMPUTE,
    published_lineage,
)
from factor_engine import (
    METRIC_DIR,
    SECTOR_MIN_PEERS,
    WEIGHT_PROFILE_LABELS,
    published_weight_profiles,
)
from history import build_history
from stock_summary import build_summary

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _find_latest_run() -> Path:
    """Find the most recent non-test run directory.

    Uses the start_time field inside meta.json (not filesystem mtime)
    to avoid issues with OneDrive or other sync tools touching files.
    """
    runs_dir = ROOT / "runs"
    candidates = []
    for d in runs_dir.iterdir():
        if d.is_dir() and not d.name.startswith("test"):
            meta = d / "meta.json"
            if meta.exists():
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError("No valid run directories found in runs/")

    def _run_start_time(d: Path) -> str:
        try:
            with open(d / "meta.json") as f:
                return json.load(f).get("start_time", "")
        except (OSError, json.JSONDecodeError, KeyError):
            return ""

    candidates.sort(key=_run_start_time, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_raw_fetch(run_dir: Path) -> Path | None:
    """Find 00_raw_fetch.parquet — first in run_dir, then in any recent run.

    Cache-hit runs don't re-fetch data, so 00_raw_fetch.parquet only exists
    in runs that actually called the Yahoo Finance API.  Fall back to the
    most recent run that has the file so the dashboard always shows price
    target data.
    """
    local = run_dir / "00_raw_fetch.parquet"
    if local.exists():
        return local

    # Search other runs (newest first)
    runs_root = run_dir.parent
    if not runs_root.exists():
        return None

    candidates = [d for d in runs_root.iterdir()
                  if d.is_dir() and d != run_dir and (d / "00_raw_fetch.parquet").exists()]
    if not candidates:
        return None

    def _start_time(d: Path) -> str:
        try:
            with open(d / "meta.json") as f:
                return json.load(f).get("start_time", "")
        except (OSError, json.JSONDecodeError, KeyError):
            return ""

    candidates.sort(key=_start_time, reverse=True)
    return candidates[0] / "00_raw_fetch.parquet"


def load_run_data(run_dir: Path) -> dict:
    """Load all data needed for the dashboard from a run directory."""
    # Final scored data
    scored_path = run_dir / "05_final_scored.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing {scored_path}")
    df = pd.read_parquet(scored_path)

    # Merge price target + fundamental fields from raw fetch if not in final scored
    raw_path = _find_raw_fetch(run_dir)
    if raw_path is not None:
        raw = pd.read_parquet(raw_path)
        merge_cols = ["Ticker"]
        # Price target fields
        for src, dst in [("currentPrice", "_current_price"),
                         ("targetMeanPrice", "_target_mean"),
                         ("targetHighPrice", "_target_high"),
                         ("targetLowPrice", "_target_low"),
                         ("numberOfAnalystOpinions", "_num_analysts")]:
            if src in raw.columns and dst not in df.columns:
                raw = raw.rename(columns={src: dst})
                merge_cols.append(dst)
        # Fundamental financial data for Company Snapshot
        for src, dst in [("marketCap", "_mcap"),
                         ("enterpriseValue", "_ev_raw"),
                         ("totalRevenue", "_total_revenue"),
                         ("totalRevenue_prior", "_total_revenue_prior"),
                         ("grossProfit", "_gross_profit"),
                         ("netIncome", "_net_income"),
                         ("netIncome_prior", "_net_income_prior"),
                         ("ebitda", "_ebitda_raw"),
                         ("operatingCashFlow", "_ocf"),
                         ("capex", "_capex"),
                         ("totalDebt", "_total_debt"),
                         ("totalCash", "_total_cash"),
                         ("cash_bs", "_cash_bs"),
                         ("totalAssets", "_total_assets"),
                         ("totalEquity", "_total_equity"),
                         ("dividendRate", "_dividend_rate"),
                         ("payoutRatio", "_payout_ratio"),
                         ("sharesOutstanding", "_shares_out"),
                         ("trailingEps", "_trailing_eps"),
                         ("forwardEps", "_forward_eps"),
                         ("shortRatio", "_short_ratio"),
                         # Display-only descriptive fields for the drilldown.
                         ("longBusinessSummary", "_about"),
                         ("industry", "_industry"),
                         # Next scheduled earnings date - display only, never
                         # scored. See `_earnings_block` for the conventions.
                         ("earningsTimestampStart", "_earn_start"),
                         ("earningsTimestampEnd", "_earn_end"),
                         ("isEarningsDateEstimate", "_earn_est")]:
            if src in raw.columns and dst not in df.columns:
                raw = raw.rename(columns={src: dst})
                merge_cols.append(dst)
        if len(merge_cols) > 1:
            df = df.merge(raw[merge_cols], on="Ticker", how="left")
        # The figures behind each metric, as fetched, under an `_in_` prefix so they
        # can never collide with a scored column. Read afresh: the loop above renames
        # some of these columns away.
        try:
            raw_full = pd.read_parquet(raw_path)
            in_cols = [k for k in tuple(INPUT_KEYS) + PROVENANCE_KEYS if k in raw_full.columns]
            if in_cols:
                df = df.merge(raw_full[["Ticker"] + in_cols].rename(
                    columns={k: "_in_" + k for k in in_cols}), on="Ticker", how="left")
        except (OSError, ValueError):
            pass  # no inputs for this run: the page shows formulas without numbers

    # Metadata
    meta_path = run_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Effective weights
    weights_path = run_dir / "effective_weights.json"
    weights = {}
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)

    # Load defensibility artifacts (optional — graceful fallback)
    sens_df = None
    sens_path = run_dir / "07_weight_sensitivity.parquet"
    if sens_path.exists():
        try:
            sens_df = pd.read_parquet(sens_path)
        except (OSError, ValueError):
            sens_df = None  # Graceful fallback — section won't render

    corr_df = None
    corr_path = run_dir / "06_factor_correlation.parquet"
    if corr_path.exists():
        try:
            corr_df = pd.read_parquet(corr_path)
        except (OSError, ValueError):
            corr_df = None  # Graceful fallback — section won't render

    # Load config snapshot for trap filter thresholds
    cfg_path = run_dir / "config.yaml"
    run_cfg = {}
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            run_cfg = yaml.safe_load(f) or {}

    return {
        "df": df,
        "meta": meta,
        "weights": weights,
        "sens_df": sens_df,
        "corr_df": corr_df,
        "cfg": run_cfg,
    }


# ---------------------------------------------------------------------------
# Prepare JSON data for the dashboard
# ---------------------------------------------------------------------------

def _clean_text(value) -> str:
    """Normalise a provider text field for JSON embedding.

    Returns "" for NaN/None so the front end can test truthiness directly
    rather than rendering the string "nan" to a reader.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


def _run_date_for_history(meta: dict) -> str | None:
    """The run's calendar date, matching how snapshots are keyed.

    ``improvement_engine`` names snapshots ``{run_date}_{run_id}.parquet``, so
    the dashboard has to agree on the date or it would append a duplicate point
    for the run it is publishing.
    """
    raw = meta.get("run_date") or meta.get("start_time") or ""
    if not raw:
        return None
    return str(raw)[:10]


def _earnings_block(row) -> dict | None:
    """The next scheduled earnings window for one stock, or ``None``.

    Display-only (plan/dashboard-north-star.md gap 4). Returns
    ``{"d": "YYYY-MM-DD", "end": "YYYY-MM-DD"|absent, "est": bool}``.

    Three conventions, each of which exists because of something measured:

    * **Only `earningsTimestampStart`/`End` are read.** `earningsTimestamp`
      means the last report for some tickers and the next for others
      (`factor_engine._fetch_single_ticker_inner`), so it is not captured.
    * **`end` is emitted only when it differs from the start.** Measured across
      all 503 tickers on 2026-09-29 the two were equal every time, so a window
      is the rare case and carrying a duplicate date on every stock is pure
      payload.
    * **`est` is always present, never defaulted to False.** 209 of the 492
      future dates (42.5%) were provider estimates; a missing flag read as
      "confirmed" would overstate four dates in ten.

    Reading the UTC date is safe: the provider stamps these at 12:30 or 20:00
    UTC only (08:30 / 16:00 US/Eastern, before the open or after the close), so
    a UTC read and an Eastern read disagreed for 0 of 503.
    """
    start = _epoch_to_date(row.get("_earn_start"))
    if not start:
        return None
    block: dict = {"d": start}
    end = _epoch_to_date(row.get("_earn_end"))
    if end and end != start:
        block["end"] = end
    block["est"] = bool(row.get("_earn_est")) if pd.notna(row.get("_earn_est")) else False
    return block


def _epoch_to_date(value) -> str | None:
    """A yfinance UNIX timestamp as an ISO date, or ``None`` if unusable."""
    try:
        if value is None or pd.isna(value):
            return None
        return datetime.fromtimestamp(float(value), UTC).date().isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


CATEGORIES = ["valuation", "quality", "growth", "momentum",
              "risk", "revisions", "size", "investment"]

# The cadence the methodology is built for, versus the cadence the site
# refreshes at. Until 2026-09-17 the dashboard stated neither.
#
# `research/2026-09-14-sell-discipline-and-hold-bands.md` §8.2, measured on this
# repo's own snapshots: acting on the strict top-25 rule at every run implies
# **121.8%** monthly one-sided turnover, against **24.0%** reviewing the same
# rule monthly. Novy-Marx & Velikov (2016, *RFS* 29(1) 104-147) find anomalies
# under roughly 50% monthly one-sided turnover mostly survive trading costs and
# few above it do - so daily action sits 2.4x outside the surviving region, a
# gap that tolerates a large error in the estimate before it reverses.
#
# This is a product defect rather than a methodology one. `config.yaml` has
# recorded a quarterly cadence since launch; the dashboard regenerated every
# weekday and said nothing, which implicitly invites a reader to act on every
# redraw. The fix is to say it, not to stop refreshing - fresh data is the point
# of the data loop, and the evidence base depends on it.
CADENCE_LABELS = {"monthly": "monthly", "quarterly": "quarterly"}
CADENCE_TURNOVER = {
    # cadence -> (monthly one-sided turnover %, the label used in the copy)
    "every run": 121.8,
    "monthly": 24.0,
}
# Novy-Marx & Velikov's survivability boundary, in monthly one-sided turnover %.
NMV_TURNOVER_CEILING = 50.0


def _cadence_block(cfg: dict) -> dict:
    """What review cadence to tell the reader the tool is built for.

    Read from the *run's* config snapshot rather than the working tree, so the
    published page states what the run it describes was configured for. Falls
    back to quarterly, which is what `config.yaml` has recorded since launch -
    an unknown cadence must not render as "no cadence", because "no cadence" is
    exactly the daily-action reading this block exists to correct.
    """
    portfolio = (cfg or {}).get("portfolio") or {}
    raw = str(portfolio.get("review_cadence") or "").strip().lower()
    cadence = raw if raw in CADENCE_LABELS else "quarterly"
    num_stocks = portfolio.get("num_stocks")
    return {
        "review": cadence,
        "label": CADENCE_LABELS[cadence],
        # False when the run's config snapshot predates `review_cadence` or
        # carries an unrecognised value, so the fallback is visible in the
        # payload instead of being indistinguishable from a real setting.
        "configured": raw in CADENCE_LABELS,
        "num_stocks": int(num_stocks) if isinstance(num_stocks, (int, float)) else None,
        "turnover_every_run": CADENCE_TURNOVER["every run"],
        "turnover_monthly": CADENCE_TURNOVER["monthly"],
        "turnover_ceiling": NMV_TURNOVER_CEILING,
    }


def _effective_weight_row(weights: dict, present: list) -> dict:
    """Per-stock weights after redistributing away categories with no score.

    Mirrors ``factor_engine.compute_factor_contributions`` exactly: a category
    with no data drops out and the survivors are renormalised over the weight
    that remains. The JS does the same arithmetic for display; this is the
    Python side used to verify it.
    """
    live = {c: weights.get(c, 0) for c in present if weights.get(c, 0) > 0}
    total = sum(live.values())
    if total <= 0:
        return {c: 0.0 for c in CATEGORIES}
    return {c: (live[c] / total * 100.0 if c in live else 0.0)
            for c in CATEGORIES}


def _derive_factor_weights(df) -> dict | None:
    """Recover the weights actually used, from the scored rows themselves.

    For a stock with all eight categories populated the weights are not
    renormalised, so ``contrib = score * weight / 100`` and the weight falls
    straight out. Taking the median over every such stock is robust to the
    2-decimal rounding on ``*_contrib``.

    Used only as a cross-check and last-resort fallback: the weights are
    supposed to arrive from ``effective_weights.json``.
    """
    score_cols = {c: f"{c}_score" for c in CATEGORIES}
    contrib_cols = {c: f"{c}_contrib" for c in CATEGORIES}
    if any(col not in df.columns for col in
           list(score_cols.values()) + list(contrib_cols.values())):
        return None

    full = df.dropna(subset=list(score_cols.values()))
    if len(full) < 20:
        return None

    derived = {}
    for cat in CATEGORIES:
        s, c = full[score_cols[cat]], full[contrib_cols[cat]]
        usable = s > 1.0  # tiny scores make contrib/score rounding-dominated
        if usable.sum() < 20:
            return None
        derived[cat] = float((c[usable] / s[usable] * 100.0).median())

    total = sum(derived.values())
    if not (99.0 <= total <= 101.0):
        return None
    return {c: round(v, 2) for c, v in derived.items()}


def _reconcile_factor_weights(df, weights: dict) -> dict:
    """Make sure the weights we publish reproduce the contributions we publish.

    The dashboard prints its own arithmetic to the reader - "Score 65.3 x 13%
    = 9.76 pts". That is the tool's central teaching claim, and between
    2026-02 and 2026-08-28 it was false for 501 of 502 stocks: the volatility
    regime adjustment was applied to the composite but never reached
    ``effective_weights.json``, so the page showed 13% where 14.95% had been
    used. Nothing failed, because nobody was checking that the sum added up.

    So check it here, every build. Weights that do not reproduce the
    contributions are not published.
    """
    fw = dict(weights.get("factor_weights") or {})
    if not fw or df is None or df.empty:
        return weights

    derived = _derive_factor_weights(df)
    if derived is None:
        return weights  # cannot verify (e.g. tiny universe) - leave as-is

    worst = max(abs(derived[c] - fw.get(c, 0)) for c in CATEGORIES)
    if worst <= 0.05:
        return weights  # recorded weights check out

    print(f"  [WEIGHTS] Recorded factor weights do not reproduce the published "
          f"contributions (worst gap {worst:.2f}pp). Publishing the weights the "
          f"scores were actually built from.")
    for cat in CATEGORIES:
        if abs(derived[cat] - fw.get(cat, 0)) > 0.05:
            print(f"    {cat}: recorded {fw.get(cat, 0)} -> actual {derived[cat]}")

    weights = dict(weights)
    weights.setdefault("base_factor_weights", fw)
    weights["factor_weights"] = derived
    weights["factor_weights_adjusted"] = derived != dict(
        weights.get("base_factor_weights") or {})
    weights["factor_weights_derived"] = True
    return weights


def _with_weight_profiles(weights: dict, run_cfg: dict) -> dict:
    """Attach the metric-weight tables, the labels that explain them, and the
    coverage-discount parameters, so every number in a score's workings is published.

    The engine saves the tables it used in ``effective_weights.json``. A run that
    predates that falls back to resolving them from its own config snapshot with the
    engine's own function - never a second implementation of the rules.
    """
    weights = dict(weights)
    profiles = weights.get("profiles")
    if not profiles:
        cfg = dict(run_cfg)
        # `metric_weights` in effective_weights.json is the post-auto-reduce set the
        # scorer really saw; the snapshot predates that mutation.
        if weights.get("metric_weights"):
            cfg["metric_weights"] = weights["metric_weights"]
        if cfg.get("metric_weights"):
            profiles = published_weight_profiles(cfg)
    if profiles:
        weights["profiles"] = profiles
        weights["profile_labels"] = dict(WEIGHT_PROFILE_LABELS)
    cov = (run_cfg.get("data_quality") or {}).get("coverage_discount") or {}
    if cov:
        weights["coverage_discount"] = {
            "enabled": bool(cov.get("enabled", False)),
            "threshold": cov.get("threshold", 0.80),
            "rate": cov.get("penalty_rate", 0.15),
        }
    return weights


# Dates of the statements each stock's figures come from (display only).
PROVENANCE_KEYS = ("_stmt_date_balance_sheet", "_stmt_date_cashflow", "_stmt_date_financials")


def _compact(v):
    """JSON-safe number: whole floats become ints, so 4,891,001,487,360.0 ships as
    4891001487360 - the figure is identical and the payload smaller."""
    v = _safe(v)
    if isinstance(v, float) and v == v and abs(v) < 1e15 and v == round(v):
        return int(v)
    return v


def _stock_inputs(row) -> dict:
    """The reported figures behind this stock's metrics, as the scorer used them."""
    inp = {}
    for k in INPUT_KEYS:
        v = _compact(row.get("_in_" + k))
        if v is not None:
            inp[k] = v
    for k in ENGINE_KEYS:
        v = _compact(row.get("_" + k))
        if v is not None:
            inp[k] = v
    return inp


def _input_mismatches(inp: dict, raw: dict) -> list:
    """Metrics whose published value does NOT rebuild from the inputs published
    beside it. The page says so for these instead of showing a formula that fails."""
    bad = []
    for m, fn in RECOMPUTE.items():
        pub = raw.get(m)
        if pub is None:
            continue
        calc = fn(inp)
        if calc is None or abs(calc - pub) > 1e-4 * max(1.0, abs(calc)) + 1e-4:
            bad.append(m)
    return bad


def _sector_stats(df, metrics) -> dict:
    """Per sector and metric: [valid count, lower quartile, median, upper quartile].

    Descriptive statistics of the raw values the percentiles were ranked over, so the
    page can say who a stock was ranked against. A sector with fewer than
    ``SECTOR_MIN_PEERS`` valid values is ranked against the universe instead - the same
    constant the scorer reads.
    """
    out = {}
    for sector, grp in df.groupby("Sector"):
        out[sector] = {}
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna().astype(float)
            if len(vals) == 0:
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            out[sector][m] = [int(len(vals)), round(float(q1), 4), round(float(med), 4), round(float(q3), 4)]
    return out


def _lineage_check(stock_detail: dict) -> dict:
    """Per metric: [stocks whose inputs reproduce the value, stocks that have one]."""
    out = {m: [0, 0] for m in RECOMPUTE}
    for d in stock_detail.values():
        bad = set(d.get("inp_bad") or ())
        for m in RECOMPUTE:
            if (d.get("raw") or {}).get(m) is None:
                continue
            out[m][1] += 1
            if m not in bad:
                out[m][0] += 1
    return out


class CalculationMismatch(RuntimeError):
    """The published weights and numbers do not reproduce the published scores."""


def _reconcile_scores(weights: dict, stock_detail: dict, run_cfg: dict) -> None:
    """Refuse to build a page whose arithmetic does not add up.

    Skipped, loudly, when the run carries no weight tables (an older run directory)
    - the page then makes no per-metric weight claim either. With factor
    neutralisation on, category scores are rewritten before the composite, so only
    the category-level check applies.
    """
    if not weights.get("profiles"):
        print("  [CALC] No weight profiles for this run - per-metric workings not "
              "published and not checked.")
        return
    neutralised = bool((run_cfg.get("factor_neutralization") or {}).get("enabled", False))
    result = verify_payload({"weights": weights, "stock_detail": stock_detail},
                            check_composite=not neutralised)
    if result["failing_stocks"]:
        sample = list(result["failures"].items())[:5]
        lines = "; ".join(f"{t}: {p[0]}" for t, p in sample)
        raise CalculationMismatch(
            f"{result['failing_stocks']} of {result['stocks']} stocks do not reproduce "
            f"their published scores from the published weights ({lines}). "
            f"Not publishing.")
    print(f"  [CALC] Reproduced {result['category_pairs']} category scores and "
          f"{result['stocks']} composites from the published payload alone.")


def prepare_dashboard_data(run_data: dict) -> str:
    """Convert run data into a JSON string for embedding in HTML."""
    df = run_data["df"]
    meta = run_data["meta"]

    weights = _reconcile_factor_weights(df, run_data.get("weights", {}))
    weights = _with_weight_profiles(weights, run_data.get("cfg") or {})

    # --- Raw metric columns, percentile columns, contribution columns ---
    raw_metrics = [
        "ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales", "pb_ratio",
        "roic", "gross_profit_assets", "debt_equity", "net_debt_to_ebitda",
        "piotroski_f_score", "accruals", "operating_leverage", "beneish_m_score",
        "roe", "roa", "equity_ratio",
        "forward_eps_growth", "peg_ratio", "revenue_growth", "revenue_cagr_3yr", "sustainable_growth",
        "return_12_1", "return_6m", "jensens_alpha",
        "volatility", "beta", "sharpe_ratio", "sortino_ratio", "max_drawdown_1y",
        "fy1_revision_3m",
        "analyst_surprise", "price_target_upside", "earnings_acceleration", "consecutive_beat_streak",
        "short_interest_ratio",
        "size_log_mcap", "asset_growth",
    ]
    pct_cols = [m + "_pct" for m in raw_metrics]
    contrib_cols = ["valuation_contrib", "quality_contrib", "growth_contrib",
                    "momentum_contrib", "risk_contrib", "revisions_contrib",
                    "size_contrib", "investment_contrib"]

    # --- Universe table data (summary) ---
    table_cols = ["Ticker", "Company", "Sector", "Composite", "Rank",
                  "valuation_score", "quality_score", "growth_score",
                  "momentum_score", "risk_score", "revisions_score",
                  "size_score", "investment_score",
                  "Value_Trap_Flag", "Growth_Trap_Flag",
                  "Value_Trap_Severity", "Growth_Trap_Severity"]
    table_data = []
    for _, row in df.iterrows():
        table_data.append({c: _safe(row.get(c)) for c in table_cols})

    # --- Pre-compute sector peer groups for comparison ---
    _peer_cols = ["Ticker", "Company", "Sector", "Composite", "Rank",
                  "valuation_score", "quality_score",
                  "revenue_growth", "earnings_yield", "roic", "roe"]
    _peer_cols_present = [c for c in _peer_cols if c in df.columns]
    _sector_groups = {}
    for sector, grp in df.groupby("Sector"):
        # Sort by market cap (use _mcap if available, else _mc)
        mcap_col = "_mcap" if "_mcap" in grp.columns else "_mc"
        if mcap_col in grp.columns:
            grp = grp.sort_values(mcap_col, ascending=False, na_position="last")
        _sector_groups[sector] = grp[_peer_cols_present + [mcap_col]].copy() if mcap_col in grp.columns else grp[_peer_cols_present].copy()

    def _get_peers(ticker, sector, n=5):
        """Get n closest peers by market cap in the same sector."""
        grp = _sector_groups.get(sector)
        if grp is None or len(grp) < 2:
            return []
        mcap_col = "_mcap" if "_mcap" in grp.columns else "_mc"
        # Exclude self, take top n by market cap proximity
        others = grp[grp["Ticker"] != ticker]
        if mcap_col in grp.columns:
            self_mcap = grp.loc[grp["Ticker"] == ticker, mcap_col].values
            if len(self_mcap) > 0 and pd.notna(self_mcap[0]) and self_mcap[0] > 0:
                others = others.copy()
                others["_mcap_dist"] = (others[mcap_col] / self_mcap[0] - 1).abs()
                others = others.sort_values("_mcap_dist").head(n)
            else:
                others = others.head(n)
        else:
            others = others.head(n)
        peers = []
        for _, r in others.iterrows():
            peer = {
                "ticker": r.get("Ticker", ""),
                "company": r.get("Company", ""),
                "composite": _safe(r.get("Composite")),
                "rank": _safe(r.get("Rank")),
                "val_score": _safe(r.get("valuation_score")),
                "qual_score": _safe(r.get("quality_score")),
                "rev_growth": _safe(r.get("revenue_growth")),
                "pe_ratio": round(1.0 / r["earnings_yield"], 1) if pd.notna(r.get("earnings_yield")) and r["earnings_yield"] > 0 else None,
                "roic": _safe(r.get("roic")),
                "roe": _safe(r.get("roe")),
                "mcap": _safe(r.get("_mcap") if "_mcap" in r.index else r.get("_mc")),
            }
            peers.append(peer)
        return peers

    # --- Per-stock detail data (for drill-down) keyed by ticker ---
    stock_detail = {}
    for _, row in df.iterrows():
        ticker = row["Ticker"]
        detail = {}
        # Raw metric values
        detail["raw"] = {m: _safe(row.get(m)) for m in raw_metrics}
        # Percentile ranks
        detail["pct"] = {m: _safe(row.get(m + "_pct")) for m in raw_metrics}
        # Category scores
        detail["cat_scores"] = {
            "valuation": _safe(row.get("valuation_score")),
            "quality": _safe(row.get("quality_score")),
            "growth": _safe(row.get("growth_score")),
            "momentum": _safe(row.get("momentum_score")),
            "risk": _safe(row.get("risk_score")),
            "revisions": _safe(row.get("revisions_score")),
            "size": _safe(row.get("size_score")),
            "investment": _safe(row.get("investment_score")),
        }
        # Contributions to composite
        detail["contrib"] = {
            "valuation": _safe(row.get("valuation_contrib")),
            "quality": _safe(row.get("quality_contrib")),
            "growth": _safe(row.get("growth_contrib")),
            "momentum": _safe(row.get("momentum_contrib")),
            "risk": _safe(row.get("risk_contrib")),
            "revisions": _safe(row.get("revisions_contrib")),
            "size": _safe(row.get("size_contrib")),
            "investment": _safe(row.get("investment_contrib")),
        }
        detail["composite"] = _safe(row.get("Composite"))
        detail["rank"] = _safe(row.get("Rank"))
        detail["sector"] = row.get("Sector", "")
        detail["company"] = row.get("Company", "")
        # Display-only context. Not scored; sourced verbatim from the data
        # provider so a reader can see what the company actually does.
        detail["industry"] = _clean_text(row.get("_industry"))
        detail["about"] = _clean_text(row.get("_about"))
        # Next scheduled earnings date. Display only - `tests/test_earnings_date.py`
        # asserts it never reaches `raw`/`pct`, the same guard `about` carries.
        _earn = _earnings_block(row)
        if _earn:
            detail["earn"] = _earn
        detail["vt"] = _safe(row.get("Value_Trap_Flag"))
        detail["gt"] = _safe(row.get("Growth_Trap_Flag"))
        # Analyst price targets (dollar values)
        detail["price"] = _safe(row.get("_current_price"))
        detail["pt_mean"] = _safe(row.get("_target_mean"))
        detail["pt_high"] = _safe(row.get("_target_high"))
        detail["pt_low"] = _safe(row.get("_target_low"))
        detail["num_analysts"] = _safe(row.get("_num_analysts"))
        # Data provenance fields
        detail["eps_mismatch"] = bool(row.get("_eps_basis_mismatch")) if pd.notna(row.get("_eps_basis_mismatch")) else False
        detail["eps_ratio"] = _safe(row.get("_eps_ratio"))
        detail["data_source"] = row.get("_data_source", None)
        detail["metric_count"] = _safe(row.get("_metric_count"))
        detail["metric_total"] = _safe(row.get("_metric_total"))
        # The reported figures behind the metrics, the nine Piotroski signals and the
        # eight Beneish indices. Display only: none of it is scored here.
        detail["inp"] = _stock_inputs(row)
        _pio = row.get("_pio_signals")
        if isinstance(_pio, str) and _pio:
            detail["pio"] = _pio
        _bn = row.get("_beneish_idx")
        if isinstance(_bn, str) and _bn:
            detail["bn"] = _bn
        _asof = {}
        for _k, _src in (("bs", "_stmt_date_balance_sheet"), ("cf", "_stmt_date_cashflow"),
                         ("is", "_stmt_date_financials")):
            _v = row.get("_in_" + _src)
            if isinstance(_v, str) and _v:
                _asof[_k] = _v[:10]
        if _asof:
            detail["asof"] = _asof
        _bad = _input_mismatches(detail["inp"], detail["raw"])
        if _bad:
            detail["inp_bad"] = _bad
        # Which weight table each category was scored with (omitted when it is the
        # generic one) and the coverage the composite discount actually read.
        _wp = {c: row.get("_wp_" + c) for c in
               ("valuation", "quality", "growth", "momentum", "risk",
                "revisions", "size", "investment")}
        _wp = {c: v for c, v in _wp.items() if isinstance(v, str) and v != "generic"}
        if _wp:
            detail["wp"] = _wp
        if pd.notna(row.get("_cov_applicable")):
            _n = int(row["_cov_present"])
            _of = int(row["_cov_applicable"])
            detail["cov"] = {"n": _n, "of": _of}
            _disc = _safe(row.get("_cov_discount"))
            if _disc:
                detail["cov"]["disc"] = round(_disc, 6)
            # "N of M metrics" now uses the same count the discount reads. It used
            # a hard-coded list of 18 while the discount used 35 or 41.
            detail["metric_count"] = _n
            detail["metric_total"] = _of

        # --- Company Snapshot (financials) ---
        _mcap = _safe(row.get("_mcap"))
        _rev = _safe(row.get("_total_revenue"))
        _rev_p = _safe(row.get("_total_revenue_prior"))
        _ni = _safe(row.get("_net_income"))
        _ni_p = _safe(row.get("_net_income_prior"))
        _gp = _safe(row.get("_gross_profit"))
        _ocf = _safe(row.get("_ocf"))
        _capex_v = _safe(row.get("_capex"))
        _debt = _safe(row.get("_total_debt"))
        _cash = _safe(row.get("_total_cash"))
        if _cash is None:
            _cash = _safe(row.get("_cash_bs"))
        _price = _safe(row.get("_current_price"))
        _div_rate = _safe(row.get("_dividend_rate"))

        detail["financials"] = {
            "market_cap": _mcap,
            "enterprise_value": _safe(row.get("_ev_raw")),
            "revenue": _rev,
            "revenue_growth_yoy": round((_rev - _rev_p) / abs(_rev_p), 4) if (_rev is not None and _rev_p is not None and abs(_rev_p) > 0) else None,
            "net_income": _ni,
            "ni_growth_yoy": round((_ni - _ni_p) / abs(_ni_p), 4) if (_ni is not None and _ni_p is not None and abs(_ni_p) > 0) else None,
            "ebitda": _safe(row.get("_ebitda_raw")),
            "gross_margin": round(_gp / _rev, 4) if (_gp is not None and _rev is not None and _rev > 0) else None,
            "net_margin": round(_ni / _rev, 4) if (_ni is not None and _rev is not None and _rev > 0) else None,
            "fcf": round(_ocf - abs(_capex_v), 2) if (_ocf is not None and _capex_v is not None) else None,
            "total_debt": _debt,
            "total_cash": _cash,
            "net_debt": round(_debt - _cash, 2) if (_debt is not None and _cash is not None) else None,
            "dividend_yield": round(_div_rate / _price, 4) if (_div_rate is not None and _price is not None and _price > 0) else None,
            "payout_ratio": _safe(row.get("_payout_ratio")),
            "shares_outstanding": _safe(row.get("_shares_out")),
            "trailing_eps": _safe(row.get("_trailing_eps")),
            "forward_eps": _safe(row.get("_forward_eps")),
            "avg_daily_dollar_vol": _safe(row.get("avg_daily_dollar_volume")),
            "short_ratio": _safe(row.get("_short_ratio")),
        }

        # --- Flags & Warnings ---
        detail["flags"] = {
            "vt_severity": _safe(row.get("Value_Trap_Severity")),
            "gt_severity": _safe(row.get("Growth_Trap_Severity")),
            "is_bank": bool(row.get("_is_bank_like")) if pd.notna(row.get("_is_bank_like")) else False,
            "fin_caveat": bool(row.get("Financial_Sector_Caveat")) if pd.notna(row.get("Financial_Sector_Caveat")) else False,
            "beneish_flag": bool(row.get("_beneish_flag")) if pd.notna(row.get("_beneish_flag")) else False,
            "channel_stuffing": bool(row.get("_channel_stuffing_flag")) if pd.notna(row.get("_channel_stuffing_flag")) else False,
            "recv_rev_divergence": _safe(row.get("_recv_rev_divergence")),
            "ev_flag": bool(row.get("_ev_flag")) if pd.notna(row.get("_ev_flag")) else False,
            "beta_overlap_pct": _safe(row.get("_beta_overlap_pct")),
            "ltm_annualized": bool(row.get("_ltm_annualized")) if pd.notna(row.get("_ltm_annualized")) else False,
            "stale_data": bool(row.get("_stale_data")) if pd.notna(row.get("_stale_data")) else False,
            "stmt_age_days": _safe(row.get("_stmt_age_days")),
        }

        # --- Sector Peers ---
        detail["peers"] = _get_peers(ticker, row.get("Sector", ""))
        # Self metrics for comparison highlight
        detail["self_metrics"] = {
            "rev_growth": _safe(row.get("revenue_growth")),
            "pe_ratio": round(1.0 / row["earnings_yield"], 1) if pd.notna(row.get("earnings_yield")) and row["earnings_yield"] > 0 else None,
            "net_margin": round(_ni / _rev, 4) if (_ni is not None and _rev is not None and _rev > 0) else None,
            "roic": _safe(row.get("roic")),
            "roe": _safe(row.get("roe")),
            "debt_equity": _safe(row.get("debt_equity")),
            "div_yield": round(_div_rate / _price, 4) if (_div_rate is not None and _price is not None and _price > 0) else None,
            "fcf_yield": _safe(row.get("fcf_yield")),
            "mcap": _mcap,
        }

        stock_detail[ticker] = detail

    # --- Sector stats ---
    sectors = sorted(df["Sector"].unique())
    sector_composition = {s: int(c) for s, c in df["Sector"].value_counts().items()}

    # --- Composite histogram ---
    hist_values, hist_edges = np.histogram(df["Composite"].dropna(), bins=20, range=(0, 100))
    hist_labels = [f"{int(hist_edges[i])}-{int(hist_edges[i+1])}" for i in range(len(hist_values))]

    # --- Value trap by sector ---
    vt_by_sector = {}
    for sector in sectors:
        s_df = df[df["Sector"] == sector]
        total = len(s_df)
        flagged = int(s_df["Value_Trap_Flag"].sum()) if "Value_Trap_Flag" in s_df.columns else 0
        vt_by_sector[sector] = {"total": total, "flagged": flagged,
                                "rate": round(flagged / total * 100, 1) if total > 0 else 0}

    # --- Growth trap by sector ---
    gt_by_sector = {}
    for sector in sectors:
        s_df = df[df["Sector"] == sector]
        total = len(s_df)
        flagged = int(s_df["Growth_Trap_Flag"].sum()) if "Growth_Trap_Flag" in s_df.columns else 0
        gt_by_sector[sector] = {"total": total, "flagged": flagged,
                                "rate": round(flagged / total * 100, 1) if total > 0 else 0}

    # --- Factor score distributions by sector (for boxplots) ---
    factor_cols_all = ["Composite", "valuation_score", "quality_score", "growth_score",
                       "momentum_score", "risk_score", "revisions_score",
                       "size_score", "investment_score"]
    # Only include factors that exist in this run's data
    factor_cols = [c for c in factor_cols_all if c in df.columns]
    sector_distributions = {}
    for factor in factor_cols:
        sector_distributions[factor] = {}
        for sector in sectors:
            vals = df[df["Sector"] == sector][factor].dropna().tolist()
            if vals:
                vals_sorted = sorted(vals)
                n = len(vals_sorted)
                sector_distributions[factor][sector] = {
                    "min": round(vals_sorted[0], 1),
                    "q1": round(vals_sorted[max(0, n // 4)], 1),
                    "median": round(vals_sorted[n // 2], 1),
                    "mean": round(sum(vals) / n, 1),
                    "q3": round(vals_sorted[min(n - 1, 3 * n // 4)], 1),
                    "max": round(vals_sorted[-1], 1),
                    "count": n,
                }

    # --- KPIs ---
    kpis = {
        "run_timestamp": meta.get("start_time", ""),
        "universe_size": len(df),
        "stocks_scored": int(df["Composite"].notna().sum()),
        "value_traps": int(df["Value_Trap_Flag"].sum()) if "Value_Trap_Flag" in df.columns else 0,
        "growth_traps": int(df["Growth_Trap_Flag"].sum()) if "Growth_Trap_Flag" in df.columns else 0,
        "avg_composite": round(float(df["Composite"].mean()), 1),
        "median_composite": round(float(df["Composite"].median()), 1),
    }

    # Metric display metadata
    metric_meta = {
        "ev_ebitda": {"label": "EV/EBITDA", "fmt": "ratio", "category": "valuation"},
        "fcf_yield": {"label": "FCF Yield", "fmt": "pct", "category": "valuation"},
        "earnings_yield": {"label": "Earnings Yield", "fmt": "pct", "category": "valuation"},
        "ev_sales": {"label": "EV/Sales", "fmt": "ratio", "category": "valuation"},
        "pb_ratio": {"label": "P/B Ratio", "fmt": "ratio", "category": "valuation"},
        "roic": {"label": "ROIC", "fmt": "pct", "category": "quality"},
        "gross_profit_assets": {"label": "Gross Profit/Assets", "fmt": "pct", "category": "quality"},
        "debt_equity": {"label": "Debt/Equity", "fmt": "ratio", "category": "reference"},  # Reference only; not scored
        "net_debt_to_ebitda": {"label": "Net Debt/EBITDA", "fmt": "ratio", "category": "quality"},
        "piotroski_f_score": {"label": "Piotroski F-Score", "fmt": "int", "category": "quality"},
        "accruals": {"label": "Accruals", "fmt": "pct", "category": "quality"},
        "operating_leverage": {"label": "Operating Leverage", "fmt": "ratio", "category": "quality"},
        "beneish_m_score": {"label": "Beneish M-Score", "fmt": "ratio", "category": "quality"},
        "roe": {"label": "ROE", "fmt": "pct", "category": "quality"},
        "roa": {"label": "ROA", "fmt": "pct", "category": "quality"},
        "equity_ratio": {"label": "Equity Ratio", "fmt": "pct", "category": "quality"},
        "forward_eps_growth": {"label": "Fwd EPS Growth", "fmt": "pct", "category": "growth"},
        "peg_ratio": {"label": "PEG Ratio", "fmt": "ratio", "category": "growth"},
        "revenue_growth": {"label": "Revenue Growth", "fmt": "pct", "category": "growth"},
        "revenue_cagr_3yr": {"label": "Revenue CAGR (3Y)", "fmt": "pct", "category": "growth"},
        "sustainable_growth": {"label": "Sustainable Growth", "fmt": "pct", "category": "growth"},
        "return_12_1": {"label": "12-1M Return", "fmt": "pct", "category": "momentum"},
        "return_6m": {"label": "6M Return", "fmt": "pct", "category": "momentum"},
        "jensens_alpha": {"label": "Jensen's Alpha", "fmt": "pct", "category": "momentum"},
        "volatility": {"label": "Volatility", "fmt": "pct", "category": "risk"},
        "beta": {"label": "Beta", "fmt": "ratio", "category": "risk"},
        "sharpe_ratio": {"label": "Sharpe Ratio", "fmt": "ratio", "category": "risk"},
        "sortino_ratio": {"label": "Sortino Ratio", "fmt": "ratio", "category": "risk"},
        "max_drawdown_1y": {"label": "Max Drawdown (1Y)", "fmt": "pct", "category": "risk"},
        "fy1_revision_3m": {"label": "FY1 EPS Revision (90d)", "fmt": "bp", "category": "revisions"},
        "analyst_surprise": {"label": "Analyst Surprise", "fmt": "pct", "category": "revisions"},
        "price_target_upside": {"label": "Price Target Upside", "fmt": "pct", "category": "revisions"},
        "earnings_acceleration": {"label": "Earnings Accel.", "fmt": "ratio", "category": "revisions"},
        "consecutive_beat_streak": {"label": "Beat Score", "fmt": "int", "category": "revisions"},
        "short_interest_ratio": {"label": "Short Interest Ratio", "fmt": "ratio", "category": "revisions"},
        "size_log_mcap": {"label": "Size (-log MCap)", "fmt": "ratio", "category": "size"},
        "asset_growth": {"label": "Asset Growth", "fmt": "pct", "category": "investment"},
    }

    # Which way is good?  Derived from the scorer's own METRIC_DIR rather than
    # written out here, so the page cannot disagree with how the number was
    # actually ranked.  `compute_sector_percentiles()` does `100 - rank` when
    # METRIC_DIR is False, which means the published percentile always reads
    # "better", never "larger" - HON's EV/EBITDA of 6.95 is the 99th percentile
    # and AXON's 98.61 is the 0th.  Nothing on the page said so before
    # 2026-09-11, so a reader had no way to tell those apart from a raw rank.
    #
    # METRIC_DIR describes the direction of the *same* number shown in the Raw
    # Value column, which is what makes this safe for the two transformed
    # metrics: `size_log_mcap` displays -log(mcap) and `max_drawdown_1y`
    # displays a negative fraction, and "higher is better" is literally true of
    # both as displayed.
    for _m, _meta in metric_meta.items():
        _meta["dir"] = "higher" if METRIC_DIR.get(_m, True) else "lower"

    # --- Factor-level correlation (8x8 Spearman from category scores) ---
    factor_score_cols = ["valuation_score", "quality_score", "growth_score",
                         "momentum_score", "risk_score", "revisions_score",
                         "size_score", "investment_score"]
    available_score_cols = [c for c in factor_score_cols if c in df.columns]
    factor_corr_data = None
    if len(available_score_cols) >= 2:
        corr_matrix = df[available_score_cols].corr(method="spearman")
        labels = [c.replace("_score", "").title() for c in available_score_cols]
        matrix_values = []
        for _, corr_row in corr_matrix.iterrows():
            matrix_values.append([round(float(v), 3) if pd.notna(v) else None for v in corr_row])
        factor_corr_data = {"labels": labels, "matrix": matrix_values}

    # --- Weight sensitivity (from parquet artifact) ---
    sens_df = run_data.get("sens_df")
    weight_sens_data = []
    if sens_df is not None and len(sens_df) > 0:
        for cat in sens_df["category"].unique():
            cat_rows = sens_df[sens_df["category"] == cat]
            plus_row = cat_rows[cat_rows["direction"] == "+"]
            minus_row = cat_rows[cat_rows["direction"] == "-"]
            plus_j = float(plus_row["jaccard_similarity"].iloc[0]) if len(plus_row) > 0 else None
            minus_j = float(minus_row["jaccard_similarity"].iloc[0]) if len(minus_row) > 0 else None
            orig_w = float(cat_rows["original_weight"].iloc[0]) if len(cat_rows) > 0 else None
            vals = [v for v in [plus_j, minus_j] if v is not None]
            avg_j = sum(vals) / len(vals) if vals else None
            plus_changed = str(plus_row["changed_tickers"].iloc[0]) if len(plus_row) > 0 and pd.notna(plus_row["changed_tickers"].iloc[0]) else ""
            minus_changed = str(minus_row["changed_tickers"].iloc[0]) if len(minus_row) > 0 and pd.notna(minus_row["changed_tickers"].iloc[0]) else ""
            weight_sens_data.append({
                "category": cat.title(),
                "original_weight": orig_w,
                "plus_jaccard": plus_j,
                "minus_jaccard": minus_j,
                "avg_jaccard": round(avg_j, 3) if avg_j is not None else None,
                "plus_changed": plus_changed,
                "minus_changed": minus_changed,
            })

    # --- Data quality summary ---
    eps_mismatch_count = 0
    avg_metric_coverage = None
    if "_eps_basis_mismatch" in df.columns:
        eps_mismatch_count = int(df["_eps_basis_mismatch"].sum())
    if "_metric_count" in df.columns and "_metric_total" in df.columns:
        valid = df[df["_metric_total"] > 0]
        if len(valid) > 0:
            avg_metric_coverage = round(float((valid["_metric_count"] / valid["_metric_total"]).mean()), 3)
    data_freshness = meta.get("start_time", "")

    data_quality_summary = {
        "eps_mismatch_count": eps_mismatch_count,
        "avg_metric_coverage": avg_metric_coverage,
        "data_freshness": data_freshness,
    }

    # `config_traps` lived here until 2026-09-08. It carried the four trap
    # thresholds solely so the AI chat could put them in its system prompt;
    # nothing rendered them. With the chat gone it had no consumer, and the
    # same thresholds are already published in the Methodology section, which
    # `run_screener.generate_screener_overview()` templates from `config.yaml`.
    # Same reasoning that retired `spx_weights` on 2026-08-26.

    # --- Historical spine: rank/score movement across prior runs ---
    # The dashboard's biggest documented gap is that it has no time dimension
    # (plan/dashboard-north-star.md gap 1). `history` supplies it from
    # the snapshots the data loop already writes. Never let a history problem
    # take down the whole build: a run with no usable history is still a
    # perfectly good snapshot dashboard.
    try:
        history_block = build_history(
            current_df=df,
            current_date=_run_date_for_history(meta),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: history unavailable ({type(exc).__name__}: {exc})")
        history_block = {"available": False, "dates": [], "series": {},
                         "excluded": [], "noise": None,
                         "compare": {"prev": None, "m1": None},
                         "movers": {}, "delta": {}}

    # --- Deterministic per-stock summaries ---
    # Priority 4 / owner directive 2026-08-10: the plain-English "why does this
    # rank here" block that replaces the browser-side AI chat. Built here so it
    # ships in the artifact - identical for every viewer, diffable, and
    # impossible to hallucinate. Runs last because it reads `metric_meta` and
    # the history spine, both of which are assembled above.
    #
    # Never let a summary problem take down the build: a stock with no summary
    # is a slightly plainer drilldown, not a broken page.
    _summary_failures = 0
    for _ticker, _detail in stock_detail.items():
        try:
            _detail["summary"] = build_summary(
                _detail,
                universe_size=kpis.get("universe_size", len(stock_detail)),
                metric_meta=metric_meta,
                metric_weights=weights.get("metric_weights", {}),
                history_delta=(history_block.get("delta") or {}).get(_ticker),
                history_compare=history_block.get("compare"),
                run_date=_run_date_for_history(meta),
            )
        except Exception as exc:  # noqa: BLE001
            _summary_failures += 1
            if _summary_failures == 1:
                print(f"WARNING: summary failed for {_ticker} "
                      f"({type(exc).__name__}: {exc})")
            _detail["summary"] = []
    if _summary_failures:
        print(f"WARNING: {_summary_failures} stock summaries could not be built.")

    # The summary sentence has now read each stock's full peer rows (rank, composite,
    # scores). The browser reads only the ticker: it rebuilds every other peer column
    # from `stock_detail` (`buildPeerRow`). Shipping the rest was ~195 KB gzipped of
    # pure duplication - more than the per-metric inputs added on 2026-10-07 cost.
    for _detail in stock_detail.values():
        _detail["peers"] = [{"ticker": p["ticker"]} for p in (_detail.get("peers") or [])]

    # A payload that cannot reproduce its own scores is not published. Recomputes
    # every category score and composite from the published numbers alone, with an
    # implementation that shares no code with the engine (calc_trace.py).
    _reconcile_scores(weights, stock_detail, run_data.get("cfg") or {})

    dashboard_json = {
        "kpis": kpis,
        "cadence": _cadence_block(run_data.get("cfg") or {}),
        "history": history_block,
        "table_data": table_data,
        "stock_detail": stock_detail,
        "weights": weights,
        "lineage": published_lineage(),
        "lineage_check": _lineage_check(stock_detail),
        "sector_stats": _sector_stats(df, raw_metrics),
        "sector_min_peers": SECTOR_MIN_PEERS,
        "metric_meta": metric_meta,
        "sectors": sectors,
        "sector_composition": sector_composition,
        "histogram": {"labels": hist_labels, "values": [int(v) for v in hist_values]},
        "vt_by_sector": vt_by_sector,
        "gt_by_sector": gt_by_sector,
        "sector_distributions": sector_distributions,
        "factor_correlation": factor_corr_data,
        "weight_sensitivity": weight_sens_data,
        "data_quality": data_quality_summary,
    }

    return json.dumps(dashboard_json, default=str)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(data_json: str = "", methodology_html: str = "", data_timestamp: str = "", data_version: str = "") -> str:
    """Build the complete dashboard HTML string.

    Data is loaded from the companion `dashboard_data.js` file (written by
    `generate_dashboard`) rather than being inlined in the HTML.  This keeps
    the HTML under ~200 KB while the data file carries the bulk.

    The `data_json` parameter is accepted for backward compatibility but is
    no longer embedded in the HTML.
    """
    # Escape braces in methodology_html so f-string doesn't choke
    methodology_escaped = methodology_html.replace("{", "{{").replace("}", "}}")
    version = data_version or "latest"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Factor Screener Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="./dashboard_data.js?v={version}"></script>
    <style>
{_css()}
    </style>
    <style>
{_css_ux()}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header" id="top-bar">
            <div class="header-left">
                <h1>Multi-Factor Screener</h1>
                <span class="run-info" id="run-info"></span>
            </div>
            <nav class="header-nav" aria-label="Sections">
                <a href="#sec-top5" onclick="goToSection('sec-top5');return false">Top 5</a>
                <a href="#sec-universe" onclick="goToSection('sec-universe');return false">Rankings</a>
                <a href="#sec-holdings" onclick="goToSection('sec-holdings');return false">Holdings</a>
                <a href="#sec-changed" onclick="goToSection('sec-changed');return false" id="nav-changed">What changed</a>
                <a href="#sec-analytics" onclick="goToSection('sec-analytics');return false">Analytics</a>
                <a href="#sec-defensibility" onclick="goToSection('sec-defensibility');return false">Diagnostics</a>
            </nav>
            <div class="header-right">
                <button class="cmdk-btn" type="button" onclick="openPalette()" aria-label="Search stocks and sections" aria-keyshortcuts="Control+K Meta+K">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>
                    <span class="cmdk-btn-text">Search stocks</span><kbd class="cmdk-kbd" id="cmdk-kbd">Ctrl K</kbd>
                </button>
                <button class="methodology-btn" onclick="openMethodology()">Methodology</button>
            </div>
        </header>

        <!-- First-visit guide: three sentences on how to read the page. Dismissed once,
             per browser; reopened from the search palette. -->
        <section class="guide" id="guide" hidden aria-label="How to read this screener">
            <ol class="guide-steps">
                <li><span class="guide-n">1</span><div><strong>Eight scores per stock.</strong> Valuation, Quality, Growth, Momentum, Risk, Revisions, Size and Investment, each 0&ndash;100 against the stock's own sector. Around 50 is typical for the sector.</div></li>
                <li><span class="guide-n">2</span><div><strong>One composite.</strong> The eight scores, weighted and added up. The rank is just the composite in order &mdash; a description of the numbers, not a verdict on the company.</div></li>
                <li><span class="guide-n">3</span><div><strong>Every number is checkable.</strong> Open any stock to see the inputs, formulas and peers behind each score, and the arithmetic that adds them up.</div></li>
            </ol>
            <div class="guide-foot">
                <span>A screening tool for research and teaching. Not investment advice.</span>
                <button type="button" class="guide-close" onclick="dismissGuide()">Got it</button>
            </div>
        </section>

        <!-- KPI Row -->
        <section class="kpi-row" id="kpi-row"></section>

        <!-- Top 5 Stocks -->
        <section class="section collapsible-section" id="sec-top5">
            <div class="section-header" onclick="toggleSection('sec-top5')">
                <h2 class="section-title" style="margin:0">Top 5 Stocks</h2>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <div class="top5-row" id="top5-row"></div>
            </div>
        </section>

        <!-- My Holdings -->
        <!--
            Priority 5 / north-star gap 2: the sell-side workflow. The shape of
            this surface is set by research/2026-09-14-sell-discipline-and-hold-bands.md
            and three of its properties are load-bearing rather than stylistic:
            it lists every name every time, it is ordered by rank and never by
            size of move, and it never asks for a cost basis. See the footnote
            copy in renderHoldings() for the sources.
        -->
        <section class="section collapsible-section collapsed" id="sec-holdings">
            <div class="section-header" onclick="toggleSection('sec-holdings')">
                <h2 class="section-title" style="margin:0">My Holdings <span class="holdings-count" id="holdings-count"></span></h2>
                <span class="sec-meta" id="meta-holdings"></span>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <p class="section-desc">Add the names you own or are following, and the screener will put what it knows about <em>all</em> of them in front of you each run &mdash; what moved, which category moved it, and what the score does and does not rest on. Stored in this browser only; nothing is sent anywhere, and no purchase price is asked for.</p>
                <div class="holdings-add-bar">
                    <div class="holdings-search-wrap">
                        <input type="text" id="holdings-search-input"
                               class="peer-search-input"
                               placeholder="Add by ticker or company name..."
                               autocomplete="off" />
                        <div id="holdings-search-results" class="peer-search-results"></div>
                    </div>
                    <button class="holdings-clear-btn" id="holdings-clear-btn" onclick="clearHoldings()" style="display:none">Clear list</button>
                </div>
                <div class="holdings-fit" id="holdings-fit"></div>
                <div id="holdings-body"></div>
                <details class="why-panel"><summary>Why this panel works this way</summary>
                    <div class="holdings-footnote" id="holdings-footnote"></div>
                </details>
            </div>
        </section>

        <!-- What Changed -->
        <section class="section collapsible-section collapsed" id="sec-changed" style="display:none">
            <div class="section-header" onclick="toggleSection('sec-changed')">
                <h2 class="section-title" style="margin:0">What Changed</h2>
                <span class="sec-meta" id="meta-changed"></span>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <div class="changed-controls">
                    <div class="seg-control" id="changed-range"></div>
                    <span class="changed-caption" id="changed-caption"></span>
                </div>
                <div class="movers-grid">
                    <div class="movers-col">
                        <h3 class="chart-title">Moved up the rankings</h3>
                        <div id="movers-up"></div>
                    </div>
                    <div class="movers-col">
                        <h3 class="chart-title">Moved down the rankings</h3>
                        <div id="movers-down"></div>
                    </div>
                </div>
                <details class="why-panel"><summary>How to read this</summary>
                    <div class="changed-footnote" id="changed-footnote"></div>
                </details>
            </div>
        </section>

        <!-- Factor Analytics Section -->
        <section class="section collapsible-section collapsed" id="sec-analytics">
            <div class="section-header" onclick="toggleSection('sec-analytics')">
                <h2 class="section-title" style="margin:0">Factor Analytics</h2>
                <span class="sec-meta">Sector &times; factor scores &middot; trap rates by sector</span>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <div class="chart-row">
                    <div class="chart-container" style="flex:1.4">
                        <div class="chart-title-row">
                            <h3 class="chart-title" style="margin-bottom:0">Where each sector scores</h3>
                            <div class="toggle-btns">
                                <button class="toggle-btn active" id="btn-median" onclick="setSectorStat('median')">Median</button>
                                <button class="toggle-btn" id="btn-mean" onclick="setSectorStat('mean')">Average</button>
                            </div>
                        </div>
                        <p class="chart-note">The score of the typical stock in each sector, 0&ndash;100. Shading is relative within each column, so it shows where a sector stands out on that factor rather than an absolute level. The Composite column is why most sectors look alike overall: the differences are in the factors. Select a sector to list its stocks.</p>
                        <div id="sector-matrix" class="sector-matrix" role="region" aria-label="Sector by factor scores"></div>
                    </div>
                    <div class="chart-container" style="flex:0.5">
                        <div class="chart-title-row">
                            <h3 class="chart-title" style="margin-bottom:0">Trap Rate by Sector</h3>
                            <div class="toggle-btns">
                                <button class="toggle-btn active" id="btn-trap-vt" onclick="setTrapType('vt')">Value</button>
                                <button class="toggle-btn" id="btn-trap-gt" onclick="setTrapType('gt')">Growth</button>
                            </div>
                        </div>
                        <p class="chart-note" id="trap-note"></p>
                        <div id="trap-bars" class="trap-bars" role="list" aria-label="Share of each sector carrying a trap flag"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Defensibility & Diagnostics -->
        <section class="section collapsible-section defensibility-section collapsed" id="sec-defensibility">
            <div class="section-header" onclick="toggleSection('sec-defensibility')">
                <div class="defensibility-header-left">
                    <h2 class="section-title" style="margin:0">Defensibility &amp; Diagnostics</h2>
                    <div class="defensibility-summary" id="defensibility-summary"></div>
                </div>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <p class="section-desc">These diagnostics help you evaluate how robust and trustworthy the screener's output is. They answer: <em>"Would the same stocks be picked if I tweaked the weights slightly?"</em> and <em>"Are any of the 8 factors just measuring the same thing?"</em></p>
                <div class="defensibility-kpis" id="defensibility-kpis"></div>
                <div class="defensibility-row">
                    <div class="chart-container" style="flex:1">
                        <h3 class="chart-title">How Stable Is the Ranking?</h3>
                        <p class="chart-desc">If we nudge each factor's weight &plusmn;5%, how much does the top-20 list change? <strong>Higher Jaccard = more stable.</strong> A score of 0.85 or more means almost no change; below 0.70 means the ranking is sensitive to that weight.</p>
                        <div id="sensitivity-table"></div>
                    </div>
                    <div class="chart-container" style="flex:1">
                        <h3 class="chart-title">Are the Factors Independent?</h3>
                        <p class="chart-desc">Spearman correlations between the 8 factor scores. <strong>Low values = independent signals.</strong> Shading deepens with the size of the correlation, and pairs above 0.7 are outlined: they are measuring similar things, which reduces the effective number of independent dimensions.</p>
                        <div id="correlation-heatmap"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Full Universe Table -->
        <section class="section collapsible-section" id="sec-universe">
            <div class="section-header" onclick="toggleSection('sec-universe')">
                <h2 class="section-title" style="margin:0">Full Universe Rankings</h2>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <div class="filters-bar" role="search">
                    <div class="filter-search">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>
                        <input type="text" id="filter-search" placeholder="Search ticker or company" aria-label="Search ticker or company" autocomplete="off" spellcheck="false">
                        <kbd aria-hidden="true">/</kbd>
                    </div>
                    <div class="filter-group">
                        <label for="filter-sector">Sector</label>
                        <select id="filter-sector"><option value="all">All sectors</option></select>
                    </div>
                    <div class="filter-group">
                        <label for="filter-vt">Trap flags</label>
                        <select id="filter-vt">
                            <option value="all">All</option>
                            <option value="clean">No flag</option>
                            <option value="vt">Value trap</option>
                            <option value="gt">Growth trap</option>
                            <option value="any">Any flag</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label for="filter-comp-min">Composite min</label>
                        <input type="number" id="filter-comp-min" min="0" max="100" value="0" step="5" inputmode="numeric">
                    </div>
                    <div class="filter-group filter-sort-m">
                        <label for="sort-by">Sort by</label>
                        <select id="sort-by" aria-label="Sort the rankings by">
                            <option value="Rank">Rank</option>
                            <option value="Composite">Composite</option>
                            <option value="valuation_score">Valuation</option>
                            <option value="quality_score">Quality</option>
                            <option value="growth_score">Growth</option>
                            <option value="momentum_score">Momentum</option>
                            <option value="risk_score">Risk</option>
                            <option value="revisions_score">Revisions</option>
                            <option value="size_score">Size</option>
                            <option value="investment_score">Investment</option>
                        </select>
                    </div>
                    <button type="button" class="filter-clear" id="filter-clear" onclick="clearFilters()" hidden>Clear filters</button>
                    <span class="result-count" id="result-count" aria-live="polite"></span>
                </div>
                <div class="table-section">
                    <table class="data-table" id="universe-table" aria-label="Full universe rankings" aria-rowcount="-1">
                        <thead><tr>
                            <th data-sort="Rank">Rank</th>
                            <th data-sort="_rank_delta" id="th-rank-delta" title="Change in rank since the previous comparable run. Positive means the stock moved up the table.">&Delta;</th>
                            <th data-sort="Ticker">Ticker</th>
                            <th data-sort="Company">Company</th>
                            <th data-sort="Sector">Sector</th>
                            <th data-sort="Composite" title="The weighted blend of all eight category scores below. This is the ranking key. Higher is better; 0-100.">Composite</th>
                            <th data-sort="valuation_score" title="Valuation - is it cheap? FCF yield (45%), EV/EBITDA (25%), earnings yield (20%), EV/Sales (10%). Banks are scored instead on P/B (60%) and earnings yield (40%), because enterprise value and free cash flow do not mean the same thing for a bank. Higher score = cheaper than its sector; 0-100.">Val</th>
                            <th data-sort="quality_score" title="Quality - is the business sound? ROIC (27%), gross profit/assets (20%), net debt/EBITDA (18%), Piotroski F-Score (15%), operating leverage (8%), Beneish M-Score (7%), accruals (5%). Banks use ROE (35%), ROA (25%), equity ratio (15%), Piotroski (15%) and accruals (10%). Higher score = better quality; 0-100.">Qual</th>
                            <th data-sort="growth_score" title="Growth - is it expanding? Forward EPS growth (45%), revenue growth (25%), 3-year revenue CAGR (15%), sustainable growth (15%). PEG carries no weight: P/E divided by growth double-counts valuation. Higher score = faster growth; 0-100.">Grow</th>
                            <th data-sort="momentum_score" title="Momentum - has the price been rising? 12-month return excluding the last month (40%), 6-month return (35%), Jensen's alpha (25%). The most recent month is skipped deliberately: short-horizon returns tend to reverse. Higher score = stronger trend; 0-100.">Mom</th>
                            <th data-sort="risk_score" title="Risk - how bumpy is the ride? Volatility (42.9%), beta (28.6%), max 1-year drawdown (28.6%). Higher score = calmer and less drawdown-prone, so a high Risk score means LOW risk; 0-100.">Risk</th>
                            <th data-sort="revisions_score" title="Revisions - are analysts turning more positive? 90-day FY1 EPS revision (35%), earnings acceleration (20%), earnings surprise (15%), price-target upside (10%), beat streak (10%), short interest (10%). Higher score = improving expectations; 0-100.">Rev</th>
                            <th data-sort="size_score" title="Size - the small-cap premium. Scored from -log(market cap), so within the S&amp;P 500 a higher score means a smaller company; 0-100.">Size</th>
                            <th data-sort="investment_score" title="Investment - is the balance sheet growing conservatively? Asset growth. Higher score = slower asset growth, which historically predicts better returns; 0-100.">Inv</th>
                            <th data-sort="Value_Trap_Flag" title="Value-trap and growth-trap flags (see Methodology). A blank cell means the stock carries no flag.">Trap flags</th>
                        </tr></thead>
                        <tbody id="universe-tbody"></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Stock Detail Modal -->
        <div class="modal-overlay" id="stock-modal" style="display:none" onclick="if(event.target===this)closeModal()">
            <div class="modal-content" role="dialog" aria-modal="true" aria-labelledby="modal-ticker">
                <div class="modal-header">
                    <div>
                        <h2 class="modal-ticker" id="modal-ticker"></h2>
                        <span class="modal-company" id="modal-company"></span>
                        <span class="modal-sector" id="modal-sector"></span>
                    </div>
                    <div class="modal-headline" id="modal-headline"></div>
                    <div class="modal-tools" role="toolbar" aria-label="Stock tools">
                        <button type="button" class="mt-btn mt-step" id="mt-prev" onclick="stepStock(-1)" aria-label="Previous stock" title="Previous stock in the list (K)"><svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="15 18 9 12 15 6"/></svg></button>
                        <span class="mt-pos" id="mt-pos" aria-live="polite"></span>
                        <button type="button" class="mt-btn mt-step" id="mt-next" onclick="stepStock(1)" aria-label="Next stock" title="Next stock in the list (J)"><svg viewBox="0 0 24 24" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg></button>
                        <button type="button" class="mt-btn mt-text" id="mt-hold" onclick="toggleHoldingCurrent()" aria-pressed="false" title="Keep this stock on My Holdings, saved in this browser (H)"><span class="mt-lg">Add to&nbsp;</span>Holdings</button>
                        <button type="button" class="mt-btn mt-text" id="mt-compare" onclick="toggleCompareCurrent()" aria-pressed="false" title="Add to the side-by-side comparison (C)">Compare</button>
                        <button type="button" class="mt-btn mt-text" id="mt-link" onclick="copyStockLink()" title="Copy a link that opens this stock"><span class="mt-lg">Copy&nbsp;</span>Link</button>
                    </div>
                    <button class="modal-close" onclick="closeModal()" aria-label="Close">&times;</button>
                </div>
                <nav class="modal-nav" aria-label="In this stock">
                    <a href="#modal-summary" onclick="goToModal('modal-summary');return false">Why it ranks here</a>
                    <a href="#modal-score-row" onclick="goToModal('modal-score-row');return false">Scores</a>
                    <a href="#section-contribution" onclick="goToModal('section-contribution');return false">How it adds up</a>
                    <a href="#section-categories" onclick="goToModal('section-categories');return false">The workings</a>
                    <a href="#section-history" onclick="goToModal('section-history');return false" id="mnav-history">History</a>
                    <a href="#section-price-targets" onclick="goToModal('section-price-targets');return false">Price targets</a>
                    <a href="#section-peers" onclick="goToModal('section-peers');return false">Peers</a>
                    <a href="#section-provenance" onclick="goToModal('section-provenance');return false">Data</a>
                </nav>
                <div class="modal-body">
                    <!-- Why it ranks here: deterministic summary, built at
                         run time from this run's own numbers. Replaced the
                         browser-side AI chat (owner directive 2026-08-10). -->
                    <div class="summary-block" id="modal-summary" style="display:none">
                        <div class="summary-head">
                            <span class="summary-title">Why it ranks here</span>
                        </div>
                        <div class="summary-body" id="modal-summary-body"></div>
                        <div class="summary-source">Assembled from this run's numbers by a fixed template &mdash; every figure appears somewhere below and is identical for every reader. It explains <em>where the stock ranks and why</em>. It is not investment advice and never says whether to buy, sell or hold.</div>
                    </div>

                    <!-- What the company does (provider description, display only) -->
                    <div class="about-block" id="modal-about" style="display:none">
                        <div class="about-head">
                            <span class="about-title">About</span>
                            <span class="about-industry" id="modal-industry"></span>
                        </div>
                        <p class="about-text clamped" id="modal-about-text"></p>
                        <button class="about-toggle" id="modal-about-toggle"
                                onclick="toggleAbout()">Show more</button>
                        <div class="about-source">Business description supplied by Yahoo Finance. Descriptive only &mdash; it is not scored and does not affect the ranking.</div>
                    </div>

                    <!-- Score summary row -->
                    <div class="modal-score-row" id="modal-score-row"></div>

                    <!-- Contribution breakdown -->
                    <div class="collapsible" id="section-contribution">
                        <div class="collapsible-header" onclick="toggleSection('section-contribution')">
                            <span>How it adds up</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body">
                            <div class="modal-chart-section">
                                <p class="modal-chart-desc">Each category score (0&ndash;100) times the weight it was <strong>actually multiplied by</strong> for this stock gives its points, and the points add up to the composite. The weights can differ from the defaults in Methodology; any gap is explained underneath. Select a row to see how that score is built.</p>
                                <div id="contrib-visual"></div>
                                <div class="contrib-total-row" id="contrib-total"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Category detail sections -->
                    <div class="collapsible" id="section-categories">
                        <div class="collapsible-header" onclick="toggleSection('section-categories')">
                            <span>The workings</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-categories"></div>
                    </div>

                    <!-- Rank history -->
                    <div class="collapsible" id="section-history" style="display:none">
                        <div class="collapsible-header" onclick="toggleSection('section-history')">
                            <span>Rank History</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-history"></div>
                    </div>

                    <!-- Analyst Price Targets -->
                    <div class="collapsible" id="section-price-targets">
                        <div class="collapsible-header" onclick="toggleSection('section-price-targets')">
                            <span>Analyst Price Targets</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-price-targets"></div>
                    </div>

                    <!-- Company Snapshot -->
                    <div class="collapsible collapsed" id="section-snapshot">
                        <div class="collapsible-header" onclick="toggleSection('section-snapshot')">
                            <span>Company Snapshot</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-snapshot"></div>
                    </div>

                    <!-- Sector Peers -->
                    <div class="collapsible collapsed" id="section-peers">
                        <div class="collapsible-header" onclick="toggleSection('section-peers')">
                            <span>Sector Peers</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-peers"></div>
                    </div>

                    <!-- Data Provenance -->
                    <div class="collapsible collapsed" id="section-provenance">
                        <div class="collapsible-header" onclick="toggleSection('section-provenance')">
                            <span>Data Provenance</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-provenance"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Methodology Modal -->
        <div class="modal-overlay methodology-modal" id="methodology-modal" style="display:none" onclick="if(event.target===this)closeMethodology()">
            <div class="modal-content methodology-content">
                <div class="modal-header">
                    <div>
                        <h2 class="modal-ticker">Methodology</h2>
                        <span class="modal-company">How the screener works, what it measures, and why</span>
                    </div>
                    <button class="modal-close" onclick="closeMethodology()">&times;</button>
                </div>
                <div class="modal-body methodology-body">
                    {methodology_escaped}
                </div>
            </div>
        </div>

        <!-- Search palette (Ctrl/Cmd+K): any stock, any section, from anywhere -->
        <div class="pal-overlay" id="palette" hidden onclick="if(event.target===this)closePalette()">
            <div class="pal" role="dialog" aria-modal="true" aria-label="Search stocks and sections">
                <div class="pal-input-row">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>
                    <input id="pal-input" type="text" role="combobox" aria-expanded="true" aria-controls="pal-list" aria-autocomplete="list" autocomplete="off" spellcheck="false" placeholder="Search a ticker, company or sector">
                    <kbd>Esc</kbd>
                </div>
                <div class="pal-list" id="pal-list" role="listbox" aria-label="Results"></div>
                <div class="pal-foot" aria-hidden="true"><span><kbd>&uarr;</kbd><kbd>&darr;</kbd> move</span><span><kbd>Enter</kbd> open</span><span><kbd>?</kbd> all shortcuts</span></div>
            </div>
        </div>

        <!-- Side-by-side comparison -->
        <div class="cmp-tray" id="cmp-tray" hidden role="region" aria-label="Comparison tray">
            <span class="cmp-tray-label">Compare</span>
            <div class="cmp-chips" id="cmp-chips"></div>
            <button type="button" class="cmp-open" id="cmp-open" onclick="openCompare()">Side by side</button>
            <button type="button" class="cmp-clear" onclick="clearCompare()" aria-label="Clear the comparison">Clear</button>
        </div>
        <div class="modal-overlay cmp-overlay" id="compare-modal" style="display:none" onclick="if(event.target===this)closeCompare()">
            <div class="modal-content cmp-content" role="dialog" aria-modal="true" aria-labelledby="cmp-title">
                <div class="modal-header">
                    <div>
                        <h2 class="modal-ticker" id="cmp-title">Side by side</h2>
                        <span class="modal-company">The same numbers as each stock's drilldown, lined up. The highest score in each row is marked.</span>
                    </div>
                    <button class="modal-close" onclick="closeCompare()" aria-label="Close">&times;</button>
                </div>
                <div class="modal-body cmp-body" id="cmp-body"></div>
            </div>
        </div>

        <!-- Keyboard shortcuts -->
        <div class="pal-overlay" id="shortcuts" hidden onclick="if(event.target===this)closeShortcuts()">
            <div class="pal kb-sheet" role="dialog" aria-modal="true" aria-labelledby="kb-title">
                <h2 id="kb-title">Keyboard shortcuts</h2>
                <dl class="kb-list">
                    <dt><kbd id="kb-mod">Ctrl</kbd><kbd>K</kbd></dt><dd>Search any stock or section</dd>
                    <dt><kbd>/</kbd></dt><dd>Filter the rankings table</dd>
                    <dt><kbd>J</kbd> <kbd>K</kbd></dt><dd>Next / previous stock, in the table's current order</dd>
                    <dt><kbd>C</kbd></dt><dd>Add the open stock to the comparison</dd>
                    <dt><kbd>H</kbd></dt><dd>Add the open stock to My Holdings</dd>
                    <dt><kbd>Esc</kbd></dt><dd>Close whatever is open</dd>
                    <dt><kbd>?</kbd></dt><dd>This list</dd>
                </dl>
                <p class="kb-note">Links to a stock open its drilldown directly: the address bar updates as you browse, so you can share or bookmark it.</p>
            </div>
        </div>

        <div class="toast" id="toast" role="status" aria-live="polite"></div>

        <footer class="dashboard-footer">
            Multi-Factor Screener Dashboard &bull; Data as of <span id="gen-time">{data_timestamp}</span><br>
            <span class="footer-disclaimer">Screening tool, not investment advice &bull; data from Yahoo Finance (yfinance), not institutional-grade &bull; no covariance risk model &bull; see <a href="#" onclick="openMethodology();return false;">Methodology</a> for limitations</span>
        </footer>
    </div>

    <script>
    // =====================================================================
    // DATA — loaded from companion dashboard_data.js (see generate_dashboard.py)
    // =====================================================================
    const D = window.SCREENER_DATA || {{}};

    function setDefault(obj, key, value) {{
        if (obj[key] === undefined || obj[key] === null) obj[key] = value;
    }}

    // Defensive defaults: keep the page usable even if a stale payload is cached.
    setDefault(D, 'kpis', {{ universe_size: 0, value_traps: 0, growth_traps: 0, run_timestamp: null }});
    setDefault(D, 'table_data', []);
    setDefault(D, 'stock_detail', {{}});
    setDefault(D, 'vt_by_sector', {{}});
    setDefault(D, 'gt_by_sector', {{}});
    setDefault(D, 'sector_distributions', {{ composite_score: {{}} }});
    setDefault(D, 'weight_sensitivity', []);
    setDefault(D, 'factor_correlation', {{ labels: [], matrix: [] }});
    setDefault(D, 'data_quality', {{}});

    // =====================================================================
    // COLOUR PALETTE
    // =====================================================================
    // Design tokens, JS side - keep in sync with :root (design-system doc).
    const ACCENT = '#3987e5';
    const COLORS = [ACCENT];
    const POS = '#0ca30c', NEG = '#e66767';

    // Chart.js is gone (2026-10-07 pass 2): the last canvas chart, trap rates, became HTML
    // bars, and the library was a render-blocking ~200 KB script in <head> for no chart.

    // =====================================================================
    // UTILITIES
    // =====================================================================
    function escapeHtml(str) {{
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }}

    function debounce(fn, ms) {{
        let timer;
        return function() {{
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, arguments), ms);
        }};
    }}

    function fmt(v, type) {{
        if (v === null || v === undefined) return '—';
        if (type === 'pct') return v.toFixed(1);
        if (type === 'int') return Math.round(v).toLocaleString();
        if (type === 'score') return v.toFixed(1);
        return v.toString();
    }}

    // =====================================================================
    // KPI CARDS
    // =====================================================================
    function renderKPIs() {{
        const k = D.kpis;
        const html = [
            kpiCard('Universe', k.universe_size, 'stocks scored'),
            kpiCard('Value-trap flags', k.value_traps, `${{(k.value_traps/k.universe_size*100).toFixed(0)}}% of the universe`),
            kpiCard('Growth-trap flags', k.growth_traps || 0, `${{((k.growth_traps||0)/k.universe_size*100).toFixed(0)}}% of the universe`),
            kpiCard('Metric coverage', (D.data_quality && D.data_quality.avg_metric_coverage != null)
                ? (D.data_quality.avg_metric_coverage * 100).toFixed(0) + '%' : '&mdash;', 'of applicable metrics, on average'),
        ].join('');
        document.getElementById('kpi-row').innerHTML = html;
        const ts = k.run_timestamp ? new Date(k.run_timestamp) : null;
        const fmtDate = ts ? ts.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }}) : '';
        const fmtTime = ts ? ts.toLocaleTimeString('en-US', {{ hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }}) : '';
        document.getElementById('run-info').textContent = ts ? `Last updated ${{fmtDate}} at ${{fmtTime}}` : '';
        // Only set gen-time via JS if it wasn't already embedded as a static value
        const genEl = document.getElementById('gen-time');
        if (genEl && !genEl.textContent.trim()) {{
            genEl.textContent = new Date().toLocaleString();
        }}
    }}

    // Jump to a section, opening it first if it is collapsed.
    function goToSection(id) {{
        const sec = document.getElementById(id);
        if (!sec) return;
        if (sec.style.display === 'none') return;
        if (sec.classList.contains('collapsed')) toggleSection(id);
        sec.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}

    function kpiCard(label, value, sub) {{
        return `<div class="kpi-card">
            <div class="kpi-label">${{label}}</div>
            <div class="kpi-value">${{value}}</div>
            <div class="kpi-sub">${{sub}}</div>
        </div>`;
    }}

    // =====================================================================
    // TOP 5 STOCKS
    // =====================================================================
    function renderTop5() {{
        // Top 5 of the ranking, excluding trap-flagged names. Previously this
        // read the model-portfolio holdings; that surface was removed and the
        // sector cap (8 of 25) cannot bind on five rows, so the list is
        // unchanged - see METHODOLOGY_CHANGELOG 2026-08-26 (evening).
        const top5 = D.table_data
            .filter(function(r) {{ return !r.Value_Trap_Flag && !r.Growth_Trap_Flag; }})
            .sort(function(a, b) {{ return a.Rank - b.Rank; }})
            .slice(0, 5)
            .map(function(r) {{
                return {{
                    rank: r.Rank, ticker: r.Ticker, company: r.Company, sector: r.Sector,
                    composite: r.Composite, valuation: r.valuation_score,
                    quality: r.quality_score, growth: r.growth_score,
                    momentum: r.momentum_score, risk: r.risk_score,
                    revisions: r.revisions_score, size: r.size_score,
                    investment: r.investment_score,
                    vt: r.Value_Trap_Flag, gt: r.Growth_Trap_Flag
                }};
            }});
        const catKeys = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        const catLabels = {{ valuation:'Val', quality:'Qual', growth:'Grow', momentum:'Mom', risk:'Risk', revisions:'Rev', size:'Size', investment:'Inv' }};
        // One hue for all categories: the bars encode magnitude, the labels
        // carry identity. A hue per category was decoration - see
        // plan/dashboard-design-system.md.
        const catColors = {{
            valuation: ACCENT, quality: ACCENT, growth: ACCENT,
            momentum: ACCENT, risk: ACCENT, revisions: ACCENT,
            size: ACCENT, investment: ACCENT
        }};

        document.getElementById('top5-row').innerHTML = top5.map((h, i) => {{
            const cells = catKeys.map(c => {{
                const val = h[c] || 0;
                return `<div class="top5-factor">
                    <span class="top5-factor-label">${{catLabels[c]}}</span>
                    <span class="top5-factor-val">${{fmt(val,'score')}}</span>
                    <div class="top5-factor-bar"><div class="top5-factor-fill" style="width:${{val}}%"></div></div>
                </div>`;
            }}).join('');

            return `<div class="top5-card" role="button" tabindex="0" onclick="openStockDetail('${{escapeHtml(h.ticker)}}')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();openStockDetail('${{escapeHtml(h.ticker)}}')}}">
                <div class="top5-top"><span class="top5-rank-bar">#${{h.rank}}</span><span class="top5-sector">${{escapeHtml(h.sector)}}</span></div>
                <div class="top5-id">
                    <div class="top5-who">
                        <span class="top5-ticker">${{escapeHtml(h.ticker)}}</span>
                        <div class="top5-company">${{escapeHtml(h.company)}}</div>
                    </div>
                    <div class="top5-score"><span class="top5-composite">${{fmt(h.composite,'score')}}</span><span class="top5-composite-label">Composite</span></div>
                </div>
                <div class="top5-factors">${{cells}}</div>
            </div>`;
        }}).join('');
    }}

    // =====================================================================
    // COMPOSITE HISTOGRAM
    // =====================================================================
    // VALUE TRAP BAR CHART
    // =====================================================================
    let currentTrapType = 'vt';

    // Trap rates as HTML bars rather than a canvas chart: the canvas clipped long
    // sector names ("onsumer Discretionary") at the widths this panel actually gets,
    // could not be read by a screen reader, and needed a hover to show the counts
    // the rate is made of. Each row now prints the rate and the count together.
    function renderTrapChart() {{
        const host = document.getElementById('trap-bars');
        if (!host) return;
        const src = currentTrapType === 'gt' ? D.gt_by_sector : D.vt_by_sector;
        const sectors = Object.keys(src || {{}}).sort((a, b) => src[b].rate - src[a].rate || a.localeCompare(b));
        const max = Math.max(1, ...sectors.map(s => src[s].rate));
        const flagged = sectors.reduce((n, s) => n + (src[s].flagged || 0), 0);
        const total = sectors.reduce((n, s) => n + (src[s].total || 0), 0);
        const kind = currentTrapType === 'gt' ? 'growth-trap' : 'value-trap';
        const note = document.getElementById('trap-note');
        if (note) note.textContent = 'Share of each sector carrying a ' + kind + ' flag. ' + flagged + ' of ' + total +
            ' stocks overall. A flag is a caveat shown beside the score; see Methodology for the rules. Select a sector to list the flagged names.';
        host.innerHTML = sectors.map(s => {{
            const d = src[s];
            return '<div class="trap-row" role="listitem" data-sector="' + escapeHtml(s) + '" tabindex="0" title="Show these stocks in the rankings">' +
                '<span class="trap-name" title="' + escapeHtml(s) + '">' + escapeHtml(s) + '</span>' +
                '<span class="trap-track"><i style="width:' + (d.rate / max * 100).toFixed(1) + '%"></i></span>' +
                '<span class="trap-val"><strong>' + Number(d.rate).toFixed(1) + '%</strong><span>' + d.flagged + ' of ' + d.total + '</span></span></div>';
        }}).join('') || '<div class="mover-none">No trap flags in this run.</div>';
    }}

    function setTrapType(type) {{
        currentTrapType = type;
        document.getElementById('btn-trap-vt').classList.toggle('active', type === 'vt');
        document.getElementById('btn-trap-gt').classList.toggle('active', type === 'gt');
        renderTrapChart();
    }}

    // =====================================================================
    // FACTOR SCORES BY SECTOR (single chart with factor + stat toggles)
    // =====================================================================
    let sectorDistChart = null;
    let currentSectorStat = 'median';

    const FACTOR_COLOR_MAP = {{
        'Composite': ACCENT,
        'valuation_score': ACCENT, 'quality_score': ACCENT,
        'growth_score': ACCENT, 'momentum_score': ACCENT,
        'risk_score': ACCENT, 'revisions_score': ACCENT,
        'size_score': ACCENT, 'investment_score': ACCENT
    }};

    function updateSectorDist() {{
        const host = document.getElementById('sector-matrix');
        if (!host) return;
        const stat = currentSectorStat;
        const cols = [['Composite', 'Composite'], ['valuation_score', 'Val'], ['quality_score', 'Qual'],
                      ['growth_score', 'Grow'], ['momentum_score', 'Mom'], ['risk_score', 'Risk'],
                      ['revisions_score', 'Rev'], ['size_score', 'Size'], ['investment_score', 'Inv']];
        const dist = D.sector_distributions || {{}};
        const comp = dist['Composite'] || {{}};
        const sectors = Object.keys(comp).sort((a, b) => (comp[b][stat] || 0) - (comp[a][stat] || 0));
        // Shade relative to each column's own spread - otherwise a column where every
        // sector sits at 48-52 looks identical to one that varies from 20 to 80.
        const span = {{}};
        cols.forEach(c => {{
            const vals = sectors.map(s => ((dist[c[0]] || {{}})[s] || {{}})[stat]).filter(v => v !== undefined && v !== null);
            span[c[0]] = vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1];
        }});
        let h = '<table class="sm-table"><thead><tr><th>Sector</th>' + cols.map(c => '<th class="num">' + c[1] + '</th>').join('') + '<th class="num">Stocks</th></tr></thead><tbody>';
        sectors.forEach(s => {{
            h += '<tr data-sector="' + escapeHtml(s) + '" tabindex="0" title="Show ' + escapeHtml(s) + ' in the rankings"><td class="sm-sector">' + escapeHtml(s) + '</td>';
            cols.forEach(c => {{
                const d = (dist[c[0]] || {{}})[s];
                const v = d ? d[stat] : null;
                if (v === null || v === undefined) {{ h += '<td class="num sm sm-na">&mdash;</td>'; return; }}
                const [lo, hi] = span[c[0]];
                const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
                h += '<td class="num sm" style="--t:' + t.toFixed(3) + '">' + v.toFixed(1) + '</td>';
            }});
            h += '<td class="num sm-n">' + (comp[s] ? comp[s].count : '') + '</td></tr>';
        }});
        host.innerHTML = h + '</tbody></table>';
    }}

    function setSectorStat(stat) {{
        currentSectorStat = stat;
        document.getElementById('btn-median').classList.toggle('active', stat === 'median');
        document.getElementById('btn-mean').classList.toggle('active', stat === 'mean');
        updateSectorDist();
    }}

    // =====================================================================
    // FULL UNIVERSE TABLE with sort and filter
    // =====================================================================
    let tableState = {{
        data: D.table_data,
        filtered: D.table_data,
        sortCol: 'Rank',
        sortDir: 'asc',
    }};

    // --- Peer comparison state ---
    let peerState = {{
        currentTicker: null,
        defaultPeers: [],
        customPeers: [],
    }};

    function buildPeerRow(peerTicker) {{
        const d = D.stock_detail[peerTicker];
        if (!d) return null;
        const ey = d.raw ? d.raw.earnings_yield : null;
        return {{
            ticker: peerTicker,
            company: d.company || '',
            rev_growth: d.raw ? d.raw.revenue_growth : null,
            pe_ratio: (ey !== null && ey !== undefined && ey > 0) ? Math.round(10.0 / ey) / 10 : null,
            net_margin: d.financials ? d.financials.net_margin : null,
            roic: d.raw ? d.raw.roic : null,
            roe: d.raw ? d.raw.roe : null,
            debt_equity: d.raw ? d.raw.debt_equity : null,
            div_yield: d.financials ? d.financials.dividend_yield : null,
            fcf_yield: d.raw ? d.raw.fcf_yield : null,
            mcap: d.financials ? d.financials.market_cap : null,
        }};
    }}

    function arraysEqual(a, b) {{
        return a.length === b.length && a.every((v, i) => v === b[i]);
    }}

    function addPeer(peerTicker) {{
        if (peerTicker === peerState.currentTicker) return;
        if (peerState.customPeers.includes(peerTicker)) return;
        if (peerState.customPeers.length >= 10) return;
        peerState.customPeers.push(peerTicker);
        const s = D.stock_detail[peerState.currentTicker];
        renderPeerComparison(peerState.currentTicker, s);
    }}

    function removePeer(peerTicker) {{
        peerState.customPeers = peerState.customPeers.filter(t => t !== peerTicker);
        const s = D.stock_detail[peerState.currentTicker];
        renderPeerComparison(peerState.currentTicker, s);
    }}

    function resetPeers() {{
        peerState.customPeers = [...peerState.defaultPeers];
        const s = D.stock_detail[peerState.currentTicker];
        renderPeerComparison(peerState.currentTicker, s);
    }}

    function setupPeerSearch() {{
        const input = document.getElementById('peer-search-input');
        const dropdown = document.getElementById('peer-search-results');
        if (!input || !dropdown) return;

        input.addEventListener('input', function() {{
            const q = this.value.toLowerCase().trim();
            if (q.length < 1) {{ dropdown.innerHTML = ''; dropdown.style.display = 'none'; return; }}

            const excluded = new Set([peerState.currentTicker, ...peerState.customPeers]);
            const matches = D.table_data
                .filter(r => !excluded.has(r.Ticker) &&
                            (r.Ticker.toLowerCase().includes(q) ||
                             (r.Company || '').toLowerCase().includes(q)))
                .slice(0, 8);

            if (matches.length === 0) {{
                dropdown.innerHTML = '<div class="peer-search-empty">No matches</div>';
                dropdown.style.display = 'block';
                return;
            }}

            dropdown.innerHTML = matches.map(r =>
                `<div class="peer-search-item" onmousedown="addPeer('${{escapeHtml(r.Ticker)}}')">
                    <span class="peer-search-ticker">${{escapeHtml(r.Ticker)}}</span>
                    <span class="peer-search-company">${{escapeHtml(r.Company || '')}}</span>
                    <span class="peer-search-score">${{r.Composite !== null ? r.Composite.toFixed(0) : '--'}}</span>
                </div>`
            ).join('');
            dropdown.style.display = 'block';
        }});

        input.addEventListener('blur', function() {{
            setTimeout(() => {{ dropdown.style.display = 'none'; }}, 200);
        }});
        input.addEventListener('focus', function() {{
            if (this.value.trim().length > 0) this.dispatchEvent(new Event('input'));
        }});
    }}

{_js_table()}

    // Sort click handler
    document.querySelectorAll('#universe-table th[data-sort]').forEach(th => {{
        th.tabIndex = 0;
        th.setAttribute('role', 'columnheader');
        th.addEventListener('click', () => sortTable(th.dataset.sort));
        th.addEventListener('keydown', e => {{
            if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); sortTable(th.dataset.sort); }}
        }});
    }});

    // =====================================================================
    // WHAT CHANGED - the dashboard's time dimension
    // =====================================================================
    const H = D.history || {{}};
    // Defaults to the ~1-month comparison, not the previous run. On the run
    // this was built against, every one of the 10 material one-day movers was
    // a round-trip (an excursion that returned to base), while 169 of 193
    // one-month movers were genuine trends. Day-to-day rank changes at this
    // cadence are dominated by metric noise, so leading with them would invite
    // trading on artifacts.
    let changedRange = (H.compare && H.compare.m1) ? 'm1' : 'prev';

    function sparkline(ranks, dir) {{
        // Rank 1 is best, so the y-axis is inverted: a line going up means the
        // stock climbed the table. No axes or gridlines - a sparkline's job is
        // shape, and the exact values are in the row beside it.
        const pts = ranks.map((r, i) => [i, r]).filter(p => p[1] != null);
        if (pts.length < 2) return '<span class="spark-empty">&mdash;</span>';
        const W = 84, HT = 22, PAD = 3;
        const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
        const x0 = Math.min(...xs), x1 = Math.max(...xs);
        let y0 = Math.min(...ys), y1 = Math.max(...ys);
        if (y1 === y0) {{ y1 = y0 + 1; }}
        const sx = i => PAD + (x1 === x0 ? 0 : (i - x0) / (x1 - x0)) * (W - 2 * PAD);
        const sy = r => PAD + (r - y0) / (y1 - y0) * (HT - 2 * PAD);
        const d = pts.map((p, k) => (k ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' ');
        const last = pts[pts.length - 1];
        // The endpoint is only coloured when the latest move clears the
        // materiality floor. Colouring a 3-rank drift green would assert a
        // significance the measurement does not support.
        const floor = (H.noise && H.noise.material_threshold) || 0;
        const sig = Math.abs(dir || 0) >= floor ? dir : 0;
        const dotColor = sig > 0 ? 'var(--green)' : (sig < 0 ? 'var(--red)' : 'var(--text-secondary)');
        return `<svg class="spark" width="${{W}}" height="${{HT}}" viewBox="0 0 ${{W}} ${{HT}}" aria-hidden="true">
            <path d="${{d}}" fill="none" stroke="var(--text-secondary)" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round" opacity=".75"/>
            <circle cx="${{sx(last[0]).toFixed(1)}}" cy="${{sy(last[1]).toFixed(1)}}" r="2.5" fill="${{dotColor}}"/>
        </svg>`;
    }}

    function moverRow(m) {{
        const det = D.stock_detail[m.t] || {{}};
        const series = (H.series && H.series[m.t]) ? H.series[m.t].r : [];
        // Direction is carried by a glyph and a signed number as well as by
        // colour, so the row is readable without colour vision.
        const up = m.dr > 0;
        const arrow = up ? '▲' : '▼';
        const cls = up ? 'pos' : 'neg';
        const drv = m.drv
            ? `<span class="mover-driver">${{CAT_LABELS[m.drv[0]] || m.drv[0]}} ${{m.drv[1] > 0 ? '+' : ''}}${{m.drv[1].toFixed(1)}}</span>`
            : '<span class="mover-driver muted">&mdash;</span>';
        const rt = m.rt
            ? `<span class="rt-badge" title="Round-trip: this stock's rank made a large excursion and came back to within ${{H.noise.material_threshold}} ranks of where it started. That pattern is usually a metric dropping out and returning rather than a real change - treat it as a data-quality flag, not news.">round-trip</span>`
            : '';
        return `<div class="mover-row" onclick="openStockDetail('${{escapeHtml(m.t)}}')">
            <div class="mover-id">
                <span class="mover-ticker">${{escapeHtml(m.t)}}</span>
                <span class="mover-name">${{escapeHtml(det.company || '')}}</span>
            </div>
            ${{sparkline(series, m.dr)}}
            <div class="mover-delta ${{cls}}"><span class="mover-arrow">${{arrow}}</span>${{Math.abs(m.dr)}}</div>
            <div class="mover-meta">#${{det.rank != null ? det.rank : '?'}} ${{drv}} ${{rt}}</div>
        </div>`;
    }}

    function renderChanged() {{
        const sec = document.getElementById('sec-changed');
        if (!H.available || !H.compare || !H.movers) return;
        const cmp = H.compare[changedRange];
        const mv = H.movers[changedRange];
        if (!cmp || !mv) return;
        sec.style.display = '';

        // Range switch
        const opts = [];
        if (H.compare.prev) opts.push(['prev', 'Since last run']);
        if (H.compare.m1) opts.push(['m1', 'Since ~1 month']);
        document.getElementById('changed-range').innerHTML = opts.map(([k, label]) =>
            `<button class="seg-btn${{k === changedRange ? ' active' : ''}}" onclick="setChangedRange('${{k}}')">${{label}}</button>`
        ).join('');

        const n = H.noise || {{}};
        document.getElementById('changed-caption').innerHTML =
            `Comparing <strong>${{escapeHtml(H.current_date || '')}}</strong> with <strong>${{escapeHtml(cmp.date)}}</strong> `
            + `(${{cmp.gap_days}} ${{cmp.gap_days === 1 ? 'day' : 'days'}}). A move counts as material past <strong>&plusmn;${{n.material_threshold}} ranks</strong>`
            + (n.source === 'measured'
                ? ` &mdash; the 95th percentile of ordinary run-to-run variation, measured across ${{n.n_pairs}} paired runs (${{n.n_observations.toLocaleString()}} observations).`
                : ` &mdash; a default used until there are enough paired runs to measure it here.`);

        const up = mv.up.map(moverRow).join('') || '<div class="mover-none">No material moves up.</div>';
        const down = mv.down.map(moverRow).join('') || '<div class="mover-none">No material moves down.</div>';
        document.getElementById('movers-up').innerHTML = up + moversMore('movers-up', mv.up.length);
        document.getElementById('movers-down').innerHTML = down + moversMore('movers-down', mv.down.length);

        // Footnote: what was truncated, what was flagged, what was excluded.
        const notes = [];
        const shownUp = Math.min(mv.up.length, mv.n_up), shownDown = Math.min(mv.down.length, mv.n_down);
        notes.push(`${{mv.n_up}} stocks moved up materially and ${{mv.n_down}} moved down; showing the largest ${{shownUp}} and ${{shownDown}}.`);
        if (mv.n_round_trip) {{
            notes.push(`<strong>${{mv.n_round_trip}} of ${{mv.n_up + mv.n_down}}</strong> are flagged round-trip &mdash; a large excursion that returned to base, which usually means a metric dropped out and came back rather than the stock changing.`);
        }}
        notes.push(`Built from ${{H.dates.length}} comparable runs (${{escapeHtml(H.dates[0])}} to ${{escapeHtml(H.dates[H.dates.length - 1])}}).`);
        if (H.excluded && H.excluded.length) {{
            const ex = H.excluded.map(e => `${{escapeHtml(e.date)}} (${{escapeHtml(e.detail)}})`).join('; ');
            notes.push(`Excluded as not comparable: ${{ex}}.`);
        }}
        notes.push('Rank changes are not a recommendation. A stock that fell may be cheaper, not worse &mdash; open it to see which categories moved and why.');
        // The panel that shows the largest moves is exactly where a reader is
        // most likely to read a daily redraw as a daily decision.
        if (cadenceText(false)) notes.push(cadenceText(false));
        document.getElementById('changed-footnote').innerHTML = notes.join(' ');
    }}

    // Five rows each way by default; the rest on request. Nothing is dropped from the
    // list - the footnote still reports how many moved - it is only folded away.
    function moversMore(id, n) {{
        return n > 5 ? `<button type="button" class="show-more" onclick="expandMovers('${{id}}')">Show all ${{n}}</button>` : '';
    }}
    function expandMovers(id) {{
        const el = document.getElementById(id);
        if (!el) return;
        el.classList.add('expanded');
        const b = el.querySelector('.show-more');
        if (b) b.remove();
    }}

    function setChangedRange(k) {{
        changedRange = k;
        renderChanged();
    }}

    // Rank delta for the universe table, relative to the previous run.
    function rankDelta(ticker) {{
        const d = H.delta && H.delta[ticker] && H.delta[ticker].prev;
        if (!d || d.new || d.dr == null) return null;
        return d.dr;
    }}

    // =====================================================================
    // MY HOLDINGS  (priority 5 / north-star gap 2 - the sell-side workflow)
    // =====================================================================
    // Three properties of this panel come from
    // research/2026-09-14-sell-discipline-and-hold-bands.md. They are not
    // style choices and should not be "tidied up":
    //
    //  1. It renders EVERY saved name on every render, never a filtered
    //     subset. Akepanidtaworn, Di Mascio, Imas & Schmidt (2023, JF 78(6))
    //     trace an 80 bp/year institutional selling deficit to a restricted
    //     consideration set: PMs sell positions that are extreme on prior
    //     returns at rates >50% higher than middling ones. A queue that
    //     surfaces only the big movers is that heuristic, implemented.
    //  2. Rows are ordered by current rank, never by size of move - same
    //     reason. The rank change is shown for context but is not the sort key
    //     and is not a filter.
    //  3. No cost basis, share count or profit-and-loss field exists anywhere
    //     in this code or in what it stores. Gain/loss against purchase price
    //     is the reference point that produces the disposition effect (Odean
    //     1998, JF 53(5): PGR/PLR = 1.50, t = -32).
    //
    // There is deliberately no sell signal. The screener has one test - the
    // top 25 - and the evidence says the hold test should be a different,
    // wider test; its width has not been set here yet.
    //
    // Measured constraint behind all of this: the movers panel's 43-rank
    // material threshold fires for a top-25 name 0.15% of the time (1 in 675
    // holding-days over 32 comparable runs), so a holdings surface cannot be
    // built on it.
    const HOLDINGS_KEY = 'screener_holdings_v1';
    const HOLDINGS_MAX = 60;
    // Which baked summary facts belong on a review row. All four are built by
    // stock_summary.py at build time and screened for advice language there,
    // so this panel renders reviewed prose instead of composing its own.
    // `input_churn` sits immediately after the two sentences it qualifies.
    // Without it a reader meets "moved down 63 places" with no way to tell that
    // two of the inputs behind the score changed availability over the same
    // window - which triples the median rank move and is not information about
    // the company (research/2026-09-14-... section 8.3).
    // `earnings` is here because this is the surface the evidence for it points
    // at: announcement-day sells are the only sells in Akepanidtaworn et al.
    // (2023) that beat their counterfactual, by more than +150 bp/year. The
    // order is the summary's, not this array's - the filter below preserves it.
    const HOLDINGS_FACTS = ['change', 'change_driver', 'input_churn', 'flags', 'confidence', 'earnings'];

    let holdings = [];

    function holdingsStorage() {{
        // Safari private mode and some embedded browsers throw on access
        // rather than returning null. A blocked store must degrade to an
        // in-memory list for the session, not blank the section.
        try {{ return window.localStorage; }} catch (e) {{ return null; }}
    }}

    function loadHoldings() {{
        const store = holdingsStorage();
        if (!store) return [];
        let raw = null;
        try {{ raw = store.getItem(HOLDINGS_KEY); }} catch (e) {{ return []; }}
        if (!raw) return [];
        let parsed;
        try {{ parsed = JSON.parse(raw); }} catch (e) {{ return []; }}
        if (!Array.isArray(parsed)) return [];
        // Tickers only, and only tickers this run actually scored. Anything
        // else in the key - from a hand edit, or a future build that tried to
        // store more - is dropped on read and not written back, so a cost
        // basis cannot survive in storage even if something put one there.
        const seen = {{}};
        const out = [];
        parsed.forEach(function(item) {{
            const t = (typeof item === 'string') ? item.trim().toUpperCase() : '';
            if (!t || seen[t] || !D.stock_detail[t]) return;
            seen[t] = true;
            out.push(t);
        }});
        return out.slice(0, HOLDINGS_MAX);
    }}

    function saveHoldings() {{
        const store = holdingsStorage();
        if (!store) return;
        try {{ store.setItem(HOLDINGS_KEY, JSON.stringify(holdings)); }} catch (e) {{ /* quota or blocked */ }}
    }}

    function addHolding(ticker) {{
        const t = String(ticker || '').trim().toUpperCase();
        if (!t || !D.stock_detail[t]) return;
        if (holdings.indexOf(t) !== -1 || holdings.length >= HOLDINGS_MAX) return;
        holdings.push(t);
        saveHoldings();
        const input = document.getElementById('holdings-search-input');
        if (input) input.value = '';
        const dd = document.getElementById('holdings-search-results');
        if (dd) {{ dd.innerHTML = ''; dd.style.display = 'none'; }}
        renderHoldings();
    }}

    function removeHolding(ticker) {{
        holdings = holdings.filter(function(t) {{ return t !== ticker; }});
        saveHoldings();
        renderHoldings();
    }}

    function clearHoldings() {{
        if (holdings.length && !window.confirm('Remove all ' + holdings.length + ' names from this list?')) return;
        holdings = [];
        saveHoldings();
        renderHoldings();
    }}

    // The baseline a holdings row quotes. Same preference order as
    // stock_summary._pick_comparison, so the chips and the sentences below
    // them cannot end up describing different windows: the ~1-month window
    // first, because every material one-day mover measured on 2026-08-25 was a
    // round trip while 169 of 193 one-month moves were genuine trends.
    function holdingDelta(ticker) {{
        const d = H.delta && H.delta[ticker];
        if (!d) return null;
        const keys = ['m1', 'prev'];
        for (let i = 0; i < keys.length; i++) {{
            const e = d[keys[i]];
            if (!e || e.new || e.dr === null || e.dr === undefined) continue;
            const cmp = (H.compare && H.compare[keys[i]]) || {{}};
            return {{ dr: e.dr, dc: e.dc, cat: e.cat || {{}},
                     date: cmp.date, gap: cmp.gap_days }};
        }}
        return null;
    }}

    // =====================================================================
    // REVIEW CADENCE  (research/2026-09-14-... section 8.2)
    // =====================================================================
    // The site refreshes every weekday; the methodology is built for a
    // quarterly review. Until 2026-09-17 it stated neither, so a reader met a
    // freshly-redrawn ranking every morning with nothing to say how often
    // acting on it was intended. Measured on this repo's snapshots, acting on
    // the strict top-25 rule at every run implies 121.8% monthly one-sided
    // turnover against 24.0% at monthly review, where Novy-Marx & Velikov
    // (2016) find few anomalies survive costs above ~50%.
    //
    // It is a sentence, not a lock. The tool does not know what a reader is
    // doing and must not pretend to; naming the cadence it was built for is
    // decision support, refusing to show a number until a date would not be.
    // Returns the sentence(s) only. Callers wrap it, because one surface wants
    // a standalone note and the other wants it inside an existing paragraph.
    function cadenceText(long) {{
        const c = D.cadence;
        if (!c) return '';
        const refresh = 'This page is rebuilt every weekday, but the ranking is '
            + 'built for <strong>' + escapeHtml(c.label) + '</strong> review.';
        if (!long) {{
            return refresh
                + ' A rank that moved since yesterday is not by itself a reason to act.';
        }}
        return refresh
            + ' Treating every refresh as a decision point has a measured cost: on this '
            + 'screener\\u2019s own history, acting on the top '
            + (c.num_stocks || 25) + ' at every run implies <strong>'
            + c.turnover_every_run.toFixed(0) + '%</strong> monthly one-sided turnover '
            + 'against <strong>' + c.turnover_monthly.toFixed(0) + '%</strong> reviewing '
            + 'the same rule monthly. Novy-Marx &amp; Velikov (<em>Review of Financial '
            + 'Studies</em> 29(1), 2016) find anomalies under roughly <strong>'
            + c.turnover_ceiling.toFixed(0) + '%</strong> mostly survive trading costs, '
            + 'and few above it do.';
    }}

    function cadenceLine(long) {{
        const text = cadenceText(long);
        return text ? '<p class="cadence-note">' + text + '</p>' : '';
    }}

    function renderHoldings() {{
        const body = document.getElementById('holdings-body');
        const fit = document.getElementById('holdings-fit');
        const count = document.getElementById('holdings-count');
        const clearBtn = document.getElementById('holdings-clear-btn');
        if (!body || !fit || !count || !clearBtn) return;

        count.textContent = holdings.length ? '(' + holdings.length + ')' : '';
        clearBtn.style.display = holdings.length ? '' : 'none';
        renderHoldingsFootnote();

        if (!holdings.length) {{
            fit.innerHTML = cadenceLine(false);
            body.innerHTML = '<div class="holdings-empty">'
                + '<p>Nothing on the list yet. Add a ticker above, or press <strong>Add to Holdings</strong> in any stock&rsquo;s drilldown, and it appears here after every run, with what moved and which category moved it.</p>'
                + '<p class="holdings-empty-sub">Works the same whether you own the name or are only watching it &mdash; nothing here assumes you hold a position.</p>'
                + '</div>';
            return;
        }}

        // Ordered by current rank, best first. Deliberately NOT by size of
        // move - see property 2 in the block comment above HOLDINGS_KEY.
        const rows = holdings
            .map(function(t) {{ return {{ t: t, s: D.stock_detail[t] }}; }})
            .filter(function(r) {{ return !!r.s; }})
            .sort(function(a, b) {{
                const ra = (a.s.rank === null || a.s.rank === undefined) ? 1e9 : a.s.rank;
                const rb = (b.s.rank === null || b.s.rank === undefined) ? 1e9 : b.s.rank;
                return ra - rb;
            }});

        fit.innerHTML = cadenceLine(false) + holdingsFitLine(rows) + holdingsConcentration(rows);
        body.innerHTML = rows.map(holdingCard).join('');
    }}

    // North-star question 4 - "how much / does it fit?" - as far as this tool
    // can honestly answer it. It reports concentration, not position sizes:
    // the list holds no weights, so any sizing number would be invented.
    function holdingsFitLine(rows) {{
        const n = rows.length;
        const bySector = {{}};
        rows.forEach(function(r) {{
            const sec = r.s.sector || 'Unclassified';
            bySector[sec] = (bySector[sec] || 0) + 1;
        }});
        const sectors = Object.keys(bySector);
        let topSector = sectors[0], topCount = bySector[sectors[0]];
        sectors.forEach(function(sec) {{
            if (bySector[sec] > topCount) {{ topSector = sec; topCount = bySector[sec]; }}
        }});
        const universe = D.kpis.universe_size || D.table_data.length;
        const inTop25 = rows.filter(function(r) {{ return r.s.rank <= 25; }}).length;
        const inTop100 = rows.filter(function(r) {{ return r.s.rank <= 100; }}).length;
        const flagged = rows.filter(function(r) {{ return r.s.vt || r.s.gt; }}).length;
        const parts = [
            '<strong>' + n + '</strong> name' + (n === 1 ? '' : 's')
              + ' across <strong>' + sectors.length + '</strong> sector' + (sectors.length === 1 ? '' : 's'),
            'largest concentration <strong>' + escapeHtml(topSector) + '</strong> ('
              + topCount + ' of ' + n + ', ' + Math.round(topCount / n * 100) + '%)',
            '<strong>' + inTop25 + '</strong> inside the top 25 of ' + universe
              + ', <strong>' + inTop100 + '</strong> inside the top 100'
        ];
        if (flagged) {{
            parts.push('<strong>' + flagged + '</strong> carrying a trap flag');
        }}
        return '<div class="holdings-fit-line">' + parts.join(' &bull; ') + '</div>';
    }}

    // Published counts for a diversified portfolio. Every one is measured on
    // RANDOMLY selected portfolios - see section 3.4 of
    // research/2026-09-21-position-sizing-and-how-much.md. They are literature
    // constants rather than run output, so they live here and cost the payload
    // nothing.
    const NAME_MARKS = [
        {{n: 30, label: '30&ndash;40 (Statman, <em>JFQA</em> 1987)'}},
        {{n: 50, label: 'about 50 (Campbell, Lettau, Malkiel &amp; Xu, <em>JF</em> 2001)'}},
        {{n: 63, label: '63 for a 10% shortfall risk over 20 years (Domian, Louton &amp; Racine, <em>Financial Review</em> 2007)'}}
    ];

    // The second half of north-star question 4 - "how much / does it fit?".
    // The fit line above reports what the list contains; this reports the two
    // things a reader needs to size it, and neither is a position weight:
    // where the name count sits against the published thresholds, and how far
    // apart the holdings are on risk.
    //
    // It never emits a target weight for a stock. That is the distinction that
    // got the Model Portfolio deleted on 2026-08-26, and the equal-split line
    // stays on the right side of it by being arithmetic on the LENGTH of the
    // list - identical for every name on it - rather than a per-stock number.
    function holdingsConcentration(rows) {{
        const n = rows.length;
        if (!n) return '';
        const parts = [];

        const cleared = NAME_MARKS.filter(function(m) {{ return n >= m.n; }}).length;
        const standing = cleared === 0
            ? 'below all three'
            : (cleared === NAME_MARKS.length
                ? 'at or above all three'
                : 'above ' + cleared + ' of the three');
        parts.push(
            '<p><strong>' + n + '</strong> name' + (n === 1 ? '' : 's') + ' &mdash; <strong>'
            + standing + '</strong> of the published counts for a diversified portfolio: '
            + NAME_MARKS.map(function(m) {{ return m.label; }}).join('; ')
            + '. All three measure <em>randomly chosen</em> portfolios, so a pre-screened '
            + 'large-cap list carries less single-stock risk than the raw numbers imply. '
            + 'They bound the question rather than settle it.</p>'
        );

        const split = n === 1
            ? 'A single name is the whole of the list &mdash; <strong>100%</strong> in one position.'
            : 'An equal split across ' + n + ' names is <strong>' + (100 / n).toFixed(1)
              + '%</strong> a position.';
        parts.push(
            '<p>' + split + ' Published caps on a single holding, '
            + 'for scale: <strong>5%</strong> of net assets under UCITS (10% provided everything '
            + 'above 5% stays under 40%); <strong>25%</strong> under the US RIC 25/5/50 rule; '
            + 'S&amp;P Dow Jones re-caps a Select Sector constituent above <strong>24%</strong>. '
            + 'Those are limits on the top end, not targets &mdash; and this line is arithmetic '
            + 'on the length of the list, not a weight for any stock on it.</p>'
        );

        // Raw annualised volatility, deliberately NOT the volatility percentile
        // the payload also carries. That percentile is sector-relative
        // (factor_engine.compute_sector_percentiles groups by Sector) and
        // direction-inverted (METRIC_DIR['volatility'] is False, so a HIGH
        // percentile is a LOW-volatility stock), which makes it unable to rank
        // risk across a mixed-sector list.
        //
        // Measured on the live run by
        // research/measurements/2026-09-22-holdings-risk-comparability.py:
        // the percentile ranks risk backwards for 23.9% of the 111,417
        // cross-sector pairs, worst case a name reading as the safer holding
        // while carrying 2.00x the volatility. The raw number is directly
        // comparable and is what the "equal dollars" arithmetic needs.
        const withVol = rows
            .map(function(r) {{ return {{t: r.t, v: (r.s.raw || {{}}).volatility}}; }})
            .filter(function(x) {{ return typeof x.v === 'number' && isFinite(x.v) && x.v > 0; }})
            .sort(function(a, b) {{ return b.v - a.v; }});
        if (withVol.length >= 2) {{
            const hi = withVol[0], lo = withVol[withVol.length - 1];
            const ratio = hi.v / lo.v;
            parts.push(
                '<p>Widest risk gap on the list: <strong>' + escapeHtml(hi.t) + '</strong> at <strong>'
                + (hi.v * 100).toFixed(0) + '%</strong> annualised volatility against <strong>'
                + escapeHtml(lo.t) + '</strong> at <strong>' + (lo.v * 100).toFixed(0)
                + '%</strong>, a <strong>' + ratio.toFixed(1) + '&times;</strong> spread. Held in '
                + 'equal dollar amounts, ' + escapeHtml(hi.t) + ' moves the value of the list about '
                + ratio.toFixed(1) + '&times; as much as ' + escapeHtml(lo.t)
                + '. This is the past year measured, not a forecast.</p>'
            );
        }}

        return '<div class="holdings-concentration">'
            + '<h4 class="holdings-conc-title">Concentration</h4>'
            + parts.join('') + '</div>';
    }}

    function holdingCard(r) {{
        const t = r.t, s = r.s;
        const d = holdingDelta(t);
        const cats = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];

        let deltaChip = '<span class="holding-delta holding-delta-none">no comparable history</span>';
        if (d) {{
            const cls = d.dr > 0 ? 'up' : (d.dr < 0 ? 'down' : 'flat');
            const arrow = d.dr > 0 ? '\\u25B2' : (d.dr < 0 ? '\\u25BC' : '\\u2013');
            const label = d.dr === 0
                ? 'rank unchanged'
                : (Math.abs(d.dr) + ' rank' + (Math.abs(d.dr) === 1 ? '' : 's'));
            const since = d.date ? (' since ' + escapeHtml(d.date)) : '';
            deltaChip = '<span class="holding-delta holding-delta-' + cls + '" '
                + 'title="Rank change since the comparison run. Shown for context - this list is not sorted or filtered by it.">'
                + arrow + ' ' + label + since + '</span>';
        }}

        const strip = cats.map(function(c) {{
            const v = (s.cat_scores || {{}})[c];
            if (v === null || v === undefined) {{
                return '<div class="holding-cat holding-cat-nodata" title="'
                    + CAT_LABELS[c] + ' could not be scored for this stock, so the other categories were reweighted">'
                    + '<span class="holding-cat-name">' + CAT_LABELS[c] + '</span>'
                    + '<span class="holding-cat-val">no data</span></div>';
            }}
            const dv = (d && d.cat && d.cat[c] !== undefined && d.cat[c] !== null) ? d.cat[c] : null;
            let move = '';
            if (dv !== null) {{
                move = '<span class="holding-cat-move holding-cat-move-' + (dv > 0 ? 'up' : 'down') + '">'
                    + (dv > 0 ? '+' : '') + dv.toFixed(1) + '</span>';
            }}
            return '<div class="holding-cat" style="border-top-color:' + CAT_COLORS[c] + '">'
                + '<span class="holding-cat-name">' + CAT_LABELS[c] + '</span>'
                + '<span class="holding-cat-val">' + v.toFixed(0) + move + '</span></div>';
        }}).join('');

        const facts = Array.isArray(s.summary) ? s.summary : [];
        const notes = facts
            .filter(function(f) {{ return HOLDINGS_FACTS.indexOf(f.k) !== -1; }})
            .map(function(f) {{
                const kind = String(f.k || '').replace(/[^a-z_]/g, '');
                return '<p class="holding-note holding-note-' + kind + '">'
                    + escapeHtml(f.t || '') + '</p>';
            }}).join('');

        let trap = '';
        if (s.vt) trap += '<span class="holding-flag">Value trap</span>';
        if (s.gt) trap += '<span class="holding-flag">Growth trap</span>';

        return '<div class="holding-card">'
          + '<div class="holding-head">'
          +   '<span class="holding-rank">#' + ((s.rank === null || s.rank === undefined) ? '--' : s.rank) + '</span>'
          +   '<button class="holding-ticker" onclick="openStockDetail(&quot;' + escapeHtml(t) + '&quot;)" title="Open the full breakdown">' + escapeHtml(t) + '</button>'
          +   '<span class="holding-company">' + escapeHtml(s.company || '') + '</span>'
          +   '<span class="holding-sector">' + escapeHtml(s.sector || '') + '</span>'
          +   trap
          +   '<span class="holding-spacer"></span>'
          +   deltaChip
          +   '<span class="holding-composite" title="Composite score - a universe percentile">'
          +     ((s.composite === null || s.composite === undefined) ? '--' : s.composite.toFixed(1)) + '</span>'
          +   '<button class="holding-remove" onclick="removeHolding(&quot;' + escapeHtml(t) + '&quot;)" title="Remove from list">&times;</button>'
          + '</div>'
          + '<div class="holding-cats">' + strip + '</div>'
          + (notes ? '<div class="holding-notes">' + notes + '</div>' : '')
          + '</div>';
    }}

    // The teaching half. Every claim on this panel that a reader might
    // reasonably want to argue with is sourced here, because the constraints
    // above look arbitrary without their evidence and the next person to touch
    // this file will otherwise "fix" them.
    function renderHoldingsFootnote() {{
        const el = document.getElementById('holdings-footnote');
        if (!el) return;
        el.innerHTML = [
            '<p><strong>How often this is meant to be acted on.</strong> ' + cadenceText(true) + '</p>',
            '<p><strong>Why this list shows everything, every time.</strong> It is never filtered or sorted by how much a name moved. Institutional managers dispose of the best and worst performers in a portfolio at rates more than 50% higher than middling positions, and that habit is the identified cause of an 80 basis-point-a-year shortfall in their disposal decisions against a random-disposal benchmark over 4.4 million trades (Akepanidtaworn, Di Mascio, Imas &amp; Schmidt, <em>Journal of Finance</em> 78(6), 2023). A queue that surfaces only the big movers automates that habit. Rows here are ordered by current rank; the rank change is shown for context only.</p>',
            '<p><strong>Why it never asks what you paid.</strong> Measuring a position against its purchase price is the reference point behind the disposition effect: investors realise gains about 1.5&times; as readily as losses, and the winners they disposed of went on to beat the losers they kept by 3.4 percentage points over the following year (Odean, <em>Journal of Finance</em> 53(5), 1998). No cost basis, share count or profit-and-loss figure is stored or shown here, which is also why this works equally as a watchlist.</p>',
            '<p><strong>Why there is no exit signal.</strong> This screener has one test &mdash; the top 25 &mdash; and both the literature and index practice say the test for continued holding should be a different, wider one. A buy/hold spread is "the single most effective simple cost mitigation strategy" in Novy-Marx &amp; Velikov (<em>Review of Financial Studies</em> 29(1), 2016); MSCI buffers its momentum indexes between rank 250 and 750 against a 500-name target, and S&amp;P Dow Jones Indices states the principle outright: "the addition criteria are for addition to an index, not for continued membership." That second threshold has not been set for this screener yet, so this panel shows the evidence and leaves the decision where it belongs. Trading more often has a measured cost: the most active households in Barber &amp; Odean (<em>Journal of Finance</em> 55(2), 2000) earned 11.4% a year against a market return of 17.9%.</p>',
            '<p><strong>Why the next earnings date is here.</strong> It is the one place the same study finds attention is well spent: sells executed on a holding&rsquo;s earnings-announcement day beat non-announcement-day sells by more than <strong>150 basis points a year</strong>, and are the only sells in the sample that beat the random-disposal benchmark at all. The authors read that as attention rather than skill &mdash; an announcement is a pre-scheduled, external reason to look at a position you would otherwise not re-examine. It matters mechanically here too: this screener&rsquo;s fundamental inputs come from filings, and between filings they barely move. Measured across a month of this site&rsquo;s own runs, the largest Quality-score move was <strong>one stock in 500</strong>, against 34% for Risk. A report is when the Valuation, Quality and Growth numbers above are actually replaced. The date is descriptive only &mdash; it is never scored, never ranked and does not affect any number on this page &mdash; and where the data provider has not confirmed it, the line says so: measured across all 503 names on 2026-09-29, <strong>209 of the 492 forthcoming dates (42.5%)</strong> were the provider&rsquo;s estimate rather than a company-announced schedule. Where the provider has not scheduled the next report at all &mdash; 11 names that day, mostly off-calendar reporters &mdash; the line is omitted rather than showing a date that has already passed.</p>',
            '<p><strong>Why the Concentration block gives you no position size.</strong> Sizing by conviction is the one thing the estimation-error literature singles out as dangerous: errors in expected returns do roughly <strong>20&times;</strong> the damage of errors in covariances, and about <strong>100&times;</strong> for an investor near zero risk aversion (Chopra &amp; Ziemba, 1993, via Ziemba &amp; MacLean, <em>Stochastic Optimization Methods in Finance and Energy</em>, Springer, 2011). Across 14 optimisation models and 7 datasets, none consistently beat a plain equal split out of sample, which would need an estimation window of roughly <strong>3,000 months</strong> for 25 assets to do reliably (DeMiguel, Garlappi &amp; Uppal, <em>Review of Financial Studies</em> 22(5), 2009). In every documented institutional scheme &mdash; equal, cap, inverse-volatility or optimiser weight &mdash; the alpha signal drives <em>selection</em> and weighting is a separate, risk-driven decision. So this panel reports the facts about your list and the published external limits, and leaves the number to you.</p>',
            '<p><strong>Why the risk line quotes raw volatility and not a percentile.</strong> Every metric percentile on this site is <em>sector</em>-relative and direction-adjusted, so on volatility a high percentile means a stock is calm <em>for its sector</em>. That cannot rank risk across a list spanning several sectors: measured on this run, the percentile orders the pair backwards for <strong>23.9%</strong> of all cross-sector pairs, in the worst case making a name read as the safer holding while carrying <strong>2.00&times;</strong> the volatility. The annualised figure above is the raw one-year number and is directly comparable between any two names.</p>',
            '<p class="holdings-storage-note">Saved in this browser only, under the <code>' + HOLDINGS_KEY + '</code> key in <code>localStorage</code>. It is not an account and it is not backed up &mdash; clearing site data removes it, and it will not follow you to another device. Tickers only, up to ' + HOLDINGS_MAX + ' names. This is decision support, not investment advice.</p>'
        ].join('');
    }}

    function setupHoldingsSearch() {{
        const input = document.getElementById('holdings-search-input');
        const dropdown = document.getElementById('holdings-search-results');
        if (!input || !dropdown) return;

        input.addEventListener('input', function() {{
            const q = this.value.toLowerCase().trim();
            if (q.length < 1) {{ dropdown.innerHTML = ''; dropdown.style.display = 'none'; return; }}
            const held = {{}};
            holdings.forEach(function(t) {{ held[t] = true; }});
            const matches = D.table_data
                .filter(function(r) {{
                    return !held[r.Ticker] &&
                        (r.Ticker.toLowerCase().includes(q) ||
                         (r.Company || '').toLowerCase().includes(q));
                }})
                .slice(0, 8);
            if (!matches.length) {{
                dropdown.innerHTML = '<div class="peer-search-empty">No matches</div>';
                dropdown.style.display = 'block';
                return;
            }}
            dropdown.innerHTML = matches.map(function(r) {{
                return '<div class="peer-search-item" onmousedown="addHolding(&quot;' + escapeHtml(r.Ticker) + '&quot;)">'
                    + '<span class="peer-search-ticker">' + escapeHtml(r.Ticker) + '</span>'
                    + '<span class="peer-search-company">' + escapeHtml(r.Company || '') + '</span>'
                    + '<span class="peer-search-score">' + (r.Composite !== null ? r.Composite.toFixed(0) : '--') + '</span>'
                    + '</div>';
            }}).join('');
            dropdown.style.display = 'block';
        }});

        input.addEventListener('blur', function() {{
            setTimeout(function() {{ dropdown.style.display = 'none'; }}, 200);
        }});
        input.addEventListener('focus', function() {{
            if (this.value.trim().length > 0) this.dispatchEvent(new Event('input'));
        }});
    }}

    function initHoldings() {{
        holdings = loadHoldings();
        // A saved list means the panel is in use, so open it. The empty
        // default stays collapsed: the owner's landing view is Top 5 plus the
        // full table, with everything else one click away (2026-08-26).
        if (holdings.length) {{
            const sec = document.getElementById('sec-holdings');
            if (sec) sec.classList.remove('collapsed');
        }}
        setupHoldingsSearch();
        renderHoldings();
    }}

    // =====================================================================
    // STOCK DETAIL MODAL
    // =====================================================================
    // One hue for all categories (see plan/dashboard-design-system.md):
    // identity rides the labels, color marks magnitude fills only.
    const CAT_COLORS = {{
        valuation: ACCENT, quality: ACCENT, growth: ACCENT,
        momentum: ACCENT, risk: ACCENT, revisions: ACCENT,
        size: ACCENT, investment: ACCENT
    }};
    const CAT_LABELS = {{
        valuation: 'Valuation', quality: 'Quality', growth: 'Growth',
        momentum: 'Momentum', risk: 'Risk', revisions: 'Revisions',
        size: 'Size', investment: 'Investment'
    }};

    // Percentiles are direction-adjusted: compute_sector_percentiles() does
    // `100 - rank` for every metric whose METRIC_DIR is False, so 100 always
    // means "best in sector" and never "largest number". Stating this is not
    // decoration - without it a reader sees EV/EBITDA 6.95 at the 99th
    // percentile and reasonably concludes the percentile tracks the raw value,
    // which is backwards for 15 of the 37 published metrics.
    // The count is read off the payload rather than written here, so it stays
    // true as metrics are added or their direction changes.
    const PCTILE_CONVENTION = (() => {{
        const mm = (typeof D !== 'undefined' && D.metric_meta) || {{}};
        const n = Object.keys(mm).filter(m => mm[m].dir === 'lower').length;
        const total = Object.keys(mm).length;
        return '100 = best in its sector, not largest. For the ' + n + ' of '
            + total + ' metrics marked ↓ better (EV/EBITDA, Beta and PEG among '
            + 'them) a low raw value earns a high percentile. Percentiles '
            + 'compare a stock with its own sector, not the whole index.';
    }})();

    // Reads metric_meta.dir, which generate_dashboard derives from the scorer's
    // METRIC_DIR - so this marker cannot drift from the ranking it describes.
    function dirChip(meta) {{
        if (!meta || !meta.dir) return '';
        const lower = meta.dir === 'lower';
        const arrow = lower ? '↓' : '↑';
        const tip = lower
            ? 'Lower is better: a smaller value earns a higher percentile.'
            : 'Higher is better: a larger value earns a higher percentile.';
        return ` <span class="metric-dir ${{lower ? 'metric-dir-lower' : 'metric-dir-higher'}}" title="${{tip}}">${{arrow}} better</span>`;
    }}
    // "What does this company actually do?" - the one question the screener
    // could not answer before 2026-08-26. Verbatim provider text, never scored.
    function renderAbout(s) {{
        const box = document.getElementById('modal-about');
        const txt = document.getElementById('modal-about-text');
        const btn = document.getElementById('modal-about-toggle');
        const ind = document.getElementById('modal-industry');
        const about = (s && s.about) ? String(s.about).trim() : '';
        ind.textContent = (s && s.industry) ? s.industry : '';
        if (!about) {{ box.style.display = 'none'; return; }}
        box.style.display = '';
        txt.textContent = about;
        txt.classList.add('clamped');
        btn.textContent = 'Show more';
        // Only offer the toggle when the text is actually being cut off.
        // Measured after layout: renderAbout runs while the modal is still
        // display:none, where both heights read 0 and the button would be
        // hidden on every stock.
        btn.style.display = 'none';
        requestAnimationFrame(function() {{
            btn.style.display = (txt.scrollHeight > txt.clientHeight + 2) ? '' : 'none';
        }});
    }}

    // "Why it ranks here" - the deterministic replacement for the AI chat.
    // The sentences are built in stock_summary.py at run time and baked into
    // the payload, so what renders here is exactly what shipped: no request,
    // no API key, no per-viewer variation. Text only, escaped, no links.
    function renderSummary(s) {{
        const box = document.getElementById('modal-summary');
        const body = document.getElementById('modal-summary-body');
        if (!box || !body) return;
        let facts = (s && Array.isArray(s.summary)) ? s.summary : [];
        if (!facts.length) {{
            box.style.display = 'none';
            body.innerHTML = '';
            return;
        }}
        box.style.display = '';
        // The lead line is the headline; the rest sit under quiet group labels so
        // eleven sentences stop reading as one block of equal weight.
        const GROUPS = {{
            drivers: 'What drives the score', weakest: 'What drives the score',
            best_inputs: 'What drives the score', worst_input: 'What drives the score',
            change: 'What changed', change_driver: 'What changed', input_churn: 'What changed',
            target: 'Context', peers: 'Context', earnings: 'Context',
            flags: 'Read with care', confidence: 'Read with care'
        }};
        // Reorder so each group is contiguous, in order of first appearance, and the
        // lead sentence stays first - otherwise a label repeats when kinds interleave.
        const lead = [], order = [], byGroup = {{}};
        facts.forEach(function(f) {{
            const group = GROUPS[String(f.k || '').replace(/[^a-z_]/g, '')] || '';
            if (!group) {{ lead.push(f); return; }}
            if (!byGroup[group]) {{ byGroup[group] = []; order.push(group); }}
            byGroup[group].push(f);
        }});
        facts = lead.concat.apply(lead, order.map(function(g) {{ return byGroup[g]; }}));
        let lastGroup = '';
        body.innerHTML = facts.map(function(f) {{
            const kind = String(f.k || '').replace(/[^a-z_]/g, '');
            const group = GROUPS[kind] || '';
            let label = '';
            if (group && group !== lastGroup) {{
                label = '<div class="summary-group-label">' + group + '</div>';
                lastGroup = group;
            }}
            return label + '<p class="summary-fact summary-' + kind + '">' +
                   escapeHtml(f.t || '') + '</p>';
        }}).join('');
    }}

    function toggleAbout() {{
        const txt = document.getElementById('modal-about-text');
        const btn = document.getElementById('modal-about-toggle');
        const clamped = txt.classList.toggle('clamped');
        btn.textContent = clamped ? 'Show more' : 'Show less';
    }}

    // Weights for ONE stock, after dropping categories it has no score for.
    //
    // Two things move a weight away from the headline number in the
    // Methodology page, and both have to be shown or the arithmetic on this
    // page does not add up:
    //   1. the run-level volatility-regime adjustment, already baked into
    //      D.weights.factor_weights; and
    //   2. this renormalisation, when a category was withheld for this stock
    //      (e.g. a rejected price series takes out Momentum and Risk).
    // Mirrors factor_engine.compute_factor_contributions.
    function effWeights(s) {{
        const fw = D.weights.factor_weights || {{}};
        const cats = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        let total = 0;
        cats.forEach(c => {{
            const w = fw[c] || 0;
            if (w > 0 && s.cat_scores[c] !== null && s.cat_scores[c] !== undefined) total += w;
        }});
        const out = {{}};
        cats.forEach(c => {{
            const w = fw[c] || 0;
            const live = w > 0 && s.cat_scores[c] !== null && s.cat_scores[c] !== undefined;
            out[c] = (live && total > 0) ? (w / total) * 100 : 0;
        }});
        out._renormalised = total > 0 && Math.abs(total - 100) > 0.01;
        return out;
    }}

    // How a weight should read: "13%", or "15.0%" once it stops being round.
    function fmtWeight(w) {{
        if (w === null || w === undefined) return '—';
        return (Math.abs(w - Math.round(w)) < 0.05 ? Math.round(w) : w.toFixed(1)) + '%';
    }}

    function openStockDetail(ticker) {{
        const s = D.stock_detail[ticker];
        if (!s) return;

        document.getElementById('modal-ticker').textContent = ticker;
        document.getElementById('modal-company').textContent = s.company;
        document.getElementById('modal-sector').textContent = s.sector;
        const hl = document.getElementById('modal-headline');
        if (hl) hl.innerHTML = `<span class="mh-item"><span class="mh-k">Rank</span><strong>#${{s.rank}}</strong><span class="mh-of">of ${{D.kpis.universe_size}}</span></span>` +
            `<span class="mh-item"><span class="mh-k">Composite</span><strong>${{fmt(s.composite,'score')}}</strong></span>`;

        renderSummary(s);
        renderAbout(s);

        // Score summary cards
        const cats = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        let scoreHtml = `<div class="modal-score-card composite">
            <div class="modal-score-label">Composite</div>
            <div class="modal-score-val">${{fmt(s.composite,'score')}}</div>
            <div class="modal-score-sub">Rank #${{s.rank}} of ${{D.kpis.universe_size}}</div>
        </div>`;
        const ew = effWeights(s);
        cats.forEach(c => {{
            const score = s.cat_scores[c];
            const contrib = s.contrib[c];
            const weight = (score === null || score === undefined) ? null : ew[c];
            const wtText = weight === null ? 'no data' : fmtWeight(weight) + ' wt';
            const sv = (score === null || score === undefined) ? 0 : Math.max(0, Math.min(100, score)) / 100;
            scoreHtml += `<div class="modal-score-card" style="--v:${{sv.toFixed(3)}}" onclick="openWorkings('${{c}}')" title="Show how this score is built">
                <div class="modal-score-label">${{CAT_LABELS[c]}}</div>
                <div class="modal-score-val">${{fmt(score,'score')}}</div>
                <div class="modal-score-sub">${{fmt(contrib,'score')}} pts (${{wtText}})</div>
            </div>`;
        }});
        document.getElementById('modal-score-row').innerHTML = scoreHtml;

        // Analyst price targets
        renderPriceTargets(s);

        // Company snapshot (financials) + peer comparison
        renderCompanySnapshot(s);
        renderPeerComparison(ticker, s);

        // Rank history over prior comparable runs
        renderStockHistory(ticker);

        // Data provenance
        renderProvenance(s);

        // Contribution breakdown visual
        renderContribVisual(s, cats);

        // Category detail sections with metric breakdowns
        renderCategoryDetails(ticker, s, cats);

        modalReturnFocus = document.activeElement;
        document.getElementById('stock-modal').style.display = 'flex';
        document.body.style.overflow = 'hidden';
        const mb = document.querySelector('#stock-modal .modal-body');
        if (mb) mb.scrollTop = 0;
        const closeBtn = document.querySelector('#stock-modal .modal-close');
        if (closeBtn) closeBtn.focus({{ preventScroll: true }});
    }}

    let modalReturnFocus = null;

    function closeModal() {{
        const m = document.getElementById('stock-modal');
        if (m.style.display === 'none') return;
        m.style.display = 'none';
        document.body.style.overflow = '';
        if (modalReturnFocus && modalReturnFocus.focus && document.contains(modalReturnFocus)) {{
            modalReturnFocus.focus({{ preventScroll: true }});
        }}
        modalReturnFocus = null;
    }}

    // Jump within the drilldown, opening the target first if it is folded away.
    function goToModal(id) {{
        const el = document.getElementById(id);
        if (!el) return;
        if (el.classList.contains('collapsed')) el.classList.remove('collapsed');
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}

    function toggleSection(id) {{
        const el = document.getElementById(id);
        if (el) el.classList.toggle('collapsed');
        if (id === 'sec-universe') requestAnimationFrame(() => renderWindow(true));
    }}

    // ESC to close any open modal
    document.addEventListener('keydown', e => {{
        if (e.key === 'Escape') {{
            closeModal();
            closeMethodology();
        }}
    }});

    function renderPriceTargets(s) {{
        const container = document.getElementById('modal-price-targets');
        const price = s.price;
        const ptMean = s.pt_mean;
        const ptHigh = s.pt_high;
        const ptLow = s.pt_low;
        const nAnalysts = s.num_analysts;

        // If no price target data, hide section
        if (!ptMean && !ptHigh && !ptLow) {{
            container.innerHTML = '';
            return;
        }}

        const fmtDollar = v => v !== null && v !== undefined ? '$' + v.toFixed(2) : '—';
        const fmtPct = (target, cur) => {{
            if (!target || !cur || cur === 0) return '';
            const pct = ((target - cur) / cur * 100);
            const sign = pct >= 0 ? '+' : '';
            const cls = pct >= 0 ? 'pt-up' : 'pt-down';
            return `<span class="${{cls}}">${{sign}}${{pct.toFixed(1)}}%</span>`;
        }};

        // Compute range bar positions (if we have low, mean, high, and price)
        let rangeBarHtml = '';
        if (ptLow && ptHigh && price) {{
            // Range from lowest of (ptLow, price) to highest of (ptHigh, price)
            const rangeMin = Math.min(ptLow, price) * 0.95;
            const rangeMax = Math.max(ptHigh, price) * 1.05;
            const span = rangeMax - rangeMin;
            const pctLow = ((ptLow - rangeMin) / span * 100).toFixed(1);
            const pctHigh = ((ptHigh - rangeMin) / span * 100).toFixed(1);
            const pctPrice = ((price - rangeMin) / span * 100).toFixed(1);
            const pctMean = ptMean ? ((ptMean - rangeMin) / span * 100).toFixed(1) : null;

            rangeBarHtml = `
                <div class="pt-range-bar">
                    <div class="pt-range-track">
                        <div class="pt-range-fill" style="left:${{pctLow}}%;width:${{(pctHigh - pctLow).toFixed(1)}}%"></div>
                        <div class="pt-marker pt-marker-price" style="left:${{pctPrice}}%" title="Current: ${{fmtDollar(price)}}">
                            <div class="pt-marker-line"></div>
                            <div class="pt-marker-label">Current</div>
                        </div>
                        ${{pctMean ? `<div class="pt-marker pt-marker-mean" style="left:${{pctMean}}%" title="Avg Target: ${{fmtDollar(ptMean)}}">
                            <div class="pt-marker-line"></div>
                            <div class="pt-marker-label">Avg</div>
                        </div>` : ''}}
                    </div>
                    <div class="pt-range-labels">
                        <span style="left:${{pctLow}}%">Low ${{fmtDollar(ptLow)}}</span>
                        <span style="left:${{pctHigh}}%">High ${{fmtDollar(ptHigh)}}</span>
                    </div>
                </div>`;
        }}

        container.innerHTML = `
            <div class="pt-section">
                <div class="pt-cards">
                    <div class="pt-card">
                        <div class="pt-card-label">Current Price</div>
                        <div class="pt-card-value">${{fmtDollar(price)}}</div>
                    </div>
                    <div class="pt-card pt-card-accent">
                        <div class="pt-card-label">Avg Target</div>
                        <div class="pt-card-value">${{fmtDollar(ptMean)}} ${{fmtPct(ptMean, price)}}</div>
                    </div>
                    <div class="pt-card">
                        <div class="pt-card-label">Low Target</div>
                        <div class="pt-card-value">${{fmtDollar(ptLow)}} ${{fmtPct(ptLow, price)}}</div>
                    </div>
                    <div class="pt-card">
                        <div class="pt-card-label">High Target</div>
                        <div class="pt-card-value">${{fmtDollar(ptHigh)}} ${{fmtPct(ptHigh, price)}}</div>
                    </div>
                    ${{nAnalysts ? `<div class="pt-card"><div class="pt-card-label">Analysts</div><div class="pt-card-value">${{Math.round(nAnalysts)}}</div></div>` : ''}}
                </div>
                ${{rangeBarHtml}}
            </div>`;
    }}

    // Shared formatters for snapshot + peers
    const fmtBig = (v) => {{
        if (v === null || v === undefined) return '\u2014';
        const abs = Math.abs(v);
        const sign = v < 0 ? '-' : '';
        if (abs >= 1e12) return sign + '$' + (abs / 1e12).toFixed(2) + 'T';
        if (abs >= 1e9)  return sign + '$' + (abs / 1e9).toFixed(1) + 'B';
        if (abs >= 1e6)  return sign + '$' + (abs / 1e6).toFixed(0) + 'M';
        return sign + '$' + abs.toLocaleString();
    }};
    const fmtPctChg = (v) => {{
        if (v === null || v === undefined) return '';
        const pct = (v * 100).toFixed(1);
        const sign = v >= 0 ? '+' : '';
        const cls = v >= 0 ? 'snap-up' : 'snap-down';
        return `<span class="${{cls}}">${{sign}}${{pct}}%</span>`;
    }};
    const fmtPct2 = (v) => {{
        if (v === null || v === undefined) return '\u2014';
        return (v * 100).toFixed(1) + '%';
    }};

    function renderCompanySnapshot(s) {{
        const container = document.getElementById('modal-snapshot');
        if (!container) return;
        const f = s.financials;
        if (!f) {{ container.innerHTML = ''; return; }}

        const fmtShares = (v) => {{
            if (v === null || v === undefined) return '\u2014';
            if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B';
            if (v >= 1e6) return (v / 1e6).toFixed(0) + 'M';
            return v.toLocaleString();
        }};

        const groups = [
            {{ label: 'Size & Valuation', color: '#898781', items: [
                {{ label: 'Market Cap',      value: fmtBig(f.market_cap) }},
                {{ label: 'Enterprise Value', value: fmtBig(f.enterprise_value) }},
                {{ label: 'EPS (TTM / Fwd)', value:
                    (f.trailing_eps !== null ? '$' + f.trailing_eps.toFixed(2) : '\u2014') + ' / ' +
                    (f.forward_eps !== null ? '$' + f.forward_eps.toFixed(2) : '\u2014') }},
            ]}},
            {{ label: 'Profitability', color: '#898781', items: [
                {{ label: 'Revenue (LTM)',   value: fmtBig(f.revenue),
                   sub: f.revenue_growth_yoy !== null ? fmtPctChg(f.revenue_growth_yoy) + ' YoY' : '' }},
                {{ label: 'Net Income',      value: fmtBig(f.net_income),
                   sub: f.ni_growth_yoy !== null ? fmtPctChg(f.ni_growth_yoy) + ' YoY' : '' }},
                {{ label: 'EBITDA',          value: fmtBig(f.ebitda) }},
                {{ label: 'Gross Margin',    value: fmtPct2(f.gross_margin) }},
                {{ label: 'Net Margin',      value: fmtPct2(f.net_margin) }},
            ]}},
            {{ label: 'Cash Flow & Leverage', color: '#898781', items: [
                {{ label: 'Free Cash Flow',  value: fmtBig(f.fcf) }},
                {{ label: 'Total Debt',      value: fmtBig(f.total_debt) }},
                {{ label: 'Cash & Equiv.',   value: fmtBig(f.total_cash) }},
                {{ label: 'Net Debt',        value: fmtBig(f.net_debt) }},
            ]}},
            {{ label: 'Shareholder', color: '#898781', items: [
                {{ label: 'Dividend Yield',  value: f.dividend_yield !== null ? fmtPct2(f.dividend_yield) : 'None' }},
                {{ label: 'Payout Ratio',    value: f.payout_ratio !== null ? fmtPct2(f.payout_ratio) : '\u2014' }},
                {{ label: 'Shares Out',      value: fmtShares(f.shares_outstanding) }},
            ]}},
        ];

        // Trading group (conditional items)
        const tradingItems = [];
        if (f.avg_daily_dollar_vol !== null) tradingItems.push({{ label: 'Avg Daily $ Vol', value: fmtBig(f.avg_daily_dollar_vol) }});
        if (f.short_ratio !== null) tradingItems.push({{ label: 'Short Interest', value: f.short_ratio.toFixed(1) + ' days to cover' }});
        if (tradingItems.length > 0) groups.push({{ label: 'Trading', color: '#898781', items: tradingItems }});

        let html = '<div class="snapshot-section">';
        html += '<div class="snapshot-header">';
        html += '<span class="snapshot-hint">LTM = Last Twelve Months</span></div>';

        groups.forEach(g => {{
            html += `<div class="snapshot-group">
                <div class="snapshot-group-label" style="border-color:${{g.color}}">
                    <span style="color:${{g.color}}">${{g.label}}</span>
                </div>
                <div class="snapshot-grid">`;
            g.items.forEach(item => {{
                html += `<div class="snapshot-item">
                    <div class="snapshot-label">${{item.label}}</div>
                    <div class="snapshot-value">${{item.value}}</div>
                    ${{item.sub ? `<div class="snapshot-sub">${{item.sub}}</div>` : ''}}
                </div>`;
            }});
            html += '</div></div>';
        }});

        html += '</div>';
        container.innerHTML = html;
    }}

    function renderPeerComparison(ticker, s) {{
        const container = document.getElementById('modal-peers');
        if (!container) return;
        const self = s.self_metrics;
        if (!self) {{ container.innerHTML = ''; return; }}

        // Initialize peer state on first render for this ticker
        if (peerState.currentTicker !== ticker) {{
            peerState.currentTicker = ticker;
            peerState.defaultPeers = (s.peers || []).map(p => p.ticker);
            peerState.customPeers = [...peerState.defaultPeers];
        }}

        const fmtRG = (v) => {{
            if (v === null || v === undefined) return '\u2014';
            const p = (v * 100).toFixed(1);
            return (v >= 0 ? '+' : '') + p + '%';
        }};
        const fmtPE = (v) => v !== null && v !== undefined ? v.toFixed(1) + 'x' : '\u2014';
        const fmtPct1 = (v) => v !== null && v !== undefined ? (v * 100).toFixed(1) + '%' : '\u2014';
        const fmtDE = (v) => v !== null && v !== undefined ? v.toFixed(2) + 'x' : '\u2014';

        // No green/red verdicts on peers' figures. They used to colour a peer green when it
        // "beat" this stock on a rule of the table's own (a lower P/E, a higher dividend yield,
        // with unexplained 80%/120% cut-offs) - judgements the screener does not make: dividend
        // yield is not scored at all. The numbers stand plain, with the peer median beneath them
        // as the reference point, and this stock's row marked.
        const median = (vals) => {{
            const v = vals.filter(x => x !== null && x !== undefined && isFinite(x)).sort((a, b) => a - b);
            if (!v.length) return null;
            const m = Math.floor(v.length / 2);
            return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
        }};

        // Build rows dynamically from peerState
        const selfRow = {{ ticker: ticker, company: s.company, isSelf: true, ...self }};
        const peerRows = peerState.customPeers
            .map(t => buildPeerRow(t))
            .filter(r => r !== null);
        const allRows = [selfRow, ...peerRows];
        const isCustomized = !arraysEqual(peerState.customPeers, peerState.defaultPeers);

        let html = `<div class="peer-section">
            <div class="peer-header">
                <span class="peer-sector">${{s.sector}}</span>
            </div>
            <div class="peer-add-bar">
                <div class="peer-search-wrap">
                    <input type="text" id="peer-search-input"
                           class="peer-search-input"
                           placeholder="Add peer by ticker or name..."
                           autocomplete="off" />
                    <div id="peer-search-results" class="peer-search-results"></div>
                </div>
                ${{isCustomized ? '<button class="peer-reset-btn" onclick="resetPeers()">Reset defaults</button>' : ''}}
            </div>
            <div class="peer-table-wrap">
            <table class="peer-table">
            <thead><tr>
                <th class="peer-th-ticker">Ticker</th>
                <th>Mkt Cap</th>
                <th>P/E</th>
                <th>Rev Growth</th>
                <th>Net Margin</th>
                <th>${{s.flags && s.flags.is_bank ? 'ROE' : 'ROIC'}}</th>
                <th>D/E</th>
                <th>Div Yield</th>
                <th>FCF Yield</th>
                <th class="peer-th-action"></th>
            </tr></thead><tbody>`;

        allRows.forEach(r => {{
            const cls = r.isSelf ? 'peer-row-self' : 'peer-row';
            const useROE = s.flags && s.flags.is_bank;
            const profitMetric = useROE ? r.roe : r.roic;
            html += `<tr class="${{cls}}">
                <td class="peer-td-ticker">
                    <span class="peer-ticker">${{escapeHtml(r.ticker)}}</span>
                    ${{r.isSelf ? '<span class="peer-you">YOU</span>' : ''}}
                </td>
                <td class="peer-td-num">${{fmtBig(r.mcap)}}</td>
                <td class="peer-td-num">${{fmtPE(r.pe_ratio)}}</td>
                <td class="peer-td-num">${{fmtRG(r.rev_growth)}}</td>
                <td class="peer-td-num">${{fmtPct1(r.net_margin)}}</td>
                <td class="peer-td-num">${{fmtPct1(profitMetric)}}</td>
                <td class="peer-td-num">${{fmtDE(r.debt_equity)}}</td>
                <td class="peer-td-num">${{fmtPct1(r.div_yield)}}</td>
                <td class="peer-td-num">${{fmtPct1(r.fcf_yield)}}</td>
                <td class="peer-td-action">${{r.isSelf ? '' : `<button class="peer-remove-btn" onclick="removePeer('${{escapeHtml(r.ticker)}}')" title="Remove">&times;</button>`}}</td>
            </tr>`;
        }});

        const others = allRows.filter(r => !r.isSelf);
        if (others.length >= 2) {{
            const useROE = s.flags && s.flags.is_bank;
            const med = k => median(others.map(r => k === 'profit' ? (useROE ? r.roe : r.roic) : r[k]));
            html += `<tr class="peer-row-median">
                <td class="peer-td-ticker"><span class="peer-ticker">Peer median</span></td>
                <td class="peer-td-num">${{fmtBig(med('mcap'))}}</td>
                <td class="peer-td-num">${{fmtPE(med('pe_ratio'))}}</td>
                <td class="peer-td-num">${{fmtRG(med('rev_growth'))}}</td>
                <td class="peer-td-num">${{fmtPct1(med('net_margin'))}}</td>
                <td class="peer-td-num">${{fmtPct1(med('profit'))}}</td>
                <td class="peer-td-num">${{fmtDE(med('debt_equity'))}}</td>
                <td class="peer-td-num">${{fmtPct1(med('div_yield'))}}</td>
                <td class="peer-td-num">${{fmtPct1(med('fcf_yield'))}}</td>
                <td class="peer-td-action"></td>
            </tr>`;
        }}
        html += '</tbody></table></div>';
        html += '<p class="modal-note">The five closest ' + escapeHtml(s.sector) + ' names by market cap, as published by this run; add or remove names to compare others. These figures are context: they are reported values, not the percentiles the scores are built from (those are in The workings, ranked against the whole sector).</p></div>';
        container.innerHTML = html;

        // Attach autocomplete after rendering
        setupPeerSearch();
    }}

    function renderFlagsWarnings(s) {{
        const container = document.getElementById('modal-flags');
        if (!container) return;
        const fl = s.flags;
        if (!fl) {{ container.innerHTML = ''; return; }}

        const badges = [];

        if (s.vt && fl.vt_severity !== null && fl.vt_severity > 0) {{
            const sev = fl.vt_severity;
            const cls = sev >= 70 ? 'flag-severe' : sev >= 40 ? 'flag-warn' : 'flag-mild';
            badges.push(`<span class="flag-badge ${{cls}}">
                <span class="flag-icon">\u26a0\ufe0f</span> Value Trap
                <span class="flag-sev">${{sev.toFixed(0)}}/100</span>
            </span>`);
        }}

        if (s.gt && fl.gt_severity !== null && fl.gt_severity > 0) {{
            const sev = fl.gt_severity;
            const cls = sev >= 70 ? 'flag-severe' : sev >= 40 ? 'flag-warn' : 'flag-mild';
            badges.push(`<span class="flag-badge ${{cls}}">
                <span class="flag-icon">\u26a0\ufe0f</span> Growth Trap
                <span class="flag-sev">${{sev.toFixed(0)}}/100</span>
            </span>`);
        }}

        if (fl.beneish_flag) {{
            badges.push('<span class="flag-badge flag-severe">' +
                '<span class="flag-icon">\u26d4</span> Earnings Manipulation Risk (Beneish)</span>');
        }}

        if (fl.channel_stuffing) {{
            const div = fl.recv_rev_divergence;
            badges.push(`<span class="flag-badge flag-warn">
                <span class="flag-icon">\u26a0\ufe0f</span> Channel Stuffing Risk
                ${{div !== null ? '<span class="flag-detail">(' + (div * 100).toFixed(0) + '% divergence)</span>' : ''}}
            </span>`);
        }}

        if (fl.ev_flag) {{
            badges.push('<span class="flag-badge flag-warn">' +
                '<span class="flag-icon">\u26a0\ufe0f</span> EV Data Discrepancy</span>');
        }}

        if (fl.stale_data) {{
            const age = fl.stmt_age_days;
            badges.push(`<span class="flag-badge flag-warn">
                <span class="flag-icon">\u26a0\ufe0f</span> Stale Financials
                ${{age !== null ? '<span class="flag-detail">(' + Math.round(age) + ' days old)</span>' : ''}}
            </span>`);
        }}

        if (s.eps_mismatch) {{
            badges.push('<span class="flag-badge flag-warn">' +
                '<span class="flag-icon">\u26a0\ufe0f</span> EPS Basis Mismatch (GAAP vs Non-GAAP)</span>');
        }}

        if (fl.beta_overlap_pct !== null && fl.beta_overlap_pct < 80) {{
            badges.push(`<span class="flag-badge flag-mild">
                <span class="flag-icon">\u2139\ufe0f</span> Beta Overlap ${{fl.beta_overlap_pct.toFixed(0)}}%
                <span class="flag-detail">(< 80% required)</span>
            </span>`);
        }}

        if (fl.ltm_annualized) {{
            badges.push('<span class="flag-badge flag-mild">' +
                '<span class="flag-icon">\u2139\ufe0f</span> LTM Partially Annualized (3 of 4 quarters)</span>');
        }}

        if (fl.is_bank) {{
            badges.push('<span class="flag-badge flag-info">' +
                '<span class="flag-icon">\U0001f3e6</span> Bank-Like \u2014 uses P/B, ROE, ROA metrics</span>');
        }}

        if (fl.fin_caveat && !fl.is_bank) {{
            badges.push('<span class="flag-badge flag-info">' +
                '<span class="flag-icon">\u2139\ufe0f</span> Financial Sector \u2014 review with caution</span>');
        }}

        if (badges.length === 0) {{
            container.innerHTML = '';
            return;
        }}

        container.innerHTML = `<div class="flags-section">
            <div class="flags-badges">${{badges.join('')}}</div>
        </div>`;
    }}

    // Explain, in words, any gap between the weights used above and the
    // headline weights on the Methodology page. A reader who checks the
    // arithmetic and finds 14.9% where the docs say 13% should be told why,
    // not left to conclude the tool is wrong.
    function weightNote(ew, anyWithheld) {{
        const base = D.weights.base_factor_weights;
        const cats = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        let parts = [];

        if (base && D.weights.factor_weights) {{
            const moved = cats.filter(c =>
                Math.abs((D.weights.factor_weights[c] || 0) - (base[c] || 0)) > 0.05);
            if (moved.length) {{
                const desc = moved.map(c =>
                    `${{CAT_LABELS[c]}} ${{fmtWeight(base[c] || 0)}} &rarr; <strong>${{fmtWeight(D.weights.factor_weights[c] || 0)}}</strong>`
                ).join(', ');
                parts.push(`<strong>This run's weights differ from the published defaults.</strong> ` +
                    `The screener scales momentum with the market's volatility regime &mdash; ` +
                    `momentum is cut in turbulent markets, where momentum crashes cluster, and raised in calm ones. ` +
                    `For this run: ${{desc}}.`);
            }}
        }}

        if (anyWithheld) {{
            parts.push(`<strong>One or more categories could not be scored for this stock</strong> &mdash; ` +
                `usually missing fundamentals, or a price series the integrity check rejected. ` +
                `Rather than score it as zero, which would penalise a stock for a data gap, ` +
                `the screener drops the category and shares its weight across the rest. ` +
                `That is why the weights above do not match the defaults, and why they still total 100%.`);
        }}

        if (!parts.length) return '';
        return `<div class="contrib-weight-note">` +
            parts.map(p => `<p>${{p}}</p>`).join('') +
            `<p class="contrib-weight-note-foot">The weights shown above are the ones this stock's scores were ` +
            `actually multiplied by, so every row is an equation you can check.</p></div>`;
    }}

    function renderContribVisual(s, cats) {{
        const ew = effWeights(s);
        const base = D.weights.base_factor_weights || {{}};
        let html = '';
        let runningTotal = 0;
        let anyWithheld = false;

        cats.forEach(c => {{
            const score = s.cat_scores[c];
            const hasScore = score !== null && score !== undefined;
            if (!hasScore) anyWithheld = true;
            const contrib = s.contrib[c] || 0;
            // The weight this stock's score was actually multiplied by, so the
            // "score x weight = pts" line below is an equation the reader can
            // check rather than an approximation.
            const weight = ew[c] || 0;
            const maxContrib = weight; // max contribution = weight * (100/100)
            const fillPct = maxContrib > 0 ? (contrib / maxContrib) * 100 : 0;
            runningTotal += contrib;

            if (!hasScore) {{
                html += `<div class="contrib-row contrib-row-na">
                    <div class="contrib-label">
                        <span class="contrib-cat-dot" style="background:${{CAT_COLORS[c]}};opacity:.35"></span>
                        <span class="contrib-cat-name">${{CAT_LABELS[c]}}</span>
                        <span class="contrib-cat-weight">no data</span>
                    </div>
                    <div class="contrib-bar-area">
                        <div class="contrib-bar-annotation">
                            <span class="contrib-score-val">Not scored for this stock &mdash; its ${{fmtWeight(base[c] || 0)}} was shared out across the categories below.</span>
                        </div>
                    </div>
                    <div class="contrib-pts">0.0</div>
                </div>`;
                return;
            }}

            // Score quality label
            let qualLabel = '', qualClass = '';
            if (score !== null && score !== undefined) {{
                if (score >= 75) {{ qualLabel = 'Strong'; qualClass = 'qual-strong'; }}
                else if (score >= 50) {{ qualLabel = 'Average'; qualClass = 'qual-avg'; }}
                else if (score >= 25) {{ qualLabel = 'Weak'; qualClass = 'qual-weak'; }}
                else {{ qualLabel = 'Very Weak'; qualClass = 'qual-vweak'; }}
            }}

            html += `<div class="contrib-row contrib-row-link" onclick="openWorkings('${{c}}')" title="Show how this score is built">
                <div class="contrib-label">
                    <span class="contrib-cat-dot" style="background:${{CAT_COLORS[c]}}"></span>
                    <span class="contrib-cat-name">${{CAT_LABELS[c]}}</span>
                    <span class="contrib-cat-weight">${{fmtWeight(weight)}} weight</span>
                </div>
                <div class="contrib-bar-area">
                    <div class="contrib-bar-track">
                        <div class="contrib-bar-fill" style="width:${{fillPct.toFixed(1)}}%;background:${{CAT_COLORS[c]}}">
                            ${{fillPct > 15 ? `<span class="contrib-bar-inner-label">${{contrib.toFixed(1)}}</span>` : ''}}
                        </div>
                        ${{fillPct <= 15 ? `<span class="contrib-bar-outer-label">${{contrib.toFixed(1)}}</span>` : ''}}
                        <div class="contrib-bar-max-marker" style="left:100%" title="Max possible: ${{maxContrib.toFixed(2)}} pts"></div>
                    </div>
                    <div class="contrib-bar-annotation">
                        <span class="contrib-score-val">Score: ${{fmt(score,'score')}}/100</span>
                        <span class="contrib-qual ${{qualClass}}">${{qualLabel}}</span>
                        <span class="contrib-math">× ${{fmtWeight(weight)}} = <strong>${{contrib.toFixed(1)}} pts</strong></span>
                    </div>
                </div>
                <div class="contrib-pts">${{contrib.toFixed(1)}}<span class="contrib-pts-max">/${{maxContrib.toFixed(1)}}</span></div>
            </div>`;
        }});

        html += weightNote(ew, anyWithheld);

        document.getElementById('contrib-visual').innerHTML = html;

        // Total: category points, then the coverage discount when one applies, then
        // the composite. The chain sums on screen for every stock.
        const cov = s.cov || null;
        const disc = cov && cov.disc ? cov.disc : 0;
        const cd = (D.weights || {{}}).coverage_discount || null;
        const composite = (s.composite !== null && s.composite !== undefined) ? s.composite : runningTotal;
        const totalPct = Math.max(0, Math.min(100, composite)).toFixed(0);
        let chain = `<div class="chain-line"><span>Category points add up to</span><strong>${{runningTotal.toFixed(1)}}</strong></div>`;
        if (disc > 0) {{
            const covPct = cov.of ? (cov.n / cov.of * 100).toFixed(0) : '?';
            chain += `<div class="chain-line chain-discount"><span>Coverage discount &mdash; values for ${{cov.n}} of ${{cov.of}} applicable metrics (${{covPct}}%), below the ${{cd ? Math.round(cd.threshold * 100) : 80}}% threshold, so the composite is reduced by ${{(disc * 100).toFixed(1)}}%</span><strong>&minus;${{(runningTotal - composite).toFixed(1)}}</strong></div>`;
        }}
        document.getElementById('contrib-total').innerHTML = `
            <div class="contrib-total-bar-area">
                <div class="contrib-total-track">
                    <div class="contrib-total-fill" style="width:${{totalPct}}%"></div>
                </div>
            </div>
            ${{chain}}
            <div class="contrib-total-label">
                Composite Score: <strong>${{fmt(composite,'score')}}</strong> / 100
            </div>`;
    }}

{_js_workings()}
{_js_ux()}

    function fmtMetric(v, type) {{
        if (v === null || v === undefined) return '—';
        if (type === 'pct') return (v * 100).toFixed(1) + '%';
        // Basis points of price. 'pct' at one decimal would round most of the
        // universe to 0.0%/0.1% and manufacture display ties in a metric that
        // has none (0.0% ties measured on all 502 names).
        if (type === 'bp') return (v * 10000).toFixed(0) + ' bp';
        if (type === 'int') return Math.round(v).toString();
        if (type === 'ratio') return v.toFixed(2);
        return v.toString();
    }}

    function pctBarColor(pct, cat) {{
        if (pct === null) return '#898781';
        return CAT_COLORS[cat] || ACCENT;
    }}

    // =====================================================================
    // METHODOLOGY MODAL
    // =====================================================================
    function openMethodology() {{
        buildMethodologyToc();
        document.getElementById('methodology-modal').style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }}

    // The methodology is ~30 headings long. Give it a "contents" rail built from its own
    // second-level headings, so a reader can jump instead of scrolling for a minute.
    function buildMethodologyToc() {{
        const body = document.querySelector('#methodology-modal .methodology-body');
        if (!body || body.querySelector('.method-toc')) return;
        const heads = body.querySelectorAll('h2');
        if (heads.length < 4) return;
        let h = '<nav class="method-toc" aria-label="Methodology sections"><div class="method-toc-title">On this page</div>';
        heads.forEach(function(el, i) {{
            el.id = el.id || 'm-sec-' + i;
            h += '<a href="#' + el.id + '" data-m="' + el.id + '" onclick="return goToMethod(this.dataset.m)">' + escapeHtml(el.textContent) + '</a>';
        }});
        body.insertAdjacentHTML('afterbegin', h + '</nav>');
    }}

    function goToMethod(id) {{
        const el = document.getElementById(id);
        if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        return false;
    }}
    function closeMethodology() {{
        document.getElementById('methodology-modal').style.display = 'none';
        document.body.style.overflow = '';
    }}

    // =====================================================================
    // COLLAPSIBLE SECTIONS
    // =====================================================================
    function toggleSection(sectionId) {{
        const sec = document.getElementById(sectionId);
        if (!sec) return;
        sec.classList.toggle('collapsed');
        if (sectionId === 'sec-universe') requestAnimationFrame(() => renderWindow(true));
    }}

    // =====================================================================
    // DEFENSIBILITY & DIAGNOSTICS
    // =====================================================================

    // Magnitude in one hue. Correlation is not good or bad in itself - it only says
    // two categories overlap - so green/amber/red would assert a verdict the number
    // does not carry. Pairs past 0.7 get an outline instead (see .corr-high).
    function corrColor(v, isDiag) {{
        if (isDiag) return 'var(--bg-elevated)';
        if (v === null || v === undefined) return 'transparent';
        return 'rgba(57,135,229,' + (0.05 + Math.min(1, Math.abs(v)) * 0.5).toFixed(3) + ')';
    }}

    function renderDefensibility() {{
        const dq = D.data_quality || {{}};
        const sens = D.weight_sensitivity || [];
        const corr = D.factor_correlation;

        // --- Summary badges (always visible in header) ---
        let summaryHtml = '';
        if (sens.length > 0) {{
            const vals = sens.filter(function(s) {{ return s.avg_jaccard !== null; }}).map(function(s) {{ return s.avg_jaccard; }});
            const avgJ = vals.length > 0 ? (vals.reduce(function(a,b) {{ return a+b; }}, 0) / vals.length) : null;
            if (avgJ !== null) {{
                const jColor = avgJ >= 0.85 ? '#0ca30c' : avgJ >= 0.70 ? '#fab219' : '#e66767';
                const jLabel = avgJ >= 0.85 ? 'Robust' : avgJ >= 0.70 ? 'Moderate' : 'Sensitive';
                summaryHtml += '<span class="def-badge"><i class="def-dot" style="background:' + jColor + '"></i>Stability: ' + jLabel + ' (' + (avgJ * 100).toFixed(0) + '%)</span>';
            }}
        }}
        if (dq.eps_mismatch_count !== undefined) {{
            const mColor = dq.eps_mismatch_count === 0 ? '#0ca30c' : '#fab219';
            summaryHtml += '<span class="def-badge"><i class="def-dot" style="background:' + mColor + '"></i>EPS Flags: ' + dq.eps_mismatch_count + '</span>';
        }}
        if (dq.avg_metric_coverage !== null && dq.avg_metric_coverage !== undefined) {{
            const cov = (dq.avg_metric_coverage * 100).toFixed(0);
            const cColor = dq.avg_metric_coverage >= 0.80 ? '#0ca30c' : dq.avg_metric_coverage >= 0.60 ? '#fab219' : '#e66767';
            summaryHtml += '<span class="def-badge"><i class="def-dot" style="background:' + cColor + '"></i>Data: ' + cov + '% complete</span>';
        }}
        document.getElementById('defensibility-summary').innerHTML = summaryHtml;

        // --- Data quality KPI cards (above the two panels) ---
        let kpiHtml = '';
        const freshTs = dq.data_freshness ? new Date(dq.data_freshness) : null;
        const freshStr = freshTs && !isNaN(freshTs) ? freshTs.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }}) : '\u2014';
        kpiHtml += dqKpi('Data Freshness', freshStr, 'when the data was last fetched from Yahoo Finance', null);
        kpiHtml += dqKpi('Metric Coverage',
            dq.avg_metric_coverage !== null && dq.avg_metric_coverage !== undefined ? (dq.avg_metric_coverage * 100).toFixed(0) + '%' : '\u2014',
            'of the metrics that apply to each stock have a value, on average', dq.avg_metric_coverage >= 0.80 ? '#0ca30c' : '#fab219');
        kpiHtml += dqKpi('EPS Mismatch', dq.eps_mismatch_count !== undefined ? dq.eps_mismatch_count : '\u2014',
            'stocks have a GAAP vs. non-GAAP EPS discrepancy that may distort growth metrics',
            dq.eps_mismatch_count === 0 ? '#0ca30c' : '#fab219');
        document.getElementById('defensibility-kpis').innerHTML = kpiHtml;

        // --- Sensitivity table with visual bars ---
        if (sens.length > 0) {{
            let tHtml = '<table class="sens-table"><thead><tr>';
            tHtml += '<th>Factor</th><th>Weight</th><th style="width:45%">Stability (Jaccard Similarity)</th><th>Verdict</th>';
            tHtml += '</tr></thead><tbody>';
            sens.forEach(function(s) {{
                const aj = s.avg_jaccard;
                const jClass = aj !== null ? (aj >= 0.85 ? 'sens-cell-high' : aj >= 0.70 ? 'sens-cell-med' : 'sens-cell-low') : '';
                const verdict = aj !== null ? (aj >= 0.85 ? 'Robust' : aj >= 0.70 ? 'Moderate' : 'Sensitive') : '\u2014';
                const barPct = aj !== null ? Math.round(aj * 100) : 0;
                const barColor = aj !== null ? (aj >= 0.85 ? '#0ca30c' : aj >= 0.70 ? '#fab219' : '#e66767') : '#4a4a48';
                const pj = s.plus_jaccard !== null ? (s.plus_jaccard * 100).toFixed(0) + '%' : '\u2014';
                const mj = s.minus_jaccard !== null ? (s.minus_jaccard * 100).toFixed(0) + '%' : '\u2014';
                tHtml += '<tr>';
                tHtml += '<td style="font-weight:600">' + s.category + '</td>';
                tHtml += '<td>' + (s.original_weight !== null ? s.original_weight + '%' : '\u2014') + '</td>';
                tHtml += '<td><div class="sens-bar-track"><div class="sens-bar-fill" style="width:' + barPct + '%;background:' + barColor + '"></div></div>';
                tHtml += '<span class="sens-bar-labels"><span>+5%: ' + pj + '</span><span>-5%: ' + mj + '</span></span></td>';
                tHtml += '<td class="' + jClass + '" style="font-weight:600">' + verdict + '</td>';
                tHtml += '</tr>';
            }});
            tHtml += '</tbody></table>';
            document.getElementById('sensitivity-table').innerHTML = tHtml;
        }} else {{
            document.getElementById('sensitivity-table').innerHTML =
                '<p style="color:var(--text-muted);font-size:12px;padding:8px">No sensitivity data available for this run. Run the screener to generate.</p>';
        }}

        // --- Correlation heatmap ---
        if (corr && corr.labels && corr.matrix) {{
            const n = corr.labels.length;
            // Use abbreviated labels that are still readable
            const shortLabels = corr.labels.map(function(l) {{
                var map = {{'Valuation':'Val','Quality':'Qual','Growth':'Grow','Momentum':'Mom','Risk':'Risk','Revisions':'Rev','Size':'Size','Investment':'Inv'}};
                return map[l] || l.slice(0,4);
            }});
            let hHtml = '<div class="corr-grid" style="grid-template-columns: 50px repeat(' + n + ', 1fr)">';
            // Header row
            hHtml += '<div class="corr-label"></div>';
            shortLabels.forEach(function(l) {{ hHtml += '<div class="corr-label">' + l + '</div>'; }});
            // Data rows
            corr.matrix.forEach(function(row, i) {{
                hHtml += '<div class="corr-label" style="text-align:right;padding-right:6px">' + shortLabels[i] + '</div>';
                row.forEach(function(v, j) {{
                    const bg = corrColor(v, i === j);
                    const txt = i === j ? '\u2014' : (v !== null ? v.toFixed(2) : '');
                    const title = corr.labels[i] + ' vs ' + corr.labels[j] + ': ' + (v !== null ? v.toFixed(3) : 'N/A');
                    hHtml += '<div class="corr-cell' + (i !== j && v !== null && Math.abs(v) > 0.7 ? ' corr-high' : '') + '" style="background:' + bg + '" title="' + title + '">' + txt + '</div>';
                }});
            }});
            hHtml += '</div>';
            // Legend
            hHtml += '<div class="corr-legend">';
            hHtml += '<span class="corr-legend-item"><span class="corr-legend-swatch" style="background:rgba(57,135,229,.10)"></span>Weak (&lt;0.4): largely independent</span>';
            hHtml += '<span class="corr-legend-item"><span class="corr-legend-swatch" style="background:rgba(57,135,229,.28)"></span>Moderate (0.4&ndash;0.7)</span>';
            hHtml += '<span class="corr-legend-item"><span class="corr-legend-swatch corr-high" style="background:rgba(57,135,229,.45)"></span>Strong (&gt;0.7): overlapping, outlined</span>';
            hHtml += '</div>';

            // Auto-generated interpretation summary
            var highPairs = [];
            var modPairs = [];
            for (var i = 0; i < n; i++) {{
                for (var j = i + 1; j < n; j++) {{
                    var v = corr.matrix[i][j];
                    if (v !== null) {{
                        var abs = Math.abs(v);
                        if (abs > 0.7) highPairs.push(corr.labels[i] + ' & ' + corr.labels[j] + ' (' + v.toFixed(2) + ')');
                        else if (abs > 0.4) modPairs.push(corr.labels[i] + ' & ' + corr.labels[j] + ' (' + v.toFixed(2) + ')');
                    }}
                }}
            }}
            hHtml += '<div class="corr-summary">';
            if (highPairs.length === 0 && modPairs.length === 0) {{
                hHtml += '<strong>All 8 factors are largely independent.</strong> No pair has correlation above 0.4, meaning each factor adds unique information to the composite score.';
            }} else {{
                if (highPairs.length > 0) {{
                    hHtml += '<strong>Overlapping factors:</strong> ' + highPairs.join(', ') + '. These pairs measure similar things \u2014 the effective number of independent signals is lower than 8. ';
                }}
                if (modPairs.length > 0) {{
                    hHtml += '<strong>Moderate overlap:</strong> ' + modPairs.join(', ') + '. Some shared signal, but each still adds value. ';
                }}
                var indepCount = 8 - Math.floor(highPairs.length * 0.5 + modPairs.length * 0.2);
                if (indepCount < 8) {{
                    hHtml += 'Effective independent factors: <strong>~' + Math.max(indepCount, 4) + ' of 8</strong>.';
                }}
            }}
            hHtml += '</div>';
            document.getElementById('correlation-heatmap').innerHTML = hHtml;
        }} else {{
            document.getElementById('correlation-heatmap').innerHTML =
                '<p style="color:var(--text-muted);font-size:12px;padding:8px">No correlation data available for this run.</p>';
        }}
    }}

    function dqKpi(label, value, sub, color) {{
        return '<div class="dq-kpi-card">' +
            '<div class="kpi-label">' + label + '</div>' +
            '<div class="kpi-value"' + (color ? ' style="color:' + color + '"' : '') + '>' + value + '</div>' +
            '<div class="kpi-sub">' + sub + '</div>' +
        '</div>';
    }}

    function renderStockHistory(ticker) {{
        const wrap = document.getElementById('section-history');
        const body = document.getElementById('modal-history');
        if (!wrap || !body) return;
        const ser = H.series && H.series[ticker];
        if (!H.available || !ser) {{ wrap.style.display = 'none'; return; }}
        const pts = ser.r.map((r, i) => [i, r]).filter(p => p[1] != null);
        if (pts.length < 2) {{ wrap.style.display = 'none'; return; }}
        wrap.style.display = '';

        const d = (H.delta && H.delta[ticker]) || {{}};
        const first = pts[0], last = pts[pts.length - 1];

        // Category movement answers "what changed?" - the single most useful
        // thing for a sell decision, and the reason this section exists.
        function catTable(entry, label, date) {{
            if (!entry || entry.new || !entry.cat) return '';
            const rows = Object.entries(entry.cat)
                .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                .slice(0, 4)
                .map(([c, v]) => `<tr><td>${{CAT_LABELS[c] || c}}</td>
                    <td class="num ${{v > 0 ? 'pos' : 'neg'}}">${{v > 0 ? '+' : ''}}${{v.toFixed(1)}}</td></tr>`)
                .join('');
            if (!rows) return '';
            const dr = entry.dr;
            const drTxt = dr === 0 ? 'unchanged'
                : `${{dr > 0 ? '▲' : '▼'}}${{Math.abs(dr)}} ranks`;
            return `<div class="modal-hist-block">
                <div class="modal-hist-head">${{label}} <span class="muted">(${{escapeHtml(date)}})</span>
                    &mdash; <span class="${{dr > 0 ? 'pos' : (dr < 0 ? 'neg' : 'muted')}}">${{drTxt}}</span></div>
                <table class="mini-table"><tbody>${{rows}}</tbody></table>
            </div>`;
        }}

        const rt = (H.movers && H.movers[changedRange]
            && [].concat(H.movers[changedRange].up, H.movers[changedRange].down)
                 .some(m => m.t === ticker && m.rt));

        body.innerHTML = `
            <div class="modal-history-row">
                <span class="modal-spark">${{sparkline(ser.r, d.prev ? d.prev.dr : 0)}}</span>
                <span class="muted" style="font-size:.78rem">
                    rank ${{first[1]}} on ${{escapeHtml(H.dates[first[0]])}}
                    &rarr; ${{last[1]}} on ${{escapeHtml(H.dates[last[0]])}}
                    across ${{pts.length}} comparable runs
                </span>
                ${{rt ? '<span class="rt-badge" title="A large rank excursion that returned to base. Usually a metric dropping out and returning rather than a real change.">round-trip</span>' : ''}}
            </div>
            ${{catTable(d.prev, 'Since last run', (H.compare.prev || {{}}).date || '')}}
            ${{catTable(d.m1, 'Since ~1 month', (H.compare.m1 || {{}}).date || '')}}
            <p class="modal-note">History covers ${{H.dates.length}} runs judged comparable to each other. Score changes describe what the model saw, not what you should do.</p>
        `;
    }}

    function renderProvenance(s) {{
        const container = document.getElementById('modal-provenance');
        if (!container) return;
        if (s.metric_count === null && s.metric_count === undefined && !s.eps_mismatch) {{
            container.innerHTML = '';
            return;
        }}
        const fl = s.flags || {{}};
        let html = '<div class="provenance-section"><div class="provenance-badges">';
        if (s.metric_count !== null && s.metric_count !== undefined && s.metric_total !== null && s.metric_total !== undefined && s.metric_total > 0) {{
            const pct = (s.metric_count / s.metric_total * 100).toFixed(0);
            const cls = pct >= 80 ? 'provenance-ok' : pct >= 60 ? 'provenance-warn' : 'provenance-alert';
            html += '<span class="provenance-badge ' + cls + '">Metrics: ' + s.metric_count + '/' + s.metric_total + ' (' + pct + '%)</span>';
        }}
        if (s.data_source) {{
            const dsLabel = s.data_source === 'quarterly' ? 'LTM (Quarterly)' : s.data_source === 'annual' ? 'Annual Only' : s.data_source;
            html += '<span class="provenance-badge provenance-ok">Source: ' + dsLabel + '</span>';
        }}
        if (fl.stmt_age_days !== null && fl.stmt_age_days !== undefined) {{
            const age = Math.round(fl.stmt_age_days);
            const cls = age <= 100 ? 'provenance-ok' : age <= 200 ? 'provenance-warn' : 'provenance-alert';
            html += '<span class="provenance-badge ' + cls + '">Filing Age: ' + age + ' days</span>';
        }}
        if (s.eps_mismatch) {{
            html += '<span class="provenance-badge provenance-alert">EPS Mismatch (ratio: ' + (s.eps_ratio !== null ? s.eps_ratio : '?') + ')</span>';
        }}
        html += '</div>';
        const fmtD = function(d) {{
            const t = new Date(d + 'T00:00:00');
            return isNaN(t) ? d : t.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }});
        }};
        const a = s.asof || {{}};
        const parts = [];
        if (a.bs) parts.push('Balance sheet as of <strong>' + fmtD(a.bs) + '</strong>');
        if (a.is) parts.push('Income statement as of <strong>' + fmtD(a.is) + '</strong>');
        if (a.cf) parts.push('Cash flow as of <strong>' + fmtD(a.cf) + '</strong>');
        if (parts.length) html += '<p class="provenance-line">' + parts.join(' &middot; ') + '.</p>';
        html += '<p class="provenance-line">Income and cash-flow figures are the sum of the last four reported quarters when the quarterly source is used, and the latest annual statement otherwise. Prices are the last close at the time of the run' + (D.kpis && D.kpis.run_timestamp ? ' (' + fmtD(String(D.kpis.run_timestamp).slice(0, 10)) + ')' : '') + '.</p>';
        html += '<p class="provenance-line provenance-limit">Every figure is as reported by the company and delivered by Yahoo Finance. The screener checks that each is used consistently - the workings above rebuild every score from them - but it cannot check that they are true.</p>';
        html += '</div>';
        container.innerHTML = html;
    }}

    // INIT
    // =====================================================================
    const hasCoreData = Array.isArray(D.table_data) && D.table_data.length > 0;
    if (!hasCoreData) {{
        const root = document.querySelector('.dashboard-container');
        if (root) {{
            root.insertAdjacentHTML('afterbegin',
                '<div style="margin:12px 0;padding:12px;border:1px solid rgba(248,81,73,.35);border-radius:8px;background:rgba(248,81,73,.1);color:#ffb3ae">No screener data payload was loaded. Try a hard refresh (Ctrl+F5). If this persists, redeploy dashboard.html and dashboard_data.js together.</div>'
            );
        }}
    }} else {{
        renderKPIs();
        renderChanged();
        if (!H.available) {{
            // No comparable history yet - hide the delta column rather than
            // filling it with em-dashes on every row.
            const ut = document.getElementById('universe-table');
            if (ut) ut.classList.add('no-history');
        }}
        renderTop5();
        initHoldings();
        renderTrapChart();
        updateSectorDist();
        setupFilters();
        applyFilters();
        renderDefensibility();
        initUX();
    }}

    </script>
</body>
</html>"""


def _js_workings() -> str:
    """JS for the per-metric workings. A plain string, so braces are not doubled."""
    return r"""
    // =====================================================================
    // WORKINGS - how a category score is built, from the published numbers.
    //
    // Every figure here is arithmetic on numbers the page already carries:
    // the engine's own weight tables (D.weights.profiles), which table this stock
    // was scored with (s.wp), and its sector percentiles (s.pct). calc_trace.py
    // does the same sum in Python with no shared code, and the build refuses to
    // publish if the two disagree - so the workings below cannot drift from the
    // scores. Until 2026-10-07 this table printed the generic metric weight for
    // every stock, which was wrong for 275 of 502 (plan/calculation-transparency.md).
    // =====================================================================
    function metricLabel(m) {
        const meta = (D.metric_meta || {})[m];
        if (meta && meta.label) return meta.label;
        return m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    // The weight table this stock's `cat` score was built with, and how much of
    // it had data. A metric with no data drops out and the rest are rescaled.
    function weightProfile(cat, s) {
        const profs = (D.weights.profiles || {})[cat];
        if (!profs) return null;
        const pid = (s.wp && s.wp[cat]) || 'generic';
        const table = profs[pid] || profs.generic || {};
        let total = 0, withData = 0, weighted = 0;
        Object.keys(table).forEach(m => {
            if (table[m] > 0) {
                weighted++;
                const p = s.pct[m];
                if (p !== null && p !== undefined) { total += table[m]; withData++; }
            }
        });
        return { pid: pid, table: table, total: total, weighted: weighted, withData: withData };
    }

    // One category's metric rows: value, percentile, weight used, points.
    function categoryWorkings(cat, s) {
        const wp = weightProfile(cat, s);
        if (!wp) return null;
        const rows = [];
        const notUsed = [];
        Object.keys(wp.table).forEach(m => {
            const w = wp.table[m];
            if (!(w > 0)) { notUsed.push(m); return; }
            const p = s.pct[m];
            const has = p !== null && p !== undefined;
            const share = has && wp.total > 0 ? (w / wp.total) * 100 : null;
            rows.push({
                metric: m, raw: s.raw[m], pct: has ? p : null, configured: w,
                share: share, points: has && share !== null ? p * share / 100 : null
            });
        });
        rows.sort((a, b) => b.configured - a.configured);
        const points = rows.reduce((t, r) => t + (r.points || 0), 0);
        return { wp: wp, rows: rows, notUsed: notUsed, points: points };
    }

    let WK_CURRENT = null;

    function renderCategoryDetails(ticker, s, cats) {
        WK_CURRENT = { ticker: ticker, s: s };
        const container = document.getElementById('modal-categories');
        const ew = effWeights(s);
        const labels = D.weights.profile_labels || {};
        const nLower = Object.keys(D.metric_meta || {}).filter(m => D.metric_meta[m].dir === 'lower').length;
        let out = `<div class="pctile-convention-note">${PCTILE_CONVENTION}</div>
            <div class="wk-controls"><button type="button" class="wk-toggle-all" onclick="toggleAllWorkings(true)">Expand all</button>
            <button type="button" class="wk-toggle-all" onclick="toggleAllWorkings(false)">Collapse all</button>
            <button type="button" class="wk-toggle-all wk-download" onclick="downloadWorkings()" title="Every number behind this stock's score, as a spreadsheet">Download as CSV</button></div>`;

        cats.forEach(cat => {
            const catScore = s.cat_scores[cat];
            const contrib = s.contrib[cat];
            const scored = catScore !== null && catScore !== undefined;
            const wk = categoryWorkings(cat, s);
            const badge = scored
                ? `${fmtWeight(ew[cat])} weight → ${fmt(contrib,'score')} pts`
                : `not scored for this stock`;

            let body = '';
            if (!wk) {
                body = `<p class="wk-note">This run did not publish its weight tables, so the per-metric workings are not shown.</p>`;
            } else {
                const generic = wk.wp.pid === 'generic';
                if (!generic) {
                    body += `<p class="wk-note wk-profile"><strong>Scored with different weights.</strong> ${escapeHtml(labels[wk.wp.pid] || wk.wp.pid)}.</p>`;
                }
                if (wk.wp.withData < wk.wp.weighted && wk.wp.withData > 0) {
                    body += `<p class="wk-note">${wk.wp.withData} of the ${wk.wp.weighted} weighted metrics have data for this stock. A metric with no data drops out and the rest are scaled up to 100%, so the weights below are shares of this stock's score.</p>`;
                }
                if (!scored) {
                    body += `<p class="wk-note">No weighted metric has data for this stock, so the category is not scored and its weight is shared across the others.</p>`;
                }
                body += `<table class="wk-table"><thead><tr>
                    <th class="wk-metric">Metric</th>
                    <th class="wk-num">Value</th>
                    <th class="wk-pct" title="${PCTILE_CONVENTION}">Sector Percentile &mdash; 100 = best</th>
                    <th class="wk-num" title="Share of this category's score">Weight</th>
                    <th class="wk-num" title="Percentile x weight">Points</th></tr></thead><tbody>`;
                wk.rows.forEach(r => {
                    const meta = D.metric_meta[r.metric] || { label: r.metric, fmt: 'ratio' };
                    const has = r.pct !== null;
                    const rawStr = r.raw !== null && r.raw !== undefined ? fmtMetric(r.raw, meta.fmt) : null;
                    const tip = has
                        ? `Configured ${fmtWeight(r.configured)} of this weighting; ${fmtWeight(r.share)} of this score after rescaling to the metrics that have data.`
                        : `Configured ${fmtWeight(r.configured)}; no data for this stock, so it carries no weight here.`;
                    const hasInfo = (D.lineage || {})[r.metric];
                    body += `<tr class="metric-row${has ? '' : ' wk-nodata'}" data-metric="${r.metric}">
                        <td class="wk-metric metric-name">${hasInfo ? `<button type="button" class="wk-info" aria-expanded="false" aria-label="Show how ${escapeHtml(meta.label)} is computed" onclick="toggleMetricDetail(this)"><span aria-hidden="true">&#9656;</span></button>` : ''}${meta.label}${dirChip(meta)}</td>
                        <td class="wk-num metric-raw">${rawStr !== null ? rawStr : '<span class="metric-na">no data</span>'}</td>
                        <td class="wk-pct"><div class="metric-pct-bar-container"><div class="metric-pct-bar">${has ? `<div class="metric-pct-fill" style="width:${Math.max(1, r.pct)}%"></div>` : ''}</div>
                            <span class="metric-pct-label">${has ? r.pct.toFixed(0) : '<span class="metric-na">&mdash;</span>'}</span></div></td>
                        <td class="wk-num metric-weight" title="${escapeHtml(tip)}">${has ? fmtWeight(r.share) : '<span class="metric-na">&mdash;</span>'}</td>
                        <td class="wk-num wk-points">${has ? r.points.toFixed(1) : '<span class="metric-na">&mdash;</span>'}</td></tr>`;
                });
                body += `</tbody>`;
                if (scored) {
                    const ok = Math.abs(wk.points - catScore) < 0.06;
                    body += `<tfoot><tr class="wk-total"><td class="wk-metric">Category score</td><td></td><td></td>
                        <td class="wk-num">100%</td><td class="wk-num wk-points" title="${ok ? 'The points add up to the category score.' : 'These points do not add up to the published category score.'}">${fmt(catScore,'score')}${ok ? '' : ' !'}</td></tr></tfoot>`;
                }
                body += `</table>`;
                if (wk.notUsed.length) {
                    body += `<p class="wk-off">Not used in this weighting: ${wk.notUsed.map(metricLabel).join(', ')}.</p>`;
                }
            }

            out += `<div class="cat-detail-section collapsed" data-cat="${cat}" id="cat-detail-${cat}">
                <div class="cat-detail-header" onclick="this.parentElement.classList.toggle('collapsed')">
                    <div><h3>${CAT_LABELS[cat]}</h3><span class="cat-weight-badge">${badge}</span></div>
                    <div><span class="cat-score-badge">Score ${fmt(catScore,'score')}</span><span class="wk-chevron" aria-hidden="true">&#9662;</span></div>
                </div>
                <div class="cat-detail-body">${body}</div>
            </div>`;
        });
        container.innerHTML = out;
    }

    function toggleAllWorkings(open) {
        document.querySelectorAll('#modal-categories .cat-detail-section').forEach(el => {
            el.classList.toggle('collapsed', !open);
        });
    }


    // ---- metric detail: formula, inputs, components, who it was ranked against ----
    const PIO_SIGNALS = [
        'Net income is positive', 'Operating cash flow is positive',
        'Return on assets rose from the prior year', 'Operating cash flow exceeds net income',
        'Long-term debt relative to assets fell', 'Current ratio rose',
        'No net new shares issued', 'Gross margin rose', 'Asset turnover rose'
    ];
    const BENEISH_INDICES = [
        ['DSRI', 'Days sales in receivables', 0.920], ['GMI', 'Gross margin', 0.528],
        ['AQI', 'Asset quality', 0.404], ['SGI', 'Sales growth', 0.892],
        ['DEPI', 'Depreciation', 0.115], ['SGAI', 'SG&A expense', -0.172],
        ['LVGI', 'Leverage', -0.327], ['TATA', 'Total accruals to assets', 4.679]
    ];

    function fmtInput(v, f) {
        if (v === null || v === undefined) return '&mdash;';
        if (f === 'usd') return fmtBig(v);
        if (f === 'price') return '$' + Number(v).toFixed(2);
        if (f === 'pct') return (v * 100).toFixed(1) + '%';
        if (f === 'ratio') return Number(v).toFixed(2);
        return String(v);
    }

    // Who a stock was ranked against for one metric: its sector, or the whole universe
    // when the sector has too few values - the rule the scorer applied.
    function peerContext(m, s) {
        const mine = s.raw[m];
        if (mine === null || mine === undefined) return null;
        const meta = (D.metric_meta || {})[m] || {};
        const lower = meta.dir === 'lower';
        const st = ((D.sector_stats || {})[s.sector] || {})[m];
        const fallback = !st || st[0] < (D.sector_min_peers || 10);
        let n = 0, better = 0;
        Object.keys(D.stock_detail).forEach(t => {
            const o = D.stock_detail[t];
            if (!fallback && o.sector !== s.sector) return;
            const v = o.raw[m];
            if (v === null || v === undefined) return;
            n++;
            if (lower ? v < mine : v > mine) better++;
        });
        return { rank: better + 1, n: n, fallback: fallback, st: st, lower: lower, meta: meta };
    }

    function ordinal(n) {
        const v = n % 100;
        const suf = (v >= 11 && v <= 13) ? 'th' : ({1: 'st', 2: 'nd', 3: 'rd'}[n % 10] || 'th');
        return n + suf;
    }

    function metricDetailHtml(m, s) {
        const info = (D.lineage || {})[m];
        if (!info) return '';
        const meta = (D.metric_meta || {})[m] || { fmt: 'ratio', label: m };
        const bad = (s.inp_bad || []).indexOf(m) >= 0;
        let h = `<div class="wk-detail-body"><div class="wk-formula"><span class="wk-k">Formula</span>${escapeHtml(info.f)}</div>`;
        if (info.how) h += `<div class="wk-how">${escapeHtml(info.how)}</div>`;

        const inp = s.inp || {};
        const shown = (info.in || []).filter(x => inp[x[1]] !== undefined && inp[x[1]] !== null);
        if (shown.length) {
            h += `<div class="wk-k">Inputs, as reported</div><dl class="wk-inputs">` +
                shown.map(x => `<div><dt>${escapeHtml(x[0])}</dt><dd>${fmtInput(inp[x[1]], x[2])}</dd></div>`).join('') + `</dl>`;
        }

        // Components for the two scores that are sums of parts.
        if (m === 'piotroski_f_score' && s.pio) {
            const sig = s.pio.split('');
            h += `<div class="wk-k">The nine signals</div><ol class="wk-parts">` + sig.map((c, i) =>
                `<li class="wk-part wk-part-${c === '1' ? 'pass' : c === '0' ? 'fail' : 'na'}"><span class="wk-mark">${c === '1' ? '&#10003;' : c === '0' ? '&#10007;' : '&ndash;'}</span>${PIO_SIGNALS[i]}<span class="wk-part-val">${c === '1' ? 'pass' : c === '0' ? 'fail' : 'not testable'}</span></li>`).join('') +
                `</ol><div class="wk-sum">${sig.filter(c => c === '1').length} passed of ${sig.filter(c => c !== '-').length} testable = score ${fmtMetric(s.raw[m], 'int')}</div>`;
        }
        if (m === 'beneish_m_score' && s.bn) {
            const parts = s.bn.split('|');
            const vals = parts[0].split(',').map(Number);
            const mask = (parts[1] || '').split('');
            let total = -4.84;
            h += `<div class="wk-k">The eight indices</div><table class="wk-mini"><thead><tr><th>Index</th><th class="wk-num">Value</th><th class="wk-num">&times; weight</th><th class="wk-num">Adds</th></tr></thead><tbody>` +
                BENEISH_INDICES.map((b, i) => {
                    const add = b[2] * vals[i];
                    total += add;
                    return `<tr class="${mask[i] === '1' ? '' : 'wk-nodata'}"><td>${b[0]} <span class="wk-dim">${b[1]}${mask[i] === '1' ? '' : ' &middot; neutral default, no data'}</span></td><td class="wk-num">${vals[i].toFixed(3)}</td><td class="wk-num">${b[2].toFixed(3)}</td><td class="wk-num">${add.toFixed(3)}</td></tr>`;
                }).join('') +
                `</tbody><tfoot><tr><td colspan="3">Constant -4.840 + the eight terms</td><td class="wk-num">${total.toFixed(3)}</td></tr></tfoot></table>`;
        }

        // Does the equation hold for this stock?
        if (info.eq) {
            h += bad
                ? `<div class="wk-check wk-check-bad">These inputs do <strong>not</strong> rebuild the value shown: at least one figure the scorer used differs from the one listed. The value shown is the one that was scored.</div>`
                : (shown.length ? `<div class="wk-check wk-check-ok">&#10003; Rebuilt from the inputs above: the same ${fmtMetric(s.raw[m], meta.fmt)} that was scored.</div>` : '');
        } else if (info.k === 'series') {
            h += `<div class="wk-check">Built from a daily price or analyst history, not from a handful of figures, so no single equation is shown.</div>`;
        } else if (info.k === 'components') {
            // the component tables above are the working
        } else {
            h += `<div class="wk-check">Not rebuilt from the inputs shown: part of this calculation is a provider figure or a longer history.</div>`;
        }

        const pc = peerContext(m, s);
        const pctv = s.pct[m];
        if (pc && pctv !== null && pctv !== undefined) {
            const where = pc.fallback ? 'the whole universe (fewer than ' + (D.sector_min_peers || 10) + ' stocks in ' + escapeHtml(s.sector) + ' have a value)' : escapeHtml(s.sector);
            h += `<div class="wk-k">Who it was ranked against</div><div class="wk-peers">Ranked <strong>${ordinal(pc.rank)}</strong> of ${pc.n} in ${where}, ${pc.lower ? 'lower' : 'higher'} being better`;
            if (pc.st && !pc.fallback) {
                h += `; sector median ${fmtMetric(pc.st[2], meta.fmt)}, middle half ${fmtMetric(pc.st[1], meta.fmt)} to ${fmtMetric(pc.st[3], meta.fmt)}`;
            }
            h += `. That is the ${ordinal(Math.round(pctv))} percentile.</div>`;
        }
        if (info.cav) h += `<div class="wk-caveat"><strong>Worth knowing.</strong> ${escapeHtml(info.cav)}</div>`;
        return h + `</div>`;
    }

    function toggleMetricDetail(btn) {
        const tr = btn.closest('tr');
        if (!tr || !WK_CURRENT) return;
        const next = tr.nextElementSibling;
        if (next && next.classList.contains('wk-detail')) {
            next.remove();
            tr.classList.remove('wk-open');
            btn.setAttribute('aria-expanded', 'false');
            return;
        }
        const row = document.createElement('tr');
        row.className = 'wk-detail';
        row.innerHTML = '<td colspan="5">' + metricDetailHtml(tr.dataset.metric, WK_CURRENT.s) + '</td>';
        tr.after(row);
        tr.classList.add('wk-open');
        btn.setAttribute('aria-expanded', 'true');
    }

    // Open one category's workings and bring it into view (used by the points rows).
    function openWorkings(cat) {
        const el = document.getElementById('cat-detail-' + cat);
        if (!el) return;
        const section = document.getElementById('section-categories');
        if (section) section.classList.remove('collapsed');
        el.classList.remove('collapsed');
        el.scrollIntoView({ block: 'start', behavior: 'smooth' });
    }
"""


def _js_table() -> str:
    """JS for the rankings table. A plain string, so braces are not doubled."""
    return r"""
    // =====================================================================
    // RANKINGS TABLE - filters, sorting, and a windowed body.
    //
    // The table used to render all 502 rows (8,050 of the page's 9,923 DOM
    // nodes) inside a box that scrolled inside the page. It now scrolls with the
    // page, keeps its header stuck to the top, and only builds the rows that are
    // on screen plus a margin; spacer rows stand in for the rest so the scrollbar
    // and scroll position stay true. Row height is fixed (a CSS variable, taller
    // on phones where each row becomes a card) so the arithmetic is exact.
    // =====================================================================
    let winRange = { a: -1, b: -1 };
    let winQueued = false;
    let activeTicker = null;
    const WIN_OVERSCAN = 12;

    function rowHeight() {
        const t = document.getElementById('universe-table');
        const v = t ? parseFloat(getComputedStyle(t).getPropertyValue('--row-h')) : NaN;
        return v > 0 ? v : 40;
    }

    function filtersActive() {
        return document.getElementById('filter-sector').value !== 'all' ||
            document.getElementById('filter-vt').value !== 'all' ||
            (parseFloat(document.getElementById('filter-comp-min').value) || 0) > 0 ||
            document.getElementById('filter-search').value.trim() !== '';
    }

    function clearFilters() {
        document.getElementById('filter-sector').value = 'all';
        document.getElementById('filter-vt').value = 'all';
        document.getElementById('filter-comp-min').value = '0';
        document.getElementById('filter-search').value = '';
        applyFilters();
    }

    function setupFilters() {
        const sel = document.getElementById('filter-sector');
        D.sectors.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s; opt.textContent = s;
            sel.appendChild(opt);
        });

        // Dropdowns apply instantly; text inputs are debounced.
        sel.addEventListener('change', applyFilters);
        document.getElementById('filter-vt').addEventListener('change', applyFilters);
        document.getElementById('filter-comp-min').addEventListener('input', debounce(applyFilters, 200));
        document.getElementById('filter-search').addEventListener('input', debounce(applyFilters, 150));
        document.getElementById('sort-by').addEventListener('change', e => {
            tableState.sortCol = e.target.value;
            tableState.sortDir = (e.target.value === 'Rank') ? 'asc' : 'desc';
            sortTable(tableState.sortCol, false);
        });

        // "/" jumps to search, the way every dense data tool does.
        document.addEventListener('keydown', e => {
            const tag = (document.activeElement || {}).tagName || '';
            if (e.key === '/' && !/^(INPUT|TEXTAREA|SELECT)$/.test(tag) && !e.metaKey && !e.ctrlKey) {
                e.preventDefault();
                document.getElementById('filter-search').focus();
            }
        });

        const tb = document.getElementById('universe-tbody');
        tb.addEventListener('click', e => {
            const tr = e.target.closest('tr[data-t]');
            if (tr) openFromRow(tr.dataset.t);
        });
        tb.addEventListener('keydown', e => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const tr = e.target.closest('tr[data-t]');
            if (tr) { e.preventDefault(); openFromRow(tr.dataset.t); }
        });
        window.addEventListener('scroll', queueWindow, { passive: true });
        window.addEventListener('resize', queueWindow);
    }

    function openFromRow(ticker) {
        activeTicker = ticker;
        document.querySelectorAll('#universe-tbody tr.row-active').forEach(r => r.classList.remove('row-active'));
        const tr = document.querySelector('#universe-tbody tr[data-t="' + ticker + '"]');
        if (tr) tr.classList.add('row-active');
        openStockDetail(ticker);
    }

    function applyFilters() {
        const sector = document.getElementById('filter-sector').value;
        const vt = document.getElementById('filter-vt').value;
        const compMin = parseFloat(document.getElementById('filter-comp-min').value) || 0;
        const search = document.getElementById('filter-search').value.toLowerCase().trim();

        tableState.filtered = tableState.data.filter(row => {
            if (sector !== 'all' && row.Sector !== sector) return false;
            const isVT = row.Value_Trap_Flag;
            const isGT = row.Growth_Trap_Flag;
            if (vt === 'clean' && (isVT || isGT)) return false;
            if (vt === 'vt' && !isVT) return false;
            if (vt === 'gt' && !isGT) return false;
            if (vt === 'any' && !isVT && !isGT) return false;
            if (row.Composite !== null && row.Composite < compMin) return false;
            if (search && !row.Ticker.toLowerCase().includes(search) &&
                !(row.Company || '').toLowerCase().includes(search)) return false;
            return true;
        });

        document.getElementById('filter-clear').hidden = !filtersActive();
        sortTable(tableState.sortCol, false);
    }

    function sortTable(col, toggle = true) {
        if (toggle) {
            if (tableState.sortCol === col) {
                tableState.sortDir = tableState.sortDir === 'asc' ? 'desc' : 'asc';
            } else {
                tableState.sortCol = col;
                tableState.sortDir = (col === 'Rank' || col === 'Ticker') ? 'asc' : 'desc';
            }
        }
        // `_rank_delta` is derived from the history payload rather than being a
        // column on the row, so it needs its own accessor.
        const cellValue = row => tableState.sortCol === '_rank_delta'
            ? rankDelta(row.Ticker)
            : row[tableState.sortCol];
        tableState.filtered.sort((a, b) => {
            let av = cellValue(a), bv = cellValue(b);
            if (av === null || av === undefined) av = tableState.sortDir === 'asc' ? Infinity : -Infinity;
            if (bv === null || bv === undefined) bv = tableState.sortDir === 'asc' ? Infinity : -Infinity;
            if (typeof av === 'string') return tableState.sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
            return tableState.sortDir === 'asc' ? av - bv : bv - av;
        });
        renderUniverseTable();
    }

    const CAT_SHORT = ['Val', 'Qual', 'Grow', 'Mom', 'Risk', 'Rev', 'Size', 'Inv'];
    const CAT_KEYS = ['valuation_score', 'quality_score', 'growth_score', 'momentum_score',
                      'risk_score', 'revisions_score', 'size_score', 'investment_score'];

    // A score cell: the number stays plain ink; the cell carries a faint tint that
    // deepens with the score, so a column can be read at a glance without
    // spending colour on meaning it does not have.
    function scoreCell(v, label) {
        if (v === null || v === undefined) return '<td class="num sc sc-na" data-l="' + label + '">&mdash;</td>';
        return '<td class="num sc" data-l="' + label + '" style="--v:' + (v / 100).toFixed(3) + '">' + fmt(v, 'score') + '</td>';
    }

    function universeRowHtml(row, i) {
        const t = escapeHtml(row.Ticker);
        const rd = rankDelta(row.Ticker);
        const rdCell = rd == null
            ? '<td class="num delta-cell muted">&mdash;</td>'
            : (rd === 0
                ? '<td class="num delta-cell"></td>'
                : '<td class="num delta-cell ' + (rd > 0 ? 'pos' : 'neg') + '">' + (rd > 0 ? '&#9650;' : '&#9660;') + Math.abs(rd) + '</td>');
        const flags = (row.Value_Trap_Flag ? '<span class="flag" title="Value-trap flag">Value</span>' : '') +
                      (row.Growth_Trap_Flag ? '<span class="flag" title="Growth-trap flag">Growth</span>' : '');
        const comp = row.Composite;
        return '<tr data-t="' + t + '" tabindex="0" aria-rowindex="' + (i + 2) + '"' +
            (row.Ticker === activeTicker ? ' class="row-active"' : '') + '>' +
            '<td class="num rank">' + row.Rank + '</td>' + rdCell +
            '<td class="ticker">' + t + '</td>' +
            '<td class="company">' + escapeHtml(row.Company || '') + '</td>' +
            '<td class="sector">' + escapeHtml(row.Sector) + '</td>' +
            '<td class="num comp">' + (comp === null || comp === undefined ? '&mdash;'
                : '<span>' + fmt(comp, 'score') + '</span><i style="width:' + Math.max(0, Math.min(100, comp)).toFixed(1) + '%"></i>') + '</td>' +
            CAT_KEYS.map((k, j) => scoreCell(row[k], CAT_SHORT[j])).join('') +
            '<td class="vt-cell">' + flags + '</td></tr>';
    }

    function emptyRowsHtml(cols) {
        return '<tr class="vt-empty"><td colspan="' + cols + '"><div class="empty-state"><strong>No stocks match these filters.</strong>' +
            '<span>Try a wider sector or a lower composite minimum.</span>' +
            '<button type="button" class="filter-clear" onclick="clearFilters()">Clear filters</button></div></td></tr>';
    }

    function queueWindow() {
        if (winQueued) return;
        winQueued = true;
        requestAnimationFrame(() => { winQueued = false; renderWindow(false); });
    }

    // Build only the rows on screen (plus a margin) between two spacers.
    function renderWindow(force) {
        const table = document.getElementById('universe-table');
        const tbody = document.getElementById('universe-tbody');
        if (!table || !tbody || table.offsetParent === null) return;
        const rows = tableState.filtered;
        const total = rows.length;
        const cols = table.tHead.rows[0].cells.length;
        if (total === 0) {
            tbody.innerHTML = emptyRowsHtml(cols);
            winRange = { a: 0, b: 0 };
            return;
        }
        const H = rowHeight();
        const headH = table.tHead.offsetHeight;
        const above = Math.max(0, -(table.getBoundingClientRect().top + headH));
        const a = Math.max(0, Math.floor(above / H) - WIN_OVERSCAN);
        const b = Math.min(total, Math.floor((above + window.innerHeight) / H) + WIN_OVERSCAN + 1);
        if (!force && a === winRange.a && b === winRange.b) return;
        winRange = { a: a, b: b };

        const focused = document.activeElement && document.activeElement.closest
            ? document.activeElement.closest('#universe-tbody tr[data-t]') : null;
        const focusT = focused ? focused.dataset.t : null;

        let html = a > 0 ? '<tr class="vt-spacer" aria-hidden="true"><td colspan="' + cols + '" style="height:' + (a * H) + 'px"></td></tr>' : '';
        for (let i = a; i < b; i++) html += universeRowHtml(rows[i], i);
        if (b < total) html += '<tr class="vt-spacer" aria-hidden="true"><td colspan="' + cols + '" style="height:' + ((total - b) * H) + 'px"></td></tr>';
        tbody.innerHTML = html;
        table.setAttribute('aria-rowcount', String(total + 1));
        if (focusT) {
            const again = tbody.querySelector('tr[data-t="' + focusT + '"]');
            if (again) again.focus({ preventScroll: true });
        }
    }

    function renderUniverseTable() {
        const s = tableState;
        document.querySelectorAll('#universe-table th').forEach(th => {
            const col = th.dataset.sort;
            const on = col === s.sortCol;
            th.classList.toggle('sorted', on);
            th.setAttribute('data-dir', on ? s.sortDir : '');
            th.setAttribute('aria-sort', on ? (s.sortDir === 'asc' ? 'ascending' : 'descending') : 'none');
        });
        const sb = document.getElementById('sort-by');
        if (sb && sb.querySelector('option[value="' + s.sortCol + '"]')) sb.value = s.sortCol;

        renderWindow(true);
        document.getElementById('result-count').textContent = s.filtered.length === s.data.length
            ? s.data.length + ' stocks'
            : s.filtered.length + ' of ' + s.data.length + ' stocks';
    }
"""


def _css() -> str:
    return """
        /* ====================================================================
           DESIGN TOKENS — see plan/dashboard-design-system.md for the
           argument and sources behind every value. Change them there and
           here together, or not at all.
           Inter for UI, body and figures (tabular-nums) · JetBrains Mono
           for literal code only. Neutrals carry the UI; one accent; color
           only where it means something.
           ==================================================================== */

        :root {
            /* Tells the browser the page is dark, so native scrollbars, form
               controls and autofill render dark instead of as stark white
               chrome (the Top 5 row's scrollbar was a white bar on near-black). */
            color-scheme: dark;
            /* surfaces */
            --bg-deep: #0d0d0d;
            --bg-primary: #141413;
            --bg-card: #1a1a19;
            --bg-card-hover: #202020;
            --bg-elevated: #222221;
            --border: #262625;
            --border-bright: #343433;
            /* ink */
            --text-primary: #ffffff;
            --text-secondary: #c3c2b7;
            --text-muted: #898781;
            /* one accent: fills vs text (text step passes 4.5:1 on every surface) */
            --accent: #3987e5;
            --accent-text: #5598e7;
            --accent-glow: rgba(57,135,229,.12); /* focus ring / selected wash only */
            /* meaning colors: direction of change, caveats. Never decoration. */
            --green: #0ca30c;
            --green-dim: rgba(12,163,12,.12);
            --red: #e66767;
            --red-strong: #d03b3b;
            --red-dim: rgba(208,59,59,.12);
            --amber: #fab219;
            --amber-dim: rgba(250,178,25,.12);
            /* space, shape, elevation, motion */
            --gap: 16px;
            --radius: 8px;
            --radius-pill: 999px;
            --shadow-overlay: 0 16px 48px rgba(0,0,0,.5);
            --t-fast: 120ms;
            --t-base: 160ms;
            /* type */
            --font-heading: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
            --font-body: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
            --font-mono: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace;
        }

        /* ---- WHAT CHANGED (time dimension) ---- */
        .changed-controls {
            display: flex; align-items: baseline; gap: 14px;
            flex-wrap: wrap; margin-bottom: var(--gap);
        }
        .seg-control { display: inline-flex; border: 1px solid var(--border-bright); border-radius: 8px; overflow: hidden; flex-shrink: 0; }
        .seg-btn {
            background: transparent; border: 0; color: var(--text-secondary);
            font-family: var(--font-body); font-size: .82rem; padding: 6px 12px;
            cursor: pointer; transition: background .12s, color .12s;
        }
        .seg-btn + .seg-btn { border-left: 1px solid var(--border-bright); }
        .seg-btn:hover { background: var(--bg-card-hover); color: var(--text-primary); }
        .seg-btn.active { background: var(--accent-glow); color: var(--accent-text); }
        .changed-caption { font-size: .82rem; color: var(--text-secondary); line-height: 1.5; }
        .changed-caption strong { color: var(--text-primary); font-weight: 600; }
        .movers-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); }
        .mover-row {
            display: grid; grid-template-columns: minmax(0,1fr) auto 58px;
            grid-template-areas: "id spark delta" "meta meta meta";
            align-items: center; gap: 2px 10px;
            padding: 8px 10px; border-radius: 8px; cursor: pointer;
            border: 1px solid transparent; transition: background .12s, border-color .12s;
        }
        .mover-row:hover { background: var(--bg-card-hover); border-color: var(--border-bright); }
        .mover-id { grid-area: id; display: flex; align-items: baseline; gap: 8px; min-width: 0; }
        .mover-ticker { font-family: var(--font-body); font-weight: 600; color: var(--text-primary); font-size: .88rem; }
        .mover-name { color: var(--text-secondary); font-size: .78rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .spark { grid-area: spark; display: block; }
        .spark-empty { grid-area: spark; color: var(--text-muted); font-size: .78rem; }
        .mover-delta {
            grid-area: delta; text-align: right; font-family: var(--font-body);
            font-size: .88rem; font-weight: 600; white-space: nowrap;
        }
        .mover-arrow { font-size: .7rem; margin-right: 2px; }
        .mover-delta.pos, .delta-cell.pos { color: var(--green); }
        .mover-delta.neg, .delta-cell.neg { color: var(--red); }
        .mover-meta { grid-area: meta; font-size: .74rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .mover-driver { color: var(--text-secondary); }
        .mover-driver.muted, .delta-cell.muted { color: var(--text-muted); }
        .mover-none { padding: 10px; color: var(--text-muted); font-size: .82rem; }
        .rt-badge {
            font-size: .66rem; text-transform: uppercase; letter-spacing: .04em;
            color: var(--amber); background: var(--amber-dim);
            border: 1px solid rgba(210,153,34,.35); border-radius: 4px;
            padding: 1px 5px; cursor: help; white-space: nowrap;
        }
        .changed-footnote {
            margin-top: var(--gap); padding-top: 12px; border-top: 1px solid var(--border);
            font-size: .76rem; color: var(--text-secondary); line-height: 1.6;
        }
        .changed-footnote strong { color: var(--amber); }
        .delta-cell { font-size: .8rem; white-space: nowrap; }
        #universe-table.no-history .delta-cell,
        #universe-table.no-history #th-rank-delta { display: none; }

        /* ---- MY HOLDINGS ---- */
        .holdings-count { color: var(--text-muted); font-weight: 400; font-size: .8em; }
        .holdings-add-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
        .holdings-search-wrap { position: relative; flex: 1; max-width: 340px; }
        .holdings-clear-btn {
            background: none; border: 1px solid var(--border-bright); color: var(--text-secondary);
            border-radius: 6px; padding: 7px 12px; font-size: 12px; cursor: pointer;
            font-family: var(--font-body); transition: color .15s, border-color .15s;
        }
        .holdings-clear-btn:hover { color: var(--red); border-color: var(--red); }
        .holdings-empty { padding: 18px 4px; color: var(--text-secondary); font-size: .84rem; line-height: 1.6; }
        .holdings-empty p { margin: 0 0 6px; }
        .holdings-empty-sub { color: var(--text-muted); font-size: .78rem; }
        .holdings-fit-line {
            font-size: .8rem; color: var(--text-secondary); line-height: 1.7;
            padding: 8px 10px; margin-bottom: 12px;
            background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px;
        }
        .holdings-fit-line strong { color: var(--text-primary); }
        .holdings-concentration {
            font-size: .78rem; color: var(--text-secondary); line-height: 1.65;
            padding: 10px 12px; margin-bottom: 12px;
            background: var(--bg-elevated); border: 1px solid var(--border);
            border-left: 3px solid var(--accent); border-radius: 8px;
        }
        .holdings-conc-title {
            margin: 0 0 6px; font-size: .72rem; letter-spacing: .08em;
            text-transform: uppercase; color: var(--text-muted); font-weight: 600;
        }
        .holdings-concentration p { margin: 0 0 7px; }
        .holdings-concentration p:last-child { margin-bottom: 0; }
        .holdings-concentration strong { color: var(--text-primary); }
        .holding-card {
            border: 1px solid var(--border); border-radius: 10px;
            background: var(--bg-card); padding: 10px 12px; margin-bottom: 10px;
        }
        .holding-card:hover { border-color: var(--border-bright); }
        .holding-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
        .holding-rank {
            font-family: var(--font-body); font-size: .8rem; color: var(--text-muted);
            min-width: 44px;
        }
        .holding-ticker {
            background: none; border: none; padding: 0; cursor: pointer;
            font-family: var(--font-heading); font-weight: 600; font-size: .95rem;
            color: var(--accent);
        }
        .holding-ticker:hover { text-decoration: underline; }
        .holding-company {
            color: var(--text-secondary); font-size: .8rem;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 260px;
        }
        .holding-sector { color: var(--text-muted); font-size: .72rem; }
        .holding-flag {
            font-size: .66rem; text-transform: uppercase; letter-spacing: .04em;
            color: var(--red); border: 1px solid rgba(248,81,73,.35);
            border-radius: 4px; padding: 1px 5px; white-space: nowrap;
        }
        .holding-spacer { flex: 1; }
        .holding-delta { font-size: .74rem; white-space: nowrap; cursor: help; }
        .holding-delta-up { color: var(--green); }
        .holding-delta-down { color: var(--red); }
        .holding-delta-flat, .holding-delta-none { color: var(--text-muted); }
        .holding-composite {
            font-family: var(--font-body); font-size: .9rem; font-weight: 600;
            color: var(--text-primary); cursor: help;
        }
        .holding-remove {
            background: none; border: none; color: var(--text-muted);
            font-size: 1.05rem; line-height: 1; cursor: pointer; padding: 0 2px;
        }
        .holding-remove:hover { color: var(--red); }
        .holding-cats {
            display: grid; grid-template-columns: repeat(8, minmax(0, 1fr));
            gap: 6px; margin-top: 10px;
        }
        .holding-cat {
            border-top: 2px solid var(--border-bright); border-radius: 0 0 5px 5px;
            background: var(--bg-elevated); padding: 5px 6px; min-width: 0;
        }
        .holding-cat-name {
            display: block; font-size: .64rem; text-transform: uppercase;
            letter-spacing: .04em; color: var(--text-muted);
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .holding-cat-val {
            display: block; font-family: var(--font-body); font-size: .82rem;
            color: var(--text-primary); white-space: nowrap;
        }
        .holding-cat-nodata .holding-cat-val { font-size: .68rem; color: var(--text-muted); }
        .holding-cat-move { font-size: .66rem; margin-left: 4px; }
        .holding-cat-move-up { color: var(--green); }
        .holding-cat-move-down { color: var(--red); }
        .holding-notes { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px; }
        .holding-note {
            margin: 0 0 4px; font-size: .78rem; line-height: 1.55;
            color: var(--text-secondary);
        }
        .holding-note:last-child { margin-bottom: 0; }
        .holding-note-change_driver { color: var(--text-primary); }
        /* A caveat on the comparison, not a warning about the company. Amber
           rule and primary text so it is not skipped, but no badge, icon or red
           - input churn scatters ranks roughly symmetrically (52.4% worse off
           against a 44.6% base rate), so styling it as bad news would invent a
           direction the measurement does not have. */
        .holding-note-input_churn {
            color: var(--text-primary); border-left: 2px solid var(--amber);
            padding-left: 8px; margin-top: 6px;
        }
        /* A scheduled fact, styled like one. No colour, no badge, no urgency
           ramp as the date approaches: the evidence says announcement days are
           when attention is well spent, not that a near date is good or bad
           news, and a countdown that turns red would assert the second. */
        .holding-note-earnings { color: var(--text-primary); }
        /* Stated, not shouted. The cadence is a framing fact a reader should
           meet before the rank changes, not an alert. */
        .cadence-note {
            margin: 0 0 8px; font-size: .78rem; line-height: 1.6;
            color: var(--text-secondary);
        }
        .cadence-note strong { color: var(--text-primary); }
        .holdings-footnote {
            margin-top: var(--gap); padding-top: 12px; border-top: 1px solid var(--border);
            font-size: .76rem; color: var(--text-secondary); line-height: 1.65;
        }
        .holdings-footnote p { margin: 0 0 8px; }
        .holdings-footnote p:last-child { margin-bottom: 0; }
        .holdings-footnote strong { color: var(--amber); }
        .holdings-storage-note { color: var(--text-muted); }
        .holdings-storage-note code { font-family: var(--font-mono); font-size: .95em; }
        @media (max-width: 900px) {
            .holding-cats { grid-template-columns: repeat(4, minmax(0, 1fr)); }
            .holding-company { max-width: 150px; }
        }
        @media (max-width: 560px) {
            .holding-cats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .holding-spacer { flex-basis: 100%; }
        }
        .modal-history-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
        .modal-spark { flex-shrink: 0; }
        .modal-hist-block { margin-top: 14px; }
        .modal-hist-head { font-size: .8rem; color: var(--text-primary); margin-bottom: 6px; }
        .modal-hist-head .pos { color: var(--green); }
        .modal-hist-head .neg { color: var(--red); }
        .mini-table { width: 100%; max-width: 320px; border-collapse: collapse; font-size: .8rem; }
        .mini-table td { padding: 3px 8px 3px 0; color: var(--text-secondary); border-bottom: 1px solid var(--border); }
        .mini-table td.num { text-align: right; font-family: var(--font-body); }
        .mini-table td.pos { color: var(--green); }
        .mini-table td.neg { color: var(--red); }
        .modal-note { margin-top: 12px; font-size: .74rem; color: var(--text-muted); line-height: 1.5; }
        .muted { color: var(--text-muted); }
        @media (max-width: 780px) {
            .movers-grid { grid-template-columns: 1fr; }
        }

        /* ---- MOTION ----
           One animation on the whole page: the modal, because opening it is
           a state change the user caused. Entrance staggers, bar-grow and
           glow pulses are deleted, not restyled - a page assembling itself
           on load is decoration that costs perceived speed. */
        @keyframes modalIn {
            from { opacity: 0; transform: translateY(-8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation: none !important;
                transition: none !important;
            }
        }

        /* ---- RESET & BASE ---- */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--font-body);
            font-size: 14px;
            background: var(--bg-deep);
            color: var(--text-primary);
            line-height: 1.55;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }
        /* Figures align vertically wherever they appear. */
        .num, .kpi-value, .metric-raw, .metric-pct-label, .metric-weight,
        .mover-delta, .holding-rank, .holding-composite, .holding-cat-val,
        .top5-composite, .top5-factor-val, .contrib-pts, .modal-score-val,
        .pt-card-value, .snapshot-value, .snapshot-sub, .result-count,
        .sens-table td, .corr-cell, .peer-table tbody td, .delta-cell {
            font-variant-numeric: tabular-nums;
        }

        .dashboard-container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px var(--gap);
        }

        /* ---- HEADER: a slim bar that stays put, with the section jump links ---- */
        :root { --bar-h: 52px; }
        .dashboard-header {
            position: sticky;
            top: 0;
            z-index: 30;
            display: flex;
            align-items: center;
            gap: 20px;
            height: var(--bar-h);
            margin: 0 calc(var(--gap) * -1) var(--gap);
            padding: 0 var(--gap);
        }
        /* The bar's surface is a pseudo-element as wide as the viewport, so it spans
           the page even though the content column stops at 1440px. */
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: 0; bottom: 0; left: 50%;
            width: 100vw;
            transform: translateX(-50%);
            background: color-mix(in srgb, var(--bg-deep) 88%, transparent);
            -webkit-backdrop-filter: blur(10px);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            z-index: -1;
        }
        body { overflow-x: clip; }
        .header-left { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
        .dashboard-header h1 {
            font-family: var(--font-heading);
            font-size: 15px;
            font-weight: 600;
            letter-spacing: -.01em;
            white-space: nowrap;
        }
        .run-info {
            font-size: 12px;
            color: var(--text-muted);
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .header-nav { display: flex; gap: 2px; margin-left: auto; overflow-x: auto; scrollbar-width: none; }
        .header-nav::-webkit-scrollbar { display: none; }
        .header-nav a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 13px;
            padding: 6px 10px;
            border-radius: var(--radius);
            white-space: nowrap;
            transition: color var(--t-fast) ease-out, background var(--t-fast) ease-out;
        }
        .header-nav a:hover { color: var(--text-primary); background: var(--bg-elevated); }
        .header-nav a:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
        .section { scroll-margin-top: calc(var(--bar-h) + 12px); }
        .data-table thead th { top: var(--bar-h); }
        .badge {
            display: inline-block;
            background: var(--bg-elevated);
            border: 1px solid var(--border-bright);
            padding: 4px 14px;
            border-radius: 20px;
            font-family: var(--font-body);
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
        }
        .badge-warn {
            background: var(--red-dim);
            border-color: rgba(248,81,73,.25);
            color: var(--red);
        }

        /* ---- KPI STRIP: one quiet bar, not a row of cards ---- */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            margin-bottom: calc(var(--gap) * 1.5);
        }
        .kpi-card { padding: 16px 22px; border-right: 1px solid var(--border); min-width: 0; }
        .kpi-card:last-child { border-right: 0; }
        .kpi-label {
            font-family: var(--font-body);
            font-size: 11px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: .05em;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-family: var(--font-body);
            font-size: 24px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -.02em;
            line-height: 1.2;
        }
        .kpi-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

        /* ---- TOP 5 ---- */
        .top5-row { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: var(--gap); }
        .top5-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 14px 16px 16px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 12px;
            transition: border-color var(--t-fast) ease-out, background var(--t-fast) ease-out;
        }
        .top5-card:hover { border-color: var(--border-bright); background: var(--bg-card-hover); }
        .top5-card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        .top5-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
        .top5-rank-bar { color: var(--text-muted); font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
        .top5-sector {
            font-size: 11px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .top5-id { display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; }
        .top5-who { min-width: 0; }
        .top5-ticker {
            font-family: var(--font-heading); font-size: 20px; font-weight: 600; letter-spacing: -.01em;
            color: var(--text-primary); line-height: 1.1;
        }
        .top5-company {
            font-size: 12px; color: var(--text-secondary); margin-top: 2px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .top5-score { display: flex; flex-direction: column; align-items: flex-end; flex: none; }
        .top5-composite { font-size: 24px; font-weight: 600; letter-spacing: -.02em; color: var(--text-primary); line-height: 1.1; }
        .top5-composite-label { font-size: 10px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }
        .top5-factors { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }
        .top5-factor {
            display: grid; grid-template-columns: 1fr auto; align-items: baseline; row-gap: 3px;
            font-size: 11px;
        }
        .top5-factor-label { color: var(--text-muted); font-size: 11px; }
        .top5-factor-val { color: var(--text-secondary); font-weight: 500; font-size: 12px; }
        .top5-factor-bar { grid-column: 1 / -1; height: 3px; background: var(--bg-elevated); border-radius: 2px; overflow: hidden; }
        .top5-factor-fill { height: 100%; background: var(--accent); border-radius: 2px; opacity: .9; }

        /* ---- SECTIONS ---- */
        .section {
            margin-bottom: calc(var(--gap) * 1.5);
        }
        .section-title {
            font-family: var(--font-heading);
            font-size: 15px;
            font-weight: 600;
            letter-spacing: -.01em;
            color: var(--text-primary);
        }
        .section-title::before { display: none; }

        /* ---- COLLAPSIBLE SECTIONS ---- */
        .collapsible-section .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            user-select: none;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
            transition: border-color var(--t-fast) ease-out;
        }
        .collapsible-section .section-header:hover { border-bottom-color: var(--border-bright); }
        .section-chevron {
            width: 18px;
            height: 18px;
            color: var(--text-muted);
            flex-shrink: 0;
            transition: transform var(--t-base) ease-out;
            margin-left: 12px;
        }
        .collapsible-section .section-body {
            overflow: hidden;
            max-height: 5000px;
            opacity: 1;
            transition: max-height var(--t-base) ease-out, opacity var(--t-base) ease-out, margin var(--t-base) ease-out;
            margin-top: 16px;
        }
        .collapsible-section.collapsed .section-body {
            max-height: 0;
            opacity: 0;
            margin-top: 0;
            pointer-events: none;
        }
        .collapsible-section.collapsed .section-chevron { transform: rotate(-90deg); }

        /* ---- PROGRESSIVE DISCLOSURE: the evidence is one click away, not in the way ---- */
        .why-panel { margin-top: 16px; border-top: 1px solid var(--border); padding-top: 12px; }
        .why-panel > summary {
            cursor: pointer; list-style: none; display: inline-flex; align-items: center; gap: 6px;
            font-size: 12.5px; color: var(--text-secondary);
            transition: color var(--t-fast) ease-out;
        }
        .why-panel > summary::-webkit-details-marker { display: none; }
        .why-panel > summary::before { content: '▸'; font-size: 10px; display: inline-block; transition: transform var(--t-fast) ease-out; }
        .why-panel[open] > summary::before { transform: rotate(90deg); }
        .why-panel > summary:hover { color: var(--text-primary); }
        .why-panel > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 4px; }
        .why-panel .holdings-footnote, .why-panel .changed-footnote { margin-top: 12px; border-top: 0; padding-top: 0; }
        .holdings-footnote strong, .changed-footnote strong { color: var(--text-primary); font-weight: 600; }
        .movers-col > div:not(.expanded) .mover-row:nth-child(n+6) { display: none; }
        .show-more {
            margin: 10px 0 0; background: none; border: 1px solid var(--border-bright); color: var(--text-secondary);
            border-radius: var(--radius); padding: 6px 12px; font: inherit; font-size: 12.5px; cursor: pointer;
            transition: color var(--t-fast) ease-out, border-color var(--t-fast) ease-out;
        }
        .show-more:hover { color: var(--text-primary); border-color: var(--text-muted); }

        /* ---- CHARTS ---- */
        .chart-row {
            display: flex;
            gap: var(--gap);
            margin-bottom: var(--gap);
            flex-wrap: wrap;
        }
        .chart-container {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px 22px;
            flex: 1;
            min-width: 340px;
            transition: border-color .2s;
        }
        .chart-container:hover { border-color: var(--border-bright); }
        .chart-container canvas { max-height: 320px; }
        .chart-small { flex: 0 0 calc(33.33% - var(--gap)); min-width: 300px; }
        .chart-small canvas { max-height: 260px; }
        .sector-dist-toggle {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 0 4px;
        }
        .sector-dist-toggle-label {
            font-size: 14px;
            font-weight: 600;
        }
        .toggle-btns {
            display: flex;
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            overflow: hidden;
        }
        .toggle-btn {
            padding: 5px 14px;
            font-family: var(--font-heading);
            font-size: 11px;
            font-weight: 600;
            border: none;
            background: var(--bg-elevated);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all .15s;
            letter-spacing: .3px;
        }
        .toggle-btn:not(:last-child) { border-right: 1px solid var(--border-bright); }
        .toggle-btn.active {
            background: var(--accent);
            color: #fff;
        }
        .toggle-btn:hover:not(.active) { background: var(--bg-card-hover); color: var(--text-primary); }
        .chart-title {
            font-family: var(--font-heading);
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-secondary);
            letter-spacing: .2px;
        }
        .chart-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            gap: 8px;
        }
        .chart-title-row select {
            padding: 5px 10px;
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            font-family: var(--font-body);
            font-size: 12px;
            background: var(--bg-elevated);
            color: var(--text-primary);
            cursor: pointer;
            transition: border-color .15s;
        }
        .chart-title-row select:hover { border-color: var(--accent); }

        /* ---- RANKINGS TABLE ----
           Scrolls with the page, header stuck to the top, rows windowed (see
           _js_table). Fixed row height is part of the contract: renderWindow()
           reads --row-h to place the spacer rows. */
        #sec-universe:not(.collapsed) .section-body { max-height: none; overflow: visible; }
        .table-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            margin-bottom: var(--gap);
            overflow: clip;
        }
        .data-table {
            --row-h: 40px;
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            table-layout: fixed;
            font-size: 13px;
        }
        .data-table thead th {
            text-align: left;
            height: 40px;
            padding: 0 10px;
            border-bottom: 1px solid var(--border-bright);
            color: var(--text-muted);
            font-family: var(--font-body);
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .05em;
            white-space: nowrap;
            user-select: none;
            cursor: pointer;
            position: sticky;
            top: var(--bar-h);
            z-index: 2;
            background: var(--bg-card);
            transition: color var(--t-fast) ease-out;
        }
        .data-table thead th:hover { color: var(--text-primary); }
        .data-table thead th.sorted { color: var(--text-primary); }
        .data-table thead th.sorted::after { content: ' ↓'; color: var(--accent-text); }
        .data-table thead th.sorted[data-dir="asc"]::after { content: ' ↑'; }
        .data-table thead th:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
        .data-table th:nth-child(1) { width: 56px; }
        .data-table th:nth-child(2) { width: 56px; }
        .data-table th:nth-child(3) { width: 76px; }
        .data-table th:nth-child(5) { width: 178px; }
        .data-table th:nth-child(6) { width: 108px; }
        .data-table th:nth-child(n+7):nth-child(-n+14) { width: 62px; }
        .data-table th:nth-child(15) { width: 132px; }
        .data-table th:nth-child(n+6):nth-child(-n+14) { text-align: right; }
        .data-table tbody tr.row-active { background: var(--accent-glow); }
        .data-table tbody tr {
            height: var(--row-h);
            cursor: pointer;
            transition: background var(--t-fast) ease-out;
        }
        .data-table tbody tr:hover { background: var(--bg-card-hover); }
        .data-table tbody tr.row-active:hover { background: var(--accent-glow); }
        .data-table tbody tr:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
        .data-table tbody td {
            padding: 0 10px;
            border-bottom: 1px solid var(--border);
            font-family: var(--font-body);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .data-table tbody tr.vt-spacer, .data-table tbody tr.vt-empty { height: auto; cursor: default; }
        .data-table tbody tr.vt-spacer td { padding: 0; border: 0; }
        .data-table tbody tr.vt-spacer:hover, .data-table tbody tr.vt-empty:hover { background: none; }
        .data-table .num { text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; }
        .data-table .rank { color: var(--text-muted); }
        .data-table .ticker { font-weight: 600; color: var(--accent-text); }
        .data-table .company { color: var(--text-secondary); }
        .data-table .sector { color: var(--text-muted); font-size: 12px; }
        .data-table .delta-cell { font-size: 12px; }
        .data-table td.comp { position: relative; font-weight: 600; color: var(--text-primary); }
        .data-table td.comp i {
            position: absolute; left: 10px; bottom: 7px; height: 2px; max-width: calc(100% - 20px);
            background: var(--accent); border-radius: 1px; opacity: .85;
        }
        .data-table td.sc {
            color: var(--text-secondary);
            background: color-mix(in srgb, var(--accent) calc(var(--v) * 20%), transparent);
        }
        .data-table td.sc-na { color: var(--text-muted); }
        .data-table .vt-cell { text-align: left; }
        .flag {
            display: inline-block; margin-right: 4px; padding: 1px 7px; border-radius: 10px;
            font-size: 11px; line-height: 16px; color: var(--amber); background: var(--amber-dim);
        }
        .empty-state { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 48px 16px; color: var(--text-muted); font-size: 13px; }
        .empty-state strong { color: var(--text-primary); font-weight: 600; font-size: 14px; }

        /* ---- FILTERS ---- */
        .filters-bar {
            display: flex;
            gap: 12px 16px;
            align-items: flex-end;
            flex-wrap: wrap;
            margin-bottom: 12px;
            padding: 12px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
        }
        .filter-group { display: flex; flex-direction: column; gap: 4px; }
        .filter-group label {
            font-family: var(--font-body);
            font-size: 11px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: .05em;
        }
        .filter-group select, .filter-group input, .filter-search input {
            height: 34px;
            padding: 0 12px;
            border: 1px solid var(--border-bright);
            border-radius: var(--radius);
            font-family: var(--font-body);
            font-size: 13px;
            background: var(--bg-elevated);
            color: var(--text-primary);
            transition: border-color var(--t-fast) ease-out, box-shadow var(--t-fast) ease-out;
        }
        .filter-group select {
            appearance: none; -webkit-appearance: none; padding-right: 30px; cursor: pointer;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12' fill='none' stroke='%23898781' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='2.5 4.5 6 8 9.5 4.5'/%3E%3C/svg%3E");
            background-repeat: no-repeat; background-position: right 10px center; background-size: 12px;
        }
        .filter-group select:hover, .filter-group input:hover, .filter-search input:hover { border-color: var(--text-muted); }
        .filter-group select:focus-visible, .filter-group input:focus-visible, .filter-search input:focus-visible {
            outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .filter-group input[type="number"] { width: 88px; }
        .filter-group input::placeholder, .filter-search input::placeholder { color: var(--text-muted); }
        .filter-search { position: relative; display: flex; align-items: center; flex: 1 1 220px; max-width: 340px; }
        .filter-search svg { position: absolute; left: 11px; width: 15px; height: 15px; color: var(--text-muted); pointer-events: none; }
        .filter-search input { width: 100%; padding-left: 34px; padding-right: 30px; }
        .filter-search kbd {
            position: absolute; right: 9px; font-family: var(--font-body); font-size: 11px; color: var(--text-muted);
            border: 1px solid var(--border-bright); border-radius: 4px; padding: 0 5px; line-height: 16px; pointer-events: none;
        }
        .filter-search input:focus ~ kbd, .filter-search input:not(:placeholder-shown) ~ kbd { display: none; }
        .filter-sort-m { display: none; }
        .filter-clear {
            height: 34px; padding: 0 12px; background: none; border: 1px solid var(--border-bright);
            border-radius: var(--radius); color: var(--text-secondary); font: inherit; font-size: 13px; cursor: pointer;
            transition: color var(--t-fast) ease-out, border-color var(--t-fast) ease-out;
        }
        .filter-clear:hover { color: var(--text-primary); border-color: var(--text-muted); }
        .filter-clear[hidden] { display: none; }
        .result-count {
            font-family: var(--font-body);
            font-size: 12px;
            color: var(--text-muted);
            margin-left: auto;
            align-self: center;
            font-variant-numeric: tabular-nums;
        }

        /* ---- RANKINGS TABLE ON NARROW SCREENS ----
           Below 1180px the Sector column goes. Below 760px each row becomes a card:
           identity and composite on the first line, the eight scores on the second,
           movement and flags on the third. Same fixed-height contract, a taller row. */
        @media (max-width: 1180px) {
            .data-table th:nth-child(5), .data-table td.sector { display: none; }
        }
        @media (max-width: 760px) {
            .filter-sort-m { display: flex; }
            .filter-search { max-width: none; flex-basis: 100%; }
            .data-table { --row-h: 96px; table-layout: auto; display: block; }
            .data-table thead { display: none; }
            .data-table tbody { display: block; }
            .data-table tbody tr {
                display: grid;
                grid-template-columns: repeat(8, 1fr);
                grid-template-rows: 36px 34px 24px;
                height: var(--row-h);
                padding: 0 12px;
                border-bottom: 1px solid var(--border);
                box-sizing: border-box;
            }
            .data-table tbody tr.vt-spacer, .data-table tbody tr.vt-empty { display: block; padding: 0; }
            .data-table tbody td { border-bottom: 0; padding: 0; display: flex; align-items: center; min-width: 0; }
            .data-table tbody td.sector { display: none; }
            .data-table td.rank { grid-column: 1; grid-row: 1; justify-content: flex-start; }
            .data-table td.ticker { grid-column: 2 / span 2; grid-row: 1; }
            .data-table td.company { grid-column: 4 / span 3; grid-row: 1; font-size: 12px; }
            .data-table td.comp { grid-column: 7 / span 2; grid-row: 1; justify-content: flex-end; }
            .data-table td.comp i { left: auto; right: 0; bottom: 4px; width: 100%; }
            .data-table td.sc {
                grid-row: 2; flex-direction: column; justify-content: center; gap: 0; font-size: 12px; border-radius: 4px;
            }
            .data-table td.sc::before {
                content: attr(data-l); font-size: 9px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em;
            }
            .data-table td.delta-cell { grid-column: 1 / span 2; grid-row: 3; justify-content: flex-start; }
            .data-table td.vt-cell { grid-column: 3 / span 6; grid-row: 3; justify-content: flex-end; }
            .data-table .vt-empty td { display: block; }
        }
        @media (prefers-reduced-motion: reduce) {
            .data-table tbody tr, .data-table thead th { transition: none; }
        }

        /* ---- FOOTER ---- */
        .dashboard-footer {
            text-align: center;
            padding: 24px 16px;
            color: var(--text-muted);
            font-size: 12px;
            font-family: var(--font-body);
            border-top: 1px solid var(--border);
            margin-top: 8px;
        }
        .footer-disclaimer {
            display: inline-block;
            margin-top: 8px;
            font-size: 11px;
            opacity: 0.85;
            max-width: 900px;
            line-height: 1.5;
        }
        .footer-disclaimer a { color: var(--accent, #3987e5); }

        /* ---- MODAL ---- */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,.7);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            z-index: 1000;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 40px 20px;
            overflow-y: auto;
        }
        .modal-content {
            background: var(--bg-primary);
            border: 1px solid var(--border-bright);
            border-radius: 14px;
            width: 100%;
            max-width: 920px;
            box-shadow: var(--shadow-overlay);
            animation: modalIn var(--t-base) ease-out;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 22px 26px 18px;
            background: var(--bg-card);
            border-radius: var(--radius) var(--radius) 0 0;
            border-bottom: 1px solid var(--border);
        }
        /* "Why it ranks here" - the deterministic summary block. Given a
           left accent rule rather than a card of its own so it reads as the
           lede of the drilldown, not another panel competing with it. */
        .summary-block {
            background: var(--bg-elevated, rgba(255,255,255,0.03));
            border: 1px solid var(--border, rgba(255,255,255,0.08));
            border-left: 3px solid var(--accent, #3987e5);
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: var(--gap, 16px);
        }
        .summary-head {
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 8px;
        }
        .summary-title {
            font-family: var(--font-heading);
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-secondary, #c3c2b7);
        }
        .summary-body { margin: 0; }
        .summary-fact {
            margin: 0 0 6px 0;
            font-size: 13.5px;
            line-height: 1.6;
            color: var(--text-primary, #ffffff);
        }
        .summary-fact:last-child { margin-bottom: 0; }
        /* The opening rank line carries the headline; the closing coverage
           and flag lines are caveats and are deliberately quieter. */
        .summary-rank { font-size: 14.5px; font-weight: 600; }
        .summary-confidence, .summary-flags {
            color: var(--text-secondary, #c3c2b7);
            font-size: 12.5px;
        }
        .summary-source {
            margin-top: 12px;
            padding-top: 10px;
            border-top: 1px solid var(--border, rgba(255,255,255,0.08));
            font-size: 11.5px;
            line-height: 1.5;
            color: var(--text-muted, #898781);
        }
        @media print {
            .summary-block { break-inside: avoid; }
        }
        .about-block {
            background: var(--bg-elevated, rgba(255,255,255,0.03));
            border: 1px solid var(--border, rgba(255,255,255,0.08));
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: var(--gap, 16px);
        }
        .about-head {
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }
        .about-title {
            font-family: var(--font-heading);
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-secondary, #c3c2b7);
        }
        .about-industry {
            font-size: 12px;
            color: var(--accent, #3987e5);
        }
        .about-text {
            margin: 0;
            font-size: 13.5px;
            line-height: 1.62;
            color: var(--text-primary, #ffffff);
            white-space: pre-wrap;
        }
        .about-text.clamped {
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .about-toggle {
            margin-top: 8px;
            background: none;
            border: none;
            padding: 0;
            font: inherit;
            font-size: 12.5px;
            color: var(--accent, #3987e5);
            cursor: pointer;
        }
        .about-toggle:hover { text-decoration: underline; }
        .about-source {
            margin-top: 10px;
            font-size: 11px;
            color: var(--text-secondary, #c3c2b7);
            opacity: 0.85;
        }
        .modal-ticker {
            font-family: var(--font-heading);
            font-size: 26px;
            font-weight: 700;
            margin: 0;
            color: var(--accent);
        }
        .modal-company {
            font-size: 14px;
            color: var(--text-secondary);
            display: block;
            margin-top: 2px;
        }
        .modal-sector {
            display: inline-block;
            background: var(--bg-elevated);
            border: 1px solid var(--border-bright);
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 11px;
            margin-top: 6px;
            color: var(--text-secondary);
        }
        .modal-close {
            background: none;
            border: 1px solid var(--border-bright);
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 22px;
            cursor: pointer;
            padding: 2px 8px;
            line-height: 1;
            transition: all .15s;
        }
        .modal-close:hover {
            color: var(--text-primary);
            border-color: var(--red);
            background: var(--red-dim);
        }
        .modal-body { padding: 22px 26px 28px; }

        /* ---- THE STOCK DRILLDOWN AS A SHEET ----
           A right-hand sheet on desktop, a bottom sheet on phones. The identity
           header and the jump links stay put; only the body scrolls. */
        #stock-modal.modal-overlay {
            align-items: stretch; justify-content: flex-end; padding: 0; overflow: hidden;
            background: rgba(0,0,0,.55);
            -webkit-backdrop-filter: none; backdrop-filter: none;
        }
        #stock-modal .modal-content {
            max-width: 880px; height: 100%; max-height: 100vh;
            display: flex; flex-direction: column;
            border-radius: 0; border-width: 0 0 0 1px;
            animation: sheetIn var(--t-base) ease-out;
        }
        #stock-modal .modal-header {
            flex: none; align-items: center; gap: 20px; padding: 18px 24px 14px;
            border-radius: 0; background: var(--bg-primary); border-bottom: 0;
        }
        #stock-modal .modal-header > div:first-child { min-width: 0; flex: 1; }
        #stock-modal .modal-ticker { color: var(--text-primary); font-size: 24px; font-weight: 600; letter-spacing: -.02em; line-height: 1.15; }
        #stock-modal .modal-company { font-size: 13px; display: inline; margin: 0 8px 0 0; }
        #stock-modal .modal-sector {
            background: none; border: 0; padding: 0; margin: 0; font-size: 12px; color: var(--text-muted);
        }
        .modal-headline { display: flex; gap: 22px; flex: none; }
        .mh-item { display: flex; flex-direction: column; align-items: flex-end; line-height: 1.2; }
        .mh-item strong { font-size: 18px; font-weight: 600; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
        .mh-k { font-size: 10px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }
        .mh-of { font-size: 11px; color: var(--text-muted); }
        .modal-nav {
            flex: none; display: flex; gap: 2px; padding: 0 20px 10px;
            border-bottom: 1px solid var(--border); overflow-x: auto; scrollbar-width: none;
            background: var(--bg-primary);
        }
        .modal-nav::-webkit-scrollbar { display: none; }
        .modal-nav a {
            color: var(--text-secondary); text-decoration: none; font-size: 12.5px; padding: 5px 10px;
            border-radius: var(--radius); white-space: nowrap;
            transition: color var(--t-fast) ease-out, background var(--t-fast) ease-out;
        }
        .modal-nav a:hover { color: var(--text-primary); background: var(--bg-elevated); }
        .modal-nav a:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
        #stock-modal .modal-body { flex: 1; overflow-y: auto; overscroll-behavior: contain; padding: 22px 24px 40px; }
        #stock-modal .modal-close {
            flex: none; width: 34px; height: 34px; padding: 0; display: grid; place-items: center;
            font-size: 20px; border-radius: var(--radius);
        }
        #stock-modal .modal-close:hover { border-color: var(--border-bright); background: var(--bg-elevated); color: var(--text-primary); }
        #stock-modal .modal-close:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        @keyframes sheetIn { from { opacity: 0; transform: translateX(24px); } to { opacity: 1; transform: none; } }
        @keyframes sheetUp { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: none; } }
        @media (max-width: 760px) {
            #stock-modal.modal-overlay { align-items: flex-end; }
            #stock-modal .modal-content {
                max-width: none; height: 94vh; border-radius: 16px 16px 0 0; border-width: 1px 0 0 0;
                animation: sheetUp var(--t-base) ease-out;
            }
            #stock-modal .modal-header { padding: 14px 16px 10px; gap: 12px; flex-wrap: wrap; }
            #stock-modal .modal-header > div:first-child { flex: 1 1 60%; }
            .modal-headline { order: 3; flex: 1 1 100%; gap: 20px; }
            .mh-item { align-items: flex-start; }
            .modal-nav { padding: 0 12px 8px; }
            #stock-modal .modal-body { padding: 16px 16px 32px; }
            .modal-score-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        /* "Why it ranks here": a headline, then quiet groups. */
        .summary-block {
            background: none; border: 0; border-radius: 0; padding: 0;
            margin-bottom: 28px; scroll-margin-top: 12px;
        }
        .summary-head { margin-bottom: 10px; }
        .summary-title {
            font-family: var(--font-body); font-size: 11px; font-weight: 500; letter-spacing: .05em; color: var(--text-muted);
        }
        .summary-rank { font-size: 16px; font-weight: 600; line-height: 1.5; letter-spacing: -.01em; margin-bottom: 14px; }
        .summary-group-label {
            font-size: 11px; font-weight: 500; letter-spacing: .05em; text-transform: uppercase; color: var(--text-muted);
            margin: 16px 0 6px; padding-top: 12px; border-top: 1px solid var(--border);
        }
        .summary-fact { font-size: 13.5px; color: var(--text-secondary); margin-bottom: 6px; }
        .summary-confidence, .summary-flags { color: var(--text-muted); font-size: 13px; }
        .summary-source { border-top: 0; padding-top: 0; margin-top: 16px; }
        .contrib-cat-dot { display: none; }
        .contrib-cat-weight { margin-left: 0; }
        .qual-strong, .qual-avg, .qual-weak, .qual-vweak { background: var(--bg-elevated); color: var(--text-muted); }
        .about-block, .modal-chart-section {
            background: none; border: 0; border-top: 1px solid var(--border); border-radius: 0; padding: 18px 0;
        }

        /* ---- COLLAPSIBLE BLOCKS INSIDE THE DRILLDOWN: hairlines, not boxes ---- */
        .collapsible { margin-bottom: 0; border: 0; border-top: 1px solid var(--border); border-radius: 0; }
        .collapsible-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 14px 0; cursor: pointer; user-select: none; background: none;
            transition: color var(--t-fast) ease-out;
        }
        .collapsible-header:hover { background: none; }
        .collapsible-header:hover span:first-child { color: var(--accent-text); }
        .collapsible-header span:first-child {
            font-family: var(--font-heading); font-size: 14px; font-weight: 600; color: var(--text-primary);
        }
        .collapsible-chevron { font-size: 10px; color: var(--text-muted); transition: transform var(--t-fast) ease-out; }
        .collapsible.collapsed .collapsible-chevron { transform: rotate(-90deg); }
        .collapsible-body { padding: 0 0 20px; background: none; }
        .collapsible.collapsed .collapsible-body { display: none; }
        .collapsible { scroll-margin-top: 12px; }

        .modal-score-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 24px;
            scroll-margin-top: 12px;
        }
        .modal-score-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 12px 12px 10px;
            text-align: left;
            transition: border-color var(--t-fast) ease-out;
        }
        .modal-score-card:hover { border-color: var(--border-bright); }
        .modal-score-card.composite {
            grid-column: 1 / -1;
            display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
            border-color: var(--border-bright);
            padding: 14px 16px;
        }
        .modal-score-card.composite .modal-score-val { font-size: 32px; }
        .modal-score-label {
            font-family: var(--font-body);
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: .05em;
            color: var(--text-muted);
            margin-bottom: 2px;
        }
        .modal-score-card.composite .modal-score-label { margin: 0; }
        .modal-score-val {
            font-family: var(--font-body);
            font-size: 22px;
            font-weight: 600;
            letter-spacing: -.02em;
            color: var(--text-primary);
            line-height: 1.2;
        }
        .modal-score-sub {
            font-family: var(--font-body);
            font-size: 11.5px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .modal-score-card.composite .modal-score-sub { margin: 0; }
        .modal-chart-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .modal-chart-section h3 {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .modal-chart-desc {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 18px;
        }

        /* ---- REFRESH BUTTON ---- */
        .methodology-btn {
            background: var(--bg-elevated);
            color: var(--text-primary);
            border: 1px solid var(--border-bright);
            padding: 6px 16px;
            border-radius: var(--radius-pill);
            font-family: var(--font-heading);
            font-size: 12.5px;
            font-weight: 500;
            cursor: pointer;
            transition: border-color var(--t-base) ease-out, color var(--t-base) ease-out;
        }
        .methodology-btn:hover {
            border-color: var(--accent);
            color: var(--accent-text);
        }
        .methodology-content {
            max-width: 880px;
        }
        .methodology-body {
            font-family: var(--font-body);
            font-size: 14px;
            line-height: 1.75;
            color: var(--text-primary);
            max-height: 70vh;
            overflow-y: auto;
            padding-right: 8px;
        }

        /* --- HEADINGS --- */
        .methodology-body h1 {
            font-family: var(--font-heading);
            font-size: 24px;
            font-weight: 700;
            margin: 36px 0 16px;
            color: var(--accent);
            padding-left: 16px;
            border-left: 4px solid var(--accent);
        }
        .methodology-body h1:first-child { margin-top: 0; }
        .methodology-body h2 {
            font-family: var(--font-heading);
            font-size: 17px;
            font-weight: 600;
            margin: 32px 0 14px;
            color: var(--text-primary);
            padding: 8px 14px;
            background: rgba(88,166,255,.06);
            border-left: 3px solid var(--accent);
            border-radius: 0 8px 8px 0;
        }
        .methodology-body h3 {
            font-family: var(--font-heading);
            font-size: 15px;
            font-weight: 600;
            margin: 24px 0 10px;
            color: var(--text-primary);
            padding-left: 12px;
            border-left: 2px solid rgba(88,166,255,.4);
        }

        /* --- TEXT --- */
        .methodology-body p {
            margin: 0 0 14px;
            color: var(--text-secondary);
        }
        .methodology-body strong {
            color: var(--text-primary);
            font-weight: 600;
        }
        .methodology-body em {
            color: var(--text-secondary);
            font-style: italic;
        }
        .methodology-body p > strong:first-child > em {
            display: inline-block;
            color: var(--accent);
            font-style: italic;
            font-weight: 400;
        }

        /* --- LISTS --- */
        .methodology-body ul, .methodology-body ol {
            margin: 0 0 16px 24px;
            color: var(--text-secondary);
        }
        .methodology-body li {
            margin-bottom: 6px;
            padding-left: 4px;
        }
        .methodology-body ol > li {
            margin-bottom: 8px;
        }
        .methodology-body ol > li::marker {
            font-family: var(--font-heading);
            font-weight: 700;
            color: var(--accent);
            font-size: 15px;
        }

        /* --- TABLES --- */
        .methodology-body table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 14px 0 20px;
            font-size: 13px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .methodology-body thead th {
            background: var(--bg-elevated);
            font-family: var(--font-heading);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .5px;
            color: var(--text-secondary);
            padding: 10px 14px;
            text-align: left;
            border-bottom: 2px solid var(--border-bright);
        }
        .methodology-body tbody td {
            padding: 9px 14px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border);
        }
        .methodology-body tbody tr:last-child td { border-bottom: none; }
        .methodology-body tbody tr:nth-child(even) td {
            background: rgba(88,166,255,.02);
        }
        .methodology-body tbody tr:hover td {
            background: var(--bg-card-hover);
        }
        .methodology-body tbody td:first-child {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* --- CODE --- */
        .methodology-body code {
            font-family: var(--font-mono);
            font-size: 12px;
            background: rgba(88,166,255,.08);
            padding: 2px 7px;
            border-radius: 4px;
            color: var(--accent);
        }
        .methodology-body pre {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 0 8px 8px 0;
            padding: 16px 20px;
            margin: 14px 0 18px;
            overflow-x: auto;
        }
        .methodology-body pre code {
            background: none;
            padding: 0;
            font-size: 12px;
            color: var(--text-primary);
            line-height: 1.6;
        }

        /* --- DIVIDERS --- */
        .methodology-body hr {
            border: none;
            height: 1px;
            background: var(--border);
            margin: 32px 0;
        }

        /* --- LINKS --- */
        .methodology-body a {
            color: var(--accent);
            text-decoration: none;
            border-bottom: 1px dotted rgba(88,166,255,.3);
            transition: border-color .15s;
        }
        .methodology-body a:hover {
            border-bottom-color: var(--accent);
            text-decoration: none;
        }

        /* --- BLOCKQUOTES --- */
        .methodology-body blockquote {
            border-left: 3px solid var(--accent);
            margin: 16px 0;
            padding: 12px 18px;
            background: rgba(88,166,255,.04);
            border-radius: 0 8px 8px 0;
            color: var(--text-secondary);
            font-style: italic;
        }
        .methodology-body blockquote p { margin-bottom: 6px; }
        .methodology-body blockquote p:last-child { margin-bottom: 0; }

        /* ---- ANALYST PRICE TARGETS ---- */
        .pt-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .pt-section h3 {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 14px;
        }
        .pt-cards {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .pt-card {
            flex: 1;
            min-width: 100px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            text-align: center;
        }
        .pt-card-accent {
            border-color: var(--accent);
        }
        .pt-card-label {
            font-family: var(--font-heading);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .6px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .pt-card-value {
            font-family: var(--font-body);
            font-size: 18px;
            font-weight: 700;
        }
        .pt-up {
            color: var(--green);
            font-size: 13px;
            font-weight: 600;
            margin-left: 4px;
        }
        .pt-down {
            color: var(--red);
            font-size: 13px;
            font-weight: 600;
            margin-left: 4px;
        }
        .pt-range-bar {
            position: relative;
            margin-top: 4px;
            padding-bottom: 22px;
        }
        .pt-range-track {
            height: 8px;
            background: var(--bg-elevated);
            border-radius: 4px;
            position: relative;
            overflow: visible;
            border: 1px solid var(--border);
        }
        .pt-range-fill {
            position: absolute;
            top: 0; bottom: 0;
            background: var(--accent);
            border-radius: 4px;
            opacity: 0.35;
        }
        .pt-marker {
            position: absolute;
            top: -6px;
            transform: translateX(-50%);
            z-index: 2;
        }
        .pt-marker-line {
            width: 2px;
            height: 20px;
            margin: 0 auto;
            border-radius: 1px;
        }
        .pt-marker-price .pt-marker-line { background: var(--text-primary); }
        .pt-marker-mean .pt-marker-line { background: var(--accent); }
        .pt-marker-label {
            font-family: var(--font-body);
            font-size: 9px;
            font-weight: 600;
            text-align: center;
            margin-top: 2px;
            color: var(--text-secondary);
            white-space: nowrap;
        }
        .pt-range-labels {
            position: relative;
            height: 16px;
            margin-top: 4px;
        }
        .pt-range-labels span {
            position: absolute;
            transform: translateX(-50%);
            font-family: var(--font-body);
            font-size: 10px;
            color: var(--text-muted);
            white-space: nowrap;
        }

        /* ---- CONTRIBUTION BREAKDOWN ---- */
        .contrib-row {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        .contrib-row:last-child { border-bottom: none; }
        .contrib-label {
            flex: 0 0 150px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .contrib-cat-dot {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }
        .contrib-cat-name {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
        }
        .contrib-cat-weight {
            font-family: var(--font-body);
            font-size: 10px;
            color: var(--text-muted);
            margin-left: 16px;
        }
        .contrib-bar-area { flex: 1; min-width: 0; }
        .contrib-bar-track {
            height: 28px;
            background: var(--bg-elevated);
            border-radius: 6px;
            position: relative;
            overflow: visible;
            border: 1px solid var(--border);
        }
        .contrib-bar-fill {
            height: 100%;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            transition: width .6s cubic-bezier(.25,.46,.45,.94);
            min-width: 2px;
        }
        .contrib-bar-inner-label {
            font-family: var(--font-body);
            font-size: 11px;
            font-weight: 700;
            color: #fff;
            text-shadow: 0 1px 2px rgba(0,0,0,.3);
        }
        .contrib-bar-outer-label {
            position: absolute;
            left: calc(2px);
            top: 50%;
            transform: translateY(-50%);
            font-family: var(--font-body);
            font-size: 11px;
            font-weight: 700;
            color: var(--text-secondary);
            margin-left: 4px;
        }
        .contrib-bar-max-marker {
            position: absolute;
            top: -4px; bottom: -4px;
            width: 2px;
            background: var(--text-muted);
            border-radius: 1px;
        }
        .contrib-bar-annotation {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-top: 4px;
            font-size: 11px;
            color: var(--text-muted);
            flex-wrap: wrap;
        }
        .contrib-score-val {
            font-family: var(--font-body);
            font-variant-numeric: tabular-nums;
        }
        .contrib-math {
            font-family: var(--font-body);
            font-variant-numeric: tabular-nums;
        }
        .contrib-qual {
            font-family: var(--font-heading);
            font-size: 9px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 10px;
            text-transform: uppercase;
            letter-spacing: .5px;
        }
        .qual-strong { background: var(--green-dim); color: var(--green); }
        .qual-avg { background: var(--amber-dim); color: var(--amber); }
        .qual-weak { background: rgba(255,140,0,.15); color: #ff8c00; }
        .qual-vweak { background: var(--red-dim); color: var(--red); }
        .contrib-pts {
            flex: 0 0 60px;
            text-align: right;
            font-family: var(--font-body);
            font-size: 18px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            line-height: 28px;
        }
        .contrib-pts-max {
            font-size: 12px;
            font-weight: 400;
            color: var(--text-muted);
        }
        .contrib-total-row {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border-bright);
        }
        .contrib-total-bar-area { margin-bottom: 8px; }
        .contrib-total-track {
            height: 8px;
            background: var(--bg-elevated);
            border-radius: 4px;
            overflow: hidden;
        }
        .contrib-total-fill {
            height: 100%;
            border-radius: 4px;
            background: var(--accent);
            transition: width var(--t-base) ease-out;
        }
        .contrib-total-label {
            font-family: var(--font-body);
            font-size: 14px;
            color: var(--text-secondary);
            text-align: right;
        }
        .contrib-total-label strong {
            font-size: 18px;
            color: var(--text-primary);
        }
        /* A category with no score for this stock: shown, not hidden, so the
           reader can see why the remaining weights add to more than the
           published defaults. */
        .contrib-row-na .contrib-cat-name { color: var(--text-muted); }
        .contrib-row-na .contrib-score-val { font-style: italic; }
        .contrib-row-na .contrib-pts { color: var(--text-muted); }
        .contrib-weight-note {
            margin-top: 14px;
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            background: var(--bg-card);
            font-size: 12.5px;
            line-height: 1.55;
            color: var(--text-secondary);
        }
        .contrib-weight-note p { margin: 0 0 8px; }
        .contrib-weight-note p:last-child { margin-bottom: 0; }
        .contrib-weight-note strong { color: var(--text-primary); }
        .contrib-weight-note-foot {
            color: var(--text-muted);
            font-size: 11.5px;
        }

        /* ---- CATEGORY DETAIL ---- */
        .cat-detail-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: border-color .2s;
        }
        .cat-detail-section:hover { border-color: var(--border-bright); }
        .cat-detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        .cat-detail-header h3 {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            margin: 0;
        }
        .cat-detail-header .cat-score-badge {
            font-family: var(--font-body);
            background: var(--bg-elevated);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        .cat-detail-header .cat-weight-badge {
            font-family: var(--font-body);
            font-size: 11px;
            color: var(--text-muted);
            margin-left: 8px;
        }
        .cat-detail-header .cat-contrib-badge {
            font-size: 12px;
            color: var(--text-secondary);
        }
        .cat-detail-body { margin-top: 12px; }
        .metric-row {
            display: flex;
            align-items: center;
            padding: 7px 0;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
            transition: background .1s;
        }
        .metric-row:hover { background: var(--bg-card-hover); margin: 0 -8px; padding: 7px 8px; border-radius: 4px; }
        .metric-row:last-child { border-bottom: none; }
        .metric-name {
            flex: 0 0 160px;
            color: var(--text-secondary);
            font-family: var(--font-body);
        }
        /* Direction marker: says which way is good for THIS metric, because the
           percentile beside it is direction-adjusted and therefore cannot be
           read off the raw value. */
        .metric-dir {
            font-size: 10px;
            font-family: var(--font-body);
            letter-spacing: .3px;
            padding: 1px 5px;
            border-radius: 8px;
            white-space: nowrap;
            cursor: help;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
        }
        .metric-dir-lower  { color: var(--amber); }
        .metric-dir-higher { color: var(--text-muted); }
        .pctile-convention-note {
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.5;
            margin: 0 0 8px 0;
        }
        .metric-raw {
            flex: 0 0 100px;
            text-align: right;
            font-family: var(--font-body);
            font-variant-numeric: tabular-nums;
            font-size: 12px;
            color: var(--text-primary);
        }
        .metric-pct-bar-container {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-left: 16px;
        }
        .metric-pct-bar {
            flex: 1;
            height: 10px;
            background: var(--bg-elevated);
            border-radius: 5px;
            overflow: hidden;
        }
        .metric-pct-fill {
            height: 100%;
            border-radius: 5px;
            transition: width .4s ease;
        }
        .metric-pct-label {
            flex: 0 0 50px;
            text-align: right;
            font-family: var(--font-body);
            font-weight: 600;
            font-size: 11px;
            color: var(--text-secondary);
        }
        .metric-weight {
            flex: 0 0 50px;
            text-align: right;
            font-family: var(--font-body);
            font-size: 11px;
            color: var(--text-muted);
        }
        .metric-na { color: var(--text-muted); font-style: italic; }

        /* ---- WORKINGS: how a category score is built ---- */
        .wk-controls { display: flex; gap: 8px; justify-content: flex-end; margin: 0 0 10px; }
        .wk-toggle-all {
            background: none; border: 1px solid var(--border); color: var(--text-secondary);
            border-radius: var(--radius); padding: 4px 10px; font-size: 12px; cursor: pointer;
            font-family: inherit; transition: border-color var(--t-fast) ease-out, color var(--t-fast) ease-out;
        }
        .wk-toggle-all:hover { border-color: var(--border-bright); color: var(--text-primary); }
        .cat-detail-section.collapsed .cat-detail-body { display: none; }
        .cat-detail-header .wk-chevron {
            display: inline-block; margin-left: 10px; color: var(--text-muted);
            transition: transform var(--t-fast) ease-out;
        }
        .cat-detail-section.collapsed .wk-chevron { transform: rotate(-90deg); }
        .wk-note {
            font-size: 12.5px; line-height: 1.55; color: var(--text-secondary);
            margin: 0 0 10px; padding: 8px 12px; background: var(--bg-elevated);
            border-radius: var(--radius); border-left: 2px solid var(--border-bright);
        }
        .wk-note strong { color: var(--text-primary); font-weight: 600; }
        .wk-profile { border-left-color: var(--accent); }
        .wk-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .wk-table th {
            text-align: left; font-weight: 500; font-size: 11px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: .04em; padding: 6px 8px;
            border-bottom: 1px solid var(--border-bright);
        }
        .wk-table th.wk-num { text-align: right; }
        .wk-table td { padding: 8px; border-bottom: 1px solid var(--border); vertical-align: middle; }
        .wk-table tr.metric-row { display: table-row; border-bottom: none; }
        .wk-table tr.metric-row:hover { background: var(--bg-card-hover); margin: 0; padding: 0; }
        .wk-table .wk-metric { color: var(--text-secondary); }
        .wk-table .wk-num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
        .wk-table .wk-pct { width: 34%; }
        .wk-table .metric-pct-bar-container { margin-left: 0; }
        .wk-table .wk-points { color: var(--text-primary); font-weight: 600; }
        .wk-table tr.wk-nodata td { color: var(--text-muted); }
        .wk-table tfoot td { border-bottom: none; border-top: 1px solid var(--border-bright); color: var(--text-primary); font-weight: 600; }
        .metric-pct-fill { background: var(--accent); }
        .wk-off { font-size: 12px; color: var(--text-muted); margin: 10px 0 0; line-height: 1.5; }
        .wk-info {
            background: none; border: 0; color: var(--text-muted); cursor: pointer;
            width: 22px; height: 22px; margin: 0 4px 0 -6px; padding: 0; border-radius: 4px;
            font-size: 11px; line-height: 1; vertical-align: middle;
            transition: color var(--t-fast) ease-out, background var(--t-fast) ease-out;
        }
        .wk-info:hover { color: var(--text-primary); background: var(--bg-elevated); }
        .wk-info span { display: inline-block; transition: transform var(--t-fast) ease-out; }
        .wk-info[aria-expanded="true"] span { transform: rotate(90deg); }
        tr.wk-open > td { border-bottom-color: transparent; }
        tr.wk-detail > td { padding: 0 8px 12px; background: var(--bg-card); }
        .wk-detail-body {
            background: var(--bg-elevated); border-radius: var(--radius); padding: 12px 14px;
            font-size: 12.5px; line-height: 1.55; color: var(--text-secondary);
        }
        .wk-detail-body > * + * { margin-top: 10px; }
        .wk-k { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); font-weight: 500; margin-right: 8px; }
        .wk-formula { color: var(--text-primary); }
        .wk-formula .wk-k { display: block; margin-bottom: 2px; }
        .wk-inputs { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 6px 16px; margin: 4px 0 0; }
        .wk-inputs div { display: flex; justify-content: space-between; gap: 10px; border-bottom: 1px solid var(--border); padding: 3px 0; }
        .wk-inputs dt { color: var(--text-muted); }
        .wk-inputs dd { margin: 0; color: var(--text-primary); font-variant-numeric: tabular-nums; white-space: nowrap; }
        .wk-parts { list-style: none; margin: 4px 0 0; padding: 0; }
        .wk-part { display: flex; gap: 8px; align-items: baseline; padding: 3px 0; border-bottom: 1px solid var(--border); }
        .wk-part .wk-mark { width: 14px; flex: none; text-align: center; }
        .wk-part .wk-part-val { margin-left: auto; color: var(--text-muted); }
        .wk-part-pass .wk-mark { color: var(--green); }
        .wk-part-fail .wk-mark { color: var(--red); }
        .wk-part-na { color: var(--text-muted); }
        .wk-sum { color: var(--text-primary); font-weight: 600; margin-top: 6px; }
        .wk-mini { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 4px; }
        .wk-mini th { text-align: left; font-weight: 500; color: var(--text-muted); padding: 3px 4px; }
        .wk-mini th.wk-num { text-align: right; }
        .wk-mini td { padding: 3px 4px; border-bottom: 1px solid var(--border); }
        .wk-mini tfoot td { border-bottom: none; color: var(--text-primary); font-weight: 600; }
        .wk-dim { color: var(--text-muted); }
        .wk-check { color: var(--text-muted); }
        .wk-check-ok { color: var(--text-secondary); }
        .wk-check-bad { color: var(--amber); }
        .wk-caveat { border-left: 2px solid var(--amber); padding-left: 10px; color: var(--text-secondary); }
        .wk-caveat strong { color: var(--text-primary); font-weight: 600; }
        .contrib-row-link { cursor: pointer; transition: background var(--t-fast) ease-out; }
        .contrib-row-link:hover { background: var(--bg-card-hover); }
        .chain-line {
            display: flex; justify-content: space-between; gap: 16px; font-size: 13px;
            color: var(--text-secondary); padding: 6px 0; font-variant-numeric: tabular-nums;
        }
        .chain-line strong { color: var(--text-primary); white-space: nowrap; }
        .chain-discount { border-left: 2px solid var(--amber); padding-left: 10px; }
        @media (max-width: 600px) {
            .wk-table .wk-pct { width: auto; }
            .wk-table .metric-pct-bar { display: none; }
            .wk-table th, .wk-table td { padding: 8px 4px; }
            .wk-table th.wk-pct { font-size: 0; }
            .wk-table th.wk-pct::after { content: "Pctile"; font-size: 11px; }
        }
        .ticker-link { cursor: pointer; }
        .ticker-link:hover { text-decoration: underline; text-underline-offset: 2px; }

        /* ---- RESPONSIVE ---- */
        @media (max-width: 768px) {
            .kpi-row { grid-template-columns: repeat(2, 1fr); }
            .chart-row { flex-direction: column; }
            .chart-container { min-width: 100%; }
            .chart-small { flex: 1 1 100%; min-width: 100%; }
            .filters-bar { flex-direction: column; }
        }

        @media (max-width: 1100px) {
            .top5-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        }
        @media (max-width: 760px) {
            .dashboard-header { position: static; height: auto; padding: 12px var(--gap); flex-wrap: wrap; gap: 8px 12px; }
            .header-left { flex-wrap: wrap; flex: 1 1 100%; gap: 2px 12px; }
            .header-nav { order: 3; flex: 1 1 100%; margin-left: -10px; }
            .header-right { margin-left: auto; order: 2; }
            .data-table thead th { top: 0; }
            .kpi-row { grid-template-columns: repeat(2, 1fr); }
            .kpi-card { padding: 14px 16px; }
            .kpi-card:nth-child(2) { border-right: 0; }
            .kpi-card:nth-child(-n+2) { border-bottom: 1px solid var(--border); }
            .top5-row { grid-template-columns: 1fr; }
        }

        /* ---- ANALYTICS & DIAGNOSTICS: one hue, status only where it means something ---- */
        .chart-note { font-size: 12.5px; line-height: 1.55; color: var(--text-muted); margin: 8px 0 14px; max-width: 70ch; }
        .sector-matrix { overflow-x: auto; }
        .sm-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
        .sm-table th {
            font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted);
            padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-bright); white-space: nowrap;
        }
        .sm-table th.num, .sm-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
        .sm-table td { padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
        .sm-sector { color: var(--text-primary); }
        .sm-table td.sm { color: var(--text-secondary); background: color-mix(in srgb, var(--accent) calc(var(--t) * 34%), transparent); }
        .sm-table td.sm-na, .sm-table td.sm-n { color: var(--text-muted); }
        .corr-cell.corr-high { box-shadow: inset 0 0 0 1px var(--amber); }
        .corr-legend-swatch.corr-high { box-shadow: inset 0 0 0 1px var(--amber); }
        .def-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; vertical-align: 1px; }
        .defensibility-summary .def-badge { color: var(--text-secondary); }

        /* The overrides below sit after the rules they replace (the cascade is the contract). */
        .contrib-cat-dot { display: none; }
        .contrib-cat-weight { margin-left: 0; }
        .contrib-qual, .qual-strong, .qual-avg, .qual-weak, .qual-vweak {
            background: var(--bg-elevated); color: var(--text-muted);
        }
        .provenance-badge.provenance-ok { background: var(--bg-elevated); color: var(--text-secondary); border-color: var(--border); }
        .provenance-line { font-size: 12.5px; line-height: 1.55; color: var(--text-secondary); margin: 10px 0 0; }
        .provenance-line strong { color: var(--text-primary); font-weight: 600; }
        .provenance-limit { color: var(--text-muted); }

        /* ---- METHODOLOGY: a reading surface ----
           A readable measure (about 72 characters), a contents rail, quiet headings. */
        .methodology-content { max-width: 1060px; }
        .methodology-body {
            display: grid; grid-template-columns: 210px minmax(0, 1fr); column-gap: 44px;
            max-height: 78vh; padding-right: 0; scroll-padding-top: 8px;
        }
        .methodology-body > * { grid-column: 2; min-width: 0; }
        .methodology-body > .method-toc {
            grid-column: 1; grid-row: 1 / span 400; position: sticky; top: 0; align-self: start;
            max-height: 74vh; overflow-y: auto; padding-right: 8px; scrollbar-width: thin;
        }
        .method-toc-title { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-bottom: 8px; }
        .method-toc a {
            display: block; color: var(--text-muted); font-size: 12.5px; line-height: 1.4; padding: 4px 0; text-decoration: none;
            transition: color var(--t-fast) ease-out;
        }
        .methodology-body .method-toc a { border-bottom: 0; }
        .method-toc a:hover { color: var(--text-primary); }
        .methodology-body p, .methodology-body ul, .methodology-body ol, .methodology-body blockquote,
        .methodology-body h1, .methodology-body h2, .methodology-body h3 { max-width: 72ch; }
        .methodology-body h1 {
            color: var(--text-primary); border-left: 0; padding-left: 0; font-weight: 600; letter-spacing: -.02em; font-size: 26px;
        }
        .methodology-body h2 {
            background: none; border-left: 0; border-radius: 0; padding: 28px 0 0; margin: 40px 0 14px;
            border-top: 1px solid var(--border); font-size: 18px; letter-spacing: -.01em;
        }
        .methodology-body h2:first-of-type { border-top: 0; padding-top: 0; margin-top: 28px; }
        .methodology-body h3 { border-left: 0; padding-left: 0; margin: 28px 0 8px; font-size: 15px; }
        @media (max-width: 900px) {
            .methodology-body { display: block; max-height: 80vh; }
            .methodology-body > .method-toc { display: none; }
        }

        /* ---- FEEL ----
           Hover feedback in a dense list should be instant: a 120ms fade on every row
           reads as lag, and costs a transition on every row. Controls keep theirs. */
        .data-table tbody tr, .mover-row { transition: none; }
        button, select, input, textarea { font-family: inherit; }

        /* ---- PRINT ---- */
        /* ---- Defensibility & Diagnostics Section ---- */
        .section-desc {
            font-size: 13px; color: var(--text-secondary); line-height: 1.7;
            margin: 4px 0 20px 0; max-width: 860px;
        }
        .section-desc em { color: var(--text-primary); font-style: italic; }
        .chart-desc {
            font-size: 12px; color: var(--text-muted); line-height: 1.6;
            margin: 0 0 16px 0;
        }
        .chart-desc strong { color: var(--text-secondary); }
        .defensibility-section { }
        .defensibility-header-left {
            display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
        }
        .defensibility-summary {
            display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
        }
        .defensibility-summary .def-badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 12px; border-radius: 12px; font-size: 11px;
            font-family: var(--font-body); font-weight: 500;
            background: var(--bg-elevated); border: 1px solid var(--border);
        }
        .defensibility-section .section-body { padding-top: 20px; }
        .defensibility-row {
            display: flex; gap: 20px; margin-top: 20px;
        }
        @media (max-width: 900px) {
            .defensibility-row { flex-direction: column; }
        }
        .defensibility-kpis {
            display: flex; gap: 14px; flex-wrap: wrap;
        }
        .dq-kpi-card {
            background: var(--bg-elevated); border-radius: 10px; padding: 16px 20px;
            min-width: 160px; flex: 1; border: 1px solid var(--border);
        }
        .dq-kpi-card .kpi-label {
            font-size: 11px; color: var(--text-secondary); text-transform: uppercase;
            letter-spacing: .5px; margin-bottom: 4px;
        }
        .dq-kpi-card .kpi-value {
            font-size: 24px; font-weight: 700; font-family: var(--font-body);
            margin: 6px 0;
        }
        .dq-kpi-card .kpi-sub {
            font-size: 11px; color: var(--text-muted); line-height: 1.5;
        }

        /* Sensitivity table */
        .sens-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .sens-table th {
            text-align: left; padding: 10px 10px 8px; font-size: 10px; text-transform: uppercase;
            letter-spacing: .5px; color: var(--text-secondary); border-bottom: 1px solid var(--border-bright);
            font-family: var(--font-heading);
        }
        .sens-table td {
            padding: 10px 10px; font-family: var(--font-body);
            border-bottom: 1px solid var(--border); vertical-align: middle;
        }
        .sens-cell-high, .sens-cell-med, .sens-cell-low { color: var(--text-secondary); }
        .dq-kpi-card .kpi-value { color: var(--text-primary) !important; }
        .sens-bar-fill { background: var(--accent) !important; opacity: .85; }
        .sens-bar-track {
            height: 10px; background: var(--bg-card); border-radius: 5px;
            overflow: hidden; margin-bottom: 6px; border: 1px solid var(--border);
        }
        .sens-bar-fill {
            height: 100%; border-radius: 5px; transition: width .3s ease;
        }
        .sens-bar-labels {
            display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted);
        }

        /* Correlation heatmap grid */
        .corr-grid {
            display: grid; gap: 3px; width: 100%;
        }
        .corr-cell {
            aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
            font-family: var(--font-body); font-size: 11px;
            border-radius: 5px; cursor: default; color: var(--text-primary);
            min-width: 0; min-height: 36px;
        }
        .corr-label {
            display: flex; align-items: center; justify-content: center;
            font-family: var(--font-heading); font-size: 10px;
            text-transform: uppercase; letter-spacing: .3px; color: var(--text-secondary);
            font-weight: 600; min-height: 36px;
        }
        .corr-legend {
            display: flex; gap: 18px; margin-top: 14px; flex-wrap: wrap; padding-top: 10px;
            border-top: 1px solid var(--border);
        }
        .corr-legend-item {
            display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-muted);
        }
        .corr-legend-swatch {
            width: 14px; height: 14px; border-radius: 3px; display: inline-block;
        }
        .corr-summary {
            margin-top: 14px; padding: 12px 14px; background: var(--bg-card);
            border-radius: 8px; border: 1px solid var(--border);
            font-size: 12px; color: var(--text-secondary); line-height: 1.6;
        }
        .corr-summary strong { color: var(--text-primary); }

        /* Provenance badges in stock modal */
        .provenance-section { margin-bottom: 16px; }
        .provenance-section h3 { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: .5px; }
        .provenance-badges { display: flex; gap: 8px; flex-wrap: wrap; }
        .provenance-badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 3px 10px; border-radius: 12px; font-size: 11px;
            font-family: var(--font-body); font-weight: 500;
        }
        .provenance-ok    { background: var(--green-dim); color: var(--green); }
        .provenance-warn  { background: var(--amber-dim); color: var(--amber); }
        .provenance-alert { background: var(--red-dim); color: var(--red); }

        /* ---- COMPANY SNAPSHOT (grouped) ---- */
        .snapshot-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .snapshot-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 16px;
        }
        .snapshot-header h3 {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            margin: 0;
            color: var(--text-primary);
        }
        .snapshot-hint {
            font-size: 11px;
            color: var(--text-muted);
            font-style: italic;
        }
        .snapshot-group {
            margin-bottom: 14px;
        }
        .snapshot-group:last-child { margin-bottom: 0; }
        .snapshot-group-label {
            font-family: var(--font-heading);
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .6px;
            padding: 0 0 6px 10px;
            margin-bottom: 8px;
            border-left: 3px solid;
        }
        .snapshot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(135px, 1fr));
            gap: 8px;
        }
        .snapshot-item {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 9px 12px;
            transition: border-color .15s;
        }
        .snapshot-item:hover { border-color: var(--border-bright); }
        .snapshot-label {
            font-family: var(--font-heading);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .4px;
            color: var(--text-secondary);
            margin-bottom: 2px;
        }
        .snapshot-value {
            font-family: var(--font-body);
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .snapshot-sub {
            font-family: var(--font-body);
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 1px;
        }
        .snap-up { color: var(--green); }
        .snap-down { color: var(--red); }

        /* ---- SECTOR PEER COMPARISON ---- */
        .peer-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .peer-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 14px;
        }
        .peer-header h3 {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            margin: 0;
            color: var(--text-primary);
        }
        .peer-sector {
            font-size: 11px;
            color: var(--text-muted);
            font-style: italic;
        }
        .peer-table-wrap { overflow-x: auto; }
        .peer-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .peer-table thead th {
            text-align: right;
            padding: 6px 10px 8px;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .4px;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-bright);
            font-family: var(--font-heading);
            font-weight: 600;
            white-space: nowrap;
        }
        .peer-th-ticker { text-align: left !important; }
        .peer-table tbody td {
            padding: 8px 10px;
            font-family: var(--font-body);
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }
        .peer-td-num { text-align: right; white-space: nowrap; }
        .peer-td-ticker { text-align: left; white-space: nowrap; }
        .peer-ticker {
            font-family: var(--font-heading);
            font-weight: 600;
            color: var(--text-primary);
        }
        .peer-you {
            font-size: 9px;
            font-family: var(--font-heading);
            font-weight: 700;
            color: var(--accent);
            background: rgba(88,166,255,.12);
            padding: 1px 5px;
            border-radius: 3px;
            margin-left: 6px;
            letter-spacing: .5px;
        }
        .peer-row-self {
            background: rgba(88,166,255,.06);
            border-left: 3px solid var(--accent);
        }
        .peer-row-self td { font-weight: 600; color: var(--text-primary); }
        .peer-row td { color: var(--text-secondary); }
        .peer-row:hover { background: var(--bg-card-hover); }

        /* ---- PEER CUSTOM SELECTION ---- */
        .peer-add-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
        }
        .peer-search-wrap {
            position: relative;
            flex: 1;
            max-width: 320px;
        }
        .peer-search-input {
            width: 100%;
            padding: 7px 12px;
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            font-family: var(--font-body);
            font-size: 12px;
            background: var(--bg-elevated);
            color: var(--text-primary);
            transition: border-color .15s;
            box-sizing: border-box;
        }
        .peer-search-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .peer-search-input::placeholder { color: var(--text-muted); }
        .peer-search-results {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            z-index: 1000;
            background: var(--bg-card);
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            margin-top: 4px;
            max-height: 260px;
            overflow-y: auto;
            box-shadow: 0 8px 24px rgba(0,0,0,.4);
        }
        .peer-search-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 12px;
            border-bottom: 1px solid var(--border);
            transition: background .1s;
        }
        .peer-search-item:last-child { border-bottom: none; }
        .peer-search-item:hover { background: var(--bg-card-hover); }
        .peer-search-ticker {
            font-family: var(--font-heading);
            font-weight: 600;
            color: var(--accent);
            min-width: 48px;
        }
        .peer-search-company {
            flex: 1;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .peer-search-score {
            font-family: var(--font-body);
            font-size: 11px;
            color: var(--text-secondary);
            min-width: 24px;
            text-align: right;
        }
        .peer-search-empty {
            padding: 12px;
            color: var(--text-muted);
            font-size: 12px;
            text-align: center;
        }
        .peer-reset-btn {
            background: none;
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            color: var(--text-secondary);
            font-family: var(--font-body);
            font-size: 11px;
            padding: 6px 12px;
            cursor: pointer;
            white-space: nowrap;
            transition: color .15s, border-color .15s;
        }
        .peer-reset-btn:hover {
            color: var(--accent);
            border-color: var(--accent);
        }
        .peer-th-action { width: 32px; }
        .peer-td-action {
            text-align: center;
            width: 32px;
            padding: 4px !important;
        }
        .peer-remove-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 16px;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 4px;
            line-height: 1;
            transition: color .15s, background .15s;
        }
        .peer-remove-btn:hover {
            color: var(--red);
            background: var(--red-dim);
        }

        /* ---- FLAGS & WARNINGS ---- */
        .flags-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .flags-section h3 {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            margin: 0 0 12px 0;
            color: var(--text-primary);
        }
        .flags-badges { display: flex; flex-wrap: wrap; gap: 8px; }
        .flag-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 12px;
            border-radius: 14px;
            font-size: 12px;
            font-family: var(--font-body);
            font-weight: 500;
            line-height: 1.3;
        }
        .flag-icon { font-size: 13px; }
        .flag-sev {
            font-family: var(--font-body);
            font-weight: 700;
            margin-left: 2px;
        }
        .flag-detail {
            font-family: var(--font-body);
            font-size: 10px;
            opacity: 0.8;
        }
        .flag-severe {
            background: var(--red-dim);
            color: var(--red);
            border: 1px solid rgba(208,59,59,.35);
        }
        .flag-warn {
            background: var(--amber-dim);
            color: var(--amber);
            border: 1px solid rgba(250,178,25,.3);
        }
        .flag-mild {
            background: var(--accent-glow);
            color: var(--accent-text);
            border: 1px solid rgba(57,135,229,.3);
        }
        .flag-info {
            background: rgba(137,135,129,.1);
            color: var(--text-secondary);
            border: 1px solid rgba(137,135,129,.2);
        }

        @media print {
            body { background: #fff; color: #000; }
            :root {
                --bg-primary: #fff; --bg-card: #fff; --bg-elevated: #f5f5f5;
                --text-primary: #000; --text-secondary: #555; --border: #ddd;
            }
            .dashboard-container { max-width: none; }
            .filters-bar { display: none; }
            .modal-overlay { display: none !important; }
            .methodology-btn, .header-nav { display: none; }
            .collapsible-section.collapsed .section-body { max-height: none; opacity: 1; pointer-events: auto; }
            .kpi-card, .chart-container, .table-section { box-shadow: none; border: 1px solid #ddd; }
        }
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _js_ux() -> str:
    """JS for the navigation layer: search palette, deep links, stepping, compare.

    A plain string, so braces are not doubled. It wraps ``openStockDetail`` and
    ``closeModal`` rather than editing them, so everything that already opens the
    drilldown (table rows, Top 5 cards, movers, holdings, peers) gets deep links and
    stepping for free.
    """
    return r"""
    // =====================================================================
    // NAVIGATION LAYER (owner-run UI pass 2, 2026-10-07)
    //
    // Four things a reader of a 502-stock tool expects and this page lacked:
    //   - reach any stock from anywhere without scrolling to the table (Ctrl/Cmd+K);
    //   - a link that opens a given stock, so a club can say "look at this one";
    //   - step to the next stock without closing the drilldown (J / K);
    //   - put two to four stocks side by side, with the composite gap between them
    //     taken apart into the categories that make it.
    // None of it computes a score. It moves between, and lines up, the numbers the
    // payload already publishes.
    // =====================================================================
    const UX = { current: null, recent: [], compare: [], pal: { items: [], sel: 0 } };
    const BASE_TITLE = document.title || 'Multi-Factor Screener';
    const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
    const COMPARE_MAX = 4;
    const UX_CATS = ['valuation', 'quality', 'growth', 'momentum', 'risk', 'revisions', 'size', 'investment'];

    // Per-viewer conveniences only (recent stocks, the comparison, a dismissed guide):
    // every read and write is guarded, and the page works the same without storage.
    const uxStore = {
        get(k, d) { try { const v = window.localStorage.getItem('screener_ux_' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
        set(k, v) { try { window.localStorage.setItem('screener_ux_' + k, JSON.stringify(v)); } catch (e) { /* blocked */ } },
    };

    function byRank() {
        if (!byRank.cache) byRank.cache = D.table_data.slice().sort((a, b) => (a.Rank || 1e9) - (b.Rank || 1e9));
        return byRank.cache;
    }

    function sheetOpen() {
        const m = document.getElementById('stock-modal');
        return !!m && m.style.display !== 'none';
    }

    let toastTimer = null;
    function toast(msg) {
        const el = document.getElementById('toast');
        if (!el) return;
        el.textContent = msg;
        el.classList.add('show');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => el.classList.remove('show'), 1800);
    }

    // ---- deep links --------------------------------------------------------
    // #stock=TICKER opens that stock. Opening from a closed sheet pushes one history
    // entry, so the phone's back gesture closes the sheet instead of leaving the site;
    // stepping between stocks replaces it, so back never walks through every stock.
    function stockFromHash() {
        const m = /^#stock=([A-Za-z0-9.\-]{1,12})$/.exec(location.hash || '');
        if (!m) return null;
        const t = decodeURIComponent(m[1]).toUpperCase();
        return D.stock_detail[t] ? t : null;
    }
    function baseUrl() { return location.pathname + location.search; }

    const _baseOpenStockDetail = openStockDetail;
    openStockDetail = function(ticker, opts) {
        opts = opts || {};
        if (!D.stock_detail[ticker]) return;
        const wasOpen = sheetOpen();
        const keepFocus = modalReturnFocus;
        _baseOpenStockDetail(ticker);
        if (wasOpen) modalReturnFocus = keepFocus;  // still return to the row that opened the first one
        UX.current = ticker;
        if (!opts.fromHistory) {
            try {
                const url = baseUrl() + '#stock=' + encodeURIComponent(ticker);
                if (wasOpen || opts.replace) history.replaceState({ sheet: ticker }, '', url);
                else history.pushState({ sheet: ticker }, '', url);
            } catch (e) { /* file:// in some browsers */ }
        }
        UX.recent = [ticker].concat(UX.recent.filter(t => t !== ticker)).slice(0, 6);
        uxStore.set('recent', UX.recent);
        document.title = ticker + ' · ' + BASE_TITLE;
        if (typeof activeTicker !== 'undefined') activeTicker = ticker;
        document.querySelectorAll('#universe-tbody tr.row-active').forEach(r => r.classList.toggle('row-active', r.dataset.t === ticker));
        const nh = document.getElementById('mnav-history');
        const sh = document.getElementById('section-history');
        if (nh && sh) nh.hidden = sh.style.display === 'none';
        updateSheetTools();
        requestAnimationFrame(modalSpy);  // after paint: it reads layout
    };

    const _baseCloseModal = closeModal;
    closeModal = function(opts) {
        opts = opts || {};
        if (!sheetOpen()) return;
        _baseCloseModal();
        UX.current = null;
        document.title = BASE_TITLE;
        if (!opts.fromHistory && stockFromHash()) {
            try { history.replaceState(null, '', baseUrl()); } catch (e) { /* ignore */ }
        }
    };

    if (window.addEventListener) window.addEventListener('popstate', () => {
        const t = stockFromHash();
        if (t) openStockDetail(t, { fromHistory: true });
        else if (sheetOpen()) closeModal({ fromHistory: true });
    });

    function copyStockLink() {
        if (!UX.current) return;
        const url = location.href.split('#')[0] + '#stock=' + encodeURIComponent(UX.current);
        const done = () => toast('Link to ' + UX.current + ' copied');
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url).then(done, () => window.prompt('Copy this link', url));
        } else {
            window.prompt('Copy this link', url);
        }
    }

    // ---- stepping ------------------------------------------------------------
    // Next and previous follow whatever the table currently shows - its filters and
    // its sort - so "the next Health Care name by momentum" is one key away. A stock
    // opened from outside that list (a mover, a peer) steps through the full ranking.
    function stepList() {
        const f = (typeof tableState !== 'undefined' && tableState.filtered) ? tableState.filtered : [];
        if (UX.current && f.some(r => r.Ticker === UX.current)) return f;
        return byRank();
    }
    function stepStock(dir) {
        if (!UX.current) return;
        const list = stepList();
        const i = list.findIndex(r => r.Ticker === UX.current);
        const j = i + dir;
        if (i < 0 || j < 0 || j >= list.length) return;
        openStockDetail(list[j].Ticker);
    }

    function updateSheetTools() {
        const t = UX.current;
        if (!t) return;
        const list = stepList();
        const i = list.findIndex(r => r.Ticker === t);
        const pos = document.getElementById('mt-pos');
        const whole = list.length === D.table_data.length;
        if (pos) pos.textContent = i < 0 ? '' : (i + 1) + ' of ' + list.length + (whole ? '' : ' shown');
        const prev = document.getElementById('mt-prev'), next = document.getElementById('mt-next');
        if (prev) prev.disabled = i <= 0;
        if (next) next.disabled = i < 0 || i >= list.length - 1;
        const cb = document.getElementById('mt-compare');
        if (cb) {
            const on = UX.compare.indexOf(t) !== -1;
            cb.setAttribute('aria-pressed', on ? 'true' : 'false');
            cb.textContent = on ? 'Comparing' : 'Compare';
        }
        const hb = document.getElementById('mt-hold');
        if (hb && typeof holdings !== 'undefined') {
            const on = holdings.indexOf(t) !== -1;
            hb.setAttribute('aria-pressed', on ? 'true' : 'false');
            hb.innerHTML = on ? '<span class="mt-lg">In&nbsp;</span>Holdings' : '<span class="mt-lg">Add to&nbsp;</span>Holdings';
        }
    }

    function toggleHoldingCurrent() {
        const t = UX.current;
        if (!t || typeof holdings === 'undefined') return;
        if (holdings.indexOf(t) !== -1) { removeHolding(t); toast(t + ' removed from My Holdings'); }
        else if (holdings.length >= HOLDINGS_MAX) { toast('My Holdings is full (' + HOLDINGS_MAX + ' names)'); }
        else { addHolding(t); toast(t + ' added to My Holdings'); }
        updateSheetTools();
        updateSectionMeta();
    }

    // ---- search palette ------------------------------------------------------
    const PAL_ACTIONS = [
        { label: 'Top 5 stocks', hint: 'Section', run: () => goToSection('sec-top5') },
        { label: 'Full rankings table', hint: 'Section', run: () => goToSection('sec-universe') },
        { label: 'My Holdings', hint: 'Section', run: () => goToSection('sec-holdings') },
        { label: 'What changed', hint: 'Section', run: () => goToSection('sec-changed'), when: () => !!(H && H.available) },
        { label: 'Factor analytics', hint: 'Section', run: () => goToSection('sec-analytics') },
        { label: 'Defensibility and diagnostics', hint: 'Section', run: () => goToSection('sec-defensibility') },
        { label: 'Methodology', hint: 'How it works', run: () => openMethodology() },
        { label: 'Open the comparison', hint: 'Compare', run: () => openCompare(), when: () => UX.compare.length >= 2 },
        { label: 'How to read this screener', hint: 'Guide', run: () => showGuide() },
        { label: 'Keyboard shortcuts', hint: 'Help', run: () => openShortcuts() },
    ];

    function hl(text, q) {
        const s = String(text || '');
        if (!q) return escapeHtml(s);
        const i = s.toLowerCase().indexOf(q);
        if (i < 0) return escapeHtml(s);
        return escapeHtml(s.slice(0, i)) + '<mark>' + escapeHtml(s.slice(i, i + q.length)) + '</mark>' + escapeHtml(s.slice(i + q.length));
    }

    function stockItem(row, q, group) {
        return { kind: 'stock', group: group, t: row.Ticker, row: row, q: q };
    }

    function paletteResults(raw) {
        const q = raw.trim().toLowerCase();
        const items = [];
        const rows = D.table_data;
        if (!q) {
            const rec = UX.recent.map(t => rows.find(r => r.Ticker === t)).filter(Boolean);
            rec.forEach(r => items.push(stockItem(r, '', 'Recent')));
            if (!rec.length) byRank().slice(0, 5).forEach(r => items.push(stockItem(r, '', 'Highest ranked')));
            PAL_ACTIONS.filter(a => !a.when || a.when()).forEach(a => items.push({ kind: 'action', group: 'Go to', a: a, q: '' }));
            return items;
        }
        const scored = [];
        rows.forEach(r => {
            const t = r.Ticker.toLowerCase();
            const c = (r.Company || '').toLowerCase();
            const sec = (r.Sector || '').toLowerCase();
            let sc = 0;
            if (t === q) sc = 1000;
            else if (t.startsWith(q)) sc = 800 - t.length;
            else if (c.startsWith(q)) sc = 600;
            else if ((' ' + c).includes(' ' + q)) sc = 500;
            else if (c.includes(q)) sc = 300;
            else if (q.length >= 3 && sec.includes(q)) sc = 100;
            if (sc) scored.push([sc - (r.Rank || 999) / 1000, r]);
        });
        scored.sort((a, b) => b[0] - a[0]);
        scored.slice(0, 8).forEach(([, r]) => items.push(stockItem(r, q, 'Stocks')));
        PAL_ACTIONS.filter(a => (!a.when || a.when()) && a.label.toLowerCase().includes(q))
            .forEach(a => items.push({ kind: 'action', group: 'Go to', a: a, q: q }));
        return items;
    }

    function renderPalette() {
        const list = document.getElementById('pal-list');
        const items = UX.pal.items;
        if (!items.length) {
            list.innerHTML = '<div class="pal-empty">No stock or section matches. Try a ticker (AAPL) or part of a name.</div>';
            return;
        }
        let html = '', group = null;
        items.forEach((it, i) => {
            if (it.group !== group) { group = it.group; html += '<div class="pal-group" role="presentation">' + escapeHtml(group) + '</div>'; }
            const sel = i === UX.pal.sel;
            html += '<div class="pal-item' + (sel ? ' sel' : '') + '" role="option" id="pal-o-' + i + '" aria-selected="' + sel + '" data-i="' + i + '">';
            if (it.kind === 'stock') {
                const r = it.row;
                const inCmp = UX.compare.indexOf(r.Ticker) !== -1;
                html += '<span class="pal-t">' + hl(r.Ticker, it.q) + '</span>' +
                    '<span class="pal-c">' + hl(r.Company, it.q) + '<span class="pal-s">' + escapeHtml(r.Sector || '') + '</span></span>' +
                    (inCmp ? '<span class="pal-tag">comparing</span>' : '') +
                    '<span class="pal-r"><span>#' + r.Rank + '</span><strong>' + fmt(r.Composite, 'score') + '</strong></span>';
            } else {
                html += '<span class="pal-a">' + hl(it.a.label, it.q) + '</span><span class="pal-h">' + escapeHtml(it.a.hint) + '</span>';
            }
            html += '</div>';
        });
        list.innerHTML = html;
        const input = document.getElementById('pal-input');
        input.setAttribute('aria-activedescendant', 'pal-o-' + UX.pal.sel);
        const selEl = document.getElementById('pal-o-' + UX.pal.sel);
        if (selEl) selEl.scrollIntoView({ block: 'nearest' });
    }

    function paletteQuery() {
        UX.pal.items = paletteResults(document.getElementById('pal-input').value);
        UX.pal.sel = 0;
        renderPalette();
    }

    function runPaletteItem(i) {
        const it = UX.pal.items[i];
        if (!it) return;
        closePalette(true);
        if (it.kind === 'stock') {
            if (document.getElementById('compare-modal').style.display !== 'none') closeCompare();
            openStockDetail(it.t);
        } else {
            it.a.run();
        }
    }

    let palReturnFocus = null;
    function openPalette() {
        const p = document.getElementById('palette');
        if (!p.hidden) return;
        palReturnFocus = document.activeElement;
        p.hidden = false;
        const input = document.getElementById('pal-input');
        input.value = '';
        input.placeholder = 'Search ' + D.table_data.length + ' stocks, a sector, or a section';
        paletteQuery();
        input.focus();
    }
    function closePalette(keepFocus) {
        const p = document.getElementById('palette');
        if (p.hidden) return;
        p.hidden = true;
        if (!keepFocus && palReturnFocus && palReturnFocus.focus && document.contains(palReturnFocus)) palReturnFocus.focus({ preventScroll: true });
    }

    function initPalette() {
        const input = document.getElementById('pal-input');
        input.addEventListener('input', paletteQuery);
        input.addEventListener('keydown', e => {
            const n = UX.pal.items.length;
            if (e.key === 'ArrowDown') { e.preventDefault(); if (n) { UX.pal.sel = (UX.pal.sel + 1) % n; renderPalette(); } }
            else if (e.key === 'ArrowUp') { e.preventDefault(); if (n) { UX.pal.sel = (UX.pal.sel - 1 + n) % n; renderPalette(); } }
            else if (e.key === 'Enter') { e.preventDefault(); runPaletteItem(UX.pal.sel); }
            else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closePalette(); }
            else if (e.key === 'Tab') { e.preventDefault(); }  // the palette is the whole dialog: keep focus in it
        });
        const list = document.getElementById('pal-list');
        list.addEventListener('mousemove', e => {
            const o = e.target.closest('.pal-item');
            if (!o) return;
            const i = +o.dataset.i;
            if (i !== UX.pal.sel) {
                UX.pal.sel = i;
                list.querySelectorAll('.pal-item').forEach(x => { const on = +x.dataset.i === i; x.classList.toggle('sel', on); x.setAttribute('aria-selected', on); });
                input.setAttribute('aria-activedescendant', 'pal-o-' + i);
            }
        });
        list.addEventListener('click', e => {
            const o = e.target.closest('.pal-item');
            if (o) runPaletteItem(+o.dataset.i);
        });
        const k = document.getElementById('cmdk-kbd');
        if (k) k.textContent = IS_MAC ? '⌘K' : 'Ctrl K';
        const km = document.getElementById('kb-mod');
        if (km) km.textContent = IS_MAC ? '⌘' : 'Ctrl';
    }

    // ---- compare ---------------------------------------------------------------
    function saveCompare() { uxStore.set('compare', UX.compare); renderCompareTray(); updateSheetTools(); }
    function toggleCompare(t) {
        const i = UX.compare.indexOf(t);
        if (i !== -1) { UX.compare.splice(i, 1); saveCompare(); toast(t + ' removed from the comparison'); return; }
        if (UX.compare.length >= COMPARE_MAX) { toast('Compare holds ' + COMPARE_MAX + ' stocks. Remove one first.'); return; }
        UX.compare.push(t);
        saveCompare();
        toast(UX.compare.length === 1 ? t + ' added. Add another to compare.' : t + ' added to the comparison');
    }
    function toggleCompareCurrent() { if (UX.current) toggleCompare(UX.current); }
    function clearCompare() { UX.compare = []; saveCompare(); closeCompare(); }

    function renderCompareTray() {
        const tray = document.getElementById('cmp-tray');
        if (!tray) return;
        tray.hidden = UX.compare.length === 0;
        document.body.classList.toggle('has-tray', UX.compare.length > 0);
        document.getElementById('cmp-chips').innerHTML = UX.compare.map(t =>
            '<span class="cmp-chip"><button type="button" class="cmp-chip-t" onclick="openStockDetail(\'' + t + '\')">' + escapeHtml(t) + '</button>' +
            '<button type="button" class="cmp-chip-x" onclick="toggleCompare(\'' + t + '\')" aria-label="Remove ' + escapeHtml(t) + '">&times;</button></span>'
        ).join('');
        const open = document.getElementById('cmp-open');
        open.disabled = UX.compare.length < 2;
        open.textContent = UX.compare.length < 2 ? 'Add one more' : 'Side by side';
    }

    function gapSentence(a, b) {
        // The composite gap between two stocks, taken apart into category points.
        // Each stock's points add up to its composite before any coverage discount
        // (calc_trace checks that for every stock at build time), so the difference
        // in points per category, plus any difference in discount, is the whole gap.
        const sa = D.stock_detail[a], sb = D.stock_detail[b];
        const gap = sa.composite - sb.composite;
        const parts = UX_CATS.map(c => [c, (sa.contrib[c] || 0) - (sb.contrib[c] || 0)]);
        const sumParts = parts.reduce((n, p) => n + p[1], 0);
        const resid = gap - sumParts;
        parts.sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
        const pts = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1);
        const rows = parts.map(([c, v]) =>
            '<div class="gap-row"><span>' + CAT_LABELS[c] + '</span>' +
            '<span class="gap-track"><i class="' + (v >= 0 ? 'up' : 'dn') + '" style="--w:' + Math.min(50, Math.abs(v) / Math.max(1, ...parts.map(p => Math.abs(p[1]))) * 50).toFixed(1) + '%"></i></span>' +
            '<span class="num">' + pts(v) + '</span></div>').join('');
        const discounted = !!((sa.cov && sa.cov.disc) || (sb.cov && sb.cov.disc));
        const discRow = discounted && Math.abs(resid) >= 0.05
            ? '<div class="gap-row"><span>Coverage discount</span><span class="gap-track"></span><span class="num">' + pts(resid) + '</span></div>' : '';
        const lead = parts.filter(p => Math.abs(p[1]) >= 0.05).slice(0, 2).map(p => CAT_LABELS[p[0]] + ' (' + pts(p[1]) + ')');
        return '<div class="gap-block"><p class="gap-lede"><strong>' + escapeHtml(a) + '</strong> scores ' +
            Math.abs(gap).toFixed(1) + ' composite points ' + (gap >= 0 ? 'above' : 'below') + ' <strong>' + escapeHtml(b) + '</strong>' +
            (lead.length ? '. The largest differences: ' + lead.join(' and ') + '.' : '.') + '</p>' +
            '<div class="gap-rows">' + rows + discRow +
            '<div class="gap-row gap-total"><span>Composite gap</span><span class="gap-track"></span><span class="num">' + pts(gap) + '</span></div></div></div>';
    }

    function openCompare() {
        if (UX.compare.length < 2) { toast('Add at least two stocks to compare'); return; }
        closePalette(true);
        const ts = UX.compare.slice().sort((a, b) => D.stock_detail[a].rank - D.stock_detail[b].rank);
        const S = ts.map(t => D.stock_detail[t]);
        const best = vals => { const v = vals.filter(x => x !== null && x !== undefined); return v.length > 1 ? Math.max(...v) : null; };
        const head = '<tr><th></th>' + ts.map((t, i) =>
            '<th><button type="button" class="cmp-head" onclick="closeCompare();openStockDetail(\'' + t + '\')">' +
            '<span class="cmp-t">' + escapeHtml(t) + '</span><span class="cmp-n">' + escapeHtml(S[i].company) + '</span>' +
            '<span class="cmp-sec">' + escapeHtml(S[i].sector) + '</span></button></th>').join('') + '</tr>';
        const compBest = best(S.map(s => s.composite));
        let body = '<tr class="cmp-comp"><th>Composite</th>' + S.map(s =>
            '<td class="' + (s.composite === compBest ? 'best' : '') + '"><strong>' + fmt(s.composite, 'score') + '</strong><span>#' + s.rank + ' of ' + D.table_data.length + '</span></td>').join('') + '</tr>';
        UX_CATS.forEach(c => {
            const vals = S.map(s => s.cat_scores[c]);
            const b = best(vals);
            body += '<tr><th>' + CAT_LABELS[c] + '</th>' + S.map((s, i) => {
                const v = vals[i];
                if (v === null || v === undefined) return '<td class="na">no data</td>';
                return '<td class="' + (v === b ? 'best' : '') + '"><div class="cmp-cell"><span class="cmp-v">' + fmt(v, 'score') + '</span>' +
                    '<span class="cmp-bar"><i style="width:' + Math.max(0, Math.min(100, v)).toFixed(1) + '%"></i></span>' +
                    '<span class="cmp-pts">' + fmt(s.contrib[c], 'score') + ' pts</span></div></td>';
            }).join('') + '</tr>';
        });
        body += '<tr class="cmp-meta"><th>Metrics with data</th>' + S.map(s => '<td>' + (s.cov ? s.cov.n + ' of ' + s.cov.of : '&mdash;') + '</td>').join('') + '</tr>';
        body += '<tr class="cmp-meta"><th>Trap flags</th>' + S.map(s => '<td>' + ([s.vt ? 'Value' : '', s.gt ? 'Growth' : ''].filter(Boolean).join(', ') || 'None') + '</td>').join('') + '</tr>';
        body += '<tr class="cmp-meta"><th>Price vs mean analyst target</th>' + S.map(s => {
            if (!s.price || !s.pt_mean) return '<td>&mdash;</td>';
            const u = (s.pt_mean / s.price - 1) * 100;
            return '<td>' + (u >= 0 ? '+' : '−') + Math.abs(u).toFixed(1) + '%<span class="cmp-sub">' + (s.num_analysts || 0) + ' analysts</span></td>';
        }).join('') + '</tr>';
        let gaps = '';
        for (let i = 1; i < ts.length; i++) gaps += gapSentence(ts[0], ts[i]);
        document.getElementById('cmp-body').innerHTML =
            '<div class="cmp-scroll"><table class="cmp-table" style="--n:' + ts.length + '"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>' +
            '<h3 class="cmp-h3">Where the composite gap comes from</h3>' +
            '<p class="modal-note">Each line is the difference in category points, so the lines add up to the gap in composite (rounding aside). It says which categories separate the two in this run, not which is the better company.</p>' +
            gaps;
        const m = document.getElementById('compare-modal');
        m.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        document.body.classList.add('compare-showing');
        const c = m.querySelector('.modal-close');
        if (c) c.focus({ preventScroll: true });
    }
    function closeCompare() {
        const m = document.getElementById('compare-modal');
        if (!m || m.style.display === 'none') return;
        m.style.display = 'none';
        document.body.classList.remove('compare-showing');
        if (!sheetOpen()) document.body.style.overflow = '';
    }

    // ---- the workings as a spreadsheet ---------------------------------------
    // Every number the drilldown shows for one stock, in one file a student can open in
    // Excel and redo with a calculator: the composite from category points, each category
    // from its metrics' percentiles and the weights actually used, and the reported inputs.
    function csvCell(v) {
        if (v === null || v === undefined) return '';
        const t = String(v);
        return /[",\n]/.test(t) ? '"' + t.replace(/"/g, '""') + '"' : t;
    }
    function round6(v) { return (v === null || v === undefined || !isFinite(v)) ? '' : Math.round(v * 1e6) / 1e6; }

    function workingsCsv(ticker) {
        const s = D.stock_detail[ticker];
        if (!s) return '';
        const ew = effWeights(s);
        const meta = D.metric_meta || {};
        const runDate = String((D.kpis || {}).run_timestamp || '').slice(0, 10);
        const L = [];
        L.push(['Multi-Factor Screener: the workings for ' + ticker]);
        L.push(['Company', s.company], ['Sector', s.sector], ['Run date', runDate],
               ['Rank', s.rank + ' of ' + D.table_data.length], ['Composite', s.composite]);
        L.push(['Not investment advice. Every figure below is from this run\'s published payload.']);
        L.push([]);
        L.push(['COMPOSITE: category score x weight used = points; points add up to the composite']);
        L.push(['Category', 'Category score (0-100)', 'Weight used (%)', 'Points']);
        let sum = 0;
        UX_CATS.forEach(c => {
            const sc = s.cat_scores[c];
            const pts = s.contrib[c];
            if (pts !== null && pts !== undefined) sum += pts;
            L.push([CAT_LABELS[c], sc === null || sc === undefined ? 'not scored' : sc, round6(ew[c]), pts === null || pts === undefined ? 0 : pts]);
        });
        L.push(['Points add up to', '', '', round6(sum)]);
        const disc = (s.cov || {}).disc;
        if (disc) L.push(['Coverage discount (%)', '', '', round6(disc * 100)]);
        L.push(['Composite', '', '', s.composite]);
        L.push([]);
        L.push(['CATEGORIES: sector percentile x share of weight = points; points add up to the category score']);
        L.push(['Category', 'Metric', 'Raw value', 'Sector percentile', 'Share of category weight (%)', 'Points', 'Weight table']);
        UX_CATS.forEach(c => {
            const wk = categoryWorkings(c, s);
            if (!wk) return;
            wk.rows.forEach(r => L.push([CAT_LABELS[c], (meta[r.metric] || {}).label || r.metric, round6(r.raw),
                r.pct === null ? 'no data' : round6(r.pct), r.share === null ? '' : round6(r.share),
                r.points === null ? '' : round6(r.points), wk.wp.pid]));
            L.push([CAT_LABELS[c], 'Category score', '', '', '', round6(wk.points), '']);
        });
        const inp = s.inp || {};
        const keys = Object.keys(inp);
        if (keys.length) {
            L.push([]);
            L.push(['INPUTS: as reported by the company and delivered by Yahoo Finance']);
            L.push(['Field', 'Value']);
            keys.sort().forEach(k => L.push([k, inp[k]]));
        }
        return L.map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
    }

    function downloadWorkings(ticker) {
        const t = ticker || UX.current;
        if (!t) return;
        const blob = new Blob(['﻿' + workingsCsv(t)], { type: 'text/csv;charset=utf-8' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = t + '-workings-' + String((D.kpis || {}).run_timestamp || '').slice(0, 10) + '.csv';
        document.body.appendChild(a);
        a.click();
        setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
        toast('Downloaded ' + a.download);
    }

    // ---- analytics -> rankings --------------------------------------------------
    // A sector row in the matrix, or a bar in the trap chart, is a question ("which stocks
    // are these?") whose answer is the rankings table filtered. One click asks it.
    function filterTable(opts) {
        const sec = document.getElementById('filter-sector');
        const vt = document.getElementById('filter-vt');
        const comp = document.getElementById('filter-comp-min');
        const q = document.getElementById('filter-search');
        if (!sec || !vt) return;
        sec.value = opts.sector || 'all';
        vt.value = opts.flag || 'all';
        if (comp) comp.value = '0';
        if (q) q.value = '';
        applyFilters();
        goToSection('sec-universe');
        const n = tableState.filtered.length;
        toast('Showing ' + n + ' ' + (opts.flag === 'vt' ? 'value-trap flagged ' : opts.flag === 'gt' ? 'growth-trap flagged ' : '') +
              (opts.sector ? opts.sector + ' ' : '') + (n === 1 ? 'stock' : 'stocks'));
    }
    function bindAnalyticsLinks() {
        const act = (e, fn) => {
            if (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ') return;
            const row = e.target.closest('[data-sector]');
            if (!row) return;
            e.preventDefault();
            fn(row.dataset.sector);
        };
        const sm = document.getElementById('sector-matrix');
        const tb = document.getElementById('trap-bars');
        ['click', 'keydown'].forEach(ev => {
            if (sm) sm.addEventListener(ev, e => act(e, sec => filterTable({ sector: sec })));
            if (tb) tb.addEventListener(ev, e => act(e, sec => filterTable({ sector: sec, flag: currentTrapType })));
        });
    }

    // ---- guide and shortcuts -------------------------------------------------
    function showGuide() {
        const g = document.getElementById('guide');
        if (!g) return;
        g.hidden = false;
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    function dismissGuide() {
        const g = document.getElementById('guide');
        if (g) g.hidden = true;
        uxStore.set('guide_done', true);
    }
    function openShortcuts() { closePalette(true); document.getElementById('shortcuts').hidden = false; }
    function closeShortcuts() { document.getElementById('shortcuts').hidden = true; }

    // ---- section previews and scroll-spy ------------------------------------
    // A collapsed section says what is inside it, so the page reads as a contents
    // list rather than a column of bare headers.
    function updateSectionMeta() {
        const mh = document.getElementById('meta-holdings');
        if (mh && typeof holdings !== 'undefined') {
            mh.textContent = holdings.length
                ? holdings.slice(0, 4).join(', ') + (holdings.length > 4 ? ' and ' + (holdings.length - 4) + ' more' : '')
                : 'Track the names you own or watch';
        }
        const mc = document.getElementById('meta-changed');
        if (mc && H && H.available && H.movers) {
            const k = H.movers.m1 ? 'm1' : 'prev';
            const mv = H.movers[k], cmp = (H.compare || {})[k];
            if (mv && cmp) mc.textContent = mv.n_up + ' up, ' + mv.n_down + ' down materially since ' + cmp.date;
        }
    }

    function initScrollSpy() {
        const links = [...document.querySelectorAll('.header-nav a')];
        const targets = links.map(a => document.getElementById(a.getAttribute('href').slice(1)));
        let queued = false;
        const update = () => {
            queued = false;
            const line = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--bar-h')) || 52) + 80;
            // The section whose top has most recently passed the reading line. Compared by
            // position, because the rankings table sits last on the page but second in the nav.
            let best = null, bestTop = -Infinity;
            targets.forEach((el, i) => {
                if (!el || el.offsetParent === null) return;
                const t = el.getBoundingClientRect().top;
                if (t <= line && t > bestTop) { bestTop = t; best = i; }
            });
            links.forEach((a, i) => a.classList.toggle('active', i === best));
        };
        window.addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(update); } }, { passive: true });
        update();
    }

    // Which part of the drilldown you are reading, lit in its nav.
    let modalSpyBound = false;
    function modalSpy() {
        const body = document.querySelector('#stock-modal .modal-body');
        const links = [...document.querySelectorAll('#stock-modal .modal-nav a')];
        if (!body || !links.length) return;
        const update = () => {
            const top = body.getBoundingClientRect().top + 24;
            let cur = links[0];
            links.forEach(a => {
                const el = document.getElementById(a.getAttribute('href').slice(1));
                if (el && el.offsetParent !== null && el.getBoundingClientRect().top <= top) cur = a;
            });
            links.forEach(a => a.classList.toggle('active', a === cur));
        };
        if (!modalSpyBound) { body.addEventListener('scroll', () => requestAnimationFrame(update), { passive: true }); modalSpyBound = true; }
        update();
    }

    // ---- keyboard --------------------------------------------------------------
    document.addEventListener('keydown', e => {
        const tag = (e.target && e.target.tagName) || '';
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(tag) || (e.target && e.target.isContentEditable);
        if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
            e.preventDefault();
            const p = document.getElementById('palette');
            if (p.hidden) openPalette(); else closePalette();
            return;
        }
        if (e.key === 'Escape') {
            if (!document.getElementById('palette').hidden) { closePalette(); e.stopImmediatePropagation(); return; }
            if (!document.getElementById('shortcuts').hidden) { closeShortcuts(); e.stopImmediatePropagation(); return; }
            if (document.getElementById('compare-modal').style.display !== 'none') { closeCompare(); e.stopImmediatePropagation(); return; }
            return;
        }
        if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.key === '?') { e.preventDefault(); openShortcuts(); return; }
        if (!sheetOpen() || !document.getElementById('palette').hidden) return;
        const k = e.key.toLowerCase();
        if (k === 'j') { e.preventDefault(); stepStock(1); }
        else if (k === 'k') { e.preventDefault(); stepStock(-1); }
        else if (k === 'c') { e.preventDefault(); toggleCompareCurrent(); }
        else if (k === 'h') { e.preventDefault(); toggleHoldingCurrent(); }
    }, true);

    function initUX() {
        UX.recent = (uxStore.get('recent', []) || []).filter(t => D.stock_detail[t]).slice(0, 6);
        UX.compare = (uxStore.get('compare', []) || []).filter(t => D.stock_detail[t]).slice(0, COMPARE_MAX);
        initPalette();
        renderCompareTray();
        updateSectionMeta();
        initScrollSpy();
        bindAnalyticsLinks();
        if (!uxStore.get('guide_done', false)) { const g = document.getElementById('guide'); if (g) g.hidden = false; }
        // Keep the holdings preview current whichever way the list changes.
        if (typeof renderHoldings === 'function') {
            const _rh = renderHoldings;
            renderHoldings = function() { _rh.apply(this, arguments); updateSectionMeta(); if (UX.current) updateSheetTools(); };
        }
        const t = stockFromHash();
        if (t) openStockDetail(t, { replace: true });
    }
"""


def _css_ux() -> str:
    """CSS for the navigation layer and the second design pass.

    Emitted as its own <style> block after ``_css()``, so at equal specificity it
    wins the cascade: the first pass lost two fights that way (the phone score grid
    and the contribution rows), and appending is cheaper to reason about than
    finding every earlier rule. Plain string: no backslash escapes, literal glyphs.
    """
    return """
        /* ---- HEADER: search button, active section ---- */
        .cmdk-btn {
            display: inline-flex; align-items: center; gap: 8px; height: 32px; padding: 0 8px 0 10px;
            background: var(--bg-card); border: 1px solid var(--border-bright); border-radius: var(--radius);
            color: var(--text-muted); font: 500 13px var(--font-body); cursor: pointer;
            transition: border-color var(--t-fast) ease-out, color var(--t-fast) ease-out;
        }
        .cmdk-btn:hover { color: var(--text-primary); border-color: var(--text-muted); }
        .cmdk-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        .cmdk-btn svg { width: 14px; height: 14px; }
        .cmdk-btn-text { min-width: 92px; text-align: left; }
        .cmdk-kbd, .pal kbd, .kb-list kbd, .pal-foot kbd {
            font: 500 11px var(--font-body); color: var(--text-muted); border: 1px solid var(--border-bright);
            border-radius: 4px; padding: 0 5px; line-height: 17px; background: var(--bg-primary); white-space: nowrap;
        }
        .header-right { display: flex; align-items: center; gap: 8px; }
        .header-nav a.active { color: var(--text-primary); background: var(--bg-elevated); }

        /* ---- FIRST-VISIT GUIDE ---- */
        .guide {
            margin: 0 0 var(--gap); padding: 18px 20px 14px; border: 1px solid var(--border-bright);
            border-radius: var(--radius); background: var(--bg-card);
        }
        .guide[hidden] { display: none; }
        .guide-steps { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
        .guide-steps li { display: flex; gap: 12px; font-size: 13px; line-height: 1.55; color: var(--text-secondary); }
        .guide-steps strong { color: var(--text-primary); font-weight: 600; display: block; margin-bottom: 2px; }
        .guide-n {
            flex: none; width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
            font-size: 12px; font-weight: 600; color: var(--accent-text); background: var(--accent-glow);
            font-variant-numeric: tabular-nums;
        }
        .guide-foot {
            display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
            margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 12px; color: var(--text-muted);
        }
        .guide-close {
            height: 30px; padding: 0 14px; border-radius: var(--radius); border: 1px solid var(--border-bright);
            background: var(--bg-elevated); color: var(--text-primary); font: 500 13px var(--font-body); cursor: pointer;
        }
        .guide-close:hover { border-color: var(--text-muted); }

        /* ---- SECTION PREVIEWS ---- */
        .collapsible-section .section-header { gap: 12px; }
        .sec-meta {
            flex: 1; min-width: 0; font-size: 13px; color: var(--text-muted); white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums;
        }
        .section-header .section-chevron { margin-left: auto; flex: none; }

        /* ---- TRAP RATE BARS (replaces the canvas chart) ---- */
        .trap-bars { display: flex; flex-direction: column; gap: 2px; }
        .trap-row {
            display: grid; grid-template-columns: minmax(0, 160px) minmax(40px, 1fr) 104px; gap: 10px;
            align-items: center; min-height: 26px; font-size: 12.5px;
        }
        .trap-name { color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .trap-track { height: 8px; border-radius: 4px; background: var(--bg-elevated); overflow: hidden; }
        .trap-track i { display: block; height: 100%; background: var(--accent); opacity: .75; border-radius: 4px; }
        .trap-val { display: flex; justify-content: flex-end; align-items: baseline; gap: 6px; white-space: nowrap; font-variant-numeric: tabular-nums; }
        .trap-val strong { font-weight: 600; color: var(--text-primary); }
        .trap-val span { font-size: 11px; color: var(--text-muted); }

        .trap-row[data-sector], .sm-table tbody tr[data-sector] { cursor: pointer; }
        .trap-row[data-sector] { border-radius: 6px; padding: 0 6px; margin: 0 -6px; }
        .trap-row[data-sector]:hover { background: var(--bg-card-hover); }
        .sm-table tbody tr[data-sector]:hover .sm-sector { color: var(--accent-text); }
        .trap-row:focus-visible, .sm-table tbody tr:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
        .methodology-body h1 { letter-spacing: -.02em; }
        @media (max-width: 760px) { .methodology-body h1 { font-size: 24px; line-height: 1.25; } }

        /* ---- DRILLDOWN TOOLBAR ---- */
        .modal-tools { display: flex; align-items: center; gap: 4px; flex: none; }
        .mt-btn {
            height: 30px; min-width: 30px; padding: 0 10px; display: inline-flex; align-items: center; justify-content: center;
            background: none; border: 1px solid var(--border-bright); border-radius: var(--radius);
            color: var(--text-secondary); font: 500 12.5px var(--font-body); cursor: pointer; white-space: nowrap;
            transition: color var(--t-fast) ease-out, border-color var(--t-fast) ease-out, background var(--t-fast) ease-out;
        }
        .mt-btn:hover:not(:disabled) { color: var(--text-primary); border-color: var(--text-muted); }
        .mt-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
        .mt-btn:disabled { opacity: .35; cursor: default; }
        .mt-step { padding: 0; }
        .mt-step svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
        .mt-btn[aria-pressed="true"] { color: var(--accent-text); border-color: var(--accent); background: var(--accent-glow); }
        .mt-pos { min-width: 72px; text-align: center; font-size: 12px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
        .mt-text { margin-left: 4px; }
        #stock-modal .modal-header { flex-wrap: wrap; row-gap: 12px; }
        #stock-modal .modal-tools { order: 4; flex: 1 1 100%; }
        .modal-nav a.active { color: var(--text-primary); background: var(--bg-elevated); }
        .modal-nav a[hidden] { display: none; }

        /* ---- DRILLDOWN SCORES: each card carries its own bar, and opens its workings ---- */
        #stock-modal .modal-score-row { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .modal-score-card { position: relative; padding-bottom: 18px; cursor: pointer; }
        .modal-score-card::after {
            content: ''; position: absolute; left: 12px; right: 12px; bottom: 9px; height: 3px; border-radius: 2px;
            background: linear-gradient(to right, var(--accent) calc(var(--v, 0) * 100%), var(--bg-elevated) 0);
        }
        .modal-score-card.composite { cursor: default; padding-bottom: 14px; }
        .modal-score-card.composite::after { display: none; }
        .modal-score-card:not(.composite):hover { border-color: var(--text-muted); }

        /* ---- HOW IT ADDS UP: a quiet ledger, not a wall of saturated bars ---- */
        #contrib-visual .contrib-row {
            display: grid; grid-template-columns: 150px minmax(0, 1fr) 84px; gap: 4px 16px; align-items: center;
            padding: 10px 6px; margin: 0 -6px; border-radius: 6px;
        }
        #contrib-visual .contrib-row-link { cursor: pointer; }
        #contrib-visual .contrib-row-link:hover { background: var(--bg-card-hover); }
        #contrib-visual .contrib-bar-track { height: 8px; border: 0; border-radius: 4px; background: var(--bg-elevated); }
        #contrib-visual .contrib-bar-fill { background: var(--accent) !important; opacity: .8; border-radius: 4px; padding: 0; transition: none; }
        #contrib-visual .contrib-bar-inner-label, #contrib-visual .contrib-bar-outer-label { display: none; }
        #contrib-visual .contrib-bar-max-marker { top: -3px; bottom: -3px; width: 1px; background: var(--border-bright); }
        #contrib-visual .contrib-bar-annotation { margin-top: 6px; font-variant-numeric: tabular-nums; }
        #contrib-visual .contrib-pts { text-align: right; font-variant-numeric: tabular-nums; }

        /* ---- LOWER DRILLDOWN BLOCKS: hairlines and type, not boxes inside boxes ---- */
        #stock-modal .pt-section, #stock-modal .snapshot-section, #stock-modal .peer-section {
            background: none; border: 0; border-radius: 0; padding: 0; box-shadow: none;
        }
        #stock-modal .pt-cards { gap: 0; }
        #stock-modal .pt-card {
            background: none; border: 0; border-left: 1px solid var(--border); border-radius: 0;
            padding: 2px 16px; text-align: left; box-shadow: none;
        }
        #stock-modal .pt-card:first-child { border-left: 0; padding-left: 0; }
        #stock-modal .pt-card-value { font-variant-numeric: tabular-nums; }
        #stock-modal .snapshot-group-label { border-left: 0; padding-left: 0; }
        #stock-modal .snapshot-item {
            background: none; border: 0; border-top: 1px solid var(--border); border-radius: 0; padding: 10px 0 4px;
        }
        #stock-modal .snapshot-value { font-variant-numeric: tabular-nums; }
        #stock-modal .peer-table td, #stock-modal .peer-table th { font-variant-numeric: tabular-nums; }
        #stock-modal .peer-row-median td { color: var(--text-muted); border-top: 1px solid var(--border-bright); }
        #stock-modal .peer-row-median .peer-ticker { font-weight: 500; color: var(--text-muted); }

        /* ---- MY HOLDINGS, populated: one quiet card per name ----
           The category strip had eight accent-coloured top rules per card (the old hue-per-
           category scheme collapsed to one blue), so a list of four names drew 32 blue lines.
           The concentration note drops its accent rail and heavy bold: it is reading, not an alert. */
        .holding-cat { border-top: 0 !important; background: var(--bg-primary); border-radius: 6px; padding: 6px 8px; }
        .holding-cat-val { font-variant-numeric: tabular-nums; font-weight: 600; }
        .holding-composite, .holding-rank, .holding-delta { font-variant-numeric: tabular-nums; }
        .holdings-concentration { border-left: 1px solid var(--border); background: var(--bg-card); }
        .holdings-concentration strong { font-weight: 600; }
        .holdings-fit-line { background: var(--bg-card); }
        @media (max-width: 760px) {
            .holding-cats { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; }
            .holding-cat { padding: 5px 6px; }
            .holding-company { max-width: 150px; }
            .holding-sector { display: none; }
            .holding-cat-name { letter-spacing: 0; font-size: 10px; }
        }

        /* ---- SEARCH PALETTE ---- */
        .pal-overlay {
            position: fixed; inset: 0; z-index: 1200; background: rgba(0,0,0,.55);
            display: flex; align-items: flex-start; justify-content: center; padding: 12vh 16px 16px;
        }
        .pal-overlay[hidden] { display: none; }
        .pal {
            width: 100%; max-width: 620px; max-height: 70vh; display: flex; flex-direction: column;
            background: var(--bg-primary); border: 1px solid var(--border-bright); border-radius: 12px;
            box-shadow: var(--shadow-overlay); overflow: hidden; animation: palIn var(--t-base) ease-out;
        }
        @keyframes palIn { from { opacity: 0; transform: translateY(-6px) scale(.99); } to { opacity: 1; transform: none; } }
        .pal-input-row { display: flex; align-items: center; gap: 10px; padding: 0 14px; border-bottom: 1px solid var(--border); }
        .pal-input-row svg { width: 17px; height: 17px; color: var(--text-muted); flex: none; }
        .pal-input-row input {
            flex: 1; min-width: 0; height: 52px; background: none; border: 0; outline: none;
            color: var(--text-primary); font: 400 16px var(--font-body);
        }
        .pal-input-row input::placeholder { color: var(--text-muted); }
        .pal-list { overflow-y: auto; padding: 6px; overscroll-behavior: contain; }
        .pal-group { padding: 10px 10px 4px; font-size: 11px; font-weight: 500; letter-spacing: .05em; text-transform: uppercase; color: var(--text-muted); }
        .pal-item {
            display: flex; align-items: center; gap: 12px; padding: 8px 10px; border-radius: var(--radius);
            cursor: pointer; min-height: 44px;
        }
        .pal-item.sel { background: var(--bg-elevated); }
        .pal-t { flex: none; width: 62px; font-weight: 600; color: var(--text-primary); font-size: 13.5px; }
        .pal-c { flex: 1; min-width: 0; color: var(--text-secondary); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .pal-s { color: var(--text-muted); font-size: 12px; margin-left: 8px; }
        .pal-r { flex: none; display: flex; gap: 10px; align-items: baseline; font-variant-numeric: tabular-nums; font-size: 12px; color: var(--text-muted); }
        .pal-r strong { color: var(--text-primary); font-size: 13px; font-weight: 600; min-width: 34px; text-align: right; }
        .pal-tag { flex: none; font-size: 11px; color: var(--accent-text); }
        .pal-a { flex: 1; color: var(--text-primary); font-size: 13.5px; }
        .pal-h { flex: none; color: var(--text-muted); font-size: 12px; }
        .pal mark { background: none; color: var(--accent-text); font-weight: 600; }
        .pal-empty { padding: 28px 16px; text-align: center; color: var(--text-muted); font-size: 13px; }
        .pal-foot { display: flex; gap: 16px; padding: 8px 14px; border-top: 1px solid var(--border); font-size: 11.5px; color: var(--text-muted); }
        .pal-foot kbd { margin-right: 3px; }

        /* ---- KEYBOARD SHORTCUTS ---- */
        .kb-sheet { padding: 20px 22px; max-width: 460px; }
        .kb-sheet h2 { font-size: 16px; font-weight: 600; margin: 0 0 14px; }
        .kb-list { display: grid; grid-template-columns: auto 1fr; gap: 10px 18px; margin: 0; font-size: 13px; }
        .kb-list dt { display: flex; gap: 4px; align-items: center; }
        .kb-list dd { margin: 0; color: var(--text-secondary); }
        .kb-note { margin: 16px 0 0; font-size: 12px; color: var(--text-muted); line-height: 1.5; }

        /* ---- TOAST ---- */
        .toast {
            position: fixed; left: 50%; bottom: 24px; z-index: 1300; transform: translate(-50%, 12px);
            padding: 9px 14px; border-radius: var(--radius); background: var(--bg-elevated); border: 1px solid var(--border-bright);
            color: var(--text-primary); font-size: 13px; box-shadow: var(--shadow-overlay);
            opacity: 0; pointer-events: none; transition: opacity var(--t-base) ease-out, transform var(--t-base) ease-out;
        }
        .toast.show { opacity: 1; transform: translate(-50%, 0); }
        body.has-tray .toast { bottom: 84px; }

        /* ---- COMPARE: tray and side-by-side ---- */
        .cmp-tray {
            position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%); z-index: 1100;
            display: flex; align-items: center; gap: 10px; padding: 8px 8px 8px 14px; max-width: calc(100vw - 32px);
            background: var(--bg-elevated); border: 1px solid var(--border-bright); border-radius: 12px; box-shadow: var(--shadow-overlay);
        }
        .cmp-tray[hidden] { display: none; }
        body.has-tray { padding-bottom: 72px; }
        .cmp-tray-label { font-size: 11px; font-weight: 500; letter-spacing: .05em; text-transform: uppercase; color: var(--text-muted); }
        .cmp-chips { display: flex; gap: 6px; overflow-x: auto; scrollbar-width: none; }
        .cmp-chip { display: inline-flex; align-items: center; border: 1px solid var(--border-bright); border-radius: var(--radius-pill); background: var(--bg-card); }
        .cmp-chip button { background: none; border: 0; color: var(--text-primary); font: 600 12.5px var(--font-body); cursor: pointer; }
        .cmp-chip-t { padding: 4px 4px 4px 10px; }
        .cmp-chip-x { padding: 4px 9px 4px 4px; color: var(--text-muted) !important; font-weight: 400 !important; font-size: 15px !important; line-height: 1; }
        .cmp-chip-x:hover { color: var(--text-primary) !important; }
        .cmp-open {
            height: 32px; padding: 0 14px; border-radius: var(--radius); border: 0; background: var(--accent); color: #fff;
            font: 600 13px var(--font-body); cursor: pointer; white-space: nowrap;
        }
        .cmp-open:disabled { background: var(--bg-card); color: var(--text-muted); cursor: default; border: 1px solid var(--border-bright); }
        .cmp-clear { height: 32px; padding: 0 10px; background: none; border: 0; color: var(--text-muted); font: 500 12.5px var(--font-body); cursor: pointer; }
        .cmp-clear:hover { color: var(--text-primary); }
        .cmp-overlay { z-index: 1050; }
        .modal-ticker { color: var(--text-primary); letter-spacing: -.01em; }
        #compare-modal .modal-ticker { font-size: 20px; font-weight: 600; }
        .cmp-content { max-width: 1040px; width: 100%; min-width: 0; max-height: calc(100vh - 48px); display: flex; flex-direction: column; }
        .cmp-content .modal-header { flex: none; }
        .cmp-body { flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; }
        body.compare-showing .cmp-tray { display: none; }
        .cmp-table tbody th, .cmp-table thead th:first-child { position: sticky; left: 0; z-index: 1; background: var(--bg-primary); }
        .cmp-scroll { overflow-x: auto; margin: 0 -4px; }
        .cmp-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: calc(150px + var(--n) * 150px); }
        .cmp-table th, .cmp-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: middle; }
        .cmp-table tbody th { font-size: 13px; font-weight: 500; color: var(--text-secondary); width: 170px; }
        .cmp-table thead th { vertical-align: bottom; border-bottom-color: var(--border-bright); }
        .cmp-head { display: flex; flex-direction: column; align-items: flex-start; gap: 1px; background: none; border: 0; padding: 0; cursor: pointer; text-align: left; color: inherit; font-family: var(--font-body); }
        .cmp-head:hover .cmp-t { color: var(--accent-text); }
        .cmp-t { font-size: 18px; font-weight: 600; color: var(--text-primary); letter-spacing: -.01em; }
        .cmp-n { font-size: 12.5px; color: var(--text-secondary); max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .cmp-sec { font-size: 11.5px; color: var(--text-muted); }
        .cmp-comp td strong { font-size: 22px; font-weight: 600; letter-spacing: -.02em; display: block; }
        .cmp-comp td span { font-size: 12px; color: var(--text-muted); }
        .cmp-cell { display: grid; grid-template-columns: 40px minmax(40px, 1fr); grid-template-rows: auto auto; gap: 2px 10px; align-items: center; }
        .cmp-v { font-size: 14px; font-weight: 500; color: var(--text-primary); }
        .cmp-bar { height: 6px; border-radius: 3px; background: var(--bg-elevated); overflow: hidden; }
        .cmp-bar i { display: block; height: 100%; background: var(--text-muted); border-radius: 3px; }
        .cmp-pts { grid-column: 2; font-size: 11px; color: var(--text-muted); }
        .cmp-table td.best .cmp-v, .cmp-table td.best strong { color: var(--text-primary); font-weight: 700; }
        .cmp-table td.best .cmp-bar i { background: var(--accent); }
        .cmp-table td.best { box-shadow: inset 2px 0 0 var(--accent); }
        .cmp-table td.na { color: var(--text-muted); font-size: 12.5px; }
        .cmp-meta td { font-size: 13px; color: var(--text-secondary); }
        .cmp-sub { display: block; font-size: 11px; color: var(--text-muted); }
        .cmp-h3 { font-size: 14px; font-weight: 600; margin: 28px 0 6px; }
        .gap-block { padding: 14px 0; border-top: 1px solid var(--border); }
        .gap-lede { font-size: 13.5px; color: var(--text-secondary); margin: 0 0 10px; line-height: 1.55; }
        .gap-lede strong { color: var(--text-primary); }
        .gap-rows { display: flex; flex-direction: column; gap: 2px; max-width: 560px; }
        .gap-row { display: grid; grid-template-columns: 150px 1fr 56px; gap: 12px; align-items: center; font-size: 12.5px; color: var(--text-secondary); min-height: 22px; }
        .gap-row .num { text-align: right; font-variant-numeric: tabular-nums; color: var(--text-primary); }
        .gap-track { position: relative; height: 8px; }
        .gap-track::before { content: ''; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--border-bright); }
        .gap-track i { position: absolute; top: 0; height: 100%; width: var(--w); border-radius: 3px; background: var(--text-muted); }
        .gap-track i.up { left: 50%; background: var(--accent); opacity: .8; }
        .gap-track i.dn { right: 50%; }
        .gap-total { border-top: 1px solid var(--border); margin-top: 4px; padding-top: 6px; font-weight: 600; color: var(--text-primary); }

        /* ---- PHONE ---- */
        @media (max-width: 760px) {
            .cmdk-btn { padding: 0 9px; }
            .cmdk-btn-text, .cmdk-kbd { display: none; }
            .guide-steps { grid-template-columns: 1fr; gap: 12px; }
            .sec-meta { display: none; }
            .filters-bar { display: grid; grid-template-columns: 1fr 1fr; align-items: end; gap: 10px 12px; flex-direction: initial; }
            .filters-bar .filter-search { grid-column: 1 / -1; max-width: none; }
            .filters-bar .filter-group select, .filters-bar .filter-group input { width: 100%; }
            .filters-bar .result-count { grid-column: 1 / -1; margin: 0; }
            .filters-bar .filter-clear { grid-column: 1 / -1; }
            #stock-modal .modal-header { padding: 12px 14px 8px; gap: 8px 10px; }
            #stock-modal .modal-ticker { font-size: 21px; }
            .modal-headline { order: 2; flex: none; gap: 14px; }
            .mh-item { align-items: flex-end; }
            .mh-item strong { font-size: 16px; }
            .mh-of { display: none; }
            #stock-modal .modal-header > div:first-child { flex: 1 1 0; }
            .modal-headline { order: 2; }
            #stock-modal .modal-close { order: 3; }
            #stock-modal .modal-tools { order: 4; gap: 4px; }
            .mt-pos { min-width: 0; padding: 0 2px; white-space: nowrap; font-size: 11.5px; }
            .mt-lg { display: none; }
            .mt-text { margin-left: 0; padding: 0 9px; }
            #mt-hold { margin-left: auto; }
            .dashboard-header { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; }
            .dashboard-header .header-nav { grid-column: 1 / -1; }
            .dashboard-header .header-right { grid-column: 2; grid-row: 1; }
            .guide { padding: 14px 14px 12px; }
            .guide-steps li { font-size: 12.5px; }
            #stock-modal .modal-score-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            #contrib-visual .contrib-row { grid-template-columns: minmax(0, 1fr) auto; grid-template-areas: "label pts" "bar bar"; }
            #contrib-visual .contrib-label { grid-area: label; flex-direction: row; align-items: baseline; gap: 8px; }
            #contrib-visual .contrib-bar-area { grid-area: bar; }
            #contrib-visual .contrib-pts { grid-area: pts; }
            .pal-overlay { padding: 8px; align-items: flex-start; }
            .pal { max-height: 80vh; }
            .pal-s, .pal-foot { display: none; }
            .cmp-tray { left: 8px; right: 8px; transform: none; max-width: none; bottom: 8px; }
            .cmp-tray-label { display: none; }
            .gap-row { grid-template-columns: 110px 1fr 48px; }
            .cmp-table { min-width: calc(96px + var(--n) * 132px); }
            .cmp-table tbody th { width: 96px; font-size: 12px; padding: 10px 8px; }
            .cmp-table th, .cmp-table td { padding: 10px 8px; }
            .cmp-content { max-height: calc(100vh - 16px); }
            .cmp-overlay { padding: 8px; }
            .trap-row { grid-template-columns: minmax(0, 1fr) 64px 100px; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: 1ms !important; transition-duration: 1ms !important; scroll-behavior: auto !important; }
        }
        @media print {
            .cmp-tray, .toast, .pal-overlay, .guide, .cmdk-btn, .modal-tools { display: none !important; }
        }
"""


def _load_methodology_html() -> str:
    """Read SCREENER_OVERVIEW.md and convert to HTML for embedding."""
    overview_path = ROOT / "SCREENER_OVERVIEW.md"
    if not overview_path.exists():
        return "<p>Methodology document not found.</p>"
    try:
        import markdown
        md_text = overview_path.read_text(encoding="utf-8")
        return markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        # Fallback: wrap raw markdown in <pre> if the markdown library is absent.
        #
        # This fallback shipped silently to the live site for three days in
        # August 2026: `markdown` was missing from requirements.txt, so on a
        # fresh machine the entire Methodology section rendered as unstyled raw
        # markdown - visible '#' and '**' characters - with nothing in any log
        # to say why. Degrading quietly is worse than failing here, because the
        # methodology page is the tool's main claim to being defensible.
        warnings.warn(
            "The 'markdown' package is not installed, so the dashboard's "
            "Methodology section will render as raw unformatted text. "
            "Install it (pip install markdown) and regenerate.",
            RuntimeWarning,
            stacklevel=2,
        )
        print("  [WARN] markdown package missing - Methodology will render unformatted")
        md_text = overview_path.read_text(encoding="utf-8")
        import html as html_mod
        return f"<pre>{html_mod.escape(md_text)}</pre>"


def generate_dashboard(run_dir: Path, output_path: Path = None) -> Path:
    """Generate dashboard HTML for a given run directory.

    Args:
        run_dir: Path to the run directory containing parquet + meta.json
        output_path: Where to write the HTML. Defaults to run_dir/dashboard.html

    Returns:
        Path to the generated HTML file.
    """
    if output_path is None:
        output_path = run_dir / "dashboard.html"

    run_data = load_run_data(run_dir)
    data_json = prepare_dashboard_data(run_data)
    methodology_html = _load_methodology_html()

    # Build a static "Data as of" label from the run start_time
    raw_ts = run_data.get("meta", {}).get("start_time", "")
    if raw_ts:
        try:
            dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            data_timestamp = dt.strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            data_timestamp = raw_ts
    else:
        data_timestamp = ""

    data_version = hashlib.md5(data_json.encode("utf-8")).hexdigest()[:12]
    html = generate_html(methodology_html=methodology_html, data_timestamp=data_timestamp,
                         data_version=data_version)

    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

    # Write companion data file — keeps the HTML lightweight and the bulk of
    # the data (~3 MB) in a separate file that can be cached independently.
    data_js_path = output_path.parent / "dashboard_data.js"
    data_js_path.write_text(f"window.SCREENER_DATA = {data_json};", encoding="utf-8")
    print(f"Data file:           {data_js_path} ({data_js_path.stat().st_size / 1024:.0f} KB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate interactive HTML dashboard")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to run directory (default: latest run)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: <run-dir>/dashboard.html)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _find_latest_run()
        print(f"Using latest run: {run_dir.name}")

    output = Path(args.output) if args.output else None
    generate_dashboard(run_dir, output)


if __name__ == "__main__":
    main()
