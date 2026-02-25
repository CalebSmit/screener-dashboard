#!/usr/bin/env python3
"""
Interactive HTML Dashboard Generator for the Multi-Factor Screener.
===================================================================
Reads run artifacts (parquet + meta.json) and the Excel output to produce
a single self-contained HTML dashboard with Chart.js visualisations,
sortable/filterable tables, and portfolio analytics.

Usage:
    python generate_dashboard.py                        # latest run
    python generate_dashboard.py --run-dir runs/abc123  # specific run
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
        except Exception:
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
        except Exception:
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
                         ("shortRatio", "_short_ratio")]:
            if src in raw.columns and dst not in df.columns:
                raw = raw.rename(columns={src: dst})
                merge_cols.append(dst)
        if len(merge_cols) > 1:
            df = df.merge(raw[merge_cols], on="Ticker", how="left")

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
        except Exception:
            pass

    corr_df = None
    corr_path = run_dir / "06_factor_correlation.parquet"
    if corr_path.exists():
        try:
            corr_df = pd.read_parquet(corr_path)
        except Exception:
            pass

    # Load model portfolio — prefer the real artifact from construct_portfolio()
    port_path = run_dir / "08_model_portfolio.parquet"
    port_df = None
    if port_path.exists():
        try:
            port_df = pd.read_parquet(port_path)
        except Exception:
            pass
    portfolio_data = _load_portfolio_from_excel(df, port_df)

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
        "portfolio": portfolio_data,
        "weights": weights,
        "sens_df": sens_df,
        "corr_df": corr_df,
        "cfg": run_cfg,
    }


def _load_portfolio_from_excel(df: pd.DataFrame, port_df: pd.DataFrame = None) -> dict:
    """Extract model portfolio data.

    If port_df is provided (the real output from construct_portfolio()),
    use it directly. Otherwise fall back to the naive top-25 approximation.
    """
    if port_df is not None and len(port_df) > 0:
        eligible = port_df.sort_values("Rank" if "Rank" in port_df.columns else "Composite",
                                       ascending="Rank" in port_df.columns).copy()
    else:
        # Fallback: naive top-25 (no sector caps)
        eligible = df[df["Value_Trap_Flag"] == False].copy()
        if "Growth_Trap_Flag" in df.columns:
            eligible = eligible[eligible["Growth_Trap_Flag"] == False]
        eligible = eligible.sort_values("Rank").head(25)

    holdings = []
    for _, row in eligible.iterrows():
        holdings.append({
            "rank": _safe(row.get("Rank")),
            "ticker": row["Ticker"],
            "company": row.get("Company", ""),
            "sector": row.get("Sector", "Unknown"),
            "composite": _safe(row.get("Composite")),
            "valuation": _safe(row.get("valuation_score")),
            "quality": _safe(row.get("quality_score")),
            "growth": _safe(row.get("growth_score")),
            "momentum": _safe(row.get("momentum_score")),
            "risk": _safe(row.get("risk_score")),
            "revisions": _safe(row.get("revisions_score")),
            "size": _safe(row.get("size_score")),
            "investment": _safe(row.get("investment_score")),
            "vt": _safe(row.get("Value_Trap_Flag")),
            "gt": _safe(row.get("Growth_Trap_Flag")),
        })

    # Sector allocation
    sector_counts = eligible["Sector"].value_counts().to_dict()

    # Portfolio beta
    avg_beta = round(float(eligible["beta"].mean()), 2) if "beta" in eligible.columns else None

    return {
        "holdings": holdings,
        "sector_counts": sector_counts,
        "avg_beta": avg_beta,
        "avg_composite": round(float(eligible["Composite"].mean()), 1),
        "num_stocks": len(eligible),
    }


# ---------------------------------------------------------------------------
# Prepare JSON data for the dashboard
# ---------------------------------------------------------------------------

def prepare_dashboard_data(run_data: dict) -> str:
    """Convert run data into a JSON string for embedding in HTML."""
    df = run_data["df"]
    meta = run_data["meta"]
    portfolio = run_data["portfolio"]

    weights = run_data.get("weights", {})

    # --- Raw metric columns, percentile columns, contribution columns ---
    raw_metrics = [
        "ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales", "pb_ratio",
        "roic", "gross_profit_assets", "debt_equity", "net_debt_to_ebitda",
        "piotroski_f_score", "accruals", "operating_leverage", "beneish_m_score",
        "roe", "roa", "equity_ratio",
        "forward_eps_growth", "peg_ratio", "revenue_growth", "revenue_cagr_3yr", "sustainable_growth",
        "return_12_1", "return_6m", "jensens_alpha",
        "volatility", "beta", "sharpe_ratio", "sortino_ratio", "max_drawdown_1y",
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

    # SPX sector weights for comparison
    spx_weights = {
        "Information Technology": 29.0, "Financials": 14.0, "Health Care": 12.0,
        "Consumer Discretionary": 10.0, "Industrials": 9.0, "Communication Services": 9.0,
        "Consumer Staples": 6.0, "Energy": 4.0, "Utilities": 3.0,
        "Real Estate": 2.0, "Materials": 2.0,
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
        "analyst_surprise": {"label": "Analyst Surprise", "fmt": "pct", "category": "revisions"},
        "price_target_upside": {"label": "Price Target Upside", "fmt": "pct", "category": "revisions"},
        "earnings_acceleration": {"label": "Earnings Accel.", "fmt": "ratio", "category": "revisions"},
        "consecutive_beat_streak": {"label": "Beat Score", "fmt": "int", "category": "revisions"},
        "short_interest_ratio": {"label": "Short Interest Ratio", "fmt": "ratio", "category": "revisions"},
        "size_log_mcap": {"label": "Size (-log MCap)", "fmt": "ratio", "category": "size"},
        "asset_growth": {"label": "Asset Growth", "fmt": "pct", "category": "investment"},
    }

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

    # Extract trap filter thresholds from config for the AI prompt
    _cfg = run_data.get("cfg", {})
    _vtf = _cfg.get("value_trap_filters", {})
    _gtf = _cfg.get("growth_trap_filters", {})
    config_traps = {
        "vt_quality": _vtf.get("quality_floor_percentile", 30),
        "vt_momentum": _vtf.get("momentum_floor_percentile", 30),
        "vt_revisions": _vtf.get("revisions_floor_percentile", 30),
        "gt_growth": _gtf.get("growth_ceiling_percentile", 70),
    }

    dashboard_json = {
        "kpis": kpis,
        "portfolio": portfolio,
        "table_data": table_data,
        "stock_detail": stock_detail,
        "weights": weights,
        "metric_meta": metric_meta,
        "sectors": sectors,
        "sector_composition": sector_composition,
        "histogram": {"labels": hist_labels, "values": [int(v) for v in hist_values]},
        "vt_by_sector": vt_by_sector,
        "gt_by_sector": gt_by_sector,
        "sector_distributions": sector_distributions,
        "spx_weights": spx_weights,
        "factor_correlation": factor_corr_data,
        "weight_sensitivity": weight_sens_data,
        "data_quality": data_quality_summary,
        "config_traps": config_traps,
    }

    return json.dumps(dashboard_json, default=str)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(data_json: str, methodology_html: str = "") -> str:
    """Build the complete dashboard HTML string."""
    # Escape braces in methodology_html so f-string doesn't choke
    methodology_escaped = methodology_html.replace("{", "{{").replace("}", "}}")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Factor Screener Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1" integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" crossorigin="anonymous"></script>
    <style>
{_css()}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header">
            <div class="header-left">
                <h1>Multi-Factor Screener</h1>
                <span class="run-info" id="run-info"></span>
            </div>
            <div class="header-right">
                <button class="methodology-btn" onclick="openMethodology()">Methodology</button>
            </div>
        </header>

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

        <!-- Model Portfolio -->
        <section class="section collapsible-section collapsed" id="sec-portfolio">
            <div class="section-header" onclick="toggleSection('sec-portfolio')">
                <h2 class="section-title" style="margin:0">Model Portfolio</h2>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <div class="kpi-row" id="portfolio-kpis"></div>
                <div class="chart-row" style="margin-bottom:var(--gap)">
                    <div class="chart-container" style="flex:1">
                        <h3 class="chart-title">Portfolio Sector Allocation vs S&P 500</h3>
                        <canvas id="sector-alloc-chart"></canvas>
                    </div>
                </div>
                <div class="table-section" id="portfolio-table"></div>
            </div>
        </section>

        <!-- Factor Analytics Section -->
        <section class="section collapsible-section" id="sec-analytics">
            <div class="section-header" onclick="toggleSection('sec-analytics')">
                <h2 class="section-title" style="margin:0">Factor Analytics</h2>
                <svg class="section-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
            <div class="section-body">
                <div class="chart-row">
                    <div class="chart-container" style="flex:1">
                        <div class="chart-title-row">
                            <h3 class="chart-title" style="margin-bottom:0">Factor Scores by Sector</h3>
                            <div style="display:flex;gap:8px;align-items:center">
                                <select id="sector-dist-factor" onchange="updateSectorDist()">
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
                                <div class="toggle-btns">
                                    <button class="toggle-btn active" id="btn-median" onclick="setSectorStat('median')">Median</button>
                                    <button class="toggle-btn" id="btn-mean" onclick="setSectorStat('mean')">Average</button>
                                </div>
                            </div>
                        </div>
                        <canvas id="sector-dist-chart"></canvas>
                    </div>
                    <div class="chart-container" style="flex:0.5">
                        <div class="chart-title-row">
                            <h3 class="chart-title" style="margin-bottom:0">Trap Rate by Sector</h3>
                            <div class="toggle-btns">
                                <button class="toggle-btn active" id="btn-trap-vt" onclick="setTrapType('vt')">Value</button>
                                <button class="toggle-btn" id="btn-trap-gt" onclick="setTrapType('gt')">Growth</button>
                            </div>
                        </div>
                        <canvas id="vt-chart"></canvas>
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
                        <h3 class="chart-title">How Stable Is the Portfolio?</h3>
                        <p class="chart-desc">If we nudge each factor's weight &plusmn;5%, how much does the top-20 list change? <strong>Higher Jaccard = more stable.</strong> Green (&ge;0.85) means almost no change. Red (&lt;0.70) means the ranking is sensitive to that weight.</p>
                        <div id="sensitivity-table"></div>
                    </div>
                    <div class="chart-container" style="flex:1">
                        <h3 class="chart-title">Are the Factors Independent?</h3>
                        <p class="chart-desc">Spearman correlations between the 8 factor scores. <strong>Low values (green) = independent signals.</strong> High values (red) mean two factors are measuring similar things, reducing the effective number of independent dimensions.</p>
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
                <div class="filters-bar">
                    <div class="filter-group">
                        <label for="filter-sector">Sector</label>
                        <select id="filter-sector"><option value="all">All Sectors</option></select>
                    </div>
                    <div class="filter-group">
                        <label for="filter-vt">Trap Flags</label>
                        <select id="filter-vt">
                            <option value="all">All</option>
                            <option value="clean">Clean Only</option>
                            <option value="vt">Value Traps</option>
                            <option value="gt">Growth Traps</option>
                            <option value="any">Any Flagged</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label for="filter-comp-min">Composite Min</label>
                        <input type="number" id="filter-comp-min" min="0" max="100" value="0" step="5">
                    </div>
                    <div class="filter-group">
                        <label for="filter-search">Search</label>
                        <input type="text" id="filter-search" placeholder="Ticker or company...">
                    </div>
                    <div class="filter-group">
                        <span class="result-count" id="result-count"></span>
                    </div>
                </div>
                <div class="table-section">
                    <table class="data-table" id="universe-table">
                        <thead><tr>
                            <th data-sort="Rank">Rank</th>
                            <th data-sort="Ticker">Ticker</th>
                            <th data-sort="Company">Company</th>
                            <th data-sort="Sector">Sector</th>
                            <th data-sort="Composite">Composite</th>
                            <th data-sort="valuation_score">Val</th>
                            <th data-sort="quality_score">Qual</th>
                            <th data-sort="growth_score">Grow</th>
                            <th data-sort="momentum_score">Mom</th>
                            <th data-sort="risk_score">Risk</th>
                            <th data-sort="revisions_score">Rev</th>
                            <th data-sort="size_score">Size</th>
                            <th data-sort="investment_score">Inv</th>
                            <th data-sort="Value_Trap_Flag">Flags</th>
                        </tr></thead>
                        <tbody id="universe-tbody"></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Stock Detail Modal -->
        <div class="modal-overlay" id="stock-modal" style="display:none" onclick="if(event.target===this)closeModal()">
            <div class="modal-content">
                <div class="modal-header">
                    <div>
                        <h2 class="modal-ticker" id="modal-ticker"></h2>
                        <span class="modal-company" id="modal-company"></span>
                        <span class="modal-sector" id="modal-sector"></span>
                    </div>
                    <button class="modal-close" onclick="closeModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <!-- Score summary row -->
                    <div class="modal-score-row" id="modal-score-row"></div>

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

                    <!-- Contribution breakdown -->
                    <div class="collapsible" id="section-contribution">
                        <div class="collapsible-header" onclick="toggleSection('section-contribution')">
                            <span>Score Contribution Breakdown</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body">
                            <div class="modal-chart-section">
                                <p class="modal-chart-desc">Each factor is scored 0–100, then multiplied by its weight to produce contribution points. The contributions add up to the composite score.</p>
                                <div id="contrib-visual"></div>
                                <div class="contrib-total-row" id="contrib-total"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Category detail sections -->
                    <div class="collapsible" id="section-categories">
                        <div class="collapsible-header" onclick="toggleSection('section-categories')">
                            <span>Category Details</span><span class="collapsible-chevron">&#9660;</span>
                        </div>
                        <div class="collapsible-body" id="modal-categories"></div>
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

        <footer class="dashboard-footer">
            Multi-Factor Screener Dashboard &bull; Auto-generated <span id="gen-time"></span>
        </footer>
    </div>

    <!-- AI Chat Panel -->
    <button class="chat-fab" id="chat-fab" onclick="toggleChat()" title="Ask AI about the data">
        <svg class="chat-fab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        <svg class="chat-fab-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
    </button>

    <div class="chat-panel" id="chat-panel">
        <div class="chat-resize-handle" id="chat-resize-handle" title="Drag to resize, double-click to reset"></div>
        <div class="chat-header">
            <div class="chat-header-left">
                <span class="chat-header-dot"></span>
                <span class="chat-header-title">Screener AI</span>
                <span class="chat-header-model" id="chat-header-model"></span>
            </div>
            <div class="chat-header-actions">
                <button class="chat-header-btn" onclick="clearChat()" title="Clear conversation">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14H7L5 6"/>
                        <path d="M10 11v6"/><path d="M14 11v6"/>
                    </svg>
                </button>
                <button class="chat-header-btn" onclick="openApiKeyDialog()" title="API Key Settings">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
                        <circle cx="12" cy="12" r="3"/>
                    </svg>
                </button>
                <button class="chat-header-btn" onclick="toggleChat()" title="Close">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </button>
            </div>
        </div>

        <div class="chat-messages" id="chat-messages"></div>

        <div class="chat-input-area" id="chat-input-area">
            <div class="chat-suggestions" id="chat-suggestions"></div>
            <div class="chat-input-row">
                <textarea class="chat-input" id="chat-input"
                    placeholder="Ask about any stock, metric, or strategy..."
                    rows="1"
                    onkeydown="handleChatKeydown(event)"
                    oninput="autoResizeInput(this)"></textarea>
                <button class="chat-send-btn" id="chat-send-btn" onclick="sendMessage()" title="Send">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="22" y1="2" x2="11" y2="13"/>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                    </svg>
                </button>
            </div>
        </div>

        <div class="chat-api-dialog" id="chat-api-dialog" style="display:none">
            <div class="chat-api-dialog-content">
                <h3>Chat Settings</h3>
                <label class="chat-api-label">OpenAI API Key</label>
                <p>Your key is stored only in your browser and sent only to OpenAI.</p>
                <input type="password" class="chat-api-input" id="chat-api-input"
                    placeholder="sk-..." autocomplete="off">
                <label class="chat-api-label" style="margin-top:12px">Model</label>
                <select class="chat-api-select" id="chat-model-select">
                    <option value="gpt-4o-mini">GPT-4o Mini &mdash; fast &amp; cheap (~$0.001/q)</option>
                    <option value="gpt-4o">GPT-4o &mdash; smarter (~$0.01/q)</option>
                    <option value="gpt-4.1-mini">GPT-4.1 Mini &mdash; fast &amp; smart (~$0.003/q)</option>
                    <option value="gpt-4.1">GPT-4.1 &mdash; latest flagship (~$0.02/q)</option>
                </select>
                <div class="chat-api-dialog-actions">
                    <button class="chat-api-btn chat-api-btn-secondary" onclick="closeApiKeyDialog()">Cancel</button>
                    <button class="chat-api-btn chat-api-btn-primary" onclick="saveSettings()">Save</button>
                </div>
                <p class="chat-api-hint">Get a key at <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener">platform.openai.com/api-keys</a></p>
            </div>
        </div>
    </div>

    <script>
    // =====================================================================
    // EMBEDDED DATA
    // =====================================================================
    const D = {data_json};

    // =====================================================================
    // COLOUR PALETTE
    // =====================================================================
    const COLORS = ['#58a6ff','#f0883e','#3fb950','#f85149','#bc8cff',
                    '#d2a8ff','#ff7b72','#79c0ff','#d29922','#56d4dd',
                    '#a5d6ff'];
    const POS = '#3fb950', NEG = '#f85149';

    // ---- Chart.js global dark-theme defaults ----
    Chart.defaults.color = '#7d8590';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = "'DM Sans', sans-serif";

    // =====================================================================
    // UTILITIES
    // =====================================================================
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
        const p = D.portfolio;
        const html = [
            kpiCard('Universe', k.universe_size, 'stocks scored'),
            kpiCard('Value Traps', k.value_traps, `${{(k.value_traps/k.universe_size*100).toFixed(0)}}% flagged`),
            kpiCard('Growth Traps', k.growth_traps || 0, `${{((k.growth_traps||0)/k.universe_size*100).toFixed(0)}}% flagged`),
        ].join('');
        document.getElementById('kpi-row').innerHTML = html;
        const ts = k.run_timestamp ? new Date(k.run_timestamp) : null;
        const fmtDate = ts ? ts.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }}) : '';
        const fmtTime = ts ? ts.toLocaleTimeString('en-US', {{ hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }}) : '';
        document.getElementById('run-info').textContent = ts ? `Last updated ${{fmtDate}} at ${{fmtTime}}` : '';
        document.getElementById('gen-time').textContent = new Date().toLocaleString();
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
        const top5 = D.portfolio.holdings.slice(0, 5);
        const catKeys = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        const catLabels = {{ valuation:'Val', quality:'Qual', growth:'Grow', momentum:'Mom', risk:'Risk', revisions:'Rev', size:'Size', investment:'Inv' }};
        const catColors = {{
            valuation: '#58a6ff', quality: '#3fb950', growth: '#f0883e',
            momentum: '#f85149', risk: '#bc8cff', revisions: '#d29922',
            size: '#56d4dd', investment: '#d2a8ff'
        }};

        // Sector color mapping for badges
        const sectorColors = {{
            'Information Technology': '#58a6ff',
            'Health Care': '#3fb950',
            'Financials': '#d29922',
            'Consumer Discretionary': '#f0883e',
            'Communication Services': '#bc8cff',
            'Industrials': '#79c0ff',
            'Consumer Staples': '#56d4dd',
            'Energy': '#f85149',
            'Utilities': '#a5d6ff',
            'Real Estate': '#d2a8ff',
            'Materials': '#ff7b72'
        }};

        document.getElementById('top5-row').innerHTML = top5.map((h, i) => {{
            const bars = catKeys.map(c => {{
                const val = h[c] || 0;
                const color = catColors[c];
                return `<div class="top5-factor">
                    <div class="top5-factor-label">${{catLabels[c]}}</div>
                    <div class="top5-factor-bar"><div class="top5-factor-fill" style="width:${{val}}%;background:${{color}}"></div></div>
                    <div class="top5-factor-val">${{fmt(val,'score')}}</div>
                </div>`;
            }}).join('');

            const sc = sectorColors[h.sector] || '#7d8590';

            return `<div class="top5-card" onclick="openStockDetail('${{h.ticker}}')">
                <div class="top5-rank-bar">#${{h.rank}}</div>
                <div class="top5-header">
                    <span class="top5-ticker">${{h.ticker}}</span>
                </div>
                <div class="top5-company">${{h.company}}</div>
                <div class="top5-sector" style="--sector-color:${{sc}}">
                    <span class="top5-sector-dot" style="background:${{sc}}"></span>${{h.sector}}
                </div>
                <div class="top5-composite-row">
                    <span class="top5-composite-label">Composite</span>
                    <span class="top5-composite">${{fmt(h.composite,'score')}}</span>
                </div>
                <div class="top5-factors">${{bars}}</div>
                <div class="top5-cta">Click for full breakdown</div>
            </div>`;
        }}).join('');
    }}

    // =====================================================================
    // MODEL PORTFOLIO TABLE (all holdings)
    // =====================================================================
    function renderPortfolio() {{
        const holdings = D.portfolio.holdings;
        const p = D.portfolio;
        if (!holdings || holdings.length === 0) return;

        // Portfolio KPI cards
        const kpis = [
            kpiCard('Holdings', p.num_stocks, 'stocks selected'),
            kpiCard('Avg Composite', p.avg_composite, 'portfolio average'),
            kpiCard('Portfolio Beta', p.avg_beta !== null ? p.avg_beta.toFixed(2) : '\u2014', 'avg systematic risk'),
            kpiCard('Sectors', Object.keys(p.sector_counts).length, 'of 11 represented'),
        ].join('');
        document.getElementById('portfolio-kpis').innerHTML = kpis;

        let html = '<table class="data-table"><thead><tr>';
        html += '<th>Rank</th><th>Ticker</th><th>Company</th><th>Sector</th>';
        html += '<th>Composite</th><th>Val</th><th>Qual</th><th>Grow</th>';
        html += '<th>Mom</th><th>Risk</th><th>Rev</th><th>Size</th><th>Inv</th>';
        html += '</tr></thead><tbody>';

        holdings.forEach(function(h) {{
            html += '<tr>';
            html += '<td class="num">' + h.rank + '</td>';
            html += '<td class="ticker ticker-link" onclick="openStockDetail(\\'' + h.ticker + '\\')">' + h.ticker + '</td>';
            html += '<td>' + (h.company || '') + '</td>';
            html += '<td>' + h.sector + '</td>';
            html += '<td class="num">' + fmt(h.composite,'score') + '</td>';
            html += '<td class="num">' + fmt(h.valuation,'score') + '</td>';
            html += '<td class="num">' + fmt(h.quality,'score') + '</td>';
            html += '<td class="num">' + fmt(h.growth,'score') + '</td>';
            html += '<td class="num">' + fmt(h.momentum,'score') + '</td>';
            html += '<td class="num">' + fmt(h.risk,'score') + '</td>';
            html += '<td class="num">' + fmt(h.revisions,'score') + '</td>';
            html += '<td class="num">' + fmt(h.size,'score') + '</td>';
            html += '<td class="num">' + fmt(h.investment,'score') + '</td>';
            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById('portfolio-table').innerHTML = html;
    }}

    // =====================================================================
    // SECTOR ALLOCATION CHART (portfolio vs SPX)
    // =====================================================================
    function renderSectorAlloc() {{
        const sc = D.portfolio.sector_counts;
        const total = D.portfolio.num_stocks;
        const allSectors = [...new Set([...Object.keys(sc), ...Object.keys(D.spx_weights)])].sort();
        const portfolioPcts = allSectors.map(s => ((sc[s] || 0) / total * 100).toFixed(1));
        const spxPcts = allSectors.map(s => D.spx_weights[s] || 0);
        const labels = allSectors.map(s => s.length > 18 ? s.slice(0, 16) + '…' : s);

        new Chart(document.getElementById('sector-alloc-chart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Portfolio %', data: portfolioPcts, backgroundColor: COLORS[0] + 'CC', borderRadius: 3 }},
                    {{ label: 'S&P 500 %', data: spxPcts, backgroundColor: COLORS[1] + 'CC', borderRadius: 3 }},
                ]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ position: 'top', labels: {{ color: '#7d8590' }} }} }},
                scales: {{
                    x: {{ beginAtZero: true, title: {{ display: true, text: 'Weight %', color: '#7d8590' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#7d8590' }} }},
                    y: {{ ticks: {{ font: {{ size: 11 }}, color: '#7d8590' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
                }}
            }}
        }});
    }}

    // =====================================================================
    // COMPOSITE HISTOGRAM
    // =====================================================================
    // VALUE TRAP BAR CHART
    // =====================================================================
    let trapChart = null;
    let currentTrapType = 'vt';

    function renderTrapChart() {{
        const dataSource = currentTrapType === 'gt' ? D.gt_by_sector : D.vt_by_sector;
        const labelText = currentTrapType === 'gt' ? 'Growth Trap Rate %' : 'Value Trap Rate %';
        const barColor = currentTrapType === 'gt' ? '#f0883e' : null;

        const sectors = Object.keys(dataSource).sort((a,b) => dataSource[b].rate - dataSource[a].rate);
        const rates = sectors.map(s => dataSource[s].rate);
        const labels = sectors.map(s => s.length > 18 ? s.slice(0,16)+'…' : s);

        if (trapChart) trapChart.destroy();

        trapChart = new Chart(document.getElementById('vt-chart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: labelText,
                    data: rates,
                    backgroundColor: barColor
                        ? rates.map(r => r > 25 ? barColor + 'CC' : r > 15 ? '#d29922CC' : POS + 'CC')
                        : rates.map(r => r > 25 ? NEG + 'CC' : r > 15 ? '#d29922CC' : POS + 'CC'),
                    borderRadius: 3,
                }}]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ beginAtZero: true, max: 40, title: {{ display: true, text: 'Flag Rate %', color: '#7d8590' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#7d8590' }} }},
                    y: {{ ticks: {{ font: {{ size: 11 }}, color: '#7d8590' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
                }}
            }}
        }});
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
        'Composite': '#58a6ff',
        'valuation_score': '#58a6ff', 'quality_score': '#3fb950',
        'growth_score': '#f0883e', 'momentum_score': '#f85149',
        'risk_score': '#bc8cff', 'revisions_score': '#d29922',
        'size_score': '#56d4dd', 'investment_score': '#d2a8ff'
    }};

    function updateSectorDist() {{
        const factor = document.getElementById('sector-dist-factor').value;
        const stat = currentSectorStat;
        const statLabel = stat === 'median' ? 'Median' : 'Average';
        const dist = D.sector_distributions[factor];
        const sectors = Object.keys(dist).sort((a,b) => (dist[b][stat] || 0) - (dist[a][stat] || 0));
        const values = sectors.map(s => dist[s][stat] || 0);
        const labels = sectors.map(s => s.length > 18 ? s.slice(0,16)+'…' : s);
        const color = FACTOR_COLOR_MAP[factor] || COLORS[0];

        if (sectorDistChart) {{
            sectorDistChart.data.labels = labels;
            sectorDistChart.data.datasets[0].data = values;
            sectorDistChart.data.datasets[0].label = statLabel + ' Score';
            sectorDistChart.data.datasets[0].backgroundColor = color + 'CC';
            sectorDistChart.update('none');
        }} else {{
            sectorDistChart = new Chart(document.getElementById('sector-dist-chart'), {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: statLabel + ' Score',
                        data: values,
                        backgroundColor: color + 'CC',
                        borderRadius: 3,
                    }}]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ beginAtZero: true, max: 70, grid: {{ color: 'rgba(255,255,255,0.04)' }}, ticks: {{ color: '#7d8590' }} }},
                        y: {{ ticks: {{ font: {{ size: 11 }}, color: '#7d8590' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }}
                    }}
                }}
            }});
        }}
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
                `<div class="peer-search-item" onmousedown="addPeer('${{r.Ticker}}')">
                    <span class="peer-search-ticker">${{r.Ticker}}</span>
                    <span class="peer-search-company">${{r.Company || ''}}</span>
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

    function setupFilters() {{
        // Populate sector dropdown
        const sel = document.getElementById('filter-sector');
        D.sectors.forEach(s => {{
            const opt = document.createElement('option');
            opt.value = s; opt.textContent = s;
            sel.appendChild(opt);
        }});

        // Event listeners
        sel.addEventListener('change', applyFilters);
        document.getElementById('filter-vt').addEventListener('change', applyFilters);
        document.getElementById('filter-comp-min').addEventListener('input', applyFilters);
        document.getElementById('filter-search').addEventListener('input', applyFilters);
    }}

    function applyFilters() {{
        const sector = document.getElementById('filter-sector').value;
        const vt = document.getElementById('filter-vt').value;
        const compMin = parseFloat(document.getElementById('filter-comp-min').value) || 0;
        const search = document.getElementById('filter-search').value.toLowerCase().trim();

        tableState.filtered = tableState.data.filter(row => {{
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
        }});

        sortTable(tableState.sortCol, false);
    }}

    function sortTable(col, toggle = true) {{
        if (toggle) {{
            if (tableState.sortCol === col) {{
                tableState.sortDir = tableState.sortDir === 'asc' ? 'desc' : 'asc';
            }} else {{
                tableState.sortCol = col;
                tableState.sortDir = (col === 'Rank' || col === 'Ticker') ? 'asc' : 'desc';
            }}
        }}
        tableState.filtered.sort((a, b) => {{
            let av = a[tableState.sortCol], bv = b[tableState.sortCol];
            if (av === null) av = tableState.sortDir === 'asc' ? Infinity : -Infinity;
            if (bv === null) bv = tableState.sortDir === 'asc' ? Infinity : -Infinity;
            if (typeof av === 'string') return tableState.sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
            return tableState.sortDir === 'asc' ? av - bv : bv - av;
        }});
        renderUniverseTable();
    }}

    function renderUniverseTable() {{
        const s = tableState;

        // Update sort indicators
        document.querySelectorAll('#universe-table th').forEach(th => {{
            const col = th.dataset.sort;
            th.classList.toggle('sorted', col === s.sortCol);
            th.setAttribute('data-dir', col === s.sortCol ? s.sortDir : '');
        }});

        const tbody = document.getElementById('universe-tbody');
        tbody.innerHTML = s.filtered.map(row => {{
            const trapClass = (row.Value_Trap_Flag || row.Growth_Trap_Flag) ? ' class="vt-row"' : '';
            const trapIcons = [];
            if (row.Value_Trap_Flag) trapIcons.push('VT');
            if (row.Growth_Trap_Flag) trapIcons.push('GT');
            const trapDisplay = trapIcons.length ? trapIcons.join('/') : '✓';
            return `<tr${{trapClass}}>
                <td class="num">${{row.Rank}}</td>
                <td class="ticker ticker-link" onclick="openStockDetail('${{row.Ticker}}')">${{row.Ticker}}</td>
                <td>${{row.Company || ''}}</td>
                <td>${{row.Sector}}</td>
                <td class="num">${{fmt(row.Composite,'score')}}</td>
                <td class="num">${{fmt(row.valuation_score,'score')}}</td>
                <td class="num">${{fmt(row.quality_score,'score')}}</td>
                <td class="num">${{fmt(row.growth_score,'score')}}</td>
                <td class="num">${{fmt(row.momentum_score,'score')}}</td>
                <td class="num">${{fmt(row.risk_score,'score')}}</td>
                <td class="num">${{fmt(row.revisions_score,'score')}}</td>
                <td class="num">${{fmt(row.size_score,'score')}}</td>
                <td class="num">${{fmt(row.investment_score,'score')}}</td>
                <td class="vt-cell">${{trapDisplay}}</td>
            </tr>`;
        }}).join('');

        document.getElementById('result-count').textContent =
            `${{s.filtered.length}} of ${{s.data.length}} stocks`;
    }}

    // Sort click handler
    document.querySelectorAll('#universe-table th[data-sort]').forEach(th => {{
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => sortTable(th.dataset.sort));
    }});

    // =====================================================================
    // STOCK DETAIL MODAL
    // =====================================================================
    const CAT_COLORS = {{
        valuation: '#58a6ff', quality: '#3fb950', growth: '#f0883e',
        momentum: '#f85149', risk: '#bc8cff', revisions: '#d29922',
        size: '#56d4dd', investment: '#d2a8ff'
    }};
    const CAT_LABELS = {{
        valuation: 'Valuation', quality: 'Quality', growth: 'Growth',
        momentum: 'Momentum', risk: 'Risk', revisions: 'Revisions',
        size: 'Size', investment: 'Investment'
    }};
    function openStockDetail(ticker) {{
        const s = D.stock_detail[ticker];
        if (!s) return;

        document.getElementById('modal-ticker').textContent = ticker;
        document.getElementById('modal-company').textContent = s.company;
        document.getElementById('modal-sector').textContent = s.sector;

        // Score summary cards
        const cats = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        let scoreHtml = `<div class="modal-score-card composite">
            <div class="modal-score-label">Composite</div>
            <div class="modal-score-val">${{fmt(s.composite,'score')}}</div>
            <div class="modal-score-sub">Rank #${{s.rank}} of ${{D.kpis.universe_size}}</div>
        </div>`;
        cats.forEach(c => {{
            const score = s.cat_scores[c];
            const contrib = s.contrib[c];
            const weight = D.weights.factor_weights ? D.weights.factor_weights[c] : '—';
            scoreHtml += `<div class="modal-score-card">
                <div class="modal-score-label">${{CAT_LABELS[c]}}</div>
                <div class="modal-score-val" style="color:${{CAT_COLORS[c]}}">${{fmt(score,'score')}}</div>
                <div class="modal-score-sub">${{fmt(contrib,'score')}} pts (${{weight}}% wt)</div>
            </div>`;
        }});
        document.getElementById('modal-score-row').innerHTML = scoreHtml;

        // Analyst price targets
        renderPriceTargets(s);

        // Company snapshot (financials) + peer comparison
        renderCompanySnapshot(s);
        renderPeerComparison(ticker, s);

        // Data provenance
        renderProvenance(s);

        // Contribution breakdown visual
        renderContribVisual(s, cats);

        // Category detail sections with metric breakdowns
        renderCategoryDetails(ticker, s, cats);

        document.getElementById('stock-modal').style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }}

    function closeModal() {{
        document.getElementById('stock-modal').style.display = 'none';
        document.body.style.overflow = '';
    }}

    function toggleSection(id) {{
        const el = document.getElementById(id);
        if (el) el.classList.toggle('collapsed');
    }}

    // ESC to close any open modal
    document.addEventListener('keydown', e => {{
        if (e.key === 'Escape') {{
            closeModal();
            closeMethodology();
            const cp = document.getElementById('chat-panel');
            if (cp && cp.classList.contains('open')) toggleChat();
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
            {{ label: 'Size & Valuation', color: '#58a6ff', items: [
                {{ label: 'Market Cap',      value: fmtBig(f.market_cap) }},
                {{ label: 'Enterprise Value', value: fmtBig(f.enterprise_value) }},
                {{ label: 'EPS (TTM / Fwd)', value:
                    (f.trailing_eps !== null ? '$' + f.trailing_eps.toFixed(2) : '\u2014') + ' / ' +
                    (f.forward_eps !== null ? '$' + f.forward_eps.toFixed(2) : '\u2014') }},
            ]}},
            {{ label: 'Profitability', color: '#3fb950', items: [
                {{ label: 'Revenue (LTM)',   value: fmtBig(f.revenue),
                   sub: f.revenue_growth_yoy !== null ? fmtPctChg(f.revenue_growth_yoy) + ' YoY' : '' }},
                {{ label: 'Net Income',      value: fmtBig(f.net_income),
                   sub: f.ni_growth_yoy !== null ? fmtPctChg(f.ni_growth_yoy) + ' YoY' : '' }},
                {{ label: 'EBITDA',          value: fmtBig(f.ebitda) }},
                {{ label: 'Gross Margin',    value: fmtPct2(f.gross_margin) }},
                {{ label: 'Net Margin',      value: fmtPct2(f.net_margin) }},
            ]}},
            {{ label: 'Cash Flow & Leverage', color: '#f0883e', items: [
                {{ label: 'Free Cash Flow',  value: fmtBig(f.fcf) }},
                {{ label: 'Total Debt',      value: fmtBig(f.total_debt) }},
                {{ label: 'Cash & Equiv.',   value: fmtBig(f.total_cash) }},
                {{ label: 'Net Debt',        value: fmtBig(f.net_debt) }},
            ]}},
            {{ label: 'Shareholder', color: '#bc8cff', items: [
                {{ label: 'Dividend Yield',  value: f.dividend_yield !== null ? fmtPct2(f.dividend_yield) : 'None' }},
                {{ label: 'Payout Ratio',    value: f.payout_ratio !== null ? fmtPct2(f.payout_ratio) : '\u2014' }},
                {{ label: 'Shares Out',      value: fmtShares(f.shares_outstanding) }},
            ]}},
        ];

        // Trading group (conditional items)
        const tradingItems = [];
        if (f.avg_daily_dollar_vol !== null) tradingItems.push({{ label: 'Avg Daily $ Vol', value: fmtBig(f.avg_daily_dollar_vol) }});
        if (f.short_ratio !== null) tradingItems.push({{ label: 'Short Interest', value: f.short_ratio.toFixed(1) + ' days to cover' }});
        if (tradingItems.length > 0) groups.push({{ label: 'Trading', color: '#d29922', items: tradingItems }});

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

        // Higher is better (green >= self, red < 80% of self)
        const higherBetter = (v, selfV) => {{
            if (v === null || v === undefined || selfV === null || selfV === undefined) return '';
            return v >= selfV ? 'color:#3fb950' : v < selfV * 0.8 ? 'color:#f85149' : '';
        }};
        // Lower is better (green <= self, red > 120% of self)
        const lowerBetter = (v, selfV) => {{
            if (v === null || v === undefined || selfV === null || selfV === undefined) return '';
            return v <= selfV ? 'color:#3fb950' : v > selfV * 1.2 ? 'color:#f85149' : '';
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
                    <span class="peer-ticker">${{r.ticker}}</span>
                    ${{r.isSelf ? '<span class="peer-you">YOU</span>' : ''}}
                </td>
                <td class="peer-td-num">${{fmtBig(r.mcap)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : lowerBetter(r.pe_ratio, self.pe_ratio)}}">${{fmtPE(r.pe_ratio)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : higherBetter(r.rev_growth, self.rev_growth)}}">${{fmtRG(r.rev_growth)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : higherBetter(r.net_margin, self.net_margin)}}">${{fmtPct1(r.net_margin)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : higherBetter(profitMetric, useROE ? self.roe : self.roic)}}">${{fmtPct1(profitMetric)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : lowerBetter(r.debt_equity, self.debt_equity)}}">${{fmtDE(r.debt_equity)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : higherBetter(r.div_yield, self.div_yield)}}">${{fmtPct1(r.div_yield)}}</td>
                <td class="peer-td-num" style="${{r.isSelf ? '' : higherBetter(r.fcf_yield, self.fcf_yield)}}">${{fmtPct1(r.fcf_yield)}}</td>
                <td class="peer-td-action">${{r.isSelf ? '' : `<button class="peer-remove-btn" onclick="removePeer('${{r.ticker}}')" title="Remove">&times;</button>`}}</td>
            </tr>`;
        }});

        html += '</tbody></table></div></div>';
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

    function renderContribVisual(s, cats) {{
        const fw = D.weights.factor_weights || {{}};
        let html = '';
        let runningTotal = 0;

        cats.forEach(c => {{
            const score = s.cat_scores[c];
            const contrib = s.contrib[c] || 0;
            const weight = fw[c] || 0;
            const maxContrib = weight; // max contribution = weight * (100/100)
            const fillPct = maxContrib > 0 ? (contrib / maxContrib) * 100 : 0;
            runningTotal += contrib;

            // Score quality label
            let qualLabel = '', qualClass = '';
            if (score !== null && score !== undefined) {{
                if (score >= 75) {{ qualLabel = 'Strong'; qualClass = 'qual-strong'; }}
                else if (score >= 50) {{ qualLabel = 'Average'; qualClass = 'qual-avg'; }}
                else if (score >= 25) {{ qualLabel = 'Weak'; qualClass = 'qual-weak'; }}
                else {{ qualLabel = 'Very Weak'; qualClass = 'qual-vweak'; }}
            }}

            html += `<div class="contrib-row">
                <div class="contrib-label">
                    <span class="contrib-cat-dot" style="background:${{CAT_COLORS[c]}}"></span>
                    <span class="contrib-cat-name">${{CAT_LABELS[c]}}</span>
                    <span class="contrib-cat-weight">${{weight}}% weight</span>
                </div>
                <div class="contrib-bar-area">
                    <div class="contrib-bar-track">
                        <div class="contrib-bar-fill" style="width:${{fillPct.toFixed(1)}}%;background:${{CAT_COLORS[c]}}">
                            ${{fillPct > 15 ? `<span class="contrib-bar-inner-label">${{contrib.toFixed(1)}}</span>` : ''}}
                        </div>
                        ${{fillPct <= 15 ? `<span class="contrib-bar-outer-label">${{contrib.toFixed(1)}}</span>` : ''}}
                        <div class="contrib-bar-max-marker" style="left:100%" title="Max possible: ${{maxContrib}} pts"></div>
                    </div>
                    <div class="contrib-bar-annotation">
                        <span class="contrib-score-val">Score: ${{fmt(score,'score')}}/100</span>
                        <span class="contrib-qual ${{qualClass}}">${{qualLabel}}</span>
                        <span class="contrib-math">× ${{weight}}% = <strong>${{contrib.toFixed(1)}} pts</strong></span>
                    </div>
                </div>
                <div class="contrib-pts">${{contrib.toFixed(1)}}<span class="contrib-pts-max">/${{maxContrib}}</span></div>
            </div>`;
        }});

        document.getElementById('contrib-visual').innerHTML = html;

        // Total row
        const totalPct = (runningTotal / 100 * 100).toFixed(0);
        document.getElementById('contrib-total').innerHTML = `
            <div class="contrib-total-bar-area">
                <div class="contrib-total-track">
                    <div class="contrib-total-fill" style="width:${{totalPct}}%"></div>
                </div>
            </div>
            <div class="contrib-total-label">
                Composite Score: <strong>${{runningTotal.toFixed(1)}}</strong> / 100
            </div>`;
    }}

    function renderCategoryDetails(ticker, s, cats) {{
        const container = document.getElementById('modal-categories');
        container.innerHTML = '';
        const mw = D.weights.metric_weights || {{}};

        cats.forEach(cat => {{
            const catMetrics = mw[cat] || {{}};
            const metricKeys = Object.keys(catMetrics);
            // Filter to metrics with non-zero weight
            const activeMetrics = metricKeys.filter(m => catMetrics[m] > 0);
            const inactiveMetrics = metricKeys.filter(m => catMetrics[m] === 0);

            const catScore = s.cat_scores[cat];
            const contrib = s.contrib[cat];
            const factorWt = D.weights.factor_weights ? D.weights.factor_weights[cat] : null;

            let html = `<div class="cat-detail-section">
                <div class="cat-detail-header" onclick="this.parentElement.querySelector('.cat-detail-body').classList.toggle('collapsed')">
                    <div>
                        <h3 style="display:inline;color:${{CAT_COLORS[cat]}}">${{CAT_LABELS[cat]}}</h3>
                        <span class="cat-weight-badge">${{factorWt}}% weight → ${{fmt(contrib,'score')}} pts</span>
                    </div>
                    <div>
                        <span class="cat-score-badge">Score: ${{fmt(catScore,'score')}}</span>
                    </div>
                </div>
                <div class="cat-detail-body">
                    <div style="display:flex;padding:4px 0;font-size:11px;color:var(--text-muted);border-bottom:1px solid var(--border-bright);margin-bottom:4px;font-family:var(--font-heading);letter-spacing:.5px;text-transform:uppercase;">
                        <span style="flex:0 0 160px">Metric</span>
                        <span style="flex:0 0 100px;text-align:right">Raw Value</span>
                        <span style="flex:1;margin-left:16px">Percentile Rank</span>
                        <span style="flex:0 0 50px;text-align:right">Weight</span>
                    </div>`;

            activeMetrics.forEach(metric => {{
                const meta = D.metric_meta[metric] || {{ label: metric, fmt: 'ratio' }};
                const raw = s.raw[metric];
                const pct = s.pct[metric];
                const weight = catMetrics[metric];
                const rawStr = fmtMetric(raw, meta.fmt);
                const pctVal = pct !== null && pct !== undefined ? pct : null;
                const barColor = pctBarColor(pctVal, cat);

                html += `<div class="metric-row">
                    <span class="metric-name">${{meta.label}}</span>
                    <span class="metric-raw">${{raw !== null && raw !== undefined ? rawStr : '<span class=metric-na>N/A</span>'}}</span>
                    <div class="metric-pct-bar-container">
                        <div class="metric-pct-bar">
                            ${{pctVal !== null ? `<div class="metric-pct-fill" style="width:${{Math.max(1, pctVal)}}%;background:${{barColor}}"></div>` : ''}}
                        </div>
                        <span class="metric-pct-label">${{pctVal !== null ? pctVal.toFixed(0) + '%' : '<span class=metric-na>—</span>'}}</span>
                    </div>
                    <span class="metric-weight">${{weight}}%</span>
                </div>`;
            }});

            if (inactiveMetrics.length > 0) {{
                html += `<div style="margin-top:8px;font-size:11px;color:var(--text-secondary);">
                    Inactive (0% weight): ${{inactiveMetrics.map(m => (D.metric_meta[m]||{{}}).label || m).join(', ')}}
                </div>`;
            }}

            html += `</div></div>`;
            container.innerHTML += html;
        }});
    }}

    function fmtMetric(v, type) {{
        if (v === null || v === undefined) return '—';
        if (type === 'pct') return (v * 100).toFixed(1) + '%';
        if (type === 'int') return Math.round(v).toString();
        if (type === 'ratio') return v.toFixed(2);
        return v.toString();
    }}

    function pctBarColor(pct, cat) {{
        if (pct === null) return '#ccc';
        return CAT_COLORS[cat] || '#4C72B0';
    }}

    // =====================================================================
    // METHODOLOGY MODAL
    // =====================================================================
    function openMethodology() {{
        document.getElementById('methodology-modal').style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }}
    function closeMethodology() {{
        document.getElementById('methodology-modal').style.display = 'none';
        document.body.style.overflow = '';
    }}

    // =====================================================================
    // AI CHAT
    // =====================================================================
    const chatState = {{ messages: [], isStreaming: false, maxHistory: 10, abortController: null }};

    const STARTER_QUESTIONS = [
        "Why are the top 5 stocks ranked so high?",
        "What sectors are strongest right now?",
        "Which stocks are flagged as value traps?",
        "How defensible is this screener's methodology?",
    ];

    function getApiKey() {{ return localStorage.getItem('screener_openai_api_key') || ''; }}
    function getChatModel() {{ return localStorage.getItem('screener_chat_model') || 'gpt-4o-mini'; }}

    function updateModelBadge() {{
        const el = document.getElementById('chat-header-model');
        if (el) el.textContent = getChatModel();
    }}

    function toggleChat() {{
        const panel = document.getElementById('chat-panel');
        const fab = document.getElementById('chat-fab');
        const isOpen = panel.classList.contains('open');
        if (isOpen) {{
            panel.classList.remove('open');
            fab.classList.remove('open');
        }} else {{
            panel.classList.add('open');
            fab.classList.add('open');
            document.getElementById('chat-input').focus();
        }}
    }}

    function openApiKeyDialog() {{
        const dialog = document.getElementById('chat-api-dialog');
        const input = document.getElementById('chat-api-input');
        input.value = getApiKey();
        var sel = document.getElementById('chat-model-select');
        if (sel) sel.value = getChatModel();
        dialog.style.display = 'flex';
        input.focus();
    }}

    function closeApiKeyDialog() {{
        document.getElementById('chat-api-dialog').style.display = 'none';
    }}

    async function saveSettings() {{
        const input = document.getElementById('chat-api-input');
        const key = input.value.trim();
        if (!key) return;
        if (!key.startsWith('sk-')) {{
            appendChatMsg('error', 'API key should start with "sk-". Please check and try again.');
            return;
        }}
        try {{
            const res = await fetch('https://api.openai.com/v1/models', {{
                headers: {{ 'Authorization': 'Bearer ' + key }}
            }});
            if (!res.ok) {{
                appendChatMsg('error', 'Invalid API key (HTTP ' + res.status + '). Please check and try again.');
                return;
            }}
        }} catch (e) {{
            appendChatMsg('error', 'Could not validate key. Check your internet connection.');
            return;
        }}
        localStorage.setItem('screener_openai_api_key', key);
        var sel = document.getElementById('chat-model-select');
        if (sel) localStorage.setItem('screener_chat_model', sel.value);
        updateModelBadge();
        closeApiKeyDialog();
        appendChatMsg('ai', 'Settings saved! Using model: ' + getChatModel());
    }}

    function appendChatMsg(type, content) {{
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'chat-msg chat-msg-' + type;
        if (type === 'ai') {{
            div.innerHTML = parseChatMd(content);
        }} else {{
            div.textContent = content;
        }}
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
        return div;
    }}

    function parseChatMd(text) {{
        return text
            .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
            .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/^[-*]\\s+(.+)$/gm, '<li>$1</li>')
            .replace(/(<li>[\\s\\S]*?<\\/li>)/g, function(m) {{ return '<ul>' + m + '</ul>'; }})
            .replace(/<\\/ul>\\s*<ul>/g, '')
            .replace(/\\n/g, '<br>');
    }}

    function showChatTyping() {{
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'chat-typing';
        div.id = 'chat-typing';
        div.innerHTML = '<div class="chat-typing-dot"></div><div class="chat-typing-dot"></div><div class="chat-typing-dot"></div>';
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }}

    function removeChatTyping() {{
        const el = document.getElementById('chat-typing');
        if (el) el.remove();
    }}

    function handleChatKeydown(e) {{
        if (e.key === 'Enter' && !e.shiftKey) {{
            e.preventDefault();
            sendMessage();
        }}
    }}

    function autoResizeInput(el) {{
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 80) + 'px';
    }}

    // ---- Context building ----
    function extractTickers(text) {{
        const upper = text.toUpperCase();
        const found = [];
        // Check every ticker in our data
        for (const t of Object.keys(D.stock_detail)) {{
            // Match as standalone word (with optional $ prefix)
            const re = new RegExp('\\\\b\\\\$?' + t + '\\\\b');
            if (re.test(upper)) found.push(t);
        }}
        // Also match company names
        const lower = text.toLowerCase();
        for (const row of D.table_data) {{
            if (!row.Company) continue;
            const firstWord = row.Company.toLowerCase().split(/[\\s,]+/)[0];
            if (firstWord.length >= 4 && lower.includes(firstWord) && !found.includes(row.Ticker)) {{
                found.push(row.Ticker);
            }}
        }}
        return found;
    }}

    function extractSectors(text) {{
        const lower = text.toLowerCase();
        return D.sectors.filter(function(s) {{ return lower.includes(s.toLowerCase()); }});
    }}

    function detectQueryType(text) {{
        const lower = text.toLowerCase();
        if (/top\\s*\\d+|best|highest|leading|strongest/i.test(text)) return 'top_n';
        if (/worst|lowest|weakest|bottom/i.test(text)) return 'bottom_n';
        if (/compar|vs\\.?|versus|differ/i.test(text)) return 'compare';
        if (/sector|industr/i.test(text)) return 'sector';
        if (/portfolio|holdings|picks/i.test(text)) return 'portfolio';
        if (/trap|flag/i.test(text)) return 'traps';
        return 'general';
    }}

    function buildStockCtx(ticker) {{
        const s = D.stock_detail[ticker];
        if (!s) return '\\n' + ticker + ': Data not available.\\n';
        const cats = ['valuation','quality','growth','momentum','risk','revisions','size','investment'];
        let ctx = '\\n### ' + ticker + ' — ' + s.company + ' (' + s.sector + ')\\n';
        ctx += 'Rank: #' + s.rank + '/' + D.kpis.universe_size + ' | Composite: ' + (s.composite != null ? s.composite.toFixed(1) : 'N/A') + '\\n';
        ctx += 'Flags: ' + (s.vt ? 'VALUE TRAP ' : '') + (s.gt ? 'GROWTH TRAP ' : '') + (!s.vt && !s.gt ? 'None' : '') + '\\n';
        ctx += 'Category Scores: ' + cats.map(function(c) {{ return c + '=' + (s.cat_scores[c] != null ? s.cat_scores[c].toFixed(1) : 'N/A'); }}).join(', ') + '\\n';
        ctx += 'Contributions: ' + cats.map(function(c) {{ return c + '=' + (s.contrib[c] != null ? s.contrib[c].toFixed(1) : '0') + 'pts'; }}).join(', ') + '\\n';
        ctx += 'Key Metrics:\\n';
        for (const [k, v] of Object.entries(s.raw)) {{
            if (v == null) continue;
            const m = D.metric_meta[k] || {{ label: k, fmt: 'ratio' }};
            const pct = s.pct[k];
            ctx += '  ' + m.label + ': ' + fmtMetric(v, m.fmt) + ' (' + (pct != null ? pct.toFixed(0) + 'th pctile' : 'N/A') + ')\\n';
        }}
        if (s.pt_mean || s.price) {{
            ctx += 'Price: $' + (s.price != null ? s.price.toFixed(2) : 'N/A');
            if (s.pt_mean) ctx += ', Mean Target: $' + s.pt_mean.toFixed(2);
            if (s.num_analysts) ctx += ' (' + s.num_analysts + ' analysts)';
            ctx += '\\n';
        }}
        // Financials (Company Snapshot data)
        if (s.financials) {{
            const f = s.financials;
            ctx += 'Financials:\\n';
            if (f.market_cap != null) ctx += '  Market Cap: $' + fmtBig(f.market_cap) + '\\n';
            if (f.enterprise_value != null) ctx += '  Enterprise Value: $' + fmtBig(f.enterprise_value) + '\\n';
            if (f.revenue != null) ctx += '  Revenue (LTM): $' + fmtBig(f.revenue) + (f.revenue_growth_yoy != null ? ' (' + (f.revenue_growth_yoy >= 0 ? '+' : '') + (f.revenue_growth_yoy * 100).toFixed(1) + '% YoY)' : '') + '\\n';
            if (f.net_income != null) ctx += '  Net Income (LTM): $' + fmtBig(f.net_income) + (f.ni_growth_yoy != null ? ' (' + (f.ni_growth_yoy >= 0 ? '+' : '') + (f.ni_growth_yoy * 100).toFixed(1) + '% YoY)' : '') + '\\n';
            if (f.ebitda != null) ctx += '  EBITDA: $' + fmtBig(f.ebitda) + '\\n';
            if (f.gross_margin != null) ctx += '  Gross Margin: ' + (f.gross_margin * 100).toFixed(1) + '%\\n';
            if (f.net_margin != null) ctx += '  Net Margin: ' + (f.net_margin * 100).toFixed(1) + '%\\n';
            if (f.fcf != null) ctx += '  Free Cash Flow: $' + fmtBig(f.fcf) + '\\n';
            if (f.total_debt != null) ctx += '  Total Debt: $' + fmtBig(f.total_debt) + '\\n';
            if (f.total_cash != null) ctx += '  Cash: $' + fmtBig(f.total_cash) + '\\n';
            if (f.net_debt != null) ctx += '  Net Debt: $' + fmtBig(f.net_debt) + '\\n';
            if (f.dividend_yield != null) ctx += '  Dividend Yield: ' + (f.dividend_yield * 100).toFixed(2) + '%\\n';
            if (f.trailing_eps != null) ctx += '  EPS (TTM): $' + f.trailing_eps.toFixed(2) + '\\n';
            if (f.forward_eps != null) ctx += '  EPS (Fwd): $' + f.forward_eps.toFixed(2) + '\\n';
        }}
        // Sector peers
        if (s.peers && s.peers.length > 0) {{
            ctx += 'Sector Peers (by market cap proximity):\\n';
            s.peers.forEach(function(p) {{
                ctx += '  ' + p.ticker + ' (' + (p.company || '') + ')';
                if (p.composite != null) ctx += ' Composite:' + p.composite.toFixed(1);
                if (p.rank != null) ctx += ' Rank:#' + p.rank;
                ctx += '\\n';
            }});
        }}
        return ctx;
    }}

    function buildUserContext(msg) {{
        const tickers = extractTickers(msg);
        const sectors = extractSectors(msg);
        const qtype = detectQueryType(msg);
        let ctx = '';

        // Always include portfolio summary
        ctx += '\\n## Portfolio Summary (Top ' + D.portfolio.holdings.length + ')\\n';
        D.portfolio.holdings.forEach(function(h) {{
            ctx += h.rank + '. ' + h.ticker + ' (' + h.sector + ') Composite:' + (h.composite != null ? h.composite.toFixed(1) : '?') +
                ' Val:' + (h.valuation != null ? h.valuation.toFixed(0) : '?') +
                ' Qual:' + (h.quality != null ? h.quality.toFixed(0) : '?') +
                ' Grow:' + (h.growth != null ? h.growth.toFixed(0) : '?') +
                ' Mom:' + (h.momentum != null ? h.momentum.toFixed(0) : '?') + '\\n';
        }});

        // Include mentioned tickers
        tickers.forEach(function(t) {{ ctx += buildStockCtx(t); }});

        // Top N queries
        if (qtype === 'top_n') {{
            const n = parseInt((msg.match(/\\d+/) || ['5'])[0]) || 5;
            D.table_data.slice(0, Math.min(n, 15)).forEach(function(row) {{
                if (!tickers.includes(row.Ticker)) ctx += buildStockCtx(row.Ticker);
            }});
        }}

        // Bottom N queries
        if (qtype === 'bottom_n') {{
            const n = parseInt((msg.match(/\\d+/) || ['5'])[0]) || 5;
            const sorted = D.table_data.slice().sort(function(a,b) {{ return (a.Composite||0) - (b.Composite||0); }});
            sorted.slice(0, Math.min(n, 15)).forEach(function(row) {{
                if (!tickers.includes(row.Ticker)) ctx += buildStockCtx(row.Ticker);
            }});
        }}

        // Sector queries
        if (qtype === 'sector' || sectors.length > 0) {{
            const targets = sectors.length > 0 ? sectors : D.sectors;
            ctx += '\\n## Sector Statistics\\n';
            targets.forEach(function(sector) {{
                const stocks = D.table_data.filter(function(r) {{ return r.Sector === sector; }});
                const avg = stocks.reduce(function(s,r) {{ return s + (r.Composite||0); }}, 0) / (stocks.length || 1);
                ctx += '**' + sector + '** (' + stocks.length + ' stocks): Avg Composite ' + avg.toFixed(1) + '\\n';
                stocks.sort(function(a,b) {{ return (b.Composite||0) - (a.Composite||0); }}).slice(0,3).forEach(function(r) {{
                    ctx += '  - #' + r.Rank + ' ' + r.Ticker + ': ' + (r.Composite != null ? r.Composite.toFixed(1) : '?') + '\\n';
                }});
            }});
        }}

        // Trap queries
        if (qtype === 'traps') {{
            const vt = D.table_data.filter(function(r) {{ return r.Value_Trap_Flag; }});
            const gt = D.table_data.filter(function(r) {{ return r.Growth_Trap_Flag; }});
            ctx += '\\n## Trap Summary\\n';
            ctx += 'Value Traps: ' + vt.length + ' stocks. Examples: ' + vt.slice(0,10).map(function(r){{ return r.Ticker; }}).join(', ') + '\\n';
            ctx += 'Growth Traps: ' + gt.length + ' stocks. Examples: ' + gt.slice(0,10).map(function(r){{ return r.Ticker; }}).join(', ') + '\\n';
        }}

        return ctx;
    }}

    function buildSystemPrompt() {{
        const fw = D.weights.factor_weights || {{}};
        const mw = D.weights.metric_weights || {{}};
        const cats = Object.entries(fw).map(function(e) {{ return e[0] + ': ' + e[1] + '%'; }}).join(', ');

        // Build per-category metric weight strings dynamically from config
        function fmtMetrics(catKey) {{
            var obj = mw[catKey] || {{}};
            return Object.entries(obj)
                .filter(function(e) {{ return e[1] > 0; }})
                .map(function(e) {{ return e[0].replace(/_/g,' ') + ' (' + e[1] + '%)'; }})
                .join(', ');
        }}

        // Read trap filter thresholds from config data (injected at generation time)
        var vtf = D.config_traps || {{}};
        var vtQual = vtf.vt_quality || 30;
        var vtMom  = vtf.vt_momentum || 30;
        var vtRev  = vtf.vt_revisions || 30;
        var gtGrow = vtf.gt_growth || 70;

        return 'You are an AI assistant embedded in a Multi-Factor Stock Screener dashboard. You have access to screener data for ' + D.kpis.universe_size + ' S&P 500 stocks.\\n\\n' +
            '## Scoring Methodology\\n' +
            'The screener ranks stocks using 8 factor categories with these weights: ' + cats + '.\\n' +
            'All flow metrics (revenue, net income, EBITDA, cash flow) use LTM (Last Twelve Months = sum of 4 most recent quarters). Balance sheet items use MRQ (Most Recent Quarter). Falls back to annual filings if quarterly data unavailable. Enterprise Value is cross-validated: if API-provided EV differs from computed (MC+Debt-Cash) by >10% (>25% for Financials, whose debt includes deposits), the computed value is used.\\n' +
            'Each stock is scored 0-100 on each category (sector-relative percentile ranking), then combined using the weights above into a Composite score (0-100). Higher = better.\\n\\n' +
            '### Category Definitions (metric weights from config):\\n' +
            '- **Valuation** (' + (fw.valuation||'?') + '%): ' + fmtMetrics('valuation') + '. Banks use P/B + Earnings Yield (see bank weights).\\n' +
            '- **Quality** (' + (fw.quality||'?') + '%): ' + fmtMetrics('quality') + '. Banks use ROE, ROA, Equity Ratio, Piotroski, Accruals.\\n' +
            '- **Growth** (' + (fw.growth||'?') + '%): ' + fmtMetrics('growth') + '. PEG Ratio removed (double-counts valuation).\\n' +
            '- **Momentum** (' + (fw.momentum||'?') + '%): ' + fmtMetrics('momentum') + '. Skip-month convention for return signals.\\n' +
            '- **Risk** (' + (fw.risk||'?') + '%): ' + fmtMetrics('risk') + '. Lower risk = higher score (Sharpe: higher = better).\\n' +
            '- **Revisions** (' + (fw.revisions||'?') + '%): ' + fmtMetrics('revisions') + '.\\n' +
            '- **Size** (' + (fw.size||'?') + '%): -log(Market Cap). Tilts toward smaller S&P 500 names.\\n' +
            '- **Investment** (' + (fw.investment||'?') + '%): YoY Asset Growth. Conservative investment = higher score.\\n\\n' +
            '### Trap Filters:\\n' +
            '- **Value Trap**: Flagged if 2-of-3: Quality < ' + vtQual + 'th pctile, Momentum < ' + vtMom + 'th pctile, Revisions < ' + vtRev + 'th pctile.\\n' +
            '- **Growth Trap**: Flagged if Growth > ' + gtGrow + 'th pctile AND fails 2-of-3 weakness checks.\\n\\n' +
            '### Portfolio: Top ' + D.portfolio.holdings.length + ' stocks by composite, sector-capped.\\n\\n' +
            '### Defensibility Features:\\n' +
            '- **Weight Sensitivity Analysis**: Each factor weight is perturbed +/-5% and the Jaccard similarity of the top-20 portfolio is measured. Jaccard >= 0.85 = robust; < 0.70 = sensitive.\\n' +
            '- **EPS Basis Mismatch Detection**: Stocks where forward/trailing EPS ratio exceeds 2.0x or is below 0.3x are flagged.\\n' +
            '- **Factor Correlation Matrix**: Spearman correlation of all category scores is computed. Correlations > 0.6 = meaningful overlap; > 0.8 = double-counting risk.\\n' +
            '- **Data Provenance**: Each stock carries `_data_source`, `_metric_count` (valid metrics out of ~33), and `_metric_total`.\\n' +
            '- **DataValidation Sheet**: Top 10 portfolio stocks shown with raw financials for manual spot-checking against Bloomberg/SEC filings.\\n\\n' +
            '## Instructions:\\n' +
            '- Answer questions using ONLY the data provided in context. Never fabricate numbers.\\n' +
            '- When explaining a stock, reference its specific scores and metrics.\\n' +
            '- Use plain language — the user may not be a quant.\\n' +
            '- Format with markdown: **bold**, bullets, `code` for metric names.\\n' +
            '- Keep responses concise (2-4 short paragraphs). Expand only when asked.\\n' +
            '- If a stock is not in the provided context, say so.\\n' +
            '- When asked about defensibility, reference the weight sensitivity, EPS mismatch, factor correlation, and data provenance features.';
    }}

    async function sendMessage() {{
        const input = document.getElementById('chat-input');
        const text = input.value.trim();
        if (!text || chatState.isStreaming) return;

        if (!getApiKey()) {{
            openApiKeyDialog();
            return;
        }}

        // Hide suggestions
        document.getElementById('chat-suggestions').innerHTML = '';

        appendChatMsg('user', text);
        input.value = '';
        autoResizeInput(input);
        chatState.messages.push({{ role: 'user', content: text }});

        chatState.isStreaming = true;
        document.getElementById('chat-send-btn').disabled = true;
        showChatTyping();

        try {{
            await callOpenAI(text);
        }} catch (err) {{
            removeChatTyping();
            if (err.name !== 'AbortError') {{
                appendChatMsg('error', 'Error: ' + err.message);
            }}
        }} finally {{
            chatState.isStreaming = false;
            document.getElementById('chat-send-btn').disabled = false;
        }}
    }}

    async function callOpenAI(userMessage) {{
        const userContext = buildUserContext(userMessage);
        const contextMsg = userContext
            ? '[CONTEXT DATA]\\n' + userContext + '\\n\\n[USER QUESTION]\\n' + userMessage
            : userMessage;

        const messages = [{{ role: 'system', content: buildSystemPrompt() }}];
        const history = chatState.messages.slice(-chatState.maxHistory);
        history.forEach(function(msg, i) {{
            if (i === history.length - 1 && msg.role === 'user') {{
                messages.push({{ role: 'user', content: contextMsg }});
            }} else {{
                messages.push({{ role: msg.role, content: msg.content }});
            }}
        }});
        if (history.length === 0) {{
            messages.push({{ role: 'user', content: contextMsg }});
        }}

        chatState.abortController = new AbortController();

        const res = await fetch('https://api.openai.com/v1/chat/completions', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + getApiKey(),
            }},
            body: JSON.stringify({{
                model: getChatModel(),
                messages: messages,
                stream: true,
                temperature: 0.3,
                max_tokens: 800,
            }}),
            signal: chatState.abortController.signal,
        }});

        if (!res.ok) {{
            const errBody = await res.text();
            if (res.status === 401) throw new Error('Invalid API key. Click the gear icon to update it.');
            if (res.status === 429) throw new Error('Rate limited. Please wait a moment and try again.');
            throw new Error('API error (' + res.status + '): ' + errBody.substring(0, 100));
        }}

        removeChatTyping();
        const aiDiv = appendChatMsg('ai', '');
        let fullContent = '';

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {{
            const {{ done, value }} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {{ stream: true }});
            const lines = buffer.split('\\n');
            buffer = lines.pop() || '';
            for (const line of lines) {{
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6);
                if (data === '[DONE]') break;
                try {{
                    const parsed = JSON.parse(data);
                    const delta = parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content;
                    if (delta) {{
                        fullContent += delta;
                        aiDiv.innerHTML = parseChatMd(fullContent);
                        document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
                    }}
                }} catch (e) {{}}
            }}
        }}

        chatState.messages.push({{ role: 'assistant', content: fullContent }});
    }}

    function clearChat() {{
        chatState.messages = [];
        if (chatState.abortController) chatState.abortController.abort();
        chatState.isStreaming = false;
        document.getElementById('chat-send-btn').disabled = false;
        initChatWelcome();
    }}

    function initChatWelcome() {{
        updateModelBadge();
        const container = document.getElementById('chat-messages');
        container.innerHTML = '';
        const welcome = document.createElement('div');
        welcome.className = 'chat-msg chat-msg-welcome';
        welcome.innerHTML = '<strong>Screener AI</strong>Ask me anything about the ' + D.kpis.universe_size + ' stocks in the screener — rankings, metrics, sectors, or comparisons.';
        container.appendChild(welcome);

        const sugEl = document.getElementById('chat-suggestions');
        sugEl.innerHTML = STARTER_QUESTIONS.map(function(q) {{
            return '<button class="chat-suggestion-btn" onclick="useSuggestion(this)">' + q + '</button>';
        }}).join('');
    }}

    function useSuggestion(btn) {{
        document.getElementById('chat-input').value = btn.textContent;
        sendMessage();
    }}

    // ---- Chat resize ----
    function loadChatSize() {{
        const panel = document.getElementById('chat-panel');
        const w = localStorage.getItem('screener_chat_width');
        const h = localStorage.getItem('screener_chat_height');
        if (w) panel.style.width = w + 'px';
        if (h) panel.style.height = h + 'px';
    }}

    function initChatResize() {{
        const handle = document.getElementById('chat-resize-handle');
        const panel = document.getElementById('chat-panel');
        if (!handle || !panel) return;

        function startResize(startX, startY) {{
            const startW = panel.offsetWidth;
            const startH = panel.offsetHeight;
            panel.classList.add('chat-resizing');

            function onMove(mx, my) {{
                const dw = startX - mx;
                const dh = startY - my;
                const newW = Math.max(320, Math.min(window.innerWidth * 0.9, startW + dw));
                const newH = Math.max(300, Math.min(window.innerHeight * 0.85, startH + dh));
                panel.style.width = newW + 'px';
                panel.style.height = newH + 'px';
            }}

            function onEnd() {{
                panel.classList.remove('chat-resizing');
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onEnd);
                document.removeEventListener('touchmove', onTouchMove);
                document.removeEventListener('touchend', onEnd);
                localStorage.setItem('screener_chat_width', panel.offsetWidth);
                localStorage.setItem('screener_chat_height', panel.offsetHeight);
            }}

            function onMouseMove(e) {{ e.preventDefault(); onMove(e.clientX, e.clientY); }}
            function onTouchMove(e) {{ e.preventDefault(); onMove(e.touches[0].clientX, e.touches[0].clientY); }}

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onEnd);
            document.addEventListener('touchmove', onTouchMove, {{ passive: false }});
            document.addEventListener('touchend', onEnd);
        }}

        handle.addEventListener('mousedown', function(e) {{
            e.preventDefault();
            startResize(e.clientX, e.clientY);
        }});
        handle.addEventListener('touchstart', function(e) {{
            e.preventDefault();
            startResize(e.touches[0].clientX, e.touches[0].clientY);
        }}, {{ passive: false }});

        handle.addEventListener('dblclick', function() {{
            panel.style.width = '';
            panel.style.height = '';
            localStorage.removeItem('screener_chat_width');
            localStorage.removeItem('screener_chat_height');
        }});
    }}

    // =====================================================================
    // COLLAPSIBLE SECTIONS
    // =====================================================================
    function toggleSection(sectionId) {{
        const sec = document.getElementById(sectionId);
        if (!sec) return;
        sec.classList.toggle('collapsed');
    }}

    // =====================================================================
    // DEFENSIBILITY & DIAGNOSTICS
    // =====================================================================

    function corrColor(v, isDiag) {{
        if (isDiag) return 'var(--bg-elevated)';
        if (v === null || v === undefined) return 'transparent';
        const abs = Math.abs(v);
        if (abs > 0.7) return 'rgba(248,81,73,' + (0.12 + abs * 0.18) + ')';
        if (abs > 0.4) return 'rgba(210,153,34,' + (0.08 + abs * 0.12) + ')';
        return 'rgba(63,185,80,' + (0.04 + (1 - abs) * 0.08) + ')';
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
                const jColor = avgJ >= 0.85 ? '#3fb950' : avgJ >= 0.70 ? '#d29922' : '#f85149';
                const jLabel = avgJ >= 0.85 ? 'Robust' : avgJ >= 0.70 ? 'Moderate' : 'Sensitive';
                summaryHtml += '<span class="def-badge" style="color:' + jColor + '">Stability: ' + jLabel + ' (' + (avgJ * 100).toFixed(0) + '%)</span>';
            }}
        }}
        if (dq.eps_mismatch_count !== undefined) {{
            const mColor = dq.eps_mismatch_count === 0 ? '#3fb950' : '#d29922';
            summaryHtml += '<span class="def-badge" style="color:' + mColor + '">EPS Flags: ' + dq.eps_mismatch_count + '</span>';
        }}
        if (dq.avg_metric_coverage !== null && dq.avg_metric_coverage !== undefined) {{
            const cov = (dq.avg_metric_coverage * 100).toFixed(0);
            const cColor = dq.avg_metric_coverage >= 0.80 ? '#3fb950' : dq.avg_metric_coverage >= 0.60 ? '#d29922' : '#f85149';
            summaryHtml += '<span class="def-badge" style="color:' + cColor + '">Data: ' + cov + '% complete</span>';
        }}
        document.getElementById('defensibility-summary').innerHTML = summaryHtml;

        // --- Data quality KPI cards (above the two panels) ---
        let kpiHtml = '';
        const freshTs = dq.data_freshness ? new Date(dq.data_freshness) : null;
        const freshStr = freshTs && !isNaN(freshTs) ? freshTs.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }}) : '\u2014';
        kpiHtml += dqKpi('Data Freshness', freshStr, 'when the data was last fetched from Yahoo Finance', null);
        kpiHtml += dqKpi('Metric Coverage',
            dq.avg_metric_coverage !== null && dq.avg_metric_coverage !== undefined ? (dq.avg_metric_coverage * 100).toFixed(0) + '%' : '\u2014',
            'of the 21 core metrics have valid data, on average per stock', dq.avg_metric_coverage >= 0.80 ? '#3fb950' : '#d29922');
        kpiHtml += dqKpi('EPS Mismatch', dq.eps_mismatch_count !== undefined ? dq.eps_mismatch_count : '\u2014',
            'stocks have a GAAP vs. non-GAAP EPS discrepancy that may distort growth metrics',
            dq.eps_mismatch_count === 0 ? '#3fb950' : '#d29922');
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
                const barColor = aj !== null ? (aj >= 0.85 ? '#3fb950' : aj >= 0.70 ? '#d29922' : '#f85149') : '#484f58';
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
                    hHtml += '<div class="corr-cell" style="background:' + bg + '" title="' + title + '">' + txt + '</div>';
                }});
            }});
            hHtml += '</div>';
            // Legend
            hHtml += '<div class="corr-legend">';
            hHtml += '<span class="corr-legend-item"><span class="corr-legend-swatch" style="background:rgba(63,185,80,.15)"></span>Low (&lt;0.4) = Independent</span>';
            hHtml += '<span class="corr-legend-item"><span class="corr-legend-swatch" style="background:rgba(210,153,34,.15)"></span>Moderate (0.4\u20130.7)</span>';
            hHtml += '<span class="corr-legend-item"><span class="corr-legend-swatch" style="background:rgba(248,81,73,.25)"></span>High (&gt;0.7) = Overlapping</span>';
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
        html += '</div></div>';
        container.innerHTML = html;
    }}

    // INIT
    // =====================================================================
    renderKPIs();
    renderTop5();
    renderPortfolio();
    renderSectorAlloc();
    renderTrapChart();
    updateSectorDist();
    setupFilters();
    applyFilters();
    renderDefensibility();
    initChatWelcome();
    loadChatSize();
    initChatResize();

    </script>
</body>
</html>"""


def _css() -> str:
    return """
        /* ====================================================================
           DARK EDITORIAL AESTHETIC — "The Terminal"
           Space Grotesk headings · DM Sans body · JetBrains Mono numbers
           ==================================================================== */

        :root {
            --bg-deep: #0a0e17;
            --bg-primary: #0d1117;
            --bg-card: #161b22;
            --bg-card-hover: #1c2333;
            --bg-elevated: #21262d;
            --border: rgba(255,255,255,.06);
            --border-bright: rgba(255,255,255,.12);
            --text-primary: #e6edf3;
            --text-secondary: #7d8590;
            --text-muted: #484f58;
            --accent: #58a6ff;
            --accent-glow: rgba(88,166,255,.15);
            --green: #3fb950;
            --green-dim: rgba(63,185,80,.15);
            --red: #f85149;
            --red-dim: rgba(248,81,73,.15);
            --amber: #d29922;
            --amber-dim: rgba(210,153,34,.15);
            --gap: 16px;
            --radius: 10px;
            --font-heading: 'Space Grotesk', sans-serif;
            --font-body: 'DM Sans', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        /* ---- ANIMATIONS ---- */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes slideRight {
            from { opacity: 0; transform: translateX(-12px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes glowPulse {
            0%, 100% { box-shadow: 0 0 0 0 var(--accent-glow); }
            50%      { box-shadow: 0 0 20px 4px var(--accent-glow); }
        }
        @keyframes barGrow {
            from { width: 0; }
        }
        @keyframes modalIn {
            from { opacity: 0; transform: translateY(-24px) scale(.97); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes chatSlideUp {
            from { opacity: 0; transform: translateY(20px) scale(.95); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes chatFabPulse {
            0%, 100% { box-shadow: 0 4px 20px rgba(88,166,255,.3); }
            50%      { box-shadow: 0 4px 30px rgba(88,166,255,.5); }
        }
        @keyframes chatDotBlink {
            0%, 80%, 100% { opacity: .3; }
            40% { opacity: 1; }
        }

        /* ---- RESET & BASE ---- */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--font-body);
            background: var(--bg-deep);
            color: var(--text-primary);
            line-height: 1.55;
            -webkit-font-smoothing: antialiased;
            /* Subtle noise texture via inline SVG */
            background-image:
                radial-gradient(ellipse at 20% 0%, rgba(88,166,255,.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 100%, rgba(63,185,80,.04) 0%, transparent 50%);
            background-attachment: fixed;
        }

        .dashboard-container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px var(--gap);
        }

        /* ---- HEADER ---- */
        .dashboard-header {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border: 1px solid var(--border-bright);
            color: var(--text-primary);
            padding: 22px 28px;
            border-radius: var(--radius);
            margin-bottom: var(--gap);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            animation: fadeUp .5s ease-out;
            position: relative;
            overflow: hidden;
        }
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent), var(--green), var(--amber), var(--red));
        }
        .dashboard-header h1 {
            font-family: var(--font-heading);
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -.3px;
        }
        .run-info {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-secondary);
        }
        .badge {
            display: inline-block;
            background: var(--bg-elevated);
            border: 1px solid var(--border-bright);
            padding: 4px 14px;
            border-radius: 20px;
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            color: var(--text-secondary);
        }
        .badge-warn {
            background: var(--red-dim);
            border-color: rgba(248,81,73,.25);
            color: var(--red);
        }

        /* ---- KPI ROW ---- */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: var(--gap);
            margin-bottom: var(--gap);
        }
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px 22px;
            animation: fadeUp .5s ease-out both;
            transition: border-color .2s, box-shadow .2s;
        }
        .kpi-card:nth-child(1) { animation-delay: .05s; }
        .kpi-card:nth-child(2) { animation-delay: .1s; }
        .kpi-card:nth-child(3) { animation-delay: .15s; }
        .kpi-card:nth-child(4) { animation-delay: .2s; }
        .kpi-card:hover {
            border-color: var(--border-bright);
            box-shadow: 0 4px 24px rgba(0,0,0,.3);
        }
        .kpi-label {
            font-family: var(--font-heading);
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: .8px;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-family: var(--font-mono);
            font-size: 28px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -.5px;
        }
        .kpi-sub {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* ---- TOP 5 ---- */
        .top5-row {
            display: flex;
            gap: var(--gap);
            overflow-x: auto;
            padding-bottom: 4px;
        }
        .top5-card {
            flex: 1;
            min-width: 210px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0 0 14px;
            cursor: pointer;
            transition: all .25s ease;
            overflow: hidden;
            animation: fadeUp .5s ease-out both;
        }
        .top5-card:nth-child(1) { animation-delay: .1s; }
        .top5-card:nth-child(2) { animation-delay: .15s; }
        .top5-card:nth-child(3) { animation-delay: .2s; }
        .top5-card:nth-child(4) { animation-delay: .25s; }
        .top5-card:nth-child(5) { animation-delay: .3s; }
        .top5-card:hover {
            border-color: var(--accent);
            box-shadow: 0 0 30px var(--accent-glow), 0 8px 32px rgba(0,0,0,.4);
            transform: translateY(-3px);
        }
        .top5-rank-bar {
            background: linear-gradient(90deg, var(--accent) 0%, rgba(88,166,255,.3) 100%);
            color: #fff;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 700;
            padding: 6px 18px;
            letter-spacing: .5px;
        }
        .top5-header { padding: 12px 18px 2px; }
        .top5-ticker {
            font-family: var(--font-heading);
            font-size: 20px;
            font-weight: 700;
            color: var(--accent);
        }
        .top5-composite-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 8px 16px;
            margin: 6px 12px 10px;
            background: var(--bg-elevated);
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        .top5-composite-label {
            font-family: var(--font-heading);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .6px;
            color: var(--text-secondary);
            font-weight: 600;
        }
        .top5-composite {
            font-family: var(--font-mono);
            font-size: 22px;
            font-weight: 700;
            color: var(--green);
        }
        .top5-company {
            font-size: 12px;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 0 18px;
            margin-bottom: 4px;
        }
        .top5-sector {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            font-weight: 500;
            background: color-mix(in srgb, var(--sector-color, #7d8590) 10%, transparent);
            border: 1px solid color-mix(in srgb, var(--sector-color, #7d8590) 25%, transparent);
            color: var(--sector-color, var(--text-secondary));
            padding: 3px 12px 3px 8px;
            border-radius: 6px;
            margin-left: 18px;
            letter-spacing: .2px;
        }
        .top5-sector-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
            box-shadow: 0 0 6px color-mix(in srgb, var(--sector-color, #7d8590) 50%, transparent);
        }
        .top5-factors {
            display: flex;
            flex-direction: column;
            gap: 4px;
            padding: 0 18px;
        }
        .top5-factor {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
        }
        .top5-factor-label {
            flex: 0 0 32px;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-weight: 500;
            font-size: 10px;
        }
        .top5-factor-bar {
            flex: 1;
            height: 6px;
            background: var(--bg-elevated);
            border-radius: 3px;
            overflow: hidden;
        }
        .top5-factor-fill {
            height: 100%;
            border-radius: 3px;
            transition: width .6s cubic-bezier(.25,.46,.45,.94);
            animation: barGrow .8s ease-out;
        }
        .top5-factor-val {
            flex: 0 0 30px;
            text-align: right;
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 11px;
            color: var(--text-secondary);
        }
        .top5-cta {
            margin-top: 10px;
            padding: 0 18px;
            font-size: 11px;
            color: var(--accent);
            text-align: center;
            opacity: 0;
            transition: opacity .2s;
            font-weight: 500;
        }
        .top5-card:hover .top5-cta { opacity: 1; }

        /* ---- SECTIONS ---- */
        .section {
            margin-bottom: calc(var(--gap) * 1.5);
            animation: fadeIn .6s ease-out both;
        }
        .section:nth-of-type(2) { animation-delay: .2s; }
        .section:nth-of-type(3) { animation-delay: .35s; }
        .section-title {
            font-family: var(--font-heading);
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 14px;
            padding-left: 4px;
            letter-spacing: -.2px;
            color: var(--text-primary);
            position: relative;
            padding-left: 14px;
        }
        .section-title::before {
            content: '';
            position: absolute;
            left: 0;
            top: 3px;
            bottom: 3px;
            width: 3px;
            background: var(--accent);
            border-radius: 2px;
        }

        /* ---- COLLAPSIBLE SECTIONS ---- */
        .collapsible-section .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            user-select: none;
            padding: 6px 4px 6px 0;
            border-radius: 8px;
            transition: background .15s ease;
        }
        .collapsible-section .section-header:hover {
            background: var(--bg-elevated);
        }
        .section-chevron {
            width: 22px;
            height: 22px;
            color: var(--text-secondary);
            flex-shrink: 0;
            transition: transform .25s ease;
            margin-left: 12px;
        }
        .collapsible-section .section-body {
            overflow: hidden;
            max-height: 5000px;
            opacity: 1;
            transition: max-height .35s ease, opacity .25s ease, margin .25s ease;
            margin-top: 14px;
        }
        .collapsible-section.collapsed .section-body {
            max-height: 0;
            opacity: 0;
            margin-top: 0;
            pointer-events: none;
        }
        .collapsible-section.collapsed .section-chevron {
            transform: rotate(-90deg);
        }

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

        /* ---- TABLES ---- */
        .table-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 12px 16px;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 700px;
            margin-bottom: var(--gap);
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .data-table thead th {
            text-align: left;
            padding: 10px 10px;
            border-bottom: 1px solid var(--border-bright);
            color: var(--text-secondary);
            font-family: var(--font-heading);
            font-weight: 600;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .8px;
            white-space: nowrap;
            user-select: none;
            transition: color .15s;
            position: sticky;
            top: 0;
            z-index: 1;
            background: var(--bg-card);
        }
        .data-table thead th:hover { color: var(--accent); }
        .data-table thead th.sorted { color: var(--accent); }
        .data-table thead th.sorted::after { content: ' ▾'; }
        .data-table thead th.sorted[data-dir="asc"]::after { content: ' ▴'; }
        .data-table tbody td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            font-family: var(--font-body);
        }
        .data-table tbody tr { transition: background .1s; }
        .data-table tbody tr:hover { background: var(--bg-card-hover); }
        .data-table .num {
            text-align: right;
            font-family: var(--font-mono);
            font-variant-numeric: tabular-nums;
            font-size: 12px;
        }
        .data-table .ticker {
            font-family: var(--font-mono);
            font-weight: 600;
            color: var(--accent);
        }
        .vt-row { background: var(--red-dim); }
        .vt-cell { text-align: center; }

        /* ---- FILTERS ---- */
        .filters-bar {
            display: flex;
            gap: 16px;
            align-items: flex-end;
            flex-wrap: wrap;
            margin-bottom: 12px;
            padding: 14px 18px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
        }
        .filter-group { display: flex; flex-direction: column; gap: 4px; }
        .filter-group label {
            font-family: var(--font-heading);
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: .8px;
        }
        .filter-group select, .filter-group input {
            padding: 7px 12px;
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            font-family: var(--font-body);
            font-size: 13px;
            background: var(--bg-elevated);
            color: var(--text-primary);
            transition: border-color .15s;
        }
        .filter-group select:focus, .filter-group input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }
        .filter-group input[type="number"] { width: 80px; }
        .filter-group input[type="text"] { width: 180px; }
        .filter-group input::placeholder { color: var(--text-muted); }
        .result-count {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-secondary);
            padding: 7px 0;
        }


        /* ---- FOOTER ---- */
        .dashboard-footer {
            text-align: center;
            padding: 24px 16px;
            color: var(--text-muted);
            font-size: 12px;
            font-family: var(--font-mono);
            border-top: 1px solid var(--border);
            margin-top: 8px;
        }

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
            box-shadow: 0 24px 80px rgba(0,0,0,.6);
            animation: modalIn .3s ease-out;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 22px 26px 18px;
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
            border-radius: 14px 14px 0 0;
            border-bottom: 1px solid var(--border);
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

        /* ---- COLLAPSIBLE SECTIONS ---- */
        .collapsible {
            margin-bottom: 16px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
        }
        .collapsible-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 18px;
            cursor: pointer;
            background: var(--bg-card);
            user-select: none;
            transition: background .15s;
        }
        .collapsible-header:hover {
            background: var(--bg-card-hover);
        }
        .collapsible-header span:first-child {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .collapsible-chevron {
            font-size: 10px;
            color: var(--text-muted);
            transition: transform .2s ease;
        }
        .collapsible.collapsed .collapsible-chevron {
            transform: rotate(-90deg);
        }
        .collapsible-body {
            padding: 0 18px 16px;
            background: var(--bg-card);
        }
        .collapsible.collapsed .collapsible-body {
            display: none;
        }

        .modal-score-row {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
            margin-bottom: 22px;
        }
        .modal-score-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 10px;
            text-align: center;
            transition: border-color .2s;
        }
        .modal-score-card:hover { border-color: var(--border-bright); }
        .modal-score-card.composite {
            border-color: var(--accent);
            box-shadow: 0 0 20px var(--accent-glow);
        }
        .modal-score-label {
            font-family: var(--font-heading);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .6px;
            color: var(--text-secondary);
            margin-bottom: 3px;
        }
        .modal-score-val {
            font-family: var(--font-mono);
            font-size: 24px;
            font-weight: 700;
        }
        .modal-score-sub {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }
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

        /* ---- METHODOLOGY BUTTON & MODAL ---- */
        .methodology-btn {
            background: linear-gradient(135deg, var(--accent), #79c0ff);
            color: #0d1117;
            border: none;
            padding: 6px 18px;
            border-radius: 20px;
            font-family: var(--font-heading);
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: .3px;
            transition: all .2s;
        }
        .methodology-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px var(--accent-glow);
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
            box-shadow: 0 2px 8px rgba(0,0,0,.15);
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
            background: linear-gradient(to right, var(--border), var(--border-bright), var(--border));
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
            box-shadow: 0 0 12px var(--accent-glow);
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
            font-family: var(--font-mono);
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
            background: linear-gradient(90deg, var(--amber), var(--green));
            border-radius: 4px;
            opacity: 0.4;
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
            font-family: var(--font-mono);
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
            font-family: var(--font-mono);
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
            animation: slideRight .4s ease-out both;
        }
        .contrib-row:nth-child(1) { animation-delay: .05s; }
        .contrib-row:nth-child(2) { animation-delay: .1s; }
        .contrib-row:nth-child(3) { animation-delay: .15s; }
        .contrib-row:nth-child(4) { animation-delay: .2s; }
        .contrib-row:nth-child(5) { animation-delay: .25s; }
        .contrib-row:nth-child(6) { animation-delay: .3s; }
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
            box-shadow: 0 0 6px currentColor;
        }
        .contrib-cat-name {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
        }
        .contrib-cat-weight {
            font-family: var(--font-mono);
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
            font-family: var(--font-mono);
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
            font-family: var(--font-mono);
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
            font-family: var(--font-mono);
            font-variant-numeric: tabular-nums;
        }
        .contrib-math {
            font-family: var(--font-mono);
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
            font-family: var(--font-mono);
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
            background: linear-gradient(90deg, var(--accent), var(--green), var(--amber), var(--red));
            transition: width .6s ease;
        }
        .contrib-total-label {
            font-family: var(--font-mono);
            font-size: 14px;
            color: var(--text-secondary);
            text-align: right;
        }
        .contrib-total-label strong {
            font-size: 18px;
            color: var(--text-primary);
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
            font-family: var(--font-mono);
            background: var(--bg-elevated);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        .cat-detail-header .cat-weight-badge {
            font-family: var(--font-mono);
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
        .metric-raw {
            flex: 0 0 100px;
            text-align: right;
            font-family: var(--font-mono);
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
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 11px;
            color: var(--text-secondary);
        }
        .metric-weight {
            flex: 0 0 50px;
            text-align: right;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }
        .metric-na { color: var(--text-muted); font-style: italic; }
        .ticker-link { cursor: pointer; }
        .ticker-link:hover { text-decoration: underline; text-underline-offset: 2px; }

        /* ---- AI CHAT PANEL ---- */
        .chat-fab {
            position: fixed; bottom: 24px; right: 24px;
            width: 52px; height: 52px; border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), #79c0ff);
            border: none; color: #0d1117; cursor: pointer; z-index: 2000;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 4px 20px rgba(88,166,255,.3);
            transition: all .25s ease;
            animation: chatFabPulse 3s ease-in-out infinite;
        }
        .chat-fab:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(88,166,255,.5); }
        .chat-fab.open { animation: none; }
        .chat-fab-icon, .chat-fab-close { width: 22px; height: 22px; transition: all .2s; }
        .chat-fab-close { display: none; }
        .chat-fab.open .chat-fab-icon { display: none; }
        .chat-fab.open .chat-fab-close { display: block; }

        .chat-panel {
            position: fixed; bottom: 88px; right: 24px;
            width: 400px; height: 520px;
            min-width: 320px; min-height: 300px;
            max-width: 90vw; max-height: 85vh;
            background: var(--bg-primary);
            border: 1px solid var(--border-bright);
            border-radius: 14px;
            box-shadow: 0 16px 60px rgba(0,0,0,.5);
            z-index: 1999; display: none; flex-direction: column;
            overflow: hidden; animation: chatSlideUp .25s ease-out;
        }
        .chat-panel.open { display: flex; }
        .chat-panel.chat-resizing { transition: none; animation: none; }

        .chat-resize-handle {
            position: absolute; top: 0; left: 0;
            width: 18px; height: 18px;
            cursor: nwse-resize; z-index: 11;
            border-radius: 14px 0 0 0;
        }
        .chat-resize-handle::after {
            content: '';
            position: absolute; top: 3px; left: 3px;
            width: 8px; height: 8px;
            border-left: 2px solid var(--text-muted);
            border-top: 2px solid var(--text-muted);
            opacity: .4; transition: opacity .15s;
        }
        .chat-resize-handle:hover::after { opacity: .9; }

        .chat-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 12px 16px;
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
            border-bottom: 1px solid var(--border); flex-shrink: 0;
        }
        .chat-header-left { display: flex; align-items: center; gap: 8px; }
        .chat-header-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--green); box-shadow: 0 0 8px rgba(63,185,80,.4);
        }
        .chat-header-title {
            font-family: var(--font-heading); font-size: 14px; font-weight: 600; color: var(--text-primary);
        }
        .chat-header-actions { display: flex; gap: 4px; }
        .chat-header-btn {
            background: none; border: 1px solid transparent; border-radius: 6px;
            color: var(--text-secondary); cursor: pointer; padding: 4px 6px;
            display: flex; align-items: center; transition: all .15s;
        }
        .chat-header-btn:hover {
            color: var(--text-primary); border-color: var(--border-bright); background: var(--bg-elevated);
        }

        .chat-messages {
            flex: 1; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
            min-height: 120px;
        }
        .chat-messages::-webkit-scrollbar { width: 4px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

        .chat-msg {
            max-width: 88%; padding: 10px 14px; border-radius: 12px;
            font-size: 13px; line-height: 1.55; animation: fadeUp .2s ease-out;
        }
        .chat-msg-user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--accent), #79c0ff);
            color: #0d1117; border-bottom-right-radius: 4px; font-weight: 500;
        }
        .chat-msg-ai {
            align-self: flex-start; background: var(--bg-card);
            border: 1px solid var(--border); color: var(--text-primary);
            border-bottom-left-radius: 4px;
        }
        .chat-msg-ai strong { color: var(--accent); }
        .chat-msg-ai code {
            font-family: var(--font-mono); font-size: 11px;
            background: var(--bg-elevated); padding: 1px 5px;
            border-radius: 3px; color: var(--accent);
        }
        .chat-msg-ai ul { margin: 6px 0 6px 16px; padding: 0; }
        .chat-msg-ai li { margin-bottom: 3px; }
        .chat-msg-error {
            align-self: center; background: var(--red-dim);
            border: 1px solid rgba(248,81,73,.25); color: var(--red);
            font-size: 12px; text-align: center;
        }
        .chat-msg-welcome {
            align-self: center; text-align: center;
            color: var(--text-secondary); font-size: 12px; padding: 8px 12px;
        }
        .chat-msg-welcome strong {
            color: var(--text-primary); display: block;
            font-family: var(--font-heading); font-size: 14px; margin-bottom: 4px;
        }
        .chat-typing {
            display: flex; gap: 4px; padding: 10px 14px; align-self: flex-start;
        }
        .chat-typing-dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--text-muted); animation: chatDotBlink 1.2s infinite;
        }
        .chat-typing-dot:nth-child(2) { animation-delay: .2s; }
        .chat-typing-dot:nth-child(3) { animation-delay: .4s; }

        .chat-input-area {
            padding: 12px 16px 14px; border-top: 1px solid var(--border);
            background: var(--bg-card); flex-shrink: 0;
        }
        .chat-suggestions { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
        .chat-suggestions:empty { display: none; margin-bottom: 0; }
        .chat-suggestion-btn {
            background: var(--bg-elevated); border: 1px solid var(--border-bright);
            color: var(--text-secondary); font-family: var(--font-body); font-size: 11px;
            padding: 4px 10px; border-radius: 14px; cursor: pointer;
            transition: all .15s; white-space: nowrap;
        }
        .chat-suggestion-btn:hover {
            color: var(--accent); border-color: var(--accent); background: var(--accent-glow);
        }
        .chat-input-row { display: flex; gap: 8px; align-items: flex-end; }
        .chat-input {
            flex: 1; background: var(--bg-primary);
            border: 1px solid var(--border-bright); border-radius: 10px;
            padding: 8px 12px; font-family: var(--font-body); font-size: 13px;
            color: var(--text-primary); resize: none; max-height: 80px;
            line-height: 1.4; outline: none; transition: border-color .15s;
        }
        .chat-input:focus { border-color: var(--accent); }
        .chat-input::placeholder { color: var(--text-muted); }
        .chat-send-btn {
            width: 36px; height: 36px; border-radius: 10px;
            background: linear-gradient(135deg, var(--accent), #79c0ff);
            border: none; color: #0d1117; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            flex-shrink: 0; transition: all .15s;
        }
        .chat-send-btn:hover { transform: scale(1.05); }
        .chat-send-btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }

        .chat-api-dialog {
            position: absolute; inset: 0;
            background: rgba(10,14,23,.92); backdrop-filter: blur(4px);
            display: flex; align-items: center; justify-content: center;
            z-index: 10; border-radius: 14px;
        }
        .chat-api-dialog-content { padding: 24px; text-align: center; }
        .chat-api-dialog-content h3 {
            font-family: var(--font-heading); font-size: 16px;
            color: var(--text-primary); margin-bottom: 8px;
        }
        .chat-api-dialog-content p {
            font-size: 12px; color: var(--text-secondary);
            margin-bottom: 14px; line-height: 1.5;
        }
        .chat-api-input {
            width: 100%; background: var(--bg-card);
            border: 1px solid var(--border-bright); border-radius: 8px;
            padding: 10px 14px; font-family: var(--font-mono); font-size: 13px;
            color: var(--text-primary); outline: none; margin-bottom: 14px;
        }
        .chat-api-input:focus { border-color: var(--accent); }
        .chat-api-dialog-actions { display: flex; gap: 8px; justify-content: center; }
        .chat-api-btn {
            padding: 8px 18px; border-radius: 8px;
            font-family: var(--font-heading); font-size: 12px; font-weight: 600;
            cursor: pointer; border: 1px solid var(--border-bright); transition: all .15s;
        }
        .chat-api-btn-secondary { background: transparent; color: var(--text-secondary); }
        .chat-api-btn-secondary:hover { color: var(--text-primary); background: var(--bg-elevated); }
        .chat-api-btn-primary {
            background: linear-gradient(135deg, var(--accent), #79c0ff);
            color: #0d1117; border-color: transparent;
        }
        .chat-api-btn-primary:hover { transform: translateY(-1px); }
        .chat-api-hint { font-size: 11px; color: var(--text-muted); margin-top: 12px; }
        .chat-api-hint a { color: var(--accent); text-decoration: none; }
        .chat-api-hint a:hover { text-decoration: underline; }
        .chat-api-label {
            display: block; text-align: left; font-family: var(--font-heading);
            font-size: 11px; font-weight: 600; color: var(--text-secondary);
            text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px;
        }
        .chat-api-select {
            width: 100%; background: var(--bg-card); color: var(--text-primary);
            border: 1px solid var(--border-bright); border-radius: 8px;
            padding: 10px 14px; font-family: var(--font-body); font-size: 13px;
            outline: none; margin-bottom: 14px; cursor: pointer;
            -webkit-appearance: none; appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%237d8590' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
            background-repeat: no-repeat; background-position: right 12px center;
        }
        .chat-api-select:focus { border-color: var(--accent); }
        .chat-api-select option { background: var(--bg-card); color: var(--text-primary); }
        .chat-header-model {
            font-family: var(--font-mono); font-size: 10px;
            color: var(--text-muted); background: var(--bg-elevated);
            border: 1px solid var(--border); border-radius: 8px;
            padding: 1px 7px; margin-left: 4px;
        }

        @media (max-width: 480px) {
            .chat-panel { right: 8px; left: 8px; width: auto; bottom: 76px; max-height: 70vh; }
            .chat-fab { bottom: 16px; right: 16px; width: 46px; height: 46px; }
        }

        /* ---- RESPONSIVE ---- */
        @media (max-width: 768px) {
            .kpi-row { grid-template-columns: repeat(2, 1fr); }
            .chart-row { flex-direction: column; }
            .chart-container { min-width: 100%; }
            .chart-small { flex: 1 1 100%; min-width: 100%; }
            .filters-bar { flex-direction: column; }
        }

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
            font-family: 'JetBrains Mono', monospace; font-weight: 500;
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
            font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
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
            font-family: 'Space Grotesk', sans-serif;
        }
        .sens-table td {
            padding: 10px 10px; font-family: 'JetBrains Mono', monospace;
            border-bottom: 1px solid var(--border); vertical-align: middle;
        }
        .sens-cell-high { background: rgba(63,185,80,.12); color: #3fb950; border-radius: 4px; padding: 4px 10px; }
        .sens-cell-med  { background: rgba(210,153,34,.12); color: #d29922; border-radius: 4px; padding: 4px 10px; }
        .sens-cell-low  { background: rgba(248,81,73,.12); color: #f85149; border-radius: 4px; padding: 4px 10px; }
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
            font-family: 'JetBrains Mono', monospace; font-size: 11px;
            border-radius: 5px; cursor: default; color: var(--text-primary);
            min-width: 0; min-height: 36px;
        }
        .corr-label {
            display: flex; align-items: center; justify-content: center;
            font-family: 'Space Grotesk', sans-serif; font-size: 10px;
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
            font-family: 'JetBrains Mono', monospace; font-weight: 500;
        }
        .provenance-ok    { background: rgba(63,185,80,.12); color: #3fb950; }
        .provenance-warn  { background: rgba(210,153,34,.12); color: #d29922; }
        .provenance-alert { background: rgba(248,81,73,.12); color: #f85149; }

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
            font-family: 'Space Grotesk', sans-serif;
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
            font-family: 'Space Grotesk', sans-serif;
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
            font-family: 'Space Grotesk', sans-serif;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .4px;
            color: var(--text-secondary);
            margin-bottom: 2px;
        }
        .snapshot-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .snapshot-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 1px;
        }
        .snap-up { color: #3fb950; }
        .snap-down { color: #f85149; }

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
            font-family: 'Space Grotesk', sans-serif;
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
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            white-space: nowrap;
        }
        .peer-th-ticker { text-align: left !important; }
        .peer-table tbody td {
            padding: 8px 10px;
            font-family: 'JetBrains Mono', monospace;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }
        .peer-td-num { text-align: right; white-space: nowrap; }
        .peer-td-ticker { text-align: left; white-space: nowrap; }
        .peer-ticker {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            color: var(--text-primary);
        }
        .peer-you {
            font-size: 9px;
            font-family: 'Space Grotesk', sans-serif;
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
            font-family: var(--font-mono);
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
            color: #f85149;
            background: rgba(248,81,73,.1);
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
            font-family: 'Space Grotesk', sans-serif;
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
            font-family: 'DM Sans', sans-serif;
            font-weight: 500;
            line-height: 1.3;
        }
        .flag-icon { font-size: 13px; }
        .flag-sev {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            margin-left: 2px;
        }
        .flag-detail {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            opacity: 0.8;
        }
        .flag-severe {
            background: rgba(248,81,73,.15);
            color: #f85149;
            border: 1px solid rgba(248,81,73,.3);
        }
        .flag-warn {
            background: rgba(210,153,34,.12);
            color: #d29922;
            border: 1px solid rgba(210,153,34,.25);
        }
        .flag-mild {
            background: rgba(88,166,255,.1);
            color: #58a6ff;
            border: 1px solid rgba(88,166,255,.2);
        }
        .flag-info {
            background: rgba(125,133,144,.1);
            color: #7d8590;
            border: 1px solid rgba(125,133,144,.2);
        }

        @media print {
            body { background: #fff; color: #000; }
            :root {
                --bg-primary: #fff; --bg-card: #fff; --bg-elevated: #f5f5f5;
                --text-primary: #000; --text-secondary: #555; --border: #ddd;
            }
            .dashboard-container { max-width: none; }
            .filters-bar { display: none; }
            .kpi-card, .chart-container, .table-section { box-shadow: none; border: 1px solid #ddd; }
        }
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        # Fallback: wrap raw markdown in <pre> if markdown library unavailable
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
    html = generate_html(data_json, methodology_html)

    output_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
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
