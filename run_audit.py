#!/usr/bin/env python3
"""
Multi-Factor Screener — Instrumented Audit Run
================================================
Runs the full scoring pipeline on 5 test tickers with comprehensive
instrumentation, then generates forensics and defensibility reports.

Usage:
    python run_audit.py                       # Default 5 tickers
    python run_audit.py --tickers AAPL,MSFT   # Custom tickers
    python run_audit.py --sample              # Force sample data (offline)

Outputs (all under ./reports/):
    run_log_full.csv            Chronological event trace
    run_log_full.md             Same, as Markdown table
    run_log_full.html           Same, as styled HTML
    composite_score_forensics.md   Per-ticker score decomposition
    defensibility_review.md     CFA/hedge-fund grade assessment
    recommended_changes.md      Prioritized improvement list
"""

import argparse
import copy
import csv
import inspect
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from instrumentation import EventLog, trace_event, trace_net_call, trace_io

# =========================================================================
# Paths
# =========================================================================
ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
CACHE_DIR = ROOT / "cache"

# Default test tickers: diversified across sectors including a bank
DEFAULT_TEST_TICKERS = ["AAPL", "JPM", "JNJ", "XOM", "AMZN"]


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Instrumented Audit Run")
    p.add_argument("--tickers", type=str, default="",
                   help="Comma-separated tickers (default: AAPL,JPM,JNJ,XOM,AMZN)")
    p.add_argument("--sample", action="store_true",
                   help="Force sample/synthetic data (skip network)")
    return p.parse_args()


# =========================================================================
# Instrumented Pipeline
# =========================================================================
def run_instrumented_pipeline(tickers: list, force_sample: bool = False):
    """Run the full pipeline with instrumentation on the given tickers.

    Returns (scored_df, cfg, event_log).
    """
    log = EventLog()

    # ---- INIT: imports ----
    with trace_event(log, "INIT", "Import factor_engine module"):
        from factor_engine import (
            get_sp500_tickers, fetch_single_ticker, fetch_all_tickers,
            fetch_market_returns, compute_metrics,
            _generate_sample_data,
            apply_universe_filters,
            winsorize_metrics, compute_sector_percentiles,
            compute_category_scores, compute_composite,
            apply_value_trap_flags, rank_stocks,
            compute_factor_correlation,
            METRIC_COLS, METRIC_DIR, CAT_METRICS,
            _BANK_ONLY_METRICS, _NONBANK_ONLY_METRICS,
        )

    # ---- INIT: config ----
    with trace_event(log, "INIT", "Load config.yaml"):
        config_path = ROOT / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    log.record("INIT", "Parse CLI / set test tickers",
               0, details=f"tickers={tickers}")

    # ---- INIT: universe ----
    with trace_event(log, "INIT", "Load S&P 500 universe from Wikipedia/fallback"):
        universe_df = get_sp500_tickers(cfg)

    # Filter to test tickers
    with trace_event(log, "INIT", f"Filter universe to {len(tickers)} test tickers"):
        filtered = universe_df[universe_df["Ticker"].isin(tickers)].copy()
        # Add any missing tickers with Unknown sector
        known = set(filtered["Ticker"])
        for t in tickers:
            if t not in known:
                filtered = pd.concat([filtered, pd.DataFrame([{
                    "Ticker": t, "Company": t, "Sector": "Unknown"
                }])], ignore_index=True)
        universe_df = filtered
        ticker_meta = universe_df.set_index("Ticker")[["Company", "Sector"]].to_dict("index")

    # ---- NET: connectivity probe ----
    USE_SAMPLE = force_sample
    if not force_sample:
        t0_probe = time.monotonic()
        try:
            with trace_event(log, "NET", f"Connectivity probe: fetch_single_ticker({tickers[0]})"):
                test_rec = fetch_single_ticker(tickers[0], max_retries=1)
                if "_error" in test_rec:
                    raise RuntimeError(test_rec["_error"])
        except Exception as e:
            log.record("NET", "Connectivity probe FAILED — falling back to sample data",
                       (time.monotonic() - t0_probe) * 1000,
                       status="FAIL", details=str(e))
            USE_SAMPLE = True

    # ---- NET/CALC: fetch data ----
    if USE_SAMPLE:
        with trace_event(log, "CALC", f"Generate sample data for {len(tickers)} tickers"):
            df = _generate_sample_data(universe_df)
        market_returns = pd.Series(dtype=float)
    else:
        # Market returns
        t0_mr = time.monotonic()
        with trace_event(log, "NET", "Fetch S&P 500 market returns (yfinance ^GSPC 1y)",
                         details="host=query1.finance.yahoo.com"):
            market_returns = fetch_market_returns()
        log.record("CALC", f"Market returns: {len(market_returns)} daily observations",
                   0, details=f"observations={len(market_returns)}")

        # Fetch each ticker individually so we can instrument per-ticker
        raw_records = []
        for ticker in tickers:
            t0_tick = time.monotonic()
            rec = fetch_single_ticker(ticker)
            elapsed_tick = (time.monotonic() - t0_tick) * 1000
            status = "OK" if "_error" not in rec else "FAIL"
            details_parts = [f"host=query1.finance.yahoo.com"]
            if "_error" in rec:
                details_parts.append(f"error={rec['_error'][:80]}")
            trace_net_call(log, f"yfinance fetch: {ticker}",
                           hostname="query1.finance.yahoo.com",
                           duration_ms=elapsed_tick, status=status,
                           retries=0,
                           caller="factor_engine:fetch_single_ticker:246")
            raw_records.append(rec)

        # Compute metrics
        with trace_event(log, "CALC", f"compute_metrics() for {len(raw_records)} tickers"):
            df = compute_metrics(raw_records, market_returns)

        # Apply wiki sector names
        for idx, row in df.iterrows():
            t = row["Ticker"]
            if t in ticker_meta:
                df.at[idx, "Sector"] = ticker_meta[t]["Sector"]
                if pd.isna(row.get("Company")) or row.get("Company") == t:
                    df.at[idx, "Company"] = ticker_meta[t]["Company"]

        # Remove failed
        skip_mask = df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)
        df = df[~skip_mask].copy()

    # ---- CALC: Coverage filter ----
    with trace_event(log, "CALC", "Coverage filter (min_data_coverage_pct)"):
        present = [c for c in METRIC_COLS if c in df.columns]
        is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False)
        applicable_count = pd.Series(0, index=df.index)
        metric_count = pd.Series(0, index=df.index)
        for c in present:
            if c in _BANK_ONLY_METRICS:
                applies = is_bank
            elif c in _NONBANK_ONLY_METRICS:
                applies = ~is_bank
            else:
                applies = pd.Series(True, index=df.index)
            applicable_count += applies.astype(int)
            metric_count += (df[c].notna() & applies).astype(int)
        coverage_pct = cfg["data_quality"]["min_data_coverage_pct"] / 100
        min_needed = (applicable_count * coverage_pct).apply(lambda x: max(1, int(x)))
        df["_mc"] = metric_count
        low = df["_mc"] < min_needed
        df = df[~low].copy()

    # ---- CALC: Revisions auto-disable ----
    with trace_event(log, "CALC", "Check revisions coverage / auto-disable"):
        rev_m = ["analyst_surprise", "price_target_upside"]
        rev_avail = sum(df[c].notna().sum() for c in rev_m if c in df.columns)
        rev_total = len(df) * len(rev_m)
        rev_pct = rev_avail / rev_total * 100 if rev_total else 0
        rev_disabled = False
        if rev_pct < 30:
            cfg["factor_weights"] = copy.deepcopy(cfg["factor_weights"])
            old_w = cfg["factor_weights"]["revisions"]
            cfg["factor_weights"]["revisions"] = 0
            others = [k for k in cfg["factor_weights"] if k != "revisions"]
            s = sum(cfg["factor_weights"][k] for k in others)
            if s > 0:
                for k in others:
                    cfg["factor_weights"][k] += old_w * cfg["factor_weights"][k] / s
                for k in cfg["factor_weights"]:
                    cfg["factor_weights"][k] = round(cfg["factor_weights"][k], 2)
            rev_disabled = True

    # ---- CALC: Winsorize ----
    with trace_event(log, "CALC", "Winsorize metrics at 1st/99th percentiles"):
        df = winsorize_metrics(df, 0.01, 0.01)

    # ---- CALC: Sector percentiles ----
    with trace_event(log, "CALC", "Compute sector-relative percentile ranks"):
        df = compute_sector_percentiles(df)

    # ---- CALC: Category scores ----
    with trace_event(log, "CALC", "Compute within-category scores (6 categories)"):
        df = compute_category_scores(df, cfg)

    # ---- CALC: Composite ----
    with trace_event(log, "CALC", "Compute composite scores (weighted category average)"):
        df = compute_composite(df, cfg)

    # ---- CALC: Value trap ----
    with trace_event(log, "CALC", "Apply value trap flags (2-of-3 majority logic)"):
        df = apply_value_trap_flags(df, cfg)

    # ---- CALC: Rank ----
    with trace_event(log, "CALC", "Rank stocks by composite score"):
        df = rank_stocks(df)

    # ---- CALC: Factor correlation ----
    with trace_event(log, "CALC", "Compute factor correlation matrix (Spearman)"):
        corr = compute_factor_correlation(df)

    # ---- WRITE: Reports will be written by the caller ----
    log.record("INIT", "Pipeline complete — preparing reports",
               0, details=f"scored={len(df)} tickers")

    return df, cfg, log, rev_disabled


# =========================================================================
# Report generators
# =========================================================================
def generate_composite_forensics(df: pd.DataFrame, cfg: dict,
                                 report_dir: Path):
    """Generate composite_score_forensics.md for the scored tickers."""
    from factor_engine import (
        METRIC_COLS, METRIC_DIR, CAT_METRICS,
        _BANK_ONLY_METRICS, _NONBANK_ONLY_METRICS,
    )

    fw = cfg["factor_weights"]
    mw = cfg["metric_weights"]
    bank_mw = cfg.get("bank_metric_weights", {})

    lines = []
    lines.append("# Composite Score Forensics Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTickers analyzed: {', '.join(df['Ticker'].tolist())}")

    # ---- A) Pipeline Map ----
    lines.append("\n## A. Pipeline Map\n")
    lines.append("```")
    lines.append("Stage 1: Universe Selection     → get_sp500_tickers()        [factor_engine.py]")
    lines.append("Stage 2: Data Fetch             → fetch_all_tickers()        [factor_engine.py]")
    lines.append("         Market Returns          → fetch_market_returns()     [factor_engine.py]")
    lines.append("Stage 3: Metric Computation     → compute_metrics()          [factor_engine.py]")
    lines.append("Stage 4: Coverage Filter        → min_data_coverage_pct      [run_screener.py]")
    lines.append("Stage 5: Winsorization          → winsorize_metrics()        [factor_engine.py]")
    lines.append("Stage 6: Sector Percentile Rank → compute_sector_percentiles [factor_engine.py]")
    lines.append("Stage 7: Category Scores        → compute_category_scores()  [factor_engine.py]")
    lines.append("Stage 8: Composite Score        → compute_composite()        [factor_engine.py]")
    lines.append("Stage 9: Value Trap Flags       → apply_value_trap_flags()   [factor_engine.py]")
    lines.append("Stage 10: Final Ranking         → rank_stocks()              [factor_engine.py]")
    lines.append("Stage 11: Portfolio Construction→ construct_portfolio()       [portfolio_constructor.py]")
    lines.append("Stage 12: Excel + Cache Output  → write_excel/parquet        [factor_engine.py]")
    lines.append("```")

    # ---- B) Composite Score Definition ----
    lines.append("\n## B. Composite Score Definition\n")
    lines.append("### Formula\n")
    lines.append("```")
    lines.append("Composite = percentile_rank(")
    lines.append("    sum(factor_weight_i * category_score_i)  /  sum(factor_weight_i for non-NaN categories)")
    lines.append(")")
    lines.append("")
    lines.append("category_score_i = sum(metric_weight_j * metric_percentile_j)  /  sum(metric_weight_j for non-NaN metrics)")
    lines.append("")
    lines.append("metric_percentile_j = sector-relative rank (0-100), inverted for 'lower-is-better' metrics")
    lines.append("```")
    lines.append("")
    lines.append("### Factor Category Weights\n")
    lines.append("| Category | Weight (%) |")
    lines.append("| --- | --- |")
    for cat, w in fw.items():
        lines.append(f"| {cat.title()} | {w} |")

    # ---- C) Factor Definitions ----
    lines.append("\n### Factor Definitions (all metrics)\n")

    metric_defs = _get_metric_definitions()

    for cat, metrics in CAT_METRICS.items():
        generic_ws = mw.get(cat, {})
        bank_ws = bank_mw.get(cat, {}) if bank_mw else {}
        lines.append(f"\n#### {cat.title()}\n")
        lines.append("| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for m in metrics:
            d = metric_defs.get(m, {})
            gw = generic_ws.get(m, 0)
            bw = bank_ws.get(m, gw)
            direction = "Higher=Better" if METRIC_DIR.get(m, True) else "Lower=Better"
            lines.append(
                f"| `{m}` | {d.get('definition', 'N/A')} | {d.get('source', 'yfinance')} | "
                f"{d.get('cleaning', 'Winsorize 1/99 pctile')} | "
                f"{d.get('normalization', 'Sector percentile rank')} | "
                f"{gw}% | {bw}% | {direction} |"
            )

    # ---- D) Per-Ticker Breakdown ----
    lines.append("\n## C. Per-Ticker Breakdown\n")

    cat_score_cols = {
        "valuation": "valuation_score",
        "quality": "quality_score",
        "growth": "growth_score",
        "momentum": "momentum_score",
        "risk": "risk_score",
        "revisions": "revisions_score",
    }

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        is_bank = bool(row.get("_is_bank_like", False))
        lines.append(f"\n### {ticker} ({row.get('Company', 'N/A')}) — {row.get('Sector', 'Unknown')}")
        if is_bank:
            lines.append("*Bank-like stock: uses bank-specific metric weights*\n")
        lines.append("")

        # Per-metric table
        lines.append("| Metric | Raw Value | Percentile | Weight (%) | Weighted Contribution |")
        lines.append("| --- | --- | --- | --- | --- |")
        for cat, metrics in CAT_METRICS.items():
            generic_ws = mw.get(cat, {})
            bank_ws = bank_mw.get(cat, {}) if bank_mw else {}
            for m in metrics:
                raw_val = row.get(m, np.nan)
                pct_val = row.get(f"{m}_pct", np.nan)
                if is_bank:
                    w = bank_ws.get(m, generic_ws.get(m, 0))
                else:
                    w = generic_ws.get(m, 0)

                raw_str = f"{raw_val:.4f}" if pd.notna(raw_val) else "NaN"
                pct_str = f"{pct_val:.1f}" if pd.notna(pct_val) else "NaN"
                if pd.notna(pct_val) and w > 0:
                    contrib = pct_val * w / 100
                    contrib_str = f"{contrib:.2f}"
                else:
                    contrib_str = "—"
                lines.append(f"| `{m}` | {raw_str} | {pct_str} | {w} | {contrib_str} |")

        # Category scores
        lines.append("\n**Category Scores:**\n")
        lines.append("| Category | Score | Weight (%) | Weighted Contribution |")
        lines.append("| --- | --- | --- | --- |")
        for cat, col in cat_score_cols.items():
            score = row.get(col, np.nan)
            w = fw.get(cat, 0)
            score_str = f"{score:.1f}" if pd.notna(score) else "NaN"
            if pd.notna(score) and w > 0:
                contrib = score * w
                contrib_str = f"{contrib:.1f}"
            else:
                contrib_str = "—"
            lines.append(f"| {cat.title()} | {score_str} | {w} | {contrib_str} |")

        composite = row.get("Composite", np.nan)
        rank = row.get("Rank", np.nan)
        vt_flag = row.get("Value_Trap_Flag", False)
        lines.append(f"\n**Composite Score: {composite:.2f}** | **Rank: {int(rank) if pd.notna(rank) else 'N/A'}** (within {len(df)} tickers)")
        lines.append(f"Value Trap Flag: {'YES' if vt_flag else 'NO'}")

    # ---- E) Rank Summary ----
    lines.append("\n## D. Rank Summary\n")
    lines.append("| Rank | Ticker | Sector | Composite | Value Trap |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, row in df.sort_values("Rank").iterrows():
        vt = "YES" if row.get("Value_Trap_Flag", False) else "NO"
        lines.append(f"| {int(row['Rank'])} | {row['Ticker']} | {row.get('Sector', '?')} | {row['Composite']:.2f} | {vt} |")

    # ---- F) Integrity Checks ----
    lines.append("\n## E. Integrity Checks\n")

    lines.append("### 1. Look-Ahead Bias\n")
    lines.append("- **Status**: PARTIALLY MITIGATED")
    lines.append("- yfinance `.info` returns CURRENT market data (price, market cap, EV)")
    lines.append("- Financial statements (`.financials`, `.balance_sheet`, `.cashflow`) return the most recent ANNUAL filing")
    lines.append("- Forward EPS comes from current analyst consensus, which is point-in-time correct for screening (not backtesting)")
    lines.append("- **Risk**: If this screener were used for backtesting, financial statement data would be stale. For live screening, it is appropriate.")

    lines.append("\n### 2. Period Alignment\n")
    lines.append("- **Status**: PARTIAL CONCERN")
    lines.append("- Market data (price, returns, beta): TTM / trailing 1 year")
    lines.append("- Financial statements: most recent annual filing (could be 1-12 months old)")
    lines.append("- Forward EPS: consensus estimate (forward-looking)")
    lines.append("- **Risk**: Mixing TTM prices with annual financials creates a temporal mismatch. For screening (not backtesting) this is standard practice.")

    lines.append("\n### 3. Currency/Units Consistency\n")
    lines.append("- **Status**: OK (S&P 500 is USD-denominated)")
    lines.append("- All financial data from yfinance is in the stock's reporting currency")
    lines.append("- S&P 500 constituents all report in USD")

    lines.append("\n### 4. Survivorship Bias\n")
    lines.append("- **Status**: PRESENT (for backtesting), ACCEPTABLE (for live screening)")
    lines.append("- Universe = current S&P 500 constituents from Wikipedia")
    lines.append("- Companies removed from the index are excluded")
    lines.append("- For live screening purposes, this is correct (you want current constituents)")

    lines.append("\n### 5. Fallback Path Consistency\n")
    lines.append("- **Status**: PARTIAL CONCERN")
    lines.append("- Sample data path (`_generate_sample_data`) uses DIFFERENT distributions than live data")
    lines.append("- Sample data has bank-like metrics only for Financials sector, not industry-granular")
    lines.append("- Scoring pipeline is identical for both paths (winsorize, percentile, composite)")

    path = report_dir / "composite_score_forensics.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


def generate_defensibility_review(df: pd.DataFrame, cfg: dict,
                                  report_dir: Path):
    """Generate defensibility_review.md — CFA/hedge fund grade."""
    lines = []
    lines.append("# Defensibility Review — CFA / Hedge Fund Grade Assessment")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nThis review assesses the Multi-Factor Stock Screener's suitability")
    lines.append("for institutional use. Each area is scored 1-5 (1=critical deficiency, 5=institutional grade).\n")

    # ---- Scoring ----
    scores = {}

    # 1. Data Provenance
    lines.append("## 1. Data Provenance & Licensing Clarity — Score: 2/5\n")
    scores["data_provenance"] = 2
    lines.append("**Assessment**: Below institutional standard.\n")
    lines.append("- **Source**: yfinance (unofficial Yahoo Finance scraper)")
    lines.append("- **License**: yfinance is open-source but scrapes Yahoo Finance without explicit API agreement")
    lines.append("- **Concern**: Yahoo Finance ToS may prohibit automated scraping for commercial use")
    lines.append("- **Concern**: No data vendor contract or SLA — data can change format or disappear without notice")
    lines.append("- **Concern**: No audit trail for when data was sourced or which API version was used")
    lines.append("- **What's defensible**: Single, well-known source; no proprietary black-box inputs")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- `factor_engine.py:_fetch_single_ticker_inner:293` — uses `yf.Ticker(ticker_str).info` which is an undocumented, unstable endpoint")
    lines.append("- `factor_engine.py:get_sp500_tickers:63` — scrapes Wikipedia for universe, which can be edited by anyone")
    lines.append("- No data licensing documentation in the repository")

    # 2. Point-in-time correctness
    lines.append("\n## 2. Point-in-Time Correctness — Score: 3/5\n")
    scores["point_in_time"] = 3
    lines.append("**Assessment**: Adequate for live screening, inadequate for backtesting.\n")
    lines.append("- **What's right**: For a live screener, using current data is correct")
    lines.append("- **What's right**: Financial statements use `.financials` which returns the latest ANNUAL filing, not TTM")
    lines.append("- **Concern**: `forwardEps` from yfinance is a consensus estimate — source/vintage is opaque")
    lines.append("- **Concern**: `trailingEps` may be TTM or fiscal year depending on yfinance's internal logic")
    lines.append("- **Concern**: `_stmt_date_financials` is tracked but not enforced (stale filings are flagged but included)")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- `factor_engine.py:compute_metrics:823` — forward EPS growth uses `forwardEps` vs `trailingEps` but these may span different periods")
    lines.append("- `backtest.py` — acknowledged survivorship + look-ahead bias; NOT suitable for performance claims")

    # 3. Metric Definitions
    lines.append("\n## 3. Metric Definitions Consistency — Score: 4/5\n")
    scores["metric_definitions"] = 4
    lines.append("**Assessment**: Good. Well-documented with clear rationale.\n")
    lines.append("- **What's right**: 21 metrics across 6 categories, each with documented formula")
    lines.append("- **What's right**: ROIC uses balance-sheet-consistent inputs (equity, debt, cash all from BS)")
    lines.append("- **What's right**: Excess cash deduction in ROIC (cash - 2% revenue)")
    lines.append("- **What's right**: Piotroski F-Score uses raw integer (not proportional normalization)")
    lines.append("- **What's right**: Bank-specific metrics (P/B, ROE, ROA, Equity Ratio) with industry classification")
    lines.append("- **Concern**: EV/EBITDA does not fall back to EBIT (documented, but reduces coverage)")
    lines.append("- **Concern**: `_stmt_val()` substring matching could match wrong line items in edge cases")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- `factor_engine.py:_stmt_val:210` — word-boundary substring fallback could match `Operating Income` when searching for `Income` in unusual filing formats")

    # 4. Robustness
    lines.append("\n## 4. Robustness to Missing/Erroneous Data — Score: 4/5\n")
    scores["robustness"] = 4
    lines.append("**Assessment**: Good. Multiple layers of protection.\n")
    lines.append("- **What's right**: Per-row weight redistribution for NaN metrics")
    lines.append("- **What's right**: Coverage filter excludes stocks with < 60% applicable metrics")
    lines.append("- **What's right**: Auto-reduce metrics with > 70% NaN")
    lines.append("- **What's right**: Winsorization at 1st/99th percentiles")
    lines.append("- **What's right**: Revisions auto-disable when coverage < 30%")
    lines.append("- **What's right**: NaN percentiles are NOT imputed to 50th (weight redistributed)")
    lines.append("- **Concern**: Earnings yield for negative-EPS stocks is correctly negative (not NaN), but this means deeply unprofitable companies get valid (very bad) valuation scores rather than being excluded")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- None critical. The missing-data handling is well-designed.")

    # 5. Reproducibility
    lines.append("\n## 5. Reproducibility — Score: 3/5\n")
    scores["reproducibility"] = 3
    lines.append("**Assessment**: Good infrastructure, but data source is inherently non-reproducible.\n")
    lines.append("- **What's right**: `RunContext` saves config snapshot, effective weights, raw fetch artifacts")
    lines.append("- **What's right**: Config-aware cache hashing prevents stale-config results")
    lines.append("- **What's right**: Git SHA recorded in run metadata")
    lines.append("- **Concern**: yfinance returns whatever Yahoo Finance has NOW — re-running next week gives different data")
    lines.append("- **Concern**: Wikipedia universe scrape changes daily")
    lines.append("- **Concern**: No way to replay a historical run with the exact same inputs")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- `factor_engine.py:get_sp500_tickers:63` — Wikipedia scrape returns different data each run")
    lines.append("- No data snapshot/freeze mechanism for institutional audit trail")

    # 6. Performance
    lines.append("\n## 6. Performance / Scalability / Caching — Score: 3/5\n")
    scores["performance"] = 3
    lines.append("**Assessment**: Adequate for S&P 500, not designed for larger universes.\n")
    lines.append("- **What's right**: Config-aware Parquet caching with configurable freshness")
    lines.append("- **What's right**: Threaded batch fetching with adaptive rate-limit throttling")
    lines.append("- **What's right**: Retry pass for failed tickers")
    lines.append("- **Concern**: Full S&P 500 fetch takes 5-15 minutes depending on throttling")
    lines.append("- **Concern**: No incremental update — re-fetches everything on cache miss")
    lines.append("- **Concern**: `compute_metrics()` iterates row-by-row (not vectorized)")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- `factor_engine.py:compute_metrics:652` — Python loop over raw_data list; O(n) with constant overhead per ticker")

    # 7. Explainability
    lines.append("\n## 7. Explainability (IC Presentation) — Score: 4/5\n")
    scores["explainability"] = 4
    lines.append("**Assessment**: Good. Every score can be decomposed.\n")
    lines.append("- **What's right**: Composite = weighted average of 6 category scores")
    lines.append("- **What's right**: Each category score = weighted average of metric percentiles")
    lines.append("- **What's right**: Each metric has documented formula and data source")
    lines.append("- **What's right**: Factor correlation matrix available for double-counting detection")
    lines.append("- **What's right**: Value trap flag logic is transparent (2-of-3 majority)")
    lines.append("- **What's right**: Per-ticker weight redistribution is documented")
    lines.append("- **Concern**: No built-in 'explain this score' function for individual stocks")
    lines.append("")
    lines.append("**Red flags:**")
    lines.append("- None. The model is interpretable by design.")

    # ---- Overall ----
    avg_score = sum(scores.values()) / len(scores)
    lines.append(f"\n## Overall Assessment\n")
    lines.append("| Area | Score |")
    lines.append("| --- | --- |")
    for area, score in scores.items():
        label = area.replace("_", " ").title()
        lines.append(f"| {label} | {score}/5 |")
    lines.append(f"| **Average** | **{avg_score:.1f}/5** |")

    lines.append(f"\n### Grade: {'B-' if avg_score >= 3.0 else 'C+'}")
    lines.append("")
    lines.append("### What IS Defensible Today")
    lines.append("- Composite score methodology (weighted percentile ranks)")
    lines.append("- Metric definitions and formulas (well-documented, academically grounded)")
    lines.append("- Missing data handling (NaN-aware weight redistribution)")
    lines.append("- Bank-specific metric treatment")
    lines.append("- Value trap filter logic")
    lines.append("- Config validation (Pydantic schemas)")
    lines.append("- Data quality logging and alerts")
    lines.append("")
    lines.append("### What is NOT Defensible Today")
    lines.append("- Data provenance (yfinance has no SLA or licensing agreement)")
    lines.append("- Reproducibility (no data snapshot/freeze; Wikipedia universe changes)")
    lines.append("- Point-in-time correctness for backtesting")
    lines.append("- Performance claims based on `backtest.py` (known biases)")

    path = report_dir / "defensibility_review.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


def generate_recommended_changes(report_dir: Path):
    """Generate recommended_changes.md — prioritized improvements."""
    lines = []
    lines.append("# Recommended Changes — Prioritized Improvements")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\nChanges are ordered by impact on institutional defensibility.\n")

    # Priority 1
    lines.append("## Priority 1: Critical (Data Foundation)\n")

    lines.append("### 1.1 Replace yfinance with a Licensed Data Provider\n")
    lines.append("**What**: Switch from yfinance to a licensed data API (e.g., Tiingo, Polygon.io, IEX Cloud, Bloomberg B-PIPE)")
    lines.append("**Why**: yfinance scrapes Yahoo Finance without a commercial license. Data can change format, disappear, or return incorrect values without notice. No institutional investor will accept a strategy built on web scraping.")
    lines.append("**Implementation**:")
    lines.append("- Create an abstract `DataProvider` class with `fetch_ticker()`, `fetch_market_returns()` methods")
    lines.append("- Implement `YFinanceProvider` (current) and `TiingoProvider` (or similar)")
    lines.append("- Config setting: `data_provider: tiingo` with API key in env var")
    lines.append("**Impact on composite**: None (metric definitions stay the same; only data source changes)")
    lines.append("")

    lines.append("### 1.2 Freeze Universe Source\n")
    lines.append("**What**: Use a versioned, static universe file instead of scraping Wikipedia")
    lines.append("**Why**: Wikipedia can be edited by anyone, and the S&P 500 list changes during the run. This creates a race condition where the universe can differ between runs.")
    lines.append("**Implementation**:")
    lines.append("- Update `sp500_tickers.json` weekly from a reliable source (S&P Global, IEX)")
    lines.append("- Add `universe_version` field to the JSON with update timestamp")
    lines.append("- Only use Wikipedia as a cross-validation check, not primary source")
    lines.append("**Impact on composite**: Minimal (same ~503 tickers)")
    lines.append("")

    # Priority 2
    lines.append("## Priority 2: High (Reproducibility)\n")

    lines.append("### 2.1 Data Snapshot for Each Run\n")
    lines.append("**What**: Save the complete raw API response for every ticker in each run")
    lines.append("**Why**: Without a data snapshot, you cannot reproduce a prior run's results. This is a fundamental requirement for audit trails.")
    lines.append("**Implementation**:")
    lines.append("- Already partially done: `ctx.save_artifact('00_raw_fetch', raw_df)` in `run_screener.py`")
    lines.append("- Extend to save `_daily_returns` (currently excluded for size)")
    lines.append("- Add a `--replay-run <run_id>` flag that loads raw data from a prior run instead of fetching")
    lines.append("**Impact on composite**: None (observational change)")
    lines.append("")

    lines.append("### 2.2 Deterministic Scoring Mode\n")
    lines.append("**What**: Add a `--deterministic` flag that pins random seeds and uses frozen data")
    lines.append("**Why**: For regression testing and validation, you need `f(same_input) = same_output`")
    lines.append("**Implementation**:")
    lines.append("- The scoring pipeline is already deterministic given fixed inputs")
    lines.append("- Need: `--replay-run` to freeze inputs, and a golden-output test")
    lines.append("**Impact on composite**: None")
    lines.append("")

    # Priority 3
    lines.append("## Priority 3: Medium (Methodology)\n")

    lines.append("### 3.1 Add Explain Function for Individual Stocks\n")
    lines.append("**What**: `explain_score(ticker)` that prints a full decomposition of how a stock got its score")
    lines.append("**Why**: IC presentations require the ability to explain any holding. Currently requires manual Excel inspection.")
    lines.append("**Implementation**:")
    lines.append("- Function that takes a scored DataFrame row and prints the per-metric → per-category → composite waterfall")
    lines.append("- Include sector context (\"AAPL's ROIC percentile is 85th vs IT sector median of 62nd\")")
    lines.append("**Impact on composite**: None (read-only)")
    lines.append("")

    lines.append("### 3.2 Multi-Period Financial Statements\n")
    lines.append("**What**: Use TTM (trailing twelve months) instead of annual filings")
    lines.append("**Why**: Annual filings can be 1-12 months stale. TTM data is more current and reduces temporal mismatch with market data.")
    lines.append("**Implementation**:")
    lines.append("- yfinance provides both `.financials` (annual) and `.quarterly_financials`")
    lines.append("- Sum last 4 quarters for TTM revenue, net income, EBITDA, etc.")
    lines.append("- Fall back to annual if < 4 quarters available")
    lines.append("**Impact on composite**: Moderate — TTM values will differ from annual, especially for seasonal businesses")
    lines.append("")

    lines.append("### 3.3 Add Sector-Neutral Composite Mode\n")
    lines.append("**What**: Option to compute composite as percentile-within-sector instead of cross-sectional")
    lines.append("**Why**: Cross-sectional ranking can overweight sectors with structurally higher quality scores (e.g., IT). Sector-neutral ranking ensures each sector contributes proportionally.")
    lines.append("**Implementation**:")
    lines.append("- Already partially implemented: `sector_relative_composite: true` in config")
    lines.append("- Currently disabled by default")
    lines.append("- Needs testing and documentation")
    lines.append("**Impact on composite**: Significant — changes ranking order by sector")
    lines.append("")

    # Priority 4
    lines.append("## Priority 4: Low (Polish)\n")

    lines.append("### 4.1 Vectorize compute_metrics()\n")
    lines.append("**What**: Replace the per-ticker Python loop with vectorized pandas operations")
    lines.append("**Why**: ~3-5x speedup for the computation phase (currently ~1s for 500 tickers, would be ~200ms)")
    lines.append("**Implementation**: Convert `for d in raw_data` loop to DataFrame operations")
    lines.append("**Impact on composite**: None (same results, faster)")
    lines.append("")

    lines.append("### 4.2 Add CI/CD Pipeline\n")
    lines.append("**What**: GitHub Actions that runs tests, linting, and a sample-data smoke test on each push")
    lines.append("**Why**: Prevents regressions and ensures the scoring pipeline always works end-to-end")
    lines.append("**Implementation**: `.github/workflows/test.yml` with `python run_screener.py --tickers AAPL,MSFT` using sample data")
    lines.append("**Impact on composite**: None")
    lines.append("")

    lines.append("### 4.3 Add Interactive Dashboard\n")
    lines.append("**What**: Streamlit or Panel dashboard for exploring scores, factor exposures, and portfolio")
    lines.append("**Why**: Excel is limiting for interactive analysis. A dashboard allows filtering, sorting, and drill-down.")
    lines.append("**Implementation**: `streamlit_app.py` that reads `factor_output.xlsx` or cached Parquet")
    lines.append("**Impact on composite**: None (read-only visualization)")

    path = report_dir / "recommended_changes.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


# =========================================================================
# Metric definition reference (for forensics report)
# =========================================================================
def _get_metric_definitions() -> dict:
    """Return a dict of metric definitions for documentation."""
    return {
        "ev_ebitda": {
            "definition": "Enterprise Value / EBITDA",
            "source": "yfinance: enterpriseValue, EBITDA from financials",
            "cleaning": "Winsorize 1/99 pctile; NaN if EBITDA<=0 or EV<=0",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "fcf_yield": {
            "definition": "Free Cash Flow / Enterprise Value",
            "source": "yfinance: operatingCashFlow - abs(capex) / EV",
            "cleaning": "Winsorize 1/99 pctile; NaN if capex missing",
            "normalization": "Sector percentile rank",
        },
        "earnings_yield": {
            "definition": "Trailing EPS / Current Price (inverse P/E)",
            "source": "yfinance: trailingEps / currentPrice",
            "cleaning": "Winsorize 1/99 pctile; can be negative",
            "normalization": "Sector percentile rank",
        },
        "ev_sales": {
            "definition": "Enterprise Value / Total Revenue",
            "source": "yfinance: enterpriseValue / totalRevenue",
            "cleaning": "Winsorize 1/99 pctile; NaN if revenue<=0",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "pb_ratio": {
            "definition": "Price-to-Book Ratio (bank-only)",
            "source": "yfinance: priceToBook or currentPrice/bookValue",
            "cleaning": "Winsorize 1/99 pctile; NaN for non-banks",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "roic": {
            "definition": "NOPAT / Invested Capital (Eq + Debt - Excess Cash)",
            "source": "yfinance: EBIT*(1-tax) / (equity + debt_bs - excess_cash)",
            "cleaning": "Winsorize 1/99 pctile; NaN if IC<=0",
            "normalization": "Sector percentile rank",
        },
        "gross_profit_assets": {
            "definition": "Gross Profit / Total Assets",
            "source": "yfinance: grossProfit / totalAssets",
            "cleaning": "Winsorize 1/99 pctile",
            "normalization": "Sector percentile rank",
        },
        "debt_equity": {
            "definition": "Total Debt / Stockholders Equity",
            "source": "yfinance: totalDebt / totalEquity",
            "cleaning": "Winsorize 1/99 pctile; NaN if equity<=0",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "piotroski_f_score": {
            "definition": "Piotroski F-Score (0-9 integer, 9 binary signals)",
            "source": "yfinance: computed from financials/BS/CF",
            "cleaning": "NaN if < 6 testable signals",
            "normalization": "Sector percentile rank",
        },
        "accruals": {
            "definition": "(Net Income - OCF) / Total Assets",
            "source": "yfinance: (netIncome - operatingCashFlow) / totalAssets",
            "cleaning": "Winsorize 1/99 pctile",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "roe": {
            "definition": "Return on Equity (bank-only)",
            "source": "yfinance: returnOnEquity or netIncome/equity",
            "cleaning": "Winsorize 1/99 pctile; NaN for non-banks",
            "normalization": "Sector percentile rank",
        },
        "roa": {
            "definition": "Return on Assets (bank-only)",
            "source": "yfinance: returnOnAssets or netIncome/totalAssets",
            "cleaning": "Winsorize 1/99 pctile; NaN for non-banks",
            "normalization": "Sector percentile rank",
        },
        "equity_ratio": {
            "definition": "Equity / Total Assets (bank-only)",
            "source": "yfinance: totalEquity / totalAssets",
            "cleaning": "Winsorize 1/99 pctile; NaN for non-banks",
            "normalization": "Sector percentile rank",
        },
        "forward_eps_growth": {
            "definition": "(Forward EPS - Trailing EPS) / max(|Trailing EPS|, $1.00)",
            "source": "yfinance: (forwardEps - trailingEps) / max(|trailingEps|, 1.0)",
            "cleaning": "Clamped to [-75%, +150%]; NaN if trailing EPS near zero",
            "normalization": "Sector percentile rank",
        },
        "peg_ratio": {
            "definition": "P/E / Forward EPS Growth Rate (%)",
            "source": "Computed from price/trailingEps / (forward_eps_growth * 100)",
            "cleaning": "NaN if growth <= 1%",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "revenue_growth": {
            "definition": "(Revenue - Revenue_Prior) / Revenue_Prior",
            "source": "yfinance: totalRevenue (col 0 vs col 1 from financials)",
            "cleaning": "Winsorize 1/99 pctile; NaN if prior=0",
            "normalization": "Sector percentile rank",
        },
        "sustainable_growth": {
            "definition": "ROE * Retention Ratio (1 - Payout Ratio)",
            "source": "yfinance: (NI/Eq) * (1 - dividendsPaid/NI)",
            "cleaning": "Winsorize 1/99 pctile; NaN if NI<=0 or equity<=0",
            "normalization": "Sector percentile rank",
        },
        "return_12_1": {
            "definition": "12-1 Month Return (skip most recent month)",
            "source": "yfinance: (price_1m_ago - price_12m_ago) / price_12m_ago",
            "cleaning": "Winsorize 1/99 pctile; uses calendar-based lookback",
            "normalization": "Sector percentile rank",
        },
        "return_6m": {
            "definition": "6-1 Month Return (skip most recent month)",
            "source": "yfinance: (price_1m_ago - price_6m_ago) / price_6m_ago",
            "cleaning": "Winsorize 1/99 pctile; uses calendar-based lookback",
            "normalization": "Sector percentile rank",
        },
        "volatility": {
            "definition": "Annualized daily return std dev (252 trading days)",
            "source": "yfinance: std(daily_log_returns) * sqrt(252)",
            "cleaning": "Winsorize 1/99 pctile; NaN if < 200 trading days",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "beta": {
            "definition": "Cov(stock, market) / Var(market) using daily returns",
            "source": "yfinance: stock daily returns vs ^GSPC daily returns",
            "cleaning": "Winsorize 1/99 pctile; NaN if < 200 common dates",
            "normalization": "Sector percentile rank (inverted: lower=better)",
        },
        "analyst_surprise": {
            "definition": "Median earnings surprise across last 4 quarters",
            "source": "yfinance: (epsActual - epsEstimate) / max(|epsEstimate|, $0.10)",
            "cleaning": "NaN if < 2 quarters; denominator floored at $0.10",
            "normalization": "Sector percentile rank",
        },
        "price_target_upside": {
            "definition": "(Analyst Target Mean Price - Current Price) / Current Price",
            "source": "yfinance: targetMeanPrice / currentPrice - 1",
            "cleaning": "Clamped [-50%, +100%]; requires >= 3 covering analysts",
            "normalization": "Sector percentile rank",
        },
    }


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()

    # Determine tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TEST_TICKERS

    print("=" * 60)
    print("  MULTI-FACTOR SCREENER — INSTRUMENTED AUDIT RUN")
    print("=" * 60)
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Mode: {'Sample data' if args.sample else 'Live data (fallback to sample)'}")
    print(f"  Report dir: {REPORT_DIR}")
    print("=" * 60)

    # ---- Run instrumented pipeline ----
    print("\nPhase 1: Running instrumented pipeline...")
    t0 = time.time()
    df, cfg, event_log, rev_disabled = run_instrumented_pipeline(
        tickers, force_sample=args.sample
    )
    pipeline_time = time.time() - t0
    print(f"  Pipeline complete: {len(df)} tickers scored in {pipeline_time:.1f}s")
    print(f"  Events recorded: {len(event_log.events)}")
    if rev_disabled:
        print("  Revisions auto-disabled (coverage < 30%)")

    # ---- Flush event log ----
    print("\nPhase 2: Writing run log...")
    with trace_event(event_log, "WRITE", "Flush run log to CSV/MD/HTML",
                     details=f"path={REPORT_DIR}"):
        event_log.flush_all(REPORT_DIR)
    # Re-flush so the WRITE event itself is included
    event_log.flush_all(REPORT_DIR)
    print(f"  Written: run_log_full.csv, run_log_full.md, run_log_full.html")

    # ---- Generate forensics report ----
    print("\nPhase 3: Generating composite score forensics...")
    t0_forensics = time.time()
    forensics_path = generate_composite_forensics(df, cfg, REPORT_DIR)
    print(f"  Written: {forensics_path} ({time.time() - t0_forensics:.1f}s)")

    # ---- Generate defensibility review ----
    print("\nPhase 4: Generating defensibility review...")
    t0_def = time.time()
    def_path = generate_defensibility_review(df, cfg, REPORT_DIR)
    print(f"  Written: {def_path} ({time.time() - t0_def:.1f}s)")

    # ---- Generate recommended changes ----
    print("\nPhase 5: Generating recommended changes...")
    t0_rec = time.time()
    rec_path = generate_recommended_changes(REPORT_DIR)
    print(f"  Written: {rec_path} ({time.time() - t0_rec:.1f}s)")

    # ---- Summary ----
    total_time = time.time() - (t0 - pipeline_time)  # Total wall time
    print("\n" + "=" * 60)
    print("  AUDIT RUN COMPLETE")
    print("=" * 60)
    print(f"  Tickers scored:           {len(df)}")
    print(f"  Events logged:            {len(event_log.events)}")
    print(f"  Revisions auto-disabled:  {'YES' if rev_disabled else 'NO'}")
    print(f"  Total time:               {pipeline_time:.1f}s")
    print()
    print("  Report files:")
    for fname in ["run_log_full.csv", "run_log_full.md", "run_log_full.html",
                   "composite_score_forensics.md", "defensibility_review.md",
                   "recommended_changes.md"]:
        fpath = REPORT_DIR / fname
        size_kb = fpath.stat().st_size / 1024 if fpath.exists() else 0
        print(f"    {fname:40s} {size_kb:.1f} KB")
    print("=" * 60)

    # Top results
    print("\n  Scored Tickers:")
    for _, row in df.sort_values("Rank").iterrows():
        vt = " [VALUE TRAP]" if row.get("Value_Trap_Flag", False) else ""
        bank = " [BANK]" if row.get("_is_bank_like", False) else ""
        print(f"    {int(row['Rank']):2d}. {row['Ticker']:6s} {str(row.get('Sector','')):26s} "
              f"Composite={row['Composite']:.2f}{vt}{bank}")


if __name__ == "__main__":
    main()
