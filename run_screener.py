#!/usr/bin/env python3
"""
Multi-Factor Stock Screener v1.0 — Master Entry Point
=======================================================
Single-command pipeline:
    python run_screener.py              # Full run
    python run_screener.py --refresh    # Force-clear cache
    python run_screener.py --tickers AAPL,MSFT,GOOGL
    python run_screener.py --no-portfolio

Reference: Multi-Factor-Screener-Blueprint.md §8, §10, Appendix C/E
"""

import argparse
import copy
import csv
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from run_context import RunContext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
VALIDATION_DIR = ROOT / "validation"
CACHE_DIR.mkdir(exist_ok=True)
VALIDATION_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data Quality Logger (Appendix C)
# ---------------------------------------------------------------------------
_DQ_LOG_ROWS: list = []

DQ_COLUMNS = ["Timestamp", "Ticker", "Issue_Type", "Severity",
              "Description", "Action_Taken"]


def dq_log(ticker: str, issue_type: str, severity: str,
           description: str, action: str):
    """Append one row to the in-memory data quality log."""
    _DQ_LOG_ROWS.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker,
        "Issue_Type": issue_type,
        "Severity": severity,
        "Description": description,
        "Action_Taken": action,
    })


def flush_dq_log():
    """Write data_quality_log.csv to ./validation/."""
    path = VALIDATION_DIR / "data_quality_log.csv"
    df = pd.DataFrame(_DQ_LOG_ROWS, columns=DQ_COLUMNS)
    df.to_csv(str(path), index=False)
    return str(path), len(df)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Factor Stock Screener v1.0")
    p.add_argument("--refresh", action="store_true",
                   help="Force-clear all cache and re-fetch everything")
    p.add_argument("--tickers", type=str, default="",
                   help="Comma-separated tickers for quick testing "
                        "(e.g. AAPL,MSFT,GOOGL)")
    p.add_argument("--no-portfolio", action="store_true",
                   help="Skip portfolio construction; only write FactorScores")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loader with error handling
# ---------------------------------------------------------------------------
def load_config_safe():
    """Load config.yaml with clear error on failure."""
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        print(f"\n  ERROR: config.yaml not found at {config_path}")
        print("  Create config.yaml from the template in Appendix B of the blueprint.")
        sys.exit(1)
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("config.yaml is empty or malformed")
        return cfg
    except Exception as e:
        print(f"\n  ERROR: Failed to parse config.yaml: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Factor Engine integration (with resilience)
# ---------------------------------------------------------------------------
def run_factor_engine(cfg, args, ctx=None):
    """Run the complete factor scoring pipeline. Returns (scored_df, stats_dict)."""
    from factor_engine import (
        get_sp500_tickers, fetch_single_ticker, fetch_all_tickers,
        fetch_market_returns, compute_metrics,
        _generate_sample_data, _find_latest_cache,
        winsorize_metrics, compute_sector_percentiles,
        compute_category_scores, compute_composite,
        apply_value_trap_flags, rank_stocks,
        compute_factor_correlation,
        write_scores_parquet, METRIC_COLS, METRIC_DIR,
    )

    stats = {
        "cache_status": "COLD",
        "tickers_api": 0,
        "tickers_cache": 0,
        "tickers_failed": 0,
        "failed_list": [],
        "fetch_time": 0.0,
        "scored": 0,
        "scoring_time": 0.0,
    }

    # ---- Universe ----
    print("Loading S&P 500 universe...")
    universe_df = get_sp500_tickers(cfg)

    # --tickers override
    if args.tickers:
        custom = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        universe_df = universe_df[universe_df["Ticker"].isin(custom)].copy()
        if universe_df.empty:
            # Create minimal entries for custom tickers not in the universe
            records = [{"Ticker": t, "Company": t, "Sector": "Unknown"} for t in custom]
            universe_df = pd.DataFrame(records)
        print(f"  Custom ticker subset: {list(universe_df['Ticker'])}")

    tickers = universe_df["Ticker"].tolist()
    universe_size = len(tickers)
    print(f"  Universe: {universe_size} tickers")
    ticker_meta = universe_df.set_index("Ticker")[["Company", "Sector"]].to_dict("index")

    # ---- Check cache freshness (config-aware) ----
    fresh_days = cfg.get("caching", {}).get("fundamental_data_refresh_days", 7)
    cfg_hash = ctx.config_hash(cfg) if ctx else None
    cached_path, cached_dt = _find_latest_cache("factor_scores", config_hash=cfg_hash)
    use_cache = False

    if args.refresh:
        print("  --refresh: clearing factor scores cache")
        for f in CACHE_DIR.glob("factor_scores_*.parquet"):
            try:
                f.unlink()
            except Exception:
                pass
        stats["cache_status"] = "COLD"
    elif cached_path is not None:
        age_days = (datetime.now() - cached_dt).days
        if age_days <= fresh_days and not args.tickers:
            use_cache = True
            stats["cache_status"] = "HOT"
            stats["tickers_cache"] = universe_size
        else:
            stats["cache_status"] = "WARM"

    if use_cache:
        print(f"  [CACHE HIT] Loading scores from {cached_path.name}")
        df = pd.read_parquet(str(cached_path))
        stats["scored"] = len(df)
        return df, stats

    # ---- Fetch data ----
    fetch_t0 = time.time()
    USE_SAMPLE = False

    print("\nTesting network connectivity...")
    try:
        # Use max_retries=1 for the probe to fail fast when offline
        test_rec = fetch_single_ticker(tickers[0], max_retries=1)
        if "_error" in test_rec:
            raise RuntimeError(test_rec["_error"])
        print("  Network OK — fetching live data")
    except Exception as e:
        print(f"  Network unavailable ({type(e).__name__})")
        print("  Generating sector-realistic sample data")
        USE_SAMPLE = True

    skipped_tickers = []

    if USE_SAMPLE:
        df = _generate_sample_data(universe_df)
        stats["tickers_cache"] = len(df)

        # Log fetch failures for sample mode
        for t in tickers:
            dq_log(t, "fetch_failure", "High",
                   "Network unavailable — using synthetic data",
                   "Generated sector-realistic sample values")
    else:
        # Live fetch with retry resilience
        print(f"\nFetching market returns...")
        market_returns = fetch_market_returns()
        print(f"  {len(market_returns)} daily observations")

        print(f"\nFetching data for {universe_size} tickers...")
        raw = fetch_all_tickers(tickers)
        stats["tickers_api"] = len(raw)

        # Identify failures and log per-ticker timing
        fetch_times = []
        for rec in raw:
            t = rec.get("Ticker", "?")
            ft = rec.get("_fetch_time_ms", 0)
            fetch_times.append(ft)
            if ctx is not None:
                ctx.log.debug(f"Fetched {t}", extra={
                    "ticker": t, "fetch_time_ms": ft,
                    "phase": "fetch",
                    "step": "error" if "_error" in rec else "ok",
                })
            if "_error" in rec:
                skipped_tickers.append(t)
                dq_log(t, "fetch_failure", "High",
                       f"yfinance error: {rec['_error'][:80]}",
                       "Excluded from scoring")
        if fetch_times:
            import statistics
            ft_arr = [x for x in fetch_times if x > 0]
            if ft_arr:
                print(f"  Fetch timing: min={min(ft_arr)}ms  mean={int(statistics.mean(ft_arr))}ms  "
                      f"max={max(ft_arr)}ms  p95={int(sorted(ft_arr)[int(len(ft_arr)*0.95)])}ms")

        print("Computing metrics...")
        df = compute_metrics(raw, market_returns)

        # Always use Wikipedia GICS sector names (yfinance uses different
        # names like "Technology" vs "Information Technology").
        for idx, row in df.iterrows():
            t = row["Ticker"]
            if t in ticker_meta:
                df.at[idx, "Sector"] = ticker_meta[t]["Sector"]
                if pd.isna(row.get("Company")) or row.get("Company") == t:
                    df.at[idx, "Company"] = ticker_meta[t]["Company"]

        # Remove fully-failed rows
        skip_mask = df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)
        skipped_tickers += df.loc[skip_mask, "Ticker"].tolist()
        df = df[~skip_mask].copy()

        # Coverage filter
        present = [c for c in METRIC_COLS if c in df.columns]
        df["_mc"] = df[present].notna().sum(axis=1)
        min_m = max(1, int(len(present) * cfg["data_quality"]["min_data_coverage_pct"] / 100))
        low = df["_mc"] < min_m
        for t in df.loc[low, "Ticker"]:
            dq_log(t, "missing_metric", "Medium",
                   f"Insufficient metric coverage (< {cfg['data_quality']['min_data_coverage_pct']}%)",
                   "Excluded from scoring")
        skipped_tickers += df.loc[low, "Ticker"].tolist()
        df = df[~low].copy()

    stats["fetch_time"] = round(time.time() - fetch_t0, 1)
    stats["tickers_failed"] = len(skipped_tickers)
    stats["failed_list"] = skipped_tickers[:20]

    # ---- Save raw metrics artifact (data lineage) ----
    if ctx is not None:
        ctx.save_artifact("01_raw_metrics", df)
        ctx.save_universe(df["Ticker"].tolist(), skipped_tickers)

    # ---- Data quality checks (§4.7) ----
    _run_data_quality_checks(df)

    # ---- Warn if > 20% failed ----
    if universe_size > 0 and len(skipped_tickers) / universe_size > 0.20:
        print(f"\n  *** WARNING: {len(skipped_tickers)}/{universe_size} tickers failed "
              f"({len(skipped_tickers)/universe_size*100:.0f}%). Results may be unreliable. ***")

    # ---- Revisions auto-disable ----
    rev_m = ["analyst_surprise", "eps_revision_ratio", "eps_estimate_change"]
    rev_avail = sum(df[c].notna().sum() for c in rev_m if c in df.columns)
    rev_total = len(df) * len(rev_m)
    rev_pct = rev_avail / rev_total * 100 if rev_total else 0
    rev_disabled = False

    if rev_pct < 30:
        print(f"\n!! Revisions coverage {rev_pct:.1f}% < 30%; auto-disabling")
        # Deep copy to avoid mutating the original config dict
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

    stats["rev_disabled"] = rev_disabled

    # ---- Scoring pipeline ----
    score_t0 = time.time()
    print("Winsorizing at 1st / 99th percentiles...")
    df = winsorize_metrics(df, 0.01, 0.01)

    # Log winsorized outliers
    for col in METRIC_COLS:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 10:
                p01, p99 = s.quantile(0.01), s.quantile(0.99)
                clipped = ((s <= p01) | (s >= p99)).sum()
                if clipped > 0:
                    dq_log("UNIVERSE", "outlier_winsorized", "Low",
                           f"{col}: {clipped} values clipped at 1st/99th pctile",
                           "Winsorized to boundary values")

    if ctx is not None:
        ctx.save_artifact("02_winsorized", df)

    print("Computing sector-relative percentile ranks...")
    df = compute_sector_percentiles(df)

    if ctx is not None:
        ctx.save_artifact("03_percentiles", df)

    print("Computing within-category scores...")
    df = compute_category_scores(df, cfg)

    if ctx is not None:
        ctx.save_artifact("04_category_scores", df)

    print("Computing composite scores...")
    df = compute_composite(df, cfg)

    print("Applying value trap flags...")
    df = apply_value_trap_flags(df, cfg)

    print("Ranking stocks...")
    df = rank_stocks(df)

    if ctx is not None:
        ctx.save_artifact("05_final_scored", df)

    stats["scoring_time"] = round(time.time() - score_t0, 1)
    stats["scored"] = len(df)

    # ---- Missing data stats ----
    labels = [
        ("EV/EBITDA", "ev_ebitda"), ("FCF Yield", "fcf_yield"),
        ("Earnings Yield", "earnings_yield"), ("EV/Sales", "ev_sales"),
        ("ROIC", "roic"), ("Gross Profit/Assets", "gross_profit_assets"),
        ("Debt/Equity", "debt_equity"), ("Piotroski F-Score", "piotroski_f_score"),
        ("Accruals", "accruals"), ("Forward EPS Growth", "forward_eps_growth"),
        ("Revenue Growth", "revenue_growth"), ("Sustainable Growth", "sustainable_growth"),
        ("12-1 Month Return", "return_12_1"), ("6-Month Return", "return_6m"),
        ("Volatility", "volatility"), ("Beta", "beta"),
        ("Analyst Surprise", "analyst_surprise"),
    ]
    stats["missing_pct"] = {}
    for lbl, col in labels:
        pct = df[col].isna().sum() / len(df) * 100 if col in df.columns else 100
        stats["missing_pct"][lbl] = round(pct, 1)

    # ---- Metric coverage drift alerts ----
    alert_threshold = cfg.get("data_quality", {}).get("metric_alert_threshold_pct", 50)
    drift_alerts = 0
    for lbl, pct in stats["missing_pct"].items():
        if pct > alert_threshold:
            print(f"  WARNING: {lbl} is {pct:.1f}% missing (threshold: {alert_threshold}%)")
            dq_log("UNIVERSE", "metric_drift", "High",
                   f"{lbl} missing {pct:.1f}% > {alert_threshold}% threshold",
                   "Flagged for review")
            drift_alerts += 1
    stats["drift_alerts"] = drift_alerts

    # ---- Factor correlation matrix (for transparency) ----
    corr = compute_factor_correlation(df)
    if ctx is not None and not corr.empty:
        ctx.save_artifact("06_factor_correlation", corr.reset_index())

    # ---- Write Parquet cache (config-aware) ----
    print("Writing cache Parquet...")
    try:
        write_scores_parquet(df, config_hash=cfg_hash)
    except Exception as e:
        print(f"  WARNING: Failed to write Parquet cache: {e}")

    return df, stats


def _run_data_quality_checks(df: pd.DataFrame):
    """Run §4.7 data quality guardrails and log issues (vectorized)."""
    if df.empty:
        return

    # Market cap outlier (vectorized)
    if "marketCap" in df.columns:
        mc = df["marketCap"]
        low_mask = mc.notna() & (mc < 100e6)
        for idx in df.index[low_mask]:
            v = mc.at[idx]
            dq_log(df.at[idx, "Ticker"],
                   "market_cap_outlier", "High",
                   f"Market Cap = ${v/1e6:.0f}M (< $100M threshold)",
                   "Flagged for review")
        high_mask = mc.notna() & (mc > 5e12)
        for idx in df.index[high_mask]:
            v = mc.at[idx]
            dq_log(df.at[idx, "Ticker"],
                   "market_cap_outlier", "High",
                   f"Market Cap = ${v/1e12:.1f}T (> $5T threshold)",
                   "Flagged for review")

    # Negative EV (vectorized)
    if "enterpriseValue" in df.columns:
        ev = df["enterpriseValue"]
        neg_mask = ev.notna() & (ev < 0)
        for idx in df.index[neg_mask]:
            v = ev.at[idx]
            dq_log(df.at[idx, "Ticker"], "negative_ev", "High",
                   f"EV = ${v/1e6:.0f}M (negative enterprise value)",
                   "EV-based metrics set to NaN")

    # Revenue discontinuity (vectorized)
    if "totalRevenue" in df.columns and "totalRevenue_prior" in df.columns:
        rev = df["totalRevenue"]
        rev_p = df["totalRevenue_prior"]
        disc_mask = rev.notna() & rev_p.notna() & (rev_p > 0) & (rev < 0.10 * rev_p)
        for idx in df.index[disc_mask]:
            r, rp = rev.at[idx], rev_p.at[idx]
            dq_log(df.at[idx, "Ticker"], "revenue_discontinuity", "High",
                   f"Revenue TTM = ${r/1e6:.0f}M vs prior ${rp/1e6:.0f}M ({r/rp*100:.0f}%)",
                   "Flagged for manual review")

    # Missing critical metrics (vectorized)
    critical = ["ev_ebitda", "roic", "return_12_1"]
    for col in critical:
        if col in df.columns:
            miss_mask = df[col].isna()
            for idx in df.index[miss_mask]:
                dq_log(df.at[idx, "Ticker"], "missing_metric", "Medium",
                       f"{col} is missing/NaN",
                       "Assigned median percentile (50th)")


# ---------------------------------------------------------------------------
# Portfolio construction integration
# ---------------------------------------------------------------------------
def run_portfolio_construction(df, cfg):
    """Run portfolio construction. Returns (portfolio_df, stats_dict)."""
    from portfolio_constructor import (
        construct_portfolio, compute_portfolio_stats,
    )

    stats = {"construction_time": 0.0}
    port_t0 = time.time()

    print("\nConstructing model portfolio...")
    port = construct_portfolio(df, cfg)
    stats_data = compute_portfolio_stats(port, cfg)

    stats["construction_time"] = round(time.time() - port_t0, 1)
    stats.update(stats_data)

    # Detect capped sectors
    max_sec = cfg.get("portfolio", {}).get("max_sector_concentration", 8)
    sec_cts = port["Sector"].value_counts()
    stats["capped_sectors"] = [s for s, c in sec_cts.items() if c >= max_sec]

    return port, stats


# ---------------------------------------------------------------------------
# Excel writer integration
# ---------------------------------------------------------------------------
def write_excel_safe(df, port, port_stats, cfg, no_portfolio):
    """Write factor_output.xlsx with error handling for locked files."""
    out_path = ROOT / cfg["output"]["excel_file"]

    try:
        if no_portfolio:
            # Write single-sheet FactorScores only
            from factor_engine import write_excel
            write_excel(df, cfg)
            return str(out_path), 1
        else:
            from portfolio_constructor import write_full_excel
            write_full_excel(df, port, port_stats, cfg)
            return str(out_path), 3
    except PermissionError:
        print(f"\n  ERROR: Cannot write {cfg['output']['excel_file']}.")
        print(f"  Close the file in Excel and re-run the screener.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: Failed to write Excel: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Diagnostics printer
# ---------------------------------------------------------------------------
def print_full_summary(args, fe_stats, port_stats, dq_counts,
                       excel_path, n_sheets, n_cache_files, total_time):
    # CLI flags string
    flags = []
    if args.refresh:
        flags.append("--refresh")
    if args.tickers:
        flags.append(f"--tickers {args.tickers}")
    if args.no_portfolio:
        flags.append("--no-portfolio")
    flags_str = ", ".join(flags) if flags else "none"

    print()
    print("============================================")
    print("  MULTI-FACTOR SCREENER v1.0 — FULL RUN")
    print("============================================")
    print(f"Config loaded:            config.yaml")
    print(f"Universe:                 S&P 500 ({fe_stats['scored']} tickers)")
    print(f"CLI flags:                {flags_str}")
    print("--------------------------------------------")

    # DATA FETCH
    print("DATA FETCH:")
    print(f"  Cache status:           {fe_stats['cache_status']}")
    print(f"  Tickers fetched (API):  {fe_stats['tickers_api']}")
    print(f"  Tickers loaded (cache): {fe_stats['tickers_cache']}")
    print(f"  Tickers failed:         {fe_stats['tickers_failed']}"
          + (f"  {fe_stats['failed_list']}" if fe_stats['failed_list'] else ""))
    print(f"  Fetch time:             {fe_stats['fetch_time']}s")
    print("--------------------------------------------")

    # SCORING
    print("SCORING:")
    print(f"  Tickers scored:         {fe_stats['scored']}")
    print("  Missing % by metric:")
    for lbl, pct in fe_stats.get("missing_pct", {}).items():
        print(f"    {lbl + ':':<24s} {pct:.1f}%")
    print(f"  Revisions auto-disabled: {'YES' if fe_stats.get('rev_disabled') else 'NO'}")
    print(f"  Scoring time:           {fe_stats['scoring_time']}s")
    print("--------------------------------------------")

    # PORTFOLIO
    if port_stats:
        print("PORTFOLIO:")
        print(f"  Holdings:               {port_stats.get('n_stocks', 0)} stocks")
        capped = port_stats.get("capped_sectors", [])
        print(f"  Sectors capped:         {', '.join(capped) if capped else 'none'}")
        print(f"  Avg Composite:          {port_stats.get('avg_composite', 0)}")
        print(f"  Portfolio Beta:          {port_stats.get('avg_beta', 0):.2f}")
        print(f"  Est. Yield:             {port_stats.get('est_div_yield', 0):.2f}%")
        print(f"  Construction time:       {port_stats.get('construction_time', 0)}s")
        print("--------------------------------------------")
    else:
        print("PORTFOLIO:                (skipped — --no-portfolio)")
        print("--------------------------------------------")

    # DATA QUALITY
    print("DATA QUALITY:")
    total_issues = sum(dq_counts.values())
    print(f"  Issues logged:          {total_issues} total")
    print(f"    High severity:        {dq_counts.get('High', 0)}")
    print(f"    Medium severity:      {dq_counts.get('Medium', 0)}")
    print(f"    Low severity:         {dq_counts.get('Low', 0)}")
    print(f"    Drift alerts:         {fe_stats.get('drift_alerts', 0)}")
    print(f"  Log file:               validation/data_quality_log.csv")
    print("--------------------------------------------")

    # OUTPUT
    check = "OK"
    sheet_str = f"{n_sheets} sheet{'s' if n_sheets > 1 else ''}"
    print("OUTPUT:")
    print(f"  factor_output.xlsx      {check}  ({sheet_str})")
    print(f"  cache/*.parquet         {check}  ({n_cache_files} files)")
    print(f"  data_quality_log.csv    {check}")
    print(f"  README.md               {check}")
    print("--------------------------------------------")
    print(f"Total runtime:            {total_time}s")
    print("============================================")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    args = parse_args()

    # Clear any residual DQ log entries from prior import/run
    _DQ_LOG_ROWS.clear()

    # ---- 0. Run context (reproducibility) ----
    ctx = RunContext()

    print("============================================")
    print(f"  MULTI-FACTOR SCREENER v1.0  [run_id={ctx.run_id}]")
    print("============================================")

    # ---- 1. Config ----
    print("Loading configuration...")
    cfg = load_config_safe()
    ctx.save_config(cfg)
    ctx.log.info("Config loaded", extra={"phase": "init"})

    # ---- 2. Clear factor scores cache if --refresh ----
    if args.refresh:
        print("Clearing factor scores cache...")
        for f in CACHE_DIR.glob("factor_scores_*.parquet"):
            try:
                f.unlink()
            except PermissionError:
                print(f"  WARNING: Could not delete {f.name} (locked)")
            except Exception as e:
                print(f"  WARNING: Could not delete {f.name}: {e}")

    # ---- 3. Factor engine ----
    df, fe_stats = run_factor_engine(cfg, args, ctx=ctx)

    # ---- 4. Portfolio construction ----
    port = None
    port_stats = None
    if not args.no_portfolio:
        port, port_stats = run_portfolio_construction(df, cfg)
        print(f"  Selected {port_stats['n_stocks']} stocks")
    else:
        print("\nSkipping portfolio construction (--no-portfolio)")

    # ---- 5. Write Excel ----
    print("\nWriting Excel workbook...")
    excel_path, n_sheets = write_excel_safe(
        df, port, port_stats if port_stats else {}, cfg, args.no_portfolio)
    print(f"  Written: {excel_path} ({n_sheets} sheets)")

    # ---- 6. Data quality log ----
    dq_path, dq_total = flush_dq_log()

    # Count by severity
    dq_counts = {"High": 0, "Medium": 0, "Low": 0}
    for row in _DQ_LOG_ROWS:
        sev = row.get("Severity", "Low")
        dq_counts[sev] = dq_counts.get(sev, 0) + 1

    # ---- 7. Count cache files ----
    n_cache_files = len(list(CACHE_DIR.glob("*.parquet")))

    # ---- 8. Ensure README exists ----
    readme_exists = (ROOT / "README.md").exists()

    total_time = round(time.time() - t0, 1)

    # ---- 9. Print full diagnostics ----
    print_full_summary(args, fe_stats, port_stats, dq_counts,
                       excel_path, n_sheets, n_cache_files, total_time)

    # ---- 10. Save run metadata ----
    ctx.save_effective_weights(cfg)
    ctx.save_metadata({
        "cli_flags": {
            "refresh": args.refresh,
            "tickers": args.tickers or None,
            "no_portfolio": args.no_portfolio,
        },
        "factor_engine_stats": fe_stats,
        "portfolio_stats": port_stats,
        "data_quality_counts": dq_counts,
        "total_time_seconds": total_time,
    })
    print(f"\n  Run artifacts saved to: runs/{ctx.run_id}/")


if __name__ == "__main__":
    main()
