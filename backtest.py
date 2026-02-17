#!/usr/bin/env python3
"""
Multi-Factor Stock Screener — Phase 2: Backtest & IC Validation
================================================================
Simplified backtesting module that:
  A. Fetches monthly price history (2020-01 to present).
  B. Simulates monthly factor scores (static fundamentals, dynamic momentum/risk).
  C. Runs decile backtest with transaction costs.
  D. Computes Information Coefficients (Spearman) per factor category.
  E. Validates value trap filters (Pure Value vs Filtered).
  F. Writes CSV results to ./validation/.

Reference: Multi-Factor-Screener-Blueprint.md §5

IMPORTANT DISCLAIMERS:
  * Survivorship bias: Uses current S&P 500 constituents throughout the
    backtest period. Stocks that were removed or went bankrupt are excluded.
  * Look-ahead bias: Fundamental scores (Valuation, Quality, Growth,
    Revisions) are held constant from the Phase 1 snapshot. They were NOT
    available at each historical rebalance date. Only Momentum and Risk
    metrics are recomputed from trailing prices.
  * These results are for MODEL VALIDATION ONLY and do NOT represent
    achievable live trading performance.
"""

import copy
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# Import shared logic from factor_engine
from factor_engine import (
    load_config,
    get_sp500_tickers,
    METRIC_COLS,
    METRIC_DIR,
    CAT_METRICS,
    winsorize_metrics,
    compute_sector_percentiles,
    compute_category_scores,
    compute_composite,
    ROOT,
    CACHE_DIR,
)

VALIDATION_DIR = ROOT / "validation"
VALIDATION_DIR.mkdir(exist_ok=True)
HIST_SCORES_DIR = CACHE_DIR / "historical_scores"
HIST_SCORES_DIR.mkdir(exist_ok=True)

BACKTEST_START = "2020-01-01"
RF_ANNUAL = 0.045          # Risk-free rate for Sharpe
TCOST_BPS = 10             # 10 bps one-way transaction cost


# =========================================================================
# A. Historical price data
# =========================================================================
def _load_or_fetch_prices(tickers: list, start: str) -> pd.DataFrame:
    """Return DataFrame of monthly adjusted close prices (columns = tickers)."""
    cache_path = CACHE_DIR / "backtest_prices.parquet"

    # Reuse cache if < 7 days old
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime) < timedelta(days=7):
            print("[CACHE HIT] Loading backtest_prices from cache")
            return pd.read_parquet(cache_path)

    print("[CACHE MISS] Fetching backtest_prices from API")
    try:
        import yfinance as yf
        data = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            interval="1mo",
            group_by="ticker",
            threads=True,
        )
        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.xs("Close", axis=1, level=1)
        else:
            prices = data[["Close"]].copy()
            prices.columns = tickers[:1]
        prices.to_parquet(str(cache_path))
        return prices
    except Exception as e:
        print(f"  yfinance download failed: {e}")
        return pd.DataFrame()


def _generate_sample_prices(tickers: list, sectors: dict,
                            start: str, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic monthly prices for offline/sandbox testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, datetime.now(), freq="ME")

    _sector_drift = {
        "Information Technology": 0.014, "Health Care": 0.010,
        "Financials": 0.009, "Consumer Discretionary": 0.011,
        "Communication Services": 0.010, "Industrials": 0.009,
        "Consumer Staples": 0.007, "Energy": 0.008,
        "Utilities": 0.006, "Real Estate": 0.007, "Materials": 0.008,
    }
    _sector_vol = {
        "Information Technology": 0.065, "Health Care": 0.055,
        "Financials": 0.060, "Consumer Discretionary": 0.065,
        "Communication Services": 0.060, "Industrials": 0.050,
        "Consumer Staples": 0.035, "Energy": 0.075,
        "Utilities": 0.035, "Real Estate": 0.055, "Materials": 0.055,
    }

    records = {}
    for t in tickers:
        sec = sectors.get(t, "Industrials")
        mu = _sector_drift.get(sec, 0.008)
        sigma = _sector_vol.get(sec, 0.055)
        # Generate log-normal price path
        monthly_ret = rng.normal(mu, sigma, size=len(dates))
        # Add a shared market component for realism
        mkt = rng.normal(0.008, 0.04, size=len(dates))
        total_ret = monthly_ret * 0.6 + mkt * 0.4
        prices = 100 * np.exp(np.cumsum(total_ret))
        # Add some idiosyncratic variation to starting price
        prices *= rng.uniform(0.3, 5.0)
        records[t] = prices

    df = pd.DataFrame(records, index=dates)
    cache_path = CACHE_DIR / "backtest_prices.parquet"
    df.to_parquet(str(cache_path))
    return df


# =========================================================================
# B. Historical factor score simulation
# =========================================================================
def _recompute_momentum_risk(prices_df: pd.DataFrame, month_end: pd.Timestamp,
                             ticker: str) -> dict:
    """Recompute momentum and risk metrics from trailing price data."""
    result = {}
    # Get prices up to month_end
    mask = prices_df.index <= month_end
    if ticker not in prices_df.columns:
        return {k: np.nan for k in ["return_12_1", "return_6m", "volatility", "beta"]}

    series = prices_df.loc[mask, ticker].dropna()
    if len(series) < 7:
        return {k: np.nan for k in ["return_12_1", "return_6m", "volatility", "beta"]}

    latest = series.iloc[-1]

    # 12-1 month return (skip most recent month)
    if len(series) >= 13:
        p_12m_ago = series.iloc[-13]
        p_1m_ago = series.iloc[-2]
        result["return_12_1"] = (p_1m_ago - p_12m_ago) / p_12m_ago if p_12m_ago > 0 else np.nan
    else:
        result["return_12_1"] = np.nan

    # 6-1 month return (exclude most recent month for consistency with 12-1M)
    if len(series) >= 7:
        p_6m_ago = series.iloc[-7]
        p_1m_ago = series.iloc[-2]
        result["return_6m"] = (p_1m_ago - p_6m_ago) / p_6m_ago if p_6m_ago > 0 else np.nan
    else:
        result["return_6m"] = np.nan

    # Volatility (annualized from monthly returns)
    if len(series) >= 13:
        monthly_ret = series.pct_change().dropna().iloc[-12:]
        result["volatility"] = float(monthly_ret.std() * np.sqrt(12))
    else:
        result["volatility"] = np.nan

    # Beta — use ^GSPC column if present, otherwise skip
    if "^GSPC" in prices_df.columns and len(series) >= 13:
        mkt = prices_df.loc[mask, "^GSPC"].dropna()
        if len(mkt) >= 13:
            stock_ret = series.pct_change().dropna().iloc[-12:]
            mkt_ret = mkt.pct_change().dropna().iloc[-12:]
            # Align
            common = stock_ret.index.intersection(mkt_ret.index)
            if len(common) >= 10:
                sr = stock_ret.loc[common].values
                mr = mkt_ret.loc[common].values
                var_m = np.var(mr, ddof=1)
                result["beta"] = np.cov(sr, mr)[0, 1] / var_m if var_m > 0 else np.nan
            else:
                result["beta"] = np.nan
        else:
            result["beta"] = np.nan
    else:
        result["beta"] = np.nan

    return result


def simulate_monthly_scores(base_scores: pd.DataFrame,
                            prices_df: pd.DataFrame,
                            cfg: dict) -> dict:
    """For each month-end, blend static fundamentals with dynamic momentum/risk.

    Returns dict of {YYYYMM: DataFrame with scores}.
    """
    months = sorted(prices_df.index)
    # Need at least 13 months for 12-1 momentum
    if len(months) < 14:
        months = months[13:]
    else:
        months = months[13:]

    # Static fundamental columns (everything except momentum and risk)
    dynamic_cols = ["return_12_1", "return_6m", "volatility", "beta"]
    static_cols = [c for c in METRIC_COLS if c not in dynamic_cols]

    all_scores = {}
    tickers = base_scores["Ticker"].tolist()

    for i, month_end in enumerate(months):
        yyyymm = month_end.strftime("%Y%m")
        cache_path = HIST_SCORES_DIR / f"scores_{yyyymm}.parquet"

        # Check cache
        if cache_path.exists():
            all_scores[yyyymm] = pd.read_parquet(str(cache_path))
            continue

        # Start with static fundamentals from Phase 1 snapshot
        df = base_scores[["Ticker", "Company", "Sector"] + static_cols].copy()

        # Recompute dynamic metrics
        for idx, row in df.iterrows():
            t = row["Ticker"]
            dyn = _recompute_momentum_risk(prices_df, month_end, t)
            for k, v in dyn.items():
                df.at[idx, k] = v

        # Ensure all METRIC_COLS exist
        for c in METRIC_COLS:
            if c not in df.columns:
                df[c] = np.nan

        # Winsorize
        df = winsorize_metrics(df, 0.01, 0.01)

        # Sector percentiles
        df = compute_sector_percentiles(df)

        # Category scores
        df = compute_category_scores(df, cfg)

        # Composite
        df = compute_composite(df, cfg)

        # Save to cache
        keep = [c for c in df.columns if not c.startswith("_")]
        df[keep].to_parquet(str(cache_path), index=False)
        all_scores[yyyymm] = df

        if (i + 1) % 12 == 0 or i == len(months) - 1:
            print(f"  Scored {i+1}/{len(months)} months (latest: {yyyymm})")

    return all_scores


# =========================================================================
# C. Decile backtest
# =========================================================================
def _monthly_forward_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-month forward returns for each ticker at each month-end."""
    return prices_df.pct_change().shift(-1)


def _assign_deciles(scores: pd.Series, n_deciles: int = 10) -> pd.Series:
    """Assign decile labels (1 = worst, 10 = best) based on composite score."""
    # Use labels=False to avoid label/bin count mismatch when duplicates are dropped
    return pd.qcut(scores, q=n_deciles, labels=False,
                   duplicates="drop") + 1


def run_decile_backtest(monthly_scores: dict, forward_returns: pd.DataFrame,
                        prices_df: pd.DataFrame) -> dict:
    """Run decile backtest. Returns dict with performance data."""
    months = sorted(monthly_scores.keys())
    # Align: we need forward returns for each month
    fwd = forward_returns.copy()

    decile_returns = {d: [] for d in range(1, 11)}
    decile_months = []
    prev_decile_members = {d: set() for d in range(1, 11)}

    for yyyymm in months:
        scores_df = monthly_scores[yyyymm]
        # Map month string to datetime
        month_dt = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:]}-01") + pd.offsets.MonthEnd(0)

        # Find closest price date
        valid_dates = fwd.index[fwd.index >= month_dt - timedelta(days=5)]
        if len(valid_dates) == 0:
            continue
        price_date = valid_dates[0]

        # Check forward returns exist
        if price_date not in fwd.index:
            continue

        # Get forward returns for this month
        fwd_row = fwd.loc[price_date]

        # Assign deciles
        valid = scores_df[["Ticker", "Composite"]].dropna(subset=["Composite"])
        if len(valid) < 20:
            continue

        try:
            valid = valid.copy()
            valid["Decile"] = _assign_deciles(valid["Composite"])
        except Exception:
            continue

        decile_months.append(yyyymm)

        for d in range(1, 11):
            members = valid.loc[valid["Decile"] == d, "Ticker"].tolist()
            # Compute equal-weighted return
            rets = []
            for t in members:
                if t in fwd_row.index and pd.notna(fwd_row[t]):
                    rets.append(fwd_row[t])
            if len(rets) == 0:
                decile_returns[d].append(0.0)
                continue

            port_ret = np.mean(rets)

            # Transaction cost: 10 bps per one-way for turnover
            old_members = prev_decile_members[d]
            new_members = set(members)
            if len(old_members) > 0:
                turnover = len(new_members.symmetric_difference(old_members)) / max(len(new_members), 1)
            else:
                turnover = 1.0  # First month = full turnover
            tcost = turnover * (TCOST_BPS / 10000)
            port_ret -= tcost

            decile_returns[d].append(port_ret)
            prev_decile_members[d] = new_members

    # Benchmark (S&P 500)
    bench_returns = []
    if "^GSPC" in prices_df.columns:
        bench_fwd = prices_df["^GSPC"].pct_change().shift(-1)
        for yyyymm in decile_months:
            month_dt = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:]}-01") + pd.offsets.MonthEnd(0)
            valid_dates = bench_fwd.index[bench_fwd.index >= month_dt - timedelta(days=5)]
            if len(valid_dates) > 0 and pd.notna(bench_fwd.loc[valid_dates[0]]):
                bench_returns.append(bench_fwd.loc[valid_dates[0]])
            else:
                bench_returns.append(0.0)
    else:
        # Approximate: equal-weight all tickers
        for i in range(len(decile_months)):
            all_rets = []
            for d in range(1, 11):
                if i < len(decile_returns[d]):
                    all_rets.append(decile_returns[d][i])
            bench_returns.append(np.mean(all_rets) if all_rets else 0.0)

    return {
        "decile_returns": decile_returns,
        "bench_returns": bench_returns,
        "months": decile_months,
    }


def _perf_metrics(monthly_returns: list) -> dict:
    """Compute annualized return, vol, Sharpe, max drawdown from monthly returns."""
    arr = np.array(monthly_returns)
    n = len(arr)
    if n == 0:
        return {"ann_return": 0, "ann_vol": 0, "sharpe": 0, "max_dd": 0}

    # Geometric annualized return
    cum = np.prod(1 + arr)
    years = n / 12
    if cum <= 0:
        ann_ret = -1.0  # Total loss
    elif years > 0:
        ann_ret = cum ** (1 / years) - 1
    else:
        ann_ret = 0

    # Annualized vol
    ann_vol = np.std(arr, ddof=1) * np.sqrt(12) if n > 1 else 0

    # Sharpe
    rf_monthly = (1 + RF_ANNUAL) ** (1 / 12) - 1
    excess = arr - rf_monthly
    sharpe = (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(12) if np.std(excess, ddof=1) > 0 else 0

    # Max drawdown
    cum_ret = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cum_ret)
    drawdowns = cum_ret / running_max - 1
    max_dd = np.min(drawdowns)

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


# =========================================================================
# D. Information Coefficient calculation
# =========================================================================
def compute_ic_series(monthly_scores: dict, forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly Spearman IC for composite and each category."""
    months = sorted(monthly_scores.keys())
    categories = ["Composite", "valuation_score", "quality_score", "growth_score",
                   "momentum_score", "risk_score", "revisions_score"]
    records = []

    for yyyymm in months:
        scores_df = monthly_scores[yyyymm]
        month_dt = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:]}-01") + pd.offsets.MonthEnd(0)

        valid_dates = forward_returns.index[
            forward_returns.index >= month_dt - timedelta(days=5)
        ]
        if len(valid_dates) == 0:
            continue
        price_date = valid_dates[0]
        if price_date not in forward_returns.index:
            continue

        fwd_row = forward_returns.loc[price_date]
        rec = {"month": yyyymm}

        for cat in categories:
            if cat not in scores_df.columns:
                rec[cat] = np.nan
                continue
            # Merge scores with forward returns
            merged = scores_df[["Ticker", cat]].dropna()
            merged = merged.set_index("Ticker")
            fwd_vals = fwd_row.reindex(merged.index).dropna()
            common = merged.index.intersection(fwd_vals.index)
            if len(common) < 20:
                rec[cat] = np.nan
                continue
            rho, _ = sp_stats.spearmanr(merged.loc[common, cat], fwd_vals.loc[common])
            rec[cat] = rho

        records.append(rec)

    return pd.DataFrame(records)


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IC stats: mean, std, t-stat, % positive."""
    categories = [c for c in ic_df.columns if c != "month"]
    rows = []
    for cat in categories:
        vals = ic_df[cat].dropna()
        n = len(vals)
        if n < 3:
            rows.append({"factor": cat, "mean_ic": np.nan, "ic_std": np.nan,
                         "ic_tstat": np.nan, "pct_positive": np.nan})
            continue
        mean_ic = vals.mean()
        ic_std = vals.std(ddof=1)
        ic_tstat = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
        pct_pos = (vals > 0).sum() / n * 100
        rows.append({
            "factor": cat,
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_tstat": ic_tstat,
            "pct_positive": pct_pos,
        })
    return pd.DataFrame(rows)


# =========================================================================
# E. Value trap filter validation
# =========================================================================
def run_value_trap_comparison(monthly_scores: dict,
                              forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Build 3 portfolios monthly, compute performance."""
    months = sorted(monthly_scores.keys())

    port_returns = {"Pure_Value": [], "Val_Qual": [], "Val_Qual_Mom": []}

    for yyyymm in months:
        scores_df = monthly_scores[yyyymm]
        month_dt = pd.Timestamp(f"{yyyymm[:4]}-{yyyymm[4:]}-01") + pd.offsets.MonthEnd(0)

        valid_dates = forward_returns.index[
            forward_returns.index >= month_dt - timedelta(days=5)
        ]
        if len(valid_dates) == 0:
            continue
        price_date = valid_dates[0]
        fwd_row = forward_returns.loc[price_date]

        df = scores_df[["Ticker", "valuation_score", "quality_score",
                         "momentum_score"]].dropna(subset=["valuation_score"]).copy()
        n = len(df)
        if n < 20:
            continue
        top_n = max(1, n // 5)  # Top quintile

        def _port_ret(tickers_list):
            rets = [fwd_row[t] for t in tickers_list
                    if t in fwd_row.index and pd.notna(fwd_row[t])]
            return np.mean(rets) if rets else 0.0

        # A: Pure Value — top 20% by valuation_score
        a = df.nlargest(top_n, "valuation_score")
        port_returns["Pure_Value"].append(_port_ret(a["Ticker"].tolist()))

        # B: Value + Quality — exclude bottom 30% quality, then top 20% by value
        qual_thresh = df["quality_score"].quantile(0.30)
        b_pool = df[df["quality_score"] >= qual_thresh]
        b_top = b_pool.nlargest(top_n, "valuation_score")
        port_returns["Val_Qual"].append(_port_ret(b_top["Ticker"].tolist()))

        # C: Value + Quality + Momentum — exclude bottom 30% quality AND momentum
        mom_thresh = df["momentum_score"].quantile(0.30)
        c_pool = df[(df["quality_score"] >= qual_thresh) &
                     (df["momentum_score"] >= mom_thresh)]
        c_top = c_pool.nlargest(top_n, "valuation_score")
        port_returns["Val_Qual_Mom"].append(_port_ret(c_top["Ticker"].tolist()))

    # Compute metrics
    rows = []
    for name, rets in port_returns.items():
        m = _perf_metrics(rets)
        rows.append({"Portfolio": name, **m})
    return pd.DataFrame(rows)


# =========================================================================
# F. Output & diagnostics
# =========================================================================
_BACKTEST_DISCLAIMER = (
    "# DISCLAIMER: ILLUSTRATIVE ONLY — NOT REPRESENTATIVE OF ACHIEVABLE PERFORMANCE\n"
    "# This backtest uses current financial data and today's index constituents applied\n"
    "# retroactively. Survivorship bias and look-ahead bias are present.\n"
    "# Fundamental scores (Valuation, Quality, Growth, Revisions) are held constant from\n"
    "# the most recent snapshot. Only Momentum and Risk are recomputed monthly.\n"
    "# Do NOT use these results for strategy validation or investor marketing.\n"
)


def write_outputs(decile_perf: dict, ic_summary: pd.DataFrame,
                  ic_df: pd.DataFrame, vtf: pd.DataFrame):
    """Write all three CSV files with disclaimers."""
    for path, df in [
        (VALIDATION_DIR / "backtest_results.csv", decile_perf),
        (VALIDATION_DIR / "factor_ic_timeseries.csv", ic_df),
        (VALIDATION_DIR / "value_trap_comparison.csv", vtf),
    ]:
        with open(path, "w", newline="") as f:
            f.write(_BACKTEST_DISCLAIMER)
            df.to_csv(f, index=False)


def print_summary(bt_result, decile_perf_df, ic_summary, vtf_df,
                  n_months, avg_universe, start_dt, end_dt, t0):
    elapsed = round(time.time() - t0, 1)
    hist_files = list(HIST_SCORES_DIR.glob("*.parquet"))

    print()
    print("============================================")
    print("  BACKTEST — RUN SUMMARY")
    print("============================================")
    print(f"Backtest period:          {start_dt} to {end_dt}")
    print(f"Months tested:            {n_months}")
    print(f"Universe size (avg):      {avg_universe} tickers/month")
    print(f"Survivorship bias:        YES (using current constituents)")
    print(f"Fundamental scores:       STATIC (Phase 1 snapshot)")
    print(f"Momentum/Risk scores:     DYNAMIC (recomputed monthly)")
    print("--------------------------------------------")
    print("DECILE PERFORMANCE (Annualized):")
    print(f"  {'Decile':<14s}| {'Return':>7s} | {'Vol':>6s} | {'Sharpe':>6s} | {'MaxDD':>7s}")

    for _, row in decile_perf_df.iterrows():
        label = row["Decile"]
        ar = row["ann_return"] * 100
        av = row["ann_vol"] * 100
        sh = row["sharpe"]
        md = row["max_dd"] * 100
        print(f"  {label:<14s}| {ar:>6.1f}% | {av:>5.1f}% | {sh:>6.2f} | {md:>6.1f}%")

    # Long-short
    d10 = decile_perf_df.loc[decile_perf_df["Decile"] == "D10", "ann_return"]
    d1 = decile_perf_df.loc[decile_perf_df["Decile"] == "D1", "ann_return"]
    if len(d10) > 0 and len(d1) > 0:
        ls = (d10.values[0] - d1.values[0]) * 100
        print(f"  {'L/S (D10-D1)':<14s}| {ls:>6.1f}%")

    # Monotonicity check
    decile_rets = []
    for d in range(1, 11):
        row = decile_perf_df.loc[decile_perf_df["Decile"] == f"D{d}"]
        if len(row) > 0:
            decile_rets.append(row["ann_return"].values[0])
    inversions = sum(1 for i in range(len(decile_rets) - 1)
                     if decile_rets[i] > decile_rets[i + 1])
    mono = "PASS" if inversions <= 2 else f"FAIL ({inversions} inversions)"
    print("--------------------------------------------")
    print(f"MONOTONICITY CHECK: {mono}")
    print(f"(PASS if D10 > D9 > ... > D1 with <= 2 inversions)")

    # IC table
    print("--------------------------------------------")
    print("INFORMATION COEFFICIENTS:")
    print(f"  {'Factor':<14s}| {'Mean IC':>8s} | {'IC t-stat':>9s} | {'% Positive':>10s}")
    name_map = {
        "Composite": "Composite", "valuation_score": "Valuation",
        "quality_score": "Quality", "growth_score": "Growth",
        "momentum_score": "Momentum", "risk_score": "Risk",
        "revisions_score": "Revisions",
    }
    for _, row in ic_summary.iterrows():
        label = name_map.get(row["factor"], row["factor"])
        mic = row["mean_ic"]
        tst = row["ic_tstat"]
        pp = row["pct_positive"]
        if pd.notna(mic):
            print(f"  {label:<14s}| {mic:>8.3f} | {tst:>9.2f} | {pp:>9.0f}%")
        else:
            print(f"  {label:<14s}|      N/A |       N/A |        N/A")

    # Value trap comparison
    print("--------------------------------------------")
    print("VALUE TRAP FILTER COMPARISON:")
    print(f"  {'Portfolio':<14s}| {'Ann. Return':>11s} | {'Sharpe':>6s} | {'MaxDD':>7s}")
    for _, row in vtf_df.iterrows():
        label = row["Portfolio"]
        ar = row["ann_return"] * 100
        sh = row["sharpe"]
        md = row["max_dd"] * 100
        print(f"  {label:<14s}| {ar:>10.1f}% | {sh:>6.2f} | {md:>6.1f}%")

    print("--------------------------------------------")
    print("DISCLAIMER:")
    print("  Look-ahead bias:        YES (static fundamentals from latest snapshot)")
    print("  Survivorship bias:      YES (current constituents used throughout)")
    print("  *** FOR MODEL VALIDATION ONLY -- NOT REPRESENTATIVE OF LIVE PERFORMANCE ***")
    print("--------------------------------------------")
    print("Output files:")
    print(f"  validation/backtest_results.csv")
    print(f"  validation/factor_ic_timeseries.csv")
    print(f"  validation/value_trap_comparison.csv")
    print("Cache files:")
    print(f"  cache/backtest_prices.parquet")
    print(f"  cache/historical_scores/scores_YYYYMM.parquet (x{len(hist_files)})")
    print(f"Total runtime:            {elapsed}s")
    print("============================================")


# =========================================================================
# MAIN
# =========================================================================
def main():
    t0 = time.time()

    # ---- Load config & universe ----
    print("Loading configuration...")
    cfg = load_config()

    # Handle revisions auto-disable (same logic as factor_engine)
    # We'll check after loading base scores

    print("Loading S&P 500 universe...")
    universe_df = get_sp500_tickers(cfg)
    tickers = universe_df["Ticker"].tolist()
    sectors = dict(zip(universe_df["Ticker"], universe_df["Sector"]))
    print(f"  {len(tickers)} tickers")

    # ---- Load Phase 1 base scores ----
    print("Loading Phase 1 factor scores...")
    from factor_engine import _find_latest_cache
    scores_path, _ = _find_latest_cache("factor_scores")
    if scores_path is None:
        print("  ERROR: No Phase 1 factor_scores cache found. Run factor_engine.py first.")
        return
    base_scores = pd.read_parquet(str(scores_path))
    print(f"  Loaded {len(base_scores)} scored tickers from {scores_path.name}")

    # Check revisions coverage and auto-disable if needed
    rev_m = ["analyst_surprise", "eps_revision_ratio", "eps_estimate_change"]
    rev_avail = sum(base_scores[c].notna().sum() for c in rev_m if c in base_scores.columns)
    rev_total = len(base_scores) * len(rev_m)
    rev_pct = rev_avail / rev_total * 100 if rev_total else 0
    if rev_pct < 30:
        print(f"  Revisions coverage {rev_pct:.1f}% < 30%; auto-disabling revisions weight")
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

    # ---- A. Historical price data ----
    print(f"\nFetching monthly prices ({BACKTEST_START} to present)...")
    # Add ^GSPC for benchmark / beta
    all_tickers = tickers + ["^GSPC"]
    prices_df = _load_or_fetch_prices(all_tickers, BACKTEST_START)

    if prices_df.empty or len(prices_df) < 14:
        print("  Network unavailable — generating synthetic monthly prices")
        print("  NOTE: Survivorship bias present (current constituents used throughout)")
        prices_df = _generate_sample_prices(all_tickers, sectors, BACKTEST_START)
    else:
        print(f"  Loaded {len(prices_df)} months x {len(prices_df.columns)} tickers")
        print("  NOTE: Survivorship bias present (current constituents used throughout)")

    # ---- B. Historical score simulation ----
    print("\nSimulating monthly factor scores...")
    print("  WARNING: Fundamental metrics (Valuation, Quality, Growth, Revisions) held")
    print("  constant from Phase 1 snapshot. Only Momentum/Risk are recomputed monthly.")
    monthly_scores = simulate_monthly_scores(base_scores, prices_df, cfg)
    n_months = len(monthly_scores)
    print(f"  {n_months} monthly score snapshots generated")

    avg_universe = int(np.mean([len(v) for v in monthly_scores.values()]))

    # ---- C. Decile backtest ----
    print("\nRunning decile backtest...")
    fwd_returns = _monthly_forward_returns(prices_df)
    bt = run_decile_backtest(monthly_scores, fwd_returns, prices_df)

    # Build decile performance table
    rows = []
    for d in range(1, 11):
        m = _perf_metrics(bt["decile_returns"][d])
        rows.append({"Decile": f"D{d}", **m})
    # Benchmark
    bm = _perf_metrics(bt["bench_returns"])
    rows.append({"Decile": "Benchmark", **bm})
    decile_perf_df = pd.DataFrame(rows)

    # ---- D. Information Coefficients ----
    print("Computing Information Coefficients...")
    ic_df = compute_ic_series(monthly_scores, fwd_returns)
    ic_summary = summarize_ic(ic_df)

    # ---- E. Value trap filter validation ----
    print("Running value trap filter comparison...")
    vtf_df = run_value_trap_comparison(monthly_scores, fwd_returns)

    # ---- F. Write outputs ----
    print("Writing output files...")
    write_outputs(decile_perf_df, ic_summary, ic_df, vtf_df)

    # Determine backtest date range
    if bt["months"]:
        start_dt = bt["months"][0][:4] + "-" + bt["months"][0][4:]
        end_dt = bt["months"][-1][:4] + "-" + bt["months"][-1][4:]
    else:
        start_dt, end_dt = "N/A", "N/A"

    print_summary(bt, decile_perf_df, ic_summary, vtf_df,
                  n_months, avg_universe, start_dt, end_dt, t0)


if __name__ == "__main__":
    main()
