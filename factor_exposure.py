#!/usr/bin/env python3
"""
Factor-Exposure Diagnostics — Standalone Script
=================================================
Regresses the portfolio's daily excess returns against the Fama-French 5
factors + Momentum (UMD) to quantify how much of the portfolio's return
is explained by known factor premia vs. genuine alpha.

Usage:
    python factor_exposure.py
    python factor_exposure.py --start 2024-01-01 --end 2025-12-31

Requirements (beyond the main screener):
    pip install pandas-datareader statsmodels

Output: printed table of factor betas, t-stats, p-values, R-squared,
        and annualized alpha.
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lazy imports (these are optional deps)
# ---------------------------------------------------------------------------
def _check_deps():
    try:
        import pandas_datareader  # noqa: F401
        import statsmodels  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e.name}")
        print("Install with:  pip install pandas-datareader statsmodels")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Fama-French factor data
# ---------------------------------------------------------------------------
def fetch_ff_factors(start: str, end: str) -> pd.DataFrame:
    """Download FF5 + Momentum factors from Kenneth French Data Library."""
    import pandas_datareader.data as web

    # FF 5 factors (daily)
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                         start=start, end=end)[0]
    ff5.index = pd.to_datetime(ff5.index.astype(str))

    # Momentum factor (daily)
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                         start=start, end=end)[0]
    mom.index = pd.to_datetime(mom.index.astype(str))

    # FF data is in percent — convert to decimal
    ff5 = ff5 / 100.0
    mom = mom / 100.0

    # Merge
    factors = ff5.join(mom, how="inner")
    # Standardize column names
    factors.columns = [c.strip() for c in factors.columns]
    # Rename Mom factor to UMD if needed
    if "Mom" in factors.columns:
        factors = factors.rename(columns={"Mom": "UMD"})
    elif "WML" in factors.columns:
        factors = factors.rename(columns={"WML": "UMD"})

    return factors


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------
def compute_portfolio_returns(tickers: list, start: str, end: str) -> pd.Series:
    """Fetch daily prices and compute equal-weighted portfolio returns."""
    import yfinance as yf

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    # Daily returns
    rets = prices.pct_change().dropna(how="all")

    # Equal-weighted portfolio return
    port_ret = rets.mean(axis=1)
    port_ret.name = "Portfolio"
    return port_ret


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def run_factor_regression(port_ret: pd.Series, factors: pd.DataFrame):
    """OLS: portfolio excess return ~ MktRF + SMB + HML + RMW + CMA + UMD."""
    import statsmodels.api as sm

    # Align dates
    merged = pd.concat([port_ret, factors], axis=1, join="inner").dropna()
    if len(merged) < 30:
        print(f"Only {len(merged)} overlapping observations — too few for regression.")
        sys.exit(1)

    # Excess return = portfolio return - risk-free rate
    y = merged["Portfolio"] - merged["RF"]

    # Factor regressors
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    if "UMD" in merged.columns:
        factor_cols.append("UMD")

    X = merged[factor_cols]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model, len(merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Factor-exposure diagnostics")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD). Default: 1 year ago.")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD). Default: today.")
    args = parser.parse_args()

    _check_deps()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Load portfolio from latest screener run
    print("Loading latest portfolio...")
    try:
        from portfolio_constructor import load_latest_scores, construct_portfolio
        from factor_engine import load_config
        cfg = load_config()
        df, _ = load_latest_scores()
        portfolio = construct_portfolio(df, cfg)
        tickers = portfolio["Ticker"].tolist()
    except Exception as e:
        print(f"Failed to load portfolio: {e}")
        print("Run the screener first (python run_screener.py)")
        sys.exit(1)

    if not tickers:
        print("Empty portfolio — run the screener first.")
        sys.exit(1)

    print(f"Portfolio: {len(tickers)} stocks")
    print(f"Period: {start_date} to {end_date}")

    # Fetch data
    print("\nFetching Fama-French factors...")
    factors = fetch_ff_factors(start_date, end_date)
    print(f"  {len(factors)} trading days of factor data")

    print("Fetching portfolio daily prices...")
    port_ret = compute_portfolio_returns(tickers, start_date, end_date)
    print(f"  {len(port_ret)} trading days of portfolio returns")

    # Regression
    print("\nRunning OLS regression: Portfolio_ExRet ~ Mkt-RF + SMB + HML + RMW + CMA + UMD")
    print("=" * 75)
    model, n_obs = run_factor_regression(port_ret, factors)

    # Results
    print(f"\nObservations: {n_obs}")
    print(f"R-squared:    {model.rsquared:.4f}")
    print(f"Adj R-sq:     {model.rsquared_adj:.4f}")
    print(f"F-stat:       {model.fvalue:.2f} (p={model.f_pvalue:.4e})")

    # Annualized alpha (daily alpha * 252)
    daily_alpha = model.params.get("const", 0)
    ann_alpha = daily_alpha * 252 * 100  # in percent
    alpha_t = model.tvalues.get("const", 0)
    alpha_p = model.pvalues.get("const", 1)

    print(f"\nAnnualized Alpha: {ann_alpha:+.2f}% (t={alpha_t:.2f}, p={alpha_p:.4f})")
    sig = "***" if alpha_p < 0.01 else "**" if alpha_p < 0.05 else "*" if alpha_p < 0.10 else ""
    print(f"  Significance: {sig if sig else 'Not significant at 10%'}")

    print(f"\n{'Factor':<12} {'Beta':>8} {'t-stat':>8} {'p-value':>10}")
    print("-" * 40)
    for factor in model.params.index:
        if factor == "const":
            continue
        beta = model.params[factor]
        t = model.tvalues[factor]
        p = model.pvalues[factor]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"{factor:<12} {beta:>8.4f} {t:>8.2f} {p:>10.4f} {sig}")

    print("\nInterpretation:")
    print("  - Mkt-RF beta ≈ 1.0 means market-neutral exposure")
    print("  - SMB > 0 → small-cap tilt; HML > 0 → value tilt")
    print("  - RMW > 0 → quality/profitability tilt")
    print("  - CMA > 0 → conservative investment tilt")
    print("  - UMD > 0 → momentum tilt")
    print("  - Positive alpha → returns unexplained by known factors")


if __name__ == "__main__":
    main()
