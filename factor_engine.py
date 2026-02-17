#!/usr/bin/env python3
"""
Multi-Factor Stock Screener - Phase 1: Factor Engine
=====================================================
Computes composite factor scores for the S&P 500 universe using six
factor categories (Valuation, Quality, Growth, Momentum, Risk, Analyst
Revisions) and writes results to Excel + Parquet cache.

Reference: Multi-Factor-Screener-Blueprint.md (Version 2.0)

Network behaviour
-----------------
* Primary path: scrape S&P 500 from Wikipedia, fetch data via yfinance.
* Fallback path: if network is unavailable (sandbox / CI), load tickers
  from sp500_tickers.json and generate sector-realistic sample data so
  the full scoring pipeline can be validated end-to-end.
"""

import copy
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mstats
from openpyxl import Workbook

warnings.filterwarnings("ignore")

# =========================================================================
# A. Load configuration
# =========================================================================
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================================================================
# B. Get S&P 500 ticker list
# =========================================================================
def get_sp500_tickers(cfg: dict) -> pd.DataFrame:
    """Return DataFrame with Ticker, Company, Sector columns.

    Primary: scrape Wikipedia.
    Fallback: load from sp500_tickers.json shipped with the project.
    """
    df = None

    # --- Primary: Wikipedia scrape ---
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["Ticker", "Company", "Sector"]
        df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
        print("  Loaded S&P 500 list from Wikipedia")
    except Exception as e:
        print(f"  Wikipedia scrape failed ({type(e).__name__}), using local fallback")

    # --- Fallback: embedded JSON ---
    if df is None or df.empty:
        fallback = ROOT / "sp500_tickers.json"
        if fallback.exists():
            with open(fallback) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"  Loaded {len(df)} tickers from sp500_tickers.json")
        else:
            raise FileNotFoundError(
                "No network access and sp500_tickers.json not found. "
                "Cannot determine universe."
            )

    # Apply config exclusions
    exclude_tickers = cfg["universe"].get("exclude_tickers", [])
    exclude_sectors = cfg["universe"].get("exclude_sectors", [])
    if exclude_tickers:
        df = df[~df["Ticker"].isin(exclude_tickers)]
    if exclude_sectors:
        df = df[~df["Sector"].isin(exclude_sectors)]

    df = df.drop_duplicates(subset="Ticker").reset_index(drop=True)
    return df


# =========================================================================
# C. Tiered Parquet caching (Blueprint SS7.4)
# =========================================================================
def _find_latest_cache(tier_name: str):
    """Find most recent cache file for a tier. Returns (path, date) or (None, None)."""
    files = sorted(CACHE_DIR.glob(f"{tier_name}_*.parquet"), reverse=True)
    for f in files:
        try:
            date_str = f.stem.rsplit("_", 1)[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            return f, file_date
        except ValueError:
            continue
    return None, None


def cache_is_fresh(tier_name: str, max_age_days: int) -> bool:
    path, dt = _find_latest_cache(tier_name)
    if path is None:
        return False
    return (datetime.now() - dt) < timedelta(days=max_age_days)


def load_cache(tier_name: str) -> pd.DataFrame:
    path, _ = _find_latest_cache(tier_name)
    print(f"[CACHE HIT] Loading {tier_name} from cache")
    return pd.read_parquet(path)


def save_cache(tier_name: str, df: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y%m%d")
    path = CACHE_DIR / f"{tier_name}_{today}.parquet"
    df.to_parquet(str(path), index=False)
    return str(path)


# =========================================================================
# D. Fetch data from yfinance in batches
# =========================================================================
def _safe(d: dict, key: str, default=np.nan):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default


def _stmt_val(stmt, label, col=0, default=np.nan):
    """Pull a value from a yfinance financial-statement DataFrame.

    Matching strategy: exact match first, then word-boundary substring
    fallback (requires target to appear as a contiguous word sequence).
    """
    try:
        if stmt is None or stmt.empty:
            return default
        target = label.lower().strip()
        # Pass 1: exact match (case-insensitive, stripped)
        for idx in stmt.index:
            if target == str(idx).lower().strip():
                vals = stmt.loc[idx].dropna()
                if len(vals) > col:
                    return float(vals.iloc[col])
        # Pass 2: word-boundary substring fallback — the target words
        # must appear as a contiguous sequence within the index label,
        # but only when the label starts with or ends with the target
        # (avoids "operating income" matching "net income from
        # continuing operation").
        for idx in stmt.index:
            idx_low = str(idx).lower().strip()
            if idx_low.startswith(target) or idx_low.endswith(target):
                vals = stmt.loc[idx].dropna()
                if len(vals) > col:
                    return float(vals.iloc[col])
        return default
    except Exception:
        return default


_NON_RETRYABLE_PATTERNS = ["404", "no data", "not found", "delisted"]


def fetch_single_ticker(ticker_str: str, max_retries: int = 3) -> dict:
    """Fetch all required data for one ticker via yfinance.

    Implements exponential backoff retry (1s / 2s / 4s) per §10.3.
    Returns dict with '_fetch_time_ms' for per-ticker timing.
    Non-retryable errors (404, delisted) fail immediately.
    """
    t_start = time.time()
    rec = {"Ticker": ticker_str}
    last_err = None
    for attempt in range(max_retries):
        try:
            rec = _fetch_single_ticker_inner(ticker_str)
            if "_error" not in rec:
                rec["_fetch_time_ms"] = round((time.time() - t_start) * 1000)
                return rec
            last_err = rec.get("_error", "unknown")
            # Skip retry for permanent errors
            if rec.get("_non_retryable", False):
                break
        except Exception as exc:
            last_err = str(exc)
        # Exponential backoff: 1s, 2s, 4s
        if attempt < max_retries - 1:
            delay = 2 ** attempt
            time.sleep(delay)
    rec = {"Ticker": ticker_str, "_error": f"Failed after {max_retries} retries: {last_err}"}
    rec["_fetch_time_ms"] = round((time.time() - t_start) * 1000)
    return rec


def _fetch_single_ticker_inner(ticker_str: str) -> dict:
    """Inner fetch logic for one ticker (called by retry wrapper)."""
    rec = {"Ticker": ticker_str}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker_str)
        info = t.info or {}

        # ---- info fields ----
        rec["marketCap"]          = _safe(info, "marketCap")
        rec["enterpriseValue"]    = _safe(info, "enterpriseValue")
        rec["trailingEps"]        = _safe(info, "trailingEps")
        rec["forwardEps"]         = _safe(info, "forwardEps")
        rec["currentPrice"]       = _safe(info, "currentPrice",
                                          _safe(info, "regularMarketPrice"))
        rec["totalDebt"]          = _safe(info, "totalDebt")
        rec["totalCash"]          = _safe(info, "totalCash")
        rec["sharesOutstanding"]  = _safe(info, "sharesOutstanding")
        rec["sector"]             = _safe(info, "sector", "Unknown")
        rec["shortName"]          = _safe(info, "shortName", ticker_str)
        rec["earningsGrowth"]     = _safe(info, "earningsGrowth")
        rec["dividendRate"]       = _safe(info, "dividendRate")

        # ---- financial statements (most recent annual) ----
        try:    fins = t.financials
        except: fins = None
        try:    bs = t.balance_sheet
        except: bs = None
        try:    cf = t.cashflow
        except: cf = None

        rec["totalRevenue"]           = _stmt_val(fins, "Total Revenue")
        rec["totalRevenue_prior"]     = _stmt_val(fins, "Total Revenue", 1)
        rec["grossProfit"]            = _stmt_val(fins, "Gross Profit")
        rec["grossProfit_prior"]      = _stmt_val(fins, "Gross Profit", 1)
        rec["ebit"]                   = _stmt_val(fins, "EBIT")
        if np.isnan(rec["ebit"]):
            rec["ebit"]               = _stmt_val(fins, "Operating Income")
        rec["ebitda"]                 = _stmt_val(fins, "EBITDA")
        rec["netIncome"]              = _stmt_val(fins, "Net Income")
        rec["netIncome_prior"]        = _stmt_val(fins, "Net Income", 1)
        rec["incomeTaxExpense"]       = _stmt_val(fins, "Tax Provision")
        if np.isnan(rec["incomeTaxExpense"]):
            rec["incomeTaxExpense"]    = _stmt_val(fins, "Income Tax")
        rec["pretaxIncome"]           = _stmt_val(fins, "Pretax Income")
        rec["costOfRevenue"]          = _stmt_val(fins, "Cost Of Revenue")

        rec["totalAssets"]            = _stmt_val(bs, "Total Assets")
        rec["totalAssets_prior"]      = _stmt_val(bs, "Total Assets", 1)
        rec["totalEquity"]            = _stmt_val(bs, "Stockholders Equity")
        if np.isnan(rec["totalEquity"]):
            rec["totalEquity"]        = _stmt_val(bs, "Total Stockholder")
        rec["totalDebt_bs"]           = _stmt_val(bs, "Total Debt")
        rec["longTermDebt"]           = _stmt_val(bs, "Long Term Debt")
        rec["longTermDebt_prior"]     = _stmt_val(bs, "Long Term Debt", 1)
        rec["currentLiabilities"]     = _stmt_val(bs, "Current Liabilities")
        rec["currentAssets"]          = _stmt_val(bs, "Current Assets")
        rec["currentAssets_prior"]    = _stmt_val(bs, "Current Assets", 1)
        rec["currentLiabilities_prior"] = _stmt_val(bs, "Current Liabilities", 1)
        rec["cash_bs"]                = _stmt_val(bs, "Cash And Cash Equivalents")
        rec["sharesBS"]               = _stmt_val(bs, "Ordinary Shares Number")
        rec["sharesBS_prior"]         = _stmt_val(bs, "Ordinary Shares Number", 1)
        if np.isnan(rec["sharesBS"]):
            rec["sharesBS"]           = _stmt_val(bs, "Share Issued")
            rec["sharesBS_prior"]     = _stmt_val(bs, "Share Issued", 1)

        rec["operatingCashFlow"]      = _stmt_val(cf, "Operating Cash Flow")
        if np.isnan(rec["operatingCashFlow"]):
            rec["operatingCashFlow"]  = _stmt_val(cf, "Total Cash From Operating")
        rec["capex"]                  = _stmt_val(cf, "Capital Expenditure")
        if np.isnan(rec["capex"]):
            rec["capex"]              = _stmt_val(cf, "Capital Expenditures")
        rec["dividendsPaid"]          = _stmt_val(cf, "Common Stock Dividend")
        if np.isnan(rec["dividendsPaid"]):
            rec["dividendsPaid"]      = _stmt_val(cf, "Dividends Paid")

        # ---- data freshness: record most recent filing date ----
        try:
            for stmt_name, stmt_obj in [("financials", fins), ("balance_sheet", bs), ("cashflow", cf)]:
                if stmt_obj is not None and not stmt_obj.empty:
                    most_recent = stmt_obj.columns[0]
                    rec[f"_stmt_date_{stmt_name}"] = str(most_recent.date()) if hasattr(most_recent, "date") else str(most_recent)
        except Exception:
            pass

        # ---- price history (13 months for 12-1 momentum) ----
        try:
            hist = t.history(period="13mo", auto_adjust=True)
            if hist is not None and len(hist) >= 10:
                closes = hist["Close"].dropna()
                daily_ret = np.log(closes / closes.shift(1)).dropna()
                rec["price_latest"] = float(closes.iloc[-1])

                # Calendar-based lookback: find the closest trading day
                # to each target date instead of using fixed index offsets
                # (iloc[-22] ≈ 1 month but varies with holidays).
                last_date = closes.index[-1]
                for label, delta_days in [("price_1m_ago", 30),
                                          ("price_6m_ago", 182),
                                          ("price_12m_ago", 365)]:
                    target = last_date - pd.Timedelta(days=delta_days)
                    # Find the closest trading day on or before the target
                    mask = closes.index <= target
                    if mask.any():
                        rec[label] = float(closes.loc[mask].iloc[-1])
                    else:
                        rec[label] = np.nan

                rec["volatility_1y"] = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) >= 200 else np.nan
                rec["_daily_returns"] = {
                    dt.strftime("%Y-%m-%d"): v
                    for dt, v in zip(daily_ret.index, daily_ret.values)
                }
        except Exception:
            pass

        # ---- earnings surprises ----
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                surs = []
                for _, row in eh.tail(4).iterrows():
                    a, e = row.get("epsActual", np.nan), row.get("epsEstimate", np.nan)
                    if pd.notna(a) and pd.notna(e) and abs(e) > 0.001:
                        surs.append((a - e) / abs(e))
                rec["analyst_surprise"] = float(np.mean(surs)) if len(surs) >= 2 else np.nan
        except Exception:
            pass

    except Exception as exc:
        err_str = str(exc)
        rec["_error"] = err_str
        rec["_non_retryable"] = any(p in err_str.lower() for p in _NON_RETRYABLE_PATTERNS)
    return rec


def fetch_all_tickers(tickers: list, batch_size: int = 50) -> list:
    """Fetch data for all tickers in batches with delays."""
    results = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        batch = tickers[bi * batch_size : (bi + 1) * batch_size]
        print(f"  Batch {bi+1}/{n_batches}  ({batch[0]}..{batch[-1]})")
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = {pool.submit(fetch_single_ticker, t): t for t in batch}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result(timeout=60))
                except Exception as e:
                    results.append({"Ticker": futs[fut], "_error": str(e)})
        if bi < n_batches - 1:
            time.sleep(2)
    return results


def fetch_market_returns(max_retries: int = 3) -> pd.Series:
    """Fetch S&P 500 daily returns for beta calculation.

    Implements exponential backoff retry (1s / 2s / 4s) per §10.3.
    """
    for attempt in range(max_retries):
        try:
            import yfinance as yf
            hist = yf.Ticker("^GSPC").history(period="1y", auto_adjust=True)
            closes = hist["Close"].dropna()
            return np.log(closes / closes.shift(1)).dropna()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return pd.Series(dtype=float)


# =========================================================================
# D-alt. Offline sample-data generator
# =========================================================================
# When yfinance is unreachable we generate sector-aware random data drawn
# from realistic distributions so every downstream step exercises real code.

_SECTOR_PROFILES = {
    "Information Technology": {"ev_ebitda": (20, 8), "fcf_yield": (0.04, 0.02), "roic": (0.22, 0.10), "gpa": (0.35, 0.12), "de": (0.5, 0.4), "vol": (0.30, 0.08), "beta": (1.15, 0.20), "rev_g": (0.12, 0.10), "mom": (0.15, 0.20)},
    "Health Care":            {"ev_ebitda": (18, 7), "fcf_yield": (0.05, 0.02), "roic": (0.18, 0.09), "gpa": (0.45, 0.15), "de": (0.7, 0.5), "vol": (0.28, 0.07), "beta": (0.90, 0.20), "rev_g": (0.08, 0.08), "mom": (0.08, 0.18)},
    "Financials":             {"ev_ebitda": (12, 4), "fcf_yield": (0.06, 0.03), "roic": (0.10, 0.05), "gpa": (0.20, 0.08), "de": (2.5, 1.5), "vol": (0.25, 0.06), "beta": (1.10, 0.20), "rev_g": (0.06, 0.06), "mom": (0.10, 0.15)},
    "Consumer Discretionary": {"ev_ebitda": (16, 6), "fcf_yield": (0.04, 0.02), "roic": (0.15, 0.08), "gpa": (0.30, 0.10), "de": (1.0, 0.7), "vol": (0.32, 0.08), "beta": (1.20, 0.25), "rev_g": (0.07, 0.08), "mom": (0.12, 0.22)},
    "Communication Services": {"ev_ebitda": (14, 5), "fcf_yield": (0.05, 0.02), "roic": (0.14, 0.07), "gpa": (0.40, 0.12), "de": (0.8, 0.5), "vol": (0.28, 0.07), "beta": (1.05, 0.20), "rev_g": (0.08, 0.07), "mom": (0.10, 0.18)},
    "Industrials":            {"ev_ebitda": (14, 4), "fcf_yield": (0.05, 0.02), "roic": (0.14, 0.06), "gpa": (0.28, 0.08), "de": (1.0, 0.6), "vol": (0.24, 0.06), "beta": (1.05, 0.15), "rev_g": (0.06, 0.05), "mom": (0.09, 0.15)},
    "Consumer Staples":       {"ev_ebitda": (15, 4), "fcf_yield": (0.05, 0.01), "roic": (0.18, 0.07), "gpa": (0.35, 0.10), "de": (1.2, 0.7), "vol": (0.18, 0.04), "beta": (0.70, 0.15), "rev_g": (0.04, 0.03), "mom": (0.05, 0.12)},
    "Energy":                 {"ev_ebitda": (7, 3),  "fcf_yield": (0.08, 0.04), "roic": (0.12, 0.08), "gpa": (0.25, 0.10), "de": (0.6, 0.4), "vol": (0.32, 0.08), "beta": (1.10, 0.25), "rev_g": (0.03, 0.12), "mom": (0.06, 0.20)},
    "Utilities":              {"ev_ebitda": (12, 3), "fcf_yield": (0.04, 0.01), "roic": (0.06, 0.02), "gpa": (0.18, 0.05), "de": (1.5, 0.5), "vol": (0.18, 0.04), "beta": (0.60, 0.15), "rev_g": (0.03, 0.03), "mom": (0.04, 0.10)},
    "Real Estate":            {"ev_ebitda": (18, 6), "fcf_yield": (0.04, 0.02), "roic": (0.05, 0.03), "gpa": (0.20, 0.08), "de": (1.8, 0.8), "vol": (0.22, 0.05), "beta": (0.85, 0.20), "rev_g": (0.05, 0.05), "mom": (0.06, 0.14)},
    "Materials":              {"ev_ebitda": (10, 3), "fcf_yield": (0.06, 0.03), "roic": (0.12, 0.06), "gpa": (0.25, 0.08), "de": (0.7, 0.4), "vol": (0.26, 0.06), "beta": (1.05, 0.20), "rev_g": (0.05, 0.06), "mom": (0.07, 0.16)},
}
_DEFAULT_PROF = {"ev_ebitda": (14, 5), "fcf_yield": (0.05, 0.02), "roic": (0.12, 0.06), "gpa": (0.28, 0.10), "de": (1.0, 0.6), "vol": (0.25, 0.07), "beta": (1.0, 0.20), "rev_g": (0.06, 0.06), "mom": (0.08, 0.16)}


def _generate_sample_data(universe_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate realistic sector-aware sample data for the full universe."""
    rng = np.random.default_rng(seed)
    records = []
    for _, row in universe_df.iterrows():
        sector = row["Sector"]
        p = _SECTOR_PROFILES.get(sector, _DEFAULT_PROF)

        # Helper: draw from truncated normal (positive where needed)
        def tn(mu, sigma, low=None, high=None):
            v = rng.normal(mu, sigma)
            if low is not None:
                v = max(v, low)
            if high is not None:
                v = min(v, high)
            return v

        ev_ebitda       = tn(*p["ev_ebitda"], low=2)
        fcf_yield       = tn(*p["fcf_yield"])
        price           = tn(150, 80, low=10)
        eps             = tn(price * 0.04, price * 0.02)
        earnings_yield  = eps / price if price > 0 else np.nan
        rev             = tn(30e9, 25e9, low=1e9)
        mc              = tn(rev * 3, rev * 1.5, low=2e9)
        ev              = mc * tn(1.1, 0.15, low=0.5)
        ev_sales        = ev / rev if rev > 0 else np.nan
        roic            = tn(*p["roic"])
        gpa             = tn(*p["gpa"], low=0)
        de              = tn(*p["de"], low=0)
        ni              = rev * tn(0.10, 0.06)
        ocf             = ni * tn(1.3, 0.3, low=0.2)
        ta              = rev * tn(1.8, 0.5, low=0.5)
        f_score         = int(tn(6, 1.5, low=0, high=9))
        accruals        = (ni - ocf) / ta if ta > 0 else np.nan
        fwd_eps         = eps * (1 + tn(0.08, 0.10))
        fwd_eps_growth  = (fwd_eps - eps) / abs(eps) if abs(eps) > 0.01 else np.nan
        pe_sample       = price / eps if (eps > 0.01 and price > 0) else np.nan
        earnings_growth = tn(0.15, 0.10, low=-0.3)
        peg_ratio       = (pe_sample / (earnings_growth * 100)) if (pd.notna(pe_sample) and earnings_growth > 0.01) else np.nan
        rev_growth      = tn(*p["rev_g"])
        roe             = ni / (ta * 0.4) if ta > 0 else 0
        retention       = max(0, min(1, tn(0.65, 0.2)))
        sust_growth     = roe * retention
        mom_12_1        = tn(*p["mom"])
        mom_6m          = tn(p["mom"][0] * 0.6, p["mom"][1] * 0.8)
        vol             = tn(*p["vol"], low=0.08)
        beta            = tn(*p["beta"], low=-0.5)
        # Analyst surprise: sparse — ~40% of tickers have it
        analyst_surprise = tn(0.05, 0.08) if rng.random() < 0.40 else np.nan

        records.append({
            "Ticker": row["Ticker"],
            "Company": row["Company"],
            "Sector": sector,
            "ev_ebitda": round(ev_ebitda, 2),
            "fcf_yield": round(fcf_yield, 4),
            "earnings_yield": round(earnings_yield, 4),
            "ev_sales": round(ev_sales, 2) if pd.notna(ev_sales) else np.nan,
            "roic": round(roic, 4),
            "gross_profit_assets": round(gpa, 4),
            "debt_equity": round(de, 2),
            "piotroski_f_score": f_score,
            "accruals": round(accruals, 4) if pd.notna(accruals) else np.nan,
            "forward_eps_growth": round(fwd_eps_growth, 4) if pd.notna(fwd_eps_growth) else np.nan,
            "peg_ratio": round(peg_ratio, 2) if pd.notna(peg_ratio) else np.nan,
            "revenue_growth": round(rev_growth, 4),
            "sustainable_growth": round(sust_growth, 4),
            "return_12_1": round(mom_12_1, 4),
            "return_6m": round(mom_6m, 4),
            "volatility": round(vol, 4),
            "beta": round(beta, 2),
            "analyst_surprise": round(analyst_surprise, 4) if pd.notna(analyst_surprise) else np.nan,
            "eps_revision_ratio": np.nan,
            "eps_estimate_change": np.nan,
        })
    return pd.DataFrame(records)


# =========================================================================
# E. Compute all 17 individual metrics (from live yfinance data)
# =========================================================================
def compute_metrics(raw_data: list, market_returns: pd.Series) -> pd.DataFrame:
    """Compute all 17 factor metrics from raw yfinance ticker data."""
    records = []

    for d in raw_data:
        rec = {
            "Ticker": d.get("Ticker"),
            "Company": d.get("shortName", d.get("Ticker")),
            "Sector": d.get("sector", "Unknown"),
        }

        if "_error" in d and not d.get("marketCap"):
            rec["_skipped"] = True
            records.append(rec)
            continue

        # -- Common intermediates (used across multiple metrics) --
        # Use info['totalCash'] consistently (includes short-term
        # investments, matching yfinance's own EV definition).
        mc = d.get("marketCap", np.nan)
        ev = d.get("enterpriseValue", np.nan)
        # Canonical cash/debt figures reused across EV and ROIC
        _cash = d.get("totalCash", d.get("cash_bs", np.nan))
        _debt = d.get("totalDebt", d.get("totalDebt_bs", np.nan))
        if pd.isna(ev) or ev == 0:
            # Only compute fallback EV when all components are available
            if pd.notna(mc) and pd.notna(_debt) and pd.notna(_cash):
                ev = mc + _debt - _cash
            else:
                ev = np.nan

        ta = d.get("totalAssets", np.nan)
        ni = d.get("netIncome", np.nan)
        eq_v = d.get("totalEquity", np.nan)
        ocf = d.get("operatingCashFlow", np.nan)
        rev_c = d.get("totalRevenue", np.nan)
        rev_p = d.get("totalRevenue_prior", np.nan)
        ticker = rec["Ticker"]

        # -- Valuation metrics (1-4) --
        try:
            # 1. EV/EBITDA — use EBITDA only; do not fall back to EBIT
            # because D&A can differ materially (e.g. XOM EBIT/EBITDA = 0.68)
            ebitda = d.get("ebitda", np.nan)
            rec["ev_ebitda"] = (ev / ebitda) if (pd.notna(ev) and pd.notna(ebitda) and ebitda > 0 and ev > 0) else np.nan

            # 2. FCF Yield
            capex = d.get("capex", np.nan)
            fcf = np.nan
            if pd.notna(ocf):
                fcf = (ocf - abs(capex)) if (pd.notna(capex) and capex < 0) else (ocf - capex if pd.notna(capex) else ocf)
            rec["fcf_yield"] = (fcf / ev) if (pd.notna(fcf) and pd.notna(ev) and ev > 0) else np.nan

            # 3. Earnings Yield
            eps = d.get("trailingEps", np.nan)
            price = d.get("currentPrice", d.get("price_latest", np.nan))
            rec["earnings_yield"] = (eps / price) if (pd.notna(eps) and pd.notna(price) and price > 0) else np.nan

            # 4. EV/Sales
            rec["ev_sales"] = (ev / rev_c) if (pd.notna(ev) and pd.notna(rev_c) and rev_c > 0 and ev > 0) else np.nan
        except Exception as e:
            warnings.warn(f"{ticker}: valuation metrics failed: {e}")

        # -- Quality metrics (5-9) --
        try:
            # 5. ROIC (Invested Capital = Equity + Total Debt - Cash)
            # Reuse _cash / _debt from EV calculation for consistency.
            ebit_v = d.get("ebit", np.nan)
            if pd.notna(ebit_v):
                tax_exp = d.get("incomeTaxExpense", np.nan)
                pretax = d.get("pretaxIncome", np.nan)
                tax_rate = 0.21
                if pd.notna(tax_exp) and pd.notna(pretax) and pretax > 0:
                    tax_rate = max(0, min(tax_exp / pretax, 0.5))
                nopat = ebit_v * (1 - tax_rate)
                if pd.notna(eq_v) and pd.notna(_debt) and pd.notna(_cash):
                    ic = eq_v + _debt - _cash
                    rec["roic"] = (nopat / ic) if ic > 0 else np.nan
                else:
                    rec["roic"] = np.nan
            else:
                rec["roic"] = np.nan

            # 6. Gross Profit / Assets
            gp = d.get("grossProfit", np.nan)
            rec["gross_profit_assets"] = (gp / ta) if (pd.notna(gp) and pd.notna(ta) and ta > 0) else np.nan

            # 7. Debt/Equity (reuse _debt for consistency)
            if pd.notna(_debt) and pd.notna(eq_v):
                rec["debt_equity"] = 999.0 if eq_v <= 0 else _debt / eq_v
            else:
                rec["debt_equity"] = np.nan

            # 8. Piotroski F-Score
            ni_p = d.get("netIncome_prior", np.nan)
            ta_p = d.get("totalAssets_prior", np.nan)
            ocfv = ocf
            ltd  = d.get("longTermDebt", np.nan)
            ltd_p = d.get("longTermDebt_prior", np.nan)
            ca_c = d.get("currentAssets", np.nan)
            cl_c = d.get("currentLiabilities", np.nan)
            ca_p = d.get("currentAssets_prior", np.nan)
            cl_p = d.get("currentLiabilities_prior", np.nan)
            sh   = d.get("sharesBS", np.nan)
            sh_p = d.get("sharesBS_prior", np.nan)
            gp_v = d.get("grossProfit", np.nan)
            gp_p = d.get("grossProfit_prior", np.nan)

            f = 0
            n_testable = 0
            if pd.notna(ni):
                n_testable += 1; f += int(ni > 0)
            if pd.notna(ocfv):
                n_testable += 1; f += int(ocfv > 0)
            if all(pd.notna(x) for x in [ni, ni_p, ta, ta_p]) and ta > 0 and ta_p > 0:
                n_testable += 1; f += int((ni/ta) > (ni_p/ta_p))
            if pd.notna(ocfv) and pd.notna(ni):
                n_testable += 1; f += int(ocfv > ni)
            if all(pd.notna(x) for x in [ltd, ltd_p, ta, ta_p]) and ta > 0 and ta_p > 0:
                n_testable += 1; f += int((ltd/ta) < (ltd_p/ta_p))
            if all(pd.notna(x) for x in [ca_c, cl_c, ca_p, cl_p]) and cl_c > 0 and cl_p > 0:
                n_testable += 1; f += int((ca_c/cl_c) > (ca_p/cl_p))
            if pd.notna(sh) and pd.notna(sh_p):
                n_testable += 1; f += int(sh <= sh_p)
            if all(pd.notna(x) for x in [gp_v, gp_p, rev_c, rev_p]) and rev_c > 0 and rev_p > 0:
                n_testable += 1; f += int((gp_v/rev_c) > (gp_p/rev_p))
            if all(pd.notna(x) for x in [rev_c, rev_p, ta, ta_p]) and ta > 0 and ta_p > 0:
                n_testable += 1; f += int((rev_c/ta) > (rev_p/ta_p))
            # Use raw integer score (0-9).  Do NOT proportionally normalize —
            # a company that passes 7 of 7 testable signals is NOT the same
            # quality as one passing 9 of 9; it simply has less data.
            rec["piotroski_f_score"] = f if n_testable >= 4 else np.nan

            # 9. Accruals
            rec["accruals"] = ((ni - ocfv) / ta) if (pd.notna(ni) and pd.notna(ocfv) and pd.notna(ta) and ta > 0) else np.nan
        except Exception as e:
            warnings.warn(f"{ticker}: quality metrics failed: {e}")

        # -- Growth metrics (10-12) --
        try:
            # 10. Forward EPS Growth
            fwd = d.get("forwardEps", np.nan)
            trail = d.get("trailingEps", np.nan)
            rec["forward_eps_growth"] = ((fwd - trail) / abs(trail)) if (pd.notna(fwd) and pd.notna(trail) and abs(trail) > 0.01) else np.nan

            # PEG Ratio = (P/E) / (Earnings Growth Rate %)
            _price = d.get("currentPrice", d.get("price_latest", np.nan))
            _pe = (_price / trail) if (pd.notna(_price) and pd.notna(trail) and trail > 0.01) else np.nan
            _eg = d.get("earningsGrowth", np.nan)
            rec["peg_ratio"] = (_pe / (_eg * 100)) if (pd.notna(_pe) and pd.notna(_eg) and _eg > 0.01) else np.nan

            # 11. Revenue Growth
            rec["revenue_growth"] = ((rev_c - rev_p) / rev_p) if (pd.notna(rev_c) and pd.notna(rev_p) and rev_p > 0) else np.nan

            # 12. Sustainable Growth
            if pd.notna(ni) and pd.notna(eq_v) and eq_v > 0 and ni > 0:
                roe = ni / eq_v
                _divs_raw = d.get("dividendsPaid", np.nan)
                if pd.isna(_divs_raw):
                    # Cashflow missing — estimate from info dividendRate
                    _div_rate = d.get("dividendRate", np.nan)
                    _shares = d.get("sharesOutstanding", np.nan)
                    if pd.notna(_div_rate) and pd.notna(_shares):
                        divs = abs(_div_rate * _shares)
                    else:
                        divs = 0  # truly unknown; assume full retention
                else:
                    divs = abs(_divs_raw)
                ret = max(0, 1 - divs / ni) if ni > 0 else 0
                rec["sustainable_growth"] = roe * ret
            else:
                rec["sustainable_growth"] = np.nan
        except Exception as e:
            warnings.warn(f"{ticker}: growth metrics failed: {e}")

        # -- Momentum metrics (13-14) --
        try:
            # 13. 12-1 Month Return
            p12 = d.get("price_12m_ago", np.nan)
            p1m = d.get("price_1m_ago", np.nan)
            rec["return_12_1"] = ((p1m - p12) / p12) if (pd.notna(p12) and pd.notna(p1m) and p12 > 0) else np.nan

            # 14. 6-1 Month Return (exclude most recent month to match 12-1M convention)
            p6m = d.get("price_6m_ago", np.nan)
            rec["return_6m"] = ((p1m - p6m) / p6m) if (pd.notna(p6m) and pd.notna(p1m) and p6m > 0) else np.nan
        except Exception as e:
            warnings.warn(f"{ticker}: momentum metrics failed: {e}")

        # -- Risk metrics (15-16) --
        try:
            # 15. Volatility
            rec["volatility"] = d.get("volatility_1y", np.nan)

            # 16. Beta (date-aligned)
            dr = d.get("_daily_returns")
            if dr and isinstance(dr, dict) and len(market_returns) >= 200:
                mr_dates = {dt.strftime("%Y-%m-%d"): v
                            for dt, v in zip(market_returns.index, market_returns.values)}
                common = sorted(set(dr.keys()) & set(mr_dates.keys()))
                if len(common) >= 200:
                    sr = np.array([dr[dt] for dt in common])
                    mr = np.array([mr_dates[dt] for dt in common])
                    cov = np.cov(sr, mr)[0, 1]
                    var = np.var(mr, ddof=1)
                    rec["beta"] = cov / var if var > 0 else np.nan
                else:
                    rec["beta"] = np.nan
            elif dr and isinstance(dr, list):
                # Backward compat: legacy list format (no date info)
                if len(dr) >= 200 and len(market_returns) >= 200:
                    sr = np.array(dr[-len(market_returns):])
                    mr = market_returns.values[-len(sr):]
                    ml = min(len(sr), len(mr))
                    sr, mr = sr[-ml:], mr[-ml:]
                    if ml >= 200:
                        cov = np.cov(sr, mr)[0, 1]
                        var = np.var(mr, ddof=1)
                        rec["beta"] = cov / var if var > 0 else np.nan
                    else:
                        rec["beta"] = np.nan
                else:
                    rec["beta"] = np.nan
            else:
                rec["beta"] = np.nan
        except Exception as e:
            warnings.warn(f"{ticker}: risk metrics failed: {e}")

        # -- Revisions (17) --
        try:
            rec["analyst_surprise"] = d.get("analyst_surprise", np.nan)
            rec["eps_revision_ratio"] = np.nan
            rec["eps_estimate_change"] = np.nan
        except Exception as e:
            warnings.warn(f"{ticker}: revisions metrics failed: {e}")

        # -- Data freshness check --
        try:
            stale_days = 400  # Flag if most recent filing is > 400 days old
            stmt_date_str = d.get("_stmt_date_financials")
            if stmt_date_str:
                stmt_date = pd.Timestamp(stmt_date_str)
                age_days = (pd.Timestamp.now() - stmt_date).days
                rec["_stmt_age_days"] = age_days
                if age_days > stale_days:
                    rec["_stale_data"] = True
                    warnings.warn(f"{ticker}: financial data is {age_days} days old (>{stale_days}d)")
        except Exception:
            pass

        records.append(rec)

    return pd.DataFrame(records)


# =========================================================================
# F. Winsorize raw metrics (SS4.7)
# =========================================================================
METRIC_COLS = [
    "ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales",
    "roic", "gross_profit_assets", "debt_equity", "piotroski_f_score", "accruals",
    "forward_eps_growth", "peg_ratio", "revenue_growth", "sustainable_growth",
    "return_12_1", "return_6m",
    "volatility", "beta",
    "analyst_surprise", "eps_revision_ratio", "eps_estimate_change",
]


def winsorize_metrics(df: pd.DataFrame, lo: float = 0.01, hi: float = 0.01):
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) >= 10:
            w = mstats.winsorize(vals, limits=(lo, hi))
            df.loc[vals.index, col] = w
    return df


# =========================================================================
# G. Sector-relative percentile ranks (SS3.3)
# =========================================================================
METRIC_DIR = {
    "ev_ebitda": False, "fcf_yield": True, "earnings_yield": True, "ev_sales": False,
    "roic": True, "gross_profit_assets": True, "debt_equity": False,
    "piotroski_f_score": True, "accruals": False,
    "forward_eps_growth": True, "peg_ratio": False, "revenue_growth": True, "sustainable_growth": True,
    "return_12_1": True, "return_6m": True,
    "volatility": False, "beta": False,
    "analyst_surprise": True, "eps_revision_ratio": True, "eps_estimate_change": True,
}


def compute_sector_percentiles(df: pd.DataFrame):
    pct = {c: f"{c}_pct" for c in METRIC_COLS}
    for c in pct.values():
        df[c] = np.nan

    for _, grp in df.groupby("Sector"):
        for col in METRIC_COLS:
            pc = pct[col]
            if col not in df.columns:
                df.loc[grp.index, pc] = 50.0
                continue
            valid = grp[col].dropna()
            if len(valid) < 3:
                df.loc[grp.index, pc] = 50.0
                continue
            ranks = grp[col].rank(pct=True, na_option="keep") * 100
            if not METRIC_DIR.get(col, True):
                ranks = 100 - ranks
            df.loc[grp.index, pc] = ranks.fillna(50.0)
    return df


# =========================================================================
# H. Within-category scores (SS3.1)
# =========================================================================
CAT_METRICS = {
    "valuation": ["ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales"],
    "quality":   ["roic", "gross_profit_assets", "debt_equity", "piotroski_f_score", "accruals"],
    "growth":    ["forward_eps_growth", "peg_ratio", "revenue_growth", "sustainable_growth"],
    "momentum":  ["return_12_1", "return_6m"],
    "risk":      ["volatility", "beta"],
    "revisions": ["eps_revision_ratio", "eps_estimate_change", "analyst_surprise"],
}


def compute_category_scores(df: pd.DataFrame, cfg: dict):
    mw = cfg["metric_weights"]
    for cat, metrics in CAT_METRICS.items():
        ws = mw.get(cat, {})
        col = f"{cat}_score"
        df[col] = 0.0
        tw = 0
        for m in metrics:
            pc = f"{m}_pct"
            w = ws.get(m, 0) / 100.0
            if pc not in df.columns:
                continue
            # Skip metrics that are entirely NaN (placeholder / unavailable)
            # so their weight is redistributed to metrics with real data.
            if df[pc].isna().all() or (df[pc] == 50.0).all():
                # Check if the underlying raw metric is entirely NaN
                if m in df.columns and df[m].isna().all():
                    continue  # truly unavailable — skip entirely
            df[col] += df[pc].fillna(50) * w
            tw += w
        if tw > 0 and abs(tw - 1.0) > 0.01:
            df[col] /= tw
    return df


# =========================================================================
# I. Composite score (SS3.2)
# =========================================================================
def compute_composite(df: pd.DataFrame, cfg: dict):
    if df.empty:
        df["Composite"] = pd.Series(dtype=float)
        return df
    fw = cfg["factor_weights"]
    tw = sum(fw.values())
    if tw == 0:
        tw = 1  # Prevent division by zero if all weights are 0
    col_map = {
        "valuation": "valuation_score", "quality": "quality_score",
        "growth": "growth_score", "momentum": "momentum_score",
        "risk": "risk_score", "revisions": "revisions_score",
    }
    df["Composite"] = sum(df[col_map[c]] * (fw[c] / tw) for c in col_map)

    sector_relative = cfg.get("sector_neutral", {}).get("sector_relative_composite", False)
    if sector_relative and "Sector" in df.columns:
        for _, grp in df.groupby("Sector"):
            lo, hi = grp["Composite"].min(), grp["Composite"].max()
            if hi > lo:
                df.loc[grp.index, "Composite"] = ((grp["Composite"] - lo) / (hi - lo)) * 100
            else:
                df.loc[grp.index, "Composite"] = 50.0
    else:
        lo, hi = df["Composite"].min(), df["Composite"].max()
        if hi > lo:
            df["Composite"] = ((df["Composite"] - lo) / (hi - lo)) * 100
        else:
            df["Composite"] = 50.0

    df["Composite"] = df["Composite"].round(2)
    return df


# =========================================================================
# J. Value trap flags (SS2.3)
# =========================================================================
def apply_value_trap_flags(df: pd.DataFrame, cfg: dict):
    vtf = cfg.get("value_trap_filters", {})
    if not vtf.get("enabled", True):
        df["Value_Trap_Flag"] = False
        return df

    qual_floor = vtf.get("quality_floor_percentile", 30) / 100.0
    mom_floor = vtf.get("momentum_floor_percentile", 30) / 100.0
    rev_floor = vtf.get("revisions_floor_percentile", 30) / 100.0

    # Layer 1 - Quality: below quality floor percentile
    # NaN values should NOT trigger flags (missing data != poor quality)
    qual_col = "quality_score" if "quality_score" in df.columns else None
    if qual_col:
        qual_thr = df[qual_col].quantile(qual_floor)
        l1 = df[qual_col].le(qual_thr).fillna(False)
    else:
        l1 = pd.Series(False, index=df.index)

    # Layer 2 - Momentum: below momentum floor percentile
    mom_col = "momentum_score" if "momentum_score" in df.columns else None
    if mom_col:
        mom_thr = df[mom_col].quantile(mom_floor)
        l2 = df[mom_col].le(mom_thr).fillna(False)
    else:
        l2 = pd.Series(False, index=df.index)

    # Layer 3 - Revisions: below revisions floor percentile
    rev_col = "revisions_score" if "revisions_score" in df.columns else None
    if rev_col:
        rev_thr = df[rev_col].quantile(rev_floor)
        l3 = df[rev_col].le(rev_thr).fillna(False)
    else:
        l3 = pd.Series(False, index=df.index)

    df["Value_Trap_Flag"] = l1 | l2 | l3
    return df


# =========================================================================
# K. Rank stocks
# =========================================================================
def rank_stocks(df: pd.DataFrame):
    df["Rank"] = df["Composite"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("Rank").reset_index(drop=True)


# =========================================================================
# L. Write to Excel (openpyxl only, data-only FactorScores sheet)
# =========================================================================
def write_excel(df: pd.DataFrame, cfg: dict) -> str:
    out_path = ROOT / cfg["output"]["excel_file"]
    sheet = cfg["output"]["factor_scores_sheet"]

    col_map = [
        ("Ticker", "Ticker"), ("Company", "Company"), ("Sector", "Sector"),
        ("valuation_score", "Val_Pct"), ("quality_score", "Qual_Pct"),
        ("growth_score", "Grow_Pct"), ("momentum_score", "Mom_Pct"),
        ("risk_score", "Risk_Pct"), ("revisions_score", "Rev_Pct"),
        ("Composite", "Composite"), ("Rank", "Rank"),
        ("Value_Trap_Flag", "Value_Trap_Flag"),
    ]

    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append([h for _, h in col_map])

    for _, row in df.iterrows():
        vals = []
        for src, _ in col_map:
            v = row.get(src)
            if isinstance(v, float) and np.isnan(v):
                vals.append(None)
            elif src in ("valuation_score", "quality_score", "growth_score",
                         "momentum_score", "risk_score", "revisions_score"):
                vals.append(round(v, 1) if pd.notna(v) else None)
            else:
                vals.append(v)
        ws.append(vals)

    wb.save(str(out_path))
    return str(out_path)


# =========================================================================
# M. Write scored DataFrame to cache Parquet
# =========================================================================
def write_scores_parquet(df: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y%m%d")
    path = CACHE_DIR / f"factor_scores_{today}.parquet"
    keep = [c for c in df.columns if not c.startswith("_")]
    df[keep].to_parquet(str(path), index=False)
    return str(path)


# =========================================================================
# Diagnostics
# =========================================================================
def print_summary(df, universe_size, skipped, cfg, cache_files, excel_path, t0):
    scored = len(df)
    elapsed = round(time.time() - t0, 1)

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

    rev_cols = ["analyst_surprise", "eps_revision_ratio", "eps_estimate_change"]
    rev_avail = sum(df[c].notna().sum() for c in rev_cols if c in df.columns)
    rev_total = len(df) * len(rev_cols)
    rev_pct = rev_avail / rev_total * 100 if rev_total else 0
    rev_w = cfg["factor_weights"].get("revisions", 0)
    vt = int(df["Value_Trap_Flag"].sum()) if "Value_Trap_Flag" in df.columns else 0

    print()
    print("============================================")
    print("  FACTOR ENGINE — RUN SUMMARY")
    print("============================================")
    print(f"Universe requested:       {universe_size} tickers")
    print(f"Successfully scored:      {scored} tickers")
    print(f"Skipped (data errors):    {len(skipped)} tickers")
    print(f"Skipped ticker list:      {skipped[:20]}{'...' if len(skipped)>20 else ''}")
    print("--------------------------------------------")
    print("Missing data % by metric:")
    for lbl, col in labels:
        pct = df[col].isna().sum() / len(df) * 100 if col in df.columns else 100
        print(f"  {lbl:24s} {pct:.1f}%")
    print("--------------------------------------------")
    print(f"Revisions coverage:       {rev_pct:.1f}%")
    print(f"Revisions weight used:    {rev_w}% {'(auto-disabled)' if rev_w == 0 else ''}")
    print("--------------------------------------------")
    print(f"Value trap flags:         {vt} stocks flagged")
    print("--------------------------------------------")
    print("Top 10 by Composite:")
    for _, r in df.nsmallest(10, "Rank").iterrows():
        print(f"  {int(r['Rank']):3d}. {r['Ticker']:6s} {str(r['Sector']):26s} {r['Composite']:.1f}")
    print("Bottom 5 by Composite:")
    for _, r in df.nlargest(5, "Rank").iterrows():
        print(f"  {int(r['Rank']):3d}. {r['Ticker']:6s} {str(r['Sector']):26s} {r['Composite']:.1f}")
    print("--------------------------------------------")
    print(f"Cache files written:      {cache_files}")
    print(f"Excel written:            {excel_path}")
    print(f"Total runtime:            {elapsed}s")
    print("============================================")


# =========================================================================
# MAIN
# =========================================================================
def main():
    t0 = time.time()

    # A. Config
    print("Loading configuration...")
    cfg = load_config()

    # B. Universe
    print("Fetching S&P 500 constituent list...")
    universe_df = get_sp500_tickers(cfg)
    tickers = universe_df["Ticker"].tolist()
    universe_size = len(tickers)
    print(f"  Universe: {universe_size} tickers after exclusions")
    ticker_meta = universe_df.set_index("Ticker")[["Company", "Sector"]].to_dict("index")

    # C/D. Attempt live data; fall back to sample data if network blocked
    USE_SAMPLE = False
    print("\nTesting network connectivity...")
    try:
        test = fetch_single_ticker(tickers[0])
        if "_error" in test:
            raise RuntimeError(test["_error"])
        print("  Network OK — will fetch live data from yfinance")
    except Exception as e:
        print(f"  Network unavailable ({type(e).__name__})")
        print("  Generating sector-realistic sample data for pipeline validation")
        USE_SAMPLE = True

    skipped_tickers = []

    if USE_SAMPLE:
        # Generate sample data — all 17 metrics pre-computed
        df = _generate_sample_data(universe_df)
    else:
        # Live path
        print(f"\nFetching S&P 500 market returns...")
        market_returns = fetch_market_returns()
        print(f"  {len(market_returns)} daily observations")

        print(f"\nFetching data for {universe_size} tickers...")
        raw = fetch_all_tickers(tickers)

        print("Computing metrics...")
        df = compute_metrics(raw, market_returns)

        # Always use Wikipedia GICS sector names (yfinance uses different
        # names: "Technology" vs "Information Technology", "Consumer Cyclical"
        # vs "Consumer Discretionary", etc.).  Sector-relative percentile
        # ranking depends on consistent sector groups across all tickers.
        for idx, row in df.iterrows():
            t = row["Ticker"]
            if t in ticker_meta:
                df.at[idx, "Sector"] = ticker_meta[t]["Sector"]
                if pd.isna(row.get("Company")) or row.get("Company") == t:
                    df.at[idx, "Company"] = ticker_meta[t]["Company"]

        # Remove fully-failed rows
        skip_mask = df.get("_skipped", pd.Series(False, index=df.index)).fillna(False)
        skipped_tickers = df.loc[skip_mask, "Ticker"].tolist()
        df = df[~skip_mask].copy()

        # Coverage filter
        present = [c for c in METRIC_COLS if c in df.columns]
        df["_mc"] = df[present].notna().sum(axis=1)
        min_m = int(len(present) * cfg["data_quality"]["min_data_coverage_pct"] / 100)
        low = df["_mc"] < min_m
        skipped_tickers += df.loc[low, "Ticker"].tolist()
        df = df[~low].copy()

    # --- Check revisions coverage & auto-disable if <30% ---
    rev_m = ["analyst_surprise", "eps_revision_ratio", "eps_estimate_change"]
    rev_avail = sum(df[c].notna().sum() for c in rev_m if c in df.columns)
    rev_total = len(df) * len(rev_m)
    rev_pct = rev_avail / rev_total * 100 if rev_total else 0

    if rev_pct < 30:
        print(f"\n!! Revisions coverage {rev_pct:.1f}% < 30% threshold")
        print("   Auto-setting revisions weight to 0; redistributing proportionally.")
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

    # F. Winsorize
    print("Winsorizing at 1st / 99th percentiles...")
    df = winsorize_metrics(df, 0.01, 0.01)

    # G. Sector percentiles
    print("Computing sector-relative percentile ranks...")
    df = compute_sector_percentiles(df)

    # H. Category scores
    print("Computing within-category scores...")
    df = compute_category_scores(df, cfg)

    # I. Composite
    print("Computing composite scores...")
    df = compute_composite(df, cfg)

    # J. Value trap flags
    print("Applying value trap flags...")
    df = apply_value_trap_flags(df, cfg)

    # K. Rank
    print("Ranking stocks...")
    df = rank_stocks(df)

    # L. Excel
    print("Writing Excel...")
    excel_path = write_excel(df, cfg)

    # M. Cache Parquet
    print("Writing cache Parquet...")
    pq_path = write_scores_parquet(df)

    cache_files = [p.name for p in CACHE_DIR.glob("*.parquet")]
    print_summary(df, universe_size, skipped_tickers, cfg,
                  cache_files, cfg["output"]["excel_file"], t0)


if __name__ == "__main__":
    main()
