#!/usr/bin/env python3
"""
Multi-Factor Stock Screener - Phase 1: Factor Engine
=====================================================
Computes composite factor scores for the S&P 500 universe using eight
factor categories (Valuation, Quality, Growth, Momentum, Risk, Analyst
Revisions, Size, Investment) across ~33 metrics and writes results to
Excel + Parquet cache.

Reference: Multi-Factor-Screener-Blueprint.md (Version 2.0)

Network behaviour
-----------------
* Primary path: load S&P 500 from GitHub CSV (fallback: Wikipedia), fetch data via yfinance.
* Fallback path: if network is unavailable (sandbox / CI), load tickers
  from sp500_tickers.json and generate sector-realistic sample data so
  the full scoring pipeline can be validated end-to-end.
"""

import copy
import json
import logging
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

    Source priority:
      1. GitHub CSV (datasets/s-and-p-500-companies) — fast, reliable
      2. Wikipedia HTML scrape — secondary fallback
      3. Local sp500_tickers.json — offline last resort

    When a network source succeeds, sp500_tickers.json is auto-updated
    so the local fallback stays current.
    """
    import requests
    from io import StringIO

    df = None
    fallback = ROOT / "sp500_tickers.json"

    # --- Primary: GitHub-hosted CSV ---
    _GITHUB_URL = (
        "https://raw.githubusercontent.com/datasets/"
        "s-and-p-500-companies/main/data/constituents.csv"
    )
    try:
        resp = requests.get(_GITHUB_URL, timeout=10)
        resp.raise_for_status()
        gh = pd.read_csv(StringIO(resp.text))
        df = gh[["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["Ticker", "Company", "Sector"]
        df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
        print(f"  Loaded S&P 500 list from GitHub ({len(df)} tickers)")
    except Exception as e:
        print(f"  GitHub CSV failed ({type(e).__name__}), trying Wikipedia...")

    # --- Secondary: Wikipedia scrape ---
    if df is None:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
            df.columns = ["Ticker", "Company", "Sector"]
            df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
            print(f"  Loaded S&P 500 list from Wikipedia ({len(df)} tickers)")
        except Exception as e:
            print(f"  Wikipedia scrape failed ({type(e).__name__})")

    # --- Cross-validate network source against local fallback ---
    if df is not None and not df.empty and fallback.exists():
        with open(fallback) as f:
            local_data = json.load(f)
        local_tickers = {r["Ticker"] for r in local_data}
        net_tickers = set(df["Ticker"])
        added = net_tickers - local_tickers
        removed = local_tickers - net_tickers
        if added or removed:
            drift_pct = (len(added) + len(removed)) / max(len(local_tickers), 1) * 100
            print(f"  Universe drift: +{len(added)} added, -{len(removed)} removed "
                  f"({drift_pct:.1f}% change vs local fallback)")
            if added:
                print(f"    Added:   {sorted(added)[:10]}{'...' if len(added) > 10 else ''}")
            if removed:
                print(f"    Removed: {sorted(removed)[:10]}{'...' if len(removed) > 10 else ''}")
            if drift_pct > 10:
                warnings.warn(
                    f"Universe drift {drift_pct:.1f}% exceeds 10% threshold."
                )
        # Auto-update local fallback so it stays current
        fresh = df[["Ticker", "Company", "Sector"]].to_dict(orient="records")
        with open(fallback, "w") as f:
            json.dump(fresh, f, indent=2)
        print(f"  Updated sp500_tickers.json ({len(fresh)} tickers)")

    # --- Last resort: local JSON ---
    if df is None or df.empty:
        if fallback.exists():
            with open(fallback) as f:
                local_data = json.load(f)
            df = pd.DataFrame(local_data)
            print(f"  Loaded {len(df)} tickers from sp500_tickers.json (offline)")
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


def apply_universe_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply min_market_cap and min_avg_volume filters from config.

    These filters are applied post-fetch since the universe source
    (Wikipedia S&P 500 list) doesn't include market cap / volume data.
    """
    ucfg = cfg.get("universe", {})
    min_mc = float(ucfg.get("min_market_cap", 0))
    # min_avg_volume is not currently fetched from yfinance, so we
    # only enforce market cap for now (volume would require an
    # additional API call that isn't in the fetch pipeline).

    if min_mc > 0 and "marketCap" in df.columns:
        mc = pd.to_numeric(df["marketCap"], errors="coerce")
        low_mc = mc.notna() & (mc < min_mc)
        n_dropped = low_mc.sum()
        if n_dropped > 0:
            print(f"  Filtered {n_dropped} tickers below ${min_mc/1e9:.1f}B market cap")
            df = df[~low_mc].copy()

    return df


# =========================================================================
# C. Tiered Parquet caching (Blueprint SS7.4)
# =========================================================================
def _find_latest_cache(tier_name: str, config_hash: str | None = None):
    """Find most recent cache file for a tier. Returns (path, date) or (None, None).

    If config_hash is provided, only matches files whose name contains
    that hash (e.g. ``factor_scores_a1b2c3_20260217.parquet``).
    """
    pattern = f"{tier_name}_*.parquet"
    files = sorted(CACHE_DIR.glob(pattern), reverse=True)
    for f in files:
        try:
            parts = f.stem.split("_")
            date_str = parts[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            # If a config_hash was provided, require it to appear in the filename
            if config_hash and config_hash not in f.stem:
                continue
            return f, file_date
        except ValueError:
            continue
    return None, None


def cache_is_fresh(tier_name: str, max_age_days: int, config_hash: str | None = None) -> bool:
    path, dt = _find_latest_cache(tier_name, config_hash=config_hash)
    if path is None:
        return False
    return (datetime.now() - dt) < timedelta(days=max_age_days)


def load_cache(tier_name: str, config_hash: str | None = None) -> pd.DataFrame:
    path, _ = _find_latest_cache(tier_name, config_hash=config_hash)
    print(f"[CACHE HIT] Loading {tier_name} from cache")
    return pd.read_parquet(path)


def save_cache(tier_name: str, df: pd.DataFrame, config_hash: str | None = None) -> str:
    today = datetime.now().strftime("%Y%m%d")
    if config_hash:
        path = CACHE_DIR / f"{tier_name}_{config_hash}_{today}.parquet"
    else:
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
    except Exception as e:
        warnings.warn(f"_safe failed for key='{key}': {e}")
        return default


_STMT_VAL_STRICT = False  # Set True to track all statement lookup misses
_STMT_VAL_MISSES: list = []  # Collected when _STMT_VAL_STRICT is True


def set_stmt_val_strict(enabled: bool = True):
    """Enable/disable strict mode for _stmt_val() lookups."""
    global _STMT_VAL_STRICT
    _STMT_VAL_STRICT = enabled


def get_stmt_val_misses() -> list:
    """Return list of recorded _stmt_val misses (each is a dict)."""
    return list(_STMT_VAL_MISSES)


def clear_stmt_val_misses():
    """Clear the recorded _stmt_val misses."""
    _STMT_VAL_MISSES.clear()


def _find_stmt_label(stmt, label):
    """Find a label in a financial-statement DataFrame index using fuzzy matching.

    Matching strategy: exact match first (case-insensitive), then
    word-boundary substring fallback (startswith/endswith).

    Returns the matched index label, or None if not found.
    """
    target = label.lower().strip()
    # Pass 1: exact match (case-insensitive, stripped)
    for idx in stmt.index:
        if target == str(idx).lower().strip():
            return idx
    # Pass 2: word-boundary substring fallback — the target words
    # must appear as a contiguous sequence within the index label,
    # but only when the label starts with or ends with the target
    # (avoids "operating income" matching "net income from
    # continuing operation").
    for idx in stmt.index:
        idx_low = str(idx).lower().strip()
        if idx_low.startswith(target) or idx_low.endswith(target):
            return idx
    return None


def _stmt_val(stmt, label, col=0, default=np.nan):
    """Pull a value from a yfinance financial-statement DataFrame.

    Matching strategy: exact match first, then word-boundary substring
    fallback (requires target to appear as a contiguous word sequence).

    When _STMT_VAL_STRICT is True, records every miss (label not found or
    empty statement) to _STMT_VAL_MISSES for downstream reporting.
    """
    try:
        if stmt is None or stmt.empty:
            if _STMT_VAL_STRICT:
                reason = "empty_statement" if stmt is not None else "null_statement"
                _STMT_VAL_MISSES.append({
                    "label": label, "col": col, "reason": reason,
                    "available_labels": [],
                })
            return default
        matched_idx = _find_stmt_label(stmt, label)
        if matched_idx is None:
            if _STMT_VAL_STRICT:
                _STMT_VAL_MISSES.append({
                    "label": label, "col": col, "reason": "label_not_found",
                    "available_labels": [str(i) for i in stmt.index[:15]],
                })
            return default
        vals = stmt.loc[matched_idx].dropna()
        if len(vals) > col:
            return float(vals.iloc[col])
        # Miss: column index out of range
        if _STMT_VAL_STRICT:
            _STMT_VAL_MISSES.append({
                "label": label, "col": col, "reason": f"col_out_of_range:{len(vals)}",
                "available_labels": [str(i) for i in stmt.index[:15]],
            })
        return default
    except (KeyError, IndexError, TypeError, ValueError) as e:
        if _STMT_VAL_STRICT:
            _STMT_VAL_MISSES.append({
                "label": label, "col": col, "reason": f"exception:{type(e).__name__}",
                "available_labels": [],
            })
        warnings.warn(f"_stmt_val failed for label='{label}', col={col}: {type(e).__name__}: {e}")
        return default


def _stmt_val_ltm(stmt, label, n_quarters=4, offset=0, default=np.nan,
                   partial_labels=None):
    """Compute LTM (sum of N quarters) from a quarterly statement DataFrame.

    Parameters
    ----------
    stmt : pd.DataFrame or None
        Quarterly financial statement from yfinance (columns = dates,
        most recent first; rows = line items).
    label : str
        Line item label to look up (same fuzzy matching as _stmt_val).
    n_quarters : int
        Number of quarters to sum (default 4 for LTM).
    offset : int
        Starting column offset. 0 = most recent LTM, 4 = prior-year LTM.
    default : float
        Value returned if data is unavailable.
    partial_labels : list or None
        If provided, the label is appended when partial annualization
        (3-of-4 quarters) is used, so callers can flag the ticker.

    Returns
    -------
    float
        Sum of the N quarterly values, or default if insufficient data.
        If only 3 of 4 quarters are available (offset=0), annualizes
        as sum * (4/3).
    """
    try:
        if stmt is None or stmt.empty:
            if _STMT_VAL_STRICT:
                reason = "empty_statement" if stmt is not None else "null_statement"
                _STMT_VAL_MISSES.append({
                    "label": label, "col": f"ltm:{offset}:{offset+n_quarters}",
                    "reason": reason, "available_labels": [],
                })
            return default

        matched_idx = _find_stmt_label(stmt, label)
        if matched_idx is None:
            if _STMT_VAL_STRICT:
                _STMT_VAL_MISSES.append({
                    "label": label, "col": f"ltm:{offset}:{offset+n_quarters}",
                    "reason": "label_not_found",
                    "available_labels": [str(i) for i in stmt.index[:15]],
                })
            return default

        row = stmt.loc[matched_idx].dropna()
        needed = offset + n_quarters
        if len(row) >= needed:
            return sum(float(row.iloc[offset + i]) for i in range(n_quarters))

        # Partial data: if 3 of 4 quarters available at offset=0, annualize
        available = len(row) - offset
        if available >= 3 and n_quarters == 4 and offset == 0:
            partial = sum(float(row.iloc[offset + i]) for i in range(available))
            if partial_labels is not None:
                partial_labels.append(label)
            return partial * (4 / available)

        if _STMT_VAL_STRICT:
            _STMT_VAL_MISSES.append({
                "label": label, "col": f"ltm:{offset}:{offset+n_quarters}",
                "reason": f"insufficient_quarters:{len(row)}",
                "available_labels": [],
            })
        return default
    except (KeyError, IndexError, TypeError, ValueError) as e:
        if _STMT_VAL_STRICT:
            _STMT_VAL_MISSES.append({
                "label": label, "col": f"ltm:{offset}:{offset+n_quarters}",
                "reason": f"exception:{type(e).__name__}",
                "available_labels": [],
            })
        warnings.warn(f"_stmt_val_ltm failed for label='{label}', offset={offset}: {type(e).__name__}: {e}")
        return default


_NON_RETRYABLE_PATTERNS = ["404", "no data", "not found", "delisted"]
_RATE_LIMIT_PATTERNS = ["429", "too many requests", "rate limit"]


def _is_rate_limited(err_str: str) -> bool:
    """Check if an error string indicates Yahoo Finance rate limiting."""
    return any(p in err_str.lower() for p in _RATE_LIMIT_PATTERNS)


def _compute_beneish_mscore(d: dict):
    """Compute Beneish M-Score from annual financial statement data.

    Returns (m_score, flag) where flag is True if M-Score > -2.22
    (indicating potential earnings manipulation).

    Uses the 8-variable model from Beneish (1999):
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Missing individual index inputs default to 1.0 (neutral), except
    revenue and total assets which are required for both years.

    Returns (NaN, False) if fewer than 5 of 8 indices can be computed
    from actual data (analogous to Piotroski's n_testable >= 6 gate).
    """
    # Extract required fields (all prefixed _beneish_ from fetch layer)
    rec_t  = d.get("_beneish_net_receivables", np.nan)
    rec_p  = d.get("_beneish_net_receivables_p", np.nan)
    rev_t  = d.get("_beneish_revenue", np.nan)
    rev_p  = d.get("_beneish_revenue_p", np.nan)
    cogs_t = d.get("_beneish_cogs", np.nan)
    cogs_p = d.get("_beneish_cogs_p", np.nan)
    ca_t   = d.get("_beneish_current_assets", np.nan)
    ca_p   = d.get("_beneish_current_assets_p", np.nan)
    ppe_t  = d.get("_beneish_ppe", np.nan)
    ppe_p  = d.get("_beneish_ppe_p", np.nan)
    ta_t   = d.get("_beneish_total_assets", np.nan)
    ta_p   = d.get("_beneish_total_assets_p", np.nan)
    dep_t  = d.get("_beneish_depreciation", np.nan)
    dep_p  = d.get("_beneish_depreciation_p", np.nan)
    sga_t  = d.get("_beneish_sga", np.nan)
    sga_p  = d.get("_beneish_sga_p", np.nan)
    ltd_t  = d.get("_beneish_lt_debt", np.nan)
    ltd_p  = d.get("_beneish_lt_debt_p", np.nan)
    cl_t   = d.get("_beneish_current_liab", np.nan)
    cl_p   = d.get("_beneish_current_liab_p", np.nan)
    ni_t   = d.get("_beneish_net_income", np.nan)
    ocf_t  = d.get("_beneish_ocf", np.nan)

    # Minimum required: revenue and total assets for both years
    if any(pd.isna(x) or x == 0 for x in [rev_t, rev_p, ta_t, ta_p]):
        return np.nan, False

    n_computed = 0  # Track how many indices are computed from real data

    # 1. DSRI (Days Sales in Receivables Index)
    if pd.notna(rec_t) and pd.notna(rec_p) and rec_p > 0:
        dsri = (rec_t / rev_t) / (rec_p / rev_p)
        n_computed += 1
    else:
        dsri = 1.0  # neutral

    # 2. GMI (Gross Margin Index)
    gm_t = (rev_t - cogs_t) / rev_t if (pd.notna(cogs_t) and rev_t > 0) else np.nan
    gm_p = (rev_p - cogs_p) / rev_p if (pd.notna(cogs_p) and rev_p > 0) else np.nan
    if pd.notna(gm_t) and pd.notna(gm_p) and gm_t > 0:
        gmi = gm_p / gm_t
        n_computed += 1
    else:
        gmi = 1.0

    # 3. AQI (Asset Quality Index)
    if all(pd.notna(x) for x in [ca_t, ppe_t, ta_t, ca_p, ppe_p, ta_p]):
        aq_t = 1 - (ca_t + ppe_t) / ta_t
        aq_p = 1 - (ca_p + ppe_p) / ta_p
        aqi = (aq_t / aq_p) if aq_p != 0 else 1.0
        n_computed += 1
    else:
        aqi = 1.0

    # 4. SGI (Sales Growth Index) — always computable (rev guaranteed above)
    sgi = rev_t / rev_p
    n_computed += 1

    # 5. DEPI (Depreciation Index)
    if (all(pd.notna(x) for x in [dep_t, dep_p, ppe_t, ppe_p])
            and (ppe_t + dep_t) > 0 and (ppe_p + dep_p) > 0):
        depi = (dep_p / (ppe_p + dep_p)) / (dep_t / (ppe_t + dep_t))
        n_computed += 1
    else:
        depi = 1.0

    # 6. SGAI (SGA Expense Index) — set to 1.0 (neutral) if SGA missing
    if (all(pd.notna(x) for x in [sga_t, sga_p])
            and rev_t > 0 and rev_p > 0 and sga_p > 0):
        sgai = (sga_t / rev_t) / (sga_p / rev_p)
        n_computed += 1
    else:
        sgai = 1.0

    # 7. LVGI (Leverage Index)
    if (all(pd.notna(x) for x in [ltd_t, cl_t, ta_t, ltd_p, cl_p, ta_p])
            and ta_t > 0 and ta_p > 0
            and (ltd_p + cl_p) > 0):
        lvgi = ((ltd_t + cl_t) / ta_t) / ((ltd_p + cl_p) / ta_p)
        n_computed += 1
    else:
        lvgi = 1.0

    # 8. TATA (Total Accruals to Total Assets)
    if pd.notna(ni_t) and pd.notna(ocf_t) and ta_t > 0:
        tata = (ni_t - ocf_t) / ta_t
        n_computed += 1
    else:
        tata = 0.0  # neutral

    # Minimum-data gate: require >= 5 of 8 indices computed from real data.
    # With < 5 indices, the M-Score is dominated by neutral defaults (1.0)
    # and loses discriminating power — analogous to Piotroski's n_testable >= 6.
    if n_computed < 5:
        return np.nan, False

    m_score = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi
               + 0.892 * sgi + 0.115 * depi - 0.172 * sgai
               + 4.679 * tata - 0.327 * lvgi)

    return m_score, (m_score > -2.22)


def fetch_single_ticker(ticker_str: str, max_retries: int = 3,
                        per_request_delay: float = 0.0) -> dict:
    """Fetch all required data for one ticker via yfinance.

    Implements exponential backoff retry (1s / 2s / 4s) per §10.3.
    Returns dict with '_fetch_time_ms' for per-ticker timing.
    Non-retryable errors (404, delisted) fail immediately.
    Rate-limit errors (429) are tagged '_rate_limited' and returned
    immediately so the batch coordinator can pause and adapt.
    """
    t_start = time.time()
    rec = {"Ticker": ticker_str}
    last_err = None
    for attempt in range(max_retries):
        if per_request_delay > 0:
            time.sleep(per_request_delay)
        try:
            rec = _fetch_single_ticker_inner(ticker_str)
            if "_error" not in rec:
                rec["_fetch_time_ms"] = round((time.time() - t_start) * 1000)
                return rec
            last_err = rec.get("_error", "unknown")
            # Skip retry for permanent errors
            if rec.get("_non_retryable", False):
                break
            # Rate limit: tag and return immediately (don't waste retries)
            if _is_rate_limited(last_err):
                rec["_rate_limited"] = True
                rec["_fetch_time_ms"] = round((time.time() - t_start) * 1000)
                return rec
        except Exception as exc:
            last_err = str(exc)
            if _is_rate_limited(last_err):
                rec = {"Ticker": ticker_str, "_error": last_err,
                       "_rate_limited": True,
                       "_fetch_time_ms": round((time.time() - t_start) * 1000)}
                return rec
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
        rec["payoutRatio"]        = _safe(info, "payoutRatio")
        # Bank-specific info fields (zero additional API cost — same .info dict)
        rec["returnOnEquity"]    = _safe(info, "returnOnEquity")
        rec["returnOnAssets"]    = _safe(info, "returnOnAssets")
        rec["priceToBook"]       = _safe(info, "priceToBook")
        rec["bookValue"]         = _safe(info, "bookValue")
        rec["industry"]          = _safe(info, "industry", "")
        # Analyst price target fields (zero additional API cost — same .info dict)
        rec["targetMeanPrice"]         = _safe(info, "targetMeanPrice")
        rec["targetHighPrice"]         = _safe(info, "targetHighPrice")
        rec["targetLowPrice"]          = _safe(info, "targetLowPrice")
        rec["numberOfAnalystOpinions"] = _safe(info, "numberOfAnalystOpinions")
        # Short interest (days to cover). Updated bi-monthly by exchanges with 1-2 week lag.
        rec["shortRatio"]             = _safe(info, "shortRatio")

        # ---- quarterly financial statements (for LTM / MRQ) ----
        # LTM = sum of last 4 quarters (flow metrics: IS + CF)
        # MRQ = most recent quarter (balance sheet items)
        # Annual statements kept as fallback for tickers without quarterly data.
        try:
            q_fins = t.quarterly_financials
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: quarterly_financials fetch failed: {type(e).__name__}: {e}")
            q_fins = None
        try:
            q_bs = t.quarterly_balance_sheet
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: quarterly_balance_sheet fetch failed: {type(e).__name__}: {e}")
            q_bs = None
        try:
            q_cf = t.quarterly_cashflow
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: quarterly_cashflow fetch failed: {type(e).__name__}: {e}")
            q_cf = None

        # ---- annual statements (fallback) ----
        try:
            fins = t.financials
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: financials fetch failed: {type(e).__name__}: {e}")
            fins = None
        try:
            bs = t.balance_sheet
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: balance_sheet fetch failed: {type(e).__name__}: {e}")
            bs = None
        try:
            cf = t.cashflow
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: cashflow fetch failed: {type(e).__name__}: {e}")
            cf = None

        # ---- Income statement: LTM (sum of last 4 quarters) ----
        # Current-period flow metrics use LTM (sum of Q0..Q3).
        # Prior-period flow metrics try LTM (Q4..Q7) first, but yfinance
        # typically provides only 4-5 quarters. When prior-year LTM is
        # unavailable, fall back to annual statement col=1 (prior year).
        _ltm_partial = []  # tracks labels where 3-of-4 quarter annualization was used
        rec["totalRevenue"]           = _stmt_val_ltm(q_fins, "Total Revenue", partial_labels=_ltm_partial)
        rec["totalRevenue_prior"]     = _stmt_val_ltm(q_fins, "Total Revenue", offset=4)
        rec["grossProfit"]            = _stmt_val_ltm(q_fins, "Gross Profit", partial_labels=_ltm_partial)
        rec["grossProfit_prior"]      = _stmt_val_ltm(q_fins, "Gross Profit", offset=4)
        rec["ebit"]                   = _stmt_val_ltm(q_fins, "EBIT", partial_labels=_ltm_partial)
        if np.isnan(rec["ebit"]):
            rec["ebit"]               = _stmt_val_ltm(q_fins, "Operating Income", partial_labels=_ltm_partial)
        rec["ebitda"]                 = _stmt_val_ltm(q_fins, "EBITDA", partial_labels=_ltm_partial)
        rec["netIncome"]              = _stmt_val_ltm(q_fins, "Net Income", partial_labels=_ltm_partial)
        rec["netIncome_prior"]        = _stmt_val_ltm(q_fins, "Net Income", offset=4)
        rec["incomeTaxExpense"]       = _stmt_val_ltm(q_fins, "Tax Provision", partial_labels=_ltm_partial)
        if np.isnan(rec["incomeTaxExpense"]):
            rec["incomeTaxExpense"]   = _stmt_val_ltm(q_fins, "Income Tax", partial_labels=_ltm_partial)
        rec["pretaxIncome"]           = _stmt_val_ltm(q_fins, "Pretax Income", partial_labels=_ltm_partial)
        rec["costOfRevenue"]          = _stmt_val_ltm(q_fins, "Cost Of Revenue", partial_labels=_ltm_partial)

        # Fallback: if quarterly IS produced all NaN, try annual for everything
        if all(np.isnan(rec.get(k, np.nan)) for k in ["totalRevenue", "netIncome", "ebit"]):
            rec["_data_source"] = "annual"
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
                rec["incomeTaxExpense"]   = _stmt_val(fins, "Income Tax")
            rec["pretaxIncome"]           = _stmt_val(fins, "Pretax Income")
            rec["costOfRevenue"]          = _stmt_val(fins, "Cost Of Revenue")
        else:
            rec["_data_source"] = "quarterly"
            # Prior-year fallback: yfinance typically provides only 4-5
            # quarterly columns, so prior-year LTM (offset=4) often fails.
            # Fall back to annual statement col=1 for prior-year values.
            if np.isnan(rec["totalRevenue_prior"]):
                rec["totalRevenue_prior"] = _stmt_val(fins, "Total Revenue", 1)
            if np.isnan(rec["grossProfit_prior"]):
                rec["grossProfit_prior"]  = _stmt_val(fins, "Gross Profit", 1)
            if np.isnan(rec["netIncome_prior"]):
                rec["netIncome_prior"]    = _stmt_val(fins, "Net Income", 1)

        # Revenue 3 years ago from annual financials (col=3) for 3-year CAGR.
        # Annual financials typically provides 4 columns (indices 0-3).
        rec["totalRevenue_3yr_ago"] = _stmt_val(fins, "Total Revenue", 3)

        # EBIT prior year from annual financials (col=1) for operating leverage (DOL).
        rec["ebit_prior"] = _stmt_val(fins, "EBIT", 1)
        if np.isnan(rec["ebit_prior"]):
            rec["ebit_prior"] = _stmt_val(fins, "Operating Income", 1)

        # ---- Balance sheet: MRQ (most recent quarter) ----
        # Use quarterly BS for current values; col=4 for year-ago MRQ.
        # Fall back to annual BS if quarterly is unavailable.
        _bs_src = q_bs if (q_bs is not None and not q_bs.empty) else bs
        _bs_is_quarterly = (_bs_src is q_bs)
        _bs_prior_col = 4 if (_bs_is_quarterly and _bs_src is not None
                               and len(_bs_src.columns) >= 5) else 1

        rec["totalAssets"]            = _stmt_val(_bs_src, "Total Assets")
        rec["totalAssets_prior"]      = _stmt_val(_bs_src, "Total Assets", _bs_prior_col)
        rec["totalEquity"]            = _stmt_val(_bs_src, "Stockholders Equity")
        if np.isnan(rec["totalEquity"]):
            rec["totalEquity"]        = _stmt_val(_bs_src, "Total Stockholder")
        rec["totalEquity_prior"]      = _stmt_val(_bs_src, "Stockholders Equity", _bs_prior_col)
        if np.isnan(rec["totalEquity_prior"]):
            rec["totalEquity_prior"]  = _stmt_val(_bs_src, "Total Stockholder", _bs_prior_col)
        rec["totalDebt_bs"]           = _stmt_val(_bs_src, "Total Debt")
        rec["longTermDebt"]           = _stmt_val(_bs_src, "Long Term Debt")
        rec["longTermDebt_prior"]     = _stmt_val(_bs_src, "Long Term Debt", _bs_prior_col)
        rec["currentLiabilities"]     = _stmt_val(_bs_src, "Current Liabilities")
        rec["currentAssets"]          = _stmt_val(_bs_src, "Current Assets")
        rec["currentAssets_prior"]    = _stmt_val(_bs_src, "Current Assets", _bs_prior_col)
        rec["currentLiabilities_prior"] = _stmt_val(_bs_src, "Current Liabilities", _bs_prior_col)
        rec["cash_bs"]                = _stmt_val(_bs_src, "Cash And Cash Equivalents")
        rec["sharesBS"]               = _stmt_val(_bs_src, "Ordinary Shares Number")
        rec["sharesBS_prior"]         = _stmt_val(_bs_src, "Ordinary Shares Number", _bs_prior_col)
        if np.isnan(rec["sharesBS"]):
            rec["sharesBS"]           = _stmt_val(_bs_src, "Share Issued")
            rec["sharesBS_prior"]     = _stmt_val(_bs_src, "Share Issued", _bs_prior_col)

        # ---- Cash flow: LTM (sum of last 4 quarters) ----
        rec["operatingCashFlow"]      = _stmt_val_ltm(q_cf, "Operating Cash Flow", partial_labels=_ltm_partial)
        if np.isnan(rec["operatingCashFlow"]):
            rec["operatingCashFlow"]  = _stmt_val_ltm(q_cf, "Total Cash From Operating", partial_labels=_ltm_partial)
        rec["capex"]                  = _stmt_val_ltm(q_cf, "Capital Expenditure", partial_labels=_ltm_partial)
        if np.isnan(rec["capex"]):
            rec["capex"]              = _stmt_val_ltm(q_cf, "Capital Expenditures", partial_labels=_ltm_partial)
        rec["dividendsPaid"]          = _stmt_val_ltm(q_cf, "Common Stock Dividend", partial_labels=_ltm_partial)
        if np.isnan(rec["dividendsPaid"]):
            rec["dividendsPaid"]      = _stmt_val_ltm(q_cf, "Dividends Paid", partial_labels=_ltm_partial)
        # D&A from cashflow (for computing GAAP EBITDA = EBIT + D&A)
        rec["da_cf"]                  = _stmt_val_ltm(q_cf, "Depreciation And Amortization", partial_labels=_ltm_partial)
        if np.isnan(rec["da_cf"]):
            rec["da_cf"]              = _stmt_val_ltm(q_cf, "Reconciled Depreciation", partial_labels=_ltm_partial)
        if np.isnan(rec["da_cf"]):
            rec["da_cf"]              = _stmt_val_ltm(q_cf, "Depreciation Amortization Depletion", partial_labels=_ltm_partial)

        # Fallback: if quarterly CF produced all NaN, try annual
        if all(np.isnan(rec.get(k, np.nan)) for k in ["operatingCashFlow", "capex"]):
            rec["operatingCashFlow"]      = _stmt_val(cf, "Operating Cash Flow")
            if np.isnan(rec["operatingCashFlow"]):
                rec["operatingCashFlow"]  = _stmt_val(cf, "Total Cash From Operating")
            rec["capex"]                  = _stmt_val(cf, "Capital Expenditure")
            if np.isnan(rec["capex"]):
                rec["capex"]              = _stmt_val(cf, "Capital Expenditures")
            rec["dividendsPaid"]          = _stmt_val(cf, "Common Stock Dividend")
            if np.isnan(rec["dividendsPaid"]):
                rec["dividendsPaid"]      = _stmt_val(cf, "Dividends Paid")
            rec["da_cf"]                  = _stmt_val(cf, "Depreciation And Amortization")
            if np.isnan(rec["da_cf"]):
                rec["da_cf"]              = _stmt_val(cf, "Reconciled Depreciation")
            if np.isnan(rec["da_cf"]):
                rec["da_cf"]              = _stmt_val(cf, "Depreciation Amortization Depletion")

        # ---- LTM partial-annualization flag ----
        # If any current-period LTM metric used 3-of-4 quarter annualization,
        # flag the ticker so downstream consumers know data may be less precise.
        if _ltm_partial:
            rec["_ltm_annualized"] = True
            rec["_ltm_annualized_labels"] = list(set(_ltm_partial))
        else:
            rec["_ltm_annualized"] = False

        # ---- Beneish M-Score data: ANNUAL statements (col 0 = current, col 1 = prior year) ----
        # Uses annual (not LTM/MRQ) because Beneish was designed for annual data
        # and year-over-year comparison requires the same reporting basis.
        rec["_beneish_net_receivables"]   = _stmt_val(bs, "Net Receivable")
        if np.isnan(rec["_beneish_net_receivables"]):
            rec["_beneish_net_receivables"] = _stmt_val(bs, "Receivables")
        if np.isnan(rec["_beneish_net_receivables"]):
            rec["_beneish_net_receivables"] = _stmt_val(bs, "Accounts Receivable")
        rec["_beneish_net_receivables_p"] = _stmt_val(bs, "Net Receivable", 1)
        if np.isnan(rec["_beneish_net_receivables_p"]):
            rec["_beneish_net_receivables_p"] = _stmt_val(bs, "Receivables", 1)
        if np.isnan(rec["_beneish_net_receivables_p"]):
            rec["_beneish_net_receivables_p"] = _stmt_val(bs, "Accounts Receivable", 1)
        rec["_beneish_revenue"]           = _stmt_val(fins, "Total Revenue")
        rec["_beneish_revenue_p"]         = _stmt_val(fins, "Total Revenue", 1)
        rec["_beneish_cogs"]              = _stmt_val(fins, "Cost Of Revenue")
        rec["_beneish_cogs_p"]            = _stmt_val(fins, "Cost Of Revenue", 1)
        rec["_beneish_current_assets"]    = _stmt_val(bs, "Current Assets")
        rec["_beneish_current_assets_p"]  = _stmt_val(bs, "Current Assets", 1)
        rec["_beneish_ppe"]               = _stmt_val(bs, "Net PPE")
        if np.isnan(rec["_beneish_ppe"]):
            rec["_beneish_ppe"]           = _stmt_val(bs, "Property Plant Equipment")
        rec["_beneish_ppe_p"]             = _stmt_val(bs, "Net PPE", 1)
        if np.isnan(rec["_beneish_ppe_p"]):
            rec["_beneish_ppe_p"]         = _stmt_val(bs, "Property Plant Equipment", 1)
        rec["_beneish_total_assets"]      = _stmt_val(bs, "Total Assets")
        rec["_beneish_total_assets_p"]    = _stmt_val(bs, "Total Assets", 1)
        rec["_beneish_depreciation"]      = _stmt_val(cf, "Depreciation And Amortization")
        if np.isnan(rec["_beneish_depreciation"]):
            rec["_beneish_depreciation"]  = _stmt_val(cf, "Depreciation")
        rec["_beneish_depreciation_p"]    = _stmt_val(cf, "Depreciation And Amortization", 1)
        if np.isnan(rec["_beneish_depreciation_p"]):
            rec["_beneish_depreciation_p"] = _stmt_val(cf, "Depreciation", 1)
        rec["_beneish_sga"]               = _stmt_val(fins, "Selling General And Administration")
        if np.isnan(rec["_beneish_sga"]):
            rec["_beneish_sga"]           = _stmt_val(fins, "Selling General And Admin")
        rec["_beneish_sga_p"]             = _stmt_val(fins, "Selling General And Administration", 1)
        if np.isnan(rec["_beneish_sga_p"]):
            rec["_beneish_sga_p"]         = _stmt_val(fins, "Selling General And Admin", 1)
        rec["_beneish_lt_debt"]           = _stmt_val(bs, "Long Term Debt")
        rec["_beneish_lt_debt_p"]         = _stmt_val(bs, "Long Term Debt", 1)
        rec["_beneish_current_liab"]      = _stmt_val(bs, "Current Liabilities")
        rec["_beneish_current_liab_p"]    = _stmt_val(bs, "Current Liabilities", 1)
        rec["_beneish_net_income"]        = _stmt_val(fins, "Net Income")
        rec["_beneish_ocf"]               = _stmt_val(cf, "Operating Cash Flow")
        if np.isnan(rec["_beneish_ocf"]):
            rec["_beneish_ocf"]           = _stmt_val(cf, "Total Cash From Operating")

        # ---- data freshness: record most recent filing date ----
        try:
            for stmt_name, stmt_obj in [
                ("financials", q_fins if (q_fins is not None and not q_fins.empty) else fins),
                ("balance_sheet", _bs_src),
                ("cashflow", q_cf if (q_cf is not None and not q_cf.empty) else cf),
            ]:
                if stmt_obj is not None and not stmt_obj.empty:
                    most_recent = stmt_obj.columns[0]
                    rec[f"_stmt_date_{stmt_name}"] = str(most_recent.date()) if hasattr(most_recent, "date") else str(most_recent)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: data freshness check failed: {type(e).__name__}: {e}")

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

                # Avg daily dollar volume (63 trading days ≈ 3 months)
                if "Volume" in hist.columns:
                    dv = hist["Close"] * hist["Volume"]
                    dv_63 = dv.tail(63).dropna()
                    rec["avg_daily_dollar_volume"] = (
                        float(dv_63.mean()) if len(dv_63) >= 20 else np.nan
                    )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: price history extraction failed: {type(e).__name__}: {e}")

        # ---- earnings surprises ----
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                surs = []
                ordered_surs = []  # Per-quarter surprises in chronological order
                for _, row in eh.tail(4).iterrows():
                    a, e = row.get("epsActual", np.nan), row.get("epsEstimate", np.nan)
                    if pd.notna(a) and pd.notna(e) and abs(e) > 0.001:
                        # Floor denominator at $0.10 to prevent near-zero
                        # estimates from producing extreme surprise ratios.
                        sur = (a - e) / max(abs(e), 0.10)
                        surs.append(sur)
                        ordered_surs.append(sur)
                    else:
                        ordered_surs.append(np.nan)
                # Median is robust to a single outlier quarter.
                rec["analyst_surprise"] = float(np.median(surs)) if len(surs) >= 2 else np.nan

                # --- Earnings Acceleration ---
                # Continuous: delta between most recent and prior quarter surprise %.
                # Positive = accelerating beats, negative = decelerating.
                valid_ordered = [s for s in ordered_surs if pd.notna(s)]
                if len(valid_ordered) >= 2:
                    rec["earnings_acceleration"] = valid_ordered[-1] - valid_ordered[-2]
                else:
                    rec["earnings_acceleration"] = np.nan

                # --- Recency-Weighted Beat Score ---
                # Each quarter's beat weighted by recency: Q1(oldest)=1, Q2=2, Q3=3, Q4(newest)=4.
                # Beat = weight, miss = 0. Range: 0 to 10 (=1+2+3+4) for 4 quarters.
                # More granular than the old 0-4 streak counter.
                if len(valid_ordered) >= 2:
                    n = len(valid_ordered)
                    weights = list(range(1, n + 1))
                    score = sum(w * (1.0 if s > 0 else 0.0)
                                for w, s in zip(weights, valid_ordered))
                    rec["consecutive_beat_streak"] = float(score)
                else:
                    rec["consecutive_beat_streak"] = np.nan
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: earnings surprise extraction failed: {type(e).__name__}: {e}")

    except Exception as exc:
        err_str = str(exc)
        rec["_error"] = err_str
        rec["_non_retryable"] = any(p in err_str.lower() for p in _NON_RETRYABLE_PATTERNS)
    return rec


def fetch_all_tickers(tickers: list, batch_size: int = 30,
                      max_workers: int = 3,
                      inter_batch_delay: float = 3.0) -> list:
    """Fetch data for all tickers with adaptive rate-limit throttling.

    Starts with the configured concurrency.  When a rate-limit (HTTP 429)
    is detected in any batch result the pipeline:
      1. Pauses for an escalating backoff (30 s / 60 s / 120 s cap).
      2. Reduces worker count by 1 (minimum 1).
      3. Increases inter-batch delay by 2 s (maximum 15 s).
    """
    results: list[dict] = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    current_workers = max_workers
    current_delay = inter_batch_delay
    rate_limit_backoffs = 0

    for bi in range(n_batches):
        batch = tickers[bi * batch_size : (bi + 1) * batch_size]
        print(f"  Batch {bi+1}/{n_batches}  ({batch[0]}..{batch[-1]})  "
              f"[workers={current_workers}, delay={current_delay:.0f}s]")

        batch_results: list[dict] = []
        batch_rate_limited = False

        with ThreadPoolExecutor(max_workers=current_workers) as pool:
            futs = {pool.submit(fetch_single_ticker, t): t for t in batch}
            for fut in as_completed(futs):
                try:
                    rec = fut.result(timeout=120)
                    batch_results.append(rec)
                    if rec.get("_rate_limited"):
                        batch_rate_limited = True
                except Exception as e:
                    batch_results.append({"Ticker": futs[fut], "_error": str(e)})

        results.extend(batch_results)

        # Adaptive throttling on rate-limit detection
        if batch_rate_limited:
            rate_limit_backoffs += 1
            backoff_time = min(30 * (2 ** (rate_limit_backoffs - 1)), 120)
            print(f"  ** Rate limit detected — pausing {backoff_time}s "
                  f"(backoff #{rate_limit_backoffs}) **")
            time.sleep(backoff_time)
            current_workers = max(1, current_workers - 1)
            current_delay = min(current_delay + 2, 15)

        if bi < n_batches - 1:
            time.sleep(current_delay)

    n_rate_limited = sum(1 for r in results if r.get("_rate_limited"))
    if n_rate_limited > 0:
        print(f"  Adaptive throttling: {rate_limit_backoffs} backoff(s), "
              f"{n_rate_limited} tickers rate-limited")
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
        except Exception as e:
            warnings.warn(f"Market returns fetch attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return pd.Series(dtype=float)


def fetch_risk_free_rate(max_retries: int = 3) -> float:
    """Fetch the 13-week T-bill yield (^IRX) as an annualized risk-free rate.

    Returns the most recent closing yield as a decimal (e.g. 0.045 for 4.5%).
    Falls back to 4.5% if the fetch fails, with a logged warning.
    Used for Jensen's Alpha and Sharpe Ratio calculations.
    """
    for attempt in range(max_retries):
        try:
            import yfinance as yf
            hist = yf.Ticker("^IRX").history(period="5d", auto_adjust=True)
            if hist is not None and not hist.empty:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    # ^IRX is quoted in percentage points (e.g. 4.5 means 4.5%)
                    last_close = float(closes.iloc[-1])
                    rf = last_close / 100.0
                    return rf
        except Exception as e:
            warnings.warn(f"Risk-free rate fetch attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    warnings.warn("All risk-free rate (^IRX) fetch attempts failed; using default 4.5%")
    return 0.045


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
        fwd_eps_growth  = float(np.clip((fwd_eps - eps) / max(abs(eps), 1.0), -0.75, 1.50)) if abs(eps) > 0.01 else np.nan
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
        # Price target upside: similar sparsity to analyst surprise
        price_target_upside = tn(0.10, 0.15) if rng.random() < 0.40 else np.nan
        # Earnings acceleration and beat streak: same sparsity as analyst surprise
        earnings_accel = tn(0.02, 0.05) if pd.notna(analyst_surprise) else np.nan
        beat_streak = round(tn(5, 3, low=0, high=10)) if pd.notna(analyst_surprise) else np.nan

        rec = {
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
            "return_12m": round(mom_12_1 + tn(0.01, 0.02), 4),  # Full 12m ≈ 12-1 + recent month effect
            "jensens_alpha": round(tn(0.03, 0.10), 4),
            "volatility": round(vol, 4),
            "beta": round(beta, 2),
            "sharpe_ratio": round((mom_12_1 - 0.045) / vol, 2) if vol > 0 else np.nan,
            "analyst_surprise": round(analyst_surprise, 4) if pd.notna(analyst_surprise) else np.nan,
            "price_target_upside": round(price_target_upside, 4) if pd.notna(price_target_upside) else np.nan,
            "earnings_acceleration": round(earnings_accel, 4) if pd.notna(earnings_accel) else np.nan,
            "consecutive_beat_streak": float(beat_streak) if pd.notna(beat_streak) else np.nan,
            "size_log_mcap": round(-np.log(mc), 4) if mc > 0 else np.nan,
            "net_debt_to_ebitda": round(tn(2.0, 1.5, low=0.0, high=8.0), 2),
            "operating_leverage": round(tn(1.5, 1.0, low=0.2, high=5.0), 2),
            "revenue_cagr_3yr": round(tn(0.08, 0.06, low=-0.10, high=0.40), 4),
            "short_interest_ratio": round(tn(3.0, 2.5, low=0.1, high=15.0), 2) if rng.random() < 0.70 else np.nan,
            "asset_growth": round(rng.normal(0.08, 0.15), 4),
            "avg_daily_dollar_volume": round(rng.lognormal(np.log(50e6), 1.0), 0),
        }

        # Bank-specific metrics for Financials sector
        if sector == "Financials":
            rec["_is_bank_like"] = True
            rec["pb_ratio"] = round(tn(1.2, 0.4, low=0.3, high=3.5), 2)
            rec["roe"] = round(tn(0.12, 0.04, low=0.02), 4)
            rec["roa"] = round(tn(0.01, 0.005, low=0.002), 4)
            rec["equity_ratio"] = round(tn(0.10, 0.03, low=0.05, high=0.20), 4)
            # Null out meaningless generic metrics
            rec["ev_ebitda"] = np.nan
            rec["ev_sales"] = np.nan
            rec["roic"] = np.nan
            rec["gross_profit_assets"] = np.nan
            rec["debt_equity"] = np.nan
            rec["net_debt_to_ebitda"] = np.nan  # Banks: skip
            rec["operating_leverage"] = np.nan   # Banks: skip
        else:
            rec["_is_bank_like"] = False
            rec["pb_ratio"] = np.nan
            rec["roe"] = np.nan
            rec["roa"] = np.nan
            rec["equity_ratio"] = np.nan

        records.append(rec)
    return pd.DataFrame(records)


# =========================================================================
# E. Compute all 30 individual metrics (from live yfinance data)
# =========================================================================
def compute_metrics(raw_data: list, market_returns: pd.Series,
                    cfg: dict | None = None,
                    risk_free_rate: float = 0.045) -> pd.DataFrame:
    """Compute all factor metrics (~33 metrics) from raw yfinance ticker data."""
    clamps = (cfg or {}).get("metric_clamps", {})
    feg_lo, feg_hi = clamps.get("forward_eps_growth", [-0.75, 1.50])
    ptu_lo, ptu_hi = clamps.get("price_target_upside", [-0.50, 1.0])
    peg_max_cap = clamps.get("peg_max_cap", 50)
    records = []

    # Market 12-month total return (computed once, reused for all tickers).
    # Convert cumulative log returns to simple return.
    if len(market_returns) >= 200:
        market_12m_return = float(np.exp(market_returns.sum()) - 1)
    else:
        market_12m_return = float('nan')

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
        mc = d.get("marketCap", np.nan)
        ev = d.get("enterpriseValue", np.nan)
        # Debt figures:
        # _debt_info: from .info (includes short-term). Used for EV fallback
        #   and D/E ratio (matches yfinance's own EV definition).
        # _debt_bs: from balance sheet. Used for ROIC invested capital
        #   (consistent source with equity and cash, which are also from BS).
        _debt_info = d.get("totalDebt", d.get("totalDebt_bs", np.nan))
        _debt_bs = d.get("totalDebt_bs", d.get("totalDebt", np.nan))
        # Cash: use info totalCash for EV (matches yfinance's own EV
        # definition which includes short-term investments), but use
        # balance sheet Cash & Cash Equivalents for ROIC (stricter
        # definition of invested capital).
        _cash_ev = d.get("totalCash", d.get("cash_bs", np.nan))
        _cash_bs = d.get("cash_bs", d.get("totalCash", np.nan))
        if pd.isna(ev) or ev == 0:
            # Only compute fallback EV when all components are available
            if pd.notna(mc) and pd.notna(_debt_info) and pd.notna(_cash_ev):
                ev = mc + _debt_info - _cash_ev
            else:
                ev = np.nan
        elif pd.notna(mc) and pd.notna(_debt_info) and pd.notna(_cash_ev):
            # EV cross-validation (Audit finding H4): yfinance has known
            # parsing bugs that can return EV values 4x+ off (e.g. TSM,
            # Issue #2507).  Compare API-provided EV against computed
            # MC + Debt - Cash; if discrepancy exceeds threshold, use
            # computed value.  Financials use a wider 25% threshold
            # because their "debt" includes customer deposits and other
            # liabilities that legitimately diverge from simple EV math.
            _ev_computed = mc + _debt_info - _cash_ev
            if _ev_computed > 0 and ev > 0:
                _ev_ratio = ev / _ev_computed
                _ev_tol = 0.25 if rec["Sector"] in _FINANCIAL_SECTORS else 0.10
                if _ev_ratio > (1 + _ev_tol) or _ev_ratio < (1 - _ev_tol):
                    rec["_ev_flag"] = (
                        f"API EV={ev/1e9:.1f}B vs computed={_ev_computed/1e9:.1f}B "
                        f"(ratio={_ev_ratio:.2f})")
                    ev = _ev_computed

        ta = d.get("totalAssets", np.nan)
        ni = d.get("netIncome", np.nan)
        eq_v = d.get("totalEquity", np.nan)
        ocf = d.get("operatingCashFlow", np.nan)
        rev_c = d.get("totalRevenue", np.nan)
        rev_p = d.get("totalRevenue_prior", np.nan)
        ticker = rec["Ticker"]

        # Pre-compute bank classification (needed early for Beneish exclusion)
        _sector = rec["Sector"]
        _industry = d.get("industry", "")
        _is_bank = _is_bank_like(ticker, _sector, _industry)

        # -- Valuation metrics (1-4) --
        try:
            # 1. EV/EBITDA — compute GAAP EBITDA as EBIT + D&A when both
            # components are available.  yfinance's reported EBITDA can
            # include non-operating items that distort the multiple.
            # Fall back to reported EBITDA if D&A is unavailable.
            _ebit_for_ebitda = d.get("ebit", np.nan)
            _da = d.get("da_cf", np.nan)
            if pd.notna(_ebit_for_ebitda) and pd.notna(_da) and _da >= 0:
                ebitda = _ebit_for_ebitda + _da
            else:
                ebitda = d.get("ebitda", np.nan)  # fallback to reported
            rec["ev_ebitda"] = (ev / ebitda) if (pd.notna(ev) and pd.notna(ebitda) and ebitda > 0 and ev > 0) else np.nan

            # 2. FCF Yield
            # Require both OCF and CapEx to compute FCF. When CapEx is
            # missing, FCF is NaN (not OCF — assuming zero capex would
            # dramatically overstate free cash flow for capital-intensive
            # companies, and FCF Yield has the highest valuation weight).
            capex = d.get("capex", np.nan)
            fcf = np.nan
            if pd.notna(ocf) and pd.notna(capex):
                fcf = (ocf - abs(capex)) if capex < 0 else (ocf - capex)
            rec["fcf_yield"] = (fcf / ev) if (pd.notna(fcf) and pd.notna(ev) and ev > 0) else np.nan

            # 3. Earnings Yield  (LTM Net Income / Market Cap)
            # Uses LTM net income from quarterly statements (same source as all
            # other fundamental metrics) instead of the opaque trailingEps from
            # the yfinance .info dict, for pipeline consistency.
            # Falls back to trailingEps / price when LTM NI or MC unavailable.
            if pd.notna(ni) and pd.notna(mc) and mc > 0:
                rec["earnings_yield"] = ni / mc
            else:
                _eps_fb = d.get("trailingEps", np.nan)
                _price_fb = d.get("currentPrice", d.get("price_latest", np.nan))
                rec["earnings_yield"] = (_eps_fb / _price_fb) if (pd.notna(_eps_fb) and pd.notna(_price_fb) and _price_fb > 0) else np.nan

            # 4. EV/Sales
            rec["ev_sales"] = (ev / rev_c) if (pd.notna(ev) and pd.notna(rev_c) and rev_c > 0 and ev > 0) else np.nan
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: valuation metrics failed: {type(e).__name__}: {e}")

        # -- Quality metrics (5-9) --
        try:
            # 5. ROIC (Invested Capital = Equity + Total Debt - Excess Cash)
            # Use balance-sheet debt and cash for IC — all three IC
            # components (equity, debt, cash) come from the same balance
            # sheet filing for temporal consistency.
            # Excess cash = max(0, cash - 2% of revenue). Deducting ALL
            # cash inflates ROIC for cash-rich companies (e.g. AAPL, GOOG).
            ebit_v = d.get("ebit", np.nan)
            if pd.notna(ebit_v):
                tax_exp = d.get("incomeTaxExpense", np.nan)
                pretax = d.get("pretaxIncome", np.nan)
                tax_rate = 0.21
                if pd.notna(pretax) and pretax <= 0:
                    # Tax-loss position: company wouldn't pay tax on operating
                    # earnings.  Using 21% here would create a fictional tax
                    # hit that understates NOPAT.  (Audit finding H3)
                    tax_rate = 0.0
                elif pd.notna(tax_exp) and pd.notna(pretax) and pretax > 0:
                    tax_rate = max(0, min(tax_exp / pretax, 0.5))
                nopat = ebit_v * (1 - tax_rate)
                if pd.notna(eq_v) and pd.notna(_debt_bs) and pd.notna(_cash_bs):
                    # Excess cash: cash beyond 2% of revenue (operating cash needs)
                    # Cap at 50% of total cash to prevent near-total IC elimination
                    # for cash-heavy companies (e.g. EXPE, asset-light platforms).
                    _operating_cash = 0.02 * rev_c if pd.notna(rev_c) and rev_c > 0 else 0
                    _excess_cash = max(0, _cash_bs - _operating_cash)
                    _excess_cash = min(_excess_cash, 0.5 * _cash_bs)
                    ic = eq_v + _debt_bs - _excess_cash
                    # Floor IC at 10% of Total Assets — prevents denominator
                    # collapse for asset-light or cash-heavy companies.
                    if pd.notna(ta) and ta > 0:
                        ic = max(ic, 0.10 * ta)
                    rec["roic"] = (nopat / ic) if ic > 0 else np.nan
                else:
                    rec["roic"] = np.nan
            else:
                rec["roic"] = np.nan

            # 6. Gross Profit / Assets
            gp = d.get("grossProfit", np.nan)
            rec["gross_profit_assets"] = (gp / ta) if (pd.notna(gp) and pd.notna(ta) and ta > 0) else np.nan

            # 7. Debt/Equity — computed for reference/DataValidation output only — not scored.
            # Replaced by net_debt_to_ebitda in quality scoring (negative equity
            # distorts D/E for buyback-heavy companies e.g. MCD, MO, LOW, Boeing).
            if pd.notna(_debt_bs) and pd.notna(eq_v) and eq_v > 0:
                rec["debt_equity"] = _debt_bs / eq_v
            else:
                rec["debt_equity"] = np.nan

            # 7b. Net Debt / EBITDA — replaces Debt/Equity in quality scoring.
            # Net Debt = Total Debt - Cash; negative net debt (net cash) → 0.0
            # (net cash companies are the best-case scenario, treated as floor).
            # Guard: negative EBITDA makes the ratio uninterpretable → NaN.
            # Banks: skip (return NaN, weight redistributes to other metrics).
            if not _is_bank:
                _ebit_nd = d.get("ebit", np.nan)
                _da_nd = d.get("da_cf", np.nan)
                if pd.notna(_ebit_nd) and pd.notna(_da_nd) and _da_nd >= 0:
                    _ebitda_nd = _ebit_nd + _da_nd
                else:
                    _ebitda_nd = d.get("ebitda", np.nan)
                if pd.notna(_debt_bs) and pd.notna(_ebitda_nd) and _ebitda_nd > 0:
                    _net_debt = _debt_bs - (_cash_bs if pd.notna(_cash_bs) else 0.0)
                    if _net_debt <= 0:
                        rec["net_debt_to_ebitda"] = 0.0  # Net cash position
                    else:
                        rec["net_debt_to_ebitda"] = _net_debt / _ebitda_nd
                elif pd.notna(_ebitda_nd) and _ebitda_nd <= 0:
                    rec["net_debt_to_ebitda"] = np.nan  # Negative EBITDA: ratio undefined
                else:
                    rec["net_debt_to_ebitda"] = np.nan
            else:
                rec["net_debt_to_ebitda"] = np.nan  # Banks: skip

            # 7c. Operating Leverage (Degree of Operating Leverage = DOL)
            # DOL = (%Δ EBIT) / (%Δ Revenue) using annual data (current vs prior year).
            # Lower DOL = less earnings sensitivity to revenue changes = more durable.
            # Guards: flat revenue (<1% change) → NaN, zero denominators → NaN.
            # Banks: skip (return NaN, weight redistributes).
            if not _is_bank:
                _ebit_curr = d.get("ebit", np.nan)
                _ebit_prev = d.get("ebit_prior", np.nan)
                if (pd.notna(_ebit_curr) and pd.notna(_ebit_prev) and _ebit_prev != 0
                        and pd.notna(rev_c) and pd.notna(rev_p) and rev_p > 0):
                    _rev_pct_change = (rev_c - rev_p) / abs(rev_p)
                    if abs(_rev_pct_change) < 0.01:  # Flat revenue: DOL undefined
                        rec["operating_leverage"] = np.nan
                    else:
                        _ebit_pct_change = (_ebit_curr - _ebit_prev) / abs(_ebit_prev)
                        rec["operating_leverage"] = _ebit_pct_change / _rev_pct_change
                else:
                    rec["operating_leverage"] = np.nan
            else:
                rec["operating_leverage"] = np.nan  # Banks: skip

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
            # Require >= 6 testable signals for a meaningful score (with
            # only 4-5 signals, the score range is too compressed to
            # discriminate quality reliably).
            rec["piotroski_f_score"] = f if n_testable >= 6 else np.nan

            # 9. Accruals
            rec["accruals"] = ((ni - ocfv) / ta) if (pd.notna(ni) and pd.notna(ocfv) and pd.notna(ta) and ta > 0) else np.nan

            # 10. Beneish M-Score (earnings manipulation detection)
            # Non-bank only; uses ANNUAL statements (t vs t-1), not LTM.
            # Banks excluded: no COGS, no PPE, Beneish assumptions break.
            if (cfg or {}).get("enable_beneish", True) and not _is_bank:
                _mscore, _mflag = _compute_beneish_mscore(d)
                rec["beneish_m_score"] = _mscore
                rec["_beneish_flag"] = _mflag
            else:
                rec["beneish_m_score"] = np.nan
                rec["_beneish_flag"] = False
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: quality metrics failed: {type(e).__name__}: {e}")

        # -- Receivables-to-revenue growth divergence (channel-stuffing flag) --
        # If receivables are growing significantly faster than revenue, it may
        # indicate aggressive revenue recognition or channel stuffing.
        try:
            _recv_t = d.get("_beneish_net_receivables", np.nan)
            _recv_p = d.get("_beneish_net_receivables_p", np.nan)
            _rev_t = d.get("totalRevenue", np.nan)
            _rev_p = d.get("totalRevenue_prior", np.nan)
            if (pd.notna(_recv_t) and pd.notna(_recv_p) and _recv_p > 0
                    and pd.notna(_rev_t) and pd.notna(_rev_p) and _rev_p > 0):
                _recv_growth = (_recv_t / _recv_p) - 1
                _rev_growth = (_rev_t / _rev_p) - 1
                _divergence = _recv_growth - _rev_growth
                rec["_recv_rev_divergence"] = _divergence
                # Flag if receivables growth exceeds revenue growth by >15pp
                rec["_channel_stuffing_flag"] = _divergence > 0.15
            else:
                rec["_recv_rev_divergence"] = np.nan
                rec["_channel_stuffing_flag"] = False
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            rec["_recv_rev_divergence"] = np.nan
            rec["_channel_stuffing_flag"] = False

        # -- Growth metrics (10-12) --
        try:
            # 10. Forward EPS Growth
            # Floor denominator at $1.00 to prevent near-zero trailing
            # EPS from producing extreme growth ratios (same principle
            # as the analyst_surprise $0.10 floor).  Clamp to configured
            # bounds (default [-75%, +300%]) because yfinance mixes GAAP
            # trailing EPS with normalised forward consensus.
            fwd = d.get("forwardEps", np.nan)
            trail = d.get("trailingEps", np.nan)
            if pd.notna(fwd) and pd.notna(trail) and abs(trail) > 0.01:
                _raw_growth = (fwd - trail) / max(abs(trail), 1.0)
                rec["forward_eps_growth"] = float(np.clip(_raw_growth, feg_lo, feg_hi))
            else:
                rec["forward_eps_growth"] = np.nan

            # PEG Ratio = (P/E) / (Forward EPS Growth Rate %)
            # Uses the already-computed forward EPS growth instead of the
            # undocumented yfinance 'earningsGrowth' field, which is a
            # black-box input with no verifiable definition.
            # NaN when growth <= 0 or P/E <= 0: negative/zero growth makes
            # PEG meaningless (not a growth stock). NaN lets the per-row
            # weight redistribution handle it rather than injecting a false signal.
            _price = d.get("currentPrice", d.get("price_latest", np.nan))
            _pe = (_price / trail) if (pd.notna(_price) and pd.notna(trail) and trail > 0.01) else np.nan
            _fwd_growth = rec.get("forward_eps_growth", np.nan)
            if pd.notna(_pe) and pd.notna(_fwd_growth) and _fwd_growth > 0:
                rec["peg_ratio"] = min(_pe / (_fwd_growth * 100), peg_max_cap)
            else:
                rec["peg_ratio"] = np.nan

            # 11. Revenue Growth (1-year)
            rec["revenue_growth"] = ((rev_c - rev_p) / rev_p) if (pd.notna(rev_c) and pd.notna(rev_p) and rev_p > 0) else np.nan

            # 11b. 3-Year Revenue CAGR — smoothed growth signal from annual filings.
            # Uses LTM revenue (current) vs annual col=3 (3 years prior).
            _rev_3yr = d.get("totalRevenue_3yr_ago", np.nan)
            if pd.notna(rev_c) and pd.notna(_rev_3yr) and _rev_3yr > 0:
                rec["revenue_cagr_3yr"] = (rev_c / _rev_3yr) ** (1.0 / 3.0) - 1.0
            else:
                rec["revenue_cagr_3yr"] = np.nan

            # 12. Sustainable Growth = ROE * Retention Ratio
            # ROE uses average equity (current + prior / 2) to smooth
            # single-year distortions from buybacks or one-time items.
            # Payout ratio prefers .info payoutRatio (reliable), then
            # falls back to dividendsPaid from cashflow.
            # SGR clamped to [0%, 100%].
            if pd.notna(ni) and pd.notna(eq_v) and eq_v > 0 and ni > 0:
                eq_prior = d.get("totalEquity_prior", np.nan)
                if pd.notna(eq_prior) and eq_prior > 0:
                    avg_eq = (eq_v + eq_prior) / 2
                else:
                    avg_eq = eq_v  # fall back to single-year
                roe = ni / avg_eq

                # Payout ratio: prefer .info payoutRatio, then cashflow
                _payout = d.get("payoutRatio", np.nan)
                if pd.notna(_payout) and 0 <= _payout <= 2.0:
                    ret = max(0, 1 - min(_payout, 1.0))
                else:
                    _divs_raw = d.get("dividendsPaid", np.nan)
                    if pd.isna(_divs_raw):
                        _div_rate = d.get("dividendRate", np.nan)
                        _shares = d.get("sharesOutstanding", np.nan)
                        if pd.notna(_div_rate) and pd.notna(_shares):
                            divs = abs(_div_rate * _shares)
                        else:
                            divs = np.nan
                    else:
                        divs = abs(_divs_raw)
                    if pd.isna(divs):
                        rec["sustainable_growth"] = np.nan
                        roe = np.nan  # signal to skip final assignment
                    else:
                        ret = max(0, 1 - divs / ni) if ni > 0 else 0

                if pd.notna(roe):
                    sgr = roe * ret
                    rec["sustainable_growth"] = float(np.clip(sgr, 0.0, 1.0))
            else:
                rec["sustainable_growth"] = np.nan
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: growth metrics failed: {type(e).__name__}: {e}")

        # -- GAAP/Normalized EPS mismatch flag --
        # EPS basis mismatch flag — relevant to forward_eps_growth metric.
        # Note: PEG ratio has been removed from scoring as of this update,
        # so this flag's primary impact is now on forward_eps_growth and
        # the Growth category score only.
        # Flag when the ratio is extreme (>2x or <0.3x), which suggests
        # large non-recurring items distorting the growth metric.
        try:
            _trail = d.get("trailingEps", np.nan)
            _fwd = d.get("forwardEps", np.nan)
            if pd.notna(_trail) and pd.notna(_fwd) and abs(_trail) > 0.10:
                _eps_ratio = _fwd / _trail
                if _eps_ratio > 2.0 or _eps_ratio < 0.3:
                    rec["_eps_basis_mismatch"] = True
                    rec["_eps_ratio"] = round(_eps_ratio, 2)
                else:
                    rec["_eps_basis_mismatch"] = False
            else:
                rec["_eps_basis_mismatch"] = False
        except (TypeError, ValueError, ZeroDivisionError):
            rec["_eps_basis_mismatch"] = False

        # -- Momentum metrics (13-14) --
        try:
            # 13. 12-1 Month Return (skip-month per Jegadeesh-Titman)
            p12 = d.get("price_12m_ago", np.nan)
            p1m = d.get("price_1m_ago", np.nan)
            rec["return_12_1"] = ((p1m - p12) / p12) if (pd.notna(p12) and pd.notna(p1m) and p12 > 0) else np.nan

            # 14. 6-1 Month Return (exclude most recent month to match 12-1M convention)
            p6m = d.get("price_6m_ago", np.nan)
            rec["return_6m"] = ((p1m - p6m) / p6m) if (pd.notna(p6m) and pd.notna(p1m) and p6m > 0) else np.nan

            # 14b. Full 12-month return (no skip-month) — used for Sharpe
            # Ratio and Jensen's Alpha, which measure realized return, not
            # the momentum signal.  The skip-month convention is appropriate
            # for momentum ranking but distorts risk-adjusted return metrics.
            _p_now = d.get("price_latest", d.get("currentPrice", np.nan))
            rec["return_12m"] = ((_p_now - p12) / p12) if (pd.notna(p12) and pd.notna(_p_now) and p12 > 0) else np.nan
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: momentum metrics failed: {type(e).__name__}: {e}")

        # -- Risk metrics (15-16) --
        try:
            # 15. Volatility
            rec["volatility"] = d.get("volatility_1y", np.nan)

            # 16. Beta (date-aligned with overlap validation)
            dr = d.get("_daily_returns")
            if dr and isinstance(dr, dict) and len(market_returns) >= 200:
                mr_dates = {dt.strftime("%Y-%m-%d"): v
                            for dt, v in zip(market_returns.index, market_returns.values)}
                common = sorted(set(dr.keys()) & set(mr_dates.keys()))
                _overlap_ratio = len(common) / len(mr_dates) if mr_dates else 0
                rec["_beta_overlap_pct"] = round(_overlap_ratio * 100, 1)
                if len(common) >= 200 and _overlap_ratio >= 0.80:
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

            # 16b. Sharpe Ratio (annualized, trailing 12 months)
            # Sharpe = (R_i - R_f) / sigma_i
            # Uses return_12m (full 12-month, no skip) — the skip-month
            # convention is for momentum ranking, not risk-adjusted returns.
            _vol_sr = rec.get("volatility", float("nan"))
            _ret_12m_sr = rec.get("return_12m", float("nan"))
            if (pd.notna(_vol_sr) and _vol_sr > 0 and pd.notna(_ret_12m_sr)):
                rec["sharpe_ratio"] = (_ret_12m_sr - risk_free_rate) / _vol_sr
            else:
                rec["sharpe_ratio"] = np.nan

            # 16c. Sortino Ratio (annualized, trailing 12 months)
            # Sortino = (R_i - R_f) / downside_deviation
            # Downside deviation = std dev of daily returns below the daily
            # risk-free rate, annualized by sqrt(252).
            if dr and isinstance(dr, dict) and pd.notna(_ret_12m_sr):
                _daily_rf = (1 + risk_free_rate) ** (1/252) - 1
                _daily_vals = np.array(list(dr.values()))
                _downside = _daily_vals[_daily_vals < _daily_rf] - _daily_rf
                if len(_downside) >= 20:
                    _dd = np.std(_downside, ddof=1) * np.sqrt(252)
                    rec["sortino_ratio"] = ((_ret_12m_sr - risk_free_rate) / _dd
                                            if _dd > 0 else np.nan)
                else:
                    rec["sortino_ratio"] = np.nan
            elif dr and isinstance(dr, list) and pd.notna(_ret_12m_sr):
                _daily_rf = (1 + risk_free_rate) ** (1/252) - 1
                _daily_vals = np.array(dr)
                _downside = _daily_vals[_daily_vals < _daily_rf] - _daily_rf
                if len(_downside) >= 20:
                    _dd = np.std(_downside, ddof=1) * np.sqrt(252)
                    rec["sortino_ratio"] = ((_ret_12m_sr - risk_free_rate) / _dd
                                            if _dd > 0 else np.nan)
                else:
                    rec["sortino_ratio"] = np.nan
            else:
                rec["sortino_ratio"] = np.nan

            # 16d. Max Drawdown (trailing 12 months)
            # Peak-to-trough decline from cumulative return series.
            # Expressed as a negative fraction (e.g., -0.25 = 25% drawdown).
            if dr and isinstance(dr, dict) and len(dr) >= 50:
                _dd_vals = np.array(list(dr.values()))
                _cum = np.cumprod(1 + _dd_vals)
                _peak = np.maximum.accumulate(_cum)
                _drawdowns = (_cum - _peak) / _peak
                rec["max_drawdown_1y"] = float(np.min(_drawdowns))
            elif dr and isinstance(dr, list) and len(dr) >= 50:
                _dd_vals = np.array(dr)
                _cum = np.cumprod(1 + _dd_vals)
                _peak = np.maximum.accumulate(_cum)
                _drawdowns = (_cum - _peak) / _peak
                rec["max_drawdown_1y"] = float(np.min(_drawdowns))
            else:
                rec["max_drawdown_1y"] = np.nan
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: risk metrics failed: {type(e).__name__}: {e}")

        # -- Jensen's Alpha (requires beta + return_12m from sections above) --
        # Read values safely from rec dict (not local variables that may be
        # out of scope if momentum or risk try blocks raised caught exceptions).
        # Uses return_12m (full 12-month, no skip) for the CAPM realized return.
        try:
            _beta_ja = rec.get("beta", float("nan"))
            _ret_12m_ja = rec.get("return_12m", float("nan"))
            if (pd.notna(_beta_ja) and pd.notna(_ret_12m_ja)
                    and pd.notna(market_12m_return)):
                expected_return = risk_free_rate + _beta_ja * (market_12m_return - risk_free_rate)
                rec["jensens_alpha"] = _ret_12m_ja - expected_return
            else:
                rec["jensens_alpha"] = np.nan
                logging.debug(f"{ticker}: jensens_alpha skipped — beta or return NaN")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: Jensen's alpha failed: {type(e).__name__}: {e}")
            rec["jensens_alpha"] = np.nan

        # -- Revisions --
        # NOTE: An EPS forecast revision metric (change in consensus forward EPS
        # over 3-6 months) would strengthen this category, but yfinance only
        # provides static forwardEps — no historical consensus data.
        # Future enhancement: integrate I/B/E/S data from FactSet or Refinitiv.
        try:
            rec["analyst_surprise"] = d.get("analyst_surprise", np.nan)
            rec["earnings_acceleration"] = d.get("earnings_acceleration", np.nan)
            rec["consecutive_beat_streak"] = d.get("consecutive_beat_streak", np.nan)

            # Price target upside: consensus analyst target vs current price.
            # Require >= 3 covering analysts for a meaningful consensus.
            # Clamped to [-50%, +100%] to guard against extreme targets.
            _target = d.get("targetMeanPrice", np.nan)
            _cur_price = d.get("currentPrice", d.get("price_latest", np.nan))
            _n_analysts = d.get("numberOfAnalystOpinions", np.nan)
            if (pd.notna(_target) and pd.notna(_cur_price) and _cur_price > 0
                    and pd.notna(_n_analysts) and _n_analysts >= 3):
                rec["price_target_upside"] = float(np.clip(
                    (_target - _cur_price) / _cur_price, ptu_lo, ptu_hi))
            else:
                rec["price_target_upside"] = np.nan

            # Short Interest Ratio (days to cover = shares short / avg daily volume).
            # Lower = less bearish sentiment = better. Yahoo updates bi-monthly
            # with a 1-2 week lag; this delay is inherent and acceptable.
            _short_ratio = d.get("shortRatio", np.nan)
            rec["short_interest_ratio"] = (
                _short_ratio if (pd.notna(_short_ratio) and _short_ratio >= 0) else np.nan
            )
        except (KeyError, TypeError, ValueError) as e:
            warnings.warn(f"{ticker}: revisions metrics failed: {type(e).__name__}: {e}")

        # -- Passthrough: analyst price targets for dashboard display --
        rec["_current_price"]     = d.get("currentPrice", d.get("price_latest", np.nan))
        rec["_target_mean"]       = d.get("targetMeanPrice", np.nan)
        rec["_target_high"]       = d.get("targetHighPrice", np.nan)
        rec["_target_low"]        = d.get("targetLowPrice", np.nan)
        rec["_num_analysts"]      = d.get("numberOfAnalystOpinions", np.nan)

        # -- Bank-specific metrics (conditional on sector) --
        try:
            # _is_bank pre-computed near top of loop (needed by Beneish check)
            rec["_is_bank_like"] = _is_bank

            if _is_bank:
                # Bank Valuation: Price-to-Book
                ptb = d.get("priceToBook", np.nan)
                if pd.notna(ptb) and ptb > 0:
                    rec["pb_ratio"] = ptb
                else:
                    bv = d.get("bookValue", np.nan)
                    price = d.get("currentPrice", d.get("price_latest", np.nan))
                    rec["pb_ratio"] = (price / bv) if (pd.notna(price) and pd.notna(bv) and bv > 0) else np.nan

                # Bank Quality: ROE
                roe_info = d.get("returnOnEquity", np.nan)
                if pd.notna(roe_info):
                    rec["roe"] = roe_info
                elif pd.notna(ni) and pd.notna(eq_v) and eq_v > 0:
                    rec["roe"] = ni / eq_v
                else:
                    rec["roe"] = np.nan

                # Bank Quality: ROA
                roa_info = d.get("returnOnAssets", np.nan)
                if pd.notna(roa_info):
                    rec["roa"] = roa_info
                elif pd.notna(ni) and pd.notna(ta) and ta > 0:
                    rec["roa"] = ni / ta
                else:
                    rec["roa"] = np.nan

                # Bank Quality: Equity Ratio
                if pd.notna(eq_v) and pd.notna(ta) and ta > 0:
                    rec["equity_ratio"] = eq_v / ta
                else:
                    rec["equity_ratio"] = np.nan

                # Null out meaningless generic metrics for banks
                rec["ev_ebitda"] = np.nan
                rec["ev_sales"] = np.nan
                rec["roic"] = np.nan
                rec["gross_profit_assets"] = np.nan
                rec["debt_equity"] = np.nan
            else:
                # Non-bank: null out bank metrics
                rec["pb_ratio"] = np.nan
                rec["roe"] = np.nan
                rec["roa"] = np.nan
                rec["equity_ratio"] = np.nan
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: bank metrics failed: {type(e).__name__}: {e}")

        # -- Data freshness check --
        try:
            stale_days = 200  # Flag if most recent filing is > 200 days old
            stmt_date_str = d.get("_stmt_date_financials")
            if stmt_date_str:
                stmt_date = pd.Timestamp(stmt_date_str)
                age_days = (pd.Timestamp.now() - stmt_date).days
                rec["_stmt_age_days"] = age_days
                if age_days > stale_days:
                    rec["_stale_data"] = True
                    warnings.warn(f"{ticker}: financial data is {age_days} days old (>{stale_days}d)")
        except (KeyError, TypeError, ValueError) as e:
            warnings.warn(f"{ticker}: data freshness check failed: {type(e).__name__}: {e}")

        # -- Size metric --
        rec["size_log_mcap"] = -np.log(mc) if (pd.notna(mc) and mc > 0) else np.nan

        # -- Investment metric (asset growth, Fama-French CMA proxy) --
        # Uses MRQ balance sheet data (quarterly_balance_sheet col=0 vs col=4).
        # Falls back to annual balance sheet if quarterly unavailable.
        _ta_curr = d.get("totalAssets", np.nan)
        _ta_prior = d.get("totalAssets_prior", np.nan)
        rec["asset_growth"] = ((_ta_curr - _ta_prior) / _ta_prior
                               if (pd.notna(_ta_curr) and pd.notna(_ta_prior) and _ta_prior > 0)
                               else np.nan)

        # -- Liquidity passthrough (for portfolio filter, not scored) --
        rec["avg_daily_dollar_volume"] = d.get("avg_daily_dollar_volume", np.nan)

        # -- Data provenance summary --
        rec["_data_source"] = d.get("_data_source", "unknown")
        rec["_ltm_annualized"] = d.get("_ltm_annualized", False)
        _metric_keys = [
            "ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales",
            "roic", "gross_profit_assets", "debt_equity", "piotroski_f_score",
            "accruals", "forward_eps_growth", "revenue_growth", "sustainable_growth",
            "return_12_1", "return_6m", "volatility", "beta",
            "analyst_surprise", "price_target_upside",
        ]
        _n_present = sum(1 for k in _metric_keys if pd.notna(rec.get(k)))
        rec["_metric_count"] = _n_present
        rec["_metric_total"] = len(_metric_keys)

        records.append(rec)

    return pd.DataFrame(records)


# =========================================================================
# F. Winsorize raw metrics (SS4.7)
# =========================================================================
METRIC_COLS = [
    "ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales",
    "pb_ratio",                                                      # bank valuation
    "roic", "gross_profit_assets", "debt_equity", "net_debt_to_ebitda",
    "piotroski_f_score", "accruals", "operating_leverage",
    "beneish_m_score",                                               # earnings manipulation (non-bank)
    "roe", "roa", "equity_ratio",                                    # bank quality
    "forward_eps_growth", "peg_ratio", "revenue_growth", "revenue_cagr_3yr", "sustainable_growth",
    "return_12_1", "return_6m", "jensens_alpha",                      # momentum + risk-adjusted alpha
    "volatility", "beta", "sharpe_ratio", "sortino_ratio",              # risk + risk-adjusted return
    "max_drawdown_1y",                                                   # tail risk: max peak-to-trough
    "analyst_surprise", "price_target_upside",
    "earnings_acceleration", "consecutive_beat_streak",              # fundamental momentum
    "short_interest_ratio",                                          # short interest sentiment
    "size_log_mcap",                                                 # size factor
    "asset_growth",                                                  # investment (CMA proxy)
]

# Metrics that only apply to bank-like or non-bank stocks.
# Used by the coverage filter to avoid penalizing stocks for
# structurally absent metrics.
_BANK_ONLY_METRICS = {"pb_ratio", "roe", "roa", "equity_ratio"}
_NONBANK_ONLY_METRICS = {"ev_ebitda", "ev_sales", "roic", "gross_profit_assets",
                         "net_debt_to_ebitda", "operating_leverage", "beneish_m_score"}


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
    "pb_ratio": False,                                    # lower P/B = cheaper = better
    "roic": True, "gross_profit_assets": True, "debt_equity": False,
    "net_debt_to_ebitda": False,                         # lower net debt/EBITDA = less leveraged = better
    "piotroski_f_score": True, "accruals": False,
    "operating_leverage": False,                         # lower DOL = less earnings sensitivity = better
    "beneish_m_score": False,                             # lower M-Score = less manipulation risk = better
    "roe": True, "roa": True, "equity_ratio": True,      # bank quality: higher = better
    "forward_eps_growth": True, "peg_ratio": False, "revenue_growth": True,
    "revenue_cagr_3yr": True,                                        # higher revenue CAGR = better
    "sustainable_growth": True,
    "return_12_1": True, "return_6m": True,
    "jensens_alpha": True,                                   # higher alpha = more excess return above CAPM = better
    "volatility": False, "beta": False,
    "sharpe_ratio": True,                                    # higher Sharpe = better risk-adjusted return
    "sortino_ratio": True,                                   # higher Sortino = better downside-adjusted return
    "max_drawdown_1y": True,                                 # less negative = smaller drawdown = better
    "analyst_surprise": True, "price_target_upside": True,
    "earnings_acceleration": True, "consecutive_beat_streak": True,  # fundamental momentum
    "short_interest_ratio": False,                                   # lower days-to-cover = less short pressure = better
    "size_log_mcap": True,                                # -log(mcap): higher = smaller = size premium
    "asset_growth": False,                                # lower asset growth = conservative investment = better
}


def compute_sector_percentiles(df: pd.DataFrame):
    pct = {c: f"{c}_pct" for c in METRIC_COLS}
    for c in pct.values():
        df[c] = np.nan

    # Pre-compute universe-wide ranks for small-sector fallback.
    # When a sector has < 10 valid values for a metric, we fall back
    # to universe-wide percentile ranking instead of assigning a flat
    # 50th percentile (which penalizes good stocks and rewards bad
    # ones in small sectors).
    universe_ranks = {}
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        ranks = df[col].rank(pct=True, na_option="keep") * 100
        if not METRIC_DIR.get(col, True):
            ranks = 100 - ranks
        universe_ranks[col] = ranks

    for _, grp in df.groupby("Sector"):
        for col in METRIC_COLS:
            pc = pct[col]
            if col not in df.columns:
                df.loc[grp.index, pc] = 50.0
                continue
            valid = grp[col].dropna()
            if len(valid) < 10:
                # Fall back to universe-wide ranking for this metric
                if col in universe_ranks:
                    df.loc[grp.index, pc] = universe_ranks[col].loc[grp.index]
                else:
                    df.loc[grp.index, pc] = 50.0
                continue
            ranks = grp[col].rank(pct=True, na_option="keep") * 100
            if not METRIC_DIR.get(col, True):
                ranks = 100 - ranks
            # NaN raw values → NaN percentile (not imputed to 50th).
            # The category score function handles per-row weight
            # redistribution for missing metrics.
            df.loc[grp.index, pc] = ranks
    return df


# =========================================================================
# G½. Optional non-linear percentile transform
# =========================================================================
def apply_percentile_transform(df: pd.DataFrame, cfg: dict):
    """Apply optional non-linear transform to percentile-ranked columns.

    If enabled, applies a logistic S-curve that compresses the middle ranks
    and stretches the extremes, rewarding truly exceptional scores.
    Default: disabled (identity transform preserves current behavior).

    Config keys (under 'percentile_transform'):
        enabled: bool (default False)
        method: "identity" | "logistic" (default "logistic")
        logistic_steepness: float (default 0.08) — higher = sharper S-curve
    """
    pt_cfg = cfg.get("percentile_transform", {})
    if not pt_cfg.get("enabled", False):
        return df

    method = pt_cfg.get("method", "logistic")
    if method == "identity":
        return df

    if method == "logistic":
        k = pt_cfg.get("logistic_steepness", 0.08)
        pct_cols = [c for c in df.columns if c.endswith("_pct")]
        for col in pct_cols:
            mask = df[col].notna()
            if mask.any():
                raw = df.loc[mask, col]
                # Logistic: 100 / (1 + exp(-k * (pct - 50)))
                transformed = 100.0 / (1.0 + np.exp(-k * (raw - 50.0)))
                df.loc[mask, col] = transformed
        return df

    # Unknown method — leave unchanged
    return df


# =========================================================================
# H. Within-category scores (SS3.1)
# =========================================================================
CAT_METRICS = {
    "valuation": ["ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales", "pb_ratio"],
    "quality":   ["roic", "gross_profit_assets", "net_debt_to_ebitda",
                  "piotroski_f_score", "accruals", "operating_leverage",
                  "beneish_m_score", "roe", "roa", "equity_ratio"],
    "growth":    ["forward_eps_growth", "peg_ratio", "revenue_growth", "revenue_cagr_3yr", "sustainable_growth"],
    "momentum":  ["return_12_1", "return_6m", "jensens_alpha"],
    "risk":      ["volatility", "beta", "sharpe_ratio", "sortino_ratio", "max_drawdown_1y"],
    "revisions": ["analyst_surprise", "price_target_upside", "earnings_acceleration", "consecutive_beat_streak", "short_interest_ratio"],
    "size":       ["size_log_mcap"],
    "investment": ["asset_growth"],
}


def compute_category_scores(df: pd.DataFrame, cfg: dict):
    mw = cfg["metric_weights"]
    bank_mw = cfg.get("bank_metric_weights", None)
    is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False)

    # Piotroski conditional weighting config
    pio_cfg = cfg.get("piotroski_conditional", {})
    pio_enabled = pio_cfg.get("enabled", False)
    pio_val_threshold = pio_cfg.get("valuation_threshold", 50)
    pio_reduction = pio_cfg.get("reduction_factor", 0.5)
    pio_redistribute_to = set(pio_cfg.get("redistribute_to", ["roic", "gross_profit_assets"]))

    # Growth-trap Piotroski conditional config (Extension of above)
    pio_gt_enabled = pio_enabled and pio_cfg.get("growth_trap_enabled", False)
    pio_gt_growth_thr = pio_cfg.get("growth_trap_growth_threshold", 70)
    pio_gt_quality_thr = pio_cfg.get("growth_trap_quality_threshold", 35)
    pio_gt_redistribute_to = set(pio_cfg.get("growth_trap_redistribute_to", ["accruals", "gross_profit_assets"]))

    for cat, metrics in CAT_METRICS.items():
        generic_ws = mw.get(cat, {})
        bank_ws = bank_mw.get(cat, generic_ws) if bank_mw else generic_ws
        col = f"{cat}_score"

        # Pre-compute Piotroski conditional adjustment for quality category
        # Only applies to non-bank stocks with low valuation scores
        pio_adjust = False
        is_low_val = pd.Series(False, index=df.index)
        # Growth-trap-like Piotroski adjustment (high growth + low quality, non-bank)
        pio_gt_adjust = False
        is_growth_trap_like = pd.Series(False, index=df.index)
        if (cat == "quality" and pio_enabled
                and "valuation_score" in df.columns):
            is_low_val = (df["valuation_score"] < pio_val_threshold).fillna(False)
            # Only adjust non-bank rows (bank quality weights don't use ROIC/GPA)
            is_low_val = is_low_val & ~is_bank
            if is_low_val.any():
                pio_adjust = True
                # Compute freed weight from piotroski reduction
                pio_generic_w = generic_ws.get("piotroski_f_score", 0) / 100.0
                freed_w = pio_generic_w * (1 - pio_reduction)
                # Proportional redistribution: split freed weight in proportion to
                # base weights of recipient metrics (not equal split).
                pio_redist_weights = {m: generic_ws.get(m, 0) for m in pio_redistribute_to
                                      if generic_ws.get(m, 0) > 0}
                pio_redist_total = sum(pio_redist_weights.values())
                pio_redist_shares = ({m: w / pio_redist_total
                                      for m, w in pio_redist_weights.items()}
                                     if pio_redist_total > 0 else {})

            # Growth-trap Piotroski: high-growth + low-quality non-bank stocks
            # get Piotroski weight halved, freed weight → accruals + gross_profit_assets
            if (pio_gt_enabled
                    and "growth_score" in df.columns and "quality_score" in df.columns):
                g_thr = df["growth_score"].quantile(pio_gt_growth_thr / 100.0)
                q_thr = df["quality_score"].quantile(pio_gt_quality_thr / 100.0)
                is_growth_trap_like = (
                    (df["growth_score"] >= g_thr)
                    & (df["quality_score"] <= q_thr)
                    & ~is_bank
                    & ~is_low_val  # Don't double-apply; growth trap takes precedence on redistribution targets
                )
                if is_growth_trap_like.any():
                    pio_gt_adjust = True
                    gt_pio_w = generic_ws.get("piotroski_f_score", 0) / 100.0
                    gt_freed_w = gt_pio_w * (1 - pio_reduction)
                    # Proportional redistribution for growth-trap variant
                    gt_redist_weights = {m: generic_ws.get(m, 0) for m in pio_gt_redistribute_to
                                         if generic_ws.get(m, 0) > 0}
                    gt_redist_total = sum(gt_redist_weights.values())
                    gt_redist_shares = ({m: w / gt_redist_total
                                         for m, w in gt_redist_weights.items()}
                                        if gt_redist_total > 0 else {})

        # Per-row weighted average: only count metrics that have data.
        # NaN percentiles are excluded (not imputed to 50th), and each
        # row's score uses its own effective weight denominator.
        # Bank-like stocks use bank_metric_weights; others use generic.
        weighted_sum = pd.Series(0.0, index=df.index)
        weight_sum = pd.Series(0.0, index=df.index)
        skipped_metrics = []
        for m in metrics:
            pc = f"{m}_pct"
            w_generic = generic_ws.get(m, 0) / 100.0
            w_bank = bank_ws.get(m, 0) / 100.0
            # Per-row weight: bank weight for bank stocks, generic for others
            w = pd.Series(w_generic, index=df.index)
            w[is_bank] = w_bank

            # Piotroski conditional: reduce piotroski weight for low-val non-bank stocks,
            # redistribute freed weight proportionally to specified recipient metrics
            if pio_adjust:
                if m == "piotroski_f_score":
                    w[is_low_val] = w_generic * pio_reduction
                elif m in pio_redist_shares:
                    w[is_low_val] = w_generic + freed_w * pio_redist_shares[m]

            # Growth-trap Piotroski conditional: same reduction, proportional redistribution
            if pio_gt_adjust:
                if m == "piotroski_f_score":
                    w[is_growth_trap_like] = w_generic * pio_reduction
                elif m in gt_redist_shares:
                    w[is_growth_trap_like] = w_generic + gt_freed_w * gt_redist_shares[m]

            if pc not in df.columns:
                continue
            # Skip metrics that are entirely NaN (unavailable data source)
            if m in df.columns and df[m].isna().all():
                skipped_metrics.append(m)
                continue
            has_data = df[pc].notna()
            weighted_sum += df[pc].fillna(0) * w * has_data.astype(float)
            weight_sum += w * has_data.astype(float)
        if skipped_metrics:
            print(f"  [{cat}] Skipped unavailable metrics: {skipped_metrics}")
        # Where weight_sum > 0, compute weighted average; else NaN
        df[col] = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
    return df


# =========================================================================
# H½. Volatility-scaled momentum weight (adaptive regime)
# =========================================================================
def adjust_momentum_weight(df: pd.DataFrame, cfg: dict, root_dir: str):
    """Adjust momentum factor weight based on realized momentum-score volatility.

    Compares current run's momentum_score dispersion to historical runs.
    High-vol regime → reduce momentum weight (whipsaw risk), redistribute
    to quality + valuation.  Low-vol regime → increase momentum weight,
    funded by valuation reduction.

    Returns a (potentially modified) deep-copy of cfg.
    """
    import copy, csv, os
    from datetime import date

    if "momentum_score" not in df.columns or df["momentum_score"].dropna().empty:
        return cfg

    current_vol = float(df["momentum_score"].dropna().std())
    hist_path = os.path.join(root_dir, "factor_vol_history.csv")

    # Load historical vol data
    hist_vols = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        hist_vols.append(float(row["momentum_vol"]))
                    except (ValueError, KeyError):
                        pass
        except Exception:
            pass

    # Append current run
    write_header = not os.path.exists(hist_path) or os.path.getsize(hist_path) == 0
    try:
        with open(hist_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["date", "momentum_vol"])
            writer.writerow([date.today().isoformat(), f"{current_vol:.4f}"])
    except Exception:
        pass

    # Need >= 20 historical observations to establish regime thresholds
    if len(hist_vols) < 20:
        print(f"  [MOM-VOL] Current vol={current_vol:.2f}, history={len(hist_vols)} runs (need 20+). Skipping regime scaling.")
        return cfg

    p25 = float(np.percentile(hist_vols, 25))
    p75 = float(np.percentile(hist_vols, 75))

    cfg = copy.deepcopy(cfg)
    fw = cfg["factor_weights"]
    mom_w = fw.get("momentum", 0)

    if current_vol > p75:
        # HIGH VOL regime: reduce momentum, boost quality + valuation
        scale = 0.70
        freed = mom_w * (1 - scale)
        fw["momentum"] = round(mom_w * scale, 2)
        fw["quality"] = round(fw.get("quality", 0) + freed / 2, 2)
        fw["valuation"] = round(fw.get("valuation", 0) + freed / 2, 2)
        regime = "HIGH VOL"
    elif current_vol < p25:
        # LOW VOL regime: increase momentum, reduce valuation
        scale = 1.15
        added = mom_w * (scale - 1)
        fw["momentum"] = round(mom_w * scale, 2)
        fw["valuation"] = round(fw.get("valuation", 0) - added, 2)
        regime = "LOW VOL"
    else:
        regime = "NORMAL"

    print(f"  [MOM-VOL] vol={current_vol:.2f} | p25={p25:.2f} p75={p75:.2f} | Regime: {regime}")
    if regime != "NORMAL":
        print(f"  [MOM-VOL] Adjusted weights: momentum={fw['momentum']}, quality={fw.get('quality')}, valuation={fw.get('valuation')}")
    return cfg


# =========================================================================
# I. Composite score (SS3.2)
# =========================================================================
def compute_composite(df: pd.DataFrame, cfg: dict):
    if df.empty:
        df["Composite"] = pd.Series(dtype=float)
        return df
    fw = cfg["factor_weights"]
    col_map = {
        "valuation": "valuation_score", "quality": "quality_score",
        "growth": "growth_score", "momentum": "momentum_score",
        "risk": "risk_score", "revisions": "revisions_score",
        "size": "size_score", "investment": "investment_score",
    }
    # Per-row weighted average: only count categories that have data.
    # NaN category scores are excluded and their weight is redistributed
    # to available categories (mirrors compute_category_scores() logic).
    # Without this, a single NaN category propagates to NaN composite
    # and the stock silently vanishes from the ranking.
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for cat, col in col_map.items():
        w = fw.get(cat, 0)
        if col not in df.columns or w == 0:
            continue
        has_data = df[col].notna()
        weighted_sum += df[col].fillna(0) * w * has_data.astype(float)
        weight_sum += w * has_data.astype(float)
    df["Composite"] = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

    # Coverage discount: mildly penalize stocks with many missing metrics.
    # Stocks with >=threshold coverage get no penalty; below that, the
    # composite is reduced proportionally to the gap.
    cov_cfg = cfg.get("data_quality", {}).get("coverage_discount", {})
    if cov_cfg.get("enabled", False):
        threshold = cov_cfg.get("threshold", 0.80)
        penalty_rate = cov_cfg.get("penalty_rate", 0.15)
        is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False).astype(bool)

        # Count applicable metrics per stock (bank vs non-bank differ)
        all_metrics = [c for c in METRIC_COLS if c in df.columns]
        for idx in df.index:
            row_bank = is_bank.loc[idx] if idx in is_bank.index else False
            if row_bank:
                applicable = [m for m in all_metrics if m not in _NONBANK_ONLY_METRICS]
            else:
                applicable = [m for m in all_metrics if m not in _BANK_ONLY_METRICS]
            n_applicable = len(applicable)
            if n_applicable == 0:
                continue
            n_present = sum(1 for m in applicable if pd.notna(df.at[idx, m]))
            coverage = n_present / n_applicable
            if coverage < threshold:
                discount = (threshold - coverage) * penalty_rate
                df.at[idx, "Composite"] = df.at[idx, "Composite"] * (1 - discount)

    sector_relative = cfg.get("sector_neutral", {}).get("sector_relative_composite", False)
    if sector_relative and "Sector" in df.columns:
        for _, grp in df.groupby("Sector"):
            if len(grp) >= 3:
                df.loc[grp.index, "Composite"] = grp["Composite"].rank(pct=True) * 100
            else:
                df.loc[grp.index, "Composite"] = 50.0
    else:
        # Cross-sectional percentile rank: rank 1 does NOT automatically
        # get 100.0. A stock's score reflects its position relative to
        # the full universe (e.g., 98.5 means "better than 98.5% of
        # stocks"), which is meaningful across runs and time periods.
        df["Composite"] = df["Composite"].rank(pct=True) * 100

    df["Composite"] = df["Composite"].round(2)
    return df


# =========================================================================
# I-b. Factor contribution waterfall
# =========================================================================
def compute_factor_contributions(df: pd.DataFrame, cfg: dict):
    """Compute each factor category's contribution to the composite score.

    For each stock, the contribution of category C is:
        contrib_C = category_score_C * weight_C / sum_of_available_weights

    This shows how many "points" each category adds to the weighted-average
    composite (before the final percentile-rank transform). The contributions
    sum to the pre-rank composite for each stock.
    """
    if df.empty:
        return df

    fw = cfg["factor_weights"]
    col_map = {
        "valuation": "valuation_score", "quality": "quality_score",
        "growth": "growth_score", "momentum": "momentum_score",
        "risk": "risk_score", "revisions": "revisions_score",
        "size": "size_score", "investment": "investment_score",
    }

    # Compute weight sums per row (same logic as compute_composite)
    weight_sum = pd.Series(0.0, index=df.index)
    for cat, col in col_map.items():
        w = fw.get(cat, 0)
        if col not in df.columns or w == 0:
            continue
        has_data = df[col].notna()
        weight_sum += w * has_data.astype(float)

    # Compute contributions
    for cat, col in col_map.items():
        w = fw.get(cat, 0)
        contrib_col = f"{cat}_contrib"
        if col not in df.columns or w == 0:
            df[contrib_col] = 0.0
            continue
        has_data = df[col].notna()
        # effective_weight = w / weight_sum (redistributed weight for this row)
        eff_w = np.where(weight_sum > 0, w / weight_sum, 0)
        df[contrib_col] = np.where(
            has_data,
            df[col].fillna(0) * eff_w,
            0.0,
        )
        df[contrib_col] = df[contrib_col].round(2)

    return df


# =========================================================================
# I-c. Weight sensitivity analysis
# =========================================================================
def run_weight_sensitivity(df: pd.DataFrame, cfg: dict,
                           perturbation_pct: float = 5.0,
                           top_n: int = 20) -> pd.DataFrame:
    """Perturb each category weight ±perturbation_pct and measure top-N stability.

    For each category, creates two scenarios:
      - weight + perturbation_pct (others scaled down proportionally)
      - weight - perturbation_pct (others scaled up proportionally)
    Re-computes composite and checks how many of the baseline top-N change.

    Returns a DataFrame with columns:
      category, direction, original_weight, perturbed_weight,
      top_n_unchanged, top_n_changed, changed_tickers, jaccard_similarity
    """
    import copy

    fw = cfg["factor_weights"]
    col_map = {
        "valuation": "valuation_score", "quality": "quality_score",
        "growth": "growth_score", "momentum": "momentum_score",
        "risk": "risk_score", "revisions": "revisions_score",
        "size": "size_score", "investment": "investment_score",
    }

    # Baseline top-N
    baseline_top = set(df.nsmallest(top_n, "Rank")["Ticker"].tolist())

    results = []
    for cat in fw:
        if fw[cat] == 0:
            continue
        for direction, delta in [("+", perturbation_pct), ("-", -perturbation_pct)]:
            new_w = fw[cat] + delta
            if new_w < 0:
                continue

            # Build perturbed config: adjust this category, scale others proportionally
            cfg_p = copy.deepcopy(cfg)
            fw_p = cfg_p["factor_weights"]
            old_others_sum = sum(fw_p[k] for k in fw_p if k != cat)
            fw_p[cat] = new_w
            if old_others_sum > 0:
                scale = (100 - new_w) / old_others_sum
                for k in fw_p:
                    if k != cat:
                        fw_p[k] = round(fw_p[k] * scale, 2)

            # Re-compute composite on the same scored data (no re-ranking of percentiles)
            df_p = df.copy()
            weighted_sum = pd.Series(0.0, index=df_p.index)
            weight_sum = pd.Series(0.0, index=df_p.index)
            for c, col in col_map.items():
                w = fw_p.get(c, 0)
                if col not in df_p.columns or w == 0:
                    continue
                has_data = df_p[col].notna()
                weighted_sum += df_p[col].fillna(0) * w * has_data.astype(float)
                weight_sum += w * has_data.astype(float)
            comp = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
            df_p["_sens_composite"] = pd.Series(comp, index=df_p.index).rank(
                pct=True) * 100

            # Top-N under perturbed weights
            perturbed_top = set(
                df_p.nlargest(top_n, "_sens_composite")["Ticker"].tolist())

            unchanged = baseline_top & perturbed_top
            changed = (baseline_top - perturbed_top) | (perturbed_top - baseline_top)
            jaccard = len(baseline_top & perturbed_top) / len(
                baseline_top | perturbed_top) if len(baseline_top | perturbed_top) > 0 else 1.0

            results.append({
                "category": cat,
                "direction": direction,
                "original_weight": fw[cat],
                "perturbed_weight": round(new_w, 1),
                "top_n_unchanged": len(unchanged),
                "top_n_changed": len(changed),
                "changed_tickers": ", ".join(sorted(changed)[:10]),
                "jaccard_similarity": round(jaccard, 3),
            })

    return pd.DataFrame(results)


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

    # Majority logic (2-of-3): flag only if at least 2 of the 3
    # dimensions breach their floors.  OR logic (any 1 breach) was
    # too aggressive — with three 30th-percentile thresholds it
    # flagged ~60% of the universe.  Majority logic catches stocks
    # with genuinely broad weakness while tolerating a single weak
    # dimension (e.g. a quality stock with one bad momentum quarter).
    df["Value_Trap_Flag"] = (l1.astype(int) + l2.astype(int) + l3.astype(int)) >= 2

    # Continuous severity score (0-100): how deeply a stock is in trap territory.
    # For each dimension, severity = max(0, (threshold - score) / threshold) * 100.
    # Average across the three dimensions. 0 = no trap risk, 100 = extreme.
    _vt_severity = pd.Series(0.0, index=df.index)
    _n_dims = 0
    if qual_col and qual_thr > 0:
        _q_sev = ((qual_thr - df[qual_col]).clip(lower=0) / qual_thr * 100).fillna(0)
        _vt_severity += _q_sev
        _n_dims += 1
    if mom_col and mom_thr > 0:
        _m_sev = ((mom_thr - df[mom_col]).clip(lower=0) / mom_thr * 100).fillna(0)
        _vt_severity += _m_sev
        _n_dims += 1
    if rev_col and rev_thr > 0:
        _r_sev = ((rev_thr - df[rev_col]).clip(lower=0) / rev_thr * 100).fillna(0)
        _vt_severity += _r_sev
        _n_dims += 1
    df["Value_Trap_Severity"] = (_vt_severity / max(_n_dims, 1)).round(1)

    # Insufficient_Data_Flag: stocks with NaN in any of the three value-trap
    # dimensions. These are NOT flagged as value traps (NaN != poor quality),
    # but the missing data means the value-trap filter cannot fully evaluate them.
    has_nan = pd.Series(False, index=df.index)
    for col_name in ["quality_score", "momentum_score", "revisions_score"]:
        if col_name in df.columns:
            has_nan = has_nan | df[col_name].isna()
    df["Insufficient_Data_Flag"] = has_nan

    return df


def apply_growth_trap_flags(df: pd.DataFrame, cfg: dict):
    """Flag high-growth stocks with weak fundamentals (growth traps).

    Mirror of value-trap logic but for the opposite scenario: stocks with
    high growth scores but poor quality and/or revisions.
    Uses 2-of-3 majority logic (growth ceiling + quality floor + revisions floor).
    """
    gtf = cfg.get("growth_trap_filters", {})
    if not gtf.get("enabled", False):
        df["Growth_Trap_Flag"] = False
        return df

    growth_ceil = gtf.get("growth_ceiling_percentile", 70) / 100.0
    qual_floor = gtf.get("quality_floor_percentile", 35) / 100.0
    rev_floor = gtf.get("revisions_floor_percentile", 35) / 100.0

    # Layer 1 - Growth: ABOVE growth ceiling (high growth = suspect)
    grow_col = "growth_score" if "growth_score" in df.columns else None
    if grow_col:
        grow_thr = df[grow_col].quantile(growth_ceil)
        g1 = df[grow_col].ge(grow_thr).fillna(False)
    else:
        g1 = pd.Series(False, index=df.index)

    # Layer 2 - Quality: below quality floor
    qual_col = "quality_score" if "quality_score" in df.columns else None
    if qual_col:
        qual_thr = df[qual_col].quantile(qual_floor)
        g2 = df[qual_col].le(qual_thr).fillna(False)
    else:
        g2 = pd.Series(False, index=df.index)

    # Layer 3 - Revisions: below revisions floor
    rev_col = "revisions_score" if "revisions_score" in df.columns else None
    if rev_col:
        rev_thr = df[rev_col].quantile(rev_floor)
        g3 = df[rev_col].le(rev_thr).fillna(False)
    else:
        g3 = pd.Series(False, index=df.index)

    # Majority logic (2-of-3): flag only if at least 2 dimensions breach
    df["Growth_Trap_Flag"] = (g1.astype(int) + g2.astype(int) + g3.astype(int)) >= 2

    # Continuous severity score (0-100): how deeply in growth-trap territory.
    # Growth dimension: how far above the ceiling. Quality/revisions: how far below floors.
    _gt_severity = pd.Series(0.0, index=df.index)
    _n_dims = 0
    if grow_col and grow_thr < 100:
        _g_sev = ((df[grow_col] - grow_thr).clip(lower=0) / (100 - grow_thr) * 100).fillna(0)
        _gt_severity += _g_sev
        _n_dims += 1
    if qual_col and qual_thr > 0:
        _q_sev = ((qual_thr - df[qual_col]).clip(lower=0) / qual_thr * 100).fillna(0)
        _gt_severity += _q_sev
        _n_dims += 1
    if rev_col and rev_thr > 0:
        _r_sev = ((rev_thr - df[rev_col]).clip(lower=0) / rev_thr * 100).fillna(0)
        _gt_severity += _r_sev
        _n_dims += 1
    df["Growth_Trap_Severity"] = (_gt_severity / max(_n_dims, 1)).round(1)

    return df


# =========================================================================
# K. Rank stocks
# =========================================================================
def compute_factor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-metric Spearman rank correlation matrix.

    Returns a DataFrame of correlations between percentile-ranked metrics.
    Useful for detecting double-counting (e.g. EV/EBITDA ~ EV/Sales).
    """
    pct_cols = [f"{m}_pct" for m in METRIC_COLS if f"{m}_pct" in df.columns]
    if not pct_cols:
        return pd.DataFrame()
    return df[pct_cols].corr(method="spearman").round(3)


_FINANCIAL_SECTORS = {"Financials", "Financial Services", "Financial"}

# Industries within Financials that should use bank-specific metrics.
# These companies have balance sheets where deposits are liabilities,
# lending is the core business, and EV/EBITDA/ROIC/D-E are meaningless.
_BANK_LIKE_INDUSTRIES = {
    "Banks—Diversified", "Banks—Regional",
    "Banks - Diversified", "Banks - Regional",
    "Diversified Banks", "Regional Banks",
    "Insurance—Diversified", "Insurance—Life",
    "Insurance—Property & Casualty", "Insurance—Specialty",
    "Insurance - Diversified", "Insurance - Life",
    "Insurance - Property & Casualty", "Insurance - Specialty",
    "Life & Health Insurance", "Multi-line Insurance",
    "Property & Casualty Insurance", "Reinsurance",
    "Credit Services", "Mortgage Finance",
}

# Financials-sector tickers that should NOT use bank metrics
# (payment processors, exchanges, analytics — conventional P&Ls).
_NON_BANK_FINANCIALS = {
    "V", "MA", "PYPL", "CPAY", "GPN",
    "FIS", "FISV", "JKHY",
    "FDS", "SPGI", "MCO", "MSCI",
    "ICE", "CME", "CBOE", "NDAQ",
}


def _is_bank_like(ticker: str, sector: str, industry: str) -> bool:
    """Determine if a stock should use bank-specific metrics.

    Priority: explicit override list > industry name > sector fallback.
    """
    if ticker in _NON_BANK_FINANCIALS:
        return False
    if sector not in _FINANCIAL_SECTORS:
        return False
    if industry and industry in _BANK_LIKE_INDUSTRIES:
        return True
    # Default: unknown Financials use bank metrics (conservative —
    # P/B + ROE is better than EV/EBITDA for an unknown financial).
    return True


def add_financial_sector_caveat(df: pd.DataFrame) -> pd.DataFrame:
    """Flag financial-sector stocks and identify bank-like treatment.

    - Financial_Sector_Caveat: True for all Financials stocks
    - _is_bank_like is already set during compute_metrics(); copy to
      public column for Excel output.
    """
    if "Sector" in df.columns:
        df["Financial_Sector_Caveat"] = df["Sector"].isin(_FINANCIAL_SECTORS)
    else:
        df["Financial_Sector_Caveat"] = False
    return df


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
        ("size_score", "Size_Pct"), ("investment_score", "Invest_Pct"),
        ("Composite", "Composite"), ("Rank", "Rank"),
        ("valuation_contrib", "Val_Contrib"), ("quality_contrib", "Qual_Contrib"),
        ("growth_contrib", "Grow_Contrib"), ("momentum_contrib", "Mom_Contrib"),
        ("risk_contrib", "Risk_Contrib"), ("revisions_contrib", "Rev_Contrib"),
        ("size_contrib", "Size_Contrib"), ("investment_contrib", "Invest_Contrib"),
        ("Value_Trap_Flag", "Value_Trap_Flag"),
        ("Value_Trap_Severity", "VT_Severity"),
        ("Growth_Trap_Flag", "Growth_Trap_Flag"),
        ("Growth_Trap_Severity", "GT_Severity"),
        ("Financial_Sector_Caveat", "Fin_Caveat"),
        ("_is_bank_like", "Is_Bank"),
        ("pb_ratio", "P/B"), ("roe", "ROE"), ("roa", "ROA"),
        ("equity_ratio", "Eq_Ratio"),
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
                         "momentum_score", "risk_score", "revisions_score",
                         "size_score", "investment_score"):
                vals.append(round(v, 1) if pd.notna(v) else None)
            elif src.endswith("_contrib"):
                vals.append(round(v, 2) if pd.notna(v) else None)
            else:
                vals.append(v)
        ws.append(vals)

    wb.save(str(out_path))
    return str(out_path)


# =========================================================================
# M. Write scored DataFrame to cache Parquet
# =========================================================================
def write_scores_parquet(df: pd.DataFrame, config_hash: str | None = None) -> str:
    today = datetime.now().strftime("%Y%m%d")
    if config_hash:
        path = CACHE_DIR / f"factor_scores_{config_hash}_{today}.parquet"
    else:
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
        ("P/B Ratio (bank)", "pb_ratio"),
        ("ROIC", "roic"), ("Gross Profit/Assets", "gross_profit_assets"),
        ("Debt/Equity", "debt_equity"), ("Piotroski F-Score", "piotroski_f_score"),
        ("Accruals", "accruals"),
        ("ROE (bank)", "roe"), ("ROA (bank)", "roa"),
        ("Equity Ratio (bank)", "equity_ratio"),
        ("Forward EPS Growth", "forward_eps_growth"),
        ("Revenue Growth", "revenue_growth"), ("Sustainable Growth", "sustainable_growth"),
        ("12-1 Month Return", "return_12_1"), ("6-Month Return", "return_6m"),
        ("Volatility", "volatility"), ("Beta", "beta"),
        ("Analyst Surprise", "analyst_surprise"),
        ("Price Target Upside", "price_target_upside"),
    ]

    rev_cols = ["analyst_surprise", "price_target_upside"]
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
    rev_m = ["analyst_surprise", "price_target_upside"]
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

    # K. Financial sector caveat + Rank
    print("Flagging financial sector caveats...")
    df = add_financial_sector_caveat(df)
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
