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
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook

# Phase 13 (F15): do NOT blanket-suppress warnings. A bare
# filterwarnings("ignore") silenced the screener's OWN schema-drift and
# staleness alerts (_stmt_val label misses, _stale_data), so a yfinance schema
# change would silently NaN metrics with no signal to the operator. Instead we
# suppress only the noisy third-party categories and let our UserWarnings pass.
warnings.simplefilter("default")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")

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
        # GICS sub-industry kept since 2026-10-09: it decides which financials are scored with
        # the bank metric set (_is_bank_like; research/2026-10-09-bank-like-financials.md).
        df = gh[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
        df.columns = ["Ticker", "Company", "Sector", "SubIndustry"]
        df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
        print(f"  Loaded S&P 500 list from GitHub ({len(df)} tickers)")
    except Exception as e:
        print(f"  GitHub CSV failed ({type(e).__name__}), trying Wikipedia...")

    # --- Secondary: Wikipedia scrape ---
    if df is None:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
            df.columns = ["Ticker", "Company", "Sector", "SubIndustry"]
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
        fresh = df[[c for c in ("Ticker", "Company", "Sector", "SubIndustry") if c in df.columns]].to_dict(orient="records")
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


def cache_age_days(cached_dt, now: datetime | None = None) -> int:
    """Whole calendar days between a cache's date and ``now``.

    Cache dates come from the filename (``factor_scores_<hash>_20260812``)
    and are therefore midnight-anchored. A cache written today is age 0; one
    written yesterday is age 1 regardless of the clock time, which is the
    right unit here - what matters is whether a new market close exists,
    not how many hours have elapsed.
    """
    now = now or datetime.now()
    return (now - cached_dt).days


def cache_is_usable(cached_dt, max_age_days: int, now: datetime | None = None) -> bool:
    """Is a cache dated ``cached_dt`` still reusable?

    The rule: ``<tier>_refresh_days: N`` means the cache is reusable for N
    calendar days *starting with the day it was written*. So N=1 means
    "reuse only a cache written today", N=7 means "reuse a cache up to six
    days old". The bound is exclusive.

    This is the semantics :func:`cache_is_fresh` has always implemented via
    ``timedelta``; it is spelled out here because ``run_screener.py`` used to
    hand-roll a looser ``<=`` comparison and reused a stale cache for a week
    (see ``tests/test_cache_freshness.py``).
    """
    if cached_dt is None:
        return False
    return cache_age_days(cached_dt, now=now) < max_age_days


def factor_scores_cache_max_age_days(caching_cfg: dict) -> int:
    """Max age for which the ``factor_scores`` cache may be reused.

    ``factor_scores`` is the *fully scored* dataset. It carries price-derived
    metrics - 12-1 momentum, 6-month return, volatility, beta, Sharpe,
    max drawdown, 52-week proximity and analyst price-target upside -
    alongside the slow-moving fundamentals. **A cache is only as fresh as its
    fastest-moving contents**, so the bound is the price tier, not the
    fundamental tier.

    Bounding it by ``fundamental_data_refresh_days`` (7) is what let a single
    fetch suppress the following week of daily runs, because the warm-start
    path returns before a new cache file is written and so never advances the
    cache date.
    """
    price_days = int(caching_cfg.get("price_data_refresh_days", 1))
    fundamental_days = int(caching_cfg.get("fundamental_data_refresh_days", 7))
    return min(price_days, fundamental_days)


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


def _coalesce(d: dict, *keys):
    """First value in `d` that is present and not NaN/None, else NaN.

    This exists because the nested form - `d.get(A, d.get(B, np.nan))` - does
    NOT fall back when key A is present with a NaN value: `dict.get` returns
    the NaN and never evaluates the default.  A test bars that form from this
    module outright (`tests/test_nan_source_fallback.py`), which is why it is
    spelled with placeholders here.  Every field the fetcher reads is written
    unconditionally (`_safe()` and `_stmt_val()` both return NaN rather than
    omitting the key), so the key is always there and the nested-get form can
    only fire on an exception path, never on the missing-data path it was
    written for.  Measured cost of that: 4 of 502 names lost roic,
    net_debt_to_ebitda, ev_ebitda or ev_sales on every run
    (`research/measurements/2026-09-25-dead-two-source-fallbacks.py`).

    `0.0` is a value, not a miss - `.info` reports exactly 0.0 total debt for
    debt-free companies and that reading must reach invested capital.  `inf`
    is not special-cased, because no call site treats it specially today.
    """
    for k in keys:
        v = d.get(k, np.nan)
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass  # non-scalar: treat as present
        return v
    return np.nan


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


def _estimate(v) -> float:
    """A consensus EPS estimate from Yahoo's eps_trend, or NaN.

    Yahoo fills a period it has no estimate for with exactly 0.0. Read as a number, that made
    AMCR's 90-days-ago consensus zero and its three-month revision +9.6% of price (100th
    percentile in its sector), and LIN's current consensus zero (-3.7%, 4th percentile) - about
    2 composite points each (2026-10-09 audit). A real consensus is an average of analysts'
    figures and is never exactly zero to the cent, so exact zero is treated as missing."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if (not np.isfinite(f) or f == 0.0) else f


def _populated_columns(stmt) -> list:
    """The statement's periods, newest first, leaving out any column Yahoo lists without data.

    Yahoo sometimes adds a period's column before filling it (AMZN listed 2026-06-30 while its
    figures ended 2026-03-31), so a column counts only if it holds at least half as many values
    as the fullest column. A blank cell in a populated column stays blank: it is that period's
    value, missing - not a reason to read the next period in its place (2026-10-09 audit:
    BRK-B's trailing net income summed Q2'25 for a blank Q3'25 and came out 21% low)."""
    try:
        cols = sorted(stmt.columns, key=lambda c: pd.Timestamp(c), reverse=True)
    except (TypeError, ValueError):
        cols = list(stmt.columns)
    counts = stmt[cols].notna().sum()
    top = counts.max() if len(counts) else 0
    if not top:
        return []
    return [c for c in cols if counts[c] >= 0.5 * top]


def _period_row(stmt, matched_idx) -> pd.Series:
    """The matched line item's value for each populated period, newest first (NaN kept)."""
    cols = _populated_columns(stmt)
    row = stmt.loc[matched_idx, cols]
    if isinstance(row, pd.DataFrame):          # a duplicated label: first non-null per period
        row = row.bfill().iloc[0]
    return row


def _quarters_adjacent(cols, start: int, n: int) -> bool:
    """True when the n period-end dates from ``start`` are consecutive quarters (75-105 days apart)."""
    try:
        d = [pd.Timestamp(c) for c in cols[start:start + n]]
    except (TypeError, ValueError):
        return True                             # undated columns: nothing to check against
    return len(d) == n and all(75 <= (d[i] - d[i + 1]).days <= 105 for i in range(n - 1))


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
        vals = _period_row(stmt, matched_idx)
        if len(vals) > col:
            v = vals.iloc[col]
            return float(v) if pd.notna(v) else default
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

        row = _period_row(stmt, matched_idx)
        cols = list(row.index)
        # n consecutive quarters with a value each, from ``offset`` - by period, never by
        # skipping a blank one (2026-10-09 audit).
        if (len(row) >= offset + n_quarters and row.iloc[offset:offset + n_quarters].notna().all()
                and _quarters_adjacent(cols, offset, n_quarters)):
            return float(row.iloc[offset:offset + n_quarters].sum())

        # Partial data: the latest 3 consecutive quarters at offset=0, annualized
        if (n_quarters == 4 and offset == 0 and len(row) >= 3 and row.iloc[:3].notna().all()
                and _quarters_adjacent(cols, 0, 3)):
            if partial_labels is not None:
                partial_labels.append(label)
            return float(row.iloc[:3].sum()) * (4 / 3)

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


BENEISH_INDEX_NAMES = ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA")


def _beneish_parts(d: dict):
    """The eight Beneish indices and which of them came from real data.

    Returns ``(indices, real)`` - ``indices`` in ``BENEISH_INDEX_NAMES`` order,
    ``real[i]`` True when index ``i`` was computed from reported figures and False
    when it fell back to its neutral value (1.0, TATA 0.0) - or ``None`` when revenue
    or total assets are missing for either year. Split out of
    ``_compute_beneish_mscore`` so the page can show the components behind the
    score; the arithmetic is unchanged.
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
        return None

    real = []

    # 1. DSRI (Days Sales in Receivables Index)
    if pd.notna(rec_t) and pd.notna(rec_p) and rec_p > 0:
        dsri = (rec_t / rev_t) / (rec_p / rev_p)
        real.append(True)
    else:
        dsri = 1.0  # neutral
        real.append(False)

    # 2. GMI (Gross Margin Index)
    gm_t = (rev_t - cogs_t) / rev_t if (pd.notna(cogs_t) and rev_t > 0) else np.nan
    gm_p = (rev_p - cogs_p) / rev_p if (pd.notna(cogs_p) and rev_p > 0) else np.nan
    if pd.notna(gm_t) and pd.notna(gm_p) and gm_t > 0:
        gmi = gm_p / gm_t
        real.append(True)
    else:
        gmi = 1.0
        real.append(False)

    # 3. AQI (Asset Quality Index)
    if all(pd.notna(x) for x in [ca_t, ppe_t, ta_t, ca_p, ppe_p, ta_p]):
        aq_t = 1 - (ca_t + ppe_t) / ta_t
        aq_p = 1 - (ca_p + ppe_p) / ta_p
        aqi = (aq_t / aq_p) if aq_p != 0 else 1.0
        real.append(True)
    else:
        aqi = 1.0
        real.append(False)

    # 4. SGI (Sales Growth Index) - always computable (rev guaranteed above)
    sgi = rev_t / rev_p
    real.append(True)

    # 5. DEPI (Depreciation Index)
    if (all(pd.notna(x) for x in [dep_t, dep_p, ppe_t, ppe_p])
            and (ppe_t + dep_t) > 0 and (ppe_p + dep_p) > 0):
        depi = (dep_p / (ppe_p + dep_p)) / (dep_t / (ppe_t + dep_t))
        real.append(True)
    else:
        depi = 1.0
        real.append(False)

    # 6. SGAI (SGA Expense Index) - set to 1.0 (neutral) if SGA missing
    if (all(pd.notna(x) for x in [sga_t, sga_p])
            and rev_t > 0 and rev_p > 0 and sga_p > 0):
        sgai = (sga_t / rev_t) / (sga_p / rev_p)
        real.append(True)
    else:
        sgai = 1.0
        real.append(False)

    # 7. LVGI (Leverage Index)
    if (all(pd.notna(x) for x in [ltd_t, cl_t, ta_t, ltd_p, cl_p, ta_p])
            and ta_t > 0 and ta_p > 0
            and (ltd_p + cl_p) > 0):
        lvgi = ((ltd_t + cl_t) / ta_t) / ((ltd_p + cl_p) / ta_p)
        real.append(True)
    else:
        lvgi = 1.0
        real.append(False)

    # 8. TATA (Total Accruals to Total Assets)
    if pd.notna(ni_t) and pd.notna(ocf_t) and ta_t > 0:
        tata = (ni_t - ocf_t) / ta_t
        real.append(True)
    else:
        tata = 0.0  # neutral
        real.append(False)

    return (dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata), real


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
    parts = _beneish_parts(d)
    if parts is None:
        return np.nan, False
    (dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata), real = parts

    # Minimum-data gate: require >= 5 of 8 indices computed from real data.
    # With < 5 indices, the M-Score is dominated by neutral defaults (1.0)
    # and loses discriminating power - analogous to Piotroski's n_testable >= 6.
    if sum(real) < 5:
        return np.nan, False

    m_score = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi
               + 0.892 * sgi + 0.115 * depi - 0.172 * sgai
               + 4.679 * tata - 0.327 * lvgi)

    return m_score, (m_score > -2.22)


# =========================================================================
# E½. Price-series integrity (split-scale consistency)
# =========================================================================
# Nine of the 44 metrics - the whole risk category and three quarters of
# momentum - are derived from one `Ticker.history()` call.  Nothing used to
# check that the series it returns is internally consistent, and on
# 2026-08-26 it was not: Yahoo's 13-month series for MNST alternated between
# pre- and post-split prices across its 2026-08-11 2:1 split, so
# `return_12_1` was computed as (unadjusted July price) / (adjusted 2025
# price) = +50%, landing MNST in the 97th percentile of momentum when its
# true split-adjusted 12-1 return was about -25% (3rd percentile).  Setting
# `auto_adjust=False` returned byte-identical numbers, so the adjustment was
# never applied at all.  See METHODOLOGY_CHANGELOG.md 2026-08-26.
#
# The test below is exact rather than heuristic: it uses the split ratio
# Yahoo itself reports.  A correctly back-adjusted series contains no day
# whose close-to-close price ratio is near 1/k or k for a declared split of
# ratio k - if one does, the series is mixing two price scales.

# Arm the check only for splits big enough to be distinguishable from an
# ordinary trading day.  Measured 2026-08-26 over 137,313 ticker-days
# (503 S&P 500 names, 13 months): p99.9 of |daily return| is 17.2% and only
# 21 days in the whole sample exceed 30%.  A ratio implying a jump smaller
# than 25% cannot be told apart from real trading, so it is left alone -
# which excludes the small spin-off "ratios" (SPGI 1.057, HON 1.061,
# CMCSA 1.067, FDX 1.241, BDX 1.272) that Yahoo also reports as splits.
_SPLIT_MIN_JUMP = 0.25

# Tolerance in log-price-ratio space.  MNST's seven artifact days sat
# 0.008-0.041 from the exact 2:1 ratio, so 0.06 catches them with margin
# while staying far from the 25% arming floor.
_SPLIT_RATIO_TOL_LOG = 0.06


def check_price_series_integrity(closes, splits=None) -> str | None:
    """Return None if a price series is self-consistent, else why it is not.

    `closes` is the Close column of a yfinance history frame; `splits` is the
    matching "Stock Splits" column (zero on ordinary days).  When a split of
    ratio k is declared inside the window, a properly back-adjusted series
    must not contain a day whose price ratio is close to 1/k (unadjusted) or
    k (adjusted history against an unadjusted quote).

    Verified 2026-08-26 against all 17 S&P 500 split events of the previous
    13 months: 11 were large enough to arm the check, and it fired on
    exactly one - MNST, the known-bad series - with no false positives.
    """
    if closes is None or splits is None or len(closes) < 3:
        return None

    try:
        ratios = pd.Series(splits).reindex(closes.index).fillna(0.0)
    except (TypeError, ValueError):
        return None

    declared = {float(k) for k in ratios.values if pd.notna(k) and k not in (0.0, 1.0)}
    if not declared:
        return None

    day_ratio = (closes / closes.shift(1)).dropna()
    if day_ratio.empty:
        return None
    log_day = np.log(day_ratio.where(day_ratio > 0))

    for k in sorted(declared):
        if k <= 0:
            continue
        # The jump an unadjusted series would show, and its mirror image.
        if abs(1.0 / k - 1.0) < _SPLIT_MIN_JUMP and abs(k - 1.0) < _SPLIT_MIN_JUMP:
            continue
        for target in (1.0 / k, k):
            if abs(target - 1.0) < _SPLIT_MIN_JUMP:
                continue
            hits = log_day[(log_day - np.log(target)).abs() <= _SPLIT_RATIO_TOL_LOG]
            if len(hits):
                when = ", ".join(str(d.date()) for d in hits.index[:3])
                return (
                    f"series mixes pre- and post-split prices across a "
                    f"{k:g}:1 split - {len(hits)} day(s) move by ~{target:.3g}x "
                    f"({when}{'...' if len(hits) > 3 else ''})"
                )
    return None


# Everything computed from the 13-month price history.  When the series
# fails its integrity check these are all withheld, and `factor_engine`'s
# existing missing-data path (`na_option="keep"` plus the `has_data` mask in
# compute_category_scores) renormalises the surviving weights.  `price_latest`
# is deliberately NOT in this list: it is a single point from the most recent
# bar, and the defect is in relationships *between* prices at different dates -
# which is exactly what the withheld metrics measure.  A single bar cannot
# mix two scales with itself.
#
# This note used to add that `info["currentPrice"]` "takes precedence over it
# everywhere it is used".  That was false at one of the seven sites and the
# claim is gone (2026-09-25): `return_12m` prefers `price_latest`, because it
# is the one price site that divides two dates and so must not cross sources.
# The other six prefer `info["currentPrice"]` and fall back to `price_latest`
# via `_coalesce`.  Withholding `price_latest` would therefore also cost
# `return_12m`, whose far endpoint is withheld anyway - so the conclusion
# stands, for a different reason than the one written here before.
# Context layer (2026-10-08): the option-chain and insider requests run in a separate pass
# AFTER the core fetch (context_fetch.py) - inside it, they tripped the rate limiter and slowed
# the scored data. Only the trend context, which needs no extra request, is computed here.

PRICE_SERIES_DERIVED_FIELDS = (
    "price_1m_ago", "price_6m_ago", "price_12m_ago",
    "volatility_1y", "_daily_returns", "avg_daily_dollar_volume",
)


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
        # Plain-English business description for the dashboard drilldown.
        # Zero additional API cost - same .info dict fetched above. Display
        # only: never scored, never ranked, never fed to a metric.
        rec["longBusinessSummary"] = _safe(info, "longBusinessSummary", "")
        # Next scheduled earnings date. Zero additional API cost - same .info
        # dict fetched above. Display only: never scored, never ranked, never
        # fed to a metric (plan/dashboard-north-star.md gap 4).
        #
        # `earningsTimestamp` is deliberately NOT captured. Measured live
        # 2026-09-29: it holds the *last* report for some tickers (AAPL
        # 2026-07-30, EXPE 2026-08-05) and the *next* for others (HST, JPM and
        # NVDA all returned their forthcoming date), so there is no label that
        # is true of every row. Start/End are unambiguously the next window.
        #
        # Measured over all 503 tickers 2026-09-29: 503 carry a start date,
        # start and end were equal every time, and the timestamp is only ever
        # 12:30 or 20:00 UTC - 08:30 and 16:00 US/Eastern, before the open or
        # after the close. Nothing sits near a date boundary, so reading the
        # UTC date and reading the Eastern date disagree for 0 of 503.
        rec["earningsTimestampStart"] = _safe(info, "earningsTimestampStart")
        # End of the current (not yet reported) fiscal year, for the 12-month-forward EPS blend.
        rec["_next_fy_end"] = _safe(info, "nextFiscalYearEnd")
        rec["earningsTimestampEnd"]   = _safe(info, "earningsTimestampEnd")
        # 209 of the 492 future dates (42.5%) are the provider's estimate
        # rather than a confirmed schedule, and the flag was present on every
        # one. This is what keeps the dashboard from presenting a guess with
        # the same confidence as a company-announced date.
        rec["isEarningsDateEstimate"] = _safe(info, "isEarningsDateEstimate", None)
        # Analyst price target fields (zero additional API cost — same .info dict)
        rec["targetMeanPrice"]         = _safe(info, "targetMeanPrice")
        rec["targetHighPrice"]         = _safe(info, "targetHighPrice")
        rec["targetLowPrice"]          = _safe(info, "targetLowPrice")
        rec["numberOfAnalystOpinions"] = _safe(info, "numberOfAnalystOpinions")
        # Short interest (days to cover). Updated bi-monthly by exchanges with 1-2 week lag.
        rec["shortRatio"]             = _safe(info, "shortRatio")
        # Candidate metric .info fields (Phase 11: Metric Evolution)
        # Always fetched even when weight=0, so improvement engine can evaluate IC.
        # Zero incremental API cost — same .info dict already fetched above.
        rec["fiftyTwoWeekHigh"]        = _safe(info, "fiftyTwoWeekHigh")
        rec["heldPercentInsiders"]     = _safe(info, "heldPercentInsiders")
        rec["heldPercentInstitutions"] = _safe(info, "heldPercentInstitutions")
        rec["shortPercentOfFloat"]     = _safe(info, "shortPercentOfFloat")
        rec["recommendationMean"]      = _safe(info, "recommendationMean")

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
        # Operating income first (2026-10-09 audit): Yahoo's "EBIT" row is pretax income plus
        # interest expense, so it carries non-operating gains - GOOGL's read $301.5B against
        # $147.6B of operating income, doubling its ROIC. ROIC, EV/EBITDA and net debt / EBITDA
        # are all operating measures (Greenblatt 2006; Koller et al., *Valuation*). "EBIT" only
        # where no operating-income line exists.
        rec["ebit"]                   = _stmt_val_ltm(q_fins, "Operating Income", partial_labels=_ltm_partial)
        if np.isnan(rec["ebit"]):
            rec["ebit"]               = _stmt_val_ltm(q_fins, "EBIT", partial_labels=_ltm_partial)
        rec["ebitda"]                 = _stmt_val_ltm(q_fins, "EBITDA", partial_labels=_ltm_partial)
        rec["netIncome"]              = _stmt_val_ltm(q_fins, "Net Income", partial_labels=_ltm_partial)
        rec["netIncome_prior"]        = _stmt_val_ltm(q_fins, "Net Income", offset=4)
        rec["incomeTaxExpense"]       = _stmt_val_ltm(q_fins, "Tax Provision", partial_labels=_ltm_partial)
        if np.isnan(rec["incomeTaxExpense"]):
            rec["incomeTaxExpense"]   = _stmt_val_ltm(q_fins, "Income Tax", partial_labels=_ltm_partial)
        rec["pretaxIncome"]           = _stmt_val_ltm(q_fins, "Pretax Income", partial_labels=_ltm_partial)
        rec["costOfRevenue"]          = _stmt_val_ltm(q_fins, "Cost Of Revenue", partial_labels=_ltm_partial)
        # Interest Expense: needed for interest_coverage candidate metric
        rec["interestExpense"]        = _stmt_val_ltm(q_fins, "Interest Expense", partial_labels=_ltm_partial)
        if np.isnan(rec["interestExpense"]):
            rec["interestExpense"]    = _stmt_val_ltm(q_fins, "Interest Expense Non Operating", partial_labels=_ltm_partial)

        # Fallback: if quarterly IS produced all NaN, try annual for everything
        if all(np.isnan(rec.get(k, np.nan)) for k in ["totalRevenue", "netIncome", "ebit"]):
            rec["_data_source"] = "annual"
            rec["totalRevenue"]           = _stmt_val(fins, "Total Revenue")
            rec["totalRevenue_prior"]     = _stmt_val(fins, "Total Revenue", 1)
            rec["grossProfit"]            = _stmt_val(fins, "Gross Profit")
            rec["grossProfit_prior"]      = _stmt_val(fins, "Gross Profit", 1)
            rec["ebit"]                   = _stmt_val(fins, "Operating Income")
            if np.isnan(rec["ebit"]):
                rec["ebit"]               = _stmt_val(fins, "EBIT")
            rec["ebitda"]                 = _stmt_val(fins, "EBITDA")
            rec["netIncome"]              = _stmt_val(fins, "Net Income")
            rec["netIncome_prior"]        = _stmt_val(fins, "Net Income", 1)
            rec["incomeTaxExpense"]       = _stmt_val(fins, "Tax Provision")
            if np.isnan(rec["incomeTaxExpense"]):
                rec["incomeTaxExpense"]   = _stmt_val(fins, "Income Tax")
            rec["pretaxIncome"]           = _stmt_val(fins, "Pretax Income")
            rec["costOfRevenue"]          = _stmt_val(fins, "Cost Of Revenue")
            rec["interestExpense"]        = _stmt_val(fins, "Interest Expense")
            if np.isnan(rec["interestExpense"]):
                rec["interestExpense"]    = _stmt_val(fins, "Interest Expense Non Operating")
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

        # Revenue growth on a true 12-month window (2026-10-09, CLAUDE.md 0.9(b)): the latest
        # quarter against the same quarter a year earlier - the seasonal comparison quarterly
        # revenue is modelled on (Jegadeesh & Livnat 2006). Kept only when the two quarters
        # really are a year apart. research/2026-10-09-revenue-growth-window.md
        try:
            if q_fins is not None and "Total Revenue" in q_fins.index:
                _qr = q_fins.loc["Total Revenue"]
                _qc = sorted([c for c in _qr.index if pd.notna(_qr[c])], reverse=True)
                if len(_qc) >= 5:
                    _gap = (pd.Timestamp(_qc[0]) - pd.Timestamp(_qc[4])).days
                    if 350 <= _gap <= 380 and float(_qr[_qc[4]]) > 0:
                        rec["_rev_q0"] = float(_qr[_qc[0]])
                        rec["_rev_q4"] = float(_qr[_qc[4]])
                        rec["_rev_q0_date"] = str(pd.Timestamp(_qc[0]).date())
                        rec["_rev_q4_date"] = str(pd.Timestamp(_qc[4]).date())
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: quarterly revenue YoY unavailable: {type(e).__name__}: {e}")

        # Revenue 3 years ago from annual financials (col=3) for 3-year CAGR.
        # Annual financials typically provides 4 columns (indices 0-3).
        rec["totalRevenue_3yr_ago"] = _stmt_val(fins, "Total Revenue", 3)
        # Phase 13 (F20): matching ANNUAL current revenue (col=0) so the 3-year
        # CAGR uses same-basis endpoints (annual vs annual) instead of mixing
        # LTM current with annual-3yr-ago.
        rec["totalRevenue_annual"] = _stmt_val(fins, "Total Revenue", 0)

        # EBIT prior year from annual financials (col=1) for operating leverage (DOL).
        rec["ebit_prior"] = _stmt_val(fins, "Operating Income", 1)
        # Phase 13 (F22): matching ANNUAL current EBIT (col=0) so DOL uses
        # same-basis endpoints (annual vs annual) instead of LTM vs annual.
        # Operating income first, as for the trailing figure (2026-10-09).
        rec["ebit_annual"] = _stmt_val(fins, "Operating Income", 0)
        if np.isnan(rec["ebit_annual"]):
            rec["ebit_annual"] = _stmt_val(fins, "EBIT", 0)
        rec["totalRevenue_annual_prior"] = _stmt_val(fins, "Total Revenue", 1)
        # Annual figures for Piotroski's three year-on-year signals (2026-10-09): net income
        # and gross profit for the latest two fiscal years, total assets at the end of the
        # latest three - Piotroski (2000) scales ROA and turnover by BEGINNING-of-year assets.
        rec["_ni_a0"] = _stmt_val(fins, "Net Income", 0)
        rec["_ni_a1"] = _stmt_val(fins, "Net Income", 1)
        rec["_gp_a0"] = _stmt_val(fins, "Gross Profit", 0)
        rec["_gp_a1"] = _stmt_val(fins, "Gross Profit", 1)
        rec["_ta_a1"] = _stmt_val(bs, "Total Assets", 1)
        rec["_ta_a2"] = _stmt_val(bs, "Total Assets", 2)
        if np.isnan(rec["ebit_prior"]):
            rec["ebit_prior"] = _stmt_val(fins, "EBIT", 1)

        # ---- Balance sheet: MRQ (most recent quarter) ----
        # Use quarterly BS for current values; col=4 for year-ago MRQ.
        # Fall back to annual BS if quarterly is unavailable.
        _bs_src = q_bs if (q_bs is not None and not q_bs.empty) else bs
        _bs_is_quarterly = (_bs_src is q_bs)
        _bs_prior_col = 4 if (_bs_is_quarterly and _bs_src is not None
                               and len(_bs_src.columns) >= 5) else 1
        # The "prior" balance sheet must be a year earlier: with a four-column quarterly sheet
        # column 1 is the previous QUARTER, and asset growth / Piotroski 5-7 would compare three
        # months while calling it a year. Then no prior is used (2026-10-09; latent - measured
        # 40 of 40 sampled sheets had 5+ columns). research/2026-10-09-revenue-growth-window.md
        try:
            # judged on the populated periods, which are what _stmt_val reads
            _bs_cols = _populated_columns(_bs_src) if _bs_src is not None else []
            if len(_bs_cols) <= _bs_prior_col:
                _bs_prior_col = 10_000
            else:
                _gap = (pd.Timestamp(_bs_cols[0]) - pd.Timestamp(_bs_cols[_bs_prior_col])).days
                if not 330 <= _gap <= 400:
                    _bs_prior_col = 10_000          # out of range: every *_prior reads NaN
        except (TypeError, ValueError):
            pass

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
        # Cash plus short-term investments on the same balance sheet - what enterprise value
        # nets off (Yahoo's totalCash), and so what net debt nets off too (2026-10-09).
        rec["cash_sti_bs"]            = _stmt_val(_bs_src, "Cash Cash Equivalents And Short Term Investments")
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
        # No quarterly D&A (DAL, UAL, MAS on 2026-10-09): the last fiscal year's, from the annual
        # cash-flow statement. D&A moves slowly, and without it EBITDA fell back to Yahoo's
        # "EBITDA" row, which for these companies equals EBIT (2026-10-09 audit).
        if np.isnan(rec["da_cf"]):
            for _da_label in ("Depreciation And Amortization", "Reconciled Depreciation",
                              "Depreciation Amortization Depletion"):
                rec["da_cf"] = _stmt_val(cf, _da_label)
                if not np.isnan(rec["da_cf"]):
                    rec["_da_annual"] = True
                    break

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
                    _pc = _populated_columns(stmt_obj)
                    most_recent = _pc[0] if _pc else stmt_obj.columns[0]
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

                # Refuse to derive metrics from a series that mixes two price
                # scales.  Withholding them is the honest outcome: the
                # alternative is a number indistinguishable from analysis that
                # is wrong by the split ratio.  Repair is not attempted -
                # MNST's series flipped scale on seven separate days, so there
                # is no single factor that puts it right.
                _integrity = check_price_series_integrity(
                    closes, hist.get("Stock Splits"))
                if _integrity is not None:
                    rec["_price_series_rejected"] = _integrity
                    for _f in PRICE_SERIES_DERIVED_FIELDS:
                        rec[_f] = np.nan
                    warnings.warn(
                        f"{ticker_str}: price history rejected - {_integrity}; "
                        f"momentum and risk metrics withheld")
                else:
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
                    # The daily figure it is annualised from, kept so the page can show
                    # "daily sd x 252^0.5" (metric_lineage.EQUATIONS["volatility"]).
                    rec["_vol_daily_sd"] = float(daily_ret.std()) if len(daily_ret) >= 200 else np.nan
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

                    # Context only (2026-10-08, plan/context-layer.md): trend, range, recent
                    # move and volume from the same history - no extra API call, never scored.
                    try:
                        from context_signals import price_context
                        rec.update(price_context(closes, hist.get("Volume")))
                    except Exception as e:  # noqa: BLE001 - display-only, must not fail a fetch
                        warnings.warn(f"{ticker_str}: price context unavailable: {type(e).__name__}: {e}")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: price history extraction failed: {type(e).__name__}: {e}")

        # ---- earnings surprises ----
        try:
            eh = t.earnings_history
            # Oldest first by quarter date: Yahoo does not always return them in order (ACN's
            # came back shuffled on 2026-10-09), and every metric below reads positions.
            if eh is not None and not eh.empty:
                try:
                    eh = eh.loc[sorted(eh.index, key=lambda x: pd.Timestamp(x))]
                except (TypeError, ValueError):
                    pass
            # A history whose newest quarter is long past is stale, not current: AMCR's ended
            # Dec-2025 while it had reported Jun-2026 (2026-10-09 audit). Kept for display,
            # not scored.
            _eh_stale = False
            if eh is not None and not eh.empty:
                try:
                    _eh_stale = (pd.Timestamp.now().normalize() - pd.Timestamp(eh.index[-1]).tz_localize(None)).days > EH_MAX_AGE_DAYS
                except (TypeError, ValueError):
                    _eh_stale = False
            if eh is not None and not eh.empty:
                surs = []
                ordered_surs = []  # Per-quarter surprises in chronological order
                _quarters = []     # [date, actual, estimate, surprise] - published, see below
                for _qd, row in eh.tail(4).iterrows():
                    a, e = row.get("epsActual", np.nan), row.get("epsEstimate", np.nan)
                    if pd.notna(a) and pd.notna(e) and abs(e) > 0.001:
                        # Floor denominator at $0.10 to prevent near-zero
                        # estimates from producing extreme surprise ratios.
                        sur = (a - e) / max(abs(e), 0.10)
                        surs.append(sur)
                        ordered_surs.append(sur)
                    else:
                        sur = None
                        ordered_surs.append(np.nan)
                    _quarters.append([str(_qd)[:10],
                                      float(a) if pd.notna(a) else None,
                                      float(e) if pd.notna(e) else None,
                                      float(sur) if sur is not None else None])
                # The four quarters the three surprise metrics are made from, so the page
                # can show them (CLAUDE.md 0.10(d)); the surprises are this loop's own.
                rec["_eps_quarters"] = json.dumps(_quarters, separators=(",", ":"))
                if _eh_stale:
                    rec["_eps_quarters_stale"] = True
                    surs, ordered_surs = [], [np.nan] * len(ordered_surs)
                # Median is robust to a single outlier quarter.
                rec["analyst_surprise"] = float(np.median(surs)) if len(surs) >= 2 else np.nan

                # --- Earnings Acceleration ---
                # The latest quarter's surprise minus the quarter before's - both must be
                # there; with one missing, "the one before" would be from half a year earlier.
                if len(ordered_surs) >= 2 and pd.notna(ordered_surs[-1]) and pd.notna(ordered_surs[-2]):
                    rec["earnings_acceleration"] = ordered_surs[-1] - ordered_surs[-2]
                else:
                    rec["earnings_acceleration"] = np.nan

                # --- Recency-Weighted Beat Score, 0-10 ---
                # Each quarter's position counted back from the newest (newest 4, then 3, 2, 1)
                # is its weight; the score is the beating quarters' share of the weight of the
                # quarters with data, times 10. With four quarters this is exactly the old
                # 1+2+3+4 sum; with fewer it no longer caps the score (three beats out of three
                # quarters read 6 before 2026-10-09, the same as missing the newest of four).
                _pos = [(4 - (len(ordered_surs) - 1 - i), s) for i, s in enumerate(ordered_surs) if pd.notna(s)]
                if len(_pos) >= 2:
                    rec["consecutive_beat_streak"] = float(
                        10.0 * sum(w for w, s in _pos if s > 0) / sum(w for w, _ in _pos))
                else:
                    rec["consecutive_beat_streak"] = np.nan
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: earnings surprise extraction failed: {type(e).__name__}: {e}")

        # ---- FY1 consensus EPS revision (raw components only) ----
        # Ticker.eps_trend is a 4x5 frame: rows are the forecast period
        # ('0q', '+1q', '0y', '+1y'), columns the consensus as it stood
        # 'current' / '7daysAgo' / ... / '90daysAgo'.  We take the '0y' (FY1)
        # row, per research/2026-09-07 SS5: FY1 is the horizon Chan, Jegadeesh &
        # Lakonishok (1996) and Barra's Sentiment descriptors both use, and the
        # quarterly row rolls over mid-window, which would put the 'current'
        # and '90daysAgo' figures on different fiscal periods.
        #
        # Only the two raw endpoints are stored here.  The metric itself is
        # built in compute_metrics(), where the price denominator lives, so it
        # is computable from a plain dict in tests and in the golden fixture.
        try:
            et = t.eps_trend
            if et is not None and not et.empty and "0y" in et.index:
                _row = et.loc["0y"]
                for _col, _key in [("current", "_fy1_eps_current"),
                                   ("90daysAgo", "_fy1_eps_90d_ago")]:
                    if _col in _row.index:
                        rec[_key] = _estimate(_row[_col])
            # Next fiscal year's consensus (FY2), for MSCI's 12-month forward EPS (2026-10-09).
            if et is not None and not et.empty and "+1y" in et.index and "current" in et.columns:
                rec["_fy2_eps_current"] = _estimate(et.loc["+1y", "current"])
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            warnings.warn(f"{ticker_str}: eps_trend extraction failed: {type(e).__name__}: {e}")

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

    # Attempt to use tqdm for progress bar; fall back to print if unavailable
    try:
        from tqdm import tqdm as tqdm_lib
        batch_range = tqdm_lib(range(n_batches), desc="Fetching tickers", unit="batch")
    except ImportError:
        batch_range = range(n_batches)

    for bi in batch_range:
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
    """Fetch S&P 500 daily returns for beta and Jensen's alpha.

    The total-return index (^SP500TR, dividends reinvested) first: each stock's return here is
    dividend-adjusted, so a price-only index tilted every alpha up by beta x the index's
    dividend return - 1.37pp over the year to 2026-10-09, up to ~5.5pp for a beta-4 stock, and
    it moved 167 stocks' alpha percentile (2026-10-09 audit). The price index (^GSPC) only if the
    total-return series cannot be fetched; the series' ``attrs["source"]`` says which.

    Implements exponential backoff retry (1s / 2s / 4s) per §10.3.
    """
    for symbol in ("^SP500TR", "^GSPC"):
        for attempt in range(max_retries):
            try:
                import yfinance as yf
                hist = yf.Ticker(symbol).history(period="1y", auto_adjust=True)
                closes = hist["Close"].dropna()
                if len(closes) < 200:
                    raise ValueError(f"only {len(closes)} closes")
                out = np.log(closes / closes.shift(1)).dropna()
                out.attrs["source"] = symbol
                if symbol != "^SP500TR":
                    warnings.warn("Market returns: total-return index unavailable; using the price index ^GSPC")
                return out
            except Exception as e:
                warnings.warn(f"Market returns ({symbol}) fetch attempt {attempt+1}/{max_retries} failed: {e}")
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


def _generate_sample_data(universe_df: pd.DataFrame, seed: int = 42, risk_free_rate: float = 0.045) -> pd.DataFrame:
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
        # NOTE: fy1_revision_3m is deliberately NOT synthesized. This generator
        # emits finished metric values and carries no price field, while the
        # revision is built in compute_metrics() from two raw consensus
        # endpoints and a price denominator - so a fabricated value here would
        # be silently overwritten with NaN, exactly as price_target_upside and
        # proximity_52w_high already are on this path. Leaving it missing is
        # the honest outcome and the existing has_data renormalisation handles
        # it. Adding a price field to make it compute would change five
        # unrelated metrics on this path; that is a separate change.

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
            "sharpe_ratio": round((mom_12_1 - risk_free_rate) / vol, 2) if vol > 0 else np.nan,
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
            # Phase 13 (F26): fill remaining METRIC_COLS so the offline sample
            # path scores the SAME factor set as the live path (was 39 cols).
            "sortino_ratio": round((mom_12_1 - risk_free_rate) / max(vol * 0.7, 0.05), 2),
            "max_drawdown_1y": round(-abs(tn(0.20, 0.12, low=0.02, high=0.80)), 4),
            "beneish_m_score": round(tn(-2.4, 0.6, low=-4.0, high=1.0), 2),
            # Candidate metrics (weight 0 by default; improvement engine may activate)
            "proximity_52w_high": round(tn(0.85, 0.12, low=0.3, high=1.0), 4),
            "operating_margin": round(tn(0.15, 0.10, low=-0.2, high=0.5), 4),
            "current_ratio": round(tn(1.8, 0.7, low=0.3, high=5.0), 2),
            "dividend_yield": round(max(0.0, tn(0.018, 0.015)), 4),
            "insider_ownership": round(max(0.0, tn(0.03, 0.05, high=0.4)), 4),
            "short_pct_float": round(max(0.0, tn(0.03, 0.03, high=0.30)), 4),
            "analyst_rating": round(tn(2.3, 0.6, low=1.0, high=5.0), 2),
            "interest_coverage": round(tn(8.0, 6.0, low=-2.0, high=40.0), 2),
            "earnings_variability": round(abs(tn(0.06, 0.05, low=0.0, high=1.0)), 4),
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
            rec["beneish_m_score"] = np.nan      # Banks excluded from Beneish
            rec["operating_margin"] = np.nan     # Candidate; non-bank only
            rec["interest_coverage"] = np.nan    # Candidate; non-bank only
            rec["current_ratio"] = np.nan        # Candidate; non-bank only
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
    # The day the metrics describe - for the months left in each fiscal year (forward EPS).
    _as_of = pd.Timestamp(datetime.now(timezone.utc).date())

    # Market 12-month total return (computed once, reused for all tickers).
    # Convert cumulative log returns to simple return.
    if len(market_returns) >= 200:
        market_12m_return = float(np.exp(market_returns.sum()) - 1)
    else:
        market_12m_return = float('nan')

    for d in raw_data:
        rec = {
            "Ticker": d.get("Ticker"),
            "Company": _coalesce(d, "shortName", "Ticker"),
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
        _debt_info = _coalesce(d, "totalDebt", "totalDebt_bs")
        _debt_bs = _coalesce(d, "totalDebt_bs", "totalDebt")
        # Cash: use info totalCash for EV (matches yfinance's own EV
        # definition which includes short-term investments), but use
        # balance sheet Cash & Cash Equivalents for ROIC (stricter
        # definition of invested capital).
        _cash_ev = _coalesce(d, "totalCash", "cash_bs")
        _cash_bs = _coalesce(d, "cash_bs", "totalCash")
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

        # The EV the scorer actually used (the API value unless it was missing or
        # failed the cross-check above) - published so the page shows this, not Yahoo's
        # raw figure, beside the multiples built from it.
        rec["_ev_used"] = ev

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
        _is_bank = _is_bank_like(ticker, d.get("_gics_sector") or _sector, _industry, d.get("_gics_sub"))
        # Beneish's M-score and its receivables index were estimated on non-financial companies:
        # his sample excluded financial firms, whose sales and receivables mean something else
        # (an insurance broker's receivables are premiums it holds for insurers - AON's DSRI read
        # 3.52). So neither the M-score nor the channel-stuffing flag is applied to any Financials
        # stock, bank-like or not (2026-10-09, with the GICS bank-like rule).
        _beneish_applies = (not _is_bank) and ((d.get("_gics_sector") or _sector) not in _FINANCIAL_SECTORS)
        # ...and so it is not "missing" for them either: applicable_coverage reads this.
        rec["_beneish_na"] = (not _is_bank) and not _beneish_applies

        # -- Valuation metrics (1-4) --
        try:
            # 1. EV/EBITDA — compute GAAP EBITDA as EBIT + D&A when both
            # components are available.  yfinance's reported EBITDA can
            # include non-operating items that distort the multiple.
            # Fall back to reported EBITDA if D&A is unavailable.
            _ebit_for_ebitda = d.get("ebit", np.nan)
            _da = d.get("da_cf", np.nan)
            # Phase 13 (F36): D&A is a positive add-back regardless of the sign
            # yfinance's cashflow row happens to carry. The old `_da >= 0` gate
            # sent exactly the negative-sign rows to the distrusted reported
            # EBITDA, creating two EBITDA definitions in one percentile rank.
            if pd.notna(_ebit_for_ebitda) and pd.notna(_da):
                ebitda = _ebit_for_ebitda + abs(_da)
            else:
                ebitda = d.get("ebitda", np.nan)  # fallback to reported
                # ...unless the reported "EBITDA" is just EBIT again - Yahoo's row equalled its
                # EBIT for all six stocks that reached this fallback on 2026-10-09.
                _ebit_raw = d.get("ebit", np.nan)
                if pd.notna(ebitda) and pd.notna(_ebit_raw) and abs(ebitda - _ebit_raw) <= 0.005 * abs(ebitda):
                    ebitda = np.nan
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
            # Phase 13 (F19): SINGLE definition across the universe. Previously
            # this fell back to trailingEps/price (a different, GAAP-EPS-based
            # definition) when LTM NI or MC was missing, so the fallback cohort
            # was ranked on an incomparable basis inside one percentile rank.
            # Now: if LTM NI or MC is unavailable, return NaN and let per-row
            # weight redistribution handle the gap.
            if pd.notna(ni) and pd.notna(mc) and mc > 0:
                rec["earnings_yield"] = ni / mc
            else:
                rec["earnings_yield"] = np.nan

            # 4. EV/Sales
            rec["ev_sales"] = (ev / rev_c) if (pd.notna(ev) and pd.notna(rev_c) and rev_c > 0 and ev > 0) else np.nan
            # The EBITDA and free cash flow the multiples above used, for display.
            rec["_ebitda_used"] = ebitda
            rec["_fcf_used"] = fcf
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: valuation metrics failed: {type(e).__name__}: {e}")

        # -- Quality metrics (5-9) --
        try:
            # 5. ROIC (Invested Capital = Equity + Total Debt - Excess Cash)
            # Use balance-sheet debt and cash for IC — all three IC
            # components (equity, debt, cash) then come from the same balance
            # sheet filing, for temporal consistency.
            #
            # Caveat, made live 2026-09-25: `_debt_bs`/`_cash_bs` fall back to
            # the `.info` figure when the filing carries no Total Debt / Cash
            # line, so the debt component may come from a different vintage.
            # The alternative is losing ROIC (weight 27) and
            # net_debt_to_ebitda (18) outright - 45 of 100 quality weight -
            # which is what happened to ANET and ISRG on every run until then
            # (ERIE is also rescuable here but stays NaN, its EBIT is missing
            # too).  The mixing error is bounded by construction: a filing that
            # omits Total Debt is a filing with little or no debt, and on those
            # names `.info` debt differs from the annual filing's long-term
            # debt by <= 1.2% of invested capital.
            #
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
                    _ic_floored = False
                    if pd.notna(ta) and ta > 0 and ic < 0.10 * ta:
                        ic = 0.10 * ta
                        _ic_floored = True
                    elif pd.notna(ta) and ta > 0:
                        ic = max(ic, 0.10 * ta)
                    rec["_roic_ic_floored"] = _ic_floored
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
                # The SAME EBITDA EV/EBITDA uses (2026-10-09, CLAUDE.md 0.9(c)): this block
                # had its own copy with the old `D&A >= 0` gate that Phase 13 (F36) removed
                # from the valuation block, so a stock whose D&A row carried a negative sign
                # was measured on two different EBITDAs. Identical for all 442 stocks with
                # both on 2026-10-09; one definition so they cannot drift apart.
                _ebitda_nd = rec.get("_ebitda_used", np.nan)
                if _ebitda_nd is None:
                    _ebitda_nd = np.nan
                rec["_ebitda_nd_used"] = _ebitda_nd
                # Net debt nets cash AND short-term investments, as enterprise value does
                # (2026-10-09 audit: with cash alone, 14 non-banks - MSFT, NVDA, GOOGL among
                # them - were net cash by the EV definition and net debt by this one).
                _cash_nd = _coalesce(d, "cash_sti_bs", "cash_bs", "totalCash")
                if pd.notna(_debt_bs) and pd.notna(_ebitda_nd) and _ebitda_nd > 0:
                    _net_debt = _debt_bs - (_cash_nd if pd.notna(_cash_nd) else 0.0)
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
            # Phase 13 (F22): (1) use ANNUAL EBIT & revenue for BOTH endpoints
            # (previously current EBIT was LTM, prior was annual — a basis
            # mismatch inside one ratio); (2) NaN when EBIT changes sign
            # year-over-year — DOL is undefined across a profit/loss transition,
            # and abs(prev) in the denominator otherwise produces a sign-scrambled
            # magnitude dominated by how close prior EBIT was to zero.
            if not _is_bank:
                _ebit_curr = _coalesce(d, "ebit_annual", "ebit")
                _ebit_prev = d.get("ebit_prior", np.nan)
                _rev_curr = d.get("totalRevenue_annual", rev_c)
                _rev_prev = d.get("totalRevenue_annual_prior", rev_p)
                _sign_flip = (pd.notna(_ebit_curr) and pd.notna(_ebit_prev)
                              and (_ebit_curr > 0) != (_ebit_prev > 0))
                if (pd.notna(_ebit_curr) and pd.notna(_ebit_prev) and _ebit_prev != 0
                        and pd.notna(_rev_curr) and pd.notna(_rev_prev) and _rev_prev > 0
                        and not _sign_flip):
                    _rev_pct_change = (_rev_curr - _rev_prev) / abs(_rev_prev)
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

            # Nine signals in Piotroski's order: 1 pass, 0 fail, None = untestable
            # (a missing input is untestable, not a fail). Recorded as a string so
            # the drilldown can show the nine behind the score.
            _sig = [None] * 9
            if pd.notna(ni):
                _sig[0] = int(ni > 0)
            if pd.notna(ocfv):
                _sig[1] = int(ocfv > 0)
            # Signals 3, 8 and 9 compare the latest FISCAL YEAR with the one before, as
            # Piotroski (2000) defines them; ROA and turnover use beginning-of-year assets.
            # Until 2026-10-09 they compared TTM flows with the fiscal year before last - a
            # 12-23 month span (research/2026-10-09-revenue-growth-window.md). Missing annual
            # inputs leave the signal untestable rather than falling back to that mix.
            _ni0, _ni1 = d.get("_ni_a0", np.nan), d.get("_ni_a1", np.nan)
            _gp0, _gp1 = d.get("_gp_a0", np.nan), d.get("_gp_a1", np.nan)
            _ta1, _ta2 = d.get("_ta_a1", np.nan), d.get("_ta_a2", np.nan)
            _rv0, _rv1 = d.get("totalRevenue_annual", np.nan), d.get("totalRevenue_annual_prior", np.nan)
            if all(pd.notna(x) for x in [_ni0, _ni1, _ta1, _ta2]) and _ta1 > 0 and _ta2 > 0:
                _sig[2] = int((_ni0 / _ta1) > (_ni1 / _ta2))
            if pd.notna(ocfv) and pd.notna(ni):
                _sig[3] = int(ocfv > ni)
            if all(pd.notna(x) for x in [ltd, ltd_p, ta, ta_p]) and ta > 0 and ta_p > 0:
                _sig[4] = int((ltd/ta) < (ltd_p/ta_p))
            if all(pd.notna(x) for x in [ca_c, cl_c, ca_p, cl_p]) and cl_c > 0 and cl_p > 0:
                _sig[5] = int((ca_c/cl_c) > (ca_p/cl_p))
            if pd.notna(sh) and pd.notna(sh_p):
                _sig[6] = int(sh <= sh_p)
            if all(pd.notna(x) for x in [_gp0, _gp1, _rv0, _rv1]) and _rv0 > 0 and _rv1 > 0:
                _sig[7] = int((_gp0 / _rv0) > (_gp1 / _rv1))
            if all(pd.notna(x) for x in [_rv0, _rv1, _ta1, _ta2]) and _ta1 > 0 and _ta2 > 0:
                _sig[8] = int((_rv0 / _ta1) > (_rv1 / _ta2))
            n_testable = sum(x is not None for x in _sig)
            f = sum(x for x in _sig if x is not None)
            rec["_pio_signals"] = "".join("-" if x is None else str(x) for x in _sig)
            # Use raw integer score (0-9).  Do NOT proportionally normalize -
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
            if (cfg or {}).get("enable_beneish", True) and _beneish_applies:
                _mscore, _mflag = _compute_beneish_mscore(d)
                rec["beneish_m_score"] = _mscore
                rec["_beneish_flag"] = _mflag
                # The eight indices behind the score, for the drilldown: "v1,..,v8|11110111"
                # (the mask marks indices computed from real data, 0 = neutral default).
                _bparts = _beneish_parts(d)
                if _bparts is not None:
                    rec["_beneish_idx"] = (",".join(f"{x:.4f}" for x in _bparts[0]) + "|"
                                           + "".join("1" if r else "0" for r in _bparts[1]))
            else:
                rec["beneish_m_score"] = np.nan
                rec["_beneish_flag"] = False
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: quality metrics failed: {type(e).__name__}: {e}")

        # -- Receivables outgrowing revenue (channel-stuffing flag) --
        # If receivables grow much faster than revenue, it may indicate aggressive revenue
        # recognition or channel stuffing. Both sides come from the same two annual statements
        # (fiscal year-end receivables, fiscal-year revenue); until 2026-10-09 revenue was the
        # trailing twelve months against usually the prior fiscal year - a 12-21 month window
        # beside a 12-month one. The flag is Beneish's (1999) days-sales-in-receivables index,
        # DSRI = (receivables / revenue) over the prior year's: its mean was 1.465 among his
        # earnings manipulators and 1.031 among the rest, so the flag fires at 1.465. It replaced
        # an unsourced "receivables growth more than 15pp above revenue growth" rule. Not for
        # bank-like stocks (receivables mean something else on a bank's balance sheet - the
        # reason Beneish skips them too). research/2026-10-09-trap-flags.md
        try:
            _recv_t = d.get("_beneish_net_receivables", np.nan)
            _recv_p = d.get("_beneish_net_receivables_p", np.nan)
            _rev_t = d.get("_beneish_revenue", np.nan)
            _rev_p = d.get("_beneish_revenue_p", np.nan)
            if (_beneish_applies and pd.notna(_recv_t) and pd.notna(_recv_p) and _recv_p > 0
                    and pd.notna(_rev_t) and _rev_t > 0 and pd.notna(_rev_p) and _rev_p > 0):
                _recv_growth = (_recv_t / _recv_p) - 1
                _rev_growth = (_rev_t / _rev_p) - 1
                rec["_recv_rev_divergence"] = _recv_growth - _rev_growth
                rec["_recv_growth"], rec["_rev_growth_fy"] = _recv_growth, _rev_growth
                rec["_dsri"] = (_recv_t / _rev_t) / (_recv_p / _rev_p)
                rec["_channel_stuffing_flag"] = bool(rec["_dsri"] >= DSRI_FLAG)
            else:
                rec["_recv_rev_divergence"] = np.nan
                rec["_dsri"] = np.nan
                rec["_channel_stuffing_flag"] = False
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            rec["_recv_rev_divergence"] = np.nan
            rec["_dsri"] = np.nan
            rec["_channel_stuffing_flag"] = False

        # -- Growth metrics (10-12) --
        try:
            # 10. Forward EPS Growth
            # Floor denominator at $1.00 to prevent near-zero trailing
            # EPS from producing extreme growth ratios (same principle
            # as the analyst_surprise $0.10 floor).  Clamp to configured
            # bounds (default [-75%, +300%]) because yfinance mixes GAAP
            # trailing EPS with normalised forward consensus.
            # Since 2026-10-09: MSCI's short-term forward EPS growth (Fundamental Data
            # Methodology, EGRSF): (EPS12F - EPS12B) / |EPS12B|, where EPS12F blends the
            # current- and next-fiscal-year consensus by the months M left in the current
            # fiscal year, (M*FY1 + (12-M)*FY2)/12, and EPS12B is the last four reported
            # quarters' actual EPS on the same (consensus) basis. Every company is measured
            # over the next 12 months. The old form - Yahoo's forwardEps (the year AFTER the
            # current one) over GAAP trailingEps - spanned 13-24 months by fiscal calendar and
            # mixed accounting bases (research/2026-10-09-forward-eps-growth.md). It remains
            # the fallback only where the consensus inputs are missing, with its F5 guard.
            _e1, _e2 = _estimate(d.get("_fy1_eps_current", np.nan)), _estimate(d.get("_fy2_eps_current", np.nan))
            _nfy = d.get("_next_fy_end", np.nan)
            _b12 = np.nan
            try:
                _q = json.loads(d["_eps_quarters"]) if isinstance(d.get("_eps_quarters"), str) else []
                _acts = [r[1] for r in _q[-4:]]
                # a stale history (newest quarter > EH_MAX_AGE_DAYS old) is not "the last four
                # quarters" - AMCR's ended Dec-2025 (2026-10-09 review)
                if len(_acts) == 4 and all(a is not None for a in _acts) and not d.get("_eps_quarters_stale"):
                    _b12 = float(sum(_acts))
            except (TypeError, ValueError, IndexError, KeyError):
                _b12 = np.nan
            _M = np.nan
            if pd.notna(_nfy):
                try:
                    _M = min(12.0, max(0.0, (datetime.fromtimestamp(int(_nfy), tz=timezone.utc).replace(tzinfo=None)
                                             - _as_of).days / 30.4375))
                except (TypeError, ValueError, OSError, OverflowError):
                    _M = np.nan
            fwd = d.get("forwardEps", np.nan)
            trail = d.get("trailingEps", np.nan)
            if all(pd.notna(x) for x in (_e1, _e2, _b12, _M)):
                _f12 = (_M * _e1 + (12.0 - _M) * _e2) / 12.0
                rec["_feg_f12"], rec["_feg_b12"], rec["_feg_m"] = float(_f12), float(_b12), float(_M)
                if _b12 <= 0:
                    # A growth rate from a loss has no meaning: its sign flips and its size is set
                    # by how small the loss was. GILD's last four quarters summed to -$0.39 (a
                    # one-off acquired-R&D charge) and scored +150%, the cap (2026-10-09; 8 stocks).
                    rec["forward_eps_growth"] = np.nan
                    rec["_feg_basis"] = "loss_base"
                else:
                    rec["forward_eps_growth"] = float(np.clip((_f12 - _b12) / max(abs(_b12), 1.0), feg_lo, feg_hi))
                    rec["_feg_basis"] = "msci_12m"
            elif pd.notna(fwd) and pd.notna(trail) and abs(trail) > 0.01:
                rec["_feg_basis"] = "fy2_over_trailing"
                # Phase 13 (F5): trailingEps is GAAP, forwardEps is normalized
                # consensus. When the two bases diverge extremely (ratio >2x or
                # <0.3x — the signature of large restructuring/impairment items
                # in trailing GAAP EPS), the growth ratio is contaminated and
                # manufactures fake growth. Route those rows to NaN so per-row
                # weight redistribution drops the contaminated signal rather than
                # scoring it. forward_eps_growth is 45% of the Growth category.
                _ratio = fwd / trail if abs(trail) > 1e-9 else np.nan
                if pd.notna(_ratio) and (_ratio > 2.0 or _ratio < 0.3):
                    rec["forward_eps_growth"] = np.nan
                else:
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
            _price = _coalesce(d, "currentPrice", "price_latest")
            _pe = (_price / trail) if (pd.notna(_price) and pd.notna(trail) and trail > 0.01) else np.nan
            _fwd_growth = rec.get("forward_eps_growth", np.nan)
            if pd.notna(_pe) and pd.notna(_fwd_growth) and _fwd_growth > 0:
                rec["peg_ratio"] = min(_pe / (_fwd_growth * 100), peg_max_cap)
            else:
                rec["peg_ratio"] = np.nan

            # 11. Revenue Growth (1-year), on a window that really is one year (2026-10-09).
            # Until then: TTM revenue over `totalRevenue_prior`, which for 502 of 502 stocks
            # fell back to the fiscal year *before* the latest one - a span of 12 to 23 months
            # depending on the fiscal calendar (median 18). Now: the latest quarter over the
            # same quarter a year earlier; where those are missing, the latest fiscal year
            # over the one before. Both are exactly 12 months. `_revg_basis` records which.
            _q0, _q4 = d.get("_rev_q0", np.nan), d.get("_rev_q4", np.nan)
            _a0, _a1 = d.get("totalRevenue_annual", np.nan), d.get("totalRevenue_annual_prior", np.nan)
            if pd.notna(_q0) and pd.notna(_q4) and _q4 > 0:
                rec["revenue_growth"] = (_q0 - _q4) / _q4
                rec["_revg_basis"] = "quarter"
            elif pd.notna(_a0) and pd.notna(_a1) and _a1 > 0:
                rec["revenue_growth"] = (_a0 - _a1) / _a1
                rec["_revg_basis"] = "annual"
            else:
                rec["revenue_growth"] = np.nan
                rec["_revg_basis"] = None

            # 11b. 3-Year Revenue CAGR — smoothed growth signal from annual filings.
            # Phase 13 (F20): use ANNUAL current (col=0) vs annual 3yr-ago (col=3)
            # so both endpoints share the same basis. Previously the numerator was
            # LTM (ending at the latest quarter) while the denominator was a full
            # fiscal year, so the flat 1/3 exponent systematically over/understated
            # CAGR. Fall back to LTM current only if annual current is unavailable.
            _rev_3yr = d.get("totalRevenue_3yr_ago", np.nan)
            _rev_cur_annual = d.get("totalRevenue_annual", np.nan)
            _rev_num = _rev_cur_annual if pd.notna(_rev_cur_annual) else rev_c
            if pd.notna(_rev_num) and pd.notna(_rev_3yr) and _rev_3yr > 0 and _rev_num > 0:
                rec["revenue_cagr_3yr"] = (_rev_num / _rev_3yr) ** (1.0 / 3.0) - 1.0
            else:
                rec["revenue_cagr_3yr"] = np.nan

            # 12. Sustainable Growth = ROE * Retention Ratio
            # ROE uses average equity (current + prior / 2) to smooth
            # single-year distortions from buybacks or one-time items.
            # Phase 13 (F21): the retention ratio now PREFERS cash dividends-paid
            # / net income (both GAAP, consistent with the GAAP netIncome in the
            # ROE numerator), and only falls back to the .info payoutRatio (which
            # yahoo derives from possibly-normalized EPS) as a last resort. The
            # old order preferred payoutRatio, making retention internally
            # inconsistent with ROE and contradicting the config's own
            # "cash-flow > accounting" rationale.
            # SGR clamped to [0%, 100%].
            if pd.notna(ni) and pd.notna(eq_v) and eq_v > 0 and ni > 0:
                eq_prior = d.get("totalEquity_prior", np.nan)
                if pd.notna(eq_prior) and eq_prior > 0:
                    avg_eq = (eq_v + eq_prior) / 2
                else:
                    avg_eq = eq_v  # fall back to single-year
                roe = ni / avg_eq

                # Retention: 1) cash dividendsPaid/NI, 2) dividendRate*shares/NI,
                # 3) .info payoutRatio (last resort).
                ret = None
                _divs_raw = d.get("dividendsPaid", np.nan)
                if pd.notna(_divs_raw):
                    ret = max(0, 1 - abs(_divs_raw) / ni)
                else:
                    _div_rate = d.get("dividendRate", np.nan)
                    _shares = d.get("sharesOutstanding", np.nan)
                    if pd.notna(_div_rate) and pd.notna(_shares):
                        ret = max(0, 1 - abs(_div_rate * _shares) / ni)
                if ret is None:
                    _payout = d.get("payoutRatio", np.nan)
                    if pd.notna(_payout) and 0 <= _payout <= 2.0:
                        ret = max(0, 1 - min(_payout, 1.0))

                if ret is not None:
                    sgr = roe * ret
                    rec["sustainable_growth"] = float(np.clip(sgr, 0.0, 1.0))
                else:
                    rec["sustainable_growth"] = np.nan
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
            #
            # DELIBERATELY single-source, and the only price site that is.
            # Both endpoints must come from the same `t.history(
            # auto_adjust=True)` series: `info["currentPrice"]` is not
            # split-adjusted against `price_12m_ago`, and dividing an
            # unadjusted price by an adjusted one across a split is exactly
            # the defect that published MNST at momentum 71.5 when its true
            # 12-1 return was the 3rd percentile (fixed 2026-08-26, see
            # check_price_series_integrity).  A `currentPrice` fallback stood
            # here until 2026-09-25 and was deleted rather than repaired: it
            # could never fire anyway, because `price_latest` and
            # `price_12m_ago` are written by the same `len(hist) >= 10` block,
            # so whenever the near endpoint is missing the far one is too.
            _p_now = _coalesce(d, "price_latest")
            rec["return_12m"] = ((_p_now - p12) / p12) if (pd.notna(p12) and pd.notna(_p_now) and p12 > 0) else np.nan
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: momentum metrics failed: {type(e).__name__}: {e}")

        # -- Risk metrics (15-16) --
        try:
            # 15. Volatility
            rec["volatility"] = d.get("volatility_1y", np.nan)
            # Published beside it as the engine's own figure (ENGINE_KEYS "vol_sd").
            rec["_vol_sd"] = d.get("_vol_daily_sd", np.nan)

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
                    # The two numbers the slope is, published so the page can show
                    # the division (metric_lineage.EQUATIONS["beta"]). Annualised
                    # (x252) only so they read as percentages-squared rather than
                    # 0.0001s; the ratio is unchanged by the common factor.
                    if var > 0:
                        rec["_beta_cov"] = float(cov * 252)
                        rec["_beta_var"] = float(var * 252)
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
            # Phase 13 (F35): require >=200 total daily observations (same gate
            # as volatility) before annualizing by sqrt(252). Previously a stock
            # with a thin history (recent IPO/spinoff, gappy yfinance series)
            # could get a Sortino from as few as 20 downside days, annualized as
            # if it were a full year, then ranked against full-history peers.
            _daily_all = None
            _daily_dates = None
            if dr and isinstance(dr, dict):
                # Sorted by date, not by insertion order: the drawdown below is
                # order-dependent (Sortino's is not), and relying on the fetch
                # having inserted chronologically is a silent dependency.  The
                # fetch does, so this is a no-op today.
                _daily_dates = sorted(dr.keys())
                _daily_all = np.array([dr[k] for k in _daily_dates])
            elif dr and isinstance(dr, list):
                _daily_all = np.array(dr)
            if _daily_all is not None and len(_daily_all) >= 200 and pd.notna(_ret_12m_sr):
                _daily_rf = (1 + risk_free_rate) ** (1/252) - 1
                _downside = _daily_all[_daily_all < _daily_rf] - _daily_rf
                if len(_downside) >= 20:
                    _dd = np.std(_downside, ddof=1) * np.sqrt(252)
                    rec["sortino_ratio"] = ((_ret_12m_sr - risk_free_rate) / _dd
                                            if _dd > 0 else np.nan)
                else:
                    rec["sortino_ratio"] = np.nan
            else:
                rec["sortino_ratio"] = np.nan

            # 16d. Max Drawdown (trailing 12 months)
            # Largest peak-to-trough fall of the *price path*, as a negative
            # fraction (-0.25 = a 25% fall).
            # Phase 13 (F35): require >=200 obs (was >=50) to match the vol/
            # Sortino gate and avoid ranking thin-history MDDs against full-year.
            #
            # 2026-10-08: the path is `exp(cumsum(log r))`, not the former
            # `cumprod(1 + log r)`.  `_daily_returns` holds **log** returns
            # (the fetch computes `log(close / close.shift(1))`), so compounding
            # them as if they were simple returns measured a series that is
            # neither the price path nor the log path: since ln(1+r) <= r it
            # drifts below the real path, and the drift is path-dependent, so
            # the peak-to-trough ratio taken on it was not the stock's actual
            # largest fall.  Measured over 50 real 13-month histories it
            # overstated the fall for 50 of 50 tickers - median 1.09pp, max
            # 4.70pp (AMD -32.46% against -27.76%) - and the bias grows with
            # volatility, so it fell hardest on exactly the stocks a tail-risk
            # metric is meant to separate.  Reproduce with
            # research/measurements/2026-10-08-max-drawdown-log-return-compounding.py;
            # METHODOLOGY_CHANGELOG.md 2026-10-08.
            #
            # `_mdd_peak` / `_mdd_trough` are the two points the fall is
            # measured between, published so the page can show the arithmetic
            # (metric_lineage.EQUATIONS["max_drawdown_1y"]).  They are rebased
            # to the window's first close so they read as prices; the ratio is
            # scale-invariant, so rebasing cannot change the metric.  They are
            # taken from this one computation - the page never recomputes them
            # (CLAUDE.md priority 0.8).
            if _daily_all is not None and len(_daily_all) >= 200:
                _cum = np.exp(np.cumsum(_daily_all))
                _peak = np.maximum.accumulate(_cum)
                _drawdowns = (_cum - _peak) / _peak
                _i = int(np.argmin(_drawdowns))
                rec["max_drawdown_1y"] = float(_drawdowns[_i])
                # The window's first close: `_cum` starts one day in, so the
                # close the series is rebased on is price_latest / _cum[-1].
                _p0 = d.get("price_latest", np.nan)
                _base = (_p0 / _cum[-1] if pd.notna(_p0) and _cum[-1] > 0 else 1.0)
                rec["_mdd_peak"] = float(_peak[_i] * _base)
                rec["_mdd_trough"] = float(_cum[_i] * _base)
                if _daily_dates is not None:
                    _j = int(np.argmax(_cum[: _i + 1]))
                    rec["_mdd_peak_date"] = str(_daily_dates[_j])
                    rec["_mdd_trough_date"] = str(_daily_dates[_i])
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
                # Every term of the CAPM line, from this one computation, so the
                # page prints the arithmetic instead of describing it
                # (metric_lineage.EQUATIONS["jensens_alpha"], CLAUDE.md 0.10a).
                rec["_ja_ret12"] = float(_ret_12m_ja)
                rec["_ja_rf"] = float(risk_free_rate)
                rec["_ja_beta"] = float(_beta_ja)
                rec["_ja_mkt"] = float(market_12m_return)
            else:
                rec["jensens_alpha"] = np.nan
                logging.debug(f"{ticker}: jensens_alpha skipped — beta or return NaN")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: Jensen's alpha failed: {type(e).__name__}: {e}")
            rec["jensens_alpha"] = np.nan

        # -- Revisions --
        try:
            rec["analyst_surprise"] = d.get("analyst_surprise", np.nan)
            rec["earnings_acceleration"] = d.get("earnings_acceleration", np.nan)
            rec["consecutive_beat_streak"] = d.get("consecutive_beat_streak", np.nan)
            rec["_eps_q"] = d.get("_eps_quarters")

            # FY1 consensus EPS revision over 90 days, scaled by price.
            # This is the category's only actual *revision* metric and, since
            # 2026-09-10, its heaviest (weight 35).  See
            # research/2026-09-07-revisions-category-has-no-revisions.md.
            #
            # Scaled by price, NOT by |estimate|, and the reason is not
            # cosmetic: on the full 502-name universe the estimate-scaled
            # denominator hits zero for at least one name, making its mean
            # literally +inf and its sd undefined (SS8.1).  Price scaling is
            # also CJL (1996)'s own construction.  It does NOT reduce the
            # correlation with momentum - SS8.3 measured five reconstructions,
            # including a sign-only variant with no denominator at all, and
            # every one carries ~0.32-0.43.  That overlap is economic
            # (Novy-Marx 2015), not an artifact of this formula; do not try to
            # "fix" it with a cleverer denominator without re-running SS8.3.
            _fy1_now = _estimate(d.get("_fy1_eps_current", np.nan))
            _fy1_then = _estimate(d.get("_fy1_eps_90d_ago", np.nan))
            # `_coalesce`, not `d.get(a, d.get(b))` - see its docstring.  This
            # site was guarded by hand on 2026-09-10; the helper generalised
            # that fix to the other eight two-source inputs on 2026-09-25.
            _rev_price = _coalesce(d, "currentPrice", "price_latest")
            if (pd.notna(_fy1_now) and pd.notna(_fy1_then)
                    and pd.notna(_rev_price) and _rev_price > 0):
                rec["fy1_revision_3m"] = float(
                    (_fy1_now - _fy1_then) / _rev_price)
            else:
                rec["fy1_revision_3m"] = np.nan

            # Price target upside: consensus analyst target vs current price.
            # Require >= 3 covering analysts for a meaningful consensus.
            # Clamped to [-50%, +100%] to guard against extreme targets.
            _target = d.get("targetMeanPrice", np.nan)
            _cur_price = _coalesce(d, "currentPrice", "price_latest")
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

        # -- Candidate metrics (Phase 11: Metric Evolution) --
        # Always computed even when weight=0; improvement engine evaluates IC.
        try:
            # C1. Proximity to 52-Week High (Momentum)
            # George & Hwang (2004): nearness to 52W high is one of the
            # strongest momentum signals. Ratio in [0, 1]; higher = nearer peak.
            _52w_high = d.get("fiftyTwoWeekHigh", np.nan)
            _cur_price_c = _coalesce(d, "currentPrice", "price_latest")
            rec["proximity_52w_high"] = (
                (_cur_price_c / _52w_high)
                if (pd.notna(_cur_price_c) and pd.notna(_52w_high) and _52w_high > 0)
                else np.nan
            )

            # C2. Operating Margin (Quality)
            # EBIT / Total Revenue. Uses LTM figures already fetched.
            _ebit_om = d.get("ebit", np.nan)
            _rev_om = d.get("totalRevenue", np.nan)
            if not _is_bank:
                rec["operating_margin"] = (
                    (_ebit_om / _rev_om)
                    if (pd.notna(_ebit_om) and pd.notna(_rev_om) and _rev_om > 0)
                    else np.nan
                )
            else:
                rec["operating_margin"] = np.nan  # Banks: skip

            # C3. Current Ratio (Quality)
            # Current Assets / Current Liabilities. MRQ figures already fetched.
            _ca_cr = d.get("currentAssets", np.nan)
            _cl_cr = d.get("currentLiabilities", np.nan)
            if not _is_bank:
                rec["current_ratio"] = (
                    (_ca_cr / _cl_cr)
                    if (pd.notna(_ca_cr) and pd.notna(_cl_cr) and _cl_cr > 0)
                    else np.nan
                )
            else:
                rec["current_ratio"] = np.nan  # Banks: meaningless

            # C4. Dividend Yield (Valuation)
            # dividendRate / currentPrice. Both already fetched.
            _div_rate_c = d.get("dividendRate", np.nan)
            rec["dividend_yield"] = (
                (_div_rate_c / _cur_price_c)
                if (pd.notna(_div_rate_c) and pd.notna(_cur_price_c) and _cur_price_c > 0)
                else np.nan
            )

            # C5. Insider Ownership (Quality)
            # heldPercentInsiders from .info. Higher = more skin in the game.
            _insider = d.get("heldPercentInsiders", np.nan)
            rec["insider_ownership"] = (
                _insider if (pd.notna(_insider) and 0 <= _insider <= 1) else np.nan
            )

            # C6. Short % of Float (Revisions)
            # shortPercentOfFloat from .info. Lower = less bearish = better.
            _short_pct = d.get("shortPercentOfFloat", np.nan)
            rec["short_pct_float"] = (
                _short_pct if (pd.notna(_short_pct) and _short_pct >= 0) else np.nan
            )

            # C7. Analyst Rating (Revisions)
            # recommendationMean from .info. 1=Strong Buy, 5=Sell.
            # Inverted in METRIC_DIR (lower = better).
            _rec_mean = d.get("recommendationMean", np.nan)
            rec["analyst_rating"] = (
                _rec_mean if (pd.notna(_rec_mean) and 1 <= _rec_mean <= 5) else np.nan
            )

            # C8. Interest Coverage (Quality)
            # EBIT / Interest Expense. Higher = more cushion.
            # Banks: skip (interest is their core business).
            _int_exp = d.get("interestExpense", np.nan)
            if not _is_bank:
                rec["interest_coverage"] = (
                    (_ebit_om / abs(_int_exp))
                    if (pd.notna(_ebit_om) and pd.notna(_int_exp) and abs(_int_exp) > 0)
                    else np.nan
                )
            else:
                rec["interest_coverage"] = np.nan

            # C9. Earnings variability (Quality candidate, 2026-10-09): standard deviation
            # of five fiscal years of ROE from each company's 10-K (SEC companyfacts), attached to the raw
            # record by run_screener (sec_fundamentals). Banks included - ROE is their
            # native profitability measure. research/2026-10-09-operating-leverage.md.
            rec["earnings_variability"] = d.get("_evol", np.nan)
            rec["_roe5"] = d.get("_roe5")
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warnings.warn(f"{ticker}: candidate metrics failed: {type(e).__name__}: {e}")

        # -- Passthrough: analyst price targets for dashboard display --
        rec["_current_price"]     = _coalesce(d, "currentPrice", "price_latest")
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
                    price = _coalesce(d, "currentPrice", "price_latest")
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
            stale_days = (cfg or {}).get("data_quality", {}).get(
                "stale_data_threshold_days", 120)  # Default 120 days (~4 months)
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
        # Why momentum/risk are blank for this name, when they are.
        rec["_price_series_rejected"] = d.get("_price_series_rejected", None)
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
# F. Flag outliers in the raw metrics (SS4.7) — report only, never clip
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
    "fy1_revision_3m",                                               # FY1 consensus EPS revision, 90d
    "analyst_surprise", "price_target_upside",
    "earnings_acceleration", "consecutive_beat_streak",              # fundamental momentum
    "short_interest_ratio",                                          # short interest sentiment
    "size_log_mcap",                                                 # size factor
    "asset_growth",                                                  # investment (CMA proxy)
    # Phase 11 candidate metrics (weight=0 until improvement engine activates)
    "proximity_52w_high",                                            # momentum candidate
    "operating_margin",                                              # quality candidate
    "current_ratio",                                                 # quality candidate
    "dividend_yield",                                                # valuation candidate
    "insider_ownership",                                             # quality candidate
    "short_pct_float",                                               # revisions candidate
    "analyst_rating",                                                # revisions candidate
    "interest_coverage",                                             # quality candidate
    "earnings_variability",                                          # quality candidate (2026-10-09)
]

# Metrics that only apply to bank-like or non-bank stocks.
# Used by the coverage filter to avoid penalizing stocks for
# structurally absent metrics.
_BANK_ONLY_METRICS = {"pb_ratio", "roe", "roa", "equity_ratio"}
_NONBANK_ONLY_METRICS = {"ev_ebitda", "ev_sales", "roic", "gross_profit_assets",
                         "net_debt_to_ebitda", "operating_leverage", "beneish_m_score",
                         "operating_margin", "current_ratio", "interest_coverage"}


def flag_metric_outliers(df: pd.DataFrame, lo: float = 0.01,
                         hi: float = 0.01) -> dict:
    """Report which metric values sit in the distribution tails. Never mutates.

    This replaced ``winsorize_metrics`` on 2026-09-01. Every column in
    ``METRIC_COLS`` is scored through ``compute_sector_percentiles``, which is
    ``Series.rank(pct=True)`` — a pure rank transform, and therefore invariant
    under *any* monotone transform of its input. Clipping the tails first could
    not change a single ordering. All it could do was collapse the clipped
    values onto one number, which ``rank`` then resolved to a shared average
    rank, and corrupt the value the dashboard publishes as the stock's ``raw``
    figure. See METHODOLOGY_CHANGELOG.md 2026-09-01.

    Outliers remain worth *knowing about* — an implausible value is usually a
    data defect, and clipping it hid exactly the signal that would have caught
    one (changelog 2026-08-26, MNST). So the tails are reported, not altered.

    Returns ``{column: {...}}`` for each metric with at least 10 non-null
    values, carrying the tail cut-offs and how many values fall at or beyond
    them. Metrics with fewer than 10 values are omitted: a tail is not
    meaningful there.
    """
    report: dict = {}
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        lo_cut = float(vals.quantile(lo))
        hi_cut = float(vals.quantile(1.0 - hi))
        n_low = int((vals <= lo_cut).sum())
        n_high = int((vals >= hi_cut).sum())
        if n_low == 0 and n_high == 0:
            continue
        report[col] = {
            "lo_cut": lo_cut,
            "hi_cut": hi_cut,
            "n_low": n_low,
            "n_high": n_high,
            "n_valid": int(len(vals)),
        }
    return report


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
    "fy1_revision_3m": True,                                 # upward consensus revision = better
    "analyst_surprise": True, "price_target_upside": True,
    "earnings_acceleration": True, "consecutive_beat_streak": True,  # fundamental momentum
    "short_interest_ratio": False,                                   # lower days-to-cover = less short pressure = better
    "size_log_mcap": True,                                # -log(mcap): higher = smaller = size premium
    "asset_growth": False,                                # lower asset growth = conservative investment = better
    # Phase 11 candidate metrics
    "proximity_52w_high": True,     # closer to 52W high = stronger momentum = better
    "operating_margin": True,        # higher margin = better quality
    "current_ratio": True,           # higher = better liquidity
    "dividend_yield": True,          # higher yield = better (for dividend strategy)
    "insider_ownership": True,       # more insider ownership = better alignment
    "short_pct_float": False,        # lower short interest = less bearish = better
    "analyst_rating": False,         # lower = more bullish (1=Strong Buy, 5=Sell)
    "interest_coverage": True,       # higher = more interest payment cushion = better
    "earnings_variability": False,   # lower = steadier ROE over five years = better (MSCI EVAR, AQR EVOL)
}


# A sector needs at least this many valid values for a metric before that metric is
# ranked within the sector; below it the stock is ranked against the whole universe
# instead. Shared with the dashboard so the page states the rule the scorer applied.
SECTOR_MIN_PEERS = 10


def _directed_pct(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """Percentile rank 0-100 with the same centre whichever way the metric points.

    ``rank(pct=True)`` runs from 1/n to 1, so flipping it for a lower-is-better metric gave 0 to
    1 - 1/n: a higher-is-better metric averaged 50 + 50/n and a lower-is-better one 50 - 50/n.
    With n the size of a sector, that tilted every composite by the sector's size and the
    weighted balance of metric directions - ~1.0 point for Energy against ~0.3 for Industrials
    (2026-10-09 audit). The midpoint rank, (rank - 0.5) / n, is symmetric: both directions
    average exactly 50, and flipping it is exact."""
    r = s.rank(method="average", na_option="keep")
    p = (r - 0.5) / s.notna().sum() * 100
    return p if higher_is_better else 100 - p


def compute_sector_percentiles(df: pd.DataFrame) -> pd.DataFrame:
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
        universe_ranks[col] = _directed_pct(df[col], METRIC_DIR.get(col, True))

    for _, grp in df.groupby("Sector"):
        for col in METRIC_COLS:
            pc = pct[col]
            if col not in df.columns:
                df.loc[grp.index, pc] = 50.0
                continue
            valid = grp[col].dropna()
            if len(valid) < SECTOR_MIN_PEERS:
                # Fall back to universe-wide ranking for this metric
                if col in universe_ranks:
                    df.loc[grp.index, pc] = universe_ranks[col].loc[grp.index]
                else:
                    df.loc[grp.index, pc] = 50.0
                continue
            ranks = _directed_pct(grp[col], METRIC_DIR.get(col, True))
            # NaN raw values → NaN percentile (not imputed to 50th).
            # The category score function handles per-row weight
            # redistribution for missing metrics.
            df.loc[grp.index, pc] = ranks
    return df


# =========================================================================
# G½. Optional non-linear percentile transform
# =========================================================================
def apply_percentile_transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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
    "valuation": ["ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales", "pb_ratio",
                   "dividend_yield"],
    "quality":   ["roic", "gross_profit_assets", "net_debt_to_ebitda",
                  "piotroski_f_score", "accruals", "operating_leverage",
                  "beneish_m_score", "roe", "roa", "equity_ratio",
                  "operating_margin", "current_ratio", "insider_ownership", "interest_coverage",
                  "earnings_variability"],
    "growth":    ["forward_eps_growth", "peg_ratio", "revenue_growth", "revenue_cagr_3yr", "sustainable_growth"],
    "momentum":  ["return_12_1", "return_6m", "jensens_alpha", "proximity_52w_high"],
    "risk":      ["volatility", "beta", "sharpe_ratio", "sortino_ratio", "max_drawdown_1y"],
    "revisions": ["fy1_revision_3m",
                  "analyst_surprise", "price_target_upside", "earnings_acceleration", "consecutive_beat_streak",
                  "short_interest_ratio", "short_pct_float", "analyst_rating"],
    "size":       ["size_log_mcap"],
    "investment": ["asset_growth"],
}


# Plain-English reason a stock is scored on something other than the generic
# metric weights. The dashboard prints these verbatim, so the sentence that
# explains a weight lives next to the code that applies it.
WEIGHT_PROFILE_LABELS = {
    "generic": "the standard metric weights",
    "bank": "bank weighting - banks and insurers are scored on P/B, ROE, ROA and "
            "equity ratio instead of EV-based and operating metrics",
    "pio_lowval": "Piotroski halved - its valuation score is below the "
                  "threshold, so the F-Score carries less weight and the freed "
                  "weight moves to other quality metrics",
    "pio_gt": "Piotroski halved - high growth with low quality, so the F-Score "
              "carries less weight and the freed weight moves to other quality metrics",
}


def metric_weight_profiles(cfg: dict, cat: str) -> dict:
    """Every metric-weight set the scorer can apply to a stock in ``cat``.

    Returns ``{profile_id: {metric: weight_as_a_fraction}}``. ``generic`` is always
    present; ``bank`` when the config gives banks their own weights for this
    category; ``pio_lowval`` / ``pio_gt`` for Quality when Piotroski conditional
    weighting is on.

    **This is the single place metric weights are resolved.** ``compute_category_scores``
    scores from these tables and the dashboard publishes the same tables, so the
    weights shown beside a score cannot differ from the weights that produced it.
    Until 2026-10-07 the page printed the generic weight for every stock while the
    scorer used bank, Piotroski-conditional and renormalised weights - the displayed
    arithmetic was wrong for 275 of 502 stocks (``plan/calculation-transparency.md``).
    """
    generic_ws = cfg["metric_weights"].get(cat, {})
    bank_mw = cfg.get("bank_metric_weights", None)
    metrics = CAT_METRICS[cat]

    profiles = {"generic": {m: generic_ws.get(m, 0) / 100.0 for m in metrics}}

    if bank_mw and cat in bank_mw:
        bank_ws = bank_mw[cat]
        profiles["bank"] = {m: bank_ws.get(m, 0) / 100.0 for m in metrics}

    pio_cfg = cfg.get("piotroski_conditional", {})
    if cat == "quality" and pio_cfg.get("enabled", False):
        reduction = pio_cfg.get("reduction_factor", 0.5)
        pio_w = generic_ws.get("piotroski_f_score", 0) / 100.0
        freed = pio_w * (1 - reduction)

        def _shifted(recipients):
            # Proportional redistribution: the freed weight is split in
            # proportion to the recipients' base weights, not equally.
            base = {m: generic_ws.get(m, 0) for m in recipients if generic_ws.get(m, 0) > 0}
            total = sum(base.values())
            shares = {m: w / total for m, w in base.items()} if total > 0 else {}
            table = {}
            for m in metrics:
                w_generic = generic_ws.get(m, 0) / 100.0
                if m == "piotroski_f_score":
                    table[m] = w_generic * reduction
                elif m in shares:
                    table[m] = w_generic + freed * shares[m]
                else:
                    table[m] = w_generic
            return table

        profiles["pio_lowval"] = _shifted(
            set(pio_cfg.get("redistribute_to", ["roic", "gross_profit_assets"])))
        if pio_cfg.get("growth_trap_enabled", False):
            profiles["pio_gt"] = _shifted(
                set(pio_cfg.get("growth_trap_redistribute_to",
                                ["accruals", "gross_profit_assets"])))
    return profiles


def published_weight_profiles(cfg: dict) -> dict:
    """``metric_weight_profiles`` for every category, as percentages, for the page.

    ``{category: {profile_id: {metric: weight_percent}}}``. Rounded to six decimals,
    which is far inside the 4dp the payload stores scores to.
    """
    return {
        cat: {pid: {m: round(w * 100.0, 6) for m, w in table.items()}
              for pid, table in metric_weight_profiles(cfg, cat).items()}
        for cat in CAT_METRICS
    }


def compute_category_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False)

    # Piotroski conditional weighting config
    pio_cfg = cfg.get("piotroski_conditional", {})
    pio_enabled = pio_cfg.get("enabled", False)
    pio_val_threshold = pio_cfg.get("valuation_threshold", 50)

    # Growth-trap Piotroski conditional config (Extension of above)
    pio_gt_enabled = pio_enabled and pio_cfg.get("growth_trap_enabled", False)
    pio_gt_growth_thr = pio_cfg.get("growth_trap_growth_threshold", 70)
    pio_gt_quality_thr = pio_cfg.get("growth_trap_quality_threshold", 35)

    for cat, metrics in CAT_METRICS.items():
        col = f"{cat}_score"
        # The weight tables for this category. Resolved once, in one function,
        # so the dashboard can publish exactly what is applied below.
        profiles = metric_weight_profiles(cfg, cat)

        # Which table each stock is scored with. Precedence is bank, then the
        # Piotroski low-valuation rule, then the growth-trap variant; the three
        # masks are disjoint by construction.
        profile = pd.Series("generic", index=df.index, dtype=object)
        if "bank" in profiles:
            profile[is_bank.astype(bool)] = "bank"

        if (cat == "quality" and pio_enabled
                and "valuation_score" in df.columns):
            is_low_val = (df["valuation_score"] < pio_val_threshold).fillna(False)
            # Only adjust non-bank rows (bank quality weights don't use ROIC/GPA)
            is_low_val = is_low_val & ~is_bank
            if is_low_val.any():
                profile[is_low_val] = "pio_lowval"

            # Growth-trap Piotroski: high-growth + low-quality non-bank stocks
            # get Piotroski weight halved, freed weight -> accruals + gross_profit_assets
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
                    profile[is_growth_trap_like] = "pio_gt"

        # Recorded so the dashboard can say which table each stock was scored
        # with. Internal column (leading underscore): never written to Excel.
        df[f"_wp_{cat}"] = profile

        # Per-row weighted average: only count metrics that have data.
        # NaN percentiles are excluded (not imputed to 50th), and each
        # row's score uses its own effective weight denominator.
        weighted_sum = pd.Series(0.0, index=df.index)
        weight_sum = pd.Series(0.0, index=df.index)
        skipped_metrics = []
        for m in metrics:
            pc = f"{m}_pct"
            w = pd.Series(profiles["generic"][m], index=df.index)
            for pid, table in profiles.items():
                if pid != "generic":
                    w[profile == pid] = table[m]

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
MOMENTUM_REGIME_SCALE = {"HIGH VOL": 0.70, "LOW VOL": 1.15}


def apply_momentum_regime(fw: dict, regime: str) -> dict:
    """The volatility-regime rule on a set of factor weights, as one pure function.

    HIGH VOL: momentum x0.70, the freed weight split between quality and valuation.
    LOW VOL: momentum x1.15, funded from valuation. NORMAL: unchanged. Used by
    ``adjust_momentum_weight`` for the run, and by the dashboard's investor profiles so
    a profile is weighted exactly as ``run_screener.py --preset <name>`` would weight it
    on the same day (2026-10-09, plan/investor-profiles.md)."""
    fw = dict(fw)
    mom_w = fw.get("momentum", 0)
    if regime == "HIGH VOL":
        scale = MOMENTUM_REGIME_SCALE[regime]
        freed = mom_w * (1 - scale)
        fw["momentum"] = round(mom_w * scale, 2)
        fw["quality"] = round(fw.get("quality", 0) + freed / 2, 2)
        fw["valuation"] = round(fw.get("valuation", 0) + freed / 2, 2)
    elif regime == "LOW VOL":
        scale = MOMENTUM_REGIME_SCALE[regime]
        added = mom_w * (scale - 1)
        fw["momentum"] = round(mom_w * scale, 2)
        fw["valuation"] = round(fw.get("valuation", 0) - added, 2)
    return fw


def infer_momentum_regime(base: dict, adjusted: dict) -> str:
    """Which regime turned ``base`` into ``adjusted`` (the run records both, not the name)."""
    for regime in MOMENTUM_REGIME_SCALE:
        if all(abs(apply_momentum_regime(base, regime).get(k, 0) - adjusted.get(k, 0)) < 1e-6
               for k in set(base) | set(adjusted)):
            return regime
    return "NORMAL"


def adjust_momentum_weight(df: pd.DataFrame, cfg: dict, root_dir: str) -> dict:
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
        except (OSError, csv.Error) as e:
            warnings.warn(f"[MOM-VOL] Could not read vol history {hist_path}: "
                          f"{type(e).__name__}: {e}")

    # Record this run - **one row per date**, replacing any row this date already
    # has rather than appending beside it.
    #
    # 2026-10-08: this appended unconditionally, so a day with two runs put two
    # observations of one day's data into the distribution `current_vol` is then
    # ranked against - and that percentile is what sets the momentum weight
    # below. Measured that day the file held 2026-02-21 **nine** times, 2026-02-24
    # five and 2026-07-28 four, out of 71 rows. Same defect, and the same fix, as
    # `improvement_engine.record_dispersion` and the snapshot directory
    # (CLAUDE.md priority 0.6); `tests/test_one_observation_per_run_date.py`
    # covers all three.
    today = date.today().isoformat()
    rows = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r", newline="") as f:
                rows = [r for r in csv.DictReader(f)
                        if str(r.get("date", "")) != today]
        except (OSError, csv.Error) as e:
            warnings.warn(f"[MOM-VOL] Could not rewrite vol history {hist_path}: "
                          f"{type(e).__name__}: {e}")
            rows = []
    try:
        with open(hist_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "momentum_vol"])
            for r in rows:
                writer.writerow([r.get("date", ""), r.get("momentum_vol", "")])
            writer.writerow([today, f"{current_vol:.4f}"])
    except OSError as e:
        warnings.warn(f"[MOM-VOL] Could not append vol history {hist_path}: "
                      f"{type(e).__name__}: {e}")

    # Switched off 2026-10-09 (config ``momentum_regime.enabled``). ``current_vol`` is the spread
    # of momentum_score ACROSS STOCKS, and momentum_score is built from within-sector percentile
    # ranks whose spread is fixed by construction: it moves only with how closely the three
    # momentum metrics' ranks agree, not with how volatile the market is (rank-predicted 25.03
    # against 25.07 measured; correlation with the S&P 500's realised volatility +0.37). It
    # called 30 of the 33 runs it acted on "LOW VOL" and never "HIGH VOL", so in practice it
    # raised momentum's weight 13 -> 14.95 most days. The history is still recorded.
    # research/2026-10-09-momentum-regime.md
    if not (cfg.get("momentum_regime") or {}).get("enabled", False):
        print(f"  [MOM-VOL] Score dispersion={current_vol:.2f} recorded; the regime rule is off (config momentum_regime.enabled).")
        return cfg

    # Need >= 20 historical observations to establish regime thresholds
    if len(hist_vols) < 20:
        print(f"  [MOM-VOL] Current vol={current_vol:.2f}, history={len(hist_vols)} runs (need 20+). Skipping regime scaling.")
        return cfg

    p25 = float(np.percentile(hist_vols, 25))
    p75 = float(np.percentile(hist_vols, 75))

    cfg = copy.deepcopy(cfg)
    if current_vol > p75:
        regime = "HIGH VOL"
    elif current_vol < p25:
        regime = "LOW VOL"
    else:
        regime = "NORMAL"
    cfg["factor_weights"] = apply_momentum_regime(cfg["factor_weights"], regime)
    fw = cfg["factor_weights"]

    print(f"  [MOM-VOL] vol={current_vol:.2f} | p25={p25:.2f} p75={p75:.2f} | Regime: {regime}")
    if regime != "NORMAL":
        print(f"  [MOM-VOL] Adjusted weights: momentum={fw['momentum']}, quality={fw.get('quality')}, valuation={fw.get('valuation')}")
    return cfg


def compute_effective_dimensionality(df: pd.DataFrame, cfg: dict) -> float:
    """Effective number of independent factors among the 8 category scores.

    Phase 13 (F4): computed from the eigenvalues of the category-score
    correlation matrix as (Σλ)² / Σλ² (participation ratio). ~8 means the
    categories are independent; a value near 3-4 confirms heavy redundancy.
    Reporting helper — does not alter ranking.
    """
    cat_cols = [f"{c}_score" for c in
                ["valuation", "quality", "growth", "momentum",
                 "risk", "revisions", "size", "investment"]]
    present = [c for c in cat_cols if c in df.columns]
    sub = df[present].dropna()
    if sub.shape[0] < 3 or sub.shape[1] < 2:
        return float(len(present))
    corr = sub.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 0]
    if eig.size == 0:
        return float(len(present))
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def neutralize_category_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Optionally orthogonalize the 8 category-score columns (Phase 13, F4).

    OFF by default (config factor_neutralization.enabled). When enabled with
    method 'gram_schmidt', each category (in configured order) is residualized
    against the earlier categories via OLS, then re-standardized back to the
    original 0-100 percentile scale so downstream weighting/labels are unchanged
    in RANGE (only the redundant, collinear component is removed). Earlier
    categories in the order keep their full signal. 'report_only' does nothing to
    the scores (diagnostics only). Returns df unchanged when disabled.
    """
    nz = cfg.get("factor_neutralization", {}) or {}
    if not nz.get("enabled", False):
        return df
    method = nz.get("method", "gram_schmidt")
    if method == "report_only":
        return df
    order = nz.get("order", ["valuation", "quality", "growth", "momentum",
                             "revisions", "risk", "size", "investment"])
    cols = [f"{c}_score" for c in order if f"{c}_score" in df.columns]
    if len(cols) < 2:
        return df

    def _rescale_like(orig: pd.Series, resid: pd.Series) -> pd.Series:
        # Map residuals back to the original column's rank scale so the
        # composite stays on a 0-100 basis (percentile of the residual).
        return resid.rank(pct=True) * 100

    built = []  # already-neutralized columns (as design matrix)
    for col in cols:
        y = df[col]
        mask = y.notna()
        if not built or mask.sum() < 5:
            df[col] = y  # first factor unchanged (or too few rows)
            built.append(col)
            continue
        X = df.loc[mask, built].fillna(df[built].mean())
        Xy = np.column_stack([np.ones(mask.sum()), X.to_numpy()])
        yv = y[mask].to_numpy()
        try:
            beta, *_ = np.linalg.lstsq(Xy, yv, rcond=None)
            resid = yv - Xy @ beta
            resid_s = pd.Series(np.nan, index=df.index)
            resid_s[mask] = resid
            df[col] = _rescale_like(y, resid_s)
        except np.linalg.LinAlgError:
            df[col] = y
        built.append(col)
    return df


# =========================================================================
# I. Composite score (SS3.2)
# =========================================================================
def weighted_metric_sets(cfg: dict) -> tuple[set, set]:
    """(metrics with weight for most stocks, metrics with weight for bank-like stocks).

    Read from ``metric_weight_profiles`` - the one place weights are resolved - so a
    metric counts exactly when it can move a score. A category with no bank table
    scores banks on the generic weights, as ``compute_category_scores`` does."""
    gen, bank = set(), set()
    for cat in CAT_METRICS:
        prof = metric_weight_profiles(cfg, cat)
        g = {m for m, w in prof["generic"].items() if w > 0}
        gen |= g
        bank |= ({m for m, w in prof["bank"].items() if w > 0} if "bank" in prof else g)
    return gen, bank


def applicable_coverage(df: pd.DataFrame, cfg: dict | None = None):
    """Per stock: (metrics present, metrics that apply to it).

    **Since 2026-10-09 "apply" means "carry weight in the table this stock is scored
    with"** (``weighted_metric_sets``). Before, it was every entry in ``METRIC_COLS``
    less the bank-only or non-bank-only ones - which counted 12 metrics that carry no
    weight at all (candidates, and Sharpe, Sortino, PEG, D/E), so a stock could be
    discounted for missing data that never enters its score (Loews, 2026-10-09), and
    adding a weight-0 candidate moved composites. Without ``cfg`` the old rule applies.

    This is the coverage the composite's discount reads, published as-is (the page
    used to show "N of 18" from a hard-coded list; ``plan/calculation-transparency.md``).
    """
    all_metrics = [c for c in METRIC_COLS if c in df.columns]
    is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    present = pd.Series(0, index=df.index)
    applicable = pd.Series(0, index=df.index)
    sets = weighted_metric_sets(cfg) if (cfg and cfg.get("metric_weights")) else None
    for m in all_metrics:
        if sets is not None:
            gen, bank = sets
            applies = pd.Series(np.where(is_bank, m in bank, m in gen), index=df.index)
        elif m in _BANK_ONLY_METRICS:
            applies = is_bank
        elif m in _NONBANK_ONLY_METRICS:
            applies = ~is_bank
        else:
            applies = pd.Series(True, index=df.index)
        if m == "beneish_m_score" and "_beneish_na" in df.columns:
            # Beneish is not computed for Financials (2026-10-09), so a generic-set financial
            # is not short of it.
            applies = applies & ~df["_beneish_na"].fillna(False).astype(bool)
        applicable += applies.astype(int)
        present += (df[m].notna() & applies).astype(int)
    return present, applicable


def compute_composite(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        df["Composite"] = pd.Series(dtype=float)
        return df
    # Phase 13 (F4): optional factor neutralization (OFF by default → no-op).
    df = neutralize_category_scores(df, cfg)
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
    # Always recorded, whether or not the discount is enabled, so the page can
    # state the coverage figure and the discount actually applied.
    _cov_present, _cov_applicable = applicable_coverage(df, cfg)
    df["_cov_present"] = _cov_present
    df["_cov_applicable"] = _cov_applicable
    df["_cov_discount"] = 0.0
    if cov_cfg.get("enabled", False):
        threshold = cov_cfg.get("threshold", 0.80)
        penalty_rate = cov_cfg.get("penalty_rate", 0.15)
        for idx in df.index:
            n_applicable = int(_cov_applicable.loc[idx])
            if n_applicable == 0:
                continue
            coverage = int(_cov_present.loc[idx]) / n_applicable
            if coverage < threshold:
                discount = (threshold - coverage) * penalty_rate
                df.at[idx, "Composite"] = df.at[idx, "Composite"] * (1 - discount)
                df.at[idx, "_cov_discount"] = discount

    # === Phase 13 (F1): preserve composite CARDINALITY =====================
    # Previously the cardinal weighted-average composite was OVERWRITTEN by its
    # own percentile rank (rank(pct=True)*100), which destroyed all magnitude —
    # #1-by-20pts and #1-by-0.1pts both mapped to 100.0, and the top-N was spaced
    # exactly 100/N apart. Conviction/edge information never reached portfolio
    # construction. We now KEEP the cardinal weighted-average as the ranking key
    # ("Composite") and expose the percentile as a SECONDARY display column
    # ("Composite_Pct", "better than X% of the universe"). Ranking, portfolio
    # score-weighting, and contribution waterfalls all now reconcile against a
    # true cardinal score.
    sector_relative = cfg.get("sector_neutral", {}).get("sector_relative_composite", False)
    if sector_relative and "Sector" in df.columns:
        # Sector-relative percentile as the DISPLAY column; cardinal composite
        # remains the ranking key.
        pct = pd.Series(np.nan, index=df.index)
        for _, grp in df.groupby("Sector"):
            if len(grp) >= 3:
                pct.loc[grp.index] = grp["Composite"].rank(pct=True) * 100
            else:
                pct.loc[grp.index] = 50.0
        df["Composite_Pct"] = pct.round(2)
    else:
        # Cross-sectional percentile rank as the DISPLAY column: 98.5 means
        # "better than 98.5% of the universe". Meaningful across runs, but it is
        # NOT the ranking key (that is the cardinal weighted-average below).
        df["Composite_Pct"] = (df["Composite"].rank(pct=True) * 100).round(2)

    # === Composite Confidence Score ===
    # Confidence is based on:
    # 1. Metric coverage (what % of applicable metrics are available)
    # 2. Category score variance (consistency across categories)
    # Confidence = coverage_ratio * (1 - normalized_variance)
    # Range: 0-100, where 100 is perfect coverage and perfect consistency
    is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    all_metrics = [c for c in METRIC_COLS if c in df.columns]
    
    confidence_scores = []
    cat_scores = [col_map[cat] for cat in col_map.keys() if col_map[cat] in df.columns]
    
    for idx in df.index:
        row_bank = is_bank.loc[idx] if idx in is_bank.index else False
        
        # Metric coverage ratio
        if row_bank:
            applicable = [m for m in all_metrics if m not in _NONBANK_ONLY_METRICS]
        else:
            applicable = [m for m in all_metrics if m not in _BANK_ONLY_METRICS]
        
        if len(applicable) > 0:
            n_present = sum(1 for m in applicable if pd.notna(df.at[idx, m]))
            coverage_ratio = n_present / len(applicable)
        else:
            coverage_ratio = 0.5  # Default midpoint
        
        # Category score variance (normalized)
        # Get all available category scores for this stock
        cat_vals = []
        for col in cat_scores:
            if col in df.columns and pd.notna(df.at[idx, col]):
                cat_vals.append(df.at[idx, col])
        
        if len(cat_vals) >= 2:
            # Normalize variance: std_dev / mean (coefficient of variation)
            # Invert so high consistency = high confidence penalty
            mean_val = np.mean(cat_vals)
            if mean_val > 0:
                cv = np.std(cat_vals) / mean_val
                # Sigmoid-like transformation: cv ranges 0-1+, we want consistency penalty 0-1
                variance_penalty = min(cv / (1 + cv), 1.0)  # 0 = perfect consistency, 1 = extreme variance
            else:
                variance_penalty = 0.5
        else:
            # If only 1 or 0 categories, no variance (use neutral)
            variance_penalty = 0.3 if len(cat_vals) == 1 else 0.5
        
        # Combine: coverage_ratio * (1 - variance_penalty)
        confidence = coverage_ratio * (1 - variance_penalty) * 100
        confidence_scores.append(max(0, min(100, confidence)))  # Clamp to 0-100
    
    df["Composite_Confidence"] = pd.Series(confidence_scores, index=df.index).round(1)

    df["Composite"] = df["Composite"].round(2)
    return df


# =========================================================================
# I-b. Factor contribution waterfall
# =========================================================================
def compute_factor_contributions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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
def apply_value_trap_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    vtf = cfg.get("value_trap_filters", {})
    if not vtf.get("enabled", True):
        df["Value_Trap_Flag"] = False
        return df

    qual_floor = vtf.get("quality_floor_percentile", 30) / 100.0
    mom_floor = vtf.get("momentum_floor_percentile", 30) / 100.0
    rev_floor = vtf.get("revisions_floor_percentile", 30) / 100.0
    cheap_floor = vtf.get("valuation_percentile", 70) / 100.0

    # Layer 0 - the stock must be cheap. A value trap is a cheap stock that is cheap for a
    # reason: Piotroski (2000) separates winners from losers *within* the highest
    # book-to-market stocks. Until 2026-10-09 this layer did not exist and the flag fired on
    # any broadly weak stock - 122 of 501, with a median valuation percentile of 0.51
    # (research/2026-10-09-trap-flags.md).
    if "valuation_score" in df.columns:
        cheap = df["valuation_score"].ge(df["valuation_score"].quantile(cheap_floor)).fillna(False)
    else:
        cheap = pd.Series(False, index=df.index)

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
    df["Value_Trap_Flag"] = cheap & ((l1.astype(int) + l2.astype(int) + l3.astype(int)) >= 2)

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


def apply_growth_trap_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flag high-growth stocks with weak fundamentals (growth traps).

    Mirror of value-trap logic but for the opposite scenario: a growth score above the
    ceiling percentile AND quality or revisions below its floor.
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

    # High growth is the defining condition, with weak quality or weak revisions beside it -
    # Mohanram (2005) separates winners from losers *within* low book-to-market (growth)
    # stocks. Until 2026-10-09 this was 2-of-3 with growth as one of the three, so a stock with
    # low growth, low quality and low revisions was called a growth trap: 34 of 125 flagged
    # were in the bottom half on growth (research/2026-10-09-trap-flags.md).
    df["Growth_Trap_Flag"] = g1 & (g2 | g3)

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

# An earnings history whose newest quarter ended longer ago than this is not scored: a quarter
# reports within ~90 days of its end, so 200 days means at least one report is missing.
EH_MAX_AGE_DAYS = 200

# Beneish (1999): mean days-sales-in-receivables index among the earnings manipulators in his
# sample (1.031 among non-manipulators). The channel-stuffing flag fires at or above it.
DSRI_FLAG = 1.465

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


# GICS sub-industries scored with the bank set: balance sheets whose liabilities are an operating
# input (deposits, insurance float, customer funds), where EV, EBITDA, ROIC and gross profit /
# assets lose their meaning and P/B against ROE is the practitioner standard (Damodaran,
# *Investment Valuation* ch. 21; Fama & French 1992 exclude financials for the same reason).
_BANK_SUBINDUSTRIES = {
    "Diversified Banks", "Regional Banks", "Consumer Finance",
    "Commercial & Residential Mortgage Finance", "Life & Health Insurance",
    "Multi-line Insurance", "Property & Casualty Insurance", "Reinsurance",
    "Multi-Sector Holdings", "Investment Banking & Brokerage", "Diversified Capital Markets",
}
# Fee businesses with conventional P&Ls, valued in practice on EV/EBITDA and P/E.
_GENERIC_FIN_SUBINDUSTRIES = {
    "Insurance Brokers", "Asset Management & Custody Banks", "Financial Exchanges & Data",
    "Transaction & Payment Processing Services",
}
# Inside "Asset Management & Custody Banks", the ones whose balance sheets are a bank's or an
# insurer's. Each needs its reason.
_BANK_OVERRIDE_TICKERS = {
    "BNY": "custody bank taking deposits",
    "BK": "custody bank taking deposits (the symbol before BNY)",
    "STT": "custody bank taking deposits",
    "NTRS": "custody bank taking deposits",
    "APO": "consolidates the insurer Athene",
    "KKR": "consolidates the insurer Global Atlantic",
    "AMP": "owns a bank and a life insurer; equity 3% of assets",
}
# Yahoo-only fallback (no GICS sub-industry): Yahoo files these under Asset Management.
_YAHOO_BANK_OVERRIDES = {"PFG", "RJF"}
_YAHOO_GENERIC_FIN_INDUSTRIES = {"Insurance Brokers", "Asset Management", "Financial Data & Stock Exchanges"}
# Tickers that reached the bank set only by the default for an unrecognised financial - logged
# by the run, and a test holds the current universe at none (26 did, silently, until 2026-10-09).
BANK_DEFAULTED: set = set()


def _is_bank_like(ticker: str, sector: str, industry: str, sub_industry: str | None = None) -> bool:
    """Whether a stock is scored with the bank metric set.

    With the GICS sub-industry (the run attaches it from the S&P 500 list, since 2026-10-09):
    the override tickers, then the sub-industry lists. Without it, Yahoo's industry, as before
    but with dashes normalised and fee businesses sent to the generic set. Either way an
    unrecognised Financials stock defaults to the bank set - the safer guess for an unseen
    lender - and is recorded in ``BANK_DEFAULTED``.
    research/2026-10-09-bank-like-financials.md
    """
    if sector not in _FINANCIAL_SECTORS:
        return False
    if isinstance(sub_industry, str) and sub_industry:
        if ticker in _BANK_OVERRIDE_TICKERS:
            return True
        if sub_industry in _BANK_SUBINDUSTRIES:
            return True
        if sub_industry in _GENERIC_FIN_SUBINDUSTRIES:
            return False
        BANK_DEFAULTED.add(ticker)
        return True
    if ticker in _NON_BANK_FINANCIALS:
        return False
    ind = (industry or "").replace("\u2014", " - ").strip()
    if ticker in _BANK_OVERRIDE_TICKERS or ticker in _YAHOO_BANK_OVERRIDES:
        return True
    if ind in _YAHOO_GENERIC_FIN_INDUSTRIES:
        return False
    if ind in _BANK_LIKE_INDUSTRIES or ind in ("Capital Markets", "Insurance - Reinsurance"):
        return True
    BANK_DEFAULTED.add(ticker)
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


def rank_stocks(df: pd.DataFrame) -> pd.DataFrame:
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

    # C/D. Attempt live data; refuse rather than fabricate if network blocked.
    #
    # This is a standalone entry point (`python factor_engine.py`), separate
    # from run_screener.py's own pipeline and its `--allow-synthetic` refusal
    # gate added 2026-08-11. That gate never covered this path, so running
    # this file directly still silently fabricated a full universe of
    # sector-realistic sample data with no warning - the exact 2026-08-06
    # credibility bug run_screener.py was fixed for, left open here because
    # nothing runs this file directly in the scheduled loops. Found 2026-09-01
    # while auditing for exactly this shape of gap. There is no supported way
    # to opt into sample data through this entry point - use
    # `python run_screener.py --tickers ... --allow-synthetic` instead, which
    # is the one place that path is intentional and labelled as such.
    print("\nTesting network connectivity...")
    try:
        test = fetch_single_ticker(tickers[0])
        if "_error" in test:
            raise RuntimeError(test["_error"])
        print("  Network OK — will fetch live data from yfinance")
    except Exception as e:
        print(f"  Network unavailable ({type(e).__name__}): {e}")
        print("  REFUSING to run: synthetic data would be indistinguishable from real.")
        print("  For sample-data pipeline testing, use:")
        print("    python run_screener.py --tickers AAPL,MSFT,GOOGL --allow-synthetic")
        sys.exit(2)

    # Reaching here means the network probe above succeeded, so this is
    # always live data now. USE_SAMPLE stays as a plain constant (rather than
    # deleting the branch below) to avoid a large, risk-for-no-reason
    # de-indent of the live-fetch path that follows it.
    USE_SAMPLE = False
    skipped_tickers = []

    if USE_SAMPLE:
        # Unreachable - the refusal above exits before this can ever be True.
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

        # Coverage filter — Phase 13 (F41): count only metrics applicable to
        # each stock type. Bank-only metrics are structurally NaN for non-banks
        # (and vice versa); including them in the denominator penalized banks in
        # this legacy path, diverging from the corrected run_screener.py filter.
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
        df["_mc"] = metric_count
        min_needed = (applicable_count * coverage_pct).apply(lambda x: max(1, int(x)))
        low = df["_mc"] < min_needed
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

    # F. Flag outliers (report only — the values are scored as fetched)
    print("Flagging outliers beyond the 1st / 99th percentiles...")
    _outliers = flag_metric_outliers(df, 0.01, 0.01)
    print(f"  {len(_outliers)} metric(s) have values in the tails")

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
