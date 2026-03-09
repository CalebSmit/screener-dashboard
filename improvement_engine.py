#!/usr/bin/env python3
"""
Self-Improving Screener Engine
==============================
Tracks forward returns of screener recommendations, computes live
Information Coefficients (IC), proposes weight changes, and applies
approved modifications to config.yaml.

Architecture:
  - Phase A: Forward return tracking (record_run_snapshot + compute_forward_returns)
  - Phase B: Live IC computation (compute_live_ic + analyze_ic_trends)
  - Phase C: Weight optimization (propose_weight_changes + propose_metric_weight_changes)
  - Phase D: Regime detection (compute_dispersion + detect_regime)
  - Phase E: Reporting & approval (generate_improvement_report + apply_changes)
"""

import copy
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

logger = logging.getLogger("screener.improvement")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
IMPROVEMENT_DIR = ROOT / "improvement"
SNAPSHOTS_DIR = IMPROVEMENT_DIR / "snapshots"
PRICE_CACHE_DIR = IMPROVEMENT_DIR / "price_cache"
PROPOSALS_DIR = IMPROVEMENT_DIR / "proposals"
VALIDATION_DIR = ROOT / "validation"

for _d in (IMPROVEMENT_DIR, SNAPSHOTS_DIR, PRICE_CACHE_DIR, PROPOSALS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORY_SCORES = [
    "valuation_score", "quality_score", "growth_score", "momentum_score",
    "risk_score", "revisions_score", "size_score", "investment_score",
]
CATEGORY_NAMES = [c.replace("_score", "") for c in CATEGORY_SCORES]

CATEGORY_BOUNDS = {
    "valuation":  (5.0, 35.0),
    "quality":    (5.0, 35.0),
    "growth":     (3.0, 25.0),
    "momentum":   (3.0, 25.0),
    "risk":       (2.0, 20.0),
    "revisions":  (2.0, 20.0),
    "size":       (0.0, 15.0),
    "investment": (0.0, 15.0),
}

MAX_CHANGE_PER_CYCLE = 3.0       # ±3% max per category per improvement cycle
MAX_METRIC_CHANGE_PER_CYCLE = 5.0  # ±5% max per metric per improvement cycle
AUTO_APPLY_THRESHOLD = 2.0        # Auto-apply if all changes ≤ this

PERFORMANCE_HISTORY_PATH = IMPROVEMENT_DIR / "performance_history.csv"
LIVE_IC_HISTORY_PATH = IMPROVEMENT_DIR / "live_ic_history.csv"
DISPERSION_HISTORY_PATH = IMPROVEMENT_DIR / "dispersion_history.csv"
CHANGE_LOG_PATH = IMPROVEMENT_DIR / "change_log.csv"
METRIC_IC_HISTORY_PATH = IMPROVEMENT_DIR / "metric_ic_history.csv"

PERF_COLUMNS = [
    "run_date", "ticker", "composite_score", "rank", "in_portfolio",
    *CATEGORY_SCORES,
    "fwd_return_1w", "fwd_return_1m", "fwd_return_3m",
]

IC_COLUMNS = [
    "run_date", "horizon", "n_tickers",
    "composite_ic", *[f"{c}_ic" for c in CATEGORY_NAMES],
]


# =========================================================================
# Phase A: Forward Return Tracking
# =========================================================================

def record_run_snapshot(
    run_id: str,
    run_date: str,
    scored_df: pd.DataFrame,
    portfolio_df: pd.DataFrame | None,
    cfg: dict,
) -> Path:
    """Record a snapshot of this run's scores for future return tracking.

    Saves to improvement/snapshots/{run_date}_{run_id}.parquet.
    Also triggers forward return computation for prior snapshots.
    Also records dispersion for regime detection.
    """
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build snapshot with scores + portfolio membership + price
    keep_cols = ["Ticker", "Sector", "Composite", "Rank"] + CATEGORY_SCORES
    # Phase 11: Include metric percentile columns for per-metric IC computation
    pct_cols = [c for c in scored_df.columns if c.endswith("_pct")]
    keep_cols.extend(pct_cols)
    available = [c for c in keep_cols if c in scored_df.columns]
    snap = scored_df[available].copy()

    # Mark portfolio membership
    if portfolio_df is not None and "Ticker" in portfolio_df.columns:
        port_tickers = set(portfolio_df["Ticker"].tolist())
        snap["in_portfolio"] = snap["Ticker"].isin(port_tickers)
    else:
        snap["in_portfolio"] = False

    # Store current price if available
    if "_current_price" in scored_df.columns:
        snap["price_at_scoring"] = scored_df["_current_price"]
    elif "price_latest" in scored_df.columns:
        snap["price_at_scoring"] = scored_df["price_latest"]

    snap["run_date"] = run_date
    snap["run_id"] = run_id

    path = SNAPSHOTS_DIR / f"{run_date}_{run_id}.parquet"
    snap.to_parquet(path, index=False)
    logger.info(f"Improvement snapshot saved: {path.name} ({len(snap)} tickers)")

    # Record dispersion for regime detection
    try:
        disp = compute_dispersion(scored_df)
        record_dispersion(run_date, disp)
    except Exception as e:
        logger.warning(f"Dispersion recording failed: {e}")

    # Compute forward returns for prior snapshots
    try:
        new_rows = compute_forward_returns(run_date)
        if new_rows is not None and len(new_rows) > 0:
            logger.info(f"Computed forward returns for {len(new_rows)} ticker-date pairs")
    except Exception as e:
        logger.warning(f"Forward return computation failed: {e}")

    return path


def compute_forward_returns(current_date: str) -> pd.DataFrame | None:
    """Compute forward returns for prior snapshots that are old enough.

    For each prior snapshot from date D:
      - 1w return: computed if current_date >= D + 7 days
      - 1m return: computed if current_date >= D + 30 days
      - 3m return: computed if current_date >= D + 90 days

    Returns newly computed rows (appended to performance_history.csv).
    """
    current_dt = pd.Timestamp(current_date)

    # Find all snapshots
    snapshot_files = sorted(SNAPSHOTS_DIR.glob("*.parquet"))
    if not snapshot_files:
        return None

    # Load existing performance history to avoid recomputation
    existing_dates = set()
    if PERFORMANCE_HISTORY_PATH.exists():
        try:
            existing = pd.read_csv(PERFORMANCE_HISTORY_PATH)
            existing_dates = set(existing["run_date"].unique())
        except Exception:
            pass

    new_rows = []
    for snap_path in snapshot_files:
        # Extract date from filename: YYYY-MM-DD_runid.parquet
        snap_date_str = snap_path.stem.split("_")[0]
        try:
            snap_dt = pd.Timestamp(snap_date_str)
        except Exception:
            continue

        # Skip if already processed
        if snap_date_str in existing_dates:
            continue

        # Need at least 7 calendar days for 1w return
        days_elapsed = (current_dt - snap_dt).days
        if days_elapsed < 7:
            continue

        # Load snapshot
        try:
            snap = pd.read_parquet(snap_path)
        except Exception:
            continue

        tickers = snap["Ticker"].dropna().unique().tolist()
        if not tickers:
            continue

        # Fetch prices for forward return windows
        prices = _fetch_prices_for_returns(
            tickers,
            snap_date_str,
            current_date,
        )
        if prices is None or prices.empty:
            continue

        # Phase 11: detect metric percentile columns in this snapshot
        pct_cols_in_snap = [c for c in snap.columns if c.endswith("_pct")]

        for _, row in snap.iterrows():
            ticker = row.get("Ticker")
            if ticker not in prices.columns:
                continue

            ticker_prices = prices[ticker].dropna()
            if ticker_prices.empty:
                continue

            # Find price at scoring date (nearest available)
            base_price = _nearest_price(ticker_prices, snap_dt)
            if base_price is None or base_price <= 0:
                continue

            # Compute returns for each horizon
            fwd_1w = _compute_fwd_return(ticker_prices, snap_dt, 7, base_price)
            fwd_1m = _compute_fwd_return(ticker_prices, snap_dt, 30, base_price) if days_elapsed >= 30 else np.nan
            fwd_3m = _compute_fwd_return(ticker_prices, snap_dt, 90, base_price) if days_elapsed >= 90 else np.nan

            if np.isnan(fwd_1w):
                continue

            new_rows.append({
                "run_date": snap_date_str,
                "ticker": ticker,
                "composite_score": row.get("Composite", np.nan),
                "rank": row.get("Rank", np.nan),
                "in_portfolio": row.get("in_portfolio", False),
                **{cs: row.get(cs, np.nan) for cs in CATEGORY_SCORES},
                # Phase 11: metric percentile columns for per-metric IC
                **{col: row.get(col, np.nan) for col in pct_cols_in_snap},
                "fwd_return_1w": fwd_1w,
                "fwd_return_1m": fwd_1m,
                "fwd_return_3m": fwd_3m,
            })

    if not new_rows:
        return None

    new_df = pd.DataFrame(new_rows)

    # Append to performance history
    if PERFORMANCE_HISTORY_PATH.exists():
        existing = pd.read_csv(PERFORMANCE_HISTORY_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(PERFORMANCE_HISTORY_PATH, index=False)
    return new_df


def _fetch_prices_for_returns(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Fetch daily close prices from yfinance. Caches in price_cache/."""
    cache_key = f"{start_date}_{end_date}"
    cache_path = PRICE_CACHE_DIR / f"prices_{cache_key}.parquet"

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            # Check if we have enough tickers
            missing = set(tickers) - set(cached.columns)
            if len(missing) < len(tickers) * 0.1:  # < 10% missing
                return cached
        except Exception:
            pass

    try:
        import yfinance as yf
        # Extend range slightly for nearest-price lookups
        start_dt = pd.Timestamp(start_date) - timedelta(days=5)
        end_dt = pd.Timestamp(end_date) + timedelta(days=5)
        data = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]]
            prices.columns = tickers[:1]

        prices.to_parquet(cache_path)
        return prices
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
        return None


def _nearest_price(prices: pd.Series, target_dt: pd.Timestamp) -> float | None:
    """Find the nearest available price to a target date (within 5 days)."""
    prices = prices.dropna()
    if prices.empty:
        return None

    # Ensure index is Timestamp
    prices.index = pd.DatetimeIndex(prices.index)

    # Find nearest date within 5 business days
    diffs = abs(prices.index - target_dt)
    min_idx = diffs.argmin()
    if diffs[min_idx].days > 7:
        return None
    return float(prices.iloc[min_idx])


def _compute_fwd_return(
    prices: pd.Series,
    base_dt: pd.Timestamp,
    days_forward: int,
    base_price: float,
) -> float:
    """Compute forward return from base_dt + days_forward."""
    target_dt = base_dt + timedelta(days=days_forward)
    fwd_price = _nearest_price(prices, target_dt)
    if fwd_price is None or fwd_price <= 0:
        return np.nan
    return (fwd_price - base_price) / base_price


def backfill_from_existing_runs() -> int:
    """Process existing runs/ artifacts to bootstrap performance_history.csv.

    Returns the number of snapshots created.
    """
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return 0

    count = 0
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "meta.json"
        scored_path = run_dir / "05_final_scored.parquet"
        portfolio_path = run_dir / "08_model_portfolio.parquet"

        if not meta_path.exists() or not scored_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())
            run_id = meta.get("run_id", run_dir.name)
            start_time = meta.get("start_time", "")
            if not start_time:
                continue
            run_date = start_time[:10]  # YYYY-MM-DD

            # Skip if snapshot already exists
            existing = list(SNAPSHOTS_DIR.glob(f"{run_date}_{run_id}*"))
            if existing:
                continue

            scored_df = pd.read_parquet(scored_path)
            portfolio_df = None
            if portfolio_path.exists():
                portfolio_df = pd.read_parquet(portfolio_path)

            # Record snapshot (without triggering forward returns yet)
            keep_cols = ["Ticker", "Sector", "Composite", "Rank"] + CATEGORY_SCORES
            available = [c for c in keep_cols if c in scored_df.columns]
            snap = scored_df[available].copy()

            if portfolio_df is not None and "Ticker" in portfolio_df.columns:
                port_tickers = set(portfolio_df["Ticker"].tolist())
                snap["in_portfolio"] = snap["Ticker"].isin(port_tickers)
            else:
                snap["in_portfolio"] = False

            snap["run_date"] = run_date
            snap["run_id"] = run_id

            path = SNAPSHOTS_DIR / f"{run_date}_{run_id}.parquet"
            snap.to_parquet(path, index=False)
            count += 1
            logger.info(f"Backfilled snapshot: {path.name}")
        except Exception as e:
            logger.warning(f"Backfill failed for {run_dir.name}: {e}")

    # Now compute forward returns for all backfilled snapshots
    if count > 0:
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            compute_forward_returns(today)
        except Exception as e:
            logger.warning(f"Backfill forward return computation failed: {e}")

    return count


# =========================================================================
# Phase B: Live IC Computation
# =========================================================================

def compute_live_ic(horizon: str = "1m") -> pd.DataFrame | None:
    """Compute Information Coefficient per category score vs realized returns.

    For each run_date, computes Spearman rank IC between category scores
    and forward returns at the specified horizon.

    Args:
        horizon: "1w", "1m", or "3m"

    Returns DataFrame of IC values per run_date, or None if no data.
    """
    if not PERFORMANCE_HISTORY_PATH.exists():
        return None

    perf = pd.read_csv(PERFORMANCE_HISTORY_PATH)
    return_col = f"fwd_return_{horizon}"
    if return_col not in perf.columns:
        return None

    # Filter to rows with non-NaN returns
    valid = perf.dropna(subset=[return_col])
    if valid.empty:
        return None

    results = []
    for run_date, group in valid.groupby("run_date"):
        if len(group) < 30:  # Need enough tickers for meaningful IC
            continue

        row = {"run_date": run_date, "horizon": horizon, "n_tickers": len(group)}

        # Composite IC
        returns = group[return_col].values
        composite = group["composite_score"].values
        mask = ~(np.isnan(composite) | np.isnan(returns))
        if mask.sum() >= 30:
            ic, _ = spearmanr(composite[mask], returns[mask])
            row["composite_ic"] = ic
        else:
            row["composite_ic"] = np.nan

        # Per-category IC
        for cat in CATEGORY_NAMES:
            score_col = f"{cat}_score"
            if score_col not in group.columns:
                row[f"{cat}_ic"] = np.nan
                continue
            scores = group[score_col].values
            mask = ~(np.isnan(scores) | np.isnan(returns))
            if mask.sum() >= 30:
                ic, _ = spearmanr(scores[mask], returns[mask])
                row[f"{cat}_ic"] = ic
            else:
                row[f"{cat}_ic"] = np.nan

        results.append(row)

    if not results:
        return None

    ic_df = pd.DataFrame(results)

    # Save/append to live IC history
    if LIVE_IC_HISTORY_PATH.exists():
        existing = pd.read_csv(LIVE_IC_HISTORY_PATH)
        # Remove duplicates by run_date + horizon
        existing_keys = set(
            existing.apply(lambda r: f"{r['run_date']}_{r['horizon']}", axis=1)
        )
        new_rows = ic_df[
            ~ic_df.apply(lambda r: f"{r['run_date']}_{r['horizon']}", axis=1).isin(existing_keys)
        ]
        if not new_rows.empty:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(LIVE_IC_HISTORY_PATH, index=False)
    else:
        ic_df.to_csv(LIVE_IC_HISTORY_PATH, index=False)

    return ic_df


def analyze_ic_trends(
    lookback_months: int = 12,
    ewm_halflife: int = 6,
) -> dict:
    """Analyze IC trends across categories.

    Returns dict per category with: mean_ic, ewm_ic, trend, ic_ir, pct_positive.
    Returns empty dict with _warning if insufficient data.
    """
    if not LIVE_IC_HISTORY_PATH.exists():
        return {"_warning": "No live IC history available. Run more screener cycles."}

    ic_df = pd.read_csv(LIVE_IC_HISTORY_PATH)
    # Use 1m horizon by default; fall back to 1w if no 1m data
    ic_1m = ic_df[ic_df["horizon"] == "1m"].sort_values("run_date")
    horizon_used = "1m"
    if len(ic_1m) == 0:
        ic_1m = ic_df[ic_df["horizon"] == "1w"].sort_values("run_date")
        horizon_used = "1w"

    n_obs = len(ic_1m)
    if n_obs < 6:
        return {
            "_warning": f"Only {n_obs} observations (need 6+ for trends). "
                        f"Keep running the screener daily.",
            "_n_observations": n_obs,
        }

    result = {"_n_observations": n_obs}

    for cat in CATEGORY_NAMES:
        ic_col = f"{cat}_ic"
        if ic_col not in ic_1m.columns:
            continue

        series = ic_1m[ic_col].dropna()
        if len(series) < 4:
            continue

        # Raw stats
        mean_ic = series.mean()
        std_ic = series.std()
        ic_ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
        pct_positive = (series > 0).mean()

        # EWM IC (more recent observations weighted higher)
        ewm_ic = series.ewm(halflife=ewm_halflife, min_periods=3).mean().iloc[-1]

        # Rolling windows for trend detection
        n = len(series)
        mean_3m = series.tail(min(3, n)).mean() if n >= 3 else np.nan
        mean_6m = series.tail(min(6, n)).mean() if n >= 6 else np.nan
        mean_12m = series.tail(min(12, n)).mean() if n >= 12 else np.nan

        # Trend classification
        trend = "stable"
        if n >= 12:
            first_half = series.iloc[:n // 2].mean()
            second_half = series.iloc[n // 2:].mean()
            delta = second_half - first_half
            if delta > 0.02:
                trend = "improving"
            elif delta < -0.02:
                trend = "declining"

        result[cat] = {
            "mean_ic": round(mean_ic, 4),
            "ewm_ic": round(ewm_ic, 4),
            "mean_ic_3m": round(mean_3m, 4) if not np.isnan(mean_3m) else None,
            "mean_ic_6m": round(mean_6m, 4) if not np.isnan(mean_6m) else None,
            "mean_ic_12m": round(mean_12m, 4) if not np.isnan(mean_12m) else None,
            "ic_trend": trend,
            "ic_ir": round(ic_ir, 3),
            "pct_positive": round(pct_positive, 3),
            "n_observations": len(series),
        }

    return result


def merge_historical_and_live_ic() -> pd.DataFrame | None:
    """Combine biased historical IC with unbiased live IC for context.

    Historical IC from factor_ic_timeseries.csv is labeled as BIASED.
    Live IC from live_ic_history.csv is labeled as UNBIASED.
    """
    frames = []

    # Historical (biased)
    hist_path = VALIDATION_DIR / "factor_ic_timeseries.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["source"] = "historical_BIASED"
        hist["horizon"] = "1m"
        # Rename columns to match
        rename_map = {
            "month": "run_date",
            "Composite": "composite_ic",
        }
        for cat in CATEGORY_NAMES:
            old_col = f"{cat}_score"
            if old_col in hist.columns:
                rename_map[old_col] = f"{cat}_ic"
        hist = hist.rename(columns=rename_map)
        frames.append(hist)

    # Live (unbiased)
    if LIVE_IC_HISTORY_PATH.exists():
        live = pd.read_csv(LIVE_IC_HISTORY_PATH)
        live["source"] = "live_UNBIASED"
        frames.append(live)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


# =========================================================================
# Phase B2: Metric-Level IC Computation (Phase 11)
# =========================================================================

def compute_metric_level_ic(
    horizon: str = "1m",
    min_tickers: int = 30,
) -> pd.DataFrame | None:
    """Compute per-metric Information Coefficient vs forward returns.

    For each run_date, computes Spearman rank IC between each metric's
    sector-relative percentile and realized forward returns.

    Returns DataFrame with columns: run_date, horizon, n_tickers,
    {metric}_ic for each metric, or None if insufficient data.
    """
    if not PERFORMANCE_HISTORY_PATH.exists():
        return None

    perf = pd.read_csv(PERFORMANCE_HISTORY_PATH)
    return_col = f"fwd_return_{horizon}"
    if return_col not in perf.columns:
        return None

    # Check which metric _pct columns are available
    available_pct = [c for c in perf.columns if c.endswith("_pct")]
    if not available_pct:
        return None

    valid = perf.dropna(subset=[return_col])
    if valid.empty:
        return None

    results = []
    for run_date, group in valid.groupby("run_date"):
        if len(group) < min_tickers:
            continue

        row = {"run_date": run_date, "horizon": horizon, "n_tickers": len(group)}
        returns = group[return_col].values

        for pct_col in available_pct:
            metric_name = pct_col.replace("_pct", "")
            scores = group[pct_col].values
            mask = ~(np.isnan(scores) | np.isnan(returns))
            if mask.sum() >= min_tickers:
                ic, _ = spearmanr(scores[mask], returns[mask])
                row[f"{metric_name}_ic"] = ic
            else:
                row[f"{metric_name}_ic"] = np.nan

        results.append(row)

    if not results:
        return None

    ic_df = pd.DataFrame(results)

    # Save/append to metric IC history
    if METRIC_IC_HISTORY_PATH.exists():
        existing = pd.read_csv(METRIC_IC_HISTORY_PATH)
        existing_keys = set(
            existing.apply(lambda r: f"{r['run_date']}_{r['horizon']}", axis=1)
        )
        new_rows = ic_df[
            ~ic_df.apply(lambda r: f"{r['run_date']}_{r['horizon']}", axis=1).isin(existing_keys)
        ]
        if not new_rows.empty:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(METRIC_IC_HISTORY_PATH, index=False)
    else:
        ic_df.to_csv(METRIC_IC_HISTORY_PATH, index=False)

    return ic_df


def analyze_metric_ic_trends(
    category: str | None = None,
    ewm_halflife: int = 6,
) -> dict:
    """Analyze per-metric IC trends, optionally filtered by category.

    Returns dict keyed by metric name with:
      mean_ic, ewm_ic, ic_trend, pct_positive, n_observations,
      consecutive_positive, consecutive_nonpositive, status
    where status is one of: "strong", "moderate", "weak", "negative", "insufficient"
    """
    if not METRIC_IC_HISTORY_PATH.exists():
        return {"_warning": "No metric IC history. Run screener with extended snapshots."}

    ic_df = pd.read_csv(METRIC_IC_HISTORY_PATH)
    ic_1m = ic_df[ic_df["horizon"] == "1m"].sort_values("run_date")
    if len(ic_1m) == 0:
        ic_1m = ic_df[ic_df["horizon"] == "1w"].sort_values("run_date")

    if len(ic_1m) < 4:
        return {"_warning": f"Only {len(ic_1m)} observations (need 4+)."}

    # Optionally filter to metrics in a specific category
    from factor_engine import CAT_METRICS
    if category:
        metrics = CAT_METRICS.get(category, [])
    else:
        metrics = [c.replace("_ic", "") for c in ic_1m.columns if c.endswith("_ic")]

    result = {"_n_observations": len(ic_1m)}
    for metric in metrics:
        ic_col = f"{metric}_ic"
        if ic_col not in ic_1m.columns:
            continue

        series = ic_1m[ic_col].dropna()
        if len(series) < 4:
            result[metric] = {"status": "insufficient", "n_observations": len(series)}
            continue

        mean_ic = series.mean()
        ewm_ic = series.ewm(halflife=ewm_halflife, min_periods=3).mean().iloc[-1]
        pct_positive = (series > 0).mean()

        # Consecutive positive/nonpositive streak from tail
        consecutive_positive = 0
        for val in reversed(series.values):
            if val > 0:
                consecutive_positive += 1
            else:
                break

        consecutive_nonpositive = 0
        for val in reversed(series.values):
            if val <= 0:
                consecutive_nonpositive += 1
            else:
                break

        # Status classification
        if ewm_ic >= 0.05 and pct_positive >= 0.7:
            status = "strong"
        elif ewm_ic >= 0.02 and pct_positive >= 0.55:
            status = "moderate"
        elif ewm_ic > 0:
            status = "weak"
        else:
            status = "negative"

        result[metric] = {
            "mean_ic": round(mean_ic, 4),
            "ewm_ic": round(ewm_ic, 4),
            "pct_positive": round(pct_positive, 3),
            "n_observations": len(series),
            "consecutive_positive": consecutive_positive,
            "consecutive_nonpositive": consecutive_nonpositive,
            "status": status,
        }

    return result


# =========================================================================
# Phase C: Weight Optimization
# =========================================================================

def _load_current_config() -> dict:
    """Load current config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_current_factor_weights() -> dict[str, float]:
    """Extract current category weights from config."""
    cfg = _load_current_config()
    fw = cfg.get("factor_weights", {})
    return {cat: float(fw.get(cat, 0)) for cat in CATEGORY_NAMES}


def _ic_weighted_allocation(
    ic_means: dict[str, float],
    total: float = 100.0,
) -> dict[str, float]:
    """Allocate weights proportional to positive IC.

    Categories with IC <= 0 get their minimum bound weight.
    """
    # Clip negative ICs to zero
    positive_ics = {cat: max(ic, 0.0) for cat, ic in ic_means.items()}
    total_positive = sum(positive_ics.values())

    if total_positive < 1e-8:
        # All ICs are zero/negative — return minimum bounds
        result = {cat: CATEGORY_BOUNDS[cat][0] for cat in CATEGORY_NAMES}
        residual = total - sum(result.values())
        if residual > 0:
            # Distribute residual equally
            per_cat = residual / len(CATEGORY_NAMES)
            result = {cat: v + per_cat for cat, v in result.items()}
        return result

    # Allocate proportionally
    min_weight_total = sum(
        CATEGORY_BOUNDS[cat][0]
        for cat in CATEGORY_NAMES
        if positive_ics[cat] <= 1e-8
    )
    available = total - min_weight_total

    result = {}
    for cat in CATEGORY_NAMES:
        if positive_ics[cat] <= 1e-8:
            result[cat] = CATEGORY_BOUNDS[cat][0]
        else:
            result[cat] = (positive_ics[cat] / total_positive) * available

    return result


def _apply_constraints(
    proposed: dict[str, float],
    current: dict[str, float],
    max_change: float = MAX_CHANGE_PER_CYCLE,
) -> dict[str, float]:
    """Enforce bounds, max change per cycle, and sum-to-100."""
    constrained = {}

    for cat in CATEGORY_NAMES:
        lo, hi = CATEGORY_BOUNDS[cat]
        val = proposed.get(cat, current.get(cat, 0))

        # Enforce bounds
        val = max(lo, min(hi, val))

        # Enforce max change per cycle
        cur = current.get(cat, val)
        delta = val - cur
        if abs(delta) > max_change:
            val = cur + max_change * (1 if delta > 0 else -1)

        # Re-enforce bounds after clamping
        val = max(lo, min(hi, val))

        constrained[cat] = round(val, 1)

    # Normalize to sum to 100 using largest-remainder method
    total = sum(constrained.values())
    if abs(total - 100.0) > 0.01:
        # Scale proportionally
        factor = 100.0 / total if total > 0 else 1.0
        constrained = {cat: round(v * factor, 1) for cat, v in constrained.items()}

        # Fix rounding: adjust largest weight
        diff = 100.0 - sum(constrained.values())
        if abs(diff) > 0.01:
            largest = max(constrained, key=constrained.get)
            constrained[largest] = round(constrained[largest] + diff, 1)

    return constrained


def propose_weight_changes(
    min_observations: int = 8,
    ewm_halflife: int = 6,
    shrinkage: float = 0.5,
) -> dict:
    """Propose new category weights based on live IC data.

    Returns dict with: current_weights, proposed_weights, changes,
    rationale, confidence, n_observations, can_auto_apply.
    """
    trends = analyze_ic_trends(ewm_halflife=ewm_halflife)

    n_obs = trends.get("_n_observations", 0)
    if "_warning" in trends and n_obs < min_observations:
        return {
            "status": "insufficient_data",
            "message": trends["_warning"],
            "n_observations": n_obs,
            "min_required": min_observations,
        }

    current = _get_current_factor_weights()

    # Extract EWM ICs for optimization
    ewm_ics = {}
    for cat in CATEGORY_NAMES:
        if cat in trends and isinstance(trends[cat], dict):
            ewm_ics[cat] = trends[cat].get("ewm_ic", 0.0)
        else:
            ewm_ics[cat] = 0.0

    # IC-weighted allocation
    ic_optimal = _ic_weighted_allocation(ewm_ics)

    # Shrink toward current weights
    proposed_raw = {}
    for cat in CATEGORY_NAMES:
        proposed_raw[cat] = shrinkage * current[cat] + (1 - shrinkage) * ic_optimal[cat]

    # Apply constraints
    proposed = _apply_constraints(proposed_raw, current)

    # Compute changes and rationale
    changes = {}
    rationale = {}
    for cat in CATEGORY_NAMES:
        delta = proposed[cat] - current[cat]
        changes[cat] = round(delta, 1)

        cat_info = trends.get(cat, {})
        if isinstance(cat_info, dict):
            ewm = cat_info.get("ewm_ic", 0)
            trend = cat_info.get("ic_trend", "unknown")
            if delta > 0.05:
                rationale[cat] = f"IC {trend} (ewm={ewm:.3f}), increasing weight by {delta:+.1f}%"
            elif delta < -0.05:
                rationale[cat] = f"IC {trend} (ewm={ewm:.3f}), decreasing weight by {delta:+.1f}%"
            else:
                rationale[cat] = f"IC {trend} (ewm={ewm:.3f}), weight unchanged"
        else:
            rationale[cat] = "Insufficient data for this category"

    # Confidence level
    if n_obs >= 50:
        confidence = "high"
    elif n_obs >= 24:
        confidence = "medium"
    else:
        confidence = "low"

    # Auto-apply eligibility
    max_abs_change = max(abs(v) for v in changes.values())
    can_auto_apply = (
        confidence in ("medium", "high")
        and max_abs_change <= AUTO_APPLY_THRESHOLD
    )

    return {
        "status": "proposal_ready",
        "current_weights": current,
        "proposed_weights": proposed,
        "changes": changes,
        "rationale": rationale,
        "confidence": confidence,
        "n_observations": n_obs,
        "can_auto_apply": can_auto_apply,
        "max_abs_change": round(max_abs_change, 1),
        "ewm_ics": {cat: round(v, 4) for cat, v in ewm_ics.items()},
    }


def propose_metric_weight_changes(
    category: str,
    min_observations: int = 24,
    shrinkage: float = 0.5,
) -> dict:
    """Propose metric weight changes within a single category.

    Uses per-metric IC from the extended snapshot data to optimize
    individual metric weights within a category.
    """
    # Ensure metric-level IC has been computed
    compute_metric_level_ic(horizon="1m")

    # Delegate to metric IC analysis
    trends = analyze_metric_ic_trends(category=category)
    if "_warning" in trends:
        return {"status": "insufficient_data", "message": trends["_warning"]}

    n_obs = trends.get("_n_observations", 0)
    if n_obs < min_observations:
        return {
            "status": "insufficient_data",
            "message": f"Only {n_obs} dates (need {min_observations}+).",
            "n_observations": n_obs,
        }

    # Load current weights
    cfg = _load_current_config()
    cat_weights = cfg.get("metric_weights", {}).get(category, {})
    if not cat_weights:
        return {"status": "error", "message": f"No metric weights for '{category}'"}

    # IC-proportional weight allocation within category
    ewm_ics = {}
    for metric, info in trends.items():
        if isinstance(info, dict) and "ewm_ic" in info:
            ewm_ics[metric] = info["ewm_ic"]

    # Shrink toward current
    positive_ics = {m: max(ic, 0.0) for m, ic in ewm_ics.items()}
    total_pos = sum(positive_ics.values())

    proposed = {}
    if total_pos > 1e-8:
        for m in cat_weights:
            if m in positive_ics and positive_ics[m] > 0:
                ic_alloc = (positive_ics[m] / total_pos) * 100
            else:
                ic_alloc = 0
            proposed[m] = round(shrinkage * cat_weights.get(m, 0) + (1 - shrinkage) * ic_alloc, 1)
    else:
        proposed = dict(cat_weights)

    # Enforce max change per metric
    for m in proposed:
        current = cat_weights.get(m, 0)
        delta = proposed[m] - current
        if abs(delta) > MAX_METRIC_CHANGE_PER_CYCLE:
            proposed[m] = round(current + MAX_METRIC_CHANGE_PER_CYCLE * (1 if delta > 0 else -1), 1)
        proposed[m] = max(0, proposed[m])

    # Normalize to 100
    total = sum(proposed.values())
    if total > 0:
        proposed = {m: round(v * 100 / total, 1) for m, v in proposed.items()}

    changes = {m: round(proposed.get(m, 0) - cat_weights.get(m, 0), 1) for m in cat_weights}

    return {
        "status": "proposal_ready",
        "category": category,
        "current_weights": dict(cat_weights),
        "proposed_weights": proposed,
        "changes": changes,
        "n_observations": n_obs,
        "metric_ics": {m: round(ewm_ics.get(m, 0), 4) for m in cat_weights},
    }


# =========================================================================
# Phase D: Regime Detection
# =========================================================================

def compute_dispersion(scored_df: pd.DataFrame) -> dict[str, float]:
    """Compute cross-sectional standard deviation of each category score.

    Higher dispersion = more discriminating power = factor has signal.
    Lower dispersion = scores clustered = factor provides less differentiation.
    """
    result = {}
    for cat in CATEGORY_NAMES:
        score_col = f"{cat}_score"
        if score_col in scored_df.columns:
            vals = scored_df[score_col].dropna()
            if len(vals) >= 10:
                result[cat] = round(float(vals.std()), 4)
    return result


def record_dispersion(run_date: str, dispersion: dict[str, float]) -> None:
    """Append dispersion to improvement/dispersion_history.csv."""
    row = {"date": run_date, **{f"{cat}_disp": dispersion.get(cat, np.nan) for cat in CATEGORY_NAMES}}
    df_row = pd.DataFrame([row])

    if DISPERSION_HISTORY_PATH.exists():
        existing = pd.read_csv(DISPERSION_HISTORY_PATH)
        combined = pd.concat([existing, df_row], ignore_index=True)
    else:
        combined = df_row

    combined.to_csv(DISPERSION_HISTORY_PATH, index=False)


def detect_regime(min_history: int = 12) -> dict[str, str]:
    """Classify current dispersion regime per category.

    Returns: {"valuation": "normal", "quality": "high", ...}
    """
    if not DISPERSION_HISTORY_PATH.exists():
        return {}

    hist = pd.read_csv(DISPERSION_HISTORY_PATH)
    if len(hist) < min_history:
        return {}

    result = {}
    for cat in CATEGORY_NAMES:
        col = f"{cat}_disp"
        if col not in hist.columns:
            continue

        series = hist[col].dropna()
        if len(series) < min_history:
            continue

        current = series.iloc[-1]
        p25 = series.quantile(0.25)
        p75 = series.quantile(0.75)

        if current > p75:
            result[cat] = "high"
        elif current < p25:
            result[cat] = "low"
        else:
            result[cat] = "normal"

    return result


def apply_regime_adjustments(
    proposed_weights: dict[str, float],
    regimes: dict[str, str],
    scale: float = 0.10,
) -> dict[str, float]:
    """Apply gentle regime-based adjustments to proposed weights.

    High-dispersion categories: +scale (more signal → increase weight)
    Low-dispersion categories: -scale (less signal → decrease weight)
    """
    adjusted = {}
    for cat in CATEGORY_NAMES:
        base = proposed_weights.get(cat, 0)
        regime = regimes.get(cat, "normal")

        if regime == "high":
            adjusted[cat] = base * (1 + scale)
        elif regime == "low":
            adjusted[cat] = base * (1 - scale)
        else:
            adjusted[cat] = base

    # Normalize to sum to 100
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {cat: round(v * 100 / total, 1) for cat, v in adjusted.items()}

    return adjusted


# =========================================================================
# Phase D2: Metric Evolution (Phase 11)
# =========================================================================

# Candidate metrics — pre-implemented with weight=0, evaluated for activation
CANDIDATE_METRICS = {
    "proximity_52w_high",
    "operating_margin",
    "current_ratio",
    "dividend_yield",
    "insider_ownership",
    "short_pct_float",
    "analyst_rating",
    "interest_coverage",
}


def propose_metric_evolution(
    min_observations: int = 12,
    ewm_halflife: int = 6,
) -> dict:
    """Propose metric additions and removals based on per-metric IC.

    Returns dict with:
      - activate_proposals: candidates to turn on
      - deactivate_proposals: existing metrics to turn off
      - weight_adjustments: per-category weight redistribution plan
      - status: "proposal_ready" | "insufficient_data" | "no_changes"
    """
    try:
        cfg = _load_current_config()
    except FileNotFoundError:
        return {"status": "insufficient_data", "message": "Config file not found."}
    imp_cfg = cfg.get("improvement", {})
    ic_threshold = imp_cfg.get("candidate_ic_threshold", 0.02)
    activation_obs = imp_cfg.get("min_observations_for_activation", 12)
    deactivation_obs = imp_cfg.get("min_observations_for_deactivation", 12)
    initial_weight = imp_cfg.get("candidate_initial_weight", 5)
    max_activations = imp_cfg.get("max_candidate_activations_per_cycle", 1)

    # Compute metric-level IC if not already done
    compute_metric_level_ic(horizon="1m")

    # Analyze trends
    trends = analyze_metric_ic_trends(ewm_halflife=ewm_halflife)
    if "_warning" in trends:
        return {"status": "insufficient_data", "message": trends["_warning"]}

    n_obs = trends.get("_n_observations", 0)
    if n_obs < min_observations:
        return {
            "status": "insufficient_data",
            "n_observations": n_obs,
            "min_required": min_observations,
        }

    # Identify current metric weights from config
    metric_weights = cfg.get("metric_weights", {})

    activate_proposals = []
    deactivate_proposals = []

    from factor_engine import CAT_METRICS

    for cat, metrics in CAT_METRICS.items():
        cat_weights = metric_weights.get(cat, {})

        for metric in metrics:
            info = trends.get(metric, {})
            if not isinstance(info, dict) or info.get("status") == "insufficient":
                continue

            current_weight = cat_weights.get(metric, 0)
            ewm_ic = info.get("ewm_ic", 0)
            consec_pos = info.get("consecutive_positive", 0)
            consec_nonpos = info.get("consecutive_nonpositive", 0)
            pct_pos = info.get("pct_positive", 0)

            # --- ACTIVATION: candidate metric with weight=0 showing promise ---
            if metric in CANDIDATE_METRICS and current_weight == 0:
                if (ewm_ic >= ic_threshold
                        and consec_pos >= activation_obs
                        and pct_pos >= 0.60):
                    activate_proposals.append({
                        "metric": metric,
                        "category": cat,
                        "proposed_weight": initial_weight,
                        "ewm_ic": ewm_ic,
                        "consecutive_positive_ic": consec_pos,
                        "pct_positive": pct_pos,
                        "rationale": (
                            f"Candidate shows consistent predictive power: "
                            f"EWM IC={ewm_ic:.3f}, {consec_pos} consecutive "
                            f"positive IC observations, {pct_pos:.0%} positive rate."
                        ),
                    })

            # --- DEACTIVATION: existing metric with weight>0 showing no signal ---
            elif metric not in CANDIDATE_METRICS and current_weight > 0:
                if (ewm_ic <= 0 and consec_nonpos >= deactivation_obs):
                    deactivate_proposals.append({
                        "metric": metric,
                        "category": cat,
                        "current_weight": current_weight,
                        "proposed_weight": max(0, current_weight - 5),
                        "ewm_ic": ewm_ic,
                        "consecutive_nonpositive_ic": consec_nonpos,
                        "rationale": (
                            f"Metric shows sustained negative/zero IC: "
                            f"EWM IC={ewm_ic:.3f}, {consec_nonpos} consecutive "
                            f"non-positive observations. Proposing weight reduction."
                        ),
                    })

    # Limit activations per cycle
    activate_proposals.sort(key=lambda p: p["ewm_ic"], reverse=True)
    activate_proposals = activate_proposals[:max_activations]

    if not activate_proposals and not deactivate_proposals:
        return {
            "status": "no_changes",
            "n_observations": n_obs,
            "message": "No metric changes proposed. All metrics within expected IC ranges.",
        }

    # Build weight redistribution plan for each affected category
    weight_adjustments = _compute_weight_redistribution(
        activate_proposals, deactivate_proposals, metric_weights
    )

    return {
        "status": "proposal_ready",
        "n_observations": n_obs,
        "activate_proposals": activate_proposals,
        "deactivate_proposals": deactivate_proposals,
        "weight_adjustments": weight_adjustments,
        "metric_ic_summary": {
            m: trends[m] for m in sorted(trends.keys())
            if isinstance(trends.get(m), dict)
        },
    }


def _compute_weight_redistribution(
    activations: list[dict],
    deactivations: list[dict],
    current_weights: dict,
) -> dict[str, dict]:
    """Compute per-category weight redistribution for proposed changes.

    When adding a metric: proportionally reduce existing metrics' weights.
    When removing weight: proportionally increase remaining metrics' weights.

    Returns {category: {metric: new_weight, ...}}.
    """
    adjustments = {}

    for cat in set(
        [p["category"] for p in activations] +
        [p["category"] for p in deactivations]
    ):
        cat_weights = dict(current_weights.get(cat, {}))

        # Process deactivations first (free up weight)
        freed_weight = 0
        for deact in deactivations:
            if deact["category"] != cat:
                continue
            m = deact["metric"]
            reduction = cat_weights.get(m, 0) - deact["proposed_weight"]
            freed_weight += reduction
            cat_weights[m] = deact["proposed_weight"]

        # Process activations (consume weight)
        needed_weight = 0
        for act in activations:
            if act["category"] != cat:
                continue
            m = act["metric"]
            needed_weight += act["proposed_weight"]
            cat_weights[m] = act["proposed_weight"]

        # Net weight to redistribute
        net_change = needed_weight - freed_weight
        if net_change > 0:
            # Need to take weight from existing metrics (proportionally)
            existing_active = {
                m: w for m, w in cat_weights.items()
                if w > 0 and m not in [a["metric"] for a in activations]
            }
            total_existing = sum(existing_active.values())
            if total_existing > 0:
                for m, w in existing_active.items():
                    reduction = (w / total_existing) * net_change
                    cat_weights[m] = round(max(0, w - reduction), 1)

        elif net_change < 0:
            # Weight freed — redistribute to existing active metrics
            existing_active = {
                m: w for m, w in cat_weights.items()
                if w > 0 and m not in [d["metric"] for d in deactivations]
            }
            total_existing = sum(existing_active.values())
            if total_existing > 0:
                bonus = abs(net_change)
                for m, w in existing_active.items():
                    addition = (w / total_existing) * bonus
                    cat_weights[m] = round(w + addition, 1)

        # Normalize to 100
        total = sum(cat_weights.values())
        if abs(total - 100) > 0.5 and total > 0:
            factor = 100.0 / total
            cat_weights = {m: round(w * factor, 1) for m, w in cat_weights.items()}
            diff = 100.0 - sum(cat_weights.values())
            if abs(diff) > 0.01:
                largest = max(cat_weights, key=cat_weights.get)
                cat_weights[largest] = round(cat_weights[largest] + diff, 1)

        adjustments[cat] = cat_weights

    return adjustments


def apply_metric_changes(
    proposal: dict,
    reason: str = "",
    dry_run: bool = False,
) -> dict:
    """Apply proposed metric weight changes to config.yaml.

    Handles both activations and deactivations by updating
    metric_weights in config.yaml.
    """
    if proposal.get("status") != "proposal_ready":
        return {"applied": False, "error": "No actionable proposal"}

    cfg = _load_current_config()

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ROOT / f"config.yaml.bak.{timestamp}"
    if not dry_run:
        import shutil
        shutil.copy2(CONFIG_PATH, backup_path)

    weight_adjustments = proposal.get("weight_adjustments", {})
    mw = cfg.get("metric_weights", {})

    old_values = {}
    for cat, new_weights in weight_adjustments.items():
        old_values[cat] = dict(mw.get(cat, {}))
        mw[cat] = new_weights

    if dry_run:
        return {
            "applied": False,
            "dry_run": True,
            "proposed_metric_weights": weight_adjustments,
            "old_values": old_values,
        }

    # Validate via Pydantic
    try:
        from schemas import RunConfig
        RunConfig(**cfg)
    except Exception as e:
        return {"applied": False, "error": f"Validation failed: {e}"}

    # Write updated config
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Log changes
    log_rows = []
    for act in proposal.get("activate_proposals", []):
        log_rows.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "change_type": "metric_activation",
            "category": act["category"],
            "metric": act["metric"],
            "old_value": 0,
            "new_value": act["proposed_weight"],
            "reason": act.get("rationale", reason),
            "n_observations": proposal.get("n_observations", ""),
        })
    for deact in proposal.get("deactivate_proposals", []):
        log_rows.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "change_type": "metric_deactivation",
            "category": deact["category"],
            "metric": deact["metric"],
            "old_value": deact["current_weight"],
            "new_value": deact["proposed_weight"],
            "reason": deact.get("rationale", reason),
            "n_observations": proposal.get("n_observations", ""),
        })

    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if CHANGE_LOG_PATH.exists():
            existing = pd.read_csv(CHANGE_LOG_PATH)
            combined = pd.concat([existing, log_df], ignore_index=True)
        else:
            combined = log_df
        combined.to_csv(CHANGE_LOG_PATH, index=False)

    return {
        "applied": True,
        "backup_path": str(backup_path),
        "activations": [a["metric"] for a in proposal.get("activate_proposals", [])],
        "deactivations": [d["metric"] for d in proposal.get("deactivate_proposals", [])],
    }


# =========================================================================
# Phase E: Reporting & Approval
# =========================================================================

def generate_improvement_report() -> str:
    """Generate a comprehensive improvement analysis report.

    Returns markdown string. Also saves to improvement/proposals/.
    """
    lines = []
    lines.append("# Screener Improvement Report")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # --- 1. Data Availability ---
    lines.append("## 1. Data Availability\n")
    n_snapshots = len(list(SNAPSHOTS_DIR.glob("*.parquet")))
    n_perf_rows = 0
    n_perf_dates = 0
    if PERFORMANCE_HISTORY_PATH.exists():
        perf = pd.read_csv(PERFORMANCE_HISTORY_PATH)
        n_perf_rows = len(perf)
        n_perf_dates = perf["run_date"].nunique()

    lines.append(f"- Snapshots recorded: {n_snapshots}")
    lines.append(f"- Performance history: {n_perf_rows} ticker-date pairs across {n_perf_dates} run dates")

    if n_perf_dates < 8:
        lines.append(f"- **Status:** Collecting data ({n_perf_dates}/8 minimum for proposals)")
        lines.append(f"- Run the screener {8 - n_perf_dates} more times to unlock weight proposals.\n")
    elif n_perf_dates < 24:
        lines.append(f"- **Status:** Low confidence ({n_perf_dates}/24 for metric-level optimization)")
        lines.append(f"- Category-level proposals available. Metric-level unlocks at 24 observations.\n")
    elif n_perf_dates < 50:
        lines.append(f"- **Status:** Medium confidence ({n_perf_dates}/50 for high confidence)")
        lines.append(f"- Both category and metric-level proposals available. Auto-apply enabled for small changes.\n")
    else:
        lines.append(f"- **Status:** High confidence ({n_perf_dates} observations)")
        lines.append(f"- All optimization features active.\n")

    # --- 2. Portfolio Performance ---
    lines.append("## 2. Portfolio Performance\n")
    if PERFORMANCE_HISTORY_PATH.exists() and n_perf_rows > 0:
        perf = pd.read_csv(PERFORMANCE_HISTORY_PATH)

        # Portfolio vs universe
        port = perf[perf["in_portfolio"] == True]  # noqa: E712
        univ = perf

        for horizon in ["1w", "1m", "3m"]:
            col = f"fwd_return_{horizon}"
            if col not in perf.columns:
                continue
            port_ret = port[col].dropna()
            univ_ret = univ[col].dropna()
            if len(port_ret) > 0 and len(univ_ret) > 0:
                lines.append(f"**{horizon.upper()} Forward Returns:**")
                lines.append(f"- Portfolio (top 25): mean={port_ret.mean():.3%}, median={port_ret.median():.3%}")
                lines.append(f"- Universe:           mean={univ_ret.mean():.3%}, median={univ_ret.median():.3%}")
                lines.append(f"- Spread:             {port_ret.mean() - univ_ret.mean():+.3%}\n")

        # Quintile analysis
        if "fwd_return_1m" in perf.columns:
            valid = perf.dropna(subset=["composite_score", "fwd_return_1m"])
            if len(valid) >= 50:
                valid["quintile"] = pd.qcut(valid["composite_score"], 5, labels=[1, 2, 3, 4, 5])
                q_returns = valid.groupby("quintile")["fwd_return_1m"].mean()
                lines.append("**Composite Score Quintile Analysis (1M Returns):**")
                lines.append("| Quintile | Avg 1M Return |")
                lines.append("|----------|--------------|")
                for q, ret in q_returns.items():
                    lines.append(f"| Q{q} ({'worst' if q == 1 else 'best' if q == 5 else 'mid'}) | {ret:.3%} |")
                lines.append("")

                # Monotonicity check
                if q_returns.is_monotonic_increasing:
                    lines.append("Monotonic quintile spread: higher scores → higher returns. **Model is working.**\n")
                else:
                    lines.append("Non-monotonic quintile spread. Some factor weights may need adjustment.\n")
    else:
        lines.append("_No performance data yet. Run the screener to start tracking._\n")

    # --- 3. Factor IC Analysis ---
    lines.append("## 3. Factor IC Analysis\n")

    # Compute live ICs
    for h in ["1w", "1m"]:
        compute_live_ic(horizon=h)

    trends = analyze_ic_trends()
    n_obs = trends.get("_n_observations", 0)

    if "_warning" in trends:
        lines.append(f"_{trends['_warning']}_\n")
    else:
        lines.append(f"Observations: {n_obs} | Horizon: 1-month forward returns\n")
        lines.append("| Category | EWM IC | 3M IC | 6M IC | 12M IC | Trend | IR | % Positive |")
        lines.append("|----------|--------|-------|-------|--------|-------|----|-----------|")
        for cat in CATEGORY_NAMES:
            info = trends.get(cat, {})
            if not isinstance(info, dict):
                continue
            ewm = info.get("ewm_ic", 0)
            m3 = info.get("mean_ic_3m")
            m6 = info.get("mean_ic_6m")
            m12 = info.get("mean_ic_12m")
            trend = info.get("ic_trend", "?")
            ir = info.get("ic_ir", 0)
            pct = info.get("pct_positive", 0)
            lines.append(
                f"| {cat:10s} | {ewm:+.3f} | "
                f"{m3:+.3f} | " if m3 is not None else f"| {cat:10s} | {ewm:+.3f} | N/A   | "
            )
            # Simplified formatting
            m3_str = f"{m3:+.3f}" if m3 is not None else "N/A"
            m6_str = f"{m6:+.3f}" if m6 is not None else "N/A"
            m12_str = f"{m12:+.3f}" if m12 is not None else "N/A"
            trend_emoji = {"improving": "^", "declining": "v", "stable": "=", "unknown": "?"}.get(trend, "?")
            lines.pop()  # Remove malformed line
            lines.append(
                f"| {cat:10s} | {ewm:+.3f} | {m3_str:>6s} | {m6_str:>6s} | "
                f"{m12_str:>6s} | {trend_emoji} {trend:9s} | {ir:.2f} | {pct:.0%} |"
            )
        lines.append("")

    # --- 4. Regime Analysis ---
    lines.append("## 4. Regime Analysis\n")
    regimes = detect_regime()
    if regimes:
        lines.append("| Category | Dispersion Regime |")
        lines.append("|----------|------------------|")
        for cat in CATEGORY_NAMES:
            regime = regimes.get(cat, "insufficient data")
            flag = " !" if regime in ("high", "low") else ""
            lines.append(f"| {cat:10s} | {regime}{flag} |")
        lines.append("")
    else:
        lines.append("_Insufficient dispersion history (need 12+ observations)._\n")

    # --- 5. Proposed Weight Changes ---
    lines.append("## 5. Proposed Weight Changes\n")
    proposal = propose_weight_changes()

    if proposal.get("status") == "insufficient_data":
        lines.append(f"_{proposal['message']}_\n")
    elif proposal.get("status") == "proposal_ready":
        current = proposal["current_weights"]
        proposed = proposal["proposed_weights"]
        changes = proposal["changes"]
        rationale = proposal["rationale"]

        lines.append(f"**Confidence:** {proposal['confidence']} ({proposal['n_observations']} observations)")
        if proposal["can_auto_apply"]:
            lines.append(f"**Auto-apply eligible:** Yes (max change: {proposal['max_abs_change']}%)\n")
        else:
            lines.append(f"**Auto-apply eligible:** No (max change: {proposal['max_abs_change']}%)\n")

        lines.append("| Category | Current | Proposed | Change | Rationale |")
        lines.append("|----------|---------|----------|--------|-----------|")
        for cat in CATEGORY_NAMES:
            cur = current[cat]
            prop = proposed[cat]
            chg = changes[cat]
            rat = rationale.get(cat, "")
            marker = " *" if abs(chg) > 0.05 else ""
            lines.append(f"| {cat:10s} | {cur:.1f}% | {prop:.1f}% | {chg:+.1f}%{marker} | {rat} |")
        lines.append("")

    # --- 5.5 Metric Evolution ---
    lines.append("## 5.5 Metric Evolution\n")

    metric_proposal = propose_metric_evolution()
    if metric_proposal.get("status") == "insufficient_data":
        lines.append(f"_{metric_proposal.get('message', 'Insufficient data for metric evolution')}_\n")
    elif metric_proposal.get("status") == "no_changes":
        lines.append("No metric add/remove proposals. All metrics within expected IC ranges.\n")
    elif metric_proposal.get("status") == "proposal_ready":
        # Activation proposals
        activations = metric_proposal.get("activate_proposals", [])
        if activations:
            lines.append("### Proposed Activations\n")
            lines.append("| Metric | Category | Proposed Weight | EWM IC | Consecutive + | % Positive |")
            lines.append("|--------|----------|----------------|--------|--------------|-----------|")
            for act in activations:
                lines.append(
                    f"| {act['metric']} | {act['category']} | {act['proposed_weight']}% | "
                    f"{act['ewm_ic']:+.3f} | {act['consecutive_positive_ic']} | "
                    f"{act['pct_positive']:.0%} |"
                )
            lines.append("")

        # Deactivation proposals
        deactivations = metric_proposal.get("deactivate_proposals", [])
        if deactivations:
            lines.append("### Proposed Deactivations\n")
            lines.append("| Metric | Category | Current Weight | Proposed | EWM IC | Consecutive 0/- |")
            lines.append("|--------|----------|---------------|----------|--------|----------------|")
            for deact in deactivations:
                lines.append(
                    f"| {deact['metric']} | {deact['category']} | {deact['current_weight']}% | "
                    f"{deact['proposed_weight']}% | {deact['ewm_ic']:+.3f} | "
                    f"{deact['consecutive_nonpositive_ic']} |"
                )
            lines.append("")

        # Per-metric IC summary
        ic_summary = metric_proposal.get("metric_ic_summary", {})
        if ic_summary:
            lines.append("### Per-Metric IC Summary\n")
            lines.append("| Metric | EWM IC | Status | % Positive | Observations |")
            lines.append("|--------|--------|--------|-----------|-------------|")
            for m, info in sorted(ic_summary.items(), key=lambda x: x[1].get("ewm_ic", 0), reverse=True):
                status_marker = {"strong": "++", "moderate": "+", "weak": "~", "negative": "-"}.get(
                    info.get("status", ""), "?"
                )
                lines.append(
                    f"| {m} | {info.get('ewm_ic', 0):+.3f} | {status_marker} {info.get('status', '')} | "
                    f"{info.get('pct_positive', 0):.0%} | {info.get('n_observations', 0)} |"
                )
            lines.append("")

    # --- 6. Ranking Impact ---
    lines.append("## 6. Ranking Impact\n")
    if proposal.get("status") == "proposal_ready":
        impact = _estimate_ranking_impact(proposal)
        if impact:
            lines.append(f"- Top-20 Jaccard similarity: {impact.get('top20_jaccard', 'N/A')}")
            enters = impact.get("top25_enters", [])
            exits = impact.get("top25_exits", [])
            if enters:
                lines.append(f"- Would enter top 25: {', '.join(enters)}")
            if exits:
                lines.append(f"- Would exit top 25: {', '.join(exits)}")
            lines.append("")
    else:
        lines.append("_No proposal available._\n")

    # --- 7. Risk Assessment ---
    lines.append("## 7. Risk Assessment\n")
    if n_obs < 24:
        lines.append("- **Overfitting risk: HIGH** — fewer than 24 observations. "
                      "IC estimates are noisy. Shrinkage is protecting against overcorrection.")
    elif n_obs < 50:
        lines.append("- **Overfitting risk: MODERATE** — 24-50 observations. "
                      "IC trends are becoming meaningful but could still reverse.")
    else:
        lines.append("- **Overfitting risk: LOW** — 50+ observations provide "
                      "statistically robust IC estimates.")

    lines.append("- All changes bounded: max ±3% per category per cycle.")
    lines.append("- Shrinkage (50%) prevents wild swings from noisy IC estimates.")
    lines.append("- Config backup created before any change.\n")

    # --- 8. Change History ---
    lines.append("## 8. Change History\n")
    if CHANGE_LOG_PATH.exists():
        log = pd.read_csv(CHANGE_LOG_PATH)
        if not log.empty:
            lines.append("| Date | Category | Old | New | Reason |")
            lines.append("|------|----------|-----|-----|--------|")
            for _, row in log.tail(20).iterrows():
                lines.append(
                    f"| {row.get('date', '')} | {row.get('category', '')} | "
                    f"{row.get('old_value', '')} | {row.get('new_value', '')} | "
                    f"{row.get('reason', '')} |"
                )
            lines.append("")
        else:
            lines.append("_No changes applied yet._\n")
    else:
        lines.append("_No changes applied yet._\n")

    report = "\n".join(lines)

    # Save to proposals directory
    date_str = datetime.now().strftime("%Y%m%d")
    report_path = PROPOSALS_DIR / f"improve_{date_str}.md"
    report_path.write_text(report, encoding="utf-8")

    return report


def _estimate_ranking_impact(proposal: dict) -> dict | None:
    """Estimate how proposed weights would change the ranking.

    Uses the most recent scored parquet from runs/.
    """
    try:
        from factor_engine import compute_composite, load_config

        # Find most recent final scored parquet
        runs_dir = ROOT / "runs"
        latest_scored = None
        latest_time = ""
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            scored = run_dir / "05_final_scored.parquet"
            meta = run_dir / "meta.json"
            if scored.exists() and meta.exists():
                m = json.loads(meta.read_text())
                t = m.get("start_time", "")
                if t > latest_time:
                    latest_time = t
                    latest_scored = scored

        if latest_scored is None:
            return None

        df = pd.read_parquet(latest_scored)

        # Current top 25
        current_top25 = set(df.nsmallest(25, "Rank")["Ticker"].tolist())
        current_top20 = set(df.nsmallest(20, "Rank")["Ticker"].tolist())

        # Recompute composite with proposed weights
        proposed = proposal["proposed_weights"]
        score_cols = {f"{cat}_score": proposed[cat] / 100 for cat in CATEGORY_NAMES}
        df["proposed_composite"] = 0
        total_w = 0
        for col, w in score_cols.items():
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, "proposed_composite"] += df.loc[mask, col] * w
                total_w += w
        if total_w > 0:
            df["proposed_composite"] /= total_w
        df["proposed_composite"] = df["proposed_composite"].rank(pct=True) * 100

        proposed_top25 = set(df.nlargest(25, "proposed_composite")["Ticker"].tolist())
        proposed_top20 = set(df.nlargest(20, "proposed_composite")["Ticker"].tolist())

        jaccard_20 = len(current_top20 & proposed_top20) / len(current_top20 | proposed_top20)

        return {
            "top20_jaccard": round(jaccard_20, 3),
            "top25_enters": sorted(proposed_top25 - current_top25),
            "top25_exits": sorted(current_top25 - proposed_top25),
        }
    except Exception as e:
        logger.warning(f"Ranking impact estimation failed: {e}")
        return None


def preview_ranking_impact(
    current_cfg: dict,
    proposed_cfg: dict,
    scored_df: pd.DataFrame,
) -> dict:
    """Recompute composite scores with proposed weights and compare rankings."""
    current_top25 = set(scored_df.nsmallest(25, "Rank")["Ticker"].tolist())
    current_top20 = set(scored_df.nsmallest(20, "Rank")["Ticker"].tolist())

    proposed_fw = proposed_cfg.get("factor_weights", {})
    score_cols = {f"{cat}_score": proposed_fw.get(cat, 0) / 100 for cat in CATEGORY_NAMES}

    df = scored_df.copy()
    df["proposed_composite"] = 0
    total_w = 0
    for col, w in score_cols.items():
        if col in df.columns:
            mask = df[col].notna()
            df.loc[mask, "proposed_composite"] += df.loc[mask, col] * w
            total_w += w
    if total_w > 0:
        df["proposed_composite"] /= total_w
    df["proposed_composite"] = df["proposed_composite"].rank(pct=True) * 100

    proposed_top25 = set(df.nlargest(25, "proposed_composite")["Ticker"].tolist())
    proposed_top20 = set(df.nlargest(20, "proposed_composite")["Ticker"].tolist())

    jaccard_20 = len(current_top20 & proposed_top20) / len(current_top20 | proposed_top20) if (current_top20 | proposed_top20) else 1.0

    return {
        "top20_jaccard": round(jaccard_20, 3),
        "top25_enters": sorted(proposed_top25 - current_top25),
        "top25_exits": sorted(current_top25 - proposed_top25),
        "max_rank_change": 0,  # TODO: compute actual max rank delta
    }


def apply_changes(
    changes: dict[str, float],
    reason: str = "",
    dry_run: bool = False,
) -> dict:
    """Apply proposed weight changes to config.yaml.

    Creates backup, validates via schema, updates change_log.csv.
    """
    cfg = _load_current_config()

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ROOT / f"config.yaml.bak.{timestamp}"
    if not dry_run:
        shutil.copy2(CONFIG_PATH, backup_path)

    # Apply changes to factor_weights
    fw = cfg.get("factor_weights", {})
    old_values = {}
    for cat, delta in changes.items():
        if cat in fw:
            old_values[cat] = fw[cat]
            fw[cat] = round(fw[cat] + delta, 1)

    # Validate sum
    total = sum(fw.get(cat, 0) for cat in CATEGORY_NAMES)
    if abs(total - 100) > 0.5:
        return {"applied": False, "error": f"Weights sum to {total}, not 100"}

    if dry_run:
        return {
            "applied": False,
            "dry_run": True,
            "proposed_factor_weights": dict(fw),
            "old_values": old_values,
        }

    # Validate via Pydantic
    try:
        from schemas import RunConfig
        RunConfig(**cfg)
    except Exception as e:
        # Restore backup
        shutil.copy2(backup_path, CONFIG_PATH)
        return {"applied": False, "error": f"Validation failed: {e}"}

    # Write updated config
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Append to change log
    log_rows = []
    for cat, delta in changes.items():
        if abs(delta) > 0.01:
            log_rows.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "change_type": "factor_weight",
                "category": cat,
                "old_value": old_values.get(cat, "?"),
                "new_value": fw.get(cat, "?"),
                "reason": reason,
                "n_observations": "",
                "confidence": "",
                "applied_by": "run-improve",
            })

    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if CHANGE_LOG_PATH.exists():
            existing = pd.read_csv(CHANGE_LOG_PATH)
            combined = pd.concat([existing, log_df], ignore_index=True)
        else:
            combined = log_df
        combined.to_csv(CHANGE_LOG_PATH, index=False)

    print("  REMINDER: Golden scores must be regenerated after weight changes.")
    print("  Run: pytest tests/test_golden.py --regen")

    return {
        "applied": True,
        "backup_path": str(backup_path),
        "changes_applied": {cat: {"old": old_values.get(cat), "new": fw.get(cat)} for cat in changes if abs(changes[cat]) > 0.01},
        "validation": "PASSED",
    }


def regenerate_golden_scores() -> bool:
    """Re-run the golden file test with --regen to capture new scoring baseline."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_golden.py", "--regen", "-x"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Golden score regeneration failed: {e}")
        return False
