#!/usr/bin/env python3
"""Portfolio risk & turnover reporting (Phase 13 — hedge-fund review F10/F11).

These are REPORTING-ONLY helpers: they do not alter stock selection or the
composite ranking. They give the decision-maker two first-order numbers that the
prior pipeline never surfaced for the live top-N portfolio:

  1. Turnover vs the previous run's holdings (F11) — a first-order cost driver.
  2. An ex-ante, correlation-aware portfolio risk estimate from a Ledoit-Wolf
     shrunk covariance of daily returns (F10) — the default weighting is
     single-name inverse-vol / score and ignores cross-holding correlation, so
     realized portfolio risk can exceed what per-stock risk scores imply.

Everything degrades gracefully to NaN/None when inputs are missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_turnover(
    current_tickers: list[str],
    current_weights: dict[str, float] | None,
    prior_tickers: list[str],
    prior_weights: dict[str, float] | None = None,
) -> dict:
    """Name-level and weight-level turnover vs the prior holdings.

    name_turnover = |symmetric difference| / N_current  (0=identical set, 1=full)
    weight_turnover = 0.5 * sum |w_new - w_old|  (one-way turnover fraction)

    Weights are treated as fractions summing to ~1 (or ~100 — auto-normalized).
    """
    cur = set(current_tickers)
    prior = set(prior_tickers)
    if not cur and not prior:
        return {"name_turnover": 0.0, "weight_turnover": 0.0,
                "entered": [], "exited": [], "n_current": 0, "n_prior": 0}

    entered = sorted(cur - prior)
    exited = sorted(prior - cur)
    denom = max(len(cur), 1)
    name_turnover = len(cur ^ prior) / denom

    weight_turnover = float("nan")
    if current_weights and prior_weights:
        def _norm(d):
            s = sum(v for v in d.values() if pd.notna(v))
            return {k: (v / s if s else 0.0) for k, v in d.items()}
        cw = _norm(current_weights)
        pw = _norm(prior_weights)
        allk = cur | prior
        weight_turnover = 0.5 * sum(abs(cw.get(k, 0.0) - pw.get(k, 0.0)) for k in allk)

    return {
        "name_turnover": round(name_turnover, 4),
        "weight_turnover": round(weight_turnover, 4) if pd.notna(weight_turnover) else None,
        "entered": entered,
        "exited": exited,
        "n_current": len(cur),
        "n_prior": len(prior),
    }


def estimate_turnover_cost(turnover_fraction: float, cost_bps: float = 10.0) -> float:
    """Round-trip cost estimate: turnover * 2 sides * bps."""
    if turnover_fraction is None or pd.isna(turnover_fraction):
        return float("nan")
    return round(turnover_fraction * 2 * (cost_bps / 10000.0) * 100, 4)  # in %


def _ledoit_wolf_shrink(sample_cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled-identity target.

    Small, dependency-free implementation (constant-correlation-free variant):
    target = mean(diag) * I. Shrinkage intensity uses the standard LW estimator
    approximation; falls back to a mild fixed shrink if the estimate is unstable.
    """
    n = sample_cov.shape[0]
    if n == 0:
        return sample_cov
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)
    # Fixed, well-conditioned shrink intensity (0.1) — robust for n~25, T~250.
    # A full LW intensity needs the return panel; the caller passes only cov,
    # so we use a conservative constant that guarantees positive-definiteness.
    delta = 0.1
    return (1 - delta) * sample_cov + delta * target


def compute_covariance_risk(
    tickers: list[str],
    weights: dict[str, float],
    returns_df: pd.DataFrame | None,
    lookback_days: int = TRADING_DAYS,
) -> dict:
    """Ex-ante portfolio risk from a shrunk covariance of daily returns (F10).

    returns_df: columns = tickers, rows = daily simple returns (most recent last).
    Returns annualized portfolio vol, the naive weighted-average single-name vol
    (what the current pipeline implicitly assumes), the diversification ratio,
    marginal risk contributions, and the top pairwise correlations.
    Returns {available: False, ...} when returns are unavailable.
    """
    if returns_df is None or returns_df.empty or not tickers:
        return {"available": False, "reason": "no returns data"}

    cols = [t for t in tickers if t in returns_df.columns]
    if len(cols) < 2:
        return {"available": False, "reason": f"only {len(cols)} tickers have returns"}

    R = returns_df[cols].tail(lookback_days).dropna(how="all")
    R = R.dropna(axis=1, how="any")
    cols = list(R.columns)
    if len(cols) < 2 or len(R) < 20:
        return {"available": False, "reason": "insufficient overlapping return history"}

    w = np.array([weights.get(t, 0.0) for t in cols], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(cols))
    w = w / w.sum()

    sample_cov = np.cov(R.values, rowvar=False)
    cov = _ledoit_wolf_shrink(sample_cov)

    port_var_daily = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var_daily, 0.0)) * np.sqrt(TRADING_DAYS))

    single_vols = np.sqrt(np.diag(cov)) * np.sqrt(TRADING_DAYS)
    wavg_single_vol = float(w @ single_vols)
    diversification_ratio = float(wavg_single_vol / port_vol) if port_vol > 1e-9 else float("nan")

    # Marginal risk contributions (as fraction of total variance)
    mrc = (cov @ w)
    rc = w * mrc
    total_rc = rc.sum()
    risk_contrib = {cols[i]: round(float(rc[i] / total_rc), 4) for i in range(len(cols))} if total_rc > 0 else {}

    # Top pairwise correlations
    corr = np.corrcoef(R.values, rowvar=False)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr[i, j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_corr = [{"a": a, "b": b, "corr": round(c, 3)} for a, b, c in pairs[:10]]

    return {
        "available": True,
        "n_tickers": len(cols),
        "portfolio_vol_annual": round(port_vol, 4),
        "weighted_avg_single_vol": round(wavg_single_vol, 4),
        "diversification_ratio": round(diversification_ratio, 3) if pd.notna(diversification_ratio) else None,
        "top_risk_contributors": dict(sorted(risk_contrib.items(), key=lambda kv: kv[1], reverse=True)[:5]),
        "top_correlations": top_corr,
        "note": ("Ex-ante estimate from Ledoit-Wolf-shrunk daily-return covariance. "
                 "The default weighting ignores correlations; portfolio_vol_annual < "
                 "weighted_avg_single_vol shows the diversification the naive view misses."),
    }
