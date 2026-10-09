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
import logging
import math
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Phase 13 (F15): targeted suppression only — keep the screener's own
# schema-drift / staleness UserWarnings visible.
warnings.simplefilter("default")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)  # yfinance sqlite cache noise
warnings.filterwarnings("ignore", message=".*urllib3.*")

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


def flush_sector_coverage(sector_stats: dict):
    """Write sector_coverage.csv to ./validation/."""
    if not sector_stats:
        return None
    rows = []
    for sector, info in sorted(sector_stats.items()):
        row = {"Sector": sector, "Stocks": info["n_stocks"],
               "Avg_Coverage_Pct": info["avg_coverage"],
               "Worst_Metric": info["worst_metric"],
               "Worst_Metric_Pct": info["worst_pct"]}
        # Add per-metric columns
        for m, pct in sorted(info.get("metric_coverage", {}).items()):
            row[f"cov_{m}"] = pct
        rows.append(row)
    path = VALIDATION_DIR / "sector_coverage.csv"
    pd.DataFrame(rows).to_csv(str(path), index=False)
    return str(path)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Factor Stock Screener v1.0")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Permit fabricated 'sector-realistic' values when the "
                        "network is unavailable. OFF by default: without it the "
                        "run refuses rather than emitting fiction that looks "
                        "exactly like analysis. Pipeline testing only.")
    p.add_argument("--refresh", action="store_true",
                   help="Force-clear all cache and re-fetch everything")
    p.add_argument("--tickers", type=str, default="",
                   help="Comma-separated tickers for quick testing "
                        "(e.g. AAPL,MSFT,GOOGL)")
    p.add_argument("--no-portfolio", action="store_true",
                   help="Skip portfolio construction; only write FactorScores")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config, test network, and check output paths without running the full pipeline (~10s)")
    p.add_argument("--show-weights", action="store_true",
                   help="Display effective factor weights after config application and exit")
    p.add_argument("--preset", type=str, default="",
                   help="Apply a configuration preset (balanced, value, growth, momentum) to override factor_weights")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loader with error handling
# ---------------------------------------------------------------------------
def load_config_safe():
    """Load config.yaml with clear error on failure.

    Validates the config against the RunConfig Pydantic schema to catch
    misconfigurations (negative weights, weights not summing to 100, etc.)
    before the pipeline runs.
    """
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

        # Validate against Pydantic schema
        from schemas import RunConfig
        RunConfig(**cfg)

        return cfg
    except Exception as e:
        print(f"\n  ERROR: Failed to parse config.yaml: {e}")
        sys.exit(1)


def weighting_description(scheme: str, num_stocks: int) -> str:
    """Describe the configured position-weighting scheme for the public docs.

    Extracted 2026-09-23 from an inline two-branch ternary in
    ``generate_screener_overview()``.  That ternary read "equal, else
    risk-parity" over a *four*-option setting, so with the shipped
    ``portfolio.weighting: 'score'`` the published SCREENER_OVERVIEW.md --
    and the methodology panel embedded in the live site -- told every reader
    the portfolio was inverse-volatility weighted.  It was composite-score
    weighted, which tilts the opposite way: toward the *highest-scoring*
    names rather than the calmest ones.  See METHODOLOGY_CHANGELOG.md
    2026-09-23 and ``tests/test_weighting_disclosure.py``.

    Every scheme ``schemas.PortfolioConfig.validate_weighting`` accepts must
    have its own branch here, and an unrecognised one names itself rather
    than borrowing another scheme's description.
    """
    scheme = (scheme or "equal").strip().lower()
    if scheme == "equal":
        return f"Equal weight (each stock gets ~{round(100 / num_stocks)}%)"
    if scheme == "score":
        return (
            "Composite-score proportional (each stock's weight is its composite "
            "score divided by the sum of the selected stocks' scores, so "
            "higher-ranked stocks get more weight)"
        )
    if scheme in ("inverse_vol", "risk_parity"):
        return (
            "Risk-parity (inverse-volatility weighting — lower-volatility "
            "stocks get more weight)"
        )
    if scheme == "markowitz":
        return (
            "Minimum-variance (experimental; requires scipy and a price-return "
            "history, and falls back to composite-score proportional when "
            "either is unavailable)"
        )
    return f"`{scheme}` (unrecognised scheme — see config.yaml `portfolio.weighting`)"


def _max_pos_note(scheme: str, num_stocks: int, max_pos: float) -> str:
    """Say so when `max_position_pct` cannot bind, instead of implying it can.

    Under equal weighting every position is exactly ``100 / num_stocks``, so
    the cap binds only if the portfolio holds fewer than ``100 / max_pos``
    names -- 20 at the shipped 5%.  Reporting a cap that arithmetic forbids
    from ever firing is the same failure shape as the always-firing bank-metric
    alarm fixed 2026-09-01: it reads as a live safety control and is not one.
    The cap is kept rather than deleted because it becomes live the moment
    ``num_stocks`` falls.  Changelog 2026-09-23.
    """
    scheme = (scheme or "equal").strip().lower()
    if scheme != "equal" or num_stocks <= 0 or max_pos <= 0:
        return ""
    equal_wt = 100.0 / num_stocks
    if equal_wt > max_pos:
        return ""  # the cap is infeasible here; portfolio_constructor warns
    threshold = math.ceil(100.0 / max_pos)
    return (
        f" — not binding under equal weighting, where every position is "
        f"{equal_wt:.2f}%. It would bind only below {threshold} holdings."
    )


# ---------------------------------------------------------------------------
# Factor Engine integration (with resilience)
# ---------------------------------------------------------------------------
def build_screener_overview(cfg: dict) -> str:
    """Build the text of SCREENER_OVERVIEW.md from the live config.

    Split out from :func:`generate_screener_overview` on 2026-09-28 so the
    document's claims can be asserted against what the generator *produces*
    rather than against the committed file. The distinction is not academic:
    on 2026-09-25 a session corrected four false statements by editing the
    committed markdown, every test passed, and the 2026-09-28 02:00 data run
    regenerated the file and published the false statements back to the live
    site. Tests that read only the artifact cannot see that coming.
    """
    fw = cfg.get("factor_weights", {})
    mw = cfg.get("metric_weights", {})
    bmw = cfg.get("bank_metric_weights", {})
    vtf = cfg.get("value_trap_filters", {})
    gtf = cfg.get("growth_trap_filters", {})
    pio = cfg.get("piotroski_conditional", {})
    pcfg = cfg.get("portfolio", {})
    dq = cfg.get("data_quality", {})
    fetch = cfg.get("fetch", {})
    pt = cfg.get("percentile_transform", {})
    clamps = cfg.get("metric_clamps", {})
    cov_disc = dq.get("coverage_discount", {})

    # Step 5's description of what happens *after* the weighted average.
    # Read from config rather than written out, so the page cannot drift from
    # the rule the engine applies (rule 10's lesson). Until 2026-10-07 Step 5
    # said the composite was "converted to a cross-sectional percentile rank
    # (0-100), so a score of 95 means better than 95% of stocks" - which has
    # been false since Phase 13 (F1) made the composite cardinal, and which the
    # same document's Limitation 8 already contradicted.
    _cov_thr = int(cov_disc.get("threshold", 0.80) * 100)
    _cov_rate = int(cov_disc.get("penalty_rate", 0.15) * 100)
    # Counted from the engine's own weight tables so the page cannot drift from them.
    from factor_engine import weighted_metric_sets
    _w_gen, _w_bank = weighted_metric_sets(cfg)
    cov_step = (
        f" One adjustment is applied after that weighted average: a stock below "
        f"{_cov_thr}% metric coverage has its composite multiplied by "
        f"`1 - (({_cov_thr}% - coverage) x {_cov_rate}%)` - the coverage discount "
        f"described under Data Quality Safeguards below - so for those stocks the "
        f"composite sits slightly below the sum of the category contributions. "
        f"**\"Coverage\" here means the share of the metrics that carry weight "
        f"in that stock's score** - {len(_w_bank)} for a bank-like stock and "
        f"{len(_w_gen)} for every other ({len(_w_gen) - 1} for other financials, which are "
        f"never given the Beneish score); a metric with no weight (a candidate, or "
        f"one shown for reference) cannot lower it (since 2026-10-09). The drilldown's \"Metrics: n/m\" badge and its \"The score rests on "
        f"n of m metrics\" sentence read **this same figure**, taken from the engine "
        f"rather than recounted (until 2026-10-07 they counted a fixed 18-metric "
        f"list, which made 62 stocks look under-covered when only 3 were "
        f"discounted). The drilldown shows the discount as its own line, so the "
        f"category points, the discount and the composite add up on screen."
        if cov_disc.get("enabled", False) else ""
    )

    # Helper to format metric weight tables
    def _metric_table(weights: dict, descriptions: dict) -> str:
        rows = []
        for metric, weight in weights.items():
            if weight == 0:
                continue
            desc = descriptions.get(metric, "")
            label = _METRIC_LABELS.get(metric, metric)
            rows.append(f"| **{label}** | {weight}% | {desc} |")
        return "\n".join(rows)

    def _metric_table_with_zeros(weights: dict, descriptions: dict) -> str:
        rows = []
        for metric, weight in weights.items():
            desc = descriptions.get(metric, "")
            label = _METRIC_LABELS.get(metric, metric)
            note = " (bank-only)" if weight == 0 else ""
            if weight == 0:
                rows.append(f"| **{label}** | {weight}%{note} | {desc} |")
            else:
                rows.append(f"| **{label}** | {weight}% | {desc} |")
        return "\n".join(rows)

    # Valuation
    val_w = mw.get("valuation", {})
    val_bank_w = bmw.get("valuation", {})
    qual_w = mw.get("quality", {})
    qual_bank_w = bmw.get("quality", {})
    grow_w = mw.get("growth", {})
    mom_w = mw.get("momentum", {})
    risk_w = mw.get("risk", {})
    rev_w = mw.get("revisions", {})
    size_w = mw.get("size", {})
    inv_w = mw.get("investment", {})

    # The heaviest metric in a category is named in that category's prose. Derive
    # it rather than writing it down: the 2026-09-10 reweight left the Revisions
    # paragraph asserting "Analyst Surprise gets the highest weight" when it had
    # dropped to 15% against FY1 EPS Revision's 35%, and the sentence sat two
    # lines under a table that contradicted it until 2026-09-25.
    def _heaviest(weights: dict) -> str:
        live = {m: w for m, w in weights.items() if w}
        if not live:
            return "no metric"
        metric = max(live, key=lambda m: live[m])
        return _METRIC_LABELS.get(metric, metric)

    rev_heaviest = _heaviest(rev_w)

    # Build composite formula
    factor_order = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]
    factor_labels = {"valuation": "Valuation", "quality": "Quality", "growth": "Growth",
                     "momentum": "Momentum", "risk": "Risk", "revisions": "Revisions",
                     "size": "Size", "investment": "Investment"}
    composite_parts = []
    for f in factor_order:
        w = fw.get(f, 0)
        if w > 0:
            composite_parts.append(f"{w}% × {factor_labels[f]}")

    # How many categories are active?
    active_factors = [f for f in factor_order if fw.get(f, 0) > 0]
    n_factors = len(active_factors)

    # Total generic metrics count
    generic_metrics = set()
    for cat in ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]:
        for m, w in mw.get(cat, {}).items():
            if w > 0:
                generic_metrics.add(m)
    bank_only = set()
    for cat in ["valuation", "quality"]:
        for m, w in bmw.get(cat, {}).items():
            if w > 0 and m not in generic_metrics:
                bank_only.add(m)
    n_generic = len(generic_metrics)
    n_bank_only = len(bank_only)
    n_total = n_generic + n_bank_only
    # Phase 13 (F12/F31): also report the full registry size (scored + the
    # candidate metrics carried at weight 0 that the improvement engine may
    # activate) so the count reconciles with METRIC_COLS and the other docs.
    try:
        from factor_engine import METRIC_COLS as _MC
        n_registry = len(_MC)
    except Exception:
        n_registry = n_total
    n_candidate = max(0, n_registry - n_total)

    # Build the generic valuation formula
    val_formula_parts = []
    for m, w in val_w.items():
        if w > 0:
            label = _METRIC_LABELS.get(m, m)
            val_formula_parts.append(f"{w}% × {label.replace(' ', '_')}_pct")
    val_formula = " + ".join(val_formula_parts)

    # Bank valuation formula
    bank_val_parts = []
    for m, w in val_bank_w.items():
        if w > 0:
            label = _METRIC_LABELS.get(m, m)
            bank_val_parts.append(f"{w}% × {label.replace(' ', '_')}_pct")
    bank_val_formula = " + ".join(bank_val_parts)

    # Pio conditional text
    pio_text = ""
    if pio.get("enabled"):
        threshold = pio.get("valuation_threshold", 50)
        reduction = pio.get("reduction_factor", 0.5)
        redist_to = pio.get("redistribute_to", [])
        redist_labels = [_METRIC_LABELS.get(m, m) for m in redist_to]
        reduction_pct = int(reduction * 100)
        pio_text = f"""## Piotroski Conditional Weighting

The Piotroski F-Score is a broad checklist of financial health signals — but its predictive power varies depending on how expensive a stock is. For cheap stocks (high valuation score), the F-Score is highly predictive: it separates genuinely undervalued companies from deteriorating ones. For expensive stocks (low valuation score), the F-Score is less informative because the market has already priced in quality.

When enabled (current: **on**), the screener reduces the Piotroski F-Score weight by {reduction_pct}% for non-bank stocks with a valuation score below {threshold} (i.e., the more expensive half of the universe). The freed weight is redistributed equally to {' and '.join(redist_labels)}, which are more robust quality signals for expensive stocks.

Bank-like stocks are unaffected — their quality weights are already tailored.

---
"""
    else:
        pio_text = """## Piotroski Conditional Weighting

Currently **disabled**. The Piotroski F-Score weight is applied uniformly regardless of valuation score.

---
"""

    # Percentile transform text
    if pt.get("enabled"):
        method = pt.get("method", "logistic")
        steepness = pt.get("logistic_steepness", 0.08)
        pt_text = f"**Optional percentile transform:** A {method} S-curve transform is **enabled** (steepness={steepness}), compressing middle-range percentiles and stretching the extremes, rewarding truly exceptional scores more aggressively. Each percentile p is transformed via `100 / (1 + exp(-{steepness} × (p - 50)))`."
    else:
        pt_text = "**Optional percentile transform:** Currently **disabled** (default). Percentile ranks are used as-is without non-linear transformation."

    # Value trap text
    vt_quality_floor = vtf.get("quality_floor_percentile", 30)
    vt_mom_floor = vtf.get("momentum_floor_percentile", 30)
    vt_rev_floor = vtf.get("revisions_floor_percentile", 30)
    vt_flag_only = vtf.get("flag_only", False)
    vt_cheap = vtf.get("valuation_percentile", 70)
    regime_state = "on" if (cfg.get("momentum_regime") or {}).get("enabled") else "off"
    vt_action = "**flagged but not excluded**" if vt_flag_only else "**excluded**"

    # Growth trap text
    gt_growth_ceil = gtf.get("growth_ceiling_percentile", 70)
    gt_quality_floor = gtf.get("quality_floor_percentile", 35)
    gt_rev_floor = gtf.get("revisions_floor_percentile", 35)
    gt_flag_only = gtf.get("flag_only", False)
    gt_action = "**flagged but not excluded**" if gt_flag_only else "**excluded**"

    # Portfolio settings
    num_stocks = pcfg.get("num_stocks", 25)
    weighting = pcfg.get("weighting", "equal")
    max_pos = pcfg.get("max_position_pct", 5.0)
    max_sector = pcfg.get("max_sector_concentration", 8)
    min_adv = float(pcfg.get("min_avg_dollar_volume", 10e6))
    min_adv_m = min_adv / 1e6

    # Data quality
    out_lo, out_hi = dq.get("outlier_report_percentiles",
                            dq.get("winsorize_percentiles", [1, 99]))
    min_coverage = dq.get("min_data_coverage_pct", 60)
    auto_reduce_thresh = dq.get("auto_reduce_nan_threshold_pct", 70)
    alert_thresh = dq.get("metric_alert_threshold_pct", 50)

    # Fetch settings
    batch_size = fetch.get("batch_size", 30)
    max_workers = fetch.get("max_workers", 3)
    inter_batch_delay = fetch.get("inter_batch_delay", 3.0)

    md = f"""# Multi-Factor Stock Screener — How It Works

**A plain-language guide to what the screener does, why it does it, and how it arrives at its picks.**

---

## What Is This?

This is a quantitative stock screener. It takes every company in the S&P 500 (roughly 500 stocks), measures each one across up to {n_total} financial metrics, combines those measurements into a single composite score (0-100), and ranks the entire universe from best to worst.

Not every stock sees all {n_total} metrics. The screener uses {n_generic} generic metrics for most stocks and a separate set of {n_bank_only} bank-specific metrics for financial companies (banks, insurers, credit companies). In practice, any individual stock is scored on about {n_generic} metrics — the set just differs depending on whether the company is a bank or not. The full metric registry (`METRIC_COLS`) has {n_registry} entries: {n_total} carry scoring weight today ({n_generic} generic + {n_bank_only} bank-specific) plus {n_candidate} candidate metrics held at weight 0 that the self-improving engine may activate if they demonstrate predictive power.

The core idea: no single number tells you whether a stock is a good investment. A stock can look cheap but be cheap for a reason (declining business, high risk). By scoring across multiple independent dimensions — {', '.join(factor_labels[f] for f in active_factors).lower()} — the screener surfaces companies that are strong across the board, not just on one axis.

---

## Where Does the Data Come From?

All data is pulled from **Yahoo Finance** via the `yfinance` Python library. For each stock, the screener fetches:

- **Financial statements** — income statement, balance sheet, and cash flow statement (annual + prior year for trend comparisons)
- **Price history** — 13 months of daily closing prices and volume (calendar-based lookbacks for momentum, volatility, and liquidity)
- **Summary statistics** — market cap, enterprise value, P/E ratios, EPS estimates, analyst price targets, number of covering analysts
- **Earnings history** — last 4 quarters of actual vs. estimated EPS (for earnings surprise calculations)

The S&P 500 member list is pulled primarily from a **GitHub-hosted CSV** (`datasets/s-and-p-500-companies`), with Wikipedia as a secondary fallback and a local backup (`sp500_tickers.json`) as a last resort. The local JSON is auto-updated whenever a network source succeeds.

Data is fetched in batches of {batch_size} tickers with {max_workers} concurrent threads per batch and a {inter_batch_delay}-second inter-batch delay to manage Yahoo Finance rate limits. Failed tickers are automatically retried in a second pass with conservative settings (single-threaded, 30-second cooldown).

---

## The {n_factors} Factor Categories

Every stock is evaluated in {n_factors} categories. Each category captures a different dimension of investment merit.

### 1. Valuation ({fw.get('valuation', 0)}% of final score)

**Question it answers:** *Is this stock priced attractively relative to what the business generates?*

**Generic stocks:**

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(val_w, _VAL_DESCRIPTIONS)}

**Bank-like stocks** use a different weight set (see [Bank-Specific Scoring](#bank-specific-scoring) below):

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(val_bank_w, _VAL_DESCRIPTIONS)}

**Why these?** Traditional P/E ratios are distorted by capital structure, one-time charges, and accounting choices. Enterprise value-based metrics strip away those distortions. FCF Yield gets the heaviest weight because cash flow is the hardest number for management to manipulate — it's cash in the door. For banks, EV-based metrics are meaningless (deposits are both liabilities and the core business), so P/B replaces them.

---

### 2. Quality ({fw.get('quality', 0)}% of final score)

**Question it answers:** *Is this a well-run business with durable competitive advantages?*

**Generic stocks:**

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(qual_w, _QUAL_DESCRIPTIONS)}

**Bank-like stocks:**

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(qual_bank_w, _QUAL_DESCRIPTIONS)}

**Why these?** A cheap stock is only a good investment if the underlying business is sound. ROIC is the single best measure of business quality — the ROIC formula deducts only *excess* cash (cash beyond 2% of revenue) from invested capital, preventing cash-rich companies like AAPL or GOOG from showing artificially inflated returns. For banks, ROIC is meaningless (invested capital = deposits + equity), so ROE and ROA replace it. The Piotroski F-Score catches deteriorating businesses by checking 9 binary signals about whether profitability, leverage, and efficiency are improving or declining. Accruals catch companies whose reported earnings aren't backed by real cash.

---

### 3. Growth ({fw.get('growth', 0)}% of final score)

**Question it answers:** *Is this business growing, and can it sustain that growth?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(grow_w, _GROWTH_DESCRIPTIONS)}

**Why these?** Forward EPS Growth gets the most weight because it's forward-looking (the market prices in the future, not the past). Revenue growth and the three-year revenue CAGR measure what has already happened, at two horizons. Sustainable Growth acts as a sanity check — if a company is growing faster than its sustainable rate, it may need external financing to keep it up. The PEG ratio is computed and shown but carries no weight: it divides a valuation by a growth rate, so it would count Valuation a second time inside Growth.

---

### 4. Momentum ({fw.get('momentum', 0)}% of final score)

**Question it answers:** *Has the market been rewarding this stock recently?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(mom_w, _MOM_DESCRIPTIONS)}

**Why these?** Decades of academic research (Jegadeesh & Titman, 1993) show that stocks that have gone up tend to keep going up over 3-12 month horizons. The skip-month convention (excluding the most recent month) is the standard academic momentum signal — the last month is excluded because very recent winners tend to experience a brief pullback. Both metrics use calendar-based date targeting instead of fixed index offsets, which ensures consistent lookback periods regardless of holidays or trading day variations.

**Momentum regime rule - currently {regime_state}.** Momentum strategies crash most often in volatile markets (Daniel & Moskowitz 2016), and scaling momentum down when its own volatility is high improves it (Barroso & Santa-Clara 2015). Until 2026-10-09 the screener tried to do this, but the input it used was the spread of the momentum score across stocks (`factor_vol_history.csv`), which is fixed by the percentile construction and does not measure market volatility: it called 30 of 33 runs "low volatility" and raised momentum's weight most days. The rule is switched off until it is rebuilt on a real volatility measure; the spread is still recorded each run.

---

### 5. Risk ({fw.get('risk', 0)}% of final score)

**Question it answers:** *How bumpy is the ride?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(risk_w, _RISK_DESCRIPTIONS)}

**Why these?** All else equal, less volatile stocks are preferable — the "low volatility anomaly" is one of the most robust findings in finance. Volatility measures total risk, Beta measures systematic risk, and Max Drawdown captures worst-case loss — a stock that drops 50% needs a 100% gain to recover. All three are *dispersion* measures: they describe how much a stock moves, not how well it did.

**Why not Sharpe and Sortino?** They were scored here until 2026-09-02, at 15% each. Both are `(12-month return − risk-free rate) ÷ some measure of dispersion`, so they share their numerator with the momentum signal. Across the S&P 500 the spread in returns is far wider than the spread in volatility, so the numerator dominates: measured on the published payload, Sharpe correlates **+0.944** with the 12-1 month return but only **+0.025** with volatility. Scoring them inside Risk meant a stock was rated safer because it had gone up — which pushed the Risk and Momentum category scores to a **+0.516** correlation, the highest of any pair in the screener. Removing them drops that to **+0.150**. Both ratios are still computed and shown on each stock's detail page; they are simply no longer scored as risk. See `METHODOLOGY_CHANGELOG.md` 2026-09-02.

**Why not operating leverage?** It was 8% of Quality until 2026-10-09, scored lower-is-better as "more durable earnings". As built it was one year's percentage change in operating profit divided by one year's percentage change in revenue, and that ratio does not measure cost structure: when profit and revenue move in opposite directions it goes negative, and on the 2026-10-09 run **85 of its 95 negative values were companies whose revenue grew while operating profit fell** - shrinking margins - which the score ranked at the 84th percentile of their sectors. A small revenue change also inflates it. The research points the other way too: firms with more operating leverage have historically earned *higher* returns as compensation for the risk (Novy-Marx 2011), and neither MSCI's nor AQR's published quality definitions use it - both measure durability as how variable earnings have been over several years. Its weight went to the other Quality metrics in proportion; it is still computed and shown. See `research/2026-10-09-operating-leverage.md`.

---

### 6. Analyst Revisions ({fw.get('revisions', 0)}% of final score)

**Question it answers:** *What do Wall Street analysts think — and are they getting more or less optimistic?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(rev_w, _REV_DESCRIPTIONS)}

**Why these?** Estimate revisions and analyst targets are among the most powerful short-term return predictors. **{rev_heaviest} gets the highest weight** because it is the one metric here that measures what the category is named for — analysts revising their forecasts. Chan, Jegadeesh & Lakonishok (1996, *Journal of Finance* 51(5)) found the analyst-revision leg of earnings momentum to be the strongest of the three they tested, a **+7.7% six-month decile spread** on IBES data 1977–93.

The surprise family — Analyst Surprise and Beat Score — is *backward*-looking: it records companies beating a past estimate and bets that the price keeps drifting afterwards. That drift is what the literature calls post-earnings-announcement drift, and Martineau (2022, *Critical Finance Review* 11(4)) finds it has been **absent in large caps since 2006**, with a significantly *negative* coefficient over 2016–19. Since this is an S&P 500 screener, that is exactly this universe — which is why the surprise family was cut from 78% of the category to 45% on 2026-09-10, and to 25% on 2026-10-09 when Earnings Acceleration (the latest surprise minus the one before) left the score: it marked a stock down for having beaten the previous quarter, though surprises tend to repeat. The revision metric now carries nearly half the category.

This category is weighted at only {fw.get('revisions', 0)}% of the composite because coverage can be sparse (not all stocks have active analyst coverage), and when coverage drops below usable levels, the weight automatically redistributes to the other categories.

*Note on the limits of this data: yfinance's estimate history reaches back only **90 days**, so the screener can see a one-quarter revision but not whether a revision trend has **persisted**. Chan, Jegadeesh & Lakonishok's strongest result used a six-month window, which remains out of reach without a paid consensus feed (FactSet, Refinitiv I/B/E/S). The metric itself is not out of reach and has been live since 2026-09-10 — an earlier version of this page said otherwise.*

---

### 7. Size ({fw.get('size', 0)}% of final score)

**Question it answers:** *Does this stock benefit from the small-cap premium?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(size_w, _SIZE_DESCRIPTIONS)}

**Why this?** The Fama-French SMB (Small Minus Big) factor captures the historical tendency for smaller companies to outperform larger ones over long horizons. Within the S&P 500 this tilts toward mid-cap names (still large-cap by absolute standards) rather than megacaps.

**What the log does, and what it does not do.** The metric is stored as `-log(marketCap)`, but no stock's score depends on the log. Every metric here is converted to a **sector percentile rank** (see "How scoring works" below), and a rank is unchanged by any transformation that preserves order. Measured on the live universe, `rank(-log mcap)`, `rank(-mcap)` and `rank(-sqrt mcap)` are identical to ten decimal places. The log makes the stored number easier to read; it does not compress the tilt.

**So how strong is the tilt?** Stronger than the metric's name suggests, which is worth stating plainly. Because the score is linear in rank, ordinary market-cap gaps become large score gaps: on the current universe CAT ($364B) scores 1 and UAL ($35B) scores 59, so a 10x cap ratio becomes a 57-point gap. For comparison, MSCI's Low Size indexes weight holdings in proportion to 1/ln(mcap), which across this same universe turns a **798x** spread in market cap into a **1.295x** spread in weight. This screener runs an equal-weight-style size tilt, not a log-compressed one. That is a defensible design - it is close to the bet the S&P 500 Equal Weight index makes - but it is a *stronger* bet than the name implies, and holding it to 5% of the composite is what keeps it proportionate.

**Known weakness.** No published study establishes a size premium *within* the largest two market-cap deciles, which is the entire S&P 500. Applying SMB here extrapolates from research run on much broader universes, and the payoff is regime-dependent: S&P 500 Equal Weight has returned roughly +63 bps/yr since 1990 but trailed cap weighting by about 32% over 2023-2025. Detail and citations: `research/2026-08-31-size-factor-in-a-large-cap-universe.md`.

---

### 8. Investment ({fw.get('investment', 0)}% of final score)

**Question it answers:** *Is this company investing conservatively or aggressively expanding its asset base?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(inv_w, _INV_DESCRIPTIONS)}

**Why this?** The Fama-French CMA (Conservative Minus Aggressive) factor captures the historical tendency for companies that invest conservatively to outperform those that aggressively expand their asset base. High asset growth often signals empire-building, dilutive acquisitions, or capex that won't generate adequate returns. The screener rewards companies that grow efficiently rather than just growing big.

When coverage drops below 30% (e.g., many stocks lack prior-year asset data), the Investment category is automatically disabled and its weight redistributes to the other categories.

---

## Bank-Specific Scoring

Traditional financial metrics like EV/EBITDA, ROIC, and Debt/Equity are meaningless for banks, insurers, and credit companies. Their "debt" is deposits (the raw material of their business), they don't have conventional capital expenditures, and enterprise value metrics break down when liabilities include customer deposits.

The screener decides by **GICS sub-industry** (since 2026-10-09), from the S&P 500 list itself:

1. **Bank set** — Diversified and Regional Banks, Consumer Finance, Mortgage Finance, Life & Health / Multi-line / Property & Casualty Insurance, Reinsurance, Multi-Sector Holdings (Berkshire) and Investment Banking & Brokerage: businesses whose liabilities — deposits, insurance float, customer funds — are an operating input.
2. **Generic set** — Insurance Brokers, Asset Management, Financial Exchanges & Data, and Payment Processing: fee businesses with conventional profit and loss statements, valued in practice on EV/EBITDA and P/E.
3. **Named exceptions** inside Asset Management & Custody Banks use the bank set, each for a stated reason: the custody banks BNY, State Street and Northern Trust (they take deposits), Apollo and KKR (they consolidate the insurers Athene and Global Atlantic) and Ameriprise (it owns a bank and a life insurer).

A Financials stock in a sub-industry on neither list defaults to the bank set — the safer guess for an unseen lender — and the run logs it by name; a test keeps the current universe at none. Until 2026-10-09 the rule read Yahoo's industry names, and 26 of the 59 stocks scored as banks — asset managers and insurance brokers among them — reached the bank set only through that default. Detail: `research/2026-10-09-bank-like-financials.md`.

Bank-like stocks get an entirely different set of metric weights within the Valuation and Quality categories (see the tables in sections 1 and 2 above). Growth, Momentum, Risk, Revisions, Size, and Investment use the same generic weights for all stocks.

All financial-sector stocks receive a `Financial_Sector_Caveat` flag in the output, reminding the user that financial companies require additional scrutiny regardless of classification.

---

## How the Score Is Calculated

The scoring pipeline has six steps:

### Step 1: Collect Raw Data
For each of the ~500 stocks, the screener pulls quarterly financial statements, price data, earnings history, and analyst estimates from Yahoo Finance. Flow metrics (income statement and cash flow) use **LTM** (Last Twelve Months = sum of 4 most recent quarters); balance sheet items use **MRQ** (Most Recent Quarter). Falls back to annual filings if quarterly data is unavailable. Enterprise Value is cross-validated against computed MC + Debt - Cash; discrepancies > 10% (25% for Financials) trigger automatic correction.

Several inputs arrive from **two places** — Yahoo's summary fields and the filed statements — and the screener prefers one but uses the other when the first is missing: total debt and cash for Enterprise Value prefer the summary figure (it matches Yahoo's own EV definition), while invested capital for ROIC prefers the balance sheet (so equity, debt and cash come from one filing). Since 2026-09-25 those fallbacks actually fire; before that a bug meant they never did, and three S&P 500 companies lost a metric on every run despite the data being present. See `METHODOLOGY_CHANGELOG.md` 2026-09-25. Data is cached locally in Parquet format (refreshed daily for prices, weekly for fundamentals) to avoid unnecessary API calls. Cache files are config-aware — changing weights or settings automatically invalidates stale caches.

### Step 2: Flag Outliers (but do not change them)
Every metric below is scored by its **rank** within its sector, and a rank does not care how far away an outlier is — only that it is last. A company with a Debt/Equity of 50x when everyone else is under 5x ranks worst either way. So the screener does **not** clip extreme values: it records them in the data-quality log (the tails beyond the {out_lo}st and {out_hi}th percentiles) and scores the number it actually fetched.

Until 2026-09-01 it did clip them, and that was a mistake in two directions. Clipping could not improve a single ranking, because ranking is unaffected by it. What it could do — and did — was flatten several companies onto one identical number, which then made them tie in the ranking, and publish that clipped number as the company's real figure. On the last run before the fix, six companies were all shown with a market capitalisation of $2,802B; Nvidia's true figure was $5,331B.

### Step 3: Rank Within Sectors
Each metric is converted to a **sector-relative percentile** (0-100). A stock's EV/EBITDA isn't compared to all 500 companies — it's compared only to other companies in the same GICS sector (Technology vs. Technology, Energy vs. Energy, etc.). This is critical because a "cheap" utility trades at a very different multiple than a "cheap" tech company. Sector-relative ranking makes apples-to-apples comparisons possible.

For metrics where lower is better (like EV/EBITDA, Debt/Equity, Volatility, P/B, PEG Ratio, Asset Growth), the percentile is flipped so that a higher percentile always means "better." The percentile uses the midpoint rule - the k-th lowest of n stocks scores (k - 0.5) / n x 100 - so a metric averages exactly 50 whichever way it points. (Until 2026-10-09 it was k / n, which averaged 50 + 50/n for higher-is-better metrics and 50 - 50/n once flipped, a tilt that grew in small sectors.)

**Small-sector fallback:** When a sector has fewer than 10 stocks with valid data for a metric, ranking within that tiny group produces noisy percentiles. In these cases, the screener falls back to universe-wide percentile ranking for that metric, which provides a more stable signal than the previous approach of assigning a flat 50th percentile.

{pt_text}

### Step 4: Combine Into Category Scores
Within each of the {n_factors} categories, the individual metric percentiles are combined using the configured weights. For example, the generic Valuation score is:

```
Valuation = {val_formula}
```

For bank-like stocks, the weights come from the bank-specific weight table instead:

```
Valuation (bank) = {bank_val_formula}
```

**Missing data handling:** When a metric has no data for a particular stock (NaN), that metric is excluded and its weight is redistributed proportionally across the metrics that do have data. This means a stock isn't penalized for a missing metric — it's scored on whatever data is available. If an entire metric is NaN across the full universe (e.g., a data source outage), it is automatically skipped for the category.

This produces {n_factors} category scores (0-100 each).

### Step 5: Combine Into Composite Score
The {n_factors} category scores are combined using the category weights:

```
Raw Composite = {' + '.join(composite_parts)}
```

The same missing-data redistribution logic applies: if a category score is NaN (e.g., all revisions data missing for a stock), its weight is redistributed to available categories rather than producing a NaN composite.

**The weights above are the configured defaults, and an individual run may not use them.** Two rules move them, both described in this document:

1. **The momentum regime rule**, when switched on, changes the momentum weight for the whole run (see the Momentum section). It is off since 2026-10-09.
2. **Missing-data redistribution** changes them for one stock, whenever a category could not be scored for it.

So a stock's momentum score may be multiplied by something other than the {fw.get("momentum", 0)}% printed above. Rather than ask you to take that on trust, the dashboard's stock drilldown shows **the weight each score was actually multiplied by**, and explains any gap against this page — every row there is an equation you can check with a calculator. The run's own weights are also written to `runs/<run_id>/effective_weights.json`.

**The composite is cardinal, and it is not a percentile.** The weighted average above is kept as a 0-100 score with its magnitude intact, and that score is the ranking key — so a stock twenty points clear of the field and one a tenth of a point clear are not both reported as 100.{cov_step} The universe percentile is a **separate** column, `Composite_Pct` (`rank(pct=True) * 100`, "better than X% of stocks"), carried for display only. So do not read a composite of 95 as "better than 95% of the universe" — read it as 95 points out of 100. See Limitation 8.

### Step 6: Apply Trap Filters & Rank
After computing composite scores, the screener applies value trap and growth trap filters (see below), then produces the final ranking.

---

{pio_text}

## Data Quality Safeguards

The screener includes several layers of data quality protection:

- **Denominator floors:** Analyst surprise uses a $0.10 floor on estimated EPS; forward EPS growth uses a $1.00 floor on trailing EPS. These prevent near-zero denominators from producing extreme ratios.
- **Output clamping (configurable):** Forward EPS growth is clamped to [{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[0] * 100)}%, +{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100)}%]; price target upside is clamped to [{int(clamps.get('price_target_upside', [-0.50, 1.0])[0] * 100)}%, +{int(clamps.get('price_target_upside', [-0.50, 1.0])[1] * 100)}%]. These bounds are configurable in `config.yaml` under `metric_clamps`. They limit the impact of data anomalies (e.g., GAAP vs. normalized EPS mismatches, extreme analyst targets) while still allowing meaningful differentiation among high-growth stocks.
- **Coverage filter:** Stocks with fewer than {min_coverage}% of the metrics that carry weight in their score available are excluded from the ranking entirely - the same coverage the composite's discount reads.
- **Coverage discount:** Stocks that pass the coverage filter but still have many missing metrics receive a mild composite discount. Below {int(cov_disc.get('threshold', 0.80) * 100)}% metric coverage, the composite is reduced by up to {int(cov_disc.get('penalty_rate', 0.15) * 100)}% per unit of coverage gap (e.g., a stock at 56% coverage gets a ~3.6% discount). This prevents stocks with sparse data from ranking artificially high due to weight redistribution concentrating the score on a few favorable metrics. {'**Currently enabled.**' if cov_disc.get('enabled', False) else '**Currently disabled.**'}
- **Auto-disable (category-level):** If the Revisions or Investment category has fewer than 30% of its metrics populated, the entire category's weight is zeroed and redistributed proportionally to the remaining categories.
- **Auto-reduce (metric-level):** If any individual metric has more than {auto_reduce_thresh}% NaN across the universe (e.g., a data source outage), its weight is automatically set to zero and redistributed within its category.
- **Metric-level alerts:** A warning is printed if any metric has more than {alert_thresh}% missing data across the universe.
- **LTM / MRQ data freshness:** All flow metrics (revenue, net income, EBITDA, cash flow) use LTM (Last Twelve Months = sum of 4 most recent quarters). Balance sheet items use MRQ (Most Recent Quarter). This reduces data staleness from up to 12 months (annual filings) to ~3 months. Falls back to annual filings if quarterly data is unavailable; prior-year comparisons fall back to annual col=1 when quarterly history is insufficient (< 8 quarters).
- **EV cross-validation:** The API-provided Enterprise Value is cross-checked against computed MC + Debt - Cash. If the discrepancy exceeds 10% (or 25% for Financials, whose "debt" includes customer deposits that legitimately diverge from simple EV math), the computed value is used and the ticker is flagged (`_ev_flag`). This catches known yfinance EV parsing bugs (4x+ discrepancy for some tickers).
- **LTM partial annualization tracking:** When only 3 of 4 quarters are available for a flow metric, the screener annualizes (sum × 4/3) but flags the ticker with `_ltm_annualized = True` and records which fields were affected. This transparency lets users know which metrics are based on extrapolated rather than complete data.
- **Channel-stuffing detection:** Compares receivables with revenue over the last fiscal year, both from the same two annual statements, using Beneish's days-sales-in-receivables index (receivables / revenue, over the prior year's). At **1.465 or more** - the average among the earnings manipulators in Beneish (1999), against 1.031 among the rest - the stock is flagged with `_channel_stuffing_flag = True`. This can indicate aggressive revenue recognition or deteriorating collection quality. Not applied to any Financials stock: Beneish's sample excluded financial firms.
- **Beta overlap validation:** Beta computation requires at least 80% date overlap between the stock's daily returns and the S&P 500 market returns. Stocks with insufficient overlap get `beta = NaN` rather than a potentially misleading value. The overlap percentage is recorded in `_beta_overlap_pct`.
- **Data quality log:** Every data issue (missing fields, stale data, rate-limit failures) is logged to `validation/data_quality_log.csv` with ticker, severity, description, and action taken.
- **Structured pipeline logging:** A Python `logging`-based structured logger (`screener.pipeline`) records coverage statistics, filter actions, and scoring stage completions for machine-parseable diagnostics.

---

## Value Trap Detection

A stock can score well on valuation (cheap!) but be cheap for a reason — declining business, negative momentum, or analysts cutting estimates. A stock is flagged as a potential value trap only if it is **cheap** — a Valuation score in the top {100 - vt_cheap}% of the universe — **and** it falls in the bottom {vt_quality_floor}% of **at least two** of these three categories:

- Quality Score (floor: {vt_quality_floor}th percentile)
- Momentum Score (floor: {vt_mom_floor}th percentile)
- Revisions Score (floor: {vt_rev_floor}th percentile)

The cheapness condition is what makes it a *value* trap: Piotroski (2000) separates the cheap stocks that go on to do well from those that do not using exactly this kind of fundamental weakness, within the cheapest stocks. Without it (before 2026-10-09) the flag fired on about a quarter of the universe, whatever the valuation. The 2-of-3 majority logic tolerates a single weak dimension (e.g., a quality stock with one bad momentum quarter); "any 1 breach" flagged roughly 60% of the universe.

Missing data (NaN) in any of the three dimensions does **not** trigger a value trap flag — missing data is not the same as poor quality. These stocks receive a separate `Insufficient_Data_Flag`.

Each flagged stock also receives a **Value Trap Severity** score (0-100): for each of the three dimensions, how far below its threshold the stock falls (as a share of the threshold, zero if above it), averaged over the three. A severity of 80 means the stock is deep in trap territory; a severity of 20 means it barely crossed the thresholds. This provides more granularity than the binary flag alone.

By default, value-trap-flagged stocks are {vt_action} from the model portfolio (configurable to flag-only mode).

---

## Growth Trap Detection

The mirror image of a value trap: a stock can score well on growth but be growing unsustainably — high growth with poor quality and/or deteriorating analyst sentiment. A stock is flagged as a potential growth trap only if its Growth Score is **above** the {gt_growth_ceil}th percentile (high growth) **and** at least one of these holds:

- Quality Score **below** the {gt_quality_floor}th percentile (low quality)
- Revisions Score **below** the {gt_rev_floor}th percentile (deteriorating sentiment)

High growth is required (since 2026-10-09; it used to be one of three conditions, so a low-growth stock with low quality and low revisions could be called a growth trap). Mohanram (2005) separates winners from losers within growth stocks using fundamental strength, the mirror of Piotroski's test within value stocks.

This catches "growth at any price" stocks — companies that are growing fast but burning cash, carrying deteriorating fundamentals, or losing analyst confidence.

Each flagged stock also receives a **Growth Trap Severity** score (0-100): how far beyond each of the three thresholds the stock falls (zero where it does not cross one), averaged over the three. Higher severity means deeper in trap territory.

By default, growth-trap-flagged stocks are {gt_action} from the model portfolio (configurable to flag-only mode).

---

## Portfolio Construction

**This is an Excel/artifact output, not a dashboard feature.** The dashboard was
never the place for it and stopped showing it on 2026-08-26: a named, fixed list
of holdings published to a public site reads as a recommendation, and this tool
is decision support - it shows you *why* a stock ranks where it does and leaves
the buy/sell/size judgement to you. The ranking on the dashboard is the product.
What follows describes the `ModelPortfolio` sheet in the generated workbook, for
research use by the owner.

After scoring and ranking, the screener builds a **model portfolio** from the top-ranked stocks:

- **Number of holdings:** Top {num_stocks} stocks (configurable)
- **Weighting:** {weighting_description(weighting, num_stocks)}
- **Sector cap:** Maximum {max_sector} stocks from any single sector, to avoid overconcentration
- **Position limits:** No single stock above {max_pos}%{_max_pos_note(weighting, num_stocks, max_pos)}
- **Liquidity filter:** Stocks with less than ${min_adv_m:.0f}M average daily dollar volume (63-day average) are excluded from the portfolio. Stocks with missing volume data are also excluded (conservative default).
- **Trap exclusions:** Value-trap and growth-trap flagged stocks are excluded (unless configured as flag-only)

If a sector would exceed its cap, the excess stocks are dropped and replaced by the next-highest-ranked stocks from other sectors. Weights are redistributed proportionally.

---

## What Gets Output

The screener produces an **Excel workbook** (`factor_output.xlsx`) with up to 7 sheets (a ReadMe/Disclaimers sheet leads the workbook):

### Sheet 1: Factor Scores
Every stock in the universe with all raw metrics, {n_factors} category scores, the composite score, rank, value trap flag (with severity 0-100), growth trap flag (with severity 0-100), financial sector caveat flag, bank classification, and bank-specific metrics (P/B, ROE, ROA, Equity Ratio) where applicable. Each stock also carries a data provenance tag (`_data_source`), metric coverage count, and an EPS basis mismatch flag. Score columns use quartile-based coloring (Q1=red, Q2=yellow, Q3=light green, Q4=green) for at-a-glance assessment.

### Sheet 2: Screener Dashboard
The top 50 stocks, formatted for quick review. Includes rank, composite score (quartile-colored), all {n_factors} category scores, and the value trap and growth trap flags with severity scores. Color-coded cells highlight strengths and weaknesses.

### Sheet 3: Model Portfolio
The final portfolio with ticker, sector, composite score, position weights, and portfolio-level statistics (weighted average beta, dividend yield, sector allocation breakdown).

### Sheet 4: DataValidation
The top 10 stocks with raw financial values (market cap, revenue, EPS, etc.) displayed for manual spot-checking. Highlights potential issues including EPS basis mismatches (GAAP vs. normalized), stale data, EV cross-validation discrepancies, LTM partial annualization flags, channel-stuffing flags (receivables rising much faster than revenue over the fiscal year), and beta overlap warnings. Also includes a sector-median context table showing 25th/median/75th percentile for 8 key metrics across each sector.

### Sheet 5: Weight Sensitivity (when available)
Results of the weight sensitivity analysis. For each factor category, the sheet shows what happens to the top-20 ranking when that category's weight is perturbed ±5%. Jaccard similarity measures how stable the ranking is — higher values (≥0.85) mean the ranking is robust to small weight changes. Color-coded: green (≥0.85), yellow (0.70–0.84), red (<0.70).

### Sheet 6: Factor Correlation (when available)
Spearman rank correlation matrix of all {n_factors} category scores across the universe. Highlights potential double-counting: correlations above 0.6 (orange) or 0.8 (red) indicate factor overlap. Useful for understanding effective independent factor count.

Additional outputs:
- **Parquet cache** (`cache/factor_scores_<hash>_<date>.parquet`) — full scored dataset for programmatic access, tagged with a config hash for reproducibility.
- **Data quality log** (`validation/data_quality_log.csv`) — every data issue encountered during the run.
- **Run artifacts** (`runs/<run_id>/`) — raw fetch data, scored data, and config snapshot for each run, enabling full reproducibility via `RunContext`.

---

## Factor-Exposure Diagnostics

A standalone script (`factor_exposure.py`) is available for analyzing how much of the portfolio's returns are explained by known academic risk factors. It runs a Fama-French 5-factor + Momentum (UMD) regression:

```
Portfolio_ExcessReturn ~ Mkt-RF + SMB + HML + RMW + CMA + UMD
```

This tells you:
- **Alpha** — returns not explained by any known factor (genuine stock selection skill)
- **Factor betas** — how much the portfolio tilts toward market risk, size, value, profitability, investment, and momentum
- **R-squared** — what fraction of portfolio return variation is explained by the factors

Usage:
```bash
python factor_exposure.py
python factor_exposure.py --start 2024-01-01 --end 2025-12-31
```

Requires: `pandas-datareader` and `statsmodels` (listed in `requirements.txt`).

---

## Reproducibility

Every screener run is assigned a unique run ID and tracked via `RunContext`. This provides:

- **Run artifacts:** Raw fetch data, scored results, and the config snapshot used are saved to `runs/<run_id>/`.
- **Config-aware caching:** Cache filenames include a hash of the scoring configuration, so changing weights or thresholds automatically invalidates stale caches.
- **Deterministic scoring:** Given the same input data and configuration, the scoring pipeline produces identical results.

---

## Defensibility & Transparency Features

The screener includes several features designed to make its outputs auditable and defensible:

### Weight Sensitivity Analysis
After scoring, the pipeline perturbs each factor category weight by ±5% (one at a time) and measures how much the top-20 ranking changes using **Jaccard similarity** (intersection / union of the two top-20 sets). A Jaccard of 1.0 means the ranking is completely unchanged; below 0.70 suggests the ranking is sensitive to that factor's weight. Results are printed to the console and saved in the Weight Sensitivity Excel sheet. This lets you verify that small weight changes don't drastically alter the output — a key requirement for any defensible quantitative process.

### EPS Basis Mismatch Detection
Yahoo Finance provides GAAP trailing EPS but normalized (non-GAAP) forward consensus EPS. When the ratio of forward-to-trailing EPS exceeds 2.0× or falls below 0.3× (and trailing EPS is above $0.10), the stock is flagged with `_eps_basis_mismatch = True`. This alerts users that the forward EPS growth and PEG ratio metrics may be distorted by a GAAP/non-GAAP mismatch rather than a genuine change in earnings expectations. Flagged stocks appear highlighted in the DataValidation sheet.

### Factor Correlation Matrix
A Spearman rank correlation matrix of all category scores is computed and written to the Factor Correlation Excel sheet. This makes explicit the degree of overlap between factors — for example, Momentum's two sub-metrics (12-1M and 6-1M return) share ~6 months of overlap, and EV-based valuation metrics are structurally correlated. Correlations above 0.6 are highlighted orange; above 0.8 are highlighted red. This transparency allows users to assess the effective number of independent signals.

### Data Provenance
Every stock carries three provenance fields: `_data_source` (where the data came from — e.g., "yfinance", "cache", "sample"), `_metric_count` (how many of the metrics that apply to that stock have a value), and `_metric_total` (how many apply to it: fewer for a bank than for a non-bank). This makes per-stock data completeness visible at a glance, and it is the same count the composite's coverage discount reads. The drilldown also states the date of the balance sheet, income statement and cash-flow statement each figure comes from.

### The Workings Behind Every Score
Open any stock and **The workings** section rebuilds its score from the bottom. For each category it lists every metric with a weight: the raw value, the sector percentile it earned, the weight it actually carried for *this* stock (bank weighting, the Piotroski conditional adjustment, and rescaling when a metric has no data all change it), and the points it contributed. The points add up to the category score, the category points add up to the composite, and a coverage discount, where one applies, is shown as its own line. Each metric row opens to its formula, the reported figures behind it (market cap, EBIT, cash flow, prices, and so on), who it was ranked against and where it placed, and a plain note wherever the code does something its name would not suggest.

The page does not ask to be taken on trust. The data run rebuilds every category score and composite from the published weights before it will publish, and refuses to publish a day on which any stock fails to reproduce; `scripts/audit_stock.py` repeats the whole calculation for any stock with code that shares nothing with the scoring engine, reading only the published data file. What it cannot do is check that the underlying figures are true: they are as reported by each company and delivered by Yahoo Finance, and the workings show that they were used consistently.

### DataValidation Sheet
The top 10 portfolio stocks are displayed with raw financial values (market cap, revenue, net income, EPS, price) for manual spot-checking against external sources (e.g., Bloomberg, SEC filings). The sheet highlights six types of potential issues: EPS basis mismatches, stale data (price targets that may be outdated), EV cross-validation discrepancies, LTM partial annualization (3-of-4 quarters extrapolated to LTM), channel-stuffing flags (Beneish's receivables index at 1.465 or more), and beta overlap warnings (<80% date overlap with market). A sector-median context table shows 25th/median/75th percentile for 8 key metrics across each sector, enabling quick sanity checks.

---

## Key Design Decisions & Why

| Decision | Why |
|----------|-----|
| **{n_factors} factor categories** ({', '.join(factor_labels[f] for f in active_factors)}) | Captures the 5 Fama-French factors (MktRF, SMB, HML, RMW, CMA) plus momentum and analyst sentiment. Broad coverage reduces reliance on any single factor. |
| **Sector-relative percentiles** (not universe-wide) | A 10x EV/EBITDA is cheap for Tech but expensive for Utilities. Ranking within sectors makes comparisons fair. |
| **Small-sector fallback to universe-wide ranking** | Sectors with <10 stocks produce noisy percentiles. Falling back to universe ranking is more informative than a flat 50th percentile. |
| **Valuation + Quality as the two largest categories** ({fw.get('valuation',0)}% each) | These are the two most robust factors in academic literature. Growth and Momentum get {fw.get('growth',0)}% each — they're powerful but noisier. Size and Investment get {fw.get('size',0)}% each as supplementary signals. |
| **FCF Yield as the top valuation metric** ({val_w.get('fcf_yield',0)}% weight) | Cash flow is harder to manipulate than earnings. FCF Yield is the purest measure of how much cash a business generates per dollar of value. |
| **Bank-specific metric weights** | EV/EBITDA, ROIC, and D/E are meaningless for banks. P/B, ROE, ROA, and Equity Ratio are the standard bank analysis toolkit. |
| **ROIC excess cash deduction** (cash - 2% revenue) | Deducting ALL cash inflates ROIC for cash-rich companies (e.g. AAPL, GOOG). Keeping 2% of revenue as operating cash provides a more accurate invested capital base. |
| **ROIC tax-loss handling** (0% tax rate when pretax < 0) | Companies with negative pretax income are in a tax-loss position and would not pay tax. Using the statutory 21% rate would create a fictional tax charge that understates NOPAT. |
| **EV cross-validation** (API vs MC+Debt-Cash) | yfinance has known EV parsing bugs (4x+ discrepancy for some tickers). When the API-provided EV differs from the computed value by more than 10% (25% for Financials), the computed value is used and the discrepancy is flagged. Financials use a wider tolerance because their "debt" includes deposits and other liabilities that structurally diverge from simple EV math. |
| **Momentum skip-month** (12-1 and 6-1, not 12-0) | The most recent month's return tends to reverse. Skipping it improves signal quality (standard in academic momentum literature). |
| **Calendar-based lookbacks** | Using calendar dates (e.g., 182 days ago) instead of fixed index offsets ensures consistent lookback periods regardless of holidays. |
| **Denominator floors** ($0.10 for surprise, $1.00 for EPS growth) | Near-zero denominators produce extreme ratios that dominate rankings. Floors bound the maximum possible ratio. |
| **Outliers flagged, never clipped** ({out_lo}%/{out_hi}% tails) | Sector ranking is a rank transform, so clipping cannot change any ordering — it can only create artificial ties and misreport the company's real figure. Extreme values are logged as a data-quality signal instead, which is also how a bad feed gets caught. |
| **Value trap: cheap, and weak on 2 of 3** | A value trap is a cheap stock that is cheap for a reason (Piotroski 2000), so cheapness is required. OR logic (any 1 breach) flagged ~60% of the universe; majority logic tolerates one bad dimension. |
| **Growth trap: high growth, and weak quality or revisions** | Mirror of value trap (Mohanram 2005). High growth is required, so a low-growth stock is never called a growth trap. |
| **Liquidity filter** (${min_adv_m:.0f}M daily dollar volume) | Ensures portfolio stocks are tradeable at scale. NaN volume is excluded conservatively. |
| **Revisions led by the estimate revision** (FY1 revision + Surprise + Beat Score + Target + Short interest) | The 90-day change in the consensus estimate carries the most weight because revisions, unlike surprises, still predict returns in large caps (Chan, Jegadeesh & Lakonishok 1996; Martineau 2022 on the surprise drift's absence since 2006). Earnings Acceleration is computed but unweighted since 2026-10-09. |
| **Momentum regime rule: off** | Momentum crashes in high-volatility markets, but the input the rule used (the spread of momentum scores across stocks) does not measure volatility. Off until rebuilt on a real volatility measure. |
| **3-metric risk category** (Vol + Beta + MaxDD), dispersion only | Volatility captures total risk, Beta systematic risk, Max Drawdown tail risk. Sharpe and Sortino were dropped from scoring on 2026-09-02: both divide the *same* trailing return by a dispersion measure, so they correlated +0.993 with each other and +0.944 with the momentum signal, but only +0.025 with volatility. They were five metrics in name and three in substance, and the two extras were momentum wearing a risk label. Institutional risk models are built the same way — the Barra US Equity Model's volatility style factors use dispersion descriptors (daily standard deviation, cumulative range, residual sigma), not return/risk ratios. |
| **Quartile-based Excel coloring** | Absolute thresholds (e.g., >80 = green) assume a stable score distribution. Quartile-based coloring adapts to the actual distribution, ensuring roughly 25% of cells in each color band regardless of market conditions. |
| **Trap severity scores** (0-100 continuous) | Binary flags lose information. Severity scores quantify how deep in trap territory a stock is — severity 80 is much worse than severity 20, but both would be flagged as True. |
| **Beta overlap validation** (≥80% required) | Stocks with limited trading history (IPOs, relisted) can produce misleading beta values from sparse overlap with the market index. The 80% threshold ensures the regression uses substantially the same time period as the market. |
| **Channel-stuffing detection** (Beneish's receivables index) | Receivables rising far faster than revenue over the same fiscal year may indicate aggressive revenue recognition; the cut, 1.465, is the manipulators' average in Beneish (1999). The flag is informational (not used in scoring). |

---

## Limitations to Be Aware Of

1. **Data source:** All data comes from Yahoo Finance (free, unofficial API). Occasional field name changes, rate limiting, or missing data are handled gracefully (the screener returns NaN and continues), but the data quality is not institutional-grade — individual fields go missing for individual companies, and the screener drops the affected metric rather than guessing at it.

   **Fetch reliability, measured:** the 18 scheduled runs from 2026-09-02 to 2026-09-25 report **0 fetch failures across 9,036 ticker-fetches**. This page previously put the failure rate at 10-25% of the universe per run, attributing it to HTTP 429 rate limiting. That figure dated from the project's launch period and nobody had re-checked it in the months since; *corrected 2026-09-25*. Rate limiting remains possible on a free API, which is why a run that fetches badly is **discarded before publication** rather than shipped: `scripts/check_run_health.py` refuses a run with price coverage below 90%, analyst-target coverage below 50%, or factor dispersion more than 20% below its trailing median. A number on this site has passed that gate.

2. **GAAP vs. normalized EPS:** Yahoo Finance provides GAAP trailing EPS but normalized forward consensus. For companies with large non-cash charges, write-downs, or unrealized gains (e.g., insurers like CINF), the two bases diverge. As of the 2026-07 review, when the forward/trailing EPS ratio is extreme (>2x or <0.3x — the signature of this contamination), `forward_eps_growth` is set to NaN and its weight is redistributed, rather than scoring a fabricated growth figure.

3. **Point-in-time:** The screener uses the latest available financial data. It does not reconstruct what was known at a past date, which means backtests carry look-ahead bias for fundamental metrics.

4. **Analyst coverage:** The Revisions category relies on analyst estimate and price target data, which is sparse for some stocks. When individual metrics are missing, their weight is redistributed within the category. When the entire category is unavailable, its weight redistributes to the other categories.

5. **EPS revisions reach back only 90 days:** the Revisions category *does* include a forward-EPS-consensus-change metric — FY1 EPS Revision (3-month), its heaviest at {rev_w.get('fy1_revision_3m', 0)}%, live since 2026-09-10. What is still missing is **depth**: yfinance's estimate history covers about 90 days, so the screener can see a recent revision but not whether the trend has persisted over the six-month window in which Chan, Jegadeesh & Lakonishok (1996) measured the effect most strongly. That longer window would require a paid consensus feed (FactSet, Refinitiv I/B/E/S). *Corrected 2026-09-25: this limitation previously said the metric was not possible at all, which stopped being true on 2026-09-10.*

6. **Rebalance frequency:** The model portfolio is a snapshot. It should be re-run at the configured frequency (monthly or quarterly) to stay current.

7. **No covariance / correlation portfolio risk model:** None of the default weighting schemes uses cross-holding correlation. `equal` uses no risk input at all, `score` uses the composite only, and `inverse_vol` uses single-name volatility; none of them accounts for how the holdings move together. Portfolio-level risk may be understated for correlated holdings. An ex-ante covariance-aware risk report (Ledoit-Wolf-shrunk daily-return covariance: portfolio vol, diversification ratio, top pairwise correlations) is now printed in the run summary for transparency, and an experimental minimum-variance weighting exists, but correlation is not neutralized in the default portfolio.

8. **Composite is cardinal; percentile is separate:** The `Composite` column is the cardinal weighted-average of the 0-100 category scores (the ranking key, preserving magnitude/conviction). The `Composite_Pct` column is the universe percentile ("better than X% of stocks"). Do not read the cardinal Composite as a percentile.

9. **Self-improving engine is governed & human-approval-only by default:** The engine can propose factor-weight changes from live IC, but auto-apply is OFF by default (`improvement.allow_auto_apply: false`) and, even when enabled, requires statistical significance (IC information ratio), the correct optimization horizon, an anti-drift cap, and FDR control on candidate-metric activation. All changes are logged with full provenance.

10. **Not investment advice:** This is a screening tool, not a recommendation engine. The output is a ranked list to narrow your research — not a list of stocks to blindly buy.

---

## Quick Start

```bash
# Run the screener on the full S&P 500
py run_screener.py --refresh

# Run on specific tickers only
py run_screener.py --refresh --tickers AAPL,MSFT,GOOGL,AMZN,META

# Use cached data (no new downloads)
py run_screener.py

# Skip portfolio construction (scoring only)
py run_screener.py --no-portfolio

# Output: factor_output.xlsx (up to 7 sheets, incl. a ReadMe/Disclaimers sheet)

# Run factor-exposure diagnostics on the latest portfolio
py factor_exposure.py --start 2024-01-01 --end 2025-12-31
```

---

## Summary

The screener answers one question: **"Which S&P 500 stocks look best when measured across {', '.join(factor_labels[f] for f in active_factors).lower()} — all at once?"**

It does this by:
1. Pulling financial data for ~500 stocks from Yahoo Finance
2. Computing up to {n_total} financial metrics across {n_factors} categories ({n_generic} generic + {n_bank_only} bank-specific, depending on company type)
3. Ranking each metric within its sector (so comparisons are fair)
4. Weighting and combining into a single 0-100 composite score (with bank-specific weights for financial companies and conditional Piotroski weighting)
5. Flagging potential value traps (cheap and weak on 2 of 3) and growth traps (high growth and weak quality or revisions)
6. Applying a liquidity filter to ensure tradeability
7. Reporting how stable that ranking is when the weights are nudged

The result is a disciplined, repeatable, multi-dimensional ranking that avoids the tunnel vision of looking at any single metric in isolation.
"""
    return md


def generate_screener_overview(cfg: dict) -> None:
    """Write SCREENER_OVERVIEW.md from the live config.

    The file is **generated**, not hand-maintained: every full
    ``run_screener.py`` run overwrites it (step 11), so a correction made to
    the markdown survives only until the next 02:00 data run. Edit
    :func:`build_screener_overview` instead. ``CLAUDE.md`` rule 10 lists this
    file for the same reason.
    """
    overview_path = ROOT / "SCREENER_OVERVIEW.md"
    overview_path.write_text(build_screener_overview(cfg), encoding="utf-8")


# Metric label lookups for the overview generator
_METRIC_LABELS = {
    "ev_ebitda": "EV/EBITDA", "fcf_yield": "FCF Yield", "earnings_yield": "Earnings Yield",
    "ev_sales": "EV/Sales", "pb_ratio": "Price-to-Book (P/B)",
    "roic": "ROIC", "gross_profit_assets": "Gross Profit / Assets",
    "debt_equity": "Debt/Equity", "piotroski_f_score": "Piotroski F-Score",
    "accruals": "Accruals", "roe": "ROE", "roa": "ROA", "equity_ratio": "Equity Ratio",
    "forward_eps_growth": "Forward EPS Growth", "peg_ratio": "PEG Ratio",
    "revenue_growth": "Revenue Growth", "sustainable_growth": "Sustainable Growth",
    "return_12_1": "12-1 Month Return", "return_6m": "6-1 Month Return",
    "volatility": "Volatility", "beta": "Beta",
    "sharpe_ratio": "Sharpe Ratio", "sortino_ratio": "Sortino Ratio",
    "max_drawdown_1y": "Max Drawdown (13M)",
    "fy1_revision_3m": "FY1 EPS Revision (3-month)",
    "analyst_surprise": "Analyst Surprise", "price_target_upside": "Price Target Upside",
    "earnings_acceleration": "Earnings Acceleration", "consecutive_beat_streak": "Beat Score",
    "short_interest_ratio": "Short Interest Ratio",
    "size_log_mcap": "Log Market Cap", "asset_growth": "Asset Growth",
    "net_debt_to_ebitda": "Net Debt / EBITDA",
    "operating_leverage": "Operating Leverage",
    "beneish_m_score": "Beneish M-Score",
    "revenue_cagr_3yr": "Revenue CAGR (3Y)",
    "jensens_alpha": "Jensen's Alpha",
}

_VAL_DESCRIPTIONS = {
    "ev_ebitda": "Enterprise value divided by earnings before interest, taxes, depreciation, and amortization. A capital-structure-neutral price tag. Lower = cheaper.",
    "fcf_yield": "Free cash flow (operating cash flow minus capital expenditures) divided by enterprise value. How much cash the business generates per dollar of total value. Higher = cheaper.",
    "earnings_yield": "LTM Net Income divided by Market Cap (inverse of P/E). Uses LTM for consistency with other flow metrics. Higher = cheaper.",
    "ev_sales": "Enterprise value divided by revenue. Useful for comparing companies with different margin profiles. Lower = cheaper.",
    "pb_ratio": "Share price divided by book value per share. THE key bank valuation metric — banks' assets are mostly financial instruments carried near fair value. Lower = cheaper.",
}

_QUAL_DESCRIPTIONS = {
    "roic": "Return on Invested Capital — NOPAT divided by invested capital (equity + debt - excess cash). Excess cash is cash beyond 2% of revenue. Tax rate: actual effective rate (clamped 0-50%) when pretax income is positive; 0% for tax-loss positions (negative pretax); 21% default when data is missing. Higher = better use of capital.",
    "gross_profit_assets": "Gross profit divided by total assets. Measures asset-light profitability (Novy-Marx quality factor).",
    "debt_equity": "Total debt divided by shareholder equity. Lower = less financial leverage and risk.",
    "net_debt_to_ebitda": "(Total Debt - Cash and short-term investments) / EBITDA, the same cash enterprise value nets off. Measures leverage relative to earnings power. Lower = less leveraged = better. Replaces Debt/Equity (negative equity from buybacks distorts D/E).",
    "piotroski_f_score": "A 0-9 checklist scoring profitability, leverage, liquidity, and efficiency trends. Higher = healthier fundamentals.",
    "accruals": "(Net Income - Operating Cash Flow) / Total Assets. Lower (more negative) = higher earnings quality (Sloan 1996).",
    "operating_leverage": "Degree of Operating Leverage (%Δ EBIT / %Δ Revenue), one year. Recorded but given no weight since 2026-10-09 - see 'Why not operating leverage?'. Banks skip this metric.",
    "beneish_m_score": "8-variable earnings manipulation detection model (Beneish 1999). More negative = lower manipulation risk. Requires ≥5 of 8 variables. Not computed for Financials stocks (Beneish's sample excluded financial firms).",
    "roe": "Return on equity — the key bank profitability metric. Higher = better.",
    "roa": "Return on assets — key bank efficiency metric. Higher = better.",
    "equity_ratio": "Total equity divided by total assets. Solvency measure — higher = more capital = safer.",
}

_GROWTH_DESCRIPTIONS = {
    "forward_eps_growth": "Expected EPS over the next 12 months (current- and next-fiscal-year consensus, weighted by the months left in the current year - MSCI's construction) versus the last four reported quarters, on the same basis. Denominator floored at $1.00. Clamped to [-75%, +150%]. Higher = faster expected growth. Since 2026-10-09; before, it compared a fiscal year 13-24 months out with GAAP trailing EPS.",
    "revenue_growth": "Year-over-year revenue increase from financial statements. Higher = growing top line.",
    "revenue_cagr_3yr": "3-year compound annual revenue growth rate from annual filings. Smooths lumpy single-year revenue growth.",
    "peg_ratio": "P/E ratio divided by forward EPS growth. A PEG of 1.0 means fairly valued relative to growth. Lower = better.",
    "sustainable_growth": "ROE × retention rate (1 - dividend payout ratio). Higher = more internally funded growth capacity.",
}

_MOM_DESCRIPTIONS = {
    "return_12_1": "Total price return from 12 months ago to 1 month ago. Skips the most recent month to avoid short-term reversal noise.",
    "return_6m": "Total price return from 6 months ago to 1 month ago. Also skips the most recent month.",
    "jensens_alpha": "Risk-adjusted excess return above CAPM prediction. Measures outperformance unexplained by market beta. Uses full 12-month return (no skip-month).",
}

_RISK_DESCRIPTIONS = {
    "volatility": "Annualized standard deviation of daily returns over about 13 months of trading (the same price history the momentum signals use). Lower = smoother ride.",
    "beta": "Covariance of stock returns with S&P 500 returns divided by variance of market returns. Requires ≥80% date overlap with market. Lower = less market-driven risk.",
    "sharpe_ratio": "(12-month return - risk-free rate) / volatility. Risk-adjusted return per unit of total risk. Higher = more efficient risk-taking.",
    "sortino_ratio": "(12-month return - risk-free rate) / downside deviation. Like Sharpe but only penalizes downside volatility. Higher = better downside-adjusted return.",
    "max_drawdown_1y": "Largest peak-to-trough fall in the closing price over about 13 months. Less negative = smaller worst-case loss.",
}

_REV_DESCRIPTIONS = {
    "fy1_revision_3m": "Change in the consensus current-fiscal-year EPS estimate over the last 90 days, divided by the share price — so it reads in **basis points of price** and is comparable across a $20 stock and a $400 one. Positive = analysts have raised their forecast. This is the category's only true *revision* metric: it measures analysts changing their minds, not companies beating a past estimate.",
    "analyst_surprise": "Median of (Actual - Estimated EPS) / max(|Estimated|, $0.10) over last 4 quarters. Positive = beat expectations.",
    "price_target_upside": "(Mean Analyst Price Target - Current Price) / Current Price. Clamped to [-50%, +100%]. Higher = more analyst optimism.",
    "earnings_acceleration": "Difference between most recent quarter's surprise % and prior quarter's surprise %. Positive = accelerating beats, negative = decelerating. Continuous; extreme values are flagged in the data-quality log but scored as fetched.",
    "consecutive_beat_streak": "Recency-weighted beat score: each of the last 4 quarters' beats weighted by recency (Q1=1, Q2=2, Q3=3, Q4=4). Range 0-10. A stock beating all 4 quarters scores 10; beating only the most recent scores 4. With fewer quarters on record the score is the beating quarters' share of the weight of those that are, times 10, so missing data does not cap it; a history whose newest quarter ended over 200 days ago is not scored.",
    "short_interest_ratio": "Days to cover (short interest shares / average daily volume). Lower = less bearish sentiment from short sellers. Contrarian signal.",
}

_SIZE_DESCRIPTIONS = {
    "size_log_mcap": "Negative natural log of market capitalization: -log(marketCap). Smaller companies get higher values. Scored by sector percentile rank, which the log does not affect - see the note below.",
}

_INV_DESCRIPTIONS = {
    "asset_growth": "Year-over-year change in total assets. Lower = better (conservative investment, Fama-French CMA).",
}


# ---------------------------------------------------------------------------
def _metric_missing_pct(df, col: str, is_bank) -> float:
    """% of *applicable* rows where ``col`` is missing.

    Bank-only and non-bank-only metrics (``_BANK_ONLY_METRICS`` /
    ``_NONBANK_ONLY_METRICS`` in factor_engine.py) are structurally NaN outside
    their population - about 58 of 502 S&P 500 stocks are banks - so scoring
    "missing" against the whole universe manufactures a permanent false
    positive: a bank-only metric absent from every non-bank reads as ~88%
    missing forever, which is exactly what it should read as zero-signal
    noise did in `validation/data_quality_log.csv` from launch through
    2026-09-01, flagged "High severity" on every run. The coverage filter a
    few hundred lines below this already scopes the same way; this mirrors it.
    """
    from factor_engine import _BANK_ONLY_METRICS, _NONBANK_ONLY_METRICS

    if col not in df.columns:
        return 100.0
    if col in _BANK_ONLY_METRICS:
        applies = is_bank
    elif col in _NONBANK_ONLY_METRICS:
        applies = ~is_bank
    else:
        applies = pd.Series(True, index=df.index)
    denom = int(applies.sum())
    if denom == 0:
        return 0.0
    return float(df.loc[applies, col].isna().sum()) / denom * 100


def _init_pipeline_logger():
    """Initialize structured pipeline logger (coexists with existing print() UX)."""
    logger = logging.getLogger("screener.pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def should_write_score_cache(args) -> bool:
    """A ``--tickers`` run must not write the scored cache.

    The cache is keyed by config and date, not by universe, and a full run the same day reads
    the newest file for its key as a HOT cache. On 2026-10-09 a 32-ticker trial run wrote
    ``factor_scores_<hash>_20261009.parquet``; the evening's full run would have loaded it and
    published a 32-stock dashboard. Reading already skipped the cache for ``--tickers``; writing
    did not."""
    return not getattr(args, "tickers", None)


def run_factor_engine(cfg, args, ctx=None):
    """Run the complete factor scoring pipeline. Returns (scored_df, stats_dict)."""
    pipeline_log = _init_pipeline_logger()

    from factor_engine import (
        get_sp500_tickers, fetch_single_ticker, fetch_all_tickers,
        fetch_market_returns, fetch_risk_free_rate, compute_metrics,
        _generate_sample_data, _find_latest_cache,
        cache_age_days, cache_is_usable, factor_scores_cache_max_age_days,
        apply_universe_filters,
        flag_metric_outliers, compute_sector_percentiles,
        apply_percentile_transform,
        compute_category_scores, adjust_momentum_weight, compute_composite,
        apply_value_trap_flags, apply_growth_trap_flags, rank_stocks,
        compute_factor_correlation, run_weight_sensitivity,
        write_scores_parquet, METRIC_COLS, METRIC_DIR,
        _BANK_ONLY_METRICS, _NONBANK_ONLY_METRICS,
        set_stmt_val_strict, get_stmt_val_misses, clear_stmt_val_misses,
        compute_factor_contributions,
    )

    # ---- Weight validation ----
    print("\n=== METRIC WEIGHT VALIDATION ===")
    mw = cfg.get("metric_weights", {})
    bank_mw = cfg.get("bank_metric_weights", {})
    for cat_name in ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]:
        cat_ws = mw.get(cat_name, {})
        generic_sum = sum(cat_ws.values())
        # Build non-zero weight string: "25+45+20+10"
        nonzero = [str(int(v)) for v in cat_ws.values() if v > 0]
        wt_str = "+".join(nonzero) if nonzero else "0"
        status = "OK" if generic_sum == 100 else f"FAIL ({generic_sum})"
        bank_cat_ws = bank_mw.get(cat_name, cat_ws)
        bank_sum = sum(bank_cat_ws.values())
        bank_status = f"  bank={bank_sum}" if cat_name in bank_mw else ""
        print(f"  [WEIGHT CHECK] {cat_name:12s}  {wt_str} = {generic_sum}% {status}{bank_status}")
    print()

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
        valid_tickers = set(universe_df["Ticker"].values)
        
        # Validate tickers and warn about invalid ones
        invalid_tickers = [t for t in custom if t not in valid_tickers]
        valid_custom = [t for t in custom if t in valid_tickers]
        
        if invalid_tickers:
            print(f"\n  WARNING: {len(invalid_tickers)} ticker(s) not in S&P 500 universe, skipping:")
            for t in invalid_tickers:
                print(f"    - {t}")
            print()
        
        if not valid_custom:
            print(f"\n  ERROR: No valid tickers specified. All provided tickers are invalid.")
            sys.exit(1)
        
        universe_df = universe_df[universe_df["Ticker"].isin(valid_custom)].copy()
        print(f"  Custom ticker subset: {list(universe_df['Ticker'])}")

    tickers = universe_df["Ticker"].tolist()
    universe_size = len(tickers)
    print(f"  Universe: {universe_size} tickers")
    ticker_meta = universe_df.set_index("Ticker")[["Company", "Sector"]].to_dict("index")

    # ---- Check cache freshness (config-aware) ----
    # Bound by the PRICE tier, not the fundamental tier: factor_scores is the
    # fully scored dataset and carries momentum, volatility and analyst
    # price-target metrics that go stale with every market close. See
    # factor_scores_cache_max_age_days() and tests/test_cache_freshness.py.
    fresh_days = factor_scores_cache_max_age_days(cfg.get("caching", {}))
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
        age_days = cache_age_days(cached_dt)
        if cache_is_usable(cached_dt, fresh_days) and not args.tickers:
            use_cache = True
            stats["cache_status"] = "HOT"
            stats["tickers_cache"] = universe_size
        else:
            print(
                f"  Cache {cached_path.name} is {age_days} day(s) old "
                f"(max {fresh_days}) - refetching"
            )
            stats["cache_status"] = "WARM"

    if use_cache:
        print(f"  [CACHE HIT] Loading scores from {cached_path.name}")
        df = pd.read_parquet(str(cached_path))
        stats["scored"] = len(df)
        # Save to run directory so dashboard generation can find it
        if ctx is not None:
            ctx.save_artifact("05_final_scored", df)
        return df, stats

    # ---- Enable strict mode for _stmt_val() lookups ----
    stmt_strict = cfg.get("data_quality", {}).get("stmt_val_strict", False)
    if stmt_strict:
        clear_stmt_val_misses()
        set_stmt_val_strict(True)

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
        # Refuse by default. On 2026-08-06 this path ran with no network,
        # fabricated all 503 tickers, and produced a normal-looking 2.6 MB
        # dashboard payload; only a failed push kept invented stock scores off
        # the public site. A screener that silently emits fiction when its data
        # source is down is a credibility bug, not a robustness feature - the
        # output is indistinguishable from a real run to anyone who does not
        # read validation/data_quality_log.csv.
        #
        # scripts/data-run.ps1 gates on this downstream, but the gate only
        # protects the scheduled loop. Anyone running the screener directly
        # got fiction. This refuses at source.
        if not getattr(args, "allow_synthetic", False):
            print(f"  Network unavailable ({type(e).__name__}): {e}")
            print("  REFUSING to run: synthetic data would be published as if real.")
            print("  Fix the connection and re-run, or pass --allow-synthetic")
            print("  if you genuinely want fabricated values for pipeline testing.")
            raise SystemExit(2)

        print(f"  Network unavailable ({type(e).__name__})")
        print("  --allow-synthetic given: generating sector-realistic SAMPLE data.")
        print("  *** THIS OUTPUT IS FABRICATED. DO NOT PUBLISH IT. ***")
        USE_SAMPLE = True

    skipped_tickers = []

    # Fetch RF rate regardless of sample/live path; fall back to default if unavailable
    try:
        risk_free_rate = fetch_risk_free_rate()
        print(f"  [RUN] Risk-free rate (^IRX): {risk_free_rate*100:.2f}%")
    except Exception:
        risk_free_rate = 0.045
        print("  [RUN] Risk-free rate unavailable — using default 4.50%")

    if USE_SAMPLE:
        df = _generate_sample_data(universe_df, risk_free_rate=risk_free_rate)
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
        stats["market_series"] = market_returns.attrs.get("source")
        print(f"  {len(market_returns)} daily observations ({stats['market_series'] or 'unavailable'})")

        fetch_cfg = cfg.get("fetch", {})
        print(f"\nFetching data for {universe_size} tickers...")
        raw = fetch_all_tickers(
            tickers,
            batch_size=fetch_cfg.get("batch_size", 30),
            max_workers=fetch_cfg.get("max_workers", 3),
            inter_batch_delay=fetch_cfg.get("inter_batch_delay", 3.0),
        )

        # ---- Retry pass: re-fetch failed tickers with conservative settings ----
        if fetch_cfg.get("retry_failed", True):
            failed = [r["Ticker"] for r in raw
                      if "_error" in r and not r.get("_non_retryable", False)]
            if failed and len(failed) <= universe_size * 0.5:
                cooldown = fetch_cfg.get("retry_cooldown", 30)
                print(f"\n  Retry pass: {len(failed)} tickers failed — "
                      f"cooling down {cooldown}s then retrying...")
                time.sleep(cooldown)
                retry_raw = fetch_all_tickers(
                    failed, batch_size=10, max_workers=1,
                    inter_batch_delay=5.0,
                )
                retry_ok = {r["Ticker"]: r for r in retry_raw
                            if "_error" not in r}
                if retry_ok:
                    raw = [retry_ok.get(r["Ticker"], r) if "_error" in r else r
                           for r in raw]
                    print(f"  Retry recovered {len(retry_ok)}/{len(failed)} tickers")
                else:
                    print(f"  Retry pass: no additional tickers recovered")

        stats["tickers_api"] = len(raw)

        # GICS sector and sub-industry from the S&P 500 list, on each raw record before anything
        # reads it: scoring decides the bank metric set from the sub-industry (2026-10-09,
        # research/2026-10-09-bank-like-financials.md), and Yahoo's own sector names differ.
        _gics = universe_df.set_index("Ticker").to_dict("index") if "Ticker" in universe_df.columns else {}
        for _r in raw:
            _m = _gics.get(_r.get("Ticker")) or {}
            if _m.get("Sector"):
                _r["_gics_sector"] = _m["Sector"]
            if isinstance(_m.get("SubIndustry"), str) and _m["SubIndustry"]:
                _r["_gics_sub"] = _m["SubIndustry"]

        # Earnings variability (Quality candidate, weight 0): five years of annual ROE
        # from each company's own 10-K figures (SEC companyfacts) - Yahoo carries four years.
        # A --tickers run reads the cache but never rewrites it (it would hold only the subset).
        # ~25 cached requests for the whole universe. research/2026-10-09-operating-leverage.md
        try:
            import json as _json_ev
            import sec_fundamentals as _sf
            _roe = _sf.roe_history([_r["Ticker"] for _r in raw if _r.get("Ticker") and "_error" not in _r],
                                   refresh=should_write_score_cache(args))
            _n_ev = 0
            for _r in raw:
                _rows = _roe.get(_r.get("Ticker"))
                if _rows:
                    _r["_roe5"] = _json_ev.dumps(_rows, separators=(",", ":"))
                    _ev = _sf.earnings_variability(_rows)
                    if _ev is not None:
                        _r["_evol"] = _ev
                        _n_ev += 1
            if _roe:
                print(f"  Earnings variability: 5 years of ROE for {_n_ev} of {len(raw)} stocks (SEC XBRL)")
        except Exception as e:  # noqa: BLE001 - a candidate metric must never stop a run
            print(f"  WARNING: earnings variability unavailable: {e}")

        # Valuation against the stock's own five-year history (display only, context): earnings
        # and free-cash-flow yield at each month-end from the SEC filings cache and one batched
        # monthly price download. research/2026-10-09-valuation-vs-own-history.md
        if cfg.get("context", {}).get("enabled", True):
            try:
                import valuation_history
                _n_vh = valuation_history.attach(raw, refresh=should_write_score_cache(args))
                stats["context_valuation_history"] = _n_vh
                print(f"  Context: valuation against its own 5-year history for {_n_vh} of {len(raw)} stocks")
            except Exception as e:  # noqa: BLE001 - context must never stop a run
                print(f"  WARNING: valuation history unavailable: {e}")

        # Context layer (display only, plan/context-layer.md). Runs only now that the core data
        # is fetched: option chains and insider trades in their own paced, time-budgeted pass
        # (context_fetch.py), then each stock's sensitivity to the 10-year yield, computed here
        # because the daily returns it needs are not saved.
        if cfg.get("context", {}).get("enabled", True):
            try:
                import context_fetch
                stats["context_pass"] = context_fetch.enrich(
                    raw, budget_seconds=cfg.get("context", {}).get("budget_seconds", 900),
                    root=Path(__file__).resolve().parent)
            except Exception as e:  # noqa: BLE001 - context must never stop a run
                print(f"  WARNING: context pass unavailable: {e}")
            # Insider trades from the SEC's own Form 4 filings where an identity with a contact
            # email is configured (outside the repo - insider_activity.user_agent); each stock
            # whose SEC record is fresh replaces the Yahoo rows the pass above fetched.
            try:
                import json as _json
                from datetime import date as _date
                import insider_activity as _ia
                if _ia.user_agent():
                    _today = _date.today()
                    _cache = _ia.refresh([_r["Ticker"] for _r in raw if _r.get("Ticker") and "_error" not in _r],
                                         today=_today,
                                         budget_seconds=cfg.get("context", {}).get("sec_budget_seconds", 900))
                    _n_sec = 0
                    for _r in raw:
                        _rows = _ia.sec_rows_for(_cache, _r.get("Ticker"), _today)
                        if _rows is not None:
                            _r["_ctx_insider"] = _json.dumps(_rows, separators=(",", ":"))
                            _r["_ctx_insider_src"] = "sec"
                            _n_sec += 1
                    stats["context_sec_insider"] = _n_sec
                    print(f"  Context: insider trades from SEC Form 4 for {_n_sec} stocks; Yahoo's feed for the rest")
            except Exception as e:  # noqa: BLE001 - context must never stop a run
                print(f"  WARNING: SEC insider refresh unavailable: {e}")
            try:
                from market_context import yield_changes
                from context_signals import rate_sensitivity
                _yc = yield_changes()
                _n_rate = 0
                for _r in raw:
                    _rs = rate_sensitivity(_r.get("_daily_returns") or {}, _yc)
                    if _rs:
                        _r["_ctx_rate_beta"], _r["_ctx_rate_r2"], _r["_ctx_rate_n"] = _rs["beta"], _rs["r2"], _rs["n"]
                        _n_rate += 1
                print(f"  Context: rate sensitivity for {_n_rate} stocks")
            except Exception as e:  # noqa: BLE001 - context must never stop a run
                print(f"  WARNING: rate sensitivity unavailable: {e}")

        # H5: Save raw API responses for reproducibility / debugging.
        # Exclude _daily_returns (large nested dict) to keep artifact lean.
        if ctx is not None:
            raw_for_save = []
            for r in raw:
                row = {k: v for k, v in r.items() if k != "_daily_returns"}
                raw_for_save.append(row)
            raw_df = pd.DataFrame(raw_for_save)
            ctx.save_artifact("00_raw_fetch", raw_df)

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
        from factor_engine import BANK_DEFAULTED as _bank_defaulted
        _bank_defaulted.clear()
        df = compute_metrics(raw, market_returns, cfg, risk_free_rate=risk_free_rate)
        if _bank_defaulted:
            stats["bank_set_by_default"] = sorted(_bank_defaulted)
            print(f"  WARNING: {len(_bank_defaulted)} financials reached the bank metric set only by default: "
                  f"{sorted(_bank_defaulted)[:12]} - add their sub-industry to factor_engine's lists")

        # ---- Log _stmt_val() misses (strict mode) ----
        if stmt_strict:
            misses = get_stmt_val_misses()
            set_stmt_val_strict(False)
            if misses:
                # Aggregate by label to avoid flooding the DQ log
                from collections import Counter
                miss_counts = Counter(m["label"] for m in misses)
                for label, count in miss_counts.most_common():
                    sample = next(m for m in misses if m["label"] == label)
                    dq_log("UNIVERSE", "stmt_val_miss", "Low",
                           f"_stmt_val miss: '{label}' col={sample['col']} "
                           f"({count} occurrences, reason={sample['reason']})",
                           "Returned NaN default")
                stats["stmt_val_misses"] = len(misses)
                stats["stmt_val_unique_labels"] = len(miss_counts)
                print(f"  _stmt_val() strict: {len(misses)} misses across "
                      f"{len(miss_counts)} unique labels")

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

        # Coverage filter - the same coverage the composite's discount reads: the metrics that
        # carry weight in the table this stock is scored with (factor_engine.applicable_coverage,
        # since 2026-10-09). Until then it counted every registered metric, so a stock could be
        # dropped from the universe for missing candidates that never enter a score.
        from factor_engine import applicable_coverage
        metric_count, applicable_count = applicable_coverage(df, cfg)
        coverage_pct = cfg["data_quality"]["min_data_coverage_pct"] / 100
        df["_mc"] = metric_count
        min_needed = (applicable_count * coverage_pct).apply(lambda x: max(1, int(x)))
        low = df["_mc"] < min_needed
        for t in df.loc[low, "Ticker"]:
            dq_log(t, "missing_metric", "Medium",
                   f"Insufficient metric coverage (< {cfg['data_quality']['min_data_coverage_pct']}%)",
                   "Excluded from scoring")
        n_coverage_dropped = int(low.sum())
        skipped_tickers += df.loc[low, "Ticker"].tolist()
        df = df[~low].copy()
        pipeline_log.info("Coverage filter: %d stocks excluded (< %d%% metric coverage), %d remaining",
                          n_coverage_dropped, cfg["data_quality"]["min_data_coverage_pct"], len(df))

    stats["fetch_time"] = round(time.time() - fetch_t0, 1)
    stats["tickers_failed"] = len(skipped_tickers)
    stats["failed_list"] = skipped_tickers[:20]

    # ---- Apply universe filters (min_market_cap) ----
    pre_filter = len(df)
    df = apply_universe_filters(df, cfg)
    n_filtered = pre_filter - len(df)
    if n_filtered > 0:
        pipeline_log.info("Universe filter: %d stocks excluded (min_market_cap), %d remaining",
                          n_filtered, len(df))
    if n_filtered > 0:
        for t in set(df["Ticker"].tolist()) ^ set(df["Ticker"].tolist()):
            dq_log(t, "universe_filter", "Medium",
                   "Below minimum market cap", "Excluded from scoring")

    # ---- Save raw metrics artifact (data lineage) ----
    if ctx is not None:
        ctx.save_artifact("01_raw_metrics", df)
        ctx.save_universe(df["Ticker"].tolist(), skipped_tickers)

    # ---- Data quality checks (§4.7) ----
    _run_data_quality_checks(df)

    # ---- Financial sector advisory ----
    _check_financial_sector_metrics(df)

    # ---- Per-sector coverage reporting ----
    sector_cov = _report_sector_coverage(df, [c for c in METRIC_COLS if c in df.columns])
    stats["sector_coverage"] = sector_cov

    # ---- Warn if > 20% failed ----
    if universe_size > 0 and len(skipped_tickers) / universe_size > 0.20:
        print(f"\n  *** WARNING: {len(skipped_tickers)}/{universe_size} tickers failed "
              f"({len(skipped_tickers)/universe_size*100:.0f}%). Results may be unreliable. ***")

    # ---- Revisions auto-disable ----
    rev_m = ["analyst_surprise", "price_target_upside", "earnings_acceleration", "consecutive_beat_streak"]
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
        pipeline_log.warning("Revisions auto-disabled: %.1f%% coverage < 30%%", rev_pct)

    stats["rev_disabled"] = rev_disabled

    # ---- Investment auto-disable (mirror revisions logic) ----
    inv_m = ["asset_growth"]
    inv_avail = sum(df[c].notna().sum() for c in inv_m if c in df.columns)
    inv_total = len(df) * len(inv_m)
    inv_pct = inv_avail / inv_total * 100 if inv_total else 0
    inv_disabled = False

    if inv_pct < 30:
        print(f"\n!! Investment coverage {inv_pct:.1f}% < 30%; auto-disabling")
        cfg["factor_weights"] = copy.deepcopy(cfg["factor_weights"])
        old_w = cfg["factor_weights"].get("investment", 0)
        cfg["factor_weights"]["investment"] = 0
        others = [k for k in cfg["factor_weights"] if k != "investment"]
        s = sum(cfg["factor_weights"][k] for k in others)
        if s > 0:
            for k in others:
                cfg["factor_weights"][k] += old_w * cfg["factor_weights"][k] / s
            for k in cfg["factor_weights"]:
                cfg["factor_weights"][k] = round(cfg["factor_weights"][k], 2)
        inv_disabled = True
        pipeline_log.warning("Investment auto-disabled: %.1f%% coverage < 30%%", inv_pct)

    stats["inv_disabled"] = inv_disabled

    # ---- Auto-reduce high-NaN metrics ----
    _auto_reduce_high_nan_metrics(df, cfg, pipeline_log=pipeline_log)

    # ---- Scoring pipeline ----
    # Outliers are reported, never clipped. compute_sector_percentiles() below
    # is Series.rank(pct=True), which is invariant under any monotone transform
    # of its input, so clipping the tails first could not change an ordering -
    # it could only manufacture ties and corrupt the value published as `raw`.
    # Removed 2026-09-01; see METHODOLOGY_CHANGELOG.md and flag_metric_outliers().
    _tails = cfg.get("data_quality", {}).get(
        "outlier_report_percentiles",
        cfg.get("data_quality", {}).get("winsorize_percentiles", [1, 99]),
    )
    _lo_frac = _tails[0] / 100.0
    _hi_frac = (100 - _tails[1]) / 100.0
    score_t0 = time.time()
    print(f"Flagging outliers beyond the {_tails[0]}th / {_tails[1]}th percentiles "
          f"(values are reported, not clipped)...")
    _outliers = flag_metric_outliers(df, _lo_frac, _hi_frac)

    for col, info in _outliers.items():
        dq_log("UNIVERSE", "outlier_flagged", "Low",
               f"{col}: {info['n_low']} at/below {info['lo_cut']:.4g}, "
               f"{info['n_high']} at/above {info['hi_cut']:.4g} "
               f"({_tails[0]}th/{_tails[1]}th pctile of {info['n_valid']} values)",
               "Reported only - the value is scored and published as fetched")

    if ctx is not None:
        ctx.save_artifact("02_outliers_flagged", df)
    pipeline_log.info("Outlier flagging complete: %d stocks, %d metrics with tail values",
                      len(df), len(_outliers))

    print("Computing sector-relative percentile ranks...")
    df = compute_sector_percentiles(df)

    if ctx is not None:
        ctx.save_artifact("03_percentiles", df)
    pipeline_log.info("Sector-relative percentile ranking complete: %d stocks", len(df))

    # ---- Optional percentile transform (default: disabled) ----
    df = apply_percentile_transform(df, cfg)
    if cfg.get("percentile_transform", {}).get("enabled", False):
        pipeline_log.info("Percentile transform applied: method=%s",
                          cfg["percentile_transform"].get("method", "logistic"))

    print("Computing within-category scores...")
    df = compute_category_scores(df, cfg)

    if ctx is not None:
        ctx.save_artifact("04_category_scores", df)
    pipeline_log.info("Category scores complete: %d stocks", len(df))

    print("Adjusting momentum weight for vol regime...")
    cfg = adjust_momentum_weight(df, cfg, str(ROOT))

    # adjust_momentum_weight returns a deep copy, so from here on `cfg` is a
    # different object from the caller's. The revisions/investment auto-disables
    # above mutate the shared dict and therefore reach main() on their own; this
    # one cannot, and for 183 days it did not - the published
    # effective_weights.json recorded the *configured* weights while the
    # composite was built from the regime-adjusted ones. The dashboard prints
    # that arithmetic to the user ("Score x 13% = 9.76 pts"), so the sum shown
    # on the public site did not add up. Hand the real weights back explicitly.
    stats["_effective_factor_weights"] = dict(cfg.get("factor_weights", {}))

    print("Computing composite scores...")
    df = compute_composite(df, cfg)
    pipeline_log.info("Composite scores complete: %d stocks", len(df))

    print("Computing factor contributions...")
    df = compute_factor_contributions(df, cfg)

    print("Applying value trap flags...")
    df = apply_value_trap_flags(df, cfg)
    vt_count = int(df["Value_Trap_Flag"].sum()) if "Value_Trap_Flag" in df.columns else 0
    pipeline_log.info("Value trap flags: %d stocks flagged", vt_count)

    print("Applying growth trap flags...")
    df = apply_growth_trap_flags(df, cfg)
    gt_count = int(df["Growth_Trap_Flag"].sum()) if "Growth_Trap_Flag" in df.columns else 0
    pipeline_log.info("Growth trap flags: %d stocks flagged", gt_count)

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
        ("Earnings Acceleration", "earnings_acceleration"),
        ("Beat Score", "consecutive_beat_streak"),
        ("Log Market Cap (size)", "size_log_mcap"),
        ("Asset Growth", "asset_growth"),
    ]
    # Recomputed here rather than reusing the `is_bank` from the coverage
    # filter above: `df` has been through several row-filtering and
    # reassignment steps since then, and a stale mask risks index misalignment.
    _is_bank_now = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False)
    stats["missing_pct"] = {}
    for lbl, col in labels:
        pct = _metric_missing_pct(df, col, _is_bank_now)
        stats["missing_pct"][lbl] = round(pct, 1)
        pipeline_log.debug("Metric coverage: %s = %.1f%% missing", lbl, pct)

    # ---- Per-category score coverage ----
    for cat in ["valuation", "quality", "growth", "momentum", "risk", "revisions",
                "size", "investment"]:
        col = f"{cat}_score"
        if col in df.columns:
            pop_pct = round((1 - df[col].isna().sum() / len(df)) * 100, 1)
            pipeline_log.info("Category coverage: %s_score = %.1f%% populated", cat, pop_pct)

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

    # ---- Stale financial data summary ----
    if "_stale_data" in df.columns or "_stmt_age_days" in df.columns:
        n_stale = df.get("_stale_data", pd.Series(dtype=bool)).fillna(False).sum()
        stale_threshold = cfg.get("data_quality", {}).get("stale_data_threshold_days", 120)
        if n_stale > 0:
            print(f"  WARNING: {int(n_stale)} tickers have financial data "
                  f"> {stale_threshold} days old")
            if "_stmt_age_days" in df.columns:
                oldest = df["_stmt_age_days"].max()
                print(f"    Oldest filing: {int(oldest)} days ago")
        stats["n_stale_tickers"] = int(n_stale)

    # ---- ROIC IC-floor diagnostic ----
    if "_roic_ic_floored" in df.columns:
        n_floored = int(df["_roic_ic_floored"].fillna(False).sum())
        if n_floored > len(df) * 0.10:
            print(f"  NOTE: {n_floored} tickers had ROIC invested capital floored "
                  f"at 10% of total assets (check for cash-rich distortion)")
        stats["n_roic_ic_floored"] = n_floored

    # ---- Analyst Revisions coverage table ----
    _rev_metrics = [
        ("analyst_surprise", "Analyst Surprise"),
        ("price_target_upside", "Price Target Upside"),
        ("earnings_acceleration", "Earnings Acceleration"),
        ("consecutive_beat_streak", "Consecutive Beat Streak"),
        ("short_interest_ratio", "Short Interest Ratio"),
    ]
    rev_coverage = {}
    for col, label in _rev_metrics:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            rev_coverage[col] = pct
    if rev_coverage:
        print("  ANALYST REVISIONS COVERAGE:")
        for col, label in _rev_metrics:
            if col in rev_coverage:
                pct = rev_coverage[col]
                flag = " ⚠" if pct < 30 else ""
                print(f"    {label:<30s}: {pct:5.1f}%{flag}")
    stats["_rev_coverage"] = rev_coverage

    # ---- Factor correlation matrix (for transparency) ----
    corr = compute_factor_correlation(df)
    if ctx is not None and not corr.empty:
        ctx.save_artifact("06_factor_correlation", corr.reset_index())
    stats["_corr_df"] = corr

    # ---- High-correlation metric pair warnings ----
    if not corr.empty:
        high_corr_threshold = cfg.get("data_quality", {}).get("high_corr_alert_threshold", 0.70)
        cols = list(corr.columns)
        high_pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) > high_corr_threshold:
                    m1 = cols[i].replace("_pct", "")
                    m2 = cols[j].replace("_pct", "")
                    high_pairs.append((m1, m2, round(float(val), 2)))
        if high_pairs:
            print(f"  HIGH-CORRELATION METRIC PAIRS (|r| > {high_corr_threshold}):")
            for m1, m2, r in sorted(high_pairs, key=lambda x: -abs(x[2])):
                print(f"    {m1} <-> {m2:35s} r={r:+.2f}")
            print("    (See FactorCorrelation sheet for full matrix)")
        stats["high_corr_pairs"] = high_pairs

    # ---- Weight sensitivity analysis ----
    print("Running weight sensitivity analysis...")
    sens_df = run_weight_sensitivity(df, cfg, perturbation_pct=5.0, top_n=20)
    if not sens_df.empty:
        avg_jaccard = sens_df["jaccard_similarity"].mean()
        min_jaccard = sens_df["jaccard_similarity"].min()
        most_sensitive = sens_df.loc[sens_df["jaccard_similarity"].idxmin(), "category"] if len(sens_df) > 0 else "N/A"
        print(f"  Avg Jaccard similarity: {avg_jaccard:.3f} (1.0 = perfectly stable)")
        print(f"  Most sensitive category: {most_sensitive} (Jaccard={min_jaccard:.3f})")
        if ctx is not None:
            ctx.save_artifact("07_weight_sensitivity", sens_df)
    stats["_sens_df"] = sens_df

    # ---- EPS mismatch summary ----
    if "_eps_basis_mismatch" in df.columns:
        n_mismatch = df["_eps_basis_mismatch"].sum()
        if n_mismatch > 0:
            print(f"  EPS basis mismatch (GAAP/normalized): {n_mismatch} tickers flagged")
            for _, row in df[df["_eps_basis_mismatch"] == True].head(5).iterrows():
                print(f"    {row['Ticker']}: fwd/trail EPS ratio = {row.get('_eps_ratio', '?')}")

    # ---- LTM partial-annualization summary ----
    if "_ltm_annualized" in df.columns:
        n_ltm = df["_ltm_annualized"].sum()
        if n_ltm > 0:
            print(f"  LTM partial annualization (3-of-4 quarters): {n_ltm} tickers flagged")

    # ---- Write Parquet cache (config-aware) ----
    if should_write_score_cache(args):
        print("Writing cache Parquet...")
        try:
            write_scores_parquet(df, config_hash=cfg_hash)
        except Exception as e:
            print(f"  WARNING: Failed to write Parquet cache: {e}")
    else:
        print("Not writing the score cache: a --tickers run scores a subset, not the universe.")

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

    # Rejected price series (see factor_engine.check_price_series_integrity).
    # Reported separately from "missing metric" because the cause is
    # different and actionable: the upstream series mixed two price scales,
    # so 23% of composite weight (momentum 13 + risk 10) was withheld rather
    # than computed wrong.
    if "_price_series_rejected" in df.columns:
        rej = df["_price_series_rejected"]
        rej_mask = rej.notna() & (rej.astype(str).str.len() > 0)
        n_rej = int(rej_mask.sum())
        if n_rej:
            print(f"\n  !! {n_rej} ticker(s) had their price history rejected; "
                  f"momentum and risk withheld:")
            for idx in df.index[rej_mask]:
                tkr = df.at[idx, "Ticker"]
                print(f"       {tkr}: {rej.at[idx]}")
                dq_log(tkr, "price_series_rejected", "High",
                       str(rej.at[idx]),
                       "Momentum and risk metrics withheld; "
                       "remaining category weights renormalized")

    # Missing critical metrics (vectorized)
    critical = ["ev_ebitda", "roic", "return_12_1"]
    for col in critical:
        if col in df.columns:
            miss_mask = df[col].isna()
            for idx in df.index[miss_mask]:
                dq_log(df.at[idx, "Ticker"], "missing_metric", "Medium",
                       f"{col} is missing/NaN",
                       "Left as NaN; weight redistributed to available metrics")


def _check_financial_sector_metrics(df: pd.DataFrame):
    """Print an advisory note about financial sector scoring.

    Bank-like financials use P/B, ROE, ROA, Equity Ratio instead of
    EV/EBITDA, ROIC, Gross Profit/Assets, Debt/Equity.
    Non-bank financials (V, MA, etc.) use standard generic metrics.
    """
    if "Sector" not in df.columns:
        return
    fin_mask = df["Sector"].str.contains("Financial", case=False, na=False)
    n_fin = fin_mask.sum()
    if n_fin > 0:
        bank_mask = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False)
        n_bank = (fin_mask & bank_mask).sum()
        n_nonbank_fin = n_fin - n_bank
        print(f"\n  Financial sector: {n_fin} stocks total")
        print(f"    Bank-like (using P/B, ROE, ROA, Equity Ratio): {n_bank}")
        print(f"    Non-bank (using standard metrics): {n_nonbank_fin}")


def _report_sector_coverage(df: pd.DataFrame, metric_cols: list):
    """Print per-sector metric coverage table and log low-coverage sectors.

    For each GICS sector, reports the average % of metrics that have data.
    Sectors with <70% average coverage are flagged in the DQ log.
    """
    if "Sector" not in df.columns or df.empty:
        return {}

    present = [c for c in metric_cols if c in df.columns]
    if not present:
        return {}

    print("\n  PER-SECTOR METRIC COVERAGE:")
    print(f"  {'Sector':<30} {'Stocks':>6} {'Avg Coverage':>13} {'Worst Metric':>25}")
    print(f"  {'-'*30} {'-'*6} {'-'*13} {'-'*25}")

    sector_stats = {}
    for sector, grp in sorted(df.groupby("Sector"), key=lambda x: x[0]):
        n = len(grp)
        # Compute per-metric coverage for this sector
        metric_cov = {}
        for m in present:
            pct = grp[m].notna().sum() / n * 100 if n > 0 else 0
            metric_cov[m] = round(pct, 1)
        avg_cov = sum(metric_cov.values()) / len(metric_cov) if metric_cov else 0
        worst_m = min(metric_cov, key=metric_cov.get) if metric_cov else "N/A"
        worst_pct = metric_cov.get(worst_m, 0)

        sector_stats[sector] = {
            "n_stocks": n, "avg_coverage": round(avg_cov, 1),
            "metric_coverage": metric_cov,
            "worst_metric": worst_m, "worst_pct": worst_pct,
        }

        flag = " !" if avg_cov < 70 else ""
        print(f"  {sector:<30} {n:>6} {avg_cov:>12.1f}% "
              f"{worst_m} ({worst_pct:.0f}%){flag}")

        if avg_cov < 70:
            dq_log(sector, "sector_low_coverage", "Medium",
                   f"Sector avg metric coverage {avg_cov:.1f}% < 70% "
                   f"(worst: {worst_m} at {worst_pct:.0f}%)",
                   "Flagged — scores for this sector may be less reliable")

    return sector_stats


def _auto_reduce_high_nan_metrics(df: pd.DataFrame, cfg: dict, pipeline_log=None):
    """Auto-zero weight for metrics that exceed the NaN threshold.

    If a metric is >70% missing (configurable), its weight is set to 0 and
    redistributed proportionally within its category. This prevents sparse
    metrics from contributing noise to scores.
    """
    threshold = cfg.get("data_quality", {}).get("auto_reduce_nan_threshold_pct", 70)
    if threshold <= 0 or threshold > 100:
        return

    cat_metrics = {
        "valuation": ["ev_ebitda", "fcf_yield", "earnings_yield", "ev_sales", "pb_ratio"],
        "quality":   ["roic", "gross_profit_assets", "debt_equity", "piotroski_f_score", "accruals",
                      "roe", "roa", "equity_ratio"],
        "growth":    ["forward_eps_growth", "peg_ratio", "revenue_growth", "sustainable_growth"],
        "momentum":  ["return_12_1", "return_6m"],
        "risk":      ["volatility", "beta"],
        "revisions": ["analyst_surprise", "price_target_upside", "earnings_acceleration", "consecutive_beat_streak"],
        "size":      ["size_log_mcap"],
        "investment": ["asset_growth"],
    }

    mw = cfg.get("metric_weights", {})
    n = len(df)
    if n == 0:
        return

    reduced = []
    for cat, metrics in cat_metrics.items():
        cat_weights = mw.get(cat, {})
        zeroed = []
        for m in metrics:
            if m not in df.columns:
                continue
            nan_pct = df[m].isna().sum() / n * 100
            if nan_pct > threshold and cat_weights.get(m, 0) > 0:
                zeroed.append((m, cat_weights[m], nan_pct))
                cat_weights[m] = 0

        if zeroed:
            # Redistribute zeroed weight proportionally to remaining metrics
            total_zeroed = sum(w for _, w, _ in zeroed)
            remaining = {m: cat_weights.get(m, 0) for m in metrics if cat_weights.get(m, 0) > 0}
            remaining_sum = sum(remaining.values())
            if remaining_sum > 0:
                for m in remaining:
                    cat_weights[m] += total_zeroed * remaining[m] / remaining_sum
                    cat_weights[m] = round(cat_weights[m], 2)
            for m, old_w, pct in zeroed:
                reduced.append(f"{m} ({pct:.0f}% NaN, was {old_w}%)")
                dq_log("UNIVERSE", "auto_reduce_metric", "Medium",
                       f"{m} is {pct:.0f}% missing (>{threshold}%), weight zeroed",
                       f"Weight {old_w} redistributed within {cat}")
                if pipeline_log:
                    pipeline_log.warning("Auto-reduced metric: %s (%.0f%% NaN, was %d%%), redistributed within %s",
                                         m, pct, old_w, cat)

    if reduced:
        print(f"\n  Auto-reduced metrics (>{threshold}% NaN): {', '.join(reduced)}")


# ---------------------------------------------------------------------------
# Portfolio construction integration
# ---------------------------------------------------------------------------
def _fetch_price_returns(tickers: list, days: int = 252) -> pd.DataFrame:
    """Fetch daily simple returns for a list of tickers (used by Markowitz mode).

    Returns a DataFrame of daily returns (dates × tickers). Columns that fail
    to download are silently dropped. Returns an empty DataFrame on failure.
    """
    try:
        import yfinance as yf
        raw = yf.download(
            tickers,
            period=f"{days + 10}d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()
        # Handle multi/single ticker output differences
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
        returns = prices.pct_change().dropna(how="all")
        return returns
    except Exception as e:
        warnings.warn(f"Markowitz price fetch failed: {e}")
        return pd.DataFrame()


def run_portfolio_construction(df, cfg):
    """Run portfolio construction. Returns (portfolio_df, stats_dict)."""
    from portfolio_constructor import (
        construct_portfolio, compute_portfolio_stats,
    )

    stats = {"construction_time": 0.0}
    port_t0 = time.time()

    # If Markowitz mode is requested, fetch price returns and inject into config
    pcfg = cfg.get("portfolio", {})
    if pcfg.get("weighting") == "markowitz":
        print("  [Markowitz] Fetching price history for covariance estimation...")
        tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
        price_returns = _fetch_price_returns(tickers, days=252)
        if price_returns.empty:
            warnings.warn("Markowitz: price fetch returned empty data; will fall back to greedy.")
        cfg = {**cfg, "portfolio": {**pcfg, "_price_returns_df": price_returns}}

    print("\nConstructing model portfolio...")
    port = construct_portfolio(df, cfg)
    stats_data = compute_portfolio_stats(port, cfg)

    stats["construction_time"] = round(time.time() - port_t0, 1)
    stats.update(stats_data)

    # Detect capped sectors
    max_sec = cfg.get("portfolio", {}).get("max_sector_concentration", 8)
    sec_cts = port["Sector"].value_counts()
    stats["capped_sectors"] = [s for s, c in sec_cts.items() if c >= max_sec]

    # --- Phase 13 (F11): turnover vs prior holdings (reporting only) --------
    try:
        stats["turnover"] = _compute_live_turnover(port, cfg)
    except Exception as e:
        warnings.warn(f"Turnover report failed: {type(e).__name__}: {e}")
        stats["turnover"] = None

    # --- Phase 13 (F10): ex-ante covariance-aware portfolio risk (reporting) -
    try:
        import portfolio_risk as pr
        wt_col = _portfolio_weight_col(port, cfg)
        port_tickers = port["Ticker"].tolist()
        weights = dict(zip(port_tickers, pd.to_numeric(port[wt_col], errors="coerce") / 100.0))
        # Reuse Markowitz returns if already fetched, else fetch for the top-N.
        pr_returns = cfg.get("portfolio", {}).get("_price_returns_df")
        if pr_returns is None or pr_returns.empty:
            pr_returns = _fetch_price_returns(port_tickers, days=252)
        stats["risk_report"] = pr.compute_covariance_risk(port_tickers, weights, pr_returns)
    except Exception as e:
        warnings.warn(f"Covariance risk report failed: {type(e).__name__}: {e}")
        stats["risk_report"] = {"available": False, "reason": str(e)}

    return port, stats


def _portfolio_weight_col(port, cfg) -> str:
    """Return the active weight column name for the configured scheme."""
    scheme = cfg.get("portfolio", {}).get("weighting", "equal")
    if scheme in ("risk_parity", "inverse_vol") and "InvVol_Weight_Pct" in port.columns:
        return "InvVol_Weight_Pct"
    if scheme == "score" and "Score_Weight_Pct" in port.columns:
        return "Score_Weight_Pct"
    return "Equal_Weight_Pct" if "Equal_Weight_Pct" in port.columns else port.columns[-1]


def _compute_live_turnover(port, cfg) -> dict | None:
    """Compute turnover vs the most recent PRIOR portfolio snapshot (F11)."""
    import portfolio_risk as pr
    from improvement_engine import SNAPSHOTS_DIR
    snaps = sorted(SNAPSHOTS_DIR.glob("*.parquet")) if SNAPSHOTS_DIR.exists() else []
    if not snaps:
        return {"name_turnover": None, "note": "no prior snapshot to compare"}
    # Most recent prior snapshot that has portfolio membership
    prior_tickers = []
    for snap_path in reversed(snaps):
        try:
            snap = pd.read_parquet(snap_path)
        except (OSError, ValueError):
            continue
        if "in_portfolio" in snap.columns:
            prior_tickers = snap.loc[snap["in_portfolio"] == True, "Ticker"].tolist()
            if prior_tickers:
                break
    if not prior_tickers:
        return {"name_turnover": None, "note": "no prior portfolio membership found"}
    cur_tickers = port["Ticker"].tolist()
    res = pr.compute_turnover(cur_tickers, None, prior_tickers, None)
    cost_bps = cfg.get("backtesting", {}).get("transaction_cost_bps", 10)
    res["est_roundtrip_cost_pct"] = pr.estimate_turnover_cost(res["name_turnover"], cost_bps)
    return res


# ---------------------------------------------------------------------------
# Excel writer integration
# ---------------------------------------------------------------------------
def write_excel_safe(df, port, port_stats, cfg, no_portfolio,
                     sens_df=None, corr_df=None):
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
            n_sheets = 4  # ReadMe + FactorScores + ScreenerDashboard + ModelPortfolio
            write_full_excel(df, port, port_stats, cfg,
                             sens_df=sens_df, corr_df=corr_df)
            # Count actual sheets
            if sens_df is not None and not sens_df.empty:
                n_sheets += 1
            if corr_df is not None and not corr_df.empty:
                n_sheets += 1
            n_sheets += 1  # DataValidation always written
            return str(out_path), n_sheets
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
def print_full_summary(args, cfg, fe_stats, port_stats, dq_counts,
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
        # Phase 13 (F11): turnover vs prior holdings
        _turn = port_stats.get("turnover")
        if _turn and _turn.get("name_turnover") is not None:
            print(f"  Turnover vs prior:      {_turn['name_turnover']*100:.0f}% "
                  f"(in {len(_turn.get('entered', []))}, out {len(_turn.get('exited', []))})")
            if _turn.get("est_roundtrip_cost_pct") is not None:
                print(f"  Est. rebalance cost:    {_turn['est_roundtrip_cost_pct']:.2f}%")
        # Phase 13 (F10): ex-ante covariance-aware portfolio risk
        _rr = port_stats.get("risk_report")
        if _rr and _rr.get("available"):
            print(f"  Ex-ante port vol (ann): {_rr['portfolio_vol_annual']*100:.1f}%  "
                  f"(wt-avg single {_rr['weighted_avg_single_vol']*100:.1f}%, "
                  f"div ratio {_rr['diversification_ratio']})")
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
    n_stale = fe_stats.get("n_stale_tickers", 0)
    stale_thresh = cfg.get("data_quality", {}).get("stale_data_threshold_days", 120)
    print(f"    Stale filings (>{stale_thresh}d): {n_stale}")
    print(f"  Log file:               validation/data_quality_log.csv")
    print("--------------------------------------------")

    # OUTPUT
    check = "OK"
    sheet_str = f"{n_sheets} sheet{'s' if n_sheets > 1 else ''}"
    print("OUTPUT:")
    print(f"  factor_output.xlsx      {check}  ({sheet_str})")
    print(f"  cache/*.parquet         {check}  ({n_cache_files} files)")
    print(f"  data_quality_log.csv    {check}")
    print(f"  sector_coverage.csv     {check}")
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

    # ---- 1.1. Apply preset if specified ----
    if args.preset:
        from presets import apply_preset, PRESETS
        if args.preset not in PRESETS:
            print(f"\n  ERROR: Unknown preset '{args.preset}'.")
            print(f"  Available presets: {', '.join(PRESETS.keys())}")
            sys.exit(1)
        cfg = apply_preset(cfg, args.preset)
        print(f"  Preset '{args.preset.upper()}' applied (factor_weights overridden)")

    # ---- 1.5. Dry-run validation (early exit if --dry-run) ----
    if args.dry_run:
        print("\n" + "="*50)
        print("  DRY-RUN MODE: Validating setup...")
        print("="*50)
        
        try:
            # Check 1: Config is valid (already done above)
            print("  [OK] Config parsed and validated")

            # Check 2: Load universe (tests network)
            print("  Loading universe...", end=" ", flush=True)
            from factor_engine import get_sp500_tickers
            universe = get_sp500_tickers(cfg)
            print(f"[OK] ({len(universe)} tickers)")

            # Check 3: Fetch a few tickers (tests yfinance)
            print("  Testing yfinance fetch (3 random tickers)...", end=" ", flush=True)
            from factor_engine import fetch_single_ticker
            import random
            test_tickers = random.sample(list(universe["Ticker"].values)[:20], min(3, len(universe)))
            test_results = []
            for t in test_tickers:
                result = fetch_single_ticker(t, max_retries=1)
                if not result.get("_error"):
                    test_results.append(True)
            if test_results:
                print(f"[OK] ({len(test_results)}/3 successful)")
            else:
                raise Exception("All test fetches failed")

            # Check 4: Output path is writable
            print("  Checking output paths...", end=" ", flush=True)
            excel_dir = ROOT / "output"
            excel_dir.mkdir(exist_ok=True)
            test_file = excel_dir / "_dry_run_test.txt"
            test_file.write_text("test")
            test_file.unlink()
            print("[OK]")

            print("\n" + "="*50)
            print("  DRY-RUN PASSED: All systems operational")
            print("="*50)
            print("\n  Run without --dry-run to execute the full screener.\n")
            sys.exit(0)

        except Exception as e:
            print(f"\n  [FAIL] DRY-RUN FAILED: {e}\n")
            sys.exit(1)

    # ---- 1.6. Show weights (early exit if --show-weights) ----
    if args.show_weights:
        print("\n" + "="*70)
        print("  EFFECTIVE FACTOR WEIGHTS")
        print("="*70)
        
        factor_weights = cfg.get("factor_weights", {})
        total_weight = sum(factor_weights.values())
        
        print(f"\nCategory Weights (Total: {total_weight}):\n")
        print(f"  {'Factor':<20} {'Weight':>10} {'%':>8}")
        print("  " + "-"*38)
        
        for factor, weight in sorted(factor_weights.items(), key=lambda x: x[1], reverse=True):
            pct = (weight / total_weight * 100) if total_weight > 0 else 0
            print(f"  {factor:<20} {weight:>10.2f} {pct:>7.1f}%")
        
        print("  " + "-"*38)
        print(f"  {'TOTAL':<20} {total_weight:>10.2f} {100.0:>7.1f}%")
        
        # Show metric-level weights
        metric_weights = cfg.get("metric_weights", {})
        if metric_weights:
            print(f"\n\nMetric-Level Weights (within each category):\n")
            for category, metrics in sorted(metric_weights.items()):
                if not isinstance(metrics, dict):
                    continue
                cat_total = sum(v for v in metrics.values() if isinstance(v, (int, float)))
                if cat_total == 0:
                    continue
                print(f"  {category.upper()}:")
                for metric, weight in sorted(metrics.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
                    if isinstance(weight, (int, float)) and weight > 0:
                        pct = (weight / cat_total * 100) if cat_total > 0 else 0
                        print(f"    {metric:<35} {weight:>6.1f}%")
                print()
        
        print("="*70)
        print("\n")
        sys.exit(0)

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
        ctx.save_artifact("08_model_portfolio", port)
        print(f"  Selected {port_stats['n_stocks']} stocks")
    else:
        print("\nSkipping portfolio construction (--no-portfolio)")

    # ---- 5. Write Excel ----
    # Extract correlation matrix and sensitivity analysis from stats
    sens_df = fe_stats.pop("_sens_df", None)
    corr_df = fe_stats.pop("_corr_df", None)
    # Weights actually used to build the composite (see run_factor_engine).
    # Absent on the warm-cache path, which returns before scoring runs.
    effective_fw = fe_stats.pop("_effective_factor_weights", None)

    print("\nWriting Excel workbook...")
    excel_path, n_sheets = write_excel_safe(
        df, port, port_stats if port_stats else {}, cfg, args.no_portfolio,
        sens_df=sens_df, corr_df=corr_df)
    print(f"  Written: {excel_path} ({n_sheets} sheets)")

    # ---- 6. Data quality log ----
    dq_path, dq_total = flush_dq_log()
    flush_sector_coverage(fe_stats.get("sector_coverage", {}))

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
    print_full_summary(args, cfg, fe_stats, port_stats, dq_counts,
                       excel_path, n_sheets, n_cache_files, total_time)

    # ---- 10. Save run metadata ----
    ctx.save_effective_weights(cfg, factor_weights=effective_fw)
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

    # ---- 11. Auto-regenerate SCREENER_OVERVIEW.md from live config ----
    try:
        generate_screener_overview(cfg)
        print("  SCREENER_OVERVIEW.md regenerated from live config")
    except Exception as e:
        print(f"  WARNING: Overview generation failed: {e}")

    # ---- 11.5. Context layer (display only; plan/context-layer.md) ----
    # The market backdrop, the ranking's track record, and a dated log of every context signal
    # so each one builds an out-of-sample record. None of it touches a score; any failure here
    # leaves the page with the previous day's context rather than stopping the run.
    # The track record, the signal log and its evaluation are records of the whole universe: a
    # --tickers run must not replace today's with 32 names (2026-10-09; data/context_log is
    # committed evidence). The market backdrop is universe-free and is still refreshed.
    _universe_run = should_write_score_cache(args)
    if cfg.get("context", {}).get("enabled", True):
        run_day = datetime.now().strftime("%Y-%m-%d")
        try:
            import market_context
            market_context.build()
            print("  Context: market backdrop refreshed (FRED)")
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: market backdrop unavailable: {e}")
    if cfg.get("context", {}).get("enabled", True) and not _universe_run:
        print("  Context: subset run - track record, signal log and evaluation left as they were")
    elif cfg.get("context", {}).get("enabled", True):
        try:
            import track_record
            live = dict(zip(df["Ticker"], df["Rank"])) if "Rank" in df.columns else None
            tr_out = track_record.build_from_disk(live=(run_day, live) if live else None)
            print(f"  Context: track record {'built' if tr_out.get('available') else 'unavailable'}")
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: track record unavailable: {e}")
        try:
            import context_signals
            context_signals.write_context_log(ctx.run_dir, run_day)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: context log not written: {e}")
        # The record each context signal is building (plan/context-layer.md item 5):
        # one-month ICs from the logs themselves, reporting only.
        try:
            import context_eval
            from datetime import date as _d_ev
            _ev = context_eval.evaluate(today=_d_ev.fromisoformat(run_day))
            print("  " + context_eval.summary_line(_ev))
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: context evaluation unavailable: {e}")

    # ---- 12. Generate interactive dashboard ----
    try:
        from generate_dashboard import generate_dashboard
        import shutil
        dash_path = generate_dashboard(ctx.run_dir)
        # Copy to project root as the canonical dashboard location
        main_dash = ROOT / "dashboard.html"
        shutil.copy2(dash_path, main_dash)
        # `index.html` is what GitHub Pages serves, so it gets the dashboard
        # itself - the same copy `data-run.ps1` makes after this step.
        #
        # 2026-10-08: this used to write a ~350-byte `<meta refresh>` stub to
        # `index.html` instead. The scheduled path never showed it, because
        # `data-run.ps1` overwrites the file with `dashboard.html` immediately
        # afterwards; a plain `python run_screener.py` left the published page
        # as a redirect and the tree **failing ship gate 3**, which requires
        # `index.html` to exceed 50,000 bytes. The weaker of two paths doing the
        # same job was the bug, as usual. Keeping one line of Python do it means
        # the two paths cannot disagree again - `tests/test_index_is_the_dashboard.py`.
        shutil.copy2(dash_path, ROOT / "index.html")
        # Copy companion data file (generated alongside the dashboard HTML)
        src_data_js = ctx.run_dir / "dashboard_data.js"
        if src_data_js.exists():
            shutil.copy2(src_data_js, ROOT / "dashboard_data.js")
        # The context layer's own file (plan/context-layer.md) travels with it.
        src_ctx_js = ctx.run_dir / "dashboard_context.js"
        if src_ctx_js.exists():
            shutil.copy2(src_ctx_js, ROOT / "dashboard_context.js")
        print(f"  Dashboard: {main_dash}")
    except Exception as e:
        print(f"  WARNING: Dashboard generation failed: {e}")


    # ---- 13. Improvement tracking snapshot ----
    # Phase 13 governance: honor the improvement.enabled kill switch. When the
    # engine is disabled, we record NOTHING (no snapshot, no dispersion) so an
    # operator who sets `enabled: false` genuinely turns the engine off.
    if cfg.get("improvement", {}).get("enabled", True):
        try:
            from improvement_engine import record_run_snapshot
            record_run_snapshot(ctx.run_id, datetime.now().strftime("%Y-%m-%d"), df, port, cfg)
            print("  Improvement snapshot recorded")
        except Exception as e:
            print(f"  WARNING: Improvement tracking failed: {e}")
    else:
        print("  Improvement engine disabled (improvement.enabled=false) — snapshot skipped")


if __name__ == "__main__":
    main()
