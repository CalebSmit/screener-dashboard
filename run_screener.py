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


# ---------------------------------------------------------------------------
# Factor Engine integration (with resilience)
# ---------------------------------------------------------------------------
def generate_screener_overview(cfg: dict) -> None:
    """Auto-generate SCREENER_OVERVIEW.md from the live config.

    Reads metric weights, factor weights, thresholds, and filter settings
    from cfg and templates them into a human-readable methodology document.
    This ensures the overview always matches the actual screener configuration.
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
    win_lo, win_hi = dq.get("winsorize_percentiles", [1, 99])
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

This is a quantitative stock screener. It takes every company in the S&P 500 (roughly 500 stocks), measures each one across up to {n_total} financial metrics, combines those measurements into a single composite score (0-100), and ranks the entire universe from best to worst. The top-ranked stocks form a model portfolio.

Not every stock sees all {n_total} metrics. The screener uses {n_generic} generic metrics for most stocks and a separate set of {n_bank_only} bank-specific metrics for financial companies (banks, insurers, credit companies). In practice, any individual stock is scored on about {n_generic} metrics — the set just differs depending on whether the company is a bank or not.

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

**Why these?** Growth without overpaying is the sweet spot. Forward EPS Growth gets the most weight because it's forward-looking (the market prices in the future, not the past). The PEG Ratio bridges valuation and growth into a single number — it penalizes stocks with high P/E ratios relative to their growth, preventing the screener from chasing expensive growers. Sustainable Growth acts as a sanity check — if a company is growing faster than its sustainable rate, it may need external financing to keep it up.

---

### 4. Momentum ({fw.get('momentum', 0)}% of final score)

**Question it answers:** *Has the market been rewarding this stock recently?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(mom_w, _MOM_DESCRIPTIONS)}

**Why these?** Decades of academic research (Jegadeesh & Titman, 1993) show that stocks that have gone up tend to keep going up over 3-12 month horizons. The skip-month convention (excluding the most recent month) is the standard academic momentum signal — the last month is excluded because very recent winners tend to experience a brief pullback. Both metrics use calendar-based date targeting instead of fixed index offsets, which ensures consistent lookback periods regardless of holidays or trading day variations.

**Volatility-regime adjustment:** The screener tracks market-wide volatility across runs (stored in `cache/vol_history.csv`). Once 20+ historical observations are available, it classifies the current volatility environment as HIGH, NORMAL, or LOW (using 25th/75th percentile thresholds of historical volatility). In HIGH-vol regimes, momentum weight is reduced by 30% (freed weight goes to Quality + Valuation), because momentum crashes are most common during volatile markets. In LOW-vol regimes, momentum weight is increased by 15% (taken from Valuation), because calm markets are where momentum works best. This adaptive scaling requires at least 20 screener runs before activating.

---

### 5. Risk ({fw.get('risk', 0)}% of final score)

**Question it answers:** *How bumpy is the ride?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(risk_w, _RISK_DESCRIPTIONS)}

**Why these?** All else equal, less volatile stocks are preferable — the "low volatility anomaly" is one of the most robust findings in finance. Volatility and Beta measure total and systematic risk respectively. Sharpe and Sortino Ratios reward stocks that deliver more return per unit of risk (Sortino penalizes only downside volatility, which matters more to investors). Max Drawdown captures worst-case loss — a stock that drops 50% needs a 100% gain to recover. Together these five metrics favor steadier, more risk-efficient companies.

---

### 6. Analyst Revisions ({fw.get('revisions', 0)}% of final score)

**Question it answers:** *What do Wall Street analysts think — and are they getting more or less optimistic?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(rev_w, _REV_DESCRIPTIONS)}

**Why these?** Estimate revisions and analyst targets are among the most powerful short-term return predictors. When a company consistently beats earnings estimates, the stock price usually follows — but with a lag, which creates an opportunity. Analyst Surprise gets the highest weight because it's a harder, backward-looking signal with less optimism bias than forward price targets. Earnings Acceleration (a continuous delta, not binary) and Beat Score (recency-weighted, not a simple streak counter) capture the *trajectory* and *consistency* of earnings beats — a company whose surprise % is improving quarter-over-quarter, and which has beaten in recent quarters with higher recency weight, signals genuine fundamental momentum. This category is weighted at only {fw.get('revisions', 0)}% because coverage can be sparse (not all stocks have active analyst coverage), and when coverage drops below usable levels, the weight automatically redistributes to the other categories.

*Note: EPS forecast revision (change in consensus forward EPS over 3-6 months) would be ideal as a fifth metric here but is not feasible with yfinance, which does not provide historical consensus data. Future enhancement: integrate I/B/E/S data from FactSet or Refinitiv.*

---

### 7. Size ({fw.get('size', 0)}% of final score)

**Question it answers:** *Does this stock benefit from the small-cap premium?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
{_metric_table(size_w, _SIZE_DESCRIPTIONS)}

**Why this?** The Fama-French SMB (Small Minus Big) factor captures the historical tendency for smaller companies to outperform larger ones over long horizons. Within the S&P 500, this creates a mild tilt toward mid-cap names (which are still large-cap by absolute standards) rather than megacaps. Using the log transform compresses the enormous range of market caps ($2B to $3T+) into a more linear scale that ranks sensibly.

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

The screener detects bank-like stocks using a three-tier classification:

1. **Explicit exclusion list** — Payment processors and financial data companies (V, MA, PYPL, FIS, FISV, SPGI, MCO, ICE, CME, etc.) have conventional P&Ls and use generic metrics despite being in the Financials sector.
2. **Industry matching** — Companies in banking, insurance, credit services, or mortgage finance industries use bank metrics.
3. **Sector fallback** — Unknown Financials-sector companies default to bank metrics (conservative — P/B + ROE is a safer default than EV/EBITDA for an unknown financial).

Bank-like stocks get an entirely different set of metric weights within the Valuation and Quality categories (see the tables in sections 1 and 2 above). Growth, Momentum, Risk, Revisions, Size, and Investment use the same generic weights for all stocks.

All financial-sector stocks receive a `Financial_Sector_Caveat` flag in the output, reminding the user that financial companies require additional scrutiny regardless of classification.

---

## How the Score Is Calculated

The scoring pipeline has six steps:

### Step 1: Collect Raw Data
For each of the ~500 stocks, the screener pulls quarterly financial statements, price data, earnings history, and analyst estimates from Yahoo Finance. Flow metrics (income statement and cash flow) use **LTM** (Last Twelve Months = sum of 4 most recent quarters); balance sheet items use **MRQ** (Most Recent Quarter). Falls back to annual filings if quarterly data is unavailable. Enterprise Value is cross-validated against computed MC + Debt - Cash; discrepancies > 10% (25% for Financials) trigger automatic correction. Data is cached locally in Parquet format (refreshed daily for prices, weekly for fundamentals) to avoid unnecessary API calls. Cache files are config-aware — changing weights or settings automatically invalidates stale caches.

### Step 2: Clean the Data (Winsorization)
Extreme outliers can distort rankings. The screener clips each metric at the {win_lo}st and {win_hi}th percentiles — for example, if one company has a Debt/Equity ratio of 50x while the rest are under 5x, that 50x gets clipped down to the {win_hi}th percentile value. This prevents a single extreme value from dominating the score.

### Step 3: Rank Within Sectors
Each metric is converted to a **sector-relative percentile** (0-100). A stock's EV/EBITDA isn't compared to all 500 companies — it's compared only to other companies in the same GICS sector (Technology vs. Technology, Energy vs. Energy, etc.). This is critical because a "cheap" utility trades at a very different multiple than a "cheap" tech company. Sector-relative ranking makes apples-to-apples comparisons possible.

For metrics where lower is better (like EV/EBITDA, Debt/Equity, Volatility, P/B, PEG Ratio, Asset Growth), the percentile is flipped so that a higher percentile always means "better."

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

The raw composite is then converted to a cross-sectional percentile rank (0-100), so a score of 95 means "better than 95% of stocks in the universe."

### Step 6: Apply Trap Filters & Rank
After computing composite scores, the screener applies value trap and growth trap filters (see below), then produces the final ranking.

---

{pio_text}

## Data Quality Safeguards

The screener includes several layers of data quality protection:

- **Denominator floors:** Analyst surprise uses a $0.10 floor on estimated EPS; forward EPS growth uses a $1.00 floor on trailing EPS. These prevent near-zero denominators from producing extreme ratios.
- **Output clamping (configurable):** Forward EPS growth is clamped to [{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[0] * 100)}%, +{int(clamps.get('forward_eps_growth', [-0.75, 3.0])[1] * 100)}%]; price target upside is clamped to [{int(clamps.get('price_target_upside', [-0.50, 1.0])[0] * 100)}%, +{int(clamps.get('price_target_upside', [-0.50, 1.0])[1] * 100)}%]. These bounds are configurable in `config.yaml` under `metric_clamps`. They limit the impact of data anomalies (e.g., GAAP vs. normalized EPS mismatches, extreme analyst targets) while still allowing meaningful differentiation among high-growth stocks.
- **Coverage filter:** Stocks with fewer than {min_coverage}% of their applicable metrics available are excluded from the ranking entirely.
- **Coverage discount:** Stocks that pass the coverage filter but still have many missing metrics receive a mild composite discount. Below {int(cov_disc.get('threshold', 0.80) * 100)}% metric coverage, the composite is reduced by up to {int(cov_disc.get('penalty_rate', 0.15) * 100)}% per unit of coverage gap (e.g., a stock at 56% coverage gets a ~3.6% discount). This prevents stocks with sparse data from ranking artificially high due to weight redistribution concentrating the score on a few favorable metrics. {'**Currently enabled.**' if cov_disc.get('enabled', False) else '**Currently disabled.**'}
- **Auto-disable (category-level):** If the Revisions or Investment category has fewer than 30% of its metrics populated, the entire category's weight is zeroed and redistributed proportionally to the remaining categories.
- **Auto-reduce (metric-level):** If any individual metric has more than {auto_reduce_thresh}% NaN across the universe (e.g., a data source outage), its weight is automatically set to zero and redistributed within its category.
- **Metric-level alerts:** A warning is printed if any metric has more than {alert_thresh}% missing data across the universe.
- **LTM / MRQ data freshness:** All flow metrics (revenue, net income, EBITDA, cash flow) use LTM (Last Twelve Months = sum of 4 most recent quarters). Balance sheet items use MRQ (Most Recent Quarter). This reduces data staleness from up to 12 months (annual filings) to ~3 months. Falls back to annual filings if quarterly data is unavailable; prior-year comparisons fall back to annual col=1 when quarterly history is insufficient (< 8 quarters).
- **EV cross-validation:** The API-provided Enterprise Value is cross-checked against computed MC + Debt - Cash. If the discrepancy exceeds 10% (or 25% for Financials, whose "debt" includes customer deposits that legitimately diverge from simple EV math), the computed value is used and the ticker is flagged (`_ev_flag`). This catches known yfinance EV parsing bugs (4x+ discrepancy for some tickers).
- **LTM partial annualization tracking:** When only 3 of 4 quarters are available for a flow metric, the screener annualizes (sum × 4/3) but flags the ticker with `_ltm_annualized = True` and records which fields were affected. This transparency lets users know which metrics are based on extrapolated rather than complete data.
- **Channel-stuffing detection:** Compares receivables growth vs. revenue growth. When receivables growth exceeds revenue growth by more than 15 percentage points, the stock is flagged with `_channel_stuffing_flag = True`. This can indicate aggressive revenue recognition or deteriorating collection quality.
- **Beta overlap validation:** Beta computation requires at least 80% date overlap between the stock's daily returns and the S&P 500 market returns. Stocks with insufficient overlap get `beta = NaN` rather than a potentially misleading value. The overlap percentage is recorded in `_beta_overlap_pct`.
- **Data quality log:** Every data issue (missing fields, stale data, rate-limit failures) is logged to `validation/data_quality_log.csv` with ticker, severity, description, and action taken.
- **Structured pipeline logging:** A Python `logging`-based structured logger (`screener.pipeline`) records coverage statistics, filter actions, and scoring stage completions for machine-parseable diagnostics.

---

## Value Trap Detection

A stock can score well on valuation (cheap!) but be cheap for a reason — declining business, negative momentum, or analysts cutting estimates. The screener uses **majority logic (2-of-3)** to flag potential value traps: a stock is flagged only if it falls in the bottom {vt_quality_floor}% of **at least two** of these three categories:

- Quality Score (floor: {vt_quality_floor}th percentile)
- Momentum Score (floor: {vt_mom_floor}th percentile)
- Revisions Score (floor: {vt_rev_floor}th percentile)

This is more balanced than the alternative "any 1 breach" approach, which flagged roughly 60% of the universe — too aggressive to be useful. The 2-of-3 majority logic catches stocks with genuinely broad weakness while tolerating a single weak dimension (e.g., a quality stock with one bad momentum quarter). About 30% of stocks are typically flagged.

Missing data (NaN) in any of the three dimensions does **not** trigger a value trap flag — missing data is not the same as poor quality. These stocks receive a separate `Insufficient_Data_Flag`.

Each flagged stock also receives a **Value Trap Severity** score (0-100), computed as the average of how far below each threshold the stock falls across the dimensions that triggered the flag. A severity of 80 means the stock is deep in trap territory; a severity of 20 means it barely crossed the thresholds. This provides more granularity than the binary flag alone.

By default, value-trap-flagged stocks are {vt_action} from the model portfolio (configurable to flag-only mode).

---

## Growth Trap Detection

The mirror image of a value trap: a stock can score well on growth but be growing unsustainably — high growth with poor quality and/or deteriorating analyst sentiment. The screener uses the same **majority logic (2-of-3)** to flag potential growth traps: a stock is flagged only if **at least two** of these three conditions are met:

- Growth Score **above** the {gt_growth_ceil}th percentile (high growth)
- Quality Score **below** the {gt_quality_floor}th percentile (low quality)
- Revisions Score **below** the {gt_rev_floor}th percentile (deteriorating sentiment)

This catches "growth at any price" stocks — companies that are growing fast but burning cash, carrying deteriorating fundamentals, or losing analyst confidence.

Each flagged stock also receives a **Growth Trap Severity** score (0-100), computed as the average of how far above/below each threshold the stock falls across the dimensions that triggered the flag. Higher severity means deeper in trap territory.

By default, growth-trap-flagged stocks are {gt_action} from the model portfolio (configurable to flag-only mode).

---

## Portfolio Construction

After scoring and ranking, the screener builds a **model portfolio** from the top-ranked stocks:

- **Number of holdings:** Top {num_stocks} stocks (configurable)
- **Weighting:** {'Equal weight (each stock gets ~' + str(round(100/num_stocks)) + '%)' if weighting == 'equal' else 'Risk-parity (inverse-volatility weighting — lower-volatility stocks get more weight)'}
- **Sector cap:** Maximum {max_sector} stocks from any single sector, to avoid overconcentration
- **Position limits:** No single stock above {max_pos}%
- **Liquidity filter:** Stocks with less than ${min_adv_m:.0f}M average daily dollar volume (63-day average) are excluded from the portfolio. Stocks with missing volume data are also excluded (conservative default).
- **Trap exclusions:** Value-trap and growth-trap flagged stocks are excluded (unless configured as flag-only)

If a sector would exceed its cap, the excess stocks are dropped and replaced by the next-highest-ranked stocks from other sectors. Weights are redistributed proportionally.

---

## What Gets Output

The screener produces an **Excel workbook** (`factor_output.xlsx`) with up to 6 sheets:

### Sheet 1: Factor Scores
Every stock in the universe with all raw metrics, {n_factors} category scores, the composite score, rank, value trap flag (with severity 0-100), growth trap flag (with severity 0-100), financial sector caveat flag, bank classification, and bank-specific metrics (P/B, ROE, ROA, Equity Ratio) where applicable. Each stock also carries a data provenance tag (`_data_source`), metric coverage count, and an EPS basis mismatch flag. Score columns use quartile-based coloring (Q1=red, Q2=yellow, Q3=light green, Q4=green) for at-a-glance assessment.

### Sheet 2: Screener Dashboard
The top 50 stocks, formatted for quick review. Includes rank, composite score (quartile-colored), all {n_factors} category scores, and the value trap and growth trap flags with severity scores. Color-coded cells highlight strengths and weaknesses.

### Sheet 3: Model Portfolio
The final portfolio with ticker, sector, composite score, position weights, and portfolio-level statistics (weighted average beta, dividend yield, sector allocation breakdown).

### Sheet 4: DataValidation
The top 10 stocks with raw financial values (market cap, revenue, EPS, etc.) displayed for manual spot-checking. Highlights potential issues including EPS basis mismatches (GAAP vs. normalized), stale data, EV cross-validation discrepancies, LTM partial annualization flags, channel-stuffing flags (receivables growth diverging from revenue growth), and beta overlap warnings. Also includes a sector-median context table showing 25th/median/75th percentile for 8 key metrics across each sector.

### Sheet 5: Weight Sensitivity (when available)
Results of the weight sensitivity analysis. For each factor category, the sheet shows what happens to the top-20 portfolio when that category's weight is perturbed ±5%. Jaccard similarity measures how stable the portfolio is — higher values (≥0.85) mean the ranking is robust to small weight changes. Color-coded: green (≥0.85), yellow (0.70–0.84), red (<0.70).

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
After scoring, the pipeline perturbs each factor category weight by ±5% (one at a time) and measures how much the top-20 portfolio changes using **Jaccard similarity** (intersection / union of the two top-20 sets). A Jaccard of 1.0 means the portfolio is completely unchanged; below 0.70 suggests the ranking is sensitive to that factor's weight. Results are printed to the console and saved in the Weight Sensitivity Excel sheet. This lets you verify that small weight changes don't drastically alter the output — a key requirement for any defensible quantitative process.

### EPS Basis Mismatch Detection
Yahoo Finance provides GAAP trailing EPS but normalized (non-GAAP) forward consensus EPS. When the ratio of forward-to-trailing EPS exceeds 2.0× or falls below 0.3× (and trailing EPS is above $0.10), the stock is flagged with `_eps_basis_mismatch = True`. This alerts users that the forward EPS growth and PEG ratio metrics may be distorted by a GAAP/non-GAAP mismatch rather than a genuine change in earnings expectations. Flagged stocks appear highlighted in the DataValidation sheet.

### Factor Correlation Matrix
A Spearman rank correlation matrix of all category scores is computed and written to the Factor Correlation Excel sheet. This makes explicit the degree of overlap between factors — for example, Momentum's two sub-metrics (12-1M and 6-1M return) share ~6 months of overlap, and EV-based valuation metrics are structurally correlated. Correlations above 0.6 are highlighted orange; above 0.8 are highlighted red. This transparency allows users to assess the effective number of independent signals.

### Data Provenance
Every stock carries three provenance fields: `_data_source` (where the data came from — e.g., "yfinance", "cache", "sample"), `_metric_count` (how many core metrics have valid data), and `_metric_total` (total possible metrics for that stock type). This makes per-stock data completeness visible at a glance.

### DataValidation Sheet
The top 10 portfolio stocks are displayed with raw financial values (market cap, revenue, net income, EPS, price) for manual spot-checking against external sources (e.g., Bloomberg, SEC filings). The sheet highlights six types of potential issues: EPS basis mismatches, stale data (price targets that may be outdated), EV cross-validation discrepancies, LTM partial annualization (3-of-4 quarters extrapolated to LTM), channel-stuffing flags (receivables growth outpacing revenue growth by >15pp), and beta overlap warnings (<80% date overlap with market). A sector-median context table shows 25th/median/75th percentile for 8 key metrics across each sector, enabling quick sanity checks.

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
| **Winsorization at {win_lo}%/{win_hi}%** | Prevents a single extreme data point from blowing up the rankings. Conservative clip — keeps {win_hi - win_lo}% of the distribution intact. |
| **Value trap 2-of-3 majority logic** | OR logic (any 1 breach) flagged ~60% of the universe — too aggressive. Majority logic catches genuinely weak stocks while tolerating one bad dimension. |
| **Growth trap 2-of-3 majority logic** | Mirror of value trap for the opposite scenario. Catches high-growth stocks with poor quality and/or deteriorating sentiment. |
| **Liquidity filter** (${min_adv_m:.0f}M daily dollar volume) | Ensures portfolio stocks are tradeable at scale. NaN volume is excluded conservatively. |
| **4-metric revisions category** (Surprise + Target + Acceleration + Beat Score) | Broadens the analyst sentiment signal beyond a single backward-looking and forward-looking metric. Earnings Acceleration (continuous delta) and Beat Score (recency-weighted) capture the trajectory and consistency of beats with much higher granularity than binary signals. |
| **Volatility-regime momentum scaling** | Momentum crashes in high-vol markets. Reducing momentum weight in turbulent conditions and boosting it in calm markets improves risk-adjusted returns (requires 20+ historical runs to activate). |
| **5-metric risk category** (Vol + Beta + Sharpe + Sortino + MaxDD) | Volatility and Beta capture total and systematic risk; Sharpe and Sortino capture risk-adjusted efficiency; Max Drawdown captures tail risk. Five metrics give a more complete risk picture than two. |
| **Quartile-based Excel coloring** | Absolute thresholds (e.g., >80 = green) assume a stable score distribution. Quartile-based coloring adapts to the actual distribution, ensuring roughly 25% of cells in each color band regardless of market conditions. |
| **Trap severity scores** (0-100 continuous) | Binary flags lose information. Severity scores quantify how deep in trap territory a stock is — severity 80 is much worse than severity 20, but both would be flagged as True. |
| **Beta overlap validation** (≥80% required) | Stocks with limited trading history (IPOs, relisted) can produce misleading beta values from sparse overlap with the market index. The 80% threshold ensures the regression uses substantially the same time period as the market. |
| **Channel-stuffing detection** (receivables vs revenue divergence) | When receivables growth exceeds revenue growth by >15pp, it may indicate aggressive revenue recognition. The flag is informational (not used in scoring) but appears in the DataValidation sheet. |

---

## Limitations to Be Aware Of

1. **Data source:** All data comes from Yahoo Finance (free, unofficial API). Occasional field name changes, rate limiting, or missing data are handled gracefully (the screener returns NaN and continues), but the data quality is not institutional-grade. Approximately 10-25% of tickers may fail to fetch on a given run due to Yahoo Finance rate limiting (HTTP 429).

2. **GAAP vs. normalized EPS:** Yahoo Finance provides GAAP trailing EPS but normalized forward consensus. For companies with large non-cash charges, write-downs, or unrealized gains (e.g., insurers like CINF), the forward EPS growth metric may show misleading declines. The $1.00 denominator floor and [-75%, +150%] clamp mitigate extreme cases but don't fully solve this inherent data limitation.

3. **Point-in-time:** The screener uses the latest available financial data. It does not reconstruct what was known at a past date, which means backtests carry look-ahead bias for fundamental metrics.

4. **Analyst coverage:** The Revisions category relies on analyst estimate and price target data, which is sparse for some stocks. When individual metrics are missing, their weight is redistributed within the category. When the entire category is unavailable, its weight redistributes to the other categories.

5. **No EPS revision data:** yfinance does not provide historical consensus EPS estimates, so the Revisions category cannot include the single most powerful revisions signal (change in forward EPS consensus over time). This would require a paid data source like FactSet or Refinitiv I/B/E/S.

6. **Rebalance frequency:** The model portfolio is a snapshot. It should be re-run at the configured frequency (monthly or quarterly) to stay current.

7. **Not investment advice:** This is a screening tool, not a recommendation engine. The output is a ranked list to narrow your research — not a list of stocks to blindly buy.

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

# Output: factor_output.xlsx (up to 6 sheets)

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
5. Flagging potential value traps and growth traps (2-of-3 majority logic)
6. Applying a liquidity filter to ensure tradeability
7. Building a diversified model portfolio from the top picks

The result is a disciplined, repeatable, multi-dimensional ranking that avoids the tunnel vision of looking at any single metric in isolation.
"""
    overview_path = ROOT / "SCREENER_OVERVIEW.md"
    overview_path.write_text(md, encoding="utf-8")


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
    "max_drawdown_1y": "Max Drawdown (1Y)",
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
    "net_debt_to_ebitda": "(Total Debt - Cash) / EBITDA. Measures leverage relative to earnings power. Lower = less leveraged = better. Replaces Debt/Equity (negative equity from buybacks distorts D/E).",
    "piotroski_f_score": "A 0-9 checklist scoring profitability, leverage, liquidity, and efficiency trends. Higher = healthier fundamentals.",
    "accruals": "(Net Income - Operating Cash Flow) / Total Assets. Lower (more negative) = higher earnings quality (Sloan 1996).",
    "operating_leverage": "Degree of Operating Leverage (%Δ EBIT / %Δ Revenue). Lower = more durable earnings (less sensitivity to revenue swings). Banks skip this metric.",
    "beneish_m_score": "8-variable earnings manipulation detection model (Beneish 1999). More negative = lower manipulation risk. Requires ≥5 of 8 variables. Non-bank only.",
    "roe": "Return on equity — the key bank profitability metric. Higher = better.",
    "roa": "Return on assets — key bank efficiency metric. Higher = better.",
    "equity_ratio": "Total equity divided by total assets. Solvency measure — higher = more capital = safer.",
}

_GROWTH_DESCRIPTIONS = {
    "forward_eps_growth": "(Forward EPS - Trailing EPS) / Trailing EPS. Denominator floored at $1.00. Clamped to [-75%, +150%]. Higher = faster expected growth.",
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
    "volatility": "Annualized standard deviation of daily returns over the past year. Lower = smoother ride.",
    "beta": "Covariance of stock returns with S&P 500 returns divided by variance of market returns. Requires ≥80% date overlap with market. Lower = less market-driven risk.",
    "sharpe_ratio": "(12-month return - risk-free rate) / volatility. Risk-adjusted return per unit of total risk. Higher = more efficient risk-taking.",
    "sortino_ratio": "(12-month return - risk-free rate) / downside deviation. Like Sharpe but only penalizes downside volatility. Higher = better downside-adjusted return.",
    "max_drawdown_1y": "Maximum peak-to-trough decline from cumulative daily return series over the past year. Less negative = smaller worst-case loss.",
}

_REV_DESCRIPTIONS = {
    "analyst_surprise": "Median of (Actual - Estimated EPS) / max(|Estimated|, $0.10) over last 4 quarters. Positive = beat expectations.",
    "price_target_upside": "(Mean Analyst Price Target - Current Price) / Current Price. Clamped to [-50%, +100%]. Higher = more analyst optimism.",
    "earnings_acceleration": "Difference between most recent quarter's surprise % and prior quarter's surprise %. Positive = accelerating beats, negative = decelerating. Continuous, winsorized at 1st/99th percentiles.",
    "consecutive_beat_streak": "Recency-weighted beat score: each of the last 4 quarters' beats weighted by recency (Q1=1, Q2=2, Q3=3, Q4=4). Range 0-10. A stock beating all 4 quarters scores 10; beating only the most recent scores 4.",
    "short_interest_ratio": "Days to cover (short interest shares / average daily volume). Lower = less bearish sentiment from short sellers. Contrarian signal.",
}

_SIZE_DESCRIPTIONS = {
    "size_log_mcap": "Negative natural log of market capitalization: -log(marketCap). Smaller companies get higher values.",
}

_INV_DESCRIPTIONS = {
    "asset_growth": "Year-over-year change in total assets. Lower = better (conservative investment, Fama-French CMA).",
}


# ---------------------------------------------------------------------------
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


def run_factor_engine(cfg, args, ctx=None):
    """Run the complete factor scoring pipeline. Returns (scored_df, stats_dict)."""
    pipeline_log = _init_pipeline_logger()

    from factor_engine import (
        get_sp500_tickers, fetch_single_ticker, fetch_all_tickers,
        fetch_market_returns, fetch_risk_free_rate, compute_metrics,
        _generate_sample_data, _find_latest_cache,
        apply_universe_filters,
        winsorize_metrics, compute_sector_percentiles,
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

        risk_free_rate = fetch_risk_free_rate()
        print(f"  [RUN] Risk-free rate (^IRX): {risk_free_rate*100:.2f}%")

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
        df = compute_metrics(raw, market_returns, cfg, risk_free_rate=risk_free_rate)

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

        # Coverage filter — count only metrics applicable to each stock type.
        # Bank-only metrics are structurally NaN for non-banks and vice versa;
        # including them in the denominator would penalize stocks unfairly.
        present = [c for c in METRIC_COLS if c in df.columns]
        is_bank = df.get("_is_bank_like", pd.Series(False, index=df.index)).fillna(False)
        applicable_count = pd.Series(0, index=df.index)
        metric_count = pd.Series(0, index=df.index)
        for c in present:
            # Determine which rows this metric applies to
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
    pipeline_log.info("Winsorization complete: %d stocks", len(df))

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
    stats["missing_pct"] = {}
    for lbl, col in labels:
        pct = df[col].isna().sum() / len(df) * 100 if col in df.columns else 100
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

    # ---- Factor correlation matrix (for transparency) ----
    corr = compute_factor_correlation(df)
    if ctx is not None and not corr.empty:
        ctx.save_artifact("06_factor_correlation", corr.reset_index())
    stats["_corr_df"] = corr

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
            n_sheets = 3
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

    # ---- 11. Auto-regenerate SCREENER_OVERVIEW.md from live config ----
    try:
        generate_screener_overview(cfg)
        print("  SCREENER_OVERVIEW.md regenerated from live config")
    except Exception as e:
        print(f"  WARNING: Overview generation failed: {e}")

    # ---- 12. Generate interactive dashboard ----
    try:
        from generate_dashboard import generate_dashboard
        import shutil
        dash_path = generate_dashboard(ctx.run_dir)
        # Copy to project root as the canonical dashboard location
        main_dash = ROOT / "dashboard.html"
        shutil.copy2(dash_path, main_dash)
        # Also copy to index.html so GitHub Pages serves it at the root URL
        shutil.copy2(dash_path, ROOT / "index.html")
        print(f"  Dashboard: {main_dash}")
    except Exception as e:
        print(f"  WARNING: Dashboard generation failed: {e}")


if __name__ == "__main__":
    main()
