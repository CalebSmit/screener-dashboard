# Multi-Factor Stock Screener

A standalone, multi-factor equity screening pipeline that scores S&P 500 stocks
across **eight factor categories**, constructs a sector-constrained model
portfolio, writes a formatted Excel workbook, and publishes a live dashboard.

> **Canonical methodology reference:** For the full, plain-language explanation of
> how the screener works — every metric, weight, filter, and design decision — see
> **[SCREENER_OVERVIEW.md](SCREENER_OVERVIEW.md)**. That document is the source of
> truth for methodology; this README is a quick operational guide.

## What It Does

The screener measures every S&P 500 company across a registry of financial
metrics, combines them into a single 0-100 composite score, ranks the universe,
and builds a model portfolio from the top-ranked names.

### The 8 Factor Categories

| Category | Weight | Captures |
|----------|--------|----------|
| Valuation | 22% | Is the stock priced attractively? |
| Quality | 22% | Is this a well-run, durable business? |
| Growth | 13% | Is the business growing sustainably? |
| Momentum | 13% | Has the market been rewarding it? |
| Risk | 10% | How volatile / drawdown-prone is it? |
| Revisions | 10% | What do analysts think, and is sentiment improving? |
| Size | 5% | Small-cap premium tilt |
| Investment | 5% | Conservative vs. aggressive asset growth |

Category weights sum to 100. Bank-like stocks (banks, insurers, credit
companies) use a bank-specific metric set within Valuation and Quality.

### Metric Registry

The metric registry (`METRIC_COLS` in `factor_engine.py`) has **46 entries**:

- **28 scored generic metrics** — carry non-zero weight, applied to non-bank stocks.
- **4 bank-specific metrics** — P/B, ROE, ROA, Equity Ratio — substituted for
  certain generic metrics on financial companies (25 weighted metrics in all for a bank).
- **14 candidate metrics at weight 0** — computed and shown, but unscored. A
  candidate joins the score only with a research note and a changelog entry.

*(2026-10-09: `operating_leverage` moved to weight 0 and `earnings_variability`
was added as a candidate. Re-derive these counts from `config.yaml` with
`factor_engine.weighted_metric_sets(cfg)` rather than editing them by hand.)*

*(Counts corrected 2026-09-10. The total was right but the split had read
32/4/8 since before `sharpe_ratio` and `sortino_ratio` were moved to weight 0
on 2026-09-02 — the errors cancelled, so the sum stayed plausible. Verified
against `METRIC_COLS` and `config.yaml` rather than incremented by hand.)*

### Composite Score

Category scores are weighted and averaged into a composite that is **cardinal**:
a 0-100 score which keeps its magnitude, and which is the ranking key. A stock
below the configured metric-coverage threshold then has its composite reduced by
the coverage discount. **The composite is not a percentile** — the universe
percentile is a separate column, `Composite_Pct` (`rank(pct=True) * 100`), kept
for display. So a composite of 95 means 95 points out of 100, *not* "better than
95% of the universe".

*(Corrected 2026-10-07. This section previously said the composite was converted
to a cross-sectional percentile rank, which stopped being true at Phase 13 (F1) —
`compute_composite` preserves cardinality deliberately so that conviction reaches
portfolio construction. See `SCREENER_OVERVIEW.md` Step 5 and Limitation 8.)*

## Quick Start

```bash
# 1. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. Run the full pipeline
python run_screener.py
```

On a machine with TLS interception (e.g. Avast), set `CURL_CA_BUNDLE` to
`.certs/combined_ca.pem` before running — yfinance's curl_cffi backend honors
`CURL_CA_BUNDLE`, and other SSL env vars do not affect the data path under
Python 3.13.

## Configuration

All tuneable parameters live in **`config.yaml`**:

- **`universe`** — index selection, minimum market cap & volume, sector/ticker exclusions.
- **`factor_weights`** — category-level weights (must sum to 100).
- **`metric_weights`** / **`bank_metric_weights`** — within-category metric weights (each category sums to 100).
- **`sector_neutral`** — toggle sector-relative scoring; GICS level and cap multiplier.
- **`value_trap_filters`** — quality/momentum/revisions floor percentiles; `flag_only` to flag without excluding.
- **`portfolio`** — number of stocks, weighting scheme, position/sector caps, rebalance frequency.
- **`caching`** — refresh intervals and format (`parquet` or `csv`); caches are config-hash-aware.
- **`data_quality`** — outlier-report percentiles (flagged, never clipped), coverage thresholds, metric clamps.
- **`improvement`** — self-improvement / metric-evolution engine settings.
- **`output`** — Excel filename and sheet names.

## Output Files

| File | Description |
|------|-------------|
| `factor_output.xlsx` | Up to **6-sheet** workbook: **FactorScores** (full universe, all 8 category scores + composite), **ScreenerDashboard** (top names, color-coded), **ModelPortfolio** (holdings, weights, sector allocation), **DataValidation** (raw values + data-quality flags), **WeightSensitivity** (±5% perturbation / Jaccard, when available), **FactorCorrelation** (Spearman matrix of category scores, when available) |
| `cache/factor_scores_<hash>_YYYYMMDD.parquet` | Scored universe cached for fast warm-start (config-hash tagged) |
| `runs/<run_id>/` | Raw fetch data, scored data, and config snapshot per run (reproducibility) |
| `validation/data_quality_log.csv` | Per-ticker data quality issues |
| `improvement/` | Snapshots, performance & live-IC history for the self-improvement engine |

## Dashboard

The live dashboard is hosted at: https://calebsmit.github.io/screener-dashboard/

`generate_dashboard.py` writes a lightweight `dashboard.html` plus a
`dashboard_data.js` payload (lazy-loaded) and a separate context file. Never
edit those outputs by hand. The scheduled data loop (`scripts/data-run.ps1`)
publishes them after its gates pass; see `CLAUDE.md` for the gates.

What the page holds (full list: `plan/dashboard-inventory.md`):

- **Rankings** for every S&P 500 stock, with a **Weighting** menu (Balanced,
  Value, Growth, Momentum). The engine computes each one at build time with
  the run's own momentum regime.
- **Stock sheet**: open any row (or press Ctrl/Cmd+K) to see each category
  score. Every metric opens to its formula, the stock's own inputs, and its
  rank among sector peers. Every score is rebuilt from the published inputs
  before the page is allowed to publish.
- **What Changed**: a short overview of what moved across the whole run,
  plus the movers since the last run and since a month ago.
- **Reporting Soon**: companies reporting within 7 or 14 days, with last
  quarter's earnings surprise.
- **My Holdings**: names you save, kept in your browser only. No cost
  basis and no share counts.
- **Before you decide**: trend, options, insider-trading and market-backdrop
  context. It is shown beside the score and never enters it.
- **Track Record**: how past top-ranked lists did afterwards. This is
  illustration, not evidence for the method.

## CLI Flags

```
python run_screener.py [OPTIONS]

Options:
  --refresh          Force-clear the Parquet cache and re-fetch data
  --tickers T1,T2    Score only the listed tickers (quick test mode)
  --top-n N          Number of holdings in the model portfolio
  --preset NAME      Apply a weighting preset (balanced / value / growth / momentum)
  --show-weights     Print the effective weights and exit
  --dry-run          Validate config and wiring without fetching or scoring
  --no-portfolio     Skip portfolio construction; write FactorScores only
```

### Examples

```bash
python run_screener.py                              # full run (default)
python run_screener.py --refresh                    # force fresh data
python run_screener.py --tickers AAPL,MSFT,GOOGL    # quick test on 3 stocks
python run_screener.py --preset value               # value-tilted weights
python run_screener.py --no-portfolio               # scores only
```

## Known Limitations

1. **yfinance dependency** — data quality depends on Yahoo Finance's free API,
   which may throttle (HTTP 429) or return stale fields. Roughly 10-25% of
   tickers may fail to fetch on a given run; failures are retried and logged.
2. **No portfolio risk model** — default weighting uses single-name volatility
   only; there is no covariance/correlation model, so portfolio risk may be
   understated for correlated holdings.
3. **Look-ahead bias in backtests** — the screener uses latest-available
   fundamentals and does not reconstruct point-in-time data.
4. **Analyst coverage sparsity** — the Revisions category auto-redistributes its
   weight when coverage is insufficient.
5. **No intraday data** — all price data is daily close.
6. **Not real-time** — designed for end-of-day batch runs, not live trading.

See [SCREENER_OVERVIEW.md](SCREENER_OVERVIEW.md) for the complete limitations
discussion.

## Disclaimer

This tool is provided for **educational and research purposes only**. It does
not constitute investment advice. The model portfolio is a quantitative screen,
not a recommendation to buy or sell any security. Past performance of any
backtested strategy does not guarantee future results. Always perform your own
due diligence and consult a qualified financial advisor before making investment
decisions.

## Dependencies

See `requirements.txt` for the full list. Core libraries:

- **pandas / numpy / scipy** — data wrangling and statistics
- **yfinance** — market data (S&P 500 constituents, fundamentals, price history)
- **openpyxl** — Excel workbook creation with formatting
- **PyYAML** — configuration file parsing
- **pyarrow** — Parquet cache I/O
