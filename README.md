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

The metric registry (`METRIC_COLS` in `factor_engine.py`) has **44 entries**:

- **32 scored generic metrics** — carry non-zero weight, applied to non-bank stocks.
- **4 bank-specific metrics** — P/B, ROE, ROA, Equity Ratio — substituted for
  certain generic metrics on financial companies.
- **8 candidate metrics at weight 0** — pre-implemented but inactive; the
  self-improvement engine may activate them over time based on live
  information-coefficient evidence.

### Composite Score

Category scores are weighted and summed into a raw composite, which is then
converted to a **cross-sectional percentile rank** (`rank(pct=True) * 100`) — so a
score of 95 means "better than 95% of the universe." (This is a percentile-rank
transform, not min-max scaling.)

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
`dashboard_data.js` payload (lazy-loaded). To publish the latest version to
GitHub Pages:

```powershell
git add -A; git commit -m "Update dashboard"; git push
```

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
