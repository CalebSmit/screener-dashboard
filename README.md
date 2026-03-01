# Multi-Factor Stock Screener v1.0

A standalone, multi-factor equity screening pipeline that scores S&P 500 stocks
across six factor categories—Valuation, Quality, Growth, Momentum, Risk, and
Analyst Revisions—then constructs a sector-constrained model portfolio and writes
everything to a formatted Excel workbook.

## Quick Start

```bash
# 1. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. Run the full pipeline
python run_screener.py
```

That's it. On first run the screener will attempt to fetch live data via
yfinance; if network access is unavailable it falls back to sector-realistic
synthetic data so the entire pipeline can be validated end-to-end.

## Configuration

All tuneable parameters live in **`config.yaml`**:

- **`universe`** — index selection, minimum market cap & volume, sector/ticker
  exclusions.
- **`factor_weights`** — category-level weights (must sum to 100). Set
  `revisions: 0` to disable analyst data.
- **`metric_weights`** — within-category metric weights (each category sums to
  100).
- **`sector_neutral`** — toggle sector-relative scoring; set GICS level and cap
  multiplier.
- **`value_trap_filters`** — quality/momentum/revisions floor percentiles;
  `flag_only: true` flags but does not exclude.
- **`portfolio`** — number of stocks, weighting scheme (`equal` or
  `risk_parity`), position-size caps, sector concentration limit, rebalance
  frequency.
- **`caching`** — refresh intervals for price / fundamental / estimate data;
  format (`parquet` or `csv`).
- **`data_quality`** — winsorize percentiles, minimum coverage thresholds.
- **`output`** — Excel filename and sheet names.

## Output Files

| File | Description |
|------|-------------|
| `factor_output.xlsx` | 3-sheet workbook: **FactorScores** (full universe), **ScreenerDashboard** (top 25, sector summary, score distribution), **ModelPortfolio** (holdings, sector allocation, factor exposure) |
| `cache/factor_scores_YYYYMMDD.parquet` | Scored universe cached for fast warm-start |
| `validation/data_quality_log.csv` | Per-ticker data quality issues (Appendix C format) |
| `validation/backtest_results.csv` | Monthly decile returns (Phase 2) |
| `validation/factor_ic_timeseries.csv` | Spearman IC by factor over time (Phase 2) |
| `validation/value_trap_comparison.csv` | Value trap filter impact analysis (Phase 2) |

## Publishing the Dashboard

The live dashboard is hosted at: https://calebsmit.github.io/screener-dashboard/

After running the screener, `dashboard.html` and `index.html` are updated
automatically. To push the latest version to GitHub Pages:

```powershell
git add -A; git commit -m "Update dashboard"; git push
```

GitHub Pages will rebuild and the site will be live within ~1 minute.

## CLI Flags

```
python run_screener.py [OPTIONS]

Options:
  --refresh          Force-clear all Parquet cache and re-fetch data
  --tickers T1,T2    Score only the listed tickers (quick test mode)
  --no-portfolio     Skip portfolio construction; write FactorScores only
```

### Examples

```bash
# Full run (default)
python run_screener.py

# Force fresh data
python run_screener.py --refresh

# Quick test on 3 stocks
python run_screener.py --tickers AAPL,MSFT,GOOGL

# Scores only, no portfolio sheet
python run_screener.py --no-portfolio
```

## Known Limitations

1. **yfinance dependency** — data quality depends on Yahoo Finance's free API,
   which may throttle or return stale fields. The screener retries failed
   fetches up to 3 times with exponential backoff (1 s / 2 s / 4 s).
2. **Analyst revisions coverage** — if fewer than 30 % of tickers have revision
   data the category is auto-disabled and its weight is redistributed
   proportionally across the remaining categories.
3. **No intraday data** — all price data is daily close; momentum factors use
   monthly price points derived from daily history.
4. **Sandbox / offline mode** — when the network is unavailable the pipeline
   generates sector-realistic synthetic data. Scores are structurally valid but
   carry no real investment signal.
5. **Single-threaded Excel writes** — openpyxl does not support concurrent
   writes. Large universes (>1 000 stocks) may take several seconds to write.
6. **No real-time signals** — the screener is designed for end-of-day batch runs,
   not live trading.

## Disclaimer

This tool is provided for **educational and research purposes only**. It does
not constitute investment advice. The model portfolio is a quantitative screen,
not a recommendation to buy or sell any security. Past performance of any
backtested strategy does not guarantee future results. Always perform your own
due diligence and consult a qualified financial advisor before making investment
decisions.

## Dependencies

See `requirements.txt` for the full list. Core libraries:

- **pandas / numpy / scipy** — data wrangling, statistics, winsorization
- **yfinance** — market data (S&P 500 constituents, fundamentals, price history)
- **openpyxl** — Excel workbook creation with formatting
- **PyYAML** — configuration file parsing
- **pyarrow** — Parquet cache I/O
