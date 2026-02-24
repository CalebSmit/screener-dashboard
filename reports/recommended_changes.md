# Recommended Changes — Prioritized Improvements

Generated: 2026-02-19 21:24:01

Changes are ordered by impact on institutional defensibility.

## Priority 1: Critical (Data Foundation)

### 1.1 Replace yfinance with a Licensed Data Provider

**What**: Switch from yfinance to a licensed data API (e.g., Tiingo, Polygon.io, IEX Cloud, Bloomberg B-PIPE)
**Why**: yfinance scrapes Yahoo Finance without a commercial license. Data can change format, disappear, or return incorrect values without notice. No institutional investor will accept a strategy built on web scraping.
**Implementation**:
- Create an abstract `DataProvider` class with `fetch_ticker()`, `fetch_market_returns()` methods
- Implement `YFinanceProvider` (current) and `TiingoProvider` (or similar)
- Config setting: `data_provider: tiingo` with API key in env var
**Impact on composite**: None (metric definitions stay the same; only data source changes)

### 1.2 Freeze Universe Source

**What**: Use a versioned, static universe file instead of scraping Wikipedia
**Why**: Wikipedia can be edited by anyone, and the S&P 500 list changes during the run. This creates a race condition where the universe can differ between runs.
**Implementation**:
- Update `sp500_tickers.json` weekly from a reliable source (S&P Global, IEX)
- Add `universe_version` field to the JSON with update timestamp
- Only use Wikipedia as a cross-validation check, not primary source
**Impact on composite**: Minimal (same ~503 tickers)

## Priority 2: High (Reproducibility)

### 2.1 Data Snapshot for Each Run

**What**: Save the complete raw API response for every ticker in each run
**Why**: Without a data snapshot, you cannot reproduce a prior run's results. This is a fundamental requirement for audit trails.
**Implementation**:
- Already partially done: `ctx.save_artifact('00_raw_fetch', raw_df)` in `run_screener.py`
- Extend to save `_daily_returns` (currently excluded for size)
- Add a `--replay-run <run_id>` flag that loads raw data from a prior run instead of fetching
**Impact on composite**: None (observational change)

### 2.2 Deterministic Scoring Mode

**What**: Add a `--deterministic` flag that pins random seeds and uses frozen data
**Why**: For regression testing and validation, you need `f(same_input) = same_output`
**Implementation**:
- The scoring pipeline is already deterministic given fixed inputs
- Need: `--replay-run` to freeze inputs, and a golden-output test
**Impact on composite**: None

## Priority 3: Medium (Methodology)

### 3.1 Add Explain Function for Individual Stocks

**What**: `explain_score(ticker)` that prints a full decomposition of how a stock got its score
**Why**: IC presentations require the ability to explain any holding. Currently requires manual Excel inspection.
**Implementation**:
- Function that takes a scored DataFrame row and prints the per-metric → per-category → composite waterfall
- Include sector context ("AAPL's ROIC percentile is 85th vs IT sector median of 62nd")
**Impact on composite**: None (read-only)

### 3.2 Multi-Period Financial Statements

**What**: Use TTM (trailing twelve months) instead of annual filings
**Why**: Annual filings can be 1-12 months stale. TTM data is more current and reduces temporal mismatch with market data.
**Implementation**:
- yfinance provides both `.financials` (annual) and `.quarterly_financials`
- Sum last 4 quarters for TTM revenue, net income, EBITDA, etc.
- Fall back to annual if < 4 quarters available
**Impact on composite**: Moderate — TTM values will differ from annual, especially for seasonal businesses

### 3.3 Add Sector-Neutral Composite Mode

**What**: Option to compute composite as percentile-within-sector instead of cross-sectional
**Why**: Cross-sectional ranking can overweight sectors with structurally higher quality scores (e.g., IT). Sector-neutral ranking ensures each sector contributes proportionally.
**Implementation**:
- Already partially implemented: `sector_relative_composite: true` in config
- Currently disabled by default
- Needs testing and documentation
**Impact on composite**: Significant — changes ranking order by sector

## Priority 4: Low (Polish)

### 4.1 Vectorize compute_metrics()

**What**: Replace the per-ticker Python loop with vectorized pandas operations
**Why**: ~3-5x speedup for the computation phase (currently ~1s for 500 tickers, would be ~200ms)
**Implementation**: Convert `for d in raw_data` loop to DataFrame operations
**Impact on composite**: None (same results, faster)

### 4.2 Add CI/CD Pipeline

**What**: GitHub Actions that runs tests, linting, and a sample-data smoke test on each push
**Why**: Prevents regressions and ensures the scoring pipeline always works end-to-end
**Implementation**: `.github/workflows/test.yml` with `python run_screener.py --tickers AAPL,MSFT` using sample data
**Impact on composite**: None

### 4.3 Add Interactive Dashboard

**What**: Streamlit or Panel dashboard for exploring scores, factor exposures, and portfolio
**Why**: Excel is limiting for interactive analysis. A dashboard allows filtering, sorting, and drill-down.
**Implementation**: `streamlit_app.py` that reads `factor_output.xlsx` or cached Parquet
**Impact on composite**: None (read-only visualization)