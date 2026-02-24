# Defensibility Review — CFA / Hedge Fund Grade Assessment

Generated: 2026-02-19 21:24:01

This review assesses the Multi-Factor Stock Screener's suitability
for institutional use. Each area is scored 1-5 (1=critical deficiency, 5=institutional grade).

## 1. Data Provenance & Licensing Clarity — Score: 2/5

**Assessment**: Below institutional standard.

- **Source**: yfinance (unofficial Yahoo Finance scraper)
- **License**: yfinance is open-source but scrapes Yahoo Finance without explicit API agreement
- **Concern**: Yahoo Finance ToS may prohibit automated scraping for commercial use
- **Concern**: No data vendor contract or SLA — data can change format or disappear without notice
- **Concern**: No audit trail for when data was sourced or which API version was used
- **What's defensible**: Single, well-known source; no proprietary black-box inputs

**Red flags:**
- `factor_engine.py:_fetch_single_ticker_inner:293` — uses `yf.Ticker(ticker_str).info` which is an undocumented, unstable endpoint
- `factor_engine.py:get_sp500_tickers:63` — scrapes Wikipedia for universe, which can be edited by anyone
- No data licensing documentation in the repository

## 2. Point-in-Time Correctness — Score: 3/5

**Assessment**: Adequate for live screening, inadequate for backtesting.

- **What's right**: For a live screener, using current data is correct
- **What's right**: Financial statements use `.financials` which returns the latest ANNUAL filing, not TTM
- **Concern**: `forwardEps` from yfinance is a consensus estimate — source/vintage is opaque
- **Concern**: `trailingEps` may be TTM or fiscal year depending on yfinance's internal logic
- **Concern**: `_stmt_date_financials` is tracked but not enforced (stale filings are flagged but included)

**Red flags:**
- `factor_engine.py:compute_metrics:823` — forward EPS growth uses `forwardEps` vs `trailingEps` but these may span different periods
- `backtest.py` — acknowledged survivorship + look-ahead bias; NOT suitable for performance claims

## 3. Metric Definitions Consistency — Score: 4/5

**Assessment**: Good. Well-documented with clear rationale.

- **What's right**: 21 metrics across 6 categories, each with documented formula
- **What's right**: ROIC uses balance-sheet-consistent inputs (equity, debt, cash all from BS)
- **What's right**: Excess cash deduction in ROIC (cash - 2% revenue)
- **What's right**: Piotroski F-Score uses raw integer (not proportional normalization)
- **What's right**: Bank-specific metrics (P/B, ROE, ROA, Equity Ratio) with industry classification
- **Concern**: EV/EBITDA does not fall back to EBIT (documented, but reduces coverage)
- **Concern**: `_stmt_val()` substring matching could match wrong line items in edge cases

**Red flags:**
- `factor_engine.py:_stmt_val:210` — word-boundary substring fallback could match `Operating Income` when searching for `Income` in unusual filing formats

## 4. Robustness to Missing/Erroneous Data — Score: 4/5

**Assessment**: Good. Multiple layers of protection.

- **What's right**: Per-row weight redistribution for NaN metrics
- **What's right**: Coverage filter excludes stocks with < 60% applicable metrics
- **What's right**: Auto-reduce metrics with > 70% NaN
- **What's right**: Winsorization at 1st/99th percentiles
- **What's right**: Revisions auto-disable when coverage < 30%
- **What's right**: NaN percentiles are NOT imputed to 50th (weight redistributed)
- **Concern**: Earnings yield for negative-EPS stocks is correctly negative (not NaN), but this means deeply unprofitable companies get valid (very bad) valuation scores rather than being excluded

**Red flags:**
- None critical. The missing-data handling is well-designed.

## 5. Reproducibility — Score: 3/5

**Assessment**: Good infrastructure, but data source is inherently non-reproducible.

- **What's right**: `RunContext` saves config snapshot, effective weights, raw fetch artifacts
- **What's right**: Config-aware cache hashing prevents stale-config results
- **What's right**: Git SHA recorded in run metadata
- **Concern**: yfinance returns whatever Yahoo Finance has NOW — re-running next week gives different data
- **Concern**: Wikipedia universe scrape changes daily
- **Concern**: No way to replay a historical run with the exact same inputs

**Red flags:**
- `factor_engine.py:get_sp500_tickers:63` — Wikipedia scrape returns different data each run
- No data snapshot/freeze mechanism for institutional audit trail

## 6. Performance / Scalability / Caching — Score: 3/5

**Assessment**: Adequate for S&P 500, not designed for larger universes.

- **What's right**: Config-aware Parquet caching with configurable freshness
- **What's right**: Threaded batch fetching with adaptive rate-limit throttling
- **What's right**: Retry pass for failed tickers
- **Concern**: Full S&P 500 fetch takes 5-15 minutes depending on throttling
- **Concern**: No incremental update — re-fetches everything on cache miss
- **Concern**: `compute_metrics()` iterates row-by-row (not vectorized)

**Red flags:**
- `factor_engine.py:compute_metrics:652` — Python loop over raw_data list; O(n) with constant overhead per ticker

## 7. Explainability (IC Presentation) — Score: 4/5

**Assessment**: Good. Every score can be decomposed.

- **What's right**: Composite = weighted average of 6 category scores
- **What's right**: Each category score = weighted average of metric percentiles
- **What's right**: Each metric has documented formula and data source
- **What's right**: Factor correlation matrix available for double-counting detection
- **What's right**: Value trap flag logic is transparent (2-of-3 majority)
- **What's right**: Per-ticker weight redistribution is documented
- **Concern**: No built-in 'explain this score' function for individual stocks

**Red flags:**
- None. The model is interpretable by design.

## Overall Assessment

| Area | Score |
| --- | --- |
| Data Provenance | 2/5 |
| Point In Time | 3/5 |
| Metric Definitions | 4/5 |
| Robustness | 4/5 |
| Reproducibility | 3/5 |
| Performance | 3/5 |
| Explainability | 4/5 |
| **Average** | **3.3/5** |

### Grade: B-

### What IS Defensible Today
- Composite score methodology (weighted percentile ranks)
- Metric definitions and formulas (well-documented, academically grounded)
- Missing data handling (NaN-aware weight redistribution)
- Bank-specific metric treatment
- Value trap filter logic
- Config validation (Pydantic schemas)
- Data quality logging and alerts

### What is NOT Defensible Today
- Data provenance (yfinance has no SLA or licensing agreement)
- Reproducibility (no data snapshot/freeze; Wikipedia universe changes)
- Point-in-time correctness for backtesting
- Performance claims based on `backtest.py` (known biases)