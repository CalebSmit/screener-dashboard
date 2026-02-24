# Composite Score Forensics Report

Generated: 2026-02-19 21:24:01

Tickers analyzed: JNJ, AAPL, JPM, XOM, AMZN

## A. Pipeline Map

```
Stage 1: Universe Selection     → get_sp500_tickers()        [factor_engine.py]
Stage 2: Data Fetch             → fetch_all_tickers()        [factor_engine.py]
         Market Returns          → fetch_market_returns()     [factor_engine.py]
Stage 3: Metric Computation     → compute_metrics()          [factor_engine.py]
Stage 4: Coverage Filter        → min_data_coverage_pct      [run_screener.py]
Stage 5: Winsorization          → winsorize_metrics()        [factor_engine.py]
Stage 6: Sector Percentile Rank → compute_sector_percentiles [factor_engine.py]
Stage 7: Category Scores        → compute_category_scores()  [factor_engine.py]
Stage 8: Composite Score        → compute_composite()        [factor_engine.py]
Stage 9: Value Trap Flags       → apply_value_trap_flags()   [factor_engine.py]
Stage 10: Final Ranking         → rank_stocks()              [factor_engine.py]
Stage 11: Portfolio Construction→ construct_portfolio()       [portfolio_constructor.py]
Stage 12: Excel + Cache Output  → write_excel/parquet        [factor_engine.py]
```

## B. Composite Score Definition

### Formula

```
Composite = percentile_rank(
    sum(factor_weight_i * category_score_i)  /  sum(factor_weight_i for non-NaN categories)
)

category_score_i = sum(metric_weight_j * metric_percentile_j)  /  sum(metric_weight_j for non-NaN metrics)

metric_percentile_j = sector-relative rank (0-100), inverted for 'lower-is-better' metrics
```

### Factor Category Weights

| Category | Weight (%) |
| --- | --- |
| Valuation | 25 |
| Quality | 25 |
| Growth | 15 |
| Momentum | 15 |
| Risk | 10 |
| Revisions | 10 |

### Factor Definitions (all metrics)


#### Valuation

| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ev_ebitda` | Enterprise Value / EBITDA | yfinance: enterpriseValue, EBITDA from financials | Winsorize 1/99 pctile; NaN if EBITDA<=0 or EV<=0 | Sector percentile rank (inverted: lower=better) | 25% | 0% | Lower=Better |
| `fcf_yield` | Free Cash Flow / Enterprise Value | yfinance: operatingCashFlow - abs(capex) / EV | Winsorize 1/99 pctile; NaN if capex missing | Sector percentile rank | 40% | 0% | Higher=Better |
| `earnings_yield` | Trailing EPS / Current Price (inverse P/E) | yfinance: trailingEps / currentPrice | Winsorize 1/99 pctile; can be negative | Sector percentile rank | 20% | 40% | Higher=Better |
| `ev_sales` | Enterprise Value / Total Revenue | yfinance: enterpriseValue / totalRevenue | Winsorize 1/99 pctile; NaN if revenue<=0 | Sector percentile rank (inverted: lower=better) | 15% | 0% | Lower=Better |
| `pb_ratio` | Price-to-Book Ratio (bank-only) | yfinance: priceToBook or currentPrice/bookValue | Winsorize 1/99 pctile; NaN for non-banks | Sector percentile rank (inverted: lower=better) | 0% | 60% | Lower=Better |

#### Quality

| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `roic` | NOPAT / Invested Capital (Eq + Debt - Excess Cash) | yfinance: EBIT*(1-tax) / (equity + debt_bs - excess_cash) | Winsorize 1/99 pctile; NaN if IC<=0 | Sector percentile rank | 30% | 0% | Higher=Better |
| `gross_profit_assets` | Gross Profit / Total Assets | yfinance: grossProfit / totalAssets | Winsorize 1/99 pctile | Sector percentile rank | 25% | 0% | Higher=Better |
| `debt_equity` | Total Debt / Stockholders Equity | yfinance: totalDebt / totalEquity | Winsorize 1/99 pctile; NaN if equity<=0 | Sector percentile rank (inverted: lower=better) | 20% | 0% | Lower=Better |
| `piotroski_f_score` | Piotroski F-Score (0-9 integer, 9 binary signals) | yfinance: computed from financials/BS/CF | NaN if < 6 testable signals | Sector percentile rank | 15% | 15% | Higher=Better |
| `accruals` | (Net Income - OCF) / Total Assets | yfinance: (netIncome - operatingCashFlow) / totalAssets | Winsorize 1/99 pctile | Sector percentile rank (inverted: lower=better) | 10% | 10% | Lower=Better |
| `roe` | Return on Equity (bank-only) | yfinance: returnOnEquity or netIncome/equity | Winsorize 1/99 pctile; NaN for non-banks | Sector percentile rank | 0% | 35% | Higher=Better |
| `roa` | Return on Assets (bank-only) | yfinance: returnOnAssets or netIncome/totalAssets | Winsorize 1/99 pctile; NaN for non-banks | Sector percentile rank | 0% | 25% | Higher=Better |
| `equity_ratio` | Equity / Total Assets (bank-only) | yfinance: totalEquity / totalAssets | Winsorize 1/99 pctile; NaN for non-banks | Sector percentile rank | 0% | 15% | Higher=Better |

#### Growth

| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `forward_eps_growth` | (Forward EPS - Trailing EPS) / max(|Trailing EPS|, $1.00) | yfinance: (forwardEps - trailingEps) / max(|trailingEps|, 1.0) | Clamped to [-75%, +150%]; NaN if trailing EPS near zero | Sector percentile rank | 35% | 35% | Higher=Better |
| `peg_ratio` | P/E / Forward EPS Growth Rate (%) | Computed from price/trailingEps / (forward_eps_growth * 100) | NaN if growth <= 1% | Sector percentile rank (inverted: lower=better) | 20% | 20% | Lower=Better |
| `revenue_growth` | (Revenue - Revenue_Prior) / Revenue_Prior | yfinance: totalRevenue (col 0 vs col 1 from financials) | Winsorize 1/99 pctile; NaN if prior=0 | Sector percentile rank | 30% | 30% | Higher=Better |
| `sustainable_growth` | ROE * Retention Ratio (1 - Payout Ratio) | yfinance: (NI/Eq) * (1 - dividendsPaid/NI) | Winsorize 1/99 pctile; NaN if NI<=0 or equity<=0 | Sector percentile rank | 15% | 15% | Higher=Better |

#### Momentum

| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `return_12_1` | 12-1 Month Return (skip most recent month) | yfinance: (price_1m_ago - price_12m_ago) / price_12m_ago | Winsorize 1/99 pctile; uses calendar-based lookback | Sector percentile rank | 50% | 50% | Higher=Better |
| `return_6m` | 6-1 Month Return (skip most recent month) | yfinance: (price_1m_ago - price_6m_ago) / price_6m_ago | Winsorize 1/99 pctile; uses calendar-based lookback | Sector percentile rank | 50% | 50% | Higher=Better |

#### Risk

| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `volatility` | Annualized daily return std dev (252 trading days) | yfinance: std(daily_log_returns) * sqrt(252) | Winsorize 1/99 pctile; NaN if < 200 trading days | Sector percentile rank (inverted: lower=better) | 60% | 60% | Lower=Better |
| `beta` | Cov(stock, market) / Var(market) using daily returns | yfinance: stock daily returns vs ^GSPC daily returns | Winsorize 1/99 pctile; NaN if < 200 common dates | Sector percentile rank (inverted: lower=better) | 40% | 40% | Lower=Better |

#### Revisions

| Metric | Raw Definition | Data Source | Cleaning | Normalization | Generic Weight | Bank Weight | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `analyst_surprise` | Median earnings surprise across last 4 quarters | yfinance: (epsActual - epsEstimate) / max(|epsEstimate|, $0.10) | NaN if < 2 quarters; denominator floored at $0.10 | Sector percentile rank | 50% | 50% | Higher=Better |
| `price_target_upside` | (Analyst Target Mean Price - Current Price) / Current Price | yfinance: targetMeanPrice / currentPrice - 1 | Clamped [-50%, +100%]; requires >= 3 covering analysts | Sector percentile rank | 50% | 50% | Higher=Better |

## C. Per-Ticker Breakdown


### JNJ (Johnson & Johnson) — Health Care

| Metric | Raw Value | Percentile | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- | --- |
| `ev_ebitda` | 18.4700 | 25.0 | 25 | 6.25 |
| `fcf_yield` | 0.0558 | 100.0 | 40 | 40.00 |
| `earnings_yield` | 0.0109 | 20.0 | 20 | 4.00 |
| `ev_sales` | 2.3000 | 50.0 | 15 | 7.50 |
| `pb_ratio` | NaN | NaN | 0 | — |
| `roic` | 0.1552 | 75.0 | 30 | 22.50 |
| `gross_profit_assets` | 0.6742 | 100.0 | 25 | 25.00 |
| `debt_equity` | 0.2700 | 50.0 | 20 | 10.00 |
| `piotroski_f_score` | 6.0000 | 60.0 | 15 | 9.00 |
| `accruals` | 0.0198 | 0.0 | 10 | 0.00 |
| `roe` | NaN | NaN | 0 | — |
| `roa` | NaN | NaN | 0 | — |
| `equity_ratio` | NaN | NaN | 0 | — |
| `forward_eps_growth` | 0.1386 | 100.0 | 35 | 35.00 |
| `peg_ratio` | 4.1700 | 0.0 | 20 | 0.00 |
| `revenue_growth` | 0.1435 | 80.0 | 30 | 24.00 |
| `sustainable_growth` | 0.1405 | 80.0 | 15 | 12.00 |
| `return_12_1` | -0.0032 | 40.0 | 50 | 20.00 |
| `return_6m` | 0.1715 | 100.0 | 50 | 50.00 |
| `volatility` | 0.2666 | 60.0 | 60 | 36.00 |
| `beta` | 0.6400 | 80.0 | 40 | 32.00 |
| `analyst_surprise` | -0.0236 | 50.0 | 50 | 25.00 |
| `price_target_upside` | 0.1214 | 50.0 | 50 | 25.00 |

**Category Scores:**

| Category | Score | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- |
| Valuation | 57.8 | 25 | 1443.8 |
| Quality | 66.5 | 25 | 1662.5 |
| Growth | 71.0 | 15 | 1065.0 |
| Momentum | 70.0 | 15 | 1050.0 |
| Risk | 68.0 | 10 | 680.0 |
| Revisions | 50.0 | 10 | 500.0 |

**Composite Score: 100.00** | **Rank: 1** (within 5 tickers)
Value Trap Flag: NO

### AAPL (Apple Inc.) — Information Technology

| Metric | Raw Value | Percentile | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- | --- |
| `ev_ebitda` | 22.4400 | 0.0 | 25 | 0.00 |
| `fcf_yield` | 0.0192 | 20.0 | 40 | 8.00 |
| `earnings_yield` | 0.0588 | 100.0 | 20 | 20.00 |
| `ev_sales` | 2.2400 | 75.0 | 15 | 11.25 |
| `pb_ratio` | NaN | NaN | 0 | — |
| `roic` | 0.1884 | 100.0 | 30 | 30.00 |
| `gross_profit_assets` | 0.3480 | 75.0 | 25 | 18.75 |
| `debt_equity` | 0.1600 | 75.0 | 20 | 15.00 |
| `piotroski_f_score` | 7.0000 | 100.0 | 15 | 15.00 |
| `accruals` | -0.0444 | 60.0 | 10 | 6.00 |
| `roe` | NaN | NaN | 0 | — |
| `roa` | NaN | NaN | 0 | — |
| `equity_ratio` | NaN | NaN | 0 | — |
| `forward_eps_growth` | 0.1268 | 80.0 | 35 | 28.00 |
| `peg_ratio` | 2.6500 | 20.0 | 20 | 4.00 |
| `revenue_growth` | 0.1569 | 100.0 | 30 | 30.00 |
| `sustainable_growth` | 0.0955 | 40.0 | 15 | 6.00 |
| `return_12_1` | 0.3257 | 100.0 | 50 | 50.00 |
| `return_6m` | 0.0820 | 60.0 | 50 | 30.00 |
| `volatility` | 0.2852 | 40.0 | 60 | 24.00 |
| `beta` | 1.0100 | 40.0 | 40 | 16.00 |
| `analyst_surprise` | NaN | NaN | 50 | — |
| `price_target_upside` | NaN | NaN | 50 | — |

**Category Scores:**

| Category | Score | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- |
| Valuation | 39.2 | 25 | 981.2 |
| Quality | 84.8 | 25 | 2118.8 |
| Growth | 68.0 | 15 | 1020.0 |
| Momentum | 80.0 | 15 | 1200.0 |
| Risk | 40.0 | 10 | 400.0 |
| Revisions | NaN | 10 | — |

**Composite Score: 80.00** | **Rank: 2** (within 5 tickers)
Value Trap Flag: NO

### JPM (JPMorgan Chase) — Financials
*Bank-like stock: uses bank-specific metric weights*


| Metric | Raw Value | Percentile | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- | --- |
| `ev_ebitda` | NaN | NaN | 0 | — |
| `fcf_yield` | 0.0472 | 80.0 | 0 | — |
| `earnings_yield` | 0.0525 | 80.0 | 40 | 32.00 |
| `ev_sales` | NaN | NaN | 0 | — |
| `pb_ratio` | 0.8400 | 0.0 | 60 | 0.00 |
| `roic` | NaN | NaN | 0 | — |
| `gross_profit_assets` | NaN | NaN | 0 | — |
| `debt_equity` | NaN | NaN | 0 | — |
| `piotroski_f_score` | 6.0000 | 60.0 | 15 | 9.00 |
| `accruals` | -0.0114 | 40.0 | 10 | 4.00 |
| `roe` | 0.1049 | 100.0 | 35 | 35.00 |
| `roa` | 0.0165 | 100.0 | 25 | 25.00 |
| `equity_ratio` | 0.0893 | 100.0 | 15 | 15.00 |
| `forward_eps_growth` | 0.1247 | 60.0 | 35 | 21.00 |
| `peg_ratio` | 0.8800 | 80.0 | 20 | 16.00 |
| `revenue_growth` | 0.0541 | 40.0 | 30 | 12.00 |
| `sustainable_growth` | 0.1011 | 60.0 | 15 | 9.00 |
| `return_12_1` | 0.0880 | 60.0 | 50 | 30.00 |
| `return_6m` | -0.1425 | 20.0 | 50 | 10.00 |
| `volatility` | 0.1632 | 80.0 | 60 | 48.00 |
| `beta` | 0.8400 | 60.0 | 40 | 24.00 |
| `analyst_surprise` | NaN | NaN | 50 | — |
| `price_target_upside` | NaN | NaN | 50 | — |

**Category Scores:**

| Category | Score | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- |
| Valuation | 32.0 | 25 | 800.0 |
| Quality | 88.0 | 25 | 2200.0 |
| Growth | 58.0 | 15 | 870.0 |
| Momentum | 40.0 | 15 | 600.0 |
| Risk | 72.0 | 10 | 720.0 |
| Revisions | NaN | 10 | — |

**Composite Score: 60.00** | **Rank: 3** (within 5 tickers)
Value Trap Flag: NO

### XOM (ExxonMobil) — Energy

| Metric | Raw Value | Percentile | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- | --- |
| `ev_ebitda` | 9.2100 | 75.0 | 25 | 18.75 |
| `fcf_yield` | 0.0427 | 60.0 | 40 | 24.00 |
| `earnings_yield` | 0.0210 | 40.0 | 20 | 8.00 |
| `ev_sales` | 3.5800 | 25.0 | 15 | 3.75 |
| `pb_ratio` | NaN | NaN | 0 | — |
| `roic` | 0.1548 | 50.0 | 30 | 15.00 |
| `gross_profit_assets` | 0.2738 | 50.0 | 25 | 12.50 |
| `debt_equity` | 0.3600 | 25.0 | 20 | 5.00 |
| `piotroski_f_score` | 6.0000 | 60.0 | 15 | 9.00 |
| `accruals` | -0.0028 | 20.0 | 10 | 2.00 |
| `roe` | NaN | NaN | 0 | — |
| `roa` | NaN | NaN | 0 | — |
| `equity_ratio` | NaN | NaN | 0 | — |
| `forward_eps_growth` | 0.0822 | 40.0 | 35 | 14.00 |
| `peg_ratio` | 1.5400 | 40.0 | 20 | 8.00 |
| `revenue_growth` | 0.0013 | 20.0 | 30 | 6.00 |
| `sustainable_growth` | 0.0096 | 20.0 | 15 | 3.00 |
| `return_12_1` | 0.0959 | 80.0 | 50 | 40.00 |
| `return_6m` | 0.0712 | 40.0 | 50 | 20.00 |
| `volatility` | 0.4287 | 0.0 | 60 | 0.00 |
| `beta` | 1.3100 | 0.0 | 40 | 0.00 |
| `analyst_surprise` | 0.1671 | 100.0 | 50 | 50.00 |
| `price_target_upside` | NaN | NaN | 50 | — |

**Category Scores:**

| Category | Score | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- |
| Valuation | 54.5 | 25 | 1362.5 |
| Quality | 43.5 | 25 | 1087.5 |
| Growth | 31.0 | 15 | 465.0 |
| Momentum | 60.0 | 15 | 900.0 |
| Risk | 0.0 | 10 | 0.0 |
| Revisions | 100.0 | 10 | 1000.0 |

**Composite Score: 40.00** | **Rank: 4** (within 5 tickers)
Value Trap Flag: NO

### AMZN (Amazon) — Consumer Discretionary

| Metric | Raw Value | Percentile | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- | --- |
| `ev_ebitda` | 13.4300 | 50.0 | 25 | 12.50 |
| `fcf_yield` | 0.0330 | 40.0 | 40 | 16.00 |
| `earnings_yield` | 0.0473 | 60.0 | 20 | 12.00 |
| `ev_sales` | 5.1800 | 0.0 | 15 | 0.00 |
| `pb_ratio` | NaN | NaN | 0 | — |
| `roic` | 0.1175 | 25.0 | 30 | 7.50 |
| `gross_profit_assets` | 0.2488 | 25.0 | 25 | 6.25 |
| `debt_equity` | 0.4300 | 0.0 | 20 | 0.00 |
| `piotroski_f_score` | 4.0000 | 20.0 | 15 | 3.00 |
| `accruals` | -0.0502 | 80.0 | 10 | 8.00 |
| `roe` | NaN | NaN | 0 | — |
| `roa` | NaN | NaN | 0 | — |
| `equity_ratio` | NaN | NaN | 0 | — |
| `forward_eps_growth` | -0.0024 | 20.0 | 35 | 7.00 |
| `peg_ratio` | 0.9800 | 60.0 | 20 | 12.00 |
| `revenue_growth` | 0.1295 | 60.0 | 30 | 18.00 |
| `sustainable_growth` | 0.1490 | 100.0 | 15 | 15.00 |
| `return_12_1` | -0.0264 | 20.0 | 50 | 10.00 |
| `return_6m` | 0.1129 | 80.0 | 50 | 40.00 |
| `volatility` | 0.3293 | 20.0 | 60 | 12.00 |
| `beta` | 1.2500 | 20.0 | 40 | 8.00 |
| `analyst_surprise` | NaN | NaN | 50 | — |
| `price_target_upside` | 0.2018 | 100.0 | 50 | 50.00 |

**Category Scores:**

| Category | Score | Weight (%) | Weighted Contribution |
| --- | --- | --- | --- |
| Valuation | 40.5 | 25 | 1012.5 |
| Quality | 24.8 | 25 | 618.8 |
| Growth | 52.0 | 15 | 780.0 |
| Momentum | 50.0 | 15 | 750.0 |
| Risk | 20.0 | 10 | 200.0 |
| Revisions | 100.0 | 10 | 1000.0 |

**Composite Score: 20.00** | **Rank: 5** (within 5 tickers)
Value Trap Flag: YES

## D. Rank Summary

| Rank | Ticker | Sector | Composite | Value Trap |
| --- | --- | --- | --- | --- |
| 1 | JNJ | Health Care | 100.00 | NO |
| 2 | AAPL | Information Technology | 80.00 | NO |
| 3 | JPM | Financials | 60.00 | NO |
| 4 | XOM | Energy | 40.00 | NO |
| 5 | AMZN | Consumer Discretionary | 20.00 | YES |

## E. Integrity Checks

### 1. Look-Ahead Bias

- **Status**: PARTIALLY MITIGATED
- yfinance `.info` returns CURRENT market data (price, market cap, EV)
- Financial statements (`.financials`, `.balance_sheet`, `.cashflow`) return the most recent ANNUAL filing
- Forward EPS comes from current analyst consensus, which is point-in-time correct for screening (not backtesting)
- **Risk**: If this screener were used for backtesting, financial statement data would be stale. For live screening, it is appropriate.

### 2. Period Alignment

- **Status**: PARTIAL CONCERN
- Market data (price, returns, beta): TTM / trailing 1 year
- Financial statements: most recent annual filing (could be 1-12 months old)
- Forward EPS: consensus estimate (forward-looking)
- **Risk**: Mixing TTM prices with annual financials creates a temporal mismatch. For screening (not backtesting) this is standard practice.

### 3. Currency/Units Consistency

- **Status**: OK (S&P 500 is USD-denominated)
- All financial data from yfinance is in the stock's reporting currency
- S&P 500 constituents all report in USD

### 4. Survivorship Bias

- **Status**: PRESENT (for backtesting), ACCEPTABLE (for live screening)
- Universe = current S&P 500 constituents from Wikipedia
- Companies removed from the index are excluded
- For live screening purposes, this is correct (you want current constituents)

### 5. Fallback Path Consistency

- **Status**: PARTIAL CONCERN
- Sample data path (`_generate_sample_data`) uses DIFFERENT distributions than live data
- Sample data has bank-like metrics only for Financials sector, not industry-granular
- Scoring pipeline is identical for both paths (winsorize, percentile, composite)