# FORENSIC AUDIT REPORT — Multi-Factor Stock Screener
**Date:** 2026-02-17
**Auditor:** Claude (automated code forensics)
**Scope:** All source code, config, and outputs in Screener-1/
**Status:** READ-ONLY — no code was modified during this audit

---

## STEP 1 — REPO ORIENTATION & CALL GRAPH

### 1.1 Entry Points

| Entry Point | File | Purpose |
|---|---|---|
| **Primary** | `run_screener.py` | Full pipeline with CLI flags (`--refresh`, `--tickers`, `--no-portfolio`) |
| **Legacy** | `run_pipeline.py` | Runs `factor_engine.py` then `portfolio_constructor.py` as subprocesses |
| **Standalone** | `factor_engine.py` (has `if __name__`) | Factor scoring only |
| **Backtest** | `backtest.py` | Historical backtest (standalone or called externally) |
| **Validation** | `validate_data_accuracy.py` | Data accuracy checks |

### 1.2 Pipeline Call Graph

```
run_screener.py::main()
│
├─ 1. load_config_safe()                         # Read config.yaml
│
├─ 2. run_factor_engine(cfg, args)
│   │
│   ├─ get_sp500_tickers(cfg)                     # Universe: Wikipedia → sp500_tickers.json fallback
│   │   └─ pd.read_html("wikipedia.org/...")      # DATA SOURCE: Wikipedia HTML scrape
│   │   └─ json.load("sp500_tickers.json")        # DATA SOURCE: Local JSON fallback
│   │
│   ├─ _find_latest_cache("factor_scores")        # Check cache/factor_scores_YYYYMMDD.parquet
│   │   └─ [if HOT: return cached df, skip fetch]
│   │
│   ├─ fetch_single_ticker(tickers[0])            # Network probe (max_retries=1)
│   │   └─ _fetch_single_ticker_inner()           # DATA SOURCE: yfinance API
│   │       ├─ yf.Ticker(ticker).info             #   → ~12 info fields
│   │       ├─ t.financials                       #   → income statement (current + prior year)
│   │       ├─ t.balance_sheet                    #   → balance sheet (current + prior year)
│   │       ├─ t.cashflow                         #   → cash flow statement
│   │       ├─ t.history(period="13mo")           #   → daily close prices
│   │       └─ t.earnings_history                 #   → last 4 quarterly EPS surprises
│   │
│   ├─ [if offline: _generate_sample_data()]      # Synthetic sector-realistic data (seed=42)
│   │
│   ├─ fetch_market_returns()                     # DATA SOURCE: yf.Ticker("^GSPC").history("1y")
│   │
│   ├─ fetch_all_tickers(tickers)                 # Batch fetch: 50/batch, 5 threads, 2s delay
│   │   └─ ThreadPoolExecutor → fetch_single_ticker() × N
│   │
│   ├─ compute_metrics(raw, market_returns)       # CORE: 17 metrics from raw data
│   │   ├─ Valuation: ev_ebitda, fcf_yield, earnings_yield, ev_sales
│   │   ├─ Quality:   roic, gross_profit_assets, debt_equity, piotroski_f_score, accruals
│   │   ├─ Growth:    forward_eps_growth, peg_ratio, revenue_growth, sustainable_growth
│   │   ├─ Momentum:  return_12_1, return_6m
│   │   ├─ Risk:      volatility, beta
│   │   └─ Revisions: analyst_surprise, [eps_revision_ratio=NaN], [eps_estimate_change=NaN]
│   │
│   ├─ _run_data_quality_checks(df)               # Market cap outliers, neg EV, revenue gaps
│   │
│   ├─ [Revisions auto-disable if <30% coverage]  # Redistributes weights proportionally
│   │
│   ├─ winsorize_metrics(df, 0.01, 0.01)          # scipy mstats.winsorize at 1st/99th pctile
│   │
│   ├─ compute_sector_percentiles(df)             # Rank within sector → 0-100 percentile
│   │   └─ For each Sector group, for each metric:
│   │       rank(pct=True) * 100, invert if "lower=better"
│   │       NaN / small-sector → 50.0
│   │
│   ├─ compute_category_scores(df, cfg)           # Weighted avg of metric percentiles per category
│   │   └─ category_score = Σ(metric_pct × weight) / total_weight
│   │
│   ├─ compute_composite(df, cfg)                 # Weighted avg of category scores → min-max 0-100
│   │   └─ Composite = Σ(cat_score × cat_weight) / total_weight
│   │   └─ Min-Max scale to [0, 100] (global or per-sector)
│   │
│   ├─ apply_value_trap_flags(df, cfg)            # OR(quality<30p, momentum<30p, revisions<30p)
│   │
│   ├─ rank_stocks(df)                            # Rank by Composite desc, method="min"
│   │
│   └─ write_scores_parquet(df)                   # Save cache/factor_scores_YYYYMMDD.parquet
│
├─ 3. run_portfolio_construction(df, cfg)
│   ├─ construct_portfolio(df, cfg)               # Top 25, sector-capped, equal/risk-parity weights
│   │   ├─ Filter: Composite ≥ median
│   │   ├─ Filter: metric coverage ≥ 60%
│   │   ├─ Sector cap: max 8 per sector
│   │   ├─ Equal weight: 100/N per stock
│   │   ├─ Risk-parity: 1/vol_i normalized
│   │   └─ Position cap: 5% max, iterative redistribution
│   └─ compute_portfolio_stats(port, cfg)         # Weighted avg composite, beta, div yield, etc.
│
├─ 4. write_excel_safe(df, port, stats, cfg)
│   └─ write_full_excel()                         # 3-sheet Excel: FactorScores, Dashboard, Portfolio
│
├─ 5. flush_dq_log()                             # Write validation/data_quality_log.csv
│
└─ 6. print_full_summary()                       # Console diagnostics
```

### 1.3 Data Sources Inventory

| # | Source | Library/Method | File:Line | Fields Used | Frequency |
|---|--------|---------------|-----------|-------------|-----------|
| 1 | Wikipedia S&P 500 list | `pd.read_html()` | `factor_engine.py:64` | Symbol, Security, GICS Sector | 1× per run |
| 2 | Local ticker JSON | `json.load()` | `factor_engine.py:78` | Ticker, Company, Sector | Fallback only |
| 3 | yfinance `.info` | `yf.Ticker().info` | `factor_engine.py:214` | marketCap, enterpriseValue, trailingEps, forwardEps, currentPrice, totalDebt, totalCash, sharesOutstanding, sector, shortName, earningsGrowth, dividendRate | Per ticker |
| 4 | yfinance `.financials` | `t.financials` | `factor_engine.py:232` | Total Revenue (curr+prior), Gross Profit (curr+prior), EBIT, EBITDA, Net Income (curr+prior), Tax Provision, Pretax Income, Cost of Revenue | Per ticker |
| 5 | yfinance `.balance_sheet` | `t.balance_sheet` | `factor_engine.py:234` | Total Assets (curr+prior), Stockholders Equity, Total Debt, Long Term Debt (curr+prior), Current Liabilities (curr+prior), Current Assets (curr+prior), Cash, Shares Outstanding (curr+prior) | Per ticker |
| 6 | yfinance `.cashflow` | `t.cashflow` | `factor_engine.py:236` | Operating Cash Flow, Capital Expenditure, Dividends Paid | Per ticker |
| 7 | yfinance `.history` | `t.history("13mo")` | `factor_engine.py:286` | Daily Close prices (13 months) | Per ticker |
| 8 | yfinance `.earnings_history` | `t.earnings_history` | `factor_engine.py:304` | epsActual, epsEstimate (last 4 quarters) | Per ticker |
| 9 | yfinance S&P 500 index | `yf.Ticker("^GSPC").history("1y")` | `factor_engine.py:350` | Daily Close prices (1 year) | 1× per run |
| 10 | yfinance batch download | `yf.download()` | `backtest.py:80` | Monthly adjusted Close (2020-01-01 to present) | 1× per backtest |

---

## STEP 2 — HEDGE FUND DEFENSIBILITY AUDIT

### A) Metric Definitions — Detailed Analysis

#### Metric 1: EV/EBITDA
- **Formula:** `EV / EBITDA` where EV=yfinance enterpriseValue (fallback: MC + Debt - Cash), EBITDA=yfinance EBITDA from income statement
- **Code:** [factor_engine.py:506](factor_engine.py#L506)
- **Required raw fields:** `enterpriseValue`, `marketCap`, `totalDebt`, `totalCash`, `ebitda`
- **Time alignment:** Uses most recent annual financial statement (col=0 in `_stmt_val`). yfinance returns TTM for `.info` fields but annual for `.financials`. **RISK: EV (current market data) divided by EBITDA (last annual report) creates a time mismatch.**
- **Missing/outliers:** NaN if any component missing, EBITDA ≤ 0, or EV ≤ 0
- **Negative denominator handling:** Explicit check `ebitda > 0` — correct
- **Look-ahead bias:** EV uses current market cap (real-time) + latest reported debt/cash. For live screening this is acceptable. For backtesting, this is NOT point-in-time safe — yfinance reports the *latest* financials, not as-of-date financials.

#### Metric 2: FCF Yield
- **Formula:** `(OCF - |CapEx|) / EV` — [factor_engine.py:508-513](factor_engine.py#L508-L513)
- **Required raw fields:** `operatingCashFlow`, `capex`, `enterpriseValue`
- **Capex sign handling:** If capex < 0, takes `abs(capex)`. If capex > 0, subtracts as-is. If capex is NaN, uses OCF alone (FCF = OCF). **ISSUE: Using OCF as FCF when capex is missing overstates FCF for capital-intensive firms.**
- **Time alignment:** OCF and CapEx from annual cash flow statement; EV from current market. Same time mismatch as EV/EBITDA.
- **Look-ahead bias:** Same as EV/EBITDA.

#### Metric 3: Earnings Yield
- **Formula:** `trailingEps / currentPrice` — [factor_engine.py:516-518](factor_engine.py#L516-L518)
- **Required raw fields:** `trailingEps`, `currentPrice`
- **Time alignment:** trailingEps is TTM from yfinance `.info`. currentPrice is real-time. Reasonably aligned for live screening.
- **Missing/outliers:** NaN if price ≤ 0 or EPS missing
- **ISSUE:** Earnings yield with negative EPS is set to NaN (not negative). This means loss-making companies silently receive the median (50th pctile) rank, which is charitable. A pro system would flag these separately.

#### Metric 4: EV/Sales
- **Formula:** `EV / Total Revenue` — [factor_engine.py:521](factor_engine.py#L521)
- **Required raw fields:** `enterpriseValue`, `totalRevenue`
- **Time alignment:** Same mismatch as EV/EBITDA (current EV, annual revenue)
- **Missing/outliers:** NaN if revenue ≤ 0 or EV ≤ 0

#### Metric 5: ROIC
- **Formula:** `NOPAT / Invested Capital` where `NOPAT = EBIT × (1 - tax_rate)`, `IC = Equity + Debt - Cash` — [factor_engine.py:529-543](factor_engine.py#L529-L543)
- **Tax rate:** Computed as `Income Tax Expense / Pretax Income`, clamped to [0%, 50%]. Defaults to 21% if pretax income ≤ 0 or missing. **ISSUE: Using statutory 21% for companies with negative pretax income is methodologically questionable — NOPAT for a loss-making company should arguably be negative.**
- **Invested Capital:** `Equity + Total Debt - Cash`. **ISSUE: Uses `totalDebt` from `.info` (which includes short-term debt) and `totalCash` from `.info` (which includes short-term investments). This is internally consistent but differs from academic definitions that use only long-term capital.**
- **Look-ahead bias:** Same as other fundamental metrics.

#### Metric 6: Gross Profit / Assets
- **Formula:** `Gross Profit / Total Assets` — [factor_engine.py:547](factor_engine.py#L547)
- **Based on:** Novy-Marx (2013) profitability factor
- **Time alignment:** Both from annual financial statements — internally consistent
- **Missing/outliers:** NaN if total assets ≤ 0

#### Metric 7: Debt/Equity
- **Formula:** `Total Debt / Total Equity` — [factor_engine.py:550-553](factor_engine.py#L550-L553)
- **Special case:** If equity ≤ 0, sets to 999.0 (distress signal). **DEFENSIBLE:** This is a reasonable sentinel that ensures high-leverage scores for distressed firms.
- **Time alignment:** Both from balance sheet

#### Metric 8: Piotroski F-Score
- **Formula:** 9 binary signals (standard Piotroski 2000), normalized — [factor_engine.py:556-591](factor_engine.py#L556-L591)
- **ISSUE: Non-standard normalization.** Original Piotroski F-Score is an integer 0-9. This code computes `(signals_passed / signals_testable) × 9`, which is a *proportional* score. Example: if only 5 of 9 signals are testable and 3 pass, the score is `(3/5) × 9 = 5.4`. This conflates data availability with quality. A company with 4 testable signals (all passing) gets 9.0, same as a company with 9 testable signals (all passing). **This rewards data sparsity.**
- **Minimum threshold:** Requires ≥4 testable signals, otherwise NaN. Reasonable.
- **Time alignment:** Uses current and prior year financials — correct for Piotroski methodology.

#### Metric 9: Accruals
- **Formula:** `(Net Income - Operating Cash Flow) / Total Assets` — [factor_engine.py:594](factor_engine.py#L594)
- **Interpretation:** Lower (more negative) = higher quality earnings
- **Direction:** Correctly marked as `False` in METRIC_DIR (lower is better)
- **Time alignment:** All from annual statements — consistent

#### Metric 10: Forward EPS Growth
- **Formula:** `(Forward EPS - Trailing EPS) / |Trailing EPS|` — [factor_engine.py:603](factor_engine.py#L603)
- **ISSUE:** Uses yfinance `forwardEps` which is a consensus estimate. **This introduces analyst estimate data that is NOT point-in-time safe for backtesting.** For live screening, it's acceptable.
- **Small denominator guard:** `abs(trail) > 0.01` — prevents division by near-zero EPS

#### Metric 11: PEG Ratio
- **Formula:** `(Price / Trailing EPS) / (earningsGrowth × 100)` — [factor_engine.py:606-609](factor_engine.py#L606-L609)
- **ISSUE:** `earningsGrowth` from yfinance `.info` is poorly documented — it may be trailing or forward, and its definition varies by data provider. **This is a black-box input.**
- **Denominator guard:** `_eg > 0.01` — only computes PEG for positive growth companies

#### Metric 12: Revenue Growth
- **Formula:** `(Revenue_current - Revenue_prior) / Revenue_prior` — [factor_engine.py:612](factor_engine.py#L612)
- **"Current" vs "Prior":** Uses `_stmt_val(fins, "Total Revenue", 0)` and `_stmt_val(fins, "Total Revenue", 1)` — columns 0 and 1 of the financials DataFrame. These are the two most recent annual periods.
- **ISSUE:** yfinance `.financials` columns are sorted by date descending, so col=0 is the most recent annual report and col=1 is the prior year. However, the column dates may not align perfectly with fiscal year ends for all companies.

#### Metric 13: Sustainable Growth
- **Formula:** `ROE × (1 - Payout Ratio)` where `ROE = NI/Equity`, `Payout = Dividends/NI` — [factor_engine.py:615-631](factor_engine.py#L615-L631)
- **Dividend estimation fallback:** If cash flow dividends unavailable, estimates from `dividendRate × sharesOutstanding`. If both unavailable, assumes 0 dividends (full retention). **ISSUE: Assuming full retention biases growth estimates upward for companies that do pay dividends but lack data.**
- **Guard:** Only computed if NI > 0 and Equity > 0

#### Metric 14: 12-1 Month Return (Momentum)
- **Formula:** `(Price_1m_ago - Price_12m_ago) / Price_12m_ago` — [factor_engine.py:640](factor_engine.py#L640)
- **Price indexing:** Uses `closes.iloc[-22]` for 1 month ago and `closes.iloc[-252]` for 12 months ago. **ISSUE: Uses fixed index offsets, not calendar dates. Trading day counts can vary due to holidays, and 13 months of data may not always contain 252+ observations.**
- **Momentum skip:** Correctly excludes the most recent month (Jegadeesh & Titman 1993 convention)
- **ISSUE:** Prices are `auto_adjust=True` from yfinance, which adjusts for splits and dividends. This is correct for total return momentum.

#### Metric 15: 6-Month Return
- **Formula:** `(Price_now - Price_6m_ago) / Price_6m_ago` — [factor_engine.py:645](factor_engine.py#L645)
- **ISSUE:** Unlike 12-1 momentum, this INCLUDES the most recent month. This is inconsistent — 12-1 momentum skips the last month (to avoid short-term reversal), but 6M does not. **A pro system would use 6-1 momentum for consistency.**

#### Metric 16: Volatility
- **Formula:** `std(daily_log_returns) × √252` — [factor_engine.py:294](factor_engine.py#L294)
- **Minimum observations:** 200 daily returns required
- **ISSUE:** Uses log returns for volatility calculation (line 289: `np.log(closes / closes.shift(1))`) but this is acceptable — log returns are standard for annualized volatility.

#### Metric 17: Beta
- **Formula:** `Cov(stock_returns, market_returns) / Var(market_returns)` — [factor_engine.py:655-684](factor_engine.py#L655-L684)
- **Date alignment:** Correctly aligns stock and market returns by date (dict-based matching)
- **Minimum observations:** 200 common trading days
- **Returns type:** Log returns for both stock and market — consistent
- **ISSUE:** Uses `ddof=1` for market variance but `np.cov` also uses ddof=1 by default — internally consistent. However, some implementations use `ddof=0`. Minor difference with 200+ observations.

#### Metrics 18-20: Analyst Revisions
- **analyst_surprise:** `mean((actual - estimate) / |estimate|)` over last 4 quarters — [factor_engine.py:303-313](factor_engine.py#L303-L313)
- **eps_revision_ratio:** Always NaN (not implemented) — [factor_engine.py:691](factor_engine.py#L691)
- **eps_estimate_change:** Always NaN (not implemented) — [factor_engine.py:692](factor_engine.py#L692)
- **CRITICAL: 2 of 3 revision metrics are permanently NaN.** The revisions category receives 75% of its weight from metrics that never have data. Auto-disable at <30% coverage handles this, but if analyst_surprise alone exceeds 30% coverage, the category activates with only 1 of 3 metrics populated. The other two default to 50th percentile, diluting the signal.

---

### B) Data Integrity Risks

#### B1. Survivorship Bias
- **Severity: HIGH**
- **Location:** `get_sp500_tickers()` at [factor_engine.py:54-96](factor_engine.py#L54-L96)
- **Issue:** The universe is the *current* S&P 500 constituent list (scraped from Wikipedia or loaded from a static JSON). For live screening, this is acceptable. **For backtesting, this is a textbook survivorship bias** — we only score companies that survived to today's index, excluding companies that were removed (bankrupt, acquired, delisted). The backtest module (`backtest.py`) uses the same current-date universe for all historical months.
- **Impact:** Backtest results will overstate returns because failed companies are excluded from the historical universe.

#### B2. Look-Ahead Bias
- **Severity: HIGH (backtest) / LOW (live screening)**
- **Location:** `_fetch_single_ticker_inner()` at [factor_engine.py:208-319](factor_engine.py#L208-L319)
- **Issue:** yfinance always returns the *latest available* financial data. For live screening, this is the intended behavior. For backtesting, the `simulate_monthly_scores()` function in `backtest.py` reuses the same current fundamentals to generate "historical" scores — it does NOT fetch point-in-time data for each historical date.
- **Impact:** Backtest uses future information. Results are unreliable for strategy validation.

#### B3. Restatement Issues
- **Severity: MEDIUM**
- **Issue:** yfinance financial statements reflect the *most recently filed* version, including any restatements. There is no mechanism to detect or account for post-filing revisions.
- **Impact:** A company that restated earnings downward would show the corrected (lower) figure, not the originally reported figure that would have been available at decision time.

#### B4. Corporate Actions
- **Severity: LOW (prices) / MEDIUM (fundamentals)**
- **Prices:** `auto_adjust=True` is used, which adjusts for splits and dividends. Correct.
- **Fundamentals:** No split-adjustment for per-share metrics like EPS. yfinance should handle this, but there is no verification.
- **Ticker changes:** The `"."` to `"-"` replacement at [factor_engine.py:68](factor_engine.py#L68) handles BRK.B → BRK-B, but other ticker changes (mergers, spin-offs) are not addressed.

#### B5. Stale Data / Timestamp Ambiguity
- **Severity: HIGH**
- **Issue:** There is **no record of when each data point was fetched**, no "as-of" timestamp on individual fields, and no check for data staleness at the field level. A ticker might return stale `trailingEps` from a filing 11 months ago alongside a `currentPrice` from today.
- **Location:** `_fetch_single_ticker_inner()` — no timestamp capture
- **Impact:** Users cannot verify data freshness. Two runs on the same day could produce different results if yfinance updates mid-run, and there would be no way to detect this.

#### B6. Universe Timing
- **Severity: MEDIUM**
- **Issue:** The Wikipedia S&P 500 list reflects *today's* constituents. If a company was just added/removed, the universe changes between runs. The `sp500_tickers.json` fallback is a static snapshot with no documented date.
- **Location:** [factor_engine.py:64](factor_engine.py#L64)

---

### C) Statistical Validity

#### C1. Scaling/Normalization
- **Method:** Sector-relative percentile ranking → weighted category scores → min-max to [0,100]
- **Winsorization:** 1st/99th percentile, applied *before* ranking. **Correct order.**
- **ISSUE: Double normalization.** The pipeline applies: (1) winsorization, (2) sector-relative percentile ranking, (3) weighted average category scores, (4) min-max scaling to [0,100]. Steps 2 and 4 are both rank-transforming operations, which is redundant but not harmful — percentile ranking already produces a uniform distribution, so the final min-max just rescales.
- **ISSUE: Percentile ranking destroys magnitude information.** A stock with EV/EBITDA of 5× and one with 7× in the same sector may receive similar percentile ranks if the sector is tightly clustered, while a stock with 50× would rank similarly to one with 200× if both are at the top. This is an intentional design choice (rank-based composites are robust to outliers) but loses information about *how much* better one stock is vs. another.

#### C2. Factor Correlation / Double-Counting
- **ISSUE: Moderate correlation between metrics within categories.**
  - **Valuation:** EV/EBITDA and EV/Sales both use EV in the numerator — highly correlated for similar margin companies. Earnings yield and FCF yield are also correlated. This means the "4 valuation metrics" effectively provide ~2 independent signals.
  - **Quality:** ROIC and Gross Profit/Assets are correlated (both measure profitability). Accruals and Piotroski F-Score partially overlap (Piotroski signal 4 checks OCF > NI, which is the accruals concept).
  - **Growth:** Forward EPS growth and PEG ratio share the P/E component; revenue growth and sustainable growth can diverge meaningfully.
  - **Momentum:** 12-1M and 6M returns overlap significantly (~6 months of shared return window).
  - **Risk:** Volatility and beta are correlated (~0.6-0.8 empirically).
- **Impact:** Effective number of independent factors is likely 8-10 rather than 17. The equal-within-category weighting partially mitigates this.

#### C3. Regime Sensitivity
- **ISSUE:** Factor weights are static (config.yaml). In regime changes (e.g., from growth to value rotation), the screener has no adaptive mechanism. Momentum and value factors tend to have negative correlation in regime shifts.
- **Mitigation needed:** Document that weights are static; consider regime-conditional weighting in a future version.

#### C4. Overfitting Risk
- **Risk level: LOW for live screening, HIGH for backtest-based weight tuning**
- The current weights (25/25/15/15/10/10) appear to be heuristic, not optimized. This is actually a strength — backtested-optimized weights would be overfit.
- **ISSUE:** The metric weights within categories (e.g., FCF Yield = 40%, EV/EBITDA = 25%) appear chosen judgmentally. No documentation of *why* these specific weights were chosen.

---

### D) Implementation Robustness

#### D1. Error Handling & Retries
- **Strength:** Exponential backoff retry (1s/2s/4s) with non-retryable error detection — [factor_engine.py:181-205](factor_engine.py#L181-L205). Good.
- **Strength:** Batch fetching with 2s inter-batch delay to avoid rate limiting — [factor_engine.py:322-339](factor_engine.py#L322-L339). Good.
- **ISSUE:** Bare `except Exception` in multiple places (lines 232-236, 299-300, 312-313) silently swallows errors. A yfinance schema change or API rate-limit error would produce NaN silently.
- **ISSUE:** `_stmt_val()` has a final `except Exception: return default` that masks all errors in financial statement parsing. A column name change in yfinance would silently break all fundamental metrics with no logged warning.

#### D2. Caching Strategy
- **Mechanism:** Date-stamped Parquet files in `cache/` directory
- **Freshness:** 7-day cache for factor scores, 1-day for prices
- **ISSUE: Cache is per-day, not per-configuration.** If you change `config.yaml` weights and re-run within the cache window, you get stale scores with the old weights. The cache does not hash the config.
- **ISSUE: No cache invalidation on code changes.** If a metric formula changes, cached scores are stale.
- **ISSUE:** Cache date is based on filename parsing (`factor_scores_YYYYMMDD.parquet`), not file modification time. Robust to filesystem timezone issues.

#### D3. Performance
- **Fetch time:** ~500 tickers × ~2s each (with threading) ≈ 3-5 minutes for a full S&P 500 run
- **Scoring time:** Sub-second (vectorized pandas operations)
- **Bottleneck:** yfinance API calls — each ticker requires 5-6 separate API calls (info, financials, balance_sheet, cashflow, history, earnings_history)
- **ISSUE: No per-ticker timing.** Cannot identify slow tickers or timeout issues.

#### D4. Logging Quality
- **Data Quality Log:** `validation/data_quality_log.csv` — captures fetch failures, missing metrics, outliers, market cap anomalies, negative EV, revenue discontinuities. **This is good.**
- **MISSING:** No structured JSON logging. Console output is unstructured `print()` statements.
- **MISSING:** No per-ticker fetch timing.
- **MISSING:** No run-level metadata (run_id, start time, config hash, git commit).
- **MISSING:** No intermediate data snapshots (raw fetched data, post-winsorize data, post-percentile data).

#### D5. Thread Safety
- **ISSUE:** `_fetch_single_ticker_inner()` imports `yfinance` inside the function body (line 212). This is done per-call, which is safe but wasteful. The `ThreadPoolExecutor` with 5 workers should be fine for yfinance's thread safety model.

---

### E) Output Auditability

#### E1. Reproducibility
- **Can we reproduce a historical run?** **NO.**
  - No run_id is generated or stored
  - No config snapshot is saved with each run
  - No data snapshot is saved (raw fetched data is discarded after scoring)
  - yfinance data changes over time (earnings restatements, price corrections)
  - Wikipedia S&P 500 list changes with index rebalances
  - The only artifact is the final `factor_scores_YYYYMMDD.parquet` — intermediate stages are lost

#### E2. Versioning
- **Config version:** `config.yaml` has a comment "Version 2.0 | Last Updated: 2026-02-12" but this is a manual string, not enforced
- **Code version:** No git tags, no `__version__` variable, no requirements.txt pinning (uses `>=` not `==`)
- **Data version:** No version tracking for `sp500_tickers.json`

#### E3. Intermediate Tables
- **Saved:** Final factor scores (Parquet), Excel output, data quality log
- **NOT saved:** Raw fetched data per ticker, post-winsorize table, sector percentile table, category scores before composite
- **Impact:** Cannot debug "why did stock X get this score?" without re-running

#### E4. Parameter Recording
- **NOT recorded:** Which tickers were fetched, which failed, which were excluded, what the actual factor weights were (after revisions auto-disable), what the winsorization boundaries were, how many sector groups existed
- **Impact:** Cannot answer "what happened during this run?" from saved artifacts alone

---

## STEP 3 — GAP LIST (What Makes This "Non-Pro" Today?)

### CRITICAL Severity

| # | Issue | File / Function | Why It Breaks Defensibility | Minimal Fix | Gold Standard |
|---|-------|----------------|---------------------------|-------------|---------------|
| 1 | **No run reproducibility** | Entire pipeline | Cannot re-create any historical run. A PM asking "why was stock X ranked #5 last month?" gets no answer. | Save run_id + config snapshot + raw data dump per run | Full data lineage: raw → cleaned → scored → ranked with immutable storage |
| 2 | **Backtest survivorship bias** | `backtest.py` / `get_sp500_tickers()` | Backtest results are unreliable — only tests stocks that survived to today | Document this limitation prominently; add disclaimer to backtest outputs | Use point-in-time index membership from a paid data provider (e.g., Compustat, S&P IQ) |
| 3 | **Backtest look-ahead bias** | `backtest.py` / `simulate_monthly_scores()` | Historical scores use current fundamentals, not as-of-date data. Backtest is scientifically invalid. | Document limitation; label backtest as "illustrative only" | Point-in-time fundamental data from institutional provider |
| 4 | **2 of 3 revisions metrics permanently NaN** | `factor_engine.py:691-692` | The "revisions" factor category is functionally broken — only analyst_surprise works. If it auto-enables, 75% of weight falls on NaN→50th percentile padding. | Set `eps_revision_ratio` and `eps_estimate_change` weights to 0 in config; restructure revisions to be analyst_surprise only | Integrate IBES or similar for consensus estimate revisions |

### HIGH Severity

| # | Issue | File / Function | Why It Breaks Defensibility | Minimal Fix | Gold Standard |
|---|-------|----------------|---------------------------|-------------|---------------|
| 5 | **No data staleness detection** | `_fetch_single_ticker_inner()` | Cannot distinguish fresh vs. 11-month-old EPS. Mixing current prices with stale fundamentals produces misleading valuations. | Log `t.info` timestamps where available; add staleness check for quarterly reporting dates | Require fundamental data recency < 100 days; flag stale tickers |
| 6 | **Piotroski F-Score normalization rewards sparsity** | `factor_engine.py:591` | A stock with 4/4 signals passing gets 9.0, same as 9/9 passing. Penalizes data-rich companies. | Use raw integer score (0-9) based on available signals, don't normalize | Raw Piotroski score with minimum 7 testable signals required |
| 7 | **No intermediate data artifacts** | Pipeline-wide | Cannot debug individual stock scores. No audit trail from raw data to final rank. | Save raw_data.parquet, winsorized.parquet, percentiles.parquet per run | Immutable data lineage store with column-level provenance |
| 8 | **Cache ignores config changes** | `_find_latest_cache()` | Changing weights in config.yaml returns stale scores from pre-change run | Include config hash in cache filename | Content-addressable cache keyed on (config_hash, code_hash, data_date) |
| 9 | **Silent error swallowing** | `_stmt_val()`, lines 232-236, 299-300, 312 | Bare `except Exception` masks data provider changes, schema breaks, and rate limits. A field rename in yfinance would silently zero out metrics. | Log warnings for each caught exception with ticker + field name | Typed schema validation with explicit expected-field assertions |
| 10 | **6M momentum includes recent month** | `factor_engine.py:645` | Inconsistent with 12-1M momentum convention. Short-term reversal effect contaminates signal. | Change to 6-1M momentum (exclude last month) | Use consistent skip-month convention across all momentum windows |

### MEDIUM Severity

| # | Issue | File / Function | Why It Breaks Defensibility | Minimal Fix | Gold Standard |
|---|-------|----------------|---------------------------|-------------|---------------|
| 11 | **No structured logging** | Entire pipeline | Console `print()` statements are not parseable, not searchable, not archivable | Add Python `logging` module with JSON formatter | Structured logging with correlation IDs, shipped to central log store |
| 12 | **No deterministic config file** | `config.yaml` | No config hash, no `run_id`, no way to tie outputs to inputs | Generate and log `run_id = uuid4()` at pipeline start; save config alongside outputs | Full provenance chain: code commit + config + data snapshot → output |
| 13 | **EV time mismatch** | `compute_metrics()` | Current market cap + annual balance sheet debt/cash = mismatch in EV computation timing | Document the limitation; note it's standard for screening (not backtest) | Use quarterly balance sheet data with TTM income statement |
| 14 | **Missing earnings yield for loss-making firms** | `factor_engine.py:518` | Companies with negative EPS get NaN → 50th percentile, hiding their unprofitability | Compute negative earnings yield and score normally (lower rank) | Separate treatment: exclude loss-makers or use alternative valuation metric |
| 15 | **PEG ratio black-box input** | `factor_engine.py:608` | `earningsGrowth` from yfinance `.info` is undocumented — may be trailing or forward | Document that this is a yfinance-provided figure; add warning | Compute growth rate explicitly from historical EPS |
| 16 | **Factor correlation / double-counting** | `compute_category_scores()` | Within-category metrics are correlated (EV/EBITDA ~ EV/Sales, volatility ~ beta). Effective degrees of freedom < 17 | Document in spec; compute and report correlation matrix | Orthogonalize factors or use PCA-weighted composites |
| 17 | **Dividend estimation fallback** | `factor_engine.py:618-625` | Assumes full retention when dividends unknown, biasing sustainable growth upward | Set sustainable_growth to NaN when dividends unknown | Cross-reference dividend data from multiple sources |
| 18 | **Fixed-index price lookback** | `factor_engine.py:291-293` | `closes.iloc[-22]` assumes 22 trading days = 1 month. Holidays and shortened histories can misalign. | Use calendar-date based lookback with `pd.DateOffset` | Proper business-day calendar alignment |
| 19 | **No dependency pinning** | `requirements.txt` | Uses `>=` not `==`. A yfinance update could break field names silently. | Pin exact versions in `requirements.txt` | Use a lockfile (poetry.lock / pip-compile) with hash verification |

### LOW Severity

| # | Issue | File / Function | Why It Breaks Defensibility | Minimal Fix | Gold Standard |
|---|-------|----------------|---------------------------|-------------|---------------|
| 20 | **Hardcoded S&P sector weights** | `portfolio_constructor.py:35-47` | Sector benchmark weights are approximate and not dated | Add a comment with the source date and reference | Dynamic sector weights from index provider API |
| 21 | **Value trap flag uses OR logic** | `factor_engine.py:867` | Any single floor breach triggers the flag. A stock with excellent quality but poor momentum gets flagged. | Document the OR logic clearly; consider AND (intersection) | Configurable AND/OR with per-layer severity |
| 22 | **No git integration** | Entire repo | Not a git repo — no version history, no ability to track code changes | `git init` + initial commit | CI/CD with automated testing on each commit |
| 23 | **`run_pipeline.py` is redundant** | `run_pipeline.py` | Runs `factor_engine.py` and `portfolio_constructor.py` as subprocesses — duplicates `run_screener.py` | Remove or deprecate | Single canonical entry point |
| 24 | **`n_ull` stub file** | `n_ull` | 52-byte orphan file in repo root | Delete it | Repo hygiene enforcement via CI |

---

## ASSUMPTIONS MADE DURING THIS AUDIT

1. **yfinance data freshness:** We assume yfinance `.financials` returns the most recent annual filing. For some companies, this may be stale by up to 12 months.
2. **Auto-adjust correctness:** We assume yfinance `auto_adjust=True` correctly handles all corporate actions (splits, dividends). This has historically had bugs in yfinance.
3. **Wikipedia scrape stability:** We assume the Wikipedia S&P 500 table format remains stable. It has changed format before.
4. **Sector classification consistency:** yfinance sector names may differ from GICS sectors (the code corrects this via ticker_meta mapping in run_screener.py:230-235).
5. **Statistical significance:** With ~500 stocks and 11 sectors, some sectors may have very few stocks (e.g., Real Estate ~30, Materials ~25). Sector-relative percentiles with <30 stocks are noisy.

---

## NEXT STEPS (Awaiting Your Approval)

Before modifying any code, I will:
1. Write `SCREENER_DEFENSIBILITY_SPEC.md` (Step 4)
2. Write the phased implementation plan (Step 5)
3. Design the test suite (Step 6)

**I will NOT change any ranking logic or code behavior until you review and approve this report.**
