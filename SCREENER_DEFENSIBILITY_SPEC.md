# SCREENER DEFENSIBILITY SPEC
**Version:** 1.0
**Date:** 2026-02-17
**Status:** DRAFT — pending owner review

---

## 1. Factor Definitions & Rationale

Each factor below exists to capture a distinct dimension of stock attractiveness. Together, they form a multi-factor model that diversifies across well-documented equity return premia.

### 1.1 Valuation (Weight: 25%)

Rationale: Value stocks (cheap relative to fundamentals) have historically outperformed over long horizons (Fama & French 1992, 1993).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **EV/EBITDA** | 25% | `Enterprise Value / EBITDA` | Capital-structure-neutral valuation; preferred over P/E for cross-sector comparability because it removes the effect of leverage and tax differences |
| **FCF Yield** | 40% | `(Operating Cash Flow - CapEx) / Enterprise Value` | Cash-based valuation; resistant to accounting manipulation. Highest weight because cash flow is the most reliable indicator of intrinsic value. |
| **Earnings Yield** | 20% | `Trailing EPS / Current Price` | Inverse of P/E; directly comparable across stocks. Uses trailing (realized) earnings, not estimates. |
| **EV/Sales** | 15% | `Enterprise Value / Total Revenue` | Revenue-based valuation; useful for loss-making or early-stage companies where earnings-based metrics are unavailable. Lowest weight because it ignores margins entirely. |

### 1.2 Quality (Weight: 25%)

Rationale: High-quality companies (profitable, low leverage, strong fundamentals) command a premium and experience lower drawdowns (Novy-Marx 2013, Asness et al. 2019).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **ROIC** | 30% | `EBIT × (1 - tax_rate) / (Equity + Debt - Cash)` | Measures management's ability to deploy capital profitably. Tax rate = actual (clamped 0-50%) or 21% default. |
| **Gross Profit / Assets** | 25% | `Gross Profit / Total Assets` | Novy-Marx quality factor; closer to "economic profitability" than net margin because it's above operating expenses. |
| **Debt/Equity** | 20% | `Total Debt / Stockholders' Equity` | Leverage risk; lower is better. Set to 999.0 for negative equity (distress signal). |
| **Piotroski F-Score** | 15% | 9 binary signals scored 0-9 | Composite financial health indicator covering profitability, leverage, liquidity, and operating efficiency (Piotroski 2000). |
| **Accruals** | 10% | `(Net Income - OCF) / Total Assets` | Earnings quality; lower accruals = more cash-backed earnings = higher quality (Sloan 1996). |

### 1.3 Growth (Weight: 15%)

Rationale: Earnings and revenue growth are priced by markets. Forward growth expectations (when reliable) add information beyond trailing metrics.

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Forward EPS Growth** | 35% | `(Forward EPS - Trailing EPS) / \|Trailing EPS\|` | Consensus-based forward-looking growth. Highest weight because forward estimates incorporate market expectations. |
| **PEG Ratio** | 20% | `(Price / Trailing EPS) / (Earnings Growth % × 100)` | Valuation-adjusted growth; identifies growth at a reasonable price. |
| **Revenue Growth** | 30% | `(Revenue_TTM - Revenue_Prior) / Revenue_Prior` | Top-line growth; harder to manipulate than earnings growth. |
| **Sustainable Growth** | 15% | `ROE × (1 - Dividend Payout Ratio)` | Internally funded growth capacity; companies that can grow without external financing. |

### 1.4 Momentum (Weight: 15%)

Rationale: Price momentum is one of the most robust anomalies in finance (Jegadeesh & Titman 1993). The 12-1 month window is standard.

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **12-1 Month Return** | 50% | `(Price_1m_ago - Price_12m_ago) / Price_12m_ago` | Classic momentum signal. Skips the most recent month to avoid short-term reversal contamination. |
| **6-1 Month Return** | 50% | `(Price_1m_ago - Price_6m_ago) / Price_6m_ago` | Medium-term momentum. Uses 6-1M convention (excludes most recent month) for consistency with 12-1M and to avoid short-term reversal contamination. |

### 1.5 Risk (Weight: 10%)

Rationale: Lower-risk stocks have historically delivered superior risk-adjusted returns (low-volatility anomaly; Baker et al. 2011).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Volatility** | 60% | `std(daily_log_returns) × sqrt(252)` | Annualized historical volatility; direct measure of stock-level risk. |
| **Beta** | 40% | `Cov(stock, market) / Var(market)` | Systematic risk exposure relative to S&P 500. |

### 1.6 Analyst Revisions (Weight: 10%)

Rationale: Upward revisions in analyst estimates predict future outperformance (Glushkov 2009).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Analyst Surprise** | 25% (100% effective) | `mean((Actual EPS - Estimate) / \|Estimate\|)` over last 4 quarters | Measures whether the company consistently beats expectations. |
| **EPS Revision Ratio** | 40% (NOT IMPLEMENTED) | Placeholder — always NaN | **Requires institutional data source (IBES/FactSet).** |
| **EPS Estimate Change** | 35% (NOT IMPLEMENTED) | Placeholder — always NaN | **Requires institutional data source (IBES/FactSet).** |

**Auto-disable rule:** If <30% of the universe has any revisions data, the entire category weight (10%) is redistributed proportionally to the other 5 categories.

---

## 2. Data Contract

### 2.1 Raw Input Fields

Each ticker fetch must produce the following fields. Types and acceptable ranges are specified for validation.

| Field | Type | Source | Acceptable Range | Required? |
|-------|------|--------|-----------------|-----------|
| `marketCap` | float | yf.info | > 0 | Yes (for EV fallback) |
| `enterpriseValue` | float | yf.info | any (can be negative for cash-rich firms) | Preferred |
| `trailingEps` | float | yf.info | any | Yes |
| `forwardEps` | float | yf.info | any | No |
| `currentPrice` | float | yf.info | > 0 | Yes |
| `totalDebt` | float | yf.info | >= 0 | Yes (for EV/ROIC) |
| `totalCash` | float | yf.info | >= 0 | Yes (for EV/ROIC) |
| `sharesOutstanding` | float | yf.info | > 0 | No |
| `earningsGrowth` | float | yf.info | any | No |
| `dividendRate` | float | yf.info | >= 0 | No |
| `totalRevenue` | float | financials[0] | > 0 | Yes |
| `totalRevenue_prior` | float | financials[1] | > 0 | No |
| `grossProfit` | float | financials[0] | any | Yes (for quality) |
| `ebit` | float | financials[0] | any | Yes (for ROIC) |
| `ebitda` | float | financials[0] | any | Yes (for valuation) |
| `netIncome` | float | financials[0] | any | Yes |
| `incomeTaxExpense` | float | financials[0] | any | No |
| `pretaxIncome` | float | financials[0] | any | No |
| `totalAssets` | float | balance_sheet[0] | > 0 | Yes |
| `totalEquity` | float | balance_sheet[0] | any | Yes (for ROIC, D/E) |
| `longTermDebt` | float | balance_sheet[0] | >= 0 | No |
| `currentAssets` | float | balance_sheet[0] | >= 0 | No |
| `currentLiabilities` | float | balance_sheet[0] | >= 0 | No |
| `operatingCashFlow` | float | cashflow[0] | any | Yes |
| `capex` | float | cashflow[0] | any (usually negative) | Yes |
| `dividendsPaid` | float | cashflow[0] | any (usually negative) | No |
| Daily Close prices | Series[float] | history("13mo") | > 0, length >= 200 | Yes |
| `epsActual` | float | earnings_history | any | No |
| `epsEstimate` | float | earnings_history | any | No |

### 2.2 Field Naming Convention
- Raw yfinance fields use camelCase (e.g., `marketCap`, `trailingEps`)
- Computed metrics use snake_case (e.g., `ev_ebitda`, `fcf_yield`)
- Prior-period fields have `_prior` suffix (e.g., `totalRevenue_prior`)
- Percentile-ranked fields have `_pct` suffix (e.g., `ev_ebitda_pct`)
- Category scores have `_score` suffix (e.g., `valuation_score`)

---

## 3. No Look-Ahead / Point-in-Time Policy

### 3.1 Live Screening (Acceptable)
For live screening (the primary use case), the screener uses current market data and the most recently available financial reports. This is acceptable because:
- Market prices are real-time by definition
- Financial statements reflect the latest available public information
- Analyst estimates reflect current consensus

### 3.2 Backtesting (NOT Point-in-Time Safe)
**The current backtest implementation is NOT point-in-time safe.**

Specific violations:
1. **Universe:** Uses today's S&P 500 constituents for all historical periods (survivorship bias)
2. **Fundamentals:** Uses current financial data for all historical scores (look-ahead bias)
3. **Estimates:** Forward EPS and analyst data reflect current consensus, not historical estimates

**Required disclaimer on all backtest outputs:**
> "ILLUSTRATIVE ONLY. This backtest uses current financial data and today's index constituents applied retroactively. Results do not represent achievable historical performance. Survivorship bias and look-ahead bias are present. Do not use for strategy validation or investor marketing."

### 3.3 Future Point-in-Time Requirements
To achieve point-in-time safety, the system would need:
- Historical index membership data (S&P adds/removes by date)
- As-of-date financial statements (not restated figures)
- Historical analyst consensus estimates by date
- These require institutional data providers (Compustat, IBES, S&P Capital IQ)

---

## 4. Missing Data, Outliers, and Negative Denominators

### 4.1 Missing Data Rules

| Scenario | Rule | Rationale |
|----------|------|-----------|
| Individual metric is NaN | Assign 50th percentile for that metric's sector rank | Conservative neutral assumption; does not reward or penalize |
| Sector has < 3 valid values for a metric | All stocks in sector get 50th percentile for that metric | Insufficient data for meaningful ranking |
| Stock has < 60% of metrics populated | **Exclude from scoring** | Insufficient coverage for reliable composite |
| Revisions category < 30% coverage universe-wide | **Auto-disable category**, redistribute weight | Prevents noise from dominating the composite |

### 4.2 Outlier Treatment

| Step | Method | Parameters | Applied When |
|------|--------|------------|-------------|
| 1. Winsorization | `scipy.stats.mstats.winsorize` | limits=(0.01, 0.01) | Before percentile ranking; requires >= 10 non-NaN values |
| 2. Percentile ranking | `pd.rank(pct=True)` within sector | - | Inherently robust to remaining outliers |
| 3. Debt/Equity sentinel | Set to 999.0 if equity <= 0 | - | At metric computation time |

### 4.3 Negative Denominator Rules

| Metric | Denominator | If <= 0 |
|--------|-------------|---------|
| EV/EBITDA | EBITDA | NaN |
| EV/EBITDA | EV | NaN |
| FCF Yield | EV | NaN |
| Earnings Yield | Price | NaN (should not occur) |
| EV/Sales | Revenue | NaN |
| EV/Sales | EV | NaN |
| ROIC | Invested Capital | NaN |
| ROIC | Pretax Income (for tax rate) | Use default 21% tax rate |
| Gross Profit/Assets | Total Assets | NaN |
| Debt/Equity | Equity | Set to 999.0 (distress) |
| Piotroski | Various | Signal marked as untestable |
| Accruals | Total Assets | NaN |
| Forward EPS Growth | Trailing EPS | NaN if |trail| < 0.01 |
| PEG | Earnings Growth | NaN if growth <= 0.01 |
| Revenue Growth | Prior Revenue | NaN |
| Sustainable Growth | Equity or NI | NaN if either <= 0 |
| Momentum | Prior Price | NaN if <= 0 |
| Beta | Market Variance | NaN if <= 0 |

---

## 5. Scoring Methodology

### 5.1 Pipeline Steps (in order)
1. **Compute raw metrics** (17 metrics from yfinance data)
2. **Winsorize** at 1st/99th percentiles per metric (universe-wide)
3. **Sector-relative percentile rank** per metric (within each GICS Sector)
4. **Weighted category scores** = Σ(metric_pct × metric_weight) per category
5. **Composite score** = Σ(category_score × category_weight), min-max scaled to [0, 100]
6. **Value trap flags** = OR(quality < 30th pctile, momentum < 30th pctile, revisions < 30th pctile)
7. **Final ranking** by Composite descending (ties use "min" method)

### 5.2 Weight Configuration

**Factor Weights** (sum to 100):
```
Valuation: 25  |  Quality: 25  |  Growth: 15  |  Momentum: 15  |  Risk: 10  |  Revisions: 10
```

**Metric Weights** (each category sums to 100):
```
Valuation:  FCF Yield 40, EV/EBITDA 25, Earnings Yield 20, EV/Sales 15
Quality:    ROIC 30, GP/Assets 25, Debt/Equity 20, Piotroski 15, Accruals 10
Growth:     Forward EPS 35, Revenue Growth 30, PEG 20, Sustainable Growth 15
Momentum:   12-1M Return 50, 6M Return 50
Risk:       Volatility 60, Beta 40
Revisions:  EPS Revision 40, EPS Est Change 35, Analyst Surprise 25
```

### 5.3 Normalization Justification
- **Why percentile ranking?** Rank-based scoring is robust to outliers and non-normal distributions. Raw z-scores would be dominated by extreme values even after winsorization.
- **Why sector-relative?** Different sectors have structurally different metric distributions (e.g., Financials have higher D/E than Tech). Sector-relative ranking ensures fair comparison.
- **Why min-max final scaling?** Converts the composite to an intuitive 0-100 scale for portfolio construction and display.

### 5.4 Tie-Breaking
- Ranking uses `method="min"`: tied scores receive the same rank, next rank is skipped.
- Example: Scores [85, 85, 83] → Ranks [1, 1, 3]
- For portfolio construction, ties are broken by `sort_values("Composite", ascending=False)` which uses pandas' default stable sort (preserves order of equal elements from input).

---

## 6. Reproducibility Requirements

### 6.1 Current State
- Run_id generated (UUID4) per pipeline run via `RunContext`
- Config snapshot saved as `runs/{run_id}/config.yaml`
- 5 intermediate Parquet artifacts saved per run (raw, winsorized, percentiles, category scores, final)
- Universe and effective weights saved per run
- Structured JSON logging with per-ticker timing
- Git repo initialized with pinned dependencies
- Config-aware caching (cache includes config hash in filename)

### 6.2 Required (Phase 1 Implementation)
Each run must produce:
1. **run_id**: UUID4 generated at pipeline start
2. **Config snapshot**: Copy of config.yaml saved as `runs/{run_id}/config.yaml`
3. **Run metadata**: JSON file with start_time, end_time, code_version, python_version, dependency versions, universe_size, tickers_scored, tickers_failed
4. **Factor scores output**: Parquet with run_id column
5. **Data quality log**: CSV with run_id column

### 6.3 Desired (Phase 2+)
6. **Raw data snapshot**: Parquet of raw fetched data before any computation
7. **Intermediate tables**: Parquet files for post-winsorize, post-percentile, post-category-score stages
8. **Immutable storage**: Append-only run history (never overwrite past results)
9. **Git integration**: Code commit SHA recorded with each run

---

## 7. Disclaimers & Limitations

### 7.1 Data Source Limitations
- **yfinance is an unofficial Yahoo Finance API.** It has no SLA, no guaranteed uptime, and data may be delayed or incorrect. Yahoo Finance is not a Bloomberg or FactSet-grade data source.
- **Financial statements from yfinance may lag.** Some companies may show data from 12+ months ago if they have not yet filed their annual report.
- **Analyst data is sparse.** Only ~40% of S&P 500 tickers have earnings history available through yfinance. The EPS revision and estimate change metrics are not available.
- **Wikipedia S&P 500 list may be stale.** The list reflects the most recent index rebalance and may not match the official S&P Dow Jones list exactly.

### 7.2 Methodological Limitations
- **Factor weights are heuristic, not optimized.** They reflect reasonable priors, not backtested optimal values. This is intentional — optimized weights would be overfit to historical data.
- **Sector-relative ranking with small sectors is noisy.** Sectors with < 25 stocks (e.g., Materials, Real Estate) produce less reliable percentile rankings. Sectors with < 3 valid values for a metric receive the neutral 50th percentile.
- **Momentum and value factors can conflict.** In regime transitions, the composite may produce confusing rankings. Factor weights are static and do not adapt to market regimes.
- **No transaction cost modeling in ranking.** The screener does not account for liquidity, market impact, or trading costs.
- **Factor correlation / double-counting.** Within-category metrics are correlated: EV/EBITDA ~ EV/Sales (both use EV), Volatility ~ Beta (~0.6-0.8 empirically), 12-1M ~ 6-1M momentum (~6 months overlap). Effective independent factors are ~8-10 rather than 17. A correlation matrix is computed at runtime via `compute_factor_correlation()` for transparency.
- **EV time mismatch.** Enterprise Value uses current market cap (real-time) combined with balance sheet debt/cash from the most recent annual filing — up to 12 months stale. Standard for live screening but not point-in-time safe.
- **PEG ratio input opacity.** The `earningsGrowth` field from yfinance `.info` is a black-box — it may be trailing or forward-looking, and its definition varies. The PEG ratio should be treated as approximate.
- **Value trap filter uses OR logic.** Any single floor breach (quality < 30th, momentum < 30th, or revisions < 30th percentile) triggers the flag. A stock with excellent quality but poor momentum will be flagged. This is documented in config.yaml.

### 7.3 What This Screener Is NOT
- It is NOT a recommendation to buy or sell any security
- It is NOT a substitute for professional financial advice
- It is NOT point-in-time safe for historical backtesting
- It is NOT suitable for regulatory or compliance reporting
- It does NOT guarantee future performance
