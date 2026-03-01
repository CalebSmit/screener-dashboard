# SCREENER DEFENSIBILITY SPEC
**Version:** 1.1
**Date:** 2026-02-22
**Status:** DRAFT — pending owner review

---

## 1. Factor Definitions & Rationale

Each factor below exists to capture a distinct dimension of stock attractiveness. Together, they form a multi-factor model that diversifies across well-documented equity return premia.

### 1.1 Valuation (Weight: 22%)

Rationale: Value stocks (cheap relative to fundamentals) have historically outperformed over long horizons (Fama & French 1992, 1993).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **EV/EBITDA** | 25% | `Enterprise Value / (EBIT + D&A)` where D&A is from cashflow; falls back to reported EBITDA if D&A unavailable | Capital-structure-neutral valuation; EBITDA is recomputed from components (EBIT + D&A) for consistency, as yfinance's reported EBITDA can include non-operating items |
| **FCF Yield** | 45% | `(Operating Cash Flow - CapEx) / Enterprise Value` | Cash-based valuation; resistant to accounting manipulation. Highest weight because cash flow is the most reliable indicator of intrinsic value. |
| **Earnings Yield** | 20% | `LTM Net Income / Market Cap`; falls back to `Trailing EPS / Price` when LTM data unavailable | LTM-based inverse of P/E; internally consistent with other LTM flow metrics. Directly comparable across stocks. |
| **EV/Sales** | 10% | `Enterprise Value / Total Revenue` | Revenue-based valuation; useful for loss-making or early-stage companies where earnings-based metrics are unavailable. Lowest weight because it ignores margins and correlates with EV/EBITDA. |

### 1.2 Quality (Weight: 22%)

Rationale: High-quality companies (profitable, low leverage, strong fundamentals) command a premium and experience lower drawdowns (Novy-Marx 2013, Asness et al. 2019).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **ROIC** | 27% | `EBIT × (1 - tax_rate) / IC` where IC = Equity + Debt_BS - ExcessCash; ExcessCash = min(max(0, Cash - 2%×Rev), 50%×Cash); IC = max(IC, 10%×TotalAssets) | Measures management's ability to deploy capital profitably. Tax rate: actual effective rate (clamped 0-50%) when pretax > 0; **0% when pretax ≤ 0** (tax-loss position — no fictional tax charge); 21% default when both pretax and tax expense are missing. Excess cash capped at 50% of total cash; IC floored at 10% of total assets to prevent denominator collapse. |
| **Gross Profit / Assets** | 20% | `Gross Profit / Total Assets` | Novy-Marx quality factor; closer to "economic profitability" than net margin because it's above operating expenses. |
| **Net Debt / EBITDA** | 18% | `(Total Debt - Cash) / EBITDA`; net cash → 0.0; negative EBITDA → NaN; banks → NaN | Replaces Debt/Equity: negative equity from buybacks (MCD, MO, LOW) distorts D/E. Net Debt/EBITDA works across all capital structures. |
| **Piotroski F-Score** | 15% | 9 binary signals scored 0-9 | Composite financial health indicator covering profitability, leverage, liquidity, and operating efficiency (Piotroski 2000). |
| **Accruals** | 5% | `(Net Income - OCF) / Total Assets` | Earnings quality; lower accruals = more cash-backed earnings = higher quality (Sloan 1996). |
| **Operating Leverage** | 8% | `(%Δ EBIT) / (%Δ Revenue)`; \|Δ Rev\| < 1% → NaN; banks → NaN | Degree of Operating Leverage (DOL). Lower = more durable earnings (less sensitivity to revenue swings). |
| **Beneish M-Score** | 7% | 8-variable model from Beneish (1999) on annual financials; requires ≥ 5 of 8 index variables to have real data (otherwise NaN) | Earnings manipulation detection. More negative = lower manipulation risk. Non-bank only. Minimum-data gate prevents spurious scores when too many inputs are missing. |

### 1.3 Growth (Weight: 13%)

Rationale: Earnings and revenue growth are priced by markets. Forward growth expectations (when reliable) add information beyond trailing metrics.

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Forward EPS Growth** | 45% | `(Forward EPS - Trailing EPS) / max(\|Trailing EPS\|, $1.00)`, clamped to [-75%, +150%] | Consensus-based forward-looking growth. Denominator floored at $1.00. Absorbs PEG's removed weight. |
| **PEG Ratio** | 0% | `(Price / Trailing EPS) / (Forward EPS Growth % × 100)` | **Removed from scoring**: P/E ÷ growth double-counts valuation (Earnings Yield = 1/PE). Still computed for reference. |
| **Revenue Growth** | 25% | `(Revenue_TTM - Revenue_Prior) / Revenue_Prior` | Top-line growth; harder to manipulate than earnings growth. |
| **Revenue CAGR (3Y)** | 15% | 3-year compound annual revenue growth rate from annual filings | Smooths lumpy single-year revenue growth. |
| **Sustainable Growth** | 15% | `ROE × (1 - Payout Ratio)`, clamped to [0%, 100%]. ROE uses avg equity (current + prior year). Payout ratio prefers `payoutRatio` from info; falls back to `dividendsPaid/netIncome`. | Internally funded growth capacity; companies that can grow without external financing. |

### 1.4 Momentum (Weight: 13%)

Rationale: Price momentum is one of the most robust anomalies in finance (Jegadeesh & Titman 1993). The 12-1 month window is standard.

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **12-1 Month Return** | 40% | `(Price_1m_ago - Price_12m_ago) / Price_12m_ago` | Classic momentum signal. Skips the most recent month to avoid short-term reversal contamination. |
| **6-1 Month Return** | 35% | `(Price_1m_ago - Price_6m_ago) / Price_6m_ago` | Medium-term momentum. Uses 6-1M convention (excludes most recent month) for consistency with 12-1M. |
| **Jensen's Alpha** | 25% | `R_i - (R_f + β × (R_m - R_f))`; R_f from ^IRX (13-week T-bill); R_i = full 12-month return (no skip-month) | Risk-adjusted excess return above CAPM prediction. Higher = outperformance unexplained by market beta. Uses full 12-month return for consistency with Sharpe Ratio. |

### 1.5 Risk (Weight: 10%)

Rationale: Lower-risk stocks have historically delivered superior risk-adjusted returns (low-volatility anomaly; Baker et al. 2011).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Volatility** | 30% | `std(daily_returns) × sqrt(252)` | Annualized historical volatility; direct measure of stock-level risk. |
| **Beta** | 20% | `Cov(stock, market) / Var(market)`; requires ≥80% date overlap with market returns | Systematic risk exposure relative to S&P 500. Overlap validation ensures covariance is computed from aligned data. |
| **Sharpe Ratio** | 15% | `(R_12m - R_f) / Volatility`; R_f from ^IRX; R_12m = full 12-month return (no skip-month) | Risk-adjusted return per unit of total risk. Higher = more efficient risk-taking. Uses full 12-month return (not skip-month) because the Sharpe denominator is full-year volatility. |
| **Sortino Ratio** | 15% | `(R_12m - R_f) / Downside_Deviation`; DD = std of daily returns below daily R_f, annualized × √252 | Like Sharpe but only penalizes downside volatility. Investors care more about losses than general variability. Requires ≥20 downside observations. |
| **Max Drawdown (1Y)** | 20% | `min((Cum_t - Peak_t) / Peak_t)` over trailing 252 trading days | Maximum peak-to-trough decline from cumulative daily return series. Captures tail risk that volatility and beta miss. Expressed as negative fraction (e.g., -0.25 = 25% drawdown). |

### 1.6 Analyst Revisions (Weight: 10%)

Rationale: Upward revisions in analyst estimates predict future outperformance (Glushkov 2009).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Analyst Surprise** | 38% | `median((Actual EPS - Estimate) / max(\|Estimate\|, $0.10))` over last 4 quarters | Measures whether the company consistently beats expectations. Hardest backward-looking signal. |
| **Price Target Upside** | 12% | `(Mean Target Price - Current Price) / Current Price`, clamped [-50%, +100%] | Forward-looking analyst sentiment. Reduced weight due to sell-side optimism bias. |
| **Earnings Acceleration** | 20% | Delta between most recent and prior quarter surprise %. Continuous, winsorized. | Captures improving/deteriorating trajectory of beats. |
| **Beat Score** | 20% | Recency-weighted beat score: each of last 4 quarters weighted by recency (Q1=1..Q4=4). Range 0-10. | Captures consistency and recency of beats. |
| **Short Interest Ratio** | 10% | Days to cover: short interest shares / average daily volume (`shortRatio`) | Lower = less bearish sentiment from short sellers. Contrarian signal. |

**Auto-disable rule:** If <30% of the universe has any revisions data, the entire category weight (10%) is redistributed proportionally to the other categories.

### 1.7 Size (Weight: 5%)

Rationale: The Fama-French SMB factor captures the historical tendency for smaller companies to outperform larger ones (Fama & French 1993).

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Log Market Cap** | 100% | `-log(marketCap)` | Smaller companies get higher values. Within S&P 500, creates a mild mid-cap tilt. |

### 1.8 Investment (Weight: 5%)

Rationale: The Fama-French CMA factor captures the tendency for conservative-investment firms to outperform aggressive ones.

| Metric | Weight | Formula | Why It Exists |
|--------|--------|---------|---------------|
| **Asset Growth** | 100% | `(Total Assets_yr0 - Total Assets_yr1) / Total Assets_yr1` | Lower asset growth = conservative investment = higher score. Uses annual balance sheet data. |

**Auto-disable rule:** If <30% of the universe has asset growth data, the category weight is redistributed.

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
| `totalRevenue` | float | LTM: sum(quarterly_financials[0:4]) | > 0 | Yes |
| `totalRevenue_prior` | float | Prior-year LTM: sum(quarterly_financials[4:8]) | > 0 | No |
| `grossProfit` | float | LTM: sum(quarterly_financials[0:4]) | any | Yes (for quality) |
| `ebit` | float | LTM: sum(quarterly_financials[0:4]) | any | Yes (for ROIC) |
| `ebitda` | float | LTM: sum(quarterly_financials[0:4]) | any | Yes (for valuation) |
| `netIncome` | float | LTM: sum(quarterly_financials[0:4]) | any | Yes |
| `incomeTaxExpense` | float | LTM: sum(quarterly_financials[0:4]) | any | No |
| `pretaxIncome` | float | LTM: sum(quarterly_financials[0:4]) | any | No |
| `totalAssets` | float | MRQ: quarterly_balance_sheet[0] | > 0 | Yes |
| `totalEquity` | float | MRQ: quarterly_balance_sheet[0] | any | Yes (for ROIC, D/E) |
| `longTermDebt` | float | MRQ: quarterly_balance_sheet[0] | >= 0 | No |
| `currentAssets` | float | MRQ: quarterly_balance_sheet[0] | >= 0 | No |
| `currentLiabilities` | float | MRQ: quarterly_balance_sheet[0] | >= 0 | No |
| `operatingCashFlow` | float | LTM: sum(quarterly_cashflow[0:4]) | any | Yes |
| `capex` | float | LTM: sum(quarterly_cashflow[0:4]) | any (usually negative) | Yes |
| `dividendsPaid` | float | LTM: sum(quarterly_cashflow[0:4]) | any (usually negative) | No |
| `da_cf` | float | LTM: sum(quarterly_cashflow[0:4]) | >= 0 | No (fallback: reported EBITDA) |
| `totalEquity_prior` | float | Year-ago MRQ: quarterly_balance_sheet[4] | any | No (fallback: current equity) |
| `payoutRatio` | float | yf.info | [0, 2.0] | No (fallback: dividendsPaid/NI) |
| Daily Close prices | Series[float] | history("13mo") | > 0, length >= 200 | Yes |
| `epsActual` | float | earnings_history | any | No |
| `epsEstimate` | float | earnings_history | any | No |

### 2.2 Data Frequency: LTM / MRQ

All financial statement data uses **quarterly** filings from yfinance for maximum freshness:

- **Flow metrics** (income statement + cash flow): **LTM** = sum of the 4 most recent quarterly filings. This reduces data staleness from up to 12 months (annual filings) to at most 3 months.
- **Balance sheet items**: **MRQ** (Most Recent Quarter) = col=0 of `quarterly_balance_sheet`. Prior-period balance sheet items use col=4 (same quarter one year ago).
- **Prior-period comparisons**: Revenue growth, asset growth, and Piotroski YoY signals compare current LTM/MRQ against prior-year LTM (quarters 4-7) or year-ago MRQ (col=4).
- **Partial data**: If only 3 of 4 quarters are available for LTM, the sum is annualized as `sum × (4/3)`. If fewer than 3 quarters are available, the field returns NaN.
- **Annual fallback**: If quarterly financial statements are entirely unavailable for a ticker (e.g., foreign ADRs, very recent IPOs), the screener falls back to annual filings.

### 2.3 Field Naming Convention
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
| Individual metric is NaN | Left as NaN; weight redistributed to available metrics within the category | Honest handling — missing data contributes 0 weight, not a fake 50th percentile |
| Sector has < 10 valid values for a metric | Fall back to universe-wide percentile ranking for that metric | Insufficient data for meaningful within-sector ranking; universe-wide is more informative than a flat 50th percentile |
| Stock has < 60% of metrics populated | **Exclude from scoring** | Insufficient coverage for reliable composite |
| Revisions category < 30% coverage universe-wide | **Auto-disable category**, redistribute weight | Prevents noise from dominating the composite |

### 4.2 Outlier Treatment

| Step | Method | Parameters | Applied When |
|------|--------|------------|-------------|
| 1. Winsorization | `scipy.stats.mstats.winsorize` | limits=(0.01, 0.01) | Before percentile ranking; requires >= 10 non-NaN values |
| 2. Percentile ranking | `pd.rank(pct=True)` within sector | - | Inherently robust to remaining outliers |
| 3. Net Debt/EBITDA negative EBITDA | NaN (negative EBITDA = loss-making) | - | At metric computation time |

### 4.3 Negative Denominator Rules

| Metric | Denominator | If <= 0 |
|--------|-------------|---------|
| EV/EBITDA | EBITDA | NaN |
| EV/EBITDA | EV | NaN |
| FCF Yield | EV | NaN |
| Earnings Yield | Price | NaN (should not occur) |
| EV/Sales | Revenue | NaN |
| EV/Sales | EV | NaN |
| ROIC | Invested Capital (after 50% cash cap + 10% TA floor) | NaN |
| ROIC | Pretax Income ≤ 0 (tax-loss) | Use 0% tax rate (no fictional tax on losses) |
| ROIC | Pretax Income missing (NaN) | Use default 21% tax rate |
| Gross Profit/Assets | Total Assets | NaN |
| Net Debt/EBITDA | EBITDA | NaN (negative EBITDA = loss-making; net cash → 0.0) |
| Piotroski | Various | Signal marked as untestable |
| Accruals | Total Assets | NaN |
| Forward EPS Growth | Trailing EPS | NaN if |trail| < 0.01 |
| PEG | Earnings Growth | NaN if growth <= 0.01 |
| Revenue Growth | Prior Revenue | NaN |
| Sustainable Growth | Equity or NI | NaN if either <= 0 |
| Operating Leverage | Revenue change | NaN if |Δ Revenue %| < 1% or EBIT_prior = 0 |
| Sharpe Ratio | Volatility | NaN if volatility = 0 or NaN |
| Sortino Ratio | Downside Deviation | NaN if DD = 0, NaN, or < 20 downside observations |
| Max Drawdown (1Y) | Daily returns | NaN if < 50 daily return observations |
| Jensen's Alpha | Beta or Return | NaN if beta or return is NaN |
| Beta | Date overlap | NaN if < 200 common dates or < 80% overlap with market |
| Momentum | Prior Price | NaN if <= 0 |
| Beta | Market Variance | NaN if <= 0 |

---

## 5. Scoring Methodology

### 5.1 Pipeline Steps (in order)
1. **Compute raw metrics** (32 generic + 4 bank-specific = 36 total metrics from yfinance data; 30 generic carry non-zero weight)
2. **Winsorize** at 1st/99th percentiles per metric (universe-wide)
3. **Sector-relative percentile rank** per metric (within each GICS Sector)
4. **Weighted category scores** = Σ(metric_pct × metric_weight) per category
5. **Composite score** = Σ(category_score × category_weight) with per-row weight redistribution for NaN categories, then percentile-ranked to [0, 100]
6. **Value trap flags** = 2-of-3 majority: flagged if at least 2 of (quality < 30th pctile, momentum < 30th pctile, revisions < 30th pctile)
7. **Final ranking** by Composite descending (ties use "min" method)

### 5.2 Weight Configuration

**Factor Weights** (sum to 100):
```
Valuation: 22  |  Quality: 22  |  Growth: 13  |  Momentum: 13  |  Risk: 10  |  Revisions: 10  |  Size: 5  |  Investment: 5
```

**Metric Weights** (each category sums to 100):
```
Valuation:  FCF Yield 45, EV/EBITDA 25 (EBIT+D&A), Earnings Yield 20 (LTM NI/MC), EV/Sales 10
Quality:    ROIC 27 (50% cash cap, 10% TA floor, 0% tax for losses), GP/Assets 20, Net Debt/EBITDA 18, Piotroski 15, Operating Leverage 8, Beneish M-Score 7, Accruals 5
Growth:     Forward EPS 45 (clamp [-75%,+150%]), Revenue Growth 25, Revenue CAGR 3Y 15, Sustainable Growth 15 (avg equity, SGR [0%,100%]), PEG 0 (removed — double-counts valuation)
Momentum:   12-1M Return 40, 6-1M Return 35, Jensen's Alpha 25
Risk:       Volatility 30, Beta 20, Sharpe 15, Sortino 15, Max Drawdown 20
Revisions:  Analyst Surprise 38, Earnings Acceleration 20, Beat Score 20, Price Target Upside 12, Short Interest Ratio 10
Size:       -log(Market Cap) 100
Investment: Asset Growth 100
```

### 5.3 Normalization Justification
- **Why percentile ranking?** Rank-based scoring is robust to outliers and non-normal distributions. Raw z-scores would be dominated by extreme values even after winsorization.
- **Why sector-relative?** Different sectors have structurally different metric distributions (e.g., Financials have higher D/E than Tech). Sector-relative ranking ensures fair comparison.
- **Why percentile-rank final scaling?** Converts the weighted composite to a 0-100 scale via `rank(pct=True) * 100`. Unlike min-max scaling, percentile ranking is robust to a single extreme composite pulling the entire scale.

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

## 7. Defensibility & Transparency Features

### 7.1 Weight Sensitivity Analysis

**Purpose:** Demonstrate that the portfolio output is robust to reasonable changes in factor weights — a key requirement for any defensible quantitative process.

**Implementation:**
- After scoring, each factor category weight is perturbed ±5% (one at a time, others renormalized to sum to 100)
- The composite score is recomputed for each perturbation
- **Jaccard similarity** of the top-20 portfolio (perturbed vs. baseline) is calculated: `|intersection| / |union|`
- Results: one row per category showing upward and downward Jaccard values

**Interpretation:**
| Jaccard | Assessment |
|---------|-----------|
| ≥ 0.85 | **Robust** — small weight changes barely affect the portfolio |
| 0.70–0.84 | **Moderate** — some sensitivity, acceptable for most purposes |
| < 0.70 | **Sensitive** — the ranking depends heavily on this factor's exact weight |

**Output:** Excel "WeightSensitivity" sheet with color-coded Jaccard values (green ≥ 0.85, yellow 0.70–0.84, red < 0.70).

### 7.2 EPS Basis Mismatch Detection

**Purpose:** Flag stocks where the forward EPS growth and PEG ratio metrics may be unreliable due to GAAP vs. normalized (non-GAAP) EPS inconsistency.

**Problem:** Yahoo Finance provides GAAP trailing EPS but normalized forward consensus EPS. For companies with large non-cash charges, write-downs, or unrealized gains, the ratio between these can produce misleading growth signals.

**Implementation:**
- For each stock, compute `eps_ratio = forwardEps / trailingEps`
- If `|trailingEps| > $0.10` and `eps_ratio > 2.0` or `eps_ratio < 0.3`: set `_eps_basis_mismatch = True`
- The `_eps_ratio` field stores the raw ratio for inspection

**Output:**
- `_eps_basis_mismatch` boolean column in scored data
- `_eps_ratio` float column
- Highlighted in the DataValidation Excel sheet
- Console summary printed during pipeline run (count + example tickers)

### 7.3 Factor Correlation Matrix

**Purpose:** Quantify and make transparent the degree of overlap (double-counting) between factor categories.

**Implementation:**
- Spearman rank correlation computed between all pairs of `*_score` columns (8×8 matrix)
- Known structural correlations: EV/EBITDA ~ EV/Sales (both use EV), Volatility ~ Beta (~0.6-0.8), 12-1M ~ 6-1M momentum (~6 months overlap)

**Interpretation:**
| Correlation | Assessment |
|-------------|-----------|
| > 0.80 | **High overlap** (red) — these factors may be measuring the same thing |
| 0.60–0.80 | **Moderate overlap** (orange) — some shared signal |
| < 0.60 | **Acceptable** — factors capture distinct information |

**Output:** Excel "FactorCorrelation" sheet with color-coded correlation values. The effective number of independent factors is typically 5-6 rather than the nominal 8.

### 7.4 Data Provenance

**Purpose:** Per-stock transparency about data source and completeness.

**Implementation:**
- `_data_source`: origin of the data ("yfinance", "cache", "sample")
- `_metric_count`: number of the core metrics that have valid (non-NaN) values
- `_metric_total`: total possible core metrics

**Use:** Low `_metric_count` stocks are scored on fewer signals, making their composite less reliable even if they pass the 60% coverage filter.

### 7.5 DataValidation Sheet

**Purpose:** Enable manual verification of the screener's top picks against external sources (Bloomberg, SEC filings, company investor relations).

**Implementation:**
- Top 10 portfolio stocks displayed with raw financial values: market cap, total revenue, net income, trailing EPS, forward EPS, current price, and key computed metrics
- Six types of issues are highlighted:
  1. **EPS basis mismatch** — `_eps_basis_mismatch = True` (GAAP vs. normalized discrepancy)
  2. **Stale data** — price target data that may be outdated
  3. **EV discrepancy** — stocks flagged by EV cross-validation (`_ev_flag`)
  4. **LTM partial annualization** — `_ltm_annualized = True` when 3-of-4 quarter annualization was used
  5. **Channel-stuffing risk** — `_channel_stuffing_flag = True` when receivables growth exceeds revenue growth by >15pp
  6. **Beta data overlap** — `_beta_overlap_pct` shows percentage of market dates covered by the stock's return series
- **Sector Context table** shows sector 25th/median/75th percentile for 8 key raw metrics, enabling quick comparison of a stock's values against its sector norms

**Output:** Excel "DataValidation" sheet with conditional formatting highlighting flagged values (red for high-severity, yellow for warnings).

---

## 8. Disclaimers & Limitations

### 8.1 Data Source Limitations
- **yfinance is an unofficial Yahoo Finance API.** It has no SLA, no guaranteed uptime, and data may be delayed or incorrect. Yahoo Finance is not a Bloomberg or FactSet-grade data source.
- **Financial statements from yfinance may lag.** Some companies may show data from 12+ months ago if they have not yet filed their annual report.
- **Analyst data is sparse.** Only ~40% of S&P 500 tickers have earnings history available through yfinance. The EPS revision and estimate change metrics are not available.
- **S&P 500 constituent list is community-maintained.** The primary source is a GitHub-hosted CSV (`datasets/s-and-p-500-companies`), with Wikipedia as a secondary fallback. The local backup (`sp500_tickers.json`) is auto-updated on each successful network fetch. The list may lag official S&P Dow Jones index changes by a few days.

### 8.2 Methodological Limitations
- **Factor weights are heuristic, not optimized.** They reflect reasonable priors, not backtested optimal values. This is intentional — optimized weights would be overfit to historical data.
- **Sector-relative ranking with small sectors is noisy.** Sectors with < 25 stocks (e.g., Materials, Real Estate) produce less reliable percentile rankings. Sectors with < 10 valid values for a metric fall back to universe-wide percentile ranking.
- **Momentum and value factors can conflict.** In regime transitions, the composite may produce confusing rankings. Factor weights are static and do not adapt to market regimes.
- **No transaction cost modeling in ranking.** The screener does not account for liquidity, market impact, or trading costs.
- **Factor correlation / double-counting.** Within-category metrics are correlated: EV/EBITDA ~ EV/Sales (both use EV), Volatility ~ Beta (~0.6-0.8 empirically), Sharpe ~ Sortino (similar inputs), 12-1M ~ 6-1M momentum (~6 months overlap). Effective independent factors are ~8-10 rather than 36. A Spearman correlation matrix is computed at runtime and written to the "FactorCorrelation" Excel sheet (see §7.3) for transparency.
- **EV time mismatch.** Enterprise Value uses current market cap (real-time) combined with balance sheet debt/cash from MRQ (most recent quarterly filing) — up to 3 months stale (reduced from up to 12 months with annual data). Standard for live screening but not point-in-time safe. **Mitigated by cross-validation:** API-provided EV is checked against computed MC + Debt - Cash; discrepancies > 10% (25% for Financials, whose "debt" includes deposits) are auto-corrected and flagged (`_ev_flag`).
- **PEG ratio removed from scoring.** PEG (P/E ÷ growth) double-counts valuation already captured by Earnings Yield (= 1/PE). It is still computed for reference output but has weight=0 in the Growth category. The EPS Basis Mismatch Detection (see §7.2) remains relevant for Forward EPS Growth, which retains a 45% weight.
- **Value trap filter uses 2-of-3 majority logic.** A stock is flagged only if at least 2 of 3 conditions are met (quality < 30th, momentum < 30th, or revisions < 30th percentile). This is more balanced than OR logic, which flagged ~60% of the universe.

### 8.3 What This Screener Is NOT
- It is NOT a recommendation to buy or sell any security
- It is NOT a substitute for professional financial advice
- It is NOT point-in-time safe for historical backtesting
- It is NOT suitable for regulatory or compliance reporting
- It does NOT guarantee future performance
