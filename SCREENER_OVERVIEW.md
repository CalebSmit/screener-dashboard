# Multi-Factor Stock Screener — How It Works

**A plain-language guide to what the screener does, why it does it, and how it arrives at its picks.**

---

## What Is This?

This is a quantitative stock screener. It takes every company in the S&P 500 (roughly 500 stocks), measures each one across up to 34 financial metrics, combines those measurements into a single composite score (0-100), and ranks the entire universe from best to worst. The top-ranked stocks form a model portfolio.

Not every stock sees all 34 metrics. The screener uses 30 generic metrics for most stocks and a separate set of 4 bank-specific metrics for financial companies (banks, insurers, credit companies). In practice, any individual stock is scored on about 30 metrics — the set just differs depending on whether the company is a bank or not.

The core idea: no single number tells you whether a stock is a good investment. A stock can look cheap but be cheap for a reason (declining business, high risk). By scoring across multiple independent dimensions — valuation, quality, growth, momentum, risk, revisions, size, investment — the screener surfaces companies that are strong across the board, not just on one axis.

---

## Where Does the Data Come From?

All data is pulled from **Yahoo Finance** via the `yfinance` Python library. For each stock, the screener fetches:

- **Financial statements** — income statement, balance sheet, and cash flow statement (annual + prior year for trend comparisons)
- **Price history** — 13 months of daily closing prices and volume (calendar-based lookbacks for momentum, volatility, and liquidity)
- **Summary statistics** — market cap, enterprise value, P/E ratios, EPS estimates, analyst price targets, number of covering analysts
- **Earnings history** — last 4 quarters of actual vs. estimated EPS (for earnings surprise calculations)

The S&P 500 member list is pulled primarily from a **GitHub-hosted CSV** (`datasets/s-and-p-500-companies`), with Wikipedia as a secondary fallback and a local backup (`sp500_tickers.json`) as a last resort. The local JSON is auto-updated whenever a network source succeeds.

Data is fetched in batches of 30 tickers with 3 concurrent threads per batch and a 3.0-second inter-batch delay to manage Yahoo Finance rate limits. Failed tickers are automatically retried in a second pass with conservative settings (single-threaded, 30-second cooldown).

---

## The 8 Factor Categories

Every stock is evaluated in 8 categories. Each category captures a different dimension of investment merit.

### 1. Valuation (22% of final score)

**Question it answers:** *Is this stock priced attractively relative to what the business generates?*

**Generic stocks:**

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **EV/EBITDA** | 25% | Enterprise value divided by earnings before interest, taxes, depreciation, and amortization. A capital-structure-neutral price tag. Lower = cheaper. |
| **FCF Yield** | 45% | Free cash flow (operating cash flow minus capital expenditures) divided by enterprise value. How much cash the business generates per dollar of total value. Higher = cheaper. |
| **Earnings Yield** | 20% | LTM Net Income divided by Market Cap (inverse of P/E). Uses LTM for consistency with other flow metrics. Higher = cheaper. |
| **EV/Sales** | 10% | Enterprise value divided by revenue. Useful for comparing companies with different margin profiles. Lower = cheaper. |

**Bank-like stocks** use a different weight set (see [Bank-Specific Scoring](#bank-specific-scoring) below):

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Earnings Yield** | 40% | LTM Net Income divided by Market Cap (inverse of P/E). Uses LTM for consistency with other flow metrics. Higher = cheaper. |
| **Price-to-Book (P/B)** | 60% | Share price divided by book value per share. THE key bank valuation metric — banks' assets are mostly financial instruments carried near fair value. Lower = cheaper. |

**Why these?** Traditional P/E ratios are distorted by capital structure, one-time charges, and accounting choices. Enterprise value-based metrics strip away those distortions. FCF Yield gets the heaviest weight because cash flow is the hardest number for management to manipulate — it's cash in the door. For banks, EV-based metrics are meaningless (deposits are both liabilities and the core business), so P/B replaces them.

---

### 2. Quality (22% of final score)

**Question it answers:** *Is this a well-run business with durable competitive advantages?*

**Generic stocks:**

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **ROIC** | 27% | Return on Invested Capital — NOPAT divided by invested capital (equity + debt - excess cash). Excess cash is cash beyond 2% of revenue. Tax rate: actual effective rate (clamped 0-50%) when pretax income is positive; 0% for tax-loss positions (negative pretax); 21% default when data is missing. Higher = better use of capital. |
| **Gross Profit / Assets** | 20% | Gross profit divided by total assets. Measures asset-light profitability (Novy-Marx quality factor). |
| **Net Debt / EBITDA** | 18% | (Total Debt - Cash) / EBITDA. Measures leverage relative to earnings power. Lower = less leveraged = better. Replaces Debt/Equity (negative equity from buybacks distorts D/E). |
| **Piotroski F-Score** | 15% | A 0-9 checklist scoring profitability, leverage, liquidity, and efficiency trends. Higher = healthier fundamentals. |
| **Accruals** | 5% | (Net Income - Operating Cash Flow) / Total Assets. Lower (more negative) = higher earnings quality (Sloan 1996). |
| **Operating Leverage** | 8% | Degree of Operating Leverage (%Δ EBIT / %Δ Revenue). Lower = more durable earnings (less sensitivity to revenue swings). Banks skip this metric. |
| **Beneish M-Score** | 7% | 8-variable earnings manipulation detection model (Beneish 1999). More negative = lower manipulation risk. Requires ≥5 of 8 variables. Non-bank only. |

**Bank-like stocks:**

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Piotroski F-Score** | 15% | A 0-9 checklist scoring profitability, leverage, liquidity, and efficiency trends. Higher = healthier fundamentals. |
| **Accruals** | 10% | (Net Income - Operating Cash Flow) / Total Assets. Lower (more negative) = higher earnings quality (Sloan 1996). |
| **ROE** | 35% | Return on equity — the key bank profitability metric. Higher = better. |
| **ROA** | 25% | Return on assets — key bank efficiency metric. Higher = better. |
| **Equity Ratio** | 15% | Total equity divided by total assets. Solvency measure — higher = more capital = safer. |

**Why these?** A cheap stock is only a good investment if the underlying business is sound. ROIC is the single best measure of business quality — the ROIC formula deducts only *excess* cash (cash beyond 2% of revenue) from invested capital, preventing cash-rich companies like AAPL or GOOG from showing artificially inflated returns. For banks, ROIC is meaningless (invested capital = deposits + equity), so ROE and ROA replace it. The Piotroski F-Score catches deteriorating businesses by checking 9 binary signals about whether profitability, leverage, and efficiency are improving or declining. Accruals catch companies whose reported earnings aren't backed by real cash.

---

### 3. Growth (13% of final score)

**Question it answers:** *Is this business growing, and can it sustain that growth?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Forward EPS Growth** | 45% | (Forward EPS - Trailing EPS) / Trailing EPS. Denominator floored at $1.00. Clamped to [-75%, +150%]. Higher = faster expected growth. |
| **Revenue Growth** | 25% | Year-over-year revenue increase from financial statements. Higher = growing top line. |
| **Revenue CAGR (3Y)** | 15% | 3-year compound annual revenue growth rate from annual filings. Smooths lumpy single-year revenue growth. |
| **Sustainable Growth** | 15% | ROE × retention rate (1 - dividend payout ratio). Higher = more internally funded growth capacity. |

**Why these?** Growth without overpaying is the sweet spot. Forward EPS Growth gets the most weight because it's forward-looking (the market prices in the future, not the past). The PEG Ratio bridges valuation and growth into a single number — it penalizes stocks with high P/E ratios relative to their growth, preventing the screener from chasing expensive growers. Sustainable Growth acts as a sanity check — if a company is growing faster than its sustainable rate, it may need external financing to keep it up.

---

### 4. Momentum (13% of final score)

**Question it answers:** *Has the market been rewarding this stock recently?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **12-1 Month Return** | 40% | Total price return from 12 months ago to 1 month ago. Skips the most recent month to avoid short-term reversal noise. |
| **6-1 Month Return** | 35% | Total price return from 6 months ago to 1 month ago. Also skips the most recent month. |
| **Jensen's Alpha** | 25% | Risk-adjusted excess return above CAPM prediction. Measures outperformance unexplained by market beta. Uses full 12-month return (no skip-month). |

**Why these?** Decades of academic research (Jegadeesh & Titman, 1993) show that stocks that have gone up tend to keep going up over 3-12 month horizons. The skip-month convention (excluding the most recent month) is the standard academic momentum signal — the last month is excluded because very recent winners tend to experience a brief pullback. Both metrics use calendar-based date targeting instead of fixed index offsets, which ensures consistent lookback periods regardless of holidays or trading day variations.

**Volatility-regime adjustment:** The screener tracks market-wide volatility across runs (stored in `cache/vol_history.csv`). Once 20+ historical observations are available, it classifies the current volatility environment as HIGH, NORMAL, or LOW (using 25th/75th percentile thresholds of historical volatility). In HIGH-vol regimes, momentum weight is reduced by 30% (freed weight goes to Quality + Valuation), because momentum crashes are most common during volatile markets. In LOW-vol regimes, momentum weight is increased by 15% (taken from Valuation), because calm markets are where momentum works best. This adaptive scaling requires at least 20 screener runs before activating.

---

### 5. Risk (10% of final score)

**Question it answers:** *How bumpy is the ride?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Volatility** | 30% | Annualized standard deviation of daily returns over the past year. Lower = smoother ride. |
| **Beta** | 20% | Covariance of stock returns with S&P 500 returns divided by variance of market returns. Requires ≥80% date overlap with market. Lower = less market-driven risk. |
| **Sharpe Ratio** | 15% | (12-month return - risk-free rate) / volatility. Risk-adjusted return per unit of total risk. Higher = more efficient risk-taking. |
| **Sortino Ratio** | 15% | (12-month return - risk-free rate) / downside deviation. Like Sharpe but only penalizes downside volatility. Higher = better downside-adjusted return. |
| **Max Drawdown (1Y)** | 20% | Maximum peak-to-trough decline from cumulative daily return series over the past year. Less negative = smaller worst-case loss. |

**Why these?** All else equal, less volatile stocks are preferable — the "low volatility anomaly" is one of the most robust findings in finance. Volatility and Beta measure total and systematic risk respectively. Sharpe and Sortino Ratios reward stocks that deliver more return per unit of risk (Sortino penalizes only downside volatility, which matters more to investors). Max Drawdown captures worst-case loss — a stock that drops 50% needs a 100% gain to recover. Together these five metrics favor steadier, more risk-efficient companies.

---

### 6. Analyst Revisions (10% of final score)

**Question it answers:** *What do Wall Street analysts think — and are they getting more or less optimistic?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Analyst Surprise** | 38% | Median of (Actual - Estimated EPS) / max(|Estimated|, $0.10) over last 4 quarters. Positive = beat expectations. |
| **Price Target Upside** | 12% | (Mean Analyst Price Target - Current Price) / Current Price. Clamped to [-50%, +100%]. Higher = more analyst optimism. |
| **Earnings Acceleration** | 20% | Difference between most recent quarter's surprise % and prior quarter's surprise %. Positive = accelerating beats, negative = decelerating. Continuous, winsorized at 1st/99th percentiles. |
| **Beat Score** | 20% | Recency-weighted beat score: each of the last 4 quarters' beats weighted by recency (Q1=1, Q2=2, Q3=3, Q4=4). Range 0-10. A stock beating all 4 quarters scores 10; beating only the most recent scores 4. |
| **Short Interest Ratio** | 10% | Days to cover (short interest shares / average daily volume). Lower = less bearish sentiment from short sellers. Contrarian signal. |

**Why these?** Estimate revisions and analyst targets are among the most powerful short-term return predictors. When a company consistently beats earnings estimates, the stock price usually follows — but with a lag, which creates an opportunity. Analyst Surprise gets the highest weight because it's a harder, backward-looking signal with less optimism bias than forward price targets. Earnings Acceleration (a continuous delta, not binary) and Beat Score (recency-weighted, not a simple streak counter) capture the *trajectory* and *consistency* of earnings beats — a company whose surprise % is improving quarter-over-quarter, and which has beaten in recent quarters with higher recency weight, signals genuine fundamental momentum. This category is weighted at only 10% because coverage can be sparse (not all stocks have active analyst coverage), and when coverage drops below usable levels, the weight automatically redistributes to the other categories.

*Note: EPS forecast revision (change in consensus forward EPS over 3-6 months) would be ideal as a fifth metric here but is not feasible with yfinance, which does not provide historical consensus data. Future enhancement: integrate I/B/E/S data from FactSet or Refinitiv.*

---

### 7. Size (5% of final score)

**Question it answers:** *Does this stock benefit from the small-cap premium?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Log Market Cap** | 100% | Negative natural log of market capitalization: -log(marketCap). Smaller companies get higher values. |

**Why this?** The Fama-French SMB (Small Minus Big) factor captures the historical tendency for smaller companies to outperform larger ones over long horizons. Within the S&P 500, this creates a mild tilt toward mid-cap names (which are still large-cap by absolute standards) rather than megacaps. Using the log transform compresses the enormous range of market caps ($2B to $3T+) into a more linear scale that ranks sensibly.

---

### 8. Investment (5% of final score)

**Question it answers:** *Is this company investing conservatively or aggressively expanding its asset base?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Asset Growth** | 100% | Year-over-year change in total assets. Lower = better (conservative investment, Fama-French CMA). |

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
Extreme outliers can distort rankings. The screener clips each metric at the 1st and 99th percentiles — for example, if one company has a Debt/Equity ratio of 50x while the rest are under 5x, that 50x gets clipped down to the 99th percentile value. This prevents a single extreme value from dominating the score.

### Step 3: Rank Within Sectors
Each metric is converted to a **sector-relative percentile** (0-100). A stock's EV/EBITDA isn't compared to all 500 companies — it's compared only to other companies in the same GICS sector (Technology vs. Technology, Energy vs. Energy, etc.). This is critical because a "cheap" utility trades at a very different multiple than a "cheap" tech company. Sector-relative ranking makes apples-to-apples comparisons possible.

For metrics where lower is better (like EV/EBITDA, Debt/Equity, Volatility, P/B, PEG Ratio, Asset Growth), the percentile is flipped so that a higher percentile always means "better."

**Small-sector fallback:** When a sector has fewer than 10 stocks with valid data for a metric, ranking within that tiny group produces noisy percentiles. In these cases, the screener falls back to universe-wide percentile ranking for that metric, which provides a more stable signal than the previous approach of assigning a flat 50th percentile.

**Optional percentile transform:** Currently **disabled** (default). Percentile ranks are used as-is without non-linear transformation.

### Step 4: Combine Into Category Scores
Within each of the 8 categories, the individual metric percentiles are combined using the configured weights. For example, the generic Valuation score is:

```
Valuation = 25% × EV/EBITDA_pct + 45% × FCF_Yield_pct + 20% × Earnings_Yield_pct + 10% × EV/Sales_pct
```

For bank-like stocks, the weights come from the bank-specific weight table instead:

```
Valuation (bank) = 40% × Earnings_Yield_pct + 60% × Price-to-Book_(P/B)_pct
```

**Missing data handling:** When a metric has no data for a particular stock (NaN), that metric is excluded and its weight is redistributed proportionally across the metrics that do have data. This means a stock isn't penalized for a missing metric — it's scored on whatever data is available. If an entire metric is NaN across the full universe (e.g., a data source outage), it is automatically skipped for the category.

This produces 8 category scores (0-100 each).

### Step 5: Combine Into Composite Score
The 8 category scores are combined using the category weights:

```
Raw Composite = 22% × Valuation + 22% × Quality + 13% × Growth + 13% × Momentum + 10% × Risk + 10% × Revisions + 5% × Size + 5% × Investment
```

The same missing-data redistribution logic applies: if a category score is NaN (e.g., all revisions data missing for a stock), its weight is redistributed to available categories rather than producing a NaN composite.

The raw composite is then converted to a cross-sectional percentile rank (0-100), so a score of 95 means "better than 95% of stocks in the universe."

### Step 6: Apply Trap Filters & Rank
After computing composite scores, the screener applies value trap and growth trap filters (see below), then produces the final ranking.

---

## Piotroski Conditional Weighting

The Piotroski F-Score is a broad checklist of financial health signals — but its predictive power varies depending on how expensive a stock is. For cheap stocks (high valuation score), the F-Score is highly predictive: it separates genuinely undervalued companies from deteriorating ones. For expensive stocks (low valuation score), the F-Score is less informative because the market has already priced in quality.

When enabled (current: **on**), the screener reduces the Piotroski F-Score weight by 50% for non-bank stocks with a valuation score below 50 (i.e., the more expensive half of the universe). The freed weight is redistributed equally to ROIC and Gross Profit / Assets, which are more robust quality signals for expensive stocks.

Bank-like stocks are unaffected — their quality weights are already tailored.

---


## Data Quality Safeguards

The screener includes several layers of data quality protection:

- **Denominator floors:** Analyst surprise uses a $0.10 floor on estimated EPS; forward EPS growth uses a $1.00 floor on trailing EPS. These prevent near-zero denominators from producing extreme ratios.
- **Output clamping (configurable):** Forward EPS growth is clamped to [-75%, +150%]; price target upside is clamped to [-50%, +100%]. These bounds are configurable in `config.yaml` under `metric_clamps`. They limit the impact of data anomalies (e.g., GAAP vs. normalized EPS mismatches, extreme analyst targets) while still allowing meaningful differentiation among high-growth stocks.
- **Coverage filter:** Stocks with fewer than 60% of their applicable metrics available are excluded from the ranking entirely.
- **Coverage discount:** Stocks that pass the coverage filter but still have many missing metrics receive a mild composite discount. Below 80% metric coverage, the composite is reduced by up to 15% per unit of coverage gap (e.g., a stock at 56% coverage gets a ~3.6% discount). This prevents stocks with sparse data from ranking artificially high due to weight redistribution concentrating the score on a few favorable metrics. **Currently enabled.**
- **Auto-disable (category-level):** If the Revisions or Investment category has fewer than 30% of its metrics populated, the entire category's weight is zeroed and redistributed proportionally to the remaining categories.
- **Auto-reduce (metric-level):** If any individual metric has more than 70% NaN across the universe (e.g., a data source outage), its weight is automatically set to zero and redistributed within its category.
- **Metric-level alerts:** A warning is printed if any metric has more than 50% missing data across the universe.
- **LTM / MRQ data freshness:** All flow metrics (revenue, net income, EBITDA, cash flow) use LTM (Last Twelve Months = sum of 4 most recent quarters). Balance sheet items use MRQ (Most Recent Quarter). This reduces data staleness from up to 12 months (annual filings) to ~3 months. Falls back to annual filings if quarterly data is unavailable; prior-year comparisons fall back to annual col=1 when quarterly history is insufficient (< 8 quarters).
- **EV cross-validation:** The API-provided Enterprise Value is cross-checked against computed MC + Debt - Cash. If the discrepancy exceeds 10% (or 25% for Financials, whose "debt" includes customer deposits that legitimately diverge from simple EV math), the computed value is used and the ticker is flagged (`_ev_flag`). This catches known yfinance EV parsing bugs (4x+ discrepancy for some tickers).
- **LTM partial annualization tracking:** When only 3 of 4 quarters are available for a flow metric, the screener annualizes (sum × 4/3) but flags the ticker with `_ltm_annualized = True` and records which fields were affected. This transparency lets users know which metrics are based on extrapolated rather than complete data.
- **Channel-stuffing detection:** Compares receivables growth vs. revenue growth. When receivables growth exceeds revenue growth by more than 15 percentage points, the stock is flagged with `_channel_stuffing_flag = True`. This can indicate aggressive revenue recognition or deteriorating collection quality.
- **Beta overlap validation:** Beta computation requires at least 80% date overlap between the stock's daily returns and the S&P 500 market returns. Stocks with insufficient overlap get `beta = NaN` rather than a potentially misleading value. The overlap percentage is recorded in `_beta_overlap_pct`.
- **Data quality log:** Every data issue (missing fields, stale data, rate-limit failures) is logged to `validation/data_quality_log.csv` with ticker, severity, description, and action taken.
- **Structured pipeline logging:** A Python `logging`-based structured logger (`screener.pipeline`) records coverage statistics, filter actions, and scoring stage completions for machine-parseable diagnostics.

---

## Value Trap Detection

A stock can score well on valuation (cheap!) but be cheap for a reason — declining business, negative momentum, or analysts cutting estimates. The screener uses **majority logic (2-of-3)** to flag potential value traps: a stock is flagged only if it falls in the bottom 30% of **at least two** of these three categories:

- Quality Score (floor: 30th percentile)
- Momentum Score (floor: 30th percentile)
- Revisions Score (floor: 30th percentile)

This is more balanced than the alternative "any 1 breach" approach, which flagged roughly 60% of the universe — too aggressive to be useful. The 2-of-3 majority logic catches stocks with genuinely broad weakness while tolerating a single weak dimension (e.g., a quality stock with one bad momentum quarter). About 30% of stocks are typically flagged.

Missing data (NaN) in any of the three dimensions does **not** trigger a value trap flag — missing data is not the same as poor quality. These stocks receive a separate `Insufficient_Data_Flag`.

Each flagged stock also receives a **Value Trap Severity** score (0-100), computed as the average of how far below each threshold the stock falls across the dimensions that triggered the flag. A severity of 80 means the stock is deep in trap territory; a severity of 20 means it barely crossed the thresholds. This provides more granularity than the binary flag alone.

By default, value-trap-flagged stocks are **excluded** from the model portfolio (configurable to flag-only mode).

---

## Growth Trap Detection

The mirror image of a value trap: a stock can score well on growth but be growing unsustainably — high growth with poor quality and/or deteriorating analyst sentiment. The screener uses the same **majority logic (2-of-3)** to flag potential growth traps: a stock is flagged only if **at least two** of these three conditions are met:

- Growth Score **above** the 70th percentile (high growth)
- Quality Score **below** the 35th percentile (low quality)
- Revisions Score **below** the 35th percentile (deteriorating sentiment)

This catches "growth at any price" stocks — companies that are growing fast but burning cash, carrying deteriorating fundamentals, or losing analyst confidence.

Each flagged stock also receives a **Growth Trap Severity** score (0-100), computed as the average of how far above/below each threshold the stock falls across the dimensions that triggered the flag. Higher severity means deeper in trap territory.

By default, growth-trap-flagged stocks are **excluded** from the model portfolio (configurable to flag-only mode).

---

## Portfolio Construction

After scoring and ranking, the screener builds a **model portfolio** from the top-ranked stocks:

- **Number of holdings:** Top 25 stocks (configurable)
- **Weighting:** Risk-parity (inverse-volatility weighting — lower-volatility stocks get more weight)
- **Sector cap:** Maximum 8 stocks from any single sector, to avoid overconcentration
- **Position limits:** No single stock above 5.0%
- **Liquidity filter:** Stocks with less than $10M average daily dollar volume (63-day average) are excluded from the portfolio. Stocks with missing volume data are also excluded (conservative default).
- **Trap exclusions:** Value-trap and growth-trap flagged stocks are excluded (unless configured as flag-only)

If a sector would exceed its cap, the excess stocks are dropped and replaced by the next-highest-ranked stocks from other sectors. Weights are redistributed proportionally.

---

## What Gets Output

The screener produces an **Excel workbook** (`factor_output.xlsx`) with up to 6 sheets:

### Sheet 1: Factor Scores
Every stock in the universe with all raw metrics, 8 category scores, the composite score, rank, value trap flag (with severity 0-100), growth trap flag (with severity 0-100), financial sector caveat flag, bank classification, and bank-specific metrics (P/B, ROE, ROA, Equity Ratio) where applicable. Each stock also carries a data provenance tag (`_data_source`), metric coverage count, and an EPS basis mismatch flag. Score columns use quartile-based coloring (Q1=red, Q2=yellow, Q3=light green, Q4=green) for at-a-glance assessment.

### Sheet 2: Screener Dashboard
The top 50 stocks, formatted for quick review. Includes rank, composite score (quartile-colored), all 8 category scores, and the value trap and growth trap flags with severity scores. Color-coded cells highlight strengths and weaknesses.

### Sheet 3: Model Portfolio
The final portfolio with ticker, sector, composite score, position weights, and portfolio-level statistics (weighted average beta, dividend yield, sector allocation breakdown).

### Sheet 4: DataValidation
The top 10 stocks with raw financial values (market cap, revenue, EPS, etc.) displayed for manual spot-checking. Highlights potential issues including EPS basis mismatches (GAAP vs. normalized), stale data, EV cross-validation discrepancies, LTM partial annualization flags, channel-stuffing flags (receivables growth diverging from revenue growth), and beta overlap warnings. Also includes a sector-median context table showing 25th/median/75th percentile for 8 key metrics across each sector.

### Sheet 5: Weight Sensitivity (when available)
Results of the weight sensitivity analysis. For each factor category, the sheet shows what happens to the top-20 portfolio when that category's weight is perturbed ±5%. Jaccard similarity measures how stable the portfolio is — higher values (≥0.85) mean the ranking is robust to small weight changes. Color-coded: green (≥0.85), yellow (0.70–0.84), red (<0.70).

### Sheet 6: Factor Correlation (when available)
Spearman rank correlation matrix of all 8 category scores across the universe. Highlights potential double-counting: correlations above 0.6 (orange) or 0.8 (red) indicate factor overlap. Useful for understanding effective independent factor count.

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
| **8 factor categories** (Valuation, Quality, Growth, Momentum, Risk, Revisions, Size, Investment) | Captures the 5 Fama-French factors (MktRF, SMB, HML, RMW, CMA) plus momentum and analyst sentiment. Broad coverage reduces reliance on any single factor. |
| **Sector-relative percentiles** (not universe-wide) | A 10x EV/EBITDA is cheap for Tech but expensive for Utilities. Ranking within sectors makes comparisons fair. |
| **Small-sector fallback to universe-wide ranking** | Sectors with <10 stocks produce noisy percentiles. Falling back to universe ranking is more informative than a flat 50th percentile. |
| **Valuation + Quality as the two largest categories** (22% each) | These are the two most robust factors in academic literature. Growth and Momentum get 13% each — they're powerful but noisier. Size and Investment get 5% each as supplementary signals. |
| **FCF Yield as the top valuation metric** (45% weight) | Cash flow is harder to manipulate than earnings. FCF Yield is the purest measure of how much cash a business generates per dollar of value. |
| **Bank-specific metric weights** | EV/EBITDA, ROIC, and D/E are meaningless for banks. P/B, ROE, ROA, and Equity Ratio are the standard bank analysis toolkit. |
| **ROIC excess cash deduction** (cash - 2% revenue) | Deducting ALL cash inflates ROIC for cash-rich companies (e.g. AAPL, GOOG). Keeping 2% of revenue as operating cash provides a more accurate invested capital base. |
| **ROIC tax-loss handling** (0% tax rate when pretax < 0) | Companies with negative pretax income are in a tax-loss position and would not pay tax. Using the statutory 21% rate would create a fictional tax charge that understates NOPAT. |
| **EV cross-validation** (API vs MC+Debt-Cash) | yfinance has known EV parsing bugs (4x+ discrepancy for some tickers). When the API-provided EV differs from the computed value by more than 10% (25% for Financials), the computed value is used and the discrepancy is flagged. Financials use a wider tolerance because their "debt" includes deposits and other liabilities that structurally diverge from simple EV math. |
| **Momentum skip-month** (12-1 and 6-1, not 12-0) | The most recent month's return tends to reverse. Skipping it improves signal quality (standard in academic momentum literature). |
| **Calendar-based lookbacks** | Using calendar dates (e.g., 182 days ago) instead of fixed index offsets ensures consistent lookback periods regardless of holidays. |
| **Denominator floors** ($0.10 for surprise, $1.00 for EPS growth) | Near-zero denominators produce extreme ratios that dominate rankings. Floors bound the maximum possible ratio. |
| **Winsorization at 1%/99%** | Prevents a single extreme data point from blowing up the rankings. Conservative clip — keeps 98% of the distribution intact. |
| **Value trap 2-of-3 majority logic** | OR logic (any 1 breach) flagged ~60% of the universe — too aggressive. Majority logic catches genuinely weak stocks while tolerating one bad dimension. |
| **Growth trap 2-of-3 majority logic** | Mirror of value trap for the opposite scenario. Catches high-growth stocks with poor quality and/or deteriorating sentiment. |
| **Liquidity filter** ($10M daily dollar volume) | Ensures portfolio stocks are tradeable at scale. NaN volume is excluded conservatively. |
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

The screener answers one question: **"Which S&P 500 stocks look best when measured across valuation, quality, growth, momentum, risk, revisions, size, investment — all at once?"**

It does this by:
1. Pulling financial data for ~500 stocks from Yahoo Finance
2. Computing up to 34 financial metrics across 8 categories (30 generic + 4 bank-specific, depending on company type)
3. Ranking each metric within its sector (so comparisons are fair)
4. Weighting and combining into a single 0-100 composite score (with bank-specific weights for financial companies and conditional Piotroski weighting)
5. Flagging potential value traps and growth traps (2-of-3 majority logic)
6. Applying a liquidity filter to ensure tradeability
7. Building a diversified model portfolio from the top picks

The result is a disciplined, repeatable, multi-dimensional ranking that avoids the tunnel vision of looking at any single metric in isolation.
