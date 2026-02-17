# Multi-Factor Stock Screener — How It Works

**A plain-language guide to what the screener does, why it does it, and how it arrives at its picks.**

---

## What Is This?

This is a quantitative stock screener. It takes every company in the S&P 500 (roughly 500 stocks), measures each one across 17 financial metrics, combines those measurements into a single composite score (0-100), and ranks the entire universe from best to worst. The top-ranked stocks form a model portfolio.

The core idea: no single number tells you whether a stock is a good investment. A stock can look cheap but be cheap for a reason (declining business, high risk). By scoring across multiple independent dimensions — valuation, quality, growth, momentum, risk, and analyst sentiment — the screener surfaces companies that are strong across the board, not just on one axis.

---

## Where Does the Data Come From?

All data is pulled from **Yahoo Finance** via the `yfinance` Python library. For each stock, the screener fetches:

- **Financial statements** — income statement, balance sheet, and cash flow statement (annual)
- **Price history** — 13 months of daily closing prices
- **Summary statistics** — market cap, enterprise value, P/E ratios, EPS estimates, analyst targets
- **Prior-year financials** — for year-over-year comparisons (profitability trends, debt changes)

The S&P 500 member list is pulled from Wikipedia's regularly updated table. A local backup (`sp500_tickers.json`) is used if the network is unavailable.

---

## The Six Factor Categories

Every stock is evaluated in six categories. Each category captures a different dimension of investment merit.

### 1. Valuation (25% of final score)

**Question it answers:** *Is this stock priced attractively relative to what the business generates?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **FCF Yield** | 40% | Free cash flow divided by enterprise value. How much cash the business generates per dollar of total value (equity + debt). Higher = cheaper. |
| **EV/EBITDA** | 25% | Enterprise value divided by earnings before interest, taxes, depreciation, and amortization. A capital-structure-neutral price tag. Lower = cheaper. |
| **Earnings Yield** | 20% | Trailing EPS divided by share price (the inverse of the P/E ratio). Higher = cheaper. |
| **EV/Sales** | 15% | Enterprise value divided by revenue. Useful for comparing companies with different margin profiles. Lower = cheaper. |

**Why these?** Traditional P/E ratios are distorted by capital structure, one-time charges, and accounting choices. Enterprise value-based metrics strip away those distortions. FCF Yield gets the heaviest weight because cash flow is the hardest number for management to manipulate — it's cash in the door.

---

### 2. Quality (25% of final score)

**Question it answers:** *Is this a well-run business with durable competitive advantages?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **ROIC** | 30% | Return on Invested Capital — how much profit the business earns on the money invested in it (equity + debt - cash). Higher = better use of capital. |
| **Gross Profit / Assets** | 25% | Gross profit divided by total assets. Measures asset-light profitability, a hallmark of businesses with pricing power. |
| **Debt/Equity** | 20% | Total debt divided by shareholder equity. Lower = less financial leverage and risk. |
| **Piotroski F-Score** | 15% | A 0-9 checklist scoring profitability, leverage, liquidity, and efficiency trends. Higher = healthier fundamentals. |
| **Accruals** | 10% | (Net Income - Operating Cash Flow) / Total Assets. Lower (more negative) = higher earnings quality, because more of the reported profit is backed by actual cash. |

**Why these?** A cheap stock is only a good investment if the underlying business is sound. ROIC is the single best measure of business quality — Warren Buffett's favorite metric. The Piotroski F-Score catches deteriorating businesses by checking 9 binary yes/no signals about whether profitability, leverage, and efficiency are improving or declining. Accruals catch companies whose reported earnings aren't backed by real cash.

---

### 3. Growth (15% of final score)

**Question it answers:** *Is this business growing, and can it sustain that growth?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Forward EPS Growth** | 35% | Analyst consensus for next year's earnings vs. trailing earnings. Higher = faster expected growth. |
| **Revenue Growth** | 30% | Year-over-year revenue increase. Higher = growing top line. |
| **PEG Ratio** | 20% | P/E ratio divided by the year-over-year earnings growth rate (sourced from Yahoo Finance). Answers "how much am I paying per unit of growth?" A PEG of 1.0 means the stock is fairly valued relative to its growth; below 1.0 suggests undervalued growth; above 2.0 suggests expensive growth. Only calculated when earnings growth is positive. Lower = better. |
| **Sustainable Growth** | 15% | ROE times the earnings retention rate (what % of profits the company reinvests rather than paying as dividends). Estimates how fast the company can grow without issuing new debt or equity. |

**Why these?** Growth without overpaying is the sweet spot. Forward EPS Growth gets the most weight because it's forward-looking (the market prices in the future, not the past). The PEG Ratio bridges valuation and growth into a single number — it penalizes stocks with high P/E ratios relative to their growth, preventing the screener from chasing expensive growers. Sustainable Growth acts as a sanity check — if a company is growing faster than its sustainable rate, it may need external financing to keep it up.

---

### 4. Momentum (15% of final score)

**Question it answers:** *Has the market been rewarding this stock recently?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **12-1 Month Return** | 50% | Total price return from 12 months ago to 1 month ago. Skips the most recent month to avoid short-term reversal noise. |
| **6-Month Return** | 50% | Total price return over the last 6 months. Captures medium-term trend. |

**Why these?** Decades of academic research (Jegadeesh & Titman, 1993) show that stocks that have gone up tend to keep going up over 3-12 month horizons. The 12-1 month variant (skipping the most recent month) is the standard academic momentum signal — the last month is excluded because very recent winners tend to experience a brief pullback.

---

### 5. Risk (10% of final score)

**Question it answers:** *How bumpy is the ride?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Volatility** | 60% | Annualized standard deviation of daily returns over the past year. Lower = smoother ride. |
| **Beta** | 40% | Sensitivity to the overall market (S&P 500). A beta of 1.0 means the stock moves in lockstep with the market. Lower = less market-driven risk. |

**Why these?** All else equal, less volatile stocks are preferable — the "low volatility anomaly" is one of the most robust findings in finance. Beta measures systematic risk (how much your stock drops when the market drops). Both metrics favor steadier, less speculative companies.

---

### 6. Analyst Revisions (10% of final score)

**Question it answers:** *Are Wall Street analysts becoming more or less optimistic?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **EPS Revision Ratio** | 40% | Ratio of upward to downward EPS estimate revisions in the last 90 days. |
| **EPS Estimate Change** | 35% | Percentage change in consensus EPS estimate over the last 90 days. |
| **Analyst Surprise** | 25% | How much the last reported earnings beat or missed the consensus estimate. Positive = beat. |

**Why these?** Estimate revisions are among the most powerful short-term return predictors. When analysts raise their earnings forecasts, the stock price usually follows — but with a lag, which creates an opportunity. This category is weighted at only 10% because coverage is sparse (not all stocks have active analyst coverage).

---

## How the Score Is Calculated

The scoring pipeline has five steps:

### Step 1: Collect Raw Data
For each of the ~500 stocks, the screener pulls financial statements, price data, and analyst estimates from Yahoo Finance. Data is cached locally (refreshed daily for prices, weekly for fundamentals) to avoid unnecessary API calls.

### Step 2: Clean the Data (Winsorization)
Extreme outliers can distort rankings. The screener clips each metric at the 1st and 99th percentiles — for example, if one company has a Debt/Equity ratio of 50x while the rest are under 5x, that 50x gets clipped down to the 99th percentile value. This prevents a single extreme value from dominating the score.

### Step 3: Rank Within Sectors
Each metric is converted to a **sector-relative percentile** (0-100). A stock's EV/EBITDA isn't compared to all 500 companies — it's compared only to other companies in the same GICS sector (Technology vs. Technology, Energy vs. Energy, etc.). This is critical because a "cheap" utility trades at a very different multiple than a "cheap" tech company. Sector-relative ranking makes apples-to-apples comparisons possible.

For metrics where lower is better (like EV/EBITDA, Debt/Equity, Volatility), the percentile is flipped so that a higher percentile always means "better."

### Step 4: Combine Into Category Scores
Within each of the six categories, the individual metric percentiles are combined using the configured weights. For example, the Valuation score is:

```
Valuation = 40% * FCF_Yield_pct + 25% * EV_EBITDA_pct + 20% * Earnings_Yield_pct + 15% * EV_Sales_pct
```

This produces six category scores (0-100 each).

### Step 5: Combine Into Composite Score
The six category scores are combined using the category weights:

```
Raw Composite = 25% * Valuation + 25% * Quality + 15% * Growth
              + 15% * Momentum + 10% * Risk + 10% * Revisions
```

The raw composite is then normalized to a 0-100 scale (min-max normalization across the full universe), so the best stock scores 100 and the worst scores 0.

---

## Value Trap Detection

A stock can score well on valuation (cheap!) but be cheap for a reason — declining business, negative momentum, or analysts cutting estimates. The screener flags potential "value traps" when a stock falls in the **bottom 30%** of any of these three categories:

- Quality Score
- Momentum Score
- Revisions Score

Flagged stocks aren't automatically removed (that's configurable), but they're visually marked in the output so you can investigate before buying.

---

## Portfolio Construction

After scoring and ranking, the screener builds a **model portfolio** from the top-ranked stocks:

- **Number of holdings:** Top 25 stocks (configurable)
- **Weighting:** Equal weight (each stock gets ~4%) or risk-parity (lower-volatility stocks get more weight)
- **Sector cap:** Maximum 8 stocks from any single sector, to avoid overconcentration
- **Position limits:** No single stock below 2% or above 5%

If a sector would exceed its cap, the excess stocks are dropped and replaced by the next-highest-ranked stocks from other sectors. Weights are redistributed proportionally.

---

## What Gets Output

The screener produces an **Excel workbook** (`factor_output.xlsx`) with three sheets:

### Sheet 1: Factor Scores
Every stock in the universe with all 17 raw metrics, 6 category scores, the composite score, rank, and value trap flag. This is the full dataset for analysis.

### Sheet 2: Screener Dashboard
The top 50 stocks, formatted for quick review. Includes rank, composite score, all six category scores, and the value trap flag. Color-coded cells highlight strengths and weaknesses.

### Sheet 3: Model Portfolio
The final portfolio with ticker, sector, composite score, position weights, and portfolio-level statistics (weighted average beta, dividend yield, sector allocation breakdown).

A Parquet cache file is also saved for programmatic access.

---

## Key Design Decisions & Why

| Decision | Why |
|----------|-----|
| **Sector-relative percentiles** (not universe-wide) | A 10x EV/EBITDA is cheap for Tech but expensive for Utilities. Ranking within sectors makes comparisons fair. |
| **Equal weighting on Valuation + Quality** (25% each) | These are the two most robust factors in academic literature. Growth and Momentum get 15% each — they're powerful but noisier. |
| **FCF Yield as the top valuation metric** (40% weight) | Cash flow is harder to manipulate than earnings. FCF Yield is the purest measure of how much cash a business generates per dollar of value. |
| **Momentum skip-month** (12-1, not 12-0) | The most recent month's return tends to reverse. Skipping it improves signal quality (standard in academic momentum literature). |
| **Winsorization at 1%/99%** | Prevents a single extreme data point from blowing up the rankings. Conservative clip — keeps 98% of the distribution intact. |
| **Value trap flags** | Cheap stocks with deteriorating fundamentals or negative momentum are statistically more likely to keep falling. The flag is a safety net. |
| **Risk-adjusted scoring (low vol = good)** | The low-volatility anomaly: less volatile stocks have historically delivered higher risk-adjusted returns. The screener tilts toward lower-risk names. |
| **Piotroski normalization** | A company with 3 available data points scoring 3/3 shouldn't lose to a company with 9 data points scoring 7/9. The F-Score is normalized to a 0-9 scale based on however many tests were actually testable. |
| **PEG Ratio (lower = better)** | PEG bridges valuation and growth in one number. A stock growing earnings at 20%/year with a P/E of 15 (PEG = 0.75) is more attractive than one growing at 10% with a P/E of 30 (PEG = 3.0). Only meaningful for positive growth — negative-growth stocks get NaN. |

---

## Limitations to Be Aware Of

1. **Data source:** All data comes from Yahoo Finance (free, unofficial API). Occasional field name changes or missing data are handled gracefully (the screener returns NaN and continues), but the data quality is not institutional-grade.

2. **Point-in-time:** The screener uses the latest available financial data. It does not reconstruct what was known at a past date, which means backtests carry look-ahead bias for fundamental metrics.

3. **Analyst coverage:** The Revisions category relies on analyst estimate data, which is sparse. When coverage drops below 30%, the screener auto-disables the Revisions category and redistributes its weight to the other five categories.

4. **Rebalance frequency:** The model portfolio is a snapshot. It should be re-run at the configured frequency (monthly or quarterly) to stay current.

5. **Not investment advice:** This is a screening tool, not a recommendation engine. The output is a ranked list to narrow your research — not a list of stocks to blindly buy.

---

## Quick Start

```bash
# Run the screener on the full S&P 500
py run_screener.py --refresh

# Run on specific tickers only
py run_screener.py --refresh --tickers AAPL,MSFT,GOOGL,AMZN,META

# Use cached data (no new downloads)
py run_screener.py

# Output: factor_output.xlsx (3 sheets)
```

---

## Summary

The screener answers one question: **"Which S&P 500 stocks look best when measured across valuation, quality, growth, momentum, risk, and analyst sentiment — all at once?"**

It does this by:
1. Pulling financial data for ~500 stocks from Yahoo Finance
2. Computing 17 financial metrics across 6 categories
3. Ranking each metric within its sector (so comparisons are fair)
4. Weighting and combining into a single 0-100 composite score
5. Flagging potential value traps
6. Building a diversified model portfolio from the top picks

The result is a disciplined, repeatable, multi-dimensional ranking that avoids the tunnel vision of looking at any single metric in isolation.
