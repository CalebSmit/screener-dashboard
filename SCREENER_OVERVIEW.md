# Multi-Factor Stock Screener — How It Works

**A plain-language guide to what the screener does, why it does it, and how it arrives at its picks.**

---

## What Is This?

This is a quantitative stock screener. It takes every company in the S&P 500 (roughly 500 stocks), measures each one across up to 31 financial metrics, combines those measurements into a single composite score (0-100), and ranks the entire universe from best to worst.

Not every stock sees all 31 metrics. The screener uses 27 generic metrics for most stocks and a separate set of 4 bank-specific metrics for financial companies (banks, insurers, credit companies). In practice, any individual stock is scored on about 27 metrics — the set just differs depending on whether the company is a bank or not. The full metric registry (`METRIC_COLS`) has 46 entries: 31 carry scoring weight today (27 generic + 4 bank-specific) plus 15 candidate metrics held at weight 0 that the self-improving engine may activate if they demonstrate predictive power.

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
| **ROIC** | 29% | Return on Invested Capital — NOPAT divided by invested capital (equity + debt - excess cash). Excess cash is cash beyond 2% of revenue. Tax rate: actual effective rate (clamped 0-50%) when pretax income is positive; 0% for tax-loss positions (negative pretax); 21% default when data is missing. Higher = better use of capital. |
| **Gross Profit / Assets** | 22% | Gross profit divided by total assets. Measures asset-light profitability (Novy-Marx quality factor). |
| **Net Debt / EBITDA** | 20% | (Total Debt - Cash) / EBITDA. Measures leverage relative to earnings power. Lower = less leveraged = better. Replaces Debt/Equity (negative equity from buybacks distorts D/E). |
| **Piotroski F-Score** | 16% | A 0-9 checklist scoring profitability, leverage, liquidity, and efficiency trends. Higher = healthier fundamentals. |
| **Accruals** | 5% | (Net Income - Operating Cash Flow) / Total Assets. Lower (more negative) = higher earnings quality (Sloan 1996). |
| **Beneish M-Score** | 8% | 8-variable earnings manipulation detection model (Beneish 1999). More negative = lower manipulation risk. Requires ≥5 of 8 variables. Non-bank only. |

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
| **Forward EPS Growth** | 45% | Expected EPS over the next 12 months (current- and next-fiscal-year consensus, weighted by the months left in the current year - MSCI's construction) versus the last four reported quarters, on the same basis. Denominator floored at $1.00. Clamped to [-75%, +150%]. Higher = faster expected growth. Since 2026-10-09; before, it compared a fiscal year 13-24 months out with GAAP trailing EPS. |
| **Revenue Growth** | 25% | Year-over-year revenue increase from financial statements. Higher = growing top line. |
| **Revenue CAGR (3Y)** | 15% | 3-year compound annual revenue growth rate from annual filings. Smooths lumpy single-year revenue growth. |
| **Sustainable Growth** | 15% | ROE × retention rate (1 - dividend payout ratio). Higher = more internally funded growth capacity. |

**Why these?** Forward EPS Growth gets the most weight because it's forward-looking (the market prices in the future, not the past). Revenue growth and the three-year revenue CAGR measure what has already happened, at two horizons. Sustainable Growth acts as a sanity check — if a company is growing faster than its sustainable rate, it may need external financing to keep it up. The PEG ratio is computed and shown but carries no weight: it divides a valuation by a growth rate, so it would count Valuation a second time inside Growth.

---

### 4. Momentum (13% of final score)

**Question it answers:** *Has the market been rewarding this stock recently?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **12-1 Month Return** | 40% | Total price return from 12 months ago to 1 month ago. Skips the most recent month to avoid short-term reversal noise. |
| **6-1 Month Return** | 35% | Total price return from 6 months ago to 1 month ago. Also skips the most recent month. |
| **Jensen's Alpha** | 25% | Risk-adjusted excess return above CAPM prediction. Measures outperformance unexplained by market beta. Uses full 12-month return (no skip-month). |

**Why these?** Decades of academic research (Jegadeesh & Titman, 1993) show that stocks that have gone up tend to keep going up over 3-12 month horizons. The skip-month convention (excluding the most recent month) is the standard academic momentum signal — the last month is excluded because very recent winners tend to experience a brief pullback. Both metrics use calendar-based date targeting instead of fixed index offsets, which ensures consistent lookback periods regardless of holidays or trading day variations.

**Momentum regime rule - currently off.** Momentum strategies crash most often in volatile markets (Daniel & Moskowitz 2016), and scaling momentum down when its own volatility is high improves it (Barroso & Santa-Clara 2015). Until 2026-10-09 the screener tried to do this, but the input it used was the spread of the momentum score across stocks (`factor_vol_history.csv`), which is fixed by the percentile construction and does not measure market volatility: it called 30 of 33 runs "low volatility" and raised momentum's weight most days. The rule is switched off until it is rebuilt on a real volatility measure; the spread is still recorded each run.

---

### 5. Risk (10% of final score)

**Question it answers:** *How bumpy is the ride?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Volatility** | 42.86% | Annualized standard deviation of daily returns over about 13 months of trading (the same price history the momentum signals use). Lower = smoother ride. |
| **Beta** | 28.57% | Covariance of stock returns with S&P 500 returns divided by variance of market returns. Requires ≥80% date overlap with market. Lower = less market-driven risk. |
| **Max Drawdown (13M)** | 28.57% | Largest peak-to-trough fall in the closing price over about 13 months. Less negative = smaller worst-case loss. |

**Why these?** All else equal, less volatile stocks are preferable — the "low volatility anomaly" is one of the most robust findings in finance. Volatility measures total risk, Beta measures systematic risk, and Max Drawdown captures worst-case loss — a stock that drops 50% needs a 100% gain to recover. All three are *dispersion* measures: they describe how much a stock moves, not how well it did.

**Why not Sharpe and Sortino?** They were scored here until 2026-09-02, at 15% each. Both are `(12-month return − risk-free rate) ÷ some measure of dispersion`, so they share their numerator with the momentum signal. Across the S&P 500 the spread in returns is far wider than the spread in volatility, so the numerator dominates: measured on the published payload, Sharpe correlates **+0.944** with the 12-1 month return but only **+0.025** with volatility. Scoring them inside Risk meant a stock was rated safer because it had gone up — which pushed the Risk and Momentum category scores to a **+0.516** correlation, the highest of any pair in the screener. Removing them drops that to **+0.150**. Both ratios are still computed and shown on each stock's detail page; they are simply no longer scored as risk. See `METHODOLOGY_CHANGELOG.md` 2026-09-02.

**Why not operating leverage?** It was 8% of Quality until 2026-10-09, scored lower-is-better as "more durable earnings". As built it was one year's percentage change in operating profit divided by one year's percentage change in revenue, and that ratio does not measure cost structure: when profit and revenue move in opposite directions it goes negative, and on the 2026-10-09 run **85 of its 95 negative values were companies whose revenue grew while operating profit fell** - shrinking margins - which the score ranked at the 84th percentile of their sectors. A small revenue change also inflates it. The research points the other way too: firms with more operating leverage have historically earned *higher* returns as compensation for the risk (Novy-Marx 2011), and neither MSCI's nor AQR's published quality definitions use it - both measure durability as how variable earnings have been over several years. Its weight went to the other Quality metrics in proportion; it is still computed and shown. See `research/2026-10-09-operating-leverage.md`.

---

### 6. Analyst Revisions (10% of final score)

**Question it answers:** *What do Wall Street analysts think — and are they getting more or less optimistic?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **FY1 EPS Revision (3-month)** | 48% | Change in the consensus current-fiscal-year EPS estimate over the last 90 days, divided by the share price — so it reads in **basis points of price** and is comparable across a $20 stock and a $400 one. Positive = analysts have raised their forecast. This is the category's only true *revision* metric: it measures analysts changing their minds, not companies beating a past estimate. |
| **Analyst Surprise** | 15% | Median of (Actual - Estimated EPS) / max(|Estimated|, $0.10) over last 4 quarters. Positive = beat expectations. |
| **Price Target Upside** | 13.5% | (Mean Analyst Price Target - Current Price) / Current Price. Clamped to [-50%, +100%]. Higher = more analyst optimism. |
| **Beat Score** | 10% | Recency-weighted beat score: each of the last 4 quarters' beats weighted by recency (Q1=1, Q2=2, Q3=3, Q4=4). Range 0-10. A stock beating all 4 quarters scores 10; beating only the most recent scores 4. |
| **Short Interest Ratio** | 13.5% | Days to cover (short interest shares / average daily volume). Lower = less bearish sentiment from short sellers. Contrarian signal. |

**Why these?** Estimate revisions and analyst targets are among the most powerful short-term return predictors. **FY1 EPS Revision (3-month) gets the highest weight** because it is the one metric here that measures what the category is named for — analysts revising their forecasts. Chan, Jegadeesh & Lakonishok (1996, *Journal of Finance* 51(5)) found the analyst-revision leg of earnings momentum to be the strongest of the three they tested, a **+7.7% six-month decile spread** on IBES data 1977–93.

The surprise family — Analyst Surprise and Beat Score — is *backward*-looking: it records companies beating a past estimate and bets that the price keeps drifting afterwards. That drift is what the literature calls post-earnings-announcement drift, and Martineau (2022, *Critical Finance Review* 11(4)) finds it has been **absent in large caps since 2006**, with a significantly *negative* coefficient over 2016–19. Since this is an S&P 500 screener, that is exactly this universe — which is why the surprise family was cut from 78% of the category to 45% on 2026-09-10, and to 25% on 2026-10-09 when Earnings Acceleration (the latest surprise minus the one before) left the score: it marked a stock down for having beaten the previous quarter, though surprises tend to repeat. The revision metric now carries nearly half the category.

This category is weighted at only 10% of the composite because coverage can be sparse (not all stocks have active analyst coverage), and when coverage drops below usable levels, the weight automatically redistributes to the other categories.

*Note on the limits of this data: yfinance's estimate history reaches back only **90 days**, so the screener can see a one-quarter revision but not whether a revision trend has **persisted**. Chan, Jegadeesh & Lakonishok's strongest result used a six-month window, which remains out of reach without a paid consensus feed (FactSet, Refinitiv I/B/E/S). The metric itself is not out of reach and has been live since 2026-09-10 — an earlier version of this page said otherwise.*

---

### 7. Size (5% of final score)

**Question it answers:** *Does this stock benefit from the small-cap premium?*

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Log Market Cap** | 100% | Negative natural log of market capitalization: -log(marketCap). Smaller companies get higher values. Scored by sector percentile rank, which the log does not affect - see the note below. |

**Why this?** The Fama-French SMB (Small Minus Big) factor captures the historical tendency for smaller companies to outperform larger ones over long horizons. Within the S&P 500 this tilts toward mid-cap names (still large-cap by absolute standards) rather than megacaps.

**What the log does, and what it does not do.** The metric is stored as `-log(marketCap)`, but no stock's score depends on the log. Every metric here is converted to a **sector percentile rank** (see "How scoring works" below), and a rank is unchanged by any transformation that preserves order. Measured on the live universe, `rank(-log mcap)`, `rank(-mcap)` and `rank(-sqrt mcap)` are identical to ten decimal places. The log makes the stored number easier to read; it does not compress the tilt.

**So how strong is the tilt?** Stronger than the metric's name suggests, which is worth stating plainly. Because the score is linear in rank, ordinary market-cap gaps become large score gaps: on the current universe CAT ($364B) scores 1 and UAL ($35B) scores 59, so a 10x cap ratio becomes a 57-point gap. For comparison, MSCI's Low Size indexes weight holdings in proportion to 1/ln(mcap), which across this same universe turns a **798x** spread in market cap into a **1.295x** spread in weight. This screener runs an equal-weight-style size tilt, not a log-compressed one. That is a defensible design - it is close to the bet the S&P 500 Equal Weight index makes - but it is a *stronger* bet than the name implies, and holding it to 5% of the composite is what keeps it proportionate.

**Known weakness.** No published study establishes a size premium *within* the largest two market-cap deciles, which is the entire S&P 500. Applying SMB here extrapolates from research run on much broader universes, and the payoff is regime-dependent: S&P 500 Equal Weight has returned roughly +63 bps/yr since 1990 but trailed cap weighting by about 32% over 2023-2025. Detail and citations: `research/2026-08-31-size-factor-in-a-large-cap-universe.md`.

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
For each of the ~500 stocks, the screener pulls quarterly financial statements, price data, earnings history, and analyst estimates from Yahoo Finance. Flow metrics (income statement and cash flow) use **LTM** (Last Twelve Months = sum of 4 most recent quarters); balance sheet items use **MRQ** (Most Recent Quarter). Falls back to annual filings if quarterly data is unavailable. Enterprise Value is cross-validated against computed MC + Debt - Cash; discrepancies > 10% (25% for Financials) trigger automatic correction.

Several inputs arrive from **two places** — Yahoo's summary fields and the filed statements — and the screener prefers one but uses the other when the first is missing: total debt and cash for Enterprise Value prefer the summary figure (it matches Yahoo's own EV definition), while invested capital for ROIC prefers the balance sheet (so equity, debt and cash come from one filing). Since 2026-09-25 those fallbacks actually fire; before that a bug meant they never did, and three S&P 500 companies lost a metric on every run despite the data being present. See `METHODOLOGY_CHANGELOG.md` 2026-09-25. Data is cached locally in Parquet format (refreshed daily for prices, weekly for fundamentals) to avoid unnecessary API calls. Cache files are config-aware — changing weights or settings automatically invalidates stale caches.

### Step 2: Flag Outliers (but do not change them)
Every metric below is scored by its **rank** within its sector, and a rank does not care how far away an outlier is — only that it is last. A company with a Debt/Equity of 50x when everyone else is under 5x ranks worst either way. So the screener does **not** clip extreme values: it records them in the data-quality log (the tails beyond the 1st and 99th percentiles) and scores the number it actually fetched.

Until 2026-09-01 it did clip them, and that was a mistake in two directions. Clipping could not improve a single ranking, because ranking is unaffected by it. What it could do — and did — was flatten several companies onto one identical number, which then made them tie in the ranking, and publish that clipped number as the company's real figure. On the last run before the fix, six companies were all shown with a market capitalisation of $2,802B; Nvidia's true figure was $5,331B.

### Step 3: Rank Within Sectors
Each metric is converted to a **sector-relative percentile** (0-100). A stock's EV/EBITDA isn't compared to all 500 companies — it's compared only to other companies in the same GICS sector (Technology vs. Technology, Energy vs. Energy, etc.). This is critical because a "cheap" utility trades at a very different multiple than a "cheap" tech company. Sector-relative ranking makes apples-to-apples comparisons possible.

For metrics where lower is better (like EV/EBITDA, Debt/Equity, Volatility, P/B, PEG Ratio, Asset Growth), the percentile is flipped so that a higher percentile always means "better." The percentile uses the midpoint rule - the k-th lowest of n stocks scores (k - 0.5) / n x 100 - so a metric averages exactly 50 whichever way it points. (Until 2026-10-09 it was k / n, which averaged 50 + 50/n for higher-is-better metrics and 50 - 50/n once flipped, a tilt that grew in small sectors.)

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

**The weights above are the configured defaults, and an individual run may not use them.** Two rules move them, both described in this document:

1. **The momentum regime rule**, when switched on, changes the momentum weight for the whole run (see the Momentum section). It is off since 2026-10-09.
2. **Missing-data redistribution** changes them for one stock, whenever a category could not be scored for it.

So a stock's momentum score may be multiplied by something other than the 13% printed above. Rather than ask you to take that on trust, the dashboard's stock drilldown shows **the weight each score was actually multiplied by**, and explains any gap against this page — every row there is an equation you can check with a calculator. The run's own weights are also written to `runs/<run_id>/effective_weights.json`.

**The composite is cardinal, and it is not a percentile.** The weighted average above is kept as a 0-100 score with its magnitude intact, and that score is the ranking key — so a stock twenty points clear of the field and one a tenth of a point clear are not both reported as 100. One adjustment is applied after that weighted average: a stock below 80% metric coverage has its composite multiplied by `1 - ((80% - coverage) x 15%)` - the coverage discount described under Data Quality Safeguards below - so for those stocks the composite sits slightly below the sum of the category contributions. **"Coverage" here means the share of the metrics that carry weight in that stock's score** - 24 for a bank-like stock and 27 for every other; a metric with no weight (a candidate, or one shown for reference) cannot lower it (since 2026-10-09). The drilldown's "Metrics: n/m" badge and its "The score rests on n of m metrics" sentence read **this same figure**, taken from the engine rather than recounted (until 2026-10-07 they counted a fixed 18-metric list, which made 62 stocks look under-covered when only 3 were discounted). The drilldown shows the discount as its own line, so the category points, the discount and the composite add up on screen. The universe percentile is a **separate** column, `Composite_Pct` (`rank(pct=True) * 100`, "better than X% of stocks"), carried for display only. So do not read a composite of 95 as "better than 95% of the universe" — read it as 95 points out of 100. See Limitation 8.

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
- **Coverage filter:** Stocks with fewer than 60% of the metrics that carry weight in their score available are excluded from the ranking entirely - the same coverage the composite's discount reads.
- **Coverage discount:** Stocks that pass the coverage filter but still have many missing metrics receive a mild composite discount. Below 80% metric coverage, the composite is reduced by up to 15% per unit of coverage gap (e.g., a stock at 56% coverage gets a ~3.6% discount). This prevents stocks with sparse data from ranking artificially high due to weight redistribution concentrating the score on a few favorable metrics. **Currently enabled.**
- **Auto-disable (category-level):** If the Revisions or Investment category has fewer than 30% of its metrics populated, the entire category's weight is zeroed and redistributed proportionally to the remaining categories.
- **Auto-reduce (metric-level):** If any individual metric has more than 70% NaN across the universe (e.g., a data source outage), its weight is automatically set to zero and redistributed within its category.
- **Metric-level alerts:** A warning is printed if any metric has more than 50% missing data across the universe.
- **LTM / MRQ data freshness:** All flow metrics (revenue, net income, EBITDA, cash flow) use LTM (Last Twelve Months = sum of 4 most recent quarters). Balance sheet items use MRQ (Most Recent Quarter). This reduces data staleness from up to 12 months (annual filings) to ~3 months. Falls back to annual filings if quarterly data is unavailable; prior-year comparisons fall back to annual col=1 when quarterly history is insufficient (< 8 quarters).
- **EV cross-validation:** The API-provided Enterprise Value is cross-checked against computed MC + Debt - Cash. If the discrepancy exceeds 10% (or 25% for Financials, whose "debt" includes customer deposits that legitimately diverge from simple EV math), the computed value is used and the ticker is flagged (`_ev_flag`). This catches known yfinance EV parsing bugs (4x+ discrepancy for some tickers).
- **LTM partial annualization tracking:** When only 3 of 4 quarters are available for a flow metric, the screener annualizes (sum × 4/3) but flags the ticker with `_ltm_annualized = True` and records which fields were affected. This transparency lets users know which metrics are based on extrapolated rather than complete data.
- **Channel-stuffing detection:** Compares receivables with revenue over the last fiscal year, both from the same two annual statements, using Beneish's days-sales-in-receivables index (receivables / revenue, over the prior year's). At **1.465 or more** - the average among the earnings manipulators in Beneish (1999), against 1.031 among the rest - the stock is flagged with `_channel_stuffing_flag = True`. This can indicate aggressive revenue recognition or deteriorating collection quality. Not applied to banks and insurers.
- **Beta overlap validation:** Beta computation requires at least 80% date overlap between the stock's daily returns and the S&P 500 market returns. Stocks with insufficient overlap get `beta = NaN` rather than a potentially misleading value. The overlap percentage is recorded in `_beta_overlap_pct`.
- **Data quality log:** Every data issue (missing fields, stale data, rate-limit failures) is logged to `validation/data_quality_log.csv` with ticker, severity, description, and action taken.
- **Structured pipeline logging:** A Python `logging`-based structured logger (`screener.pipeline`) records coverage statistics, filter actions, and scoring stage completions for machine-parseable diagnostics.

---

## Value Trap Detection

A stock can score well on valuation (cheap!) but be cheap for a reason — declining business, negative momentum, or analysts cutting estimates. A stock is flagged as a potential value trap only if it is **cheap** — a Valuation score in the top 30% of the universe — **and** it falls in the bottom 30% of **at least two** of these three categories:

- Quality Score (floor: 30th percentile)
- Momentum Score (floor: 30th percentile)
- Revisions Score (floor: 30th percentile)

The cheapness condition is what makes it a *value* trap: Piotroski (2000) separates the cheap stocks that go on to do well from those that do not using exactly this kind of fundamental weakness, within the cheapest stocks. Without it (before 2026-10-09) the flag fired on about a quarter of the universe, whatever the valuation. The 2-of-3 majority logic tolerates a single weak dimension (e.g., a quality stock with one bad momentum quarter); "any 1 breach" flagged roughly 60% of the universe.

Missing data (NaN) in any of the three dimensions does **not** trigger a value trap flag — missing data is not the same as poor quality. These stocks receive a separate `Insufficient_Data_Flag`.

Each flagged stock also receives a **Value Trap Severity** score (0-100): for each of the three dimensions, how far below its threshold the stock falls (as a share of the threshold, zero if above it), averaged over the three. A severity of 80 means the stock is deep in trap territory; a severity of 20 means it barely crossed the thresholds. This provides more granularity than the binary flag alone.

By default, value-trap-flagged stocks are **excluded** from the model portfolio (configurable to flag-only mode).

---

## Growth Trap Detection

The mirror image of a value trap: a stock can score well on growth but be growing unsustainably — high growth with poor quality and/or deteriorating analyst sentiment. A stock is flagged as a potential growth trap only if its Growth Score is **above** the 70th percentile (high growth) **and** at least one of these holds:

- Quality Score **below** the 35th percentile (low quality)
- Revisions Score **below** the 35th percentile (deteriorating sentiment)

High growth is required (since 2026-10-09; it used to be one of three conditions, so a low-growth stock with low quality and low revisions could be called a growth trap). Mohanram (2005) separates winners from losers within growth stocks using fundamental strength, the mirror of Piotroski's test within value stocks.

This catches "growth at any price" stocks — companies that are growing fast but burning cash, carrying deteriorating fundamentals, or losing analyst confidence.

Each flagged stock also receives a **Growth Trap Severity** score (0-100): how far beyond each of the three thresholds the stock falls (zero where it does not cross one), averaged over the three. Higher severity means deeper in trap territory.

By default, growth-trap-flagged stocks are **excluded** from the model portfolio (configurable to flag-only mode).

---

## Portfolio Construction

**This is an Excel/artifact output, not a dashboard feature.** The dashboard was
never the place for it and stopped showing it on 2026-08-26: a named, fixed list
of holdings published to a public site reads as a recommendation, and this tool
is decision support - it shows you *why* a stock ranks where it does and leaves
the buy/sell/size judgement to you. The ranking on the dashboard is the product.
What follows describes the `ModelPortfolio` sheet in the generated workbook, for
research use by the owner.

After scoring and ranking, the screener builds a **model portfolio** from the top-ranked stocks:

- **Number of holdings:** Top 25 stocks (configurable)
- **Weighting:** Equal weight (each stock gets ~4%)
- **Sector cap:** Maximum 8 stocks from any single sector, to avoid overconcentration
- **Position limits:** No single stock above 5.0% — not binding under equal weighting, where every position is 4.00%. It would bind only below 20 holdings.
- **Liquidity filter:** Stocks with less than $10M average daily dollar volume (63-day average) are excluded from the portfolio. Stocks with missing volume data are also excluded (conservative default).
- **Trap exclusions:** Value-trap and growth-trap flagged stocks are excluded (unless configured as flag-only)

If a sector would exceed its cap, the excess stocks are dropped and replaced by the next-highest-ranked stocks from other sectors. Weights are redistributed proportionally.

---

## What Gets Output

The screener produces an **Excel workbook** (`factor_output.xlsx`) with up to 7 sheets (a ReadMe/Disclaimers sheet leads the workbook):

### Sheet 1: Factor Scores
Every stock in the universe with all raw metrics, 8 category scores, the composite score, rank, value trap flag (with severity 0-100), growth trap flag (with severity 0-100), financial sector caveat flag, bank classification, and bank-specific metrics (P/B, ROE, ROA, Equity Ratio) where applicable. Each stock also carries a data provenance tag (`_data_source`), metric coverage count, and an EPS basis mismatch flag. Score columns use quartile-based coloring (Q1=red, Q2=yellow, Q3=light green, Q4=green) for at-a-glance assessment.

### Sheet 2: Screener Dashboard
The top 50 stocks, formatted for quick review. Includes rank, composite score (quartile-colored), all 8 category scores, and the value trap and growth trap flags with severity scores. Color-coded cells highlight strengths and weaknesses.

### Sheet 3: Model Portfolio
The final portfolio with ticker, sector, composite score, position weights, and portfolio-level statistics (weighted average beta, dividend yield, sector allocation breakdown).

### Sheet 4: DataValidation
The top 10 stocks with raw financial values (market cap, revenue, EPS, etc.) displayed for manual spot-checking. Highlights potential issues including EPS basis mismatches (GAAP vs. normalized), stale data, EV cross-validation discrepancies, LTM partial annualization flags, channel-stuffing flags (receivables rising much faster than revenue over the fiscal year), and beta overlap warnings. Also includes a sector-median context table showing 25th/median/75th percentile for 8 key metrics across each sector.

### Sheet 5: Weight Sensitivity (when available)
Results of the weight sensitivity analysis. For each factor category, the sheet shows what happens to the top-20 ranking when that category's weight is perturbed ±5%. Jaccard similarity measures how stable the ranking is — higher values (≥0.85) mean the ranking is robust to small weight changes. Color-coded: green (≥0.85), yellow (0.70–0.84), red (<0.70).

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
After scoring, the pipeline perturbs each factor category weight by ±5% (one at a time) and measures how much the top-20 ranking changes using **Jaccard similarity** (intersection / union of the two top-20 sets). A Jaccard of 1.0 means the ranking is completely unchanged; below 0.70 suggests the ranking is sensitive to that factor's weight. Results are printed to the console and saved in the Weight Sensitivity Excel sheet. This lets you verify that small weight changes don't drastically alter the output — a key requirement for any defensible quantitative process.

### EPS Basis Mismatch Detection
Yahoo Finance provides GAAP trailing EPS but normalized (non-GAAP) forward consensus EPS. When the ratio of forward-to-trailing EPS exceeds 2.0× or falls below 0.3× (and trailing EPS is above $0.10), the stock is flagged with `_eps_basis_mismatch = True`. This alerts users that the forward EPS growth and PEG ratio metrics may be distorted by a GAAP/non-GAAP mismatch rather than a genuine change in earnings expectations. Flagged stocks appear highlighted in the DataValidation sheet.

### Factor Correlation Matrix
A Spearman rank correlation matrix of all category scores is computed and written to the Factor Correlation Excel sheet. This makes explicit the degree of overlap between factors — for example, Momentum's two sub-metrics (12-1M and 6-1M return) share ~6 months of overlap, and EV-based valuation metrics are structurally correlated. Correlations above 0.6 are highlighted orange; above 0.8 are highlighted red. This transparency allows users to assess the effective number of independent signals.

### Data Provenance
Every stock carries three provenance fields: `_data_source` (where the data came from — e.g., "yfinance", "cache", "sample"), `_metric_count` (how many of the metrics that apply to that stock have a value), and `_metric_total` (how many apply to it: fewer for a bank than for a non-bank). This makes per-stock data completeness visible at a glance, and it is the same count the composite's coverage discount reads. The drilldown also states the date of the balance sheet, income statement and cash-flow statement each figure comes from.

### The Workings Behind Every Score
Open any stock and **The workings** section rebuilds its score from the bottom. For each category it lists every metric with a weight: the raw value, the sector percentile it earned, the weight it actually carried for *this* stock (bank weighting, the Piotroski conditional adjustment, and rescaling when a metric has no data all change it), and the points it contributed. The points add up to the category score, the category points add up to the composite, and a coverage discount, where one applies, is shown as its own line. Each metric row opens to its formula, the reported figures behind it (market cap, EBIT, cash flow, prices, and so on), who it was ranked against and where it placed, and a plain note wherever the code does something its name would not suggest.

The page does not ask to be taken on trust. The data run rebuilds every category score and composite from the published weights before it will publish, and refuses to publish a day on which any stock fails to reproduce; `scripts/audit_stock.py` repeats the whole calculation for any stock with code that shares nothing with the scoring engine, reading only the published data file. What it cannot do is check that the underlying figures are true: they are as reported by each company and delivered by Yahoo Finance, and the workings show that they were used consistently.

### DataValidation Sheet
The top 10 portfolio stocks are displayed with raw financial values (market cap, revenue, net income, EPS, price) for manual spot-checking against external sources (e.g., Bloomberg, SEC filings). The sheet highlights six types of potential issues: EPS basis mismatches, stale data (price targets that may be outdated), EV cross-validation discrepancies, LTM partial annualization (3-of-4 quarters extrapolated to LTM), channel-stuffing flags (Beneish's receivables index at 1.465 or more), and beta overlap warnings (<80% date overlap with market). A sector-median context table shows 25th/median/75th percentile for 8 key metrics across each sector, enabling quick sanity checks.

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
| **Outliers flagged, never clipped** (1%/99% tails) | Sector ranking is a rank transform, so clipping cannot change any ordering — it can only create artificial ties and misreport the company's real figure. Extreme values are logged as a data-quality signal instead, which is also how a bad feed gets caught. |
| **Value trap: cheap, and weak on 2 of 3** | A value trap is a cheap stock that is cheap for a reason (Piotroski 2000), so cheapness is required. OR logic (any 1 breach) flagged ~60% of the universe; majority logic tolerates one bad dimension. |
| **Growth trap: high growth, and weak quality or revisions** | Mirror of value trap (Mohanram 2005). High growth is required, so a low-growth stock is never called a growth trap. |
| **Liquidity filter** ($10M daily dollar volume) | Ensures portfolio stocks are tradeable at scale. NaN volume is excluded conservatively. |
| **Revisions led by the estimate revision** (FY1 revision + Surprise + Beat Score + Target + Short interest) | The 90-day change in the consensus estimate carries the most weight because revisions, unlike surprises, still predict returns in large caps (Chan, Jegadeesh & Lakonishok 1996; Martineau 2022 on the surprise drift's absence since 2006). Earnings Acceleration is computed but unweighted since 2026-10-09. |
| **Momentum regime rule: off** | Momentum crashes in high-volatility markets, but the input the rule used (the spread of momentum scores across stocks) does not measure volatility. Off until rebuilt on a real volatility measure. |
| **3-metric risk category** (Vol + Beta + MaxDD), dispersion only | Volatility captures total risk, Beta systematic risk, Max Drawdown tail risk. Sharpe and Sortino were dropped from scoring on 2026-09-02: both divide the *same* trailing return by a dispersion measure, so they correlated +0.993 with each other and +0.944 with the momentum signal, but only +0.025 with volatility. They were five metrics in name and three in substance, and the two extras were momentum wearing a risk label. Institutional risk models are built the same way — the Barra US Equity Model's volatility style factors use dispersion descriptors (daily standard deviation, cumulative range, residual sigma), not return/risk ratios. |
| **Quartile-based Excel coloring** | Absolute thresholds (e.g., >80 = green) assume a stable score distribution. Quartile-based coloring adapts to the actual distribution, ensuring roughly 25% of cells in each color band regardless of market conditions. |
| **Trap severity scores** (0-100 continuous) | Binary flags lose information. Severity scores quantify how deep in trap territory a stock is — severity 80 is much worse than severity 20, but both would be flagged as True. |
| **Beta overlap validation** (≥80% required) | Stocks with limited trading history (IPOs, relisted) can produce misleading beta values from sparse overlap with the market index. The 80% threshold ensures the regression uses substantially the same time period as the market. |
| **Channel-stuffing detection** (Beneish's receivables index) | Receivables rising far faster than revenue over the same fiscal year may indicate aggressive revenue recognition; the cut, 1.465, is the manipulators' average in Beneish (1999). The flag is informational (not used in scoring). |

---

## Limitations to Be Aware Of

1. **Data source:** All data comes from Yahoo Finance (free, unofficial API). Occasional field name changes, rate limiting, or missing data are handled gracefully (the screener returns NaN and continues), but the data quality is not institutional-grade — individual fields go missing for individual companies, and the screener drops the affected metric rather than guessing at it.

   **Fetch reliability, measured:** the 18 scheduled runs from 2026-09-02 to 2026-09-25 report **0 fetch failures across 9,036 ticker-fetches**. This page previously put the failure rate at 10-25% of the universe per run, attributing it to HTTP 429 rate limiting. That figure dated from the project's launch period and nobody had re-checked it in the months since; *corrected 2026-09-25*. Rate limiting remains possible on a free API, which is why a run that fetches badly is **discarded before publication** rather than shipped: `scripts/check_run_health.py` refuses a run with price coverage below 90%, analyst-target coverage below 50%, or factor dispersion more than 20% below its trailing median. A number on this site has passed that gate.

2. **GAAP vs. normalized EPS:** Yahoo Finance provides GAAP trailing EPS but normalized forward consensus. For companies with large non-cash charges, write-downs, or unrealized gains (e.g., insurers like CINF), the two bases diverge. As of the 2026-07 review, when the forward/trailing EPS ratio is extreme (>2x or <0.3x — the signature of this contamination), `forward_eps_growth` is set to NaN and its weight is redistributed, rather than scoring a fabricated growth figure.

3. **Point-in-time:** The screener uses the latest available financial data. It does not reconstruct what was known at a past date, which means backtests carry look-ahead bias for fundamental metrics.

4. **Analyst coverage:** The Revisions category relies on analyst estimate and price target data, which is sparse for some stocks. When individual metrics are missing, their weight is redistributed within the category. When the entire category is unavailable, its weight redistributes to the other categories.

5. **EPS revisions reach back only 90 days:** the Revisions category *does* include a forward-EPS-consensus-change metric — FY1 EPS Revision (3-month), its heaviest at 48%, live since 2026-09-10. What is still missing is **depth**: yfinance's estimate history covers about 90 days, so the screener can see a recent revision but not whether the trend has persisted over the six-month window in which Chan, Jegadeesh & Lakonishok (1996) measured the effect most strongly. That longer window would require a paid consensus feed (FactSet, Refinitiv I/B/E/S). *Corrected 2026-09-25: this limitation previously said the metric was not possible at all, which stopped being true on 2026-09-10.*

6. **Rebalance frequency:** The model portfolio is a snapshot. It should be re-run at the configured frequency (monthly or quarterly) to stay current.

7. **No covariance / correlation portfolio risk model:** None of the default weighting schemes uses cross-holding correlation. `equal` uses no risk input at all, `score` uses the composite only, and `inverse_vol` uses single-name volatility; none of them accounts for how the holdings move together. Portfolio-level risk may be understated for correlated holdings. An ex-ante covariance-aware risk report (Ledoit-Wolf-shrunk daily-return covariance: portfolio vol, diversification ratio, top pairwise correlations) is now printed in the run summary for transparency, and an experimental minimum-variance weighting exists, but correlation is not neutralized in the default portfolio.

8. **Composite is cardinal; percentile is separate:** The `Composite` column is the cardinal weighted-average of the 0-100 category scores (the ranking key, preserving magnitude/conviction). The `Composite_Pct` column is the universe percentile ("better than X% of stocks"). Do not read the cardinal Composite as a percentile.

9. **Self-improving engine is governed & human-approval-only by default:** The engine can propose factor-weight changes from live IC, but auto-apply is OFF by default (`improvement.allow_auto_apply: false`) and, even when enabled, requires statistical significance (IC information ratio), the correct optimization horizon, an anti-drift cap, and FDR control on candidate-metric activation. All changes are logged with full provenance.

10. **Not investment advice:** This is a screening tool, not a recommendation engine. The output is a ranked list to narrow your research — not a list of stocks to blindly buy.

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

# Output: factor_output.xlsx (up to 7 sheets, incl. a ReadMe/Disclaimers sheet)

# Run factor-exposure diagnostics on the latest portfolio
py factor_exposure.py --start 2024-01-01 --end 2025-12-31
```

---

## Summary

The screener answers one question: **"Which S&P 500 stocks look best when measured across valuation, quality, growth, momentum, risk, revisions, size, investment — all at once?"**

It does this by:
1. Pulling financial data for ~500 stocks from Yahoo Finance
2. Computing up to 31 financial metrics across 8 categories (27 generic + 4 bank-specific, depending on company type)
3. Ranking each metric within its sector (so comparisons are fair)
4. Weighting and combining into a single 0-100 composite score (with bank-specific weights for financial companies and conditional Piotroski weighting)
5. Flagging potential value traps (cheap and weak on 2 of 3) and growth traps (high growth and weak quality or revisions)
6. Applying a liquidity filter to ensure tradeability
7. Reporting how stable that ranking is when the weights are nudged

The result is a disciplined, repeatable, multi-dimensional ranking that avoids the tunnel vision of looking at any single metric in isolation.
