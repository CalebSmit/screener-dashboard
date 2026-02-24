# INSTITUTIONAL-GRADE AUDIT REPORT v2.0

# Multi-Factor Stock Screener — Full System Audit

**Date:** 2026-02-18
**Auditor:** Senior Quantitative Analyst / CFA Charterholder (simulated)
**Scope:** Complete codebase, data sources, scoring logic, portfolio construction, backtest module
**Prior Audits:** FORENSIC_AUDIT_REPORT.md (2026-02-17), INSTITUTIONAL_AUDIT_REPORT.md (2026-02-17)
**Status:** READ-ONLY — no code was modified during this audit

---

## EXECUTIVE SUMMARY

This screener implements a 6-category, 17-metric multi-factor model scoring S&P 500 equities using data exclusively from yfinance (an unofficial, unsupported Yahoo Finance scraper). The system has undergone two prior rounds of remediation that fixed several critical issues (Piotroski normalization, momentum skip-month, negative earnings yield masking, calendar-based lookbacks, reproducibility infrastructure). However, **fundamental architectural limitations remain that prevent institutional deployment.**

### Overall Defensibility Rating: **C+ (Retail-Grade with Caveats)**

| Grade | Meaning | This System |
|-------|---------|-------------|
| A | Institutional-grade | No |
| B | Usable with caveats for professional idea generation | Almost — blocked by data source |
| **C+** | **Retail-grade, suitable for personal research** | **Yes — current state** |
| D | Unreliable / misleading | No |

**Why not B:** The sole data source (yfinance) has no SLA, known parsing bugs for `enterpriseValue`, undocumented field semantics, and can break without notice. No institutional investor would deploy capital based on a system whose entire data pipeline relies on an unofficial API scraping undocumented Yahoo Finance endpoints.

**Why not D:** The scoring methodology is financially sound, metric formulas are mostly correct, the pipeline has proper safeguards (winsorization, NaN handling, value trap filters), reproducibility infrastructure exists, and the codebase has professional-grade test coverage. If the data source were replaced with an institutional provider (Bloomberg, FactSet, S&P Capital IQ), this system would be a solid B.

---

## 1. DATA SOURCE VALIDATION (yfinance v0.2.66)

### 1.1 Field-by-Field Assessment

| Field | yfinance Source | Time Period | Reliability | Risk Level |
|-------|----------------|-------------|-------------|------------|
| `marketCap` | `summaryDetail` | Real-time | Reliable for S&P 500 (<2% missing) | LOW |
| `enterpriseValue` | `defaultKeyStatistics` | Mixed (real-time MC + lagging BS) | **Known parsing bugs** (Issue #2507: 4.3x discrepancy for TSM) | **HIGH** |
| `trailingEps` | `defaultKeyStatistics` | TTM (sum of last 4 quarters) | Reliable, updates after each earnings release | LOW |
| `forwardEps` | `defaultKeyStatistics` | NTM consensus | Depends on analyst coverage (5-10% missing) | MEDIUM |
| `currentPrice` | `financialData` | Real-time | Very reliable (<1% missing) | LOW |
| `totalDebt` | `financialData` | MRQ | Can differ from BS for financial sector (5-10% missing) | MEDIUM |
| `totalCash` | `financialData` | MRQ | Includes short-term investments; differs from strict cash definition | MEDIUM |
| `sharesOutstanding` | `defaultKeyStatistics` | Latest filing | Reliable (<2% missing) | LOW |
| `earningsGrowth` | `financialData` | QoQ YoY (single quarter) | **Undocumented definition**; the screener correctly avoids using it | **HIGH** |
| `dividendRate` | `summaryDetail` | Annualized forward | Forward-looking (not historical); timing mismatch with trailing fundamentals | MEDIUM |
| `t.financials` | Annual income statement | Annual (4 years, col=0 newest) | Reliable but can be 1-12 months stale | MEDIUM |
| `t.balance_sheet` | Annual balance sheet | Annual (4 years, col=0 newest) | Same staleness concern as financials | MEDIUM |
| `t.cashflow` | Annual cash flow | Annual (4 years, col=0 newest) | Same staleness concern | MEDIUM |
| `t.earnings_history` | EPS surprises | Last 4 quarters | **30-60% missing rate**; most problematic field | **HIGH** |
| `t.history("13mo")` | Daily adjusted close | 13 months | Reliable; `auto_adjust=True` handles splits/dividends | LOW |

### 1.2 Structural Data Source Risks

| Risk | Severity | Description |
|------|----------|-------------|
| **No SLA** | CRITICAL | Yahoo can change or shut down endpoints at any time. yfinance reverse-engineers undocumented APIs. |
| **Silent field changes** | HIGH | yfinance `_stmt_val()` does fuzzy label matching. A Yahoo schema change (e.g., renaming "Total Revenue" to "TotalRevenue") would silently break metrics. |
| **Enterprise Value bug** | HIGH | Known yfinance parsing bug (Issue #2507) can return values 4x+ higher than actual for some tickers (especially international ADRs). The fallback formula `MC + Debt - Cash` omits preferred stock, minority interest, and pension obligations. |
| **No point-in-time** | HIGH | yfinance always returns the latest available data. Cannot retrieve historical snapshots for backtesting. |
| **Rate limiting** | MEDIUM | Yahoo progressively tightens restrictions; 5-thread concurrent fetching may trigger HTTP 429 errors unpredictably. |
| **Data provenance unknown** | MEDIUM | Yahoo Finance aggregates from undisclosed upstream providers. Cannot verify data accuracy against primary sources. |
| **Restatement blindness** | MEDIUM | Financial statements reflect the most recently filed version including restatements. No mechanism to detect post-filing revisions. |

### 1.3 Data Source Verdict

**yfinance is the single most critical blocker for institutional use.** Every metric in the system is exactly as reliable as its weakest data input. For a system intended to deploy real capital, the minimum data standard is a provider with:
- Contractual SLA
- Documented field definitions
- Point-in-time historical data
- Cross-referenced multi-source validation
- Error notification and correction alerts

yfinance provides none of these.

---

## 2. METRIC-BY-METRIC VERIFICATION

### Metric Verification Table

| # | Metric | Formula in Code | Correct Financial Definition | Verdict | Issues |
|---|--------|-----------------|------------------------------|---------|--------|
| 1 | EV/EBITDA | `EV / EBITDA` (EBITDA > 0 required) | Enterprise Value / Earnings Before Interest, Taxes, Depreciation & Amortization | **✔ Correct** | EV time mismatch (real-time MC + lagging BS debt/cash). Does not fall back to EBIT (correct — D&A can differ materially). |
| 2 | FCF Yield | `(OCF - \|CapEx\|) / EV` | Free Cash Flow to Firm / Enterprise Value | **✔ Correct** | Uses FCF/EV (capital-structure-neutral) rather than FCF/MC (equity-focused). CapEx sign normalization handles both conventions. Requires both OCF and CapEx present (NaN capex → NaN FCF, not OCF). |
| 3 | Earnings Yield | `trailingEps / currentPrice` | E/P = Earnings per Share / Price | **✔ Correct** | Correctly allows negative values (penalizes loss-makers). trailingEps is TTM, currentPrice is real-time — reasonably aligned. |
| 4 | EV/Sales | `EV / Total Revenue` (rev > 0 required) | Enterprise Value / Total Revenue | **✔ Correct** | Same EV time mismatch as #1. Revenue is from most recent annual filing. |
| 5 | ROIC | `EBIT × (1 - tax_rate) / (Equity + Debt_BS - Cash_BS)` | NOPAT / Invested Capital | **⚠ Questionable** | (a) When pretax income ≤ 0, defaults to 21% tax rate — should use 0% when company is in tax-loss position. (b) Subtracts ALL cash, not excess cash — systematically inflates ROIC for cash-rich companies (AAPL, GOOG). (c) Tax rate capped at 50% — some legitimate rates exceed this. |
| 6 | Gross Profit / Assets | `Gross Profit / Total Assets` | Novy-Marx (2013) quality factor | **✔ Correct** | Both from annual statements — temporally consistent. |
| 7 | Debt/Equity | `totalDebt (info) / totalEquity` | Total Leverage Ratio | **✔ Correct** | Uses `.info` totalDebt (includes short-term debt) for comprehensive leverage view. Negative/zero equity → NaN (not sentinel value). |
| 8 | Piotroski F-Score | 9 binary signals, raw integer 0-9 | Piotroski (2000) | **⚠ Questionable** | (a) Signal 9 (asset turnover) uses end-of-year assets instead of beginning-of-year per original paper. (b) n_testable ≥ 6 threshold is reasonable but introduces data availability selection bias. (c) Raw integer scoring (fixed in Phase 2) is correct — no longer uses proportional normalization. |
| 9 | Accruals | `(NI - OCF) / Total Assets` | Sloan (1996) accrual anomaly | **✔ Correct** | Direction correctly set as "lower is better" (more negative = higher earnings quality). |
| 10 | Forward EPS Growth | `(forwardEps - trailingEps) / \|trailingEps\|` | Consensus estimate growth rate | **✔ Correct** | Uses absolute value in denominator (handles negative trailing EPS). Small denominator guard at \|trail\| > 0.01. forwardEps is NTM consensus. |
| 11 | PEG Ratio | `(Price / trailingEps) / (forward_eps_growth × 100)` | Price-Earnings-to-Growth | **✔ Correct** | Fixed in Phase 2 to use computed forward_eps_growth instead of undocumented `earningsGrowth` field. Only computed when growth > 1%. |
| 12 | Revenue Growth | `(Rev_current - Rev_prior) / Rev_prior` | Year-over-year revenue growth | **⚠ Questionable** | Uses annual-to-annual growth, not TTM-to-TTM. For companies with different fiscal year ends, the "current" and "prior" periods may not align perfectly with calendar time. |
| 13 | Sustainable Growth | `ROE × (1 - Payout Ratio)` | Higgins (1977) SGR | **⚠ Questionable** | (a) When dividendsPaid is NaN and dividendRate is available, uses forward indicated rate × shares — timing mismatch (forward dividend with historical earnings). (b) When both dividend sources are NaN, SGR is NaN (fixed in Phase 2 — no longer assumes full retention). |
| 14 | 12-1 Month Return | `(price_1m_ago - price_12m_ago) / price_12m_ago` | Jegadeesh-Titman (1993) momentum | **✔ Correct** | Correctly excludes most recent month. Uses calendar-based lookback (fixed in Phase 2 — no longer uses fixed index offsets). |
| 15 | 6-1 Month Return | `(price_1m_ago - price_6m_ago) / price_6m_ago` | Medium-term momentum (skip-month) | **✔ Correct** | Fixed in Phase 2 to exclude most recent month (6-1 convention), consistent with 12-1 momentum. |
| 16 | Volatility | `std(daily_log_returns) × √252` | Annualized historical volatility | **✔ Correct** | Requires ≥ 200 daily observations. Log returns are standard for volatility estimation. |
| 17 | Beta | `Cov(r_stock, r_market) / Var(r_market)` | CAPM market beta | **✔ Correct** | Date-aligned log returns, ddof=1 (unbiased), ≥ 200 common trading days. 10-month window is shorter than the 2-year academic standard but acceptable for a screener. |
| — | Analyst Surprise | `mean((actual - estimate) / \|estimate\|)` over last 4Q | Standardized earnings surprise | **✔ Correct** | 30-60% missing rate from yfinance. Auto-disabled when coverage < 30%. Only surviving revisions metric. |

### Verdict Summary

| Category | Count |
|----------|-------|
| ✔ Correct | 13 of 17 |
| ⚠ Questionable | 4 of 17 (ROIC, Piotroski, Revenue Growth, Sustainable Growth) |
| ❌ Incorrect | 0 of 17 |

**No metrics are outright incorrect.** The four questionable metrics have defensible implementations with documented trade-offs. The most material issue is ROIC's treatment of cash-rich companies (inflated ROIC) and the tax rate default for loss-making companies.

---

## 3. ACCOUNTING & FINANCIAL LOGIC CHECK

### 3.1 Enterprise Value Construction

| Component | Source | Timing | Issue |
|-----------|--------|--------|-------|
| Market Cap | `info.marketCap` | Real-time | Correct |
| Total Debt | `info.totalDebt` | MRQ (most recent quarter) | 1-3 month lag vs market cap |
| Total Cash | `info.totalCash` | MRQ | Includes short-term investments (broader than strict cash) |
| Preferred Stock | Not included | — | **Missing**: can be material for financials |
| Minority Interest | Not included | — | **Missing**: can be material for conglomerates |

**Impact:** EV is slightly incorrect for ~15-20% of S&P 500 companies where preferred stock or minority interest is material. For the majority of companies, the error is < 2% of EV.

### 3.2 ROIC Invested Capital

The code correctly uses balance sheet sources for all three IC components (equity, debt, cash) for temporal consistency. However:

- `IC = Equity + Total_Debt_BS - Cash_BS`: subtracting ALL cash (not excess cash) inflates ROIC for cash-rich companies. AAPL with ~$60B cash would have IC reduced by $60B, significantly boosting ROIC.
- **Quantitative estimate:** For a company with $40B equity, $20B debt, and $30B cash: IC = $30B. If excess cash is only $10B (the rest needed for operations), true IC = $50B. ROIC would be overstated by 67%.

### 3.3 Financial Sector Treatment

EV/EBITDA, Debt/Equity, and Gross Profit/Assets are meaningless for banks and insurance companies (~14% of S&P 500). The system handles this through weight redistribution (NaN metrics → weight shifted to available metrics within each category), which is the correct approach for a screener. However, the effective quality score for financials is based on only 2-3 of 5 quality metrics, reducing discriminatory power.

### 3.4 Negative Earnings Treatment

| Scenario | Metric | Treatment | Assessment |
|----------|--------|-----------|------------|
| Negative EPS | Earnings Yield | Negative E/P (ranks at bottom) | **✔ Correct** |
| Negative EPS | PEG Ratio | NaN (requires positive trailing EPS) | **✔ Correct** |
| Negative Net Income | Sustainable Growth | NaN | **✔ Correct** |
| Negative Net Income | Piotroski (Signal 1) | Scores 0 on profitability signal | **✔ Correct** |
| Negative EBITDA | EV/EBITDA | NaN | **✔ Correct** |
| Negative Equity | Debt/Equity | NaN | **✔ Correct** |
| Negative IC | ROIC | NaN | **✔ Correct** |
| Negative Pretax Income | ROIC tax rate | Defaults to 21% | **⚠ Should use 0%** |

---

## 4. TIME CONSISTENCY & POINT-IN-TIME VALIDITY

### 4.1 Data Freshness Matrix

| Component | Source | Freshness | Potential Lag |
|-----------|--------|-----------|---------------|
| Share price | `currentPrice` | Real-time | Minutes |
| Market cap | `marketCap` | Real-time | Minutes |
| Enterprise value | `enterpriseValue` | Mixed | Days (MC) to months (debt/cash) |
| Trailing EPS | `trailingEps` | TTM, updates quarterly | 0-90 days from last earnings |
| Forward EPS | `forwardEps` | NTM consensus | Days (analyst updates) |
| Financial statements | `t.financials` col=0 | Most recent annual filing | **1-12 months stale** |
| Balance sheet | `t.balance_sheet` col=0 | Most recent annual filing | **1-12 months stale** |
| Cash flow statement | `t.cashflow` col=0 | Most recent annual filing | **1-12 months stale** |
| Price history | `t.history("13mo")` | Up to prior close | 1 day |

### 4.2 Time Mismatch Assessment

**Most Critical Mismatch:** Current market cap (today) ÷ annual EBITDA (from filing up to 12 months ago) = EV/EBITDA can be significantly distorted for companies with:
- Rapidly changing earnings (tech companies in growth/contraction)
- Recent M&A activity
- Significant debt restructuring since last annual filing

**Data Staleness Detection:** The system now records `_stmt_date_financials` and flags data > 200 days old (added in Phase 2). This is a reasonable safeguard, though 200 days is generous — 120 days (one quarter + grace period) would be tighter.

### 4.3 Point-in-Time Violations

| Component | Live Screening | Backtesting |
|-----------|---------------|-------------|
| Universe (S&P 500 list) | Current — acceptable | **VIOLATION**: today's constituents applied to all historical dates |
| Fundamental data | Latest filing — acceptable | **VIOLATION**: today's fundamentals applied to all historical dates |
| Price data | Real-time — acceptable | Correct (historical prices properly fetched) |
| Analyst estimates | Current consensus — acceptable | **VIOLATION**: current estimates applied retroactively |

**Backtest point-in-time verdict:** The backtest has **two severe biases** (survivorship + look-ahead) that make it scientifically invalid for strategy validation. The code properly disclaims this with prominent warnings. Estimated combined inflation: **2.5-6% annually** on top-decile returns, **0.3-0.6** on Sharpe ratio.

---

## 5. RANKING & SCORING LOGIC AUDIT

### 5.1 Pipeline Overview

```
Raw metrics → Winsorize (1/99 pctile) → Sector percentile rank → Weighted category avg → Weighted composite → Cross-sectional percentile rank → Value trap flags → Final rank
```

### 5.2 Winsorization

- **Method:** scipy `mstats.winsorize` at 1st/99th percentile
- **Minimum sample:** 10 non-NaN values (below this, no winsorization)
- **Assessment:** ✔ Correct order (before ranking). Appropriate bounds. NaN preserved.

### 5.3 Sector-Relative Percentile Ranking

- **Method:** `pandas.rank(pct=True, na_option="keep") * 100`
- **Direction handling:** "Lower is better" metrics inverted via `100 - ranks`
- **Small sector fallback:** < 10 valid values → all stocks get 50.0
- **NaN handling:** NaN raw values → NaN percentiles (not imputed to 50th)

**Issues identified:**
1. **Small-sector bias:** Stocks in sectors with < 10 valid observations for a metric get a forced 50th percentile regardless of actual value. This rewards poor stocks and penalizes good stocks in small sectors. Affected sectors: Real Estate (~30 stocks, but after NaN removal some metrics may have < 10), Materials (~28 stocks).
2. **All METRIC_DIR directions verified correct** (18 of 18).

### 5.4 Category Score Weight Redistribution

When a metric is NaN for a given stock, its weight is redistributed proportionally to available metrics within the same category. This is mathematically sound but has an edge case:

**Single-metric dominance:** If a stock has only 1 of 5 quality metrics available (e.g., only `accruals`, configured weight 10%), that single metric becomes 100% of the quality score. A quality assessment based entirely on one low-confidence metric is not comparable to one based on all 5 metrics.

**Quantitative example:**
- Stock A: All 5 quality metrics available → quality score is a weighted blend
- Stock B: Only `accruals` available → quality score = accruals percentile alone
- Stock A and Stock B are compared as if their quality scores are equivalent — they are not.

### 5.5 Composite Scoring

The final composite is a weighted average of category scores, then percentile-ranked cross-sectionally. **This double percentile ranking compresses information:**

- Sector percentiles → weighted average → cross-sectional percentile
- A stock that is #1 by 20 points and one that is #1 by 0.1 points receive the same composite of 100.0

This is an intentional design choice (robust to outliers) but means the composite cannot distinguish HOW MUCH better one stock is vs. another.

### 5.6 Value Trap Flags

**OR logic:** Flagged if quality < 30th pctile OR momentum < 30th pctile OR revisions < 30th pctile.

**Expected flag rate with independent conditions:**
```
P(flagged) = 1 - (1 - 0.30)^3 = 1 - 0.343 = 65.7%
```

**This is extremely aggressive.** Roughly two-thirds of stocks would be flagged. A stock like Berkshire Hathaway with outstanding quality but poor momentum in a given quarter would be flagged. The config documents this behavior clearly, and the flag can be configured to "flag_only" mode (display but don't exclude).

### 5.7 Factor Correlation / Double-Counting

| Pair | Estimated Correlation | Impact |
|------|----------------------|--------|
| EV/EBITDA ↔ EV/Sales | 0.6-0.8 | Both use EV numerator; high correlation for similar-margin companies |
| Volatility ↔ Beta | 0.6-0.8 | Both measure risk; effectively ~1.5 independent signals, not 2 |
| 12-1M Return ↔ 6M Return | 0.7-0.9 | ~6 months of shared return window; highly redundant |
| ROIC ↔ Gross Profit/Assets | 0.4-0.6 | Both measure profitability; moderate correlation |
| Forward EPS Growth ↔ PEG Ratio | 0.3-0.5 | Share the growth component; PEG adds valuation dimension |

**Effective independent factors:** Likely 8-10 rather than 17. The within-category weighting partially mitigates double-counting, but investors should understand the effective dimensionality is lower than the metric count suggests.

---

## 6. EDGE CASE TESTING

| Scenario | Handling | Assessment |
|----------|----------|------------|
| **Negative earnings company** | Earnings yield is negative (ranked at bottom); PEG/sustainable growth → NaN; Piotroski signal 1 fails | ✔ Correct |
| **Financial institution (bank)** | EV/EBITDA, D/E, GPA → NaN; quality score based on 2-3 of 5 metrics via weight redistribution | ⚠ Adequate but reduced discriminatory power |
| **Highly leveraged firm** | Negative equity → D/E = NaN; negative IC → ROIC = NaN; relies on remaining quality metrics | ✔ Correct |
| **Microcap (< $2B)** | Filtered out by `min_market_cap: 2e9` config | ✔ Correct for S&P 500 universe |
| **Company with all NaN fundamentals** | Coverage filter (< 60% metrics → excluded) catches this | ✔ Correct |
| **Extreme outlier (EV/EBITDA = 500)** | Winsorized to 99th percentile before ranking | ✔ Correct |
| **Stock with only 1 metric in a category** | Gets category score = that one metric's percentile. Comparable to stocks with all metrics. | ⚠ Problematic (see §5.4) |
| **Sector with 5 stocks** | All metrics → 50th percentile fallback | ⚠ Bias (see §5.3) |
| **Company that recently IPO'd** | Short price history → volatility/beta NaN; likely fails coverage filter | ✔ Correct |
| **Cash-rich tech company** | ROIC inflated due to all-cash subtraction from IC | ⚠ Systematic upward bias |

---

## 7. DEFENSIBILITY ASSESSMENT

### Would a professional hedge fund consider this screener's outputs trustworthy?

**No, not for direct capital deployment. Yes, for idea generation with heavy caveats.**

#### Arguments Against Institutional Use:
1. **Data source is an unofficial API with no SLA** — any production system is one Yahoo Finance change away from silent failure
2. **Enterprise Value has known parsing bugs** — a 4x discrepancy for a major company (TSM) was documented and closed as "not planned"
3. **No cross-validation against a second data source** — cannot verify any metric's accuracy
4. **Financial statements can be 1-12 months stale** — mixing with real-time prices creates time mismatches
5. **Backtest is scientifically invalid** — cannot validate the strategy's historical performance

#### Arguments For "Usable with Caveats" Rating:
1. **Scoring methodology is financially sound** — factor selection is academically motivated, formulas are correct, pipeline handles edge cases well
2. **Reproducibility infrastructure is professional** — run_id, config snapshots, intermediate artifacts, structured logging, git integration
3. **Test coverage is comprehensive** — 100+ unit tests covering metrics, scoring, edge cases, config validation
4. **NaN handling is sophisticated** — per-row weight redistribution, auto-disable of sparse metrics, coverage filters
5. **Value trap filter adds a defensive layer** — catches cheap-but-deteriorating stocks
6. **The system knows its limitations** — backtest disclaimers, financial sector advisories, data quality logging

### Classification: **C+ (Retail-Grade with Caveats)**

| Use Case | Appropriate? |
|----------|-------------|
| Personal stock research and idea generation | ✔ Yes |
| Academic factor modeling exercise | ✔ Yes |
| Screening for manual deep-dive analysis | ✔ Yes, with verification |
| Systematic portfolio allocation with real capital | ❌ No — data source risk |
| Client-facing investment recommendations | ❌ No — not defensible |
| Marketing materials or performance claims | ❌ No — backtest invalid |

---

## 8. REPRODUCIBILITY CHECK

| Dimension | Status | Assessment |
|-----------|--------|------------|
| **Deterministic scoring** | ✔ | No randomness in the scoring pipeline (sample data has seed=42) |
| **Run ID tracking** | ✔ | UUID4 hex[:12] generated per run; saved in metadata |
| **Config snapshot** | ✔ | Config YAML saved per run; config hash used for cache keying |
| **Intermediate artifacts** | ✔ | 6 intermediate DataFrames saved as Parquet (raw → winsorized → percentiles → categories → final) |
| **Effective weights** | ✔ | Post-auto-disable weights saved to `effective_weights.json` |
| **Universe tracking** | ✔ | Scored and failed ticker lists saved |
| **Git integration** | ✔ | Git SHA captured in run metadata |
| **Package versions** | ✔ | Key dependency versions recorded |
| **Raw API responses** | ✔ | Raw fetch data saved as artifact (excluding `_daily_returns`) |
| **Sensitivity to API changes** | ❌ | yfinance schema changes would silently break metrics via fuzzy `_stmt_val()` matching |
| **Data revision vulnerability** | ❌ | Cannot detect if yfinance returns restated vs. originally-reported figures |
| **Runtime timing dependency** | ⚠ | Intraday price movements affect `currentPrice`; fetching at market open vs. close produces different results |

**Reproducibility verdict:** Two runs executed within the same cache window (7 days) will produce identical results. Runs separated by more than 7 days may differ due to yfinance data updates, universe changes, and price movements. The system cannot reproduce historical runs even from a few weeks ago, because the raw yfinance data is not point-in-time.

---

## 9. IMPROVEMENT RECOMMENDATIONS

### HIGH PRIORITY — Must Fix for Professional Use

| # | Issue | Impact | Recommendation |
|---|-------|--------|----------------|
| H1 | **yfinance as sole data source** | All metrics limited by unofficial API | Integrate at least one institutional data provider (Bloomberg B-PIPE, FactSet, Refinitiv, or budget: Polygon.io/Intrinio). Keep yfinance as fallback for cost-sensitive users. |
| H2 | **ROIC cash deduction inflates quality for cash-rich companies** | AAPL, GOOG, MSFT systematically overrated on quality | Use `max(0, cash - estimated_operating_cash)` or simply use `totalDebt` in IC denominator without cash deduction. Alternative: use ROIC from a data provider that computes it correctly. |
| H3 | **ROIC tax rate default for negative pretax income** | Companies in tax-loss position get fictional 21% tax hit | When `pretax_income <= 0` and `tax_expense <= 0`, use `tax_rate = 0`. When pretax is negative but tax is positive, the 21% default is acceptable. |
| H4 | **Enterprise Value validation** | Known yfinance parsing bugs (4x+ discrepancy for some tickers) | Always cross-check API-provided EV against computed `MC + Debt - Cash`. Flag and log any discrepancy > 10%. |
| H5 | **Value trap OR logic flags ~60% of stocks** | Too aggressive; penalizes stocks with one weak dimension | Change to require at least 2 of 3 conditions breached (majority logic), or switch to AND (all three breached). |
| H6 | **Backtest `pd.qcut` with `duplicates="drop"`** | Can produce fewer than 10 deciles, making D9/D10 returns zero | Use rank-based decile assignment: `np.ceil(scores.rank(pct=True) * 10).clip(1, 10)`. |

### MEDIUM PRIORITY — Improves Robustness

| # | Issue | Impact | Recommendation |
|---|-------|--------|----------------|
| M1 | **Single-metric category dominance** | A stock scored on 1 of 5 quality metrics is incomparable to one scored on 5 of 5 | Require minimum 2 non-NaN metrics per category; if not met, set category score to NaN (triggers composite weight redistribution). |
| M2 | **Small-sector 50th percentile fallback** | Biases stocks in sectors with < 10 valid observations | Fall back to cross-sector percentile ranking rather than forcing 50.0 |
| M3 | **Data staleness threshold at 200 days** | Too generous — allows stale annual filings | Tighten to 120 days (one quarter + 30-day reporting grace period) |
| M4 | **Financial sector metrics** | EV/EBITDA, D/E, GPA meaningless for banks/insurance | Consider separate metric set for financials (Price/Book, NIM, Efficiency Ratio, Tier 1 Capital) or exclude financials and note the limitation |
| M5 | **Revenue growth is annual-to-annual, not TTM** | Does not reflect most recent operational performance | Use quarterly statements (`t.quarterly_financials`) to compute TTM revenue growth |
| M6 | **Piotroski signal 9 uses end-of-year assets** | Slight deviation from original paper | Use `totalAssets_prior` as denominator for current-year ATO (beginning-of-year convention) |
| M7 | **Portfolio sector cap can silently produce < 25 holdings** | No warning between 20-25 stocks | Log a warning when `len(selected) < num_stocks` after backfill loop |
| M8 | **SPX_SECTOR_WEIGHTS sum to 101%** | Data error in benchmark weights | Correct to sum to exactly 100% |
| M9 | **Hardcoded backtest risk-free rate (4.5%)** | Slightly conservative for 2020-2022 period | Use time-varying risk-free rate (3-month T-bill at each rebalance date) |

### LOW PRIORITY — Enhancements

| # | Issue | Impact | Recommendation |
|---|-------|--------|----------------|
| L1 | **Double percentile ranking** | Compresses information — can't distinguish HOW MUCH better #1 is vs #2 | Consider using raw weighted average for final ranking (already on 0-100 scale) |
| L2 | **Factor correlation not surfaced to users** | ~17 metrics provide ~8-10 independent signals | Display correlation matrix in output; consider orthogonalizing or using PCA |
| L3 | **Beta window (10 months) is short** | Noisier than the 2-year academic standard | Fetch 2+ years of history and use 1-year rolling window |
| L4 | **Inverse-vol "risk parity" ignores correlations** | Not true risk parity; correlated positions get outsized risk allocation | Document clearly; consider correlation-adjusted weighting if institutional use is targeted |
| L5 | **Per-category minimum coverage** | The 60% coverage filter is cross-category; allows lopsided coverage | Add per-category minimum (e.g., at least 1 metric per category) |
| L6 | **1-month IC horizon for slow factors** | Understates predictive power of Quality and Value | Add 3-month and 6-month forward return IC calculations |
| L7 | **Sustainable growth dividend fallback timing** | Forward indicated rate with historical earnings | Use only cash flow dividendsPaid; if NaN, set SGR to NaN |
| L8 | **Missing metrics professionals would expect** | No Price/Book, no Dividend Yield, no Short Interest, no Insider Activity | Add Price/Book as a valuation metric; Dividend Yield as income metric |

---

## 10. BIASES AND FAILURE MODES

### Known Biases

| Bias | Severity | Component | Mitigation |
|------|----------|-----------|------------|
| **Survivorship bias** | HIGH (backtest) / LOW (live) | Universe from current S&P 500 | Properly disclaimed; affects backtest only |
| **Look-ahead bias** | HIGH (backtest) / N/A (live) | Current fundamentals applied retroactively | Properly disclaimed; affects backtest only |
| **Cash-rich company bias** | MEDIUM | ROIC calculation | Not mitigated; systematically favors cash-heavy tech |
| **Small-sector bias** | MEDIUM | Percentile fallback to 50th | Not mitigated; affects Real Estate, Materials |
| **Value trap over-flagging** | MEDIUM | OR logic on 3 dimensions | Can be configured to AND logic via config |
| **Data freshness bias** | MEDIUM | Mixing real-time prices with stale fundamentals | 200-day staleness warning (should be 120 days) |
| **Financial sector bias** | MEDIUM | Inappropriate metrics for banks | Weight redistribution partially mitigates |
| **Short-history bias** | LOW | 10-month beta window | Acceptable for screening |

### Failure Modes

| Mode | Likelihood | Consequence |
|------|-----------|-------------|
| **yfinance API breaks** | HIGH (has happened multiple times) | All metrics return NaN; system falls back to sample data |
| **Yahoo schema change** | MEDIUM | Specific metrics silently become NaN; may not trigger coverage filter |
| **Rate limiting (HTTP 429)** | MEDIUM | Partial universe scored; warning logged |
| **Wikipedia table format change** | LOW | Fallback to `sp500_tickers.json` |
| **Extreme market dislocation** | LOW | Winsorization and percentile ranking handle well |

---

## APPENDIX A: RECALCULATION VERIFICATION

### A.1 EV/EBITDA Recalculation

Given: `EV = $110B`, `EBITDA = $18B`
- Code result: `110 / 18 = 6.11`
- Manual: `110 / 18 = 6.111...`
- **Match: ✔**

### A.2 FCF Yield Recalculation

Given: `OCF = $14B`, `CapEx = -$3B` (negative = outflow), `EV = $110B`
- Code result: `FCF = 14 - abs(-3) = 11B`, `Yield = 11/110 = 0.10`
- Manual: `FCF = OCF - |CapEx| = 14 - 3 = 11B`, `Yield = 11/110 = 10.0%`
- **Match: ✔**

### A.3 ROIC Recalculation

Given: `EBIT = $15B`, `Tax Expense = $3B`, `Pretax Income = $13B`, `Equity = $40B`, `Debt_BS = $20B`, `Cash_BS = $10B`
- Tax rate: `3/13 = 0.2308`, clamped to [0, 0.5] → `0.2308`
- NOPAT: `15 × (1 - 0.2308) = $11.54B`
- IC: `40 + 20 - 10 = $50B`
- ROIC: `11.54 / 50 = 0.2308 = 23.1%`
- **Match: ✔**

### A.4 Piotroski F-Score Recalculation

Given test defaults: NI=$10B, NI_prior=$9B, OCF=$14B, TA=$80B, TA_prior=$75B, LTD=$15B, LTD_prior=$16B, CA=$25B, CA_prior=$23B, CL=$15B, CL_prior=$14B, Shares=1B, Shares_prior=1.02B, GP=$25B, GP_prior=$22B, Rev=$50B, Rev_prior=$45B

1. NI > 0? Yes ✔ (+1)
2. OCF > 0? Yes ✔ (+1)
3. ROA improved? 10/80=0.125 > 9/75=0.12 ✔ (+1)
4. OCF > NI? 14 > 10 ✔ (+1)
5. LTD/TA decreased? 15/80=0.1875 < 16/75=0.2133 ✔ (+1)
6. Current ratio improved? 25/15=1.667 > 23/14=1.643 ✔ (+1)
7. No dilution? 1B <= 1.02B ✔ (+1)
8. Gross margin improved? 25/50=0.50 > 22/45=0.489 ✔ (+1)
9. Asset turnover improved? 50/80=0.625 > 45/75=0.60 ✔ (+1)

Total: 9 of 9, n_testable=9 ≥ 6 → F-Score = 9
- **Match: ✔**

### A.5 Sustainable Growth Recalculation

Given: `NI = $10B`, `Equity = $40B`, `Dividends Paid = -$2B`, `Shares = 1B`
- ROE: `10/40 = 0.25`
- Dividends: `abs(-2B) = $2B`
- Retention: `max(0, 1 - 2/10) = 0.80`
- SGR: `0.25 × 0.80 = 0.20 = 20%`
- **Match: ✔**

---

## APPENDIX B: CONFIGURATION VALIDATION

### Factor Weights (must sum to 100)

| Category | Weight | Status |
|----------|--------|--------|
| Valuation | 25% | ✔ |
| Quality | 25% | ✔ |
| Growth | 15% | ✔ |
| Momentum | 15% | ✔ |
| Risk | 10% | ✔ |
| Revisions | 10% | ✔ |
| **Total** | **100%** | **✔** |

### Metric Weights (must sum to 100 within each category)

| Category | Metrics | Sum | Status |
|----------|---------|-----|--------|
| Valuation | EV/EBITDA(25) + FCF Yield(40) + Earnings Yield(20) + EV/Sales(15) | 100 | ✔ |
| Quality | ROIC(30) + GPA(25) + D/E(20) + Piotroski(15) + Accruals(10) | 100 | ✔ |
| Growth | Fwd EPS Growth(35) + PEG(20) + Revenue Growth(30) + Sust. Growth(15) | 100 | ✔ |
| Momentum | 12-1M(50) + 6M(50) | 100 | ✔ |
| Risk | Volatility(60) + Beta(40) | 100 | ✔ |
| Revisions | Analyst Surprise(100) | 100 | ✔ |

### Config Parameters Enforced vs. Not Enforced

| Parameter | Config Value | Enforced? |
|-----------|-------------|-----------|
| `min_market_cap` | $2B | ✔ Yes |
| `min_avg_volume` | $10M | ❌ **NOT ENFORCED** (no volume filter in pipeline) |
| `sector_cap_multiplier` | 2.0× | ❌ **NOT ENFORCED** (portfolio uses `max_sector_concentration` count instead) |
| `min_position_pct` | 2.0% | ❌ **NOT ENFORCED** (equal/risk_parity weighting ignores this) |
| `rebalance_frequency` | quarterly | ❌ **NOT ENFORCED** (manual rebalance only) |
| `max_missing_metrics` | 6 | ❌ **NOT ENFORCED** (coverage filter uses `min_data_coverage_pct` instead) |
| `gics_level` | Sector | ❌ **NOT ENFORCED** (always uses Sector, not IndustryGroup) |

---

## APPENDIX C: QUESTIONS FOR THE SYSTEM OWNER

Before implementing any fixes, the following design decisions require owner input:

1. **ROIC cash deduction:** Should we subtract all cash (current behavior, inflates ROIC for cash-rich companies) or only excess cash (requires estimating operating cash needs, more complex)?

2. **Value trap logic:** The OR logic flags ~60% of stocks. Should this be changed to:
   - AND logic (all 3 conditions must be breached) — much less aggressive (~2.7% flagged)
   - Majority logic (at least 2 of 3 conditions breached) — moderate (~22% flagged)
   - Keep OR logic but lower thresholds (e.g., 15th percentile instead of 30th)?

3. **Financial sector treatment:** Should financial stocks be:
   - Scored with the current weight redistribution approach (adequate, reduced power)?
   - Scored with a separate metric set (Price/Book, NIM, etc.) — more accurate but more complex?
   - Excluded from the universe entirely?

4. **Data source upgrade:** Is there budget for a paid data provider? If so, what is the target (Bloomberg/FactSet = institutional, Polygon/Intrinio = budget)?

5. **Small-sector fallback:** Should stocks in sectors with < 10 observations:
   - Get 50th percentile (current behavior — biased)?
   - Get cross-sector percentile rank (uses broader universe)?
   - Get NaN (excluded from that metric, weight redistributed)?

6. **Double percentile ranking:** Should the final composite be:
   - Percentile-ranked (current — robust but loses magnitude information)?
   - Raw weighted average (preserves how much better #1 is vs #2)?
   - Z-score standardized (compromise between the two)?

7. **Backtest module:** Given the known biases, should the backtest module be:
   - Kept with disclaimers (current)?
   - Removed entirely (simplifies codebase, no misleading outputs)?
   - Enhanced with point-in-time data when/if an institutional provider is added?

---

*End of Audit Report*

*This report was generated on 2026-02-18 through comprehensive analysis of all source files, configuration, test suites, prior audit reports, and yfinance library internals. No code was modified during this audit.*
