# INSTITUTIONAL-GRADE AUDIT REPORT
# Multi-Factor Stock Screener — Full System Review
**Date:** 2026-02-18
**Auditor:** Claude (CFA-level quantitative audit methodology)
**Scope:** Complete codebase, data pipeline, scoring logic, portfolio construction, backtesting
**Status:** READ-ONLY — no code modified during this audit
**Prior work reviewed:** FORENSIC_AUDIT_REPORT.md, SCREENER_DEFENSIBILITY_SPEC.md, IMPLEMENTATION_PLAN.md

---

## EXECUTIVE SUMMARY

This screener implements a 17-metric, 6-category multi-factor model applied to the S&P 500 universe using yfinance as its sole data source. Significant remediation has been performed since the initial audit (2026-02-17): Piotroski F-Score normalization was fixed, 6-month momentum now correctly uses the 6-1M convention, negative earnings yield is no longer masked as NaN, revisions category was restructured to use only `analyst_surprise`, calendar-based price lookback replaced fixed-index offsets, reproducibility infrastructure was added (run_id, config snapshots, intermediate artifacts), and config-aware caching was implemented.

**Overall Defensibility Rating: C+ — Retail-grade with notable improvements, approaching "usable with caveats" for idea generation only.**

The system has real strengths: the scoring pipeline is mathematically sound, edge cases are well-tested, and the remediation work addressed the most egregious bugs. However, fundamental limitations in the data source (yfinance), remaining methodological issues, and architectural gaps prevent institutional-grade classification.

**Key remaining risks:**
1. yfinance is an unofficial, undocumented API with no SLA — a single schema change silently breaks all fundamentals
2. EV/fundamental time mismatch (real-time market cap + 3-12 month old balance sheet) produces misleading valuation ratios
3. ROIC invested capital definition inconsistently mixes `.info` and `.balance_sheet` cash figures
4. Sustainable growth dividend fallback to info `dividendRate` introduces a conceptual mismatch
5. Backtest remains scientifically invalid (survivorship + look-ahead bias)
6. No cross-validation of yfinance data against any second source

---

## 1. DATA SOURCE VALIDATION (yfinance)

### 1.1 yfinance Architecture & Reliability

| Aspect | Assessment |
|--------|-----------|
| **Official status** | Unofficial scraper of Yahoo Finance. No API key, no SLA, no guaranteed schema. |
| **Data provider chain** | Yahoo Finance ← Refinitiv/Morningstar/S&P Global ← company filings |
| **Known failure modes** | Rate limiting (429 errors), field renaming without notice, intermittent NaN for valid tickers, stale data after provider changes |
| **Institutional suitability** | **Not institutional-grade.** Bloomberg, FactSet, Capital IQ, Refinitiv Eikon are the standard. |

### 1.2 Field-by-Field Verification

#### `.info` fields

| Field | What yfinance returns | What the code assumes | Risk |
|-------|----------------------|----------------------|------|
| `marketCap` | Current shares × current price. Real-time. | Correct. Used for EV fallback and universe filter. | **LOW** — reliable field |
| `enterpriseValue` | Yahoo-computed: MC + Total Debt - Total Cash (including short-term investments). Real-time MC component. | Correctly treated as primary EV source. | **MEDIUM** — Yahoo includes short-term investments in "cash" which can overstate cash deduction, slightly understating EV for companies with large investment portfolios |
| `trailingEps` | TTM diluted EPS from last 4 quarterly reports. | Used directly in earnings yield and forward EPS growth. | **MEDIUM** — TTM figure but can lag up to 3 months after quarter-end before the latest quarter's earnings are incorporated |
| `forwardEps` | Consensus analyst mean estimate for next fiscal year. Source: Refinitiv via Yahoo. | Used in forward EPS growth calculation. | **HIGH** — this is a point-in-time estimate that changes daily. No timestamp captured. The "forward" period shifts throughout the fiscal year (refers to next FY in Q1, current FY in Q4). |
| `currentPrice` | Last trade price or `regularMarketPrice`. Real-time. | Used for earnings yield and PEG denominator. Correctly falls back to `regularMarketPrice`. | **LOW** — reliable |
| `totalDebt` | Total borrowings (short + long term). From last annual/quarterly filing. | Used in EV fallback and D/E ratio. | **MEDIUM** — can be up to 12 months stale. `.info` version may differ from balance sheet version due to different report dates. |
| `totalCash` | Cash + cash equivalents + short-term investments. | Used for EV computation. **IMPORTANT:** The code comments at line 548-549 correctly identify that `.info` totalCash includes short-term investments while balance sheet `Cash And Cash Equivalents` does not, and intentionally uses the `.info` version for EV (matching Yahoo's own EV definition) and the BS version for ROIC. This is a sophisticated and correct design decision. | **LOW** — correctly handled |
| `sharesOutstanding` | Current shares outstanding. | Used only for sustainable growth dividend estimation fallback. | **LOW** |
| `earningsGrowth` | **Undocumented field.** Appears to be YoY quarterly earnings growth rate (decimal). Source and exact definition vary by data provider. | Used as the growth rate in PEG ratio. | **HIGH** — black-box input. Could be trailing quarterly, trailing annual, or forward. The PEG ratio using this field is unreliable. |
| `dividendRate` | Annual indicated dividend per share (not yield). | Used as a fallback when cash flow dividends unavailable: `dividendRate × sharesOutstanding`. | **MEDIUM** — the indicated rate is forward-looking but the sustainable growth formula uses it as if it were a trailing payout. Conceptual mismatch. |

#### `.financials` (Income Statement)

| Field | Label searched | Matching strategy | Risk |
|-------|---------------|-------------------|------|
| Total Revenue | `"Total Revenue"` | Exact match, then startswith/endswith | **LOW** — standard label |
| Gross Profit | `"Gross Profit"` | Same | **LOW** |
| EBIT | `"EBIT"`, fallback `"Operating Income"` | **MEDIUM** — EBIT and Operating Income are *not* identical. EBIT includes non-operating income/expenses. For most S&P 500 companies the difference is small, but for companies with significant non-operating items (e.g., insurance float income, litigation settlements), this conflation distorts ROIC. |
| EBITDA | `"EBITDA"` | Exact match only (no fallback to EBIT, which is correct) | **LOW** |
| Net Income | `"Net Income"` | **MEDIUM** — the `startswith/endswith` fallback at line 203-208 could match "Net Income From Continuing Operations" or similar variants. The code mitigates this by requiring the target to appear at the start or end of the index label, which is a reasonable safeguard. |
| Tax Provision | `"Tax Provision"`, fallback `"Income Tax"` | **LOW** |
| Pretax Income | `"Pretax Income"` | **LOW** |

#### `.balance_sheet`

| Field | Risk |
|-------|------|
| Total Assets | **LOW** — standard label |
| Stockholders Equity / Total Stockholder | **MEDIUM** — the fallback to `"Total Stockholder"` is a substring match that could theoretically match other labels. In practice, yfinance labels are stable enough. |
| Total Debt (BS) | **LOW** |
| Long Term Debt | **LOW** |
| Current Assets / Current Liabilities | **LOW** |
| Cash And Cash Equivalents | **LOW** |
| Ordinary Shares Number / Share Issued | **MEDIUM** — used for Piotroski share dilution signal. Two fallback labels. |

#### `.cashflow`

| Field | Risk |
|-------|------|
| Operating Cash Flow / Total Cash From Operating | **LOW** — both labels are well-known |
| Capital Expenditure / Capital Expenditures | **MEDIUM** — yfinance reports capex as negative in most cases, but the code handles both signs correctly at line 581 |
| Common Stock Dividend / Dividends Paid | **MEDIUM** — the primary label "Common Stock Dividend" is non-standard. "Dividends Paid" fallback is more reliable. |

#### `.history(period="13mo")`

| Aspect | Assessment |
|--------|-----------|
| Data type | Adjusted daily OHLCV with `auto_adjust=True` |
| Corporate actions | Split-adjusted and dividend-adjusted (total return). Correct for momentum and volatility. |
| 13-month window | Correct for computing 12-1 month return with skip-month. |
| Calendar-based lookback | **FIXED** — now uses `pd.Timedelta(days=N)` with "closest trading day on or before" logic. This correctly handles holidays and weekends. |

#### `.earnings_history`

| Aspect | Assessment |
|--------|-----------|
| Fields | `epsActual`, `epsEstimate` from last 4 quarters |
| Surprise calculation | `mean((actual - estimate) / |estimate|)` — **correct** standard calculation |
| Coverage | ~40-60% of S&P 500 tickers. Sparse. |
| Risk | **MEDIUM** — `t.earnings_history` has been renamed/restructured in past yfinance versions. A bare `except` protects against this but silently swallows the data. |

### 1.3 Data Source Verdict

| Rating | Assessment |
|--------|-----------|
| **Overall** | ❌ **Not institutional-grade.** yfinance is acceptable for prototyping and personal research. No hedge fund compliance team would approve it as a production data source. |
| **Reliability** | Fields can change names, return NaN unexpectedly, or lag by months with no notification. |
| **Timeliness** | Market data is real-time. Fundamentals lag 1-12 months. No "as-of" date per field. |
| **Accuracy** | Generally accurate for large-cap S&P 500 companies. Known issues with ADRs, newly listed companies, and companies with complex structures. |
| **Auditability** | Cannot verify provenance. No audit trail from yfinance → original filing. |

---

## 2. METRIC-BY-METRIC VERIFICATION

### Metric 1: EV/EBITDA

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:571](factor_engine.py#L571) |
| **Formula in code** | `ev / ebitda` where `ev` = yfinance enterpriseValue or fallback `mc + debt - cash` |
| **Correct definition** | Enterprise Value / Earnings Before Interest, Taxes, Depreciation & Amortization |
| **Recalculation** | With test defaults: EV=110B, EBITDA=18B → 110/18 = 6.11. Code produces 6.11. ✓ |
| **Units** | Ratio (dimensionless). ✓ |
| **Divide-by-zero** | Guarded: `ebitda > 0 and ev > 0`. ✓ |
| **EV fallback** | When `enterpriseValue` is NaN/0, computes `mc + debt - cash`. Requires all three non-NaN. ✓ |
| **EBIT fallback** | **Removed.** Code no longer falls back to EBIT when EBITDA is missing. Comment at line 568-569 explains: "D&A can differ materially." ✓ |
| **Hidden assumption** | EV uses real-time market cap + last-filed debt/cash. Time mismatch of up to 12 months. |
| **Verdict** | ✔ **Correct** — formula and guards are sound. Time mismatch is a data limitation, not a code bug. |

### Metric 2: FCF Yield

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:578-582](factor_engine.py#L578-L582) |
| **Formula in code** | `(ocf - abs(capex)) / ev` if capex < 0; `(ocf - capex) / ev` if capex > 0 |
| **Correct definition** | Free Cash Flow / Enterprise Value, where FCF = Operating Cash Flow - Capital Expenditures |
| **Recalculation** | OCF=14B, CapEx=-3B, EV=110B → FCF = 14 - 3 = 11B → yield = 11/110 = 0.10. ✓ |
| **Missing capex** | **FIXED.** When capex is NaN, FCF is NaN (not OCF). Comment at lines 574-577 explains the rationale. ✓ |
| **Capex sign** | Handles both negative (yfinance convention) and positive correctly. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 3: Earnings Yield

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:585-587](factor_engine.py#L585-L587) |
| **Formula in code** | `trailingEps / price` where `price = currentPrice` or fallback `price_latest` |
| **Correct definition** | Trailing EPS / Current Price (inverse of trailing P/E) |
| **Negative EPS** | **FIXED.** Negative EPS now produces negative earnings yield (not NaN). This correctly penalizes loss-making companies. ✓ |
| **Tests confirm** | `TestEarningsYield.test_negative_eps` at test_metrics.py:168 verifies EPS=-3, Price=100 → -0.03. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 4: EV/Sales

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:590](factor_engine.py#L590) |
| **Formula in code** | `ev / rev_c` where rev_c = Total Revenue from most recent annual filing |
| **Guards** | NaN if revenue ≤ 0 or EV ≤ 0 or either NaN. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 5: ROIC

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:600-614](factor_engine.py#L600-L614) |
| **Formula in code** | `NOPAT / IC` where `NOPAT = EBIT × (1 - tax_rate)`, `IC = Equity + Debt - Cash_BS` |
| **Tax rate** | Computed as `income_tax_expense / pretax_income`, clamped to [0%, 50%]. Default 21% when pretax ≤ 0 or missing. |
| **Cash source for IC** | Uses `_cash_bs` (balance sheet Cash & Cash Equivalents) for IC, not `.info` totalCash. Comment at lines 597-600 explains this is the stricter definition. ✓ |
| **Recalculation** | EBIT=15B, tax_rate=3/13≈0.2308, NOPAT=15×(1-0.2308)=11.538B. IC=40+20-10=50B. ROIC=11.538/50=0.2308. Test at test_metrics.py:199-208 confirms. ✓ |
| **Negative IC** | Returns NaN (IC ≤ 0 guard). ✓ |
| **EBIT vs Operating Income** | Uses EBIT first, falls back to Operating Income. |
| **Issue** | ⚠ When pretax income is negative and EBIT is positive (company has large non-operating losses), the 21% default tax rate is applied to a positive EBIT, producing a positive ROIC for a company that is actually losing money on a pretax basis. This is misleading but rare for S&P 500 companies. |
| **Verdict** | ⚠ **Questionable** — formula is correct for normal cases. The negative-pretax-positive-EBIT edge case produces misleading results. |

### Metric 6: Gross Profit / Assets

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:617-618](factor_engine.py#L617-L618) |
| **Formula in code** | `grossProfit / totalAssets` |
| **Correct definition** | Novy-Marx (2013) profitability factor: Gross Profit / Total Assets |
| **Verdict** | ✔ **Correct** — clean, simple implementation |

### Metric 7: Debt/Equity

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:623-627](factor_engine.py#L623-L627) |
| **Formula in code** | `_debt / eq_v` where `_debt = totalDebt` (info), `eq_v = totalEquity` (BS) |
| **Negative equity** | **FIXED.** Returns NaN when equity ≤ 0 (comment: "not a sentinel value"). Previously used 999.0. ✓ |
| **Issue** | ⚠ The debt figure comes from `.info` (`totalDebt`) which may include short-term borrowings, while equity comes from balance sheet (`Stockholders Equity`). These could be from different filing dates if yfinance updates them at different times. |
| **Verdict** | ⚠ **Questionable** — correct formula, but cross-source timing risk. Minor issue for S&P 500 companies. |

### Metric 8: Piotroski F-Score

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:629-667](factor_engine.py#L629-L667) |
| **Formula in code** | 9 binary signals: NI>0, OCF>0, ROA↑, OCF>NI, LTD/TA↓, CR↑, Shares↓, GM↑, ATO↑ |
| **Normalization** | **FIXED.** Now uses raw integer score (0-9). Comment at line 665-666: "Do NOT proportionally normalize — a company that passes 7 of 7 testable signals is NOT the same quality as one passing 9 of 9." ✓ |
| **Minimum data** | Requires ≥ 4 testable signals, otherwise NaN. |
| **Test confirms** | `TestPiotroskiFScore.test_partial_data_not_inflated` verifies that 5/5 passing returns 5, not 9. ✓ |
| **Issue** | ⚠ Minimum threshold of 4 is lenient. Academic standard is typically all 9 signals testable. With only 4 testable, the score range is compressed (0-4 instead of 0-9), reducing discriminating power. |
| **Verdict** | ⚠ **Questionable** — normalization fix is correct. Minimum threshold is lenient but documented. |

### Metric 9: Accruals

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:670](factor_engine.py#L670) |
| **Formula in code** | `(ni - ocf) / ta` |
| **Correct definition** | Sloan (1996): (Net Income - Operating Cash Flow) / Total Assets |
| **Direction** | Lower (more negative) = better quality. `METRIC_DIR["accruals"] = False`. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 10: Forward EPS Growth

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:677-679](factor_engine.py#L677-L679) |
| **Formula in code** | `(forwardEps - trailingEps) / abs(trailingEps)` |
| **Small denominator guard** | `abs(trail) > 0.01`. ✓ |
| **Issue** | ⚠ Forward EPS from yfinance is a consensus estimate for the *next fiscal year*, not next 12 months. When a company is 10 months into its fiscal year, "forward" EPS refers to a period starting 2 months from now — the growth rate is compressed and not comparable to a company early in its fiscal year. |
| **Verdict** | ⚠ **Questionable** — formula correct but forward EPS fiscal year alignment varies across companies |

### Metric 11: PEG Ratio

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:682-685](factor_engine.py#L682-L685) |
| **Formula in code** | `(price / trail) / (earningsGrowth × 100)` |
| **Correct definition** | P/E ratio divided by earnings growth rate (in %) |
| **Issue 1** | ⚠ `earningsGrowth` from yfinance `.info` is a **black-box field** with no documented definition. It may be trailing quarterly YoY growth, annual growth, or forward consensus growth. Different from the manually computed forward EPS growth. |
| **Issue 2** | The `× 100` conversion assumes `earningsGrowth` is a decimal (e.g., 0.10 = 10%). If Yahoo changes the format, PEG becomes 100× too large/small with no error. |
| **Verdict** | ❌ **Incorrect** — uses an undocumented, unverifiable input. The PEG ratio from this screener is unreliable and should not be presented to professional investors without a caveat. |

### Metric 12: Revenue Growth

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:688](factor_engine.py#L688) |
| **Formula in code** | `(rev_c - rev_p) / rev_p` where rev_c = financials col 0, rev_p = financials col 1 |
| **Time periods** | Most recent annual vs. prior annual. yfinance orders columns descending by date. |
| **Issue** | ⚠ For companies with non-December fiscal year ends, the "most recent" annual report could be 3-11 months old. Revenue growth is accurate for the reported period but may not reflect the most recent quarter. |
| **Verdict** | ✔ **Correct** — formula is standard YoY revenue growth |

### Metric 13: Sustainable Growth

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:691-710](factor_engine.py#L691-L710) |
| **Formula in code** | `ROE × retention` where `ROE = NI/Equity`, `retention = max(0, 1 - divs/NI)` |
| **Dividend source** | Primary: `dividendsPaid` from cash flow statement. Fallback: `dividendRate × sharesOutstanding` from `.info`. If both NaN: **returns NaN** (not full retention assumption). |
| **FIXED** | The prior audit noted that unknown dividends defaulted to full retention. This is now NaN. Test at test_metrics.py:432-435 confirms. ✓ |
| **Issue** | ⚠ The `.info` `dividendRate` fallback is conceptually problematic: `dividendRate` is the *indicated forward* annual dividend per share, while the formula needs *trailing actual* dividends paid. A company that just initiated or cut its dividend would have a mismatch. |
| **Verdict** | ⚠ **Questionable** — unknown-dividend handling is fixed. The fallback source is a conceptual mismatch but better than the prior full-retention assumption. |

### Metric 14: 12-1 Month Return

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:717-719](factor_engine.py#L717-L719) |
| **Formula in code** | `(p1m - p12) / p12` where p1m = `price_1m_ago`, p12 = `price_12m_ago` |
| **Price lookback** | **FIXED.** Now uses calendar-based lookback: `last_date - pd.Timedelta(days=30)` for 1M, `days=365` for 12M. Finds closest trading day on or before target. ✓ |
| **Skip-month** | Correctly excludes most recent month (uses price_1m_ago, not price_latest). Jegadeesh & Titman (1993) convention. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 15: 6-1 Month Return

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:722-723](factor_engine.py#L722-L723) |
| **Formula in code** | `(p1m - p6m) / p6m` where p1m = `price_1m_ago`, p6m = `price_6m_ago` |
| **Skip-month** | **FIXED.** Now uses 6-1M convention (excludes most recent month), matching 12-1M. ✓ |
| **Price lookback** | Uses `pd.Timedelta(days=182)` for 6M. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 16: Volatility

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:357](factor_engine.py#L357) |
| **Formula in code** | `daily_ret.std() × √252` where `daily_ret = log(close / close.shift(1))` |
| **Minimum observations** | Requires ≥ 200 daily returns. ✓ |
| **Returns type** | Log returns — correct for annualization via √252 scaling. ✓ |
| **Issue** | Uses population std (pandas default `ddof=1` for .std()). With 200+ observations, this is fine. |
| **Verdict** | ✔ **Correct** |

### Metric 17: Beta

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:733-763](factor_engine.py#L733-L763) |
| **Formula in code** | `Cov(stock, market) / Var(market)` with date alignment via dict matching |
| **Date alignment** | Both stock and market returns stored as `{date_str: return}` dicts. Common dates found via set intersection. ✓ |
| **Minimum observations** | 200 common trading days required. ✓ |
| **ddof** | `np.var(mr, ddof=1)` for sample variance, `np.cov` defaults to `ddof=1` — internally consistent. ✓ |
| **Verdict** | ✔ **Correct** |

### Metric 18: Analyst Surprise

| Aspect | Detail |
|--------|--------|
| **Code location** | [factor_engine.py:366-376](factor_engine.py#L366-L376) |
| **Formula in code** | `mean((actual - estimate) / |estimate|)` over last 4 quarters, minimum 2 with valid data |
| **Correct definition** | Standardized Unexpected Earnings (SUE) averaged over recent quarters |
| **Issue** | The `abs(e) > 0.001` guard prevents division by near-zero estimates but an estimate of exactly 0.001 could produce extreme surprise ratios. |
| **Verdict** | ✔ **Correct** — standard calculation. Coverage is the main limitation. |

### Metric Verification Summary Table

| # | Metric | Category | Verdict | Key Issue |
|---|--------|----------|---------|-----------|
| 1 | EV/EBITDA | Valuation | ✔ Correct | EV time mismatch (data limitation) |
| 2 | FCF Yield | Valuation | ✔ Correct | Fixed: missing capex → NaN |
| 3 | Earnings Yield | Valuation | ✔ Correct | Fixed: negative EPS handled |
| 4 | EV/Sales | Valuation | ✔ Correct | EV time mismatch |
| 5 | ROIC | Quality | ⚠ Questionable | Negative pretax + positive EBIT edge case |
| 6 | Gross Profit/Assets | Quality | ✔ Correct | Clean implementation |
| 7 | Debt/Equity | Quality | ⚠ Questionable | Cross-source timing risk |
| 8 | Piotroski F-Score | Quality | ⚠ Questionable | Fixed: raw score. Lenient 4-signal minimum |
| 9 | Accruals | Quality | ✔ Correct | Standard Sloan (1996) |
| 10 | Forward EPS Growth | Growth | ⚠ Questionable | Fiscal year alignment varies |
| 11 | PEG Ratio | Growth | ❌ Incorrect | Black-box `earningsGrowth` input |
| 12 | Revenue Growth | Growth | ✔ Correct | Standard YoY |
| 13 | Sustainable Growth | Growth | ⚠ Questionable | Dividend fallback source mismatch |
| 14 | 12-1 Month Return | Momentum | ✔ Correct | Fixed: calendar-based lookback |
| 15 | 6-1 Month Return | Momentum | ✔ Correct | Fixed: skip-month convention |
| 16 | Volatility | Risk | ✔ Correct | Standard annualized vol |
| 17 | Beta | Risk | ✔ Correct | Date-aligned, 200-day minimum |
| 18 | Analyst Surprise | Revisions | ✔ Correct | Coverage is the limitation |

**Summary: 11 Correct, 6 Questionable, 1 Incorrect**

---

## 3. ACCOUNTING & FINANCIAL LOGIC CHECK

### 3.1 Enterprise Value Construction

```
EV = yfinance enterpriseValue (primary)
   = marketCap + totalDebt - totalCash (fallback)
```

**Issues identified:**

1. **Time mismatch (MEDIUM):** `marketCap` is real-time. `totalDebt` and `totalCash` are from the last annual or quarterly filing (up to 12 months stale). A company that took on $20B in debt last quarter but hasn't filed yet will have an understated EV.

2. **Cash definition (LOW):** The code correctly notes (lines 545-548) that `.info` `totalCash` includes short-term investments, matching Yahoo's own EV definition. The balance sheet `Cash And Cash Equivalents` (stricter) is used for ROIC invested capital. This is a thoughtful design choice.

3. **Negative EV guard (CORRECT):** EV ≤ 0 → NaN for EV-based metrics. Cash-rich companies (e.g., Berkshire Hathaway) with negative EV are correctly excluded from valuation ratios rather than producing misleading negative ratios.

### 3.2 ROIC Invested Capital

```
IC = Equity + totalDebt - cash_bs
     (BS)     (info)     (BS)
```

**Issue (MEDIUM):** The debt component comes from `.info` (`totalDebt`) while equity and cash come from the balance sheet. If `.info` and the balance sheet are from different filing dates, invested capital is internally inconsistent. The code should use `totalDebt_bs` from the balance sheet for consistency.

**Verify:** Line 544 shows `_debt = d.get("totalDebt", d.get("totalDebt_bs", np.nan))` — preferring `.info` totalDebt over BS totalDebt. For ROIC, the BS version would be more appropriate since equity and cash are both from the BS.

### 3.3 Profitability Metrics

| Check | Result |
|-------|--------|
| Net Income definition | yfinance "Net Income" — bottom-line including discontinued ops. Standard. |
| ROE = NI / Equity | Correct. Negative equity → NaN for sustainable growth. |
| Gross margin consistency | Gross Profit and Revenue from same filing. ✓ |
| Tax rate computation | `tax_expense / pretax_income`, clamped [0, 0.5]. Default 21%. Reasonable. |

### 3.4 Per-Share Calculations

| Check | Result |
|-------|--------|
| EPS | Uses `trailingEps` from `.info` — TTM diluted. Not computed manually. |
| Shares outstanding | Only used for dividend estimation fallback. Not in denominators of key metrics. |
| No total-to-per-share confusion | The code never divides total figures by shares to get per-share metrics. All per-share data comes from `.info`. ✓ |

### 3.5 Treatment of Negative Earnings / Cash Flow

| Scenario | Treatment | Assessment |
|----------|-----------|------------|
| Negative EPS | Negative earnings yield (ranks low on valuation) | ✔ Correct (fixed) |
| Negative EBITDA | EV/EBITDA = NaN | ✔ Correct |
| Negative NI | Piotroski: NI>0 signal fails. Sustainable growth = NaN. Accruals still computed. | ✔ Correct |
| Negative OCF | Piotroski: OCF>0 signal fails. FCF may be very negative. Accruals still computed. | ✔ Correct |
| Negative FCF | FCF Yield is negative (ranks low). | ✔ Correct |
| Negative equity | D/E = NaN. ROIC IC guard prevents division. | ✔ Correct (fixed from 999.0) |

### 3.6 Financial Institution Handling

**Issue (MEDIUM):** Financial institutions (banks, insurance companies, REITs) have fundamentally different financial statement structures:

- Banks: No "revenue" in the traditional sense. Net interest income is the top line. Gross profit is meaningless.
- Insurance: Premium income and investment income mix. EBITDA is not meaningful.
- REITs: FFO (Funds From Operations) is the standard profitability metric, not EPS.

The screener treats all companies identically. For S&P 500 financials (~14% of the index), metrics like EV/EBITDA, FCF Yield, and Gross Profit/Assets may produce misleading values. Sector-relative percentile ranking partially mitigates this (financials compete against each other), but within the Financials sector, comparing JPMorgan (bank) to Marsh & McLennan (insurance broker) is still problematic.

---

## 4. TIME CONSISTENCY & POINT-IN-TIME VALIDITY

### 4.1 Time Alignment Matrix

| Data Element | As-Of Date | Source | Staleness Risk |
|-------------|-----------|--------|---------------|
| Stock price | Today (real-time) | yf.Ticker.info | None |
| Market cap | Today (real-time) | yf.Ticker.info | None |
| EV (market component) | Today | Derived from MC | None |
| EV (debt/cash component) | Last filing (1-12 months ago) | yf.Ticker.info | **HIGH** |
| Trailing EPS | TTM (updated quarterly) | yf.Ticker.info | Medium (up to 3 months) |
| Forward EPS | Current consensus (changes daily) | yf.Ticker.info | None |
| Revenue / EBITDA / NI | Last annual report (1-12 months ago) | yf.Ticker.financials | **HIGH** |
| Balance sheet items | Last annual report (1-12 months ago) | yf.Ticker.balance_sheet | **HIGH** |
| Cash flow items | Last annual report (1-12 months ago) | yf.Ticker.cashflow | **HIGH** |
| Daily prices (13 months) | Today - 13 months | yf.Ticker.history | None |
| Earnings history | Last 4 quarters (up to current) | yf.Ticker.earnings_history | Low |

### 4.2 Time Mismatch Scenarios

**Scenario 1: Post-M&A EV distortion**
Company acquires a $30B target with $20B debt. Market cap drops (dilution), but balance sheet debt hasn't been filed yet. EV is understated by ~$20B. EV/EBITDA and FCF Yield are both distorted.

**Scenario 2: Fiscal year misalignment**
Two companies in the same sector: Company A has a December FY-end (filed February, 2 months old), Company B has a June FY-end (filed August, 8 months old). At any given run date, Company B's fundamentals are much staler. The screener treats them identically.

**Scenario 3: Earnings season distortion**
During earnings season (Jan-Feb, Apr-May, Jul-Aug, Oct-Nov), some companies have reported new quarters while others haven't. `trailingEps` may reflect 3 or 4 recent quarters depending on reporting status, creating an uneven playing field.

### 4.3 Data Freshness Detection

**IMPROVED:** The code now captures statement dates (lines 326-332) and computes `_stmt_age_days`. Stale data (>400 days) triggers a warning. This is a significant improvement from the prior audit where no freshness detection existed.

**Remaining gap:** The 400-day threshold is generous. A company with a December FY-end that has filed its annual report in February has "fresh" data (age ~60 days). A company with a March FY-end won't file until May/June — at which point the prior year's data is ~15 months old and flagged at 400+ days. A tighter threshold (e.g., 200 days) would catch more cases.

### 4.4 Point-in-Time Verdict

| Use Case | PIT Safety | Assessment |
|----------|-----------|------------|
| **Live screening** | ⚠ Partial | Market data is real-time. Fundamentals lag 1-12 months. Acceptable for idea generation with documented caveats. |
| **Backtesting** | ❌ Unsafe | Uses current fundamentals for all historical periods. Survivorship bias present. Results are scientifically invalid. Properly disclaimed in code. |

---

## 5. RANKING & SCORING LOGIC AUDIT

### 5.1 Scoring Pipeline (in order)

```
Raw metrics (17)
  → Winsorize at 1st/99th percentile (universe-wide)
  → Sector-relative percentile rank (within each GICS sector)
  → Weighted average category scores (6 categories)
  → Weighted average composite score
  → Percentile rank of composite (universe-wide)
  → Value trap flags
  → Final ranking
```

### 5.2 Winsorization

**Implementation:** `scipy.stats.mstats.winsorize` with limits=(0.01, 0.01). Applied to all 18 metric columns.
**Minimum sample:** 10 non-NaN values required. ✓
**NaN handling:** NaN values are excluded from winsorization, then remain NaN. ✓
**Order:** Applied BEFORE percentile ranking. ✓ (Winsorizing after ranking would be meaningless.)

**Assessment:** ✔ Correct implementation. 1%/99% is standard.

### 5.3 Sector-Relative Percentile Ranking

**Implementation:** `pandas.rank(pct=True) * 100` within each sector group.
**Direction inversion:** `100 - ranks` for "lower is better" metrics. ✓
**NaN handling:** **FIXED.** NaN raw values → NaN percentile (not imputed to 50th). The category score function handles per-row weight redistribution. ✓
**Small sector:** < 10 valid values → all get 50.0. ✓

**Issues:**

1. **Sector size disparity (MEDIUM):** S&P 500 sectors vary dramatically in size:
   - Information Technology: ~70 stocks
   - Materials: ~25 stocks
   - Real Estate: ~30 stocks

   Percentile ranking within a 25-stock sector is much noisier than within a 70-stock sector. The 50th percentile of a small sector represents a much wider quality band.

2. **Cross-sector comparability (LOW):** After sector-relative ranking, a stock at the 80th percentile in Energy (25 stocks) and one at the 80th percentile in Tech (70 stocks) are treated as equivalent. In practice, the Tech stock has been screened against much stronger competition.

**Assessment:** ⚠ Correct implementation, but structural limitations with small sectors.

### 5.4 Category Score Computation

**Implementation:** Per-row weighted average of metric percentiles, with NaN-aware weight redistribution.

```python
weighted_sum += df[pc].fillna(0) * w * has_data.astype(float)
weight_sum += w * has_data.astype(float)
df[col] = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
```

**Key behavior:** If a stock has NaN for one metric, the remaining metrics' weights are scaled up to sum to 100%. This is a standard approach (available-data reweighting).

**Issue — entirely NaN metrics skip:**
```python
if m in df.columns and df[m].isna().all():
    skipped_metrics.append(m)
    continue
```
Metrics that are entirely NaN for the entire universe are skipped entirely. This prevents phantom metrics from absorbing weight. ✓

**Assessment:** ✔ Correct implementation.

### 5.5 Composite Score

**Implementation:** Weighted average of category scores → percentile rank × 100.

```python
df["Composite"] = sum(df[col_map[c]] * (fw[c] / tw) for c in col_map)
df["Composite"] = df["Composite"].rank(pct=True) * 100
```

**Key behaviors:**
1. Factor weights are normalized to sum to 1.0 (fw/tw). ✓
2. After the weighted average, a final percentile rank converts to 0-100. This means the composite is a *rank* measure, not an *absolute* score. ✓
3. The top stock gets Composite = 100.0, the bottom gets ≈ 0.2 (1/N × 100). ✓

**Revisions auto-disable:**
When analyst_surprise coverage < 30%, the revisions weight (10%) is redistributed proportionally to other categories. This is correct and well-implemented with deep copy to avoid config mutation. ✓

**Issue — composite percentile ranking destroys information (LOW):**
After the weighted category average, stocks with similar (but different) quality spreads all get compressed to adjacent percentile ranks. A stock with category scores of [90, 90, 90, 90, 90, 90] and one with [88, 88, 88, 88, 88, 88] may end up only 1-2 percentile points apart, even though the former dominates on every dimension.

**Assessment:** ✔ Correct implementation. The percentile-rank approach is defensible for a screening tool (robust to outliers, interpretable).

### 5.6 Value Trap Flags

**Implementation:** OR logic — flagged if quality < 30th percentile OR momentum < 30th percentile OR revisions < 30th percentile.

**NaN handling:** `.le(threshold).fillna(False)` — NaN scores do NOT trigger flags. ✓
**Config documentation:** OR logic is clearly documented in config.yaml with rationale. ✓

**Issue (LOW):** OR logic is aggressive. A stock with excellent quality (95th percentile) but temporarily poor momentum (25th percentile) gets flagged. For a value trap filter, AND logic (both quality AND momentum must be poor) might be more appropriate. However, the OR logic is defensible if the goal is to avoid any single-dimension weakness.

**Assessment:** ✔ Correct implementation, with documented design choice.

### 5.7 Weight Sensitivity Analysis

**Current weights:**
```
Valuation: 25%  Quality: 25%  Growth: 15%  Momentum: 15%  Risk: 10%  Revisions: 10%
```

After revisions auto-disable (typical scenario):
```
Valuation: 27.78%  Quality: 27.78%  Growth: 16.67%  Momentum: 16.67%  Risk: 11.11%
```

**Factor correlation concern:** Within Valuation, EV/EBITDA (25%) and EV/Sales (15%) share the EV numerator. Within Momentum, 12-1M (50%) and 6-1M (50%) share ~5 months of overlap. Within Risk, volatility (60%) and beta (40%) have empirical correlation ~0.6-0.8.

**Effective independent factors:** Approximately 8-10 rather than 17. This means the composite is less diversified than the weight table implies.

**Assessment:** ⚠ Weights are heuristic (intentionally not optimized — avoids overfitting). Factor correlation reduces effective diversification.

---

## 6. EDGE CASE TESTING

### 6.1 Negative Earnings Companies

| Metric | Behavior | Assessment |
|--------|----------|------------|
| EV/EBITDA | NaN if EBITDA ≤ 0 | ✔ Correct |
| FCF Yield | Can be negative (OCF < CapEx) | ✔ Correct — ranks low |
| Earnings Yield | Negative (negative EPS / positive price) | ✔ Correct (fixed) |
| ROIC | Positive EBIT + negative pretax → misleading positive ROIC with 21% default rate | ⚠ Edge case |
| Piotroski | NI > 0 signal fails, OCF > NI comparison still works | ✔ Correct |
| Sustainable Growth | NaN (NI must be > 0) | ✔ Correct |

**Overall:** Loss-making companies are handled well. The earnings yield fix was critical.

### 6.2 Financial Institutions

| Metric | Issue |
|--------|-------|
| EV/EBITDA | EBITDA is not meaningful for banks. May be NaN or wildly different from operating profit. |
| Gross Profit/Assets | Banks have no "gross profit" in the traditional sense. yfinance may return NaN or an irrelevant number. |
| Debt/Equity | Banks have structurally high D/E (deposits are liabilities). D/E of 10× is normal for a bank but would rank poorly. |
| FCF Yield | Bank OCF includes deposit flows, making FCF meaningless. |

**Mitigation:** Sector-relative ranking means banks compete only against other Financials. But Financials includes banks, insurance, asset managers, and payment processors — very different businesses.

**Assessment:** ⚠ Known limitation. Would require sub-industry (GICS Industry Group) ranking to fix properly.

### 6.3 Highly Leveraged Firms

| Scenario | Treatment |
|----------|-----------|
| D/E > 5× | Winsorized at 99th percentile. Still ranks very poorly. ✔ |
| Negative equity | D/E = NaN. Not penalized or rewarded — neutral treatment. ⚠ Arguably should be penalized. |
| EV >> MC | EV includes debt. High-leverage firms have high EV, making EV/EBITDA and EV/Sales unfavorable. ✔ |

### 6.4 Microcaps

**Universe filter:** `min_market_cap: 2e9` ($2B minimum). S&P 500 constituents are all large/mid-cap.
**Assessment:** Microcaps are excluded by universe definition. ✔

### 6.5 Missing Data Companies

| Missing % | Treatment |
|-----------|-----------|
| < 40% metrics missing | Scored normally. Missing metrics get NaN → category weight redistribution. |
| 40-100% missing | Excluded (`min_data_coverage_pct: 60`). |
| Entirely failed fetch | `_skipped = True`, excluded from scoring. |

**Assessment:** ✔ Reasonable thresholds. The 60% coverage requirement prevents poorly-covered stocks from receiving scores based on only 3-4 metrics.

### 6.6 Extreme Outliers

**Protection layers:**
1. Winsorization at 1st/99th percentile
2. Percentile ranking (inherently outlier-robust)
3. Sector-relative comparison (limits cross-sector distortion)

**Assessment:** ✔ Three layers of outlier protection is robust.

---

## 7. DEFENSIBILITY ASSESSMENT

### Would a professional hedge fund consider this screener's outputs trustworthy?

**Rating: C+ — Retail-grade with notable improvements, approaching "usable with caveats" for idea generation only**

#### What prevents institutional-grade (A/B) classification:

1. **Data source (Critical):** yfinance is an unofficial, undocumented API scraping Yahoo Finance. No SLA, no schema guarantees, no audit trail. Any institutional fund would require Bloomberg, FactSet, Capital IQ, or Refinitiv as a primary data source.

2. **PEG ratio uses unverifiable input (High):** The `earningsGrowth` field from yfinance is a black box. A metric that cannot be verified from first principles is not defensible.

3. **No cross-validation (High):** No second data source to validate yfinance outputs. A single-source system has no way to detect data errors.

4. **Backtest is invalid (High):** While properly disclaimed, the backtest cannot provide any evidence of the screener's predictive power. An investment committee would ask "does this model work?" and the answer is "we can't prove it."

5. **EV time mismatch (Medium):** Real-time prices divided by 3-12 month old fundamentals is a known issue with no fix available from yfinance.

6. **Financial sector metrics (Medium):** ~14% of the universe (Financials) is scored with metrics that don't apply well to banks/insurance.

#### What the screener does well (preventing D rating):

1. **Metric formulas are correct:** 11 of 18 metrics are mathematically correct. The 6 "questionable" metrics have minor edge cases, not fundamental errors.
2. **Edge cases are handled:** Negative denominators, missing data, zero values — all produce NaN rather than garbage.
3. **Scoring pipeline is sound:** Winsorize → sector-relative percentile → weighted average → composite percentile is a standard, defensible approach.
4. **Significant remediation done:** The previous audit's most critical metric bugs (Piotroski normalization, 6M momentum, negative earnings yield, dividend assumption) have all been fixed.
5. **Reproducibility infrastructure:** run_id, config snapshots, intermediate artifacts, structured logging — this is professional-grade instrumentation.
6. **Test coverage:** 50+ test cases covering normal, edge, and invariant scenarios.
7. **Honest about limitations:** Backtest disclaimers, stale data warnings, and the SCREENER_DEFENSIBILITY_SPEC.md demonstrate intellectual honesty.

#### Classification justification:

| Rating | Description | This System? |
|--------|-------------|-------------|
| **A: Institutional-grade** | Bloomberg/FactSet data, point-in-time backtesting, cross-validated, compliance-approved | ❌ |
| **B: Usable with caveats** | Reliable data source, verified formulas, documented limitations, useful for idea generation | ❌ (data source fails) |
| **C: Retail-grade** | Free data, correct formulas, some edge cases, useful for personal research | ✔ (strong C+) |
| **D: Unreliable/Misleading** | Wrong formulas, no error handling, produces garbage for edge cases | ❌ (too harsh) |

---

## 8. REPRODUCIBILITY CHECK

### 8.1 Determinism

| Aspect | Deterministic? | Notes |
|--------|---------------|-------|
| Metric computation | ✔ Yes | Same inputs → same outputs. No randomness in scoring. |
| Percentile ranking | ✔ Yes | pandas.rank is deterministic for non-tied values. Ties use "average" method. |
| Composite score | ✔ Yes | Weighted sum is deterministic. |
| Ranking | ✔ Yes | sort_values with method="min" is deterministic. |
| Sample data generator | ✔ Yes | Uses `np.random.default_rng(seed=42)` — seeded PRNG. |

### 8.2 API Sensitivity

| Aspect | Vulnerability |
|--------|--------------|
| yfinance schema change | **HIGH** — field name changes silently produce NaN via bare `except`. |
| yfinance rate limiting | **MEDIUM** — retry logic with exponential backoff handles transient failures. |
| Wikipedia format change | **MEDIUM** — `pd.read_html()` depends on specific table structure. |
| Yahoo data revisions | **LOW** — earnings restatements change historical data silently. |

### 8.3 Runtime Timing Sensitivity

| Aspect | Impact |
|--------|--------|
| Market hours vs. after-hours | `currentPrice` may differ. Small impact on earnings yield. |
| Earnings season | Some companies have filed Q4 earnings, others haven't. `trailingEps` reflects different TTM windows. |
| Index rebalance day | Wikipedia list changes on S&P rebalance dates. Universe differs pre/post rebalance. |
| Mid-day run vs. end-of-day | Prices change intraday. Two runs on the same day can differ slightly. |

### 8.4 Reproducibility Infrastructure

| Feature | Status | Assessment |
|---------|--------|------------|
| run_id (UUID4) | ✔ Implemented | Unique identifier per run |
| Config snapshot | ✔ Implemented | `runs/{run_id}/config.yaml` |
| Effective weights | ✔ Implemented | Post-auto-disable weights saved |
| Universe snapshot | ✔ Implemented | Scored and failed tickers saved |
| Intermediate artifacts | ✔ Implemented | 5 Parquet files per run |
| Structured logging | ✔ Implemented | JSON log with per-ticker timing |
| Package versions | ✔ Implemented | `meta.json` records dependency versions |
| Config-aware caching | ✔ Implemented | Cache filename includes config hash |

**Assessment:** ✔ Strong reproducibility infrastructure. The major remaining gap is that raw yfinance API responses are not snapshotted — you can reproduce the scoring pipeline from intermediate artifacts, but not the raw data fetch.

---

## 9. IMPROVEMENT RECOMMENDATIONS

### HIGH PRIORITY — Must Fix for Professional Use

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| H1 | **Replace or supplement yfinance** | Data source is the single biggest credibility gap | At minimum, add a second data source for cross-validation (e.g., SEC EDGAR XBRL, Alpha Vantage, or Financial Modeling Prep). Long-term: institutional provider. |
| H2 | **Fix PEG ratio input** | Uses undocumented `earningsGrowth` field | Compute PEG using explicit forward EPS growth: `PEG = (P / trailing_EPS) / (forward_EPS_growth × 100)`. This uses data you already have. |
| H3 | **Use BS totalDebt for ROIC** | Invested Capital mixes `.info` and `.balance_sheet` sources | Change ROIC IC computation to use `totalDebt_bs` instead of `totalDebt` (info) for consistency with equity and cash, which are already from BS. |
| H4 | **Add stale data handling** | Stale fundamentals produce misleading ratios | (a) Reduce staleness threshold from 400 to 200 days. (b) When financial statements are >6 months old, flag the ticker in a "STALE_DATA" column. (c) Consider excluding tickers with >10-month-old data from portfolio construction. |
| H5 | **Save raw API responses** | Cannot reproduce the data fetch step | Serialize the raw list of dicts from `fetch_all_tickers()` as `runs/{run_id}/raw_fetch.parquet` or JSON before any transformation. |

### MEDIUM PRIORITY — Improves Robustness

| # | Issue | Fix |
|---|-------|-----|
| M1 | **Add financial sector handling** | Either (a) use GICS Industry Group for ranking instead of Sector, or (b) use alternative metrics for Financials: P/B instead of EV/EBITDA, Tier 1 Capital Ratio instead of D/E, NIM instead of Gross Profit/Assets. |
| M2 | **Improve Piotroski minimum threshold** | Raise from 4 to 6 testable signals. With only 4 signals, the score range (0-4) has very low discriminating power. |
| M3 | **Address factor correlation** | (a) Compute and report a correlation matrix (already done via `compute_factor_correlation`). (b) Consider replacing EV/Sales with a less correlated metric (e.g., P/B for non-financials). (c) Consider replacing 6-1M momentum with earnings momentum or RSI. |
| M4 | **Tighten error handling** | Replace bare `except Exception` in `_stmt_val` (line 210) and `_fetch_single_ticker_inner` (lines 273-278) with specific exception types. Log the actual exception with ticker and field name. |
| M5 | **Add missing metric alerting** | If any single metric exceeds 50% NaN rate (already partially implemented via `metric_alert_threshold_pct`), consider automatically reducing its weight rather than just logging a warning. |
| M6 | **Validate ticker universe** | Cross-check the Wikipedia S&P 500 list against a second source (e.g., S&P Dow Jones Indices website or IVV ETF holdings). Log any discrepancies. |

### LOW PRIORITY — Enhancements

| # | Issue | Fix |
|---|-------|-----|
| L1 | **Add regime detection** | Compute rolling 12-month Value vs. Growth spread. When value is underperforming by >2 standard deviations, flag "growth regime" in output. Does not change weights but helps interpretation. |
| L2 | **Update SPX sector weights** | `portfolio_constructor.py:38-50` hardcodes approximate sector weights from Q1 2025. Add a comment with the source and date, and update quarterly. |
| L3 | **Add liquidity filter** | `min_avg_volume: 10e6` is in config but not enforced (comment at line 107-108 notes this). Either implement or remove from config. |
| L4 | **Add turnover reporting** | When running the screener periodically, report which stocks entered/exited the top 25. High turnover may indicate noisy signals. |
| L5 | **Consider alternative risk metrics** | Maximum drawdown (past 12 months), downside deviation, or Value-at-Risk would provide richer risk characterization than just volatility and beta. |

---

## 10. FINAL DELIVERABLE — COMPLETE AUDIT SUMMARY

### Major Risks Identified

| # | Risk | Severity | Status |
|---|------|----------|--------|
| 1 | yfinance as sole data source | CRITICAL | Unmitigated — architectural limitation |
| 2 | PEG ratio uses unverifiable input | HIGH | Unmitigated |
| 3 | ROIC invested capital cross-source inconsistency | HIGH | Unmitigated |
| 4 | Backtest scientifically invalid | HIGH | Mitigated (properly disclaimed) |
| 5 | EV fundamental-market time mismatch | MEDIUM | Mitigated (freshness detection added) |
| 6 | Financial sector metric inappropriateness | MEDIUM | Partially mitigated (sector-relative ranking) |
| 7 | Piotroski 4-signal minimum is lenient | MEDIUM | Unmitigated |
| 8 | Forward EPS fiscal year misalignment | MEDIUM | Unmitigated |
| 9 | Sustainable growth dividend fallback | LOW | Mitigated (NaN when truly unknown) |
| 10 | Factor correlation double-counting | LOW | Mitigated (correlation matrix computed) |

### Data Reliability Assessment

| Data Category | Reliability | Notes |
|---------------|------------|-------|
| Market data (prices, market cap) | ✔ High | Real-time, accurate for S&P 500 |
| Annual financial statements | ⚠ Medium | Correct when available, but 1-12 months stale |
| Forward estimates | ⚠ Medium | Consensus estimates are reasonable but unverifiable |
| Earnings history | ⚠ Low-Medium | Sparse coverage (~40-60%), but accurate when available |
| `earningsGrowth` | ❌ Low | Undocumented black-box field |

### Biases and Failure Modes

| Bias | Present? | Impact | Mitigation |
|------|---------|--------|------------|
| Survivorship (live screening) | No | N/A | Universe is current S&P 500 constituents |
| Survivorship (backtest) | **YES** | Overstates returns by 2-4% annually (typical estimate) | Disclaimed; cannot fix without historical membership data |
| Look-ahead (live screening) | **Partial** | Fundamentals lag 1-12 months | Staleness detection added |
| Look-ahead (backtest) | **YES** | Renders backtest scientifically invalid | Disclaimed |
| Selection (PEG/sustainable growth) | **YES** | PEG only computed for positive-growth companies; sustainable growth only for profitable companies | These companies cluster in growth sectors, biasing results |
| Sector size | **YES** | Small sectors (Materials, Real Estate) have noisier percentile ranks | 50.0 fallback for < 10 valid values partially mitigates |

### Professional Defensibility Rating

## **Rating: C+ — Retail-Grade with Notable Improvements**

**Usable for:** Personal investment research, academic study, learning multi-factor modeling, idea generation (with verification).

**NOT usable for:** Hedge fund strategy deployment, institutional investor presentations, client-facing analytics, regulatory submissions, portfolio management with fiduciary duty.

**Path to B rating (Usable with Caveats):**
1. Fix PEG ratio to use computed growth (H2)
2. Fix ROIC IC consistency (H3)
3. Add one second data source for cross-validation (H1, partial)
4. Save raw API responses for reproducibility (H5)
5. Tighten stale data handling (H4)

**Path to A rating (Institutional-Grade):**
All of the above, plus:
- Replace yfinance with Bloomberg, FactSet, or Refinitiv
- Add point-in-time backtesting capability
- Add financial sector alternative metrics
- Add liquidity screening
- Formal model validation with out-of-sample testing

### Recommended Fix Plan (Priority Order)

```
IMMEDIATE (same day):
  H2: Fix PEG ratio to use computed forward EPS growth (already available)
  H3: Fix ROIC to use totalDebt_bs instead of totalDebt from .info

WEEK 1:
  H4: Tighten stale data threshold to 200 days
  H5: Save raw API responses per run
  M4: Replace bare except blocks with specific exception handling

WEEK 2:
  M2: Raise Piotroski minimum to 6 testable signals
  M6: Add universe cross-validation

MONTH 1:
  H1: Add second data source (Alpha Vantage or FMP free tier)
  M1: Add financial sector alternative metrics or sub-industry ranking

ONGOING:
  L2: Update sector weights quarterly
  L4: Add turnover reporting
```

---

## CLARIFYING QUESTIONS BEFORE MAKING CORRECTIONS

1. **PEG Ratio:** Should I fix PEG to use the already-computed `forward_eps_growth` as the growth rate, or would you prefer to drop PEG entirely and replace it with a different growth metric (e.g., EPS CAGR from multiple years)?

2. **ROIC Invested Capital:** Should I switch to `totalDebt_bs` for the ROIC IC computation? This may produce slightly different ROIC values for some companies where `.info` totalDebt differs from BS totalDebt.

3. **Piotroski minimum:** Should I raise the minimum testable signals from 4 to 6? This will increase the number of NaN Piotroski scores (more stocks won't have enough data), but the scores that are produced will be more reliable.

4. **Stale data threshold:** Should I reduce the staleness warning from 400 days to 200 days? Should stale-data tickers be excluded from scoring or just flagged?

5. **Financial sector:** Would you prefer (a) keep current treatment (sector-relative ranking handles it), (b) add GICS Industry Group sub-ranking, or (c) use completely different metrics for Financials?

6. **Bare except blocks:** Should I replace ALL bare `except Exception` blocks with specific types (e.g., `KeyError`, `TypeError`, `ValueError`), or just add logging to the existing catches?

7. **Second data source:** Would you like me to implement a cross-validation check using a free API (Alpha Vantage, Financial Modeling Prep), or is this a future consideration?
