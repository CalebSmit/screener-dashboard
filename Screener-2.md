# Composite Multi-Guru Stock Screener Blueprint

## Master Architecture

This blueprint combines all 10 professional investor frameworks into a single composite scoring system. Every stock in your universe receives a normalized sub-score for each guru, and then a weighted final composite score from 0 to 100.

---

## Step 0: Universe Definition & Data Requirements

### Minimum Universe Filters (apply before any scoring)

Remove stocks that would distort or break the scoring system.

```
Filter 1: Market cap ≥ $300M (eliminates micro-caps with unreliable data)
Filter 2: Average daily volume ≥ 100,000 shares (ensures liquidity)
Filter 3: Listed on major exchange (NYSE, NASDAQ, AMEX)
Filter 4: Not a financial holding company, REIT, SPAC, or ADR
         (optional — some gurus exclude financials; flag them separately)
Filter 5: At least 5 years of continuous financial data available
Filter 6: Positive trailing-12-month revenue (eliminates pre-revenue companies)
Filter 7: Share price ≥ $5 (eliminates penny stocks)
```

### Master Data Fields Required

Below is every data point needed across all 10 frameworks. Collect these for each stock annually (and quarterly where noted).

**Price & Market Data:**
```
Price_Current, High_52Week, MarketCap, SharesOutstanding, SharesFloat,
Volume_Daily, AvgVolume_50Day, Return_12M, MA_50, MA_200,
RS_Percentile (relative strength vs. all stocks),
IndustryGroupRank_Percentile
```

**Income Statement (annual, 10 years + quarterly, 8 quarters):**
```
Revenue, GrossProfit, EBIT, EBITDA, PreTaxIncome, NetIncome,
EPS (from continuing operations), InterestExpense
```

**Balance Sheet (annual, 10 years):**
```
CurrentAssets, CurrentLiabilities, TotalAssets, TotalLiabilities,
LongTermDebt, TotalDebt, TotalEquity, BookValuePerShare,
NetPPE (net property plant & equipment), Cash
```

**Cash Flow Statement (annual, 10 years):**
```
CashFlowFromOperations (CFO), FreeCashFlow (FCF),
SharesRepurchased_Value, DividendsPaid
```

**Valuation & Market Ratios:**
```
PE_TTM, PE_3YAvg, PB, PS, PCF (Price/CashFlowPerShare),
EV (MarketCap + TotalDebt - Cash), EV_EBITDA,
DividendYield, PayoutRatio, FCF_Yield (FCF/MarketCap)
```

**Analyst & Ownership Data:**
```
EPS_Estimate_NextFY, EPS_Estimate_60DaysAgo, EPS_Estimate_90DaysAgo,
InstOwnership_Pct, InstOwners_Current, InstOwners_PriorQtr,
InsiderBuys_3M, InsiderSells_3M
```

**Market Benchmarks (for relative comparisons):**
```
MarketPE (S&P 500 trailing P/E), MarketReturn_12M,
SP500_EPS_CAGR_5Y, SP500_EstGrowth, SP500_DividendYield,
Index_Price, Index_MA_50, Index_MA_200
```

---

## Step 1: Compute Each Guru Sub-Score

Each guru score is first computed as raw points, then normalized to a 0–10 scale so every guru contributes equally before weighting.

```
Normalization formula:
  NormalizedScore_Guru = (RawScore / MaxPossibleScore) × 10
```

---

### 1A. Benjamin Graham – Defensive Value (Max Raw: 7)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| G1 | Adequate size | `MarketCap >= 500000000` | Revenue or market cap above ~$500M |
| G2 | Current ratio | `CurrentAssets / CurrentLiabilities >= 2.0` | Strong short-term liquidity |
| G3 | Long-term debt vs NCA | `LongTermDebt <= (CurrentAssets - CurrentLiabilities)` | Conservative financing |
| G4 | Earnings stability | `COUNT(years with EPS < 0 in last 10) = 0` | No losses in a decade |
| G5 | Dividend record | `All 20 years have DPS > 0` | Uninterrupted dividends 20 years |
| G6 | Earnings growth | `(AvgEPS_Last3Y / AvgEPS_10YAgo3Y) - 1 >= 0.33` | 33%+ growth over decade |
| G7 | Combined valuation | `(PE_3YAvg) × (PB) <= 22.5` | Graham Number condition |

```
Graham_Raw = G1 + G2 + G3 + G4 + G5 + G6 + G7        (max 7)
Graham_Norm = (Graham_Raw / 7) × 10
```

---

### 1B. Warren Buffett – Quality at a Fair Price (Max Raw: 8)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| B1 | High avg ROE | `AVERAGE(ROE_Last10Y) >= 0.15` | Durable profitability |
| B2 | Consistent ROE | `COUNT(years ROE >= 0.15 in last 10) >= 8` | Not a one-time fluke |
| B3 | Strong gross margin | `GrossMargin >= 0.40` | Pricing power / moat proxy |
| B4 | Low leverage | `TotalDebt / TotalEquity <= 1.0` | Conservative balance sheet |
| B5 | Strong interest coverage | `EBIT / InterestExpense >= 5` | Can service debt easily |
| B6 | Consistent FCF | `COUNT(years FCF > 0 in last 10) >= 8` | Cash generation through cycles |
| B7 | FCF growing | `FCF_CAGR_5Y > 0` | Free cash flow trending up |
| B8 | Reasonable valuation | `FCF_Yield >= 0.05 OR PE < PE_10Y_Median` | Not overpriced for quality |

```
Buffett_Raw = B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8  (max 8)
Buffett_Norm = (Buffett_Raw / 8) × 10
```

---

### 1C. Peter Lynch – PEG & Growth Type (Max Raw: 6)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| L1 | PEG ratio | `PE / EPS_GrowthRate_5Y <= 1.0` | Growth at reasonable price |
| L2 | Dividend-adjusted PEG | `PE / (EPS_GrowthRate + DividendYield) <= 1.0` | Including income return |
| L3 | Growth sweet spot | `EPS_GrowthRate >= 0.10 AND <= 0.25` | Stalwart/fast grower range |
| L4 | Positive earnings | `EPS_TTM > 0` | Must be profitable |
| L5 | Moderate leverage | `TotalDebt / TotalEquity <= 1.5` | Not overleveraged |
| L6 | Limited dilution | `SharesOutstanding_Current <= SharesOutstanding_3YAgo * 1.05` | < 5% dilution over 3 years |

```
Lynch_Raw = L1 + L2 + L3 + L4 + L5 + L6              (max 6)
Lynch_Norm = (Lynch_Raw / 6) × 10
```

---

### 1D. Joel Greenblatt – Magic Formula (Rank-Based, Max Raw: 10)

This one uses percentile ranks, not pass/fail. Convert the combined rank into a 0–10 score.

```
Step 1: Compute metrics
  EV = MarketCap + TotalDebt - Cash
  EarningsYield = EBIT / EV
  NWC = CurrentAssets - CurrentLiabilities
  Capital = NWC + NetPPE
  ReturnOnCapital = EBIT / Capital

Step 2: Rank within universe
  EY_PctRank = PERCENTRANK(Universe_EY, Company_EY)       -- higher = better
  ROC_PctRank = PERCENTRANK(Universe_ROC, Company_ROC)     -- higher = better

Step 3: Combined score
  MF_Combined = (EY_PctRank + ROC_PctRank) / 2            -- average percentile (0 to 1)

Step 4: Normalize
  Greenblatt_Norm = MF_Combined × 10
```

---

### 1E. James O'Shaughnessy – Value Composite 2 (Rank-Based, Max Raw: 10)

Also rank-based. Six valuation metrics combined into a composite percentile.

```
Step 1: For each stock, compute percentile rank (0 = cheapest/best, 1 = most expensive/worst):
  Rank_PB  = PERCENTRANK(Universe_PB, Company_PB)             -- lower PB = better = lower rank
  Rank_PS  = PERCENTRANK(Universe_PS, Company_PS)             -- lower PS = better
  Rank_PE  = PERCENTRANK(Universe_PE, Company_PE)             -- lower PE = better
  Rank_PCF = PERCENTRANK(Universe_PCF, Company_PCF)           -- lower PCF = better
  Rank_EBITDAEV = 1 - PERCENTRANK(Universe_EBITDA_EV, Company_EBITDA_EV)  -- higher = better, invert
  Rank_SHY = 1 - PERCENTRANK(Universe_ShareholderYield, Company_SHY)      -- higher = better, invert

  Where: ShareholderYield = DividendYield + NetBuybackYield
         NetBuybackYield = SharesRepurchased_Value / MarketCap (positive if buying back)

Step 2: Average the ranks
  VC2_AvgRank = (Rank_PB + Rank_PS + Rank_PE + Rank_PCF + Rank_EBITDAEV + Rank_SHY) / 6

Step 3: Normalize (invert so lower rank = higher score)
  OShaughnessy_Norm = (1 - VC2_AvgRank) × 10
```

---

### 1F. John Neff – Low P/E + Growth + Dividend (Max Raw: 6)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| N1 | Low relative P/E | `CompanyPE / MarketPE <= 0.60` | 40%+ discount to market |
| N2 | Growth in range | `EPS_GrowthRate >= 0.07 AND <= 0.20` | Moderate sustainable growth |
| N3 | Dividend-adj PEG | `PE / (EPS_GrowthRate + DividendYield) <= 1.0` | Core Neff metric |
| N4 | Meaningful yield | `DividendYield >= MAX(SP500_DividendYield, 0.02)` | Above market or 2% floor |
| N5 | Positive EPS | `EPS_TTM > 0` | Must be profitable |
| N6 | Earnings growing | `EPS_GrowthRate_5Y > 0` | Not in secular decline |

```
Neff_Raw = N1 + N2 + N3 + N4 + N5 + N6                (max 6)
Neff_Norm = (Neff_Raw / 6) × 10
```

---

### 1G. William O'Neil – CAN SLIM (Max Raw: 10)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| O1 | Quarterly EPS growth | `QtrEPSGrowth_YoY >= 0.25` | Current earnings power |
| O2 | Annual EPS consecutive | `EPS increased each of last 5 years` | Consistent annual growth |
| O3 | 5Y EPS CAGR | `EPS_CAGR_5Y >= 0.25` | High long-term growth |
| O4 | ROE quality | `ROE >= 0.17` | Management effectiveness |
| O5 | Near 52-week high | `Price / High_52Week >= 0.85` | Technical strength |
| O6 | Low float / small supply | `SharesFloat <= 50,000,000` | Supply/demand favorable |
| O7 | Relative strength | `RS_Percentile >= 80` | Market leader |
| O8 | Industry leader | `IndustryGroupRank_Percentile >= 60` | In a leading group |
| O9 | Institutional trend | `InstOwners_Current > InstOwners_PriorQtr` | Smart money accumulating |
| O10 | Market uptrend | `Index > MA_200 AND MA_50 > MA_200` | Favorable macro backdrop |

```
ONeil_Raw = O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 + O10  (max 10)
ONeil_Norm = (ONeil_Raw / 10) × 10
```

---

### 1H. Joseph Piotroski – F-Score (Max Raw: 9)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| F1 | Positive ROA | `NetIncome / TotalAssets > 0` | Profitable |
| F2 | Positive CFO | `CashFlowFromOps > 0` | Cash-generating |
| F3 | Improving ROA | `ROA_Current > ROA_Prior` | Getting more profitable |
| F4 | Accruals quality | `CashFlowFromOps > NetIncome` | Earnings backed by cash |
| F5 | Decreasing leverage | `(LTDebt/TotalAssets)_Curr < (LTDebt/TotalAssets)_Prior` | Deleveraging |
| F6 | Improving current ratio | `CurrentRatio_Curr > CurrentRatio_Prior` | Better liquidity |
| F7 | No dilution | `SharesOutstanding_Curr <= SharesOutstanding_Prior` | No new equity issued |
| F8 | Improving gross margin | `GrossMargin_Curr > GrossMargin_Prior` | Better efficiency |
| F9 | Improving asset turnover | `(Rev/TotalAssets)_Curr > (Rev/TotalAssets)_Prior` | Better asset use |

```
Piotroski_Raw = F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8 + F9  (max 9)
Piotroski_Norm = (Piotroski_Raw / 9) × 10
```

---

### 1I. Martin Zweig – GARP (Max Raw: 10)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| Z1 | 5Y EPS CAGR ≥ 15% | `EPS_CAGR_5Y >= 0.15` | Long-term growth |
| Z2 | EPS up every year | `EPS increased each of last 5 years` | Consistency |
| Z3 | Quarterly EPS ≥ 20% | `QtrEPSGrowth_YoY >= 0.20` | Current momentum |
| Z4 | Accelerating quarterly EPS | `QtrGrowth_Latest > QtrGrowth_Prior` | Growth speeding up |
| Z5 | Sales confirm earnings | `RevGrowth_5Y >= EPS_CAGR_5Y × 0.50` | Not just cost-cutting |
| Z6 | Quarterly revenue positive | `QtrRevGrowth_YoY > 0` | Top-line confirming |
| Z7 | P/E bounded | `PE <= MIN(3 × MarketPE, 43) AND PE_Pctile >= 0.10` | Not too rich or cheap |
| Z8 | Low relative D/E | `DebtToEquity < IndustryMedian_DE` | Conservative financing |
| Z9 | Price strength | `Return_12M > MarketReturn_12M` | Outperforming market |
| Z10 | Estimate revisions up | `EPS_Estimate_Current > EPS_Estimate_60DaysAgo` | Analyst momentum |

```
Zweig_Raw = Z1 + Z2 + Z3 + Z4 + Z5 + Z6 + Z7 + Z8 + Z9 + Z10  (max 10)
Zweig_Norm = (Zweig_Raw / 10) × 10
```

---

### 1J. David Dreman – Contrarian Value (Max Raw: 10)

| # | Criterion | Formula | Pass = 1 |
|---|-----------|---------|----------|
| D1 | Contrarian gate | `Bottom 20% on ≥ 2 of: PE, PCF, PB, DivYield` | **Required gate** |
| D2 | Adequate size | `MarketCap_Percentile >= 0.30` | Large/mid cap safety |
| D3 | Current ratio | `CurrentRatio >= 2.0 OR > IndustryAvg` | Short-term strength |
| D4 | High ROE | `ROE >= TopThird_ROE_1500` | Quality earnings |
| D5 | Pre-tax margin ≥ 8% | `PreTaxIncome / Revenue >= 0.08` | Solid business |
| D6 | Historical EPS growth | `EPS_CAGR_5Y > SP500_EPS_CAGR_5Y` | Growing faster than market |
| D7 | Forward EPS growth | `EPS_EstGrowth > SP500_EstGrowth` | Expected to keep growing |
| D8 | Low leverage | `TotalLiabilities / TotalAssets < SectorAvg` | Below-average debt |
| D9 | Dividend yield | `DividendYield >= 0.015` | Income protection |
| D10 | Estimate revisions | `EPS_Estimate_Current > EPS_Estimate_90DaysAgo` | Positive surprise setup |

**Special rule:** If D1 = 0 (fails contrarian gate), the entire Dreman score = 0 regardless of other criteria.

```
Dreman_Raw = IF(D1 = 0, 0, D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10)  (max 10)
Dreman_Norm = (Dreman_Raw / 10) × 10
```

---

## Step 2: Style Classification & Adaptive Weighting

### The Problem with Equal Weights

Not every guru framework applies to every stock. A high-growth tech stock will never score well on Graham or Dreman (deep value), and a slow-growing utility will never pass O'Neil (momentum growth). Equal-weighting all 10 would dilute meaningful signals.

### Solution: Classify Each Stock, Then Apply Style Weights

First, classify each stock into one of four style buckets using simple rules:

```
IF EPS_GrowthRate_5Y >= 0.20 AND PE >= MarketPE:
    Style = "Growth"

ELSE IF EPS_GrowthRate_5Y >= 0.08 AND PE <= MarketPE × 1.5:
    Style = "GARP" (Growth at Reasonable Price)

ELSE IF PE_Percentile <= 0.30 AND DividendYield >= 0.015:
    Style = "Deep Value"

ELSE:
    Style = "Core Value"
```

### Recommended Weight Profiles (sum to 100%)

| Guru | Growth | GARP | Deep Value | Core Value |
|------|--------|------|------------|------------|
| **Graham** | 2% | 8% | 15% | 12% |
| **Buffett** | 8% | 15% | 10% | 15% |
| **Lynch** | 10% | 18% | 5% | 12% |
| **Greenblatt** | 8% | 12% | 12% | 12% |
| **O'Shaughnessy** | 5% | 8% | 15% | 10% |
| **Neff** | 2% | 8% | 15% | 10% |
| **O'Neil** | 25% | 8% | 2% | 3% |
| **Piotroski** | 10% | 8% | 15% | 12% |
| **Zweig** | 25% | 10% | 3% | 5% |
| **Dreman** | 5% | 5% | 8% | 9% |
| **Total** | 100% | 100% | 100% | 100% |

**Rationale:** Growth stocks lean heavily on O'Neil and Zweig (momentum + earnings acceleration). Deep value leans on Graham, O'Shaughnessy, Neff, and Piotroski. GARP is balanced toward Buffett and Lynch. Core value spreads evenly with a Buffett/Greenblatt tilt.

### Alternative: Equal Weights (Simpler)

If you prefer simplicity or want a style-agnostic score:
```
W_equal = 10% for each guru (10 × 10% = 100%)
```

---

## Step 3: Compute the Composite Score

```
CompositeScore = (W_Graham    × Graham_Norm)
               + (W_Buffett   × Buffett_Norm)
               + (W_Lynch     × Lynch_Norm)
               + (W_Greenblatt × Greenblatt_Norm)
               + (W_OShaughnessy × OShaughnessy_Norm)
               + (W_Neff      × Neff_Norm)
               + (W_ONeil     × ONeil_Norm)
               + (W_Piotroski × Piotroski_Norm)
               + (W_Zweig     × Zweig_Norm)
               + (W_Dreman    × Dreman_Norm)
```

Since each normalized score ranges 0–10 and weights sum to 100%:
```
CompositeScore ranges from 0.0 to 10.0
```

**To convert to a 0–100 scale for easier interpretation:**
```
FinalScore = CompositeScore × 10
```

---

## Step 4: Tier Classification & Action Signals

| Tier | Final Score | Label | Interpretation |
|------|-------------|-------|----------------|
| S | 80 – 100 | **Elite** | Passes most/all guru filters; extremely rare. Strong multi-factor conviction. |
| A | 65 – 79 | **Strong** | High marks across multiple styles. Solid buy candidates. |
| B | 50 – 64 | **Above Average** | Decent but has weaknesses in some dimensions. Watchlist material. |
| C | 35 – 49 | **Average** | Mixed signals. Needs deeper qualitative research before acting. |
| D | 20 – 34 | **Below Average** | Fails most screens. Likely overvalued, weak fundamentals, or both. |
| F | 0 – 19 | **Poor** | Red flags across the board. Avoid or potential short candidate. |

---

## Step 5: Supplemental Diagnostic Views

Beyond the single composite number, surface the following for each stock to help with final analysis.

### 5A. Guru Agreement Heatmap

For each stock, show which gurus rate it highly (green), average (yellow), or poorly (red).

```
Per-guru rating:
  IF Guru_Norm >= 7.0  → "Strong"  (green)
  IF Guru_Norm >= 4.0  → "Neutral" (yellow)
  IF Guru_Norm <  4.0  → "Weak"    (red)
```

This reveals consensus vs. divergence. A stock scoring 70 composite could be:
- **Broad consensus:** 7/10 on most gurus (safer bet).
- **Polarized:** 10/10 on 3 gurus, 2/10 on the rest (style-specific play, riskier).

### 5B. Style Consistency Score

Count how many guru sub-scores are ≥ 6.0 out of 10:

```
GuruAgreementCount = COUNT(Guru_Norm >= 6.0 for all 10 gurus)
```

| Agreement Count | Interpretation |
|-----------------|----------------|
| 7 – 10 | Multi-factor darling. Very high conviction. |
| 4 – 6 | Style-specific strength. Identify which style cluster agrees. |
| 0 – 3 | Narrow appeal or universally weak. Proceed with caution. |

### 5C. Value vs. Growth Tilt Score

Split the 10 gurus into two camps and compare:

```
Value_Camp = AVERAGE(Graham_Norm, Greenblatt_Norm, OShaughnessy_Norm,
                     Neff_Norm, Piotroski_Norm, Dreman_Norm)

Growth_Camp = AVERAGE(Buffett_Norm, Lynch_Norm, ONeil_Norm, Zweig_Norm)

Tilt = Growth_Camp - Value_Camp
```

| Tilt | Interpretation |
|------|----------------|
| > +3.0 | Strong growth characteristics |
| -1.0 to +3.0 | Balanced / GARP characteristics |
| < -1.0 | Strong value characteristics |

### 5D. Quality Floor Check

Some metrics are so fundamental that no stock should pass without them. Apply a hard quality floor regardless of composite score:

```
QualityFloor_Pass = ALL of:
  1. Positive trailing-12-month EPS
  2. Positive operating cash flow
  3. Current ratio ≥ 1.0
  4. Debt-to-equity ≤ 3.0
  5. No earnings restatements in last 2 years (if data available)
```

If QualityFloor_Pass = FALSE, flag the stock with a warning regardless of composite score.

---

## Step 6: Portfolio Construction Rules

Once you have ranked all stocks by FinalScore, build a portfolio with these guidelines:

### Selection
```
1. Sort universe by FinalScore descending.
2. Apply Quality Floor filter (Step 5D). Remove any stock that fails.
3. Select top N stocks (recommended: 20–30 positions).
4. Require minimum GuruAgreementCount ≥ 4 (at least 4 gurus rate ≥ 6/10).
```

### Diversification Constraints
```
5. Maximum 3 stocks per GICS sector.
6. Maximum 2 stocks per industry group.
7. No single position > 5% of portfolio at initiation.
```

### Weighting Options
```
Option A: Equal-weight all positions (simplest, recommended for most users).
Option B: Score-weight (allocate proportional to FinalScore).
Option C: Inverse-volatility weight (allocate more to lower-volatility names).
```

### Rebalancing
```
8. Full re-screen quarterly (after earnings season: Feb, May, Aug, Nov).
9. Between rebalances, apply stop-loss of -15% from purchase price.
10. Any stock whose FinalScore drops below 35 at rebalance is sold.
11. Replace sold positions with next-highest-scoring stock from ranked list.
```

---

## Step 7: Implementation Pseudocode (Python-Style)

```python
# ============================================================
# COMPOSITE MULTI-GURU SCREENER — FULL PIPELINE
# ============================================================

import pandas as pd
import numpy as np

def compute_graham(row):
    g1 = int(row['MarketCap'] >= 500_000_000)
    g2 = int(row['CurrentAssets'] / row['CurrentLiabilities'] >= 2.0)
    g3 = int(row['LongTermDebt'] <= (row['CurrentAssets'] - row['CurrentLiabilities']))
    g4 = int(row['NegativeEPS_Years_10'] == 0)
    g5 = int(row['DividendYears_Uninterrupted'] >= 20)
    g6 = int((row['AvgEPS_Last3Y'] / row['AvgEPS_10YAgo3Y'] - 1) >= 0.33)
    g7 = int(row['PE_3YAvg'] * row['PB'] <= 22.5)
    return (g1 + g2 + g3 + g4 + g5 + g6 + g7) / 7 * 10

def compute_buffett(row):
    b1 = int(row['ROE_10Y_Avg'] >= 0.15)
    b2 = int(row['ROE_Years_Above15_of10'] >= 8)
    b3 = int(row['GrossMargin'] >= 0.40)
    b4 = int(row['TotalDebt'] / max(row['TotalEquity'], 1) <= 1.0)
    b5 = int(row['EBIT'] / max(row['InterestExpense'], 1) >= 5)
    b6 = int(row['FCF_PositiveYears_10'] >= 8)
    b7 = int(row['FCF_CAGR_5Y'] > 0)
    b8 = int(row['FCF_Yield'] >= 0.05 or row['PE_TTM'] < row['PE_10Y_Median'])
    return (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8) / 8 * 10

def compute_lynch(row):
    l1 = int(row['PEG'] <= 1.0)
    l2 = int(row['AdjPEG'] <= 1.0)
    growth = row['EPS_GrowthRate_5Y']
    l3 = int(0.10 <= growth <= 0.25)
    l4 = int(row['EPS_TTM'] > 0)
    l5 = int(row['TotalDebt'] / max(row['TotalEquity'], 1) <= 1.5)
    l6 = int(row['SharesOutstanding'] <= row['SharesOutstanding_3YAgo'] * 1.05)
    return (l1 + l2 + l3 + l4 + l5 + l6) / 6 * 10

def compute_greenblatt(row):
    # Rank-based: uses precomputed percentile columns
    return ((row['EY_PctRank'] + row['ROC_PctRank']) / 2) * 10

def compute_oshaughnessy(row):
    avg_rank = np.mean([
        row['Rank_PB'], row['Rank_PS'], row['Rank_PE'],
        row['Rank_PCF'], row['Rank_EBITDAEV'], row['Rank_SHY']
    ])
    return (1 - avg_rank) * 10

def compute_neff(row):
    n1 = int(row['PE_TTM'] / row['MarketPE'] <= 0.60)
    growth = row['EPS_GrowthRate_5Y']
    n2 = int(0.07 <= growth <= 0.20)
    n3 = int(row['PE_TTM'] / (growth + row['DividendYield'] + 0.001) <= 1.0)
    n4 = int(row['DividendYield'] >= max(row['SP500_DividendYield'], 0.02))
    n5 = int(row['EPS_TTM'] > 0)
    n6 = int(growth > 0)
    return (n1 + n2 + n3 + n4 + n5 + n6) / 6 * 10

def compute_oneil(row):
    o1  = int(row['QtrEPSGrowth_YoY'] >= 0.25)
    o2  = int(row['AnnualEPS_Consecutive5Y'])
    o3  = int(row['EPS_CAGR_5Y'] >= 0.25)
    o4  = int(row['ROE'] >= 0.17)
    o5  = int(row['Price'] / row['High_52Week'] >= 0.85)
    o6  = int(row['SharesFloat'] <= 50_000_000)
    o7  = int(row['RS_Percentile'] >= 80)
    o8  = int(row['IndustryGroupRank_Pctile'] >= 60)
    o9  = int(row['InstOwners_Current'] > row['InstOwners_PriorQtr'])
    o10 = int(row['Index_Above_MA200'] and row['MA50_Above_MA200'])
    return (o1+o2+o3+o4+o5+o6+o7+o8+o9+o10) / 10 * 10

def compute_piotroski(row):
    f1 = int(row['NetIncome'] / max(row['TotalAssets'], 1) > 0)
    f2 = int(row['CFO'] > 0)
    f3 = int(row['ROA_Current'] > row['ROA_Prior'])
    f4 = int(row['CFO'] > row['NetIncome'])
    f5 = int(row['LTDebt_TotalAssets_Curr'] < row['LTDebt_TotalAssets_Prior'])
    f6 = int(row['CurrentRatio_Curr'] > row['CurrentRatio_Prior'])
    f7 = int(row['SharesOut_Curr'] <= row['SharesOut_Prior'])
    f8 = int(row['GrossMargin_Curr'] > row['GrossMargin_Prior'])
    f9 = int(row['AssetTurnover_Curr'] > row['AssetTurnover_Prior'])
    return (f1+f2+f3+f4+f5+f6+f7+f8+f9) / 9 * 10

def compute_zweig(row):
    z1  = int(row['EPS_CAGR_5Y'] >= 0.15)
    z2  = int(row['AnnualEPS_Consecutive5Y'])
    z3  = int(row['QtrEPSGrowth_YoY'] >= 0.20)
    z4  = int(row['QtrEPSGrowth_Latest'] > row['QtrEPSGrowth_Prior'])
    z5  = int(row['RevGrowth_5Y'] >= row['EPS_CAGR_5Y'] * 0.50)
    z6  = int(row['QtrRevGrowth_YoY'] > 0)
    z7  = int(row['PE_TTM'] <= min(3 * row['MarketPE'], 43) and row['PE_Pctile'] >= 0.10)
    z8  = int(row['DebtToEquity'] < row['IndustryMedian_DE'])
    z9  = int(row['Return_12M'] > row['MarketReturn_12M'])
    z10 = int(row['EPS_Est_Current'] > row['EPS_Est_60DaysAgo'])
    return (z1+z2+z3+z4+z5+z6+z7+z8+z9+z10) / 10 * 10

def compute_dreman(row):
    v1 = int(row['PE_Pctile'] <= 0.20)
    v2 = int(row['PCF_Pctile'] <= 0.20)
    v3 = int(row['PB_Pctile'] <= 0.20)
    v4 = int(row['DivYield_Pctile'] >= 0.80)
    gate = int((v1 + v2 + v3 + v4) >= 2)
    if gate == 0:
        return 0.0
    d2  = int(row['MarketCap_Pctile'] >= 0.30)
    d3  = int(row['CurrentRatio'] >= 2.0 or row['CurrentRatio'] > row['IndustryAvg_CR'])
    d4  = int(row['ROE'] >= row['TopThird_ROE_1500'])
    d5  = int(row['PreTaxMargin'] >= 0.08)
    d6  = int(row['EPS_CAGR_5Y'] > row['SP500_EPS_CAGR_5Y'])
    d7  = int(row['EPS_EstGrowth'] > row['SP500_EstGrowth'])
    d8  = int(row['TotalLiab_TotalAssets'] < row['SectorAvg_LiabToAssets'])
    d9  = int(row['DividendYield'] >= 0.015)
    d10 = int(row['EPS_Est_Current'] > row['EPS_Est_90DaysAgo'])
    return (gate+d2+d3+d4+d5+d6+d7+d8+d9+d10) / 10 * 10


def classify_style(row):
    """Assign a style bucket for adaptive weighting."""
    growth = row['EPS_GrowthRate_5Y']
    pe     = row['PE_TTM']
    mkt_pe = row['MarketPE']
    dy     = row['DividendYield']
    pe_pct = row['PE_Pctile']

    if growth >= 0.20 and pe >= mkt_pe:
        return 'Growth'
    elif growth >= 0.08 and pe <= mkt_pe * 1.5:
        return 'GARP'
    elif pe_pct <= 0.30 and dy >= 0.015:
        return 'Deep Value'
    else:
        return 'Core Value'


# Weight profiles by style (must sum to 1.0 for each style)
WEIGHT_PROFILES = {
    'Growth':     [0.02, 0.08, 0.10, 0.08, 0.05, 0.02, 0.25, 0.10, 0.25, 0.05],
    'GARP':       [0.08, 0.15, 0.18, 0.12, 0.08, 0.08, 0.08, 0.08, 0.10, 0.05],
    'Deep Value': [0.15, 0.10, 0.05, 0.12, 0.15, 0.15, 0.02, 0.15, 0.03, 0.08],
    'Core Value': [0.12, 0.15, 0.12, 0.12, 0.10, 0.10, 0.03, 0.12, 0.05, 0.09],
}
# Order: Graham, Buffett, Lynch, Greenblatt, OShaughnessy, Neff, ONeil, Piotroski, Zweig, Dreman


def compute_composite(row):
    """Main pipeline: compute all guru scores and weighted composite."""
    scores = [
        compute_graham(row),
        compute_buffett(row),
        compute_lynch(row),
        compute_greenblatt(row),
        compute_oshaughnessy(row),
        compute_neff(row),
        compute_oneil(row),
        compute_piotroski(row),
        compute_zweig(row),
        compute_dreman(row),
    ]

    style = classify_style(row)
    weights = WEIGHT_PROFILES[style]

    composite = sum(s * w for s, w in zip(scores, weights))
    final_score = composite * 10  # scale to 0-100

    return {
        'Graham': scores[0], 'Buffett': scores[1], 'Lynch': scores[2],
        'Greenblatt': scores[3], 'OShaughnessy': scores[4], 'Neff': scores[5],
        'ONeil': scores[6], 'Piotroski': scores[7], 'Zweig': scores[8],
        'Dreman': scores[9], 'Style': style, 'FinalScore': round(final_score, 2),
        'GuruAgreement': sum(1 for s in scores if s >= 6.0),
    }


# ============================================================
# USAGE
# ============================================================
# df = pd.read_csv('universe_data.csv')  # your master data
# results = df.apply(compute_composite, axis=1, result_type='expand')
# df = pd.concat([df, results], axis=1)
# df = df.sort_values('FinalScore', ascending=False)
# top_picks = df[df['GuruAgreement'] >= 4].head(30)
```

---

## Appendix A: Handling Missing Data

Some stocks will lack data for certain metrics (e.g., no 20-year dividend history for Graham, or no float data for O'Neil). Handle gracefully:

```
Rule 1: If a guru criterion cannot be computed due to missing data,
        score that criterion as 0 (fail). Do not skip it.

Rule 2: If more than 30% of a guru's criteria are missing for a stock,
        exclude that guru from the weighted composite for that stock.
        Re-normalize the remaining weights to sum to 100%.

Rule 3: If more than 4 gurus are excluded for a stock,
        flag it as "Insufficient Data" and do not rank it.
```

---

## Appendix B: Backtesting Recommendations

```
1. Use point-in-time data (no survivorship bias, no look-ahead bias).
2. Lag financial data by 90 days from fiscal year-end (ensures availability).
3. Rebalance quarterly to match the recommended schedule.
4. Track each guru sub-score's alpha contribution independently.
5. Report: total return, Sharpe ratio, max drawdown, turnover, and sector exposure.
6. Compare against: S&P 500 (cap-weighted), equal-weight S&P 500, and Russell 1000 Value/Growth.
```

---

## Appendix C: Quick Reference — All Criteria Across All Gurus

| Metric Category | Used By (Guru #s) |
|-----------------|-------------------|
| P/E ratio | 1, 3, 5, 6, 9, 10 |
| P/B ratio | 1, 5, 8 (pre-filter), 10 |
| P/S ratio | 5 |
| P/CF ratio | 5, 10 |
| EV/EBITDA (or inverse) | 4, 5 |
| PEG ratio | 3, 6 |
| FCF yield | 2 |
| Dividend yield | 5, 6, 10 |
| Shareholder yield | 5 |
| ROE | 2, 7, 10 |
| ROA | 8 |
| Gross margin | 2, 8 |
| Pre-tax margin | 10 |
| Current ratio | 1, 8, 10 |
| Debt-to-equity | 2, 3, 9 |
| Long-term debt vs NCA | 1 |
| Total liabilities / assets | 10 |
| Earnings growth (annual) | 1, 2, 3, 6, 7, 9, 10 |
| Earnings growth (quarterly) | 7, 9 |
| Revenue growth | 7, 9 |
| Earnings stability (no losses) | 1, 7, 9 |
| Dividend record | 1 |
| Operating cash flow | 2, 8 |
| Accruals quality | 8 |
| Share dilution | 3, 8 |
| Relative price strength | 7, 9 |
| 52-week high proximity | 7, 9 |
| Institutional ownership | 7 |
| Insider activity | 9 |
| Market direction (index trend) | 7 |
| Earnings estimate revisions | 9, 10 |
| Earnings yield (EBIT/EV) | 4 |
| Return on capital (EBIT/invested) | 4 |
| Asset turnover | 8 |
