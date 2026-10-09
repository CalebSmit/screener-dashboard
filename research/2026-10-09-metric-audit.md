# Audit of every weighted metric - 2026-10-09 (owner-run)

**Why.** Three metrics fixed earlier the same day (`revenue_growth`, `forward_eps_growth`, Piotroski
3/8/9) all measured a different window or basis than their label. Two read-only audit passes then
went through every other weighted metric, in code and on run `a2d76219dc0a` (02:00, 501 stocks), with
Yahoo's statements cross-checked against each company's SEC filings (`data/sec/pit/facts.parquet`)
and a live sample of 152-335 tickers. Each defect below was verified in code and measured; the three
headline figures were re-measured independently before this note was written (GOOGL operating
income, AMCR's zero estimate, the regime replay).

## Fixed (each with a `METHODOLOGY_CHANGELOG.md` entry)

| # | Defect | Measured | Fix |
|---|---|---|---|
| 1 | **"EBIT" was Yahoo's pretax income + interest**, so it carried non-operating gains, feeding ROIC, EV/EBITDA and net debt / EBITDA | GOOGL $301.5B vs SEC operating income $147.6B (ROIC ~doubled, 88.6th pct); MSFT 169.0 vs 155.2; 80 of 317 non-banks >10% above operating income; ROIC sector percentile moves >10 pts for 44 stocks, EV/EBITDA for 40 | Operating income first; "EBIT" only where no operating-income line exists (Greenblatt 2006; Koller et al., *Valuation*: NOPAT excludes non-operating income) |
| 2 | **The EBITDA fallback was EBIT**: with no quarterly D&A, Yahoo's "EBITDA" row was used, and it equals its EBIT row for all six stocks that reached it | DAL EV/EBITDA 12.29 -> 8.52 (sector pct ~78 -> 98), UAL 9.30 -> 6.05 | Fall back to the last fiscal year's cash-flow D&A; treat a reported EBITDA equal to EBIT as missing |
| 3 | **Statement lookups read the k-th non-blank value, not the k-th period** | BRK-B's TTM net income summed Q2'25 in place of a blank Q3'25: $67.3B vs $85.8B (-21%), feeding earnings yield, accruals, Piotroski 1/2/4; Beneish inputs from the wrong year in 10 of 127 sampled non-banks | Lookups by period over the columns that hold data; a blank cell is that period, missing; TTM needs four adjacent quarters (3 adjacent, annualised, as before) |
| 4 | **Yahoo's 0.0 placeholder in `eps_trend` read as a real estimate** | AMCR's 90-day-ago FY1 estimate 0.0 -> revision +9.6% of price (100th pct in Materials); LIN's current 0.0 -> -3.7% (4th pct); ~2 composite points each; VMRK too | Exact 0.0 is missing (a consensus is never exactly zero to the cent); also for FY2 |
| 5 | **Jensen's alpha set a total-return stock against the price index**, and the caveat misstated the effect | Index total minus price return over the year 1.37pp; the tilt is beta x 1.37pp (SNDK ~5.5pp), not "the dividend yield"; 167 of 499 alpha percentiles move | ^SP500TR for alpha and beta, ^GSPC only as a logged fallback |
| 6 | **The momentum "volatility regime" did not measure volatility** | Its input is the cross-stock spread of `momentum_score`, fixed by the percentile construction (rank-predicted 25.03 vs measured 25.07; correlation with S&P 500 realised volatility +0.37). Replayed over `factor_vol_history.csv`: **LOW in 30 of 33 runs, HIGH never** - momentum 13 -> 14.95 most days | Rule off (`momentum_regime.enabled: false`), history still recorded; the methodology page said it "tracks market-wide volatility ... in `cache/vol_history.csv`" - corrected |
| 7 | **Percentile ranks were biased by direction and sector size**: `rank(pct=True)` runs 1/n..1, flipped 0..1-1/n | Higher-is-better metrics average 50+50/n, lower-is-better 50-50/n (Energy 52.38 vs 47.62); ~1.0 composite point for Energy against ~0.3 for Industrials | Midpoint rank, (k - 0.5)/n: both directions average exactly 50 |
| 8 | **Surprise quarters**: not sorted by date (ACN shuffled), never checked for staleness (AMCR's newest Dec-2025 against a reported Jun-2026), and the beat score capped at 6 when fewer than four quarters had data | ~1-2% of stocks | Sorted by date; not scored when the newest quarter is >200 days old; acceleration needs the two latest quarters; beat score = weight share x 10 (identical with four quarters) |
| 9 | **False page text**: beta "about 13 months" (it is 12); Piotroski 5-7 "otherwise with the earliest it has" (untestable since the balance-sheet guard) | - | Corrected in `metric_lineage.py` |

## Deferred - each needs research before a change (rule 4), now on the queue

* **Which financials are "bank-like".** 26 of the 59 stocks scored on bank metrics get there only by
  the default for an unlisted Financials industry: asset managers (BLK, TROW, BEN, IVZ, ...), insurance
  brokers (AON, AJG, BRO, WTW, ...), broker-dealers and COIN. Equity ratio puts TROW at the 100th
  percentile; P/B puts AON at the 8th. Broker-dealers (GS, MS) arguably *are* bank-like; asset managers
  and brokers have conventional P&Ls. Needs a note on how practitioners split financials (MSCI/GICS
  sub-industries, Morningstar) before the list changes.
* **`earnings_acceleration` (20% of Revisions) works against its own category.** Latest minus prior
  surprise: Spearman +0.37 with the latest surprise but -0.57 with the prior one; surprises are
  positively autocorrelated (Bernard & Thomas 1990), so it penalises a stock that beat last quarter. Its
  extremes are mostly one-off GAAP items (REIT property sales). Candidates: the latest quarter's
  price-scaled surprise (SUE, Livnat & Mendenhall 2006) or He & Narayanamoorthy's earnings
  acceleration.
* **Accruals counted three times within Quality** (the accruals metric, Piotroski signal 4, Beneish
  TATA all use net income minus operating cash flow).
* **Net debt** subtracts cash only while enterprise value subtracts cash and short-term investments
  (14 non-banks net cash by one definition, net debt by the other).

## Checked and correct

12-1 and 6-1 returns (dividend- and split-adjusted, calendar endpoints, skip month); volatility (log
returns x sqrt 252); beta date alignment (cap-weighted mean 1.09); max drawdown (the 2026-10-08 fix
holds); FY1 rollover; price-target upside (no value at either clamp; >= 3 analysts); all 12 metric
directions; 3-year revenue CAGR endpoints (444 of 446 exactly three fiscal years apart); TTM timing
(net income and operating cash flow end on the same quarter); bank P/B (59 of 59 exact); size.
