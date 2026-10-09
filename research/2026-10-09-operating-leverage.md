# Operating leverage: what the Quality category was scoring, and why it stops (2026-10-09)

**Question (CLAUDE.md open item 0.9(a), opened 2026-10-07).** `operating_leverage` carried 8% of
non-bank Quality, scored lower-is-better, with the config's rationale "lower = more durable". The
transparency build found negative values ranking at the top of their sectors. Is the metric
measuring what the screener means by quality, does the literature support the direction it is
scored in, and do practitioners use it? Research-led per CLAUDE.md rule 4 - no IC series, no
backtest number is used below.

**Decision: weight 0, kept as a tracked candidate.** Its 8 points go to the other six Quality
metrics in proportion to their weights. Changelog 2026-10-09.

---

## 1. What the metric is, as built

`factor_engine.py` §7c: the one-year **degree of operating leverage**,

    DOL = (% change in annual EBIT) / (% change in annual revenue)

from the latest two fiscal years, missing when EBIT changes sign, prior EBIT is zero, or revenue
moves less than 1%. Scored lower-is-better (`METRIC_DIR = False`).

## 2. What it measures here - measured on run `a2d76219dc0a` (2026-10-09)

`research/measurements/2026-10-09-operating-leverage.py`, reproducible from the run directory, no
network.

| | |
|---|---|
| Stocks with a value | 392 |
| Negative | **95** |
| of which revenue up, EBIT down (a **margin squeeze**) | **85** |
| of which revenue down, EBIT up | 10 |
| \|DOL\| > 5 / > 10 | 94 / 49 |
| 1st / 99th percentile | -24.7 / +57.4 |
| Mean sector percentile, negative values | **84.1** |
| Mean sector percentile, positive values | 37.2 |
| Share with revenue change under 5% | 31.9% |
| Median \|DOL\| there, vs revenue change over 10% | **4.47** vs 1.64 |
| Spearman with the other six Quality metrics | -0.09 to +0.15 |

Three things follow.

1. **The sign is not cost structure.** A negative DOL is not "low operating leverage"; it means
   EBIT and revenue moved in opposite directions. In 85 of 95 cases that is a company whose sales
   grew while operating profit fell - margin deterioration - and the score ranked those at the
   84th percentile of their sectors on average. The metric was rewarding the opposite of quality.
2. **The size is mostly the denominator.** A third of the universe had revenue move by less than
   5%, and there the ratio's median magnitude is almost three times what it is for companies with
   a real revenue change. Dividing by a small number is what produces the tails.
3. **It is not redundant with anything** (correlations near zero) - but noise is not
   diversification.

## 3. Literature

- **The estimator.** DOL is defined as an elasticity. The textbook two-point ratio used here is the
  crudest estimator of it. Mandelker & Rhee (1984, *JFQA*) estimate it as the slope of log EBIT on
  log sales over a time series; O'Brien & Vanderheiden (1987) argue for a two-stage trend
  approach and that the regression measures relative growth rather than DOL; Stelk, Park & Dugan
  (2015, *J. Financial Economic Policy*) find the two estimators indistinguishable in practice.
  Lord (1998, *Financial Review*) shows that even multi-year time-series estimates tend to fall
  below one - impossible for a firm above breakeven - because prices, variable and fixed costs
  move over the window. *If the multi-year estimators are this fragile, a single-year ratio of
  changes is not an estimate of cost structure at all.*
- **The direction.** The return literature treats operating leverage as a **risk** that is paid
  for, not a defect. Novy-Marx (2011, *Review of Finance* 15(1), 103-134) builds a cost-based
  measure ((COGS + SG&A) / assets) and finds that it predicts the cross-section of returns and
  that sorts on it earn significant excess returns, and that within-industry differences in
  book-to-market reflect differences in operating leverage. García-Feijóo & Jorgensen (2010,
  *Financial Management* 39(3), 1127-1154) ask whether operating leverage can cause the value
  premium; later work finds the relation conditional (García-Feijóo et al. 2024: high-OL
  outperformance only in unconstrained funding conditions) or non-monotonic (Kogan, Li, Zhang &
  Zhu, 2025 working paper). **Nothing found supports scoring lower operating leverage as better
  in a ranking meant to find stocks with better prospects.** The most the literature supports is
  that it is a risk characteristic - and the screener already has a Risk category.
  *Access note: the Novy-Marx and García-Feijóo papers were read at abstract level; the
  Novy-Marx measure's construction is as described by a 2026 AEA paper citing it.*

## 4. Documented practice

- **MSCI Quality Indexes** (methodology 2013/2017/2022): the quality score is the z-scores of three
  winsorised variables - **return on equity, debt to equity and earnings variability**, the last
  defined as the standard deviation of year-over-year EPS growth over five years. No operating
  leverage.
- **AQR, Quality Minus Junk** (Asness, Frazzini & Pedersen 2019, *Review of Accounting Studies*
  24, 34-112), read in full: quality = profitability + growth + **safety**, and safety = low beta,
  low leverage (debt / assets), low bankruptcy risk (Ohlson O, Altman Z) and **low ROE volatility
  (EVOL: standard deviation of quarterly ROE over 60 quarters, at least 12; else annual ROE over 5
  years)**. No operating leverage.

Where the two disagree with the screener's intent, practice is clear about the concept: the
"durability" the config comment describes is measured by **how variable earnings have been**, over
years, not by one year's ratio of changes.

## 5. How this fits the screener as a whole

- Quality without it is ROIC, gross profit / assets, net debt / EBITDA, Piotroski, accruals and
  Beneish - profitability, balance-sheet safety and earnings-quality checks, which is the shape of
  both practitioner definitions above, minus their earnings-variability leg.
- Its weight moves **within Quality**, proportionally: ROIC 27 -> 29, gross profit / assets
  20 -> 22, net debt / EBITDA 18 -> 20, Piotroski 15 -> 16, accruals 5 (5.4 rounds down), Beneish
  7 -> 8. Proportional because no Quality metric's research case changed today; whole numbers
  because `schemas.py` requires each category to sum to 100.
- **Effect, measured** by re-scoring the run's own table with the engine's own functions: rank
  Spearman **0.9976**, **1** change in the top 25, 4 in the top 50, median move 5 places, largest
  36; Quality scores move 1.9 points on average.
- It **stays computed and published** as a candidate (weight 0, `NOT_USED_BECAUSE` says why), so
  the improvement engine keeps a record of it and a reader can still see it.

## 6. What this opens

**Earnings variability is the missing Quality leg** both practitioner definitions share. It needs
five or more years of earnings (MSCI) or 12+ quarters of ROE (AQR); the Yahoo statements the
fetch reads carry four annual years and about five quarters, which is not enough to compute either
faithfully. The SEC's XBRL `companyfacts` endpoint carries a decade of annual and quarterly
figures and the run now has SEC access (2026-10-08) - so an `earnings_variability` candidate is
buildable. It should arrive as a **candidate at weight 0**, with its own note, before it is
weighted.

## Sources

- Novy-Marx, R. (2011). "Operating Leverage." *Review of Finance* 15(1), 103-134. https://ideas.repec.org/a/oup/revfin/v15y2011i1p103-134.html
- García-Feijóo, L. & Jorgensen, R. D. (2010). "Can Operating Leverage Be the Cause of the Value Premium?" *Financial Management* 39(3), 1127-1154. https://ideas.repec.org/a/bla/finmgt/v39y2010i3p1127-1154.html
- Kogan, Li, Zhang & Zhu (2025), working paper on operating leverage and the risk premium, AEA 2026 program. https://www.aeaweb.org/conference/2026/program/paper/yZE4n359
- Lord, R. A. (1998). "Properties of Time-Series Estimates of Degree of Leverage Measures." *Financial Review* 33(2), 69-83. https://ideas.repec.org/a/bla/finrev/v33y1998i2p69-83.html
- Stelk, Park & Dugan (2015). *Journal of Financial Economic Policy* 7(2), 180-188. https://ideas.repec.org/a/eme/jfeppp/v7y2015i2p180-188.html
- Mandelker, G. & Rhee, S. G. (1984), *JFQA*; O'Brien, T. & Vanderheiden, P. (1987) - as summarised in the two papers above.
- MSCI Quality Indexes methodology. https://www.msci.com/eqb/methodology/meth_docs/MSCI_Quality_Indexes_Methodology_May2022.pdf
- Asness, C., Frazzini, A. & Pedersen, L. H. (2019). "Quality Minus Junk." *Review of Accounting Studies* 24, 34-112. https://research-api.cbs.dk/ws/portalfiles/portal/60211462/lasse_heje_pedersen_et_al_quality_minus_junk_publishersversion.pdf
