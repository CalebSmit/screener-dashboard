# Earnings variability: why it stays a candidate (2026-10-09)

Added this morning as a weight-0 Quality candidate (`sec_fundamentals.py`, changelog
2026-10-09): the sample standard deviation of five annual ROEs from the SEC's XBRL frames - AQR
Quality Minus Junk's annual EVOL rule. Both practitioner definitions of quality carry an
earnings-stability leg (MSCI: one third of the score; AQR: one of five safety inputs), so the
question is whether this one should carry weight here. **Measured on run `a2d76219dc0a`, the answer
is not yet, and not in this form.**

## What it overlaps

Spearman rank correlation with `earnings_variability` (higher = more variable), 442 stocks:

| Metric | rho | Reading |
|---|---|---|
| ROIC | **+0.21** | more profitable companies have *more* variable ROE |
| Gross profit / assets | **+0.25** | same |
| ROE | +0.19 | same - the standard deviation scales with the level |
| Accruals | -0.28 | |
| Return volatility (Risk) | **+0.44** | overlaps the Risk category |
| Max drawdown (Risk) | -0.27 | deeper drawdowns, more variable ROE |
| Quality category score | **+0.13** | |
| Risk category score | -0.27 | |

Within sectors the correlation with the Quality score ranges from -0.20 (Materials) to +0.51
(Energy).

**Two consequences.** (1) Scored lower-is-better inside Quality, it would partly *cancel* the
profitability leg: the measure rises with the level of ROE (a company earning 40% on equity swings
by more percentage points than one earning 8%), so it would mark down the most profitable firms
for being profitable. Apple, whose ROE runs 127-176% on a buyback-shrunk equity base, has a
variability of 0.20 against Coca-Cola's 0.013. (2) It double-counts risk: its strongest overlap is
with return volatility, which the Risk category already scores at 43% of that category.

Coverage is also uneven: 59 of 501 missing, concentrated in IT (14) and Consumer Discretionary
(13) - mostly negative or near-zero equity years, where ROE is undefined.

## What would fix it

The scale problem is specific to ROE on book equity. Two variants avoid it and are worth measuring
as candidates before anything is weighted:

- **Volatility of ROA** (net income / total assets): no equity denominator, so buybacks and
  negative equity stop mattering; still a level-scaled dispersion.
- **Variability of earnings growth** (MSCI's EVAR: standard deviation of year-over-year EPS
  growth over five years): scale-free by construction, but undefined around zero or negative EPS.

Then the test that matters: does the variant add information **after** ROIC, gross profit / assets
and the Risk category - i.e. its correlation with the residual of the Quality score. That is a
Wednesday synthesis question, and the improvement engine is already recording the candidate's
record against forward returns.

## Decision

`earnings_variability` stays at **weight 0**. No change to the scoring today. Next: add an ROA-based
variant beside it (same frames, one more concept: `Assets`), then compare the two on overlap and,
once `1m` observations accrue, on record.
