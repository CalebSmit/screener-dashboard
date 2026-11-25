# Factor Lab experiment 20: Ang, Hodrick, Xing & Zhang (2006), idiosyncratic volatility

**2026-11-25, 13:00 code session.** Queue row 6 (`plan/factor-lab.md`): *"Still open for Risk: the
paper's own headline sort (`_RESVAR`)."*
**Result: 3 of 6 figures within tolerances committed before the comparison was run (`1c9aa54`).**
Research only: **no weight, metric or threshold changed, and none is proposed.**

---

## 1. Why this experiment, and why today

The lab's 21-day bound (`health_numbers.last_advanced`) passes on 2026-11-28 and Friday 2026-11-27
is a retrospective (ISO week 48), so the morning session's Next handed the lab to the weekend. The
13:00 queue session had no carried item due today (the implied-volatility switch is dated
2026-11-26), so it took the lab's queue.

The queue's own "Next" was `EarningsSurprise` (Foster, Olsen & Shevlin 1984). **That paper was not
obtained**: it is in *The Accounting Review* 59(4), 574-603, on JSTOR, with no working-paper copy
found (two searches; Stanford GSB lists only an abstract). The lab's rule is to read the table
before writing a tolerance, so that experiment waits for a copy rather than being built from the
secondary summaries (OSAP's `SignalDoc.csv` gives 2.975 and "4 Days +1 to +60", which is not the
table). It stays first in the Revisions row's queue.

Risk's open row was the next one that could be finished in one session: the same working paper as
experiment 5, already open, and a file the library publishes in the layout experiment 5 uses.

## 2. The paper

Ang, A., R. J. Hodrick, Y. Xing and X. Zhang (2006), "The cross-section of volatility and expected
returns", *Journal of Finance* 61(1), 259-299,
[doi:10.1111/j.1540-6261.2006.00836.x](https://doi.org/10.1111/j.1540-6261.2006.00836.x).

**Read** (pypdf) as **NBER working paper 10852**, October 2004
(`https://www.nber.org/system/files/working_papers/w10852/w10852.pdf`, 301,287 bytes, `%PDF`). The
published article was **not obtained**; experiment 5 has the same limitation and the page says so.

**The table reproduced** - Table VI, Panel B (PDF page 50, printed page 48), "Portfolios Sorted by
Idiosyncratic Volatility Relative to FF-3". Value-weighted quintiles, formed monthly on the
volatility of daily residuals from the Fama-French three-factor model over the previous month
(their "1/0/1" strategy), July 1963 to December 2000, monthly means of total (not excess) returns:

| Quintile | Mean | Std dev | % market share | CAPM alpha | FF-3 alpha |
|---|---|---|---|---|---|
| 1 (least) | 1.04 | 3.83 | 53.5% | 0.11 | 0.04 [0.99] |
| 2 | 1.16 | 4.74 | 27.4% | 0.11 | 0.09 [1.51] |
| 3 | 1.20 | 5.85 | 11.9% | 0.04 | 0.08 [1.04] |
| 4 | 0.87 | 7.13 | 5.2% | -0.38 | -0.32 [-3.15] |
| 5 (most) | -0.02 | 8.16 | 1.9% | -1.27 | -1.27 [-7.68] |
| 5-1 | -1.06 [-3.10] | | | -1.38 [-4.56] | -1.31 [-7.00] |

Newey-West t-statistics in brackets, as the table prints them. The text (page 23) states the same
-1.06 and -1.31 (t -7.00).

**What the paper says about the measurement window** (Section F, page 30, Table X): with a 12-month
formation window ("12/1/1") the 5-1 FF-3 alpha is still -1.12% a month, and -0.77% for "12/1/12".
So the paper itself finds the effect does not depend on its one-month window. Its NYSE-only sort
(Section C.1, page 24) gives a 5-1 alpha of -0.66% (t -4.85) - about half the all-stock figure.

## 3. What the public portfolios are

Kenneth French's Data Library, "Portfolios Formed on Residual Variance"
(`Portfolios_Formed_on_RESVAR_CSV.zip`, built from the 202609 CRSP database - a month later than
the other sorts the lab holds, which are 202608). From the file's header and the library's detail
page (`det_port_form_resvar.html`, read 2026-11-25): monthly quintiles and deciles on **the
variance of the residuals from the FF three-factor model, estimated over 60 days (minimum 20) of
lagged daily returns, with NYSE breakpoints**, all NYSE, AMEX and NASDAQ stocks, financials and
utilities included. Value- and equal-weighted returns, July 1963 to September 2026.

The documented differences from the paper, then: a 60-day window against one month; NYSE
breakpoints against all-stock breakpoints; and (not stated by either source, so not asserted) any
difference in the share-code and price filters.

## 4. The pre-registration

Committed in `1c9aa54` before any return in the RESVAR file was read. Seen first, and disclosed in
the record: the paper's table; the file's text lines (section names and column headers, printed by
a script that skipped every numeric line); the library's detail page; and **experiment 5's result
on the total-volatility sort**, where the most volatile fifth earned 0.81% against the paper's 0.09.

- **Figures:** the five average returns and the 5-1 gap - six figures.
- **Tolerances:** 0.10 points a month for a mean, 0.20 for the gap - experiment 5's, because it is
  the same table of the same paper on the same library's construction.
- **Rule 8:** experiment 5 also scored the five volatilities, which a null result keeps at the
  paper's values; that is why its null matches 6 of 11. This record leaves the volatilities out of
  the count (they stay on the page), so a null - every fifth at 0.85, the gap at zero - matches the
  fourth fifth only: **1 of 6**, missing the headline.
- **Prediction written down:** the three calmer fifths within; the most volatile fifth and the gap
  outside; the fourth either way - **3 or 4 of 6**.

## 5. The result

July 1963 - December 2000, value-weighted, 450 months:

| Figure | Paper | Library | Difference | |
|---|---|---|---|---|
| Least idiosyncratic volatility | 1.04 | 1.07 | +0.03 | within |
| Second fifth | 1.16 | 1.10 | -0.06 | within |
| Middle fifth | 1.20 | 1.25 | +0.05 | within |
| Fourth fifth | 0.87 | 1.25 | +0.38 | outside |
| Most idiosyncratic volatility | -0.02 | 0.76 | +0.78 | outside |
| Most minus least | -1.06 | -0.32 (t -1.18) | +0.74 | outside |

**3 of 6, as predicted.** The library's volatilities rise from 3.81 to 7.45 (the paper's 3.83 to
8.16): its most volatile fifth is the tamer group, as on the total-volatility sort.

**After the paper** (2001 - September 2026, 309 months), value-weighted: gap +0.09 (t 0.24); equally
weighted -0.27 (t -0.72). Since 2010 value-weighted +0.24 (t 0.54).

### 5b. The risk-adjusted gap - measured, not scored

The paper's stronger figure is the FF-3 alpha of the gap. No tolerance was written for it before
the run, so it is **not in the count**; it is measured here and stated on the page as an open
detail, checked by a test against the published series
(`research/measurements/2026-11-25-idiosyncratic-volatility-alpha.py`; `factor_lab.ff3_alpha`,
matched to a numpy OLS with Newey-West errors to three decimals):

| Period | Factors | 5-1 FF-3 alpha | NW t (6 lags) | Loadings mkt / SMB / HML |
|---|---|---|---|---|
| 1963-07 - 2000-12 | three-factor file | -0.562 | -4.07 | +0.32 / +1.27 / -0.30 |
| 2001-01 - 2026-09 | three-factor file | -0.564 | -2.15 | +0.75 / +1.09 / -0.47 |
| 1963-07 - 2000-12 | lab's 2x3 file | -0.536 | -3.71 | +0.31 / +1.25 / -0.47 |
| 2001-01 - 2026-09 | lab's 2x3 file | -0.583 | -2.22 | +0.76 / +1.02 / -0.67 |

So the raw miss is largely composition: the library's most volatile fifth loads more than one-for-one
on the small-company factor and positively on the market, and the premia those loadings carried
make up the difference between the raw gap and the alpha (by construction of the regression, the
raw mean is the alpha plus each loading times its factor's mean). Once that is
allowed for, **about 43% of the paper's alpha remains in its own years (-0.56 against -1.31), and
about the same alpha has persisted since** - which is close to the paper's own NYSE-only figure
(-0.66), as NYSE breakpoints would lead one to expect. That last comparison is suggestive, not
tested.

## 6. What this says for the screener - and what it does not

The screener's Risk category (10 points) scores **total** volatility (42.86% of the category), beta
(28.57%) and the one-year maximum drawdown (28.57%) - not idiosyncratic volatility.

- **What it supports:** the low-volatility end is not penalised. In every period and weighting
  measured, the calmest fifth earned about what the paper found and at the lowest volatility. That
  is the half of the anomaly a long-only screener holds.
- **What it does not support:** a *return* claim for the category on raw returns. The raw gap is
  small and not distinguishable from zero in any period here, as on the total-volatility sort
  (experiment 5) and the beta sort (experiment 7). The methodology page already says the extra risk
  "has not been paid for" rather than that calm stocks earn more (`tests/test_factor_lab.py`,
  `test_the_methodology_states_risk_and_size_as_the_lab_measures_them`); nothing here changes that
  sentence.
- **The risk-adjusted half:** after the three factors, the most volatile stocks have underperformed,
  before and after the paper. Among the large companies the screener ranks, this is weaker still
  (the paper's own size-quintile table, Table VII, found the largest quintile's 5-1 alpha
  insignificant at 5%). It is a reason the Risk category is not a mistake; it is **not** a reason to
  raise its weight, and it is not large-company evidence.
- **A question for a Monday, not a proposal:** whether the Risk category should score idiosyncratic
  rather than total volatility. In this screener total volatility and beta together approximate
  it, and the two are not separable on public sorts among large companies. Rule 4 and the lab's
  route to a weight apply; nothing is queued for a build.

## 7. What was checked, and what a second reader should check

Checked this session: the paper's Table VI Panel B against its own text (page 23: -1.06, -1.31,
t -7.00 - both agree with the table); the null count from the record before the run (1 of 6, pinned
in the suite); the build's comparison recomputed in a scratch script and by
`reproduce_sort`; the alpha regression against an independent numpy implementation (identical to
three decimals) and against a planted intercept (`test_ff3_alpha_recovers_a_planted_intercept`);
the page at 1440 and 375 px (no console errors, no horizontal overflow; the menu reads "20
experiments", the verdict "3 of 6 ... A result of no effect at all would match 1 of these 6").

**For a second reader (a different session):** re-download the working paper and confirm the six
reference figures in Table VI Panel B; recompute the six library figures and the two alphas without
importing `factor_lab`; confirm in `git log` that `1c9aa54` precedes every figure; and try to
obtain the published article's Table VI.
