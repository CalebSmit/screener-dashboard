# Factor Lab experiment 19: Mohanram (2005), the growth score, on Open Source Asset Pricing

**2026-11-06, 13:00 code session.** Queue row: the Growth category (`plan/factor-lab.md`, "Cannot
be rerun on free data" - *"Next: `MS` gives Growth one on its own measures"*).
**Result: 4 of 7 figures within tolerances committed before the comparison was run
(`4a55141`).** Research only: **no weight, metric or threshold changed, and none is proposed.**

---

## 1. Why this paper, and what it corrects

Growth was the one of the screener's eight categories with **no rerunnable experiment at all**.
Worse, the lab's own category table said so in a way that turned out to be false. Its Growth row
read:

> "Mohanram (2005), which it cites, needs company accounts, and the library has no sort on sales
> or earnings growth."

The second half is still true. The first half is not: Chen & Zimmermann's Open Source Asset
Pricing publishes **portfolio returns for exactly this score** (signal `MS`), free, with the
construction code in the open. That is the audit `plan/factor-lab.md` calls "replication as an
audit of the screener's own citations" working in the direction nobody plans for - the lab's claim
about what *cannot* be checked was the thing that needed checking.

## 2. The paper

Mohanram, P. S. (2005), "Separating winners from losers among low book-to-market stocks using
financial statement analysis", *Review of Accounting Studies* 10(2-3), 133-170,
[doi:10.1007/s11142-005-1526-4](https://doi.org/10.1007/s11142-005-1526-4).

**Obtained and read** (pypdf) as the **accepted manuscript** in the University of Toronto's TSpace
repository - the April 2004 draft, 47 pages. The published version of record was **not obtained**;
a journal's revision can change a table, and that is recorded on the page.

**Method.** All COMPUSTAT firms with price and book value between **1979 and 1999**, restricted to
the **lowest book-to-market quintile** - growth stocks. Eight signals, each 0 or 1, each against
the median of the firm's **two-digit industry among low-BM firms** (PDF p. 36):

| | Signal | What it says |
|---|---|---|
| G1 | ROA >= industry median | profitable for its industry |
| G2 | cash-flow ROA >= industry median | the profits are cash |
| G3 | cash-flow ROA >= ROA | earnings are not accrual-driven |
| G4 | variance of ROA <= industry median | earnings are steady |
| G5 | variance of sales growth <= industry median | growth is steady |
| G6 | R&D / assets >= industry median | spends on research |
| G7 | capex / assets >= industry median | spends on capacity |
| G8 | advertising / assets >= industry median | spends on the brand |

G1-G3 are profitability and cash flow; G4-G5 are the "naive extrapolation" signals (the market
over-extrapolates volatile growth); G6-G8 are "conservatism" (accounting writes these off, so the
spending is hidden value).

**Table 4, Panel A** (PDF p. 39), one-year raw returns, each firm held a year from four months
after its fiscal year end, 20,866 firm-years. Scores 0 and 8 are too thin to trade (589 and 72
firm-years in 21 years), so the paper pools **LOW = 0,1** and **HIGH = 6,7,8** (p. 18):

| G_SCORE | LOW (0,1) | 2 | 3 | 4 | 5 | HIGH (6,7,8) | HIGH - LOW |
|---|---|---|---|---|---|---|---|
| mean one-year return | -4.0% | 5.1% | 7.8% | 11.6% | 16.5% | 17.4% | **21.4%** (t 10.71) |

**The paper's own caveat, which matters more to this screener than its headline.** Panel B is the
same table size-adjusted: the strong group earns **+3.3%** and the weak group **-17.9%**. The
paper says it plainly (p. 19): the strategy "is more effective in identifying potential losers or
torpedo stocks than in identifying winners". A long-only screener can use only the half that is
smaller.

## 3. What the public portfolios are

Open Source Asset Pricing, October 2025 release (v2.0.0), signal `MS`. Its construction code
(`Signals/pyCode/Predictors/MS.py` at tag v2.0.0) was read **before** the tolerances were written.
Two things decided the design of this experiment:

1. **The library pools the ends exactly as the paper does.** The last lines of `MS.py` map scores
   6-8 to 6 and scores 0-1 to 1. The score therefore takes six values, and because `MS` is
   declared *discrete*, the library's portfolio code (`Portfolios/Code/01_PortfolioFunction.R`)
   gives **one portfolio per value** rather than deciles. So its six portfolios **are the paper's
   six groups** - which is why this comparison can be row-by-row where experiment 17's (Piotroski,
   on deciles of a 0-9 score) could only be end-to-end.
2. **A discrete signal is not in the decile files.** `PredictorAltPorts_DecilesEW` holds 179
   continuous signals and not `MS`. The portfolios come from the library's own cut
   (`Portfolios/Full Sets OP/PredictorPortsFull.parquet`), and the market-value view from the same
   portfolios forced to value weights (`PredictorAltPorts_LiqScreen_VWforce.parquet`), both from
   the **same release's** Hugging Face branch, so every OSAP figure in the lab comes from one
   release.

**Differences from the paper, written down before the run** (all six are on the page under "What
this leaves open"): monthly re-weighting against a one-year buy-and-hold; the score fixed six
months after the fiscal year end rather than four; quarterly accounts (four-quarter sums over
average assets) rather than annual; the steadiness signals as 48-month standard deviations of
quarterly figures rather than the variance of three to five annual ones; strict `>` where the
paper's signals fire at or above the median; and twenty years of CRSP/Compustat revisions.

## 4. The pre-registration, and one disclosure

Tolerance **4.0 percentage points a year** on each of the seven figures: two standard errors of
the paper's own spread (21.4 / 10.71 = 1.998), which is also about the sampling error of a 24-year
average of monthly spreads. The **t-statistic is deliberately not compared** - the paper's 10.71
is a two-sample t across 5,585 firm-years and anything computed here is a t across monthly
spreads. A different quantity is not a wider version of the same one; comparing them is how
experiment 17 bought itself a guaranteed miss.

**Rule 8, worked out from the record before the run:** a result of no effect - every group at the
sample's 8.2% a year - matches **3 of 7** (the 2, 3 and 4 groups) and misses both ends, the 5
group and the spread. The build computes the same count independently and publishes it beside the
verdict; it says 3.

**The disclosure.** `SignalDoc.csv`'s summary columns for `MS` were read while reading the
documentation, and the record says so: a long-short return of 1.575% a month, t 9.14. I read that
as the library's own measured summary, and wrote in the record that the spread was therefore
expected to land inside its tolerance (1.575 x 12 = 18.9% against 21.4%).

**That reading was wrong, and the result shows it.** The column is the **paper's** figure as the
library transcribed it, not a measured one - checked afterwards against `PS`, whose entry reads
1.958333, and 1.958333 x 12 = 23.5%, Piotroski's published spread to the decimal. So the
pre-registration was blinder than its own disclosure claimed, and the expectation it stated was an
expectation about *the paper*, not a peek at the answer. The measured spread is 13.3%, outside.

## 5. The result

288 months, 1978-2001, equally weighted - the library's documented window for this paper.

| Figure | Paper | Here | Diff. | Within ±4.0 |
|---|---|---|---|---|
| Score 0-1 (weakest): % a year | -4.0 | **+3.22** | +7.22 | no |
| Score 2 | 5.1 | 5.40 | +0.30 | yes |
| Score 3 | 7.8 | 6.48 | -1.32 | yes |
| Score 4 | 11.6 | 12.75 | +1.15 | yes |
| Score 5 | 16.5 | **7.58** | -8.92 | no |
| Score 6-8 (strongest) | 17.4 | 16.48 | -0.92 | yes |
| Strongest minus weakest | 21.4 | **13.26** | -8.14 | no |

t of the measured spread, for context and not compared: **4.96**.

**Three readings.**

1. **The strong end reproduces; the weak end does not.** 16.5% against 17.4% at the top, +3.2%
   against -4.0% at the bottom. The entire shortfall in the spread is the losers the public
   construction fails to find. Given the paper's own Panel B - where the weak end is where the
   effect lives - the half that reproduces is the half a long-only screener can act on, and the
   half that does not is the half it could never use. That is a better outcome for this screener
   than the headline number suggests, and a worse one for the paper's claim.
2. **The middle is not monotonic.** Score 5 earns 7.58%, below score 4's 12.75%. The paper reports
   an "almost perfectly monotonic" rise. With one group out of order, the public construction
   supports the *ends* of the claim and not its shape.
3. **It survived its own sample, which almost nothing in this lab has.** Equally weighted, the gap
   is **5.04% a year since 2002** (t 2.32) and **6.94% since 2007** (t 2.72); by market value -
   the weighting that speaks to large companies - **3.67% since 2002** (t 1.52) and **6.01% since
   2007** (t 2.21). Of the lab's 19 experiments, this is one of a very small number whose result
   is still distinguishable from zero after publication, and the only one of those in the
   fundamentals half. The lab's front-page table now counts 23 of 27 directional results weaker
   after their samples; this is one of the four that are not.

The strongest group was also the **calmest** (6.24% a month against 7.18-7.89% for the others) -
the same pattern experiment 10 found for profitability, and an argument the Risk category usually
makes being made by a fundamentals measure.

## 6. What this says for the screener - and what it does not

**It does not justify changing Growth's weight, and nothing here is proposed as one.** The route
from the lab to a weight is `plan/factor-lab.md`, "From the lab to the weights": the lab raises a
question, a research note answers it from the literature *and* from practice, a changelog entry
makes one change at a time. This is the lab raising a question.

**The question it raises is the open Growth one - composition, not weight** (CLAUDE.md: *"The open
Growth question is composition, not weight"*). Growth is 13% of the composite and scores revenue
growth, earnings growth, forward earnings growth and operating leverage: **growth rates**. The
evidence this rerun supports is not about growth rates. It is about **sorting growth companies on
whether their fundamentals support the valuation** - and that evidence is the stronger of the two:

- Sorted on a growth rate (experiment 14, asset growth, by market value), the premium **since its
  paper is -0.07% a month, t -0.29** - about zero.
- Sorted on fundamental strength *within* growth stocks, the premium since its paper is **5.0% a
  year equally weighted, t 2.3**, and **6.0% by market value since 2007, t 2.2**.

Four of Mohanram's eight signals are things the screener already scores, but in **other
categories** - G1/G2 in Quality (ROA, cash-flow quality), G3 in Quality's accruals, G7 in
Investment. Three are not scored anywhere: the two steadiness signals (G4, G5) and research
intensity (G6). Advertising intensity (G8) is not available in the screener's data at all.

So the honest statement of the open question, for a Monday: **does the Growth category's case rest
on growth rates, which the public evidence does not support, when the evidence it cites supports
scoring growth companies on steadiness and on investment in the future instead?** Three things a
note would have to answer before anything moved:

1. Whether the steadiness signals (variance of ROA, variance of sales growth) belong in Growth, in
   Quality, or in Risk - the screener already scores volatility in Risk, and
   `research/2026-11-02-investment-inside-quality.md` shows how little a weight-preserving move
   between categories is actually neutral (renormalisation, the panic rule, the profiles, trap
   flags).
2. Whether any of this survives inside the S&P 500. Every figure here is an all-listed-stock sort
   of the lowest book-to-market *quintile*, most of them small; the screener ranks 500 large
   companies **within their sectors**. The market-value weighting is the nearest view and it
   holds, which is more than most of the lab's sorts manage - but it is not a large-company test.
3. What the practitioners do. Mohanram's score is in use (Validea runs a public "Mohanram
   G-score" screen), and the index providers' growth definitions are built on growth rates rather
   than on this. That disagreement is exactly what a Monday note is for.

## 7. What was checked, and what a second reader should check

- The paper's seven reference figures were read from the PDF table, not from memory, and each
  carries its page in the record.
- The comparison runs in the build (`reproduce_sort`) and again in the browser
  (`lab_engine.js`), and a test asserts the two agree for every experiment.
- `null_within` is computed by the code, not typed: it says 3, as the record predicted.
- The page was opened at 1440 px and 375 px: no page errors, no sideways scroll, the six bars and
  all seven comparison rows present at both widths.

**Still open for a second reader** (`OWNER_FOCUS.md` 2026-10-20 item 2 - an experiment is not
checked until someone who did not write it has read it):

- The **version of record** was not obtained; only the accepted manuscript. Table 4 Panel A should
  be re-read from the published article if a copy can be had.
- Every measured figure should be recomputed **without importing `factor_lab`**, from the two
  parquet files, as experiments 15, 17 and 18 each had done.
- The claim in section 4 that `SignalDoc`'s `Return` column is the paper's figure rather than a
  measured one rests on one check (`PS` = 23.5%). A second reader should test it on two or three
  more signals.
