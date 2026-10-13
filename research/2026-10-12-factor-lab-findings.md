# What the Factor Lab's first three reruns say about the screener - 2026-10-12 (owner-run)

**Status:** observations, not changes. Nothing here moved a weight or a threshold, and nothing
here is sufficient to. It is written down so the sessions that own the methodology start from
what was measured instead of rediscovering it.

**What these numbers are:** monthly returns of the long-short factor portfolios in the Kenneth R.
French Data Library - the papers' own kind of data, no trading costs. They are **not** a backtest
of the screener (CLAUDE.md rule 5 is about `backtest.py`, and stays in force) and they are not
this system's information-coefficient series (rule 4). They are replications of published
results, with a look at what happened after each paper's sample ended. Reproduce any of them with
`python factor_lab.py --report`, or on `lab.html`.

## 1. The three papers reproduce

| Experiment | Within tolerance | What missed, and why |
|---|---|---|
| Barroso & Santa-Clara (2015), risk-managed momentum | 13 of 14 | The risk-managed worst month (-24.16% here, -28.40% in the paper). The paper took its pre-1963 daily returns from another source. |
| Daniel & Moskowitz (2016), momentum crashes | 9 of 15 | Six figures that describe the level of the momentum portfolio. The paper builds its own deciles; they earn about three points a year more than the library's. Every figure about what follows a market decline is within. |
| Fama & French (2015), five factors | 20 of 21 | Profitability's monthly volatility (2.25% here, 2.14% in the paper), by 0.01 over the tolerance. |

Tolerances for the second and third were written before their comparisons were run; the
first's were written after, and its page says so.

## 2. After their samples ended, the results weakened

| | In the paper's sample | After it |
|---|---|---|
| Plain momentum, Sharpe ratio | 0.54 (1927-2011) | 0.20 (2012 to 2026-08) |
| Risk-managed momentum, Sharpe ratio | 1.00 | 0.37 |
| Value factor, average month (t) | +0.38% (3.36), 1963-2013 | -0.06% (-0.19), 2014 to 2026-08 |
| Profitability | +0.27% (2.94) | +0.18% (0.93) |
| Investment | +0.32% (3.99) | -0.08% (-0.42) |
| Size | +0.28% (2.26) | -0.19% (-0.79) |

**How much to read into this.** Less than the table invites. 152 months is short for a factor:
one that truly averages 0.3% a month with 2.8% monthly volatility has a standard error of about
0.23% over that span, so its expected t-statistic is only about 1.3. Seeing -0.2 instead is
roughly one and a half standard errors low - weak evidence of a smaller premium, not proof of
none. The literature expects some decay: McLean & Pontiff (2016, *Journal of Finance* 71(1))
find anomaly returns lower out of sample and lower again after publication.

**What it is worth doing with.** Three things, all research, none of them a weight change today:

1. **Monday's research queue.** "What do practitioners now assume the value and size premia are
   worth, and how did they update after 2007-2020?" is a note worth writing properly, with the
   literature on both sides (Arnott et al. on value's drawdown; Fama & French 2021, "The value
   premium"; AQR's "Is (systematic) value investing dead?"). Verify those references when
   reading them - they are listed from memory as starting points, not as checked citations.
2. **The methodology page should say it.** The page cites long-run premia. A sentence that
   the same factors have earned little since 2014, with the caveat about short samples, is the
   honest version. Do it in the generator, with a registered claim.
3. **Size.** The size factor is the weakest in both periods and the methodology already admits
   there is no published size premium within large stocks. Plan item 4 (size deciles within
   the largest two) is the experiment that speaks to the Size weight directly.

## 3. Value and investment are largely the same bet

Over the paper's sample the value factor's correlation with the investment factor is 0.70, the
other four factors explain half of its variation, and what they leave unexplained is -0.01% a
month (the paper: -0.04%). After 2013 the unexplained part is +0.04%.

The screener carries **both** a Valuation category and an Investment category. That is not
wrong - Fama and French keep value in their model too, for describing portfolios - but it means
the two categories are closer to one and a half independent votes than two. The dashboard's
factor-correlation panel measures the same thing on the screener's own scores each run; a
session designing the "coherent whole" (CLAUDE.md "Build a coherent screener") should read the
two side by side before adding anything else that leans on cheapness or low asset growth.

## 4. The momentum rule's state, on the paper's kind of data

Recorded in full in `research/2026-10-09-momentum-regime.md` (addendum, same date). In short: at
the screener's own thresholds, on all US stocks from 1927 to 2013, the panic state covers 9.1% of
months and momentum averaged -1.68% a month in it against +1.52% otherwise; and the broader
market series puts April 2020 and January 2023 in the state where the S&P 500 total-return index
the screener reads does not.

## 5. Size, among large stocks (experiment 4, added the same night)

The methodology page says: "No published study establishes a size premium *within* the largest
two market-cap deciles, which is the entire S&P 500." Experiment 4 measures it on the library's
ten size groups. The ninth group's return minus the tenth's, a month:

| Period | Equal-weighted | Value-weighted |
|---|---|---|
| 1963-07 to 1990-12 (the paper's sample) | +0.08% (t = 1.07) | +0.08% (0.84) |
| 1991-01 to 2026-08 | +0.08% (1.11) | +0.07% (0.79) |
| 2000-01 to 2026-08 | +0.11% (1.36) | +0.14% (1.24) |
| 1926-07 to 2026-08 | **+0.10% (2.39)** | +0.09% (1.57) |

The eighth group minus the tenth, over the full century: +0.15% a month either way (t = 2.65
equal-weighted, 2.17 value-weighted).

**Reading.** The sign is positive in all eight cells and the size is steady - about one
percentage point a year - but it is distinguishable from zero only over a hundred years. So the
methodology's sentence stays true as written (this is a measurement on public data, not a
published study), while "there is no size effect among large stocks" would be saying more than
the data do. **The methodology's Size section now quotes this measurement beside its own
caveat** (same claim, same night). For the Size weight that argues for what the screener already does: a small
weight, stated with its weakness, and not a larger one. Before anyone moves it, read the
literature on size after controlling for quality - Asness, Frazzini, Israel, Moskowitz &
Pedersen (2018), "Size matters, if you control your junk", and Alquist, Israel & Moskowitz
(2018), "Fact, fiction, and the size effect" are the usual starting points (listed from memory;
verify on reading).

**The replication itself mostly failed, and is published that way:** 3 of 11 figures within
tolerance. Fama & French's 1992 table covers non-financial firms that also appear in the
accounting database; the library's groups hold every listed stock, and every group of smaller
companies comes out well below the paper's figure. The requirement of accounting data is a
known source of upward bias in small-firm returns of that era (Kothari, Shanken & Sloan 1995,
*Journal of Finance* 50(1) - again from memory). It is a useful thing for a student to see:
the same sort, on a slightly different set of companies, gives a size effect about a third
smaller.

## 6. Risk: no reward for volatility, rather than a penalty (experiment 5, added the same night)

The methodology calls the low-volatility anomaly "one of the most robust findings in finance".
On the library's variance-sorted fifths (value-weighted, the paper's sample of 1963-07 to
2000-12) the calmest fifth averaged 1.03% a month with volatility of 3.57%, and the most
volatile 0.81% with volatility of 7.56%. Return per unit of risk falls steadily from the
calmest group to the most volatile (0.29, 0.26, 0.24, 0.21, 0.11).

Two things for a methodology session:

1. **The strong form of the paper's result does not reproduce here.** The paper's most volatile
   fifth earned 0.09% a month and trailed the calmest by 0.97% (t = -2.86). The library's
   trails by 0.22% (t = -0.79), and since 2001 by nothing. The library sorts on 60 days where
   the paper uses one month and draws its lines so the extreme group holds larger stocks. The
   paper's penalty lives in small, extremely volatile stocks - which the screener never ranks.
2. **The weak form is what the screener can lean on.** Among the stocks it does rank, the
   defensible statement is that volatility has not been rewarded with return, so a calmer stock
   has earned about the same with less risk. That supports a Risk category. It supports "the
   calmer stock was the better trade per unit of risk"; it does not support "the calmer stock
   earned more". **The methodology's sentence was reworded to exactly that the same night**
   (generator; claim `overview.low_risk_and_size_as_measured`): "more volatile and higher-beta
   stocks have not earned reliably more than calmer ones, so the extra risk has not been paid
   for." Baker, Bradley & Wurgler (2011) and Frazzini & Pedersen (2014) are still worth reading
   for the large-cap version (from memory; verify).

**Beta, the category's second measure (experiment 7).** Fama & French's (1992) flat relation
between beta and return reproduces: the highest-beta tenth minus the lowest is -0.03% a month in
the paper's sample (the paper: +0.04%), +0.26% since 1991 (t = 0.93) and +0.13% over the whole
period (t = 0.65), while the highest-beta group has been about twice as volatile throughout.
Same reading as for volatility: market risk has not been paid for, so a lower-beta company has
carried less risk for about the same return. Since 2010 the value-weighted high-beta groups have
earned more (the ninth tenth 1.86% a month) - with volatility to match and nowhere near
significance, but worth knowing before describing low beta as free.

## 7. Valuation: the earnings-yield spread since 2007 (experiment 6, added the same night)

Valuation is the screener's joint-largest category (22%) and its metrics are yields. The nearest
sort the public data carries is earnings against price. The highest-yield tenth minus the
lowest, a month:

| Period | Equal-weighted | Value-weighted (large companies dominate) |
|---|---|---|
| 1963-07 to 1990-12 (the paper's sample) | +0.70% (t = 4.35) | +0.41% (1.75) |
| 1991-01 to 2026-08 | +0.43% (2.80) | +0.16% (0.74) |
| 2007-01 to 2026-08 | +0.13% (0.65) | **-0.23% (-0.78)** |
| 1951-07 to 2026-08 | +0.60% (6.05) | +0.36% (2.56) |

**Reading.** Over seventy-five years this is among the strongest patterns in the lab. Over the
last twenty it is not there, and in the value-weighted version - the one closest to a universe
of large companies - the sign is negative. Twenty years is long enough to take seriously and
not long enough to call (the t-statistics say so), and this period is the well-documented
drawdown of value strategies generally.

**What it is and is not a reason for.** It is a reason to (a) write the Monday note the first
section already asks for, with this table in it; (b) check whether the screener's own valuation
metrics - free-cash-flow yield and enterprise-value yields, compared within sector - behave like
earnings-to-price or differently, which the cash-flow-yield sort (plan item 6b's remainder)
starts to answer; and (c) say so on the methodology page, which limitation 11 now does for the
factors. It is **not** a reason to cut the Valuation weight tonight: one sort on one definition
over one twenty-year window is exactly the kind of evidence rule 4 exists to stop a session
acting on, and within-sector comparison is a material difference from this sort.

## Sources

* Barroso, P. & Santa-Clara, P. (2015). Momentum has its moments. *JFE* 116(1), 111-120. Figures from the April 2012 working paper, Table 3.
* Daniel, K. & Moskowitz, T. (2016). Momentum crashes. *JFE* 122(2), 221-247. Figures from the authors' draft of 2015-06-16, Tables 1, 2, 3, 5.
* Fama, E. & French, K. (2015). A five-factor asset pricing model. *JFE* 116(1), 1-22. Tables 4 and 6.
* McLean, R. D. & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance* 71(1), 5-32.
* Data: Kenneth R. French Data Library, files dated from the 2026-08 database.
