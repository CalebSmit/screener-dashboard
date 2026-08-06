# Backtest v2 - an honest validation harness

**Priority:** 2 in `CLAUDE.md`, and arguably 1 in importance
**Status:** Not started

## Why this matters more than any feature

The system is now allowed to change its own methodology. The thing that decides
whether a methodology change was *good* is the backtest. So the backtest is
load-bearing in a way it never was before: a biased backtest doesn't merely
mislead a reader, it actively steers the self-improvement loop toward whatever
the bias favours.

`backtest.py` states its own limitations honestly in its docstring:

1. **Survivorship bias** - it uses today's S&P 500 constituents across the whole
   2020-present window. Companies that were removed, acquired, or went bankrupt
   are simply absent. Every strategy tested looks better than it was, and
   strategies tilted toward "stocks that are in the index today" look best of
   all. Value and distress-adjacent tilts are the most flattered, which is
   precisely where this screener's Valuation weighting lives.

2. **Look-ahead bias** - Valuation, Quality, Growth and Revisions scores are
   held constant from a single Phase-1 snapshot and applied backwards through
   history. Those numbers were not knowable at the historical rebalance dates.
   Only Momentum and Risk are honestly recomputed from trailing prices.

Together these mean: **a v1 backtest result cannot distinguish a genuinely
better methodology from one that better exploits hindsight.** Any changelog
entry claiming "validated by backtest" against v1 should be read sceptically.

## What v2 needs

**Point-in-time universe.** Reconstruct historical index membership rather than
projecting today's. Options worth researching: a historical constituents
dataset, or reconstructing from index-change announcements. If genuinely
unavailable for free, the fallback is to *measure and report* the survivorship
premium rather than pretend it's zero - run the same test on a
delisted-inclusive proxy universe and quote the gap.

**Point-in-time fundamentals.** Scores at each rebalance must use only data
published by that date. This means respecting reporting lags (a fiscal quarter
is not knowable on the quarter-end date - typically 30-90 days later). The SEC
EDGAR XBRL route is worth investigating; the owner has a separate working
project doing exactly this kind of primary-source pull, which may be reusable.

**Honest cost and capacity modelling.** Transaction costs exist in v1; also
consider bid-ask spread by market cap, and whether the model portfolio's
position sizes are achievable.

**A regression harness, not just a report.** The self-improvement loop needs to
ask "is candidate config B better than incumbent config A?" and get a
statistically meaningful answer - with confidence intervals, not a point
estimate. Deciding on a 0.3% return difference with no error bar is how the
system talks itself into noise.

## Suggested sequencing

1. **Quantify the damage first.** Before building anything, measure how much
   survivorship and look-ahead are worth in this specific setup. If it's 0.5%
   a year, v1 is usable with a caveat. If it's 4%, every existing validation
   claim needs retracting. This is a research task and it is the right first
   step - it tells you how hard to work on the rest.
2. Point-in-time universe (bigger bias, usually).
3. Point-in-time fundamentals with reporting lags.
4. A/B regression harness with confidence intervals.
5. Wire it into the improvement engine as the validation gate.

## Interim rule

Until v2 exists, **live IC measurements from the data loop are more trustworthy
than backtest results**, because they are genuinely out-of-sample and
forward-looking. Weight methodology decisions accordingly, and say in
`METHODOLOGY_CHANGELOG.md` which evidence type was used.
