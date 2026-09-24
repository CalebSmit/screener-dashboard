# Backtest v2 - an honest validation harness

**Priority:** 3 in `CLAUDE.md`
**Status:** Step 1 done for survivorship (2026-09-24). Steps 2-5 open.
**Deadline that matters:** 2027-02-11

---

## Step 1 result, 2026-09-24: survivorship measured

The sequencing below says "quantify the damage first", and sets the decision
rule: *"If it's 0.5% a year, v1 is usable with a caveat. If it's 4%, every
existing validation claim needs retracting."*

**It is 4.3% a year. v1 is at the retract end of that rule.**

Shipped: `universe_history.py` (point-in-time S&P 500 membership, 59 tests in
`tests/test_universe_history.py`), a committed 80-month cache at
`data/universe_history/sp500_membership.json`, and
`research/measurements/2026-09-24-survivorship-gap.py`, which reproduces every
number here.

| Measurement | Value |
|---|---|
| Constituents of 2020-01-31 absent from today's list | **116 of 505 (23.0%)** |
| ...of which are ticker renames, not exits (by SEC CIK) | 18 |
| **Survivorship bias, net of renames** | **98 of 505 (19.4%)** |
| Index removals over the window | 141 in 79 months = **21.4/yr ≈ 4.3% of the universe/yr** |
| Distinct names ever in the index, 2020-01..2026-08 | **643** |
| Names in any single v1 run | 503 — so v1 tests **78%** of the true universe |
| Gap by month | decays monotonically 23.0% (2020-01) → 0.6% (2026-08) |

That monotone decay *is* the survivorship signature: the further back the
test reaches, the more of the real universe is missing, so v1's early years
are its most biased and its recent years nearly clean.

**Renames had to be separated, and only CIK can do it.** A ticker missing from
today's list has not necessarily left the index — ANTM became ELV, FB became
META, BK became BNY. Corporate renames change the symbol *and* the company
name together, so neither symbol nor name matching works; the SEC registrant
id survives both. Reporting the gross 23.0% as survivorship would have
overstated it by 18 names.

### The feasibility finding that shapes step 2

Of the 98 genuine exits, a 30-name sample found **12 (40%) still have
downloadable price history** and 18 do not. Verified as genuine absence, not
throttling: yfinance returns `possibly delisted; no timezone found` and zero
rows for them while contemporaneous controls return full series.

**The split is not random, and it is the bad half.** Companies dropped from
the index but still publicly traded (AAL, BWA) keep a complete series;
companies acquired, taken private or wound up (ATVI, CERN, ABMD, AGN, DISH)
return nothing. So the names free data cannot restore are exactly the terminal
outcomes — precisely the returns survivorship bias is made of.

**Consequence for the plan:** reconstructing the universe from free sources
gets roughly 40% of the way. The fallback already written into "What v2 needs"
below — *measure and report the survivorship premium rather than pretend it is
zero* — is therefore not a fallback but the likely destination, unless a paid
or archival price source for delisted tickers is found. Costing that source is
the next decision, and it should be made before building steps 2-4, because it
changes what they can honestly claim.

### What was deliberately not done

`backtest.py` is **not** wired to `universe_history.py`. A point-in-time
universe without point-in-time fundamentals — and without prices for 60% of
the restored names — is differently wrong, not fixed, and this document is
explicit that a half-fixed backtest invites the false confidence the bench
period exists to prevent. A pointer was added to `backtest.py`'s docstring so
the component is findable.

**Nothing needs retracting in `METHODOLOGY_CHANGELOG.md`** — checked, not
assumed. No entry cites a backtest figure under **Evidence**; the 2026-08-11
bench rule landed before any entry could. The rule worked.

### Limitations of the reconstruction itself

* Membership is Wikipedia's contemporaneous record, not the index. The
  revision used was a median of 3 days behind its month-end (max 27, 6 of 80
  months over 14 days). At ~1.8 removals a month a 27-day lag can miss one or
  two changes — immaterial against a 19.4% figure, but it means these
  snapshots are not exact index membership and should not be described as such.
* The rename decomposition is exact only for the oldest month, where every
  name appears in that month's revision and all 116 CIKs resolve. Extending it
  across the window needs CIKs from all 80 revisions.

---

**Owner direction 2026-08-11: the current backtest decides nothing until
2027-02-11.** Until then its output is supporting colour only - see `CLAUDE.md`
rule 5. That changes what this project is *for*. It is no longer an urgent
unblocker; it is the thing that must exist and be trustworthy by the time the
bench period ends, so that six months of research-justified methodology changes
can finally be checked against something honest.

Two consequences for how to approach it:

- **There is no rush to ship a half-fixed backtest.** A v2 that fixes
  survivorship but not look-ahead is still not decision-grade, and shipping it
  early invites exactly the false confidence the bench period exists to
  prevent. Take the time and do both.
- **Build the list of things to re-check.** Every `METHODOLOGY_CHANGELOG.md`
  entry written between 2026-08-11 and 2027-02-11 is justified by research
  alone. When v2 lands, those are the first things to test. Keeping that list
  current is part of this project.

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
   **DONE for survivorship 2026-09-24 - it is 4.3%/yr, the retract end.** See
   the step 1 section at the top. **Still open: the look-ahead half**, which
   is the other of the two biases and has not been sized at all.
2. Point-in-time universe (bigger bias, usually).
   **Component built 2026-09-24** (`universe_history.py`); not wired in, and
   blocked on a price source for delisted names - only 40% of exits have one.
3. Point-in-time fundamentals with reporting lags.
4. A/B regression harness with confidence intervals.
5. Wire it into the improvement engine as the validation gate.

## Interim rule

Until v2 exists, **live IC measurements from the data loop are more trustworthy
than backtest results**, because they are genuinely out-of-sample and
forward-looking. Weight methodology decisions accordingly, and say in
`METHODOLOGY_CHANGELOG.md` which evidence type was used.
