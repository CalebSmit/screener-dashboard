# Backtest v2 - an honest validation harness

**Priority:** 3 in `CLAUDE.md`
**Status:** Step 1 done for survivorship (2026-09-24); the price-source
procurement question that gated steps 2-4 is **answered and decided**
(2026-09-30). Steps 2-5 open, and the next step is now a **free measurement**,
not a purchase - see "The procurement decision" below.
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

**Superseded 2026-09-30 by a census of all 123 exits. Two of the numbers below
were wrong; read "The procurement decision" further down instead.** Kept because
the *qualitative* claim - the split is not random and it is the bad half - was
right, and the census sharpened it from an impression into 100% vs 10%.

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

---

## The procurement decision, 2026-09-30: answered, and deliberately deferred

Full reasoning, citations and licence quotes:
`research/2026-09-30-delisted-price-source-cost.md`. Every number below is
reproduced by `research/measurements/2026-09-30-delisted-price-requirement.py`
and `research/measurements/2026-09-30-exit-reasons.py`.

**The answer: Sharadar Prices (10-year history), $19 to download and $199/yr to
keep.** Cost is not the blocker and must not be cited as one again. Norgate
Platinum ($630/yr) also bundles historical index constituents, which
`universe_history.py` already provides for free. CRSP is the only surveyed source
carrying a true delisting return (`dlret`/`dlstcd`) and has no list price at all
— institutional annual contracts only.

**The decision: do not buy yet. Size the look-ahead half first, because it is
free and it is the missing half of step 1.** This document's own rule is that a
v2 fixing survivorship but not look-ahead is not decision-grade; survivorship is
now measured twice over and look-ahead has never been measured at all; and
Sharadar's licence requires deleting the data within 30 days of the subscription
ending, so subscribing before a consumer exists pays for a panel nothing reads.

### The census that replaces the 30-name sample

All 123 genuine exits, coverage scored **month by month against each name's own
membership months** rather than by row count:

| Measurement | 2026-09-24 sample | **2026-09-30 census** |
|---|---|---|
| Rename/exit decomposition | oldest month only | **all 80 months by CIK: 142 absent = 19 renames + 123 exits + 0 unresolved** |
| Exited names with usable history | 40% (30-name sample) | **54% of names, 57.2% of name-months** |
| Survivorship as a share of the panel | not computed | **11.4%** — 4,526 exited of 39,603 name-months |
| Residual after a free-data v2 | not computed | **4.89% of the panel** — 1,938 name-months |

**So "roughly 40% of the way" was pessimistic.** Free data gets **57%** of the
way by name-months, and a paid source closes essentially all of the rest.
**11.4% is also the cleanest single statement of this backtest's survivorship
bias** — the 4.3%/yr above is a turnover rate; this is the share of the panel
that is simply absent.

### Where the free gap sits, and why a cheap feed is therefore sufficient

Exits classified by S&P DJI removal reason (116 of 123 matched):

| Removal reason | Names | Name-months needed | Free source misses |
|---|---|---|---|
| Market-cap / representation | 72 (62%) | 2,755 | 276 (**10%**) |
| Acquired / merged / taken private | 35 (30%) | 1,339 | 1,339 (**100%**) |
| Other / unparsed | 4 | 136 | 93 (68%) |
| **Bankruptcy / receivership** | **3** | 93 | 93 (100%) |
| Spin-off / restructuring | 2 | 24 | 0 |
| (unmatched) | 7 | 179 | 137 (77%) |

**The free gap is the acquisitions** — 69% of all missing name-months, none of
them supplied — while demotions, which keep trading, are 90% covered.

**And acquisitions are the easy case, per the index's own methodology.** A
company delisted by merger or acquisition is *"removed at a time announced by
S&P Dow Jones Indices, normally at the close of the last day of trading or
expiration of a tender offer"*, and where there is *"no achievable market
price"* it is removed *"at a zero or minimal price"*. So for an acquisition the
last traded close is not an estimate of the exit value — **it is the exit value
the index used**, and that is precisely what a $199/yr feed sells. Performance
delistings, the only case Shumway's missing-delisting-return corrections apply to
(−30% NYSE/AMEX, 1997; −55% Nasdaq, Shumway & Warther 1999), are **3 of 116** and
all three are the 2023 FDIC receiverships.

**Two conventions to adopt when wiring this, citing S&P DJI rather than CRSP:**
exit an acquisition at the last traded close, and a no-achievable-price deletion
at zero.

### Do not test price availability by row count

Three exits — **INFO, LB and SBNY** — return more than 200 rows and cover
**none** of their membership months. `INFO` (IHS Markit, absorbed by S&P Global
in March 2022) now returns a series beginning October 2024. Joining that to a
2020-2022 backtest would not leave a hole; it would insert **a different
company's prices under a former constituent's symbol**. Three more (AVB, EA,
LEG) return one to four days in August 2026.

**An availability check must assert coverage of the span the caller will read,
not the volume of what came back.**

### The licence, which is the real constraint

Sharadar's personal-use licence expressly permits publishing *"research outputs,
backtest results, models, summary statistics"* derived from the data, and
expressly forbids making the data itself available to others. So **v2 could run
and publish its results; the price cache could not be committed.** That is the
first input this project would be contractually unable to publish, in a
repository whose standard is that its numbers are checkable. Two questions for
the owner before any purchase: whether a personal-use grant covers a site aimed
at investment clubs, and whether a derived return series may be committed.

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
**Membership itself is DONE and free** (`universe_history.py`, 2026-09-24).
**The price half is costed** (2026-09-30): $199/yr closes it, and the fallback
above is better-founded than this paragraph assumed - a point-in-time S&P 500
panel that exits acquisitions at the last traded close and no-price deletions at
zero is close to correct rather than a consolation prize, because those are the
index's own conventions.

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
   **DONE for survivorship 2026-09-24 - it is 4.3%/yr, the retract end**, and
   restated 2026-09-30 as **11.4% of the name-month panel**, which is the figure
   a look-ahead measurement should be made comparable to. See the step 1 section
   at the top.

   **>>> THE NEXT STEP ON THIS PLAN IS THE LOOK-AHEAD HALF, AND IT IS FREE. <<<**
   It has not been sized at all. Nothing else here should be built first, and as
   of 2026-09-30 nothing else is blocked on anything but this. Method sketch: for
   each rebalance month, compare a score built only from data published by that
   date against the score `backtest.py` actually uses (one Phase-1 snapshot held
   constant across the whole window), and express the gap as a share of the panel
   so the two biases are directly comparable. Same shape as the 2026-09-24
   survivorship measurement; no vendor, no purchase, no permission needed.
2. Point-in-time universe (bigger bias, usually).
   **Component built 2026-09-24** (`universe_history.py`); not wired in.
   **No longer blocked on procurement as of 2026-09-30** - the price source is
   costed and chosen ($199/yr, Sharadar), and the decision is to buy it *after*
   step 1's look-ahead half and step 3, not before. Free data covers 57% of the
   needed name-months, not the 40% previously recorded.
3. Point-in-time fundamentals with reporting lags.
4. A/B regression harness with confidence intervals.
5. Wire it into the improvement engine as the validation gate.

## Interim rule

Until v2 exists, **live IC measurements from the data loop are more trustworthy
than backtest results**, because they are genuinely out-of-sample and
forward-looking. Weight methodology decisions accordingly, and say in
`METHODOLOGY_CHANGELOG.md` which evidence type was used.
