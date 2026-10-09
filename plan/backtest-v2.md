# Backtest v2 - an honest validation harness

**Priority:** 3 in `CLAUDE.md`
**Status:** **Step 1 is DONE, both halves.** Survivorship measured 2026-09-24 and
restated 2026-09-30 (**11.4% of the panel**); look-ahead measured 2026-10-01
(**>= 63.2% of the panel**). The price-source procurement question that gated
steps 2-4 is answered and decided (2026-09-30). Steps 2-5 open, **and the
sequencing below is now wrong: look-ahead is the bigger bias, so step 3 outranks
step 2** - see "Step 1 result, 2026-10-01" below.
**Deadline that matters:** 2027-02-11

---

## Step 1 result, 2026-10-01: look-ahead measured, and it is the bigger half

Full note: `research/2026-10-01-lookahead-bias-size.md`. Every number is
reproduced by
`research/measurements/2026-10-01-lookahead-price-component.py` (committed JSON
output beside it). Component shipped: `lookahead.py`, 43 tests in
`tests/test_lookahead.py`.

**63.2% of name-months are assigned to a different decile** once the
price-dependent metrics are restated at the rebalance month's own price; **30.3%**
move by two deciles or more; only **59.5%** of v1's top decile belongs there.
Against survivorship's **11.4% of the panel**, look-ahead is **5.5x larger on the
same unit** - and it is a **lower bound**, because 49.0 points of composite weight
are frozen in both arms of the experiment.

### The decomposition that makes the remaining work concrete

`lookahead.weight_buckets()` derives this from `config.yaml`, so a reweight cannot
leave it stale:

| Bucket | Weight | Status |
|---|---|---|
| Recomputed per rebalance | **16.9%** | honestly point-in-time |
| **Price-restatable, held constant** | **28.0%** | free to fix - the harness already holds the price panel |
| **Price-derived, held constant** | **6.1%** | free to fix - `jensens_alpha`, `max_drawdown_1y` |
| Needs point-in-time filings/estimates | **49.0%** | the expensive half; 9.0pp of it is analyst estimates with no free retrospective source |
| **Held constant, total** | **83.1%** | |

**34.1 of those 83.1 points need no vendor, no licence and no permission.** That
is the cheapest honest improvement available to v2 and it should come first.

### Two things this corrected about the module's own description

* **`backtest.py`'s docstring says "Only Momentum and Risk metrics are recomputed
  from trailing prices".** Its `dynamic_cols` list holds four names and those two
  categories carry six weighted metrics, so **6.1 of the 23 points** are frozen -
  both of them pure functions of a price history.
  `tests/test_lookahead.py::test_recomputed_matches_backtests_dynamic_cols` parses
  that literal and fails if it moves.
* **v1 backtests a weighting the site does not publish.** `run_screener.py` calls
  `adjust_momentum_weight()` between the category scores and the composite;
  `backtest.py` does not. On the 2026-10-01 run that step moves momentum
  **13 -> 14.95** and valuation **22 -> 20.05**. Adding the one call to the
  reconstruction closed its gap against the published ranking from a median of 8
  rank places to **2** (to **0** against the full-precision run cache rather than
  the 4dp payload), which is how the measurement's arm A was shown faithful.
  **v2 must apply it per rebalance month, from that month's own volatility
  regime** - the current call reads the current run's regime, which is a third
  look-ahead vector, living in the weights rather than the metrics.

### The fundamentals half - MEASURED 2026-10-09 (owner-run)

`research/2026-10-09-pit-fundamentals-census.md`: from SEC XBRL `companyfacts` (503 requests, 1.17M
facts, each with its filed date), the inputs for most of the 49.0pp are knowable point-in-time for
**~95-97% of name-months** (net income, assets, equity, revenue, operating cash flow), 68-85% for
operating income / D&A / debt / cash / capex, and only **39%** for a gross-profit tag (59% can derive
it from cost of revenue). Median first-filing lag: **33 days** after a quarter, **54** after a year.
Analyst estimates (9.0pp) and short interest have no free history, as expected. Facts cache:
`data/sec/pit/facts.parquet` (gitignored; rebuild with the measurement script).

### What step 1 said was unmeasured (kept for the record)

The **49.0pp fundamentals half**. SEC EDGAR's XBRL `companyconcept` endpoint is
free and carries a `filed` date per fact, so both the reporting lag and the drift
can be measured without a vendor for the Quality, Growth and Investment inputs.
Analyst estimates (`forward_eps_growth`, `fy1_revision_3m`, `analyst_surprise` -
9.0pp) have no free retrospective source and may have to be *reported as
permanently unmeasurable* rather than fixed.

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

`backtest.py` states its own limitations in its docstring. **Honestly on
survivorship, and not quite honestly on look-ahead** - measured 2026-10-01, item
2 below understates what the code does, because `simulate_monthly_scores`'s
`dynamic_cols` list recomputes four metrics and Momentum plus Risk carry six:

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
   **Corrected 2026-10-01: that last sentence is wrong.** `jensens_alpha` (25% of
   momentum) and `max_drawdown_1y` (28.57% of risk) are also held constant, and
   both are pure functions of a price history. **83.1% of composite weight is
   held constant, and 34.1 points of it depend on nothing but price.**

Together these mean: **a v1 backtest result cannot distinguish a genuinely
better methodology from one that better exploits hindsight.** Any changelog
entry claiming "validated by backtest" against v1 should be read sceptically.
**Measured 2026-10-01: at least 63.2% of name-months sit in the wrong decile**,
so this is not a caveat about precision - the sort itself is largely a different
sort.

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

   **DONE for look-ahead 2026-10-01 - it is >= 63.2% of the panel**, 5.5x
   survivorship on the same unit. See the step 1 section at the top. The method
   used was the price half of the comparison, which is exact and free; the
   fundamentals half of step 1 remains, and is the only part of step 1 open.

   **>>> THE NEXT STEP ON THIS PLAN IS STEP 3, NOT STEP 2. <<<**
   Look-ahead is the bigger bias here, so point-in-time fundamentals outrank the
   point-in-time universe. And **34.1 of the 83.1 held-constant weight points are
   free to fix** - the 28.0pp of price-restatable metrics plus the 6.1pp of
   price-derived ones - which makes that the cheapest honest improvement available
   and the thing to build first within step 3.
2. Point-in-time universe. **Not the bigger bias here** - the heading used to say
   "bigger bias, usually" and the 2026-10-01 measurement reversed it for this
   screener: 11.4% of the panel against >= 63.2%.
   **Component built 2026-09-24** (`universe_history.py`); not wired in.
   **No longer blocked on procurement as of 2026-09-30** - the price source is
   costed and chosen ($199/yr, Sharadar), and the decision is to buy it *after*
   step 3, not before. Free data covers 57% of the needed name-months, not the 40%
   previously recorded.
3. Point-in-time fundamentals with reporting lags. **Data layer BUILT 2026-10-09 (owner-run):**
   `pit_fundamentals.PointInTime` answers instant / annual / quarter "as filed by date d" from the
   census's XBRL facts cache (restatements count from their filing date; comparatives in later
   filings change nothing; tag switches keep the series). Not imported by v1 or production
   (`tests/test_pit_fundamentals.py`). Next: the 34.1 price points, then v2 recomputing each
   fundamentals metric per month from this layer - in one piece. **Now ahead of step 2.** Start
   with the 34.1pp that needs no data source; then size the 49.0pp fundamentals
   half via SEC EDGAR XBRL `companyconcept` (free, carries a `filed` date per
   fact); expect the 9.0pp of analyst-estimate metrics to be reportable only as
   permanently unmeasurable.
4. A/B regression harness with confidence intervals.
5. Wire it into the improvement engine as the validation gate.

## Interim rule

Until v2 exists, **live IC measurements from the data loop are more trustworthy
than backtest results**, because they are genuinely out-of-sample and
forward-looking. Weight methodology decisions accordingly, and say in
`METHODOLOGY_CHANGELOG.md` which evidence type was used.
