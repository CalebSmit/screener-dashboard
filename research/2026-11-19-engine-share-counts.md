# The research engine's share counts: four causes, three candidate rules, none accepted yet

**Date:** 2026-11-19 (13:00 session). **Scope:** `pit_record.market_fields`, which only backtest v2
and the research engine read. Nothing here touches the live screener, whose market value comes
from the data source and is cross-checked against filings every morning (`sec_crosscheck`).
**No code changed.** Nothing below is evidence about returns (CLAUDE.md rule 5); it is a
measurement of a data defect.

**Status: DESIGNED, NOT BUILT.** A build needs the 176 monthly records rebuilt (about 95 minutes
at the measured 81 months in 44) and all 32 ledger rows rerun, which did not fit after the
diagnosis. `plan/backtest-v2.md` carries the build order.

Scripts: `research/measurements/2026-11-19-engine-share-counts.py` (`--trace`, `--sweep`,
`--evaluate --version 1|2|3`) and `...-readout.py`. The per-name-month tables are written to the
gitignored `logs/`; rerun to regenerate.

## 1. What the morning found, and what it left

The 06:00 session found 16 tickers with a point-in-time market value under $100M somewhere in
the engine's history and Berkshire Hathaway under $2B in all 176 months, and wrote: "follow each
ticker's count to its filing fact before choosing a rule". This note does that.

The engine builds market value as price x a share count, the count being **the more recently
dated of the latest balance-sheet count (`CommonStockSharesOutstanding`) and the latest
cover-page count (`EntityCommonStockSharesOutstanding`)**, multiplied by the split ratios dated
after the month-end.

## 2. Four causes, each traced to the filed fact

| Cause | Example, as filed | Why the rule picks it |
|---|---|---|
| **A stale tag.** The filer stopped using the concept and the last value is still "the latest". | Simon Property: balance-sheet count **8,000**, dated 2012-12-31, picked in every month to August 2026 (its weighted average is ~303-326 million). Fox: cover count **1**, dated 2019-03-18 (the pre-spin shell), picked for 90 months. Baker Hughes: **100**, dated 2017-06-30. Berkshire: **941,481** (Class A), dated 2011-04-29. | No age limit on "latest". |
| **A units slip in the count.** The figure is filed in thousands or millions without the scaling. | Stryker 2011-2015: balance sheet **383**, cover page 382,672,989, same date. Public Storage 2016: cover **173**, balance sheet 173,078,782. Roper 2014-15: balance sheet **100,126**, cover 100,356,523. LabCorp 2012: cover **97**. | A tie on date goes to the balance sheet; otherwise the later date wins, whichever is wrong. |
| **The same slip, upward.** | Garmin (16 months over $4 trillion; $9.6 **quadrillion** in December 2016 against about $9bn), Packaging Corp ($16.4 quadrillion in June 2024), CenterPoint, Seagate, Yum, AMD, Oracle, Clorox, Carnival, Edison, Qualcomm. | The morning's check looked only for values that were too small. |
| **The split basis is taken from the month-end, not from the filing.** A count filed before a split, read after it, is multiplied by nothing, while the price is already on the new basis. | NVIDIA: **$2,697bn** in May 2024, **$304bn** in June, **$288bn** in July, **$2,928bn** in August - its 10-for-1 split was 7 June and the next 10-Q arrived in late August. | `split_after(splits, ticker, as_of)`: only splits after the month-end are applied. |

The last one is the only cause that is a logic error rather than a property of the filings, and
its signature is exact: where the candidate rule below uses the *same* source as the shipped
rule, the two counts differ by more than 1.4x in **194 name-months across 77 tickers**, and the
ratios are split ratios - 2.0 (85 name-months), 1.5 (23), 4.0 (20), 3.0 (14), 10.0 (12), 5.0 (10).
Brown-Forman 20 months, Nike 11, Monster 7, Apple 3, NVIDIA 3.

**The rule for which basis a filing is on is the SEC's, not ours.** Staff Accounting Bulletin
Topic 4.C, "Change in Capital Structure": for a stock split occurring after the latest balance
sheet but before the financial statements are released, "Such changes in the capital structure
must be given retroactive effect in the balance sheet."
(https://www.sec.gov/interps/account/sabcodet4.htm - fetched 2026-11-19; that sentence read
verbatim, the surrounding Facts paragraph read in fragments only.) So a filing is on the share
basis of the day it was **filed**: it reflects every split up to then and none after.

## 3. A reference that turned out not to be one

The obvious guard is to compare the count with the company's own weighted-average share count
from the same filings. Measured over 76,618 ticker-months (`--sweep`): the picked count is under
half the weighted average in 941 ticker-months (22 tickers) - and **over twice it in 877 (70
tickers), almost all of them at ratios of 1,000 or 1,000,000.** In those the *weighted average*
is the figure filed in thousands: ConocoPhillips for 97 months, Garmin 70, McDonald's 31 (in
millions), Coca-Cola 12, Host Hotels 12. A guard that trusted the weighted average would have
replaced ConocoPhillips' correct count with one a thousand times too small for a decade.

No single share-count concept is reliable. That is the finding that shaped everything below.

## 4. Three candidate rules, in the order they were written

Version 1 was written and committed before any evaluation was run (commit `280b983`). **Versions 2 and 3 were each written after reading the result before them, and
are labelled so here.** Version 3's four acceptance criteria were committed before it was run
(commit `8adeb3f`).

Common to all three:

1. Three sources, each the latest visible on the day: balance-sheet count, cover-page count,
   weighted-average basic count of the latest period.
2. A source dated more than **460 days** before the month-end is not used.
3. Each count goes onto today's share basis by the splits dated after its filing was **filed**
   (section 2).
4. A market value under **$50M** is unknown, never a small company.

They differ in how a disagreement is settled:

- **Version 1 - a vote.** Preference order: the later-dated instant, the other instant, the
  weighted average. Take the first that agrees within 1.5x with another source. One source alone
  stands. Several and no two agreeing: unknown.
- **Version 2 - a referee** (after reading 1). R = net income / diluted EPS for the latest
  period reporting both (|EPS| >= 0.10), which carries no share-unit scaling. Take the first
  source within 3x of R. Plus one declared exception: Berkshire reports its count in Class A
  equivalents and the line scored is Class B, 1,500 to 1.
- **Version 3 - vote, then referee** (after reading 2). Version 1's vote; the referee decides
  only where the vote gives nothing or gives a value under the backstop.

### Results (65,458 name-months with a market value in the engine's cache)

The emulation of the shipped rule is exact: the price implied by cached market value / emulated
count is within 0.1% of the engine's own close in **65,381 of 65,458 (99.88%)**.

| | Shipped | v1 | v2 | v3 |
|---|---|---|---|---|
| Name-months under $100M | 494 (16 tickers) | 0 | 0 | **0** |
| Name-months over $4T | 84 (15 tickers) | 32 (4) | 32 (4) | **32 (4)** |
| Count jumps over 5x, month to month | 97 | 3 | 2 | **3** |
| Count jumps over 1.5x | 292 | 45 | 51 | 52 |
| Newest month within 10% of the live fetch's count | 454 of 481 | 474 of 489 | 476 of 491 | **476 of 491** |
| Newest month off by more than 2x | 14 | 5 | 5 | **5** |
| Had a value, now unknown | - | 766 | 559 | **436** |
| Put under the $50M backstop | - | 66 | 152 | **21** |

The 32 name-months over $4T that remain are real: Apple from November 2025, Alphabet's two
lines from January 2026, NVIDIA from July 2025 (to $5.32T).

Why each version was superseded: version 1 turned ConocoPhillips' 97 correct months into
"unknown" (two sources, one slipped, a vote of two cannot say which) and kept Garmin's two
instants that agree with each other and are both in thousands. Version 2 fixed both and broke
Halliburton for 43 months, where the referee is the figure that is wrong. Each mechanism fails
where the other holds, hence version 3.

### Version 3 against its criteria - **NOT ACCEPTED as written: three of four**

| | Criterion (committed before the run) | Result |
|---|---|---|
| a | None under $100M; none over $4T but Apple, Alphabet, NVIDIA | **met** - 0, and exactly those four lines |
| b | "Had a value, now unknown" at most 150 name-months | **NOT met - 436** |
| c | At most 3 jumps over 5x | **met - 3** (Keurig Dr Pepper June 2022, Lennar April 2012, Simon February 2012) |
| d | Newest month: at least 474 within 10%, at most 5 off by 2x | **met - 476 and 5** |

Criterion b fails and is not being rewritten. What it counted, though, matters for whoever
builds this: **368 of the 436 are three tickers whose shipped value was wrong** - Visa (176
months; shipped count 0.25x the live count), Berkshire (123; 0.0004x) and Simon Property (69;
0x). For those, "unknown" replaces a wrong number, which the criterion did not distinguish from
losing a right one. The other 68: Fox's two lines 30, Everest Group 11, Viatris 6, Ares 5,
Smurfit Westrock 4, and 3 each for Baker Hughes, Amcor, ICE and Eaton - mostly the months around
a merger or redomicile when the only filed count belonged to a shell.

**What no rule over this cache can fix:** Visa and Berkshire (after 2013) have **no usable count
in the SEC's `companyfacts` at all** - no fresh instant, no weighted average, no EPS - because
they tag them per share class, and the facts cache holds only untagged-by-class facts. The live
fetch's five remaining 2x disagreements in the newest month are the standing structures the
2026-11-09 review already named (Carvana, Interactive Brokers, TKO: partnership units) plus
Blackstone and Waters. Getting Visa and Berkshire right needs the class-level facts from the
filings themselves, which is a fetch change, not a rule.

## 5. How large this is

The morning sized it at 494 name-months (0.75%), 655 counting Berkshire. Measured against
version 3: **1,992 name-months (3.06%) across 135 tickers have a market value that moves by more
than 10%**, 1,073 by more than 2x, 473 by more than 10x; **1,635 gain a market value they did
not have**; 436 lose one. It is spread evenly - between 2.6% and 3.5% of every year from 2012 to
2026 - so it is not a presample artefact.

What it reaches in the engine: `size_log_mcap` (the Size score) and every ratio with market
value in it - earnings yield, FCF yield, price-to-book, the enterprise-value multiples.
**Not sized:** how far any ledger statistic moves. That needs the rebuild.

## 6. What I did not do, and what is uncertain

- **No build.** See Status.
- The rule was iterated three times against the same data. It is a cleaning rule judged on
  plausibility, continuity and agreement with an outside count, not on returns, but the
  thresholds (460 days, 1.5x, 3x, $50M) were each chosen once and not swept, and three passes is
  still three passes: a build session should run the committed criteria on the rebuilt months
  and not tune further to pass b.
- The live-fetch comparison is one cross-section, with the engine's newest month (August) set
  against a November count. Charter, Kimberly-Clark and Fox's differences are the acquisitions
  the 2026-11-09 note documents, not rule errors.
- The 460-day staleness bound was reasoned (a 10-Q arrives every quarter; an annual-only tag is
  at most ~15 months old), not measured from the distribution of count ages.
- EPS has the same split-basis error as the count (`trailingEps = eps / split`, same `split`).
  Not measured here; the build should move both together.
- `sharesBS` / `sharesBS_prior` (the statement fields behind dilution metrics) read the same
  balance-sheet concept and were not examined.
- The referee uses `NetIncomeLoss` over diluted EPS. Where a company has material preferred
  dividends R overstates the count slightly; at a 3x tolerance that does not matter, and it is
  why R is a referee and never a count except as a last resort (73 name-months in version 3).

## 7. For the build session

1. Decide criterion b first, in writing, before touching code: either build version 3 and record
   that it fails b for the reason in section 4, or add the class-level count for Visa and
   Berkshire so that it passes. Do not lower the bar to 436.
2. Implement in `pit_record.market_fields` (and the EPS basis), with a unit test per cause using
   the filed values in section 2's table as fixtures.
3. Rebuild the months (`python backtest_v2.py --shard k/4`, both eras), rerun all 32 rows
   (`python scripts/rerun_experiments.py --jobs 4`), `python v2_experiments.py report`. Budget
   about two and a half hours end to end.
4. Re-run `--evaluate --version 3` logic against the rebuilt cache: the shipped column should now
   equal the candidate column.
5. Report how far the ledger moved as a *Backtest observation*, never as evidence.
