# Where twenty companies' filings went: the SEC files a combined report under one co-registrant

**Date:** 2026-11-18 (13:00 session). **Kind:** a documented failure, measured, and its fix.
**Changes no score, weight, metric or rank** - see section 5 for exactly what it does change.

This answers "Next 3" of the same morning's synthesis
(`research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md` section 5), which
found 20 of 499 scored companies with no 10-K or 10-Q in the SEC `companyfacts` cache filed in
the last 165 days, showed the gap was in the SEC's record rather than in our refresh, guessed at
"combined filings" and **did not establish why**.

## 1. What was measured

`research/measurements/2026-11-18-where-the-missing-filings-went.py` (134 requests to EDGAR
through the project's own client). For each of the twenty it reads:

- EDGAR's `submissions` record - the filings the company actually made since the newest one its
  `companyfacts` record carries;
- each such filing's header - every registrant on it, in the order filed;
- `companyfacts` for every *other* registrant on it.

Result: the twenty have made **41** periodic filings since the newest their own record holds.
All 41 are XBRL and inline XBRL. **None** has a single fact in the listed company's own record.

- **37 of the 41, for 19 companies, are in a co-registrant's record instead** - whole (283 to 557
  facts each), and in exactly one. Fourteen utilities that file with their operating
  subsidiaries and five REITs that file with their operating partnership.
- The co-registrant is not stable: Sempra's June quarter is under Southern California Gas and its
  September quarter under San Diego Gas & Electric; Exelon's under Delmarva Power, then PECO;
  Alliant's under Interstate Power, then Wisconsin Power. **I did not find the rule that picks
  it** (it is not the first registrant, which is the listed company on all 41, and not the
  highest or lowest CIK). The fix does not need it.
- Before mid-May those co-registrants' records held no periodic-filing facts at all, except
  Regency Centers LP and Mid-America Apartments LP (a few). So the earlier behaviour was "the
  whole filing under the first registrant" and the present one is "the whole filing under some
  other registrant". The newest filing in any of the nineteen's own records is dated between
  2026-04-23 and 2026-05-08; the earliest moved one among them is 2026-07-24 (United Rentals', section 3, is 2026-07-22).
- **4 of the 41 are nowhere**: Citigroup's three 10-Qs of 2026 (it files alone, so there is no
  co-registrant to look under) and FirstEnergy's June 10-Q (two registrants, in neither record).
  The frames API agrees for Citigroup: `us-gaap/Assets` at CY2025Q4 carries its 10-K's
  $2,657,202m; CY2026Q1, Q2 and Q3 carry 5,579, 5,611 and 4,856 filers and not Citigroup.
  **Why is not established.**

I found no SEC notice describing a change, and did not search for one exhaustively; this note
describes what the API returns, not what the SEC intended.

## 2. Are the moved facts the listed company's figures?

They have to be shown to be, because they sit in a record named for a subsidiary.

**The argument.** `companyfacts` carries facts reported for the filing as a whole; in a combined
report the subsidiaries' statements are tagged with a legal-entity member and the consolidated
parent's are not. That is how the same facts were the parent's when they sat in the parent's
record until May. It is a convention of the filers, stated here from the pattern in the data and
not from a rule I read.

**The measurement** (`research/measurements/2026-11-18-whose-figures-the-moved-facts-are.py`).
Every 10-Q repeats earlier figures as comparatives - the prior year-end balance sheet, the same
quarter a year earlier - and those (concept, unit, start, end) facts are already in the parent's
own record from filings the SEC did attribute to it.

| | overlapping facts | equal |
|---|---|---|
| 37 moved filings against the parent's own record | 7,326 | **7,004 (95.6%)** |
| the same 19 parents' last *un-moved* filing against their own earlier record | 2,635 | 2,511 (95.3%) |

So a moved filing agrees with the parent's history exactly as often as the parent's own filings
agree with each other. The disagreements are ordinary restatement and rounding: American Electric
Power began reporting in whole millions in 2026 (279.0 for 278.6) and accounts for 233 of the 322
in the first row and 66 of the 124 in the second; without it the rates are 98.7% and 97.6%.

**An independent check, against a different source.** With the moved facts in the cache, the
filing cross-check compares 19 more companies' total assets with the data source's:

| field | before | after |
|---|---|---|
| total assets within 0.5% | 440 of 440 | **459 of 459** |
| equity within 0.5% | 432 of 441 | 450 of 460 |
| operating cash flow within 0.5% | 430 of 431 | 444 of 445 |
| net income within 0.5% | 420 of 443 | 433 of 457 |
| revenue within 0.5% | 399 of 436 | 410 of 450 |
| over 10% apart, any of those five | 15 | 15 |

Nineteen balance sheets, nineteen agreements; no new disagreement over 10% in any field but
capex (45 to 46 of 371, a field already known to differ by definition - `DECISIONS.md` 0.14).

## 3. The 165-day bar hid four more

An age bar finds a company whose record *stopped*. Rebuilding the cache with the fix found the
same thing in four companies whose record had not: **Dow, Consolidated Edison, Edison
International and United Rentals** each had their June 10-Q under a co-registrant and their
September one under themselves. So the first refresh recovered **41 filings for 23 companies
(3,248 cached facts)** - not only utilities and REITs, and not every quarter.

That is why the fix looks for a *gap between consecutive filings*, not for age.

## 4. The fix

`sec_fundamentals.refresh_companyfacts`, weekly as before. After reading a company's own record:

1. `filing_gap`: is there a stretch of more than **150 days** with no periodic filing, between
   two of its filings or since the newest? (Two consecutive periodic filings of a large
   accelerated filer are at most about 142 days apart: a third-quarter 10-Q in mid-October, the
   10-K sixty days after the year end.) No gap, no extra request - all but 27 companies cost what
   they did before.
2. Where there is one: the company's `submissions` record names the periodic filings made after
   the gap opens that its record lacks; each one's header names the registrants.
3. **Only where the listed company is the first registrant**, the other registrants' records are
   read (one request each per refresh, shared) and the facts carrying *that accession number* are
   kept under the listed company's ticker. Nothing else from a co-registrant's record is read -
   a subsidiary's own separate filings stay out.
4. The refresh's log line says how many companies and filings were recovered, and names every
   company that **still** has a gap. A failure anywhere in the follow-up leaves the company's
   own facts as they were.

Measured cost: the refresh took 362 s for 503 companies (the 02:00 run's `fetch.sec_roe_history`
was 262 s that morning, on a day it was the only refresh). No row of the old cache is absent
from the new one (0 of 2,446,742).

**What still has a gap after it** - named in the log every week, none explained:

| | missing | what is known |
|---|---|---|
| Citigroup | three 10-Qs of 2026 | files alone; in no record and no frame |
| FirstEnergy | June 10-Q | two registrants, in neither record |
| Cardinal Health | by its calendar, the 10-Q due about May (nothing between 2026-02-05 and 2026-08-11) | not followed further |
| Clorox | by its calendar, the 10-K due about August (nothing between 2026-04-30 and 2026-11-02) | not followed further |

Only Citigroup is old enough to be in `sec_crosscheck.no_recent_filing`'s count, which read 20
this morning and reads **1** on the rebuilt cache.

## 5. What this changes, and what it does not

- **No score.** The only scored-engine reader of this cache is `earnings_variability`, a
  weight-0 candidate built from 10-K facts; every recovered fact is from a 10-Q. The composite
  and every rank are untouched.
- **The filing check** (`xchk`, the health note, the brief) compares up to 19 more companies per
  field (total assets and equity 19, revenue, net income and operating cash flow 14, capex 9). The pre-registered 2026-12-15 promotion rule
  (`research/2026-10-14-yahoo-vs-sec.md` sections 6-7) is **not changed**; its section 7 said the
  check could not see twenty companies, and now it cannot see one. Whoever applies the rule
  should know the runs before 2026-11-19 compared fewer companies than the runs after.
- **"A period behind the filings"** (`sec_crosscheck.newer_filings`) can now be true of these
  companies, where before the cache knew of no newer filing to be behind.
- **"Against its own five years"** (context, never scored) reads these companies' two newest
  quarters instead of stopping at March.
- **Backtest v2's point-in-time records** are built from this cache. They are rebuilt only on an
  explicit build, the ledger's verdict does not read this source (`cache->red`, 2026-11-10), and
  nothing they produce decides anything (rule 5). The next rebuild will include the recovered
  quarters for 23 companies; before this it would have carried their fundamentals forward from
  March.
- **The measurement in this morning's synthesis section 4** ("15 of 18 comparable") was made on
  the old cache and could not compare the companies this recovers. It is not re-run here; the
  Monday queued for 2026-12-07 should re-run it first.

## 6. What I did not do

- Explain the four filings that are nowhere. The count on the health line and the names in the
  refresh log are how a later session will know if they appear.
- Find the SEC's attribution rule, or whether it is intended.
- Check the predecessor cache (`facts_predecessor.parquet`, backtest v2 only) for the same thing.
- Check 10-Ks. No combined 10-K has been filed since the change; **the first will be in February
  2027**, and `earnings_variability` (weight 0) reads 10-Ks. The same follow-up handles any
  periodic form, and the "still have a gap" line is what to read that month.
  **Dated 2026-11-19: read it on 2027-03-02** - the day after a calendar-year large accelerated
  filer's 10-K is due (60 days after the year end), so every combined 10-K is in by then. The
  morning brief's "Coming up" carries the date (`scripts/write_brief.py`, `COMING_UP`).
