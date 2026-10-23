# Which free data would add evidence, rather than more tests on the same history?

**Date:** 2026-10-22 (delegated research session). **Kind:** a data-source evaluation.
**Not a methodology proposal.** Nothing here changes a weight, a metric or the research registry;
under CLAUDE.md rules 4 and 5 nothing a backtest says decides anything before 2027-02-11. The three
pre-registerable tests in section 4 are written in the registry's style and **deliberately not added
to `research/experiments/registry.yaml`** - a session that adopts one commits it there, before any
month of it is built.

**Proposal status: DECIDED 2026-10-23 (06:00 session, not the note's author)** - the instruction
"register tests A, B and C in `research/experiments/registry.yaml`, each in its own commit before
any of its data is built" is **REFUSED for all three, as written**, because the registry cannot hold
any of them and for B registering first would manufacture a false null. The **research stands**:
test B's data is real and independently re-verified, and B is **accepted in substance** with a
corrected pre-registration and a stated build order. Test A is refused on its own void condition
and on sequencing; test C is refused on the trade between what it costs and the 9.8% of precision
it buys. Reasoning and measurements: **section 7**.

## 1. The question, and the answer in one paragraph

The owner, 2026-10-22: *"there's so much data out there ... we have all of the data we could possibly
need to make this tool amazing."* There is a great deal of free data. Very little of it is
*evidence* for this screener, because the binding constraint is not data but **months**: a monthly
large-company screen gets about one independent observation a month, and
`research/2026-10-20-backtest-overfitting.md` showed that more tests on the same 2012-2026 months
mostly give luck more candidates to choose from. So the useful question is narrower: which free,
properly licensed sources **fix a documented weakness** or **add months nobody has looked at**?
Three do, concretely: **SEC Form 13F holdings** price the acquired members that every free price
source misses (quarterly, public-domain, so a survivorship test that can confirm as well as
falsify); **FINRA's short-interest files** carry exchange-listed days-to-cover back to at least
**2017-12-29** - not "no free history" as `research/2026-10-09-pit-fundamentals-census.md` has it, and
not "June 2021" as FINRA's own page says - which un-benches the one Revisions metric that is not an
analyst estimate; and **SEC XBRL plus Wikipedia's revision history** reach back past backtest v2's
2012 start into the XBRL phase-in, about **22 months that no registered experiment has seen**. The
weakness no free source fixes is point-in-time analyst estimates: the only archive of those is the
one this project's own data loop has been writing since February.

## 2. Weaknesses on record, and what would count as fixing one

| Weakness | Where it is measured | What a fix needs |
|---|---|---|
| Survivorship, 2012-2019 | 15,590 departed member-months; free routes close 37%, **acquisitions 51.5% of the gap and 1.2% covered** (`research/2026-10-20-free-survivorship-sources.md`) | a price for acquired members while they were members |
| No point-in-time analyst estimates | Revisions (10 points) and `forward_eps_growth` sit at weight 0 in v2 (`backtest_v2.UNMEASURABLE_METRICS`, 11 metrics) | consensus EPS and targets as they stood each month |
| Fundamentals before 2009 | XBRL starts with periods after 2009-06-15 (SEC Release 33-9002, 2009) | structured, dated financial statements before then |
| Prices from Yahoo | personal-use terms; `research/2026-10-14-yahoo-vs-sec.md` checked the *fundamentals*, not the licence | a price source whose terms allow the use |
| Too few independent months | detectable IC difference **0.040** over 2012-2019's 96 months, **0.070** over the 38 held-out months (`research/2026-10-15-research-engine-first-batch.md` section 5) | months never used, or the forward record |

## 3. The ranked table

Licence verdicts are readings of each provider's own terms page on 2026-10-22 (quoted in section 5),
not legal advice; `screener-bot/ROADMAP.md` already carries the questions for a lawyer. Columns:
**(a)** private research, **(b)** aggregate results on a public non-commercial page, **(c)**
commercial use. "Effort": S = an evening, M = a few sessions, L = a plan of its own.

| # | Source | What it adds | Weakness fixed / hypothesis made testable | History (verified how) | (a) | (b) | (c) | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **SEC Form 13F data sets** (structured institutional holdings) + CUSIP->CIK from SC 13G cover pages or OpenFIGI | As-traded **quarter-end price for nearly every member, acquired ones included**: value / shares, median over filers | **Survivorship, including the acquisition hole** - at quarterly, not monthly, frequency. Makes the Size question two-sided (section 4, test A) | Positions 2013-06-30 onward, quarterly (SEC page: "July 2013 - August 2026"); 29 of 39 within 1% of Yahoo, 60 of 86 acquired members name-matched in the 2026-10-20 test | yes | yes | yes | L |
| 2 | **FINRA equity short-interest files** (biweekly; `daysToCoverQuantity` in the file) | Days to cover, as reported, for every exchange-listed member | The one weighted Revisions metric that is not an analyst estimate, `short_interest_ratio` (13.5 of Revisions' 100, **1.35 composite points**), becomes backtestable (test B) | **Probed today:** exchange-listed rows (AAPL, JPM) present on 2017-12-29, 2018-06-29, 2018-12-31, 2019-01-15, 2019-12-31, 2021-05-28; earlier dates tried returned 403. FINRA's page says exchange-listed data starts June 2021 - **the files contradict it** | yes | yes, with FINRA named as source - via the Query API's terms, not the website's | no | S-M |
| 3 | **SEC XBRL (`companyfacts`, `frames`, Financial Statement Data Sets) 2009-2011 + Wikipedia revisions before 2012** | Backtest v2's own record-builder can reach back before 2012-01 | **New months**: about 22 (2010-03..2011-12) that no registered experiment has used (test C) | **Measured today** (`research/measurements/2026-10-22-xbrl-phase-in-coverage.py`): of January 2012's 457 CIK-mapped members, **316** filed an XBRL balance sheet for 2009-06-30 within a year, **401** for 2010-06-30, **412** for 2011-06-30. `universe_history.fetch_snapshot` parses and validates the list at 2009-06-30, 2010-03-31, 2010-06-30, 2010-12-31, 2011-06-30 and 2011-12-31 (499-500 members each) | yes | yes (Wikipedia: CC BY-SA, attribution; membership is fact) | yes | M |
| 4 | **Tiingo free plan** (EOD prices, delisted symbols listed) | Monthly prices for ~84 of 152 acquired names *if* the listed histories are real | Survivorship at **monthly** frequency, ~71% of the gap with Yahoo (list-based estimate) | Public ticker list read 2026-10-20; series need a free key | yes (personal) | **no, until Tiingo confirms in writing** | no | S once a key exists |
| 5 | **SEC Insider Transactions Data Sets** (Forms 3/4/5) | Every reported insider trade as structured data | The context layer's insider signal gets 20 years of history instead of waiting for its live record | "January 2006 - September 2026", quarterly (SEC page) | yes | yes | yes | M |
| 6 | **SEC EDGAR 8-K item 2.02 dates** (`submissions` API: form, items, acceptance time) | The day each quarter's results were announced - typically weeks before the 10-Q that `pit_record` dates them by | Tighter point-in-time timing for v2; an earnings-announcement-return signal | Item 12 from 2003-03-28, renumbered 2.02 from 2004-08-23 (SEC Releases 33-8216, 33-8400) | yes | yes | yes | M |
| 7 | **Hou, Xue & Zhang q-factors** (`global-q.org`) | Investment, ROE and expected-growth factors, daily to annual | A third independent construction (beside French and JKP) for the Investment category and for profitability | "1967-2025 sample" (their page) | yes | probably - no terms stated; publish a derived series with citation, as with French | ask | S |
| 8 | **AQR data sets** (QMJ, BAB, HML Devil, Value and Momentum Everywhere) | Quality-minus-junk from 1956 (US) - the closest published construction to the Quality category | Long-run prior for Quality | Monthly and daily; QMJ "U.S. sample 1956" (AQR page) | read only | **no** - terms forbid publishing "any information obtained from this Website" without written consent | no | S |
| 9 | **FRED / Treasury / Federal Reserve series** | Macro context | Already used by the macro backdrop | Decades | yes | yes for public-domain series, cited; **not** for copyrighted ones (VIXCLS, BAA10Y, S&P series) | public-domain series only | in use |
| 10 | **Alpaca free market data** (IEX feed; historical bars) | A second price source the bot's own broker supplies | ROADMAP 1.6 (compare the payload's prices with Alpaca's close at trade time) - an execution check, not research evidence | "7+ years" historical (their page) | yes (own account) | no - Alpaca's FAQ (2022): API data may not be redistributed | no | S |
| 11 | **Cboe put/call ratios** | Market-level option volume, 1995-2019 | Nothing new: the option-implied signals are already settled as "not score candidates" (`research/2026-10-12-option-implied-signals.md`) | 1995-2003 archive, 2006-2019 files (Cboe page) | terms not read | unknown | unknown | - |
| 12 | **FINRA Reg SHO daily short-sale volume** | Daily short-*sale* volume by venue | Not short interest - mostly market-maker flow; no hypothesis here worth a test | Consolidated file from 2018-08-01 (FINRA page) | yes | as row 2 | no | - |
| 13 | **Nasdaq Data Link "WIKI" prices** | Community EOD prices, frozen 2018-03-27 | At most a cross-check of Tiingo; delisted coverage undocumented | Secondhand sources only; licence described as open but not read at source | ? | ? | ? | - |
| - | Ken French library, JKP, Open Source Asset Pricing | Portfolio-level history to 1926/1963, 93 countries, 200+ predictors | **Already in use** - the Factor Lab's 17 experiments | `plan/factor-lab.md` | yes | yes (JKP non-commercial; French and OSAP cited) | JKP no | in use |

**Why this order.** Rows 1-3 each either close a gap on record or add months, and each is
public-domain or attributable. Row 1 is first despite its effort because survivorship is the
documented bias behind the only near-miss the research engine has produced (Size alone in
2012-2019, t 1.83-1.95). Row 2 is cheap and corrects a statement the project has repeated. Row 3 is
the only stock-level source of months that nobody has looked at. Row 4 would beat row 1 on
frequency, but nothing it produces can be published until Tiingo answers. Rows 5-6 make new
hypotheses testable, which is the overfitting note's warning case - worth doing only with a prior,
and the priors for large companies are near zero (Lakonishok & Lee 2001, *RFS* 14(1), for insider
trades - `research/2026-10-09-insider-purchases.md`; Martineau 2022, *Critical Finance Review*
11(3-4), for post-earnings drift: "For large stocks, PEAD have been non-existent since 2006", and
random-walk surprises "only predict positively returns following announcements prior to 1990" for
all-but-microcap stocks). Rows 7-8 add long-run priors the lab mostly has already.

## 4. The top three, each with a test written to be registered

Each is in the registry's fields. None has been run; **no return for any month or stock named
below was read in this session** - the only things measured were which members have filings
(row 3) and whether the FINRA files exist (row 2).

### Test A - does survivorship explain Size's lead? A two-sided version on public data

`research/2026-10-20-free-survivorship-sources.md` section 7 showed the monthly rebuild from Yahoo
can only *falsify* the survivorship explanation, because it adds the members that shrank and still
omits the acquired ones (whose announcement months carry the takeover premium - both omissions push
Size's measured lead the same way). 13F holdings price both kinds. The cost is frequency: quarterly.

```yaml
  - id: size-survivorship-13f-quarterly
    hypothesis: >-
      Size alone's 2012-2019 lead is mostly survivorship. Re-measured at quarter-ends on the
      complete membership - every member that quarter, departed ones included - Size alone's IC
      falls by at least half against the identical computation on the members backtest v2 builds
      today.
    source: >-
      Shumway, T. (1997), "The Delisting Bias in CRSP Data", Journal of Finance 52(1) (omitted
      delisting returns; about -30% for performance delistings); Linnainmaa & Roberts (2018), RFS
      31(7); research/2026-10-20-free-survivorship-sources.md sections 5(d) and 7; this note row 1.
    data: >-
      SEC Form 13F data sets, positions 2013-09-30 .. 2019-12-31 (files 2013Q4 .. 2020Q1). Price =
      median over filers of value / shares (shares >= 1,000, >= 5 filers; value in $ thousands
      before the 2023 form change). CUSIP -> CIK from SC 13G cover pages (subject-company CIK and
      CUSIP), else OpenFIGI, else the 2026-10-20 name match - the route used is recorded per name.
      Dividends added from XBRL CommonStockDividendsPerShareDeclared; splits detected as integer
      ratios in the filers' aggregate share count. Exit quarter: the deal's cash price from the
      target's merger filing, hand-checked (the eight known extraction errors corrected first), or
      for a stock deal the acquirer's 13F price times the exchange ratio; a bankruptcy with no later
      13F price takes Shumway's -30%. Market value = 13F price x the filed share count as of that
      date (pit_record). Membership: v2's point-in-time universe.
    statistic: >-
      Per formation quarter (25: 2013-09-30 .. 2019-09-30), the rank IC of Size alone (the
      registry's only-size transform) against the next quarter's return, computed twice with the
      same 13F prices for every stock: on the complete membership (IC_complete) and on the members
      v2 builds today (IC_survivors). Read the paired mean of dIC = IC_complete - IC_survivors and
      its paired standard error; also report IC_survivors.
    prediction: >-
      Survivorship explanation: mean dIC <= -0.5 x mean IC_survivors (at least half of Size's IC
      disappears once the departed members are ranked).
    refuted_if: >-
      mean dIC > -(1/6) x mean IC_survivors, or dIC >= 0 - adding the departed members leaves Size's
      IC essentially where it was, so survivorship cannot be what produced the lead. Between the two
      thresholds: inconclusive, and said so.
    void_if: >-
      mean IC_survivors <= 0 on quarterly 13F prices - there is then nothing for survivorship to
      explain at this horizon; or fewer than 80% of departed member-quarters receive a price.
    differences: >-
      quarterly where item 4(d)'s test is monthly; price returns plus filed dividends rather than an
      adjusted close; starts 2013-09 because the structured 13F sets do, so 2012 - the year with the
      largest gap - is outside it. Complements item 4(d), does not replace it.
```

### Test B - does short interest earn its 1.35 points among S&P 500 members?

```yaml
  - id: short-interest-dtc-finra
    hypothesis: >-
      Among S&P 500 members, low days to cover predicts a higher next-month return - the reason
      short_interest_ratio carries 13.5 of the Revisions category's 100 points.
    source: >-
      Boehmer, E., Huszar, Z. & Jordan, B. (2010), "The good news in short interest", Journal of
      Financial Economics 96(1), 80-97 (lightly shorted, heavily traded stocks earn positive
      abnormal returns; NYSE/AMEX/Nasdaq 1988-2005); Hong, H., Li, W., Ni, S., Scheinkman, J. & Yan,
      P. (2015), "Days to Cover and Stock Returns", NBER Working Paper 21166 - a working paper, not
      peer-reviewed (DTC long-short 1.2% a month, all stocks 1988-2012); McLean, R. D. & Pontiff, J.
      (2016), Journal of Finance 71(1) (predictor returns 58% lower after publication). Read the
      large-company and value-weighted rows of both papers from the PDFs before committing.
    data: >-
      FINRA equity short-interest files, settlement dates 2017-12-29 .. 2026-08 (exchange-listed rows
      verified present 2017-12-29 onward, 2026-10-22), field daysToCoverQuantity, each value used only
      from its FINRA dissemination date, never its settlement date (take the dates from FINRA's
      published reporting calendar; write them into the registration). Obtained through the FINRA
      Query API under its Equity Specific Terms, which permit derivative data and non-commercial
      redistribution with FINRA named as source; the website's own terms forbid building a database
      from the site. Membership and returns: backtest v2's.
    statistic: >-
      Monthly within-sector rank IC of -DTC against the next month's return, 2018-01 .. 2026-08
      (about 104 months), mean and t with month-to-month standard error.
    prediction: >-
      Mean IC positive but below 0.02 and not distinguishable from zero (|t| < 2): the published
      effect is concentrated outside the largest stocks and decays after publication.
    refuted_if: >-
      The metric's place in the weighting lacks support in this universe if the mean IC is <= 0 at
      the point estimate. It counts as supported only at t >= 3 (the engine's bar). Neither outcome
      changes a weight before 2027-02-11 (rule 5); after it, the research route applies.
    power: >-
      With a month-to-month IC spread near the baseline's 0.10-0.12, the standard error over 104
      months is about 0.010-0.012 and the smallest IC reliably seen at t >= 3 is about 0.04 - above
      the predicted effect, which is why a null here is read as "not large", not "zero".
```

Two things this would also settle. The project has written in two notes that short interest "has no
free history"; the files say otherwise for at least 104 months, and a correction belongs in the
next session that touches `pit_record.FIELD_CLASSES` (the metric is classed `no_free_history`).
And FINRA's page statement that exchange-listed data begins in June 2021 should be checked against
a full archive listing before any claim on a public page relies on either date.

### Test C - the baseline on months nobody has used

```yaml
phasein:
  months: [2010-03-31, 2011-12-31]
  hypothesis: >-
    The current weighting (the filings-and-prices screener, v2_config) ranks no better than chance
    in the XBRL phase-in months, as it did in 2012-2019 and 2020-2026.
  source: >-
    SEC Release 33-9002 (2009), Interactive Data to Improve Financial Reporting - phase 1, large
    accelerated filers above $5bn public float, periods ending after 2009-06-15; phase 2, the other
    large accelerated filers, a year later. This note's measurement (316 of 457 January-2012
    members with an XBRL balance sheet at 2009-06-30, 401 at 2010-06-30). The research engine's
    reading: baseline IC about zero in both eras (research/2026-10-15-research-engine-first-batch.md).
  data: >-
    Membership from the Wikipedia revision current at each month-end (universe_history; parsed and
    validated today at six dates 2009-06 .. 2011-12); records from pit_record as filed by that
    month-end; prices from v2's cache extended to 2009-02 for the 13-month window. A member enters
    only if pit_record builds its record - so until phase-2 companies have a tagged trailing year
    (about 2011-03) the universe leans to the largest members. Stated, not corrected.
  outcome_known_at_registration: none - no return for 2009-2011 has been computed by this project.
  prediction: >-
    Baseline IC point estimate within +-0.03 and |t| < 2 over the 22 months; no registered
    alternative differs from the baseline at |t| >= 3; the census builds a smaller share of members
    per month than 2012's 56.7%.
  refuted_if: >-
    The "no edge" reading is contradicted if the baseline IC is >= +0.05 with t >= 2 - or <= -0.05
    with t <= -2, which would say the screen ranked worse than chance where nobody had looked.
  power: >-
    About 0.10-0.12 / sqrt(22), a standard error near 0.023 for the baseline, and near 0.021 for a
    paired difference (scaled from 0.010 over 96 months). It can catch only a large effect; its
    value is that it is blind.
```

Building it touches `backtest_v2.py`, which is in `v2_experiments.RESULT_PATHS`, so the commit that
adds the months reruns every experiment (rule 3 of the engine); register first, in a commit of its
own, then build, as the 2012-2019 extension was done.

## 5. Terms, as read at the source on 2026-10-22

- **SEC** (`sec.gov/privacy`): "Information presented on sec.gov is considered public information
  and may be copied or further distributed by users of the web site without the SEC's permission";
  it asks for citation, reserves its seal and the EDGAR marks, and limits automated access to 10
  requests a second across all machines. The 13F, insider and financial-statement data-set pages
  carry an accuracy disclaimer and no other terms. Verdict: yes for all three uses.
- **FINRA website** (`finra.org/terms-of-use`): content "shall be used ONLY for your own
  non-commercial personal or professional use"; no derivative works from the website, no database
  built from it, no use in developing software or with machine learning "to predict trades".
  **FINRA Query API, Specific Terms for Equity Data** (December 2022; read from the PDF with `pypdf`):
  use "only for Developer's non-commercial personal or professional use"; redistribution of the data
  and "any derivative data or Resultant Data" to end users for their non-commercial use, free of
  charge, with FINRA "clearly" identified as owner and source; derivative data expressly permitted.
  It covers the `otcMarket` dataset group; **whether that group's short-interest dataset carries the
  exchange-listed rows the files do is unverified** and is the first thing to check with a free API
  account. The master API Terms of Service were not read. Verdict: (a) yes, (b) yes with attribution
  through the API, (c) no.
- **Tiingo**: personal use only; public aggregates need written confirmation
  (`research/2026-10-20-free-survivorship-sources.md`, addendum).
- **FRED** (`fred.stlouisfed.org/legal`): per series - "Copyrighted series contain the word
  'Copyright' in their notes"; public-domain series may be used with citation, including in reports;
  the API may not be used "in connection with the development or training of any software program
  or system or machine learning".
- **AQR** (`aqr.com/Terms-of-Use`): may not "publish any information obtained from this Website
  without AQR's express prior written consent". Private reading only.
- **global-q.org**: no terms, licence or citation request on the factors page.
- **OpenFIGI**: Bloomberg "dedicates FIGI Identifiers to the public domain", any purpose.
- **Alpaca**: free plan is the IEX feed with "7+ years" of history; its support FAQ (November 2022)
  says API data cannot be redistributed. The current market-data agreement was not found.
- **Cboe**: the historical-data page subjects the files to the Cboe websites' terms, which were not
  read. **Nasdaq Data Link WIKI**: only secondhand licence statements found.

## 6. What data cannot fix

**Months, not stocks, are the scarce thing.** A monthly IC's month-to-month spread is about 0.10-0.12
for this screener; its standard error falls with the square root of the number of months, and a
factor premium's good and bad months are common to every stock in the universe. More stocks per
month - a second universe, more fields per company - narrows the cross-sectional noise inside each
month and leaves that time-series noise untouched. That is why a true IC of 0.05 needs about **89
months** to show at t >= 3, and an IC of 0.03 about 20 years (`research/2026-10-15-research-engine-first-batch.md`
section 5). Nothing in section 3 changes that arithmetic. What adds months: the forward record (one
a month), the phase-in months of test C (22, once), and portfolio-level history - which the lab
already uses back to 1926, and across 93 countries through JKP.

**Every new signal is another draw.** Rows 5 and 6 would each be tested on the same 2012-2026 months
as the 30-odd configurations already in the ledger. The probability-of-overfitting reading was 0.63
with the one survivorship-exposed screen set aside; adding candidates raises the bar each must clear
(p < 0.05 / N, |t| >= 3) and gives luck more to choose from. A new signal earns a test only with a
published prior strong enough to predict its sign *among large companies* - and for insider trades
and earnings drift, the published priors predict nothing.

**Analyst estimates have no free past.** I/B/E/S, Zacks and Refinitiv are paid, and the no-paid-data
rule stands. Open Source Asset Pricing has portfolio returns for analyst-based predictors, but
all-stock and to 2024, not this screener's metrics on these stocks. The one point-in-time archive of
this screener's own analyst inputs is the data loop's: `improvement/snapshots/` holds 62 run
snapshots since 2026-02-20, and the recent ones carry each stock's percentile on `fy1_revision_3m`,
`analyst_surprise`, `price_target_upside` and the rest. That record is the evidence for the Revisions
category, and it accrues at one independent month a month. Say so wherever a v2 result is quoted.

**Fundamentals before mid-2009 exist only as documents.** The filings are on EDGAR as text and HTML;
there is no free structured version, and parsing fifteen years of financial statements by hand is
a project with its own error rate. Before 2009 the evidence stays portfolio-level.

**Acquired members at monthly frequency.** Free data gives them quarterly (13F) or at exit (deal
filings); monthly needs Tiingo (private) or a paid feed (excluded).

**A commercial licence.** Yahoo's terms, JKP's licence, FINRA's and FRED's copyrighted series all stop
at non-commercial use; only SEC data, public-domain FRED series and FIGI identifiers carry no such
limit. No free source makes the current scoring commercially usable - consistent with the bot's
ROADMAP, which already lists this.

**And rule 5.** Whatever these sources show, nothing a backtest says decides a weight before
2027-02-11.

## 7. Second reading and decision (2026-10-23, the 06:00 scheduled session)

Decided by a session that did not write the note, as the routine requires. Everything below was
measured today; the two scripts are committed beside this note.

### 7.1 What re-measuring found

**(a) The FINRA claim is true, and I re-ran it rather than taking it.**
`research/measurements/2026-10-23-finra-short-interest-probe.py` (+ `.json`) fetched twelve
settlement dates. 2014-12-31, 2015-12-31 and 2016-12-30 return **403**; **2017-12-29 onward return
200**, with 15,495 rows rising to 20,516 by 2025-06-30. The files are **pipe-delimited despite the
`.csv` extension** (the note does not say so; a reader following it with a comma parser gets one
column - the first run of my own probe did exactly that). `daysToCoverQuantity` is present on every
served date and is numeric and positive for **15,494 of 15,495** rows on 2017-12-29 and
**20,516 of 20,516** on 2025-06-30. `marketClassCode` on 2017-12-29: OTC 7,152, **NYSE 3,102, NNM
2,355, ARCA 1,433** - so exchange-listed rows are in the file four years before the June 2021 that
FINRA's own page gives, and the note's correction of FINRA is itself correct. All seven large
exchange-listed members probed (AAPL, JPM, MSFT, XOM, JNJ, KO, GE) are present on **every** served
date; AAPL 2017-12-29 reads 1.91 days to cover on 45,746,430 shares short against 23,901,107 average
daily volume, JPM 2.53 on 27,572,737 against 10,902,017. One construction detail the note should
carry: the field **floors at 1.00** (AAPL and JPM both read exactly 1.00 on 2018-12-31, when average
volume exceeded the short position), so it is censored below, not a clean ratio.

**(b) The registry cannot hold any of the three, and this is the decisive finding.**
`research/measurements/2026-10-23-can-the-registry-hold-these-tests.py` runs the engine's own call
sites against the note's entries instead of judging the YAML by eye. The committed registry has
exactly **two** shapes: `experiments:` - 29 entries, **every one** with a `transform` whose `kind`
is one of `baseline, config, drop_category, equal_categories, metrics, only_category, whole_market`
- and `evaluations:` - 2 entries, every one with a `rule` and **no** transform (the bot arms).
None of A, B or C is either. Against an entry shaped as the note writes it, `apply_transform` raises
**`KeyError: 'factor_weights'`**, and the bare `e["transform"]["kind"]` read raises
**`KeyError: 'transform'`** - a read that is unguarded at `v2_experiments.py:1122`, `:1156` and
`:1180` (`report`, `summary_for_page`) and in three places in `tests/test_engine_calibration.py`.
So the literal instruction does not describe an executable action: it would red the suite and the
public research page, not pre-register anything.

**(c) For test B specifically, registering first would manufacture a false null.** This is worse
than an error. `base_config()` is `v2_config(config.yaml)` (`v2_experiments.py:126-127`), which
zeroes every name in `backtest_v2.UNMEASURABLE_METRICS` - `short_interest_ratio` is in it
(verified) - and `backtest_v2._record` NaNs every field in
`pit_record.FIELD_CLASSES["no_free_history"]`, which holds **`shortRatio` and
`shortPercentOfFloat`** (verified). A `kind: metrics` transform *could* put the weight back, and the
entry would then **run without error** on an all-NaN column: zero coverage, renormalised away, a
"not distinguishable from zero" row in the ledger. Pre-registration exists to stop data-dependent
choices; here it would publish a null that is a property of the missing panel and says nothing about
short interest. **Build the panel, confirm the metric is actually scored, then register the
prediction before any return is computed.** That order is the opposite of the proposal's.

**(d) Test B's pre-registration is malformed as written.** Its `prediction` is a null (mean IC
positive, below 0.02, `|t| < 2`); its `refuted_if` is a condition on the **sign of the point
estimate** ("mean IC is <= 0 at the point estimate"). Under the entry's own predicted null the
point estimate is negative with probability **0.500** - so the entry would be declared *refuted* on
a coin flip by exactly the state of the world it predicts, while the prediction itself holds with
probability 0.466 at its stated SE of 0.011. A verdict rule that fires half the time under its own
null is not a pre-registration.

**(e) Test C has a 16.2% hole where it returns no verdict at all.** At its own stated SE of 0.023
over 22 months, the `prediction` band (`|IC| < 0.03`) binds at `|t| < 1.304` and fires with
probability **0.808** under the null; `refuted_if` (`|IC| >= 0.05` **and** `|t| >= 2`) binds at
`|t| >= 2.174` and fires with probability **0.030**. Between them sits **0.162** of the null
distribution for which the entry says nothing. At SE 0.021 it is 0.136.

**(f) What test C's 22 months actually buy.** At the engine's own month-to-month IC spread of 0.11,
adding 22 months to 2012-2019's 96 moves the standard error from **0.01123 to 0.01013** - a **9.8%**
precision gain - and the smallest IC visible at `t >= 3` from **0.0337 to 0.0304**.

### 7.2 The decisions

**Test A (`size-survivorship-13f-quarterly`): REFUSED.** Three reasons, in order of weight.
*First*, its own `void_if` requires a price for **>= 80% of departed member-quarters**, and both
coverage rates the note measured sit below that bar: **60 of 86 acquired members name-matched
(69.8%)** and **29 of 39 13F prices within 1% of Yahoo (74.4%)**. The entry proposes better
identification routes (SC 13G cover pages, then OpenFIGI), but their yield is **unmeasured** - so
the registration would commit an "L"-effort build whose feasibility precondition is unknown and
whose own stated outcome may be "void". Measure the identification rate first; it is a cheap
EDGAR exercise and it decides whether the test exists.
*Second*, the price panel is built from a long list of choices made while looking at the data -
filers with >= 1,000 shares, >= 5 filers, splits inferred as integer ratios in aggregate share
counts, deal prices hand-checked with "the eight known extraction errors corrected first",
Shumway's -30% for bankruptcies. Pre-registering the *hypothesis* does not protect a result when
the *measurement* carries that many free parameters; freezing the construction in code, before any
return is read, is what would.
*Third*, sequencing: `only-size-extended` is already queued (`OWNER_FOCUS.md` 2026-10-20 item 6)
with its one-sided reading written, is monthly rather than quarterly, and can **falsify** the
survivorship explanation - which is the decision-relevant direction, since the explanation is
currently the live excuse for a near-miss. Do the cheap falsifying test before building a quarterly
price panel from 13F filings. The YAML in section 4 stays as a draft for after it.

**Test B (`short-interest-dtc-finra`): ACCEPTED in substance, REFUSED in form.** It is the only one
of the three that tests a metric **carrying live weight** - `short_interest_ratio` is 13.5 of the
Revisions table's 100 points, **1.35 composite points** (`config.yaml:125`, verified) - and which
v2 today scores at weight 0 as unmeasurable. Its data is real, free, public-domain-adjacent and now
independently verified back to 2017-12-29 across ~104 months. Its published priors are genuine and
point the right way. It should be built. The **form** is refused for (b), (c) and (d): it cannot go
in `experiments:`, registering before the panel exists would publish a mechanical null, and its
refutation rule fires on a coin flip. The corrected build order, for the session that takes it:

1. Verify the licence route **before** collecting anything. My probe used
   `cdn.finra.org/equity/otcmarket/biweekly/`, i.e. the **website**, whose terms forbid derivative
   works and building a database from it. The route that permits publication is the **Query API**
   under the Equity Specific Terms. The CDN path containing `otcmarket` is suggestive that these are
   the same dataset group the API terms name, but it is not proof. If the API does not serve the
   exchange-listed rows, this test is private-reading only and **no figure from it goes on a public
   page** - settle that first, not last.
2. Build the panel: settlement dates 2017-12-29 onward, each value used from its FINRA
   **dissemination** date, with the reporting calendar written into the entry. Record that
   `daysToCoverQuantity` is **censored at 1.00**.
3. Wire it into v2 as a *measured* field - which means taking `shortRatio` out of
   `no_free_history` **only for the months the panel covers**, never globally. Reclassifying it
   without a panel makes `backtest_v2._record` read **today's** value into every historical month:
   a look-ahead leak of exactly the kind that class exists to prevent. Confirm non-zero coverage
   on the scored months before anything is predicted.
4. *Then* register, in its own commit, with the verdict bands disjoint and keyed to effect size
   rather than sign - e.g. **prediction**: mean IC in (0, 0.02) with `|t| < 2`; **refuted_if**:
   mean IC <= -0.02 with `t <= -2` (the metric is signed the wrong way for this universe) **or**
   mean IC >= 0.04 with `t >= 3` (the effect is large among large companies, against the decay
   prediction); **inconclusive** otherwise, and said so.
5. Read the large-company and value-weighted rows of Boehmer, Huszar & Jordan (2010) and Hong et
   al. (2015) from the PDFs first, as the entry already requires - the effect sizes in section 4
   are flagged as from abstracts and secondary summaries.

**Test C (`phasein`): REFUSED.** Not because blind months are worthless - the note is right that
months are the scarce thing - but on the trade. It buys **9.8%** of precision on the 2012-2019 era
(7.1f) for a rebuild of v2's records back to 2009 plus a full ~47-minute rerun; its confirm band
fires with probability 0.808 under the null and it leaves 16.2% of outcomes without a verdict
(7.1e); and its months are **selected on size**. The XBRL phase-in was mandated by public float
above $5bn (SEC Release 33-9002, the note's own citation), and a member enters only if `pit_record`
builds its record - so the phase-in universe is the largest members first, by construction. The one
near-miss the engine has produced is **Size alone in 2012-2019**, so these are precisely the months
that could not be pooled into the question they would most be wanted for. A test that cannot be
combined with the live question, almost certainly confirms, and has a 16% undefined region does not
earn a rebuild. **What would change this:** if the free survivorship fixes in `OWNER_FOCUS.md` item
6 are built anyway (they touch `RESULT_PATHS` and force the same rerun), the marginal cost of the
phase-in months drops to the record rebuild alone - reopen it then, with the verdict gap closed and
the size selection written into `differences` rather than left in prose.

### 7.3 Two corrections this decision leaves behind

- **"Short interest has no free history" is false and is written in three places in the code**, not
  two: `pit_record.py`'s `no_free_history` comment, `backtest_v2.py`'s `UNMEASURABLE_METRICS`
  comment, and `research/2026-10-15-pit-record-validation.md`. The accurate statement is that no
  free retrospective panel is **in use** - FINRA publishes days to cover from 2017-12-29, measured
  today. Corrected in this session's commit; the class **membership is deliberately unchanged**,
  for the look-ahead reason in 7.2 step 3.
- **The ledger was already 31 rows stale before this session**, from commit `502ca2b` (2026-10-22),
  which added `_price_bar_date`, `regularMarketTime` and `published_price_date` to `factor_engine.py`
  and `pit_record.py` - both in `RESULT_PATHS`. The change is additive and display/gating-only
  (`published_price_date` is called by `check_run_health`, never by scoring; the two fields are
  classed `price_history`, which v2 restates per month), so the rerun it owes **cannot move a
  number** - the same category as the 2026-10-18 provenance rerun, which measured 0 differences
  across 29 rows. It is provenance hygiene, not a result, and it is this session's logged Next.

## Sources

- SEC: Financial Statement Data Sets, Insider Transactions Data Sets and Form 13F Data Sets pages
  (`sec.gov/data-research/sec-markets-data/...`), and `sec.gov/privacy`; Releases 33-9002 (2009),
  33-8216 (2003), 33-8400 (2004).
- FINRA: `finra.org/terms-of-use`; Equity Short Interest data and files pages; Daily Short Sale Volume
  Files page; "FINRA API Terms of Service - Specific Terms for Equity Data" (12/2022 PDF,
  `developer.finra.org`); `cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv` probed.
- FRED legal page; AQR Datasets and Terms of Use; `global-q.org/factors.html`; OpenFIGI terms of
  service; Alpaca data page and support FAQ; Cboe historical options data page.
- Martineau, C. (2022), "Rest in Peace Post-Earnings Announcement Drift", *Critical Finance Review*
  11(3-4), 613-646, DOI 10.1561/104.00000122 (read from the CFR draft PDF with `pypdf`).
- Boehmer, Huszar & Jordan (2010), *JFE* 96(1); Hong, Li, Ni, Scheinkman & Yan (2015), NBER WP 21166;
  McLean & Pontiff (2016), *JF* 71(1); Shumway (1997), *JF* 52(1); Lakonishok & Lee (2001), *RFS*
  14(1); Linnainmaa & Roberts (2018), *RFS* 31(7). The effect sizes quoted for the first three are
  from abstracts and secondary summaries, flagged in test B as to be read from the PDFs first.
- Measurement: `research/measurements/2026-10-22-xbrl-phase-in-coverage.py` and `.json` (SEC
  `frames` API, total assets at ten quarter-ends 2009-2012; read-only, cached outside the repo).
