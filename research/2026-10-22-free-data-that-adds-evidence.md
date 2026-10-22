# Which free data would add evidence, rather than more tests on the same history?

**Date:** 2026-10-22 (delegated research session). **Kind:** a data-source evaluation.
**Not a methodology proposal.** Nothing here changes a weight, a metric or the research registry;
under CLAUDE.md rules 4 and 5 nothing a backtest says decides anything before 2027-02-11. The three
pre-registerable tests in section 4 are written in the registry's style and **deliberately not added
to `research/experiments/registry.yaml`** - a session that adopts one commits it there, before any
month of it is built.

**Proposal status: OPEN** - register tests A, B and C (section 4) in `research/experiments/registry.yaml`, each in its own commit before any of its data is built; or refuse one here with the reason.

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
