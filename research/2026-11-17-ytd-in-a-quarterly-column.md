# A year-to-date figure in a quarterly cash-flow column: how often, what it costs, and what can see it

2026-11-17, 13:00 session. The 06:00 session found that DexCom (ranked 8th) is scored on $2.209bn
of operating cash flow against $1.414bn in its filings, because the data source's column for the
quarter to 2026-09-30 carries $1.120bn - the company's figure for the **nine months** to that
date - and left three questions for research (`NIGHTLY_LOG.md`, its Next 2): how often does the
source do this, measured across the universe; can it be recognised from the statement alone; and
what follows. This note answers the first two by measurement, ships the display half of the
third, and leaves the scoring half as a dated proposal for a session that did not write it.

**Short version.** It is rare (3 of 2,300 quarterly columns; one stock inside the scoring window
today) and expensive where it lands (DexCom would rank about 20th, not 8th, and the error hid a
real fall in its cash flow). **Nothing the source returns can recognise it** - three detectors
tried, none usable. The company's filings recognise it exactly, with no false positive in 10,446
stock-days. So the page now says so on the two rows that read the figure; whether a filing may
also *correct* the score is the open question in section 7.

## 1. Why the error exists at all

A 10-Q does not contain a quarter's cash flow. Regulation S-X Rule 10-01(c)(3) (17 CFR
210.10-01, read on Cornell LII this session) requires, for Form 10-Q:

> "Interim statements of cash flows shall be provided for the period between the end of the
> preceding fiscal year and the end of the most recent fiscal quarter, and for the corresponding
> period of the preceding fiscal year."

The income statement's rule, (c)(2), names "the most recent fiscal quarter" as well; the
cash-flow rule does not. So the second and third quarters' cash flows exist in the filings only
as six- and nine-month totals, and every vendor that prints a "quarterly" cash-flow statement has
derived it by subtraction. That is documented practice on the academic side too: WRDS ships a
macro, `QUARTERIZE` - "Quarterizes Compustat YTM Cash Flow Variables in FUNDQ Dataset" - whose
code comments read "Note: Quarterize only Cash Flow items in Compustat" and "I/S items in
Compustat are quarterly numbers" (page fetched this session; it has no prose beyond its header
and comments, and the reading that it sets the first fiscal quarter equal to the year-to-date
value is from its code, not a statement on the page).

So the defect is not exotic: it is the one step every user of quarterly cash flow has to get
right, skipped for one column. The filings side of this project already does it correctly -
`sec_crosscheck.flow_ttm` builds twelve months as year-to-date + prior fiscal year - prior
year-to-date - which is why the cross-check saw DexCom at all.

## 2. Prevalence, measured

`research/measurements/2026-11-17-ytd-in-a-quarterly-column.py` fetched the source's quarterly,
annual and trailing cash-flow statements for the 499 stocks on today's page (498 returned, 497
with an operating-cash-flow row, 2,479 quarterly columns; pickled to the gitignored
`data/measurements/2026-11-17-cashflow/`). Each column was compared with what the SEC cache
implies for the same period end: the **discrete quarter** (year-to-date minus the year-to-date
one quarter earlier) and the **year-to-date** figure itself, each as first filed.

| Of the 2,300 columns the filings can judge | Columns |
|---|---|
| within 2% of the filed quarter | **2,283** (99.3%) |
| over 2% from the quarter and within 2% of the filed **year-to-date** figure | **3** |
| neither | 14 (8 over 10% from the quarter, on 7 stocks; 3 inside the four newest columns) |

First fiscal quarters cannot show the error (there the two figures are one number); of the
**1,795** columns at fiscal position two to four, 3 are year-to-date figures - **0.17%**.

The three:

| Stock | Column | Source | Filed quarter | Filed year-to-date | In the twelve months scored? |
|---|---|---|---|---|---|
| DXCM (#8) | 2026-09-30 | $1.120bn | $0.326bn | $1.120bn (nine months) | **yes** |
| DXCM | 2025-09-30 | $1.147bn | $0.660bn | $1.147bn (nine months) | no - fifth column back |
| GM (#374) | 2025-06-30 | $12.969bn | $6.908bn | $12.969bn (six months) | no |

**One stock of 497 is scored on such a column today.** Both of DexCom's are the two nine-month
figures its 2026-09-30 10-Q carries (this year's and the comparative), which says when it
happened: `--history` reads every published page git holds since `inp` was first published
(24 pages, 2026-10-08 on). DexCom's operating cash flow was **$1.749bn, equal to the filing to
the dollar, on all 16 pages to 2026-10-30**, and $2.209bn on all 8 from 2026-11-03. The error
arrived with the new 10-Q.

## 3. What it is worth to the stock it lands on

`--impact DXCM`: two scored metrics read operating cash flow, free-cash-flow yield (45% of
Valuation) and accruals (5% of Quality). Piotroski's two cash-flow tests still pass on the filed
figure ($1.414bn is positive and above net income of $0.991bn) and the Beneish score reads the
annual statement, so neither moves. Restating the two with the filed twelve months and reading
each new sector percentile off the published values of DexCom's 59 Health Care peers:

| Metric | Published | With the filed figure | Sector percentile | Composite |
|---|---|---|---|---|
| FCF yield | 5.77% | 3.12% | 83.6 -> 30.6 | **-5.25** |
| Accruals | -0.195 | -0.068 | 97.5 -> 86.6 | -0.12 |

Composite 70.56 -> about 65.2; **rank 8 -> about 20**, every other stock held still. This is an
estimate, not an engine rerun: it first rebuilds both published category scores from the
published percentiles and weights (and refuses to print if either is off by 0.01), but it holds
the peers' percentiles fixed.

The more useful reading is in the history. DexCom ranked 7-10 on every page from 2026-10-20 to
2026-10-30, on a correct $1.749bn. Its true third quarter was weak - $0.326bn against $0.660bn a year earlier - so the
filed twelve months **fell** to $1.414bn. The page showed them **rising** to $2.209bn and the
rank improving to 5-8. The error did not just flatter a number; it reversed the direction of the
one piece of news in the quarter.

## 4. Can it be recognised from the statement alone? No - three detectors, measured

Scoring reads only the data source (section 6 is why), so the first question is whether the
source's own figures give the column away. Each detector below uses nothing else, and is judged
against section 2's ground truth.

**A. "The column is at least the sum of the earlier quarters of its fiscal year"** (what a
year-to-date figure must satisfy when cash flow is positive; fiscal position taken from the
source's annual statement dates). Testable on 862 judged columns; it fires on **311 (36.1%)**.
On the newest column only - where a detector would act - testable on 379 and fires on **63: one
true case and 62 correct quarters.** By position: 246 of 431 second quarters, 59 of 385 third
quarters. A good third quarter after two ordinary ones is not rare; it is one stock in six.

**B. "A completed fiscal year's four quarters do not add up to the annual statement."** An
identity, so it cannot false-alarm: **0 of 61** stock-years differ by more than 2%. But only 61
stocks are testable at all - the source serves five quarterly columns, so a full fiscal year is
on file only around its own year end - and it is blind by construction to the current fiscal
year, which is where a new 10-Q's error lands. It would have seen neither of DexCom's columns
nor GM's.

**C. "The source disagrees with itself"**: its own trailing-twelve-month figure against the sum
of its four newest quarters. 494 stocks carry the trailing figure, 458 comparably; **DexCom's
trailing figure is $2.209bn - it repeats the error** - so C does not fire on the one stock it is
needed for. It fires on four others (DD 92%, COIN 63%, SPGI 9.8%, EBAY 3.2%) for unrelated
reasons, which is a separate observation and not pursued here.

So the morning's caution was right, and by a wide margin: the cheapest heuristic raises 62 false
alarms per true case, and the two clean ones cannot see it.

## 5. What can: an identity between the run's figure and three filed ones

If one quarterly column inside the twelve months carries the year-to-date figure, the run's
total exceeds the true twelve months by exactly the part of the year before that quarter. All of
that is checkable against the filings with no access to the source's columns:

> the run's operating cash flow, minus the filed twelve months to the statement's date, equals
> the filed year-to-date figure one quarter before one of the last four quarter ends.

For DexCom: $2.2092bn - $1.4144bn = $0.7948bn, and its filed six months to 2026-06-30 are
$0.7948bn. `sec_crosscheck.ytd_columns` implements it (match within 0.5% of the run's figure;
ignored where the excess is under 2% of it, the filing check's own bar).

`research/measurements/2026-11-17-ytd-signature-history.py` ran it over all 24 published pages:
**10,446 stock-days compared; 151 of them over 2% from the filing and 95 over 10%, for every
reason there is; the identity fires on 8 - DexCom, on exactly the 8 pages from 2026-11-03.**
None of the other 143 disagreements matches, including the nine annual-statement stocks the
06:00 session explained and the 14 "neither" columns above. It agrees with section 2's
column-level finding from an independent direction (one reads the source's columns, the other
never sees them).

It is an identity among filed numbers rather than a threshold on a ratio, which is why it has no
seasonal false positive to tune away. What it cannot do: fire when two columns are wrong at once,
or when the company's own filing is mis-tagged (the cross-check's known limit); both fail silent,
which for a note is the safe direction.

## 6. What shipped today: the page says it, where the number is

Display only. `generate_dashboard._attach_ytd_column_note` writes `srcn` for the stock, and the
free-cash-flow-yield and accruals rows print, under "Against this company's filings":

> The $2.2B of operating cash flow above counts part of the year twice. The data source's column
> for the quarter to September 30, 2026 carries $1.1B, which is this company's filed figure for
> the nine months to that date; the quarter alone was $326M as filed. With that quarter, the
> twelve months to September 30, 2026 come to $1.4B in the company's filings, 36% below the
> figure used here. This score is computed from the source's figure as delivered and has not
> been adjusted.

Data Provenance says the same in one sentence and - for this cause only - no longer follows
"operating cash flow (36%) differs" with "a difference usually means the two sources define the
line differently ... rather than that either figure is wrong", which for DexCom was false.
Claim `drilldown.ytd_column_note`, 12 registered checks, 9 of 9 mutations caught
(`research/measurements/2026-11-17-ytd-note-mutations.py`). The published payload was built
before the note, so it reaches the site with the next 02:00 run; it was looked at today on a
throwaway copy of DexCom's sheet at 1440 and 375 px
(`research/measurements/2026-11-17-ytd-note-on-the-page.py`).

## 7. The open half: should the filing correct the score?

Not built, on purpose. Three reasons it is a second session's decision:

1. **It crosses a settled line.** `sec_crosscheck` is display-only and a test fails if a scoring
   module imports it (`CLAUDE.md` 0.14). That rule exists because of
   `research/2026-10-14-yahoo-vs-sec.md` section 3c, which measured a per-stock fallback to the
   filings and refused it: the *definitions cost* of switching source for one stock (median 20.5
   rank places) was as large as the staleness it cured (17.5), and "mixing two sources inside
   one cross-sectional ranking also breaks what a percentile means."
2. **This case is narrower than the one 3c refused, and the difference is measurable.** 3c
   swapped a stock's whole record. Here one field would change, for a stock where that field is
   provably not the company's figure, to the figure the source itself published for the same
   stock until the day before the 10-Q. And for operating cash flow the definitions cost is
   close to nil: the source and the filings agree within 0.5% for 402 of 418 stocks today, and
   2,283 of 2,300 quarterly columns sit within 2% of the filed quarter. Net income is 388 of
   427, which is why this argument does not transfer to it.
3. **One case.** The identity has fired on one stock in six weeks of pages. A rule that changes
   a rank should not be justified by a single instance however clean.

**The options, for the reviewer:**

- **(a) Leave the score, keep the note** - today's state. Costs: a public rank 8 that the page's
  own row says rests on a doubled half-year; the trading bot reads rank, not the note (rank 8
  and rank 20 are both inside its top 40, so today nothing it does would differ).
- **(b) Score the stock on the filed twelve months for this one field** when, and only when, the
  identity fires - at fetch, where the quarterly columns are in hand and the single column could
  be shown to equal the filed year-to-date figure before it is replaced by the filed quarter.
- **(c) Treat operating cash flow as missing** for such a stock, so free-cash-flow yield and
  accruals go unscored and their weight is renormalised - uses the filing only to *withhold*,
  never to supply a figure. For DexCom that removes 45% of Valuation's weight from a stock whose
  true figure is known to the dollar.

My recommendation is **(b), not yet**: build nothing until the conditions below are met, and
then build (b) rather than (c) - (c) discards a known number to avoid stating it.

**Pre-registered, before any further page is read** (so they cannot be fitted to DexCom):

1. *The identity must have fired on at least two more stocks*, each confirmed by fetching the
   source's quarterly statement and finding the column equal to the filed year-to-date figure
   within 2% (`2026-11-17-ytd-in-a-quarterly-column.py`, which also re-measures section 2's
   table). **Any firing where the column is not that figure refuses (b) and (c) both** - it
   would mean the identity can be satisfied by coincidence.
2. *The definitions cost must be re-measured for the field alone*: for the stocks whose operating
   cash flow is within 0.5% of the filing, substituting the filed figure must move no rank by
   more than 1 place. If it does, 3c's objection applies after all.
3. *Reversal on the next filing must be checked*: when the source corrects a column (its columns are
   not stable: DexCom's twelve months to June matched the filing until 2026-10-30, which they
   could not have with today's 2025-09-30 column in them, so that column was rewritten to the
   nine-month figure when the 2026 10-Q arrived - an inference from the totals, the old column
   was not kept), the substituted figure and the source's corrected one must agree within
   0.5%, i.e. the rank does not jump twice.
4. If (b) is built it needs its own `DECISIONS.md` entry rewriting the display-only rule for
   exactly this field and this identity, a changelog entry, a line on the stock's row saying the
   filed figure was used, and - because `factor_engine.py` is one of the research engine's
   result paths - the 32-row rerun budgeted.

Review this no earlier than the first data run on or after **2026-12-15** (the same run the
cross-check's range table is re-read on - one sitting), by re-running
`2026-11-17-ytd-signature-history.py`. If the identity has fired on no further stock by then,
the answer is (a) and the status line below says so until it does.

**Proposal status: WAITING until 2026-12-15** - whether a stock whose operating cash flow
provably counts part of a year twice should be scored on the filed twelve months for that one
field (option b), left as published with the note (a), or have the field withheld (c), on the
four conditions in section 7. It is deliberately **not** marked OPEN: the brief makes an OPEN
proposal the next session's first job, and this one cannot be decided before its conditions can
be read. The brief's "Coming up" carries the date; the session that reads it then either decides
or rewrites this line to OPEN for the one after.

## 8. What this does not settle

- **The 14 "neither" columns** (section 2). Eight are over 10% from the filed quarter; three sit
  inside the scoring window (DTE 2025-09-30: $0.133bn against $0.632bn filed; VMRK; SKYD, whose
  filings give no year-to-date figure). They are not year-to-date errors and nothing here
  explains them. The cross-check already counts them every morning.
- **Detector C's four** (DD, COIN, SPGI, EBAY): the source's trailing figure and its own four
  quarters disagree. For COIN the four quarters equal the filing ($3.851bn), so there the
  trailing figure is the odd one - and the engine does not read it.
- **Other cumulative lines.** Capital expenditure comes from the same statement and is exposed
  to the same step; the cross-check compares it but the page does not show it, and it was not
  measured here.
- **History.** Payloads before 2026-10-08 do not carry `inp`, so whether GM was scored on its
  doubled half-year between July 2025 and July 2026 cannot be read from the record.

## Sources

- **17 CFR 210.10-01(c)(2)-(3)**, Regulation S-X, "Interim financial statements". Read this
  session at https://www.law.cornell.edu/cfr/text/17/210.10-01 (quoted above).
- **WRDS, "QUARTERIZE - Macro: Quarterizes Compustat YTM Cash Flow Variables in FUNDQ
  Dataset"**, https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/wrds-macro-quarterize/
  - header and code comments read this session; no prose description exists on the page.
- Deloitte's *Roadmap: Statement of Cash Flows*, Appendix D ("SEC Interim Reporting
  Considerations") came up in the same search as saying a quarter-only cash-flow statement is
  permitted, not required - **not read**; the rule text above is what this note relies on.
- This project's own measurements, all committed under `research/measurements/2026-11-17-ytd-*`
  with their JSON, and `research/2026-10-14-yahoo-vs-sec.md` sections 3c, 5 and 6.
