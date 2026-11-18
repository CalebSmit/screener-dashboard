# What the screener knows about its own inputs, and what it does with it - synthesis, 2026-11-18

06:00 session, the Wednesday synthesis. Subject, per the rotation: everything that has changed
the methodology since the last synthesis session (2026-11-04).

**What this note decides:** one published sentence and one badge label are corrected (section 3),
and the health line gains one count (section 5). **No score, weight, metric, threshold or rank
changes**, and nothing here is justified by a return, an information coefficient or a backtest.
Section 4 sizes a question and hands it to a Monday; it is not a proposal and carries no status
line.

Measurements, all read-only and re-runnable, under `research/measurements/`:

| Script | What it reads |
|---|---|
| `2026-11-18-what-the-page-knows-about-its-inputs.py` | every input caveat on the published page, counted |
| `2026-11-18-stale-sentence-against-the-filings.py` (`--history`) | the staleness flag against the payload's own filing facts |
| `2026-11-18-what-the-annual-fallback-leaves-out.py` | the fallback's figures against the filed twelve months |
| `2026-11-18-filings-the-sec-cache-does-not-hold.py` (`--ask`) | which companies the SEC cache has stopped receiving |
| `2026-11-18-stale-sentence-on-the-page.py` | the corrected sentence and badge, rendered at 1440 and 375 px |

## 1. The subject: five entries, one family

`METHODOLOGY_CHANGELOG.md` holds five entries after the 2026-11-04 synthesis:

| Date | Entry | Moves a score? |
|---|---|---|
| 2026-11-04 (13:00) | the options panel inverts its own mids and records which construction it used | no (context) |
| 2026-11-06 | a clamped metric row shows the figure its own arithmetic gives, then the cap | no (display) |
| 2026-11-06 (night) | a share count filed in the wrong units no longer makes a five-year yield range | no (context) |
| 2026-11-17 | figures read from the annual statement are dated by it | no (display) |
| 2026-11-17 (13:00) | operating cash flow that counts part of a year twice says so | no (display) |

None changed how a stock is scored. All five are the same kind of change: **the page learning to
say that an input is not what it appears to be.** That has been the dominant kind of change for
six weeks - the filing cross-check (10-14), the period-behind note (10-15), the lease and
special-item notes (10-15, 10-16), the stale bar (11-02) and now these. Each was justified on its
own day by a failure someone could demonstrate. Nobody had looked at the layer they add up to,
which is the question a synthesis session exists to ask.

## 2. The layer as a whole, counted

On the page published 2026-11-18 (499 scored), every caveat the page can attach to a stock's
*inputs*. "Reaches the score" is read from the code, not inferred.

| Caveat | Universe | Top 40 | Top 25 | Reaches the score? |
|---|---|---|---|---|
| Reported and normalised EPS disagree | 94 | 2 | 2 | no - a sentence and a badge |
| A weighted metric missing | 91 | 4 | 2 | **yes** - the category renormalises; the coverage discount below its bar |
| Filing note on earnings yield (a one-off charge) | 49 | 3 | 1 | no (0.14) |
| An input over 2% from the filing | 44 | 7 | 4 | no (0.14) |
| A period behind the filings | 32 | 1 | 0 | no |
| Statement flagged stale | 27 | 0 | 0 | no |
| Annual statement read instead of four quarters | 26 | 0 | 0 | **it is the route** - the score reads its output |
| Enterprise value substituted | 20 | 1 | 1 | **it is the route** |
| An input over 10% from the filing | 14 | 3 | 2 | no (0.14) |
| Three quarters scaled to twelve months | 13 | 1 | 1 | **it is the route** |
| Rank move is input churn | 12 | 0 | 0 | no - a sentence on the move |
| A category unscored | 5 | 0 | 0 | **yes** - the others are reweighted |
| Filing note on FCF yield (lease purchases) | 4 | 0 | 0 | no (0.14) |
| Twelve months rolled forward | 3 | 1 | 1 | **it is the route** |
| Source error named (year-to-date column) | 1 | 1 | 1 | no - WAITING until 2026-12-15 |
| Quarterly statement refused | 1 | 0 | 0 | **it is the route** |

**247 of 499 stocks carry at least one** (146 one, 53 two, 31 three, 17 four or more); **17 of the
top 40** - the set `signals.json` hands the trading bot - and **11 of the top 25**. The four flags
about the *company* rather than its data (growth trap 63, Beneish 44, value trap 41, channel
stuffing 14) are separate and not counted here.

Three things the table shows that no single entry could.

**(a) The layer has exactly two kinds of member, and they never mix.** Either the engine acts
(an input is missing, or a statement cannot be used, and the score reads what is left or reads
another route), or the page speaks and the score is untouched. There is no caveat that discounts
a score a little. That is the settled display-only rule of 0.14 holding everywhere, and it is the
right shape for a tool that must be explainable: a reader can be told "this number was used as
delivered" or "this number was not used", and never "this number was used at 70%".

**(b) Half the universe carries a caveat, so a caveat is not an alarm.** The count that matters
is the short one: in the top 40, three stocks have an input over 10% from the company's own
filing - DexCom (#8, operating cash flow +36%, named as a source error), CF (#17, net income
-15%) and Altria (#39, revenue -12%) - and two more reached their twelve months by a route
other than four quarters (Tapestry #10 rolled forward, Travelers #15 scaled, both stated on the
sheet). The morning brief's "Top 25, worth a look" already names the first group; nothing new is
needed there.

**(c) Three of the rows are one row.** Section 3.

## 3. Finding: two correct changes met and made a false sentence

On 2026-11-02 the stale bar moved from 120 to 165 days, so the flag would mean "a quarter or more
behind" and not "following the SEC's deadlines". On 2026-11-17 statement dates became the period
end of the statement the figures were *read from*, which is right, and which the entry said
would turn the flag on for the stocks read from their annual statement. Both changes were sound
and both were verified. Together they produced this, on Amazon's page, this morning:

> The score rests on 27 of 27 metrics. **Its filings are flagged stale (322 days old).**

Amazon filed its 10-Q for the September quarter on 2026-10-30, nineteen days earlier, and the
same payload says so two fields away (`newer_filing`). The Data Provenance badge beside it read
"Filing Age: 322 days".

Measured across the page (`2026-11-18-stale-sentence-against-the-filings.py`):

- **27 stocks** carry the flag and the sentence.
- **27 of 27** have a newer filing on record in the SEC cache; **22** of those were filed inside
  the 165-day bar itself.
- **27 of 27** have a balance sheet inside the bar (26 at 2026-09-30).
- **26 of 27** are stocks the fetch read from the annual statement because the source's
  quarterly statement lacks a quarter; the 27th is DuPont, whose quarterly statement was refused
  for a negative-revenue quarter. **None is flagged for any other reason.**

So the sentence was false of every company it was printed for. What is 322 days old is the
period end of the income statement *the score read*. The company's filings are current; the
data source's quarterly table has a hole in it.

The history says when it stopped meaning anything else (`--history`, every page under the
165-day bar):

| Run | Flagged | ...with a newer filing on record | ...balance sheet inside the bar |
|---|---|---|---|
| 2026-11-04 | 8 | 4 | 1 |
| 2026-11-05 | 8 | 4 | 1 |
| 2026-11-06 | 5 | 4 | 1 |
| 2026-11-10 .. 11-17 | 1 | 1 | 1 |
| 2026-11-18 | 27 | 27 | 27 |

**What this makes redundant.** On today's page `statement flagged stale` (27) is a strict subset
of `annual fallback of either kind` (27) and of `a period behind the filings` (32). The flag adds
no stock the other two notes do not already name, and the other two say *why*. It is kept, as
what the 11-02 entry called it - "the coarse fallback" for a statement that is old with no filing
in the cache to say so, which was half the flagged stocks on 11-04 - but it should not speak
louder or more vaguely than the notes that know the cause.

**What shipped** (display only; `METHODOLOGY_CHANGELOG.md` 2026-11-18):

- `stock_summary._sentence_confidence` speaks of the statement and never of "its filings". Where
  the fetch read the last fiscal year for the income statement, it gives that as the cause:
  *"Its income statement and cash-flow statement figures are the last fiscal year's (322 days
  old): the data source's quarterly figures could not be summed to twelve months."* Otherwise:
  *"The income statement this score read is flagged stale (322 days old)."* The condition is the
  one the Data Provenance panel's "Last fiscal year used" already uses, so the two cannot
  disagree, and the fallback is named only when the income statement is among the statements
  replaced - the age is the income statement's.
- The badge is **Statement age**. It was never the age of a filing: for a company that filed its
  September quarter twenty days ago it read "Filing Age: 49 days".
- "The case against" clears a stock with "no missing category, stale statement or EPS
  disagreement".
- Claim `summary.confidence` states the new assertion and gains three checks.

Rendered from a rebuild of today's run with the new code: 26 stocks get the first wording, 1 the
second, none the old one. Seen at 1440 px (Amazon, "Read with care" under "Why it ranks here":
three lines, nothing clipped) and 375 px (the badge row wraps to two lines, "Statement age: 322
days" in the alert colour on its own line; the sentence is five lines at 343 px; the document is
no wider than the viewport).

**The general lesson, for the next entry of this kind:** a flag's *wording* is a claim about its
cause, and a change to what a date means changes which stocks a date-driven flag fires for.
Yesterday's entry predicted the flag would turn on and was right; nobody re-read the sentence
the flag prints. When a change moves a population into an existing flag, read the flag's
sentence for the new population before shipping.

## 4. What the fallback leaves out - sized, not acted on

The implication for the eight categories. Twenty-seven stocks - 5.4% of the index - have their
income and cash-flow figures from a fiscal year that ended, for 23 of them, on 2025-12-31, beside
a balance sheet at 2026-09-30 and prices, analyst figures and returns from yesterday. Category by
category:

- **Momentum 13, Risk 10, Revisions 10, Size 5** (38 points) read prices and analyst data. Current.
- **Investment 5** reads asset growth from the balance sheet. Current.
- **Valuation 22, Quality 22, Growth 13** (57 points) read flows: earnings, cash flow, revenue,
  margins. For these stocks, most of a year old - set against a current price in Valuation's
  yields and a current balance sheet in Quality's returns.

`2026-11-18-what-the-annual-fallback-leaves-out.py` compares what was scored with the twelve
months to the newest period the companies have filed, where the SEC cache can supply both:

| | Comparable | Scored below filed | Above | Median gap | Median absolute gap | Over 10% | Over 25% |
|---|---|---|---|---|---|---|---|
| Net income | 18 | 15 | 3 | -13.7% | 32.1% | 13 | 10 |
| Operating cash flow | 17 | 10 | 6 | -5.3% | 13.5% | 10 | 5 |

Amazon: $77.7bn scored, $134.8bn filed. Boeing: $2.2bn against $7.7bn. Mondelez: $2.5bn against
$4.1bn. And where the 27 sit: mean Quality score **41.0 against 50.4** for the other 472 (-2.9
standard errors), Valuation 43.4 against 50.2 (-1.5), Investment 43.0 against 50.4 (-1.3); the
four market-fed categories are within 1.2 standard errors of the rest in either direction. Median
rank 330.

**How to read that, and how not to.** It is a measurement of *inputs against filings* on one
morning. It is consistent with the obvious mechanism - in a year when most large companies'
earnings rose, last year's earnings under this year's price reads as expensive and less
profitable - and the split between statement-fed and market-fed categories is what that
mechanism predicts. But it is 27 stocks in one cross-section, **which stocks the source fails
for is not random** (five are utilities and five are REITs, low-Quality-score sectors on this
screener's metrics before any of this), individual rows are unaudited (FedEx and FedEx Freight
are mid-separation and their "filed twelve months" should not be trusted to the dollar), and of
the nine with no comparison, five are among section 5's twenty and two more (Eversource, Ameren)
are in its 101-165 day band. It does not show that these stocks are
ranked too low. It shows the size of what is not being read.

**Why nothing is done about it today.** Three settled things stand in the way, each for a reason
that was measured:

1. Scoring may not read a filing (0.14; `research/2026-10-14-yahoo-vs-sec.md` section 3c).
2. The source's own trailing-twelve-month series reached the newer quarter for 13 of the 15
   stocks measured on 2026-10-30, but for a hole two quarters back it **cannot be corroborated
   from the source alone**
   (`research/2026-10-29-ttm-fills-the-empty-quarter.md` section 5.4: `TTM - (three quarters)`
   is not computable when one of the three is the empty column), and that series repeated
   DexCom's error to the dollar (`research/2026-11-17-ytd-in-a-quarterly-column.md` section 4).
3. The pre-registered narrow rule (10-29 section 5.5) admits a gap of **exactly one quarter**.
   On 10-30 it admitted 6 stocks. The population has since moved out from under it: it was
   written for an empty *newest* column, and on yesterday's live fetch of nine of these stocks
   the newest column (September) is populated and the hole is the June quarter behind it, so the
   rule has nothing to fill. I did not re-fetch all 26 to confirm it admits none.

**The synthesis point is that this and DexCom are one question.** Both ask whether a company's
own filing may do anything to a scored input beyond being printed beside it. DexCom's asks it of
one field of one stock and waits until 2026-12-15 on four conditions written in advance. This
asks it of two statements for 26 stocks. Deciding the small case in December without the large
one in view would settle the principle by its least representative instance. The distinction
worth researching is between a filing **replacing** a source figure (refused, 10-14 section 3c)
and a filing **corroborating** a figure the source itself supplies by another route - MSCI's
"dual vendor validation" wording in its Fundamental Data Methodology section 1.1.5, quoted in the
10-29 note section 5.2, is the documented-practice version of the second.

**Queued as a Monday, ahead of 2026-12-15:** how do point-in-time data vendors and index
providers treat a company whose quarterly series has a hole - hold the last complete twelve
months, carry the fiscal year, or admit a trailing figure once a second source agrees - and what
does each do to the 26? From the literature and from documented practice, with the 10-29 and
11-17 notes as its starting material. No design is proposed here.

## 5. Finding: the check that feeds this layer cannot see twenty stocks, and said "0 days old"

Nine of the 27 had no filed twelve months to compare in section 4. Following that up:

- **20 of 499 scored stocks have no 10-K or 10-Q in the SEC cache filed in the last 165 days.**
  Fourteen utilities, five REITs and Citigroup. Nineteen were last filed between 2026-04-23 and
  2026-05-08 - their March-quarter 10-Q - and Citigroup's newest is its 10-K of 2026-02-20.
- The filing cross-check compares **no field at all for 15 of the 20** (and 2-3 fields, against
  old periods, for the five read from their annual statement). One of the 20, Evergy, ranks 27th.
- The cache file was rebuilt the same morning. The brief's row read "filings cached 0 days ago".
- **The gap is in the SEC's `companyfacts` record, not in what the refresh keeps of it.** Asked
  directly (`--ask REG C`): Regency Centers' record carries no fact of any kind filed after
  2026-05-04; Citigroup's carries nothing from a 10-Q in 2026, though it has fee-table facts
  from prospectus supplements filed yesterday. I did not establish why. The membership -
  companies that file one combined report with their subsidiaries or operating partnership -
  suggests something about how those filings are tagged, and that is a guess, marked as one.

Nothing on the public page is false because of this: a stock with nothing to compare says "the
filing check: not available", which is true. What was wrong is one level up - **the only number
describing the cache's health was the age of the file**, which is the "tripwire wired to a number
that cannot stand still" of CLAUDE.md rule 8, in a new place.

**What shipped:** `sec_crosscheck.no_recent_filing` and one more figure on the health line every
morning, carried into the brief's row: *"20 stocks with no 10-K or 10-Q in the SEC cache filed
in the last 165 days (AEP, C, CNP, ...)"*. The bar is the filing calendar's, not a fitted one:
two periodic filings of a large accelerated filer are never more than about 142 days apart (a
third-quarter 10-Q filed in mid-October to a 10-K due 60 days after the year end), plus the same
two weeks the stale bar allows. It is a count, not an ATTENTION line and not a gate: it reads 20
on its first morning, and a line that fires every day is a line nobody reads.

**What it does not catch, stated so nobody over-reads the 20:** 18 more stocks sit between 101
and 165 days. About ten are utilities and REITs last filed in early August, which very likely
means one filing missing rather than two; the rest have September fiscal years and are simply
between their June 10-Q and a 10-K not yet due. A bar on filing dates cannot tell those apart,
so the count is of the cases that are certain.

**What it means for a commitment already made.** The filing cross-check's three fields are due
to be considered for promotion to a publish gate on 2026-12-15 (`research/2026-10-14-yahoo-vs-sec.md`
section 6). The pre-registered rule is unchanged and I have not touched it. But whoever applies it
should know the denominators: total assets was compared for 440 of 499 this morning, and for
these 20 companies a gate on the cross-check would be a gate that cannot fire. The `COMING_UP`
line now says so.

## 6. The design, written down

Three rules the accumulated notes already obey, stated once so the next one is written to them.

1. **A caveat names the thing that is impaired - a statement, a column, a field - and never the
   company.** "Its filings are stale" was a sentence about the company. Every other note in the
   table says which figure, from which statement, and what was done.
2. **The source decides the score; the filings decide what the page says about it.** Unchanged,
   and now with two open questions pressing on it from the same side (section 4). They are to
   be decided together, after research, not one at a time.
3. **A check says what it could not see.** The cross-check now does, for the companies whose
   filings never arrive. This is the 2026-11-09 lesson (`market_value_check` printing `0/0`) one
   more time: a count of agreements means nothing without the count of what was compared and the
   count of what could not be.

And one thing deliberately **not** designed: a single "data quality" badge or score per stock
that rolls these up. Half the universe carries at least one caveat, so a roll-up would either
light up for everyone or need weights between unlike things - a lease note against a missing
metric against a year-old income statement - and choosing those weights is exactly the
unexplainable number this project keeps declining to publish. The caveats stay separate, each
beside the figure it qualifies.

## Sources

No new literature was read today; the two documents this note leans on were read by the sessions
that cite them, and are cited here second-hand, as such:

- MSCI, *Fundamental Data Methodology*, June 2024, section 1.1.5 ("only upon availability of
  complete information ... or upon dual vendor validation") - quoted in
  `research/2026-10-29-ttm-fills-the-empty-quarter.md` section 5.2. **Not re-read today.**
- SEC Form 10-Q and 10-K filing deadlines for large accelerated filers (40 and 60 days) - the
  arithmetic in `METHODOLOGY_CHANGELOG.md` 2026-11-02 (night), from
  `research/2026-10-05-fundamental-reporting-lag.md`. **Not re-read today.**
- SEC EDGAR `companyfacts` API, `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`,
  queried 2026-11-18 for CIK 910606 (Regency Centers) and CIK 831001 (Citigroup).

Everything else is this project's own published payload and its SEC cache, measured by the five
scripts listed at the top.
