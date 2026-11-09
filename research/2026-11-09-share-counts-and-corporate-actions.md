# The share count behind a market value, and what a screen should do when it moves

**2026-11-09** (Monday, research; the focus was deferred to the stalled data loop first and this
note is what the diagnosis turned into)

**One sentence:** a scored stock's market value can move 25–35% overnight with its price
unchanged, because the share count behind it changed; this moved FOXA from rank **34 to 60** on
2026-11-09; it happens about **once every three runs**; index providers do exactly the same thing
on purpose and say so; and the thing this screener is missing is not the new share count but any
record that the *earnings* behind the multiple have not caught up with it.

---

## 1. What happened, and how it was found

The 02:00 data run of 2026-11-09 was refused by the published-claims gate and the live site kept
the page of 2026-11-06 (`logs/datarun-2026-11-09_020001.log`). Two tests failed. One is a
measurement defect in the Factor Lab's phone budget and is written up in its commit; the other is
this note's subject.

`tests/test_share_classes_page.py::test_the_price_gap_is_the_only_valuation_difference` asserted
that FOXA's and FOX's `earnings_yield` differ by exactly the ratio of their prices. It read
**0.6650 against 0.8929**.

The project's rule for a data-dependent gate is to ask first whether the claim is false
(`CLAUDE.md`, settled row "data->red"). So, from the filings rather than from inference:

| Source | Fact |
|---|---|
| Fox 10-Q, filed 2026-11-02, cover page | Class A (FOXA) **201,552,650**; Class B (FOX) **220,248,764**; total **421,801,414** |
| Fox 8-K, filed 2026-11-06 ("the Closing Date") | Fox **completed** its acquisition of Roku on **2026-11-06**, having entered the Merger Agreement 2026-06-14; Class A stock issued as consideration |
| Fox S-8 and S-8 POS, both filed 2026-11-06 | the post-effective amendment on Form S-8 to the Form S-4 declared effective 2026-09-01 — the registration housekeeping of a closed deal |
| Fox 10-Q, filed 2026-11-02 | weighted-average basic shares, quarter to 2026-09-30: **421,000,000** — the same total, independently |

And what the source published the next session:

| | FOXA before (runs to 2026-11-06 18:34) | FOXA on 2026-11-09 |
|---|---|---|
| `currentPrice` | 66.64 | 66.64 (**unchanged** — no session had closed) |
| `sharesOutstanding` | 201,552,650 | 346,052,650 (**+144,500,000**) |
| `marketCap` | 28,108,845,056 | 37,738,328,064 (**+34.3%**) |
| implied total shares (`marketCap / currentPrice`) | 421,801,396 | 566,301,442 |

`346,052,650 + 220,248,764 = 566,301,414`. So the source added the newly issued Class A shares to
FOXA's count and to its market value, and **left FOX's market value on the pre-merger total** —
`25,097,183,232 / 59.50 = 421,801,399`. The two listed lines of one company were on two share
counts, which is why the price ratio stopped being the market-value ratio.

**Nothing published was wrong.** `02_outliers_flagged` keeps every fetched line and the second
line is dropped before the percentiles (`run_screener.py`, the "One company, one ranked line"
step), so FOX's `earnings_yield` is never scored, ranked or published; and the page's sentence is
about the two *prices*, both of which were current. The test was asserting a property of a
discarded intermediate. It is rewritten to assert the claim instead — and the premise with it —
in commit `9817403`.

## 2. What it cost, measured

`research/measurements/2026-11-09-post-acquisition-market-value.py` (committed; reads `runs/`,
writes nothing).

**FOXA, across the two runs either side of the close.** Price identical, market value +34.3%:

| metric | 2026-11-06 18:34 | 2026-11-09 02:00 | change |
|---|---|---|---|
| `earnings_yield` | 0.07218 | 0.05376 | **−25.5%** |
| `fcf_yield` | 0.07035 | 0.05407 | −23.1% |
| `ev_ebitda` | 6.993 | 9.099 | +30.1% |
| `ev_sales` | 1.745 | 2.271 | +30.1% |
| `size_log_mcap` | −24.059 | −24.354 | +1.2% |
| **Composite** | **63.19** | **60.98** | **−2.21** |
| **Rank** | **34** | **60** | **−26 places** |

A 26-place move, on no price change and no new filing, from one input.

**How often the signature occurs.** Over every consecutive pair of the 20 committed runs
(2026-10-16 to 2026-11-09), the stocks whose implied share count moved by more than 5%:

| run pair | stocks | over 5% | which |
|---|---|---|---|
| 13 pairs, 2026-10-16 → 2026-11-03 | 503 | **0** | — |
| 2026-11-03 → 11-04 | 503 | 1 | KMB +84.5% |
| 2026-11-04 → 11-06 | 502/503 | 0 | — |
| 2026-11-06 02:00 → 11-06 18:34 | 503 | 2 | CVNA +34.6%, PCG −26.6% |
| 2026-11-06 → 11-09 | 503 | 3 | CHTR +37.8%, **FOXA +34.3%**, CPT −13.5% |

**6 moves over 5% in 19 run pairs — 0.32 a run.** Thirteen consecutive pairs saw none, so this is
episodic rather than noise.

Three of the six line up with a filing, checked on EDGAR: **KMB** filed an 8-K on 2026-11-04
reporting the "completion of the Exchange Offers" (the move appeared in the 11-04 run); **CHTR**
filed an 8-K dated 2026-11-06 (the move appeared in the 11-09 run); **FOXA** as above. **I did not
establish the other three** (CVNA, PCG, CPT) — and two of them appeared between two runs on the
*same day*, which points at a source refresh rather than at the company. Marked as unestablished
rather than assumed.

**What saw it: nothing.** That run's `check_run_health` reported every line `ok`, including
`rank continuity ... Spearman 0.997 over 499 names, 2 new in the top 25`. It is right to: one
stock moving 26 places does not dent a 499-name Spearman, the dispersion is preserved because
renormalisation preserves spread, and coverage was 100%. The gate was built for outages and
continuity, and this is neither.

## 3. What professionals do — read, not recalled

### 3.1 The share count moves with the event, not with the filing

**MSCI, *Corporate Events Methodology*, February 2026** (105 pp.; read from the PDF at
`https://www.msci.com/indexes/documents/methodology/0_MSCI_Corporate_Events_Methodology_20260210.pdf`):

- p. 5: "As a general policy, changes resulting from corporate events are implemented in the MSCI
  Equity Indexes **as they occur simultaneously with the event**." The changes listed include
  "changes in number of shares (NOS)".
- §2.6, "**Pro Forma Float Calculation for M&As**" (pp. 16–17) — the post-event float is computed
  *pro forma from the deal terms*. Where the terms are unavailable, "the resulting pro-forma free
  float is calculated as a weighted average of the free float of the shares of the pre merging
  entities."
- §2.7, p. 17: "for a merger/acquisition that involves **only cash consideration**, the acquirer's
  constituent weighting **does not change**. For any merger/acquisition that involves the
  **issuance of shares** (as all or part of the consideration), post-event constituent weighting
  would take into account the shares issued within in the deal; as determined by the NOS Inflow
  Ratio."
- p. 18: `NOS Inflow Ratio` for an acquisition = `TARGET PERCENTAGE ACQUIRED × (ACQUIRER SHARES
  ISSUED / TARGET_SHARES NEEDED)`.

**S&P Dow Jones Indices, *Equity Indices Policies & Practices*** and the companion *Corporate
Actions: Policies & Practices* (read from the public copies at
`https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-equity-indices-policies-practices.pdf`
and `.../methodology-sp-corporate-actions-policies-practices_20140213.pdf`; **these are older
editions — S&P's own host served a bot wall rather than the current August 2026 file, and that was
not worked around**, so the version caveat stands and the current edition should be re-read before
any of this is relied on):

- pp. 28–29 (and pp. 32–33 of the companion, same text): "M&A activity is implemented **when the
  target company is deleted**."
- Same pages: "Other changes of **5% or more** ... are made **weekly** and are announced after the
  market close on Fridays for implementation after the close of trading the following Friday."

So both providers move the acquirer's share count at the event, from the deal terms. **Yahoo's
behaviour on FOXA is not a data error — it is what an index provider would also have done.** The
figure this project scored on 2026-11-09 is the correct current market value of Fox Corporation.

### 3.2 Both providers set a materiality threshold on a share change

- **S&P: 5%** — changes below it wait for the quarterly share rebalancing; at or above it they go
  in weekly (above).
- **MSCI: ±1%** — from the methodology's own version history, October 2017 updates (p. 93):
  "number of shares updates smaller than ±1% are not implemented at the time of the event but at
  a subsequent Index Review."

This matters for section 5: 5% is not a number I chose, it is the threshold the larger of the two
providers already publishes, and the measurement in section 2 used it before I had read this.

### 3.3 And US practice for two listed classes is exactly this project's design

**S&P DJI, *Equity Indices Policies & Practices*, "Multiple Share Classes"** (p. 29):

> "US. In S&P Dow Jones' US indices (including the S&P BMI US), **each company is represented only
> once. Multiple classes are combined into one class with an adjusted share count. The stock price
> is based on one class, usually the most liquid class, and the share count is based on the total
> shares outstanding.**"

Two things follow.

First, the 2026-10-21 decision to score each company once on its more-traded class is this rule,
and now has the sentence behind it quoted at a page number rather than paraphrased.

Second — and this is the part that explains the failure at its root — **S&P's rule is one share
count applied to one price.** The source instead publishes `total shares × that line's price` on
*each* line, so FOXA and FOX carry two independent estimates of one quantity. While both records
are refreshed together those estimates agree and the ratio between them is the price ratio, which
is what the page says. A corporate action that reaches one record first makes them disagree. The
page's sentence survives (it is about prices), but the invariant a test was using as a proxy for
it does not, and **this is a structural property of the source, not a one-off**: any dual-class
company doing anything to its share count can reproduce it.

## 4. The question this leaves open, which is a methodology question

The share count is right. **The numerator is not yet.**

FOXA's `earnings_yield` on 2026-11-09 is Fox's standalone trailing net income over Fox-plus-Roku's
market value. Both halves are the latest reported figures; they describe different companies. The
same is true of `fcf_yield`, `ev_ebitda`, `ev_sales` and `pb_ratio`, and it will stay true until the
December quarter is filed — roughly **three months** at Fox's reporting cadence, longer before four
clean post-deal quarters exist.

Is the resulting number wrong? A defensible case says no: it is what a buyer pays today over what
the company has actually reported earning, and Roku was loss-making, so a *pro-forma* trailing
earnings figure would put the yield lower still, not higher. The distortion is also arguably
informative — paying with stock when your stock is dear is Shleifer & Vishny's mechanism, and a
mechanically-cheapened or dearer acquirer is a thing the literature has studied.

What is **not** defensible is that the page says none of this. A reader who opens FOXA sees a
Valuation score 26 places worse than three days ago, with every equation on the row reproducing
exactly, and no sentence anywhere telling them the denominator grew by a third on 2026-11-06 and
the numerator has not moved. That is a straight failure of the standing rule that **every number
on the page must be checkable from the page** (owner, 2026-10-06) — the arithmetic checks out and
the *reason* is invisible.

**Three options, not decided here.** This note deliberately does not build any of them; it is one
session's diagnosis and the choice between them wants a reader who did not write it.

1. **Say so, change nothing.** A display-only line on the Data Provenance panel and in the
   drilldown when a stock's implied share count has moved by more than the pre-registered
   threshold since the last accepted run: what it was, what it is, and that the earnings behind
   the multiple predate the change. Cheapest, no scoring change, and it is the option most
   consistent with how this project settled the growth-exposure question on 2026-11-03 (state the
   lean, leave the weight).
2. **Note it in `check_run_health` as a note, not a gate.** Same detection, printed every run and
   surfaced in the morning brief, like `sec_crosscheck`'s `ATTENTION -` line. This is the option I
   would take first, because it is the one that produces the counts needed to decide between the
   other two, and the `sec_crosscheck` precedent (promote to a gate only after weeks of counts
   show the normal range) is exactly right here.
3. **Exclude the affected metrics for the stub window.** Treat the market-value-based Valuation
   metrics as missing while the earnings predate the share change, and let the existing
   renormalisation handle it. **I am sceptical of this one** and the reason is measurable: it would
   drop a stock's largest Valuation inputs for a quarter on the strength of a share-count
   heuristic, and `MIN_CATEGORY_COVERAGE = 0.90` plus the coverage discount means the consequences
   reach well past the one metric. It also throws away the correct current market value, which
   section 3 shows is what every provider uses.

**Proposal status: DECIDED 2026-11-09 (13:00 session, which did not write this note) - option 2
ACCEPTED in substance and REFUSED as specified; option 3 REFUSED on a measured reason; option 1
left open with a pre-registration it did not have.** See section 8. In one line: the detector
this note proposes building already exists, anchored to the *filings* rather than to the previous
run - and it had been comparing **nothing** since 2026-11-08.

## 5. Pre-registered, before the numbers are in

Written now so it cannot be fitted later.

1. **The detector.** `implied_shares = marketCap / currentPrice`, compared with the last accepted
   run's. Not `sharesOutstanding`, because that field is per *class* and the market value is not —
   which is the whole of section 1.
2. **The threshold is 5%**, because it is S&P DJI's published threshold for an off-cycle share
   change (§3.2) and not because of anything in this project's data. Do not tune it to the counts.
   MSCI's ±1% is the alternative and is *stricter*; if 5% turns out to be noisy the move is to 1%,
   never upward.
3. **The normal range must be established before anything becomes a gate.** The precedent is
   `sec_crosscheck` (priority 0.14): run it as a printed note, and only promote a field to a gate
   after several weeks of counts. Today's figure is 0.32 a run over 19 pairs, from 20 runs — far
   too thin.
4. **Two runs on the same day do not count as an observation.** Two of the six moves found appeared
   between the 02:00 and 18:34 runs of 2026-11-06, and a share count that changes inside one
   session is a source refresh, not a corporate action. Count per *run date*, the rule this project
   already applies to its evidence files (priority 0.6).
5. **The void condition.** If, over four weeks, the detector fires on more than **3 stocks a run
   date** on average, it is measuring source churn rather than corporate actions and the construct
   is wrong — stop and re-derive it, do not raise the threshold to quieten it.
6. **Nothing here may change a weight, a metric or a score** without its own changelog entry. The
   whole of options 1 and 2 is display and reporting.

## 6. The literature, honestly: queued, not read

The practitioner half of this note was read from the PDFs. **The academic half was not, and is
marked accordingly rather than cited from memory** (the convention
`research/2026-10-20-size-weight.md` introduced). These are the specific locators to fetch, and the
question each bears on — whether a mechanically re-priced acquirer is only a measurement problem or
also a signal:

- Loughran, T. & Vijh, A. M. (1997), "Do Long-Term Shareholders Benefit from Corporate
  Acquisitions?", *Journal of Finance* 52(5), 1765–1790. The method-of-payment result — stock
  mergers versus cash tender offers. **— not read**
- Rau, P. R. & Vermaelen, T. (1998), "Glamour, value and the post-acquisition performance of
  acquiring firms", *Journal of Financial Economics* 49(2), 223–253. Whether the acquirer's own
  book-to-market conditions post-deal returns — directly about the valuation ratio this note is
  about. **— not read**
- Shleifer, A. & Vishny, R. W. (2003), "Stock market driven acquisitions", *Journal of Financial
  Economics* 70(3), 295–311. The theory that overvalued firms buy with stock, which is why a
  stock-funded close is not an accounting accident. **— not read**

**Do not quote an effect size from any of the three until it has been read from the paper.** The
one claim section 4 makes on their strength is hedged to "a thing the literature has studied",
which is as far as an unread citation may be taken. A session that fetches them turns section 4's
first option into a researched answer rather than a choice of three.

## 7. What this note does and does not establish

**Established, from sources read today:**

- Fox completed the Roku acquisition on 2026-11-06 and issued Class A stock (8-K; S-8 and S-8 POS
  the same day; share counts off the 10-Q cover of 2026-11-02).
- FOXA's market value rose 34.3% with its price unchanged, moving four scored metrics by 23–30% and
  its rank by 26 places (measured, committed script).
- The signature occurs 6 times in 19 run pairs, 0 of them in thirteen consecutive pairs, and
  nothing in the pipeline reports it.
- Both MSCI and S&P DJI move an acquirer's share count at the event and pro forma from the deal
  terms, and both publish a materiality threshold (5% and ±1%); S&P's US dual-class rule is one
  line, the most liquid price, and the company's total share count.
- The failing test was checking a discarded intermediate, and the published claim was not false.

**Not established:**

- Whether three of the six share-count moves were corporate actions at all. **Section 8 adds one
  observation on this** and does not close it.
- Whether scoring a post-event market value against pre-event earnings helps or hurts a ranking —
  that needs the literature in §6 and, eventually, this project's own forward returns, which at
  the `1m` horizon accrue about one independent observation a month.
- Which of §4's three options is right. That is the open proposal.

---

## 8. The review (2026-11-09, 13:00 session - not the session that wrote this note)

Measurements: `research/measurements/2026-11-09-share-count-detector-review.py`, committed, which
reads `runs/*/`, the SEC companyfacts cache and `origin/main`, and writes nothing.

### 8.1 The finding that decides it: the detector already exists, and it was broken

Section 4 asks for a detector. The project has had one since **2026-10-14**, and this note does
not mention it. `sec_crosscheck.market_value_check` computes `marketCap / price` - this note's own
`implied_shares`, the same expression - and compares it with the **latest filed**
weighted-average share count. `check_run_health.check_inputs_against_filings` calls it every run
and prints the count within 5% and the count over 10%, with an `ATTENTION -` line above 60
stocks. That is priority 0.14's own machinery, and section 5's rule 3 names `sec_crosscheck` as
the precedent to copy without noticing it is the same check.

**It had been comparing nothing.** From the run logs:

| run | the line it printed |
|---|---|
| 2026-11-05, 11-06 02:00 | `market value / price vs filed shares 446/485 within 5% (21 over 10%)` |
| 2026-11-06 18:34 | `444/485 within 5% (19 over 10%)` |
| **2026-11-09 02:00** | **`0/0 within 5% (0 over 10%)`** |

`financials` moved into `dashboard_detail.js` on **2026-11-08** (the `stock_detail` shard's second
slice). `market_value_check` reads `stock_detail[t]["financials"]["market_cap"]`;
`check_run_health.load_payload` read `dashboard_data.js` alone. So on the first run after that
commit the check skipped every stock, and **an ATTENTION line that fires above 60 can never be
reached from zero**. Measured on the 2026-11-09 run's own artifacts: **0 compared from
`dashboard_data.js`, 485 with the shards folded back, 20 over 10%** - FOXA among them, at ratio
**1.3109**, the stock this note is about.

So "what saw it: nothing" (section 2) is true of that morning, and true for the wrong reason. The
evidence to raise FOXA was already being computed; the gate reading it had stopped being able to.

**Why nothing noticed.** `tests/test_sec_crosscheck.py` builds its payload by hand with
`financials` inline, so it reports the same thing whatever the generator does with the field -
CLAUDE.md rule 10's lesson (assert against the generator's output, not an artifact) one level
down, in a gate rather than a claim. And the rule for what may move into the shard is written in
terms of the **page's** read sites, which is true of `financials` and silent about the payload's
readers that are not the page.

### 8.2 Event or state - the measurement that refuses the proposed detector

Section 5 rule 1 pre-registers a **run-to-run** delta over 5%. But the question section 4 poses is
a **state**: the numerator predates the denominator, for about three months. A run-to-run delta is
an **event** - it fires the morning the count moves and is silent for the whole window the
distortion lasts.

Measured on **KMB**, whose implied count moved +84.5% between the 11-03 and 11-04 runs (8-K of
2026-11-04, completion of the Exchange Offers):

| run date | run-to-run >5% (the proposed detector) | over 10% from the filed count (the existing one) |
|---|---|---|
| 2026-11-04 | KMB | KMB |
| 2026-11-05 | - | KMB |
| 2026-11-06 02:00 | - | KMB |
| 2026-11-06 18:34 | - | KMB |
| 2026-11-09 | FOXA, CHTR | KMB, FOXA, CHTR |

**One run against five, and still counting.** The filings-anchored comparison is also the only one
that can see a stock whose count was already stale before the project started watching, and the
only one whose anchor is the quantity the distortion is *about* - the share count in the filing
the earnings came from.

So option 2 is **accepted in substance and refused as specified**: the detection is right, the
anchor is wrong, and nothing new needed building. What was missing is the **names**.

### 8.3 Named by what is new, not by how far - measured, because the obvious version fails

The ordinary note printed a count. A session reading the brief cannot act on a count. The obvious
fix is to name the stocks furthest from their filed count, and it does not work:

- the 8 furthest on the 2026-11-09 run are **MCD, SKYD, IBKR, TKO, CVNA, VMRK, WAT, MNST** - the
  standing structure recorded on 2026-10-14 (partnership units, splits after the filed quarter,
  two filers' own scale errors);
- **FOXA is 13th of 485.** A magnitude list of eight would not have named it.

Against the page the site is serving, the set is exactly the news: **CHTR and FOXA new, 18
standing, CPT gone.** That is what shipped - `check_share_count_moves`, a note plus an
`ATTENTION -` line, never a failure. On the 11-09 run it reads:

> share counts behind the market values: 20 over 10% from the filed count, 2 of them new since
> the published page, 1 no longer (CPT)
>
> ATTENTION - 2 stock(s) now carry a market value implying a share count over 10% from the one
> their filings report, which the published page does not: CHTR +47.7%, FOXA +31.1%.

Base rate, from the 20 runs on file: new entrants on **3 of 16 run dates**, at most 2 on any one.
The note's section 5.5 void condition (>3 a run date) is in the code as `SHARE_COUNT_MOVE_VOID`
and fires as a second ATTENTION line, worded so that the response is to re-derive the detector
rather than raise the line.

**One observation on section 7's open question.** CPT was over 10% on the published page and is
not on the 11-09 run - its gap **closed**. A completed acquisition does not reverse. That is
evidence, not proof, that CPT was the source refreshing rather than the company doing something,
which is what this note suspected and marked unestablished. It also only shows up in a state
detector: a reversion is invisible to a one-shot event.

### 8.4 Option 3 refused, on a reason this note could not have

Section 4 is sceptical of option 3 (drop the market-value-based Valuation metrics for the stub
window) on blast radius, and is right. The measurement adds a harder reason: **the detector cannot
tell a corporate action from a partnership-unit structure.** 20 of 485 stocks sit over 10% on an
ordinary run and 18 of them are standing, so the exclusion would not fire for a quarter after a
deal - it would fire for **IBKR, TKO, CVNA, MCD and WAT permanently**, on the days their filings
and the source simply disagree about what a share is. Dropping a stock's largest Valuation inputs
for ever is not a stub-window rule.

### 8.5 Option 1 left open, and better founded than it was

Option 1 - a sentence on the Data Provenance panel and in the drilldown - is **not refused**, and
8.2 makes it buildable in a way it was not: a sentence must be true for the whole window a reader
might open the stock in, and only the filings-anchored **state** is. Two conditions are
pre-registered here for whoever builds it, both from 8.3 and 8.4:

1. **It may not say "changed" without a basis.** The state is "the market value implies a share
   count N% from the one this company's latest filing reports", which is checkable from the page
   (`stock_detail[t].xchk` is already that panel). "It moved on 2026-11-06" is a different claim
   and needs the run-to-run comparison beside it.
2. **It needs the discriminator 8.4 shows is missing**, or it will print a share-count sentence on
   IBKR, TKO, CVNA, MCD and WAT every day for ever, which teaches a reader to ignore it. A
   candidate the review did not test: whether the gap is **new** (which is what shipped in the
   health note) and whether the stock has a recent 8-K. Neither is on the page today.

Until then the state is reported where a session reads it and not where a reader does, which is
the `sec_crosscheck` precedent applied as 0.14 wrote it: a printed note first, a surface once its
normal range is known.

### 8.6 What the review did not do

- **It did not change the threshold.** The shipped check uses `sec_crosscheck.FAR` - the existing
  10% line, one literal for one quantity - and **not** this note's 5%. The 5% in section 5 rule 2
  is S&P DJI's threshold for an off-cycle share change, which is the right number for a
  *run-to-run* move and is not the number for a distance from a filed count. If the run-to-run
  detector is ever built, 5% stands as pre-registered.
- **It did not read the three papers in section 6.** They are still marked `- not read` and the
  question they bear on - whether a mechanically re-priced acquirer is a signal as well as a
  measurement problem - is untouched. It remains a Monday.
- **No weight, metric, threshold, score or published sentence moved.** The diff is the health
  gate, two test modules, a comment in the generator and this section.
