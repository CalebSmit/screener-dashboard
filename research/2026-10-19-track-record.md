# The track record: what eight months of a live ranking can show, and what the panel said it showed

**2026-10-19. `plan/context-layer.md` queue item 4, the last per-signal note**, plus queue item 6's
next piece (turnover and cost). The track record is the one context surface that is not a signal
about a stock. It is the ranking's own outcome: the top 25 by composite, equal-weighted, measured
forward from each monthly run against RSP (the S&P 500 in equal weights) and SPY (the S&P 500 by
size), with the top fifth against the bottom fifth beside it (`track_record.py`).

**What this note concludes, up front.**

1. **A record of this length cannot tell skill from luck, and this one does not try to.** Measured
   day by day, the top 25's lead over RSP is **0.8 standard errors** from zero after 165 trading
   days. Two is the usual minimum. If the lead went on at exactly the same pace and noise, reaching
   two would take **about 4.3 years**, and a smaller true lead takes longer (section 3). "5 of 7 periods ahead of RSP" is what a fair coin does **22.7%** of the time.
   The literature says the same in three different ways: Lo (2002) on the standard error of a
   performance ratio, Bailey & López de Prado (2012) on the minimum track record length, and McLean
   & Pontiff (2016) on why even a real published edge should be expected to shrink.
2. **The panel said more than its arithmetic does, in nine places**, and each is corrected in the
   same commit as this note (section 6). The two that mattered most: it called the record
   "measured forward from each run as it was published, with nothing backfilled" when the February
   snapshots were written on 2026-03-09 from saved run output, and the 2026-08-10 rebalance run was
   made **after that day's close** and entered at that close, a price set before the ranking
   existed. The second is a defect in the record's arithmetic, not only its wording; it is fixed in
   `track_record.py` from the run's own recorded prices, not from an assumed clock.
3. **Turnover and cost (item 6) is built and display-only.** The top 25 traded **31%** of its value
   at each rebalance on average, one way (19% to 53%). No cost is assumed or deducted; the panel
   states the one figure that needs no assumption: a cost of **1.48% on every dollar traded** would
   have erased the lead over RSP (section 7).
4. **The track record is not a candidate for the score, structurally.** It is the score's outcome,
   not an input, and CLAUDE.md rule 4 already forbids reading methodology from a thin return
   series. Nothing here moves a score, a rank or a weight.

Every DOI below was resolved through Crossref before it was written down
(`research/measurements/2026-10-19-track-record-dois.py`). The record's figures are reproducible
offline from the committed snapshots and the data loop's price cache
(`research/measurements/2026-10-19-track-record.py`; copy `data/track/prices.parquet` from the main
checkout into a fresh worktree first). Where a source was read second-hand it says so.

---

## 1. What the panel printed on 2026-10-19

From `generate_dashboard._js_context` (`renderTrack`, `contextMetas`) as committed at `bfd95a9`,
rendered from `data/track_record.json` as the 02:00 run published it:

| Where | Text | Figure |
|---|---|---|
| opening | "How the ranking has actually done since **Feb 20**, measured forward from each run as it was published, with nothing backfilled. It measures the ranking, not a portfolio to follow." | |
| stat | Top 25, total return / "equal-weight, rebalanced monthly" | +11.3% |
| stat | S&P 500 equal-weight / "RSP, the like-for-like benchmark" | +6.7% |
| stat | S&P 500 / "SPY, weighted by size" | +11.6% |
| stat | Top fifth minus bottom fifth / "does the ranking separate stocks?" | +7.7% |
| stat | Periods ahead of RSP / "monthly holding periods" | 5 of 7 |
| legend | "Methodology change (hover)" | |
| caveat | "8 months is far too short ... and the methodology changed during it (the ticks under the chart; 28 changes)." | |
| caveat | "There were no runs for a stretch before Jul 29; the previous list was held through it." | |
| caveat | "...usually companies taken private or acquired (EA, HOLX). Leaving them out can move a basket's return either way; for a company that was acquired it usually understates it, because takeovers come at a premium." | |
| caveat | "...companies that stopped trading are priced from the screener's own record of their price at each run, until they stopped (AVB, CTRA, EA, HOLX), on price alone without dividends." | |
| caveat | "Equal weights, entry at the close of the first trading day after each run, dividends included except where said, no trading costs or taxes. This record is not used to set the methodology." | |
| tile | "Top 25, vs +6.7% equal-weight, +11.6% S&P 500 · 8-month record" | +11.3% |

The section line, "Top 25 +11.3% vs equal-weight S&P 500 +6.7%, S&P 500 +11.6% since Feb 20", was
fixed on 2026-10-16 to quote both benchmarks and is fair as written.

---

## 2. What the record is, and what it is not

**It is a model record.** No money was invested and no trade was placed. The Global Investment
Performance Standards have a category for exactly this. GIPS 2020 for Firms (CFA Institute, read
from the standard's PDF, <https://www.gipsstandards.org/wp-content/uploads/2021/03/2020_gips_standards_firms.pdf>;
no DOI) calls it *theoretical performance*, "not based on the performance of actual assets", and
4.C.48 requires a firm that shows it to say so, to describe the method "including if it is based on
model performance, backtested performance, or hypothetical performance", and to "disclose whether
the theoretical performance reflects the deduction of ... transaction costs". 1.A.27: "The firm
must not link actual performance to historical theoretical performance." The panel said "it
measures the ranking, not a portfolio to follow", which is true and is not the same statement; it
never said no money was invested. It does now.

**It is out of sample in the selection, with one qualification on provenance.** Each basket is the
top of a ranking that existed before the returns it is measured on, so there is no look-ahead in
*what was picked*. But "as it was published, with nothing backfilled" is not what the history shows:

- The snapshots for 2026-02-20 to 2026-03-01 were first committed on **2026-03-09** (`9de0c98`, "Add
  self-improving engine"), the same commit that introduced the snapshot store and
  `improvement_engine.backfill_from_existing_runs`. That function builds a snapshot from each run's
  saved `05_final_scored.parquet`, dated by the run's own `meta.json` start time. So the rankings
  are each run's own output, not a re-scoring, but they were *written down* three weeks later.
- The public page began on **2026-02-23** (`852e238`, "Add index.html for GitHub Pages"). The
  record's first basket, 2026-02-20, was never on a public page.
- The 2026-04-14 snapshot the record uses (`f9b830b42916`) was committed on 2026-07-28 (`8ac9354`).

None of this changes a number. It changes what the opening sentence may claim. The corrected
sentence says the record is "measured forward from the list each run recorded on its date", which
the code makes true.

**It is not evidence for the methodology.** CLAUDE.md rule 4 and the module's own docstring already
say so; the panel says so; this note adds the arithmetic for why (section 3).

---

## 3. Why a record this short cannot discriminate skill

### 3a. The arithmetic of a standard error

A track record's average excess return is an estimate, and its standard error falls only with the
square root of the record's length. Lo (2002) derives it for the Sharpe ratio: under independent,
identically distributed returns the estimator's asymptotic variance is `V = 1 + SR²/2` per
observation (Table 1's note), so its standard error is `sqrt((1 + SR²/2) / T)`. Lo's Table 1 gives
**0.354** for a true per-period Sharpe ratio of 1.00 measured over 12 observations, and 0.158 over
60. His second point is the one that bites live records: the IID assumption "is extremely
restrictive and often violated by financial data", and with serial correlation the standard error
changes; his abstract reports annual Sharpe ratios for hedge funds overstated by as much as 65% for
that reason. *Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal 58(4),
36-52. doi:10.2469/faj.v58.n4.2453* (read from the article PDF).

Bailey & López de Prado (2012) turn the same distribution round and ask how long a record must be
before an observed Sharpe ratio is above a threshold with a given confidence, which they call the
**Minimum Track Record Length**. Their worked example: "a 2.73 years track record is required for
an annualized Sharpe of 2 to be considered greater than 1 at a 95% confidence level", on daily IID
Normal returns, and longer with negative skew or fat tails. *Bailey, D. H. & López de Prado, M.
(2012). The Sharpe Ratio Efficient Frontier. Journal of Risk 15(2), 3-44.
doi:10.21314/JOR.2012.255* (read from the authors' PDF of the forthcoming version, section 5).

### 3b. Measured on this record

On the corrected record (section 6), 2026-02-20 to 2026-10-16, the daily difference between the top
25's return and RSP's:

| | |
|---|---|
| trading days | **165** |
| mean daily difference | 3.41 basis points |
| standard deviation of the daily difference | 56.4 basis points (9.0% scaled by √252; a volatility scaling, not a return) |
| mean over its standard error, `t` | **0.78** |
| trading days to `t` = 2 at the same mean and noise | 1,094, **4.3 years** |

That last line is the most optimistic reading the record allows: it assumes the lead seen so far
is real and continues. For scale, by the same arithmetic a lead with an information ratio of 1.0
reaches two standard errors in 4 years, 0.5 in 16 years and 0.25 in 64. The page now prints the
first three rows' substance: "Measured day by day against RSP over 165 trading days, the top 25's
lead is 0.8 standard errors from zero; two is the usual minimum for telling skill from luck, and
even if the lead went on at the same pace it would take about 4.3 years of record to get there."
The `t` treats daily differences as independent, which Lo shows is a simplification, so it is a
guide to scale, not a test result (claim `context.track_luck`, caveat).

**The period count.** 5 of 7 periods ahead of RSP is what a fair coin does at least as often
22.7% of the time (29/128). The panel called them "monthly holding periods"; they ran 14, 35, 105,
12, 20, 29 and 15 days, because there was no run between 2026-04-14 and 2026-07-28 and the record
holds the previous list through a gap. The stat now reads "a coin toss does at least this well 23%
of the time".

**The spread.** Top fifth minus bottom fifth was positive in 4 of 7 periods, mean 1.24 points,
`t` across periods 0.96.

**One day moves the headline by more than the lead.** As a sensitivity, not a proposal: entering
every rebalance one trading day later moves the top 25 from **+12.7% to +17.2%**, RSP from +6.5% to
+7.2% and SPY from **+11.9% to +9.2%**. A one-day shift in the entry rule takes the top 25's gap to SPY
from +0.8 points to +8.0, and under the rule published until today it was −0.3. The sign of the
comparison a reader looks at first depends on a calendar convention.

### 3c. Luck in a cross-section, and the decay of real edges

Fama & French (2010) bootstrap the cross-section of US equity mutual funds and find that "few funds
produce benchmark-adjusted expected returns sufficient to cover their costs", with evidence of
genuine skill only "in the extreme tails" once costs are added back. Their method is the reason a
single good record is weak evidence: across many records, some look good by chance. *Fama, E. F. &
French, K. R. (2010). Luck versus Skill in the Cross-Section of Mutual Fund Returns. Journal of
Finance 65(5), 1915-1947. doi:10.1111/j.1540-6261.2010.01598.x* (abstract, via Crossref).

McLean & Pontiff (2016) study 97 published predictors: "Portfolio returns are 26% lower
out-of-sample and 58% lower post-publication", the out-of-sample drop being "an upper bound
estimate of data mining effects" and the further 32% attributed to publication-informed trading.
The screener's categories are built on published effects. So the *expected* live edge, if the
methodology is right, is a fraction of the papers' figures, which makes a short live record even
less able to show it. *McLean, R. D. & Pontiff, J. (2016). Does Academic Research Destroy Stock
Return Predictability? Journal of Finance 71(1), 5-32. doi:10.1111/jofi.12365* (abstract, via
Crossref). This agrees with the project's own longer test: the research engine finds the ranking's
IC about zero over 2012-2026 and that the design can only see an IC difference of about 0.04-0.07
(`research/2026-10-15-research-engine-first-batch.md`), which the panel already links.

---

## 4. How serious managers present a short record

GIPS 2020 for Firms, provisions read from the standard:

| Provision | Says | The panel |
|---|---|---|
| 2.A.12 | "Returns for periods of less than one year must not be annualized." | Never did; now says so while the record is under a year, and a test fails on any annual rate |
| 2.A.13 | "All returns must be calculated after the deduction of transaction costs incurred during the period." | A model record cannot know its costs; it now states its turnover and the break-even cost instead (section 7) |
| 4.C.18 | if estimated costs are used, disclose "the estimated transaction costs used and how they were determined" | Why no estimate is shown: no verified basis this session |
| 4.A.1.a | at least five years of performance, or since inception if shorter, building to ten | The panel shows since inception and says it is far too short |
| 4.A.1.j | the three-year annualized ex-post standard deviation, where 36 monthly returns exist | Not shown; there are 8 months |
| 4.C.48 | theoretical performance labelled, its method described, costs disclosed | Now: "No money was invested: this is a model record" |
| 1.A.27 | actual performance must not be linked to theoretical | Not applicable; no actual performance exists |

The practice these encode: show the record since inception without extrapolating it, show the
benchmark beside it on the same basis, state what is deducted, and do not draw a risk statistic
from fewer observations than it needs. The panel met most of this already. It did not say the
record was theoretical, and it described the benchmark as "like-for-like" when RSP's price is after
the fund's own running costs and the top 25 bears none.

---

## 5. The two picks the panel used to explain with a takeover premium

The caveat read: "for a company that was acquired it usually understates it, because takeovers
come at a premium", beside EA and HOLX, the two top-fifth picks still unpriced on the day their
February basket formed (the snapshots carry no price before 2026-03-06).

A premium is missed only if it is paid while the basket holds the stock. The screener's own
recorded prices for both are flat: EA **197.55** on 2026-03-06 to 208.87 on 2026-07-28 (a 5.7%
range over three runs), HOLX **75.66** to 75.16 (0.7%). That is how a stock trading near an agreed
takeover price looks. The announcements themselves predate the window; that was not re-checked from
a primary source this session, and the corrected sentence does not depend on it. It now says what
the payload makes true: "Leaving them out can move that basket's return either way; they are 2 of
the 703 picks in the top-fifth baskets."

---

## 6. Every sentence, against the code

| # | Sentence (before) | Verdict | Now |
|---|---|---|---|
| 1 | "measured forward from each run as it was published, with nothing backfilled" | **Overstated** (section 2) | "measured forward from the list each run recorded on its date. No money was invested: this is a model record of the ranking, not the result of real trades and not a portfolio to follow." |
| 2 | "entry at the close of the first trading day after each run" | **False for 2026-08-10**; the arithmetic is fixed | see below |
| 3 | "equal-weight, rebalanced monthly" | Loose: periods of 12 to 105 days | "equal weights, before trading costs" (the rebalance rule is in the claim and the table) |
| 4 | "RSP, the like-for-like benchmark" | **Overstated** (costs asymmetric) | "RSP: the same index in equal weights, after the fund's own costs" |
| 5 | Top fifth minus bottom fifth "+7.7%" / "does the ranking separate stocks?" | **Wrong unit**; the sub-label implies the figure answers it | "+9.1pp" / "difference in total return, percentage points" |
| 6 | "monthly holding periods" | **False** (14 to 105 days) | "a coin toss does at least this well 23% of the time" |
| 7 | "the methodology changed during it (... 28 changes)" and "Methodology change (hover)" | **Overstated**: `methodology_dates()` counts dated changelog headings, and several (2026-08-11, 08-20 process directives; 09-08 the AI chat's removal) changed no score | "The methodology changelog has entries on 28 dates inside this window; not every entry changed how stocks are scored." Legend: "Methodology changelog entry" |
| 8 | "There were no runs for a stretch before Jul 29" | True today; could become false (the gap is in *comparable* runs) | "no run the record could use" |
| 9 | "...usually understates it, because takeovers come at a premium" | **Unsupported here** (section 5) | sized by the payload's own counts |
| 10 | "priced ... until they stopped" | Loose: the record ends at the last run that recorded the stock, carried between runs | "carried from one run to the next and ending at the last run that recorded them" |
| 11 | "N months" in the panel (`days / 30`) and the tile (`days / 30.4`) | Two divisors; could disagree by a month | one `trMonths()` for both |
| 12 | "no trading costs or taxes" | True, and the only cost statement | kept, plus turnover and break-even (section 7) |
| 13 | "This record is not used to set the methodology." | True (rule 4) | kept |
| 14 | the section line and the tile's benchmarks | Fair (fixed 2026-10-16) | kept |

**The entry defect (row 2).** `track_record.build` entered each list at the close of the first
trading day **on or after** the run's date. The docstring's reason is right for the 02:00 data
loop, which scores on the previous close. But a snapshot records its date and not its time, and
not every rebalance run was at 02:00. The 2026-08-10 snapshot was committed at 19:22 CDT (`710621b`,
"full refresh - restore prices, analyst targets and correct rankings"), after the close, and its
`price_at_scoring` is **that day's** close for 47% of stocks within 0.2%, against 10% for the
previous day's. For every 02:00 run the pattern is reversed (2026-09-01: 67% match the previous
close, 9% the same day's; 2026-10-01: 95% and 12%). `test_entry_is_never_before_the_run` passed
throughout, because it compared dates, and a date cannot tell a 02:00 run from a 19:00 one: the
layer's standing lesson, a claim whose check tests the plumbing instead of the assertion.

The fix reads the time of day from the run's own prices (`track_record.runs_after_the_close`): a run
is after the close when more of its recorded prices sit within 0.2% of its date's close than of the
previous trading day's, and at least a quarter do. Such a run enters at the next close and the
previous list is held until then. Two rebalance snapshots record no prices (2026-02-20 and
2026-04-14), so they cannot be tested; they keep their own date's close, and the page names them.

**Why not simply enter every run a day later?** It was measured and rejected. It would be after
every run by construction, but it shifts five periods whose runs demonstrably came before the
close, and it moves the headline from +12.7% to +17.2% (section 3b). A correction that improves the
number by more than the defect it fixes is the kind a sceptical reader is right to distrust. The
evidence-read rule changes exactly the period the evidence says is wrong.

**Effect at the next full run**, measured on the cached prices (the published file is rewritten by
the data loop, never by hand):

| | published 2026-10-19 | corrected |
|---|---|---|
| Top 25 | +11.30% | **+12.66%** |
| RSP | +6.69% | +6.52% |
| SPY | +11.58% | +11.91% |
| Top fifth minus bottom fifth | +7.66pp | +9.08pp |
| Periods ahead of RSP | 5 of 7 | 5 of 7 |

Two periods move: 2026-07-29 now runs to 2026-08-10 (−0.3% to +0.8%) and 2026-08-10 enters on
08-11. **No score, rank or weight moves**; `dashboard_data.js` is byte-identical.

---

## 7. Turnover and cost (queue item 6)

**What is built.** At each rebalance, `track_record.one_way_turnover` measures half the sum of the
absolute weight changes from the top 25 *as it had drifted* to equal weights on the new list, which
is the share of the basket sold and equals the share bought. The first purchase is not counted.
Each period's figure is in the holding-period table ("Traded"); the stat strip shows the mean and
range; `break_even_cost` solves `(1 + top) x prod(1 - 2c x TO_k) = 1 + RSP` for the cost per dollar
traded that would erase the lead.

| rebalance | 03-09 | 04-14 | 07-29 | 08-10 | 09-01 | 10-01 |
|---|---|---|---|---|---|---|
| one-way turnover | 53% | 23% | 39% | 27% | 19% | 27% |

Mean **31%**; traded in all, both ways, 375% of the basket's value; break-even **1.48%** per dollar
traded.

**Why no estimated cost.** GIPS 2.A.13 wants real returns after costs, and 4.C.18 wants an
estimate's basis disclosed. The two serious sources on costs for a strategy like this were not
verifiable to the figure this session. Frazzini, Israel & Moskowitz ("Trading Costs", 2018,
doi:10.2139/ssrn.3229719) measure $1.7 trillion of a large manager's live trades in 21 developed
markets over 19 years and report actual costs "an order of magnitude smaller than previous studies
suggest". That is read second-hand from the SSRN abstract as quoted by search results; the PDF
returned 403. Novy-Marx & Velikov (2016) find that most anomalies with "one-sided monthly turnover
lower than 50%" keep significant net spreads after costs, and few above it do (NBER w20721 abstract;
*A Taxonomy of Anomalies and Their Trading Costs. Review of Financial Studies 29(1), 104-147.
doi:10.1093/rfs/hhv063*). The top 25's 31% a rebalance at a roughly monthly cadence sits under that
line, which is consistent with the 50% bound this project already uses for the hold band (CLAUDE.md
priority 5). Neither source gives a number the panel could print as *this* basket's cost, so it
prints the break-even instead, which needs no assumption to be true.

**Limits.** A pick held at its last price after its prices stopped (an acquisition cash-out) is
counted as sold at the next rebalance, though a cash-out costs nothing to trade; that slightly
overstates turnover. Taxes are not modelled. Claim `context.track_turnover`.

---

## 8. The benchmark question (item 6's remainder) - left open, with the reasoning

The plan asks whether RSP is the right benchmark "after cap-weighted leadership". The top 25 is an
equal-weighted selection from the S&P 500, so RSP matches its weighting and universe, and the
difference between the two is selection alone. SPY is what most readers could hold instead, so it
answers a different question: was owning the ranking's picks better than owning the index? The
panel shows both, always (the 2026-10-16 fix). Over this window the two disagree by more than the
lead itself (RSP +6.5%, SPY +11.9%), which is the cap-weighted leadership the plan names: the
large-company tilt of SPY is a bet the equal-weighted top 25 does not make. No change: dropping
either would hide one of the two questions. What would change it is a published construction of
the right benchmark for a sector-capped, equal-weighted large-company selection, which this note
did not find and does not invent.

---

## 9. Candidate for the score?

No, and that is structural rather than pending. The record is the ranking's outcome, measured on
the ranking's own picks; feeding it back would be choosing weights from a return series, which
CLAUDE.md rule 4 forbids until the evidence base is far deeper, and which is what the improvement
engine, not the dashboard, is for. `context_eval.py` evaluates *signals* across stocks; the track
record has no cross-section to evaluate.

---

## 10. What shipped with this note

- `track_record.py`: `runs_after_the_close`, `untimed_runs`, `entry_day`; `one_way_turnover`,
  `break_even_cost`; `coin_toss_chance`, `excess_record`. The payload's `track` block gains
  `after_close`, `untimed`, `turnover`, `beat_rsp_chance`, `excess` and a per-period `turnover`, at
  the next full run.
- `generate_dashboard.py`: the corrected sentences, a "Traded at each rebalance" stat, a "Traded"
  column, `trMonths()` and `cPp()`. The page regenerated presentation-only; both payload files
  byte-identical (sha256 asserted).
- `claims.py`: `context.track_record` rewritten; `context.track_luck` and `context.track_turnover`
  added; two `FORBIDDEN` patterns for the sentences corrected today.
- `tests/test_track_record_claims.py` (in the data loop's claims gate).
- No `METHODOLOGY_CHANGELOG.md` entry: nothing here changes how a stock is scored, and a dated
  heading there becomes a tick on this very chart, which row 7 above says should mean a change to
  the ranking.

**Open.** (a) The two untimed runs could be timed from the runs' `meta.json` if a copy survives
anywhere; the data loop prunes `runs/`. (b) A verified cost basis for large-cap US trades, from a
primary source, would let the panel state an estimated net figure under GIPS 4.C.18. (c) Recording
the run's timestamp in each snapshot would make section 6's inference unnecessary for future runs.
