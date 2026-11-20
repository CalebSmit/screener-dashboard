# Owner focus - item histories moved out of OWNER_FOCUS.md, 2026-11-19

Moved verbatim on 2026-11-19 (owner-run, the fresh-eyes review's decision 5): the Open section had
reached 1,170 lines for six items, and every session reads it first. Each open item in
OWNER_FOCUS.md now keeps its request, its status and its next step; the history is here.

## The 2026-10-20 item - CLOSED 2026-11-19 (all seven sub-items done or argued down)

- **2026-10-20 - Fine-tune and test what the 2026-10-20 night built.** Owner: *"do as much work as
  you can tonight before the 2 am session please! Then have the sessions fine tune things and test
  things."* The night shipped (owner-run, tags `good/2026-10-20-owner-1..`): the overfitting check on
  the research page, the engine calibration test, Factor Lab experiment 15 (Jensen, Kelly & Pedersen
  2023), the Size and bank-overlap research notes, the context layer's close-out (0.11), and the
  measurements listed in `NIGHTLY_LOG.md` 2026-10-20 (owner-run, night). **The sessions' job now is
  to check that work as a second reader and harden it, not to add surfaces.** In order:
  1. **Decide the OPEN proposals** (the morning brief lists them with their ages): share classes
     (`research/2026-10-19-share-classes.md`), and the bank P/B-ROE pairing - state it on the page
     and guard it with a test (`research/2026-10-20-bank-quality-valuation-overlap.md` section 8).
     The deciding session must not be the one that wrote the note.
     **Share classes: DECIDED 2026-10-21 (06:00) - ACCEPTED, with one change the review measured.**
     Alphabet, Fox and News Corp each had two ranked lines; each company is now scored **once**, on
     its more-traded class, as S&P DJI and FTSE Russell do. The note wanted the line re-picked from
     every run's own trading volume. Measured over 89 month-ends, Alphabet's two classes are within
     S&P's own 20% switching margin **a third of the time** and the busier line has actually changed
     **four times** - and the company ranks **179 on GOOGL against 101 on GOOG**, so that rule would
     have moved a company 78 places on a coin flip. The choice is now declared in `config.yaml` and
     a run only *reports* a crossing. Top 25 unchanged; the 16 other Communication Services stocks
     gain 1.565 composite points each (they were being held down by rows that were duplicates); 0.0
     change outside that sector. `METHODOLOGY_CHANGELOG.md` 2026-10-21.
     **The bank P/B-ROE pairing: DECIDED 2026-10-21 (13:00) - ACCEPTED, with the guard it asked
     for replaced by one that can see what it claims.** The question was whether 35 points of a
     bank's Quality score double-count its Valuation inputs. They do not: a bank's ROE is its
     earnings yield times its price-to-book, so ROE *cancels* P/B rather than adding to it, and the
     four weights net to a position on earnings yield and on equity per dollar of assets with
     **nothing on price-to-book by itself** - which is the "high ROE at a low P/B" reading
     Damodaran, Wilcox, Nissim, Mercer Capital and MSCI's multi-factor indexes all build on
     purpose. Six ways of removing the "overlap" were rescored and every one is further from it.
     **So no weight moved** (0 of 501 composites); what was wrong is that the neutrality was the
     arithmetic of four separately chosen weights and nothing said so. The page now states it, with
     every figure generated from `config.yaml`.
     **What the review added:** both of the note's refutation criteria were re-measured on a
     different run (R-squared 0.978, P/B loading -0.069 against earnings yield's +0.283; all six
     candidates reproduce to the third decimal) - and the guard the note specified was measured
     too, and **could not see the one change the note itself calls the mildest route to a
     price-to-book tilt**: moving ROE's weight to ROA reads exactly 0.00 on its first-order
     formula, identical to today, while the loading measured on the engine's own scores is 2.3x
     today's and past the note's own bar. The failing condition is now the measured one.
     `METHODOLOGY_CHANGELOG.md` 2026-10-21 (13:00 session); the note's new section 11.
     **A third proposal arrived 2026-10-22 and was DECIDED 2026-10-23 (06:00) - all three of its
     registrations REFUSED as written; test B accepted in substance.**
     `research/2026-10-22-free-data-that-adds-evidence.md` asked a later session to register three
     pre-registered tests in `research/experiments/registry.yaml`. The decisive finding is that
     **the registry cannot hold any of them**: `experiments:` is 29 weighting transforms (29 of 29
     carry one) and `evaluations:` is 2 bot rules, and shaped as the note writes them the engine
     raises `KeyError: 'transform'` in `report`, `summary_for_page` and three places in
     `tests/test_engine_calibration.py`. For the short-interest test it is worse than an error:
     `v2_config` zeroes the metric and `_record` NaNs its inputs, so the entry would have **run
     clean on an all-NaN column** and published a null caused by the missing panel. Its data is
     real, though - FINRA's files were re-probed independently: 403 before 2017, **200 from
     2017-12-29**, exchange-listed rows present four years before FINRA's own page says, ~100% of
     rows carrying days to cover. So the note's research stands and the build order is now written
     into its section 7, licence check first. Test A refused on its own void condition (both
     coverage rates measured, 69.8% and 74.4%, sit under its 80% bar); test C refused on the trade
     (22 months buy 9.8% of precision, its confirm band fires with probability 0.808 under the null,
     and the months are size-selected by SEC Release 33-9002, so they cannot be pooled into the Size
     near-miss they would most be wanted for). A false "short interest has no free history" was
     corrected in four places, with a mutation-tested guard against the reclassification that
     correction invites.
     **A fourth proposal arrived 2026-10-26 (06:00) and was DECIDED the same day (13:00) -
     REFUSED, on the pre-registered bar the proposal itself nominated as its first test.**
     `research/2026-10-26-price-target-upside.md` §6.2 asked a later session to take
     `price_target_upside`'s 13.5 points off the scored *level* of implied analyst upside and put
     them on a 90-day *change* in the consensus target. It **passed** its own first check - the
     note said "if it duplicates `fy1_revision_3m` the honest outcome is no change", and it does
     not (+0.3475 within sector at the proposal's own window, against the incumbent's −0.007, so
     on construct validity the candidate really is the improvement the note claimed). It
     **fails** the check the note deferred: `revisions ~ momentum` goes **+0.2223 → +0.3778**,
     through the **~0.35** bar `research/2026-09-07-...` §8.4 set before seeing its own number -
     the same bar that *admitted* `fy1_revision_3m` at +0.317, which is why it is not available
     to be widened now. The sharpest number is the decomposition: **94% of the candidate's
     spanned variance is momentum** (R² 0.2598 on all eight categories, 0.2431 on momentum
     alone) against the incumbent's **22%**. So the swap would not reduce the screener's
     duplication, it would concentrate it in the one category the rule protects, and flip a
     −0.32 offset against Momentum into a +0.53 alignment with it.
     **Why the note expected otherwise, which is the lesson worth keeping:** it read the
     **30-day** beta (+0.245) and applied it to a **90-day** metric, where its own table says
     +0.540 - and more generally **a level beta does not bound a rank correlation.** Measured on
     the same series: the beta is flat at ~0.245 from 7 to 30 days while the within-sector rank
     correlation of target change with price change goes +0.116 → +0.296 → +0.526 → +0.581. A
     target that moves a quarter as far as the price still moves in the price's *direction*, and
     a percentile screener keeps all of the direction and none of the attenuation.
     **Three smaller things settled, so nobody re-opens them:** MSCI's `PTGF_C` and S&P Global
     MI's ratio are **rank-identical on all 98 end dates**, so §6.2's paragraph choosing between
     the two scalings cannot matter here; the Russell coverage-shrinkage rule the note called
     "worth copying regardless" would touch **no scored stock** (1 of 496 has under 3 analysts,
     median count **20**, 92.9% have 11+), so it was measured rather than shipped; and the
     90-day window can only ever be built from this project's own payload history, which the
     **2026-08-07** payload shows going to 0% coverage after a bad run.
     **What the review added that the note could not:** its three **[VERIFY FIRST]** papers are
     now read from PDF bytes, with **three corrections** - Palley, Steffen & Zhang's sample is
     **537,519 firm-months to December 2020**, not 465,797 to June 2018; its positive region is
     only the two lowest dispersion deciles, not "low dispersion"; and Han & Kim's signal is the
     **net number of optimistic analysts**, which that paper explicitly prefers *over* consensus
     implied upside, so it is not a direct test of this metric. AQR's Table X is exact. The net
     effect is that §6.1's reading of the *incumbent* ("literature and practice disagree") is now
     too generous to it - there is one direct test for and two against - and §8.7 states that
     question with its criteria written in advance rather than acting on it.
     **And the rule is now a tripwire instead of a comment**: `tests/test_revisions_momentum_budget.py`
     (7 tests, in the publish gate) holds the published run to the §8.4 bar, calibrated across
     **all 57 committed payloads** (max +0.3206, so no ordinary day fires) and mutation-checked
     in four ways, including an 80/20 contamination that fires at +0.4777. **No weight, metric or
     threshold changed.**
     **A fifth proposal arrived 2026-10-26 (owner-run night) and was DECIDED 2026-10-27 (06:00) -
     REFUSED as written, on the one measurement the note itself nominated as deciding.**
     `research/2026-10-26-data-run-timing.md` section 5 item 1 asked a later session to overlap the
     SEC Form 4 refresh with the Yahoo ticker fetch for **~5-7 minutes**, and said to read the
     first `PHASE TIMING` line from a real 02:00 run before deciding. That line landed this
     morning and puts the step at **87.5 s, not the ~400 s the note inferred** - so the saving is
     ~1.5 min. The reason the note's own 335 s stopwatch did not transfer is that it timed EDGAR at
     **21:30**, and at 02:00 the step is **floor-bound rather than latency-bound**: 566 requests x
     the 0.13 s politeness interval this project imposes on itself is **73.6 s**, so EDGAR
     contributed ~14 s against ~262 s in the evening. Even an infinitely fast EDGAR leaves 74 s.
     **So the ranking inverts:** item 2 (EDGAR's daily form index, ~566 requests -> ~2) takes the
     same step to ~2 s - a **larger** saving than overlapping can reach at all - with no
     concurrency, no GIL contention and 92 MB/day less downloading. Item 2 is the one to build.
     **Two further corrections.** Item 6's "no 02:00 data run in the record was rate-limited" is
     **false**: the first 02:00 run that could report the count reported **39**, with a 124.7 s
     retry pass - its own pre-registered tripwire failed on day one. Read honestly that is not a
     clean refutation (the owner-run night merged 3h22m earlier with two real subset runs, inside
     the window item 6's own theory names), and its policy - leave concurrency alone - stands
     either way. And the two skips shipped that night delivered **169 s, ~2.8 min** against ~4
     predicted; `context.options_pass` is 1.0 s against the 178 s removed. The count now has a
     place a session reads it (the brief's At-a-glance), which is what the month of watching needs.
     **No weight, metric or threshold changed.**
     **A sixth proposal arrived 2026-10-29 (owner-run, evening) and was DECIDED 2026-10-30 (13:00)
     - REFUSED as written, and the opportunity it describes is bigger than it measured.**
     `research/2026-10-29-ttm-fills-the-empty-quarter.md` §5 is the review; it is kept there rather
     than here on purpose (the 2026-10-30 retrospective named a refused proposal's full argument
     under Open as one of the two sources of `CLAUDE.md` regrowth). In one paragraph: when a
     stock's newest quarterly income column is empty, score it from the source's trailing-twelve-month
     series instead. The review measured the two things the note asked for and both favour it -
     the trailing series is **definitionally identical** to the engine's own four-quarter sum
     (median relative difference **0.000000** across all nine labels on 12 control stocks, which
     preserves the 2026-10-15 operating-income decision), and **MSCI and S&P DJI both pair a
     trailing-12-month income numerator with the *latest* book value**, so the mixed-period worry
     does not stand. What refuses it is the rule: it keys on the newest SEC filing rather than on
     the column the engine actually reads, and the source's trailing series returns **exactly one
     period end, 15 of 15**. Restated correctly the opportunity is **13 of 15 stocks, not 9** - but
     **7 of those 13 are two quarters past anything the quarterly table can check them against**,
     and that uncheckable half contains every stock whose fill would move a score materially,
     Amazon and Bank of America included. Amazon is also the proof that the note's table was one
     morning's data: recorded as a match on 10-29, and under its own rule the build would fill
     nothing for it on 10-30. Plus the prior-year half cannot be built from this source at all, so
     shipping the numerator alone would turn `revenue_growth` into a **fifteen-month** growth rate
     for exactly the stocks being fixed. §5.5 pre-registers the narrow version (gap of exactly one
     quarter, reconciled, prior-year anchor moved with it: **6 of 14** stocks, no top-25 name) and
     says plainly that whether it is worth building is still open.
     **Five of six proposals have now been refused by their reviewer, and this is the first whose
     review found the note *understating* its own case.** No weight, metric, threshold or published
     sentence changed.
     **A seventh arrived 2026-11-02 (owner-run) and was DECIDED 2026-11-03 (06:00) - one option
     accepted, two refused, and the premise all three rest on does not survive the longer record.**
     `research/2026-11-02-growth-exposure-drift.md` asked how to hold the ranking near
     growth-neutral now that a fixed 13% had "drifted" from -0.01 to +0.06 in three weeks: re-fit
     the weight on a schedule, constrain the exposure as MSCI does, or state the lean and leave it.
     The review ran the note's own sweep over **every published run** instead of three of them
     (`research/measurements/2026-11-03-neutral-weight-history.py`). **The Growth weight that would
     leave a run exactly neutral sat between 8.8 and 12.0 on all 40 runs from 2026-08-12 to the
     morning of 2026-10-09, was 13.6 on the evening run of the same day, and is 8.8 again today.**
     Between those two runs are the five changelog entries that changed how `operating_leverage`,
     `revenue_growth`, Piotroski 3/8/9, EBITDA and `forward_eps_growth` are computed. So **13% was
     never the neutral point** - it was neutral for about six runs, the 2026-10-13 review measured
     in the middle of them, and the "drift" is that step decaying back to where it was.
     **Option 1 (re-fit monthly) refused**: it would have moved the weight three times in a month,
     each time following the decay of this screener's own edit. **Option 2 (neutralise the
     composite) refused as a response to this evidence** and left open on its merits - there is no
     persistent drift to constrain, the exposure being stationary in +0.02..+0.08 across the 40
     pre-change runs. **Option 3 (state the lean, leave the weight) accepted** - which is what
     shipped on 11-02, so this is ratification, with one correction: the page said the balance
     moves "with the market", and the record says the one large move was ours. It now states the
     measured range and the cause. The note's own unanswered question is answered too - it is
     **not** the bank entries: excluding Financials the rise is *larger* (+0.0949 against +0.0814)
     and it continues through five runs sharing one `methodology_version`. Note section 6; no
     weight, metric or threshold moved.
     **An eighth was DECIDED 2026-11-04 (06:00) - the first whose recommendation was ACCEPTED as
     written, and the page gained what it had been missing since 2026-10-19.**
     `research/2026-11-02-investment-inside-quality.md` asked whether the Investment category
     (`asset_growth`, 5 points) should be folded into Quality, as nine of eleven provider families
     do, and recommended keeping it standalone. **The fold is refused and the recommendation
     accepted**; no weight, metric, threshold or scoring line moved. The six proposals before it
     were refused as written, and the reason this one is not is structural: it recommends *no
     change to any number*, and it measured the costs of the change it declines before declining
     it. Section 4 was rebuilt from `factor_engine` rather than read -
     `research/measurements/2026-11-04-investment-fold-review.py`, committed, where the note's own
     script was not - and run over **eight published runs instead of one**, which closes the note's
     own "this is one cross-section" caveat: **455 of about 500 composites unchanged and the top 25
     untouched on every one of the eight**, Quality's dispersion down **15.0-15.4%** against
     `check_run_health`'s 20% refusal bar. So the fold buys no ranking change and would spend
     three-quarters of a safety margin on methodology rather than data.
     **One of the note's listed costs is struck:** §3.2 says the positional read of
     `dispersion_history.csv` means dropping a column shifts every later one. `investment_disp` is
     the file's **last** column and `check_run_health.CATEGORIES` holds **five** categories, so the
     gate never reads it at any position - the fold is marginally *cheaper* than argued, which does
     not change the verdict.
     **What the review added:** every load-bearing figure re-read from the PDF bytes - Fama &
     French's Tables 4 and 6 (fourteen figures, all exact), MSCI FaCS's 25% and its worked Microsoft
     example, Scientific Beta's Exhibit 14, Hsu/Kalesnik/Kose p. 57 and Asness p. 5, **all exact**,
     with two one-page citation slips and no wrong number. Plus two things the note did not have:
     Scientific Beta's Exhibit 15, where the low-investment and high-profitability indices share
     only ~50% of their stocks; and the coherence point that **among large companies the investment
     factor's strongest documented correlation is +0.68 with low volatility, not with profitability
     (-0.18)** - so Quality is the category it is *least* related to, and if these 5 points overlap
     anything here it is Risk and Valuation. That is a Monday, not a weight change.
     **And the sound part shipped:** four paragraphs in the methodology page's Investment section -
     which measure, the large-company caveat, the joint position with Growth, and why it is
     standalone - closing items 1-3 of the page work
     `research/2026-10-19-revenue-growth-vs-asset-growth.md` §11.2 queued on 2026-10-19. The section
     had no large-company reading where Size and Momentum both carry thorough ones. Claim
     `overview.investment_standalone`, `tests/test_investment_standalone.py` (6 tests, in the publish
     gate by derivation); `METHODOLOGY_CHANGELOG.md` 2026-11-04; note section 6.
     **A ninth was DECIDED 2026-11-04 (13:00) - ACCEPTED, the second in a row, and the cost of
     acting on it turned out to be fifteen times smaller than it first looked.**
     `research/2026-11-03-open-source-quant-stack.md` section 1 asked whether the options panel's
     implied volatility and put skew should be **our own inversion of our own bid/ask mids**
     (QuantLib, American, discrete dividends, a Treasury curve) instead of the quote response's
     undocumented `impliedVolatility` field. **Accepted.** Its pre-registered bar holds on a sample
     disjoint from its own (median absolute difference at the money **2.00** volatility points), and
     the gap its own cross-check left open is closed: an independently written binomial tree
     reproduces QuantLib on **19 of 19** contracts within half a point - **10 of 10 without a
     dividend and 9 of 9 with one**, the dividend schedule being the part the note calls hardest.
     **What decides it is the paper the panel cites, and the evidence was already in this
     repository.** Xing, Zhang & Zhao's Table 1 (p. 647, read from the paper by the 2026-10-14
     session) puts the 5th percentile of their skew at **-0.99** points - a negative skew is about a
     **one-in-sixteen** event in the data the measure comes from. The field we publish puts it
     negative for **42-60% of the universe, median 47%, on every one of the 12 logged days** (~5,800
     stock-days, read from the committed context logs). Ours: 10% after the close, 35% mid-session.
     **Three corrections to the note.** Its headline contrast (5 of 48 against 28 of 48) does not
     replicate in magnitude - ours is 9 of 26 (35%) mid-session, while the field's 58% replicates
     exactly - so the honest version is 35% against 58%. Its proposed explanation is **not
     confirmed**: the candidate it could not test after the close was tested mid-session (the last
     trade inverted against the underlying's price at the minute of that trade, from 1-minute bars)
     and reproduces the field no better than ours does, so **eight constructions across two sessions
     match it none**. Which makes the case *stronger* than the note's own: it needs no inference
     about which sign is plausible, only this project's standing rule that every number on the page
     must be checkable from the page.
     **And the sizing, which is the part worth your attention.** A switch retires the readings built
     the old way, and a first draft of the review sized that at "18 logged days of implied volatility
     and 12 of the skew" by reading the log files. Run through `context_eval` itself: **one day, 467
     readings.** The 2026-11-03 anchoring rule already excludes every earlier log. So the switch
     costs one day and about one day of calendar time, delay is the expensive option, and the
     mechanism shipped today (`_ctx_opt_iv_src`, one constant, 5 tests) makes it cost nothing at all
     while giving a **paired** record - the only way this project could ever measure which
     construction predicts better. The inversion itself is the next session's build, order written
     in the note's section 1d. `METHODOLOGY_CHANGELOG.md` 2026-11-04 (13:00); no published number,
     score or sentence moved.
     **A tenth was DECIDED 2026-11-05 (06:00) - REFUSED as written, and the rerun it was going to
     cost is not needed, because the experiment could not have measured anything.**
     `research/2026-11-03-short-interest-days-to-cover.md` section 5 asked a later session to
     register a research-engine experiment **dropping `short_interest_ratio` from Revisions**. The
     research engine has never scored that metric: `v2_config` zeroes it and the record leaves its
     column **NaN in 176 of 176 months**, which makes **Revisions the only category v2 does not
     score at all**. So the arm *is* the baseline - measured over every month on file,
     **0 of 67,113 stock-months rank differently, maximum composite difference 0.0**
     (`research/measurements/2026-11-05-short-interest-registration.py`). It would have published
     "no difference, not distinguishable from zero", which reads as *the leg is not earning its 1.35
     composite points* and is in fact an identity. The note's own research and its verdict (the
     weight stays; the placement inside Revisions is a synthesis question) are untouched.
     **The second half is blocked by the engine's own rule, not only by the missing build:** FINRA's
     files begin 2017-12-29, which is **80 of 80** post-2020 months and **25 of 96** presample
     months, and the registry requires a presample arm pointing the same way before anything is
     CONFIRMED - so a built panel could only ever be judged on 2018-2026, which is a
     `METHOD_VERSION` change.
     **What shipped is the guard**, because this is the **third** proposal refused on the same fact
     (2026-10-23 asked for weight *on* the metric; this one for its removal):
     `tests/test_registry_can_measure_what_it_registers.py`, 11 tests, fails the suite on any
     registered transform that leaves v2's configuration unchanged, weights a column the record
     leaves NaN, or leaves nothing measurable carrying weight - with the three refused entries as
     fixtures, and mutation-checked by appending each to the registry. **Measured: before today
     nothing checked this.** With the entry registered and the new module excluded the suite reports
     **2 failed, 3,906 passed** - and both failures only read the *count* of registered
     alternatives, so both go green the moment the session regenerates the page, which the rules
     require it to do. A runtime refusal in `v2_experiments.run` is queued in
     `plan/research-engine.md` for a session already paying for a rerun. **No weight, metric,
     threshold, published sentence or production line changed.**
     **An eleventh was DECIDED the same session (2026-11-05, 06:00) - REFUSED as written, and the
     reason is a defect in how the context layer would count it rather than anything in the note.**
     `research/2026-11-03-following-the-smart-money.md` section 3 proposed logging each stock's 13F
     holder count and its quarterly change as a context-log candidate. A 13F reading changes on
     **four filing dates a year** and is the same cross-section in between; `context_eval` counts
     effective observations over **spaced run dates**, which is right for a signal that moves daily
     and blind to one that does not. Measured against the data loop's weekday cadence: the gate
     would read **13 effective observations a year against 4 genuine vintages (3.25x), and 25
     against 8 over two years** (`research/measurements/2026-11-05-quarterly-signal-accrual.py`) -
     the 2026-08-10 independence defect (2.35x) in a form the existing count cannot see, on the one
     number rule 4 and priority 0 exist to protect. So the counting must change **before** such a
     signal is logged, not after; recorded in `plan/context-layer.md` under "Things not to do".
     Counted honestly the signal's decision point is **2028-08-14** (eight vintages), for a signal
     the note's own literature finds absent in large companies after the 45-day lag.
     **Two further things measured.** `heldPercentInstitutions`, the field already fetched, is not a
     substitute and should not be logged as one: median **0.9167**, 91.1% above 0.70, and **24.5% of
     the index above 1.00**, so it is not a share of the company at all. But it does **not** settle
     the note's "little to sort on" for the proposal's own construct - breadth is a *count*, and the
     2016 Q1 13F probe this project already ran found a median of **598 filers per security**, so
     that reading is left open, which is weaker than the note's. Two conditions are pre-registered
     in the note's section 4.5 for whoever builds it; the Congress verdict and sections 1-2 stand.
     **A twelfth arrived 2026-11-09 (06:00) and was DECIDED the same day (13:00) - option 2
     accepted in substance and refused as specified, option 3 refused, option 1 left open - and
     the review found the detector it asks for already built and silently broken.** The full
     argument is `research/2026-11-09-share-counts-and-corporate-actions.md` section 8, kept
     there rather than here (the 2026-10-30 retrospective named a refused proposal's full
     argument under Open as a source of regrowth). In one paragraph: Fox closed its acquisition
     of Roku on 2026-11-06, the source added the issued shares, and FOXA's market value rose
     34.3% with its price unchanged - rank 34 -> 60 on four metrics moving 23-30%. The note asked
     for a run-to-run detector on `marketCap / currentPrice`. **The project has compared that
     expression with each company's filed share count every run since 2026-10-14**
     (`sec_crosscheck.market_value_check`), and the note does not mention it while naming
     `sec_crosscheck` as the precedent to copy. **It had been comparing nothing:** `financials`
     moved into `dashboard_detail.js` on 2026-11-08 and the health gate read `dashboard_data.js`
     alone, so the 11-09 run printed `0/0 within 5% (0 over 10%)` - 0 stocks of 485, and an
     ATTENTION line that fires above 60 can never be reached from zero. Nothing saw *that*
     because the check's test builds the payload by hand with `financials` inline.
     **Three measurements decided the rest.** The note's detector is an **event** and its question
     is a **state**: on KMB, whose count moved 2026-11-04, the delta fired on **1** run and the
     filings comparison on **all 5** to 11-09. Naming the stocks by *magnitude* would have missed
     FOXA, **13th of 485** behind the standing structure - against the published page the set is
     exactly the news, **CHTR and FOXA new, 18 standing, CPT gone**. And option 3 would exclude
     **IBKR, TKO, CVNA, MCD and WAT permanently**, not for a stub window, because the detector
     cannot tell a corporate action from a partnership-unit structure.
     **What shipped:** the gate folds every payload shard back (with a tripwire derived from
     `$DataArtifacts`, so a fourth payload file cannot repeat it), and the note now names what is
     new - verified on the 11-09 run, `CHTR +47.7%, FOXA +31.1%`. Never a failure; the note's own
     void condition is in the code. 14 of 14 mutations refused. **No weight, metric, threshold,
     score or published sentence moved.** Option 1 (a sentence on the page) stays open with two
     conditions pre-registered in section 8.5.
     **No proposal is waiting for a decision** as of 2026-11-09 13:00 - read
     `write_brief.open_proposals()` rather than this line.
  2. **Second-read every new lab experiment.** For experiment 15 (and any later one from tonight):
     open the paper, re-read each reference figure against `factor_lab.py`, and check in
     `git log` that the tolerances were committed before the comparison ran. A transcription error
     found here is a defect to fix and record; a miss that is real stays a miss. **Experiment 15:
     DONE the same night** - all ten figures re-read in the paper and recomputed independently;
     the licence's copyright notice added (`research/2026-10-20-lab-jkp.md`, "Second reading").
     **Experiments 16 and 17: DONE 2026-10-23 (13:00), and both stand - nothing needed
     correcting.** They were the two merged after 15 and the last ones outstanding, so **this
     sub-item is now clear**; the next experiment built needs its own second reader.
     *16 (`size-given-profitability`, Asness et al. 2018):* the publisher's open-access PDF
     obtained and Table 3 Panel A (p. 495) re-read - **all seventeen reference figures exact**,
     including the one derived figure (the paper's column "4" minus "Big" is 0.28/0.19/0.25/0.30/
     0.25, mean **0.254**). A second implementation that does not import `factor_lab` recomputes
     every measured figure and the pre-registered criterion (**0.1661 a month, t 2.3641** against
     its bar of 0.136 and 1.5; **-0.1066, t -0.70** since 2013), which reproduces **10 of 17
     within** and "not refuted" - so **Size's weight is not reopened**, on a checked number rather
     than an unchecked one. `research/2026-10-20-size-weight.md` section 12.
     *17 (`piotroski`):* Table 3 Panel B (p. 17) re-read - **all four figures exact** (0.078,
     0.313, 0.235, t 5.594) with the firm-year counts, and every measured figure reproduced
     independently (26.07 / 30.41 / 4.34 / t 1.5175; the authors' value-weighted cut 12.58, t 3.64;
     the end deciles' average scores 2.69 and 7.53). Section 7's other-period claims were measured
     too, since no tolerance covers them, and hold. **Two source observations, neither a defect
     here:** OSAP's own `SignalDoc.csv` names Table **3A** as the key table while quoting Panel
     **B**'s t-statistic (Panel A's is 5.590 on market-adjusted returns) - the experiment cites
     Panel B, where its figures actually are, so it is right and more precise than its own data
     source; and the `open_details` figure from Table 4 ("largest third, high minus low 15.2%,
     t 1.88") is exact but market-adjusted, a basis the sentence does not name, which changes no
     number. `research/2026-10-20-lab-osap.md` section 8.
     Both pre-registrations were verified in `git log` as genuine: tolerances committed with an
     empty `verdict_note` and no `lab_data.js`, 24 and 90 minutes before the result commits.
     **Experiment 19 (`mohanram`, Mohanram 2005 on OSAP's MS groups): DONE 2026-11-07 (06:00
     weekend session, which did not build it) - it stands, nothing needed correcting**, so this
     sub-item is clear again until the next experiment is built.
     `research/2026-11-06-lab-mohanram.md` section 8;
     `research/measurements/2026-11-07-lab-mohanram-second-reading.py`, committed, which does not
     import `factor_lab`. The pre-registration is genuine (`4a55141` 13:14, +173 lines to
     `factor_lab.py` and nothing else, `verdict_note` empty; result `3f3cb98` 13:26 - **twelve
     minutes**, the shortest gap of any experiment, with all seven figures and tolerances
     byte-identical across the two). **All seven reference figures exact**, re-read from the PDF
     bytes - and in **two independent manuscripts**, Columbia's as well as TSpace's, whose Table 4
     Panel A is character-identical; the *version of record* is still behind Springer's paywall,
     so that caveat stands. Panel B's two ends and the 5,585 firm-years behind the paper's t are
     exact too. **Every measured figure reproduces to the second decimal** (+3.22, +5.40, +6.48,
     +12.75, +7.58, +16.48, spread +13.26, t 4.96; 4 of 7 within; the four later-window spreads
     and their t's; the strongest group the calmest at 6.24 against 7.18-7.89), and the library's
     own `LS` port averages 1.1047%/month = 13.26% a year, so the lab's spread is the library's.
     **The note's weakest claim is now its strongest:** section 4 said `SignalDoc.csv`'s `Return`
     column is the *paper's* figure rather than a measured one, on one check (`PS`). Tested on
     seven signals, **five differ from the library's own measured long-short** by more than
     0.05%/month, two by over 40% - and the column's values are recognisable as published
     headlines (`Mom12m` at **1.31%/month** is Jegadeesh & Titman's own number). **One slip found,
     and it is the paper's:** Mohanram's text on PDF p. 26 cites "the 21.2% difference reported in
     Table 4" while discussing raw returns - 21.2% is Panel B's spread, and Panel A's is 21.4%,
     which is what the record uses.
  3. **Test the tests.** Pick three registered claims a session (`claims.py`), break the code each
     one describes in a scratch copy, and confirm its `checked_by` test fails. A check that still
     passes is checking plumbing, not the claim (CLAUDE.md, 0.11's lesson) - fix it, and keep a
     running table in `plan/claims-mutation-audit.md`. **Started the same night:** 38 mutations on
     22 claims; **15 got through** (8 were untested boundaries, 3 checked the committed payload
     instead of the code), all 15 now caught by registered tests.
     **Second pass DONE 2026-10-25 (13:00 session) - the queue's items 1-8 and 12 are closed, and
     one of the claims turned out to be false rather than unchecked.** 24 more mutations on 11
     claims (rows 39-62): **15 caught, 4 real gaps, 5 gaps in the register, 1 false claim.**
     The four real gaps - two bars in the weakest-category sentence, the panic rule's volatility
     bar, and the capex note's 5% reconciliation - now each fail a registered check, verified by
     re-running every mutation after the fix. **The false one:** `summary.earnings` and
     `_sentence_earnings`'s docstring both said "a date at or before the run date is dropped",
     which no version of the code has done - it keeps a same-day date and reads "the day of this
     run", which is right, and is the one day the paper behind the feature says a sell beats its
     counterfactual. The sentence was corrected; nothing a reader sees moved.
     **Two things generalise.** A check may not import the number it is checking: the capex
     tolerance's only assertion read `rel=X.LEASE_RECONCILE_TOL`, so widening the constant widened
     the assertion with it and the test passed at any value. And a test's name is read far more
     often than its body - `test_the_threshold_is_strict_and_configurable` probed the volatility
     bar 0.001 either side and never *at* it, the one point that separates `>` from `>=`, so the
     name made the gap invisible to anyone looking for one.
     **And the limitation the last pass recorded is fixed:** `checked_by` can now name
     `Class::method`, which the register could not, so the good tests in `tests/test_metrics.py`
     are no longer invisible to it or to the publish gate. The plan file lists the next six items,
     two of them mechanical searches that should outperform more hand-picked mutations.
     **Fourth pass DONE 2026-10-28 (13:00 session) - the research page's statistics, and the last
     three residuals. 17 mutations on 6 claims: 11 caught, 6 gaps, four of them seen by nothing
     anywhere in the suite.** Rows 89-105; the queue's items 1-4 are now closed. The research
     surface was the one no pass had touched, and what it found is a *kind* of unchecked claim this
     audit had not met: **the only thing seeing six of its eight gaps was a drift tripwire** -
     "the committed page is what the generator makes" - and a drift check is green in exactly the
     state the change ships in. Measured, not argued: each mutation was re-run with
     `python v2_experiments.py report` in between, the way the page is actually published, and
     **all seven stayed green.** Underneath that, two claims were checked by tests that recompute
     the rule out of the code and then ask the page to agree with it (`every_zero = all(lo <= 0 <=
     hi ...)`; `published["better"]`), which is row 59's rule applied to a *rule* instead of a
     constant and much harder to see.
     **The six gaps, now each failing a registered check:** the interval boundary at exactly zero
     (the published figures are rounded to four decimals, so an endpoint of 0.0000 is a real
     state); the intervals' confidence level, in both places that builds them; the record the
     pooled estimate covers (176 months, not 80); what "cleared its bar" means for the count of
     alternatives; the Sahm trigger *at* 0.50; and the window that makes the Sahm low a
     **prior**-12-month low. The last two had real arithmetic behind them and were blind anyway,
     because **every fixture in the register was a rising unemployment series**, where the prior-12
     low and the inclusive-12 low are the same month.
     **One register finding worth the owner's attention:** `tests/test_engine_calibration.py` -
     the seven synthetic tests that are the reason to believe the engine can measure what it
     reports - was named by **no** claim, so the loop that publishes the research page had never
     run them. It is in the gate now, by derivation. **Four of the six gaps change no number
     published today**; they change what the next one would be, which is the class of defect only
     a mutation finds. Nothing about scoring, no published text, and no production code changed.
     The plan's next list is three items, one of them the three *unregistered* computational
     sentences this pass found on the research page.
     **Fifth pass DONE 2026-10-29 (06:00 session) - those three sentences are claims now, and the
     measurement is that nothing in the project could see them change.** Rows 106-117 of
     `plan/claims-mutation-audit.md`. The research page's "How long would it take to prove an
     edge?" section prints the ranking's monthly swing (`se x sqrt(n)` over 2020-2026), three
     "months of record" counts, and the smallest difference the test could see in each era. All
     three say how a number is computed and **none was in `claims.py`**, so the loop that
     publishes them had never run a check of them. They are registered now with checks on both
     sides of every figure, and **12 mutations produce 12 catches**.
     **The number worth your attention is the other one.** 12 of 12 is the expected result for
     tests written the same day, so the question asked instead was: before today, would *anything*
     have noticed? One mutation per claim, with the page regenerated the way
     `python v2_experiments.py report` publishes it, then the **whole** suite with only today's
     four new tests deselected: **3,506 passed, 1 skipped - three times out of three.** Not
     under-checked; invisible. Taking the swing off the wrong era would have moved the published
     figure from 0.10 to 0.12 and all three month counts with it, and shipped.
     **Two things generalise, both about what "caught" means.** Two of the twelve *were* caught
     before today - by a test no claim named, so the publish gate (whose membership is derived from
     `checked_by`) never ran it; "something in the suite fails" and "the loop refuses to publish"
     are different facts. And the test that caught them asserted
     `months_needed(...) == round(((3.0 + 0.84) * 0.12 / 0.05) ** 2)`, the function's own
     expression copied, which survived only because those two mutations changed the code *beside*
     the copied line rather than the line itself. So a copied-expression check reads as coverage
     until a mutation happens to land on the copied part. It is literals now (85 and 45).
     **One structural fix:** `claims.RESEARCH` is a `detect` surface at last, and the **generated**
     page rather than the committed one, so a reworded research-page sentence goes stale on the
     session that rewords it. The register could not carry a pattern for that surface before today.
     **No weight, metric, threshold or published sentence changed** - the diff is the register,
     tests and the write-up. The plan's next list is three items, and its first is the sweep this
     pass argues for: the sentences that are not registered **at all**, which is the gap
     `claims.py`'s own docstring names and which this audit cannot reach by mutation. Three more on
     the research page are named there with what each needs (the difference chart's band and
     pass-bar geometry, the overlap figures, the bot table's intervals).
     **Sixth pass DONE 2026-10-29 (13:00 session) - that sweep is built, and the number to read is
     that a brand-new sentence stating the arithmetic of a published figure is invisible to 3,515
     tests.** Rows 118-126 of `plan/claims-mutation-audit.md`. The three sentences registered this
     morning were found by *reading one section of one page*, which is attention, not a method -
     and this audit cannot reach that class at all, because mutation can only mutate what the
     register names. So `tests/test_claims_register.py` now sweeps `research_page.py` the way it
     has always swept `stock_summary.py`: it enumerates **every sentence the page prints that puts
     a number this repo computed into prose - 32 of them** - and each must carry a verdict
     (`claim:`, `checked:`, `reported:` or `open:`). A new or reworded one fails the suite until it
     gets one. `checked:` means *every* computed figure in the sentence is pinned by the named
     test; one unpinned figure makes the row `open:`, and every verdict was assigned by reading the
     assertion rather than its name.
     **What it found: 12 of the 32 sentences have at least one figure pinned by nothing.** They are
     listed by literal slug, the set only ever shrinks, and the plan carries what each needs. The
     sharpest ones: the pass bar is printed **twice** on the page and nothing pins either copy or
     that the two agree; the overfitting section's PBO words are probed at 0.21/0.5/0.73 and
     **never at their own bars, 0.45 and 0.55**; and the trading rule's monthly figure is matched
     as `[0-9.]+%`, so its sign, words and t-statistic are pinned and **the percentage itself is a
     wildcard**.
     **One closed the same session** - the universe paragraph, which this morning named as a
     residual. `research.pairing_overlap`, five checks, including the branch threshold at exactly
     99% of the screen's universe and the record the overlap mean is taken over.
     **9 mutations, 9 caught, and 9 of 9 blind to everything else** (apply, regenerate the page the
     way `v2_experiments.py report` publishes it, then the whole suite minus today's eight tests:
     3,515 passed, eight times out of eight). Two of the nine are the sweep's own teeth: adding a
     new computed sentence, and rewording an existing one. The ninth breaks the sweep itself and
     confirms the floor fires rather than passing vacuously.
     **No weight, metric, threshold or published sentence changed.** The plan's next list leads
     with the rest of this item - the same sweep over `build_screener_overview` and
     `generate_dashboard.py`, one surface per session - and the twelve open rows as a queue.
     **Eighth pass DONE 2026-10-31 (06:00 weekend session) - four of the open rows, and a
     correction to what this audit counts as "caught".** Rows 131-138 of
     `plan/claims-mutation-audit.md`. The plan called the nineteen `open:` rows a queue and named
     four; all four are closed on the **value** of the figure, 8 mutations for 8 catches, and
     `KNOWN_OPEN` drops from twelve to ten. They were: the Data quality bullet's second printing
     of all four clamps (a test's own docstring claimed to cover it and matched only the metric
     rows); the share-class switching margin, pinned twice as a literal and read off the page by
     **nothing** - `tests/test_share_classes.py` had never called `build_screener_overview`; the
     trading rule's monthly excess, matched as `[0-9.]+%`, so the published percentage was a
     wildcard in all three periods; and the two survivorship coverage percentages beside it.
     **The correction is the part worth your attention.** Run with the row's own check deselected,
     the whole suite *did* fail on three of these mutations - and **not one of the three failures
     was a check of the number**: two were drift tripwires ("the committed page is what the
     generator builds"), green the morning the 02:00 run rewrites the page, and one was the
     sentence sweep built the day before, which keys a sentence by its text and so detects *an
     edit* rather than *a wrong number*, satisfied by registering the new wording. A mutation can
     read as caught twice over and still ship; this audit no longer counts either as arithmetic
     coverage.
     **One production change**, and the reason the row was worth taking: `build_screener_overview`
     fell back to its own `0.20` literal for the switching margin while step 11 falls back to
     `share_classes.CHALLENGE_MARGIN`. They agree today, so nothing on the page was wrong - but
     two literals for one number are free to drift, and the page would then have stated a margin
     no run applied. The generator shares the constant now (row 0.8).
     **Ninth pass DONE 2026-10-31 (13:00 session) - nine more rows, and the morning's own two
     turned out not to have been closed.** Rows 139-147 of `plan/claims-mutation-audit.md`, plus
     a section on the bookkeeping. **20 mutations, 20 caught, one equivalent mutant recorded, and
     every one of the 20 a GAP against the same claim's `checked_by` before this session.** The
     `open:` queue is **six**, down from nineteen on 2026-10-29: `KNOWN_OPEN` ten -> five on the
     research page and `OVERVIEW_KNOWN_OPEN` seven -> **one** on the methodology page.
     **The part worth your attention is that this morning's two rows were not closed.** The 06:00
     commit wrote five real checks for the share-class margin and extended one for the clamp
     bullet, and touched **no register file**: both sentences still read `open:`, the slug list
     still named both, and the five new checks were in no claim's `checked_by`. Re-run against
     what the publish gate actually derives - the claim's own `checked_by`, not the module holding
     the test - the margin row was a real gap: **6 of 6 registered nodes passed both mutations**,
     while a test in the suite caught them. So "the row is closed" is three facts, not one (the
     check exists, the register names it, the verdict says so), and that is now a rule in the
     plan. The clamp row's gap was the label alone.
     **The same shape, one level up, is the bigger find:** `tests/test_overfitting.py` was named
     by **no claim**, so all eight of its checks - including the only reader of the overfitting
     section's stat cards and of its C(16,8) split count - were **outside the publish gate**.
     Exactly `tests/test_engine_calibration.py` on 2026-10-28, which is the finding the 10-29
     session flagged for you. Registered now (`research.overfitting_reading`), and the module is
     in the gate by derivation.
     **What the nine rows were:** the whole "If you had kept the best backtest" section - three
     prose PBOs and the words beside them, `_pbo_words` probed **at** 0.45 and 0.55 for the first
     time (0.21/0.5/0.73 passes for any bars between 0.21 and 0.5 and between 0.5 and 0.73), two
     out-of-sample shares, the count of alternatives, and the top winner's two era leads with the
     clause that depends on their signs; then the methodology page's three bank P/B-ROE prose
     figures and the coverage-discount worked example. The **blindest sentence this audit has
     met** is the bank residual clause: today's weights print the *neutral* wording, so the branch
     that states a measured price-to-book tilt was on a page **no test had ever generated**. It is
     reached now with weights of the test's own, bracketing the 0.25 neutrality share to ±0.013
     and probing both sides of its `against`/`on` word.
     **One correction to this audit's own record:** row 147's note said the worked example's five
     figures were "read by none of them". Its check captures **seven** and pins four; three were
     read and discarded, including a coverage percentage bound to a regex group and never asserted
     at all. A row's note is written before the work - re-read the assertion before trusting it.
     **No weight, metric, threshold, published sentence or production line changed.** Next in the
     plan's list: `pass-bar-printed` with `pass-bar-under-chart` (the same figure printed twice,
     neither pinned), then the two Size rows, `chart-axis-and-geometry` and
     `coverage-sentence-metric-counts`; `generate_dashboard.py`'s sweep still wants its own
     session.
  4. **The positive control as a ledger row** (`plan/research-engine.md` item 4(c), second half): the
     look-ahead arm as `kind: control`, rerun with everything else. The synthetic half shipped
     tonight (`tests/test_engine_calibration.py`).
     **DONE 2026-10-25 (06:00 weekend session) - the test can see a signal, and now it is a number
     anyone can check rather than a sentence.** The research page has told readers since 2026-10-15
     that "the test can see a signal when there is one", on the strength of a measurement that was
     never a ledger row and whose monthly records have since been rebuilt twice. It is now the
     registered control `lookahead-filings`: the same weighting, the same universe, the same prices,
     but every month scored on the filings as they stand today - backtest v1's look-ahead on purpose.
     **Result: the cheating ranking beats the real one by +0.0512 of monthly rank correlation over 80
     months (95% +0.0330 to +0.0694, t 5.59), against the 0.0373 this test could reliably detect -
     and inside the +0.030 to +0.075 band written down before the arm was rebuilt.** On the held-out
     half, +0.0561 with t 8.34. The baseline measured on the same stocks is **-0.0003**, which is the
     plainest statement of where the screener stands: with the filings as they actually stood,
     approximately nothing. So the twenty-eight flat verdicts are a finding about the rankings, not a
     blind test - *for differences of about this size*; a typical published IC of 0.02-0.05 is still
     below what this design can see, exactly as `research/2026-10-15-research-engine-first-batch.md`
     says.
     **What was built, not just measured:** `controls:` in the registry (a control has no
     `transform`, is never in the multiple-testing denominator and can never support a weighting
     change - the 2026-10-23 refusal is why that distinction is in code); the row names its own arm by
     SHA-256, because `content_hash` covers the point-in-time records and cannot see the arm's file;
     `scripts/rerun_experiments.py` reruns a control whose arm was rebuilt and *prints* an arm that
     is not built rather than taking a whole rerun down; and the page now has three states, including
     "the design did NOT see it - every flat result here is uninterpretable", because a page that
     could only report the reassuring outcome would be worthless. Claim
     `research.positive_control`, which added `research.html` as a claims surface, so
     `tests/test_positive_control.py` (20) is in the publish gate by derivation.
     Record: `research/2026-10-25-positive-control.md`; `plan/research-engine.md` item 4(c).
     **One cost to budget, which this exposed:** a test requires every newest ledger row to share one
     `code_hash`, so adding a single row forces a ~45-minute rerun of all 31 - whose numbers provably
     cannot move (checked: 31 ids, 24,796 statistics, **0 differences**). Both the test and the rerun
     are right; just do not start one you cannot finish, because the ledger is append-only.
  5. **Lab experiment 4b** - size inside each profitability quintile (French
     `25_Portfolios_ME_OP_5x5`), with the refutation criterion already written in
     `research/2026-10-20-size-weight.md` section 9. It is the one result that would reopen Size.
     **DONE 2026-10-20 the same night** - it shipped as experiment 16, `size-given-profitability`,
     10 of 17 within and the rule not refuted - and **second-read 2026-10-23** under item 2 above,
     which is what makes the "not refuted" reading checked. This line was still open because
     nobody had closed it; the work was finished before it was written.
  6. **Backtest v2's two free data fixes, then the extended build**
     (`research/2026-10-20-free-survivorship-sources.md`): 167 member-months of today's members
     dropped because a rename or symbol format is not carried before May 2014 (BF.B, MHP->SPGI,
     SAI->LDOS, PCS->TMUS, ACE->CB, TYC->JCI), and 773 `no_filings` member-months whose older filings
     sit under a predecessor's CIK (Exxon, BlackRock, Disney, Cigna, Alphabet...). Both touch
     `RESULT_PATHS`, so do them together: one rebuild, one rerun. Then the note's section 8 plan for
     the extended universe - **register `only-size-extended` first, with its one-sided reading**
     (free data adds the members that shrank out but almost none that were acquired, so the Size
     test can falsify the survivorship explanation and cannot confirm it).
     **The two data fixes are DONE, 2026-10-24 (13:00 session)**, with the one rebuild and the one
     rerun they force - `research/2026-10-24-v2-pre-2014-renames.md`. `no_filings` went **773 -> 91**
     in 2012-2019 and **177 -> 13** in 2020-2026; **three** of the eight uncarried renames were
     admitted (BFB, MHP, SAI - 43 member-months) and five refused with their reasons, two of them
     because the symbol they would claim was itself a member that month. **+398 and +98 member-months
     scored, none removed** - until today v2 ranked no Exxon in any month and no BlackRock in 58 of
     80 after 2020, so every published engine result had been measured without them. One rule in it
     was wrong on its first draft and measuring it is what showed that; the log entry has the
     account. **What remains of this sub-item is the extended universe** (`only-size-extended`
     registered first). The residual inside v2 is now **coverage, not registrants**: 725 records were
     built in 2012-2019 and 398 scored, because a filings-only record carries only the tags the filer
     used - Exxon files no `OperatingIncomeLoss` at all. More XBRL tags per input is the next free
     step, and it needs no new data source.
     **Late the same night:** Tiingo's public ticker list carries 84 of the 152 acquired names (65% of
     acquired member-months) - the hole free sources could not fill. If the owner opens a free Tiingo
     account, he saves the key himself in the gitignored `data/survivorship/tiingo_key.txt` (never in
     chat, never in a tracked file); then spot-check a few series, read Tiingo's terms on derived results,
     and rebuild with it - the Size test would then be able to confirm as well as falsify. Addendum in the note.
     **Licence gate (checked 2026-10-20 late):** the free Starter plan ($0; 500 symbols a month, 50 requests an
     hour, 1,000 a day) is **personal use only** - its pricing page says the data may not be displayed or
     shared with another person or organization. Whether a statistic computed from it (a monthly IC) may
     appear on the public research page or in this public repo is **not settled**. Until Tiingo confirms
     in writing that it may, **nothing derived from Tiingo prices is committed or published**: raw files,
     rebuilt months and any results stay in gitignored folders and out of the ledger. Do not rebuild v2's
     published records with them.
  7. **Watch two numbers and act before they bite:** the full suite's runtime (6:23-7:20 tonight,
     against the Bash tool's 600 s per call - speed it up or split the call, never drop tests), and
     `lab_data.js` at ~2.9 MB (measure the lab's first paint; move the JKP series to their own file
     if it is slow).
     Also: three tests (`test_claims_register::test_confidence_metric_count_is_the_discount_coverage`,
     two in `test_investor_profiles`) fail with FileNotFoundError in a fresh worktree, which has no
     gitignored `runs/` - every agent tonight had to copy it in. Make them skip loudly when no run exists.
     **All three parts DONE 2026-10-26 (owner-run night).** *Suite:* runs in parallel through
     `scripts/run_tests.py` (pytest-xdist, with the exact serial command as the fallback) - **1:51
     against 6:09 serial** on the merged tree, same 3,248 passed / 1 skipped; gate 1, the data loop's
     claims gate and the prompts all use it. *Lab:* the six heavy series now load only when their
     experiment opens - `lab_data.js` 1.18 MB -> 100 KB gzipped, a throttled phone reaches the menu
     in **0.95 s instead of 6.3 s**, every figure byte-identical. *Fresh worktrees:* the three tests
     skip with `MISSING DATA: ...` when `runs/` is absent; copy data in with `cp -rp` (plain `cp -r`
     loses mtimes, which made `test_investor_profiles` pick the wrong run for two agents that night).
  Close this item when 1-6 are done or argued down in the log.


## The 2026-10-28 item (universe expansion) - text to 2026-11-19

- **2026-10-28 - Beyond the S&P 500: add NASDAQ and NYSE stocks, phased in.** Owner, in chat:
  *"right now we only have the S&P 500 stocks, and that's good. There's been some really good
  testing done on these stocks. But over time, I would really like to get the NASDAQ and the New
  York Stock Exchange tickers in here as well ... we would have to phase it in bit by bit."* A
  long-horizon item; the ordered queue, the measurements behind it and what must not change is
  `plan/universe-expansion.md`. In short: **phase 0** is a Monday research note (which universe
  first - the S&P 1500 is the expectation to test - how professionals handle microcaps, analyst
  coverage and sector percentiles once small companies are in the pool); **phase 1** makes the
  universe a parameter and loads each stock's detail on demand, with the S&P 500 page byte-identical;
  **phase 2** scores the S&P 400 + 600 in shadow, unpublished, and writes the numbers down;
  **phase 3** publishes the S&P 1500 beside the S&P 500; **phase 4** is all NYSE and NASDAQ common
  stocks above a pre-registered floor. **Throughout: the S&P 500 ranking, `signals.json` and the
  bot are unchanged, and each universe keeps its own evidence series.** **Phase 0 written
  2026-11-02 (owner-run evening; the scheduled Monday session could not run):**
  `research/2026-11-02-universe-expansion-phase0.md` - S&P 1500 first, the phase-4 floor at the
  NYSE 20th percentile ($1.22B today), percentiles within sector *and* size band, and why.
  **Phase 0 DECIDED and CLOSED 2026-11-03 (13:00 session, which did not write the note) - all
  three pre-registrations ADOPTED, two of them corrected by the review.** Note section 9; the
  operative decisions are now in `plan/universe-expansion.md`, pinned by
  `tests/test_universe_expansion_preregistration.py` so they cannot be re-fitted later.
  **The floor's headline number is exact** - Ken French's June 2026 row re-read independently,
  n = 1,079 NYSE stocks, $1,224.13M - **but two of the five reasons given for choosing it do not
  survive the longer record and are struck.** Over 2015-2026 the line moved **+112.1% (2021),
  -34.4% (2022) and +40.7% this June**, so "it moves with the market, so it never needs re-tuning"
  is true of the rule and misleading about the line; and its agreement with S&P's own $1.2B
  small-cap entry floor is **this June's coincidence** (June 2025: $870M; June 2020: $484M). It is
  adopted on the one durable reason - it is the breakpoint Fama & French, Hou/Xue/Zhang and
  Jensen/Kelly/Pedersen all draw - and the consequence the note did not draw is now in phase 4:
  **the universe churns at its bottom edge every July, so the bot-style continuity gates must be
  re-derived from a replay that spans a July**, never an average month.
  **The band split's exit test is a measurement now, not an argument:** grouping percentiles by
  `(Sector, size_band)` with a band-wide fallback leaves the S&P 500 at **0 of 23,092 percentile
  values changed**, against **20,151 changed (up to 68.93 points)** for a genuine two-band split -
  so the zero means something. And its cost has a number the note did not have: the ten-peer
  fallback fires for **1 of 301 weight-bearing cells today** and **27-35 of 903 after the split**,
  every guaranteed one in **S&P 400 Communication Services (6 members)** - the only sector-band
  group under ten members anywhere in the S&P 1500. **The thin group is in the middle band**, so
  phasing the 600 in last does not defer it.
  **One self-contradiction in the note resolved by measuring:** section 7 said both "keep 2e9 for
  the S&P 500" and "set 0 for S&P index bands", which disagree for the one band whose
  byte-identity is the phase-1 exit test. **0 of 503 members sit below $2B** (smallest MOS at
  $6.050B, p1 $8.796B), so one rule - `min_market_cap: 0` for every S&P index band - is
  byte-identical today and needs no special case.
  **Left open on purpose:** section 4's choice of coverage rule for a stock with no analysts. It
  is a present-day methodology change (one S&P 500 stock today, Loews) and gets its own changelog
  entry, not a side effect of phase 3.
  **Phase 1, the `size_band` slice: SHIPPED 2026-11-07 (06:00 weekend session).** Percentiles are
  now ranked within `(Sector, size_band)` with a band-wide fallback, and `Composite_Pct` within
  band - pre-registration 3, in code. One universe means one band, so **nothing published moved,
  measured rather than argued**: 0 of 23,092 percentile values differ from the sector-only code
  it replaced, against 20,140 moved by a genuine cap split, so the zero is not vacuous; a
  five-ticker live run and `rehearse_publish.py` confirm it end to end. `universe.py` joined the
  research engine's `RESULT_PATHS` (it decides what a stock is ranked against) and all 32 ledger
  rows were rerun - no statistic moved. `plan/universe-expansion.md` has the record.
  **Phase 1, the per-band health gate: SHIPPED 2026-11-07 (13:00).** The analyst-target floor is
  applied inside each size band as well as universe-wide, with the band read from the run's own
  parquet so the payload stays byte-identical. The auto-disable's *trigger* was deliberately left
  universe-wide: it changes the weights, so choosing between its three per-band rules is a
  methodology decision needing phase 2's coverage measurement, and the per-band figures are
  recorded now. `plan/universe-expansion.md` has the record.
  **Phase 1, the `stock_detail` shard: the mechanism and its first slice SHIPPED 2026-11-08
  (06:00 weekend session), measured at -19.8% of first paint.** The fields only an open sheet
  reads now ship in a third payload file, `dashboard_detail.js`, fetched when the first sheet is
  laid out: **first paint 1.547 -> 1.240 MB gzipped, -307 KB**, nothing published moved (the
  business description is display-only prose by the 2026-08-26 decision). Which fields move was
  decided by measuring every field's weight *and* every one of ~85 JS read sites - the rule is
  **one read site, inside the sheet, and no number** - because the page reads
  `D.stock_detail[t]` synchronously all over, so the fields the peers panel, movers strip and
  calculation traces read cannot move until each is measured. Guarded in the same commit on both
  publish paths, with a derived tripwire so the next payload file cannot be left out of the
  rehearsal; 7 of 7 plausible mutations refused.
  **Phase 1, the shard's second slice: SHIPPED 2026-11-08 (13:00 session), and the day's total is
  -25.0% of first paint.** The morning named `financials` + `self_metrics` as next; **one of the
  two did not need sharding, it needed deleting.** The peer table built its nine columns **two
  ways** - the open stock's row from a precomputed `self_metrics` block, every peer's row from
  `buildPeerRow` in JS over `raw` + `financials` - and every stock is on both sides of that split,
  being the subject of its own sheet and a peer in its sector-mates'. Measured before touching it
  (`research/measurements/2026-11-08-peer-row-two-paths.py`): **0 of 4,491 cells disagreed**, so no
  published figure moved, though the two expressions for P/E really were different code. One path
  now; **-17 KB with no fetch, no waiting state and nothing new on demand.** Then `financials`
  moved into the shard: **-62 KB**, taking first paint **1.547 -> 1.161 MB gzipped on the day**.
  What the second half actually built is a **mechanism** - the shard could not carry a *number*
  until the loader could redraw the panels that print one, and both panels now say they are
  waiting rather than showing an em-dash that means "this run published none". The membership rule
  is restated in the plan. 7 of 7 mutations refused, the one that got through first recorded as an
  equivalent mutant and then closed. Gate 5 passed with all five published files byte-identical.
  **Phase 1, the `config.yaml` block: SHIPPED 2026-11-19 (06:00 session) - and the $2B floor it
  replaces had never excluded a stock.** `universe.index` is read now and a name with no
  definition stops the run (the old comment offered the Russell 1000 and 2000; either silently
  ran the S&P 500). The floor is 0, as decided on 2026-11-03. Checking that the change was
  byte-identical turned up why it had to be: the filter looked for a column the scoring frame
  does not carry, so 502 stocks went in and 502 came out at a floor of $1 quadrillion. A floor
  that is set is now applied, or the run stops - which phase 4 needs, because its universe is
  *defined* by one. No published number moved. A side finding went to `plan/backtest-v2.md`: the
  research engine's history has Berkshire Hathaway at under $2B in every month and 16 tickers
  under $100M somewhere (wrong share counts), not yet fixed.
  **Next in phase 1:** `inp`/`raw`/`pct`/`summary` (42.6%
  of what is left, each needing its own read-site audit), then bucketing - one file is right at 500
  stocks and wrong at 5,000, so the shard is "off first paint" today and not yet "flat in the
  universe". **Grep `tests/` for a field before moving it:** `tests/examples.py` reads `financials`
  and picks the stock four publish-gate modules open.


## The 2026-10-15 item (research engine) - text to 2026-11-19

- **2026-10-15 - A research engine: test the methodology the way automated research loops do, so it
  can be shown to people and trusted.** Owner: *"we're trying to come up with the best methodology
  to screen stocks by, and it has to be backed up, it has to be proven, so ... I can show this to
  people, and they can trust me that this screener works ... first, it's going to be a good trading
  bot, but then eventually ... the methodology behind a hedge fund type of thing that I want to
  start."* **First version built and run the same night (owner-run):** `v2_experiments.py`, 27
  experiments from the literature, each pre-registered with a prediction before it ran, judged on
  held-out months and on 2012-2019 (built only after the rule for it was committed), against a
  multiple-testing bar. **What it found, plainly: nothing beats the current weighting, and the
  current weighting does not measurably rank next month's returns on public data** (IC ~0 in both
  eras; the same harness fed leaked data finds t 4.7, so it can see a signal when there is one).
  Proving a realistic edge from any monthly large-company record takes years (an IC of 0.05 needs
  about 7½ years at t >= 3). So what can honestly be shown today is the process and the evidence for
  each factor, not a proven return - `research/2026-10-15-research-engine-first-batch.md`. The
  nightly queue is `plan/research-engine.md`.
  **The bot's own rule, run the same way** (registered before it ran; note section 9): it lagged an
  equal-weighted S&P 500 in every period - by 0.72% a month in 2020-2023 (t -2.7), 0.22% in
  2023-2026 and 0.15% in 2012-2019 (neither distinguishable) - mostly because the ranking's top was
  weak in growth-led years. Expect the paper bot to match or lag RSP, not beat it, and always show
  its record next to RSP. Nothing in the bot was changed (its rules forbid tuning to a backtest).
  **Decided by the owner, 2026-10-15: no paid data.** *"my goal is to not spend any money if possible
  on top of a claude subscription."* The $199/yr delisted-price feed is off the table; the
  survivorship gap (42% of 2012's members missing, nine in ten because they left the index) is
  stated wherever a result depends on it, and free ways to narrow it are the nightly queue's job.
  **Public since the same night:** `research.html` ("Has it been tested?"), linked from the
  dashboard's hero, footer, search palette and track record, beside a Factor Lab button.

  **Progress (update this line each session):**
  - **2026-10-19 (06:00) - the engine grew by one idea, from a research note, and the idea's own
    prediction was wrong.** This is the part of the queue (item 6) that turns reading into tested
    claims: a Monday note ends with something falsifiable, it is written down and committed
    *before* it is run, and then it is run. Today's note asked whether the screener should be
    rewarding one-year revenue growth at all - the published research says no, and one quant shop
    has measured exactly that construction at **-2% a year** - so the registered test was "take that
    leg out; the ranking should not get worse."
    **It got worse** - by 0.007 of ranking accuracy in the held-out years, which is **the opposite
    direction to the one I predicted**. It does not clear the bar the rules set in advance (it needs
    t of 3, it reached 2.5) and it shows nothing at all in 2012-2019, so the verdict is **"not
    confirmed"** and nothing was changed on the strength of it - the rules forbid that in both
    directions. But the prediction is on the public record and so is its failure, which is the whole
    reason for writing predictions down.
    **No weight was changed, and the note explains why in a way that corrected itself twice.** The
    first argument it drafted was measured and refuted; the second was undercut by a paper found
    later in the day. What survives is the honest version: the screener's two growth metrics
    together are a bet on *revenue rising faster than the balance sheet*, documented practice
    rewards exactly that (FTSE scores it, S&P frames its quality screen on it), and the academic
    evidence for it is **absent rather than unfavourable** - the one paper that would settle it is
    paywalled and the major replication never tested it. Practice and research disagree, so the
    rule written in advance says the weight stays.
    **One cost worth your knowing:** registering an experiment forces a rerun of all of them,
    because the engine refuses to compare results produced by different code. Today that was 30
    runs - and the 06:00 session ran out of time after 15 of them, so its gates failed; an
    owner-run session finished the other 15 the same day. It is the right behaviour; it just means a
    session that adds an idea must budget for it.
    **Next in the queue is still 8(c)** - the positive control, deferred twice now, and the next
    13:00 or weekend session should take it rather than a fifteenth thing.
  - **2026-10-18 (13:00 midday) - the morning's work is finished and shipped, and finishing it
    caught a bug in the new check itself.** Read this bullet rather than the one below it: **the
    06:00 session failed its ship gates** and nothing it wrote reached the site. It had built all
    the machinery described below, but of the 29 results that had to be rerun to actually carry the
    new provenance it had rerun **2**, so the claim "all 28 ideas and the evaluation were rerun" was
    written ahead of the fact and four tests were red. The gates refused the merge, the work was
    kept on a branch, and this session recovered it - which is the system working as designed, not a
    setback.
    **What it took to finish:** all 28 ideas and the trading-rule evaluation rerun in one go on a
    clean tree, and then the thing worth telling you - **it was checked rather than assumed that no
    number moved.** 29 of 29 compared against their previous run, every stored statistic at full
    precision: **0 differences anywhere, largest 0.0.** That was the expected answer, since neither
    the data nor the method changed, but the public research page quotes these figures and "it
    should be identical" is how a silent change gets published.
    **The bug the rerun caught:** the "was the code edited when this ran" marker looked at the
    *whole* project folder - and a run writes its own results into that folder, so the second half
    of every rerun got branded unreproducible because of its own output. It had already mislabelled
    the trading-rule result that way. The marker now looks at exactly the files that can change a
    number, the same ones the refusal looks at, with a test that fails if anyone widens it back. The
    lesson is written into the plan: a check that includes its own output in what it checks will
    fire in normal use, and a warning that fires in normal use gets ignored.
    **Next in the queue: 8(c)** - record the positive control (the deliberately-leaked arm that
    scores t 4.7) as a result in its own right, plus a test that injects a signal of known size and
    confirms the engine measures it back. That is the evidence that "we found nothing" means there
    was nothing to find, rather than that the test cannot see.
  - **2026-10-18 (06:00 weekend) - every result now names the code, the weighting and the records it
    ran on, and the engine refuses to run from an edited tree.** `plan/research-engine.md` item 8(b).
    This is the part of "it has to be proven" that is about *someone else* being able to check it: a
    result nobody can rebuild is not evidence, and until today each result recorded only a short
    commit - which cannot say whether the code had been edited since that commit, which is exactly
    the state research code is usually in while it is being written. Each row now records the full
    code version with a `-dirty` marker, a fingerprint of every file that can change a number, the
    baseline weighting it started from, and a fingerprint of the monthly records' **contents**. The
    last one matters most: the old fingerprint covered only *who* was in each month, so a rebuild
    that fixed a metric or re-fetched the filings could change every number while the fingerprint
    said nothing had changed. And `run` now refuses to start at all from an uncommitted tree.
    **Nothing a reader sees moved:** all 28 ideas and the trading-rule evaluation were rerun on the
    same data so every published row can name its code, and no verdict, interval or figure changed -
    the rerun was for the provenance, not the numbers.
  - **2026-10-17 (06:00 weekend) - the comparisons are now apples-to-apples, and it changed nothing,
    which is worth knowing.** `plan/research-engine.md` item 8a. Until today each idea's score was
    measured over the companies *it* could rank and compared with the current weighting's score over
    the companies *the current weighting* could rank - two different sets - so part of every
    published difference was a difference in **who was ranked** rather than in the ranking. A narrow
    screen like free-cash-flow yield covers 348 companies a month where the current weighting covers
    440. Both sides are now measured on the companies they share, every result in the ledger says
    which way it was measured, and the report refuses to put the two kinds of row side by side.
    **All 28 ideas and the trading-rule evaluation were rerun; not one of the 28 verdicts changed.**
    The largest any comparison moved was **0.0068**, against a test that cannot reliably see
    anything below about **0.040** - so the fix was needed for the claim to be true, and nothing the
    page told you was resting on it. The public page now says plainly how the comparison is made.
    Record and measurements: `research/2026-10-15-research-engine-first-batch.md` section 11.
    **Next in the queue: 8(b)**, provenance hardening - every result recording the exact code and
    data that produced it, and refusing to run from an edited tree.


## The 2026-10-12 item (the Factor Lab) - text to 2026-11-19

- **2026-10-12 - The Factor Lab: turn the papers the screener cites into experiments a reader
  can rerun. First experiment built by an owner-run session; the nightly sessions grow it.**
  Owner, verbatim, after reading about agents that turned a reading list into a browser lab:
  *"this is something i read today and thought it was a really good idea, dont know how it would
  integrate with these tools (the screener and the trading bot), but it think it would be really
  cool"*, and then: *"do whatever you think would improve these tools please! just keep looking
  for improvements, then have also make a plan for the nightly sessions to dig even deeper!"*

  **The plan is `plan/factor-lab.md` - read it first.** In short: `lab.html` reproduces a cited
  paper's headline result from public data, shows the claim, the conditions, the difference from
  the paper and what is still unverified, and lets the reader change the conditions. The queue
  there is ordered so that every weight on the methodology page ends up with either an
  experiment or an honest "cannot be rerun on free data". **Friday is the lab's day** (one
  experiment, finished); Monday adds to its queue; the context-layer item kept the other days
  until it closed on 2026-10-20 (now under Done). The lab never sets a screener parameter and no scoring module may import it.

  The same plan carries the "dig deeper" half: use replication as an audit of the screener's own
  citations - every effect size the methodology quotes gets tagged *rerun in the lab*, *cannot
  be rerun on free data* or *not yet tried*.

  This item is about the screener. The trading bot is a separate repo that nightly sessions do
  not work on (CLAUDE.md "Downstream consumer"); its own next steps are in its `ROADMAP.md`.

  **Progress (update this line each session):**
  - **2026-10-26 (13:00) - the morning's proposal was reviewed and REFUSED, and the rule that
    refused it is a test now.** The full account is item 1 above and
    `research/2026-10-26-price-target-upside.md` §8. In short: the morning proposed swapping the
    scored *level* of analyst implied upside for a 90-day *change* in the consensus target. It
    survived the check the note put first (it is not a duplicate of the category's existing
    revision metric) and failed the one the note deferred - the swap takes
    `revisions ~ momentum` from **+0.2223 to +0.3778**, through the **~0.35** bar written down in
    September, and **94% of the new metric's overlap with the rest of the screener is momentum**
    against the incumbent's 22%. So it would move the duplication rather than remove it, which is
    the opposite of what the category's 10% slot is for.
    **This is the route working in the direction that matters most.** Both halves of it did their
    job: the morning did the research and declined to build its own proposal, and a different
    session measured the thing the proposal had only predicted - and the prediction was wrong, in
    a way worth remembering (it read a 30-day number and applied it to a 90-day metric; and a
    regression slope on levels does not bound a rank correlation, which is what a percentile
    screener actually scores).
    **Four of the four proposals this routine has produced have now been refused by their
    reviewer** (Growth, capex, the three free-data registrations, this one) - and each refusal
    measured something its note had not. That is the argument for the second reader, and it is
    worth noting that none of the four was refused for being badly researched.
    **Also from the review, now on the record rather than pending:** the three papers the note
    could not verify are verified from PDF bytes, with three corrections, and the strongest
    evidence on this metric is now *against* the level the screener scores - one direct test for
    it (1996-2004) against two later ones against it. §8.7 makes that the next Monday in this
    category, with its refutation criteria written before anyone looks, and explicitly refuses a
    sign flip in advance unless someone can show the reversal in a long-only book.
    Next in this category is still `short_interest_ratio`, its other 13.5 non-revision points.
  - **2026-10-26 (06:00) - the route from the lab to the weights used on the last unresearched
    metric in the composite, and it reached a conclusion I did not expect.**
    `research/2026-10-26-price-target-upside.md`. `price_target_upside` (13.5 of the Revisions
    category's 100 points, 1.35 composite points) was the only weighted metric with no research
    note of its own. Measured from 56 committed payload versions: between consecutive published
    runs the median consensus target moves **exactly 0.0000** while prices move up to 2.7%, 60-77%
    of targets are byte-identical, and at the **30-day** horizon the review cadence uses, **71.4%
    of the variation in implied upside is the price moving**, not analysts changing their minds.
    Within sector it is **-0.40** with risk-adjusted momentum, **-0.47** with Risk and **-0.007**
    with `fy1_revision_3m`, the only actual revision metric in its own category.
    **But the weight did not move, and the pre-written reading rule is why.** Da & Schaumburg
    (2011) is the one paper that tests this exact construct, and the decisive finding is that the
    **absolute** level earns 79 bp/month (t 0.86, not significant) while the **within-sector**
    level earns 177 bp/month (t 3.55) - "it is relative and not absolute valuations that convey the
    most information". The screener scores a sector percentile, so **it is accidentally on the
    supported side of the only distinction anyone has tested.** Literature partly supports it,
    practice (Barra's three analyst descriptors, the Zacks Rank's four components - all changes, no
    target level) does not use it at all; that is a disagreement, so the weight stays. Removing it
    would anyway move the median stock 4 ranks and no top-25 name.
    **One proposal left OPEN** for a reader who did not write the note: replace the scored *level*
    with a 90-day *change* in the consensus target scaled by price. Its first step is cheap and
    decisive - if that duplicates `fy1_revision_3m`, the honest outcome is no change.
    **Worth knowing for any future note:** Brav & Lehavy's 1997-99 estimate that 9% of a
    target/price gap closes per week predicts 31%/71%/91% closed by 4/13/26 weeks; my betas at the
    matching gaps are 24.5%/54.0%/82.3% - two independent estimates of the same slow adjustment,
    27 years apart, which is what rules out "it is just a Yahoo refresh cycle".
    Next in this category: `short_interest_ratio`, its other 13.5 non-revision points, also with no
    note and with a free data history already probed.
  - **2026-10-23 (13:00 midday) - the lab's two unchecked experiments now have their second
    reader, and both stand.** The morning deferred the lab for a proposal decision; by 13:00 no
    proposal was waiting (`open_proposals()` returns `[]`), and the thing the lab most needed was
    not an eighteenth experiment but a reader for the two that had none - 16 and 17, merged on the
    owner-run night of 2026-10-20. Both were checked three ways: the paper reopened and every
    reference figure re-read, every measured figure recomputed by an implementation that does not
    import `factor_lab`, and the pre-registration confirmed in `git log`. **All 21 reference
    figures are exact, every measured figure reproduces, and both counts (10 of 17, 1 of 4) are
    right** - so nothing needed correcting, and in particular the result that keeps Size at 5%
    rests on a number someone other than its author has now checked. Two things were corrected:
    this plan's row 7c called experiment 17 "experiment 15" (the JKP one), and two source
    observations are recorded where a later reader would otherwise read them as errors - OSAP's
    `SignalDoc.csv` names the wrong panel of Piotroski's Table 3, and one `open_details` figure is
    market-adjusted where the page's main comparison is raw. Details in item 2 above.
    **The lab's seventeen experiments are now all second-read.** The next experiment (18) is still
    unstarted; the next non-retrospective Friday is **2026-11-06**, and `lab_data.js` at ~2.9 MB
    still wants its first paint measured in the same session (item 7 above).
    **Update 2026-11-02 (owner-run evening): experiment 18 SHIPPED** - analyst revisions (Chan,
    Jegadeesh & Lakonishok 1996) on OSAP `REV6`, 3 of 3 within tolerances committed first
    (`4e1f6a7`); `research/2026-11-02-lab-revisions.md`. **Second-read the same night** by a
    different reader: every paper and measured figure reproduced without importing `factor_lab`,
    the pre-registration order verified; three wording fixes to the note, no figure changed.
  - **2026-10-23 (06:00 nightly) - deferred, and why; nothing started, so nothing is half-done.**
    This was the lab's Friday, but a proposal was waiting for a decision (1 day open) and both the
    prompt and item 1 above put that ahead of the day's focus. It was decided - all three of its
    registrations refused as written, test B accepted in substance - see item 1 and
    `NIGHTLY_LOG.md` 2026-10-23. **The next experiment is still unstarted**; any session that finds
    no proposal waiting can take one, and the next non-retrospective Friday is **2026-11-06**
    (2026-10-30 falls in an even ISO week, so the runner swaps in the retrospective). One thing for
    whoever takes it: the lab's own `lab_data.js` is at ~2.9 MB and item 7 above asks for its first
    paint to be measured - worth doing in the same session as experiment 18 rather than separately.
  - **2026-10-15 (owner-run) - three experiments in a day: 10, 11 and 12.** Profitability (Novy-Marx
    2013, 6 of 6 within), cash-flow yield (Lakonishok, Shleifer & Vishny 1994, 11 of 11) and accruals
    (Sloan 1996, 6 of 11 - the direction at half the size). Each says the same thing about the weights:
    supported in its paper's sample, weaker since, and weaker still weighted by market value (where
    large companies count most) - nothing to change. Experiment 13 followed at night: value and momentum together (Asness, Moskowitz &
    Pedersen 2013), 7 of 10 within, a hedge rather than a return since 2011. Thirteen experiments, 111
    of 162 figures within; the lab now has a button on the dashboard's front page. The research engine (the item at the top) tests weightings on
    2012-2026 history; the lab and the engine answer different questions - what a paper found, and
    what this screener's weighting would have done.
  - **2026-10-14 (06:00 nightly) - deferred for a red ship gate, nothing lost.** The day's work was a
    data defect published by the 02:00 run (see the context-layer item, now under Done, and `NIGHTLY_LOG.md`).
    The lab's next day is Friday **2026-10-23** either way - 2026-10-16 is a retrospective - and the
    next experiment is still profitability (Novy-Marx 2013). No proposal is waiting for a decision:
    `write_brief.open_proposals()` returns `[]` since the Growth refusal.
  - **2026-10-13 (06:00 nightly) - the Growth proposal was reviewed and REFUSED, and the page now
    says why the weight is where it is.** The waiting proposal (Growth 13% -> 10%, Quality 22% ->
    25%) measured one point; the review measured the curve, which is what separates the two
    readings of the same evidence. Net growth exposure moves steadily with the weight, crosses zero
    at **13.5**, and stays within 0.05 of neutral only between **11 and 16** - because Growth is the
    one category of eight whose score runs *against* the others. So its weight is the dial that
    sets whether the finished ranking leans toward growing companies or away from them, and at 13%
    it leans neither way. The proposal's own argument is that growth earns nothing; if that is
    right the supported position is none, and 13 points is what delivers none. **Cutting to 10
    would not remove a position on growth - it would take one**, in a direction the four studies do
    not agree on (two flat, one negative but on 1968-1990 five-year market-wide growth, and MSCI's
    modern estimate slightly positive). The methodology page gained **"Why the weight stays at
    13%"**, with six tests behind it - including one that fails if a published run ever stops being
    growth-neutral, so the sentence cannot quietly go stale. The sound half of the proposal
    survives and moved: the category's *composition* (one metric duplicating momentum, one arguing
    with Investment about the same fact, one duplicating Quality) is now the open Growth question.
    **Found while wiring that up:** the publish gate's tripwire only noticed tests that read the
    committed methodology file, not ones that read the generator - which is the better way - so
    **three further modules holding public page claims were never run before publishing**. All four
    are in the gate now (454 claims-gate tests, up from 371).
  - **2026-10-13 (owner-run, before the 02:00 run) - the nightly sessions are set up to carry
    this on, and a ninth experiment answers the question the review left open.**
    `prompts/nightly.md` now tells each session about the lab, the route from a finding to a
    weight, the proposal waiting for review, and how to read a paper so a note can be trusted.
    The Growth proposal was **pre-flighted** in a throwaway checkout: eight tests trip, and the
    note lists what each needs. A full rehearsal of the 02:00 publish with tonight's code passed
    the whole suite. Experiment 9 reruns Israel & Moskowitz (2013) in the largest fifth of
    companies: the paper's figures there reproduce, and momentum since 2000 is 2.7% a year
    (t 0.54) - the flat stretch is in the largest fifth as well. **Your morning brief now has a
    "Waiting for a decision" section** listing any proposal a session left for review, with how
    many days it has been open - today, the Growth one.
  - **2026-10-13 (owner-run, just after midnight) - the second question is researched too: value
    and momentum among large companies.**
    `research/2026-10-12-value-and-momentum-among-large-companies.md`. Value on book value was
    never reliably there among the largest stocks, but the screener weights earnings and cash-flow
    measures, where the published large-company evidence is better, and nearly all of that premium
    is on the side a long-only book keeps. Momentum is strong in every published study and has
    been flat among big companies on the public data since 2000 - a longer, lower stretch than
    any on record, which no published study yet covers. **No weight change is proposed:** the lab
    and the literature disagree, and the rule written beforehand says the weight then stays.
    The methodology page now tells a reader all of this instead of quoting all-stock figures.
  - **2026-10-12 (owner-run, near midnight) - the first of the five questions is researched:
    Growth's 13%.** `research/2026-10-12-growth-category.md`, from six papers and MSCI's and S&P's
    methodology documents. No published return premium stands behind growth as the category
    measures it; index providers treat it as a style, not a rewarded factor; and in this
    screener it works as a counterweight - the published ranking does not lean toward growth at
    all (correlation -0.01). The methodology page now says so. **FOR THE NEXT SESSION, ahead of
    the rotation: review section 7 of that note - a proposal to move 3 points from Growth to
    Quality - and decide it.** Build it with a changelog entry, or refuse it and write why; the
    note lists both the build steps and the case against. It was not made the same night on
    purpose: the 02:00 run was the first under the rebuilt momentum rule, and the note's author
    should not be its only reviewer.
  - **2026-10-12 (owner-run, past 23:00) - your question, "will the lab be used to improve the
    methodology?", has a written answer: `plan/factor-lab.md`, "From the lab to the weights".**
    In short: yes, through research notes and the changelog, never through code; five questions
    about the weights are ranked there (Growth's 13% first - nothing stands behind it that can
    be rerun). Experiment 8 shipped with it - the factors **among large companies**, which are
    all the screener ranks: value, investment and momentum are about half-size there, and
    profitability is the one that holds. The lab's front page now shows where each category's
    weight stands on that evidence. **No weight changed.** The same experiment also shows what
    a holder who shorts nothing keeps of each factor among big companies: nearly all of value
    and investment, about half of momentum, **under half of profitability** (the rest is weak
    companies lagging, which a long-only portfolio benefits from only by not owning them).
  - **2026-10-12 (owner-run, the end of the night) - the front page compares every headline
    result inside its paper's sample with the years since** (9 of the 10 that claim a direction
    were weaker afterwards), and **two methodology sentences were corrected to what the lab
    measured** (Risk: extra risk has not been paid for, rather than "one of the most robust
    findings in finance"; Size: the measured ninth-minus-tenth difference beside the caveat).
    Also new and not about the lab: `scripts/rehearse_publish.py` - see CLAUDE.md "Ship gates" -
    after a rehearsal showed tonight's page would have made the 06:00 baseline red.
  - **2026-10-12 (owner-run, the last of the night) - experiment 7 shipped: does market risk
    pay?** Fama & French (1992) on the library's beta deciles: **7 of 11 within**, and the paper's
    point - returns flat from the lowest-beta group to the highest - reproduces, in its sample
    and since. The Risk category now has two of its three measures behind it. The methodology
    page links every category to its experiments. **Seven experiments, 64 of 94 figures within
    tolerance, six of eight categories covered.** Next for a Friday: profitability (Novy-Marx
    2013) or the cash-flow-yield sort; next for a Monday: the research notes that
    `research/2026-10-12-factor-lab-findings.md` asks for (value since 2007 first).
  - **2026-10-12 (owner-run, the same night) - experiment 6 shipped: does a high earnings yield
    pay?** Fama & French (1992) on the library's earnings-yield deciles, the nearest public sort
    to the screener's own valuation measures. **7 of 11 figures within** tolerances written
    beforehand; returns rise step by step with earnings yield in the paper's sample and for
    decades after. **Since 2007 the gap has all but closed, and among large companies it
    reversed** - the finding that matters most for the screener's biggest category, written up
    in the research note (section 7) as a question for a Monday, not acted on. Six experiments;
    57 of 83 figures within tolerance across them.
  - **2026-10-12 (owner-run, the same night) - experiment 5 shipped: do riskier stocks earn
    more?** Ang, Hodrick, Xing & Zhang (2006), the study behind the Risk category, on the
    library's volatility-sorted fifths. **5 of 11 figures within** tolerances written
    beforehand. The calmest fifth earned about as much as the rest with half the volatility; the
    paper's large penalty for the most volatile stocks does not appear on the public version of
    the sort. Built on a generic layout, so the next sorted-portfolio experiments (profitability,
    earnings and cash-flow yield, accruals) need only a record and a paper's table. Five
    experiments now; six of the eight categories have something behind them.
  - **2026-10-12 (owner-run, the same night) - a front page, and an honest line on the
    methodology page.** The lab opens on an overview: four cards (verdict and one tested sentence
    each; 45 of 61 figures within tolerance across them) and a table of what stands behind each
    of the eight categories - including the two that have nothing yet (Growth, Risk) and the one
    that probably never will (Analyst revisions: its studies need estimate history that is sold,
    not published). The methodology page gained limitation 11: the value, investment and size
    factors have averaged close to zero since 2014. The plan's queue now names the library files
    for Risk, for value measured the screener's way, and for accruals.
  - **2026-10-12 (owner-run, the same night) - experiment 4 shipped, and it is the lab's first
    replication that mostly fails:** Fama & French (1992) on size, **3 of 11 figures within**
    tolerances written beforehand, because the paper's table covers only non-financial firms
    with accounting data and the public file covers every stock. It is published as a miss, with
    the reason. The page also answers the screener's own question - is there a size effect among
    large companies? - for any period: the ninth size group has out-earned the tenth by
    0.07-0.14% a month every way it was cut, but only the full century clears the usual bar.
    The lab also gained a readout on every chart and links that reopen an exact scene.
    **Next: experiment 5, profitability (Novy-Marx 2013)**, then 2b.
  - **2026-10-12 (owner-run, the same night) - experiment 3 shipped:** Fama & French (2015), the
    five factors behind the Valuation, Quality, Investment and Size categories. **20 of 21
    figures within tolerances written beforehand.** After the paper's sample (2014 on) none of
    the four factors is distinguishable from zero, and value is almost fully explained by the
    other four in both periods. `research/2026-10-12-factor-lab-findings.md` sets out what the
    three experiments mean for the screener and what they do not - it is the "dig deeper"
    starting point, and it asks for a Monday research note on what the value and size premia
    are now worth. **Next: experiment 4, a size premium among large stocks**, then 2b.
  - **2026-10-12 (owner-run, later that night) - experiment 2 shipped**, after the owner's "keep
    improving, whatever you think would benefit these tools, be creative and start making this
    great!": Daniel & Moskowitz (2016), "Momentum crashes" - the study the momentum regime rule
    is built on. **9 of 15 figures within tolerances written before the comparison was run**;
    every figure about what follows a market decline matches, and the six misses all describe
    the portfolio itself (the paper builds its own deciles). The page sorts each month into four
    market states and shades the panic state on the chart: at the screener's own thresholds,
    1927 to 2013, momentum averaged -1.68% a month in that state and +1.52% outside it. The lab
    now switches between experiments. **Next: experiment 3, the Fama-French five factors**, and
    2b in the plan (rerun experiment 2 on the authors' own portfolios, terms permitting).
  - **2026-10-12 (owner-run) - experiment 1 shipped:** Barroso & Santa-Clara (2015),
    risk-managed momentum, on the Kenneth French library's momentum deciles. **13 of 14 figures
    within tolerance** (Sharpe 0.54 -> 1.00 here against the paper's 0.53 -> 0.97); the
    risk-managed worst month is outside (-24.16% against -28.40%) and stays marked. After the
    paper's sample (2012 on) the Sharpe ratios are 0.20 and 0.37. `factor_lab.py`,
    `lab_engine.js`, `lab_data.js`, `lab.html`, `tests/test_factor_lab.py` (15).
    **Next: experiment 2, Daniel & Moskowitz (2016), momentum crashes** - the study the momentum
    regime rule is built on. Write its tolerances before running the comparison.


## The stale "(No open owner items...)" note, removed 2026-11-19

*(No open owner items. Calculation transparency and the premium redesign were finished on
2026-10-07 in owner-run sessions; their text is archived under **Done**. Owner direction that
night: *"I actually decided that I want you to finish up all UI/UX work in here tonight. And make
the nightly sessions just focus on what it was previously focusing on, making sure the methodology
is sound. And improving this tool overall every single night."* Nightly sessions return to the
weekly rotation: research-led methodology and overall improvement. The residuals that are data or
methodology, not design, are in CLAUDE.md "Current priorities": 0.9 (four methodology questions)
and the inputs the 11 history-based metrics need kept at fetch.)*