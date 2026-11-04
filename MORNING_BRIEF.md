# Morning Brief - Wednesday 04 November 2026, 02:22

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-04 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, SNDK, APA |
| Ranking's lean on growth | +0.051 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 111 rows across all horizons), newest 1m run date 2026-10-02 (33 days ago, bound 40) |
| Context signals | 18 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 447/448 within 0.5%; operating cash flow 414/431 within 0.5%; net income 406/444 within 0.5%; market value 449/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 2 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); DXCM (#6: operating cash flow 36% off the filing); HST (#7: net income 1.46x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); CF (#12: net income 16% off the filing) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 659 s, `fetch.retry` 128 s, `context.sec_insider` 118 s; **40 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (29 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide whether to fold the Investment category (`asset_growth`, 5%) into Quality at the weight-preserving 18.52% of a 27% Quality (section 3) or keep it standalone, as this note recommends, with the...** - open 2 days - `research/2026-11-02-investment-inside-quality.md`
- **decide whether to log (not show, not score) one line per stock - the number of 13F filers holding it and the change on the prior quarter, dated by quarter end and filing date - as a context-log...** - open 1 day - `research/2026-11-03-following-the-smart-money.md`
- **decide whether the options panel's ATM implied volatility and put skew should be computed from our own closing mids with QuantLib instead of Yahoo's `impliedVolatility` (measured 2026-11-03, section...** - open 1 day - `research/2026-11-03-open-source-quant-stack.md`
- **register in `research/experiments/registry.yaml`, before running it, a research-engine experiment that drops `short_interest_ratio` from Revisions (and a second that scores FINRA days to cover from...** - open 1 day - `research/2026-11-03-short-interest-days-to-cover.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 5 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 12 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `bd1702c data: screener run 2026-11-04 - 499 scored, top: EXPE VLO MU SNDK APA`
- `9a289d9 OWNER_FOCUS: the night's later items`
- `12fc0a6 Research: days to cover (short_interest_ratio), Revisions' last unresearched leg - weight kept, experiment proposed`
- `3a7487a CLAUDE.md priority 6 and NIGHTLY_LOG: the M4 answer and the night's later items`
- `cddf0f0 Investor profiles M4: a client-side reweight can reproduce the engine - measured, with two conditions`
- `cf3d3c5 Queue: 37 named-ticker references in payload-reading tests, measured with a recipe for a daytime fix`
- `3e7e437 Rank chart: no literal universe size - the page's last '|| 500' reads the universe being run`
- `69a6a5a CLAUDE.md: experiment 18's second reading is done (2026-11-02), not pending`
- `4ba0eea Research engine: rerun all 32 rows on the new code hash (no result changed); log and CLAUDE.md`
- `24cc2f2 Publish-gate checks: no assumption a good run can break`
- `f02cf1b Universe phase 1, first slice: every size constant reads universe.py`
- `e3d1175 NIGHTLY_LOG and OWNER_FOCUS: the owner-run night; Coming up gains 2026-12-04`
- `27aafe2 Claims audit: the dashboard sweep is closed - 0 open rows`
- `8bfcd65 Claims audit: the holdings rank strip and the phone table's pages (2 open)`
- `c948214 Review cadence: the turnover figures are measured at every build and name their basis`

## The session's own account

> 2026-11-03 (owner-run, night) - "keep finding improvements!! be creative. work until the 2 am session"
> 
> **Health (rule 8, all five):** as the evening entry above - no scheduled session or data run since.
> Last code session ran? yes - `logs/nightly-2026-11-03_130000-midday.log` (completed). Data loop
> published? yes - `logs/datarun-2026-11-03_020000.log` (HEALTH: PASS). Evidence base at `1m`: 5 of 8
> needed (43 rows), newest 1m run date 2026-10-02 (32 days, bound 40). Priority 0: `allow_auto_apply`
> **false**. Top open roadmap item: **0.13, The Factor Lab** (age **22 days**).
> **Tests:** 3,879 passed / 3 skipped at the last gate run. Tags `good/2026-11-03-owner-6` to `-12`
> and this entry's. Worked in a worktree outside OneDrive; pushed only while no session ran; the
> 20:00 option refresh ran tonight's code and was measured (evening entry, item 7).
> 
> ### Did
> 
> 1. **QuantLib against Yahoo's implied volatility: pre-registered, measured, bar MET, decision OPEN**
>    (`research/2026-11-03-open-source-quant-stack.md` sections 1a-1b; pre-registration committed as
>    `57958d5` before any number). On 51 stocks: our American IVs from closing mids sit a median
>    **3.16 points below** Yahoo's at the money (90th pct 6.09); an independent CRR tree reproduces
>    QuantLib on **10 of 10** contracts; the put skew's **sign differs for 17 of 30**. Diagnostic: no
>    simple construction reproduces Yahoo's field (last trades a median 9.8 h old), and Yahoo's skew is
>    negative for **28 of 48** stocks against **5 of 48** from the closing quotes. ~14 s a refresh if
>    adopted. This session wrote the proposal, so a scheduled session decides; the brief lists it.
> 2. **The options panel's anchoring, completed and seen on the live page.** After the evening fix the
>    card withheld a pre-fix move but still printed its "at the money" volatility and skew - NOW's
>    from the 150 strike with the stock near $109. The move, the ATM IV and the skew legs share one
>    anchor, so a reading without its recorded price withholds all three, in the generator, on the page
>    (`optAnchored`) and in `context_eval`; the skew's explanation and effect sizes now appear only
>    beside a skew. Looked at the live card at 375px after each change. Simulated tomorrow's 02:00 read
>    of tonight's cache: **467** stocks will show option figures with their divisor, 12 older readings
>    withheld; NOW will read +/-11.5% divided by $109.16 at the 110 strike. 12 of 12 mutations.
> 3. **The claims audit's dashboard sweep is closed: 38 -> 0 open rows.** Every computed sentence on the
>    page now carries a verdict a test enforces - the workings' weight lines and points equation for
>    every row of every stock (the first draft rebuilt the line from its own copy of the template and a
>    mutation survived it), the Piotroski / surprise / ROE lines redoing their arithmetic from the
>    inputs published beside them, the history and Reporting Soon counts, the price-target strip (its
>    mean-target upside equals the scored one for 496 of 496), the share-class gap, the compare
>    sentence, the holdings strip, the phone table's pages, and the snapshot's net-income growth and
>    days to cover against the run's own fetch. New claims: `drilldown.percentile_convention`,
>    `drilldown.price_targets`, `history.rank_counts`, `dashboard.reporting_soon_counts`,
>    `dashboard.review_cadence_turnover`, `dashboard.phone_table_pages`. Four mutations survived first
>    drafts on today's data (every stock ranked today, no report exactly 8 days out, every company
>    scored, a template copied into the test) and were caught by hand-built cases.
> 4. **A stale figure found and fixed: the review-cadence turnover.** The note printed September's
>    "122% at every run against 24% monthly" as "this screener's own history"; re-measured the same way
>    on the longer record it is **149%** and **28%**. `history.review_turnover` now measures both at
>    every build over every stock each run ranked, and the note names its basis (changelog 2026-11-03
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

