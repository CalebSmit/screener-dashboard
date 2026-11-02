# Morning Brief - Monday 02 November 2026, 13:08

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-10-30 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, HST, APA |
| Ranking's lean on growth | +0.049 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (40 rows, but overlapping windows are not independent; 105 rows across all horizons), newest 1m run date 2026-09-29 (34 days ago, bound 40) |
| Context signals | 16 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 452/453 within 0.5%; operating cash flow 429/437 within 0.5%; net income 420/449 within 0.5%; market value 450/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); SNDK (#5: a period behind its filings); PSX (#7: net income 1.28x after-tax operating income); EG (#8: a period behind its filings); DXCM (#9: operating cash flow 36% off the filing); DECK (#10: a period behind its filings); CF (#11: net income 16% off the filing); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **22 min** total; slowest `fetch.tickers` 646 s, `fetch.sec_roe_history` 195 s, `fetch.retry` 138 s; **45 tickers rate-limited** |
| Context coverage | options usable 490/499 (9 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide how the screener holds the ranking's growth exposure near** - open today - `research/2026-11-02-growth-exposure-drift.md`

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- SESSION DID NOT RUN: session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed
- Treating this as a failure, not as 'nothing to do'. No success marker will be written,
- so the catch-up trigger will retry rather than skipping the day.
- No commits, because the session never ran. This is a lost day, not a quiet one.
- === Run FAILED: session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed ===

## What changed in the repo

- `f67e3b1 Unstall the data loop: the growth-neutral claim, and two tests that read the wrong data`
- `e839aa3 brief: code session 2026-11-02 - SESSION DID NOT RUN`
- `c6ad39a brief: data run 2026-11-02`
- `5c56aaf brief: code session 2026-11-01-midday - SESSION DID NOT RUN`
- `710c9b0 brief: code session 2026-11-01-weekend - SESSION DID NOT RUN`

## The session's own account

> 2026-11-02 (owner-run, midday) - Unstalling the data loop: the growth-neutral claim, and two tests that read the wrong data
> 
> **Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-02_060000.log`
> ends "SESSION DID NOT RUN: ... OAuth session expired and could not be refreshed", as did the
> 11-01 06:00 weekend and 11-01 13:00 sessions. `claude auth status` reports `loggedIn: false`;
> `~/.claude/.credentials.json` was last written 2026-11-01 05:43. **Logging in is the owner's to
> do** (credentials) - he said he will tonight; until then every scheduled code session fails the
> same way within two seconds. Data loop published? **no** - `logs/datarun-2026-11-02_020000.log`:
> HEALTH: PASS, both payloads parse, then "PUBLISHED CLAIMS: FAIL" (2 tests) and "Run discarded";
> the live site and `signals.json` are still the 10-30 run. Evidence base at `1m` = **40 rows,
> newest 2026-09-29 (34 days ago, bound 40), 5 effective**. Priority 0: `allow_auto_apply`
> **false**, 5 of 8 effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12,
> **age 21 days**) - not advanced; a stalled data loop outranks it.
> **Tests:** before **1 failed** at the runner's baseline (`test_special_item_note::
> test_the_published_payload_agrees_with_the_filings`, inherited); after **3,606 passed / 3 skipped**
> (two of the skips are the new premise skips below, until tonight's payload carries the stamp).
> Dry run exits 0, both payloads parse. `scripts/rehearse_publish.py` against the refused run
> 1957c294714c with this tree: **PASSED** (362 page tests, published claims, files restored).
> 
> ### The three failures, and what each was
> 
> 1. **`test_the_ranking_really_is_close_to_growth_neutral` - real.** The composite's Spearman with
>    the Growth score was +0.058 against the page's "sits close to neutral" band of 0.05. Not one bad
>    run: across the fifteen published runs since the 2026-10-13 review it drifted steadily from
>    -0.009 to +0.049 (10-30), and the weight at which the ranking is neutral moved from 13.5 to
>    about 10 (the review's own sweep script, rerun). The gate refused the run as designed. **The
>    weight did not move** - re-fitting it to a point that moved four points in three weeks is a
>    rule to design, not a weight to set. The paragraph was rewritten - the test's own named
>    alternative: it now states the review's reading, the drift and the 11-02 reading, and points at
>    **`research/2026-11-02-growth-exposure-drift.md`, a new OPEN proposal** (re-fit on a schedule,
>    constrain the exposure as MSCI does, or state the lean). The old sentence is in
>    `claims.FORBIDDEN`; the brief now carries a "Ranking's lean on growth" row every morning.
>    CLAUDE.md and DECISIONS.md updated.
> 2. **`test_valuation_history::test_the_published_blocks_are_internally_consistent` - the test.**
>    First run of a month, so October's month-end close is the run's own price and five stocks'
>    "today" sat within half a basis point of their last month. The tolerance `now + 0.00005`, done
>    in floating point, came out below the stored month (0.09985 + 0.00005 = 0.0998999...). Now
>    compared in basis points with the stored precisions (+-0.5bp per month, +-0.05bp for today).
>    0 of 743 blocks fail on the refused run, 0 of 694 on the published one; a one-month-off mutant
>    is caught exactly as often as before (679 of 1,486 vs 681 - ties are the limit in both).
> 3. **`test_special_item_note::test_the_published_payload_agrees_with_the_filings` - the premise.**
>    The data loop refreshed the weekly SEC cache at 02:16 and then refused its publish, so the
>    committed 10-30 notes were compared with 10-Qs filed after them (11 notes "missing" - AON,
>    BIIB, CARR...). The payload now records the cache it read (`sec_facts`: rows, newest filing
>    date - `generate_dashboard._facts_vintage`), and this test and its twin in
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

