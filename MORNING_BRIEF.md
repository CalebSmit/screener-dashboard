# Morning Brief - Saturday 31 October 2026, 07:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-30 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, HST, APA |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (40 rows, but overlapping windows are not independent; 105 rows across all horizons), newest 1m run date 2026-09-29 (32 days ago, bound 40) |
| Context signals | 16 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 375/376 within 0.5%; operating cash flow 361/365 within 0.5%; net income 353/372 within 0.5%; market value 439/484 within 5%; 14 scored a quarter behind a filing over 30 days old; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#5: net income 1.46x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); CF (#14: net income 16% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 663 s, `fetch.retry` 140 s, `context.sec_insider` 96 s; **42 tickers rate-limited** |
| Context coverage | options usable 490/499 (9 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Things that needed attention

- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_forward_return_download_skip.py::test_skipping_changes_no_output_only_the_downloads
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.

## What changed in the repo

- `f2d817c log: 2026-10-31 weekend - a red main fixed at its root, a clock-read sweep, and four open audit rows`
- `57a15cb claims audit: the trading rule's monthly figure stops being a wildcard, and its two coverage years`
- `a306d32 claims audit: the two open rows the plan called sharpest - the clamp bullet and the switching margin`
- `7cc51ef tests: the clock-read sweep - three red gates in three weeks, one guard`
- `e2816d5 tests: a red main - the download-skip test's scenario came from the clock, so it failed every weekend`
- `52e1bcb brief: code session 2026-10-30-midday`
- `7a21ec2 log: 2026-10-30 (13:00) - the TTM-fill proposal refused as written, and the methodology page's computed sentences swept`
- `6c59f28 claims sweep: the methodology page, and a new published sentence is invisible to 3,566 tests`
- `3c2217e decide: correct a count in the review - six one-quarter stocks, not eight`
- `c5eee94 decide: the TTM-fill proposal - REFUSED as written, and its own case understated`
- `c13b46c wip: the measurements for the TTM-fill review`
- `4ca24b4 brief: code session 2026-10-30`
- `2a40afc log: 2026-10-30 RETROSPECTIVE - a red main found at baseline, and the blind spot that let it sit`
- `d00e2f5 rotation: Wednesday's subject is whatever changed the methodology, not only Monday's note`
- `221b2f4 process: CLAUDE.md 1036 -> 959; the fifth asymmetry scored; the step-1 enumeration as a tool`

## The session's own account

> 2026-10-31 (06:00 weekend) - WEEKEND. A red `main` fixed at its root, and four of the audit's open rows
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-30_130001-midday.log`,
> ends "shipped to main". Data loop published? **yes** - `logs/datarun-2026-10-30_020001.log`,
> "Data loop complete", HEALTH: PASS, signals written (40 names, 8 gates, all ran). Evidence base at
> `1m` = **40 rows, newest 2026-09-29 (32 days ago, bound 40), 5 effective** - steady state, one day
> further on than yesterday's 31 as expected. Priority 0: `allow_auto_apply` **false**, 5 of 8
> effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 19 days**) -
> not advanced; the next lab Friday is **2026-11-06** and no experiment is half-built. (0.10 is
> older - opened 2026-10-07, **age 24 days** - and its remainder is Sharpe and Sortino at weight 0,
> which CLAUDE.md says to do only with nothing better.)
> **Tests:** before **3,569 passed / 1 failed / 1 skipped** (RED on arrival, not this session's
> doing), after **3,594 passed / 1 skipped**. Dry run exits 0. `scripts/rehearse_publish.py` PASSED
> (the diff touches the methodology generator): 362 page tests, published claims 1,834 passed,
> published files restored byte-for-byte, all four artifacts unchanged.
> **Owner queue / rotation:** the red baseline first, as Baseline requires - then `OWNER_FOCUS.md`
> item 2 sub-item 3 ("test the tests"), whose plan names the work. No proposal is waiting
> (`write_brief.open_proposals()` returns `[]`). Nothing deferred for a gate or a loop.
> `git stash list` shows `auto-rescue 2026-10-28_060001`, **still empty** (`git stash show -p` prints
> nothing) - left in place, as every session since has. No owner-session marker.
> 
> ### Did
> 
> **1. The red `main`, and its cause was the calendar.** `tests/test_forward_return_download_skip.py::test_skipping_changes_no_output_only_the_downloads`
> failed at baseline, with no commit since 2026-10-26 touching anything it covers. It built its
> snapshot dates from `datetime.now()`, and **it cannot fire the skip it tests on a Saturday or a
> Sunday**: with an empty performance history the only skippable download is the one whose 1-week
> end is today, that needs a *market-day* snapshot exactly seven days earlier, and seven days before
> a weekend is a weekend. Today is the first weekend since the test shipped on Monday 2026-10-26.
> 
> The scenario is now **a fixed week, Monday to Sunday, parametrized** - every date (snapshots,
> prices, and `_close_possible_since`'s notion of today) comes from the frozen date. It also warms
> the history up first, which is both what makes the skip reachable on all seven days and the shape
> the 02:00 loop actually meets: a run date's shorter horizons were recorded weeks ago, so the
> horizon still due is the one whose end is at hand. With an empty history the clock-built version
> exercised the skip through **one of its sixteen snapshot dates**.
> 
> **Mutation-checked, and the first draft was found wanting.**
> `research/measurements/2026-10-31-skip-test-teeth.py`, five mutations of the production rule: the
> rule always True, always False, `min` -> `max` over the due horizon ends, the scan starting a day
> late, and the weekend counted as a session. The first draft **missed `min` -> `max`** - the
> warm-up leaves exactly one horizon due per date, so the two agree - and the whole suite missed it
> too: **3,563 passed with the mutation applied.** So
> `test_the_skip_reads_the_earliest_due_end_not_the_latest` covers the case that separates them, a
> date first processed late with its 3-month end out of reach and its 1-week end long past (which is
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

