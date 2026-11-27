# Morning Brief - Friday 27 November 2026, 06:42

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## Coming up, only you can do it: the Claude CLI login runs out 2026-12-03 09:01 (in 6 days)

The code sessions run through the Claude Code CLI on this PC, and its login
lasts about 30 days. When it runs out, every scheduled session fails within
seconds until you sign in again. Doing it early loses nothing - open a
terminal on the PC and run:

    claude.cmd auth login

## At a glance

| | |
|---|---|
| Data run (2 AM) | **refused by its publish gates (published-claims check failed)** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-26 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, APA, MPC, SNDK |
| Ranking's lean on growth | +0.012 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 18 commit(s) changed what a reader sees, 3 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (59 rows, but overlapping windows are not independent; 147 rows across all horizons), newest 1m run date 2026-10-26 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 459/459 within 0.5%; operating cash flow 445/446 within 0.5%; net income 433/457 within 0.5%; market value 452/487 within 5%; 5 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 1 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#6: net income 1.29x after-tax operating income); PSX (#10: net income 1.28x after-tax operating income); DXCM (#14: operating cash flow 36% off the filing); CF (#15: net income 15% off the filing); ES (#18: a period behind its filings); TPR (#19: equity 5.2% of assets) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 624 s, `snapshot` 152 s, `fetch.retry` 111 s; **30 tickers rate-limited** |
| Code-session login | **runs out 2026-12-03 09:01 (in 6 days)** |
| Context coverage | options usable 481/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-12-04 (in 7 days) - The expected move's first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. (The ATM volatility and the skew restart under our own construction from the 2026-11-27 log - see 2026-12-28.) `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 10 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## Things that needed attention

- ATTENTION - CLI LOGIN EXPIRES 2026-12-03 09:01 (in 6 days) - after that every scheduled code session fails until the owner runs `claude.cmd auth login` in a terminal and signs in; doing it before then loses nothing.
- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.

## What changed in the repo

- `a5a54b8 NIGHTLY_LOG 2026-11-27: gate results`
- `ec96ecd NIGHTLY_LOG 2026-11-27: retrospective - the refused run and held-back branch fixed, gate 5's list derived, one retro rule`
- `66a59ba CLAUDE.md: closed items move below the open ones; 0.10 closed; a stale v1 claim corrected`
- `16a71f4 Retrospective schedule: one rule, keyed to the last retrospective that ran`
- `42e9ef6 Gate 5's ENGINE CHANGED list is derived from what the run imports, not hand-kept`
- `c0273a9 Sheet tests settle on the detail shard too: its pct redraw moves the sections the tab test clicks`
- `bf0bdb4 Test states the reading's IV construction instead of inheriting it: the 02:00 run was refused for a correct page`
- `d1ea09c brief: data run 2026-11-27`
- `5e8fb20 brief: code session 2026-11-26-midday`
- `b0bffa5 NIGHTLY_LOG 2026-11-26 (13:00): gate results`
- `ac876f1 NIGHTLY_LOG 2026-11-26 (13:00): pct leaves first paint`
- `5ebe06a docs: pct shard - plan progress, inventory, CLAUDE.md DETAIL_KEYS (and the corrected premise: the movers strip and compare drawer read neither raw nor pct), owner item`
- `b0e9cc7 wip: pct shard - waiting workings keep the weight-table note and the not-used list; browser helpers wait for the shard before reading rows`
- `a31a14d wip: pct shard tests (audit tripwire, waiting, redraw keeps open sections, CSV waits); fix the detail loader's retry loop on a failed load`
- `22225f2 wip: pct shard - the remaining readers fold the shard; score-tile line keeps its tested wording`

## The session's own account

> 2026-11-27 - RETROSPECTIVE. Today's refused run and yesterday's held-back branch fixed first; gate 5's warning was a hand-kept list; two retrospectives seven days apart
> 
> **Health (rule 8, all five):** last code session ran? **held back by its gates** -
> `logs/nightly-2026-11-26_130001-midday.log` (gate 1: one browser test timed out; work kept on
> `origin/nightly/2026-11-26`). Data loop published? **no** - `logs/datarun-2026-11-27_020001.log`, HEALTH:
> PASS, then **refused by the published-claims gate (1 failed)**; fixed this session, so the 18:00 retry
> publishes once this branch is merged. Evidence base at `1m`: **59 rows, newest `run_date` 2026-10-26 (32
> days ago, bound 40), 5 effective** of 8. Priority 0: `allow_auto_apply` **false**. Top open roadmap item:
> **0.13, the Factor Lab** (opened 2026-10-12, **age 46 days**; a standing programme, last advanced **2 days**
> ago, bound 21); also open 0.14 (44 d) - 0.10 closed today, below. Shipped to readers: 18 commits, 3
> methodology changes in the last 7 days.
> **Tests:** runner baseline **4564 passed / 1 skipped** (green). After merging the held-back branch and both
> test fixes: **4573 passed / 1 skipped**; final-tree figures under "Gates". No owner-session marker; no stash.
> 
> ### Did (before the retrospective): the refused data run and the held-back branch
> 
> **1. The 02:00 run was refused for a correct page.** One failure:
> `test_context_cards_printed.py::test_a_partial_reading_marks_its_figures[partial-False-True]`. The test
> patches option fields onto the first stock and kept that stock's own `ois` (which construction its implied
> volatilities use). The 11-26 06:00 session switched `PUBLISHED_IV_CONSTRUCTION` to `own_mid`, and **today's
> run was the first one computed under it**, so the inherited `ois` became `own_mid`. The page then
> correctly left off the caveat "the figures below are the data source's own, worked from last trades".
> The test now states `ois` and covers both cases: a partial reading from the source gets the caveat, and
> one with `own_mid` does not. Commit `bf0bdb4`.
> 
> **2. The 11-26 13:00 branch (the `pct` shard, -14.9% first paint) was held back by gate 1:**
> `test_sheet_round_three.py::test_a_chosen_tab_stays_lit_at_the_end_of_the_sheet` timed out under
> `-n auto`. **It was real, not a flake.** Since `pct` moved into `dashboard_detail.js`, the file's arrival
> redraws each category's workings and moves the sections. The test's `_settled` helper waited for the
> context payload's redraw but not this one. That session had fixed the same race in the unscored-tile
> test and missed the tab test. Merged `origin/nightly/2026-11-26` into this branch and made `_settled`
> call `sheet_warm.detail_loaded(page)` too (merge `7d5033d`, fix `c0273a9`). The module passes serially (18/18) and the full parallel
> suite is green (4,573). **Rehearsed against today's refused run (`4f43d5f75e77`): PASS**, with the claims
> gate at 2,443 passed.
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 13 log entries, 2026-11-20 13:00 to 2026-11-26 13:00 - 11 scheduled, 2 owner-run**
>   (11-22 evening phone pass, 11-23 evening unjam + phone rows). **2 of 14 code slots never started**
>   (11-23 06:00 and 13:00: `git pull --ff-only` failed on a local `main` diverged by a data-run push that
>   failed). **1 session held back** (11-26 13:00, above).
> - **Genuinely valuable: 11 of 11 scheduled, 2 of 2 owner-run | Churn: 0 | Failed gates: 1** (11-26 13:00
>   gate 1, a real race). **Data loop: 2 of 6 weekday 02:00 runs refused** (11-20, 11-27), both by the
>   claims gate and **both with a correct page and a stale test**. In both, the change the run exposed
>   had shipped by day and changed what the run *computes*, and gate 5 could not see it. 11-23 was jammed
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

