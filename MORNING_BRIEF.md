# Morning Brief - Wednesday 14 October 2026, 08:16

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-14T02:00:10.255065 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, INCY, HST, APA, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (29 rows, but overlapping windows are not independent; 81 rows across all horizons), newest 2026-10-07 |
| Context signals | Context signals: 5 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 486/501 (15 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `c45178e brief: code session 2026-10-14-midday`
- `dfd94fb routine: two extra scheduled code sessions - 13:00 daily and weekend mornings`
- `4f0c64c brief: code session 2026-10-14`
- `f8480a6 log: the four gates for 2026-10-14`
- `2f42754 log: the 2026-10-14 session, and the two constraints it added`
- `17249d4 publish: regenerate today's page so it stops saying two false things`
- `1daf857 changelog: the dividend-yield defect, its fix and the second derivation it exposed`
- `3e27f14 health: refuse a run where a weighted metric vanished across the universe`
- `53d9c5f churn: a metric the whole universe lost is a source event, not one stock's`
- `fa77fe7 fix: the source dropped dividendRate - restore the yield from the trailing rate`
- `34859a8 brief: data run 2026-10-14`
- `ccbc795 data: screener run 2026-10-14 - 501 scored, top: EXPE INCY HST APA SNDK`
- `d7c1b71 brief: code session 2026-10-13`
- `1a027ac growth: record the refusal in the docs, and generalise the open-proposal test`
- `99ec010 growth: record the decision on the page, the lab scorecard and the plan`

## The session's own account

> 2026-10-14 (owner-run, morning) - two extra scheduled code sessions a day
> 
> **Health (rule 8):** both loops ran today - `logs/datarun-2026-10-14_020002.log` ends "Data loop
> complete", HEALTH: PASS; `logs/nightly-2026-10-14_060001.log` ends "Run complete: shipped to
> main". Evidence base at `1m`: **29 rows, newest 2026-09-14 (30 days, bound 40), 4 effective**.
> Priority 0 unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, age 2
> days).
> **Tests:** before 2274 passed / 0 failed; after 2285 passed / 0 failed.
> 
> ### The owner asked
> *"i want you to be doing more, scheduling more runs. I want to speed up the development of these
> tools please"*
> 
> ### Did
> - **`Screener Midday Session` (13:00 every day) and `Screener Weekend Session` (06:00 Sat-Sun)**,
>   both `scripts/nightly-screener.ps1` with a new `-Slot`. Fourteen code sessions a week instead
>   of five. Same gates, lock, merge and tag as the 06:00 session; their own once-a-day markers,
>   logs (`nightly-<stamp>-midday.log`) and focus (`QUEUE`: carry on from the morning's Next).
> - **They are built not to collide**, because the owner works in this same folder:
>   `logs/.owner-session` stands every slot down before any git command; an extra slot that finds
>   the tree dirty or off `main` stands down without stashing or switching anything (the 06:00
>   slot still recovers real leftovers); a 429 writes `logs/.usage-limited` and the extra slots
>   stand down for 24 hours. No logon trigger and no late catch-up for either.
> - **Exercised on the real runner** before registering: on this branch with a dirty tree,
>   `-Slot midday -DryRun` stood down with "the tree is dirty on 'owner/2026-10-14-more-sessions' -
>   ... Nothing was stashed or switched"; with the owner marker present it stood down before any
>   git command; `-Slot bogus` refused. `git stash list` empty and no lock left behind after each.
> - `tests/test_extra_sessions.py`, 11 tests; the task-count test now counts five. The brief's row
>   reads "Code session (latest)". CLAUDE.md and `prompts/nightly.md` say a session may not be the
>   first today, and that owner-run sessions create and remove `logs/.owner-session`.
> 
> ### Methodology changed
> - None.
> 
> ### Gates
> 1. **2285 passed / 0 failed** (baseline 2274 / 0). 2. `run_screener.py --dry-run`
> **DRY-RUN PASSED**. 3. payload and lab files pass `node --check`. 4. `git status --porcelain`
> clean after the commit.
> 
> **Machine state, rule 11:** `scripts/register-tasks.ps1` run after the merge and all five tasks
> read back with `Get-ScheduledTask` - recorded in the commit that follows this one.

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

