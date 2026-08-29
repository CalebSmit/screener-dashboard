# Morning Brief - Saturday 29 August 2026, 12:32

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **failed** - last ran today |
| Code session (6 AM) | **failed** - last ran today |
| Dashboard data from | 2026-08-28T02:00:03.803219 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, EIX, CF |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 27 rows across all horizons), newest 2026-08-21 |

## Things that needed attention

- Could not check out main.

## What changed in the repo

- `b12ad6b docs: the stagger is committed but not registered on the machine`
- `0e64354 docs: record the catch-up trigger defect and its fix`
- `373cef8 test: cover the loop collision, and parse the scripts for real`
- `4a49f95 tasks: stagger the logon triggers, and stop two scripts writing them`
- `1b056be fix: stop the two loops racing each other for git's index`
- `d45f3ae brief: data run 2026-08-29`
- `bf98aa7 brief: code session 2026-08-28`
- `493ddeb data: republish the dashboard with weights that add up`
- `71da63c docs: the printed weights are defaults, and a run may not use them`
- `6344a72 teach: show the weight each score was actually multiplied by`
- `14027d9 fix: record the weights the composite was actually built from`
- `5c0966d brief: data run 2026-08-28`
- `7dfe548 data: screener run 2026-08-28 - 502 scored, top: HST EXPE APA EIX CF`

## The session's own account

> 2026-08-29 - CATCH-UP. Not normally scheduled. Work the single highest-value item from the priorities list.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-08-28_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | **NO - `logs/datarun-2026-08-29_121115.log` died at "Could not check out main." That is today's work.** Last good run: `datarun-2026-08-28_020001.log`, "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **27 rows, newest 2026-08-21, 3 effective observations at `1m`** (8 raw) |
> | Priority 0 | Fixed 2026-08-24, holding |
> 
> **Tests:** before 825/825, after **853/853**
> **Data loop:** **was broken this morning - fixed**
> **Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
> **Rotation:** Saturday catch-up, so there was no nominal focus to defer. The
> data-loop failure would have outranked one anyway (`CLAUDE.md`: "fixing it is
> the highest priority work available, ahead of any feature").
> 
> ### Did - the catch-up trigger killed the data run it exists to protect
> 
> Both logs are stamped the same second:
> 
> ```
> logs/datarun-2026-08-29_121115.log   [12:11:15] === Data loop 2026-08-29 ===
> logs/nightly-2026-08-29_121115.log   [12:11:15] === Code loop 2026-08-29 ===
> ```
> 
> One second later the data loop was dead:
> 
> ```
> [12:11:16] [ERROR] Could not check out main.
> ```
> 
> **Root cause.** `register-tasks.ps1` gave both scheduled tasks an at-logon
> catch-up trigger with the *same* `PT3M` delay, so on the first logon of a day
> they start together. They share one working tree and git serialises nothing for
> them. At 12:11:16 the data loop ran `git checkout main` while the code loop was
> inside `Restore-Artifacts` running `git status` - which takes `.git/index.lock`
> to refresh the index. The data loop treated a transient lock as fatal and
> stopped **before running the screener at all**.
> 
> The at-logon trigger is the fix for priority -1, the dominant failure mode
> ("sessions do not start"). It had become a way to lose a run.
> 
> **Why nothing caught it.** Each script has a single-instance lock
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

