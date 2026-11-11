# Morning Brief - Wednesday 11 November 2026, 07:38

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **failed** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-11-10 19:38 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, HST, MU, EXPE |
| Ranking's lean on growth | +0.063 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (48 rows, but overlapping windows are not independent; 119 rows across all horizons), newest 1m run date 2026-10-09 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Context coverage | options usable 448/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (**due 2 days ago - still listed**) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 5 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- Network not ready (attempt 1/10). Waiting 30s...
- Network not ready (attempt 2/10). Waiting 30s...
- Network not ready (attempt 3/10). Waiting 30s...
- Network not ready (attempt 4/10). Waiting 30s...
- Network not ready (attempt 5/10). Waiting 30s...
- Network not ready (attempt 6/10). Waiting 30s...
- Network not ready (attempt 7/10). Waiting 30s...
- Network not ready (attempt 8/10). Waiting 30s...
- Network not ready (attempt 9/10). Waiting 30s...
- Network not ready (attempt 10/10). Waiting 30s...
- No network connectivity after 5 minutes. Skipping today's run rather than publishing synthetic data.
- Could not fetch origin/main to publish the brief: fatal: unable to access 'https://github.com/CalebSmit/screener-dashboard.git/': Could not resolve host: github.com
- Brief not published; set aside in logs/MORNING_BRIEF.unpublished.md and MORNING_BRIEF.md restored, so the tree stays clean for the next session.
- Required tool 'claude' is not on PATH. Aborting.

## What changed in the repo

- `0c4b002 brief: data run 2026-11-11`
- `716b43e log: the retry task registered and its first run published Nov 10's close`
- `a426b51 brief: data run 2026-11-10`
- `c577494 data: screener run 2026-11-10 - 499 scored, top: VLO MPC HST MU EXPE`
- `9842618 A refused data run is retried the same evening; the runner updates a stale CLI`
- `901c2ea brief: code session 2026-11-10-midday - SESSION DID NOT RUN`
- `5b058e3 brief: code session 2026-11-10`
- `4b06504 log: 2026-11-10 - the advice ban read a company name, and the fingerprint read a weekly cache`
- `febe5f5 fix: the mask read window.D, which the page does not have - so it masked nothing`
- `b7f6cdc rerun: the ledger level again, and the measurement that the filings refresh moved nothing`
- `52d2a68 rerun: research ledger rows 25-30 of 32`
- `d8ba024 rerun: research ledger rows 19-24 of 32`
- `3f05095 rerun: research ledger rows 13-18 of 32`
- `6838462 rerun: research ledger rows 7-12 of 32`
- `e891914 rerun: research ledger rows 1-6 of 32 at the new fingerprint`

## The session's own account

> 2026-11-10 (owner-run, evening) - the site stayed stale because a refused run waits a day; it no longer does
> 
> **Health (rule 8, all five):** last code session ran? **no** - `logs/nightly-2026-11-10_130001-midday.log`:
> "SESSION DID NOT RUN: ... Claude Code 2.1.229 does not support this model; version 2.1.290 or newer is
> required" (the 06:00 session before it shipped). Data loop published? **no** - `logs/datarun-2026-11-10_020001.log`,
> HEALTH: PASS, then the published-claims gate refused it (`test_reporting_soon_is_a_calendar_not_a_leaderboard`,
> Best Buy's name read as advice; fixed by the 06:00 session). The live page is still the **2026-11-06 18:34**
> run. Evidence base at `1m`: **45 rows, newest 2026-10-06 (35 days ago, bound 40)**. Priority 0:
> `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **29 days**).
> **Owner direction (chat):** *"how did the run go this morning, why did the dashboard not update. make sure
> this doesnt happen again"*.
> 
> ### What happened, and the pattern
> 
> Three of the last five weekday data runs (11-06, 11-09, 11-10) were refused by the published-claims
> gate, each for a fault in a test's premise (a rank read by position; a merger splitting Fox's share
> counts; a company name read as advice), each fixed by the next 06:00 session - and each time the site
> waited for the next 02:00 run anyway. By today it was three trading days stale. The gate did its job;
> **the gap was that nothing re-ran the loop once the fix had landed.** Separately the CLI had not updated
> itself, and the 13:00 session died on the version check - as tomorrow's 06:00 would have.
> 
> ### Did
> 
> - **`Screener Data Retry`, 18:00 Mon-Fri, `data-run.ps1 -Retry`.** The data loop's success marker makes
>   it a no-op on a day that published. On a refused day it reruns every gate on that day's close. It
>   stands down on an owner session (before anything), and on a dirty tree or a branch off `main` (under
>   the repo lock, before the 02:00 run's stash-and-checkout rescue, writing no brief) - so it cannot
>   touch work in progress. No logon trigger, no late catch-up. `tests/test_data_retry.py`, 9 tests; every
>   guard removed in turn fails one. The 02:00 run is unchanged.
> - **The code runner updates the CLI and retries the session once** when the API says the CLI is too
>   old for the model (`npm install -g @anthropic-ai/claude-code@latest`; one retry, only on a successful
>   update).
> - **Updated the CLI on this PC now:** 2.1.223 -> 2.1.322, and confirmed a `--print` session runs on the
>   configured model. (npm skipped the package's postinstall script; the binary runs regardless.)
> - CLAUDE.md settled row -1 and "The two loops" say both.
> 
> ### Registered and exercised (with the owner's go-ahead, same evening)
> 
> - `scripts/register-tasks.ps1` run: six tasks Ready; `Screener Data Retry` runs
>   `data-run.ps1 -Retry` at 18:00 Mon-Fri (`schtasks /query /v` checked).
> - The retry task was started by hand at 19:37 and took the real path: "Same-day retry: today's 02:00
>   run did not publish", a full run (`logs/datarun-2026-11-10_193759.log`): HEALTH: PASS, price dates
>   **499/499 on 2026-11-10**, Spearman 0.985 vs the 11-06 page, PUBLISHED CLAIMS: PASS (2335), signals
>   written for run `4108d8310b22`, **published 20:07**. The live page reads "PRICES AT THE NOV 10 CLOSE"
>   at 1440 and 375, no errors, no sideways scroll.
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

