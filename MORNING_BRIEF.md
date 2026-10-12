# Morning Brief - Monday 12 October 2026, 02:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## THE ROUTINE IS NOT RUNNING

- **Code session** has not run since 2 days ago

Nothing below is current. A loop that stops firing writes no log, so
the rest of this page describes the last run that *did* happen, not
today. Most likely cause: the PC rebooted and nobody logged back in -
the tasks only run while a user is signed in. See NIGHTLY_LOG.md
2026-08-20 and `scripts/register-tasks.ps1`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran 2 days ago |
| Dashboard data from | 2026-10-12T02:00:03.296811 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, APA, CF, BMY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (28 rows, but overlapping windows are not independent; 78 rows across all horizons), newest 2026-10-05 |
| Context signals | Context signals: 3 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 455/501 (46 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `c620c4b data: screener run 2026-10-12 - 501 scored, top: EXPE HST APA CF BMY`
- `7cbc8e7 log: PR #1 gated on the PC and merged; a time-of-day test fixed`
- `b7cda7f test: forward EPS blend check no longer depends on the time of day the suite runs`

## The session's own account

> 2026-10-11 (late, owner-run on the PC) - PR #1 gated and merged; one time-of-day test fixed
> 
> **Health (rule 8):** last code session - `logs/nightly-2026-10-09_060001.log`, shipped. Data loop -
> `logs/datarun-2026-10-09_020001.log`, complete, HEALTH: PASS (no weekend runs; next 2026-10-12 02:00).
> Evidence base at `1m`: **26 rows, newest 2026-09-09 (32 days, bound 40), 4 effective**. Priority 0
> unchanged (`allow_auto_apply` false). Top open roadmap item: **0.12's last residual, the momentum panic
> rule** (opened 2026-10-09, age 2 days), then **3, backtest v2 step 3** (age 10 days).
> 
> **PR #1** (`tests/test_bot_payload_contract.py`, the CLAUDE.md "Downstream consumer" note and the entry
> above) was built in a cloud container that could not run gates 1 and 2. All four were run here on the PR
> head: **gate 1 2,113 passed / 0 failed; gate 2 `--dry-run` exit 0; gate 3 `index.html` 542 KB and both
> payload files pass `node --check`; gate 4 clean.** Merged with a merge commit (`951570d`), tagged
> `good/2026-10-11-owner`. No methodology change, no generated file changed.
> 
> **One thing gate 1 found that was not the PR's:** `tests/test_forward_eps_growth.py::test_the_twelve_month_blend`
> failed on `main` too. The test (written 2026-10-09) builds a fiscal-year end "six months from now" while
> the engine counts whole days from midnight, so months-left was 6 give or take a day by the time of day
> the suite ran, and the tolerance was tighter than a day of the blend. It passed every daytime run and
> failed at 23:25. Test-only fix: the arithmetic is checked against the engine's own published months-left.
> The engine is unchanged.
> 
> **Standing consequence of the merge:** the seven fields the bot reads (`kpis.run_timestamp`;
> `table_data[].Ticker/Rank/Composite/Sector/Value_Trap_Flag/Growth_Trap_Flag`; `stock_detail[t].price`)
> must not be renamed or dropped. A change that has to updates `bot/signals.py` and its tests in
> `CalebSmit/screener-bot` in the same piece of work. Nightly sessions here do not otherwise work on the bot.

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

