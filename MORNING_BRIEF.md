# Morning Brief - Saturday 17 October 2026, 07:17

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-16T02:00:03.815047 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, HST, INCY, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (30 rows, but overlapping windows are not independent; 83 rows across all horizons), newest 2026-10-08 |
| Context signals | Context signals: 7 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#18: a period behind its filings); ES (#19: a period behind its filings) |
| Context coverage | options usable 492/501 (9 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `ccd68d3 log: the 2026-10-17 weekend session - gate fix, method version 2, and the rerun's result`
- `1289e7d engine: rerun all 28 experiments and the evaluation on method version 2; no verdict changed`
- `d5ec2ad docs: a method change reruns the engine, like a data fix (rule in CLAUDE.md)`
- `5761f47 engine: comparisons are measured on the stocks both rankings share (method version 2)`
- `2268c3e test: the option-cache fixtures name a real market session, not 'today'`
- `f849b55 non-blocking fonts on the research page and the lab; the afternoon's log; two settled rows updated`
- `78c5aa6 test: the Escape check finds the modal handler among several`
- `d479628 docs: row 5's kept-name rule recorded, inventory names the ten fixes`
- `e94c227 dashboard: ten functional defects fixed, payload byte-identical`
- `6626c71 evidence base repaired: every forward return re-measured to a horizon end that exists`
- `68866b6 forward returns are measured only to a close that exists, looking forward, and once`
- `9b2f89f research docs: dated corrections and the audit's queued engine changes`
- `b9a311d research page: claim only what the ledger supports (audit, 2026-10-16)`
- `16bf997 measurements for the research-page audit: coverage by month, stocks ranked per configuration`
- `70e7ba9 ignore Claude Code agent worktrees`

## The session's own account

> 2026-10-17 (06:00 weekend session) - WEEKEND. A red gate fixed, then the research engine's comparisons made apples-to-apples (queue item 8a)
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-16_130001-midday.log`
> ends "Run complete: shipped to main"; an owner-run pass followed and `main` is at `f849b55`.
> Data loop published? **yes** - `logs/datarun-2026-10-16_020001.log` ends "Data loop complete",
> HEALTH: PASS (no Saturday run is due; the loop is Mon-Fri). Evidence base at `1m`: **30 rows,
> newest 2026-09-15 (32 days ago, bound 40), 4 effective** - inside the bound, steady state.
> Priority 0: unchanged, `allow_auto_apply` **false**, 4 of 8 effective observations.
> Top open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 9 days**).
> **Tests:** before **2619 passed, 2 skipped / 3 failed** (two a genuine red gate - see below - and
> one my own edit landing mid-run), after **2638 passed, 2 skipped / 0 failed** (+19). Dry run exits
> 0; both payloads parse; tree clean. `scripts/rehearse_publish.py` **PASSED** (990 claims-gate
> tests), run because the diff touches a page generator.
> **Owner queue / rotation:** the top open owner item is the **research engine**
> (`OWNER_FOCUS.md` 2026-10-15), whose plan's next queued stage is `plan/research-engine.md` item
> **8(a)**. Took that. **0.11 (the context layer) deferred for the fourth consecutive session** - see
> "On the deferral streak" below, which is the accounting the rule asks for.
> 
> ### Did
> 
> **1. Gate 1 was red at baseline, and it would have been red every weekend.** Two tests in
> `tests/test_option_quote_hours.py` stamped their cache fixture with today's UTC date and expected
> `options_cache` to serve it back. Two days in seven that date is a Saturday or a Sunday, and
> `reading_for` **deliberately refuses** a quote date naming a day the market never opened - the
> page states the session a quote closed in, so a stamp naming a non-session would make that
> sentence false. Production is right; the fixtures were not.
> 
> Nobody had seen it because code sessions ran Mon-Fri. Since 2026-10-14 they also run at 13:00 daily
> and 06:00 at weekends, so **this was the first Saturday session and it opened on a red baseline with
> neither loop at fault.** The 13:00 slot would have hit it today too.
> 
> Fixed: the fixtures take the most recent weekday on or before today (at most two days back, inside
> `MAX_AGE_DAYS` = 3), and a new parametrised test walks all seven days of a week and asserts the
> fixture date is always one `reading_for` accepts - so the class of fault cannot come back silently.
> Measured: the module went from 57 passed / 2 failed to **66 passed** (+7 guard tests). The suite had
> just been run on a Saturday and failed **exactly** these two, so the class is contained to them.
> 
> **2. Queue item 8(a): every comparison is now measured on the stocks the two rankings share.**
> This is the research engine's next queued stage, and it is a correctness defect in what the public
> research page reports.
> 
> Under **method version 1** each configuration's IC was computed over the stocks *it* could rank
> (`backtest_v2.rank_month` filters on coverage of that configuration's own weighted metrics) and
> differenced against the baseline's IC over the stocks the *baseline* could rank. A narrow screen
> ranks far fewer stocks - free-cash-flow yield alone 348 a month against the baseline's 440 - so
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

