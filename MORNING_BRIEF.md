# Morning Brief - Monday 26 October 2026, 02:27

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-26T02:00:04.754086 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, INCY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (36 rows, but overlapping windows are not independent; 95 rows across all horizons), newest 2026-10-16 |
| Context signals | Context signals: 13 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 409/413 within 0.5%; net income 399/422 within 0.5%; market value 443/484 within 5%; filings cached 3 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#15: net income 16% off the filing) |
| Context coverage | options usable 477/498 (21 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `18bb07d data: screener run 2026-10-26 - 498 scored, top: EXPE MU APA HST INCY`
- `30f7b28 CLAUDE.md: the bot reads signals.json since 2026-10-25 - keep write_signals.py's fields and order`
- `c637b84 brief: code session 2026-10-25-midday`
- `67c6ff8 log + OWNER_FOCUS: the 2026-10-25 midday session - the claims register's second mutation pass`
- `61890df plan: the 2026-10-25 mutation audit - 24 mutations, 11 claims, 4 real gaps and a false claim`
- `8564552 claims audit: the capex note's 5% reconciliation had no check that could see it`
- `905b2aa claims: register the two new summary.weakest boundary checks`
- `fc888b8 claims audit: two untested bars, a test named for a strictness it could not see, and a register sentence no code ever obeyed`
- `12b1b94 brief: code session 2026-10-25-weekend`
- `ef1a514 log: the 2026-10-25 weekend session - the positive control as a ledger row`
- `96ac5f6 test: the rerun planner's new shape - controls last, an unbuilt arm reported not run`
- `4cc2f50 OWNER_FOCUS: sub-item 4 done - the positive control is a ledger row`
- `f49474d the positive control ran: +0.0512 of monthly IC, t 5.59, inside its band`
- `bc25a63 wip: the plan's item 4(c), the note's method, and the log entry's health block`
- `ea102de the page's 'the test can see a signal' claim reads the control's row, or says nothing`

## The session's own account

> 2026-10-25 (13:00 midday session) - the claims register's second mutation pass: four bars nothing could see, a test named for a strictness it did not test, and one claim that was simply false
> 
> **Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 weekend session
> ended "Run complete: shipped to main" at 08:13, tag `good/2026-10-25`. Data loop published?
> **yes** - 2026-10-23 02:28, HEALTH PASS, 498 scored; **no 02:00 run on 10-24 or today, correctly -
> Saturday and Sunday**; the next is Monday 2026-10-26, which brings the first `signals.json`.
> Evidence base at `1m` = **35 rows, newest 2026-09-22 (33 days ago, bound 40), 4 effective**
> (`_n_observations` 4; 35 is the row count and never the one to quote). Priority 0:
> `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
> Factor Lab** (opened 2026-10-12, **age 13 days**) - deferred again; experiment 18 is unstarted and
> the next non-retrospective Friday is 2026-11-06, so nothing about it is time-critical, and this
> session took an owner sub-item that outranks it.
> **Tests:** before **3149 passed, 1 skipped** (364.6s) - identical to the 06:00 session's after
> count, so no drift. After **3152 passed, 1 skipped** (+3: two in `test_stock_summary.py`, one in
> `test_capex_lease_note.py`).
> **Owner queue / rotation:** `OWNER_FOCUS.md` 2026-10-20, **sub-item 3, "Test the tests"** - the
> claims mutation audit. No proposal was waiting (`open_proposals()` returns `[]`). This item had
> been named as Next by **three** consecutive sessions and never taken, and the 06:00 entry said
> plainly that a session with no proposal waiting and no broken loop should take it rather than make
> it a fourth deferral. I did. The morning's own Next - the extended universe - is deferred and
> stated below, because it needs a rebuild plus a ~45-minute serial rerun and this slot had under
> three hours.
> 
> ### Did
> 
> **`OWNER_FOCUS.md` 2026-10-20 sub-item 3: the second mutation pass over `claims.py`.**
> `plan/claims-mutation-audit.md`, rows 39-62. Method unchanged: for each registered sentence that
> states a number, make the smallest plausible edit that makes the sentence false, run **only** that
> claim's `checked_by`, and record caught or gap. **24 mutations on 11 claims: 15 caught, 4 real
> gaps, 5 gaps in the register, and 1 mutation that could not be made because the claim was already
> false.** Every gap now fails a registered check, and every mutation was **re-run after its fix and
> required to read CAUGHT** - which caught me out once: two fixes first came back as register gaps
> because I had written the test and not yet named it in `checked_by`.
> 
> **One improvement to the harness, which is the reason the numbers above are legible.** It now runs
> the claim's `checked_by` *and* the whole of each module those tests live in, so "nothing in the
> suite checks this" and "the suite checks it but the register does not say so" are separate
> verdicts. The 2026-10-20 pass had to tell them apart by hand and its rows 35 and 37 record the
> distinction informally. Five of this pass's nine gaps were the second kind.
> 
> **The four real gaps** - nothing anywhere in the suite saw them (verified: only the one module
> references each symbol):
> 
> 1. **`summary.weakest`, `LARGEST_SIZE_SCORE` 10 -> 11.** The sentence says "at 10 or under, one of
>    the largest"; the four registered checks probe Size at 2, 30 and 55, so the bar could have sat
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

