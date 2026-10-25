# Morning Brief - Sunday 25 October 2026, 08:13

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## THE ROUTINE IS NOT RUNNING

- **Data run** has not run since 2 days ago

Nothing below is current. A loop that stops firing writes no log, so
the rest of this page describes the last run that *did* happen, not
today. Most likely cause: the PC rebooted and nobody logged back in -
the tasks only run while a user is signed in. See NIGHTLY_LOG.md
2026-08-20 and `scripts/register-tasks.ps1`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran 2 days ago |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-23T02:00:04.666022 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, APA, HST, INCY, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (35 rows, but overlapping windows are not independent; 93 rows across all horizons), newest 2026-10-15 |
| Context signals | Context signals: 12 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 423/424 within 0.5%; operating cash flow 406/410 within 0.5%; net income 396/419 within 0.5%; market value 444/484 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); MU (#10: a period behind its filings); NEM (#12: a period behind its filings); CF (#16: net income 16% off the filing) |
| Context coverage | options usable 486/498 (12 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `ef1a514 log: the 2026-10-25 weekend session - the positive control as a ledger row`
- `96ac5f6 test: the rerun planner's new shape - controls last, an unbuilt arm reported not run`
- `4cc2f50 OWNER_FOCUS: sub-item 4 done - the positive control is a ledger row`
- `f49474d the positive control ran: +0.0512 of monthly IC, t 5.59, inside its band`
- `bc25a63 wip: the plan's item 4(c), the note's method, and the log entry's health block`
- `ea102de the page's 'the test can see a signal' claim reads the control's row, or says nothing`
- `344a9e9 research engine: controls, so a null result can be told from a blind test`
- `e55d36a register the positive control, before it is built or run`
- `e7ccef5 brief: code session 2026-10-24-midday`
- `5dad042 test: the research page's intervals come from the ledger, not from one audit's literals`
- `0693eda research engine: all 31 ledger rows rerun on the rebuilt records`
- `f3d757f plans: the two v2 data fixes are done; what remains of the sub-item`
- `5eedc7c wip: the log entry's body, up to the rerun`
- `9bf99fd wip: the main build's effect, measured`
- `40b2e93 wip: the presample rebuild's effect, measured`

## The session's own account

> 2026-10-25 (06:00 weekend session) - the positive control is a ledger row: the page's "the test can see a signal" is now a number, rerun with everything else
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-24 13:00 midday ended "Run
> complete: shipped to main" at 15:49, tag `good/2026-10-24-1549`. Data loop published? **yes** -
> 2026-10-23 02:28, HEALTH PASS, 498 scored, all four input gates green; **no 02:00 run on 10-24 or
> today, correctly - Saturday and Sunday** (next is Monday 2026-10-26, which brings the first
> `signals.json`). Evidence base at `1m` = **35 rows, newest 2026-09-22 (33 days ago, bound 40), 4
> effective** (`_n_observations` 4; 35 is the row count and not the one to quote). Priority 0:
> `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
> Factor Lab** (opened 2026-10-12, **age 13 days**) - deferred; the next experiment is 18 and the next
> non-retrospective Friday is 2026-11-06, so nothing about it is time-critical.
> **Tests:** before **3129 passed, 1 skipped** (369.6s) - identical to the 10-24 13:00 session's after
> count, so no drift overnight; after **3149 passed, 1 skipped** (359.3s), **+20, all in the one new
> module** (`tests/test_positive_control.py`). Gates 2 and 4 pass; gate 3 via
> `tests/test_payload_parse_gate.py` (19) and the fact that no payload file is in this diff -
> `node --check` is refused by this session's tool guard, and the runner re-runs gate 3 itself.
> **`scripts/rehearse_publish.py` run and PASSED**, because the diff touches `claims.py`, which the
> page is built from: 1,426 claims-gate tests, 3,979 category scores and 498 composites reproduced
> from the regenerated payload, published files restored byte for byte.
> **One existing test needed updating and it was the contract, not the assertion:**
> `test_a_limited_rerun_does_experiments_first_and_evaluations_once_they_are_level` pinned
> `rerun_experiments.plan`'s return as a 3-tuple. It is now a 5-tuple (controls, and the controls whose
> arm is not built), so the test pins the new shape *and* the new ordering - experiments, then
> evaluations, then controls, because a control is paired against the baseline and is worth nothing
> until that row is current - plus that an unbuilt arm comes back named with the command that builds
> it. Nothing was loosened.
> **Owner queue / rotation:** `OWNER_FOCUS.md`'s **top open item (2026-10-20), sub-item 4** - the
> positive control as a ledger row. No proposal was waiting for a decision
> (`write_brief.open_proposals()` returns `[]`), so section 3's first job was clear. Sub-item 4 is the
> one the 10-18 and 10-19 sessions both named as Next and nobody had taken since; sub-item 6's
> remaining half (the extended universe) is the larger piece and is left as Next with its reason.
> 
> ### Did
> 
> **`OWNER_FOCUS.md` 2026-10-20 sub-item 4 / `plan/research-engine.md` item 4(c): the positive
> control is a ledger row.** The thing it fixes is a sentence. Since 2026-10-15 the public research
> page has told readers, in the three-bullet answer at the top, that **"The test can see a signal when
> there is one"** - with no number beside it, resting on a measurement from 2026-10-15 that was never
> a ledger row and whose monthly records have since been rebuilt **twice** (operating income on 10-15,
> the pre-2014 renames and predecessor CIKs on 10-24). It was the one claim on that page a reader
> could not check, and the page's 28 flat verdicts are the claim that most depends on it: a row of
> nulls is exactly what a broken measurement produces.
> 
> **Why a null needs this and cannot supply it.** Harvey, Liu & Zhu (2016), *RFS* 29(1) 5-68,
> DOI 10.1093/rfs/hhv059 - whose t >= 3 bar this engine adopts, with a Bonferroni correction on top -
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

