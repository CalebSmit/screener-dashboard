# Morning Brief - Monday 21 September 2026, 02:14

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
| Dashboard data from | 2026-09-21T02:00:03.202880 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, VLO, CAH |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (13 rows, but overlapping windows are not independent; 48 rows across all horizons), newest 2026-09-14 |

## What changed in the repo

- `120daba data: screener run 2026-09-21 - 502 scored, top: EXPE HST BBY VLO CAH`

## The session's own account

> 2026-09-18 - RETROSPECTIVE. Evaluate whether this routine is producing value, and change the process where it is not.
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes.** `logs/nightly-2026-09-17_060000.log` ends "Run complete: shipped to main", tagged `good/2026-09-17` |
> | Data loop published? | **Yes.** `logs/datarun-2026-09-18_020000.log` ends "Data loop complete", HEALTH: PASS, 502 scored, 0 fetch failures, price coverage 502/502 |
> | Evidence base | **At `1m`: 11 rows, newest 2026-08-14, 3 effective observations** against a gate of 8. Read at `1m` for the first time today - see finding 1. Lag **35 days**, inside the new 40-day bound; it is the 08-15..08-19 outage still working through the pipe, and resumes 09-21 |
> | Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call touched |
> | Top open roadmap item | **Priority 3, backtest v2** - `plan/backtest-v2.md` dated 2026-08-25, **24 days**. Not taken: a retrospective does not work on the screener |
> 
> **Tests:** before **1362/1362**, after **1378/1378** (+16, `tests/test_payload_parse_gate.py`; 11 of them fail against the pre-change runners).
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty; nothing deferred. ISO week 38, Friday, even week - retrospective, per the rotation.
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 9 scheduled** (2026-09-07 to 2026-09-17). No owner-run
>   sessions this period - the first fortnight with none.
> - **Genuinely valuable: 9 | Churn: 0 | Failed gates: 0.**
> 
> **1. What fraction produced something genuinely valuable? All nine, and there
> is a merge commit per session to check it against.** 09-07 found the Revisions
> category contains no revisions; 09-08 shipped priority 4, replacing the AI chat
> with deterministic per-stock summaries; 09-09 settled the design and caught two
> of Monday's own numbers being wrong; 09-10 built `fy1_revision_3m` and the
> reweight; 09-11 fixed a published percentile whose obvious reading was
> backwards on 13 of 37 metrics; 09-14 researched sell discipline from five
> papers and three index methodologies; 09-15 shipped My Holdings; 09-16 found
> the project's independence trap for the third time, in rank statistics, and
> settled the hold band as "not yet" with a date; 09-17 shipped the cadence
> statement and the input-churn flag. Sessions ran **13.0-22.2 API-minutes**
> against a 4-hour limit; none came close.
> 
> **The quality signal worth naming: five of the nine corrected this project's
> own published claims** rather than defending them - 09-09 corrected Monday's
> +0.346 to +0.401, 09-10 corrected three stale doc claims including a registry
> split wrong since 09-02 where two errors cancelled, 09-11 caught two false
> tooltips before shipping by checking `config.yaml` instead of recalling, 09-16
> corrected two of Monday's headline numbers, 09-17 corrected the inventory's
> superseded hold-band figures. That is the habit that makes the rest credible.
> 
> **2. Which rotation day earns its place? All five, and the Mon-Wed-Thu chain
> is now demonstrably load-bearing rather than nominal.** 09-07 research →
> 09-09 synthesis → 09-10 build shipped one coherent methodology change across
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

