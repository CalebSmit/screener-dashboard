# Morning Brief - Sunday 18 October 2026, 14:56

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

- `bf34a24 log: the 2026-10-18 midday session - item 8b finished, and the two defects the rerun found`
- `07a3192 engine: rerun on the fixed marker, and measure against the pre-8b rows`
- `c70d485 engine: the provenance flag keys on the code's content, not on the locator`
- `41a2143 docs: item 8b's record, the marker defect the rerun caught, and the owner's progress line`
- `5420a9b engine: the dirty marker is scoped to the files that can change a number`
- `0a6a8fb engine: rerun all 28 experiments and the evaluation for item 8b's provenance fields`
- `affabe8 brief: code session 2026-10-18-weekend`
- `e19cf90 wip: salvaged from an interrupted session 2026-10-18 - uncommitted when the gates ran`
- `a1204a8 docs: item 8b's record, the no-override rule, and the session log so far`
- `f03b991 engine: every result names the code, the weighting and the records it ran on (item 8b)`
- `ad2d422 brief: code session 2026-10-17-midday`
- `68fb679 log: the 2026-10-17 midday session - the macro backdrop note and three corrected page sentences`
- `6924c52 docs: context-layer item 4's macro note is done; the owner queue's progress line`
- `1970c17 look at the corrected backdrop at 1440 and 375; ignore the FRED download cache`
- `b4a56c2 backdrop: three sentences corrected to what their sources' own records show`

## The session's own account

> 2026-10-18 (13:00 midday session) - QUEUE. This morning's provenance work finished, and finishing it found two defects in the check itself
> 
> **Health (rule 8, all five):** last code session ran? **it ran and its gates failed** -
> `logs/nightly-2026-10-18_060001-weekend.log` ends "SHIP GATES FAILED: tests, clean-tree. Not
> merging", so `main` is still at `affabe8` and this morning's work sat unmerged on
> `nightly/2026-10-18`. Recovering it is this session's work (below). Data loop published? **yes** -
> `logs/datarun-2026-10-16_020001.log` ends "Data loop complete", HEALTH: PASS; 2026-10-16 was a
> **Friday** and today is a **Sunday**, so Friday is the newest run due and the next is Monday 10-19.
> Evidence base at `1m`: **30 rows, newest 2026-09-15 (33 days ago, bound 40), 4 effective** - inside
> the bound, steady state. Priority 0: unchanged, `allow_auto_apply` **false**, 4 of 8 effective.
> Top open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 10 days**).
> **Tests:** before **2659 passed, 5 failed, 2 skipped** - the five are this morning's, inherited with
> its branch; `main` itself is green. After: see Gates below.
> **Owner queue / rotation:** no proposal waiting (`write_brief.open_proposals()` returns `[]`); no
> stash to rescue; no owner-session marker. Took the recovery of this morning's unmerged branch, which
> continues the same owner item it was working - the **research engine** (`OWNER_FOCUS.md`
> 2026-10-15), plan item **8(b)**. **0.11 (the context layer) deferred, streak 2** - an inherited red
> baseline outranks the queue (rule 1); the next session with nothing ahead of it should take
> context-layer item 4's last per-signal note, the track record.
> 
> ### Did
> 
> **Recovered this morning's branch and finished queue item 8(b).** The 06:00 session built the
> provenance machinery - every ledger row records `git describe`, the SHA-256 of each
> result-affecting file (`RESULT_PATHS`), a content fingerprint of the monthly records and the
> baseline weighting's own hash; `run` and `evaluate` refuse to start from an edited tree; `report`
> flags a mixture; `summary_for_page` drops a mismatched row. **That code is sound and is kept as
> written.** What it did not do is the step that makes the claim true: **rerun the experiments so the
> rows the report reads actually carry the fields.** It ran **2 of 29** (`baseline`,
> `equal-categories`), wrote "All 28 experiments and the evaluation were rerun on a clean tree" into
> `plan/research-engine.md`, left the ledger and the measurement script uncommitted, and stopped. Four
> tests were red. The runner salvaged the work to `nightly/2026-10-18`, refused the merge, and left
> `main` clean - **the gates did exactly their job**, and this is the argument for the runner
> re-running them independently (2026-09-04) rather than for trusting a progress line. A stage is done
> when its own tests pass on a clean tree, not when its description is written.
> 
> So this session merged the branch and ran the rerun: **all 28 registered experiments and the one
> evaluation**, in a single process on a clean tree. **Doing it is what found two defects in the new
> check, neither of which a reading of the code would have shown.**
> 
> **Defect 1: `git describe --dirty` reads the whole tree, and a run dirties it itself.** A run
> appends to the tracked `ledger.jsonl` and `report` rewrites `RESULTS.md`, so by the time the
> *second* entry point starts, the tree is dirty **because of the run**. The evaluation row came out
> stamped `good/...-dirty` on a rerun where nothing but the engine's own output had changed - and
> **no order of `run` and `evaluate` escapes it**, in one process or two, so the test asserting no
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

