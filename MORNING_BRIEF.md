# Morning Brief - Monday 02 November 2026, 06:08

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **stopped deliberately** - last ran today |
| Dashboard data from | 2026-10-30 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, HST, APA |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (40 rows, but overlapping windows are not independent; 105 rows across all horizons), newest 1m run date 2026-09-29 (34 days ago, bound 40) |
| Context signals | 16 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 452/453 within 0.5%; operating cash flow 429/437 within 0.5%; net income 420/449 within 0.5%; market value 450/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); SNDK (#5: a period behind its filings); PSX (#7: net income 1.28x after-tax operating income); EG (#8: a period behind its filings); DXCM (#9: operating cash flow 36% off the filing); DECK (#10: a period behind its filings); CF (#11: net income 16% off the filing); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **22 min** total; slowest `fetch.tickers` 646 s, `fetch.sec_roe_history` 195 s, `fetch.retry` 138 s; **45 tickers rate-limited** |
| Context coverage | options usable 490/499 (9 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_special_item_note.py::test_the_published_payload_agrees_with_the_filings
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.
- SESSION DID NOT RUN: session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed
- Treating this as a failure, not as 'nothing to do'. No success marker will be written,
- so the catch-up trigger will retry rather than skipping the day.
- GATE 1 tests: FAIL (=========================== short test summary info =========================== FAILED tests/test_special_item_note.py::test_the_published_payload_agrees_with_the_filings 1 failed, 3601 passed, 1 skipped, 353 warnings in 225.42s (0:03:45))
- GATE 1 attribution: 0 new, 1 inherited from a main that was already red.
- inherited: tests/test_special_item_note.py::test_the_published_payload_agrees_with_the_filings
- SHIP GATES FAILED: tests. Not merging.
- Work pushed to nightly/2026-11-02 for inspection. origin/main verified clean (reverted if the earlier push had reached it).
- Run finished with failing gates - see above.

## What changed in the repo

- `c6ad39a brief: data run 2026-11-02`
- `5c56aaf brief: code session 2026-11-01-midday - SESSION DID NOT RUN`
- `710c9b0 brief: code session 2026-11-01-weekend - SESSION DID NOT RUN`

## The session's own account

> 2026-10-31 (13:00 midday) - QUEUE. Nine more audit rows, and the morning's own two turned out not to have been closed
> 
> **Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 weekend session,
> `logs/nightly-2026-10-31_060001-weekend.log`, ends "shipped to main". Data loop published?
> **yes** - the last weekday run, `logs/datarun-2026-10-30_020001.log`, "Data loop complete",
> HEALTH: PASS, signals written (40 names, 8 gates, all ran); today is a Saturday, so no 02:00 run
> is due. Evidence base at `1m` = **40 rows, newest 2026-09-29 (32 days ago, bound 40), 5
> effective** - steady state, unchanged from this morning. Priority 0: `allow_auto_apply` **false**,
> 5 of 8 effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 19
> days**) - not advanced; the next lab Friday is **2026-11-06** and no experiment is half-built.
> (0.10 is older - opened 2026-10-07, **age 24 days** - and its remainder is Sharpe and Sortino at
> weight 0, which CLAUDE.md says to do only with nothing better.)
> **Tests:** before **3,594 passed / 1 skipped**; the runner's own baseline line agrees
> (`BASELINE main is GREEN on arrival (3594 passed, 1 skipped ... 0:04:01)`), after **3,602 passed /
> 1 skipped** (+8, all new). Dry run exits 0. `scripts/rehearse_publish.py` **PASSED** (the diff
> touches `claims.py`): 362 page tests, published claims 1,850 passed, all four published files
> unchanged and restored byte-for-byte.
> **Owner queue / rotation:** no proposal is waiting (`write_brief.open_proposals()` returns `[]`),
> so `OWNER_FOCUS.md` item 2 sub-item 3 ("test the tests"), carrying on from the 06:00 entry's
> **Next** exactly as the focus asks. Nothing deferred for a gate or a loop. `git stash list` shows
> `auto-rescue 2026-10-28_060001`; `git stash show -p` prints nothing and
> `--include-untracked --name-only` lists three gitignored data files (a context log and two price
> caches), so there is **no tracked work in it** - left in place, as every session since has. No
> owner-session marker.
> 
> **Second session today.** The 06:00 entry's Next named `_pbo_words` at its own bars, then the
> three bank P/B-ROE prose rows and the coverage-discount worked example. All of those are done -
> and so is something that entry did not know it had left behind: its own two rows.
> 
> ### Did
> 
> **1. The two rows the 06:00 session closed were never closed in the register.** This audit's own
> rule, met from the inside: *"Finishing a gap means registering the fix too. A new test that
> `checked_by` does not name reads as a gap on the next pass and is outside the publish gate's
> derivation."*
> 
> Commit `a306d32` wrote five real checks for `share-class-switching-margin` and extended one for
> `clamp-bullet-second-printing`. It touches **no register file**: both sentences still read
> `open:` in `OVERVIEW_SENTENCES`, `OVERVIEW_KNOWN_OPEN` still named both slugs, and the five new
> `TestThePagePrintsTheMargin` checks were in no claim's `checked_by`. "`KNOWN_OPEN` drops twelve to
> ten", written in that entry and in `OWNER_FOCUS.md`, was true of the **research** page's two rows
> only - the 07:00 commit did move those.
> 
> **Measured rather than inferred**, with a harness that differs from the morning's in the one way
> that matters: `research/measurements/2026-10-31-registered-row-mutations.py` runs **the claim's
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

