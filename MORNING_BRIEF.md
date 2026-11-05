# Morning Brief - Thursday 05 November 2026, 13:50

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-05 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, EXPE, MU, SNDK |
| Ranking's lean on growth | +0.060 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons), newest 1m run date 2026-10-05 (31 days ago, bound 40) |
| Context signals | 19 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 436/437 within 0.5%; operating cash flow 403/420 within 0.5%; net income 397/433 within 0.5%; market value 446/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 3 days ago |
| Top 25, worth a look | EXPE (#3: equity 4.2% of assets); HST (#6: net income 1.46x after-tax operating income); DXCM (#7: operating cash flow 36% off the filing); PSX (#8: net income 1.28x after-tax operating income); CF (#13: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 660 s, `fetch.retry` 121 s, `context.sec_insider` 107 s; **38 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (28 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 4 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 11 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.

## What changed in the repo

- `8f5f791 Log the 13:00 session: the red baseline, and the defect behind it`
- `18eeb13 The flake found: the sheet pre-warm was overwriting panels a test had just rendered`
- `2ca17f1 A red baseline keeps its transcript, like the gates do`
- `0081d38 Leader caption: the date never breaks across lines`
- `d5e85b4 Leaders show how long they have led; holdings show where each name has been`
- `0a00f0e The top-stock rank test opens the stock ranked first, not EXPE by name`
- `061953d Polish: nothing cut short, no empty slot, and no false 'ahead of 100%'`
- `d25bdb1 Show the story each section tells: what moved the scores, period by period, sector standouts`
- `a0500df Market backdrop: embed the band table indented, so the page carries no template brace pair`
- `d409e09 Market backdrop: a gauge on each reading, real charts on the ten series`
- `6f1dff0 brief: code session 2026-11-05`
- `9fe087c Log: the stale ledger, the check that sees it, and what the next build costs`
- `d8f04f3 A ledger row is checked against the code it is published beside`
- `201c21e Bring the research-engine ledger level with the code it is published beside`
- `c58d886 Both decisions' measurement outputs, and how the baseline count was obtained`

## The session's own account

> 2026-11-05 (13:00) - QUEUE. A red baseline came first, and it turned out to be a real defect
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-05_060000.log`
> (shipped to main, `good/2026-11-05`); four owner-run sessions followed it, `good/2026-11-05-owner-1..4`.
> Data loop published? yes - `logs/datarun-2026-11-05_020000.log` (Data loop complete; HEALTH: PASS).
> Evidence base at `1m`: **44 rows, newest run date 2026-10-05 - 31 days ago, bound 40, 5 effective
> observations of the 8 the engine's gate needs.** Priority 0: `allow_auto_apply` **false**. Top open
> roadmap item: **0.13, The Factor Lab**, opened 2026-10-12 - **24 days**; also open 0.10 (29d, opened
> 2026-10-07) and 0.14 (22d, opened 2026-10-14). No branch held back by the gates. `git stash list`
> still carries the one `auto-rescue 2026-10-28_060001` entry that `git stash show` reports as empty -
> left alone, as five prior sessions did.
> 
> **Tests:** before **3,976 passed / 1 skipped** (parallel, 3:02 - my own run; the runner's baseline at
> 13:03 reported 1 failure, see below), after **3,982 passed / 1 skipped** (parallel, 3:04). The six
> new tests are 1 in `tests/test_gate_ordering.py` and 5 in `tests/test_sheet_warm_is_settled.py`.
> All five gates pass on this tree.
> 
> **Owner queue / rotation:** `write_brief.open_proposals()` returns `[]` - no proposal waiting.
> `OWNER_FOCUS.md`'s open items are the universe expansion (next: phase 1) and the 2026-10-20
> fine-tune item. The 06:00 entry's **Next** was the implied-volatility inversion build. **Neither was
> taken, and the reason is the first line of the prompt**: the runner reported `main` **RED on
> arrival**, which outranks the focus - and chasing it down produced the session's work. See "Why not
> the IV build" below; nothing was deferred silently.
> 
> ### Did
> 
> 1. **A red baseline now keeps its transcript, like gates 1 and 5 do.** At 13:03:29 the runner logged
>    `BASELINE main was RED on arrival: 1 pre-existing failure(s)` and named
>    `tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken` - a
>    browser module that `claims.py` also puts in the 02:00 publish gate, where a failure **discards
>    the night's data**. And nothing else: no assertion, no traceback. Gates 1 and 5 have saved their
>    whole output beside the log since 2026-11-04, for exactly this reason ("the log said only FAIL
>    check_published_claims"); the baseline, which the session reads *first* and which the prompt makes
>    its first job, did not. Measured cost: about **40 minutes** of this session establishing that
>    `main` was in fact green.
>    `Save-GateOutput` gains a `$Level` parameter defaulting to `'ERROR'`, so gates 1 and 5 are
>    byte-unchanged, and the baseline calls it at `WARN` - it is not a gate and decides nothing - on
>    **both** red branches, including "could not be taken", the most opaque of the three states.
>    `tests/test_gate_ordering.py` (1 new test, 14 total); **5 mutations, 5 caught**
>    (`research/measurements/2026-11-05-baseline-output-mutations.py`), including the save moved inside
>    the named-failures branch and either gate made to pass its own level.
> 2. **Then the flake itself, found and fixed - it was a real defect in the harness, not luck.**
>    It never reproduced: 8 runs across 4 scenarios, including **two whole-suite parallel runs of all
>    28 Chromium modules, 737 tests each**, all green
>    (`research/measurements/2026-11-05-context-claims-flake.py`).
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

