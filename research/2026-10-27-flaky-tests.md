# Flaky tests in gate 1 - what failed under load, why, and what changed

*2026-10-27 (owner-run night, round three). Test infrastructure, not methodology: no weight,
metric, threshold, page or payload changed, so `METHODOLOGY_CHANGELOG.md` is untouched. One
finding is a defect in production code (`scripts/repo-lock.ps1`); it is measured and proposed
below, not changed here.*

**Proposal status: OPEN** - fix the two races in `scripts/repo-lock.ps1` (section 3): publish the
lock already carrying its PID instead of exposing an empty unreadable file, treat an unreadable lock
as live, and reclaim a stale lock by atomic rename instead of read-then-delete. Then add the
stale-lock race as a test.

## 1. Why

Gate 1 is the full suite, and it guards a public site and fourteen unattended code sessions a
week. Of six full runs on one tree earlier tonight, two failed once or twice in tests that pass
alone: `test_loop_mutual_exclusion.py::test_exactly_one_of_several_simultaneous_starts_wins`, and
browser tests with short fixed waits (`test_case_against.py`, `test_header_nav_fit.py`,
`test_dashboard_browser.py::test_sort_to_paint_is_fast`). A test that fails on load rather than
on a defect turns a session's baseline red and teaches its reader to re-run until green - which
CLAUDE.md forbids ("a failing test is never retried"). So each one has to become either
deterministic or a reported defect.

## 2. How it was measured

In a worktree with the canonical `runs/`, `cache/` and `data/` copied in (`cp -rp`), on the same
6-core machine while other agents were working (the full suite took 7:23 against 1:51 on a quiet
machine - that load is the condition under which the flakes appear). Scratch harnesses in the
gitignored `logs/round3-flaky/`; the lock measurement is committed as
`research/measurements/2026-10-27-repo-lock-races.py`.

**Before the fixes**

| Run | Result |
|---|---|
| Full suite, parallel | 1 run: 3,453 passed, 1 skipped |
| The 18 browser modules, `-n 8` | 1 run: clean |
| The 18 browser modules, `-n 6` | 7 runs: **2 failed** - `test_factor_lab.py::test_the_overview_loads_no_heavy_series_and_each_experiment_only_its_own`, `test_club_readiness.py::test_a_ticker_this_run_did_not_score_is_named` |
| `test_factor_lab.py` + `test_lab_reading_and_access.py`, `-n 4` | 3 runs: **1 failed** - `test_lab_reading_and_access.py::test_a_mouse_button_on_a_card_asks_for_its_series_before_the_click` |
| The mutual-exclusion race, 3 harnesses x 12 runs at once | 36 runs: 0 double winners |

The failures that tonight's main session saw were not reproduced by looping; their mechanism was
found by reading the test against the page and then forcing the timing (section 4). The ones the
loops did reproduce are new names: the same defect in more places.

## 3. The mutual-exclusion test - a real race in the lock, not the harness

**Verdict: the test is right and the lock is wrong.** `scripts/repo-lock.ps1` can let two
processes hold the repo lock at once, by two routes.

**Route 1 - the write window.** `Enter-RepoLock` creates the lock with
`File.Open(CreateNew, FileShare.None)`, *then* formats `"$PID`n$Holder`n<date>"`, writes it and
disposes the stream. Between the create and the dispose the file exists but no other process can
read it. A contender that arrives then fails its own CreateNew, finds the file, and calls
`Read-RepoLockInfo` - which swallows the sharing violation and leaves `OwnerPid = 0` (its comment
says "an unreadable lock is handled by the age ceiling, not by guessing", but the caller does not
do that). `Test-RepoLockOwnerAlive 0` is false, so the contender logs *"held by PID 0 (unknown),
which is gone. Reclaiming."*, loops on `Remove-Item` (refused while the handle is open) and deletes
the lock the instant the winner closes it - then creates its own. Both now believe they hold it.

Measured by playing the winner from Python with the identical open (CreateNew, exclusive share),
staying inside the window for a set time - which is what a winner descheduled under CPU load does -
then writing its own live PID and closing, while the real `Enter-RepoLock` contends:

| Time spent inside the window | Contender took the live lock |
|---|---|
| 1.0 s | **10 of 10** (60 "PID 0 ... Reclaiming" lines each) |
| 0.5 s (committed script) | **4 of 5** |
| 0.25 s | **10 of 10** |
| 0.12 s | 0 of 10 - shorter than the contender's own path from "the file exists" to its read |

How long the real window is, timed in fresh `powershell.exe` processes running the exact block
(the first `Get-Date -Format 'o'` in a new process is slow): **70-85 ms alone, 94-168 ms with six
starting at once, and 106-793 ms while the browser suite ran beside it.** So the window the
contender needs to land in is a tenth of a second on a quiet machine and most of a second on a
busy one - which is why the test fails only under a loaded full suite, and why 36 quiet runs did
not catch it.

**Route 2 - reclaiming a stale lock.** When a crashed run leaves its lock, every contender reads
the dead PID, deletes the file and creates its own. A contender that read the dead PID before
another's create then deletes *that live* lock: check-then-delete is not atomic. Six racers
starting together on a lock owned by PID 999999, each winner holding until all six have returned:
**4 of 8 trials and 2 of 8 trials (two batches) ended with 2, 3 or 5 simultaneous holders.** No
current test covers this; it cannot be added as a test until the lock is fixed, because it would
fail about a third of the time and a test is never marked xfail here.

**How likely in production.** The loops start minutes apart (logon delays PT3M and PT20M; slots at
02:00, 06:00, 13:00), so route 1 needs two `Enter-RepoLock` calls within a fraction of a second -
rare. Route 2 needs a crashed run's lock and two contenders polling it whose read and delete
interleave - also rare, since waiters poll every 20 s unsynchronised. Rare is not never, and two
holders is exactly the git-index corruption the lock exists to prevent (CLAUDE.md settled row -1).

**Recommended fix (not applied: `scripts/` is production code, outside tonight's remit).**

1. Never expose an unreadable lock: write the three lines to a unique temp file
   (`.repo.lock.<PID>.tmp`) and publish it with `[System.IO.File]::Move(tmp, path)`, which is
   atomic on NTFS and throws if `path` exists - the lock then carries its PID from the moment it
   exists. Contention is "Move threw and the file exists", exactly as CreateNew's is today.
2. Treat an unreadable or PID-less lock as **held**: wait and re-read; reclaim it only past the
   age ceiling, as the existing comment already claims.
3. Reclaim by atomic rename, not read-then-delete: `Move(path, path + '.stale.' + $PID)`; only the
   contender whose rename succeeded re-reads the renamed file, confirms it is the dead owner's, and
   deletes it; everyone else finds the path gone and goes back to step 1.
4. Then add route 2 as a test (`research/measurements/2026-10-27-repo-lock-races.py stale` is the
   shape), and run the window measurement against the new code - it should report 0 of N at every
   window length.

**What changed in the test (harness only; the assertion is unchanged - exactly one holder).**

- The winner held for a fixed 15 s. A racer starting more than 15 s after it would find the lock
  *released* and win it legitimately, which reads as a double win. Measured racer start spread
  here is 0.5-0.9 s, so this was a latent assumption rather than tonight's cause, but it was an
  assumption about load. The winner now holds until all six racers have returned from
  `Enter-RepoLock` (bounded at 90 s so a hung racer cannot hang the suite).
- Winners appended to one shared file. Two simultaneous `Add-Content` calls can lose a line to a
  sharing violation, which would *hide* a double win. Each racer now writes its own result file,
  and the test also asserts all six reported.
- The failure message now carries every racer's output, so a reader sees "PID 0 ... Reclaiming"
  and knows it is route 1 without re-running anything.

The test will still fail occasionally under load until the lock is fixed. That is correct.

## 4. The browser tests - fixed waits standing in for something the page does

Every one of these slept a fixed 100-400 ms in place of waiting for a specific event. A loaded
machine stretches frames, timers and the round trip between pytest and Chromium, so the test read
the page before the event. Each now waits for the event itself; **no assertion changed**, no
timeout was raised to cover a race, and nothing is retried.

| Test | What the fixed wait stood in for | Evidence | Fix |
|---|---|---|---|
| `test_factor_lab.py::test_the_overview_loads_no_heavy_series_and_each_experiment_only_its_own` | Nothing - the pointer, left on the picker, sat over the opened list while Playwright scrolled the chosen entry into view; the scroll slid `size-effect` under it, and the page's 150 ms hover-intent rule (round two, by design) asked for that series too | Forced: a 300 ms settle before the click gives `['jkp', 'size']` **3 of 3**; 0 ms gives `['jkp']` 3 of 3; with the pointer parked first, `['jkp']` **6 of 6** at both | Park the pointer in the page gutter (2, 2) before choosing the entry; (2, 2) measured to have no link under it |
| `test_lab_reading_and_access.py::test_a_mouse_button_on_a_card_asks_for_its_series_before_the_click` | "Straight across, no dwell" was two `page.mouse.move` calls - two round trips. Under load the gap passed 150 ms: the pointer *had* dwelt and the page rightly fetched | Captured: `assert page.evaluate("'sorts' in LAB_DATA") is False` -> `True` | The over and the out are dispatched in one task, so the dwell is zero by construction; the listener reads only those two events |
| `test_club_readiness.py::test_a_ticker_this_run_did_not_score_is_named` | The search box's 150 ms debounce; the test waited 200 ms - a 50 ms margin | Probe: the message still reads the old text at +100 ms and has changed by +200 ms | Wait for the box to show the new message (15 s ceiling, then the assertion reports what it saw). A slip while editing - the `fill` dropped - made it fail every time, which confirms the wait does not mask a page that never changes |
| `test_case_against.py::test_context_lines_wait_for_the_context_file` | The held context request reaching the route handler, which runs only when that event reaches pytest | Probe, quiet machine: right after the page's functions existed the request was not yet held in **1 of 10** runs, 109 ms later | Wait until the request is held (poll, 30 s ceiling); `assert held` stays |
| `test_header_nav_fit.py` (all nine) | The bar's fit, which every trigger (load, fonts, resize, a link's visibility) queues on the next animation frame | Read from the page's code: frames, not milliseconds | Wait until the state the tests read is identical for three consecutive frames (bounded at 300) - it waits for the bar to stop changing, never for it to pass |
| `test_dashboard_browser.py::test_sort_to_paint_is_fast` | A single wall-clock sample against a 250 ms budget | 60 samples (three pages of 20) on a moderately loaded machine: page medians 26-32 ms, maximum **125 ms** - one stall from a lone sample nearing the budget | The fastest of five identical sorts (the `timeit` convention). Load only adds time; a slower page raises every sample, the fastest included, so a regression still fails. Said in the docstring |
| `test_dashboard_browser.py`, seven more sleeps | Scroll, sort and filter handlers render on the next frame; the search box debounces; the sheet slides in with a CSS animation | Read from the page's code (`queueWindow`, `debounce(applyFilters, 150)`, `sheetIn`) | `_frames()` (three animation frames), a wait for the empty state, and `_animations_done()` (every finite running animation's `finished`) |

**Left alone, deliberately.** `test_row_click_to_drilldown_is_fast` has the sort test's
single-sample shape but a different purpose - the *first* click on a row is the user's experience,
and the fastest of five would hide a first-open cost. It has not failed in any run here; if it does,
its fix is a measurement that excludes scheduler stalls without excluding the first open, not a
minimum. About forty other fixed sleeps remain in `test_dashboard_navigation.py`,
`test_dashboard_interaction.py`, `test_drilldown_presentation.py` and `test_club_readiness.py`;
none failed in these runs, and converting them blind would be churn in files other sessions are
editing. The helpers above are the pattern if one does.

## 5. After the fixes

Same machine, same competing load (the browser runs took 3:04-4:55 each, slower than before):

| Run | Before | After |
|---|---|---|
| The 18 browser modules, `-n 6` | 2 of 7 runs failed | **4 of 4 clean** (446 passed each) |
| The two lab modules, `-n 4` | 1 of 3 runs failed | **3 of 3 clean** (100 passed each) |
| Each forced timing above | failed as described | passes |

Full suite on the final tree (`python scripts/run_tests.py tests/ test_screener.py -q`, parallel),
three consecutive runs under the same load: **3,453 passed, 1 skipped** each, in 7:21, 5:25 and
6:02. The mutual-exclusion test passed in all three; it will still fail now and then under load
until route 1 in section 3 is fixed, and when it does its message names the route.
