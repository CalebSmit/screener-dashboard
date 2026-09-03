# Morning Brief - Thursday 03 September 2026, 02:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **stopped deliberately** - last ran today |
| Dashboard data from | 2026-09-03T02:00:03.208847 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CF, VLO |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 31 rows across all horizons), newest 2026-08-27 |

## Things that needed attention

- GATE 1 tests: FAIL (=========================== short test summary info =========================== FAILED test_screener.py::TestCacheRoundTrip::test_parquet_roundtrip - assert ... 1 failed, 904 passed, 32 warnings in 56.69s)
- SHIP GATES FAILED: tests. Not merging.
- Session had already committed onto local main. Resetting local main back to 09d856b0; work preserved on nightly/2026-09-02.
- Work pushed to nightly/2026-09-02 for inspection. main is untouched.
- Run finished with failing gates - see above.
- Brief committed locally; push failed.

## What changed in the repo

- `115a5d6 data: screener run 2026-09-03 - 502 scored, top: HST EXPE APA CF VLO`
- `b6b3e58 docs: log the investigation into last night's ship-gate failure`
- `7875cb5 fix: the wrapper's ship-gate failure recovery could not undo an already-pushed merge`
- `579bef3 brief: code session 2026-09-02`
- `25dc8ed docs: synthesis note, changelog entry, and the 2026-09-02 session log`
- `4e55a1d fix: the data loop's daily evidence readout printed the raw row count`
- `049bc8d fix: the risk category scored two metrics that were momentum, not risk`
- `09d856b brief: data run 2026-09-02`
- `3024a59 data: screener run 2026-09-02 - 502 scored, top: HST EXPE APA CF EIX`

## The session's own account

> 2026-09-02 (evening) - Owner-run: investigating last night's first-ever ship-gate failure
> 
> Owner asked how last night's run went. Last night's 6 AM session (Wednesday,
> SYNTHESIS) logged `SHIP GATES FAILED: tests. Not merging.` - the first gate
> failure in this project's history. What follows is what that actually meant,
> found by checking directly rather than trusting the log's own conclusion.
> 
> ### What actually happened, in order
> 
> 1. The session did real work: found the risk category was scoring two metrics
>    that were momentum, not risk, fixed it, wrote a synthesis note.
> 2. Per `prompts/nightly.md`'s own instructions, the session merged and pushed
>    `main` itself, as its normal final step. That landed on `origin/main` at
>    06:24:22 - confirmed from the commit's own timestamp.
> 3. `nightly-screener.ps1`'s independent re-verification then ran a fresh copy
>    of the test suite two minutes later and hit `test_parquet_roundtrip` FAIL.
> 4. The wrapper's recovery path reset **local** `main` back to the last known
>    good commit and logged "main is untouched." **That claim was wrong.** The
>    session's own push in step 2 had already reached `origin/main` - the
>    branch GitHub Pages actually serves - before this independent check ever
>    ran. A local reset cannot undo a push it did not make.
> 
> ### Checked, not assumed
> 
> - **The live public dashboard was never at risk.** Fetched it directly:
>   serving correct data from the clean 02:00 run.
> - **Last night's risk-category fix was legitimate**, not something that
>   needed reverting. `git log origin/main` showed the merge had genuinely
>   landed; re-running the exact same suite against that exact commit found no
>   regression in it.
> - **`test_parquet_roundtrip` is not flaky - it is a genuine, reproducible
>   test-isolation bug**, and I only believed "flaky" for about ten minutes
>   before it failed again, on demand, on this machine, tonight. Root cause:
>   it calls the real `write_scores_parquet`/`_find_latest_cache` against the
>   real, shared `cache/` directory. `_find_latest_cache("factor_scores")` with
>   no hash filter globs every `factor_scores_*.parquet` for *any* date and
>   reverse-sorts filenames; a hex hash starting with a-f sorts ahead of one
>   starting with a digit, so a same-day real pipeline cache file
>   (`factor_scores_2bde439e06ad_20260902.parquet`) beat the test's own
>   `factor_scores_20260902.parquet` and the test silently read 502 real
>   production rows instead of its own 503 synthetic ones. This is exactly the
>   test-isolation gap CLAUDE.md priority 8 already named, just never traced to
>   a specific test before. Fixed: isolated to `tmp_path` via `monkeypatch`,
>   confirmed deterministic across 4 consecutive runs with the real same-day
>   cache file present and untouched throughout.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

