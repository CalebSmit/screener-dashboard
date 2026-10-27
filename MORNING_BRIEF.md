# Morning Brief - Tuesday 27 October 2026, 02:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-27T02:00:05.759618 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (38 rows, but overlapping windows are not independent; 99 rows across all horizons), newest 2026-10-19 |
| Context signals | Context signals: 14 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 424/425 within 0.5%; operating cash flow 408/412 within 0.5%; net income 398/421 within 0.5%; market value 443/484 within 5%; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#20: net income 16% off the filing) |
| Context coverage | options usable 485/498 (13 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **overlap the SEC Form 4 refresh (and the other SEC/FRED passes) with** - open 1 day - `research/2026-10-26-data-run-timing.md`

## What changed in the repo

- `7a4063a data: screener run 2026-10-27 - 498 scored, top: EXPE MU APA HST EG`
- `eda0bb8 log: 2026-10-26 owner-run night - speed (parallel suite, parallel engine reruns, lab first paint, data-run timing) and the docs it made stale`
- `d95032e research: rerun all 32 engine rows on today's code with --jobs 4 - provenance refresh, 0 statistics moved`
- `079d568 plan: row 76 - nothing else in the suite saw the Beneish coefficient move (the investor-profile failure was a copied runs/ folder's mtimes)`
- `cb870c3 rerun_experiments: --jobs N runs stale rows in worker processes, rows identical to serial`
- `047bf4b plan: claims mutation audit 2026-10-26 - the two mechanical searches, rows 63-88`
- `3ebc0f8 test: pin the points-add-up tolerance at rounding, both sides (drilldown.composite_chain, compare.composite_gap)`
- `d9830c3 research: where the 02:00 data run's 27 minutes go, what shipped, and what to cut next`
- `e4ec4cd tests: the one-observation snapshot tests stop rewriting the tracked IC history`
- `c493e01 tests: run the suite in parallel where pytest-xdist is installed, serially where not`
- `0040a94 lab: the first paint, measured (6.4 s on a phone) and fixed (0.95 s) - every figure unchanged`
- `3b96dc8 test: drilldown.inputs_rebuild gets a check that runs compute_metrics, not only the committed payload`
- `4a77109 data run: record where the time goes, and drop two kinds of request whose result nothing kept`
- `0da5189 test: register the literal pin on S&P DJI's 20% switching margin for overview.share_classes`
- `0460b74 test: claims mutation audit items 5-6 - pin the stated numbers against the code, not the constant or the payload`

## The session's own account

> 2026-10-26 (owner-run, night) - SPEED. The suite in 1:51 instead of 6-7 minutes, engine reruns in parallel, the lab's phone first paint 6.3 s -> 0.95 s, ~4 minutes off the 02:00 run, and three bot ROADMAP items
> 
> **Health (rule 8, all five):** last code session ran? **yes** - the 13:00 midday session ended
> "Run complete: shipped to main" at 13:40, tag `good/2026-10-26-1340`. Data loop published?
> **yes** - 2026-10-26 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored. Evidence base at
> `1m` = **36 rows, newest 2026-09-23 (33 days ago, bound 40), 4 effective**. Priority 0:
> `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 14 days**) - advanced (its first paint, below).
> **Tests:** before **3159 passed, 1 skipped** in 7:19 serial. After, on the merged tree: **3248
> passed, 1 skipped in 1:51** through `scripts/run_tests.py` (parallel), and 6:09 when forced serial
> on the same tree (3,249 collected either way). `python scripts/rehearse_publish.py`: PASSED,
> published files restored byte for byte.
> **Owner queue:** the owner's brief for the night was speed first. `OWNER_FOCUS.md` 2026-10-20 item
> 7 (suite runtime, `lab_data.js`, the fresh-worktree failures) is **done in all three parts**.
> Nothing scheduled was reversed; the price-target decision from 13:00 is untouched.
> 
> ### How the night was run
> 
> Measured first, then seven agents in isolated worktrees (two in the bot repo), each told not to
> push or merge; every diff was read before it was merged onto `staging/2026-10-26-owner`, the gates
> were run once on the combined tree, then `--no-ff` to `main`. The five screener branches merged with
> no conflicts; the two bot branches conflicted only in `ROADMAP.md`/`SESSION_LOG.md`.
> 
> ### Did - screener
> 
> **1. The suite runs in parallel** (`scripts/run_tests.py`, pytest-xdist 3.8.0 pinned). The slow
> tail was a long tail - the slowest test is 16 s - so distribution, not test surgery, was the lever.
> - **Gate 1** (`scripts/nightly-screener.ps1`) and the **data loop's claims gate**
>   (`scripts/check_published_claims.py`) add `-n auto` only when xdist is importable; otherwise the
>   byte-for-byte serial command. pytest exit 3/4 from a parallel run (xdist could not run the suite -
>   not a verdict) reruns serially and that run decides; a failing test is never retried.
>   `scripts/data-run.ps1` is **unchanged**.
> - **The `conftest.py` tripwire works under xdist:** the controller snapshots the protected files
>   before any worker starts and hands each worker the digests; a per-worker snapshot would have
>   snapshotted another worker's write and "restored" it. Proved with a throwaway writer under `-n 4`
>   and pinned by a real `-n 2` run in `tests/test_parallel_suite.py` (32 tests).
> - **One real isolation defect:** `test_one_observation_per_run_date`'s fixture left the two
>   improvement-history paths pointing at the repo, so every run re-read the 21 MB
>   `performance_history.csv` three times and **rewrote the tracked `improvement/live_ic_history.csv`**
>   (byte-identical, so no tripwire saw it - but two parallel writers could have dropped rows from the
>   evidence base). Now `tmp_path`; the module went 29.4 s -> 2.4 s.
> - Three git-sandbox modules build their origin+clone once per module and copy it (42.7 s -> 26.2 s).
> - **Fresh worktrees:** the three tests that needed gitignored `runs/` skip with `MISSING DATA: ...`
>   only when it is absent. **Copy data into a worktree with `cp -rp`** - two agents lost time to
>   `test_investor_profiles` choosing the wrong "newest run" from a copy whose mtimes `cp -r` reset.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

