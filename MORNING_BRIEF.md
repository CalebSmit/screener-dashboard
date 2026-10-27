# Morning Brief - Tuesday 27 October 2026, 06:32

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
| Data run timing | **19 min** total; slowest `fetch.tickers` 659 s, `snapshot` 136 s, `fetch.retry` 125 s; **39 tickers rate-limited** |
| Context coverage | options usable 485/498 (13 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `a520fe1 log: correct the after-count (3,256 passed, not 3,250)`
- `8950247 log: 2026-10-27 - flaky gate 1 fixed at the root, the data-run proposal refused on its own number, drilldowns re-verified`
- `6f47700 brief: carry the data run's phase timing and the rate-limited count`
- `a0fd9b2 research: decide the data-run timing proposal - item 1 REFUSED on the number it asked for`
- `7593c90 test: the rerun scheduler's prefix tests no longer race worker startup`
- `d6b7799 brief: data run 2026-10-27`
- `7a4063a data: screener run 2026-10-27 - 498 scored, top: EXPE MU APA HST EG`
- `eda0bb8 log: 2026-10-26 owner-run night - speed (parallel suite, parallel engine reruns, lab first paint, data-run timing) and the docs it made stale`
- `d95032e research: rerun all 32 engine rows on today's code with --jobs 4 - provenance refresh, 0 statistics moved`
- `079d568 plan: row 76 - nothing else in the suite saw the Beneish coefficient move (the investor-profile failure was a copied runs/ folder's mtimes)`
- `cb870c3 rerun_experiments: --jobs N runs stale rows in worker processes, rows identical to serial`
- `047bf4b plan: claims mutation audit 2026-10-26 - the two mechanical searches, rows 63-88`
- `3ebc0f8 test: pin the points-add-up tolerance at rounding, both sides (drilldown.composite_chain, compare.composite_gap)`
- `d9830c3 research: where the 02:00 data run's 27 minutes go, what shipped, and what to cut next`
- `e4ec4cd tests: the one-observation snapshot tests stop rewriting the tracked IC history`

## The session's own account

> 2026-10-27 (06:00 session) - PRODUCT. A flaky ship gate fixed at the root, the open proposal refused on the number it asked for, and the drilldowns re-checked against the engine
> 
> **Health (rule 8, all five):** last code session ran? **yes** - the 2026-10-26 13:00 midday
> session ends "Run complete: shipped to main", tag `good/2026-10-26-1340`. Data loop published?
> **yes** - 2026-10-27 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored, live at 02:20.
> Evidence base at `1m` = **38 rows, newest 2026-09-25 (32 days ago, bound 40), 4 effective**.
> Priority 0: `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor
> Lab** (opened 2026-10-12, **age 15 days**) - not advanced today; the next non-retrospective
> Friday is 2026-11-06 and nothing here competed for it.
> **Tests:** before **1 failed, 3247 passed, 1 skipped** - the failure was *flaky, not mine*, and
> became the session's first work (below). After: **3256 passed, 1 skipped** - 3,248 collected at
> baseline plus the scheduler test split into two (+1) and 7 new brief tests, with the flake gone.
> **Owner queue / rotation:** `OWNER_FOCUS.md` item 1 sub-item 1 ("decide the OPEN proposals") was
> waiting with one, 1 day open, so it came before the Tuesday focus as the prompt requires. The
> Tuesday focus was then done in its verification half (drilldowns against the engine) and found
> nothing broken. Nothing deferred for a loop or a gate.
> 
> ### Did
> 
> **1. Gate 1 was flaky, and the flake was a test asserting a race.** The session's baseline came
> back `1 failed, 3247 passed` on
> `test_session_ends_clean.py::test_a_failed_worker_stops_the_run_with_the_serial_prefix_appended`,
> with `assert ['older'] == ['older', 'baseline', 'e1', 'e2', 'e3']`. The same test then passed
> alone, passed under `-n auto` in its own module, and the full suite passed twice more on the
> identical tree. **A nondeterministic gate 1 is worse than a red one**: the runner re-runs all four
> gates itself, so this could refuse a merge for a reason unconnected to the day's work, and would
> look like a different fault each time.
> 
> The cause is not in the scheduler. `run_parallel`'s documented guarantee is that the ledger holds
> **a** prefix of the serial order; the test asserted the **maximal** prefix, which additionally
> requires workers 0 and 2 to reach their first row before worker 1 reaches its failing one - a race
> between three Python interpreter startups, held off only by a 0.3 s sleep on two unrelated rows.
> The parallel suite shipped 2026-10-26 puts an xdist worker on every core, which is exactly what
> makes a process lose that race. **Reproduced 100%** by delaying the other workers' first rows, the
> thing CPU contention actually does to startup: the prefix collapses to `['older']` and the failure
> is byte-identical to the baseline's.
> 
> Fixed by removing the race rather than the assertion. The fake engine gains a done-marker per
> finished row and a barrier that waits on them, so e4 cannot fail until **every row another worker
> owns has finished** (c1 excluded - worker 1 runs it *after* e4, so waiting on it would deadlock
> rather than order anything). The maximal-prefix assertion is now a fact of the run instead of a
> hope, and **strictly stronger than before**: e5, e6, v1 and v2 are all demonstrably finished and
> still withheld, which is the "nothing after it even if it finished" clause the old sleep could only
> hope for. A second test pins the outcome that actually fired on the live machine - a short prefix -
> with assertions that are universal invariants (a prefix, never a hole, nothing at or after the
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

