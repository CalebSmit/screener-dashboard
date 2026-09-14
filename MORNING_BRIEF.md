# Morning Brief - Monday 14 September 2026, 02:13

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
| Dashboard data from | 2026-09-14T02:00:05.118731 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, VLO, APA, CAH |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (11 rows, but overlapping windows are not independent; 41 rows across all horizons), newest 2026-09-07 |

## What changed in the repo

- `6b8d071 data: screener run 2026-09-14 - 502 scored, top: EXPE HST VLO APA CAH`

## The session's own account

> 2026-09-11 - HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes** - `logs/nightly-2026-09-10_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-10` |
> | Data loop published? | **Yes** - `logs/datarun-2026-09-11_020001.log` ends "Data loop complete", 502 scored, top EXPE HST APA VLO EIX |
> | Evidence base | **39 rows, newest 2026-09-04, 3 effective observations at `1m`** (10 raw) against a gate of 8 |
> | Priority 0 | Fixed 2026-08-24, not weakened today. Nothing in this session touches `_effective_observations()` or any scoring path |
> | Top open roadmap item | **Priority 5, the sell-side workflow - 37 days old**, still untouched |
> 
> **Tests:** before 1161/1161, after **1193/1193**. Zero failures either side; the
> 32 new tests are `tests/test_percentile_direction.py`.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
> deferred. Friday's focus taken as written. Priority 5 was **not** taken - see
> *Next*, fourth consecutive session its age has been written down.
> 
> ### Did
> 
> **The dashboard published a percentile whose obvious reading was backwards, and
> now it doesn't.** `compute_sector_percentiles()` does `ranks = 100 - ranks`
> wherever `METRIC_DIR` is `False`, so a published percentile always means "best
> in its sector" and never "largest". That is **13 of the 37 published metrics**,
> and nothing on the page said so. Measured on the live payload:
> 
> | Stock | EV/EBITDA | Published percentile |
> |---|---|---|
> | HON | 6.95 | **99** |
> | AXON | 98.61 | **0** |
> 
> Same shape for beta (RSG -0.37 at the 99th vs CVNA 2.35 at the 0th) and PEG
> (UAL 0.24 at the 99th vs KMI 28.34 at the 0th). In prose it read as a flat
> contradiction: *"the 97th sector percentile on EV/EBITDA (9.02)"* (HST, live).
> 
> This is the surface whose entire purpose is explaining **why** a stock ranks
> where it does. A student who learns the convention backwards misreads every
> valuation and risk metric on the site - which is the exact question this day
> exists to ask.
> 
> Three fixes, plus the category columns:
> 
> 1. **`metric_meta[m]["dir"]`, derived from `factor_engine.METRIC_DIR`** rather
>    than written out. This is the load-bearing decision: the page cannot claim a
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

