# Morning Brief - Friday 11 September 2026, 06:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-11T02:00:03.557036 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, APA, VLO, EIX |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (10 rows, but overlapping windows are not independent; 39 rows across all horizons), newest 2026-09-04 |

## What changed in the repo

- `f545136 docs: changelog, inventory and log for the percentile-direction change`
- `a3f4ef1 teach: say that the published percentile means best, not largest`
- `78b86fd brief: data run 2026-09-11`
- `88b4b46 data: screener run 2026-09-11 - 502 scored, top: EXPE HST APA VLO EIX`
- `678017d brief: code session 2026-09-10`
- `eeb79a8 docs: changelog for the revisions change; correct three stale claims`
- `40454df test: 44 tests for fy1_revision_3m; update three pinned counts`
- `1d5e162 feat(dashboard): surface the FY1 revision in basis points of price`
- `b3c9f76 feat: add fy1_revision_3m, the revisions category's first actual revision`
- `ba668bb brief: data run 2026-09-10`
- `f68979d data: screener run 2026-09-10 - 502 scored, top: HST EXPE APA VLO ALL`

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

