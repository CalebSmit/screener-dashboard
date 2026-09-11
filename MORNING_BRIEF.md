# Morning Brief - Friday 11 September 2026, 02:13

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

- `88b4b46 data: screener run 2026-09-11 - 502 scored, top: EXPE HST APA VLO EIX`
- `678017d brief: code session 2026-09-10`
- `eeb79a8 docs: changelog for the revisions change; correct three stale claims`
- `40454df test: 44 tests for fy1_revision_3m; update three pinned counts`
- `1d5e162 feat(dashboard): surface the FY1 revision in basis points of price`
- `b3c9f76 feat: add fy1_revision_3m, the revisions category's first actual revision`
- `ba668bb brief: data run 2026-09-10`
- `f68979d data: screener run 2026-09-10 - 502 scored, top: HST EXPE APA VLO ALL`

## The session's own account

> 2026-09-10 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes** - `logs/nightly-2026-09-09_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-09` |
> | Data loop published? | **Yes** - `logs/datarun-2026-09-10_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **37 rows, newest 2026-09-03, 3 effective observations at `1m`** (9 raw) against a gate of 8 |
> | Priority 0 | Fixed 2026-08-24, not weakened today. `_effective_observations()` untouched |
> | Top open roadmap item | **Priority 5, the sell-side workflow - 36 days old**, still untouched |
> 
> **Tests:** before 1117/1117, after **1161/1161**. Zero failures either side; the
> 44 new tests are `tests/test_fy1_revision.py`.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
> deferred. Thursday taken as the focus, building exactly what §8.7 of Monday's
> research note specified and Wednesday settled. Priority 5 was **not** taken -
> see *Next*, and note its age is now written down for the third consecutive
> session.
> 
> ### Did
> 
> **Shipped `fy1_revision_3m` and the revisions reweight as one change.** The
> category was named for revisions and contained none: 78 of its 100 points sat
> on the earnings-**surprise** family, whose drift Martineau (2022) documents as
> absent in large caps since 2006. It now leads with an actual revision, at
> weight 35. The category's **10% share of the composite did not change** - only
> the split inside it.
> 
> | Metric | Was | Now |
> |---|---|---|
> | `fy1_revision_3m` | - | **35** |
> | `analyst_surprise` | 38 | **15** |
> | `consecutive_beat_streak` | 20 | **10** |
> | `earnings_acceleration` | 20 | 20 |
> | `price_target_upside` | 12 | **10** |
> | `short_interest_ratio` | 10 | 10 |
> 
> Touched: `factor_engine.py` (fetch block, metric, `METRIC_COLS` / `METRIC_DIR`
> / `CAT_METRICS`), `config.yaml`, `schemas.py`, `generate_dashboard.py`,
> `stock_summary.py`, the golden fixture, three existing test modules whose
> pinned counts genuinely moved, and five documents.
> 
> **Verified live, not just against mocks.** Fetched 8 real tickers end-to-end:
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

