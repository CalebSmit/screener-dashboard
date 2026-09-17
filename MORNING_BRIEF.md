# Morning Brief - Thursday 17 September 2026, 02:13

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-17T02:00:02.915870 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, CAH, BBY, JBHT |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (11 rows, but overlapping windows are not independent; 44 rows across all horizons), newest 2026-09-10 |

## What changed in the repo

- `090abba data: screener run 2026-09-17 - 502 scored, top: HST EXPE CAH BBY JBHT`
- `dd028f6 brief: code session 2026-09-16`
- `3a10f3c docs: correct the plans this session proved wrong, plus the session log`
- `21bd821 synthesis: no hold band, and the rule for when one may be chosen`
- `28cd94a research: make a note's own numbers re-runnable`
- `e01ca25 brief: data run 2026-09-16`
- `fdd85a4 data: screener run 2026-09-16 - 502 scored, top: EXPE HST VLO CAH BBY`

## The session's own account

> 2026-09-16 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.
> 
> **Health (rule 8, all five):**
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **Yes** - `logs/nightly-2026-09-15_060000.log` ends "Run complete: shipped to main", tagged `good/2026-09-15` |
> | Data loop published? | **Yes** - `logs/datarun-2026-09-16_020000.log` ends "Data loop complete", HEALTH: PASS, 502 scored, top EXPE HST VLO CAH BBY |
> | Evidence base | **43 rows, newest 2026-09-09, 3 effective observations at `1m`** (11 raw) against a gate of 8. Up from 42 yesterday - moving |
> | Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call touched. Today **extended its lesson** to a third place - see below |
> | Top open roadmap item | **Priority 5, the sell-side workflow - 42 days old.** Its remaining half (the hold band) is now **settled as "not yet, until ~2027-04"** with a pre-registered rule rather than left vague. Next unblocked item is **Priority 3, backtest v2**, plan file dated 2026-08-25 - **22 days** |
> 
> **Tests:** before **1264/1264**, after **1264/1264**. No production code changed;
> this is a no-new-failures check, not a claim of new coverage.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
> deferred. Wednesday's synthesis taken as written, working §8 of Monday's note.
> 
> ### Did
> 
> **Wrote §8, the design section Monday's note was left open for - and in doing so
> found that two of that note's headline numbers were wrong.** The deliverable is
> the design; the correction is the more important half.
> 
> **1. §6.3's "a 25/50 band never fires" was an estimator artifact.** Monday walked
> **one path** through 18 runs and got zero. Taking instead *every ordered pair* of
> comparable runs at a given calendar spacing - 34 runs now - a 2x band fires at
> **1.65%** of weekly holding-looks. It is rare, not dead.
> 
> **2. But the pairwise estimator has the project's own oldest defect, and this is
> the finding of the day.** **34 runs yield 72 pairs at a fortnight's spacing**, so
> each run feeds many pairs and the observations are nowhere near independent -
> the identical trap `research/2026-08-10-ic-evidence-independence.md` found in the
> IC series and `improvement_engine._effective_observations()` was built to guard
> against. **Nobody had noticed it applies to rank statistics too.** Restricting to
> non-overlapping pairs:
> 
> | spacing | pairs all → disjoint | B=50 breach all → disjoint |
> |---|---|---|
> | 5-9 days | 68 → **8** | 1.65% → **4.50%** |
> | 12-18 days | 72 → **4** | 3.99% → **10.89%** |
> | 25-35 days | 37 → **2** | 1.93% → **4.00%** |
> 
> The overlapping estimator **understates wide-band breach rates by 2-3x** - the
> same order as the ~2.35x the IC note measured - and the honest sample is **8
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

