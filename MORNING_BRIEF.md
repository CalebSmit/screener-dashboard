# Morning Brief - Thursday 10 September 2026, 02:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **failed** - last ran today |
| Dashboard data from | 2026-09-10T02:00:04.141425 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, VLO, ALL |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (9 rows, but overlapping windows are not independent; 37 rows across all horizons), newest 2026-09-03 |

## What changed in the repo

- `f68979d data: screener run 2026-09-10 - 502 scored, top: HST EXPE APA VLO ALL`
- `06a897d brief: code session 2026-09-09`
- `2dab71b log: 2026-09-09 synthesis session`
- `d88f1fd changelog: confirm the 2026-09-02 risk-category change against its prediction`
- `4a74d5d research: synthesis section on the revisions category (2026-09-07 note, Â§8)`
- `bed35c0 brief: data run 2026-09-09`
- `a195b07 data: screener run 2026-09-09 - 502 scored, top: HST EXPE APA VLO ALL`

## The session's own account

> 2026-09-09 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.
> 
> ### Health numbers (rule 8, all five)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-09-08_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | `logs/datarun-2026-09-09_020001.log` - "Data loop complete", **HEALTH: PASS**, 502 scored |
> | Evidence base | **36 rows, newest 2026-09-02, 3 effective observations at `1m`** (9 raw) against a gate of 8 |
> | Priority 0 | DONE 2026-08-24, not reopened |
> | Top open roadmap item | **Priority 5, sell-side workflow** - north-star gap 2, dated 2026-08-05, **35 days open**. Not taken today; see "Owner queue / rotation" |
> 
> **Tests:** before 1117/1117, after **1117/1117** (no pre-existing failures; no
> code changed today, so no test changed)
> **Data loop:** healthy. Evidence base moved 34 -> 36 rows, newest 09-01 -> 09-02.
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty. Nothing deferred
> for a stalled loop or a failing gate. ISO week 37, Wednesday - synthesis day,
> taken as the focus, with Monday's note
> (`research/2026-09-07-revisions-category-has-no-revisions.md`) as its subject,
> exactly as that note's §6 instructed. Priority 5 was **not** taken: Monday
> deferred a specific measurement to Wednesday and a synthesis day that skips it
> leaves Thursday building on an unverified design. Priority 5 remains the top
> open north-star item and its age is written above so the trade stays visible.
> 
> ### Did
> 
> **Answered the coherence question Monday deferred, on the full 502-name
> universe, and settled the design for Thursday's build.** No code, weight,
> threshold or scoring path changed today. Output is §8 of the research note
> (~330 lines) plus a measured confirmation against the 2026-09-02 changelog
> entry.
> 
> **First, the counterfactual was made trustworthy.** Before computing anything
> hypothetical, the published composite was reproduced from its own inputs -
> the eight category scores through `compute_composite`'s per-row
> renormalisation, the revisions category rebuilt from its five published metric
> percentiles, and the coverage discount reconstructed from per-stock metric
> presence split by `_BANK_ONLY_METRICS`/`_NONBANK_ONLY_METRICS`. **Max absolute
> error 0.000000000000 on all 502 names**, including the three where the
> discount actually bites (FDXF at 62.5% coverage, FISV, L). A counterfactual is
> only worth reading if the factual reproduces first.
> 
> *Trap for the next session, now written down:* the published `Composite` is
> stored rounded to 2dp while category scores are stored at full precision, so a
> naive comparison shows a spurious ~0.005 residual on **every** name and looks
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

