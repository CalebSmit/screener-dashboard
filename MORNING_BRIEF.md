# Morning Brief - Thursday 03 September 2026, 06:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-03T02:00:03.208847 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CF, VLO |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 31 rows across all horizons), newest 2026-08-27 |

## What changed in the repo

- `0776450 log: 2026-09-03 build session - size tilt closed by measurement, remote branch sweep`
- `7438356 fix: merged nightly/* branches are now swept from origin, not just locally`
- `a8e685d docs: the size section described a compression the pipeline does not perform`
- `f4e81c6 brief: data run 2026-09-03`
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

> 2026-09-03 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-09-02_060001.log` - ran; gate failure investigated and resolved the same evening (see 09-02 evening entry) |
> | Data loop published? | `logs/datarun-2026-09-03_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored, 0 fetch failures, price coverage 502/502 (100%) |
> | Evidence base | **31 rows, newest 2026-08-27, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
> | Priority 0 | DONE 2026-08-24, not reopened |
> 
> **Tests:** before 917/917, after **938/938** (+21, no pre-existing failures)
> **Data loop:** healthy
> **Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
> **Rotation:** ISO week 36, Thursday. Build day.
> 
> ### First: the two checks 09-02 asked for, both confirmed
> 
> 1. **Wednesday's risk-category fix reached the live site.** On the 09-03
>    published run, `momentum ~ risk` is **+0.100** (it was +0.516 before the fix,
>    and 09-02 predicted "near +0.15"). `sharpe_ratio ~ volatility` is +0.012,
>    confirming the two dropped metrics carried essentially no dispersion signal.
> 2. **The evidence readout prints the honest number.** The 02:00 log reads
>    "3 effective (8 rows) at the 1m horizon" - the effective count, not the raw
>    row count that `CLAUDE.md` rule 8 names as how this went wrong the first
>    time. `scripts/report_evidence.py` works in production, which the 09-02
>    session could not verify from its sandbox.
> 
> ### Did
> 
> **1. Closed Monday's Candidate 1 - the size tilt - by measurement, and the
> answer was "do not change the methodology; the documentation was false".**
> 
> This was the week's open thread: Monday found the tilt far more aggressive than
> the practitioner standard it resembles, Wednesday deferred it to a
> pre-registered measurement, today ran it. All numbers from the 09-03 published
> run, recomputed through the **real** `factor_engine.compute_composite` (exact
> reproduction of the published composite, err 0.0).
> 
> **The pre-registered criterion did not discriminate.** "Refuted if the
> compressed version changes the top 50 by fewer than ~2 names" returns 1, 2, 3
> or 5 names depending only on which steepness constant you pick - and there is no
> evidence for any particular one. A criterion whose verdict is set by a free
> parameter is not a criterion.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

