# Morning Brief - Friday 04 September 2026, 06:24

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-04T02:00:03.088823 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CF, VLO |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 32 rows across all horizons), newest 2026-08-28 |

## What changed in the repo

- `af19547 log: 2026-09-04 retrospective - 9 of 9 sessions shipped; the gates were an audit`
- `0fbb194 process: the ship gates decide before anything reaches main, not after`
- `1904829 fix: the morning brief could publish the work the ship gates refused`
- `af1ea03 brief: data run 2026-09-04`
- `25f34f1 data: screener run 2026-09-04 - 502 scored, top: HST EXPE APA CF VLO`
- `7beb43b brief: code session 2026-09-03`
- `0776450 log: 2026-09-03 build session - size tilt closed by measurement, remote branch sweep`
- `7438356 fix: merged nightly/* branches are now swept from origin, not just locally`
- `a8e685d docs: the size section described a compression the pipeline does not perform`
- `f4e81c6 brief: data run 2026-09-03`
- `115a5d6 data: screener run 2026-09-03 - 502 scored, top: HST EXPE APA CF VLO`

## The session's own account

> 2026-09-04 - RETROSPECTIVE. Evaluate whether this routine is producing value, and change the process where it is not.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-09-03_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | `logs/datarun-2026-09-04_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored, 0 fetch failures, price coverage 502/502 (100%) |
> | Evidence base | **32 rows, newest 2026-08-28, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
> | Priority 0 | DONE 2026-08-24, not reopened |
> | Top open roadmap item | **Priority 4, deterministic per-stock summaries - owner directive 2026-08-10, open 25 days** (this row is new; see Process changes) |
> 
> **Tests:** before 938/938, after **965/965** (+27, no pre-existing failures)
> **Data loop:** healthy
> **Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
> **Rotation:** ISO week 36, Friday, even week - retrospective.
> 
> Two carried-over checks first, both confirmed:
> 
> 1. **The remote branch sweep fired.** `logs/nightly-2026-09-04_060001.log` shows
>    12 "Swept stale merged remote branch" lines; `git branch -r` is down to
>    `origin/main`, `origin/master` and `origin/HEAD`. The 09-03 session left this
>    as the one thing it could not verify from its sandbox.
> 2. **The evidence base is moving.** 31 -> 32 rows, newest 08-27 -> 08-28. `1m`
>    sits at 3 effective for the fifth session, which is structural rather than a
>    defect: `1m` rows mature only as older run dates age past the horizon.
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 9 scheduled** (2026-08-24 to 2026-09-03), plus 6
>   owner-run evening/catch-up sessions.
> - **Genuinely valuable: 9 | Churn: 0 | Failed gates: 1** (2026-09-02, gate 1).
> 
> **1. What fraction of sessions produced something genuinely valuable? All nine.**
> The last retrospective measured 2 of 11 slots producing anything and 5 never
> firing. This period: 9 of 9 fired and 9 of 9 shipped. Named - 08-24 the
> evidence-base repair (5 defects, 3 IC rows -> 23), 08-25 `history.py` and the
> dashboard's time dimension, 08-26 the split-scale price guard (MNST was live at
> ~110 ranks too high), 08-27 the GitHub Actions loop watchdog, 08-28 the
> weight-transparency fix, 08-31 the size-factor research note, 09-01 the removal
> of winsorization (six megacaps published at an identical false $2,802.0B), 09-02
> the risk category shedding two momentum metrics, 09-03 the size tilt documented
> truthfully plus the origin branch sweep. Not one is churn. Sessions run 13.6-22.4
> minutes against a 4-hour limit; nothing came close.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

