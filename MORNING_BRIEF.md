# Morning Brief - Wednesday 07 October 2026, 02:14

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **stopped deliberately** - last ran today |
| Dashboard data from | 2026-10-07T02:00:04.100170 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, DLTR, BMY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (24 rows, but overlapping windows are not independent; 71 rows across all horizons), newest 2026-09-30 |

## Things that needed attention

- SESSION DID NOT RUN: API error 429 - You've hit your session limit ┬╖ resets 11am (America/Chicago)
- Treating this as a failure, not as 'nothing to do'. No success marker will be written,
- so the catch-up trigger will retry rather than skipping the day.
- GATE 4 clean tree: FAIL - uncommitted changes remain
- M generate_dashboard.py
- ?? plan/dashboard-design-system.md
- ?? scripts/_mono_sweep.py
- ?? scripts/_shape_sweep.py
- ?? scripts/_token_sweep.py
- ?? scripts/check_contrast.py
- ?? scripts/shot_dashboard.py
- SHIP GATES FAILED: clean-tree. Not merging.
- Work pushed to nightly/2026-10-06 for inspection. origin/main verified clean (reverted if the earlier push had reached it).
- Run finished with failing gates - see above.

## What changed in the repo

- `99c2431 data: screener run 2026-10-07 - 502 scored, top: EXPE HST BBY DLTR BMY`
- `770bb9e plan: full dashboard redesign plan and calculation-transparency plan, with three live defects found while writing them`
- `d5f67ff dashboard: ship design-system stage 1, and stop an interrupted session from stranding or leaking its work`
- `54bec97 wip: salvage the 2026-10-06 session's design-system pass (cut off by a 429 before it could commit)`
- `ca6f0ac brief: code session 2026-10-06 - SESSION DID NOT RUN`
- `df40d95 brief: data run 2026-10-06`
- `2913fac data: screener run 2026-10-06 - 502 scored, top: EXPE HST BBY DLTR CAH`
- `a46eb1a owner: make the dashboard look and feel premium - queue it, brief it, standing rule`

## The session's own account

> 2026-10-06 (evening) - PLANNING, owner-run session: a surface-by-surface redesign plan and a calculation-transparency plan; three live defects found, none fixed yet
> 
> Written by an interactive session at the owner's request. **No product code
> changed.** Two plans, three research measurements, and queue/prompt edits.
> 
> ### Health numbers (rule 8, all five)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | **No** - the 06:00 session was cut off by a 429; see the entry above. Nothing new has run since |
> | Data loop published? | **Yes** - `logs/datarun-2026-10-06_020001.log` ends "Data loop complete", HEALTH: PASS |
> | Evidence base | at `1m`: **23 rows, newest 2026-09-04, 4 effective observations** (re-read with `scripts/report_evidence.py`); lag 32 days, inside the 40-day bound |
> | Priority 0 | Fixed 2026-08-24, not touched |
> | Top open roadmap item | **0.8, calculation transparency and the false claims** - opened today, **0 days**. Below it, priority 3 (backtest v2) is **42 days** old and was not taken |
> 
> **Tests:** only docs, prompt and two read-only research scripts changed;
> `tests/test_owner_focus.py`, `test_governance.py` and `test_scripts_static.py`
> pass (76). The full suite was not re-run; nothing it covers changed.
> 
> ### What the owner asked for
> 
> *"Get it set up with a full blown plan to make it look better, every single place
> on the dashboard. Also, I want it to have more data integrity, or calculation
> integrity - we can see how things score, but we don't actually see the numbers going
> into any calculations in the breakout details for each company."*
> 
> ### What was written
> 
> - `plan/dashboard-redesign-master.md` - 18 surfaces, each with what was seen on the
>   live page at 1440 and 375px and the target; stages D2-D8; measured budgets; the
>   order of work against the other plan.
> - `plan/calculation-transparency.md` - the goal, eight design principles, stages
>   T0a, T0b, T1-T6, a verification protocol, sources, and what it deliberately does
>   not do.
> - `OWNER_FOCUS.md` - a new top item for transparency; the premium item now points at
>   the master plan. `CLAUDE.md` - a standing rule and priority 0.8. `prompts/nightly.md`
>   - a paragraph saying a named plan is the brief and one stage is one session.
> - `research/measurements/2026-10-06-calculation-reproducibility.py` and
>   `...-rank-sentence-claim.py` - the two measurements below, runnable from the repo.
> 
> ### Three defects found while auditing - all on the live site, none fixed
> 
> Each was measured against the live payload; re-run the two scripts to reproduce.
> 
> 1. **The per-metric weights in the drilldown are not the weights used, for 276 of 502
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

