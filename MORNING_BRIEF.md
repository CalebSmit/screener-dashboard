# Morning Brief - Tuesday 25 August 2026, 06:25

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-25T02:00:03.840225 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | HST, EXPE, APA, CF, EIX |
| Evidence for weight changes | 2 of 8 needed at the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows across all horizons), newest 2026-08-14 |

## What changed in the repo

- `d33dd04 docs: record the history gate, its evidence, and a defect it uncovered`
- `7e7d23c test: syntax-check the emitted dashboard JavaScript`
- `99fa64d feat(dashboard): What Changed - movers, rank deltas and per-stock history`
- `db294e5 feat: history.py - a quality-gated historical spine for the dashboard`
- `0232d40 brief: data run 2026-08-25`
- `4a0a90f data: screener run 2026-08-25 - 501 scored, top: HST EXPE APA CF EIX`
- `b149c65 brief: evening session 2026-08-24`
- `b99158b log: evening session - priority 1 closed, tomorrow's evidence path de-risked`
- `8a87b3e fix: the screener refuses to fabricate data when the network is down`
- `30ee3c5 brief: code session 2026-08-24`
- `a118727 docs: priority 0 is done; record the evidence and what has not changed`
- `5200fec fix: report effective observations, not the raw row count`
- `8cc18d9 data: repair the evidence base, and the 1-month backfill it unblocks`
- `9a03e96 fix: the improvement engine's observation count was meaningless`
- `cff2675 brief: data run 2026-08-24`

## The session's own account

> 2026-08-25 - PRODUCT. The dashboard gets a time dimension.
> 
> **Health numbers:** last code session **ran and shipped** (2026-08-24,
> `good/2026-08-24`); data loop **published** 02:11 today, HEALTH: PASS, 501/501
> price coverage, 0 fetch failures, 0 synthetic substitutions; evidence base
> **23 rows, newest 2026-08-14, 2 effective observations at the `1m` horizon**;
> priority 0 **DONE** (unchanged since 08-24).
> 
> **Tests:** before **596/596**, after **627/627** (+31, no pre-existing
> failures)
> **Data loop:** healthy - see health numbers above.
> 
> ### Did
> 
> Shipped the dashboard's first **time dimension**. This was gap 1 in
> `.claude/plan/dashboard-north-star.md` ("There is no time dimension. This is
> the biggest one... **Start here.**") and priority 2 in `CLAUDE.md`. Three
> surfaces, all fed by a new `history.py` built from the snapshots the data loop
> has been writing all along:
> 
> 1. **A "What Changed" section** below the KPI row - biggest rank movers in each
>    direction, each with an inline sparkline, the category that moved furthest,
>    and the current rank.
> 2. **A Δ column** in the Full Universe table, sortable, showing the rank change
>    since the previous comparable run.
> 3. **A "Rank History" block** in the per-stock drill-down - the full rank path
>    plus the four categories that moved most, since the last run and since ~1
>    month.
> 
> **The hard part was not the arithmetic. It was deciding which runs are
> comparable to each other**, and getting that wrong would have been worse than
> shipping nothing.
> 
> `2026-07-28` is a degraded run sitting in `improvement/snapshots/`. It predates
> `check_run_health.py`, so nothing ever blocked it. Its ranks correlate with the
> run before at Spearman **0.016** and with the run after at **-0.020**. Diffed
> naively it reports **411 of 501 stocks (82%) moving more than 50 ranks**. The
> flagship new panel would have opened with 15 fictional movers.
> 
> **First attempt was wrong and the real data caught it.** I reused
> `check_run_health`'s dispersion rule (>20% below trailing median) on the theory
> that reusing an already-justified threshold beats inventing one. It excluded
> **16 of 20 runs**. Risk dispersion has drifted legitimately from 26.7 (Feb) to
> 19.5 (Aug), and because my baseline only recorded *kept* runs it froze in
> February and every later run failed against it - one exclusion cascading into
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

