# Morning Brief - Monday 24 August 2026, 06:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-24T02:00:03.959537 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | HST, EXPE, APA, EIX, CF |
| Evidence for weight changes | 2 of 8 needed at the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows across all horizons), newest 2026-08-14 |

## What changed in the repo

- `a118727 docs: priority 0 is done; record the evidence and what has not changed`
- `5200fec fix: report effective observations, not the raw row count`
- `8cc18d9 data: repair the evidence base, and the 1-month backfill it unblocks`
- `9a03e96 fix: the improvement engine's observation count was meaningless`
- `cff2675 brief: data run 2026-08-24`
- `37ab0c9 data: screener run 2026-08-24 - 501 scored, top: HST EXPE APA EIX CF`

## The session's own account

> 2026-08-24 - PRIORITY 0, all five steps. The number finally moved: 3 rows -> 23.
> 
> **Health numbers:** last code session **ran and shipped** (2026-08-21 06:16,
> `good/2026-08-21-0616`); data loop **published** 02:11 today, HEALTH: PASS, 501
> scored, 100% price coverage; evidence base **3 rows, newest 2026-02-22, 0
> effective observations at the `1m` optimization horizon** -> now **23 rows,
> newest 2026-08-14, 2 effective at `1m`**; priority 0 **FIXED**.
> 
> **Tests:** before 560/560, after **590/590**
> **Data loop:** healthy - `logs/datarun-2026-08-24_020001.log` ends "Data loop
> complete", HEALTH: PASS, 0 fetch failures, 0 synthetic substitutions.
> 
> ### Swapped the rotation focus, deliberately
> 
> Today was scheduled as **research**. I did not do research. `CLAUDE.md` priority
> 0 says the first session must fix the forward-return bug ahead of any rotation
> focus, the nightly prompt repeats that override, rule 8's three-session trigger
> had fired (the evidence numbers had not moved on 08-13, 08-21 or 08-21-later),
> and the previous session named step 5 as the single next thing. Three separate
> rules pointed at the same work. **No research note today**; Monday's slot is
> owed one, and the next session should take it.
> 
> ### Did
> 
> **All five priority-0 steps, plus a sixth defect found while fixing them.**
> Full detail in `METHODOLOGY_CHANGELOG.md` 2026-08-24. In short:
> 
> 1. `compute_forward_returns()` tracks eligibility per `(run_date, horizon)`, so
>    a date is revisited as it ages instead of being frozen at its 7-day state.
> 2. One snapshot per run date, not one per file.
> 3. `_effective_observations()` - non-overlapping windows - now feeds every gate.
> 4. Weekend run dates excluded.
> 5. `record_run_snapshot()` calls `compute_live_ic()` for all three horizons.
> 6. **New, found while fixing 1:** the price cache is keyed `(start, end)` and
>    `end` was the *current* date. My horizon fix would have turned every
>    revisited snapshot into a fresh full-universe yfinance download - about ten
>    of them in tomorrow's 02:00 run, against the rate limits that already cost
>    the loop 10-25% of its tickers. The fetch window is now bounded by the
>    horizon being measured, so the key is stable.
> 
> **Ran the real backfill rather than leaving it to discover itself overnight.**
> 2,495 new rows, 4 tickers failed out of ~500 (HOLX, CTRA, BK, EA - Yahoo
> "possibly delisted"). This is why the `1m` horizon has observations at all now.
> 
> **Fixed both places that report the evidence count to the owner.** This was a
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

