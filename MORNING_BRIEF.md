# Morning Brief - Wednesday 30 September 2026, 06:34

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-30T02:00:04.133596 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, TRV, ALL, BBY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (19 rows, but overlapping windows are not independent; 61 rows across all horizons), newest 2026-09-23 |

## What changed in the repo

- `2c6c2d6 log: 2026-09-30 - priority 3 procurement decision, rotation swap recorded`
- `e447cdd plan: record the procurement decision and make look-ahead the next step`
- `948fb7b research: the delisted-price source is $199/yr, and cost was never the blocker`
- `b37ea07 measure: census the delisted-price requirement for backtest v2`
- `ebb1068 brief: data run 2026-09-30`
- `114da5c data: screener run 2026-09-30 - 502 scored, top: EXPE HST TRV ALL BBY`
- `056fa4e brief: code session 2026-09-29`
- `8334521 log: changelog, nightly log, and both dashboard plans updated for gap 4`
- `5c49739 build: regenerate dashboard artifacts for the earnings surface`
- `733e168 feat: surface each stock's next earnings date (north-star gap 4)`
- `859ec29 brief: data run 2026-09-29`
- `826afad data: screener run 2026-09-29 - 502 scored, top: EXPE HST MPC CAH VLO`

## The session's own account

> 2026-09-30 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-29_060001.log` ends "Run complete: shipped to main",
> tagged `good/2026-09-29` | data loop published? **yes** -
> `logs/datarun-2026-09-30_020001.log` ends "Data loop complete", HEALTH: PASS,
> 502 scored, top EXPE HST TRV ALL BBY | evidence base at `1m` = **19 rows, newest
> 2026-08-31 (30 days ago, bound 40), 4 effective** - the middle of the 30-33-day
> steady state, healthy | priority 0 **fixed 2026-08-24, not weakened**
> (`_effective_observations()` still gates, `allow_auto_apply` still false, 4
> effective against a gate of 8) | top open roadmap item: **priority 3, backtest
> v2 - 36 days old**, and **today I took it** rather than deferring it a ninth
> time.
> 
> **Tests:** before **1615/1615**, after **1615/1615** - no production code
> changed, by design.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so nothing to
> claim or move to Done. **I swapped the nominal focus, and this is the record of
> it.** Wednesday synthesises Monday's research note; the 2026-09-28 Monday was
> consumed by a red ship gate and produced no note, and it said so - "the research
> rotation lost its Monday ... nothing is half-finished and no topic is owed."
> There was therefore nothing to synthesise. Rather than manufacture a synthesis,
> I took the thing the last two sessions both nominated in writing: *"the next
> session that is not carrying an owner item or a broken loop should price that
> data source and write the answer down, even if the answer is 'too expensive'."*
> Owner queue empty, both loops healthy, baseline green - that was today. Nothing
> was deferred for a stalled loop or a failing gate.
> 
> ### Did
> 
> **Answered the procurement decision that has gated backtest v2 steps 2-4 since
> 2026-09-24, and the answer is that it was never the blocker.**
> `research/2026-09-30-delisted-price-source-cost.md`, plus two measurement
> scripts that reproduce every number.
> 
> **1. Why this and not the rotation.** Priority 3 had been deferred by **eight of
> the ten sessions** before today. Every deferral was defensible on the day and
> every log entry said so, which is precisely the pattern rule 8's roadmap line
> exists to expose: the item is a *decision*, not code, so it lost every fair
> fight against work that could be finished. It cannot be un-deferred by finding a
> better day. It had to be decided.
> 
> **2. The answer: $19 to download, $199/yr to keep.** Sharadar Prices, 10-year
> history, covers 2020-2026 - the flagship full-history tier is not needed. Priced
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

