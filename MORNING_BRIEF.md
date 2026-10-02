# Morning Brief - Friday 02 October 2026, 02:15

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-02T02:00:03.760075 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, CAH, VLO |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (21 rows, but overlapping windows are not independent; 65 rows across all horizons), newest 2026-09-25 |

## What changed in the repo

- `f1de3c1 data: screener run 2026-10-02 - 502 scored, top: EXPE HST BBY CAH VLO`
- `220e8fb brief: code session 2026-10-01`
- `3624b73 log: 2026-10-01 - look-ahead sized, backtest-v2 step 1 closed`
- `2f80627 docs: correct backtest.py's look-ahead claim, reverse the plan's sequencing`
- `1de3975 research: look-ahead is >= 63.2% of the panel, 5.5x survivorship`
- `17c67a8 lookahead: diagnostic to size backtest.py's look-ahead bias`
- `358af76 brief: data run 2026-10-01`
- `7d24629 data: screener run 2026-10-01 - 502 scored, top: EXPE HST CAH MPC TRV`

## The session's own account

> 2026-10-01 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-30_060001.log` ends "Run complete: shipped to main",
> tagged `good/2026-09-30` | data loop published? **yes** -
> `logs/datarun-2026-10-01_020001.log` ends "Data loop complete", HEALTH: PASS,
> 502 scored, top EXPE HST CAH MPC TRV | evidence base at `1m` = **20 rows, newest
> 2026-09-01 (30 days ago, bound 40), 4 effective** - the middle of the 30-33-day
> steady state, healthy | priority 0 **fixed 2026-08-24, not weakened**
> (`_effective_observations()` still gates, `allow_auto_apply` still false, 4
> effective against a gate of 8) | top open roadmap item: **priority 3, backtest
> v2 - 37 days old**, and **today closed step 1**, the measurement the last two
> sessions both nominated in writing.
> 
> **Tests:** before **1615/1615**, after **1650/1650** (+35)
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so nothing to
> claim or move to Done. Took the nominal **Thursday build** focus by the route the
> prompt specifies for it: the week produced no methodology change to implement
> (Monday 09-28 was consumed by a red ship gate and produced no note; Wednesday
> 09-30 swapped to priority 3 and said so), and in that case the instruction is to
> take the top open item in "Current priorities" rather than invent a change. That
> item nominated its own next step. Nothing was deferred for a stalled loop or a
> failing gate - both loops are healthy and the baseline was green at 1615/1615.
> 
> ### Did
> 
> **Sized the look-ahead bias in `backtest.py`. It is >= 63.2% of the name-month
> panel against survivorship's 11.4% - 5.5x bigger on the same unit - and that
> closes `plan/backtest-v2.md` step 1, both halves.**
> `research/2026-10-01-lookahead-bias-size.md`; `lookahead.py`;
> `research/measurements/2026-10-01-lookahead-price-component.py` (committed JSON
> output beside it); `tests/test_lookahead.py`, **43 tests**.
> 
> **1. Why this and not something else.** The 2026-09-30 session priced the
> delisted-price feed, decided not to buy, and nominated this as the next step in
> writing: *"free, needs no vendor and no permission ... the missing half of step 1
> ... nothing else on the plan should be built first."* The 09-28 session nominated
> the same item. Owner queue empty, both loops healthy, baseline green - so there
> was no competing claim, and priority 3 had been deferred by eight of the ten
> sessions before 09-30 precisely because something always looked more urgent.
> 
> **2. The headline, on the same unit as survivorship.** Two arms of the real
> scoring chain, 81 rebalance months x 502 names = **40,662 name-months**. Arm A is
> v1: static metrics at snapshot values, momentum/risk recomputed per month exactly
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

