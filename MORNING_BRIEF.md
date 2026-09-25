# Morning Brief - Friday 25 September 2026, 02:15

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-25T02:00:03.515421 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, VLO, MPC, EIX |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (16 rows, but overlapping windows are not independent; 55 rows across all horizons), newest 2026-09-18 |

## What changed in the repo

- `aad81aa data: screener run 2026-09-25 - 502 scored, top: EXPE HST VLO MPC EIX`
- `d2e231c brief: code session 2026-09-24`
- `8c13217 log: close the stale worktree left by 2026-09-23`
- `7a08281 log: nightly 2026-09-24 - backtest v2 step 1, survivorship sized`
- `f099c4a docs: record the survivorship result where it is acted on`
- `b822a01 measure: survivorship bias in backtest.py is 4.3%/yr`
- `6206bdc backtest-v2: point-in-time S&P 500 membership`
- `b05b9fd brief: data run 2026-09-24`
- `4ca65a5 data: screener run 2026-09-24 - 502 scored, top: EXPE HST VLO BBY BMY`

## The session's own account

> 2026-09-24 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-23_060001.log` ends "Run complete: shipped to main" |
> data loop published? **yes** - `logs/datarun-2026-09-24_020001.log` ends
> "Data loop complete" | evidence base at `1m` = **15 rows, newest 2026-08-25
> (30 days ago, bound 40), 3 effective** - steady-state lag, healthy |
> priority 0 **fixed** (2026-08-24, untouched) | top open roadmap item:
> **priority 3, backtest v2, 30 days old - taken today**
> **Tests:** before **1439/1439**, after **1498/1498** (+59 new, no pre-existing
> failures)
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so the rotation
> governed. The week's research (position sizing) shipped Wednesday, so per the
> Thursday rule I took the **top open item in Current priorities** rather than
> inventing a methodology change: priority 3, deferred by six consecutive
> sessions. Nothing deferred.
> 
> ### Did
> 
> **Sized the survivorship bias in `backtest.py` - step 1 of
> `plan/backtest-v2.md` - and the answer settles that plan's own decision rule
> against v1.**
> 
> The plan set the rule in advance: *"If it's 0.5% a year, v1 is usable with a
> caveat. If it's 4%, every existing validation claim needs retracting."*
> **Measured: 4.3% a year.**
> 
> **1. `universe_history.py` - point-in-time S&P 500 membership.** Reconstructs
> who was in the index on a given date from the Wikipedia revision current on
> that date, via the MediaWiki revisions API. This is a genuine contemporaneous
> record rather than a backward projection. An 80-month cache
> (2020-01..2026-08, 500 KB) is committed at
> `data/universe_history/sp500_membership.json` with per-snapshot provenance -
> revision id, revision timestamp, staleness - so the reconstruction is
> reproducible offline and every number can be checked against the exact
> revision it came from.
> 
> **2. The measurement.**
> `research/measurements/2026-09-24-survivorship-gap.py` reproduces all of it:
> 
> | | |
> |---|---|
> | 2020-01-31 constituents absent from today's list | **116 of 505 (23.0%)** |
> | ...of which are ticker renames, not exits | 18 |
> | **Survivorship bias, net of renames** | **98 of 505 (19.4%)** |
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

