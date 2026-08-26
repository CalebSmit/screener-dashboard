# Morning Brief - Wednesday 26 August 2026, 06:23

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-26T02:00:03.870018 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, EIX, CF |
| Evidence for weight changes | 2 of 8 needed at the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows across all horizons), newest 2026-08-14 |

## What changed in the repo

- `f055475 docs: record the fix, and correct the record it was diagnosed from`
- `3bf9798 guard: bound the blast radius of a withheld price series`
- `bdda9a7 fix: refuse a price series that mixes two split scales`
- `f997570 brief: data run 2026-08-26`
- `0a61fd4 data: screener run 2026-08-26 - 502 scored, top: HST EXPE APA EIX CF`
- `a32f390 brief: evening session 2026-08-25`
- `f2f5c74 chore: remove the last things that needed a human`
- `3083968 brief: code session 2026-08-25`
- `d33dd04 docs: record the history gate, its evidence, and a defect it uncovered`
- `7e7d23c test: syntax-check the emitted dashboard JavaScript`
- `99fa64d feat(dashboard): What Changed - movers, rank deltas and per-stock history`
- `db294e5 feat: history.py - a quality-gated historical spine for the dashboard`
- `0232d40 brief: data run 2026-08-25`
- `4a0a90f data: screener run 2026-08-25 - 501 scored, top: HST EXPE APA CF EIX`
- `b149c65 brief: evening session 2026-08-24`

## The session's own account

> 2026-08-26 - SYNTHESIS. How does this fit the rest of the screener?
> 
> **Health numbers:** last code session **ran and shipped** (06:25 on 08-25,
> `good/2026-08-25-0625`); data loop **published** 02:11 today, HEALTH: PASS,
> 502/502 price coverage; evidence base **23 rows, newest 2026-08-14, 2 effective
> observations at the `1m` horizon**; priority 0 **DONE**, priority 1.5 **closed
> today**.
> 
> **Tests:** before 647/647, after **676/676**
> **Data loop:** healthy - `logs/datarun-2026-08-26_020001.log` ends "Data loop
> complete", HEALTH: PASS, 0 fetch failures, 0 synthetic substitutions.
> 
> **On the evidence base not moving.** 23 rows / 2026-08-14 / 2 effective is
> identical to 08-25, which is two consecutive sessions. That is **expected
> latency, not a stall**: the newest snapshot old enough for a `1w` IC is
> 2026-08-20, which becomes eligible on 08-27. Snapshots exist for 08-20, 08-21,
> 08-24, 08-25 and 08-26 and are queued. If the row count has not moved by the
> 08-27 session, rule 8 bites and that becomes the work.
> 
> ### Did
> 
> **Root-caused and fixed priority 1.5 - and the 08-25 diagnosis of it was
> backwards.**
> 
> MNST's `return_12_1` percentile round-trip (97.1 -> 2.9 -> 97.1) was not a
> transiently-failing metric. Yahoo's 13-month series for MNST **alternates
> between pre- and post-split prices** across its 2026-08-11 2:1 split:
> 
> ```
> 2026-08-05    94.46      <- unadjusted
> 2026-08-06    47.08      <- adjusted
> 2026-08-07    90.36      <- unadjusted
> 2026-08-11    45.53      <- split date
> ```
> 
> `auto_adjust=False` returns byte-identical numbers, so no adjustment was ever
> applied. From today's live `runs/83c9e2e2dd48/00_raw_fetch.parquet` the pipeline
> divided an unadjusted July close (93.49) by an adjusted 2025 close (62.30):
> 
>     published   return_12_1 = +0.5006  -> 97th percentile
>     correct     return_12_1 = -0.2497  ->  3rd percentile
> 
> **So 97.1 was the artifact and 2.9 was right** - the reverse of what
> `NIGHTLY_LOG.md` 08-25, `history.py` and `CLAUDE.md` priority 1.5 all said.
> MNST was live on the public site at momentum 71.5, rank 360, roughly **110
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

