# Morning Brief - Tuesday 22 September 2026, 02:13

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-22T02:00:03.386380 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, BBY, VLO, APA |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (13 rows, but overlapping windows are not independent; 49 rows across all horizons), newest 2026-09-15 |

## What changed in the repo

- `e6fa846 data: screener run 2026-09-22 - 502 scored, top: EXPE HST BBY VLO APA`
- `efccf2e brief: code session 2026-09-21`
- `b853d40 log: 2026-09-21 research session - position sizing note`
- `fe2d862 research: position sizing and the "how much" question`
- `d8efe45 brief: data run 2026-09-21`
- `120daba data: screener run 2026-09-21 - 502 scored, top: EXPE HST BBY VLO CAH`

## The session's own account

> 2026-09-21 - RESEARCH. One specific thing, learned properly from the literature AND documented practice, in one session. A dated note in research/, complete today. No production code.
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-09-18_060001.log`
> ends "Run complete: shipped to main", tagged `good/2026-09-18` | data loop published?
> **yes** - `logs/datarun-2026-09-21_020000.log` ends "Data loop complete" | evidence base
> at `1m` = **13 rows, newest 2026-08-21 (31 days ago, bound 40), 3 effective** - the
> horizon moved (it read 2026-08-14 for six sessions) and the 08-15..08-19 outage has
> cleared the pipe | priority 0 **fixed** (2026-08-24, untouched) | top open roadmap item:
> **priority 3, backtest v2, 27 days old**
> **Tests:** before 1378/1378, after 1378/1378
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so the rotation governed.
> Took Monday research. Nothing deferred.
> 
> ### Did
> 
> One research note, complete in the session, on **position sizing - the "how much"
> question**: `research/2026-09-21-position-sizing-and-how-much.md`, with its numbers as a
> re-runnable script at `research/measurements/2026-09-21-position-sizing-dispersion.py`.
> 
> Why this topic: of the four questions `plan/dashboard-north-star.md` says the dashboard
> exists to answer, **"how much / does it fit?" is the only one with no surface at all**,
> and has had none since the Model Portfolio was removed on 2026-08-26. The 09-18 session
> flagged exactly that to the owner. Research is the right first step because the obvious
> implementation - print a recommended weight - is the thing that got the Model Portfolio
> deleted.
> 
> **The note found a live defect in what this repo already does.** `config.yaml` has set
> `portfolio.weighting: 'score'` since launch - composite-score-proportional position
> sizing. Measured across **39 run dates (2026-02-20 .. 2026-09-21)**, one snapshot per
> date, two degraded 3-row February files excluded:
> 
> | | |
> |---|---|
> | Equal weight, 25 names | 4.00% |
> | Score weight, full span | **3.77% .. 4.56%** |
> | Max deviation from equal weight | **0.57 pp** (median 0.42) |
> | Active share vs equal weight, same names | median **1.31%** |
> | Heaviest/lightest ratio | **1.21x** |
> | Positions ever hitting the 5% cap | **0** |
> 
> So **`weighting: 'score'` is equal weight with noise**, and it cannot be anything else:
> composite scores are level-bounded 0-100 and the top 25 of 502 sit in a narrow band
> (today 64.75-73.48), so a 13% spread in level becomes a 13% spread in weight around 4%.
> **`max_position_pct: 5.0` is inert** for the same reason - it would need a composite 25%
> above the selected mean, which has never happened and cannot under this construction.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

