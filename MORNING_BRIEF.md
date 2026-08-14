# Morning Brief - Friday 14 August 2026, 02:11

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** |
| Code session (6 AM) | **completed** |
| Dashboard data from | 2026-08-14T02:00:04.105056 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, EIX, APA, CF |
| Evidence for weight changes | 3 of 8 needed |

## What changed in the repo

- `3fba581 data: screener run 2026-08-14 - 502 scored, top: HST EXPE EIX APA CF`
- `779272a brief: code session 2026-08-13`
- `d23e990 docs: record the stale-price cache fix; trust blocker resolved`
- `3a3090e fix: the data loop only fetched once every eight days`
- `e9b6271 brief: data run 2026-08-13`

## The session's own account

> 2026-08-13 - BUILD. The data loop was only fetching once every eight days.
> 
> **Tests:** before 530/530, after 552/552 (22 new)
> **Data loop:** was silently degraded - **fixed**. Today's 02:00 run warm-started
> from an 08-12 cache, produced no fetch artifacts, and was correctly discarded by
> the health gate. Root cause found and fixed; confirm on the 08-14 02:00 run.
> 
> ### The trust blocker is gone
> 
> The 06:00 runner logged **"Workspace trust OK"**, and `python -c`,
> `python -m pytest` and `python run_screener.py --dry-run` all executed. This is
> the **first autonomous session that could run the ship gates**, and the first to
> merge to `main`. Priority -1 in `CLAUDE.md` is marked resolved. Every session
> from 08-05 to 08-12 was blocked on this.
> 
> ### Did
> 
> Found and fixed why the data loop keeps publishing nothing. This was not a
> scheduling problem - it is that **the screener almost never fetched**.
> 
> `run_factor_engine` bounded reuse of the `factor_scores` cache by
> `caching.fundamental_data_refresh_days` (**7**), not
> `price_data_refresh_days` (**1**). Two things compound:
> 
> 1. `factor_scores` is the *fully scored* dataset, not fundamentals.
>    **18 of the 44 metrics in `METRIC_COLS` move with the daily close**, across
>    five of the eight categories - every valuation ratio (price is in all of
>    them), all three momentum metrics, six risk metrics, `price_target_upside`
>    and `size_log_mcap`. So a "fundamental" freshness bound was the wrong unit
>    entirely, and published Valuation/Momentum/Risk scores could be computed
>    from a close up to eight days old while presented as current.
> 2. The warm-start path returns at `run_screener.py:1011`, **before**
>    `write_scores_parquet` at `:1502`. A warm-started run lays down no new cache
>    file, so the cache date never advances. One real fetch therefore suppressed
>    the next seven days of runs: **one real fetch per eight daily runs.**
> 
> Also fixed an off-by-one that would have defeated the fix on its own. Cache
> dates are parsed from the filename and are midnight-anchored, so a cache from
> yesterday is `age_days == 1` no matter the clock time. Under the old
> `age_days <= fresh_days`, even `fresh_days = 1` would still have reused
> yesterday's cache at 02:00 - the daily loop would have kept warm-starting.
> The rule is now strict and stated in plain English:
> **`<tier>_refresh_days: N` means the cache is reusable for N calendar days
> starting with the day it was written.** `1` therefore means "refetch unless the
> cache is from today". A same-day manual re-run still warm-starts, which is
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

