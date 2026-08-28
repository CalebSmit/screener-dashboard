# Morning Brief - Friday 28 August 2026, 06:18

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-28T02:00:03.803219 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, EIX, CF |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 27 rows across all horizons), newest 2026-08-21 |

## Things that needed attention

- Stale lock. Reclaiming.
- Stale lock (24h). Reclaiming.

## What changed in the repo

- `493ddeb data: republish the dashboard with weights that add up`
- `71da63c docs: the printed weights are defaults, and a run may not use them`
- `6344a72 teach: show the weight each score was actually multiplied by`
- `14027d9 fix: record the weights the composite was actually built from`
- `5c0966d brief: data run 2026-08-28`
- `7dfe548 data: screener run 2026-08-28 - 502 scored, top: HST EXPE APA EIX CF`
- `bc127cd brief: code session 2026-08-27`
- `3175da4 watch: verify the workflow on GitHub, and bump the actions off Node 20`
- `1af2643 docs: both priority -1 infrastructure items are done; correct the record`
- `0fda98d fix: the morning brief lost its Top 5 when the portfolio key was removed`
- `c05faf6 watch: report a loop that stops firing, from outside the machine`
- `c52abb3 brief: data run 2026-08-27`
- `8e021ab data: screener run 2026-08-27 - 502 scored, top: HST EXPE EIX APA CF`
- `42c278d harden: ToString() the commit-subject output before trimming it`
- `9273625 fix: the data-run commit subject named sector peers, not the top five`

## The session's own account

> 2026-08-28 - HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-08-27_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | `logs/datarun-2026-08-28_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
> | Evidence base | **27 rows, newest 2026-08-21, 3 effective observations at `1m`** (8 raw) |
> | Priority 0 | Fixed 2026-08-24, holding |
> 
> **Tests:** before 791/791, after **825/825**
> **Data loop:** healthy
> **Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`, so the
> rotation stood. Nothing was deferred.
> **Rotation:** ISO week 35 is odd, so this was a normal Friday, not a
> retrospective.
> 
> **Last session's item 3 is closed first, because it was cheap.** The watchdog's
> first *scheduled* firing (run
> [33148005073](https://github.com/CalebSmit/screener-dashboard/actions/runs/33148005073),
> `schedule` trigger, 10s, success) is green. The cron works, not just the manual
> dispatch.
> 
> ### The question this day asks, asked literally
> 
> "Would a finance student understand what they are looking at?" The most
> teachable surface in the tool is the drilldown's contribution panel, because it
> does not just show a score - it shows the working:
> 
> ```
> Momentum   13% weight
> Score: 65.3/100  [Average]  x 13% = 9.76 pts
> ```
> 
> So I checked the arithmetic against the payload that was live on `main` this
> morning. **65.3 x 13% is 8.49, not 9.76.** The one worked example on the site
> did not add up.
> 
> ### Did - the weights shown were not the weights used
> 
> Solving `contrib / score` over the 491 stocks with all eight categories
> populated recovers what the composite was really built from: **valuation 20.05,
> momentum 14.95**, the other six unchanged, summing to 100.000. Those are
> exactly a LOW VOL regime - `13 x 1.15 = 14.95`, the 1.95pp taken out of
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

