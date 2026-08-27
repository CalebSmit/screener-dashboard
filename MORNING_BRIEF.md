# Morning Brief - Thursday 27 August 2026, 06:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-08-27T02:00:03.379939 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, EIX, APA, CF |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (7 rows, but overlapping windows are not independent; 25 rows across all horizons), newest 2026-08-20 |

## What changed in the repo

- `3175da4 watch: verify the workflow on GitHub, and bump the actions off Node 20`
- `1af2643 docs: both priority -1 infrastructure items are done; correct the record`
- `0fda98d fix: the morning brief lost its Top 5 when the portfolio key was removed`
- `c05faf6 watch: report a loop that stops firing, from outside the machine`
- `c52abb3 brief: data run 2026-08-27`
- `8e021ab data: screener run 2026-08-27 - 502 scored, top: HST EXPE EIX APA CF`
- `42c278d harden: ToString() the commit-subject output before trimming it`
- `9273625 fix: the data-run commit subject named sector peers, not the top five`
- `39dbfca brief: data run 2026-08-26`
- `d6074a9 data: screener run 2026-08-26 - 502 scored, top: MAA DOC KIM REG UDR`
- `5494086 brief: data run 2026-08-26`
- `4a622a8 product: Top 5 first, What Changed and Factor Analytics collapsed by default`
- `253be1d product: remove the model portfolio, give each stock an "about"`
- `9bed64f brief: code session 2026-08-26`
- `f055475 docs: record the fix, and correct the record it was diagnosed from`

## The session's own account

> 2026-08-27 - BUILD. The watchdog moves outside the thing it was watching.
> 
> ### Health numbers (rule 8)
> 
> | Check | Reading |
> |---|---|
> | Last code session ran? | `logs/nightly-2026-08-26_060001.log` - "Run complete: shipped to main" |
> | Data loop published? | `logs/datarun-2026-08-27_020001.log` - "Data loop complete", HEALTH: PASS, 0 fetch failures, 502/502 price coverage |
> | Evidence base | **25 rows, newest 2026-08-20, 3 effective observations at `1m`** (7 raw) |
> | Priority 0 | Fixed 2026-08-24, holding |
> 
> **Tests:** before 733/733, after **791/791**
> **Data loop:** healthy
> 
> **The 08-26 tripwire is released.** The previous two sessions read "23 rows,
> newest 2026-08-14, 2 effective at `1m`" and armed rule 8: if it had not moved
> today, making it move was today's work regardless of rotation. It moved -
> **23 -> 25 rows, 2 -> 3 effective at `1m`** - so the 08-24 fix is accruing
> evidence as predicted and the rotation stood. Independent 1-month observations
> still arrive about one a month; 3 of 8 is roughly five more months.
> 
> ### First, the five-minute check the last session asked for
> 
> **The MNST price-series fix landed on the live site.** Confirmed in today's
> 02:00 run rather than assumed:
> 
> - `validation/data_quality_log.csv` carries the first-ever
>   `price_series_rejected` row - MNST, "series mixes pre- and post-split prices
>   across a 2:1 split - 4 day(s) move by ~0.5x".
> - The published payload has `momentum_score: null` and `risk_score: null` for
>   MNST, and `return_12_1`/`return_6m`/`volatility`/`beta` all null in `raw`.
> - MNST moved rank **360 -> 370**, in the direction the correction implies.
> - `check_run_health` still PASS: one withheld name in 502, which is
>   `MIN_CATEGORY_COVERAGE = 0.90` doing its job rather than tripping.
> 
> ### Did - 1. A watchdog that is not inside the thing it watches
> 
> `CLAUDE.md` priority -1 listed two open infrastructure items. **One of them was
> already done and the file did not know.** `scripts/register-tasks.ps1` shipped
> 2026-08-21 (commit `33f3ca7`), yet the priority section, `ACTION_REQUIRED.md`
> and a research note all still asserted that `grep -rn "Register-ScheduledTask"
> .` "finds nothing". It finds it at `scripts/register-tasks.ps1:111`. Corrected
> in all three places - rule 9.
> 
> The second item was real, and is the session's work: **nothing watched whether
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

