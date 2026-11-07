# Morning Brief - Saturday 07 November 2026, 13:51

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-06 18:34 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, MPC, DXCM |
| Ranking's lean on growth | +0.052 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (32 days ago, bound 40) |
| Context signals | 20 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 331/332 within 0.5%; operating cash flow 305/320 within 0.5%; net income 302/332 within 0.5%; market value 444/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); DXCM (#5: operating cash flow 36% off the filing); HST (#7: net income 1.46x after-tax operating income); PSX (#9: net income 1.28x after-tax operating income); CF (#12: net income 16% off the filing); TPR (#13: equity 5.2% of assets); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 674 s, `context.options_pass` 178 s, `fetch.retry` 116 s; **33 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (26 days) |
| Context coverage | options usable 415/499 - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 2 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 9 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `16a40f7 log: the per-band health gate, and the threshold that is a methodology decision`
- `0004e80 measure: six mutations against the per-band gate, 6 of 6 caught`
- `5191b33 gate: the analyst-target floor is applied inside each size band`
- `39d7ea0 brief: code session 2026-11-07-weekend`
- `120b5cc log: the second reading of experiment 19`
- `e86408d lab: experiment 19's second reading - it stands, with a stronger answer to its own open question`
- `d8157a8 log: the size_band slice of universe-expansion phase 1`
- `d703b37 rerun the research-engine ledger after the size_band change (32 rows, no number moved)`
- `94e7ed4 wip: rerun ledger rows 21-28`
- `93adbe2 wip: rerun ledger rows 11-20`
- `c7bf580 wip: rerun ledger rows 1-10 after the size_band change`
- `0905264 wip: rank percentiles within (Sector, size_band), one band today`
- `c50a7b1 log: the night's last fixes`
- `1697ab3 The track record lists its 37 changelog entries, readable where nothing hovers`
- `5339f2d The research page's share preview counts the alternatives the page counts`

## The session's own account

> 2026-11-07 (13:00 session) - QUEUE: the per-band health gate, and the one of its three thresholds that is a methodology decision rather than engineering
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-07_060001-weekend.log` (completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **26 days**); also open: 0.10 31d; 0.14 24d.
> **Tests:** before 4130 passed / 1 skipped (211 s, parallel); after 4194 passed / 1 skipped.
> **Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting.
> Second session today. Read the 06:00 entry and **carried on from its `Next`**, which named phase
> 1's remaining bullets "in this order: the per-band health gates ... then the `stock_detail`
> shard". Took the per-band gates. Nothing deferred for a gate or a stalled loop. The stash
> `auto-rescue 2026-10-28_060001` still holds an empty diff (`git stash show -p` prints nothing) -
> left alone, as this morning did. 0.13 (the Factor Lab, 26 days) deferred again: the owner queue
> outranks the roadmap, and this morning cleared the lab's one outstanding sub-item (experiment
> 19's second reading), so the lab has nothing waiting on a reader.
> **Did not reverse this morning's decision** to leave the `config.yaml` `universe:` block alone -
> it is still a knob with one legal value, and nothing today needed it.
> 
> ### Did
> 
> **`check_run_health.MIN_TARGET_COVERAGE` (0.50) now applies inside each size band as well as to
> the universe.** This is the first of phase 1's three remaining bullets
> (`plan/universe-expansion.md`: per band, "because a universe-wide average hides a thin bottom
> band"). `check_target_coverage_by_band` fails a run naming any band below the floor; the band is
> read from the run's **own** `01_raw_metrics.parquet` (`run_size_bands`), an artifact this gate
> already requires as fetch evidence, so the check needs no new input.
> 
> - **The band is not published.** Phase 1's exit test is a byte-identical S&P 500 payload, so the
>   band stays out of it and the gate reads a parquet - which is also the rule the rest of the
>   module has followed since 2026-10-16: judge the run's own artifacts, never the last published
>   ones.
> - **A run with no band column is a note, never a failure** - every run before this morning. Same
>   trade as the unreadable-weights case beside it and the node/pytest fallbacks in both publish
>   paths: a gate that can only fail closed jams an unattended loop.
> - **It is strictly stronger than the universe-wide check it sits beside, and that is what let it
>   ship with no shadow run behind it.** The universe rate is a weighted mean of the band rates, so
>   a universe-wide failure implies some band below the same floor. `TestDominance` parametrises
>   that over five splits x seven coverage rates rather than asserting it in a comment.
> 
> **`MIN_PRICE_COVERAGE` and `MIN_CATEGORY_COVERAGE` stay universe-wide, and the reason is not
> caution.** Analyst coverage is the one input to this screener whose *availability* is a function
> of company size. A price is served for every listed stock and a price history for every one with
> a year of trading, so both of those guard a **feed failure** - which appears across the whole
> universe or not at all. Banding them would add ways to fail and no ability to see anything. Two
> tests pin the scope, so it reads as a decision rather than an oversight.
> 
> **`CATEGORY_AUTO_DISABLE_PCT` (30) is measured per band and still triggered universe-wide. That
> is the part of the bullet I did not build, on purpose.** The plan lists it beside the health gate,
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

