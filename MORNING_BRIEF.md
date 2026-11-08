# Morning Brief - Sunday 08 November 2026, 06:51

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-06 18:34 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, MPC, DXCM |
| Ranking's lean on growth | +0.052 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (33 days ago, bound 40) |
| Context signals | 20 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 331/332 within 0.5%; operating cash flow 305/320 within 0.5%; net income 302/332 within 0.5%; market value 444/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); DXCM (#5: operating cash flow 36% off the filing); HST (#7: net income 1.46x after-tax operating income); PSX (#9: net income 1.28x after-tax operating income); CF (#12: net income 16% off the filing); TPR (#13: equity 5.2% of assets); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 674 s, `context.options_pass` 178 s, `fetch.retry` 116 s; **33 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (26 days) |
| Context coverage | options usable 415/499 - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 1 day) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 8 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `a1bc585 log: the detail split, and two measurements that corrected the reasoning behind it`
- `ae07726 gate: a runner must check each payload by its assignment, not only name the file`
- `2dc0b70 measure: the pre-warm does fetch the detail file, and the inventory's 60 KB was wrong by 5x`
- `c451ebf split: the sheet-only half of stock_detail leaves the first-paint payload`
- `263cd05 brief: code session 2026-11-07-midday`
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

## The session's own account

> 2026-11-08 (06:00 weekend session) - the sheet-only half of `stock_detail` leaves first paint: -19.8% on the wire
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-07_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **45 rows, newest 2026-10-06 (33 days ago, bound 40), 5 effective** of the 8
> the engine's gate needs. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13,
> The Factor Lab** (age **27 days**); also open: 0.10 32d, 0.14 25d.
> **Tests:** before 4194 passed / 1 skipped (213 s, parallel); after **4218 passed / 1 skipped**
> (211 s) - the 24 new ones are this work's.
> **Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting. Took the
> top open `OWNER_FOCUS.md` item (universe expansion, `plan/universe-expansion.md`) and the stage its
> Progress line named next, which the 13:00 session's `Next` also named: **the `stock_detail` shard**.
> Nothing deferred for a gate or a stalled loop. 0.13 (the lab, 27 days) deferred again - the owner
> queue outranks the roadmap and the lab has no experiment waiting on a reader. The stash
> `auto-rescue 2026-10-28_060001` still shows an empty diff; left alone, as the last three sessions did.
> 
> ### Did
> 
> **`about` and `industry` now ship in a third payload file, `dashboard_detail.js`, fetched when the
> first sheet is laid out instead of at first paint. Measured: the gzipped first-paint payload goes
> 1.547 -> 1.240 MB, -307 KB, -19.8%.** Nothing published moved - no number, score, rank or sentence.
> `generate_dashboard.split_detail()`, `DETAIL_KEYS = ("about", "industry")`;
> `tests/test_detail_payload_split.py`, 21 tests.
> 
> - **The shape of the shard was decided by measurement, and it is not the obvious one.** The plan's
>   bullet says "sharded out of `dashboard_data.js` and loaded per stock". Per *stock* cannot be done
>   without rewriting the page: **~85 call sites read `D.stock_detail[t]` synchronously for an
>   arbitrary ticker** - search, peers, holdings, movers, compare, the hash router - and each would
>   have to await a fetch. Enumerating those sites by enclosing function gives a rule that works
>   today: **one read site, inside the sheet, and no number.** `about`/`industry` are read only by
>   `renderAbout`. `financials` is also read by `buildPeerRow`, `eq4` by `renderReporting` (a
>   first-paint strip), and `raw`/`pct`/`inp`/`summary` by the compare drawer, the movers strip and
>   the calculation traces - 42.6% more of first paint, each needing its own measurement. The
>   mechanism is now there for them.
> - **Guarded in the commit that created it**, which is the 2026-10-16 lesson: both runners'
>   size/header/`node --check` specs, `$DataArtifacts`, `check_published_claims.PUBLISHED_ARTIFACTS`,
>   `rehearse_publish.PUBLISHED`, `rebuild_page.PAYLOADS` and `run_screener`'s publish copy. Two new
>   tripwires generalise it: the rehearsal's install list is **derived from `$DataArtifacts`** (a
>   payload file it does not install is one gate 5 rehearses *as committed*), and the page's
>   cache-busting token must be the hash of the committed detail file (else a returning reader is
>   served yesterday's descriptions forever).
> - **Gate 5 passed with all five published files byte-identical** to what the next data run would
>   write, which is also the proof that the committed artifacts are exactly what this tree's
>   generator makes.
> 
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

