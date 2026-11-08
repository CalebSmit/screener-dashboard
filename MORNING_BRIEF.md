# Morning Brief - Sunday 08 November 2026, 14:33

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

- `78900f5 log: five full parallel runs on identical code, three one-off browser flakes`
- `77cc502 log: the shard's second slice, and the half of it that needed deleting`
- `13c18f8 fix: two modules that read `financials` before the loader merged it`
- `c28af48 shard: the detail file learns to carry numbers, and `financials` moves into it`
- `62bc52b measure: 7 of 7 mutations refused, and the one that got through first was right to`
- `e299b11 unify: the peer table's rows all go through one function`
- `fc1865f brief: code session 2026-11-08-weekend`
- `a1bc585 log: the detail split, and two measurements that corrected the reasoning behind it`
- `ae07726 gate: a runner must check each payload by its assignment, not only name the file`
- `2dc0b70 measure: the pre-warm does fetch the detail file, and the inventory's 60 KB was wrong by 5x`
- `c451ebf split: the sheet-only half of stock_detail leaves the first-paint payload`
- `263cd05 brief: code session 2026-11-07-midday`
- `16a40f7 log: the per-band health gate, and the threshold that is a methodology decision`
- `0004e80 measure: six mutations against the per-band gate, 6 of 6 caught`
- `5191b33 gate: the analyst-target floor is applied inside each size band`

## The session's own account

> 2026-11-08 (13:00 session) - QUEUE: the shard's second slice, and the half of it that needed deleting rather than moving
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-08_060001-weekend.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **45 rows, newest 2026-10-06 (33 days ago, bound 40), 5 effective** of the 8
> the engine's gate needs. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13,
> The Factor Lab** (age **27 days**); also open: 0.10 32d, 0.14 25d.
> **Tests:** before 4218 passed / 1 skipped (209 s, parallel); after **4232 passed / 1 skipped**
> (206 s) - the 14 net new ones are this work's. Gate 2 clean, gate 5 **PASSED** with all five
> published files byte-identical to what the next data run would write.
> **Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting. Nothing
> held back by the gates (`origin/nightly/2026-11-08` was level with `main`). Carried on from the
> 06:00 session's **Next**, which named the shard's next field group, under the same top open
> `OWNER_FOCUS.md` item (universe expansion, `plan/universe-expansion.md`). Nothing deferred for a
> gate or a stalled loop. 0.13 (the lab, 27 days) deferred again - the owner queue outranks the
> roadmap and the lab has no experiment waiting on a reader. The stash `auto-rescue 2026-10-28_060001`
> still shows an empty diff; left alone, as the last four sessions did.
> 
> ### Did
> 
> **First paint fell 1.240 -> 1.161 MB gzipped (-6.4%) in two steps, and 1.547 -> 1.161 MB (-25.0%)
> over the day with the morning's split.** Nothing published moved: no number, score, rank or
> sentence. The two steps are different in kind and the first is the one worth reading.
> 
> **1. The peer table's rows all go through one function; `self_metrics` is deleted, not sharded.**
> The morning's `Next` named `financials` + `self_metrics` as the next slice. `self_metrics` did not
> need moving, because it held nothing of its own. The nine-column peer table built the open stock's
> row from that precomputed block and **every peer's row from `buildPeerRow` in JS** over `raw` +
> `financials` - and every stock sits on both sides of that split, being the subject of its own sheet
> and a peer in its sector-mates'. So one company's figures could print two ways on two sheets.
> 
> - **Measured before touching it** (`research/measurements/2026-11-08-peer-row-two-paths.py`,
>   committed): **0 of 4,491 cells disagreed** on the 2026-11-06 payload - all nine columns, all 499
>   stocks - so no published figure moved. The reason to remove it anyway is CLAUDE.md row 0.8 (the
>   page shows the engine's numbers, never a second derivation) and the fact that **one path cannot
>   disagree with itself**. The two expressions really were different code: `round(1.0 /
>   earnings_yield, 1)` on the unrounded yield in Python, round-half-to-even, against
>   `Math.round(10.0 / ey) / 10` on the published 4dp yield in JS, half away from zero.
> - **-17 KB of first paint for figures the payload already carried**, with no fetch, no waiting
>   state and nothing new served on demand - the cheapest slice in phase 1's queue, and the only one
>   that makes the page *simpler* rather than more conditional.
> - The panel's empty state now asks the question its sentence makes - whether any of the nine
>   columns has a figure, not whether one block exists. Unreachable on a good run either way, which
>   the new module asserts as its own premise (0 of 499; the thinnest stock carries six columns).
> - `tests/test_peer_row_one_path.py`, 8 tests. The one that would have caught the duplication reads
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

