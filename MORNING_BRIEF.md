# Morning Brief - Friday 06 November 2026, 06:59

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-05 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, EXPE, MU, SNDK |
| Ranking's lean on growth | +0.060 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons), newest 1m run date 2026-10-05 (32 days ago, bound 40) |
| Context signals | 19 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 357/358 within 0.5%; operating cash flow 329/346 within 0.5%; net income 326/357 within 0.5%; market value 446/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); HST (#6: net income 1.46x after-tax operating income); DXCM (#7: operating cash flow 36% off the filing); PSX (#8: net income 1.28x after-tax operating income); TPR (#12: equity 5.2% of assets); CF (#15: net income 16% off the filing); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 628 s, `fetch.retry` 156 s, `context.sec_insider` 103 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (27 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 3 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 10 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.

## What changed in the repo

- `4473ac7 Both branches of the capped line are reached whichever payload is installed`
- `a9dd812 A capped metric row prints the figure its own arithmetic gives`
- `5acfa4c The holdings strip test reads each stand-in's rank, not its position`
- `f8af01f brief: data run 2026-11-06`
- `50b535f Read the flag-key scroll where it ends, not 700 ms into it`
- `9db2e4f Compare says what sets each stock apart, under its composite`
- `ffe51b5 Say in words what sets each stock apart: leaders, sheets and holdings`
- `7cf7b1f Pin the momentum rule's wiring to the one market fetch it now shares`
- `e6aab5f Rehearse the sell-off, and put the market beside the ranking in What Changed`
- `b9097bd brief: code session 2026-11-05-midday`
- `8f5f791 Log the 13:00 session: the red baseline, and the defect behind it`
- `18eeb13 The flake found: the sheet pre-warm was overwriting panels a test had just rendered`
- `2ca17f1 A red baseline keeps its transcript, like the gates do`
- `0081d38 Leader caption: the date never breaks across lines`
- `d5e85b4 Leaders show how long they have led; holdings show where each name has been`

## The session's own account

> 2026-11-06 - THE FACTOR LAB (deferred: the data loop was not publishing)
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-05_130000-midday.log`
> (completed). Data loop published? **no** - `logs/datarun-2026-11-06_020000.log` (stopped
> deliberately; HEALTH: PASS, the publish gate refused). Evidence base at `1m`: 5 of 8 needed at the
> 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons),
> newest 1m run date 2026-10-05 (**32 days ago, bound 40** - inside it). Priority 0:
> `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **25 days**);
> also open: 0.10 30d; 0.14 23d.
> **Tests:** before 4011 passed / 2 skipped (green baseline, nothing inherited), after 4020 / 2.
> **Owner queue / rotation:** no open proposal (`write_brief.open_proposals()` -> `[]`); no owner
> item taken. **Friday's lab focus deferred** - the data loop had stopped publishing, which outranks
> it (CLAUDE.md, "The two loops"). No held-back branch in the health line.
> 
> ### Did
> 
> **The data loop was discarding its run every morning's work, and the live site was a day stale.**
> The 02:00 run passed every health check (`HEALTH: PASS - safe to publish`) and was then refused by
> `scripts/check_published_claims.py` on two tests. Both are now fixed, and
> `python scripts/rehearse_publish.py` **PASSES** on the page the next run will publish - it was red
> on both counts before. Measured in `research/measurements/2026-11-06-publish-gate-failures.py`,
> which runs each failure against the discarded run's payload and the committed one side by side.
> 
> **1. The drilldown printed an equality its own figures refute** - the substantive one, and a live
> defect on the public site rather than only a test problem. `price_target_upside` is clamped to
> -50%..+100% by the scorer and its Calculation line is *exact*, so for a stock on a bound the row
> read `($2023.63 mean target - $806.43 price) / $806.43 price = 100.0%` where the arithmetic gives
> **150.9%**. That is precisely what priority 0.8 exists to stop (owner, 2026-10-06: every number on
> the page must be checkable from the page). **11 of 496 stocks** sat on a bound in the 11-06 run
> (CIEN, COHR, FLEX, LITE, MRVL, MU, NRG, ORCL, SNDK, STX, WDC) and **2 of 496** on the page that is
> live right now (NRG, FLEX).
> 
> The row now shows the cap as a step - `= 150.9% capped to 100.0%`, bounds in the title - so the
> line is true and the clamp is taught where it applies instead of only in the definition beneath.
> The uncapped figure is computed once in the build (`_capped_equations` -> `stock_detail[t].xcap`),
> only for the stocks whose bound binds, and the page displays it rather than re-deriving it (0.8).
> `metric_lineage.CLIPS` is pinned against `config.yaml`'s `metric_clamps` in **both directions**, so
> a third clamp cannot reach the scorer without the page learning about it; a clamped metric has
> exactly two honest presentations and no third (exact + published bounds, or prose that says the
> value is clipped - which `forward_eps_growth` already was).
> 
> **Looked at it** on the regenerated page (`logs/shot_capped_row.py`), desktop 1440 and 375px, SNDK
> capped and VLO uncapped. Desktop: one line, no overflow. 375px: wraps to two, `150.9%` in white and
> `100.0%` in blue, "capped to" in the muted label colour - so the emphasis stays on the scored
> value. The uncapped row is byte-unchanged (`= 21.6%`).
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

