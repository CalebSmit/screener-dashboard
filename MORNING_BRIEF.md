# Morning Brief - Friday 06 November 2026, 19:00

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
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (31 days ago, bound 40) |
| Context signals | 20 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 331/332 within 0.5%; operating cash flow 305/320 within 0.5%; net income 302/332 within 0.5%; market value 444/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); DXCM (#5: operating cash flow 36% off the filing); HST (#7: net income 1.46x after-tax operating income); PSX (#9: net income 1.28x after-tax operating income); CF (#12: net income 16% off the filing); TPR (#13: equity 5.2% of assets); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 674 s, `context.options_pass` 178 s, `fetch.retry` 116 s; **33 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (26 days) |
| Context coverage | options usable 415/499 - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 3 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 10 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `624e538 data: screener run 2026-11-06 - 499 scored, top: VLO EXPE MU MPC DXCM`
- `12d0e3d The data loop never reuses the day's scored cache, so a same-day rerun can publish`
- `dd9d628 brief: data run 2026-11-06`
- `f450757 brief: code session 2026-11-06-midday`
- `840e8a5 log: the suite count after is 4046, the same as the baseline`
- `2e918cb nightly 2026-11-06 (13:00) - the Factor Lab, experiment 19`
- `04237b4 lab: experiment 19 written up, and the overview's phone budget measured where it settles`
- `3f3cb98 lab: experiment 19, Mohanram's growth score on OSAP's six MS groups - 4 of 7 within`
- `4a55141 lab: experiment 19 tolerances, written before the comparison (Mohanram 2005, on OSAP's MS groups)`
- `e18feaf Say how old the prices are, and make two tiles tell their story`
- `17b85a1 brief: code session 2026-11-06`
- `4473ac7 Both branches of the capped line are reached whichever payload is installed`
- `a9dd812 A capped metric row prints the figure its own arithmetic gives`
- `5acfa4c The holdings strip test reads each stand-in's rank, not its position`
- `f8af01f brief: data run 2026-11-06`

## The session's own account

> 2026-11-06 (owner-run, evening) - the catch-up run could not publish, by construction. Fixed.
> 
> **Health (rule 8, all five):** last code session ran? yes - the 13:00 session shipped to main.
> Data loop published? **no, twice** - 02:00 stopped by the publish gate (fixed by the 06:00
> session), and the 16:00 catch-up run discarded as below; the live page is still the 2026-11-05
> 02:00 run. Evidence base at `1m`: **44 rows, newest 2026-10-05 (32 days ago, bound 40), 5
> effective of the 8 needed** (unchanged since 13:00: neither run today recorded a snapshot that
> survived). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor
> Lab** (age **25 days**).
> **Tests:** before 4046 passed / 2 skipped (13:00's count); after **4057 passed / 2 skipped** (+11:
> `tests/test_same_day_rerun.py`, and one fixture line in `tests/test_rehearse_publish.py`).
> 
> ### What happened at 16:00
> 
> The owner approved a one-time after-close data run so the site would not sit on Wednesday's
> close all weekend. It ran at 16:00:01 and took 90 seconds, because it never fetched: it loaded the
> 02:00 run's scored cache (`[CACHE HIT] Loading scores from factor_scores_3ed2aee7f770_20261106`).
> Dashboard generation then refused it - `CalculationMismatch: 286 of 499 stocks do not reproduce
> their published scores` (ALL valuation 63.3 published, 93.3 recomputed, "profile generic") - and
> the health check discarded the run for having no payload of its own. **The gate was right.**
> 
> **Cause.** `factor_engine.write_scores_parquet` drops every column whose name begins with `_`
> (since Phase 0, 2026-02-17; the overview even says so). Among them is `_is_bank_like`, which the
> generator's weight resolution reads, so every insurer was checked against the generic weights.
> The scores themselves were byte-identical to 02:00's. And because the run fetched nothing it had
> no `00_raw_fetch.parquet` either - which `check_run_health` requires - and its prices were the
> 02:00 fetch's, the previous session's close. **So a second full run on the same day could never
> publish, and could never bring after-close prices even if it had.** Proposing the catch-up run
> without checking this was this session's mistake.
> 
> **A second-order fault, the one that mattered for the weekend.** The discarded run was now the
> newest folder in `runs/`, and `scripts/rehearse_publish.py` (gate 5) rehearses from the newest
> full run - so every rehearsal failed with the same `CalculationMismatch`, whatever the branch.
> Every weekend session's work would have been held off `main`. Confirmed by running the rehearsal
> (failed), moving the folder to `logs/discarded-run-bce9b3ce6629` (not deleted), and running it
> again (passed - which also confirms the 06:00 fixes clear the 02:00 run's data).
> 
> ### Did
> 
> - **`run_screener.py --fresh`**: fetch and score afresh even when today's scored cache exists,
>   keeping every cache file (`--refresh` would delete all `factor_scores_*.parquet`, which
>   `backtest.py` reads as its rolling history). The decision is one function,
>   `reuse_score_cache()`.
> - **`scripts/data-run.ps1` always passes `--fresh`.** The publishing run never reuses the cache;
>   hand runs and trials still may.
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

