# Morning Brief - Saturday 07 November 2026, 07:16

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
- `9576d0d log: the options caveat and what was checked and left`
- `a345df8 A partial option reading says its figures are the source's last-trade ones`
- `3bc33aa A saved second share class opens its company's line from the holdings card`
- `6218160 log, inventory: the night's last batches`

## The session's own account

> 2026-11-07 (06:00 weekend session) - the universe becomes a parameter where it decides what a stock is compared with
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-06_130000-midday.log` (completed). Data loop published? yes - `logs/datarun-2026-11-06_183427.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (32 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **26 days**); also open: 0.10 31d; 0.14 24d.
> **Tests:** before 4117 passed / 1 skipped (195 s, parallel); after 4130 passed / 1 skipped.
> **Owner queue / rotation:** `write_brief.open_proposals()` is empty - no proposal waiting. Took
> the **top open owner item** (2026-10-28, beyond the S&P 500), whose named plan is
> `plan/universe-expansion.md` and whose Progress line said phase 1, `size_band` bullet. One stage,
> finished end to end. Nothing deferred for a gate or a stalled loop. The stash `auto-rescue
> 2026-10-28_060001` holds an empty diff (`git stash show -p` prints nothing) - left alone.
> 0.13 (the Factor Lab, 26 days) deferred: Friday is the lab's day and the owner queue outranks the
> roadmap; experiment 19 still wants its second reader.
> 
> ### Did
> 
> **Percentiles are ranked within `(Sector, size_band)`, with a band-wide fallback.** This is
> pre-registration 3 of the universe expansion, adopted 2026-11-03 and until today only written
> down. Once the S&P 400 and 600 are in the pool, "cheap for a utility" has to mean cheap against
> utilities *of about this size* - the size effect runs through most of these metrics - and a
> sector-band group below `SECTOR_MIN_PEERS` has to fall back to its **band**, because falling back
> across bands puts exactly the stocks with the thinnest peer group (small companies in a small
> sector) back in a distribution of mega caps.
> 
> - `universe.Universe` carries a `size_band`; `universe.assign_size_bands()` sets the column,
>   never overwriting one a caller set, and fills blanks because `groupby` drops a NaN key and a
>   stock dropped out of its own distribution is the failure this whole exercise is meant to catch.
> - `run_screener` calls it once the universe is settled, so `01_raw_metrics` onward carry it.
> - `factor_engine.compute_sector_percentiles` groups by `(Sector, size_band)`; the fallback rank
>   table is per band. `SECTOR_MIN_PEERS` stays 10 - only its scope changed.
> - `Composite_Pct` is ranked within band.
> - `universe.py` joined `v2_experiments.RESULT_PATHS`: it now decides what each stock is ranked
>   against, so an edit there can move every engine number, and the hash could not see it.
> 
> **Nothing published moved, and that is measured rather than argued** - `tests/test_size_bands.py`,
> 13 tests. The module keeps a transcription of the sector-only code it replaced and compares the
> two on the published run: **0 of 23,092 percentile values differ**. Its floor is a half-and-half
> cap split, which moves **20,140** - so the zero means something. (Phase 0's script reported 20,151
> on the 2026-11-03 run; that figure is a property of a morning's data, not a constant, which is
> worth knowing before someone treats it as one.) The composite percentile has the same pair:
> unchanged with one band, **492 of 499** moved by the split. The band-wide fallback has a fixture
> built so the two fallbacks *disagree* - a thin cell that is the bottom of its own band and the top
> of the universe - because with one band or one sector nothing can tell them apart, which is how a
> universe-wide fallback could have shipped under this name.
> 
> **Verified past the suite, because unit tests cannot see a shape assumption:** a five-ticker live
> `run_screener.py` (the band reaches `01`/`02`/`05`, the calc trace reproduces 40 category scores
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

