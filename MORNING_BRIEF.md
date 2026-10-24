# Morning Brief - Saturday 24 October 2026, 06:42

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-23T02:00:04.666022 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, APA, HST, INCY, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (35 rows, but overlapping windows are not independent; 93 rows across all horizons), newest 2026-10-15 |
| Context signals | Context signals: 12 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 423/424 within 0.5%; operating cash flow 406/410 within 0.5%; net income 396/419 within 0.5%; market value 444/484 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); MU (#10: a period behind its filings); NEM (#12: a period behind its filings); CF (#16: net income 16% off the filing) |
| Context coverage | options usable 486/498 (12 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `93f7d37 comment: the side file is ~10 KB, measured, not 2`
- `b8c592b log: correct the after-count to the measured 3109`
- `19d8435 log: the 2026-10-24 weekend session - owner item 2026-10-21 items 5 and 6, and the item closed`
- `ddf7abc the published run's inputs outlive the four-week prune`
- `5b83030 only the gated run is tradable: signals.json`
- `fd88886 wip: the page says which run, which commit and which raw fetch it came from`
- `86c90cb brief: code session 2026-10-23-midday`
- `b6352de log: the 2026-10-23 13:00 session - experiments 16 and 17 second-read, the rerun deferred with its argument`
- `257faa3 measurements: extend the Piotroski second reading to the note's other-period claims`
- `749064e record: experiments 16 and 17 second-read; close item 5; correct 7c's experiment number`
- `ce959a6 research: second reading of lab experiment 17 - four paper figures exact, every measured figure reproduced`
- `8b83ff5 research: second reading of lab experiment 16 - all 17 paper figures and every measured figure confirmed`
- `76dbb7b measurements: second reading of lab experiment 16 (Asness et al. 2018 Table 3A), recomputed independently`
- `99806bb measurements: second reading of lab experiment 17 (Piotroski 2000 on OSAP), recomputed independently`
- `88f5430 brief: code session 2026-10-23`

## The session's own account

> 2026-10-24 (06:00 weekend session) - the screener's half of the plan to real money is finished: only the gated run is tradable, and the run a page came from can now be checked against its own inputs
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-23 13:00 ended "Run
> complete: shipped to main" at 13:40, tag `good/2026-10-23-1340`. Data loop published? **yes** -
> 2026-10-23 02:28, HEALTH PASS, 498 scored, all four of the 2026-10-22 input gates green (Spearman
> 0.994, 1 new in the top 25, 498/498 prices on the expected session). No 02:00 run today, correctly
> - Saturday. Evidence base at `1m` = **35 rows, newest 2026-09-22 (32 days ago, bound 40), 4
> effective** (`_n_observations` 4; the raw count of 35 is the one not to quote). Priority 0:
> `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
> Factor Lab** (opened 2026-10-12, **age 12 days**) - deferred, see below.
> **Tests:** before **3064 passed, 1 skipped**; after **3109 passed, 1 skipped** (+45: three new
> modules - 22, 14 and 8 - plus one new contract test). Gates 2, 3, 4 pass. **`scripts/rehearse_publish.py` run and
> PASSED** - the diff touches `generate_dashboard.py`, which the four gates cannot see.
> **Owner queue / rotation:** `OWNER_FOCUS.md`'s **top open item, 2026-10-21, items 5 and 6** - both
> built, and the item is now **closed and moved to Done**. **Deferred: the research-engine rerun**
> batched with the 2026-10-20 item 6, which the 10-23 13:00 session named as its Next; the argument
> is under *Tried and rejected*.
> 
> ### Did
> 
> **`OWNER_FOCUS.md` 2026-10-21 item 5 - only the gated run is tradable.** The hole this closes is
> an asymmetry, the same shape as the two previous retrospectives found. `dashboard_data.js` is
> published by **two** paths: the 02:00 data loop, behind seven gates (no synthetic data, a
> fetch-failure ceiling, `check_run_health.py` with its rank-continuity, empty-record and
> price-freshness rules, a size floor, a real `node --check`, the opening assignment, and
> `check_published_claims.py`); and any code session that runs the full screener, behind **none of
> them** - its four ship gates are about the repository, not about the run. The bot reads the page as
> Pages serves it and cannot tell the two apart.
> 
> So the gated run now writes a second, small file and the bot trades that. `scripts/write_signals.py`
> is called from `data-run.ps1` after the published-claims gate and before the commit, and
> `signals.json` is in `$DataArtifacts`. `scripts/nightly-screener.ps1` never writes or stages it -
> **that asymmetry is the whole mechanism**, so one test asserts the code runner does not mention it
> and another that no other module in the repo writes it.
> 
> Three decisions in it worth recording, because each could reasonably have gone the other way:
> 
> - **It is derived from the gated payload, not from a second read of the run directory.** The
>   ranking in `signals.json` is then by construction the ranking those gates inspected. A second
>   read could disagree with the page, and the disagreement would be invisible.
> - **Gate verdicts are `pass` or `warn`, and there is no `fail`.** Two gates publish on a documented
>   fallback when the tool they need is missing (node, pytest), and a consumer is entitled to know a
>   check did not run - `all_gates_ran` is false then. A `fail` verdict could not be true in a file
>   only reached after every gate accepted, so the parser refuses one.
> - **It is not a gate.** A failed write leaves the previous `signals.json` in place, which is the
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

