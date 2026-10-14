# Morning Brief - Wednesday 14 October 2026, 06:40

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-14T02:00:10.255065 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, INCY, HST, APA, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (29 rows, but overlapping windows are not independent; 81 rows across all horizons), newest 2026-10-07 |
| Context signals | Context signals: 5 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 486/501 (15 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `f8480a6 log: the four gates for 2026-10-14`
- `2f42754 log: the 2026-10-14 session, and the two constraints it added`
- `17249d4 publish: regenerate today's page so it stops saying two false things`
- `1daf857 changelog: the dividend-yield defect, its fix and the second derivation it exposed`
- `3e27f14 health: refuse a run where a weighted metric vanished across the universe`
- `53d9c5f churn: a metric the whole universe lost is a source event, not one stock's`
- `fa77fe7 fix: the source dropped dividendRate - restore the yield from the trailing rate`
- `34859a8 brief: data run 2026-10-14`
- `ccbc795 data: screener run 2026-10-14 - 501 scored, top: EXPE INCY HST APA SNDK`
- `d7c1b71 brief: code session 2026-10-13`
- `1a027ac growth: record the refusal in the docs, and generalise the open-proposal test`
- `99ec010 growth: record the decision on the page, the lab scorecard and the plan`
- `a3d2726 growth: review the waiting proposal and refuse it, with the measurement that decided it`
- `eeff9e2 brief: data run 2026-10-13`
- `a5e84b8 data: screener run 2026-10-13 - 501 scored, top: EXPE HST INCY APA BMY`

## The session's own account

> 2026-10-14 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-13_060001.log`
> ends "Run complete: shipped to main". Data loop published? **yes** -
> `logs/datarun-2026-10-14_020002.log` ends "Data loop complete", HEALTH: PASS, claims gate 454
> passed. Evidence base at `1m`: **29 rows, newest 2026-09-14 (30 days ago, bound 40), 4 effective**
> - inside the bound, and the engine's own line in the data log agrees ("4 effective (29 rows) at
> the 1m horizon"). Priority 0: unchanged, `allow_auto_apply` false, 4 of 8 effective observations.
> Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 2 days**).
> **Tests:** before **2,234 passed / 1 failed**, after **2,274 passed / 0 failed**.
> **Owner queue / rotation:** **the baseline was red**, which outranks both the owner queue and the
> day's focus - and the failure turned out to be a real defect that had published four hours
> earlier. Both owner items (the Factor Lab, the context layer) are **deferred for that**, named
> here as rule 8 requires. The lab's next day is Friday 2026-10-23 anyway (2026-10-16 is a
> retrospective); the context layer's queue item 4 is unworked and still next.
> 
> ### Did
> 
> **The gate-1 baseline failed, and it was not flaky.**
> `tests/test_input_churn.py::test_build_history_on_real_snapshots_reports_some_churn` -
> "churn on most of the universe", 852 of 1002 comparison entries - green yesterday. It is a
> tripwire on the real snapshot directory, so the question was which metric, not how to quiet it.
> `research/measurements/2026-10-14-input-churn-which-metric.py`: **`dividend_yield_pct` went from
> 99 NaN to 495 NaN of 501 in the run published at 02:00 this morning.**
> 
> **1. The source dropped a field and the metric had no fallback.**
> Measured in the scored caches: `dividend_yield` **402 of 501 on 2026-10-13, 6 of 501 on
> 2026-10-14**. A live probe (`research/measurements/2026-10-14-dividend-field-probe.py`) found
> `.info["dividendRate"]` served for **0 of 12** tickers while `dividendYield`,
> `trailingAnnualDividendRate`, `trailingAnnualDividendYield`, `fiveYearAvgDividendYield`,
> `lastDividendValue` and `payoutRatio` were all served **12 of 12**. The metric read that one
> field and divided by the price; nothing else.
> 
> Fixed in `factor_engine.dividend_yield_from_fields`: the declared forward rate, then the trailing
> twelve months paid, over the price, bounded at 25%, publishing which basis it used.
> **`.info["dividendYield"]` is deliberately not read** - the probe measured it at **2.35 for a
> stock yielding 2.31%**, already a percent, where older yfinance releases served the same field as
> a fraction. A field whose units changed between releases is a silent 100x error waiting for the
> release that changes them back; a dollar amount over our own price cannot be misread.
> `METHODOLOGY_CHANGELOG.md` 2026-10-14, `tests/test_dividend_yield_source.py` (22).
> **No score moved: `dividend_yield` carries weight 0 in both profiles.**
> 
> **2. Fixing the engine would not have fixed the page, which is the row-0.8 failure again.**
> `generate_dashboard` derived the same quantity a **second** time from its own copy of
> `dividendRate` - for the Company Snapshot row *and* for Sector Peers. Measured on the published
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

