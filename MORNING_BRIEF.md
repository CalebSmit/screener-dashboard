# Morning Brief - Tuesday 29 September 2026, 06:26

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-09-29T02:00:04.389921 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | EXPE, HST, MPC, CAH, VLO |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (18 rows, but overlapping windows are not independent; 59 rows across all horizons), newest 2026-09-22 |

## What changed in the repo

- `8334521 log: changelog, nightly log, and both dashboard plans updated for gap 4`
- `5c49739 build: regenerate dashboard artifacts for the earnings surface`
- `733e168 feat: surface each stock's next earnings date (north-star gap 4)`
- `859ec29 brief: data run 2026-09-29`
- `826afad data: screener run 2026-09-29 - 502 scored, top: EXPE HST MPC CAH VLO`
- `e88c695 brief: code session 2026-09-28`
- `9a83f3b log: changelog, nightly log, and CLAUDE.md rule 10 now names the generated doc`
- `b4b24e1 docs: regenerate the methodology page and republish the corrected live site`
- `0d298f4 fix: the public methodology page is generated, so correct it in the generator`
- `6f3d25d brief: data run 2026-09-28`
- `2e08f62 data: screener run 2026-09-28 - 502 scored, top: EXPE HST BBY EIX MPC`

## The session's own account

> 2026-09-29 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.
> 
> **Health (rule 8, all five):** last code session ran? **yes** -
> `logs/nightly-2026-09-28_060001.log` ends "Run complete: shipped to main" |
> data loop published? **yes** - `logs/datarun-2026-09-29_020001.log` ends "Data
> loop complete", HEALTH: PASS, 0 fetch failures, 502 scored |
> evidence base at `1m` = **18 rows, newest 2026-08-28 (32 days ago, bound 40),
> 4 effective** - inside the tripwire, and the lag is the normal 30-33-day
> steady state |
> priority 0 **fixed 2026-08-24, not weakened** (`_effective_observations()`
> still gates; `allow_auto_apply` still false; 4 effective against a gate of 8) |
> top open roadmap item: **priority 3, backtest v2 - 35 days old**, deferred
> again today, see below.
> 
> **Tests:** before **1561/1561**, after **1615/1615** (+54)
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so nothing to
> claim or move to Done. Took the nominal **Tuesday product** focus. Nothing was
> deferred for a stalled loop or a failing gate - both loops are healthy and the
> baseline was clean. **Priority 3 deferred for the eighth time in ten sessions**;
> its gating step is still a procurement decision, not code (cost a price source
> for delisted tickers), and today's item closed a standing owner-directive gap
> instead - see "Tried and rejected".
> 
> ### Did
> 
> **Shipped north-star gap 4 - earnings-date proximity - which had been open 50
> days and was the last of the plan's cheap-and-high-value items.**
> `METHODOLOGY_CHANGELOG.md` 2026-09-29; `tests/test_earnings_date.py`, **54
> tests, 49 of which fail against the pre-change tree**.
> 
> **1. What the dashboard could not answer.** Opening it as a user, it can tell
> you 44 metrics about a company and cannot tell you **when that company next
> reports**. That matters twice over here. Behaviourally, it is the one selling
> behaviour the evidence positively endorses. Mechanically, this screener's
> Valuation, Quality and Growth inputs come from filings and barely move between
> them - measured 2026-09-17, the largest one-month Quality move was **one stock
> in 500** - so the report date is when those numbers are actually replaced. The
> score on screen has a shelf life and the page never said when it expires.
> 
> **2. What shipped.** Three `.info` fields captured at fetch
> (`earningsTimestampStart`, `earningsTimestampEnd`, `isEarningsDateEstimate`),
> carried as `stock_detail[t]["earn"]`, rendered as a new `earnings` sentence in
> the baked summary. **One implementation, two surfaces** - the drilldown and
> every My Holdings row - because `HOLDINGS_FACTS` lifts kinds out of the same
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

