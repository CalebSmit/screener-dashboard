# Morning Brief - Friday 23 October 2026, 02:28

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
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

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **register tests A, B and C (section 4) in `research/experiments/registry.yaml`, each in its own commit before any of its data is built; or refuse one here with the reason.** - open 1 day - `research/2026-10-22-free-data-that-adds-evidence.md`

## What changed in the repo

- `3e366b0 data: screener run 2026-10-23 - 498 scored, top: EXPE APA HST INCY SNDK`
- `1b02665 log + CLAUDE.md: the 2026-10-22 improvement wave; the bot's input gates recorded as settled`
- `5ab1eca research + OWNER_FOCUS: the input gates replayed against every published page`
- `502ca2b health: the trading bot's inputs - rank continuity, the unscored share, price dates; data-loop git hygiene`
- `8b9c494 research: which free data adds evidence, not more tests on the same history`
- `b4d0333 brief: code session 2026-10-22`
- `9bf29e2 log: the gates' numbers, the rerun check, and the three guards that fired`
- `15efe21 research engine: all 31 rows rerun level, and the page reports the stop arm`
- `3a56a63 log + OWNER_FOCUS: the 2026-10-22 stop-loss item, done and moved to Done`
- `5e9d966 research: the stop-loss arm's result, and the page reports it beside the unstopped rule`
- `513c364 research engine: register and build one stop-loss arm on the bot's rule (Han, Zhou & Zhu 2016)`
- `ad2346d brief: data run 2026-10-22`
- `9c0bc1b data: screener run 2026-10-22 - 498 scored, top: EXPE APA HST INCY EG`
- `a45fb93 OWNER_FOCUS: one pre-registered stop-loss test on the bot's rule (Han, Zhou & Zhu 2016), never a sweep`
- `eaef387 OWNER_FOCUS: the screener's half of the plan to real money - no wrong or stale ranking reaches the bot`

## The session's own account

> 2026-10-22 (owner-run, afternoon) - an improvement wave across both repos: the bot's safety and measurement, the screener's input gates, and which free data adds evidence
> 
> **Health (rule 8):** last code session ran? **yes** - 06:00 shipped (the stop-loss test). Data loop
> published? **yes** - 02:00, HEALTH PASS, claims 1,272 passed, 498 scored. Evidence base at `1m` =
> **34 rows, newest 2026-09-21 (31 days, bound 40), 4 effective**. Priority 0: `allow_auto_apply`
> **false**, 4 of 8. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 10
> days**) - not worked; the owner's request outranked it. The 13:00 slot stood down for the owner
> marker, as designed.
> 
> **Owner:** *"go and find something in these tools to improve ... maybe even coming up with a better
> plan ... so much data out there."* Answered with five agents in isolated worktrees and merged here.
> 
> ### Screener (this repo)
> - **The trading bot's input gates** (screener-bot ROADMAP 1.3-1.5, 1.9; `OWNER_FOCUS` 2026-10-21 items
>   1-4): rank continuity, the unscored share, price dates (`stock_detail[t].price_date`,
>   `scoring_schema` 16 - tonight's run refetches), and data-loop git hygiene (fail closed on a bad pull,
>   push only the data commit). **Replayed against all 72 published pages since 07-28: the four
>   known-bad runs fail, no ordinary day does.** `data-run.ps1` was parse-checked here
>   (`Parser::ParseFile`: OK) - the agent could not run PowerShell. `tests/test_input_gates.py` (75),
>   in the publish gate. CLAUDE.md row 0.5/0.7 and the data-loop gates paragraph updated.
> - **Free data that adds evidence** (`research/2026-10-22-free-data-that-adds-evidence.md`): the scarce
>   thing is months, not data. Three free, licensed sources add independent evidence or fix a recorded
>   weakness - **SEC 13F** (quarter-end prices for departed members, acquisitions included), **FINRA
>   short interest** (history from at least 2017-12, contradicting the "no free history" class for
>   `short_interest_ratio`), **SEC XBRL 2009-2011** (about 22 untouched months for backtest v2). Three
>   pre-registerable tests written, none registered: **Proposal status OPEN** for a later session.
> 
> ### screener-bot (owner-run; its SESSION_LOG has the detail)
> - Order safety (2.3-2.5), account safety (2.1, 2.2 built not switched on, 2.6, 2.7, 2.10, 2.11) and
>   measurement (3.3, 3.4, 3.6, 3.7 part) merged and joined: sells send the bot's own quantity; the
>   emergency stop keeps a wholesale close under its own name; the completeness check asks whether
>   each planned buy went in (a 95%-invested floor fired on a correct rebalance and was replaced at
>   merge). 475 tests, rehearsals clean, CI green. The setup panel now uses finished sessions only.
> 
> ### Next
> The data-scout proposal (register tests A-C or refuse them) is the open decision. Then the 2026-10-21
> OWNER_FOCUS item's parts 5-6 (a data-loop-only `signals.json` the bot trades from; the audit trail).

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

