# Morning Brief - Friday 30 October 2026, 13:59

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-30 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, HST, APA |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (40 rows, but overlapping windows are not independent; 105 rows across all horizons), newest 1m run date 2026-09-29 (31 days ago, bound 40) |
| Context signals | 16 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 375/376 within 0.5%; operating cash flow 361/365 within 0.5%; net income 353/372 within 0.5%; market value 439/484 within 5%; 14 scored a quarter behind a filing over 30 days old; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#5: net income 1.46x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); CF (#14: net income 16% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 663 s, `fetch.retry` 140 s, `context.sec_insider` 96 s; **42 tickers rate-limited** |
| Context coverage | options usable 490/499 (9 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## What changed in the repo

- `7a21ec2 log: 2026-10-30 (13:00) - the TTM-fill proposal refused as written, and the methodology page's computed sentences swept`
- `6c59f28 claims sweep: the methodology page, and a new published sentence is invisible to 3,566 tests`
- `3c2217e decide: correct a count in the review - six one-quarter stocks, not eight`
- `c5eee94 decide: the TTM-fill proposal - REFUSED as written, and its own case understated`
- `c13b46c wip: the measurements for the TTM-fill review`
- `4ca24b4 brief: code session 2026-10-30`
- `2a40afc log: 2026-10-30 RETROSPECTIVE - a red main found at baseline, and the blind spot that let it sit`
- `d00e2f5 rotation: Wednesday's subject is whatever changed the methodology, not only Monday's note`
- `221b2f4 process: CLAUDE.md 1036 -> 959; the fifth asymmetry scored; the step-1 enumeration as a tool`
- `5d19c04 runner: take gate 1's baseline before the session, and label the gate's failures against it`
- `475082f publish gate: the evidence files this loop appends to get their tripwire`
- `55b7ab0 tests: a red main - the unscored-member tests pinned yesterday's two failed fetches`
- `1fddb88 brief: data run 2026-10-30`
- `5c0cfb0 data: screener run 2026-10-30 - 499 scored, top: EXPE VLO MU INCY HST`
- `f3c5aa8 log: 2026-10-29 owner-run, last before 02:00 - unscored members, glossary, brief`

## The session's own account

> 2026-10-30 (13:00 midday) - QUEUE. The open proposal decided, then the claims sweep over the public methodology page
> 
> **Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 retrospective,
> `logs/nightly-2026-10-30_060001.log`, ends "shipped to main". Data loop published? **yes** -
> `logs/datarun-2026-10-30_020001.log`, "Data loop complete", HEALTH: PASS. Evidence base at `1m` =
> **40 rows, newest 2026-09-29 (31 days ago, bound 40), 5 effective** - steady state, unchanged from
> this morning. Priority 0: `allow_auto_apply` **false**, 5 of 8 effective. Top open roadmap item:
> **0.13, the Factor Lab** (opened 2026-10-12, **age 18 days**) - not advanced and not drifting; the
> next lab Friday is **2026-11-06** and no experiment is half-built. (0.10 is older - opened
> 2026-10-07, **age 23 days** - and its remainder is Sharpe and Sortino at weight 0, which CLAUDE.md
> calls "do them only if a session has nothing better".)
> **Tests:** before **3,566 passed / 1 skipped** (green), after **3,570 passed / 1 skipped** (+4,
> all in `tests/test_claims_register.py`). Dry run exits 0.
> **Owner queue / rotation:** `OWNER_FOCUS.md` item 1 sub-item 1 (decide the open proposal) and then
> sub-item 3 (test the tests), in that order, which is the order Orient and the item both give.
> Nothing deferred for a gate or a loop. `git stash list` shows `auto-rescue 2026-10-28_060001`,
> still empty, left in place as its own entry asks. No owner-session marker.
> 
> **Second session today - read the 06:00 retrospective's entry first.** Its **Next** was "decide the
> open proposal in `research/2026-10-29-ttm-fills-the-empty-quarter.md`", so that was this session's
> first job; nothing it decided was re-opened or reversed. Its process change 1 had its **first live
> use** on this session: `logs/nightly-2026-10-30_130001-midday.log` logs
> `BASELINE main is GREEN on arrival (3566 passed, 1 skipped ... 0:03:53)`, which matched my own
> baseline exactly, so for the first time a session could tell before starting that it had inherited
> nothing.
> 
> ### Did
> 
> **1. The open proposal, decided: REFUSED as written - and its own case was understated.**
> `research/2026-10-29-ttm-fills-the-empty-quarter.md` §5 (new), measurements
> `research/measurements/2026-10-30-trailing-series-review.py` and
> `...-trailing-column-moves.py` with their JSON. The proposal: when a stock's newest quarterly
> income column is empty (14 stocks on 2026-10-30, Amazon and Bank of America among them), score
> its income-statement inputs from the source's trailing-twelve-month series instead. It named five
> things to answer first; three are now measured.
> 
> **Two answer in its favour, and one of them decisively.**
> 
> - **The trailing series is definitionally the same rows under different names.** On the twelve
>   highest-ranked stocks that are *not* flagged - so their quarterly statement is the one
>   `_stmt_val_ltm` actually sums - across all nine income-statement labels the engine reads, the
>   trailing series against the engine's own four-quarter sum: **median relative difference
>   0.000000**, max 0.00067 (Total Revenue), 0.00061 (Operating Income), 0.00052 (Net Income); nine
>   of twelve agree **exactly** on every label, and the single worst figure anywhere is EG's interest
>   expense at 0.68%. So the substitution preserves the 2026-10-15 decision to keep the source's
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

