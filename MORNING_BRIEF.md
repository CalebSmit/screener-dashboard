# Morning Brief - Tuesday 03 November 2026, 13:54

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-03 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, SNDK, HST |
| Ranking's lean on growth | +0.073 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days ago, bound 40) |
| Context signals | 17 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 450/451 within 0.5%; operating cash flow 421/435 within 0.5%; net income 412/447 within 0.5%; market value 450/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 1 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); SNDK (#4: a period behind its filings); HST (#5: net income 1.46x after-tax operating income); DXCM (#6: operating cash flow 36% off the filing); EG (#8: a period behind its filings); PSX (#9: net income 1.28x after-tax operating income); CF (#13: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 631 s, `fetch.retry` 155 s, `snapshot` 145 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-02 10:28 (29 days) |
| Context coverage | options usable 474/499 (25 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide whether to fold the Investment category (`asset_growth`, 5%) into Quality at the weight-preserving 18.52% of a 27% Quality (section 3) or keep it standalone, as this note recommends, with the...** - open 1 day - `research/2026-11-02-investment-inside-quality.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 6 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). `CLAUDE.md`
- 2026-11-16 (in 13 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `0043f03 NIGHTLY_LOG: correct the after-count to the measured 3,788`
- `7aba7ef NIGHTLY_LOG: universe phase 0 decided, and rule 8's health line could not say 'fine'`
- `3814361 Classify the four health tests by what the mutation measured, not by their names`
- `9be2cdc Rule 8's health line could not leave the alarm state`
- `4cc1927 Owner queue: universe phase 0 decided and closed, next is phase 1`
- `6a81d10 Mutation-test the pre-registration guard: 5 gaps, all one defect, now 25 of 25`
- `7b5fb07 Decide the universe phase-0 proposal: all three pre-registrations adopted, two corrected`
- `c46387d wip: three measurements for the universe phase-0 review`
- `af2c168 brief: code session 2026-11-03`
- `b3ed924 Log and owner queue: the growth-exposure proposal decided, two still open`
- `007dbcc Decide the growth-exposure proposal: option 3, and the premise does not hold`
- `c3246c8 NIGHTLY_LOG: 11 false percentages on the live page, and the rule that is now one function`
- `2708177 Fix what the change broke, including one real coverage regression`
- `d1248cf The near-zero rule gets a tripwire, and four more formatters`
- `afc7a5d wip: one near-zero rule for every percentage the page prints`

## The session's own account

> 2026-11-03 (13:00 midday) - QUEUE. A second session today - more of the work, not a different kind. Read today's earlier NIGHTLY_LOG.md entry first and carry on from its Next; if it finished, take the next open item in OWNER_FOCUS.md, then the top of CLAUDE.md "Current priorities", then the queues in plan/factor-lab.md, plan/context-layer.md and plan/backtest-v2.md. Finish one item end to end, to the same evidence standard as any other session. Do not repeat or reverse a decision an earlier session made today without new evidence, and write down why if you do.
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-03_080346.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-03_020000.log` (completed;
> HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (43 rows, but overlapping
> windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days
> ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor
> Lab** (age **22 days**); also open: 0.10 27d; 0.14 20d.
> 
> **The line above is not the one the runner handed me, and that is this session's second finding.**
> The prompt's copy said *"last code session ran? **no** - `logs/nightly-2026-11-03_080346.log`:
> SESSION DID NOT RUN ()"*. That log ends `=== Run complete: shipped to main ===`, merged six commits
> and tagged `good/2026-11-03`. See item 2.
> 
> **Tests:** before **3,769 passed, 1 skipped** (green baseline, nothing inherited); after
> **3,788 passed, 1 skipped** - 19 new (15 pinning the pre-registrations, 4 on the health line).
> Gate 5 rehearsal PASSED (page tests 366 passed, check_published_claims 2,028 passed, published
> files restored byte for byte).
> **Owner queue / rotation:** carried on from the 06:00 session's **Next**, which asked a later
> session to decide one of the three OPEN proposals. Took the **universe-expansion phase 0**
> proposal, because it is the next step on the top item under `OWNER_FOCUS.md` **Open** (2026-10-28,
> beyond the S&P 500) - it is now **decided and closed**, and `plan/universe-expansion.md`'s next
> stage is phase 1. **Deferred: the Investment-inside-Quality proposal** (`research/2026-11-02-investment-inside-quality.md`,
> opened 2026-11-02, age 1 day), still the only one on the brief's "Waiting for a decision" list. It
> needs a measurement of its own - its section 4 lists four places a weight-preserving fold is not
> neutral - and one proposal decided properly is worth more than two skimmed. Any session from the
> 06:00 slot tomorrow on is an eligible reader; neither of today's sessions wrote it.
> 
> ### Did
> 
> **1. Decided the universe-expansion phase-0 proposal: all three pre-registrations ADOPTED, two of
> them corrected by measurements the note did not make.**
> 
> The note (`research/2026-11-02-universe-expansion-phase0.md`, owner-run, 2026-11-02) is the
> best-sourced thing in the queue and the review did not find it wrong. What it found is that two of
> its arguments rest on one month's data and one of its instructions contradicts itself. Record:
> note **section 9**; the operative decisions are now in `plan/universe-expansion.md`, pinned by
> `tests/test_universe_expansion_preregistration.py` (15 tests). Three measurement scripts are
> committed so every number can be re-derived.
> 
> - **PR2, the phase-4 floor: the headline number is exact.** Ken French's `ME_Breakpoints` was
>   downloaded again (CRSP vintage 202609, 1,210 monthly rows) and the June 2026 row read
>   independently: **n = 1,079 NYSE stocks, 20th percentile $1,224.13M** - both match the note to the
>   digit, as do all four of its "recent Junes for scale".
> - **But two of the five reasons given for it are struck.** Over 2015-2026 the line moved **+112.1%
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

