# Morning Brief - Monday 26 October 2026, 06:51

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-26T02:00:04.754086 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, INCY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (36 rows, but overlapping windows are not independent; 95 rows across all horizons), newest 2026-10-16 |
| Context signals | Context signals: 13 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 409/413 within 0.5%; net income 399/422 within 0.5%; market value 443/484 within 5%; filings cached 3 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#15: net income 16% off the filing) |
| Context coverage | options usable 477/498 (21 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **replace the scored *level* of implied upside with a 90-day *change* in the consensus target, scaled by price (section 6.2)** - open today - `research/2026-10-26-price-target-upside.md`

## What changed in the repo

- `6ed80a6 research: the delegated reads returned late and sharpened the note's reasoning`
- `6043575 research: price_target_upside - the level is 71% price, and the literature scores the change`
- `3dcc5a4 measure: price_target_upside is 71% price, not analyst opinion, at the monthly horizon`
- `6a618c8 brief: data run 2026-10-26`
- `18bb07d data: screener run 2026-10-26 - 498 scored, top: EXPE MU APA HST INCY`
- `30f7b28 CLAUDE.md: the bot reads signals.json since 2026-10-25 - keep write_signals.py's fields and order`
- `c637b84 brief: code session 2026-10-25-midday`
- `67c6ff8 log + OWNER_FOCUS: the 2026-10-25 midday session - the claims register's second mutation pass`
- `61890df plan: the 2026-10-25 mutation audit - 24 mutations, 11 claims, 4 real gaps and a false claim`
- `8564552 claims audit: the capex note's 5% reconciliation had no check that could see it`
- `905b2aa claims: register the two new summary.weakest boundary checks`
- `fc888b8 claims audit: two untested bars, a test named for a strictness it could not see, and a register sentence no code ever obeyed`
- `12b1b94 brief: code session 2026-10-25-weekend`
- `ef1a514 log: the 2026-10-25 weekend session - the positive control as a ledger row`
- `96ac5f6 test: the rerun planner's new shape - controls last, an unbuilt arm reported not run`

## The session's own account

> 2026-10-26 (06:00 session) - RESEARCH. The last unresearched weighted metric: `price_target_upside` is 71% price and 29% analyst at the horizon we score it over - and the screener is accidentally on the right side of the one thing the literature actually tested
> 
> **Health (rule 8, all five):** last code session ran? **yes** - the 2026-10-25 13:00 midday session
> ended "Run complete: shipped to main" at 13:44, tag `good/2026-10-25-1344`. Data loop published?
> **yes** - 2026-10-26 02:00, "Data loop complete", **HEALTH: PASS**, 498 scored, top EXPE MU APA HST
> INCY; this is the run that wrote the **first `signals.json`**, as the 10-24 entry predicted.
> Evidence base at `1m` = **36 rows, newest 2026-09-23 (33 days ago, bound 40), 4 effective**
> (`_n_observations` 4 against a gate of 8; 36 is the row count and never the one to quote; the
> whole file reads 95 rows / 2026-10-16, which is the number that cannot stand still). Priority 0:
> `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 14 days**) - advanced today rather than deferred, see below.
> **Tests:** before **3152 passed, 1 skipped** (376.8s) - identical to the 10-25 midday session's
> after-count, so no drift overnight. After **3152 passed, 1 skipped** - no production code changed
> today (Monday).
> **Owner queue / rotation:** `OWNER_FOCUS.md`'s top open item is 2026-10-20, whose sub-items 1, 2, 4
> and 5 are closed; **no proposal was waiting** (`open_proposals()` returns `[]`), which is the first
> thing the prompt asks. So the day's focus stood, and I took the Monday research slot in the way
> that also advances the top roadmap item: `plan/factor-lab.md`'s "From the lab to the weights" names
> five ranked weight questions, and this is the metric-level residual of question 2's category. **One
> thing deferred and stated:** sub-item 6's extended universe, which the last two sessions also
> deferred - it needs a records rebuild plus a ~45-minute serial append-only rerun, and a research
> day was not the session to start it. Sub-item 3's next mutation pass is likewise untouched.
> 
> ### Did
> 
> **Picked the one weighted metric in the composite with no research note of its own.** Everything
> around `price_target_upside` has been researched - the category was rebuilt on a 1,022-line note,
> `earnings_acceleration` was zeroed, `short_interest_ratio`'s data history was probed. This metric
> is mentioned four times across all of `research/` and examined nowhere; the category rebuild's own
> table says only "role unchanged". It carries 13.5 of the Revisions category's 100 points, so
> **1.35 composite points**, and it is the only metric in that category that is a *level* rather than
> a change - with price in its denominator.
> 
> `research/2026-10-26-price-target-upside.md`, with
> `research/measurements/2026-10-26-price-target-upside.py` + `.json`, and a second committed script
> that re-verifies every quoted paper figure against the PDF bytes.
> 
> **1. The measurement, from data already on disk - no return, IC or backtest read anywhere.**
> Implied upside in logs is exactly `log(target) - log(price)`, so its change decomposes additively.
> Reading `pt_mean` and `price` out of **56 committed versions of `dashboard_data.js`**:
> 
> - Between consecutive published runs the median absolute log change in the consensus target is
>   **exactly 0.0000** while the price moves 0.26-2.7%, and **60-77% of targets are byte-identical**.
>   corr(d upside, d price) runs **-0.82 to -0.99**.
> - The term structure, which is the part that matters, because daily is not the horizon anything is
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

