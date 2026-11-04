# Morning Brief - Wednesday 04 November 2026, 13:43

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-04 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, SNDK, APA |
| Ranking's lean on growth | +0.051 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 111 rows across all horizons), newest 1m run date 2026-10-02 (33 days ago, bound 40) |
| Context signals | 18 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 447/448 within 0.5%; operating cash flow 414/431 within 0.5%; net income 406/444 within 0.5%; market value 449/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 2 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); DXCM (#6: operating cash flow 36% off the filing); HST (#7: net income 1.46x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); CF (#12: net income 16% off the filing) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 659 s, `fetch.retry` 128 s, `context.sec_insider` 118 s; **40 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (29 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide whether to log (not show, not score) one line per stock - the number of 13F filers holding it and the change on the prior quarter, dated by quarter end and filing date - as a context-log...** - open 1 day - `research/2026-11-03-following-the-smart-money.md`
- **register in `research/experiments/registry.yaml`, before running it, a research-engine experiment that drops `short_interest_ratio` from Revisions (and a second that scores FINRA days to cover from...** - open 1 day - `research/2026-11-03-short-interest-days-to-cover.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 5 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 12 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `60e7d49 Log: the implied-volatility proposal is accepted, and the fetch has no ex-date`
- `ef9bc75 Decide the implied-volatility proposal: ACCEPTED, and the switch costs one day not a month`
- `3074106 Record which construction an implied volatility came from, so adopting the inversion splices nothing`
- `fb27f1c Pre-register the QuantLib review's replication before computing anything`
- `9f3c66a Log: the 06:00 work landed; a refused branch is named to the next session`
- `ca674ac Runner: a refused branch is named to the next session, and a failed gate keeps its output`
- `3d76cfd brief: code session 2026-11-04`
- `39c7080 Log: the auto-rescue stash is not empty - git stash show -p cannot see an -u stash`
- `05030b7 Log and owner queue: the Investment fold decided, and the page rebuilt to carry it`
- `320a2a2 shot_methodology.py: the section and the paragraph are arguments, not literals`
- `7668fda Investment stays standalone: the fold refused, and the page now says why`
- `4726ae8 Review measurement: the Investment-into-Quality fold, rebuilt independently and run over eight payloads`
- `b876aa4 brief: data run 2026-11-04`
- `bd1702c data: screener run 2026-11-04 - 499 scored, top: EXPE VLO MU SNDK APA`
- `9a289d9 OWNER_FOCUS: the night's later items`

## The session's own account

> 2026-11-04 (13:00 midday) - QUEUE. A second session today - more of the work, not a different kind. Read today's earlier NIGHTLY_LOG.md entry first and carry on from its Next; if it finished, take the next open item in OWNER_FOCUS.md, then the top of CLAUDE.md "Current priorities", then the queues in plan/factor-lab.md, plan/context-layer.md and plan/backtest-v2.md. Finish one item end to end, to the same evidence standard as any other session. Do not repeat or reverse a decision an earlier session made today without new evidence, and write down why if you do.
> 
> **Health (rule 8, all five):** last code session ran? **the runner's line said "held back by its
> gates"** - `logs/nightly-2026-11-04_060000.log` - but **nothing is actually held back**:
> `health_numbers.held_back` named no branch and `git log origin/main..origin/nightly/2026-11-04` is
> empty, because the owner-run morning session merged that work after re-running all five gates (its
> entry above). So there was no refused branch to merge, and the line is reading the 06:00 run log's
> own ending rather than where the work ended up - worth knowing before the next session spends time
> looking for it. Data loop published? yes - `logs/datarun-2026-11-04_020001.log` (completed; HEALTH:
> PASS). Evidence base at `1m`: **5 of 8 effective** observations needed at the `1m` horizon (43 rows,
> but overlapping windows are not independent; 111 rows across all horizons), newest `1m` run date
> **2026-10-02 (33 days ago, bound 40)**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
> item: **0.13, The Factor Lab** (age **23 days**); also open: 0.10 28d, 0.14 21d.
> **Tests:** before **3,896 passed / 1 skipped**; after **3,901 passed / 1 skipped**.
> **Owner queue / rotation:** I am today's **third** session. The 06:00 entry's Next named the three
> open proposals as the first job, and prompt section 1 says deciding one outranks the focus, so I
> took the one it named first. Nothing was deferred for a broken loop or a gate. The deferral-streak
> rule did not fire: this session carried an owner item (`OWNER_FOCUS.md` item 1, "Decide the OPEN
> proposals"). No decision made earlier today was reversed.
> 
> ### Did
> 
> **Decided the implied-volatility proposal: ACCEPTED - the options panel should invert its own
> bid/ask mids rather than take the quote response's `impliedVolatility` field as given.**
> `research/2026-11-03-open-source-quant-stack.md` section 1, open 1 day, written by an owner-run
> session; section 1d is the review. This is the **ninth** proposal reviewed and the **second running**
> to be accepted after six refusals. I did not write the note or its measurement.
> 
> 1. **Pre-registered my own reading before computing anything** (section 1c, commit `fb27f1c`): a
>    disjoint sample, the candidate construction the author could not test, the dividend gap in its
>    cross-check, the literature check, and the adopt/refuse rule. The author's own pre-registration
>    checks out in `git log` too - section 1a in `57958d5`, the result in `741215b`.
> 2. **Replicated on a disjoint sample, mid-session** (`recs[5::10]`, no overlap with the author's
>    `recs[::10]`): 40 of 50 stocks priced, QuantLib minus the field **median -1.78 volatility points,
>    median |difference| 2.00, max 7.51**. The 1-point bar holds on a second sample; the gap is about a
>    third narrower at 14:10 ET than the author's 3.16 at 21:00.
> 3. **Closed the gap in the author's cross-check.** Its printout said 10 of 10 ATM calls reproduced an
>    independently written binomial tree but not how many carried a **dividend** - the part the note
>    itself calls hardest. Measured on two sets chosen by the same rule: **10 of 10 with no dividend
>    and 9 of 9 with one**, 19 of 19 within half a point (largest CF, 60.46% against 60.82%).
> 4. **Found the deciding evidence already in this repository.** The note marks its key inference -
>    that a negative put skew is the unusual case - as uncited. It needn't be: **Xing, Zhang & Zhao's
>    Table 1 (p. 647)** puts the 5th percentile of their skew at **-0.99** points, so a negative skew
>    is about a **one-in-sixteen** event in the data the measure comes from. It was read from the paper
>    by the 2026-10-14 session and sits in
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

