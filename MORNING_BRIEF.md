# Morning Brief - Thursday 05 November 2026, 07:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-05 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, EXPE, MU, SNDK |
| Ranking's lean on growth | +0.060 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons), newest 1m run date 2026-10-05 (31 days ago, bound 40) |
| Context signals | 19 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 436/437 within 0.5%; operating cash flow 403/420 within 0.5%; net income 397/433 within 0.5%; market value 446/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 3 days ago |
| Top 25, worth a look | EXPE (#3: equity 4.2% of assets); HST (#6: net income 1.46x after-tax operating income); DXCM (#7: operating cash flow 36% off the filing); PSX (#8: net income 1.28x after-tax operating income); CF (#13: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 660 s, `fetch.retry` 121 s, `context.sec_insider` 107 s; **38 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (28 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 4 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 11 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## What changed in the repo

- `9fe087c Log: the stale ledger, the check that sees it, and what the next build costs`
- `d8f04f3 A ledger row is checked against the code it is published beside`
- `201c21e Bring the research-engine ledger level with the code it is published beside`
- `c58d886 Both decisions' measurement outputs, and how the baseline count was obtained`
- `ff6b8ec Log: both open proposals decided, and the guard the first one needed`
- `66d5343 OWNER_FOCUS: both open proposals decided; the queue is empty`
- `c93f568 Proposal decided: the 13F holder-count log line is refused as written`
- `dd0913f The first rate-limit position record: a contiguous wall at 352-389, not a scatter`
- `edc71ce Proposal decided: the short-interest registration is refused as written`
- `8b37648 The registry cannot hold an experiment v2 is unable to measure`
- `aa5fa7f brief: data run 2026-11-05`
- `7d7c841 data: screener run 2026-11-05 - 499 scored, top: VLO MPC EXPE MU SNDK`
- `4eb5ebf Log: the stale agent worktrees are removed`
- `6188976 Log: the data loop's blind spots - where the rate limit hits, a landed refusal, retention, named stocks`
- `2780d72 Browser tests open a stock the payload carries, so an index change cannot discard a night's data`

## The session's own account

> 2026-11-05 - BUILD. Implement what the week's research justified. Write tests alongside the code.
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-04_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-05_020000.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **5 of 8 effective observations needed (44 rows; 112 across all horizons),
> newest 1m run date 2026-10-05 - 31 days ago, bound 40.** Priority 0: `allow_auto_apply` **false**.
> Top open roadmap item: **0.13, The Factor Lab** (age **24 days**); also open 0.10 (29d), 0.14 (22d).
> No branch held back by the gates. `git stash list` carries one `auto-rescue 2026-10-28_060001` entry
> that `git stash show` reports as empty (no diff, no stat) - left alone, as four prior sessions did.
> **Tests:** after **3,926 passed / 1 skipped** (parallel, 3:04). **The baseline was not a separate
> run** - it is read off the mutation check in item 2, which ran the whole suite with this session's
> first new module excluded and reported 3,906 passed with the 2 failures the mutation itself caused,
> i.e. **3,908 passed / 1 skipped** on the arriving tree, matching the 11-04 evening entry. The 18 new
> tests (11 + 7) account for the difference exactly. All five gates pass on this tree: suite, dry-run,
> both payloads parse, clean tree, and `scripts/rehearse_publish.py`.
> 
> **Owner queue / rotation:** `OWNER_FOCUS.md` item 1 - **decide the OPEN proposals** - outranked the
> rotation, and both were listed in the brief at 2 days old. **Both decided; the queue is empty for the
> first time since 2026-11-03.** Thursday's build is the guard the first decision needed, so this is a
> build day as well as a decision day. Deferred: the implied-volatility inversion, named in the 11-04
> 13:00 entry's Next - it is the accepted ninth proposal's build and needs a session that is not also
> clearing two decisions; it is this entry's Next.
> 
> ### Did
> 
> 1. **Proposal decided - `research/2026-11-03-short-interest-days-to-cover.md` section 5: REFUSED as
>    written. The experiment it asks for cannot measure anything.** It wanted a research-engine
>    experiment dropping `short_interest_ratio` from Revisions. The engine has never scored that
>    metric: `v2_config` zeroes it and the record leaves its column **NaN in 176 of 176 months**
>    (68,197 stock-months), which makes **Revisions the only category backtest v2 does not score at
>    all**. Measured over every month on file
>    (`research/measurements/2026-11-05-short-interest-registration.py`): the arm, written the way the
>    registry's own metric-level precedent writes it (`drop-revenue-growth`, `kind: config`), is
>    **dict-identical to the baseline**, and ranks **0 of 67,113 stock-months differently - maximum
>    composite difference 0.0**. Read as `drop_category revisions` it is identical too. So the row
>    would publish "no difference, not distinguishable from zero" - the note's own prediction - which
>    a reader takes as *the leg is not earning its 1.35 composite points* and is in fact an identity.
>    The third expression the registry allows (`kind: metrics`) does move the ranking, but only by
>    putting weight back on five all-NaN analyst columns.
>    **The second half is blocked by the engine's own rule, not only by the missing panel:** FINRA's
>    files begin 2017-12-29, which is **80 of 80** months after 2020 and **25 of 96** in the presample,
>    and nothing is CONFIRMED without a presample arm pointing the same way - so a built panel could
>    only be judged on 2018-2026, a `METHOD_VERSION` change that reruns everything.
>    The note's research and its verdict (the weight stays; placement inside Revisions is a synthesis
>    question) are untouched. Note section 6.
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

