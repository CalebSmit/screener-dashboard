# Morning Brief - Friday 20 November 2026, 18:26

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-20 18:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, APA, SNDK |
| Ranking's lean on growth | +0.054 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 13 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (55 rows, but overlapping windows are not independent; 135 rows across all horizons), newest 1m run date 2026-10-20 (31 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 458/458 within 0.5%; operating cash flow 443/444 within 0.5%; net income 432/456 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#7: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); TPR (#9: equity 5.2% of assets); DXCM (#11: operating cash flow 36% off the filing); ES (#18: a period behind its filings); CF (#23: net income 15% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 644 s, `context.options_pass` 182 s, `fetch.retry` 149 s; **46 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (12 days) |
| Context coverage | options usable 494/499 - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **a Monday research session completes sections 4-5 against the criteria in section 3, which may not be changed after the first measurement runs** - open 1 day - `research/2026-11-19-what-the-ranking-is-for.md`
- **replace `jensens_alpha` in Momentum with CAPM residual momentum (option C: 36-month monthly regression on `^SP500TR` excess returns, residuals of months t-11..t-1 summed over their standard...** - open today - `research/2026-11-20-jensens-alpha-in-momentum.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (in 3 days) - Monday: CLAUDE.md 0.15, what the ranking is for given no measured edge - measure against the criteria pre-registered in section 3 (fixed 2026-11-19; do not relax them), registering each test in the engine's registry before it runs. `research/2026-11-19-what-the-ranking-is-for.md`
- 2026-11-24 (in 4 days) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 14 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`

## Things that needed attention

- ok   ATTENTION - 1 stock(s) now carry a market value implying a share count over 10% from the one their filings report, which the published page does not: SNDK +639.2%. Usually a completed acquisition or a split: the market value is right and every valuation ratio built on it has moved, while the earnings behind those ratios have not. Not a refusal - see research/2026-11-09-share-counts-and-corporate-actions.md.

## What changed in the repo

- `81f1d48 data: screener run 2026-11-20 - 499 scored, top: EXPE VLO MU APA SNDK`
- `072a500 brief: code session 2026-11-20-midday`
- `a467255 NIGHTLY_LOG 2026-11-20 (13:00): gates recorded`
- `4a1d52f Research: Jensen's alpha in Momentum is the 12-month return with a beta tilt; residual momentum proposed (OPEN)`
- `3a5d0a1 wip: research note - Jensen's alpha in Momentum (practice section pending)`
- `5377901 wip: measure how much of Jensen's alpha is the 12-month return Momentum already scores`
- `4d0f7ba brief: code session 2026-11-20`
- `3677b3c NIGHTLY_LOG 2026-11-20: retrospective - the refused run fixed, the watchdog's blind week, gate 5's blind spot`
- `a8e70a6 Retrospective prompt: the sixth hole, the watchdog cross-check, refusal counts, the split`
- `107fb8a CLAUDE.md 1,089 -> 940 lines: closed record and narrative to DECISIONS.md, verbatim`
- `99ebd1d A code run that never invoked Claude is labelled SESSION DID NOT RUN for the watchdog`
- `b2edc70 Gate 5 says when it cannot see the change: a scoring change is rehearsed on old scores`
- `a96295a The brief names the gate that refused a data run instead of "stopped deliberately"`
- `c8e40ec Tests read the missing-category rule the payload states: the 02:00 run was refused for it`
- `18f6d57 brief: data run 2026-11-20`

## The session's own account

> 2026-11-20 (13:00) - QUEUE. Jensen's alpha is the 12-month return with a beta tilt; the literature's version is residual momentum, proposed and left OPEN
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-20_060000.log`
> (completed). Data loop published? **no** - `logs/datarun-2026-11-20_020001.log` (refused by its publish
> gates (published-claims check failed); HEALTH: PASS) - fixed by this morning's retrospective (`c8e40ec`,
> merged); the 18:00 retry is the run that publishes. Evidence base at `1m`: **54 rows, newest 2026-10-19
> (32 days ago, bound 40), 5 effective of 8**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
> item: **0.15, What the ranking is for** (opened 2026-11-19, **age 1 day**); also open 0.13 (39 d, a
> standing programme), 0.10 (44 d), 0.14 (37 d). Shipped to readers: 13 commits, 8 methodology changes in 7
> days.
> **Tests:** baseline 4,447 passed / 1 skipped (the runner's, 13:04, GREEN). After: **4,447 passed / 1 skipped** (no change); dry run OK; payload parses, index.html 949,596 bytes; rehearsal (gate 5) PASSED; tree clean.
> **Owner queue / rotation:** `git stash list` empty; no owner-session marker. Today's earlier entry (the
> retrospective) left three Next items: (1) confirm the 18:00 retry - not yet possible at 13:00, and its
> precondition (the branch merged) is met: `origin/main` is `4d0f7ba`, which contains the fix; (2) Monday's
> 0.15 note - left for Monday as nominated; (3) the share-count rebuild (a 06:00 session to itself) and the
> lab's Friday on 11-27 - not mine. So I took the next open owner item: `OWNER_FOCUS.md` 2026-11-19
> item 3's remaining sub-items. Of those, **the IV inversion** is dated for a 06:00 session on or after
> 11-24 (the 11-19 review's decision: it needs a throwaway full live run, and one at 14:00 would be a second
> full fetch four hours before the 18:00 retry - the pattern behind every past throttle), so I did not
> reverse that; **the "->red" collapse** was already done 11-19; **the Jensen's-alpha note** was open, so
> that is today's work.
> 
> ### Did
> - **`research/2026-11-20-jensens-alpha-in-momentum.md`** - a complete note (literature, practice,
>   measurement, fit, options, proposal). Findings:
>   - **What the metric is.** Across one run `rf` and the market return are constants, so the scored
>     alpha is `return_12m - beta x (m12 - rf)` plus a constant: the 12-month return with a beta tilt
>     whose spread (s.d. 0.058) is an eighth of the return's (0.455). Same-window beta makes it, up to
>     compounding, the fitted intercept of a 12-month market model - the exact term Blitz, Huij & Martens
>     leave out of residual momentum on purpose.
>   - **Measured** (`research/measurements/2026-11-20-jensens-alpha-overlap.py` + `.json`, rank
>     correlations only, no forward return): today's run, sector percentiles, **0.981 with the 12-month
>     return, 0.871 with 12-1** (the fresh-eyes +0.87 reproduces), +0.18 with the latest month; month-ends
>     2012-2026 (177): median **0.969 / 0.918**, and **+0.28 with the latest month** where 12-1 is +0.02.
>     About 65 of Momentum's 100 points rank one 12-month move. Skipping the month alone leaves a 0.94-0.97
>     copy of 12-1; CAPM residual momentum (Blitz construction, 36-month fit) is **0.77** with 12-1 and
>     carries no load on the latest month.
>   - **No source uses the screener's construction.** Every momentum source skips the latest month for
>     stocks; the ones that remove market exposure use a longer-window residual, alpha excluded, skipped,
>     volatility-scaled - and report about half the volatility at a similar mean, including in the largest
>     decile (Blitz et al. Table 8: Sharpe 0.36 -> 0.60).
>   - **Fit:** the beta tilt flips sign with the market and rewards low beta a second time (Risk already
>     scores beta) in up years - the dynamic exposure the momentum literature blames for crashes,
>     entering through the adjustment meant to remove it.
> - **Proposal OPEN** (the brief lists it): replace `jensens_alpha` with CAPM residual momentum on
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

