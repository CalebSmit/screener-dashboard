# Morning Brief - Friday 20 November 2026, 07:00

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **refused by its publish gates (published-claims check failed)** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-19 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | SNDK, EXPE, HST, APA, MPC |
| Ranking's lean on growth | +0.042 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 13 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 458/458 within 0.5%; operating cash flow 443/444 within 0.5%; net income 432/456 within 0.5%; market value 453/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#6: net income 1.29x after-tax operating income); PSX (#7: net income 1.28x after-tax operating income); DXCM (#8: operating cash flow 36% off the filing); TPR (#12: equity 5.2% of assets); CF (#21: net income 15% off the filing); ES (#23: a period behind its filings) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 593 s, `fetch.retry` 152 s, `context.sec_insider` 106 s; **45 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (13 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **a Monday research session completes sections 4-5 against the criteria in section 3, which may not be changed after the first measurement runs** - open 1 day - `research/2026-11-19-what-the-ranking-is-for.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (in 3 days) - Monday: CLAUDE.md 0.15, what the ranking is for given no measured edge - measure against the criteria pre-registered in section 3 (fixed 2026-11-19; do not relax them), registering each test in the engine's registry before it runs. `research/2026-11-19-what-the-ranking-is-for.md`
- 2026-11-24 (in 4 days) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 14 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.

## What changed in the repo

- `3677b3c NIGHTLY_LOG 2026-11-20: retrospective - the refused run fixed, the watchdog's blind week, gate 5's blind spot`
- `a8e70a6 Retrospective prompt: the sixth hole, the watchdog cross-check, refusal counts, the split`
- `107fb8a CLAUDE.md 1,089 -> 940 lines: closed record and narrative to DECISIONS.md, verbatim`
- `99ebd1d A code run that never invoked Claude is labelled SESSION DID NOT RUN for the watchdog`
- `b2edc70 Gate 5 says when it cannot see the change: a scoring change is rehearsed on old scores`
- `a96295a The brief names the gate that refused a data run instead of "stopped deliberately"`
- `c8e40ec Tests read the missing-category rule the payload states: the 02:00 run was refused for it`
- `18f6d57 brief: data run 2026-11-20`
- `08b86c6 Shipped-to-readers number in the brief and health line; Set aside section and a five-line lane in the prompt`
- `c121314 Brief-skip test counts the lock-failure case; log text repaired`
- `447b820 No brief from a loop that never got the lock; CLAUDE.md 0.14 narrative to DECISIONS.md`
- `635d6de NIGHTLY_LOG: the night's gate results`
- `1386003 Drives column uses the page's percentage rule (pctPlain)`
- `ae78520 NIGHTLY_LOG 2026-11-19 (owner-run, night); the sustainable-growth fallback test inverted`
- `063486c Diagnostics shows what each category drives; a missed retrospective is owed next Friday; 0.15 pre-registered`

## The session's own account

> 2026-11-20 - RETROSPECTIVE. The 02:00 run refused and fixed first; the watchdog was blind to a week of aborted sessions; gate 5 is blind to the engine
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-11-19_130000-midday.log`
> (completed). Data loop published? **no** - `logs/datarun-2026-11-20_020001.log`, HEALTH: PASS, then
> **refused by the published-claims gate (7 failed)**; fixed this session, see below - the 18:00 retry
> publishes once this branch is merged. Evidence base at `1m`: **54 rows, newest 2026-10-19 (32 days ago,
> bound 40), 5 effective of 8**. Priority 0: `allow_auto_apply` **false**. Top open roadmap item:
> **0.15, What the ranking is for** (opened 2026-11-19, **age 1 day**); also open 0.13 (39 d, a standing
> programme), 0.10 (44 d), 0.14 (37 d). Shipped to readers: 13 commits, 8 methodology changes in 7 days.
> The health line also names `origin/nightly/2026-11-20` as held back - that is this branch, pushed
> early; not a stranded one.
> **Tests:** baseline **4,443 passed / 1 skipped** (green, matches the runner's 06:04 baseline). After: see
> "Gates" below. No owner-session marker; `git stash list` empty.
> 
> ### Did (before the retrospective): the refused data run
> 
> The 02:00 run scored 499 stocks, passed every health check, and was refused by
> `scripts/check_published_claims.py`: `test_case_against`, four in `test_dashboard_printed_figures`,
> `test_sheet_round_three`. Reproduced with `scripts/rehearse_publish.py` against run `4793e4da8a1a` (FAIL,
> same 7). **The page was right; the tests were stale.** The 2026-11-19 late-evening rule (an unscored
> category counts at 50 at full weight) shipped its generator branch keyed on the payload's
> `weights.missing_category_score`, but four tests still asserted the old "its weight is shared across the
> others" wording and a renormalised weight. Gate 5 had passed the change because it regenerated the page
> from the 11-19 run, scored *before* the rule, so the page took the old branch. Each test now checks the
> branch the payload's own field selects (and the hand-made cases set the contribution as the engine
> would). **Rehearsal against the refused run: PASS.** Commit `c8e40ec`.
> 
> (A false alarm of my own on the way, recorded so nobody repeats it: `logs/rehearse-publish/site/index.html`
> is a stale leftover from 11-06 - the rehearsal installs `dashboard.html` as the index. Copying the site
> folder by hand installs an old page.)
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 49 log entries, 2026-10-30 13:00 to 2026-11-20 00:30 - 24 scheduled, 25 owner-run**;
>   plus **15 scheduled slots that never started** (of ~43) and 1 gate-refused session (11-02 06:00, red
>   baseline inherited).
> - **Genuinely valuable: 24 of 24 scheduled entries, ~22 of 25 owner-run | Churn: 0 | Failed gates: 2**
>   (11-02 gate 1 on an inherited red `main`; 11-04 gate 5, unreproducible, merged by hand after three
>   passes) **| Never started: 15** (11-01 06:00, 11-01 13:00, 11-02 13:00 on a lapsed login; 11-10 13:00 on
>   a too-old CLI; eleven from 11-11 07:38 to 11-16 13:00 with no CLI on PATH).
> - **Data loop: 8 of 15 02:00 runs refused** (11-02, 06, 09, 10, 11, 12, 16, 20), all 8 by the
>   claims gate. **3 caught a real page defect** - a false "growth-neutral" sentence (11-02), a false
>   equality on 11 clamped rows (11-06), a 2 px sideways scroll on phones (11-12, twice); **5 were a wrong
>   test** - FOXA's share-class premise and the lab's font timing (11-09), Best Buy read as advice (11-10),
>   a footnote needing three movers (11-11 twice, 11-16), and today's stale rule. The 18:00 retry
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

