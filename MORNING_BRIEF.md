# Morning Brief - Tuesday 24 November 2026, 02:30

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-11-24 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, APA, VLO, SNDK, MU |
| Ranking's lean on growth | +0.026 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 18 commit(s) changed what a reader sees, 6 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (58 rows, but overlapping windows are not independent; 142 rows across all horizons), newest 1m run date 2026-10-23 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 453/453 within 0.5%; operating cash flow 439/440 within 0.5%; net income 427/451 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 6 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#7: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); TPR (#13: equity 5.2% of assets); DXCM (#17: operating cash flow 36% off the filing); ES (#19: a period behind its filings); CF (#21: net income 15% off the filing) |
| Data run timing | **22 min** total; slowest `fetch.tickers` 606 s, `snapshot` 382 s, `context.sec_insider` 167 s; 0 tickers rate-limited |
| Code-session login | 2026-12-03 09:01 (9 days) |
| Context coverage | options usable 474/499 (25 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (**due 1 day ago - still listed**) - Build residual momentum in place of `jensens_alpha` (ACCEPTED 2026-11-21, weight 25 unchanged) in a 06:00 session (0.15's section 4 is committed - done 2026-11-21): section 9's build list - 37-month history on the same request with the Risk inputs trimmed to 13, `^SP500TR` to 4y, lineage/equation/claims, backtest v2, the engine rerun, a throwaway full run before the changelog entry. `research/2026-11-20-jensens-alpha-in-momentum.md`
- 2026-11-24 (**today**) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 10 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 13 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## Things that needed attention

- git pull --ff-only failed - main diverged locally. Resolve by hand.
- Could not fast-forward local main to the published brief: hint: Diverging branches can't be fast-forwarded, you need to either:

## What changed in the repo

- `0679feb data: screener run 2026-11-24 - 499 scored, top: EXPE APA VLO SNDK MU`
- `8a0cf49 NIGHTLY_LOG 2026-11-23 (owner-run, evening): the jammed loops, the push retry, the phone rows`
- `84086fc Phone rows: the flag test holds in both row forms; the publish gate's layout list follows the rename`
- `172a08e Phone: a ranking row is rank, ticker, the whole name and the composite`
- `931a9b1 Data loop: a failed push is retried, and one that never lands leaves main level with origin`
- `4a646bf brief: data run 2026-11-23`
- `1d4cd76 brief: code session 2026-11-23-midday - SESSION DID NOT RUN`
- `d8632b6 brief: code session 2026-11-23 - SESSION DID NOT RUN`
- `75a652d brief: data run 2026-11-23`
- `53dc4a7 data: screener run 2026-11-23 - 499 scored, top: EXPE VLO MU APA SNDK`
- `70492a3 NIGHTLY_LOG 2026-11-22 (owner-run, evening): the phone pass - measurements, gates, what was left`
- `dbe332d Phone pass: on a phone every long part is a heading that opens`

## The session's own account

> 2026-11-23 (owner-run, evening) - The loops were jammed by one failed push: unjammed, and a failed push now retries. Then the phone's ranking rows: nine numbers a row -> rank, name and composite
> 
> **Health (rule 8, all five):** last code session ran? **no - both of today's stood down**
> (`logs/nightly-2026-11-23_060001.log` and `..._130001-midday.log`: "git pull --ff-only failed - main
> diverged locally"), for the reason below; fixed in this session. Data loop published? **the 02:00
> run passed every gate (HEALTH: PASS, 499 scored, claims PASS) and its push failed on an HTTP 408**
> (`logs/datarun-2026-11-23_020001.log`); the 18:00 retry stopped on the same divergence. **Published
> by this session at 21:25**; `scripts/check_live_site.py`: "serving this run, 64 s after the push".
> Evidence base at `1m`: **56 rows, newest `run_date` 2026-10-21 (33 days ago, bound 40), 5 effective**
> of 8. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab**
> (opened 2026-10-12, **age 42 days**; a standing programme, last advanced **17 days** ago, bound 21).
> Shipped to readers: 18 commits changed what a reader sees, 8 methodology changes, in the last 7 days.
> 
> **Owner request:** *"Yes, simplify rows"* - the follow-up offered at the end of the 2026-11-22 phone
> pass. `logs/.owner-session` held from 21:14; removed at the end.
> 
> ### Found first: nothing had published or run since 02:23
> 
> - **What happened.** The 02:00 data run passed all eight gates and committed (`53dc4a7`). Its push
>   returned `HTTP 408 ... the remote end hung up unexpectedly` at 02:23:22 and the runner stopped with
>   "Push to main failed. Local main is ahead." Seventeen seconds later, from `finally`, it published
>   the morning brief - a single-file commit built on `origin/main`, exactly as designed - and that
>   made local `main` (one data commit ahead) **diverge**. The 06:00 session, the 13:00 session and the
>   18:00 data retry each stopped on "Not possible to fast-forward". One dropped connection cost the
>   day's page and three runs, and the health line's two "did it run" answers were both no.
> - **Unjammed.** `git merge --no-ff origin/main` on local `main` (no history rewritten; the four
>   origin commits touch only `MORNING_BRIEF.md`, and `git diff --stat 53dc4a7 HEAD` showed that one
>   file). The three payload files parse, `rebuild_page.py --check` says the page is up to date, the
>   newest run directory is still that run's. Pushed at 21:25. Then the steps the runner skips after a
>   failed push, by hand: `check_live_site.py` (serving the run, 64 s), `archive_run.py` (archived
>   `2026-11-23_bcdc1725766d`, 5 files), the success marker.
> - **Fixed** (`scripts/data-run.ps1`): the push is tried **five times, 60 s apart**; if it still
>   fails the runner fetches and asks whether `origin/main` is the data commit after all (a timeout
>   can come back after the ref moved); if not, and `HEAD` is still this run's own commit, it takes
>   the commit back (`git reset -q HEAD~1`) and calls `Undo-RunArtifacts` - the discard every gate
>   already uses - so `main` is level with `origin/main`, no success marker is written and the 18:00
>   retry reruns on the day's close. The refspec is unchanged (the data commit alone).
>   `tests/test_data_push_retry.py` (5); the script parses (`Parser::ParseFile`).
> - **Not done, and why:** the 18:00 retry cannot recover an already-diverged `main` by itself. With
>   the change above the runner no longer creates that state; a merge inside an unattended runner
>   that a trading bot reads from is a bigger thing than tonight's fault justified.
> 
> ### The rows (375 x 812, the live page as committed)
> 
> | | Before | After |
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

