# Morning Brief - Friday 30 October 2026, 06:40

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

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **fill an empty newest quarter from the source's own trailing series (9 of 15 lagging stocks, Amazon among them); answer section 4 first.** - open 1 day - `research/2026-10-29-ttm-fills-the-empty-quarter.md`

## What changed in the repo

- `2a40afc log: 2026-10-30 RETROSPECTIVE - a red main found at baseline, and the blind spot that let it sit`
- `d00e2f5 rotation: Wednesday's subject is whatever changed the methodology, not only Monday's note`
- `221b2f4 process: CLAUDE.md 1036 -> 959; the fifth asymmetry scored; the step-1 enumeration as a tool`
- `5d19c04 runner: take gate 1's baseline before the session, and label the gate's failures against it`
- `475082f publish gate: the evidence files this loop appends to get their tripwire`
- `55b7ab0 tests: a red main - the unscored-member tests pinned yesterday's two failed fetches`
- `1fddb88 brief: data run 2026-10-30`
- `5c0cfb0 data: screener run 2026-10-30 - 499 scored, top: EXPE VLO MU INCY HST`
- `f3c5aa8 log: 2026-10-29 owner-run, last before 02:00 - unscored members, glossary, brief`
- `a5483e8 plan: dashboard inventory - today's surfaces and the unscored payload key`
- `05d636c page: an S&P 500 member the run could not score is named and told why`
- `1c026d6 brief: the run time to the minute, and the context row without its own label twice`
- `b072c5d methodology: the glossary spells out the abbreviations the page uses`
- `e8e7bbd test: the dead-badges flake will say which of its two causes it was`
- `80cc059 log: the dead-badges flake's traceback, and what it rules out`

## The session's own account

> 2026-10-30 - RETROSPECTIVE. Evaluate whether this routine is actually producing value, and change the process where it is not.
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-29_130001-midday.log`
> ends "Run complete: shipped to main" (tag `good/2026-10-29-1403`). Data loop published? **yes** -
> `logs/datarun-2026-10-30_020001.log` ends "Data loop complete", HEALTH: PASS, run `24d9938b511b`,
> 499 scored, `signals.json` written with 8 gates all run. Evidence base at `1m` = **40 rows, newest
> 2026-09-29 (31 days ago, bound 40), 5 effective** - in steady state; the data log's own line
> agrees. Priority 0: `allow_auto_apply` **false**, unchanged, 5 of 8 effective. Top open roadmap
> item: **0.13, the Factor Lab** (opened 2026-10-12, **age 18 days**) - not advanced and not
> drifting: today is an even ISO week so the runner swapped in the retrospective, and the next lab
> Friday is **2026-11-06**. (0.10 is older - opened 2026-10-07, **age 23 days** - but its remainder
> is Sharpe and Sortino at weight 0, which CLAUDE.md already calls "do them only if a session has
> nothing better".)
> **Tests:** before **2 failed / 3,560 passed / 1 skipped** - a red `main`, see below - after
> **3,566 passed / 1 skipped** (+6: the 2 fixed, plus 4 new - 1 stand-in guard, 1 in the publish
> gate's module, 2 in the gate-ordering module). Dry run exits 0.
> **Owner queue / rotation:** retrospective, so no rotation item, and `OWNER_FOCUS.md`'s two open
> items (universe expansion phase 0, a Monday; the 2026-10-20 fine-tune block) were read and left
> open - a retrospective does not work on the screener. **One open proposal is deliberately left**
> (`research/2026-10-29-ttm-fills-the-empty-quarter.md`, fill an empty newest quarter from the
> source's trailing series, 1 day old): a retrospective is the wrong session to decide a methodology
> question, the same call the 2026-10-16 retrospective made and recorded. `git stash list` shows
> `auto-rescue 2026-10-28_060001`, still empty (`git stash show --stat` prints nothing) and left in
> place as its own entry asks. No owner-session marker.
> 
> ### Retrospective findings
> 
> - **Sessions reviewed: 38** (2026-10-16 to 2026-10-30) - **24 scheduled, 14 owner-run.**
> - **Genuinely valuable: 35 | Churn: 0 | Lost: 1 | Stood down correctly: 2 | Failed gates: 2.**
> - **And one finding that was not in the logs at all: `main` was red when this session started.**
> 
> **0. The red `main`, found at baseline and fixed first.** `tests/test_unscored_members.py` shipped
> the evening of 2026-10-29 naming **HONA and VYLR**, the two S&P 500 members that morning's fetch
> had left unscored. The 02:00 run on 10-30 served HONA, so it is in the payload,
> `notInRunText('hona')` correctly returns `''` for a stock that *is* in the run, and two tests
> failed on a page that was right. **`main` was red from 02:21 and nothing said so**: the data run
> logged HEALTH: PASS, the brief reported a healthy run, and no commit had touched any code.
> 
> Without this session looking, the runner's gate 1 would have failed at ~08:00 on a defect the
> session did not cause, the whole retrospective would have been stranded on a branch, and the log
> would have read like its fault. Both halves are now fixed - the test, and the blindness (process
> changes 1 and 2).
> 
> **1. What fraction of sessions produced something genuinely valuable? 35 of 38, with 3 that
> were not sessions' fault.** Nothing in the churn column for the second retrospective running. The
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

