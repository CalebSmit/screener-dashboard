# Morning Brief - Saturday 31 October 2026, 14:00

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-30 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, HST, APA |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (40 rows, but overlapping windows are not independent; 105 rows across all horizons), newest 1m run date 2026-09-29 (32 days ago, bound 40) |
| Context signals | 16 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 375/376 within 0.5%; operating cash flow 361/365 within 0.5%; net income 353/372 within 0.5%; market value 439/484 within 5%; 14 scored a quarter behind a filing over 30 days old; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#5: net income 1.46x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); CF (#14: net income 16% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 663 s, `fetch.retry` 140 s, `context.sec_insider` 96 s; **42 tickers rate-limited** |
| Context coverage | options usable 490/499 (9 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## What changed in the repo

- `b40ce88 log: 2026-10-31 (13:00) - nine more audit rows, and the morning's two were never registered`
- `dfe8de6 plan: the ninth mutation pass, and five rules the pass produced`
- `66be3a9 claims audit: the methodology page's last four prose rows, and one equivalent mutant recorded`
- `58ebd6d claims audit: the whole "best backtest" section, and the module checking it joins the gate`
- `00aa22e claims audit: the two rows closed at 06:00 were never closed in the register`
- `5955e39 brief: code session 2026-10-31-weekend`
- `f2d817c log: 2026-10-31 weekend - a red main fixed at its root, a clock-read sweep, and four open audit rows`
- `57a15cb claims audit: the trading rule's monthly figure stops being a wildcard, and its two coverage years`
- `a306d32 claims audit: the two open rows the plan called sharpest - the clamp bullet and the switching margin`
- `7cc51ef tests: the clock-read sweep - three red gates in three weeks, one guard`
- `e2816d5 tests: a red main - the download-skip test's scenario came from the clock, so it failed every weekend`
- `52e1bcb brief: code session 2026-10-30-midday`
- `7a21ec2 log: 2026-10-30 (13:00) - the TTM-fill proposal refused as written, and the methodology page's computed sentences swept`
- `6c59f28 claims sweep: the methodology page, and a new published sentence is invisible to 3,566 tests`
- `3c2217e decide: correct a count in the review - six one-quarter stocks, not eight`

## The session's own account

> 2026-10-31 (13:00 midday) - QUEUE. Nine more audit rows, and the morning's own two turned out not to have been closed
> 
> **Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 weekend session,
> `logs/nightly-2026-10-31_060001-weekend.log`, ends "shipped to main". Data loop published?
> **yes** - the last weekday run, `logs/datarun-2026-10-30_020001.log`, "Data loop complete",
> HEALTH: PASS, signals written (40 names, 8 gates, all ran); today is a Saturday, so no 02:00 run
> is due. Evidence base at `1m` = **40 rows, newest 2026-09-29 (32 days ago, bound 40), 5
> effective** - steady state, unchanged from this morning. Priority 0: `allow_auto_apply` **false**,
> 5 of 8 effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 19
> days**) - not advanced; the next lab Friday is **2026-11-06** and no experiment is half-built.
> (0.10 is older - opened 2026-10-07, **age 24 days** - and its remainder is Sharpe and Sortino at
> weight 0, which CLAUDE.md says to do only with nothing better.)
> **Tests:** before **3,594 passed / 1 skipped**; the runner's own baseline line agrees
> (`BASELINE main is GREEN on arrival (3594 passed, 1 skipped ... 0:04:01)`), after **3,602 passed /
> 1 skipped** (+8, all new). Dry run exits 0. `scripts/rehearse_publish.py` **PASSED** (the diff
> touches `claims.py`): 362 page tests, published claims 1,850 passed, all four published files
> unchanged and restored byte-for-byte.
> **Owner queue / rotation:** no proposal is waiting (`write_brief.open_proposals()` returns `[]`),
> so `OWNER_FOCUS.md` item 2 sub-item 3 ("test the tests"), carrying on from the 06:00 entry's
> **Next** exactly as the focus asks. Nothing deferred for a gate or a loop. `git stash list` shows
> `auto-rescue 2026-10-28_060001`; `git stash show -p` prints nothing and
> `--include-untracked --name-only` lists three gitignored data files (a context log and two price
> caches), so there is **no tracked work in it** - left in place, as every session since has. No
> owner-session marker.
> 
> **Second session today.** The 06:00 entry's Next named `_pbo_words` at its own bars, then the
> three bank P/B-ROE prose rows and the coverage-discount worked example. All of those are done -
> and so is something that entry did not know it had left behind: its own two rows.
> 
> ### Did
> 
> **1. The two rows the 06:00 session closed were never closed in the register.** This audit's own
> rule, met from the inside: *"Finishing a gap means registering the fix too. A new test that
> `checked_by` does not name reads as a gap on the next pass and is outside the publish gate's
> derivation."*
> 
> Commit `a306d32` wrote five real checks for `share-class-switching-margin` and extended one for
> `clamp-bullet-second-printing`. It touches **no register file**: both sentences still read
> `open:` in `OVERVIEW_SENTENCES`, `OVERVIEW_KNOWN_OPEN` still named both slugs, and the five new
> `TestThePagePrintsTheMargin` checks were in no claim's `checked_by`. "`KNOWN_OPEN` drops twelve to
> ten", written in that entry and in `OWNER_FOCUS.md`, was true of the **research** page's two rows
> only - the 07:00 commit did move those.
> 
> **Measured rather than inferred**, with a harness that differs from the morning's in the one way
> that matters: `research/measurements/2026-10-31-registered-row-mutations.py` runs **the claim's
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

