# Morning Brief - Thursday 19 November 2026, 14:13

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-19 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | SNDK, EXPE, HST, APA, MPC |
| Ranking's lean on growth | +0.042 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 459/459 within 0.5%; operating cash flow 444/445 within 0.5%; net income 433/457 within 0.5%; market value 453/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 1 days ago |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); HST (#3: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); DXCM (#9: operating cash flow 36% off the filing); TPR (#12: equity 5.2% of assets); CF (#18: net income 15% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 645 s, `fetch.retry` 162 s, `context.sec_insider` 101 s; **52 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (14 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## What changed in the repo

- `344e4bd NIGHTLY_LOG 2026-11-19 (13:00): the engine's share counts traced, the rule that failed one criterion, the gates' results, what is next`
- `ffea6e3 Research note, plan, CLAUDE.md: the engine's share counts - four causes traced to filed facts, three candidate rules, the third fails one of four pre-written criteria`
- `8adeb3f wip: share-count rule versions 1 and 2 measured; version 3 and its four acceptance criteria written before it is run`
- `961b353 The withdrawn 'Its filings are flagged stale' sentence is forbidden; the first combined 10-Ks have a date in the brief`
- `280b983 wip: measurement - the engine's share counts traced to their filed facts, candidate rule written before its evaluation`
- `0465ce4 brief: code session 2026-11-19`
- `69ef668 NIGHTLY_LOG 2026-11-19: the universe block, the floor that never fired, the gates' results, what is next`
- `07c7572 Research engine: all 32 rows rerun on the new config - 27,225 statistics, 0 moved; only the config hashes differ`
- `a27370b Changelog, plans, CLAUDE.md: the market-cap floor is 0 and never fired; wrong market caps in the engine's history recorded`
- `6911cd0 wip: the universe block is read - an undefined index is refused, and a market-cap floor that is set is applied`
- `f949d3a brief: data run 2026-11-19`
- `3017bd6 data: screener run 2026-11-19 - 499 scored, top: SNDK EXPE HST APA MPC`
- `e365cfc brief: code session 2026-11-18-midday`
- `4e34d62 NIGHTLY_LOG 2026-11-18 (13:00): where the missing filings went, the gates' results, what is next`
- `04e3f32 Research note, changelog, CLAUDE.md 0.14: where twenty companies' filings went`

## The session's own account

> 2026-11-19 (13:00) - QUEUE. The engine's wrong market caps traced to their filed facts: four causes, about 3% of name-months, and a candidate rule that failed one of its four criteria - designed, not built
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-19_060001.log` (completed). Data loop published? yes - `logs/datarun-2026-11-19_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (54 rows, but overlapping windows are not independent; 133 rows across all horizons), newest 1m run date 2026-10-19 (31 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **38 days**; a standing programme, last advanced **13 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 43d; 0.14 36d.
> 
> **Tests:** before **4,393 passed / 1 skipped / 0 failed** (3:58). After **4,394 passed / 1 skipped /
> 0 failed** (4:22) - one new parametrized case. Gate 2 (dry-run) completes. Gate 5,
> `scripts/rehearse_publish.py --changed-since origin/main`: **REHEARSAL PASSED**, all published
> files unchanged byte for byte (it parses the three payload files; I did not run a separate
> `node --check`). Gate 4 clean.
> 
> **Owner queue / rotation:** second session today; this morning's entry read first. `git stash
> list` empty, no branch held back, `open_proposals()` is `[]`. `OWNER_FOCUS.md`'s top item is the
> owner's alone; the universe-expansion item advanced this morning. I carried on from the morning's
> **Next**: items 1 and 2 are tomorrow's by their nature; I did **3** (small) and the diagnosis half
> of **4**. No decision of the morning's was reversed. 0.13 is not deferred (13 days against 21;
> tomorrow is its day). Nothing deferred for a gate or a loop.
> 
> ### Did
> 
> **1. The engine's share counts, followed to the filed facts**
> (`research/2026-11-19-engine-share-counts.md`). The morning found 16 tickers under $100M and
> Berkshire under $2B and said to follow each count to its filing before choosing a rule. Four
> causes, not the three it guessed at:
> 
> - **A stale tag read as "latest".** Simon Property: 8,000 shares, dated 2012, picked in every
>   month to August 2026. Fox: 1 share, dated March 2019. Baker Hughes: 100.
> - **A units slip downward.** Stryker's balance sheet says 383; its cover page, same date,
>   382,672,989.
> - **The same slip upward - which this morning's check could not see**, because it looked only
>   for small values. Garmin reads $9.6 quadrillion in December 2016; 84 name-months across 15
>   tickers are over $4 trillion and only 32 of them (Apple, Alphabet, NVIDIA, 2025-26) are real.
> - **A logic error: the split adjustment is taken from the month-end, not from the day the count
>   was filed.** NVIDIA reads $2,697bn, **$304bn, $288bn**, $2,928bn across May-August 2024. 194
>   name-months, 77 tickers, at exact split ratios. The SEC's own rule (SAB Topic 4.C, fetched
>   today) is why the filed date is the right basis.
> 
> **Size: 1,992 name-months (3.06%, 135 tickers) are more than 10% off, 473 by more than 10x, and
> 1,635 have no market value where one is available** - against the 494-655 recorded this morning.
> Spread evenly, 2.6-3.5% of every year.
> 
> **2. A guard I expected to use and measured out.** The weighted-average share count looked like
> the natural reference. It is itself filed in thousands or millions for **56 tickers**
> (ConocoPhillips for 97 months; 598 ticker-months more than 10x off), so a guard trusting it
> would have replaced correct counts.
> 
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

