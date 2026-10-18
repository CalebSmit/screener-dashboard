# Morning Brief - Sunday 18 October 2026, 06:18

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## THE ROUTINE IS NOT RUNNING

- **Data run** has not run since 2 days ago

Nothing below is current. A loop that stops firing writes no log, so
the rest of this page describes the last run that *did* happen, not
today. Most likely cause: the PC rebooted and nobody logged back in -
the tasks only run while a user is signed in. See NIGHTLY_LOG.md
2026-08-20 and `scripts/register-tasks.ps1`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran 2 days ago |
| Code session (latest) | **stopped deliberately** - last ran today |
| Dashboard data from | 2026-10-16T02:00:03.815047 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, HST, INCY, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (30 rows, but overlapping windows are not independent; 83 rows across all horizons), newest 2026-10-08 |
| Context signals | Context signals: 7 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#18: a period behind its filings); ES (#19: a period behind its filings) |
| Context coverage | options usable 492/501 (9 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Things that needed attention

- GATE 1 tests: FAIL (FAILED tests/test_v2_experiments.py::test_the_baseline_weighting_is_hashed_separately_from_the_transformed_one FAILED tests/test_v2_experiments.py::test_the_report_says_what_its_rows_ran_on_and_flags_a_mixture 5 failed, 2659 passed, 2 skipped, 316 warnings in 279.01s (0:04:39))
- GATE 4 clean tree: FAIL - uncommitted changes remain
- M research/experiments/ledger.jsonl
- ?? research/measurements/2026-10-18-provenance-rerun-effect.py
- SHIP GATES FAILED: tests, clean-tree. Not merging.
- Committed the uncommitted work onto nightly/2026-10-18 so it is not lost. Recover it from there.
- Work pushed to nightly/2026-10-18 for inspection. origin/main verified clean (reverted if the earlier push had reached it).
- Run finished with failing gates - see above.

## What changed in the repo

- `ad2d422 brief: code session 2026-10-17-midday`
- `68fb679 log: the 2026-10-17 midday session - the macro backdrop note and three corrected page sentences`
- `6924c52 docs: context-layer item 4's macro note is done; the owner queue's progress line`
- `1970c17 look at the corrected backdrop at 1440 and 375; ignore the FRED download cache`
- `b4a56c2 backdrop: three sentences corrected to what their sources' own records show`
- `13321bf research: the market backdrop's five readings against their sources (context-layer item 4)`
- `9a60175 measure: the 10y-3m spread's own record against the curve sentence`
- `9c45e1c measure: the Sahm rule's own record against the sentence the page prints`
- `2698fdb brief: code session 2026-10-17-weekend`
- `ccd68d3 log: the 2026-10-17 weekend session - gate fix, method version 2, and the rerun's result`
- `1289e7d engine: rerun all 28 experiments and the evaluation on method version 2; no verdict changed`
- `d5ec2ad docs: a method change reruns the engine, like a data fix (rule in CLAUDE.md)`
- `5761f47 engine: comparisons are measured on the stocks both rankings share (method version 2)`
- `2268c3e test: the option-cache fixtures name a real market session, not 'today'`

## The session's own account

> 2026-10-17 (13:00 midday session) - QUEUE. The macro backdrop's research note, and the three page sentences it found were wrong about their own sources
> 
> **Health (rule 8, all five):** last code session ran? **yes** - this morning's own
> `logs/nightly-2026-10-17_060001-weekend.log` ends "Run complete: shipped to main", tagged
> `good/2026-10-17`. Data loop published? **yes** - `logs/datarun-2026-10-16_020001.log` ends
> "Data loop complete"; Friday was the last due run, the loop is Mon-Fri and today is Saturday.
> Evidence base at `1m`: **30 rows, newest 2026-09-15 (32 days ago, bound 40), 4 effective** -
> inside the bound, steady state. Priority 0: unchanged, `allow_auto_apply` **false**, 4 of 8
> effective observations. Top open roadmap item: **0.11, the context layer** (opened 2026-10-08,
> **age 9 days**) - **taken today**, see below.
> **Tests:** before **2638 passed, 2 skipped / 0 failed**, after **2651 passed, 2 skipped / 0
> failed** (+13). Dry run exits 0; `index.html` 611,099 bytes; both payloads parse (`node --check`
> 0 on `dashboard_data.js` and `dashboard_context.js`); tree clean.
> `scripts/rehearse_publish.py` **PASSED** (1,003 claims-gate tests), run because the diff touches
> `claims.py`.
> **Owner queue / rotation:** no proposal is waiting (`write_brief.open_proposals()` returns `[]`).
> This morning's session took the research engine's item 8(a) and left **0.11, the context layer**,
> as the thing to take next unless a proposal opened or a gate went red. Neither did, so I took it:
> `plan/context-layer.md` queue item 4, **the macro backdrop's research note**. That also ends the
> deferral streak the morning entry counted at four.
> 
> ### Did
> 
> **One item, end to end: `plan/context-layer.md` item 4's macro-backdrop note, plus the three page
> defects reading the sources turned up.**
> 
> **1. The note.** `research/2026-10-17-macro-backdrop.md` covers all five readings the panel prints
> (yield curve, credit spreads, volatility, inflation, labour market) and all three factor notes
> beside them, from the primary sources. Every DOI was resolved through Crossref **before** it was
> written down (`research/measurements/2026-10-17-fetch-macro-sources.py --dois`, 7 of 7 verified),
> which is the 2026-10-16 audit's finding acted on - the three most recent notes before this one
> carried none. Three papers were downloaded and read from the PDF; one (Estrella & Mishkin 1998)
> could not be, because the NY Fed's scan has no text layer, and the note says so rather than
> pretending otherwise.
> 
> **2. Three sentences on the live page are wrong about their own sources' records.** All three are
> display-only context text; **no score, rank or number moves.** They share one failure mode, which
> is the finding worth carrying: *each attached a real citation to a stronger or different statement
> than the source makes.*
> 
> - **(a) The labour-market sentence.** It said the Sahm indicator "at +0.50pp or more **has marked
>   the start** of every US recession since 1970 (Sahm 2019)". Measured on FRED's own real-time
>   series against the NBER indicator
>   (`research/measurements/2026-10-17-sahm-rule-record.py`): the crossing came **after** the
>   recession had begun in **6 of 8**, mean **+0.9 months** against the NBER peak. That is what
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

