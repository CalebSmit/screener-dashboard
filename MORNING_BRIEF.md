# Morning Brief - Thursday 29 October 2026, 06:49

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-29T02:00:05.052320 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, VLO, MU, APA, HST |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (39 rows, but overlapping windows are not independent; 103 rows across all horizons), newest 2026-10-21 |
| Context signals | Context signals: 15 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 396/397 within 0.5%; operating cash flow 381/385 within 0.5%; net income 373/393 within 0.5%; market value 443/484 within 5%; filings cached 6 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#5: net income 1.46x after-tax operating income); CF (#17: net income 16% off the filing); WDC (#22: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 646 s, `fetch.retry` 140 s, `snapshot` 129 s; **45 tickers rate-limited** |
| Context coverage | options usable 489/498 (9 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `5d1bf72 log: the fifth mutation pass - three sentences the page publishes that nothing could see change`
- `f9f0fca OWNER_FOCUS: the fifth mutation pass - three unregistered sentences, and what "caught" means`
- `93fad41 plan: rows 106-117, and the measurement that the three sentences were blind to the whole suite`
- `036dd1e wip: register the research page's three "prove an edge" sentences, with checks on both sides`
- `439a8b9 brief: data run 2026-10-29`
- `3170a19 data: screener run 2026-10-29 - 498 scored, top: EXPE VLO MU APA HST`
- `bee8f89 brief: code session 2026-10-28-midday`
- `0d20410 log: the fourth mutation pass - the research surface, and a drift tripwire that is green in the state the change ships in`
- `9f46655 plan/OWNER_FOCUS: the fourth mutation pass - the research surface's statistics and the last three residuals`
- `6c815b3 wip: pin the Sahm rule's prior-12 window and its 0.50 bar (audit rows 102-103)`
- `eaa27eb wip: pin what 'cleared its bar' means for the alternatives count (audit row 97)`
- `ec11731 wip: pin the research page's interval rule and the detectable figure against the claims they state`
- `c87757f log: the third casualty of the refused run, and the gates`
- `639e1d6 test_investor_profiles: pair the payload with the run it names, not the newest folder`
- `ac294b1 The refused 2026-10-28 data run: a signed 'case against' trend line, a discard that leaves no files, a stash judged by the tree; the universe-expansion plan`

## The session's own account

> 2026-10-29 (06:00 session) - BUILD. Three sentences the research page publishes every run, that nothing in the project could see change
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-28 13:00 ended "shipped to
> main", tag `good/2026-10-28`. Data loop published? **yes** - 02:00 today ended "Data loop complete"
> (498 scored, 45 tickers rate-limited on the retry pass, 18 min). Evidence base at `1m` = **39 rows,
> newest 2026-09-28 (31 days ago, bound 40), 5 effective**. Priority 0: `allow_auto_apply` **false**,
> unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 17 days**) -
> not advanced, and not drifting: the lab's day is a Friday, 2026-10-30 is an even ISO week so the
> runner swaps in the retrospective, and the next lab Friday is **2026-11-06**.
> **Tests:** before 3,506 passed / 1 skipped, after **3,510 passed / 1 skipped**.
> **Owner queue / rotation:** took `OWNER_FOCUS.md` item 3 of the 2026-10-20 block, "Test the
> tests" - the fifth mutation pass, and specifically the item the 2026-10-28 session deferred with
> the work specified (`plan/claims-mutation-audit.md`, next-list item 3). No proposal was waiting
> (`write_brief.open_proposals()` returns `[]`). The stash `auto-rescue 2026-10-28_060001` is
> **empty** (`git stash show -p` prints nothing, `--stat` nothing): the 2026-10-28 morning session's
> fix cleaned the refused run's files through `Undo-RunArtifacts`, so there was nothing left for the
> stash to hold - and its own entry says not to pop it. Left in place, nothing lost. The top open
> item in `OWNER_FOCUS.md` (universe expansion) is a Monday research item by its own Next line;
> nothing else deferred. Thursday's focus is BUILD and the week's research justified no methodology
> change (2026-10-26's proposal was refused on 10-26 and again confirmed 10-27), so by the rotation's
> own rule this is the top open owner item instead of an invented change.
> 
> ### Did
> 
> **The research page's "How long would it take to prove an edge?" section states three numbers, and
> `claims.py` registered none of them.** They are registered now - `research.ic_swing`,
> `research.months_to_prove`, `research.detectable_difference` - each with checks on both sides of
> its figure. Rows 106-117 of `plan/claims-mutation-audit.md`. No weight, metric, threshold or
> published sentence moved; the diff is the register, four tests, one test strengthened, and the
> write-up.
> 
> The three sentences:
> 
> - **"The ranking's monthly correlation swings by about 0.12 from one month to the next"** - the
>   standard *deviation* of the baseline's monthly rank correlation over 2020-2026, recovered from
>   that ledger row's standard error as `se x sqrt(n)`.
> - **"31 / 59 / 164 months of record"** to show a correlation of 0.05 at the usual 5% level, the
>   same at a t of 3, and 0.03 at a t of 3 - `((z + 0.84) x the swing / the correlation)^2`, the
>   record needed to see it 80% of the time.
> - **"The smallest difference the test could reliably see was about 0.037 over 2012-2019 and 0.062
>   over the held-out months"** - the engine's `detectable` applied to the **median** alternative's
>   standard error *in each period*, over that period's own month count, corrected for the number of
>   registered comparisons.
> 
> **12 mutations, 12 caught.** That is the expected result for tests written the same day and it is
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

