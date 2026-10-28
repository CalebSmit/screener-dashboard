# Morning Brief - Wednesday 28 October 2026, 14:03

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-27T02:00:05.759618 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (38 rows, but overlapping windows are not independent; 99 rows across all horizons), newest 2026-10-19 |
| Context signals | Context signals: 14 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 412/413 within 0.5%; operating cash flow 397/401 within 0.5%; net income 387/409 within 0.5%; market value 443/484 within 5%; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#17: net income 16% off the filing); WDC (#18: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 651 s, `fetch.retry` 144 s, `context.sec_insider` 98 s; **44 tickers rate-limited** |
| Context coverage | options usable 485/498 (13 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.

## What changed in the repo

- `0d20410 log: the fourth mutation pass - the research surface, and a drift tripwire that is green in the state the change ships in`
- `9f46655 plan/OWNER_FOCUS: the fourth mutation pass - the research surface's statistics and the last three residuals`
- `6c815b3 wip: pin the Sahm rule's prior-12 window and its 0.50 bar (audit rows 102-103)`
- `eaa27eb wip: pin what 'cleared its bar' means for the alternatives count (audit row 97)`
- `ec11731 wip: pin the research page's interval rule and the detectable figure against the claims they state`
- `c87757f log: the third casualty of the refused run, and the gates`
- `639e1d6 test_investor_profiles: pair the payload with the run it names, not the newest folder`
- `ac294b1 The refused 2026-10-28 data run: a signed 'case against' trend line, a discard that leaves no files, a stash judged by the tree; the universe-expansion plan`
- `a63a914 brief: code session 2026-10-28`
- `9f9fda7 brief: data run 2026-10-28`
- `b7ff1f4 log: the seventh owner pass - the repo lock's two races fixed, flaky tests, sheet, shell, lab, bot`
- `ab2a172 test_sheet_round_three: the tab tests wait for the context payload and for the jump to land, not for a fixed position (flaked twice in five parallel runs)`
- `763e791 repo lock: publish the lock whole and reclaim only under a token - both measured races fixed`
- `a22a01b page shell, round three: failure states, freshness, true run time, Inter stand-in, print, 404, home screen (payload byte-identical)`
- `8b9b0fc research: flaky tests - three full suites green on the final tree`

## The session's own account

> 2026-10-28 (13:00 midday session) - QUEUE. The research page's statistics mutation-tested: six of its eight unchecked claims were seen only by a tripwire that is green in the state the change ships in
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-27 13:00 shipped
> (`good/2026-10-27-1338`); today's 06:00 stood down and the owner-run morning session fixed both
> causes (its entry above). Data loop published? **no, correctly** - 02:00 today was HEALTH: PASS
> and then refused by the published-claims gate on a real false sentence; the live site keeps
> 2026-10-27's figures and the fix is committed, so tonight's run publishes. Evidence base at `1m`
> = **38 rows, newest 2026-09-25 (33 days ago, bound 40), 4 effective**. Priority 0:
> `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 16 days**) - not advanced, and not drifting: the lab's day is a Friday, 10-30 is
> an even ISO week so the runner swaps in the retrospective, and the next lab Friday is **2026-11-06**.
> **Tests:** before 3,500 passed / 1 skipped, after **3,506 passed / 1 skipped**.
> **Owner queue / rotation:** took `OWNER_FOCUS.md` item 3 of the 2026-10-20 block, "Test the
> tests" - the fourth mutation pass, `plan/claims-mutation-audit.md`'s queue items 4 then 1-3. No
> proposal was waiting (`write_brief.open_proposals()` returns `[]`). The top open item in that file
> (universe expansion) is a Monday research item by its own Next line; nothing else deferred.
> 
> ### Did
> 
> **17 mutations across 6 claims: 11 caught, 6 gaps. Every gap now fails a registered check.**
> Rows 89-105 of `plan/claims-mutation-audit.md`, which closes its queue items 1-4. No production
> code changed and no published sentence moved; the whole diff is tests, the register, and the
> write-up.
> 
> **The research surface was the one no pass had touched, and it had a kind of unchecked claim this
> audit had not met.** `research.html`'s sentences quote intervals, t-statistics and a
> "smallest difference the test could reliably detect" figure. Ten mutations of that arithmetic - the
> 80%-power quantile, the power term itself, the t-distribution for the bar, the interval boundary at
> zero, the intervals' confidence level in both places that builds them, the record the pooled
> estimate covers, the multiple-testing bar, the alternatives count, and the rule deciding what
> "cleared its bar" means - produced **two** caught and **eight** gaps.
> 
> **The thing worth carrying forward: for six of those eight, the only test that failed was a drift
> tripwire** - `test_the_committed_page_is_what_the_generator_makes_from_the_ledger`, or
> `test_the_methodology_page_quotes_the_engine_s_own_summary`, which asserts the committed
> `summary.json` equals a recomputation. Both of those fail under *any* numeric change and assert
> nothing about the arithmetic - **and both are green in exactly the state the change ships in.**
> That is measured rather than argued: every mutation was re-run with `python v2_experiments.py
> report` in between, which is how `research.html` and `summary.json` are actually published, and
> **all seven gaps stayed green.** It is CLAUDE.md rule 10 one level up ("assert claims against the
> generator's output, not the committed file"): a check that compares a generated artifact with its
> own regeneration cannot see a change to the generator.
> 
> **Underneath that, two claims were checked by a test that recomputes the rule out of the code and
> then asks the page to agree with it.** `research.plain_answer`'s only check opened with
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

