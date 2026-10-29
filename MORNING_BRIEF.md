# Morning Brief - Thursday 29 October 2026, 14:03

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

- `f341a9d log + OWNER_FOCUS: the sixth mutation pass - the sweep, and 12 open rows it enumerated`
- `98c9e40 plan: rows 118-126, and the measurement that a brand-new computed sentence is invisible`
- `28f7ba2 claims: sweep research.html for computed-number sentences; register the overlap paragraph`
- `f73803c log: 2026-10-29 owner-run - overnight runs healthy, claims gate log line`
- `8386997 claims gate: log pytest's result line, not the last line printed`
- `a0cacb2 brief: code session 2026-10-29`
- `5d1bf72 log: the fifth mutation pass - three sentences the page publishes that nothing could see change`
- `f9f0fca OWNER_FOCUS: the fifth mutation pass - three unregistered sentences, and what "caught" means`
- `93fad41 plan: rows 106-117, and the measurement that the three sentences were blind to the whole suite`
- `036dd1e wip: register the research page's three "prove an edge" sentences, with checks on both sides`
- `439a8b9 brief: data run 2026-10-29`
- `3170a19 data: screener run 2026-10-29 - 498 scored, top: EXPE VLO MU APA HST`
- `bee8f89 brief: code session 2026-10-28-midday`
- `0d20410 log: the fourth mutation pass - the research surface, and a drift tripwire that is green in the state the change ships in`
- `9f46655 plan/OWNER_FOCUS: the fourth mutation pass - the research surface's statistics and the last three residuals`

## The session's own account

> 2026-10-29 (13:00 midday session) - QUEUE. The sentences nothing was watching: a sweep for computed numbers on the research page that no claim names
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-29 06:00 shipped to main
> (tag `good/2026-10-29`), and an owner-run session merged a claims-gate log fix after it. Data loop
> published? **yes** - 02:00 today ended "Data loop complete", HEALTH: PASS, run `a08c2741a3ca`, 498
> scored. Evidence base at `1m` = **39 rows, newest 2026-09-28 (31 days ago, bound 40), 5
> effective**. Priority 0: `allow_auto_apply` **false**, unchanged. Top open roadmap item: **0.13,
> the Factor Lab** (opened 2026-10-12, **age 17 days**) - not advanced, and not drifting: the lab's
> day is a Friday, 2026-10-30 is an even ISO week so the runner swaps in the retrospective, and the
> next lab Friday is **2026-11-06**.
> **Tests:** before 3,515 passed / 1 skipped, after **3,523 passed / 1 skipped**.
> **Owner queue / rotation:** the third session today. Carried on from the 06:00 entry's **Next**,
> which specified this work: `OWNER_FOCUS.md` item 3 of the 2026-10-20 block ("Test the tests") and
> `plan/claims-mutation-audit.md`'s next-list item 3. No proposal was waiting
> (`write_brief.open_proposals()` returns `[]`). The stash `auto-rescue 2026-10-28_060001` is still
> empty and still untouched (`git stash show --stat` prints nothing) - left in place, as its own
> entry says. No owner-session marker. The top open item in `OWNER_FOCUS.md` (universe expansion) is
> a Monday research item by its own Next line; nothing else deferred.
> 
> ### Did
> 
> **`claims.py` registers the sentences someone thought to write down. This sweeps for the ones
> nobody did.** Rows 118-126 of `plan/claims-mutation-audit.md`. No production code, no weight, no
> threshold and no published sentence changed; the diff is one new section in
> `tests/test_claims_register.py`, five tests, one claim and the write-ups.
> 
> **The argument for doing it.** This morning's session registered three computational sentences on
> the research page and measured that each was invisible to the whole suite. It found them by
> *reading one section of one page*. That is attention, not a method - and the mutation audit this
> item has been running since 2026-10-20 **cannot reach the class at all**, because mutation can
> only mutate what the register names. The gap is the one `claims.py`'s own docstring describes:
> "nothing enumerated the *set* of computational claims the site makes".
> 
> **So the register now sweeps a generated page the way it has always swept `stock_summary.py`.**
> `test_every_sentence_builder_is_registered` works because a new `_sentence_*` cannot ship
> unregistered; no generated page has a naming convention to hang that on. `computed_sentences()`
> supplies one structurally: it walks `research_page.py`'s f-strings, reconstructs each as the text
> the page prints, splits it into sentences, and keeps those that interpolate a number **this repo
> computed** - a numeric format spec, arithmetic, or a call to one of the generator's own statistics
> helpers. **32 sentences.** Each must carry a verdict - `claim:`, `checked:`, `reported:` or
> `open:` - and a new or reworded one fails the suite until it gets one.
> 
> **Two rules in it that are worth more than the code.** `checked:` and `reported:` mean *every*
> computed figure in the sentence is pinned by the named test; **one unpinned figure makes the whole
> row `open:`**, because a row that reads as coverage it does not have is the exact failure the
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

