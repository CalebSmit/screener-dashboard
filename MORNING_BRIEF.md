# Morning Brief - Monday 09 November 2026, 06:56

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-06 18:34 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, MPC, DXCM |
| Ranking's lean on growth | +0.052 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (34 days ago, bound 40) |
| Context signals | 20 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 331/332 within 0.5%; operating cash flow 305/320 within 0.5%; net income 302/332 within 0.5%; market value 0/0 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#5: net income 1.46x after-tax operating income); DXCM (#7: operating cash flow 36% off the filing); PSX (#9: net income 1.28x after-tax operating income); CF (#10: net income 16% off the filing); TPR (#11: equity 5.2% of assets); WDC (#20: net income 2.13x after-tax operating income); ES (#23: a period behind its filings) |
| Data run timing | **17 min** total; slowest `fetch.tickers` 628 s, `fetch.retry` 153 s, `context.sec_insider` 80 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (24 days) |
| Context coverage | options usable 415/499 - insider from SEC 499/499 - earnings variability 439 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **which of options 1-3 (or none) to build for a stock whose implied share count has moved more than the threshold since the last accepted run while the earnings behind its multiples predate the move** - open today - `research/2026-11-09-share-counts-and-corporate-actions.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (**today**) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 7 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_share_classes_page.py::test_the_price_gap_is_the_only_valuation_difference
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.
- full output: C:\Users\smitc\OneDrive\Documents\Screener\logs\nightly-2026-11-09_060001.baseline.txt

## What changed in the repo

- `f14b4f1 log: 2026-11-09 - the refused data run, both faults fixed, and the share-count note`
- `2c62cfc research: the share count behind a market value, and what a screen should do when it moves`
- `8af1404 fix: the lab's card bound was read before the page settled, and had been breached`
- `9817403 fix: the share-class gate asserted a discarded intermediate, and discarded a night`
- `3a608c4 brief: data run 2026-11-09`
- `51d75a7 brief: code session 2026-11-08-midday`
- `78900f5 log: five full parallel runs on identical code, three one-off browser flakes`
- `77cc502 log: the shard's second slice, and the half of it that needed deleting`
- `13c18f8 fix: two modules that read `financials` before the loader merged it`
- `c28af48 shard: the detail file learns to carry numbers, and `financials` moves into it`
- `62bc52b measure: 7 of 7 mutations refused, and the one that got through first was right to`
- `e299b11 unify: the peer table's rows all go through one function`
- `fc1865f brief: code session 2026-11-08-weekend`
- `a1bc585 log: the detail split, and two measurements that corrected the reasoning behind it`
- `ae07726 gate: a runner must check each payload by its assignment, not only name the file`

## The session's own account

> 2026-11-09 - RESEARCH (deferred to the stalled data loop first, then written)
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-08_130001-midday.log`
> (completed). Data loop published? **NO** - `logs/datarun-2026-11-09_020001.log`: HEALTH: PASS, then
> the published-claims gate failed on 2 tests and the run was discarded, so the live site still serves
> the page of **2026-11-06 18:34**. Evidence base at `1m`: **45 rows, newest 2026-10-06 (34 days ago,
> bound 40), 5 effective of the 8 needed**. Priority 0: `allow_auto_apply` **false**. Top open roadmap
> item: **0.13, The Factor Lab** (age **28 days**); also open 0.10 (33d), 0.14 (26d).
> **Tests:** before **1 failed, 4,231 passed, 1 skipped** (a RED baseline I did not cause); after
> **4,243 passed, 1 skipped**. Gates 2, 3, 4 pass; gate 5 (`rehearse_publish.py`) PASSED.
> **Owner queue / rotation:** `OWNER_FOCUS.md` item 1 is "decide the OPEN proposals" and there were
> **none** at session start (`write_brief.open_proposals()` returned `[]`), so nothing was deferred
> there. The data loop outranked the Monday focus (CLAUDE.md: a stalled loop is the highest priority
> work available); the focus was then served by writing the diagnosis up properly. **Deferred: 0.13,
> 0.10 and 0.14, for the stalled loop** - and 0.13 has now been deferred by three consecutive
> sessions, so the deferral-streak rule binds on the next session carrying no owner item.
> 
> ### Did
> 
> **1. Diagnosed and fixed the two gate failures that discarded the 02:00 run.** Both were tests
> whose premise had broken, not production defects - but a stalled publish is a stalled publish.
> 
> **(a) `test_share_classes_page.py::test_the_price_gap_is_the_only_valuation_difference`.** It
> asserted FOXA's and FOX's `earnings_yield` differ by exactly the ratio of their prices, and read
> 0.6650 against 0.8929. This one was **also red in the baseline**, because it reads the newest `runs/`
> folder. Per CLAUDE.md's "data->red" rule I asked first whether the claim was false. Established from
> the filings, not inferred: **Fox completed its acquisition of Roku on 2026-11-06** (8-K of that date,
> "the Closing Date"; S-8 and S-8 POS the same day), the 10-Q filed 2026-11-02 puts Class A at
> 201,552,650 and Class B at 220,248,764, and the source added the **144,500,000** newly issued Class A
> shares to FOXA (market value on 566,301,414 shares) while leaving **FOX's market value on the
> pre-merger 421,801,414**. Two records, two share counts, so the price ratio stopped being the
> market-value ratio.
> **The claim was not false.** `02_outliers_flagged` keeps every fetched line and the second line is
> dropped before the percentiles, so FOX's `earnings_yield` is never scored, ranked or published, and
> the page's sentence is about the two prices, both current. The test was asserting a property of a
> **discarded intermediate**. Rewritten to assert the claim and its premise: `earnings_yield`'s ratio
> must be the market-value gap and `dividend_yield`'s the price gap (both hold on either basis - they
> are what says the filings behind the two lines are one, and only one metric was checked before);
> those two are one number only while both records count the same shares, which is now tested rather
> than assumed; and the fact that makes a divergence harmless is asserted rather than tolerated -
> **exactly one line of the company is ranked**. 7 new tests, 27 in the module, every branch exercised
> on chosen inputs because the run-reading test can only reach the branch today's data is in. The six
> audit tests joined the claim's `checked_by`, so the publish gate runs them.
> 
> **(b) `test_lab_reading_and_access.py::test_on_a_phone_the_overview_scans_as_a_list_...`.** Passed
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

