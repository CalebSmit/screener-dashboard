# Morning Brief - Friday 06 November 2026, 02:20

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-05 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, MPC, EXPE, MU, SNDK |
| Ranking's lean on growth | +0.060 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (44 rows, but overlapping windows are not independent; 112 rows across all horizons), newest 1m run date 2026-10-05 (32 days ago, bound 40) |
| Context signals | 19 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 357/358 within 0.5%; operating cash flow 329/346 within 0.5%; net income 326/357 within 0.5%; market value 446/485 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); HST (#6: net income 1.46x after-tax operating income); DXCM (#7: operating cash flow 36% off the filing); PSX (#8: net income 1.28x after-tax operating income); TPR (#12: equity 5.2% of assets); CF (#15: net income 16% off the filing); WDC (#19: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 628 s, `fetch.retry` 156 s, `context.sec_insider` 103 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (27 days) |
| Context coverage | options usable 476/499 (23 quotes-closed) - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 3 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 10 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_context_layer_claims.py::test_the_page_says_when_each_kind_of_quote_was_taken
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.

## What changed in the repo

- `50b535f Read the flag-key scroll where it ends, not 700 ms into it`
- `9db2e4f Compare says what sets each stock apart, under its composite`
- `ffe51b5 Say in words what sets each stock apart: leaders, sheets and holdings`
- `7cf7b1f Pin the momentum rule's wiring to the one market fetch it now shares`
- `e6aab5f Rehearse the sell-off, and put the market beside the ranking in What Changed`
- `b9097bd brief: code session 2026-11-05-midday`
- `8f5f791 Log the 13:00 session: the red baseline, and the defect behind it`
- `18eeb13 The flake found: the sheet pre-warm was overwriting panels a test had just rendered`
- `2ca17f1 A red baseline keeps its transcript, like the gates do`
- `0081d38 Leader caption: the date never breaks across lines`
- `d5e85b4 Leaders show how long they have led; holdings show where each name has been`
- `0a00f0e The top-stock rank test opens the stock ranked first, not EXPE by name`
- `061953d Polish: nothing cut short, no empty slot, and no false 'ahead of 100%'`
- `d25bdb1 Show the story each section tells: what moved the scores, period by period, sector standouts`
- `a0500df Market backdrop: embed the band table indented, so the page carries no template brace pair`

## The session's own account

> 2026-11-06 (owner-run, 00:00) - Compare says what sets each stock apart
> 
> **Health (rule 8):** unchanged from 2026-11-05's entries; pushed well before the 02:00 data run.
> 
> **Owner:** *"Do that now. Do everything you can before the 2 am run."* - the Compare view, named as the
> next place for the Stands out / Trails strip in the previous entry.
> 
> **Built:** Compare (Side by side) gains a **What sets it apart** row directly under Composite, one cell
> per stock, the same `traitsHtml` strip as the leader cards, so two stocks are told apart by kind before
> the eight rows of numbers (VLO: stronger price trend, lower price for its profits; AAPL: calmer price,
> trails as a larger company with weakening analyst views; WMB: nothing at 70 or more). Chips that wrap
> on a phone get a 10px corner rather than a pill. Display only; payload byte-identical.
> 
> **Checks:** `tests/test_stock_traits.py::test_the_compare_view_carries_the_words_for_each_stock`
> (added to claim `dashboard.traits`): the first, median and last stocks opened together, each column's
> words against its own published scores and in the view's rank order. Seen at 1440 and 375 with three
> stocks; no horizontal scroll, no page errors.
> 
> **Caught by gate 5, and fixed before 02:00: a flake the previous push introduced.** The rehearsal failed
> `tests/test_club_readiness.py::test_a_phone_reader_reaches_the_flag_definitions_from_the_tile` - on a
> phone, "What a flag means" scrolls to the flag definitions and the test read their position at a fixed
> 700 ms. Measured: **0 of 10** failures on the page before last night's Stands out / Trails change, about
> **1 in 3** after it (1 of 6 on the live `main`, 2 of 5 here). Cause, measured frame by frame: the leader
> cards grew a line of words, the phone's scroll to the ranking got ~400px longer, and at 700 ms the smooth
> scroll was still in flight (y = 703-806 against a 812 viewport); it landed at **y = 352 every time by
> 1.2 s**. The reader was never affected; the test's premise was. **This module is in the 02:00 publish
> gate** (claim `dashboard.score_key`), so left alone it could have discarded tonight's data run on a bad
> roll. Fix: read the position once the page has stopped scrolling (ten unchanged frames) - the same final
> assertion, the premise of settled row paint->red. 0 of 15 failures after. It was the only test in the
> suite reading a position after a fixed wait.

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

