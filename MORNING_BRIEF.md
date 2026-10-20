# Morning Brief - Tuesday 20 October 2026, 02:25

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **stopped deliberately** - last ran today |
| Dashboard data from | 2026-10-20T02:00:03.858333 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, APA, INCY, VLO |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (33 rows, but overlapping windows are not independent; 88 rows across all horizons), newest 2026-10-12 |
| Context signals | Context signals: 9 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#2: net income 1.46x after-tax operating income); CF (#8: net income 16% off the filing); MU (#15: a period behind its filings) |
| Context coverage | options usable 489/501 (12 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **for bank-like stocks, `pb_ratio = marketCap / commonEquity` (new raw field `commonEquity` = the quarterly balance sheet's "Common Stock Equity" at `_stmt_date_balance_sheet`; NaN unless both are positive; no fallback to `priceToBook`, `bookValue` or stockholders' equity), shipped with `metric_lineage` LINEAGE / RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, a `scoring_schema` bump, the generated methodology text and a changelog entry (section 8). Measured on the 2026-10-19 run: top 25 unchanged, bank-like median move 2.5 places, BRK-B 60th -> 121st (its source P/B of 0.0010 is book per Class A share against a Class B price)** - open 1 day - `research/2026-10-19-bank-price-to-book.md`
- **for bank-like stocks, `roe = netIncome / ((totalEquity + totalEquity_prior) / 2)` and `roa = netIncome / ((totalAssets + totalAssets_prior) / 2)` (NaN unless both balances are positive; no fallback to Yahoo's `returnOnEquity` / `returnOnAssets` or to an ending balance), shipped with `metric_lineage` LINEAGE / RECOMPUTE / EQUATIONS, `pit_record.FIELD_CLASSES`, `scoring_schema` 12, the generated methodology text and a changelog entry (section 7)** - open 1 day - `research/2026-10-19-bank-roe-roa.md`
- **score each company once: in `run_screener.py`, after `compute_metrics` and before `compute_sector_percentiles`, group universe rows by SEC CIK (falling back to identical `longName`), keep the line with the higher `avg_daily_dollar_volume` (S&P DJI's Designated Listing / FTSE Russell's pricing vehicle; on 2026-10-19 GOOGL, FOXA, NWSA) and remove the other lines from percentiles, scoring and ranking; publish them as unranked entries pointing to the company's ranked line with their own price and the price gap; shipped with a no-two-rows-per-CIK test, the page sentence and its claim, the generated methodology sentence and a changelog entry (section 9). Measured on the 2026-10-19 run: top 25 unchanged; FOXA 47 -> 28, GOOGL 226 -> 197, NWSA 61 -> 52; other Communication Services stocks +1.56 composite points on average** - open 1 day - `research/2026-10-19-share-classes.md`

## Things that needed attention

- GATE 1 tests: FAIL (FAILED tests/test_v2_experiments.py::test_every_ledger_row_the_report_reads_names_its_code_and_its_baseline FAILED tests/test_v2_experiments.py::test_the_report_says_what_its_rows_ran_on_and_flags_a_mixture 4 failed, 2662 passed, 1 skipped, 316 warnings in 279.04s (0:04:39))
- GATE 4 clean tree: FAIL - uncommitted changes remain
- M research.html
- M research/experiments/RESULTS.md
- M research/experiments/ledger.jsonl
- M research/experiments/summary.json
- SHIP GATES FAILED: tests, clean-tree. Not merging.
- Committed the uncommitted work onto nightly/2026-10-19 so it is not lost. Recover it from there.
- Work pushed to nightly/2026-10-19 for inspection. origin/main verified clean (reverted if the earlier push had reached it).
- Run finished with failing gates - see above.

## What changed in the repo

- `ce4f70e data: screener run 2026-10-20 - 501 scored, top: EXPE HST APA INCY VLO`
- `dfc7ad0 engine: rerun complete on the final 2026-10-19 code (30 of 30, published figures unchanged); evening log addendum`
- `d09b38b engine: rerun on the final 2026-10-19 code (chunk 7)`
- `a67c35c engine: rerun on the final 2026-10-19 code (chunk 6)`
- `a90fc19 engine: rerun on the final 2026-10-19 code (chunk 5)`
- `744da92 engine: rerun on the final 2026-10-19 code (chunk 4)`
- `d45cdf7 engine: rerun on the final 2026-10-19 code (chunk 3)`
- `ab138a8 engine: rerun on the final 2026-10-19 code (chunk 2)`
- `9d0bdb9 engine: rerun on the final 2026-10-19 code (chunk 1: baseline, equal-categories, only-valuation)`
- `59579b1 log: 2026-10-19 owner-run rounds 2 and 3`
- `4918278 notes and plans: the rebuild and rerun after the pit_record fixes - what moved, no verdict changed`
- `4c610c1 measurement: no verdict changed in the rerun on the rebuilt months; largest IC-difference shift 0.00065`
- `20038ee research engine: all 29 experiments and the bot evaluation rerun on the rebuilt months; report`
- `cd09907 research engine: rerun on the rebuilt months (27 of 30)`
- `0cc7b0d research engine: rerun on the rebuilt months (24 of 30)`

## The session's own account

> 2026-10-19 (owner-run, day) - rounds 2 and 3 of the agent pass: 13 branches merged across the screener and the bot
> 
> **Health (rule 8):** last code session ran? **it ran but failed its gates** - `logs/nightly-2026-10-19_060001.log`
> ended "Run finished with failing gates" (the session ended its reply while the engine rerun ran in the background;
> diagnosed and fixed below; its work was finished and merged here). Data loop published? **yes** -
> `logs/datarun-2026-10-19_020001.log` "Data loop complete", HEALTH: PASS. Evidence base at `1m` = **31 rows, newest
> 2026-09-16 (33 days, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**, 4 of 8. Top open roadmap item:
> **0.11, the context layer** (opened 2026-10-08, **age 11 days**) - worked today: item 4's track-record note and item
> 6's turnover/cost are done.
> 
> **Owner:** *"i want you to get back to work and keep finding and making even more improvements. Send out a bunch of
> agents to do so."* The 13:00 session stood down for the owner marker; the work it would have done (finishing the
> 06:00 branch) was done here.
> 
> ### Merged (screener main `good/2026-10-19-owner-1` to `-4`; full suite 2,815 at the last)
> - **The 06:00 session finished**: the Growth-composition note, its pre-registered test (prediction wrong, no weight
>   changed), and the engine rerun completed on one code version (30 runs, no result changed).
> - **Why sessions ended with uncommitted work** (`research/2026-10-19-session-ended-on-a-background-job.md`): not a
>   usage limit. Under `claude --print` a reply without a tool call ends the session; both failing sessions started the
>   ~47-minute rerun in the background and ended their reply. Fixed: a prompt rule, a stated deadline, foreground
>   waits (`scripts/wait_for.py`), a resumable rerun (`scripts/rerun_experiments.py`), and one continuation of the
>   same session before the gates (never after them; no gate weakened). **Not yet exercised in production.**
> - **Factor Lab experiment 14, asset growth** (Cooper, Gulen & Schill 2008): 13 of 24 within; the premium reproduces
>   in the paper's years and is about zero among large companies since 2003.
> - **Top-of-ranking data audit**: none of the top 40's inputs wrong; one backtest-reader defect fixed (`pit_record`).
> - **Methodology claims audit**: 16 false sentences on the public methodology page corrected (ROIC's formula, the cache,
>   the coverage filter, the workbook sheets, a superseded EPS description, among others); the page now reads those
>   rules from the code; 17 claims registered.
> - **Track record**: a late run's entry price fixed, nine sentences corrected, the record's power stated (0.8 SE),
>   turnover and break-even cost shown; snapshots now record their run's start time.
> - **Fetch**: the year-ago quarter and balance sheet chosen by date, not position (BRK-B, FERG, SKYD; changelog,
>   scoring_schema 12).
> - **Club readiness, two rounds** (`plan/club-readiness.md`): a key to the columns and flags, plain answers for a stock
>   not in the ranking, a way back on phones, the Size/Risk/Investment sentence explained, glosses for options and
>   insider terms, readable score cards.
> - **Research notes with OPEN proposals** for a later session to decide: bank ROE/ROA computed from filings; bank
>   price-to-book as market value over common equity (**it also fixes a live BRK-B defect: P/B 0.001 against about
>   1.52, because the source's book value is per Class A share**); share classes scored once per company.
> 
> ### Merged (screener-bot main; 324 tests, green on GitHub)
> Corporate actions in the sim (renames carried, split guesses checked against Alpaca), the chart's windows computed in
> Python, the calendar to 2030 (its tripwire skipped inside the trade gate so a date cannot halt trading), Node 24
> actions. Today's rebalance ran cleanly (no trades needed); a dry-run dispatch confirmed the journal push on checkout v7.
> 
> ### The backtest rebuild after today's `pit_record` fixes
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

