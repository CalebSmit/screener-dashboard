# Morning Brief - Tuesday 10 November 2026, 13:08

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-11-06 18:34 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, MPC, DXCM |
| Ranking's lean on growth | +0.052 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (35 days ago, bound 40) |
| Context signals | 20 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 438/439 within 0.5%; operating cash flow 407/424 within 0.5%; net income 393/433 within 0.5%; market value 452/487 within 5%; 0 scored a quarter behind a filing over 30 days old; filings cached 0 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); HST (#5: net income 1.29x after-tax operating income); DXCM (#6: operating cash flow 36% off the filing); PSX (#9: net income 1.28x after-tax operating income); TPR (#10: equity 5.2% of assets); CF (#15: net income 15% off the filing); MO (#24: revenue 12% off the filing) |
| Data run timing | **23 min** total; slowest `fetch.tickers` 632 s, `fetch.sec_roe_history` 221 s, `fetch.retry` 152 s; **48 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (23 days) |
| Context coverage | options usable 415/499 - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (**due 1 day ago - still listed**) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 6 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- ok   ATTENTION - 1 stock(s) now carry a market value implying a share count over 10% from the one their filings report, which the published page does not: FOXA +32.3%. Usually a completed acquisition or a split: the market value is right and every valuation ratio built on it has moved, while the earnings behind those ratios have not. Not a refusal - see research/2026-11-09-share-counts-and-corporate-actions.md.
- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- SESSION DID NOT RUN: API error 400 - API Error: 400 Claude Code 2.1.229 does not support this model; version 2.1.290 or newer is required. Run 'claude update', or update the Claude desktop app, the
- Treating this as a failure, not as 'nothing to do': no success marker will be written, so the catch-up trigger will retry rather than skipping the day.
- No commits, because the session never ran. This is a lost day, not a quiet one.
- === Run FAILED: API error 400 - API Error: 400 Claude Code 2.1.229 does not support this model; version 2.1.290 or newer is required. Run 'claude update', or update the Claude desktop app, the ===

## What changed in the repo

- `5b058e3 brief: code session 2026-11-10`
- `4b06504 log: 2026-11-10 - the advice ban read a company name, and the fingerprint read a weekly cache`
- `febe5f5 fix: the mask read window.D, which the page does not have - so it masked nothing`
- `b7f6cdc rerun: the ledger level again, and the measurement that the filings refresh moved nothing`
- `52d2a68 rerun: research ledger rows 25-30 of 32`
- `d8ba024 rerun: research ledger rows 19-24 of 32`
- `3f05095 rerun: research ledger rows 13-18 of 32`
- `6838462 rerun: research ledger rows 7-12 of 32`
- `e891914 rerun: research ledger rows 1-6 of 32 at the new fingerprint`
- `344eb47 fix: the research ledger went stale because a weekly cache refresh moved, not because a number did`
- `87871f9 fix: a company's name is not a recommendation - the advice ban discarded a healthy run`
- `ecd9d9e brief: data run 2026-11-10`
- `4a404cb brief: code session 2026-11-09-midday`
- `92256b7 log: 2026-11-09 (13:00) - the share-count proposal decided, and the detector that had been comparing nothing`
- `94dc9fb docs: the share-count decision in the owner queue, and the gate-reads-the-whole-payload constraint`

## The session's own account

> 2026-11-10 - PRODUCT, THROUGH THE NUMBERS. Two refused loops instead: a company's name read as a recommendation, and a weekly cache refresh read as a changed result
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-09_130001-midday.log` (completed). Data loop published? **no** - `logs/datarun-2026-11-10_020001.log` (stopped deliberately; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (35 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **29 days**); also open: 0.10 34d; 0.14 27d.
> 
> **Tests:** before **4,268 passed / 3 skipped / 1 FAILED** (the runner's baseline,
> `logs/nightly-2026-11-10_060001.baseline.txt`: `test_ledger_is_level_with_the_code.py::
> test_every_published_row_was_produced_by_the_committed_code[content_hash]`) -> after **4,291
> passed / 3 skipped / 0 failed**. Gates 1-4 pass. Gate 5 (`scripts/rehearse_publish.py`)
> **PASSED** - it failed twice first, and both failures were real; all five published files
> restored byte for byte.
> 
> **Owner queue / rotation:** `write_brief.open_proposals()` returns `[]`, and `OWNER_FOCUS.md`'s
> open items are the long-horizon universe expansion. The nominal focus was Tuesday's product day
> and I did not reach it: **two loops were refused and `main` arrived red**, which the prompt and
> CLAUDE.md both put ahead of the focus. **Deferred: the product focus, 0.13 (the Factor Lab, 29
> days), 0.10 (34d) and 0.14 (27d).** 0.13 is now a **fifth** consecutive deferral; the
> deferral-streak rule's exception does not cover a fifth, so I am recording plainly that a red
> `main` and two non-publishing loops outranked it rather than claiming the rule was met - see
> **Next**.
> 
> ### Did
> 
> **1. The advice-language ban discarded a healthy data run, because Best Buy is a company.**
> 
> The 02:00 loop refused to publish on `tests/test_dashboard_navigation.py::
> test_reporting_soon_is_a_calendar_not_a_leaderboard`, with `HEALTH: PASS` on every line above it:
> 499 scored, 499/499 priced, 497/499 with a target, Spearman 0.989 against the published page,
> every category dispersion inside its band. Reproduced against the regenerated page
> (`scripts/rehearse_publish`'s machinery, one test): the failing assertion is
> `advice_terms_in(page.inner_text("#reporting-body")) == []`, and the term found is **`Buy`**, in
> **Best Buy Co., Inc.** Best Buy's next report is 2026-11-24 - **exactly +14 days** from that run
> date - so it entered the panel's longer window that morning for the first time.
> 
> The page was right: a calendar of forthcoming reports prints the company's legal name. The
> *check's premise* was wrong - "no company reporting within fourteen days has a banned word in its
> name" is a fact about S&P 500 membership and the earnings calendar, and it went false with no
> commit touching anything. **Measured: Best Buy is the only name in the universe that trips the
> ban** - 1 of 998 names and tickers, over all 499 rows of both the 11-06 and 11-10 payloads - which
> is why a check written weeks ago fired on the fourth window it could have fired on.
> 
> - `stock_summary.without_data_names()` masks the data's own proper nouns before the ban runs:
>   longest first, each replaced by a space so no two words fuse, case-insensitive like the ban. A
>   genuine recommendation beside a masked name still matches.
> - every advice check over text a *browser* rendered now masks first - Reporting Soon, the stale
>   note, the option-quote panel, the track record.
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

