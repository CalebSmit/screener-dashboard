# Morning Brief - Friday 27 November 2026, 02:22

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## Coming up, only you can do it: the Claude CLI login runs out 2026-12-03 09:01 (in 6 days)

The code sessions run through the Claude Code CLI on this PC, and its login
lasts about 30 days. When it runs out, every scheduled session fails within
seconds until you sign in again. Doing it early loses nothing - open a
terminal on the PC and run:

    claude.cmd auth login

## At a glance

| | |
|---|---|
| Data run (2 AM) | **refused by its publish gates (published-claims check failed)** - last ran today |
| Code session (latest) | **held back by its gates (work kept on its branch, not on main)** - last ran today |
| Dashboard data from | 2026-11-26 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, APA, MPC, SNDK |
| Ranking's lean on growth | +0.012 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 14 commit(s) changed what a reader sees, 3 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (59 rows, but overlapping windows are not independent; 147 rows across all horizons), newest 1m run date 2026-10-26 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 459/459 within 0.5%; operating cash flow 445/446 within 0.5%; net income 433/457 within 0.5%; market value 452/487 within 5%; 5 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 1 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#6: net income 1.29x after-tax operating income); PSX (#10: net income 1.28x after-tax operating income); DXCM (#14: operating cash flow 36% off the filing); CF (#15: net income 15% off the filing); ES (#18: a period behind its filings); TPR (#19: equity 5.2% of assets) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 624 s, `snapshot` 152 s, `fetch.retry` 111 s; **30 tickers rate-limited** |
| Code-session login | **runs out 2026-12-03 09:01 (in 6 days)** |
| Context coverage | options usable 481/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-12-04 (in 7 days) - The expected move's first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. (The ATM volatility and the skew restart under our own construction from the 2026-11-27 log - see 2026-12-28.) `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 10 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## Things that needed attention

- ATTENTION - CLI LOGIN EXPIRES 2026-12-03 09:01 (in 6 days) - after that every scheduled code session fails until the owner runs `claude.cmd auth login` in a terminal and signs in; doing it before then loses nothing.
- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- GATE 1 tests: FAIL (=========================== short test summary info =========================== FAILED tests/test_sheet_round_three.py::test_a_chosen_tab_stays_lit_at_the_end_of_the_sheet 1 failed, 4571 passed, 1 skipped, 391 warnings in 250.21s (0:04:10))
- full output: C:\Users\smitc\OneDrive\Documents\Screener\logs\nightly-2026-11-26_130001-midday.gate1.txt
- GATE 1 attribution: 1 new, 0 inherited from a main that was already red.
- new: tests/test_sheet_round_three.py::test_a_chosen_tab_stays_lit_at_the_end_of_the_sheet
- SHIP GATES FAILED: tests. Not merging.
- Work pushed to nightly/2026-11-26 for inspection. origin/main verified clean (reverted if the earlier push had reached it).
- Run finished with failing gates - see above.

## What changed in the repo

- `5e8fb20 brief: code session 2026-11-26-midday`
- `88354de brief: code session 2026-11-26`
- `5c324fb NIGHTLY_LOG 2026-11-26: own implied volatilities published`
- `c9d85b9 own IV published: changelog, claim extended, browser test, brief dates, note step 4 done, owner item closed`
- `96f8817 wip: page rebuilt for the own-IV sentences (rebuild_page.py; payload byte-identical); skew fixture stamped as published`
- `dee457f wip: publish our own implied volatilities (PUBLISHED_IV_CONSTRUCTION own_mid); cached readings re-selected; whole-universe coverage measured`
- `f0eac97 brief: data run 2026-11-26`
- `8da9237 data: screener run 2026-11-26 - 499 scored, top: EXPE VLO APA MPC SNDK`

## The session's own account

> 2026-11-26 - BUILD (owner queue first). Our own implied volatilities published: counted over the whole universe, every ATM figure ours lacks had no usable quote, and the flip would have blanked the page for Thanksgiving without a fix
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-25_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-26_020001.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **59 rows, newest `run_date` 2026-10-26 (31 days ago, bound 40), 5 effective** of 8.
> Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 45 days**; a standing programme, last advanced **1 day** ago, bound 21 - not deferred);
> also open 0.10 (50 d), 0.14 (43 d). Shipped to readers: 21 commits, 3 methodology changes in the last
> 7 days. The "held back" branch the line names is this session's own, pushed mid-session. No stash.
> **Tests:** before 4558 passed / 1 skipped; after **4564 passed / 1 skipped** (6 new). Gate 2 dry run exit
> 0. Gate 5 (`rehearse_publish.py --changed-since main`) PASSED on the final tree (it failed once mid-session
> on `test_option_implied_claims.py`'s unstamped fixture - fixed, below); no ENGINE CHANGED. Gate 3's
> `node --check` is refused by this session's tool guard; the three payload files are byte-identical to
> `main` (`rebuild_page.py`), index.html 954 KB; the runner runs it. Gate 4 clean at the end.
> **Owner queue / rotation:** `OWNER_FOCUS.md`'s 2026-11-19 item had one part left - the implied-volatility
> switch, dated today in the brief's "Coming up". Taken first; that also satisfies Thursday's "build what
> the week justified". No open proposal (`open_proposals()` = `[]`). **The 2026-11-19 item is now CLOSED
> and moved to Done.**
> 
> ### Did
> - **The whole-universe count the 11-04 decision asked for**
>   (`research/measurements/2026-11-26-own-iv-coverage-universe.py`, the 2026-11-25 20:00 refresh, 499
>   readings; the run it read dated 402 of 402 payers' ex-dividend dates): ours **418 of the source's 486**
>   ATM IVs and **327 of its 483** skews.
>   - **All 68 ATM losses are `partial` readings with no expected move**: no usable at-the-money mid, so
>     the source's figure there came from a contract whose quote did not stand up. That loss is the point.
>   - Of 156 skew losses, **128** have our ATM call, so the gap is the out-of-the-money put; they are
>     calmer stocks (source ATM IV median 0.361 vs 0.418), where a 5%-out put is cheapest and its spread
>     widest. `MAX_REL_SPREAD` and the IV bounds untouched.
>   - Ours minus source: ATM median -2.78 pts, skew +2.03 (sign disagrees 115 of 327); skew negative for
>     13.8% of ours, 35.0% of the source's, ~6% in Xing, Zhang & Zhao's Table 1.
> - **Flipped `PUBLISHED_IV_CONSTRUCTION` to `own_mid`.**
> - **Found and fixed what the flip would have done to tomorrow's page:** the cache's main columns were
>   assembled under `source`, the page withholds a figure stamped the other way, and Thanksgiving means no
>   20:00 refresh replaces the cache before the 11-27 02:00 run - so **every stock's volatility would have
>   been withheld**. `context_signals.in_published_construction`, called by `options_cache.reading_for`,
>   re-selects a cached reading's main columns from its own paired `_ctx_opt_own_*` columns (nothing
>   recomputed). Simulated on the real cache for 11-27: ATM IV shown for 418, skew for 327, 57 withheld.
> - **The panel says why a figure is missing**: a withheld reading under `own_mid` now says ours could not
>   be worked out and the source's unpublished figures are not shown instead (was: "worked out a different
>   way", true but no help); a reading with our IV and no skew says its two skew options could not both be
>   worked out. `oniv` carries the published construction. Claim `context.iv_construction` extended; a
>   browser test sets each branch (`test_the_panel_names_our_construction_and_says_why_a_figure_is_missing`).
> - **Looked at it:** AAPL, ATO and AEE sheets opened on the rebuilt page with tomorrow's re-selected
>   context at 1440 and 375 px, text read back: AAPL shows 25% ATM, +4.5 pt skew ($335 put 30% - $355
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

