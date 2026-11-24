# Morning Brief - Tuesday 24 November 2026, 08:46

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-24 06:18 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, APA, VLO, MU, SNDK |
| Ranking's lean on growth | +0.006 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 18 commit(s) changed what a reader sees, 7 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (58 rows, but overlapping windows are not independent; 142 rows across all horizons), newest 1m run date 2026-10-23 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 453/453 within 0.5%; operating cash flow 439/440 within 0.5%; net income 427/451 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 6 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#7: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); TPR (#13: equity 5.2% of assets); DXCM (#17: operating cash flow 36% off the filing); ES (#19: a period behind its filings); CF (#21: net income 15% off the filing) |
| Data run timing | **22 min** total; slowest `fetch.tickers` 606 s, `snapshot` 382 s, `context.sec_insider` 167 s; **32 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (9 days) |
| Context coverage | options usable 474/499 (25 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-24 (**today**) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 10 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 13 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## What changed in the repo

- `c3b5f3b NIGHTLY_LOG 2026-11-24: residual momentum built and published, research engine rerun`
- `bc02a8e overview and page: the research engine's rerun figures (generated)`
- `0573aec research engine: backtest v2 months rebuilt with residual momentum, all 32 rows rerun`
- `2d67ed7 OWNER_FOCUS: residual momentum built (2026-11-19 item stays open for the IV inversion)`
- `87b7be9 wip: the full run that publishes residual momentum, its measured effect, the changelog entry`
- `3fc14ad wip: residual momentum in place of Jensen's alpha - engine, lineage, page, backtest v2 input, tests`
- `def9850 brief: data run 2026-11-24`
- `0679feb data: screener run 2026-11-24 - 499 scored, top: EXPE APA VLO SNDK MU`
- `8a0cf49 NIGHTLY_LOG 2026-11-23 (owner-run, evening): the jammed loops, the push retry, the phone rows`
- `84086fc Phone rows: the flag test holds in both row forms; the publish gate's layout list follows the rename`
- `172a08e Phone: a ranking row is rank, ticker, the whole name and the composite`
- `931a9b1 Data loop: a failed push is retried, and one that never lands leaves main level with origin`
- `4a646bf brief: data run 2026-11-23`
- `1d4cd76 brief: code session 2026-11-23-midday - SESSION DID NOT RUN`
- `d8632b6 brief: code session 2026-11-23 - SESSION DID NOT RUN`

## The session's own account

> 2026-11-24 - PRODUCT, THROUGH THE NUMBERS (owner queue first): residual momentum built in place of Jensen's alpha, with the full run that publishes it
> 
> **Health (rule 8, all five):** last code session ran? **failed** - `logs/nightly-2026-11-23_130001-midday.log`
> ("git pull --ff-only failed - main diverged locally"); both of 11-23's sessions stood down on the jam the
> owner-run session of 11-23 (evening) found and fixed (`main` level with `origin/main` at 06:05 today:
> `git rev-list --left-right --count main...origin/main` = `0 0`; nothing further to do). Data loop
> published? **yes** - `logs/datarun-2026-11-24_020001.log` (completed; HEALTH: PASS). Evidence base at
> `1m`: **58 rows, newest `run_date` 2026-10-23 (32 days ago, bound 40), 5 effective** of 8. Priority 0:
> `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 43
> days**; a standing programme, last advanced **18 days** ago, bound 21 - its day is Friday, so not deferred
> by a Tuesday); also open 0.10 (48 d), 0.14 (41 d). Shipped to readers: 18 commits, 7 methodology changes
> in the last 7 days. (The health line's "held back" branch is this session's own, mid-session.)
> **Tests:** before 4501 passed / 1 skipped (baseline, 06:05); after 4519 passed / 1 skipped (08:2x, the exact tree pushed; 18 new tests). Gate 2 dry run exit 0. Gate 5 (`rehearse_publish.py --changed-since main`): failed once on `check_published_claims` and passed on the two runs after it; the first run's output was piped to `tail` and not kept, so its cause is unknown - **if the runner's gate 5 refuses this branch, read `logs/nightly-*.gate5.txt` first**. Gate 3's `node --check` is refused by this session's tool guard; the runner runs it
> **Owner queue / rotation:** `OWNER_FOCUS.md` 2026-11-19 item - two ACCEPTED builds were due: residual
> momentum (due 11-23 06:00, a session that never ran) and the implied-volatility inversion (due today). Took
> residual momentum: it is overdue, it is a scoring change a reader sees on every stock, and the brief asked
> for "a 06:00 session with nothing else in it" for each. The IV inversion is **set aside** (below). Tuesday's
> focus is served by the same work - the third Momentum metric could not answer "is this stock's trend its
> own or the market's?", and the drilldown check the focus asks for was done on the new row (AAPL, JPM, SNDK).
> 
> ### Did
> - **Residual momentum replaces `jensens_alpha`** (25 of Momentum, weight unchanged), exactly as decided in
>   `research/2026-11-20-jensens-alpha-in-momentum.md` section 9: 36 monthly returns anchored on the run's
>   date, CAPM fit on `^SP500TR` with an intercept, residuals of months 12..2 back summed over their sd. The
>   paper's own window was re-read (Blitz, Huij & Martens, manuscript p. 11: "over the period from t-36 until
>   t-1"), and the construction matches it - the skipped month enters only the fit.
> - **The trap the note named, closed with a test:** the fetch now asks for 37 months in **the same one
>   request**, and every other price input is computed on the last 13 months of it (`risk_window`, backtest
>   v2's own rule) - `tests/test_residual_momentum.py::test_the_risk_inputs_keep_their_thirteen_months`.
>   `^SP500TR` to 4y; the panic rule's state is asserted identical on 3y and 4y of closes.
> - **A defect the new tests caught before it shipped:** a stock moving exactly with the market left
>   residuals of float noise (~1e-16) and the ratio came out as 0.41. Floored at 1e-9.
> - **Frontend in the same change (row 0.8c):** `LINEAGE`/`RECOMPUTE`/`EQUATIONS` - the row prints the
>   engine's own division `{rm_sum} ÷ {rm_sd}` with the 36-month beta, via `ENGINE_KEYS`; the Recent move
>   card's reversal note rewritten (claim `context.momentum_skip`: all three weighted Momentum measures now
>   skip the month); `scoring_schema` 16 -> 17; `SCREENER_DEFENSIBILITY_SPEC.md`, `plan/dashboard-inventory.md`.
> - **Backtest v2 builds the same input through the same function** (`residual_momentum_fields`,
>   `pit_record.FIELD_CLASSES`), its months rebuilt (both eras, 6 shards) and the research engine rerun -
>   176 months rebuilt in ~70 min (6 shards), 32 rows rerun in ~16 min (`--jobs 4`): the current weighting's IC is -0.0023 pooled (was -0.0013), 0 of 27 alternatives better or worse, control t 5.74 - a Backtest observation only. Months before ~2013-12 have no residual momentum (daily prices begin 2010-11).
> - **The full run that publishes it** (the 2026-10-20 bank precedent: a scoring change ships with the live
>   run that carries it, so the committed page and the suite agree): `HEALTH: PASS`, published-claims gate
>   2400 passed with only the stale ledger red before the rerun. Same closing session as the 02:00 page
>   (2026-11-23), so the comparison isolates the change: rank Spearman 0.996, top 25 three in (ADBE, IT, NWSA)
>   three out (ALL, DLTR, EVRG), median move 6 places, coverage 491/499
>   (`research/measurements/2026-11-24-residual-momentum-effect.py`/`.json`). **`signals.json` is untouched
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

