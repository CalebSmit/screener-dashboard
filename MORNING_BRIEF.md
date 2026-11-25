# Morning Brief - Wednesday 25 November 2026, 06:35

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-25 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, APA, VLO, SNDK, MU |
| Ranking's lean on growth | -0.001 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 18 commit(s) changed what a reader sees, 6 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (58 rows, but overlapping windows are not independent; 144 rows across all horizons), newest 1m run date 2026-10-23 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 453/453 within 0.5%; operating cash flow 439/440 within 0.5%; net income 427/451 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#6: net income 1.29x after-tax operating income); PSX (#10: net income 1.28x after-tax operating income); DXCM (#14: operating cash flow 36% off the filing); CF (#18: net income 15% off the filing); ES (#19: a period behind its filings); TPR (#21: equity 5.2% of assets) |
| Data run timing | **20 min** total; slowest `fetch.tickers` 646 s, `snapshot` 167 s, `fetch.retry` 134 s; **37 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (8 days) |
| Context coverage | options usable 483/499 (16 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-26 (in 1 day) - Publish our own implied volatilities (built 2026-11-24, recorded beside the source field as _ctx_opt_own_*). First confirm the 11-25 02:00 run's raw fetch carries exDividendDate and the 11-25 20:00 refresh's cache has _ctx_opt_own_iv for most stocks; count own vs source coverage over the whole cache, then flip context_signals.PUBLISHED_IV_CONSTRUCTION to 'own_mid' with a changelog entry. Do not loosen MAX_REL_SPREAD or the IV bounds to raise coverage. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 9 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 12 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## What changed in the repo

- `f49d35d NIGHTLY_LOG 2026-11-25: gate results`
- `90d2093 NIGHTLY_LOG 2026-11-25: synthesis entry`
- `c87c8f5 exposure_shift: a scoring change records the ranking's factor exposures before and after`
- `6d154fb wip: changelog check of residual momentum on the first data run; growth-lean break recorded`
- `f0f69c8 wip: synthesis section on residual momentum's fit (note section 10)`
- `1df46b4 wip: synthesis measurement - residual momentum's fit against the other seven categories`
- `8c06fdc brief: data run 2026-11-25`
- `c1dd9e1 data: screener run 2026-11-25 - 499 scored, top: EXPE APA VLO SNDK MU`
- `7d39fb0 brief: code session 2026-11-24-midday`
- `d906531 NIGHTLY_LOG 2026-11-24 (13:00): gate results`
- `a73cc94 NIGHTLY_LOG 2026-11-24 (13:00): the implied-volatility inversion built, the switch queued`
- `d39649d research engine: all 32 rows rerun and the report written; summary and research page unchanged`
- `4c90763 research engine: 24 of 32 rows rerun`
- `97b15af research engine: 12 of 32 rows rerun after factor_engine.py's fetch change (no scoring change)`
- `781c424 page rebuilt from the generator (the options panel's construction sentence); changelog time corrected`

## The session's own account

> 2026-11-25 - SYNTHESIS. Residual momentum read against the other seven categories on its first data run: Momentum overlaps six of them less, the ranking's growth lean stepped +0.026 -> +0.006, and the habit that would have caught it is one command now
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-24_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-25_020001.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **58 rows, newest `run_date` 2026-10-23 (33 days ago, bound 40), 5 effective** of 8.
> Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 44 days**; a standing programme, last advanced **19 days** ago, bound 21 - its day is
> Friday, so not deferred by a Wednesday); also open 0.10 (49 d), 0.14 (42 d). Shipped to readers: 19
> commits, 6 methodology changes in the last 7 days. The "held back" branch the line names is this
> session's own, pushed mid-session. No stash.
> **Tests:** before 4553 passed / 1 skipped (06:05, `logs/session-baseline-2026-11-25.txt`); after
> **4557 passed / 1 skipped** (the 4 new in `tests/test_exposure_shift.py`). Gate 2 dry run exit 0. Gate 5
> (`rehearse_publish.py --changed-since main`) PASSED (page tests, claims gate, files restored). Gate 3:
> `node --check` is refused by this session's tool guard; the three payload files and `index.html` are
> byte-identical to `main` (`git diff --stat main` empty), so the runner's check reads what it read
> yesterday. Gate 4 clean.
> **Owner queue / rotation:** no proposal open (`write_brief.open_proposals()` = `[]`). `OWNER_FOCUS.md`'s
> 2026-11-19 item has one step left, the implied-volatility switch, which is dated for 2026-11-26 (it needs
> tonight's 20:00 refresh) - not today's. So the rotation's focus: the methodology changes since the last
> synthesis (2026-11-18). Changelog read back to it: 11-18 (13:00) SEC co-registrant refresh; 11-19
> universe floor 0; 11-19 missing category at 50; 11-19 EPS-mismatch like-for-like; 11-19 sustainable
> growth's ROE needs positive equity both years; **11-24 residual momentum replaces Jensen's alpha**; 11-24
> (13:00) own implied volatilities, context only. The one that changes how stocks are compared across
> categories is residual momentum, so it is the subject; the others are placed in a line each below.
> 
> ### Did
> - **The previous entry's owed check, done:** the 02:00 run published residual momentum with HEALTH:
>   PASS, rank continuity **0.998** against the 11-24 06:00 page (1 new in the top 25), coverage **491 of
>   499** as the build said.
> - **Measured the fit** (`research/measurements/2026-11-25-momentum-fit.py` / `.json`) on three pages: the
>   last scored with Jensen's alpha (`0679feb`), the same closing session rescored with residual momentum
>   (`87b7be9`, so the difference is the change, not a day of markets), and the first data-loop page
>   (`c1dd9e1`). Rank correlations only.
>   - The note's prediction (a) **held**: residual momentum ~ `return_12_1` **0.792** (predicted 0.77),
>     where Jensen's alpha was **0.910**.
>   - Prediction (b), "removes a second low-beta reward", **could not be observed**: Jensen's alpha ~ beta
>     was already **+0.009** over this window (the 12-month return's +0.17 beta lean cancelled its beta
>     term). Residual momentum is beta-neutral by construction; the alpha was only when the year allowed.
>   - **Momentum now overlaps six of the seven other categories less**: growth~momentum +0.138 -> +0.082,
>     valuation~momentum -0.095 -> -0.045, momentum~size -0.154 -> -0.108, momentum~risk +0.251 -> +0.219,
>     quality~momentum +0.098 -> +0.069, revisions~momentum +0.209 -> +0.188 (budget bar 0.35 - headroom
>     gained, which does **not** license a momentum-adjacent Revisions metric); investment +0.03, negligible.
>   - **The ranking's lean on growth stepped +0.026 -> +0.006 from this change alone** (about one point of
>     Growth weight at the drift note's slope). The brief's daily growth-lean series has a break at
>     2026-11-24 that is a methodology step, not drift.
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

