# Morning Brief - Wednesday 25 November 2026, 02:25

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
| Shipped to readers | 19 commit(s) changed what a reader sees, 6 methodology change(s), in the last 7 days |
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

## Things that needed attention

- GATE 5 note: ENGINE CHANGED since run d687fd4972dd was scored (factor_engine.py): this rehearsal shows the new page code on that run's old scores, so the next 02:00 run is the first to put the new scores before the claims gate. A page branch keyed on a new payload field needs a test that sets the field in the browser (prompts/nightly.md, section 4).

## What changed in the repo

- `c1dd9e1 data: screener run 2026-11-25 - 499 scored, top: EXPE APA VLO SNDK MU`
- `7d39fb0 brief: code session 2026-11-24-midday`
- `d906531 NIGHTLY_LOG 2026-11-24 (13:00): gate results`
- `a73cc94 NIGHTLY_LOG 2026-11-24 (13:00): the implied-volatility inversion built, the switch queued`
- `d39649d research engine: all 32 rows rerun and the report written; summary and research page unchanged`
- `4c90763 research engine: 24 of 32 rows rerun`
- `97b15af research engine: 12 of 32 rows rerun after factor_engine.py's fetch change (no scoring change)`
- `781c424 page rebuilt from the generator (the options panel's construction sentence); changelog time corrected`
- `17a7753 own implied volatilities: changelog, research note build status, owner focus, inventory, brief date`
- `46626c4 wip: the options panel's construction sentence after both glosses`
- `ed5d428 wip: register the panel's IV-construction sentence; pin QuantLib; coverage measurement script`
- `ebf415a wip: our own implied volatilities (QuantLib, American, discrete dividends) recorded beside the source field`
- `c88a2ce brief: code session 2026-11-24`
- `c3b5f3b NIGHTLY_LOG 2026-11-24: residual momentum built and published, research engine rerun`
- `bc02a8e overview and page: the research engine's rerun figures (generated)`

## The session's own account

> 2026-11-24 (13:00) - QUEUE. The implied-volatility inversion built: our own volatilities recorded beside the source's on every reading, the panel says whose it shows; the switch waits two days for a whole-universe count
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-24_060001.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-24_020001.log` (completed; HEALTH: PASS).
> Evidence base at `1m`: **58 rows, newest `run_date` 2026-10-23 (32 days ago, bound 40), 5 effective** of 8.
> Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 43 days**; a standing programme, last advanced **18 days** ago, bound 21 - its day is
> Friday, so not deferred by a Tuesday); also open 0.10 (48 d), 0.14 (41 d). Shipped to readers: 19
> commits, 8 methodology changes in the last 7 days. No branch held back; no stash.
> **Tests:** before 4519 passed / 1 skipped (13:05); after **4553 passed / 1 skipped** (14:20, the
> committed tree; 34 new). Gate 2 dry run exit 0. Gate 5 (`rehearse_publish.py --changed-since main`)
> PASSED (page tests and claims gate, 2435 passed; it prints ENGINE CHANGED for `factor_engine.py`, which
> here is the fetch keeping two `.info` fields - no score reads them). Gate 3's `node --check` is refused by
> this session's tool guard; the payload files are byte-identical to `main` (`rebuild_page.py`), index.html
> 953 KB; the runner runs it. Gate 5 failed once mid-session on `test_club_readiness`, which reads the
> glosses by position - the construction sentence was moved after both glosses and built as a variable.
> **Owner queue / rotation:** carried on from the 06:00 entry's Next, which is also `OWNER_FOCUS.md`'s
> 2026-11-19 item: the implied-volatility inversion ACCEPTED 2026-11-04, unbuilt for twenty days.
> 
> ### Did
> - **`implied_vol.py`: our own implied volatilities**, the construction section 1d of
>   `research/2026-11-03-open-source-quant-stack.md` ADOPTED - each option's own bid/ask mid (the existing
>   `MAX_REL_SPREAD` bar, not loosened), QuantLib's American finite-difference solver with a discrete
>   dividend schedule, FRED's 1/3/6/12-month Treasury curve interpolated to the expiry and cached in
>   `data/options/` (the only place the 20:00 task may write). One inversion feeds the ATM call, ATM put
>   and both skew legs (build order step 1).
> - **No new request per stock (step 2's obstacle):** the core fetch keeps `exDividendDate` and
>   `lastDividendValue` from the `.info` it already pulls. Checked live on six tickers before writing the
>   rule: present for every payer (AAPL, JPM, KO, O, NVDA), absent for TSLA. The schedule rule: annual
>   rate in equal instalments, frequency from rate / last payment (O comes out monthly, the rest
>   quarterly), first on the ex-date. A payer with no ex-date is "unknown" and keeps the source field -
>   never priced as if no dividend fell in the option's life (step 3).
> - **A paired record from the first refresh.** Every reading carries both constructions where it can
>   (`_ctx_opt_own_*`, `_ctx_opt_src_*`), and the context log keeps every `_ctx_` column, so the
>   comparison the decision wanted ("the only way this project will ever be able to say which
>   construction predicts better") accrues from tonight. The main columns hold
>   `PUBLISHED_IV_CONSTRUCTION`'s figures, falling back - stamped - to the other.
> - **The page (step 4's guard, step 6):** the generator withholds a figure whose stamp is not the
>   published construction (`oniv`), by the same predicate `context_eval` counts with; the panel now
>   says whose volatilities it shows - today *"the data source's own figures; it does not publish how it
>   computes them"*, which the page had never said. Claim `context.iv_construction`.
> - **QuantLib 1.44 pinned** (step 5). It was already installed in `Python312`, the interpreter the
>   scheduled tasks run, so nothing was installed.
> - **Coverage, measured through the production path (step 7)**, 51 stocks mid-session
>   (`research/measurements/2026-11-24-own-iv-coverage.py`): ours 45 of the source's 47 ATM IVs, 40 of its
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

