# Morning Brief - Tuesday 03 November 2026, 02:23

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-11-03 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, SNDK, HST |
| Ranking's lean on growth | +0.073 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (43 rows, but overlapping windows are not independent; 110 rows across all horizons), newest 1m run date 2026-10-02 (32 days ago, bound 40) |
| Context signals | 17 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 450/451 within 0.5%; operating cash flow 421/435 within 0.5%; net income 412/447 within 0.5%; market value 450/486 within 5%; 4 scored a quarter behind a filing over 30 days old; filings cached 1 days ago |
| Top 25, worth a look | EXPE (#2: equity 4.2% of assets); SNDK (#4: a period behind its filings); HST (#5: net income 1.46x after-tax operating income); DXCM (#6: operating cash flow 36% off the filing); EG (#8: a period behind its filings); PSX (#9: net income 1.28x after-tax operating income); CF (#13: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 631 s, `fetch.retry` 155 s, `snapshot` 145 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-02 10:28 (29 days) |
| Context coverage | options usable 474/499 (25 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **decide how the screener holds the ranking's growth exposure near neutral now that a fixed 13% weight has drifted from -0.01 to +0.06 in three weeks (re-fit the weight on a schedule, constrain the...** - open 1 day - `research/2026-11-02-growth-exposure-drift.md`
- **decide whether to fold the Investment category (`asset_growth`, 5%) into Quality at the weight-preserving 18.52% of a 27% Quality (section 3) or keep it standalone, as this note recommends, with the...** - open 1 day - `research/2026-11-02-investment-inside-quality.md`
- **a later session (not the one that wrote this) decides whether to adopt the three pre-registrations in section 7: the S&P 1500 as the first wider universe, the NYSE 20th-percentile market cap as the...** - open 1 day - `research/2026-11-02-universe-expansion-phase0.md`

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (in 6 days) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). `CLAUDE.md`
- 2026-11-16 (in 13 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- SESSION DID NOT RUN: session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed
- Treating this as a failure, not as 'nothing to do'. No success marker will be written,
- so the catch-up trigger will retry rather than skipping the day.
- No commits, because the session never ran. This is a lost day, not a quiet one.
- === Run FAILED: session reported an error - Failed to authenticate: OAuth session expired and could not be refreshed ===

## What changed in the repo

- `5a719b6 data: screener run 2026-11-03 - 499 scored, top: VLO EXPE MU SNDK HST`
- `cf35873 Log the dashboard sweep's second round and queue the expected-move defect first`
- `55ddf5c plan/claims-mutation-audit.md: the rehearsal's catch`
- `1313f2d test_dashboard_printed_figures: the valuation-history recount allows the published rounding of both sides`
- `8207357 OWNER_FOCUS: round 2 done`
- `46dce22 NIGHTLY_LOG: owner-run night, round 2`
- `5d6429d plan/claims-mutation-audit.md: results of the late pass (37 of 77 dashboard rows closed, rows 189-271, the expected-move finding, the queue)`
- `7c688a8 Research engine: rerun all 32 rows on the new config hash (no result changed)`
- `b46ea1d Claims audit: the KPI row, the score key's median line, the analytics headline (5 more open rows, rows 263-271, 9 of 9 caught)`
- `68f7467 Claims audit: 'Against its own five years' (2 more open rows, rows 257-262, 6 of 6 caught)`
- `a1246d2 Claims audit: the expected-move card's divisor is not the price it names (measured; rows left open, marked FALSE)`
- `c608d13 Claims audit: the drilldown's rate card (2 more open rows, rows 251-256, 6 of 6 caught)`
- `acd1d23 CLAUDE.md: any config.yaml or factor_engine.py edit stales the research ledger - budget the rerun`
- `dfc8747 Claims audit: the What Changed footnote and tile counts (4 more open rows, rows 243-250, 8 of 8 caught)`
- `c51f212 Claims audit: the Methodology dialog's 'Momentum rule, this run' note (3 more open rows, rows 237-242, 6 of 6 caught)`

## The session's own account

> 2026-11-02 (owner-run, night, round 2) - "Keep finding improvements!"
> 
> **Health (rule 8):** as the evening entry above (no scheduled session or data run since); the line
> now prints from `scripts/health_numbers.py` and the runner hands it to every session in its prompt.
> **Tests:** 3,663 passed / 3 skipped at the last gate run. Tags `good/2026-11-02-owner-9`, `-10`.
> 
> ### Did
> 
> 1. **Gate 5: the runner rehearses the next publish** (`scripts/rehearse_publish.py --changed-since
>    $BaseSha`) whenever a branch changed more than prose, before gate 4; a failure keeps the work off
>    `main`; "no full run to rehearse from" (exit 3) only warns. **It earned its keep within the
>    hour**: the stale-flag fix below passed gates 1-4 locally while the rehearsal failed
>    (`test_the_baseline_weighting_is_hashed_separately_from_the_transformed_one`) - a config edit
>    stales the research ledger. CLAUDE.md now says so up front (`config.yaml` and `factor_engine.py`
>    are in `RESULT_PATHS`; budget ~20 min for `rerun_experiments.py --jobs 4`).
> 2. **The stale-filing flag waits for the SEC deadline** (display only; `config.yaml` 120 -> 165,
>    METHODOLOGY_CHANGELOG 2026-11-02 night). At 120 it told readers **320 of 499** stocks had stale
>    filings on 10-30 while their Q3 10-Qs were not yet due; every bar from 130 to 210 flags the same
>    **21** genuinely a quarter or more behind. The 32 research rows were rerun on the new config hash
>    - every difference is a provenance field; `summary.json` and `research.html` byte-identical.
> 3. **Experiment 18 second-read** by a different reader (isolated worktree): every paper figure
>    (NBER w5375, rendered pages) and every measured figure reproduced without importing
>    `factor_lab`; the published article's figures confirmed via Jegadeesh & Titman (2001); the
>    pre-registration order verified; three wording fixes to the note, no figure changed.
> 4. **Weight question 3 researched** - `research/2026-11-02-investment-inside-quality.md`: keep
>    Investment standalone and state the practitioner convention on the page (**proposal OPEN**); a
>    weight-preserving fold leaves the top 25 unchanged but is not neutral where data is missing,
>    under the panic rule, in the profiles and in the trap flags.
> 5. **Two load-sensitive browser tests in the publish gate** now measure the page, not the machine
>    (best of five clicks; the reload gets `_open`'s allowance) - one slow sample at 02:00 would have
>    discarded a night's run. No budget loosened.
> 6. **The brief's "Coming up"** lists dated commitments (2026-11-09 first one-month context
>    reading, 2026-11-16 filing-check promotion, 2027-02-11 backtest) from `COMING_UP`, each tied by a
>    test to the file that commits to it.
> 7. **Checked and left alone:** Loews (no analyst coverage in any published run; its Revisions move
>    is real short-interest change), Yahoo rate-limiting (40-45 a run since 10-27, all recovered).
> 
> ### Next
> - The OPEN proposals now waiting for a session that did not write them: growth exposure, universe
>   phase 0, Investment-inside-Quality.
> - After 02:00: the stale flag should fire for about 21 stocks, not about 320.
> - **A FALSE sentence on the live page, measured and queued (item 1 of the claims queue):** the
>   expected-move card says the move is the at-the-money straddle divided by *this run's* close, but
>   the 20:00 option refresh (`options_cache.refresh` -> `_fetch_one` -> `context_signals.options_context`)
>   is handed the previous run's `price_latest` and picks the at-the-money strike against it. On the
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

