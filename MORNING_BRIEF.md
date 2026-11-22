# Morning Brief - Sunday 22 November 2026, 06:47

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-20 18:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, APA, SNDK |
| Ranking's lean on growth | +0.054 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 16 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (55 rows, but overlapping windows are not independent; 135 rows across all horizons), newest 1m run date 2026-10-20 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 458/458 within 0.5%; operating cash flow 443/444 within 0.5%; net income 432/456 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#7: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); TPR (#9: equity 5.2% of assets); DXCM (#11: operating cash flow 36% off the filing); ES (#18: a period behind its filings); CF (#23: net income 15% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 644 s, `context.options_pass` 182 s, `fetch.retry` 149 s; **46 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (12 days) |
| Context coverage | options usable 494/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (in 1 day) - Build residual momentum in place of `jensens_alpha` (ACCEPTED 2026-11-21, weight 25 unchanged) in a 06:00 session (0.15's section 4 is committed - done 2026-11-21): section 9's build list - 37-month history on the same request with the Risk inputs trimmed to 13, `^SP500TR` to 4y, lineage/equation/claims, backtest v2, the engine rerun, a throwaway full run before the changelog entry. `research/2026-11-20-jensens-alpha-in-momentum.md`
- 2026-11-24 (in 2 days) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 12 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`

## Things that needed attention

- ok   ATTENTION - 1 stock(s) now carry a market value implying a share count over 10% from the one their filings report, which the published page does not: SNDK +639.2%. Usually a completed acquisition or a split: the market value is right and every valuation ratio built on it has moved, while the earnings behind those ratios have not. Not a refusal - see research/2026-11-09-share-counts-and-corporate-actions.md.

## What changed in the repo

- `11e048c NIGHTLY_LOG 2026-11-22: gates and what the full run caught`
- `2b398f5 shard (inp): three more published-payload readers fold the shard; docs and log`
- `2d634c2 shard: the reported inputs (inp) leave first paint - 1,125 -> 908 KB gzipped, -19.3%`
- `6cb141a brief: code session 2026-11-21-midday`
- `03de1b1 NIGHTLY_LOG 2026-11-21 (13:00): Track record tile leads with the lesson; 0.15 closed`
- `638c684 0.15 CLOSED: both section-5 page changes shipped`
- `294401c Track record tile leads with the lesson: heading 'too short to judge', caveat above the return (0.15 answer D, section 5 item 2)`
- `bbcc74c brief: code session 2026-11-21-weekend`
- `2e9da36 NIGHTLY_LOG 2026-11-21: gates recorded`
- `01063c3 NIGHTLY_LOG 2026-11-21 (weekend): Jensen proposal decided, 0.15 answered`
- `5758fa7 inventory: the hero's second tested sentence and the research section`
- `fd9541f 0.15's results on the page: the hero's tested line and a research-page section`
- `4c9583e 0.15 answered: D, a teaching tool with no return claim`
- `129ee13 0.15: fix the measurement's method and script before the first run`
- `86db426 Decide the Jensen's-alpha proposal: ACCEPTED (residual momentum), build after 0.15's measurement`

## The session's own account

> 2026-11-22 - WEEKEND. Universe expansion phase 1: each stock's reported inputs (`inp`) leave first paint - 1,125 -> 908 KB gzipped, -19.3%, nothing on the page changed
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-21_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-20_180003.log` (completed; HEALTH: PASS);
> no 02:00 run on a weekend, so nothing newer is expected. Evidence base at `1m`: **55 rows, newest
> 2026-10-20 (33 days ago, bound 40), 5 effective of 8**. Priority 0: `allow_auto_apply` **false**. Top open
> roadmap item: **0.13, the Factor Lab** (age **41 days**; a standing programme last advanced **16 days**
> ago, bound 21 - not late); also open 0.10 46d, 0.14 39d. Shipped to readers: 16 commits, 8 methodology
> changes in 7 days.
> **Tests:** baseline **4,455 passed / 1 skipped** (dry run OK). After: **4,461 passed / 1 skipped** (+6,
> the new module; no new skips). Gate 2 dry run OK; gate 3 all three payload files `node --check` rc 0 with
> their opening assignments, index.html 955,149 bytes; gate 4 clean; **gate 5 REHEARSAL PASSED** (every
> published file regenerated byte-for-byte unchanged, page tests and claims gate pass, restored).
> **What the first full run caught:** three more readers my grep for `"inp"` missed - the capex-lease and
> special-item notes are rebuilt from the published inputs *through* `sec_crosscheck` (both in the 02:00
> publish gate), and the cap test hands payload stocks to `equationHtml`. All three now fold the shard. The
> lesson for the next slice: grep for the *functions that consume* a field, not only its name - or simply
> regenerate the committed payload first and let the suite name every reader.
> **Owner queue / rotation:** `git stash list` empty; no owner-session marker; no earlier entry today; no
> proposal waiting. OWNER_FOCUS's top item (2026-11-19) has two sub-items left, both dated for weekday
> 06:00 slots with a 02:00 run behind them (residual momentum 11-23, the IV inversion 11-24) - a scoring
> change on a weekend reaches no claims gate until Monday. The 11-17 item is the owner's three decisions.
> So I took the next owner item, universe expansion, at the stage its plan names next (phase 1: the
> `stock_detail` shard's next field) - also the previous entry's Next 2.
> 
> ### Did
> - **`inp` - the reported figures behind every metric's equation - moved from `dashboard_data.js` into
>   `dashboard_detail.js`.** First paint **1,125,321 -> 907,972 bytes gzipped (-217 KB, -19.3%)**; the
>   detail file 304 -> 511 KB, fetched when the first sheet is laid out (on an untouched page, the idle
>   pre-warm). This is phase 1's stated purpose - first paint must not grow with the universe - and at
>   1,500 stocks it is the difference of ~650 KB on a phone's first load.
> - **Read-site audit before moving it, which corrected the plan.** The plan and the generator's comment
>   said `inp` was read by "the movers strip, the compare drawer and the calculation traces". It is read by
>   none of them. In the page: `metricDetailHtml` / `equationHtml` (a metric row, built on open),
>   `renderProvenance` (the recomputed-EV pair) and `workingsCsv` - all inside the open sheet. The build's
>   `calc_trace`, input-mismatch and capped-figure checks and `ranking_audit` run before `split_detail`.
>   Outside the page (the 11-09 rule - grep beyond the page): `sec_crosscheck` reads it through
>   `check_run_health.fold_shards`, which folds every shard generically, so the publish gate's filing check
>   was already safe; **`scripts/audit_stock.py` and three test modules read `dashboard_data.js` alone** and
>   now fold the shard back. Without that, `test_metric_lineage`'s fixture - the equation tests that hold
>   row 0.8c's 99% bar - would have **skipped** as "payload predates the published inputs" on Monday's
>   run: the 2026-11-09 failure shape exactly, a check of a moved field seeing nothing, quietly.
> - **Each reader says it is waiting** (the shard's membership rule): an open row's Calculation and
>   "Inputs, as reported" read "Loading the reported figures…" beside the scored value; the EV sentence
>   "Loading the figure used here…"; the loader redraws provenance and every open row when the file lands;
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

