# Morning Brief - Sunday 22 November 2026, 13:49

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
| Shipped to readers | 18 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
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

- `af2a043 docs: eq4/roe5 shard slice - plan progress, inventory, CLAUDE.md, owner item, NIGHTLY_LOG 13:00`
- `739bc89 shard (eq4): the calendar's phone line waits too, rather than vanishing until the file lands`
- `3ab8056 shard (eq4, roe5): the quarters and ROE years behind four metrics' workings leave first paint - 908 -> 801 KB gzipped, -11.8%`
- `f585248 brief: code session 2026-11-22-weekend`
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

## The session's own account

> 2026-11-22 (13:00) - QUEUE. Universe expansion phase 1, next slice: `eq4` and `roe5` leave first paint - 908 -> 801 KB gzipped, -11.8%; the first shard field with a reader outside the sheet
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-22_060001-weekend.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-20_180003.log` (completed; HEALTH: PASS);
> weekend, no newer run expected. Evidence base at `1m`: **55 rows, newest 2026-10-20 (33 days ago, bound
> 40), 5 effective of 8**. Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the
> Factor Lab** (age **41 days**; standing programme last advanced **16 days** ago, bound 21 - not late);
> also open 0.10 46d, 0.14 39d. Shipped to readers: 18 commits, 8 methodology changes in 7 days.
> **Tests:** baseline **4,461 passed / 1 skipped** (dry run OK). After: **4,467 / 1** at the first
> commit (+6, the new module); the final tree re-run is under Gates below.
> **Owner queue / rotation:** `git stash list` empty; no owner-session marker; no proposal waiting
> (`open_proposals()` = []). Carried on from today's 06:00 entry, Next 3 (phase 1's next slice). Its
> Next 1 and 2 are dated for Monday (see Set aside).
> 
> ### Did
> - **Re-measured before choosing** (`research/measurements/2026-11-08-stock-detail-field-weights.py`
>   on the committed payload): after `inp` the heaviest first-paint fields were `raw` 23.5%, `pct`
>   14.0%, `summary` 12.1%, then **`roe5` 6.9% and `eq4` 5.2%** - not on the plan's list, and moveable
>   without the redesign `raw`/`pct`/`summary` need. So this session moved those two.
> - **Read-site audit** (page and outside it): `roe5` - `metricDetailHtml` only (earnings variability's
>   five-year ROE table). `eq4` - `metricDetailHtml` (the three surprise metrics' quarter table) **and
>   `renderReporting`**, the Reporting soon calendar's *Last quarter* column: a collapsed section, not
>   the sheet. It qualifies under the shard rule's second half ("can say honestly that it is waiting"):
>   it prints "…" while the file is in flight - as its options column already does for the context
>   file - **requests the file itself** when drawn (so the "…" is not left waiting on the pre-warm by
>   chance), and the loader redraws it. Outside the page nothing in `scripts/` or any scoring module
>   reads either field; `test_sec_fundamentals`'s published-ROE check read `dashboard_data.js` alone
>   and would have **skipped** ("payload predates earnings variability") - it now folds the shard, as
>   `test_metric_lineage`, `test_methodology_claims_audit` and `test_dashboard_printed_figures` already did.
> - **Waiting states:** an open surprise row reads "Loading the quarterly figures…", earnings
>   variability "Loading the five years of filings…" (instead of "Needs all five years - not
>   computed", which would have been false); both have failure wording. `detailPending(s, k)`
>   generalises the morning's `inputsPending`.
> - **Published files regenerated** from the published run (2026-11-20 18:00), `logs/regen_eq4.py`:
>   folded whole, **0 differences**; 498 stocks carry `eq4` and 497 `roe5` in the shard, 0 in first paint.
>   First paint **907,972 -> 801,225 bytes gzipped**; the detail file 511 -> 612 KB.
> - **Looked at it** (`logs/shot_eq4.py`, first ranked stock with quarters, file held then released), 1440
>   and 375: the row waits then shows the four quarters and "Median of the 4 valid surprises (...) =
>   10.9%"; the calendar shows "…" then "+18.4%" etc. **At 375 it caught a defect in my first version:**
>   the phone line "Last quarter vs estimate" under each company was *omitted* while waiting and appeared
>   later - fixed to print "…" like the desktop column, with a test. No overflow (scrollWidth = viewport),
>   no page errors.
> - Tests: `tests/test_quarters_in_detail_shard.py` (6): per-field read-site tripwire (a new `.eq4`/`.roe5`
>   reader outside the audited functions fails), each reader checks the in-flight state, the calendar asks
>   for the file and the loader redraws it, the browser round trip for both. `test_health_reads_every_shard`'s
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

