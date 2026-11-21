# Morning Brief - Saturday 21 November 2026, 13:39

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-20 18:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, APA, SNDK |
| Ranking's lean on growth | +0.054 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 15 commit(s) changed what a reader sees, 8 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (55 rows, but overlapping windows are not independent; 135 rows across all horizons), newest 1m run date 2026-10-20 (32 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 458/458 within 0.5%; operating cash flow 443/444 within 0.5%; net income 432/456 within 0.5%; market value 452/487 within 5%; 2 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#7: net income 1.29x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); TPR (#9: equity 5.2% of assets); DXCM (#11: operating cash flow 36% off the filing); ES (#18: a period behind its filings); CF (#23: net income 15% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 644 s, `context.options_pass` 182 s, `fetch.retry` 149 s; **46 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (12 days) |
| Context coverage | options usable 494/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-23 (in 2 days) - Build residual momentum in place of `jensens_alpha` (ACCEPTED 2026-11-21, weight 25 unchanged) in a 06:00 session (0.15's section 4 is committed - done 2026-11-21): section 9's build list - 37-month history on the same request with the Risk inputs trimmed to 13, `^SP500TR` to 4y, lineage/equation/claims, backtest v2, the engine rerun, a throwaway full run before the changelog entry. `research/2026-11-20-jensens-alpha-in-momentum.md`
- 2026-11-24 (in 3 days) - Build the implied-volatility inversion ACCEPTED on 2026-11-04 and dropped from every Next list since 11-05 (found 2026-11-19). Section 1d's build order: exDividendDate into the core .info fetch, a throwaway full run, QuantLib pinned and installed by a scheduled session (never from the desktop app: its AppData is sandboxed). A 06:00 session with nothing else in it. `research/2026-11-03-open-source-quant-stack.md`
- 2026-12-04 (in 13 days) - The option signals' (expected move, ATM volatility, skew) first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. `METHODOLOGY_CHANGELOG.md`

## Things that needed attention

- ok   ATTENTION - 1 stock(s) now carry a market value implying a share count over 10% from the one their filings report, which the published page does not: SNDK +639.2%. Usually a completed acquisition or a split: the market value is right and every valuation ratio built on it has moved, while the earnings behind those ratios have not. Not a refusal - see research/2026-11-09-share-counts-and-corporate-actions.md.

## What changed in the repo

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
- `9250a6a brief: data run 2026-11-20`
- `81f1d48 data: screener run 2026-11-20 - 499 scored, top: EXPE VLO MU APA SNDK`
- `072a500 brief: code session 2026-11-20-midday`
- `a467255 NIGHTLY_LOG 2026-11-20 (13:00): gates recorded`

## The session's own account

> 2026-11-21 (13:00) - QUEUE. 0.15 closed: the Track record tile now leads with "too short to judge", the return beneath it
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-21_060001-weekend.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-20_180003.log` (completed; HEALTH: PASS);
> no 02:00 run on a Saturday, so nothing newer is expected. Evidence base at `1m`: **55 rows, newest
> 2026-10-20 (32 days ago, bound 40), 5 effective of 8**. Priority 0: `allow_auto_apply` **false**. Top
> open roadmap item: was **0.15** (age **2 days**) - **closed by this session**; the line now reads **0.13,
> the Factor Lab** (age 40 days, a standing programme last advanced 15 days ago, bound 21 - not late);
> also open 0.10 45d, 0.14 38d. Shipped to readers: 15 commits, 8 methodology changes in 7 days.
> **Tests:** baseline 4,455 passed / 1 skipped; dry run OK. After: **4,455 passed / 1 skipped** (one
> existing browser test extended, none added); dry run OK; rehearsal (gate 5): see Gates below.
> **Owner queue / rotation:** `git stash list` empty; no owner-session marker. Carried on from the 06:00
> entry's Next 2 (Next 1 is dated for the 11-23 06:00 slot, Next 3 are weekday 06:00 / Friday items). No
> open proposals waiting (the morning session decided both).
> 
> ### Did
> - **The Track record tile leads with the lesson** (`research/2026-11-19-what-the-ranking-is-for.md`
>   section 5, item 2 - the page consequence of answer D, written before the measurement). While the
>   record's day-by-day lead over RSP is under two standard errors (today 0.1):
>   - the tile's heading reads **"Track record: too short to judge"** instead of "Track record";
>   - **"It cannot yet tell skill from luck: 0.1 standard errors, where two is the usual bar"** sits
>     *above* the +4.5% (it used to sit under the sub-line, after the number);
>   - the return and both benchmarks stay, as the note requires - hiding the figure would be worse.
>   Above two the tile reverts to "Track record" with no caveat (the existing branch, now also asserting
>   the heading). `generate_dashboard.contextMetas` (JS), `.kpi-caveat-lead` CSS. Presentation only:
>   `scripts/rebuild_page.py` reported the payload byte-identical.
> - **Why this is an improvement:** the answer to 0.15 (pre-registered 11-19, measured 11-21) is that the
>   ranking carries no return claim; the tile was the one place on the first screen that led with a
>   return in the page's largest numerals. Its own figure (0.1 SE) says that return is indistinguishable
>   from luck, so the page now says that first. The figure is the panel's own (claim
>   `context.track_luck`), not a new computation.
> - **Test:** `tests/test_page_fourth_pass.py::test_the_track_record_tile_states_the_record_is_short_in_the_panels_figure`
>   now also pins the heading, a single caveat, and that the caveat precedes `.kpi-value`
>   (`compareDocumentPosition`). It ran (not skipped) against the published payload. Claim caveat in
>   `claims.py` rewritten to the new wording and order.
> - **Looked at it at 1440 and 375 px** (`logs/design/track-tile-*.png`): desktop 269x209 - heading on two
>   uppercase lines, amber-ruled caveat, then +4.5%, then the benchmarks; phone 171x243 - same order,
>   wraps cleanly; no horizontal overflow at either width.
> - **0.15 CLOSED** in CLAUDE.md (heading marked, so `health_numbers.open_items` drops it - verified, the
>   top item is now 0.13), with the constraint that survives: no surface presents the ranking's return as
>   a result. Note section 5 and `OWNER_FOCUS.md` say both page changes shipped;
>   `plan/dashboard-inventory.md` updated. The OWNER_FOCUS 2026-11-19 item stays open: the IV inversion
>   and residual momentum builds are still in it.
> 
> ### Evidence / research
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

