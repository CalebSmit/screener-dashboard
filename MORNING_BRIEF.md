# Morning Brief - Friday 27 November 2026, 14:05

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
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-26 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, APA, MPC, SNDK |
| Ranking's lean on growth | +0.012 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 20 commit(s) changed what a reader sees, 3 methodology change(s), in the last 7 days |
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

## What changed in the repo

- `150020b NIGHTLY_LOG 2026-11-27 (13:00): gate results`
- `a530971 NIGHTLY_LOG 2026-11-27 (13:00): raw shard`
- `91134ab docs: raw shard - plan progress, inventory, owner item, CLAUDE.md DETAIL_KEYS`
- `05e000d wip: raw shard tests - audit tripwire, vol kept equal, holdings risk gap with the file held (mutation-checked), peer table waits then fills`
- `30d76ac wip: raw shard - the peer table draws waiting; five readers of the published payload fold the shard (one had been half-blind since 11-08)`
- `3b37869 wip: shard (raw) - each metric's scored value leaves first paint (693 -> 494 KB gzipped); volatility stays as vol for the two readers outside the sheet; payload regenerated from the published run (folded 0 differences)`
- `40f23fd brief: code session 2026-11-27`
- `a5a54b8 NIGHTLY_LOG 2026-11-27: gate results`
- `ec96ecd NIGHTLY_LOG 2026-11-27: retrospective - the refused run and held-back branch fixed, gate 5's list derived, one retro rule`
- `66a59ba CLAUDE.md: closed items move below the open ones; 0.10 closed; a stale v1 claim corrected`
- `16a71f4 Retrospective schedule: one rule, keyed to the last retrospective that ran`
- `42e9ef6 Gate 5's ENGINE CHANGED list is derived from what the run imports, not hand-kept`
- `c0273a9 Sheet tests settle on the detail shard too: its pct redraw moves the sections the tab test clicks`
- `bf0bdb4 Test states the reading's IV construction instead of inheriting it: the 02:00 run was refused for a correct page`
- `d1ea09c brief: data run 2026-11-27`

## The session's own account

> 2026-11-27 (13:00) - QUEUE. Universe expansion phase 1: each metric's scored value leaves first paint (693 -> 494 KB gzipped, -28.8%); volatility stays behind for My Holdings, and a peer test found half-blind since 11-08
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-27_060001.log`
> (completed; the retrospective, merged as `c47a4c7`). Data loop published? **no** -
> `logs/datarun-2026-11-27_020001.log`, HEALTH: PASS, refused by the published-claims gate; the morning
> session fixed the stale test and its branch is on `main`, so the **18:00 retry** should publish. Evidence
> base at `1m`: **59 rows, newest `run_date` 2026-10-26 (32 days ago, bound 40), 5 effective** of 8.
> Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 46 days**; a standing programme, last advanced **2 days** ago, bound 21); also open 0.14
> (44 d). Shipped to readers: 20 commits, 3 methodology changes in the last 7 days.
> **Tests:** before **4574 passed / 1 skipped**; after: under "Gates".
> **Owner queue / rotation:** QUEUE. No stash, no owner-session marker, no open proposal
> (`write_brief.open_proposals()` is empty). The morning's Next 1 is the 18:00 retry (not yet due); its Next
> 2 was this slot's work - phase 1's `raw` slice (`plan/universe-expansion.md`), under the owner item
> "Beyond the S&P 500". The top owner item (Search Console, a byline, analytics) is the owner's to decide;
> nothing there for a session.
> 
> ### Did
> - **`raw` - each metric's value as scored - moved to `dashboard_detail.js`. First paint 693,214 ->
>   493,871 bytes gzipped (-199 KB, -28.8%).** Regenerated from the published run (`runs/c31edb9b89dc`,
>   11-26 02:00) with this generator: folded whole, **0 differences** from the committed payload (the
>   context file deliberately not rewritten, as on 11-26). `logs/regen_raw.py`.
> - **Read-site audit, page.** Readers of a stock's `raw`: the peer table (`buildPeerRow`, six of nine
>   columns), `categoryWorkings`, `peerContext`, `metricDetailHtml`, `equationHtml`, and the CSV through
>   the workings rows - all the open sheet - plus **two outside it, both of one figure, volatility**: My
>   Holdings' concentration line (live at first paint for a reader with saved names) and the context
>   panel's "Realised, past year" row (drawn when the context file lands, which can be first). The plan
>   offered "wait" or "keep volatility as its own small field"; I took the second, because a first-paint
>   surface should not wait on a 951 KB file for one number. `vol` (6.3 KB gzipped, 497 stocks) is copied
>   by `split_detail` (`FIRST_PAINT_RAW`), never computed twice; verified equal to `raw.volatility` for
>   every stock that has one; both readers go through `volOf`, which falls back to `raw` so hand-built
>   payloads render as before.
> - **The peer table, which had six figures at first paint, now has none until the file lands.** It is
>   drawn anyway - ticker and rank are first paint - with every figure and the peer median "…"; the "No
>   peer figures were published" sentence is kept for a file that landed with nothing. The workings
>   needed nothing: they already say "Loading the sector percentiles…" for `pct`, in the same file.
>   `metricDetailHtml` gives a stock without `raw` an empty one rather than throwing, keeping its
>   `claims.py`-registered sentence templates verbatim.
> - **Outside the page (the 11-09 rule: grep beyond the page).** `write_brief.context_coverage` counted
>   earnings variability off unfolded `raw` - it would have printed 0 every morning - and folds now.
>   Five readers failed loudly in the suite, not silently: `test_bank_pb_roe_identity` (0 banks),
>   `test_peer_row_one_path`, `test_holdings_concentration`'s source check, the growth-structure
>   measurement script (run by `test_growth_category`), and three membership tests that pinned "raw not
>   yet moved". All fold now. **Found on the way:** `test_peer_row_one_path`'s empty-state premise read
>   the payload unfolded, so since `financials` moved on 11-08 it had counted three of its nine columns as
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

