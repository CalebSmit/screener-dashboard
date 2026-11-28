# Morning Brief - Saturday 28 November 2026, 06:56

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## Coming up, only you can do it: the Claude CLI login runs out 2026-12-03 09:01 (in 5 days)

The code sessions run through the Claude Code CLI on this PC, and its login
lasts about 30 days. When it runs out, every scheduled session fails within
seconds until you sign in again. Doing it early loses nothing - open a
terminal on the PC and run:

    claude.cmd auth login

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-27 18:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, APA, MPC, HST |
| Ranking's lean on growth | +0.013 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Shipped to readers | 22 commit(s) changed what a reader sees, 3 methodology change(s), in the last 7 days |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (59 rows, but overlapping windows are not independent; 147 rows across all horizons), newest 1m run date 2026-10-26 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 459/459 within 0.5%; operating cash flow 445/446 within 0.5%; net income 433/457 within 0.5%; market value 452/487 within 5%; 5 scored a quarter behind a filing over 30 days old; 1 with no filing in the cache for 165 days; filings cached 2 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#1: equity 6.2% of assets); HST (#4: net income 1.29x after-tax operating income); PSX (#10: net income 1.28x after-tax operating income); CF (#14: net income 15% off the filing); DXCM (#16: operating cash flow 36% off the filing); TPR (#18: equity 5.2% of assets); ES (#22: a period behind its filings) |
| Data run timing | **21 min** total; slowest `fetch.tickers` 608 s, `context.options_pass` 190 s, `snapshot` 172 s; 0 tickers rate-limited |
| Code-session login | **runs out 2026-12-03 09:01 (in 5 days)** |
| Context coverage | options usable 447/499 - insider from SEC 499/499 - earnings variability 440 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-12-04 (in 6 days) - The expected move's first one-month reading: readings before 2026-11-04 recorded no price and are excluded, not spliced. (The ATM volatility and the skew restart under our own construction from the 2026-11-27 log - see 2026-12-28.) `METHODOLOGY_CHANGELOG.md`
- 2026-12-07 (in 9 days) - The last Monday before the year-to-date-column decision of 2026-12-15: research what vendors and index providers do with a hole in a company's quarterly series. 26 stocks were scored on last fiscal year's flows for that reason on 2026-11-18, and it is the same question as DexCom's (may a filing do more than sit beside a scored input) at 26 times the size. Section 4; no design is proposed there. `research/2026-11-18-synthesis-what-the-screener-knows-about-its-inputs.md`

## Things that needed attention

- ATTENTION - CLI LOGIN EXPIRES 2026-12-03 09:01 (in 5 days) - after that every scheduled code session fails until the owner runs `claude.cmd auth login` in a terminal and signs in; doing it before then loses nothing.
- BASELINE main was RED on arrival: 1 pre-existing failure(s). The session did not cause these:
- tests/test_every_panel_reads_clean.py::test_no_panel_prints_a_value_the_page_could_not_format
- BASELINE fixing a red main outranks the day's focus - gate 1 below will refuse the merge until it is green.
- full output: C:\Users\smitc\OneDrive\Documents\Screener\logs\nightly-2026-11-28_060001-weekend.baseline.txt

## What changed in the repo

- `38dca07 NIGHTLY_LOG 2026-11-28: gate results`
- `a40b635 docs: summary shard - plan progress, inventory, CLAUDE.md, owner item, log`
- `57dc0e3 wip: summary shard tests - read-site tripwire, held-file browser tests, folds`
- `2147355 wip: summary leaves first paint (hsum keeps the holdings' six kinds)`
- `1cd9a88 fix: case-against trend gap keeps decimals to its first significant figure`
- `a757fd5 brief: data run 2026-11-27`
- `ad76458 data: screener run 2026-11-27 - 499 scored, top: EXPE VLO APA MPC HST`
- `a6665cf brief: code session 2026-11-27-midday`
- `150020b NIGHTLY_LOG 2026-11-27 (13:00): gate results`
- `a530971 NIGHTLY_LOG 2026-11-27 (13:00): raw shard`
- `91134ab docs: raw shard - plan progress, inventory, owner item, CLAUDE.md DETAIL_KEYS`
- `05e000d wip: raw shard tests - audit tripwire, vol kept equal, holdings risk gap with the file held (mutation-checked), peer table waits then fills`
- `30d76ac wip: raw shard - the peer table draws waiting; five readers of the published payload fold the shard (one had been half-blind since 11-08)`
- `3b37869 wip: shard (raw) - each metric's scored value leaves first paint (693 -> 494 KB gzipped); volatility stays as vol for the two readers outside the sheet; payload regenerated from the published run (folded 0 differences)`
- `40f23fd brief: code session 2026-11-27`

## The session's own account

> 2026-11-28 - WEEKEND. A red baseline first (JPM's case-against line read "−0.00%"), then universe expansion phase 1: the "Why it ranks here" sentences leave first paint (494,961 -> 426,815 bytes gzipped, -13.8%), the six kinds My Holdings prints stay as `hsum`
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-27_130001-midday.log`
> (completed). Data loop published? yes - `logs/datarun-2026-11-27_180001.log` (the 18:00 retry; completed,
> HEALTH: PASS, "Data loop complete", brief published). Evidence base at `1m`: **59 rows, newest `run_date`
> 2026-10-26 (33 days ago, bound 40), 5 effective** of 8 (147 rows across all horizons). Priority 0:
> `allow_auto_apply` **false**. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 47
> days**; a standing programme, last advanced **3 days** ago, bound 21); also open 0.14 (45 d). Shipped to
> readers: 20 commits, 3 methodology changes in the last 7 days.
> **Tests:** before **4581 passed / 1 failed / 1 skipped** (the runner's baseline agrees: `BASELINE main was RED
> on arrival: 1 pre-existing failure`); after: under "Gates".
> **Owner queue / rotation:** WEEKEND - the top open item. No stash, no owner-session marker, no held-back
> branch, no open proposal (`write_brief.open_proposals()` is `[]`). A red baseline outranks the focus, so
> it came first. Then the owner item "Beyond the S&P 500" (phase 1, `plan/universe-expansion.md`), carrying
> on from yesterday 13:00's Next 2. The top owner item (Search Console, a byline, analytics) is the
> owner's to decide; nothing in it for a session.
> 
> ### Did
> - **The red baseline, fixed (a data-driven defect, not a code change).**
>   `tests/test_every_panel_reads_clean.py` failed on arrival: the 11-27 18:00 run published JPM closing
>   **0.004%** under its 200-day average, and The case against printed "Its last close is **−0.00%**
>   against its 200-day average of $322.97" - a signed zero, the exact class the 10-28 BXP fix (one decimal
>   -> two) was meant to end. Two decimals were still too few. The gap now keeps decimals until its first
>   significant figure shows (`gap >= 0.1 ? 1 : max(2, -floor(log10(gap)))`): JPM reads −0.004%, UAL's
>   0.0098% −0.010%. The template's key moved in `test_claims_register` (claim `drilldown.case_against`) and
>   in `test_percent_near_zero`'s exemption; `test_case_against`'s near-zero test gained three cases
>   (0.004%, 0.00001%, 0.0098%). Page rebuilt with `scripts/rebuild_page.py` (payload byte-identical).
>   Commit `1cd9a88`.
> - **`summary` moved to `dashboard_detail.js`; first paint 494,961 -> 426,815 bytes gzipped (-68 KB,
>   -13.8%).** Measured first (`logs/summary_weights.py`): `summary` 69.6 KB gzipped, of which the six
>   `HOLDINGS_FACTS` kinds 16.2 KB. Regenerated from the published run (`logs/regen_summary.py`, copied from
>   11-27's): folded **0 differences** apart from 499 new `hsum` keys, each equal to its stock's summary
>   filtered by `HOLDINGS_FACTS`; the context file came out byte-identical.
> - **Read-site audit, page.** Three readers: `renderSummary` (the sheet, every kind), `caseItems` (the
>   sheet; flags, confidence, change, input_churn, earnings - five of the six - and `target`), and My
>   Holdings' `holdingCard` (first paint for a reader with saved names; the six kinds). The six stay as
>   `hsum`, copied by `split_detail` in the summary's own order. All three read through `factsOf` (whole
>   list, else `hsum`), so a hand-built payload renders as before. Why it ranks here says "Loading why it
>   ranks here…" until the file lands (or that it could not be loaded); The case against waits only for its
>   target line ("Loading the analyst-target line…", appended to the context wait rather than overwriting
>   it); the loader redraws both. **The page's `HOLDINGS_FACTS` is now written from
>   `generate_dashboard.HOLDINGS_FACTS`**, the tuple the split filters by - two hand-kept lists that had to
>   agree became one.
> - **Outside the page** (the 11-09 rule): no script or brief reads `summary`; two tests read it off the
>   published page and now fold or wait (`test_case_against`'s `data` fixture folds the shard, its page and
> ...

---

If a run says **refused by its publish gates** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

