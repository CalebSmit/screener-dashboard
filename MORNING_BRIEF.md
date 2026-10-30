# Morning Brief - Friday 30 October 2026, 02:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-30 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | EXPE, VLO, MU, HST, APA |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (40 rows, but overlapping windows are not independent; 105 rows across all horizons), newest 1m run date 2026-09-29 (31 days ago, bound 40) |
| Context signals | 16 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 375/376 within 0.5%; operating cash flow 361/365 within 0.5%; net income 353/372 within 0.5%; market value 439/484 within 5%; 14 scored a quarter behind a filing over 30 days old; filings cached 7 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#5: net income 1.46x after-tax operating income); PSX (#8: net income 1.28x after-tax operating income); CF (#14: net income 16% off the filing); WDC (#25: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 663 s, `fetch.retry` 140 s, `context.sec_insider` 96 s; **42 tickers rate-limited** |
| Context coverage | options usable 490/499 (9 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **fill an empty newest quarter from the source's own trailing series (9 of 15 lagging stocks, Amazon among them); answer section 4 first.** - open 1 day - `research/2026-10-29-ttm-fills-the-empty-quarter.md`

## What changed in the repo

- `5c0cfb0 data: screener run 2026-10-30 - 499 scored, top: EXPE VLO MU INCY HST`
- `f3c5aa8 log: 2026-10-29 owner-run, last before 02:00 - unscored members, glossary, brief`
- `a5483e8 plan: dashboard inventory - today's surfaces and the unscored payload key`
- `05d636c page: an S&P 500 member the run could not score is named and told why`
- `1c026d6 brief: the run time to the minute, and the context row without its own label twice`
- `b072c5d methodology: the glossary spells out the abbreviations the page uses`
- `e8e7bbd test: the dead-badges flake will say which of its two causes it was`
- `80cc059 log: the dead-badges flake's traceback, and what it rules out`
- `3c83b1e log: 2026-10-29 owner-run late - lab signed zeros; the empty quarter's trailing series`
- `93fbd35 brief: count the stocks scored a quarter behind their filings`
- `e721248 research: the quarter the source leaves empty is in its own trailing series`
- `7d1c3d5 lab: a figure that rounds to zero carries no sign`
- `8868d64 log: 2026-10-29 owner-run night - three defects from sweeps of the rendered site`
- `b7e1698 methodology: plain words for "NaN" on the public page`
- `ad3e07d page: a small percent in the context panels keeps its sign; every panel swept`

## The session's own account

> 2026-10-29 (owner-run, last before the 02:00 run) - unscored members named; glossary abbreviations; brief polish
> 
> Health as the evening entry. Owner: "keep finding improvements please before the 2 am run".
> 
> - **An S&P 500 member the run could not score is named and told why.** HONA and VYLR (in the
>   index, unscored on 2026-10-29: too little data from the source) were absent from the payload, so a
>   search or a saved holding said a company "that has left the index, or whose data could not be
>   fetched" is not in the ranking - the first half false. The run already records them
>   (`universe.json` "failed"); the generator publishes them as `unscored` (`_run_unscored`, display
>   only, `{}` on any failure) and `unscoredMemberText` feeds the search, the palette and My
>   Holdings. `tests/test_unscored_members.py` (7). **This changes the payload the 02:00 run
>   publishes (+72 bytes); `rehearse_publish.py` PASSED against run `a08c2741a3ca`**, and the
>   regenerated block read `{"HONA": "Honeywell Aerospace Inc.", "VYLR": "Vylor Inc."}`.
> - **The methodology glossary spells out its abbreviations** - EPS appeared 35 times, ROE 23, P/B 21,
>   ROIC 14, most never expanded. One entry, names only (no formulas, so no new computational claim);
>   generator edited, overview regenerated, page rebuilt, payload byte-identical.
> - **Brief polish:** the run time reads "2026-10-29 02:00", not the raw ISO stamp with microseconds;
>   the context row no longer repeats its own label.
> - `plan/dashboard-inventory.md` updated (rule 9).

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

