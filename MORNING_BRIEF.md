# Morning Brief - Wednesday 18 November 2026, 13:48

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-18 02:00 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | HST, EXPE, VLO, MU, APA |
| Ranking's lean on growth | +0.043 (within the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 131 rows across all horizons), newest 1m run date 2026-10-16 (33 days ago, bound 40) |
| Context signals | up to 1 effective observations of 8 needed. Not enough for a t-statistic yet. |
| Inputs vs SEC filings | total assets 440/440 within 0.5%; operating cash flow 430/431 within 0.5%; net income 420/443 within 0.5%; market value 452/487 within 5%; 7 scored a quarter behind a filing over 30 days old; filings cached 0 days ago |
| Top 25, worth a look | HST (#1: net income 1.29x after-tax operating income); EXPE (#2: equity 6.2% of assets); DXCM (#8: operating cash flow 36% off the filing); PSX (#9: net income 1.28x after-tax operating income); TPR (#10: equity 5.2% of assets); CF (#17: net income 15% off the filing) |
| Data run timing | **22 min** total; slowest `fetch.tickers` 573 s, `fetch.sec_roe_history` 262 s, `fetch.retry` 168 s; **51 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (15 days) |
| Context coverage | options usable 494/499 (5 quotes-closed) - insider from SEC 499/499 - earnings variability 440 |

## What changed in the repo

- `4e34d62 NIGHTLY_LOG 2026-11-18 (13:00): where the missing filings went, the gates' results, what is next`
- `04e3f32 Research note, changelog, CLAUDE.md 0.14: where twenty companies' filings went`
- `6b9d514 wip: the SEC refresh follows a combined filing to the co-registrant whose record holds it`
- `9f638b9 brief: code session 2026-11-18`
- `0a4e4c0 NIGHTLY_LOG 2026-11-18: the synthesis, the gates' results, what is next`
- `1aecd56 The synthesis note commits to the date COMING_UP gives it`
- `e0f2c28 Synthesis note, changelog, CLAUDE.md 0.14: what the screener knows about its inputs`
- `cfeb7bc wip: the health line counts the stocks the filing check cannot see`
- `3751f4b wip: measurements - every input caveat on the page, counted; the reworded sentence seen at 1440 and 375`
- `743f929 wip: staleness is said of the statement the score read, never of the company's filings`
- `5f55b79 brief: data run 2026-11-18`
- `3aecb1a data: screener run 2026-11-18 - 499 scored, top: HST EXPE VLO MU APA`
- `fd62296 brief: code session 2026-11-17-midday`
- `e68a574 NIGHTLY_LOG 2026-11-17 (13:00): the gates' results`
- `7f6a8fb NIGHTLY_LOG 2026-11-17 (13:00): the year-to-date column, the health line, what is next`

## The session's own account

> 2026-11-18 (13:00) - QUEUE. Where twenty companies' filings went: the SEC now files a combined report under one co-registrant, and the weekly refresh follows it there
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-18_060000.log` (completed). Data loop published? yes - `logs/datarun-2026-11-18_020001.log` (completed; HEALTH: PASS). Evidence base at `1m`: 5 of 8 needed at the 1m horizon (53 rows, but overlapping windows are not independent; 131 rows across all horizons), newest 1m run date 2026-10-16 (33 days ago, bound 40). Priority 0: `allow_auto_apply` **false**. Top open roadmap item: **0.13, The Factor Lab** (age **37 days**; a standing programme, last advanced **12 days** ago per `plan/factor-lab.md` (bound 21)); also open: 0.10 42d; 0.14 35d.
> 
> **Tests:** the runner's baseline was GREEN (4,363 passed / 1 skipped). **My own baseline had one
> failure**, `tests/test_lab_reading_and_access.py::test_a_web_font_that_lands_late_is_waited_for_and_one_that_cannot_land_is_not`
> (card heights 208 against 183 at the moment it read them): 1 failed / 4,362 passed / 1 skipped in
> 4:39. It did not reproduce - the module alone, 35 passed - and it ran while I was making EDGAR
> requests on the same machine. Recorded as a flake, not fixed; the saved output is
> `logs/session-baseline-1300.txt`. After: **4,371 passed / 4 skipped / 0 failed** in 4:09 (11 new
> cases). **The three extra skips are mine and are by design:** `test_capex_lease_note`,
> `test_special_item_note` and `test_ytd_column_note` each skip, loudly, when the SEC cache on disk
> is not the one the committed payload was built from (2,446,742 rows then, 2,449,990 now). They
> skip the same way after every weekly refresh and run again once the 02:00 run publishes from the
> new cache. Gate 2 (dry-run) passes. Gate 3: no payload or page file touched. Gate 5,
> `scripts/rehearse_publish.py --changed-since origin/main`: **REHEARSAL PASSED**, files restored
> byte for byte. Gate 4 clean. The suite was run twice with the same counts, the second time (4:27)
> on the final tree with this entry already written; only this sentence changed after it.
> 
> **Owner queue / rotation:** second session today; I read the 06:00 entry first and took its
> **Next 3** (Next 1 is Friday's, Next 2 needs tomorrow's run). `open_proposals()` is `[]`; no
> branch was held back; no `auto-rescue` stash (`git stash list` is empty). `OWNER_FOCUS.md`: not
> re-read item by item - the 06:00 session did four hours earlier and no commit has touched it
> since; nothing moved. 0.13 is not deferred (12 days against a bound of 21; Friday is its day).
> Nothing was deferred for a gate or a loop. No decision of this morning's was reversed; one of its
> numbers is superseded (below).
> 
> ### Did
> 
> **The question:** this morning found 20 of 499 scored companies with nothing in the SEC
> `companyfacts` cache filed in 165 days, showed the SEC's record itself lacked them, and "did not
> establish why". `research/2026-11-18-combined-filings-under-a-co-registrant.md`.
> 
> **1. The cause, for 19 of the 20.** Followed filing by filing through EDGAR's `submissions`
> record and each filing's header (`research/measurements/2026-11-18-where-the-missing-filings-went.py`,
> 134 requests): the twenty have made **41** periodic filings since the newest in their own record,
> all XBRL, none with a single fact there. **37 of them, for 19 companies, are whole in exactly one
> co-registrant's record** - a subsidiary utility or an operating partnership, and a different one
> from quarter to quarter (Sempra's June quarter is under Southern California Gas, September's
> under San Diego Gas & Electric). Until mid-May the whole filing sat under the first registrant,
> the listed company. I did not find the rule that now picks the registrant, and found no SEC
> notice; the note says so.
> 
> **2. The moved facts are the parent's - measured, not assumed.** Every 10-Q repeats earlier
> figures as comparatives, and the parent's own record already holds those. 7,004 of 7,326
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

