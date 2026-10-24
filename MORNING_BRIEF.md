# Morning Brief - Saturday 24 October 2026, 15:49

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran yesterday |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-23T02:00:04.666022 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, APA, HST, INCY, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (35 rows, but overlapping windows are not independent; 93 rows across all horizons), newest 2026-10-15 |
| Context signals | Context signals: 12 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 423/424 within 0.5%; operating cash flow 406/410 within 0.5%; net income 396/419 within 0.5%; market value 444/484 within 5%; filings cached 0 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#3: net income 1.46x after-tax operating income); MU (#10: a period behind its filings); NEM (#12: a period behind its filings); CF (#16: net income 16% off the filing) |
| Context coverage | options usable 486/498 (12 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `5dad042 test: the research page's intervals come from the ledger, not from one audit's literals`
- `0693eda research engine: all 31 ledger rows rerun on the rebuilt records`
- `f3d757f plans: the two v2 data fixes are done; what remains of the sub-item`
- `5eedc7c wip: the log entry's body, up to the rerun`
- `9bf99fd wip: the main build's effect, measured`
- `40b2e93 wip: the presample rebuild's effect, measured`
- `789c940 wip: the research note and the log entry for the two v2 data fixes`
- `1306eb7 v2 data: carry the pre-2014 renames and each member's predecessor filings`
- `a36bf60 brief: code session 2026-10-24-weekend`
- `93f7d37 comment: the side file is ~10 KB, measured, not 2`
- `b8c592b log: correct the after-count to the measured 3109`
- `19d8435 log: the 2026-10-24 weekend session - owner item 2026-10-21 items 5 and 6, and the item closed`
- `ddf7abc the published run's inputs outlive the four-week prune`
- `5b83030 only the gated run is tradable: signals.json`
- `fd88886 wip: the page says which run, which commit and which raw fetch it came from`

## The session's own account

> 2026-10-24 (13:00 midday session) - the two free data fixes backtest v2's record was missing, and the rule I first wrote for one of them was wrong in both directions
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-24 06:00 weekend ended
> "Run complete: shipped to main" at 06:42, tag `good/2026-10-24`. Data loop published? **yes** -
> 2026-10-23 02:28, HEALTH PASS, 498 scored, all four input gates green; no 02:00 run today,
> correctly - Saturday. Evidence base at `1m` = **35 rows, newest 2026-09-22 (32 days ago, bound
> 40), 4 effective** (`_n_observations` 4; 35 is the row count and not the one to quote). Priority
> 0: `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13,
> the Factor Lab** (opened 2026-10-12, **age 12 days**) - deferred; the next experiment is 18 and
> the next non-retrospective Friday is 2026-11-06, so nothing about it is time-critical today.
> **Tests:** before **3109 passed, 1 skipped**; after **3129 passed, 1 skipped** (+20, one new
> module). Gates 2 and 4 pass; gate 3 via `tests/test_payload_parse_gate.py` (19) and the fact that
> no payload file is in this diff - `node --check` is refused by this session's tool guard, and the
> runner re-runs gate 3 itself. **`scripts/rehearse_publish.py` run and PASSED**, because the diff
> changes `SCREENER_OVERVIEW.md`, which `generate_dashboard` embeds in the page.
> **One test needed changing and the reason is worth reading** (see *Tried and rejected*):
> `test_every_headline_figure_carries_its_interval_from_the_ledger` pinned the 2026-10-20 audit's
> literal confidence interval, which a legitimate data fix moves.
> **Owner queue / rotation:** `OWNER_FOCUS.md`'s **top open item (2026-10-20), sub-item 6, first
> half** - the two free data fixes - which is also what the 06:00 session and the 10-23 13:00
> session both named as Next. No proposal is waiting for a decision
> (`write_brief.open_proposals()` returns `[]`), so section 3's first job was clear.
> 
> ### Did
> 
> **`OWNER_FOCUS.md` 2026-10-20, sub-item 6, first half: backtest v2's two free data fixes, done
> together because both are in `v2_experiments.RESULT_PATHS` and so force one rebuild and one
> rerun.** The write-up is `research/2026-10-24-v2-pre-2014-renames.md`; every figure below is in
> `research/measurements/2026-10-24-v2-data-fix-effect.json`.
> 
> **Re-measured before touching anything (the prompt's rule), and nothing had moved since
> 2026-10-20:** 48,282 member-months in 2012-2019, 31,919 built, `no_current_record` 15,108,
> `no_price` 434, `symbol_reused` 48, `no_filings` 773 - every figure identical. The 167 months the
> note attributes to an uncarried rename reproduce symbol by symbol (TYC 56, ACE 48, SAI 21, MHP 16,
> PCS 16, BFB 6, CMCSK 3, SAIC 1).
> 
> **Fix 1 - the renames the membership history cannot carry.** `Inputs.current_symbol` matches a
> month's member to today's record by SEC registrant number and nothing else, which is right: only
> the CIK separates a rename from a removal. But the revisions had no CIK column until **2014-05**,
> so a company renamed before then has no registrant number anywhere in the history and the matcher
> has nothing to match on. `universe_history.PRE_REVISION_CIKS` supplies three - BFB (BF.B written
> without the dot, so it normalises to a symbol that never meets today's BF-B), MHP (S&P Global) and
> SAI (Leidos) - each a *unique* name match against SEC's own `cik-lookup-data.txt` from the
> 2026-10-20 note, each verified here to resolve to a company in today's universe, to a target that
> was **not itself a member in any affected month**, and to a price series that covers those months.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

