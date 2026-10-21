# Morning Brief - Wednesday 21 October 2026, 08:48

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-21T02:00:03.974680 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, INCY, HST, VLO |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (33 rows, but overlapping windows are not independent; 89 rows across all horizons), newest 2026-10-13 |
| Context signals | Context signals: 10 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 407/411 within 0.5%; net income 398/421 within 0.5%; market value 447/487 within 5%; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#11: net income 16% off the filing); MU (#14: a period behind its filings) |
| Context coverage | options usable 492/501 (9 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **no weight or metric change; state the bank P/B-ROE pairing as the design and guard it: (1) a paragraph in `run_screener.build_screener_overview()`'s "Bank-specific scoring" section saying a bank's ROE is its earnings yield times its P/B, so its Valuation and Quality scores are read together - the 44 points rank it on ROE relative to P/B (to first order, its earnings yield) and on equity per dollar of assets, with no net weight on P/B by itself - registered in `claims.py` as `overview.bank_pb_roe_identity`; (2) the two table cells calling P/B and ROE "THE key" bank metrics reworded to say each is read against the other; (3) `tests/test_bank_pb_roe_identity.py`, in that claim's `checked_by`, failing if the first-order net loading on log P/B computed from `config.yaml` (factor weights x bank weights: -P/B + ROE + ROA, in composite points; 0.0 today) leaves +-1.5 points, or if on the published payload log ROE regressed on log earnings yield and log P/B has R-squared below 0.90 across the bank-like stocks (0.978 today); (4) a `config.yaml` comment above `bank_metric_weights` naming that constraint and this note.** - open 1 day - `research/2026-10-20-bank-quality-valuation-overlap.md`

## What changed in the repo

- `5e4ccd0 gitignore: a session's .scratch/ folder`
- `0e5c04a log: the end-to-end verification, the shape bug it caught, and the gate results`
- `1fd3cb4 page: do not print 'Alphabet Inc..' - strip a trailing stop from the company name`
- `2411b85 fix: read the fetch in both of its shapes (share_classes.fetch_maps)`
- `268ead4 log: the 2026-10-21 session entry and the owner queue's progress line`
- `1d16900 research engine: rerun after the config change (30 of 30)`
- `328235d research engine: rerun after the config change (27 of 30)`
- `9590f13 research engine: rerun after the config change (22 of 30)`
- `f9f738d research engine: rerun after the config change (17 of 30)`
- `9eb07be research engine: rerun after the config change (12 of 30)`
- `246a2af research engine: rerun after the config change (7 of 30)`
- `2b7f1cf research engine: rerun after the config change (3 of 30)`
- `fbafe2d cache: bump scoring_schema to 15 and key the cache by share_classes`
- `ff2838d methodology: one company, one ranked line - changelog and the proposal's decision`
- `b5c7f97 wip: the review's measurements, committed and reproducible`

## The session's own account

> 2026-10-21 (06:00 session) - QUEUE. The share-class proposal reviewed and ACCEPTED - but the designated line is declared, not re-picked: Alphabet is 179 on GOOGL and 101 on GOOG
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-20 13:00 ended "shipped to
> main" (`logs/nightly-2026-10-20_130002-midday.log`), and two owner-run sessions followed it.
> Data loop published? **yes** - `logs/datarun-2026-10-21_020001.log` ends "Data loop complete",
> 02:26. Evidence base at `1m` = **33 rows, newest 2026-09-18 (33 days ago, bound 40), 4 effective**
> - inside the bound, no action. Priority 0: `allow_auto_apply` **false**, 4 of 8 effective
> observations. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 9 days**) -
> not worked; its day is Friday 2026-10-23, and the owner queue outranks it today.
> **Tests:** before **2922 passed / 1 skipped**, after **2969 passed / 1 skipped** - +47, which is
> exactly the two new modules, no failures and nothing removed. Gates 2 and 3 pass (dry-run exit 0;
> `index.html` 642,631 bytes; both payloads `node --check` clean with their opening assignments).
> **Owner queue / rotation:** took `OWNER_FOCUS.md`'s 2026-10-20 item **1**, the first OPEN proposal.
> Wednesday's nominal SYNTHESIS focus was displaced by it, which the prompt requires - and the work
> was a synthesis question anyway: what the *population* of a sector percentile should be. **Deferred:**
> the second open proposal (the bank P/B-ROE guard, opened 2026-10-20) and the Factor Lab.
> 
> ### Did
> 
> **Reviewed `research/2026-10-19-share-classes.md` and accepted it, with one change the review
> measured and the note did not have.** The note is good: it inventoried every multi-class company,
> established from MSCI, S&P DJI and FTSE Russell that fundamentals are company facts and prices are
> line facts, rescored six candidate rules, and ruled out the cheap one (filling a line's *missing*
> analyst fields puts FOX at **#6**, into the top 25, on a fallback growth figure of +49.6%). What it
> proposed was S&P DJI's approach B - score the company once on its **Designated Listing** - with the
> line picked from **each run's own `avg_daily_dollar_volume`**.
> 
> **That last part is not S&P DJI's rule and is not stable, which is what a second reader is for.**
> S&P changes a Designated Listing only when the other line leads on *both* criteria by more than
> 20%. Measured over 89 month-ends (2019-06 to 2026-10, 63-day average daily dollar volume):
> 
> | | Alphabet | Fox | News Corp |
> |---|---|---|---|
> | Months the two lines are within S&P's 20% margin | **31 of 89 (34.8%)** | 0 | 0 |
> | Closest the two ever came (volume ratio) | **1.003** | 2.077 | 2.193 |
> | Times the busier line actually changed | **4** (2020-09, 11, 12; 2021-01) | 0 | 0 |
> 
> And it matters: rescoring the 2026-10-21 run both ways, **Alphabet ranks 179 scored on GOOGL and
> 101 scored on GOOG** - 78 places, with a different ticker in the published ranking - because the
> two lines carry different analyst records (the note's own section 7: the source serves GOOGL's
> quarterly "actual EPS" on a GAAP basis and GOOG's on the analysts' basis). A per-run recomputation
> would have moved a company 78 places on a liquidity tiebreak that is a coin flip a third of the
> time, with nothing in the changelog to explain it.
> 
> **So the choice of line is declared in `config.yaml` (`share_classes.designated: GOOGL, FOXA,
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

