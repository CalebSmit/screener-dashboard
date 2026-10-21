# Morning Brief - Wednesday 21 October 2026, 14:30

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

## What changed in the repo

- `0055951 log: the 2026-10-21 13:00 session - the bank P/B-ROE decision and the guard it replaced`
- `732ad0a research engine: rerun all 30 after the config.yaml comment (0 of 25,460 values moved)`
- `55b9291 record: the bank P/B-ROE decision - changelog, owner queue, section order`
- `859a7dd bank P/B-ROE proposal ACCEPTED: state the pairing on the page, guard the measured loading`
- `ba9b400 brief: code session 2026-10-21`
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

## The session's own account

> 2026-10-21 (13:00 session) - QUEUE. The bank P/B-ROE proposal reviewed and ACCEPTED - and the guard it asked for could not see the one change the note itself calls the mildest route to a P/B tilt
> 
> **Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 session ends
> "Run complete: shipped to main" at 08:48 (`logs/nightly-2026-10-21_060000.log`). Data loop
> published? **yes** - `logs/datarun-2026-10-21_020001.log` ends "Data loop complete" at 02:26,
> PUBLISHED CLAIMS PASS (1214 tests). Evidence base at `1m` = **33 rows, newest 2026-09-18 (33 days
> ago, bound 40), 4 effective** - inside the bound, no action. Priority 0: `allow_auto_apply`
> **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the Factor Lab** (opened
> 2026-10-12, **age 9 days**) - not worked; its day is Friday 2026-10-23, and the owner queue
> outranks it today.
> **Tests:** before **2969 passed / 1 skipped**, after ****2980 passed / 1 skipped**** - the delta is exactly the one new
> module, no failures and nothing removed. Gates 2 and 3 pass (dry-run exit 0; `index.html` 642,631
> bytes; both payloads `node --check` clean with their opening assignments).
> **Owner queue / rotation:** took `OWNER_FOCUS.md`'s 2026-10-20 item **1**, which named this as "the
> next session's first job", and it was the only OPEN proposal in the morning brief. **Deferred:**
> item 2 (second-read lab experiments 16 and 17), item 3 (the next 12 claim mutations) and the
> Factor Lab.
> 
> ### Did
> 
> **Reviewed `research/2026-10-20-bank-quality-valuation-overlap.md` section 8 and accepted it -
> with the guard replaced, which is the part only a second reader produces.**
> 
> The note's question was whether 35 points of a bank's Quality score double-count two of its
> Valuation inputs. Since the two changes that shipped on 2026-10-20 put ROE and P/B on nearly one
> book, a bank's ROE **is** its earnings yield times its P/B. The note's answer - no, because ROE
> duplicates earnings yield but *cancels* price-to-book, so the four weights net to "high ROE at a
> low P/B", the construction bank analysts and index builders use on purpose - holds up. I
> re-measured it on a different run and it reproduces. **No weight, metric or threshold changed.**
> 
> **Re-measured on run `4dc4acf1d780` (today 02:00) against the note's `7c12b43a07b6` (yesterday
> 13:21).** Both measurement scripts rerun; the rescoring baseline reproduces **501 of 501**
> published ranks either way:
> 
> | | Note's run | My run |
> |---|---|---|
> | **(a)** R-squared, log ROE on log E/P and log P/B (bar 0.90) | 0.978 | **0.978** |
> | **(b)** measured loading on P/B's rank | -0.067 | **-0.069** |
> | loading on E/P's rank - criterion (b)'s bar is half of it | +0.284 | **+0.283** |
> | status quo against the P/B-ROE (Wilcox) residual | -0.890 | **-0.892** |
> | the six alternatives against it | -0.85 / -0.72 / -0.84 / -0.79 / -0.84 / -0.88 | **-0.848 / -0.724 / -0.839 / -0.796 / -0.838 / -0.882** |
> 
> Every figure to the third decimal, and the status quo is still the closest of the seven
> constructions to the combined signal. Both refutation criteria hold.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

