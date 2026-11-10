# Morning Brief - Tuesday 10 November 2026, 02:26

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-11-06 18:34 |
| Stocks scored | 499 |
| With a price | 499/499 |
| With an analyst target | 497/499 |
| Top 5 | VLO, EXPE, MU, MPC, DXCM |
| Ranking's lean on growth | +0.052 (OUTSIDE the review's +-0.05 band; 2026-10-13 read -0.009) |
| Evidence for weight changes | 5 of 8 needed at the 1m horizon (45 rows, but overlapping windows are not independent; 114 rows across all horizons), newest 1m run date 2026-10-06 (35 days ago, bound 40) |
| Context signals | 20 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 438/439 within 0.5%; operating cash flow 407/424 within 0.5%; net income 393/433 within 0.5%; market value 452/487 within 5%; 0 scored a quarter behind a filing over 30 days old; filings cached 0 days ago - **see the ATTENTION note below** |
| Top 25, worth a look | EXPE (#2: equity 6.2% of assets); HST (#5: net income 1.29x after-tax operating income); DXCM (#6: operating cash flow 36% off the filing); PSX (#9: net income 1.28x after-tax operating income); TPR (#10: equity 5.2% of assets); CF (#15: net income 15% off the filing); MO (#24: revenue 12% off the filing) |
| Data run timing | **23 min** total; slowest `fetch.tickers` 632 s, `fetch.sec_roe_history` 221 s, `fetch.retry` 152 s; **48 tickers rate-limited** |
| Code-session login | 2026-12-03 09:01 (23 days) |
| Context coverage | options usable 415/499 - insider from SEC 499/499 - earnings variability 439 |

## Coming up

Dates the project has committed to. A past-due line stays until the session that
does it deletes it from `COMING_UP` in `scripts/write_brief.py`.

- 2026-11-09 (**due 1 day ago - still listed**) - The context signals' first one-month reading arrives with this run; read the brief's `context_eval` line (CLAUDE.md 0.11). The three option signals wait for 2026-12-04 (below). `CLAUDE.md`
- 2026-11-16 (in 6 days) - Re-run the filing cross-check's range table; promote total assets, equity and operating cash flow to a gate at 25 only if each stayed <= 5 on every run. `research/2026-10-14-yahoo-vs-sec.md`

## Things that needed attention

- ok   ATTENTION - 1 stock(s) now carry a market value implying a share count over 10% from the one their filings report, which the published page does not: FOXA +32.3%. Usually a completed acquisition or a split: the market value is right and every valuation ratio built on it has moved, while the earnings behind those ratios have not. Not a refusal - see research/2026-11-09-share-counts-and-corporate-actions.md.
- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.

## What changed in the repo

- `4a404cb brief: code session 2026-11-09-midday`
- `92256b7 log: 2026-11-09 (13:00) - the share-count proposal decided, and the detector that had been comparing nothing`
- `94dc9fb docs: the share-count decision in the owner queue, and the gate-reads-the-whole-payload constraint`
- `1e71c1f decide: the share-count proposal - option 2 accepted in substance, refused as specified; 3 refused; 1 left open with a pre-registration`
- `e16a88e feat: the health note names the stocks whose market value implies a share count their filings do not - newly`
- `b807239 test: the health gate must read the whole payload, derived rather than transcribed`
- `016824c fix: the health gate read one of the run's three payload files, so the market-value check compared nothing`
- `44b363d brief: code session 2026-11-09`
- `f14b4f1 log: 2026-11-09 - the refused data run, both faults fixed, and the share-count note`
- `2c62cfc research: the share count behind a market value, and what a screen should do when it moves`
- `8af1404 fix: the lab's card bound was read before the page settled, and had been breached`
- `9817403 fix: the share-class gate asserted a discarded intermediate, and discarded a night`
- `3a608c4 brief: data run 2026-11-09`
- `51d75a7 brief: code session 2026-11-08-midday`

## The session's own account

> 2026-11-09 (13:00 session) - QUEUE. The share-count proposal decided, and the detector it asks for found already built and silently broken
> 
> **Health (rule 8, all five):** last code session ran? yes - `logs/nightly-2026-11-09_060001.log`
> (completed). Data loop published? **no** - `logs/datarun-2026-11-09_020001.log` (stopped
> deliberately; HEALTH: PASS) - this morning's session diagnosed and fixed both gate failures and
> verified the rehearsal, so the next 02:00 run publishes; nothing further was needed today.
> Evidence base at `1m`: **5 effective of the 8 needed, 45 rows, newest 1m run date 2026-10-06 (34
> days ago, bound 40)**; 114 rows across all horizons. Priority 0: `allow_auto_apply` **false**. Top
> open roadmap item: **0.13, The Factor Lab** (age **28 days**); also open 0.10 (33d), 0.14 (26d).
> **Tests:** before **4,243 passed / 1 skipped** (226 s parallel, green baseline - `origin/main`
> already carried this morning's work and no branch was held back by the gates); after **4,271
> passed / 1 skipped**. Gates 1-4 pass; gate 5 (`rehearse_publish.py`) **PASSED**, all five
> published files restored byte for byte.
> **Owner queue / rotation:** the morning brief listed **one open proposal**, which the prompt and
> `OWNER_FOCUS.md` item 1 both make the first job, and I did not write it - so that was the session.
> `write_brief.open_proposals()` returns `[]` now. **Deferred: 0.13, 0.10 and 0.14** - 0.13 for the
> fourth consecutive session, and this is the deferral-streak rule's own exception rather than a
> fourth defensible deferral: the rule binds on "the next session carrying **no owner item**", and
> deciding an open proposal is an owner item. **It is the next session's first job if the queue is
> still clear.**
> 
> ### Did
> 
> **Decided the open proposal** - `research/2026-11-09-share-counts-and-corporate-actions.md`, which
> asked which of three things to do for a stock whose implied share count has moved while the
> earnings behind its multiples have not (FOXA, rank 34 -> 60 on 2026-11-09, price unchanged).
> **Option 2 accepted in substance and refused as specified; option 3 refused; option 1 left open
> with two conditions it did not have.** The argument is the note's new section 8; the measurements
> are `research/measurements/2026-11-09-share-count-detector-review.py`, committed.
> 
> **1. The finding: the detector the note asks for already existed, and had been comparing
> nothing.** `sec_crosscheck.market_value_check` has computed `marketCap / price` - the note's own
> `implied_shares`, the same expression - against each company's **filed** weighted-average share
> count every run since 2026-10-14, printed in `check_run_health`'s own note with an ATTENTION line
> above 60 stocks. The note names `sec_crosscheck` as the precedent to copy without noticing it is
> the same check.
> 
> From the run logs: `446/485 within 5% (21 over 10%)` on 11-05 and 11-06 02:00, `444/485 (19)` on
> 11-06 18:34, and **`0/0 within 5% (0 over 10%)` on 11-09**. `financials` moved into
> `dashboard_detail.js` on **2026-11-08**; the check reads
> `stock_detail[t]["financials"]["market_cap"]`; `load_payload` read `dashboard_data.js` alone. So
> from the first run after that commit it compared **0 stocks of 485**, and **an ATTENTION line that
> fires above 60 cannot be reached from zero.** Measured on the 11-09 run's own artifacts: 0
> compared from `dashboard_data.js`, **485 with the shards folded back, 20 over 10%** - FOXA among
> them at ratio **1.3109**, the stock the whole note is about. Fixed: the gate folds every shard
> ...

---

If a run says **stopped deliberately** or **held back by its gates**, that is the safety
gates working: the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

