# Morning Brief - Monday 12 October 2026, 06:33

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-12T02:00:03.296811 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, APA, CF, BMY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (28 rows, but overlapping windows are not independent; 78 rows across all horizons), newest 2026-10-05 |
| Context signals | Context signals: 3 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 455/501 (46 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `c70910c docs: name dashboard_context.js in 'Where things live'`
- `44f11ce log: gates for 2026-10-12 - 2,137 passed (baseline 2,113), all four green`
- `7fced8d log: the research note, the evidence, what was rejected, and next`
- `f1918cd docs: item 2 closed, the options research note recorded, next steps set`
- `f0e8dc2 research: what the four option-implied signals are worth (context-layer item 4)`
- `daa3198 log: the 2026-10-12 entry so far - health, the session-date defect, the repair`
- `663b6b7 repair: re-date the cached option quotes to Friday's session and republish`
- `9140ad4 fix: option quotes name the session they came from, not the next UTC day`
- `61b2720 brief: data run 2026-10-12`
- `c620c4b data: screener run 2026-10-12 - 501 scored, top: EXPE HST APA CF BMY`
- `7cbc8e7 log: PR #1 gated on the PC and merged; a time-of-day test fixed`
- `b7cda7f test: forward EPS blend check no longer depends on the time of day the suite runs`

## The session's own account

> 2026-10-12 - RESEARCH. One thing learned properly, from the literature and from documented practice. A dated note in research/, complete today.
> 
> **Health (rule 8, all five):** last code session - `logs/nightly-2026-10-09_060001.log`, ends
> "shipped to main". Data loop - `logs/datarun-2026-10-12_020001.log`, ends "Data loop complete",
> HEALTH: PASS. Evidence base at `1m`: **28 rows, newest 2026-09-11 (31 days ago, bound 40), 4
> effective**. Priority 0 unchanged (`allow_auto_apply` false; 4 effective against a gate of 8).
> Top open roadmap item: **0.12's last residual, the momentum panic rule** (opened 2026-10-09,
> age 3 days) - deferred, see below.
> **Tests:** before 2,113 passed / 0 failed.
> **Owner queue / rotation:** took the open owner item (the context layer, `plan/context-layer.md`).
> Its Progress line set two next steps: confirm the options `ok` share from the first 02:00 log
> after a 20:00 refresh, then item 4's per-signal research notes. Both done. The momentum panic
> rule is deferred a second time - the owner item outranks it, and it is not yet a three-session
> streak.
> 
> ### Did
> 
> **1. Queue item 2 is confirmed closed - and confirming it found a live false statement.**
> Today's 02:00 run was the first scheduled run after a 20:00 quote refresh. Options `ok` for
> **397 of 503**, `partial` 59, `quotes-closed` 47 - against **0 of 503** before the fix. The
> after-close cache works.
> 
> But the quotes were stamped **2026-10-10, a Saturday**, and the live site told readers of
> **455 of 501** stocks "Quotes are the close of Oct 10; ... so the last session's close is used."
> The market was shut that day. Root cause: the refresh task runs weekdays at 20:00 local = 21:00
> ET, which is **already the next day in UTC**, and `options_cache.refresh` stamped
> `datetime.now(timezone.utc).date()`. Every refresh had named the session *after* the one its
> quotes came from; the Friday case is simply the one that names a non-session.
> 
> A second consequence, measured: `MAX_AGE_DAYS = 3` behaved as **four** - a Tuesday close was
> still being served on the Saturday.
> 
> Measurements, both reproducible:
> `research/measurements/2026-10-12-option-quote-session-date.py` (the stamp, the published
> dates, the staleness window) and `research/measurements/2026-10-12-context-layer-published.py`
> (written while chasing a false lead - see "Tried and rejected").
> 
> **The fix** (`options_cache.py`, `generate_dashboard.py`, 26 tests):
> - `market_session_date()` stamps the Eastern date, weekends rolled back to Friday. Exchange
>   holidays are deliberately not resolved - `quotes_are_live` already covers them, because a shut
>   market serves no quote to accept.
> - `is_session_date()` is **one shared predicate**, used by `reading_for` *and* the generator.
>   The generator drops the whole options block rather than only the date: dropping only the date
>   falls through to "Quotes are the previous close", which is a claim about a session nobody
>   checked.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

