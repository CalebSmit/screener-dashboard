# Morning Brief - Wednesday 02 September 2026, 02:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **failed** - last ran today |
| Dashboard data from | 2026-09-02T02:00:05.461510 |
| Stocks scored | 502 |
| With a price | 502/502 |
| With an analyst target | 498/502 |
| Top 5 | HST, EXPE, APA, CF, EIX |
| Evidence for weight changes | 3 of 8 needed at the 1m horizon (8 rows, but overlapping windows are not independent; 30 rows across all horizons), newest 2026-08-26 |

## What changed in the repo

- `3024a59 data: screener run 2026-09-02 - 502 scored, top: HST EXPE APA CF EIX`
- `94afb03 docs: log the smoothness pass, record it in the owner queue`
- `891f2ee fix: three smoothness/reliability issues found in an owner-requested audit`
- `bcef439 brief: code session 2026-09-01`
- `97d5f10 docs: the winsorization rationale was false, so correct it rather than soften it`
- `f1f5855 test: lock down the no-winsorization property and the megacap regression`
- `4d47c42 fix: stop clipping metric values the screener then ranks`
- `c2cf5c4 brief: data run 2026-09-01`
- `c916b5e data: screener run 2026-09-01 - 502 scored, top: HST EXPE APA CF VLO`

## The session's own account

> 2026-09-01 (evening) - Owner-run: a general audit for smoothness, three real findings
> 
> Owner's brief: "make all necessary updates for it to run as smooth and
> effective moving forward." Interpreted as a health/reliability audit rather
> than a specific feature - checked scheduled tasks, repo hygiene, and whether
> CLAUDE.md's own priority claims still matched reality, then fixed what was
> actually wrong rather than inventing scope.
> 
> ### Did
> 
> **1. Corrected a stale claim before acting on it.** Priority 1 said "~10-25%
> of tickers fail per run." Checked the last 15 data-run logs directly rather
> than trust it: every single one, back to 2026-08-10, reports 0 fetch
> failures. Updated CLAUDE.md to say so, with a note to re-verify periodically
> rather than let the record drift stale in either direction again.
> 
> **2. Found and fixed a permanent false alarm.** `validation/data_quality_log.csv`
> flagged four bank-only metrics as "High severity - missing >50%" on every run
> since launch - 88.4%/88.2%, unchanging. Traced it: only ~58 of 502 stocks are
> banks, and these metrics are correctly absent from every non-bank by design.
> The drift check scored missing-% against the whole universe instead of the
> population a metric applies to; the coverage filter a few hundred lines away
> in the same function already did this correctly and the drift check simply
> never matched it. Extracted `_metric_missing_pct()`, scoped by the existing
> `_BANK_ONLY_METRICS`/`_NONBANK_ONLY_METRICS` sets. 7 tests.
> 
> Worth naming plainly: a permanent "High severity" alert that never means
> anything is the same failure shape the loop watchdog was explicitly designed
> to avoid (CLAUDE.md rule 7's reasoning) - it trains a reader to stop looking,
> which is exactly when a real drift would go unnoticed.
> 
> **3. Found the synthetic-data refusal only covered one of two entry points.**
> `run_screener.py` refuses to fabricate data on a failed fetch (fixed
> 2026-08-11). `factor_engine.py` has its own independent `main()` - unreachable
> from the scheduled loops, reachable by anyone running it directly - and it
> still had the exact pre-fix behaviour: unconditional fabrication, no flag, no
> refusal. Fixed the same way. 4 new tests, confirmed to fail against the
> pre-fix file before trusting them.
> 
> **4. Nightly branches were not actually self-cleaning.** `git branch` showed
> `nightly/2026-08-10` and `nightly/2026-08-27` still present, both fully
> merged weeks ago. The delete-after-merge call existed but piped its exit code
> to `Out-Null`, so a rare failure (transient lock, interrupted run) left debris
> with zero visibility. Fixed two ways: the delete now logs a `WARN` on
> failure, and every run sweeps any local `nightly/*` branch already merged
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

