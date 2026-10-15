# Morning Brief - Wednesday 14 October 2026, 22:12

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **failed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-14T02:00:10.255065 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, INCY, HST, APA, SNDK |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (29 rows, but overlapping windows are not independent; 81 rows across all horizons), newest 2026-10-07 |
| Context signals | Context signals: 5 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Context coverage | options usable 486/501 (15 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `0c8730d log: 2026-10-14 13:00 session - context-layer item 4's three findings closed`
- `1091181 context: withhold a skew whose legs cannot show the construction, and fix the window clause`
- `3cfd9b8 research: the skew construction, read from the papers, and a correction to the 10-12 note`
- `c09d343 context: the put skew is the measure the paper the page cites measured`
- `38503b7 tests: the suite no longer rewrites the published universe`
- `88a93c1 brief: report the newest code session that actually ran; a dry run publishes nothing`
- `c45178e brief: code session 2026-10-14-midday`
- `dfd94fb routine: two extra scheduled code sessions - 13:00 daily and weekend mornings`
- `4f0c64c brief: code session 2026-10-14`
- `f8480a6 log: the four gates for 2026-10-14`
- `2f42754 log: the 2026-10-14 session, and the two constraints it added`
- `17249d4 publish: regenerate today's page so it stops saying two false things`
- `1daf857 changelog: the dividend-yield defect, its fix and the second derivation it exposed`
- `3e27f14 health: refuse a run where a weighted metric vanished across the universe`
- `53d9c5f churn: a metric the whole universe lost is a source event, not one stock's`

## The session's own account

> 2026-10-14 (13:00 session) - QUEUE. The options panel now cites a paper for the number it actually computes
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-14_060001.log`
> ends "Run complete: shipped to main". Data loop published? **yes** -
> `logs/datarun-2026-10-14_020002.log` ends "Data loop complete", HEALTH: PASS. (The 21:12 data run
> stood down correctly: "already completed successfully today".) Evidence base at `1m`: **29 rows,
> newest 2026-09-14 (30 days ago, bound 40), 4 effective** - inside the bound, and the data log's own
> line agrees ("4 effective (29 rows) at the 1m horizon"). Priority 0: unchanged, `allow_auto_apply`
> false, 4 of 8 effective. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 2
> days**).
> **Tests:** before **2,288 passed / 1 error**, after **2,317 passed / 0 errors**.
> **Owner queue / rotation:** took the **context layer** (`plan/context-layer.md` item 4), which is
> what the 06:00 session's **Next** named and what it had deferred for a red gate. The Factor Lab is
> not deferred so much as **scheduled**: its own rule is one finished experiment on a Friday, and
> 2026-10-16 is a retrospective, so its next day is 2026-10-23. No proposal is waiting for a decision.
> 
> ### Did
> 
> **0. The baseline was red, and it was mine to fix before anything else.**
> `python -m pytest tests/ test_screener.py -q` gave **2,288 passed, 1 error**: the `conftest.py`
> tripwire reporting *"A test wrote published artifacts (restored): sp500_tickers.json"*. The same
> command was green at 06:40 and at the owner-run session after it.
> 
> `test_screener.py`'s **session-scoped `universe_df` fixture** called
> `factor_engine.get_sp500_tickers()`, which goes to the network and **rewrites the published
> `sp500_tickers.json` whenever a source succeeds**. So the write happened only when the network was
> up *and* the fetched universe differed from the committed one - which is why the 2026-10-09 per-test
> write detector reported "exactly three tests" and missed this one: it ran on a day the fetch did not
> change the file. The module's own docstring had been promising "no network calls" since before any
> of this.
> 
> That matters more than a flaky test, because **the runner re-runs all four gates before merging**: a
> gate-1 failure whose trigger is outside the repo blocks the unattended merge for a reason no session
> can reproduce. The fixture now reads the committed file, as `test_tiny_pipeline` already did.
> 
> The loader keeps real coverage and gains the assertion that was missing: with every network source
> refused it falls back to the committed universe and leaves it **byte-identical**.
> `tests/test_published_artifact_isolation.py` (5) is the tripwire - an AST scan over all **98** test
> sources finds every call to a loader that writes a published artifact and fails unless it is that
> one stubbed test (verified non-vacuous: it finds exactly one call site and attributes it correctly).
> Two further tests stop a later session shortening `conftest.PROTECTED_PATHS` or downgrading its
> `pytest.fail` to a silent restore.
> 
> **1. The put skew now is the measure the paper beside it measured.**
> `plan/context-layer.md` item 4 left three findings open, all the same shape: *a figure from a paper
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

