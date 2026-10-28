# Morning Brief - Wednesday 28 October 2026, 06:00

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **stopped deliberately** - last ran today |
| Code session (latest) | **failed** - last ran today |
| Dashboard data from | 2026-10-27T02:00:05.759618 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (38 rows, but overlapping windows are not independent; 99 rows across all horizons), newest 2026-10-19 |
| Context signals | Context signals: 14 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 412/413 within 0.5%; operating cash flow 397/401 within 0.5%; net income 387/409 within 0.5%; market value 443/484 within 5%; filings cached 5 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#17: net income 16% off the filing); WDC (#18: net income 2.13x after-tax operating income) |
| Data run timing | **18 min** total; slowest `fetch.tickers` 651 s, `fetch.retry` 144 s, `context.sec_insider` 98 s; **44 tickers rate-limited** |
| Context coverage | options usable 485/498 (13 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## Things that needed attention

- The regenerated artifacts contradict claims this project asserts about them.
- Refusing to publish. The live site keeps the last good version.
- Run discarded: published-claims check failed - fix the generator, not the file.
- Working tree is dirty - probably leftovers from a failed session:
- ?? data/context_log/2026-10-28.parquet
- ?? improvement/price_cache/prices_2026-07-29_2026-10-27.parquet
- ?? improvement/price_cache/prices_2026-10-20_2026-10-27.parquet
- Saved working directory and index state On main: auto-rescue 2026-10-28_060001
- warning: failed to remove output/: Permission denied
- Could not stash the leftovers. Resolve by hand: git status

## What changed in the repo

- `9f9fda7 brief: data run 2026-10-28`
- `b7ff1f4 log: the seventh owner pass - the repo lock's two races fixed, flaky tests, sheet, shell, lab, bot`
- `ab2a172 test_sheet_round_three: the tab tests wait for the context payload and for the jump to land, not for a fixed position (flaked twice in five parallel runs)`
- `763e791 repo lock: publish the lock whole and reclaim only under a token - both measured races fixed`
- `a22a01b page shell, round three: failure states, freshness, true run time, Inter stand-in, print, 404, home screen (payload byte-identical)`
- `8b9b0fc research: flaky tests - three full suites green on the final tree`
- `be8ef69 research: the flaky tests of 2026-10-27 - each flake, its cause, the fix, before/after; the repo-lock races measured and proposed`
- `df43558 tests: make the load-sensitive browser waits deterministic; harden the repo-lock race test`
- `aa91610 sheet round three: scroll-spy holds the chosen tab, Financials tab, idle warm-up of the first open, unscored categories say why, peers open and rank`
- `2d26d77 CLAUDE.md: drop the repeated LAZY_KEYS reference`
- `1333c56 CLAUDE.md: the lab's lazy parts are one per LAZY_KEYS entry, not six`
- `0d5d128 plan: round three notes - the 153 factors split by universe, the sorts by sort, measured`
- `cae470c lab: every overview card puts its count under its meter, so the counts line up`
- `27642a6 research: every t-statistic carries a real minus, as the means beside it do`
- `5d257e9 lab: each of the eight sorts is its own part, so opening one fetches its own series only`

## The session's own account

> 2026-10-27 (13:00 midday session) - QUEUE. The eleven badges that lived in dead code: four said nothing, one of them a false sentence on a top-25 stock, and one a false claim on the public methodology page
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 06:00 today, "Run complete:
> shipped to main", tag `good/2026-10-27`; five owner-run passes merged after it. Data loop
> published? **yes** - 02:00 today, "Data loop complete", **HEALTH: PASS**, 498 scored, live at
> 02:20, claims gate 1,576 passed in 65 s `[parallel, -n auto]`. Evidence base at `1m` = **38 rows,
> newest 2026-09-25 (32 days ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**,
> unchanged. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12, **age 15 days**) -
> not advanced; the next non-retrospective Friday is 2026-11-06 and experiment 18 is a Friday's work.
> **Tests:** before **3295 passed, 1 skipped**; after **3313 passed, 1 skipped** (+18, the new
> module). `scripts/rehearse_publish.py` **PASSED**.
> **Owner queue / rotation:** no proposal was waiting (`open_proposals()` returns `[]`). The 06:00
> session's **Next** was the EDGAR daily-form-index build, whose own evidence bar is two weeks of
> shadow runs - it cannot be finished end to end today, and the prompt asks for one item finished.
> I took instead the item this morning's owner-run session flagged as available to a session
> (`plan/ui-elevation.md` item 5), which is also what the 2026-10-15 session called **"the single
> most valuable next thing"** and which had then sat for **12 days**. Nothing deferred for a loop or
> a gate; nothing an earlier session decided today was reversed.
> 
> ### Did
> 
> **The defect hunt the 2026-10-15 session left open.** That session found
> `generate_dashboard.renderFlagsWarnings` dead - nothing called it, and its container
> `#modal-flags` was never written into the page, so that morning's quarterly-rejection badge could
> never have appeared. It rehomed the two provenance sentences and left the **other eleven badges**
> with the note that one session should check each against the live page and either rehome it or
> delete it. This is that session.
> 
> Measured on the 2026-10-27 published run, 498 stocks
> (`research/measurements/2026-10-27-dead-badge-audit.py`, reproducible at any commit):
> 
> | Badge | Stocks | Reaches the reader? |
> |---|---|---|
> | value trap | 46 | yes - table flag column, `summary`, "The case against" |
> | growth trap | 66 | yes - same |
> | Beneish | 43 | yes - `summary`, "The case against" |
> | channel stuffing | 14 | yes - `summary`, "The case against" |
> | stale financials | 28 | yes - Filing Age badge, `summary`, "The case against" |
> | EPS basis mismatch | 100 | yes - provenance badge, data-quality KPIs |
> | bank-like | 46 | yes - the ROE/ROIC column label, methodology page |
> | **LTM 3-of-4 annualised** | **4** | **no - and the panel said the opposite** |
> | **enterprise value substituted** | **18** | **no** |
> | **beta overlap under 80%** | **1** | **no** |
> | **financial-sector caveat** | **0** | **no - dead at source, with a false claim on the public page** |
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

