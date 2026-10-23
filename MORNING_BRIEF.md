# Morning Brief - Friday 23 October 2026, 06:35

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
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

- `523f66f log + OWNER_FOCUS: the proposal decided, the lab's Friday deferred and why`
- `3eee020 decide the open proposal: all three registrations refused as written, test B accepted in substance`
- `0fc5c75 measurements: second reading of the three proposed registrations (FINRA files, registry shape, verdict bands)`
- `8c2064b brief: data run 2026-10-23`
- `3e366b0 data: screener run 2026-10-23 - 498 scored, top: EXPE APA HST INCY SNDK`
- `1b02665 log + CLAUDE.md: the 2026-10-22 improvement wave; the bot's input gates recorded as settled`
- `5ab1eca research + OWNER_FOCUS: the input gates replayed against every published page`
- `502ca2b health: the trading bot's inputs - rank continuity, the unscored share, price dates; data-loop git hygiene`
- `8b9c494 research: which free data adds evidence, not more tests on the same history`
- `b4d0333 brief: code session 2026-10-22`
- `9bf29e2 log: the gates' numbers, the rerun check, and the three guards that fired`
- `15efe21 research engine: all 31 rows rerun level, and the page reports the stop arm`
- `3a56a63 log + OWNER_FOCUS: the 2026-10-22 stop-loss item, done and moved to Done`
- `5e9d966 research: the stop-loss arm's result, and the page reports it beside the unstopped rule`
- `513c364 research engine: register and build one stop-loss arm on the bot's rule (Han, Zhou & Zhu 2016)`

## The session's own account

> 2026-10-23 (06:00) - the open proposal decided: the registry cannot hold any of the three tests, and one of them would have manufactured a false null
> 
> **Health (rule 8, all five):** last code session ran? **yes** - 2026-10-22 06:00 ended "shipped to
> main" (the 13:00 slot stood down for the owner marker, as designed). Data loop published? **yes** -
> 02:28 today, HEALTH PASS, 498 scored, claims gate 1,360 passed. Evidence base at `1m` = **35 rows,
> newest 2026-09-22 (31 days ago, bound 40), 4 effective**. Priority 0: `allow_auto_apply` **false**,
> 4 of 8 effective observations. Top open roadmap item: **0.13, the Factor Lab** (opened 2026-10-12,
> **age 11 days**) - **deferred**, see below.
> **Tests:** before **3063 passed, 1 skipped**; after **3064 passed, 1 skipped** (one new guard).
> **Owner queue / rotation:** today was the lab's Friday. A proposal was waiting for a decision
> (1 day open), which the prompt and `OWNER_FOCUS.md` item 1 both put ahead of the focus, so **the
> Factor Lab experiment is deferred**. Nothing lab-side was started, so nothing is half-done.
> 
> ### Did
> 
> **Decided the open proposal in `research/2026-10-22-free-data-that-adds-evidence.md` - and the
> decision is a refusal of all three registrations as written, with the research left standing.**
> The note asked a later session to register three pre-registered tests in
> `research/experiments/registry.yaml`, "each in its own commit before any of its data is built".
> It was written by a different session, so this is the second reader the routine is built around.
> Three things were measured that the note had not; both scripts are committed beside it.
> 
> **1. The FINRA claim is true - re-run, not taken.** Twelve settlement dates fetched
> (`research/measurements/2026-10-23-finra-short-interest-probe.py` + `.json`). 2014-2016 return
> **403**; **2017-12-29 onward return 200**, 15,495 rows rising to 20,516 by 2025. `marketClassCode`
> on 2017-12-29: **NYSE 3,102, NNM 2,355, ARCA 1,433** beside OTC 7,152 - exchange-listed rows are in
> the file four years before the June 2021 FINRA's own page states, so the note's correction of FINRA
> stands. `daysToCoverQuantity` is numeric and positive on **15,494 of 15,495** rows then and
> **20,516 of 20,516** in 2025, and all seven large members probed (AAPL, JPM, MSFT, XOM, JNJ, KO,
> GE) are present on every served date. Two details the note should have carried: the files are
> **pipe-delimited despite the `.csv` extension** (my first probe read one column), and the field
> **floors at 1.00** - AAPL and JPM both read exactly 1.00 on 2018-12-31 - so it is censored below.
> 
> **2. The decisive finding: the registry cannot hold any of the three, so the instruction is not an
> executable action.** Measured by running the engine's own call sites rather than reading the YAML
> (`research/measurements/2026-10-23-can-the-registry-hold-these-tests.py` + `.json`). The committed
> registry has exactly **two** shapes: `experiments:` - 29 entries, **29 of 29** carrying a
> `transform` whose `kind` is one of seven known kinds - and `evaluations:` - 2 entries, each with a
> `rule` and no transform. None of A, B or C is either. Shaped as the note writes them,
> `apply_transform` raises **`KeyError: 'factor_weights'`** and the bare `e["transform"]["kind"]` read
> raises **`KeyError: 'transform'`** - unguarded at `v2_experiments.py:1122`, `:1156` and `:1180`
> (`report`, `summary_for_page`) and in three places in `tests/test_engine_calibration.py`. Following
> the instruction would have reddened the suite and the public research page, not pre-registered
> anything.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

