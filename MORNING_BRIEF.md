# Morning Brief - Friday 23 October 2026, 13:40

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

- `b6352de log: the 2026-10-23 13:00 session - experiments 16 and 17 second-read, the rerun deferred with its argument`
- `257faa3 measurements: extend the Piotroski second reading to the note's other-period claims`
- `749064e record: experiments 16 and 17 second-read; close item 5; correct 7c's experiment number`
- `ce959a6 research: second reading of lab experiment 17 - four paper figures exact, every measured figure reproduced`
- `8b83ff5 research: second reading of lab experiment 16 - all 17 paper figures and every measured figure confirmed`
- `76dbb7b measurements: second reading of lab experiment 16 (Asness et al. 2018 Table 3A), recomputed independently`
- `99806bb measurements: second reading of lab experiment 17 (Piotroski 2000 on OSAP), recomputed independently`
- `88f5430 brief: code session 2026-10-23`
- `523f66f log + OWNER_FOCUS: the proposal decided, the lab's Friday deferred and why`
- `3eee020 decide the open proposal: all three registrations refused as written, test B accepted in substance`
- `0fc5c75 measurements: second reading of the three proposed registrations (FINRA files, registry shape, verdict bands)`
- `8c2064b brief: data run 2026-10-23`
- `3e366b0 data: screener run 2026-10-23 - 498 scored, top: EXPE APA HST INCY SNDK`
- `1b02665 log + CLAUDE.md: the 2026-10-22 improvement wave; the bot's input gates recorded as settled`
- `5ab1eca research + OWNER_FOCUS: the input gates replayed against every published page`

## The session's own account

> 2026-10-23 (13:00 session) - the lab's two unchecked experiments second-read: all 21 reference figures exact, every measured figure reproduced, and the number that keeps Size at 5% is now checked
> 
> **Health (rule 8, all five):** last code session ran? **yes** - today's 06:00 ended "Run complete:
> shipped to main" at 06:35. Data loop published? **yes** - 02:28 today, HEALTH PASS, 498 scored.
> Evidence base at `1m` = **35 rows, newest 2026-09-22 (31 days ago, bound 40), 4 effective**
> (`_n_observations` 4, `_n_raw_observations` 35 - the raw count is the one not to quote). Priority 0:
> `allow_auto_apply` **false**, 4 of 8 effective observations. Top open roadmap item: **0.13, the
> Factor Lab** (opened 2026-10-12, **age 11 days**) - advanced, see below.
> **Tests:** before **3064 passed, 1 skipped**; after **3064 passed, 1 skipped**. Gates 2, 3, 4 pass
> (dry-run clean; `index.html` 651,844 bytes, both payload files parse under `node --check` and open
> with their own assignment; working tree clean). `scripts/rehearse_publish.py` **not run and not
> needed**: the diff is research notes, measurement scripts, `OWNER_FOCUS.md` and `plan/`, and
> touches nothing the page is built from.
> **Owner queue / rotation:** `OWNER_FOCUS.md` 2026-10-20 **item 2** taken and its outstanding half
> **closed**; **item 5** closed as already-done. No proposal was waiting
> (`write_brief.open_proposals()` returns `[]`). **Deferred: the research-engine rerun** the 06:00
> session named as Next - the argument is below, under *Tried and rejected*, because deferring an
> earlier session's Next on the same day needs one.
> 
> ### Did
> 
> **Second-read Factor Lab experiments 16 and 17, the two that had no reader but their author.**
> `OWNER_FOCUS.md` item 2 asks for exactly this and names what it means: reopen the paper, re-read
> every reference figure against `factor_lab.py`, and check in `git log` that the tolerances were
> committed before the comparison ran. Experiment 15 was done the night it shipped; 16 and 17 were
> merged after it on the owner-run night of 2026-10-20 and have been uncovered since. Both were
> checked three ways, and a fourth was added - recomputing every *measured* figure from the source
> files with no import of `factor_lab`, so a construction error would disagree rather than be
> reproduced. **Nothing needed correcting.** Both measurement scripts are committed.
> 
> **Experiment 16, `size-given-profitability`** (Asness, Frazzini, Israel, Moskowitz & Pedersen 2018)
> - the one that matters most, because its verdict is what kept the Size weight at 5% on 2026-10-20.
> The publisher's open-access PDF was obtained (the CBS Research Portal copy; ScienceDirect answers
> 403 to anything that is not a browser and that was **not** worked around) and **all seventeen
> reference figures are exact** against Table 3 Panel A, printed p. 495: the Small - Big column
> 0.23/0.46/0.54/0.42/0.44 with t 1.21/3.02/3.44/2.82/2.87, the Quality - Junk ends 0.62 (t 4.78) and
> 0.42 (t 2.78), and SMBQ 42 bps t 3.18 quoted verbatim from the text on the same page. The single
> **derived** figure is derived correctly - the paper's column "4" minus column "Big" runs
> 0.28/0.19/0.25/0.30/0.25 down the five rows and their mean is **0.254**. The second implementation
> reproduces every measured figure (SMBQ **0.3918**, fourth-minus-biggest **0.2414**, the five rows,
> both quality spreads), gives **10 within and 7 outside** on the committed tolerances - the published
> count, with the misses exactly where the verdict note says - and recomputes the pre-registered
> criterion at **0.1661 a month, t 2.3641** against its bar of 0.136 and 1.5, **-0.1066 (t -0.70)**
> since 2013. So "not refuted" is right. `research/2026-10-20-size-weight.md` section 12.
> 
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

