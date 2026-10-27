# Morning Brief - Tuesday 27 October 2026, 13:38

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-27T02:00:05.759618 |
| Stocks scored | 498 |
| With a price | 498/498 |
| With an analyst target | 496/498 |
| Top 5 | EXPE, MU, APA, HST, EG |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (38 rows, but overlapping windows are not independent; 99 rows across all horizons), newest 2026-10-19 |
| Context signals | Context signals: 14 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07 and its first reading comes with the run of 2026-11-09, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 424/425 within 0.5%; operating cash flow 408/412 within 0.5%; net income 398/421 within 0.5%; market value 443/484 within 5%; filings cached 4 days ago |
| Top 25, worth a look | EXPE (#1: equity 4.2% of assets); HST (#4: net income 1.46x after-tax operating income); CF (#20: net income 16% off the filing) |
| Data run timing | **19 min** total; slowest `fetch.tickers` 659 s, `snapshot` 136 s, `fetch.retry` 125 s; **39 tickers rate-limited** |
| Context coverage | options usable 485/498 (13 quotes-closed) - insider from SEC 498/498 - earnings variability 440 |

## What changed in the repo

- `bf6da57 page: rebuild index.html for the three new provenance sentences (payload byte-identical)`
- `b573f36 log: the dead-badge hunt - four defects, one a false sentence on a top-25 stock, one a false claim on the public methodology page`
- `e043051 provenance: the EV sentence states the limit that applied to this stock, not the other one`
- `2b04e17 drilldown: the dead badges' last four - a false LTM sentence, a silent EV substitution, an unexplained beta, and a flag no run ever set`
- `c148df2 docs: log and owner queue for the TradingAgents build`
- `903c796 tests: two browser tests wait for the condition, not a fixed delay a loaded suite overran`
- `e8baa51 contract: pin stock_detail[t].price_date, which the bot's swap ledger now reads; note: swap ledger done`
- `0a7f45e header: the priority-plus fit counts a rounded 1px overflow as over`
- `ccab5b3 drilldown: "The case against" - what the sheet already shows against a stock, collected (owner)`
- `3e62cb3 evidence base: a holiday run enters at the next session's close, not the one its ranking was built from`
- `952a184 research: what to take from TradingAgents - and a holiday look-ahead in the live evidence base`
- `8063ee6 header: every section link whole or under More, at every width (owner)`
- `06c54bd type: one typeface - headlines in Inter 600 tracked tight, the ticker's style; the display serif removed (owner)`
- `8a33be0 docs: 2026-10-27 owner-run UI elevation - log, owner queue, plan status and residuals, design-system note, rebuild_page in rule 10`
- `9658fb0 lab, research: the serif wordmark as the dashboard sets it; eyebrows in muted ink (accent is for what you can click)`

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

