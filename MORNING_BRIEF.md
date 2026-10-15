# Morning Brief - Thursday 15 October 2026, 14:03

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (latest) | **completed** - last ran today |
| Dashboard data from | 2026-10-15T02:00:03.638262 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, APA, HST, INCY, BMY |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (30 rows, but overlapping windows are not independent; 83 rows across all horizons), newest 2026-10-08 |
| Context signals | Context signals: 6 daily logs since 2026-10-08; the first one-month return window closes 2026-11-07, so no signal has a record yet (gate: 8 effective observations). |
| Inputs vs SEC filings | total assets 425/426 within 0.5%; operating cash flow 406/411 within 0.5%; net income 397/421 within 0.5%; market value 447/487 within 5%; filings cached 6 days ago |
| Context coverage | options usable 487/501 (14 quotes-closed) - insider from SEC 501/501 - earnings variability 443 |

## What changed in the repo

- `bdb5b75 log: the roll-forward, the dead provenance function, and the operating-income finding`
- `18f8913 research: operating income is the least source-agreeing input, and three weighted metrics stand on it`
- `a375e70 wip: the provenance sentences move to the function the page actually calls`
- `9e3949a wip: the rolled twelve months on the page, in the methodology, and as a claim`
- `e867764 wip: roll the last fiscal year forward instead of annualising three quarters`
- `5a9e42c research: a per-stock fallback to the filings when the source is behind would add more noise than it removes`
- `b6c6e86 provenance: say when a stock is scored a period behind its own filings`
- `b35015e filing check: revenue is compared with the filed line nearest the run's figure, not the largest`
- `ae36dcd fetch: a quarterly statement derived across a restatement is not summed (DuPont's revenue read a third low)`
- `4ea8f9a nightly prompt: backtest v2 is not evidence either - and not for cutting a weight`
- `56d212e backtest v1: the disclaimer quotes the measured look-ahead instead of an unsourced estimate`
- `9d6ed87 methodology page: the point-in-time limitation quotes the measured look-ahead; 72.6% corrected to 72.5%`
- `b6f7fd9 backtest v2: monthly rankings from what was known then, and the first honest look at returns`
- `243a8b3 backtest v2: the engine's record rebuilt from SEC filings as of any date, and the look-ahead's fundamentals half measured`
- `a46c53e brief: code session 2026-10-15`

## The session's own account

> 2026-10-15 (13:00 slot) - QUEUE. A quarter the source never published is rolled forward, not invented
> 
> **Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-10-15_060001.log`
> ends "Run complete: shipped to main". Data loop published? **yes** - `logs/datarun-2026-10-15_020001.log`
> ends "Data loop complete", claims gate PASS (460 passed), published to main 02:22.
> Evidence base at `1m` = **30 rows, newest 2026-09-15 (30 days ago, bound 40), 4 effective**
> - inside the band, no action. Priority 0: `allow_auto_apply` still **false**, 4 effective against a
> gate of 8. Top open roadmap item: **0.11, the context layer** (opened 2026-10-08, **age 7 days**);
> deferred today - see below.
> **Tests:** before **2396 passed, 1 skipped**; after **2414 passed, 1 skipped**. Dry run OK;
> `index.html` 558,545 bytes; `dashboard_data.js` and `dashboard_context.js` both `node --check` clean;
> `scripts/rehearse_publish.py` PASSED (300 page tests, 473 claims).
> **Owner queue / rotation:** this is the **fourth** session today (06:00, two owner-run, and this
> 13:00 slot after the owner marker cleared). Carried on from the owner-run midday entry's **Next**.
> Its Next asked whether scoring should ever read the filings when the source is behind or
> inconsistent - I did **not** reopen that (it was measured and refused twice today), and instead took
> the concrete, still-open residual the same line of work left behind:
> `research/2026-10-14-yahoo-vs-sec.md` **section 4b**, which named three-quarter annualisation as
> "a small, well-defined data fix for a later session". The context-layer owner item (macro and
> track-record notes) was deferred to finish this end to end; nothing in it was started or lost.
> 
> ### Did
> 
> **1. The main piece: the twelve months is rolled forward, not fabricated.** Re-measured section 4b
> rather than trusting it. `_ltm_annualized` was true for **4 of 503** stocks on this run. For three it
> touched one immaterial line each (COST and V: common stock dividend; GIS: interest expense). For
> **BRK-B it touched nine lines** - revenue, operating income, net income, tax, pretax income, operating
> cash flow, capex, D&A, interest - because the source does not carry its **2025-09-30 quarter at all**;
> the columns run `2026-06-30, 2026-03-31, 2025-12-31, 2025-06-30, 2025-03-31`, so `_quarters_adjacent`
> rightly refused to sum the first four and `sum(3) * 4/3` took over.
> 
> Against Berkshire's own filings for the same twelve months the published figures were **net income
> 14.54% low, operating cash flow 6.09% low, revenue 10.23% high**.
> 
> The fix is not the annual statement (measured: it is *worse* on net income, -21.9%). It is the
> identity every filer uses in its own 10-Q and that `sec_crosscheck.flow_ttm` already applies to the
> filings: **fiscal year + year-to-date - year-to-date a year earlier**. Rebuilt that way BRK-B's
> figures are **net income $85.768bn against $85.768bn filed, operating cash flow $46.634bn against
> $46.634bn, capex $22.419bn against $22.419bn** - exact to the dollar on all three.
> `factor_engine._ltm_rollforward`, tried between the four-quarter sum and the 4/3 fallback, current
> period only, with guards on every term. It reads the two statements the fetch already holds and
> **never a filing** - a test enforces that, because scoring deliberately does not touch the SEC cache
> (CLAUDE.md 0.14), which is what made it the independent yardstick here.
> 
> **Said plainly, including what it does not fix:** BRK-B's revenue goes from +10.2% to **+12.2%** -
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

