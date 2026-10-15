# Morning Brief - Thursday 15 October 2026, 06:35

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

- `fc2755a log: 2026-10-15 06:00 session - the capex proposal decided, and the capex figure taken apart`
- `4bc5cfd drilldown: the FCF-yield row says what the capex figure contains`
- `36f608a capex proposal: reviewed and refused - the construction keeps the finance arm's cash flow`
- `23afa6a brief: data run 2026-10-15`
- `cfc228c data: screener run 2026-10-15 - 501 scored, top: EXPE APA HST INCY BMY`
- `3b56c2a health: the filing check reports its SEC cache's age and flags it past two weeks`
- `98ed268 capex proposal: the filed lease tags read, Ford ruled out, the tags cached weekly`
- `5b95d97 brief: a daily row for the inputs' agreement with the SEC filings`
- `258595d methodology page: the figures are checked against the filings - say so, and say what is not`
- `31e60fc data trust: market value checked against the filed share counts`
- `185d03b research: what capex should include in free-cash-flow yield - a proposal left for review`
- `b1e5f6d drilldown: each stock's inputs checked against its own SEC filing; a dated test defused`
- `43cebfc data trust: the inputs checked against the companies' own SEC filings`
- `165596a brief: code session 2026-10-14-midday`
- `0c8730d log: 2026-10-14 13:00 session - context-layer item 4's three findings closed`

## The session's own account

> 2026-10-15 (06:00) - BUILD. The capex proposal reviewed and refused, and the capex figure taken apart on the page
> 
> **Health (rule 8, all five):** last code session ran? **yes** - the 2026-10-14 13:00 slot ends
> "shipped to main" (and an owner-run session ran after midnight today; its entry is above - this
> is a second entry for today, not an edit of it). Data loop published? **yes** - 02:00 today,
> HEALTH: PASS, "Data loop complete", 501 scored. Evidence base at `1m`: **30 rows, newest
> 2026-09-15 (30 days ago, bound 40), 4 effective** - inside the bound, and the gate is 8.
> Priority 0: `allow_auto_apply` still false, the engine still correctly refuses to propose.
> Top open roadmap item: **0.13, the Factor Lab** - age **3 days**; not advanced today (its day is
> Friday 2026-10-23, and an owner item outranked it).
> **Tests:** before **2345 passed, 0 failed**; after **2353 passed, 1 skipped, 0 failed** (the skip
> is the new test that compares the *committed* payload's notes with the filings - it has nothing to
> compare until tonight's run publishes them, and the same test runs for real inside the data loop's
> publish gate, against the regenerated payload).
> **Owner queue / rotation:** took the top open owner item - the capex proposal waiting for a
> decision since 2026-10-14. That is also a legitimate BUILD day: the decision came with code.
> The Factor Lab item and the context-layer item were not advanced.
> 
> ### Did
> 
> **Reviewed the open capex proposal and refused it, on a measurement its author had not made.**
> The proposal (`research/2026-10-14-capex-definition.md` section 4) would have taken a captive
> finance arm's lease purchases out of free-cash-flow capex for GM, DE, CAT and PCAR - GM's FCF
> yield 0.3% -> 8.0%, its sector percentile about 10 -> 90, 7.9 composite points on the heaviest
> metric in the screener (9.9 of 100). Section 4's figures all reproduce. What it had not looked at
> is the **other half of the same ratio**:
> 
> - **Its own sources define free cash flow on the segment, not on the consolidated company.** GM's
>   fiscal 2025 10-K, read today: *"net automotive cash provided by operating activities was $18.7
>   billion, capital expenditures were $9.2 billion"*. The consolidated operating cash flow for the
>   same year is **$26.87B**. The proposal removes the finance arm's capex and keeps its **$8.17B**
>   of operating cash flow. Caterpillar's 8-K of 2026-06-11 is the same shape - "MP&E free cash
>   flow", the industrial segments on both sides.
> - **So the proposal's number is not GM's.** For fiscal 2025: today's construction $1.78B
>   (**-81%** against GM's own $9.5B unadjusted, $10.6B adjusted); the proposal's $17.57B
>   (**+85%**). The error is *exactly* the consolidated-minus-automotive operating-cash-flow gap.
> - **A lease fleet is a three-legged cycle and the rule would count one and a half.** GM's own
>   tags, fiscal 2025: bought $15.79B, collected $7.80B of rentals (inside operating cash flow, so
>   **kept**), sold $10.10B at lease end (investing - counted by neither construction). Net
>   investment in the book was **$5.69B**, not $15.79B. Netting legs 1 and 3 lands nearest the
>   company's figure (+25%) and is **not buildable**: only GM of the four files the proceeds leg.
> - **It would also put 9.9 composite points on the weekly SEC cache**, which scoring deliberately
>   never reads (a test enforces it). A failed refresh would silently restate four companies' capex.
> - **Direction.** Both constructions are measurably wrong; the proposal's *promotes* a stock on a
>   number no source publishes. For a long-only screener those are not equivalent.
> ...

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

