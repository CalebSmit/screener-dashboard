# Morning Brief - Thursday 15 October 2026, 02:23

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

## Waiting for a decision

A session researched these, proposed a change and left it for review. The next
session decides and records why; you can also answer in `OWNER_FOCUS.md`.

- **FCF capex excludes captive-finance lease assets for GM, DE, CAT, PCAR (section 4 of this note)** - open 1 day - `research/2026-10-14-capex-definition.md`

## What changed in the repo

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
- `1091181 context: withhold a skew whose legs cannot show the construction, and fix the window clause`
- `3cfd9b8 research: the skew construction, read from the papers, and a correction to the 10-12 note`
- `c09d343 context: the put skew is the measure the paper the page cites measured`
- `38503b7 tests: the suite no longer rewrites the published universe`

## The session's own account

> 2026-10-15 (owner-run, after midnight) - integrity sweeps, and the filing check watches its own source
> 
> **Health (rule 8):** unchanged from the 2026-10-14 entries - both loops ran 2026-10-14; the 02:00
> run of 2026-10-15 has not happened yet. Evidence base at `1m`: **29 rows, newest 2026-09-14 (31
> days, bound 40), 4 effective**. Top open roadmap item: **0.13, the Factor Lab** (age 3 days).
> **Owner session marked** (`logs/.owner-session`), removed at the end.
> 
> ### Did
> - **Payload invariants, checked on the published run**: 501 unique tickers, table and detail keys
>   identical, ranks in composite order (a competition ranking - ties share a rank, which is what the
>   bot accepts), every price positive, every category score within 0-100 (7 nulls, all coverage
>   gaps), trap flags boolean. Nothing wrong; and the bot validates the same things before it trades
>   (`bot/signals.build_signals`), so a second gate here would add little. Not built, on purpose.
> - **Analyst targets and estimates, the one input no filing can check**: 497 stocks; every mean
>   target inside its own high-low range; target / price from 0.57 to 1.78 (median 1.17 - analysts'
>   usual optimism, which a sector percentile neutralises); one stock under 3 analysts (TPL); three
>   FY2/FY1 estimate pairs far apart (MRK 2.67 -> 9.28, ARE, MOS) - charges in the current year's
>   consensus, which the EPS-basis guard already exists to catch. No defect.
> - **The filing check now watches its own source.** It depends on the weekly SEC cache; a refresh
>   that kept failing would leave it comparing against older and older filings silently. The health
>   note now ends with the cache's age, an `ATTENTION -` line fires past 14 days, and the brief's
>   filing row says how old the cached filings are. 2 + 1 tests.

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

