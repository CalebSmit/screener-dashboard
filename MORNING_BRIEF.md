# Morning Brief - Friday 09 October 2026, 02:21

Written automatically after each run. Newest state only - the full
history is in `NIGHTLY_LOG.md`.

## At a glance

| | |
|---|---|
| Data run (2 AM) | **completed** - last ran today |
| Code session (6 AM) | **completed** - last ran today |
| Dashboard data from | 2026-10-09T02:00:03.566643 |
| Stocks scored | 501 |
| With a price | 501/501 |
| With an analyst target | 497/501 |
| Top 5 | EXPE, HST, APA, BBY, DLTR |
| Evidence for weight changes | 4 of 8 needed at the 1m horizon (26 rows, but overlapping windows are not independent; 75 rows across all horizons), newest 2026-10-02 |

## What changed in the repo

- `e8991bc data: screener run 2026-10-09 - 501 scored, top: EXPE HST APA BBY DLTR`
- `b8039f9 insiders: count only the company's own filings, keep 10%+ holders apart, one trade per filing-day`
- `48066b2 Reporting Soon, and insider trades from the SEC's own Form 4 filings`
- `88fa678 context layer: ship it in its own file, loaded after the page is usable; context pass measured`
- `db38cf3 context layer, first draft: technicals, options, insider trades, market backdrop and the ranking's track record - shown beside the score, never in it`
- `28237af brief: code session 2026-10-08`
- `bbe5758 log: 2026-10-08 session entry - gate results`
- `576951b fix: factor_vol_history keeps one row per date; clearer drawdown input labels; docs`
- `1077e83 wip: rescore with the drawdown fix; index.html is the dashboard, not a redirect stub`
- `6fc0297 wip: changelog entry, scoring_schema bump to 3, log entry started`
- `1b8613b fix: one observation per run date in the evidence base (priority 0.6)`
- `d99c20d wip: max_drawdown_1y measures the price path, and publishes the two closes it measured between`
- `f9582f7 measure: 22.6% of composite has no shown arithmetic; max_drawdown_1y compounds log returns as simple`
- `8a8c195 brief: data run 2026-10-08`
- `d704cc6 data: screener run 2026-10-08 - 502 scored, top: EXPE HST BBY APA DLTR`

## The session's own account

> 2026-10-08 (night) - OWNER-RUN: Reporting Soon; insider trades from the SEC, with two defects caught before publishing
> 
> Owner gave a contact email for the SEC's User-Agent (*"yes you can do that"*). It is stored **outside the
> repo** - `data/sec/user_agent.txt`, gitignored, or `SEC_USER_AGENT` - and a test fails if it appears in
> any file on the nightly data path. (It already sits in two older tracked files, a 2026-10-05 measurement
> script and the owner's setup script; those were left alone.)
> 
> **1. Reporting Soon** (`sec-reporting`, nav "Reporting") - every scored stock reporting within 7 or 14
> days of the run, by date then rank, never by expected move; options-implied move where the expiry spans
> the report; officer/director buyers; est./held tags; scope all / top 100 / My Holdings. **23** and
> **103** companies on this run (earnings season). Checked live at 1440 and 375 px: no horizontal scroll,
> nav still fits at 1440. Shipped as `good/2026-10-08-owner-3`.
> 
> **2. Insider trades from SEC Form 4s.** `run_screener` refreshes every Form 4 filed in 180 days after the
> context pass (900 s budget; cache `data/insider/filings.json`), and a stock whose record is fresh uses it;
> the rest keep Yahoo's rows. The card tags **plan** sales (the Form 4's Rule 10b5-1 checkbox), states the
> share of sale value on plans, links each trade's date to its filing and names its source. First fill:
> 14,643 filings, 0 failures, 43 min.
> 
> **Two defects found in the first fill and fixed before any of it published** (the 02:00 run is the first
> to publish SEC data):
> - **Wrong issuer.** A company's EDGAR list also holds Form 4s it filed as an *owner of another company* -
>   Berkshire's Lennar purchases showed as Berkshire insider buying; Goldman and Prudential likewise.
>   `parse_form4` keeps `issuerCik`; only filings whose issuer is the company count; the cache was re-read
>   (14,643 filings, 0 failures, 39 min). This removed **$6.6bn** of misattributed sales.
> - **10%+ holders.** Cascade Investment's $1.38bn of Republic Services read as "insider buying". Holders
>   who are neither officers nor directors are now listed but counted on their own line
>   (`holder_only`): **92 of 189** purchase lines, all in **3** stocks. Whether the literature supports
>   treating them differently is **not settled** - `plan/context-layer.md` item 3 has it as a research
>   question; no citation was added to the page for it.
> - Also: one filing's sale split across price tiers is one trade now (Apple's executive chair had four
>   rows for one sale).
> 
> **Measured after the fixes** (`research/measurements/2026-10-08-insider-sec-vs-yahoo.py`): SEC record for
> 502 of 503; officer/director buying at **60** stocks (SEC) vs **54** (Yahoo); sales at 338 vs 318; sale
> value $17.8bn, **72.4% on 10b5-1 plans**. Rendered from a copy of the run into scratch: main payload
> byte-identical, 501 stocks with SEC insider data, plan tags and filing links present.
> 
> **For the next session:** read the 02:00 data log for `insider refresh:` (should be ~500 requests plus
> the day's new filings, well inside 900 s) and `insider trades from SEC Form 4 for N stocks` (N ~ 500), and
> open one drilldown on the live site to confirm the SEC source line.

---

If a run says **stopped deliberately**, that is the safety gates working:
the live dashboard was left untouched rather than published with bad data.
`logs/` has the detail, and `ROLLBACK.md` covers undoing anything.

