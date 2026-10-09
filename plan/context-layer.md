# The context layer - technicals, options, insiders, macro and the track record

**Owner, 2026-10-08:** *"This is where I want people to come for all of their investing needs ... maybe
we should take like technicals into account for timing of investments? Or macro data? Or options data?
... the goal is still long term ... if free somehow ... you would just build the first rough draft of
it, then the nightly sessions would really drill into it, and make sure that it is perfect moving
forward."* He chose all four drafts - **track record, a "Before you decide" panel, a market backdrop,
insider buying** - and chose **"context only, track them"** for how they relate to the score.

## The rule that governs all of it

**Context is shown beside the score, never in it.** The composite stays research-led (CLAUDE.md rules 4
and 5). Short-horizon timing signals do not have the published backing for a long-term ranking, and a
screener whose credibility is the product cannot fold them in on intuition. Instead:

1. Every context signal is **recorded every run** in `data/context_log/YYYY-MM-DD.parquet` (one file
   per date, replaced by a same-day rerun - the rule every observation file follows since 2026-10-08).
2. Joined to the forward returns the improvement engine already computes, each column becomes an
   **out-of-sample test** of that signal in this universe.
3. A signal may be proposed for the score only with **(a)** a research note (literature *and* practice,
   the Monday standard) and **(b)** a measured record at the `1m` horizon of at least the effective
   observations the improvement engine's own gate requires (`min_observations_for_proposal`, 8). Then a
   `METHODOLOGY_CHANGELOG.md` entry, like any other metric. Until both, it stays context.

Nothing in `context_signals.py`, `market_context.py`, `insider_activity.py` or `track_record.py` may be
imported by a scoring path (`tests/test_context_layer.py::test_no_scoring_module_reads_a_context_field`).

## What was built (first draft, 2026-10-08, owner-run)

| Piece | Where | Data | Cost per night |
|---|---|---|---|
| **Trend / range / recent move / volume** | `context_signals.price_context`, called in the fetch | the 13-month history the fetch already pulls | none |
| **Options: expected move, ATM IV, put skew, put/call OI** | `context_signals.options_context`, in the fetch | yfinance option chain, one expiry per stock | 2 calls/stock |
| **Insider open-market buys and sales, 90 days** | `insider_activity.rows_from_yahoo` + `summarise_rows`, in the fetch | yfinance `insider_transactions` (compiled from Form 4) | 1 call/stock |
| **Rate sensitivity** (return per +1pp in the 10-year yield) | `context_signals.rate_sensitivity`, in `run_screener` after the fetch | the fetch's daily returns + FRED `DGS10` | none extra |
| **Market backdrop** - 10 FRED series, readings, sector rate medians, factor notes | `market_context.build` -> `data/market_context.json` | FRED CSV (no key), cached `data/market/` | 10 small requests |
| **Track record** - top 25 vs RSP/SPY, top vs bottom fifth, per period | `track_record.build_from_disk` -> `data/track_record.json` | the comparable snapshots + yfinance closes, cached `data/track/` | 1 batched download (incremental) |
| **The record** | `context_signals.write_context_log` -> `data/context_log/` | the run's raw fetch | none |

Page: **Before you decide** in every drilldown (with a one-line teaser under "Why it ranks here");
**Market Backdrop** and **Track Record** sections; a **Context** filter on the rankings (uptrend,
downtrend, insider buying, reports within 14 days). Payload: per stock `ctx`, top level `market`, `track`.

**First measurements (2026-10-08):** track record Feb 20 -> Oct 8 (230 days): top 25 **+5.7%**, RSP
**+4.9%**, SPY **+13.6%**; top fifth minus bottom fifth **+2.5pp**; top 25 ahead of RSP in **4 of 7**
monthly periods. The backdrop: 10y-3m curve +0.99pp, credit spreads tight (4th percentile of 10 years),
VIX 15.1, CPI 3.7% y/y, Sahm 0.00.

## Known limits of the draft - the nightly queue, in order

1. **Fetch load.** The context calls add three requests per stock to the 02:00 fetch, which already
   meets Yahoo's rate limit. Measure the fetch time and failure rate against the week before (the
   data loop refuses to publish above 40% failures). If context costs the core fetch anything, move the
   options and insider calls to a **separate, time-budgeted pass after scoring** so they can never
   cost a scored number. Core data first, always.
2. **Option quotes at 2 AM** are the previous close; some chains have no bid/ask. Measure the share of
   stocks with `os == "ok"` and decide whether that is good enough or the options pass belongs in a
   market-hours run. Do not loosen `MAX_REL_SPREAD` or the IV bounds to raise the number.
3. **Insider source.** Yahoo's feed has no Rule 10b5-1 flag and no filing links; the SEC's EDGAR
   archive has both, and `insider_activity.refresh` / `parse_form4` already parse it - but
   `www.sec.gov` refuses any User-Agent without a contact email. **That email is the owner's to give;
   do not add one.** Ask in the log; if he provides it, switch the source and show "planned sale".
4. **One research note per signal** (Monday standard: literature with effect sizes *and* practice):
   200-day trend (Faber 2007; Brock, Lakonishok & LeBaron 1992), one-month reversal (Jegadeesh 1990;
   Lehmann 1990), implied-volatility skew (Xing, Zhang & Zhao 2010; Cremers & Weinbaum 2010 on put-call
   IV spreads), insider purchases (Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003; Cohen, Malloy
   & Pomorski 2012), macro regimes and factors (Asness, Frazzini & Pedersen 2019; Daniel & Moskowitz
   2016). Each note checks the page's sentence about it is fair, and records whether the signal is a
   **candidate** for the score.
5. **The evaluation harness.** Join `data/context_log/` to forward returns; report each signal's IC and
   effective observations in the morning brief once there are 3+ months. Reporting only - see the rule.
6. **Track record hardening.** Prices for names that left the index (some tickers fail to download -
   EQR, EA, AVB, CTRA, HOLX, DAY, SATS, BK, MMC on 2026-10-08; none was in a top 25); a turnover/cost
   estimate; whether RSP is the right benchmark after cap-weighted leadership; never list current
   holdings (the Model Portfolio decision, 2026-08-26).
7. **Macro.** Real-time vs revised data (Sahm's rule is defined on real-time data; FRED serves the
   latest vintage) - say so or use ALFRED. Consider the dollar and oil for sector context.

## Things not to do

- Do not add RSI/MACD "signals" or anything worded as a buy/sell trigger.
- Do not let a context field reach `raw`, `pct`, `cat_scores` or the composite.
- Do not show a list of current top-25 names as "the portfolio".
- Do not use the track record as evidence for a methodology change (rule 4).
