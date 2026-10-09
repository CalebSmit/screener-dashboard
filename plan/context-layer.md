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
| **Options: expected move, ATM IV, put skew, put/call OI** | `context_signals.options_context`, in `context_fetch.enrich` (after the core fetch) | yfinance option chain, one expiry per stock | 2 calls/stock |
| **Insider open-market buys and sales, 90 days** | `insider_activity.refresh` + `rows_from_sec` in `run_screener` after the context pass; `rows_from_yahoo` in `context_fetch.enrich` as the per-stock fallback | **SEC EDGAR Form 4** (plan flag, filing links); Yahoo `insider_transactions` where a stock's SEC record is older than 3 days | ~1 SEC request/stock + new filings; 1 Yahoo call/stock |
| **Rate sensitivity** (return per +1pp in the 10-year yield) | `context_signals.rate_sensitivity`, in `run_screener` after the fetch | the fetch's daily returns + FRED `DGS10` | none extra |
| **Market backdrop** - 10 FRED series, readings, sector rate medians, factor notes | `market_context.build` -> `data/market_context.json` | FRED CSV (no key), cached `data/market/` | 10 small requests |
| **Track record** - top 25 vs RSP/SPY, top vs bottom fifth, per period | `track_record.build_from_disk` -> `data/track_record.json` | the comparable snapshots + yfinance closes, cached `data/track/` | 1 batched download (incremental) |
| **The record** | `context_signals.write_context_log` -> `data/context_log/` | the run's raw fetch | none |

Page: **Before you decide** in every drilldown (with a one-line teaser under "Why it ranks here");
**Market Backdrop** and **Track Record** sections; a **Context** filter on the rankings (uptrend,
downtrend, insider buying, reports within 14 days).

**Payload: its own file.** `generate_dashboard.split_context()` moves every stock's `ctx` and the
top-level `market`, `track` and `ctx_weeks` into `dashboard_context.js` (`window.SCREENER_CONTEXT`),
which the page loads *after* it is usable and merges into `D` (`loadContext`, `CTX_LOADED`). Inline,
the context had grown the scored payload from 1.27 to 1.68 MB gzipped; split, the main file is
**1.29 MB** and the context **0.31 MB**, with every score, summary and table row byte-identical
(2026-10-08). Until it lands the drilldown says "Loading context"; if it fails, it says so. Both
files are written by the generator, copied by `run_screener` step 12 and committed by the data loop.
`tests/test_context_layer.py::test_the_main_payload_carries_no_context`.

**First measurements (2026-10-08):** track record Feb 20 -> Oct 8 (230 days): top 25 **+5.7%**, RSP
**+4.9%**, SPY **+13.6%**; top fifth minus bottom fifth **+2.5pp**; top 25 ahead of RSP in **4 of 7**
monthly periods. The backdrop: 10y-3m curve +0.99pp, credit spreads tight (4th percentile of 10 years),
VIX 15.1, CPI 3.7% y/y, Sahm 0.00.

## Known limits of the draft - the nightly queue, in order

1. **Fetch load - DONE 2026-10-08 (owner-run), keep watching.** Inside the core fetch, the three extra
   calls per stock tripped Yahoo's limiter at batch 9 of 17 and dropped the fetch to one worker. They
   now run in `context_fetch.enrich`, a separate pass after the core fetch with its own pacing, a
   rate-limit backoff and a 900 s budget. **Measured in isolation on the full universe:** 503 of 503
   stocks in **206 s**, **0** rate limits, **0** failures; options `ok` 394, `partial` 56,
   `stale-quotes` 25, `no-atm` 18, `no-chain` 9; insider 503; 471 chosen expiries span the next
   report. Nightly: read the `Context pass:` line in `logs/datarun-*.log` and record the seconds and
   whether it stopped. If it ever stops on `rate limited` two nights running, lower `WORKERS` to 1
   before anything else. Core data first, always.
2. **Option quotes at 2 AM - MEASURED AND FIXED IN CODE 2026-10-09; one machine-level step is
   outstanding.** The share with `os == "ok"` is not "some chains have no bid/ask", it is **none of
   them**:

   | When | Hour (ET) | n | `ok` | usable |
   |---|---|---|---|---|
   | 2026-10-08, owner-run | 21:27 | 503 | 394 | **78.3%** |
   | 2026-10-09, the 02:00 data loop | 03:00 | 503 | **0** | **0.0%** |
   | live probe, 2026-10-09 | 07:03 | 10 | 0 | **0.0%** |

   484 of 503 were `stale-quotes`. Overnight Yahoo serves the chain with **`bid` and `ask` both 0.00
   and `impliedVolatility` 0.000** on every strike; `lastPrice` and `openInterest` survive. So the
   loop that publishes the site spent ~1,000 option requests a night and showed a number to nobody,
   and the panel blamed quotes "missing or too wide at the time of the fetch" - which reads as a
   transient glitch rather than the hour. The 06:00 code loop is in the same dead window. Reproduce
   with `research/measurements/2026-10-09-option-quote-availability.py` (the record half needs no
   network; `--probe N` re-runs the live half at whatever hour you run it).

   **Not fixed by loosening a bar** - there is no quote to accept, and `lastPrice` is specifically
   the wrong substitute: Battalio & Schultz (2006) show apparent option mispricings largely vanish
   when quotes replace last trade prices. Fixed by fetching when quotes exist: `options_cache.py`
   keeps usable readings with the date the quotes belong to, and `context_fetch` **probes** two or
   three real chains before fetching 500 - quotes live, fetch as before; not live, skip the option
   half and read the cache, labelled with its quote date, with `_ctx_opt_status = "quotes-closed"`
   for a stock that has none. A probe rather than a hard-coded window, because a guessed clock goes
   wrong the day the source changes and three requests cannot. 02:00 now costs **3** option
   requests, not ~1,000.

   **Outstanding, and it is the first thing to do next:** the scheduled task that fills the cache -
   `Screener Option Quotes`, weekdays 20:00, defined in `scripts/register-tasks.ps1` - **was not
   registered on the machine.** The 2026-10-09 session was hard-blocked: every PowerShell
   invocation, including a read-only `Get-ScheduledTask`, is auto-denied in a non-interactive
   session, so this was outside its reach rather than left out of caution (rule 11). Until it is
   registered nothing fills the cache and every options panel says `quotes-closed` - still better
   than today, because the requests are no longer wasted and the page states the real reason, but
   the feature is empty. To finish: run
   `powershell -ExecutionPolicy Bypass -File scripts\register-tasks.ps1` (idempotent; it
   re-registers all three tasks), then confirm with `Get-ScheduledTask`/`Get-ScheduledTaskInfo`
   that `Screener Option Quotes` exists, has **one** trigger (weekly 20:00, no logon trigger), and
   has a `NextRunTime`; the two loops must still read PT3M and PT20M. Then read the next morning's
   `logs/datarun-*.log` for `Options: skipped the fetch and read the cache - N of ~500` with N in
   the high hundreds, and `logs/options-*.log` for what the 20:00 pass kept. Measure the `ok` share
   again at that point and record it here - **78.3% is the number to beat, and it is itself only one
   evening's observation.**

   Still true: do not loosen `MAX_REL_SPREAD` or the IV bounds to raise the number.
3. **Insider source - DONE 2026-10-08 (late, owner-run).** The owner gave a contact email; it lives in
   `data/sec/user_agent.txt` (gitignored) or `SEC_USER_AGENT`, **never in a tracked file**
   (`test_sec_identity_needs_an_email_and_never_lives_in_the_repo`). `run_screener` now refreshes every
   Form 4 (not 4/A - an amendment restates a filing and would double count) filed in 180 days, cached
   in `data/insider/filings.json`, and a stock whose SEC record was checked within 3 days uses it; the
   rest keep Yahoo's rows. The page marks **plan** sales (the Form 4's Rule 10b5-1 checkbox), states
   the share of sale value on pre-set plans, links each trade's date to its filing, and names its
   source. The first fill was run by the owner session; nightly cost is ~503 submissions requests
   plus the day's new filings at <8 requests/s, inside a 900 s budget. Nightly: read the
   `insider refresh:` and `Context: insider trades from SEC Form 4 for N stocks` lines in the data
   log - N should be ~500.

   **Two defects found and fixed the same night, before anything published:** (i) a company's
   EDGAR list also holds the Form 4s it filed *as an owner of another company* - Berkshire's
   filings for its Lennar purchases appeared as Berkshire insider buying, likewise Goldman and
   Prudential. `parse_form4` now keeps `issuerCik` and `rows_from_sec` reads only filings whose
   issuer is the company; cached entries parsed before that were re-read (14,643 filings, 0
   failures, 39 min) and `sec_rows_for` falls back to Yahoo for any entry not yet re-read.
   (ii) **10%+ holders who are neither officers nor directors** (Cascade Investment's $1.38bn of
   Republic Services) are listed but kept out of the officer-and-director counts
   (`holder_only`), and the card states their total on its own line. Also: lines of one filing
   with the same date and direction are one trade (a sale split across price tiers).

   **Measured after both fixes** (`research/measurements/2026-10-08-insider-sec-vs-yahoo.py`, run
   `91fa11d3ae8e`): SEC record for 502 of 503; stocks with officer/director buying in 90 days
   **60 (SEC) vs 54 (Yahoo)**, 51 in both, 9 SEC-only, 3 Yahoo-only; with sales 338 vs 318;
   purchase lines 97 officer/director vs 92 holder-only, the latter in 3 stocks; sale value
   $17.8bn, of which **72.4% on Rule 10b5-1 plans**. Open: why 3 stocks show buys only on Yahoo
   (candidate: Form 4/A, which is not read).

   **Research questions this opens:** (a) is the officer/director-vs-10%-holder split right?
   Seyhun's work is the usual lead for an "information hierarchy" among insider types; it was not
   verified this night and is **not cited on the page**. (b) Cohen, Malloy & Pomorski (2012)
   separate *routine* from *opportunistic* insiders by each person's own trading calendar; the
   180-day cache is too short for their three-year rule, so it needs a longer retention before it can
   be built.
4. **Step one done 2026-10-09 (owner-run):** every research sentence the panel prints was checked against
   its source and three were corrected (`research/2026-10-09-context-panel-claims.md`). Still to do -
   **One research note per signal** (Monday standard: literature with effect sizes *and* practice):
   200-day trend (Faber 2007; Brock, Lakonishok & LeBaron 1992), one-month reversal (Jegadeesh 1990;
   Lehmann 1990), implied-volatility skew (Xing, Zhang & Zhao 2010; Cremers & Weinbaum 2010 on put-call
   IV spreads), insider purchases (Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003; Cohen, Malloy
   & Pomorski 2012), macro regimes and factors (Asness, Frazzini & Pedersen 2019; Daniel & Moskowitz
   2016). Each note checks the page's sentence about it is fair, and records whether the signal is a
   **candidate** for the score.
5. **The evaluation harness - BUILT 2026-10-09 (owner-run).** `context_eval.py`: for every log date,
   the forward price return to the first log 30-40 days later (from the logs' own closes - no extra
   download), and per signal the Spearman IC across stocks; effective observations are the
   improvement engine's own non-overlapping count, and a t-statistic appears only from 3 effective
   observations, computed on those alone. 13 signals (trend distances, recent returns, 52-week
   position, volume, three option readings, insider buyers and value, rate sensitivity). Runs at the
   end of every full run, writes `data/context_eval.json` (committed by the data loop) and one line
   in the morning brief. **First one-month window closes 2026-11-07; the gate (8 effective) is about
   eight months of daily logs away.** `tests/test_context_eval.py`. Nightly: nothing to do until
   November except keep the logs complete; then read the brief line monthly.
6. **Track record hardening.** Prices for names that left the index (some tickers fail to download -
   EQR, EA, AVB, CTRA, HOLX, DAY, SATS, BK, MMC on 2026-10-08; none was in a top 25); a turnover/cost
   estimate; whether RSP is the right benchmark after cap-weighted leadership; never list current
   holdings (the Model Portfolio decision, 2026-08-26).
7. **Macro - Sahm DONE 2026-10-09 (owner-run):** the backdrop reads FRED's `SAHMREALTIME` (unemployment
   as first published) and says so, falling back to the revised computation with a note. Remaining:
   Real-time vs revised data (Sahm's rule is defined on real-time data; FRED serves the
   latest vintage) - say so or use ALFRED. Consider the dollar and oil for sector context.

## Things not to do

- Do not add RSI/MACD "signals" or anything worded as a buy/sell trigger.
- Do not let a context field reach `raw`, `pct`, `cat_scores` or the composite.
- Do not show a list of current top-25 names as "the portfolio".
- Do not use the track record as evidence for a methodology change (rule 4).
