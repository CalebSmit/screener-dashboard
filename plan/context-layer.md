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

## CLOSED 2026-10-20 - what was left, what was done, and the one thing only time delivers

Three sessions running had deferred CLAUDE.md priority 0.11 while this queue was nearly finished
and nobody had written down what remained. An audit of every item below and of every context panel
against the code (`context_signals`, `context_fetch`, `options_cache`, `insider_activity`,
`market_context`, `valuation_history`, `track_record`, `context_eval`, and how `generate_dashboard`
renders `ctx` / `market` / `track`) gives this list.

**Done but not marked (marked now):** item 1 (the pass ran 177-180 s, 0 rate limits, never stopped,
on every data run 10-16 to 10-20 - read it from the run's `meta.json` `context_pass`, **not** the
`datarun` log, which never carried the line this item told sessions to read); item 6's "record each
run's timestamp in its snapshot" (shipped 2026-10-19 - snapshots carry their start time); item 7's
macro vintages (closed 10-17); OWNER_FOCUS's "(7) macro data vintages".

**Genuinely open, and done tonight** (`METHODOLOGY_CHANGELOG.md` 2026-10-20 evening,
`tests/test_context_layer_claims.py`, 31 tests in the publish gate):

1. **A live false sentence.** A live reading carried no stamp and the options panel said "Quotes
   are the previous close" - for 485 of 501 stocks after the 13:00 session's 13:21 CT publish, whose
   quotes were mid-session. Live readings now record session, Eastern time and whether the market
   was open (`options_cache.live_quote_stamp`), and the panel says which.
2. **Mid-session prices in the evidence record.** The same run replaced the data loop's 2026-10-20
   log with the day's unfinished bar; `write_context_log` now refuses a mid-session run and the log
   was restored from git (the 06:16 run's, 2026-10-19 closes).
3. **The harness spliced two skew constructions** - item 4 said to exclude the old one and the code
   did not. It now applies the generator's own rule (both legs recorded), which excludes **six** log
   dates, 2026-10-08 to **10-15** (the 10-15 run read a cache refreshed before the fix merged), not
   the five item 4 says. ATM implied volatility, shown since 10-08, is now evaluated as well.
4. **Five computational sentences had no registered claim** - trend/recent move/volume windows and
   the trend word (one rule for chip, teaser and filter, checked in a browser against
   `trend_state`), the sector median, the momentum skip-month note, the backdrop's momentum-rule and
   Quality notes, the market cards' arithmetic, and "each is logged every run". Two of them said
   more than the code and were corrected: a quarter of Momentum (Jensen's alpha) does **not** skip
   the latest month; the stat strip shows a card only **when** the cut is in force.
5. **The harness was dry-run end to end** on the real logs plus synthetic future logs: every one of
   18 signals produces an IC; the brief line now names the first reading's date.

**A measured finding, not a defect:** 51 stocks lost "Against its own five years" on 2026-10-19
(449 -> 398). SEC's `companyfacts` API carries **no facts at all** from their Q2 2026 10-Qs (ABT
filed 2026-07-28, DUK 08-04, SPGI 07-28: present in `submissions`, absent from `companyfacts`),
so their newest trailing twelve months end 2026-03-31 and crossed `TODAY_MAX_AGE_DAYS = 200` that
weekend. Withholding is the designed behaviour; the panel returns when the source ingests the
filings or the Q3 10-Qs arrive. It also thins the two valuation-history signals' record for now.

**What is left: one item, and only time delivers it.** The first one-month observation for every
signal arrives with the **2026-11-09** run (the window opens 2026-11-07, a Saturday); the put
skew's, on its new construction only, about **2026-11-16**; and the gate of 8 effective
observations, about one a month, is roughly **mid-2027**. Read the morning brief's
`context_eval` line then. Until then this item is waiting, not deferred - **do not count it as
deferred work.** The research-level questions below (insider-type hierarchy, routine vs
opportunistic insiders, the three Yahoo-only buyers, an estimated trading cost, EV/EBITDA on the
valuation spine, the dollar and oil) are optional improvements, each needing a research note
first; none is a defect.

**What still binds:** settled row "ctx" (never in the score, logged every run, joins only with a
note *and* a measured `1m` record at the engine's gate); the option-implied signals and the macro
backdrop are **not** score candidates (findings, items 4 and 7); the track record is never
methodology evidence; the skew series before 2026-10-16's log is excluded, not spliced; a mid-session
run never writes the log; every computational sentence on a context panel has a claim in
`claims.py` checked by behaviour, not by the presence of a string.

## Known limits of the draft - the nightly queue, in order

1. **Fetch load - DONE 2026-10-08 (owner-run), keep watching.** Inside the core fetch, the three extra
   calls per stock tripped Yahoo's limiter at batch 9 of 17 and dropped the fetch to one worker. They
   now run in `context_fetch.enrich`, a separate pass after the core fetch with its own pacing, a
   rate-limit backoff and a 900 s budget. **Measured in isolation on the full universe:** 503 of 503
   stocks in **206 s**, **0** rate limits, **0** failures; options `ok` 394, `partial` 56,
   `stale-quotes` 25, `no-atm` 18, `no-chain` 9; insider 503; 471 chosen expiries span the next
   report. Nightly: read `context_pass` in the run's `runs/<id>/meta.json` (seconds, `stopped`,
   `options_live`) - the `Context pass:` line is printed to the run's console, not to
   `logs/datarun-*.log` (corrected 2026-10-20). If it ever stops on `rate limited` two nights running, lower `WORKERS` to 1
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

   **CLOSED 2026-10-12.** The task was registered and verified on 2026-10-09 (owner-run:
   `Get-ScheduledTask` read back one weekly 20:00 trigger, no logon trigger; the loops still PT3M /
   PT20M), and 2026-10-12 02:00 was the **first scheduled run after a 20:00 refresh**. Measured from
   `data/context_log/2026-10-12.parquet`: options `ok` **397 of 503**, `partial` 59,
   `quotes-closed` 47 - against **0 of 503** before the fix, and against the 394/503 (78.3%) owner-run
   evening reading that was the number to beat. 456 readings came from the cache. The after-close
   cache works end to end.

   **Closing it found a live false statement, fixed the same day.** The quotes were stamped
   **2026-10-10, a Saturday**, and the published page told readers of **455 of 501** stocks "Quotes
   are the close of Oct 10 ... so the last session's close is used." `options_cache.refresh` stamped
   `datetime.now(timezone.utc).date()`, and the task runs at 20:00 local = 21:00 ET, which is already
   the next UTC day - so **every** refresh named the session after the one its quotes came from; the
   Friday case is just the one that names a non-session. It also made `MAX_AGE_DAYS = 3` behave as
   four. Fixed by `options_cache.market_session_date` (Eastern date, weekends rolled back; exchange
   holidays are left to `quotes_are_live`, since a shut market serves no quote to accept) and
   `is_session_date`, **one predicate shared by `reading_for` and the dashboard generator** so the
   display cannot drift from its source. The generator drops the whole options block rather than only
   the date, because dropping only the date falls through to "Quotes are the previous close" - a claim
   about a session nobody checked. Measurement:
   `research/measurements/2026-10-12-option-quote-session-date.py`; repair of the cache and the run's
   copy: `scripts/repair_option_quote_session_dates.py`.

   **The lesson to carry:** the claim `context.expected_move` already asserted the panel "names that
   session's date", and its registered checks only proved that *a* date was carried through. **A
   claim whose check tests the plumbing instead of the assertion is an unchecked claim.**

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
   **One research note per signal - covered 2026-10-09 (owner-run):** insider purchases
   (`research/2026-10-09-insider-purchases.md` - weakest in large caps, now said on the page), one-month
   reversal (`research/2026-10-09-short-term-reversal.md` - faded in plain form, survives sector-relative;
   the harness now tracks both), trend (21/200-day distance, AKS 2021, now recorded), put skew and the
   rates note (addenda in `research/2026-10-09-context-panel-claims.md`).

   **The macro backdrop's note is DONE 2026-10-17** - `research/2026-10-17-macro-backdrop.md`, all
   five readings and all three factor notes against their sources, every DOI resolved through
   Crossref before it was written down. **Three page sentences were wrong about their own sources'
   records and are corrected in the same commit** (claims `context.sahm`, `context.macro_curve`,
   `context.macro_credit`; `tests/test_macro_backdrop_claims.py`, 13 tests, in the publish gate):
   the Sahm sentence said the rule "marked **the start**" of every recession since 1970 when the
   crossing came *after* the start in **6 of 8** (mean +0.9 months against the NBER peak - which is
   what Sahm's own chapter says) and the record holds **two** false alarms the page did not mention,
   one of them **2024**, inside the ten-year window the panel itself charts; the curve sentence
   credited Estrella & Mishkin (1998) with a lag range (measured **6 to 17** months, not 6 to 24)
   and a false alarm that paper states neither of, and the one clear false alarm (1966-67) is
   *outside* the period the sentence names; the credit sentence had **no source at all** and
   explained the spread by expected defaults, the one component Gilchrist & Zakrajsek (2012) show
   carries no predictive content. **No score, rank or number moved.**

   **The note also settles the standing question for this signal: macro is not a score candidate,
   and that is structural rather than pending** (note section 7). A macro reading is one observation
   a month for the *whole universe*, so it carries no cross-sectional variation for a
   cross-sectional ranking to use, and `context_eval.py` - which measures each signal's IC across
   stocks - cannot evaluate it even in principle. Using it to move *weights* instead is factor
   timing, which AQR's own authors find "deceptively difficult" (Asness, Chandra, Ilmanen & Israel
   2017), and index providers keep macro series out of factor construction. Do not wait on a `1m`
   record for the backdrop; there will not be one.

   **The pattern all three defects share, and the thing to carry into the track-record note:** each
   attached a real citation to a *stronger or different* statement than the source makes. Reading
   the source found them; measuring the record proved them. So the tests **rebuild each record from
   FRED's own series** (committed under `tests/fixtures/fred/`) rather than compare a sentence with
   a stored summary of it - item 2's lesson, that a claim whose check tests the plumbing is an
   unchecked claim, applied in advance this time.

   **The track record's note is DONE 2026-10-19 - `research/2026-10-19-track-record.md`, so every
   context signal now has a note.** It finds the record cannot discriminate skill (the day-by-day lead
   over RSP is 0.8 standard errors from zero after 165 days; 4.3 years to reach two at the same pace;
   5 of 7 periods is a coin toss's 23%), quotes Lo (2002), Bailey & López de Prado (2012), Fama &
   French (2010), McLean & Pontiff (2016) and GIPS 2020 (2.A.12, 2.A.13, 4.C.18, 4.C.48), DOIs resolved
   through Crossref. **Nine panel sentences said more than the arithmetic and are corrected**
   (note section 6; claims `context.track_record`, `context.track_luck`, `context.track_turnover`;
   `tests/test_track_record_claims.py`, in the publish gate; two `FORBIDDEN` patterns), among them
   "with nothing backfilled" (the February snapshots were written on 2026-03-09 from saved output),
   "the like-for-like benchmark", "monthly holding periods", a percentage-point spread printed as a
   percent, and a takeover-premium explanation beside two picks the screener's own records price
   flat. One was a defect in the arithmetic, fixed: see item 6. **The track record is not a score
   candidate, structurally** (it is the score's outcome). **What was open before that note:** the one
   signal with no note of its own - the track record
   (the options panel's note is done, and its three findings with it; see below). Plus each signal's
   *measured* record, which `context_eval.py` starts reporting after 2026-11-07. Original text:
   **One research note per signal** (Monday standard: literature with effect sizes *and* practice):
   200-day trend (Faber 2007; Brock, Lakonishok & LeBaron 1992), one-month reversal (Jegadeesh 1990;
   Lehmann 1990), implied-volatility skew (Xing, Zhang & Zhao 2010; Cremers & Weinbaum 2010 on put-call
   IV spreads), insider purchases (Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003; Cohen, Malloy
   & Pomorski 2012), macro regimes and factors (Asness, Frazzini & Pedersen 2019; Daniel & Moskowitz
   2016). Each note checks the page's sentence about it is fair, and records whether the signal is a
   **candidate** for the score.

   **The options panel's own note is done - `research/2026-10-12-option-implied-signals.md`**, a full
   Monday note on all four of its numbers (expected move, ATM IV, put skew, put/call open interest),
   literature and practice, with effect sizes and conditions.

   **Its verdict: the option-implied signals are NOT candidates for the score, and that is a finding
   rather than a holding position.** Index providers exclude option-implied inputs from factor
   selection (MSCI's factor list is Barra fundamentals, price history and analyst sentiment);
   derivatives desks use them because they trade daily. The panel's job is timing and sizing, which
   is what the owner asked for. **The verdict stands; one of its three legs does not** - see the
   correction below.

   **Its three page-sentence findings are DONE 2026-10-14 (13:00 session)** -
   `research/2026-10-14-volatility-skew-construction.md`, `METHODOLOGY_CHANGELOG.md` 2026-10-14
   (13:00), claim `context.put_skew`, `tests/test_option_implied_claims.py` (12) and 11 new tests in
   `tests/test_context_layer.py`. **No score moved.**

   - **(a) The skew is now the paper's measure**, not a citation attached to something else.
     `context_signals._xzz_skew`: the OTM put nearest 0.95 moneyness within (0.80, 0.95), minus the
     **ATM call** nearest 1 within [0.95, 1.05], positive open interest, with the paper's 10-60 day
     window recorded. The old measure was a 0.90-moneyness skew *minus half of Cremers & Weinbaum's
     put-call IV spread* - neither published quantity. Measured on 25 names from the same chains:
     **7 of 22 (32%) disagreed on the sign**, and the old one produced +46.6 IV points for one stock
     against the paper's 95th percentile of +19.92. Both legs are now published, so the row shows
     its own subtraction.
   - **(b) The decay is stated**: McLean & Pontiff (2016), 58% lower post-publication across 97
     predictors, as an upper bound. Read from the forthcoming-JF manuscript - the 2012/2013 working
     papers say 82 and ~35%.
   - **(c) Put/call open interest is off the page** and out of the payload; still logged and still
     evaluated by `context_eval.py` as `options_put_call_oi`.

   **A correction this produced, which the next session must carry
   (`research/2026-10-14-volatility-skew-construction.md` §4):** the 10-12 note listed XZZ as
   "weekly rebalanced" among signals that decay within a month, having had the paper second-hand.
   **Table 4 Panel B measures the same spread at a four-week holding period - 6.52% a year, t 2.70 -
   and at 6-7% with rising t-statistics out to 28 weeks**; the abstract says the predictability
   "persists for at least 6 months". The note conflated rebalancing frequency with decay horizon.
   **Withdrawn for XZZ**; it stands for Pan & Poteshman, Cremers & Weinbaum and Fodor et al. The
   skew is still not a score candidate (the shrinkage above, and this layer's own observation gate),
   but **do not read the first `1m` record as confirming something already known to be spent** - the
   paper claims an effect at that horizon, so a flat reading is evidence against *it*. And the
   `_ctx_opt_skew` series before 2026-10-14 is the old construction: **exclude those log dates, do
   not splice them.** *(2026-10-20: it is six dates, 2026-10-08 to 10-15, and the harness had been
   splicing them; `context_eval.skew_is_the_papers_measure` now excludes them by the generator's
   both-legs rule.)*

   One presentation note, deliberately not acted on: at 375px the skew row's value wraps to two
   lines because the label is long. The label stays long because "out-of-the-money put vs
   at-the-money call" teaches and "OTM vs ATM" does not, and the redesign is closed.
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
6. **Track record hardening - the price half DONE 2026-10-15 (owner-run):** `track_record.fill_unpriced`
   prices a renamed pick under its new symbol (CIK) and a pick that stopped trading from the snapshots'
   `price_at_scoring` until its last record (price only - the page says so); a pick with no price on
   the day its basket formed is still left out and listed (EA and HOLX in February: the snapshots carry
   no price before 2026-03-06). Claim `context.track_record`, 2 tests.
   **Turnover/cost DONE 2026-10-19** (`research/2026-10-19-track-record.md` section 7): one-way
   turnover per rebalance from the drifted basket (`one_way_turnover`; mean 31%, 19-53%) and the
   break-even cost per dollar traded that would erase the lead over RSP (`break_even_cost`; 1.48% on
   the cached prices), display-only, claim `context.track_turnover`. **No estimated cost is shown**:
   GIPS 2020 4.C.18 wants an estimate's basis disclosed and no primary source for large-cap US costs
   was verifiable this session (Frazzini, Israel & Moskowitz 2018 read only second-hand). Adding one
   needs that source first. **The same session fixed an entry-timing defect**: the 2026-08-10
   rebalance run was made after that day's close and entered at it; `runs_after_the_close` now reads
   each run's time of day from its own `price_at_scoring` (2026-02-20 and 04-14 record no prices and
   are named on the page as untimed). Do **not** "simplify" this to entering every run a day later:
   measured, that moves five periods whose runs came before the close and the headline from +12.7% to
   +17.2% (note section 6). **The benchmark question is answered as "keep both"** (note section 8):
   RSP isolates selection, SPY answers "better than the index I could hold"; they disagreed by more
   than the lead itself this window. ~~Open: record each run's timestamp in its snapshot~~ - done
   2026-10-19 (snapshots carry their run's start time; claim `context.track_record`).
   History: Prices for names that left the index (some tickers fail to download -
   EQR, EA, AVB, CTRA, HOLX, DAY, SATS, BK, MMC on 2026-10-08; none was in a top 25). *Diagnosed
   2026-10-15 (owner-run):* **four are renames** - the SEC CIK in `data/universe_history/ticker_ciks.json`
   maps EQR->VMRK, BK->BNY, MMC->MRSH, SATS->ECHO, each priced under its new symbol (backtest v2's
   `Inputs.current_symbol` is the same rule). The rest left the index (acquired); their prices at each
   run are in `improvement/snapshots/*` as `price_at_scoring` (CTRA to 2026-07-28, EA to 07-29), which
   would let a basket hold them to delisting instead of dropping them - label those as price return,
   not total return. On 2026-10-15 the unpriced picks were q1 CTRA, EA, HOLX and q5 SATS; the top-25
   basket had none; a turnover/cost
   estimate; whether RSP is the right benchmark after cap-weighted leadership; never list current
   holdings (the Model Portfolio decision, 2026-08-26).
7. **Macro - Sahm DONE 2026-10-09 (owner-run):** the backdrop reads FRED's `SAHMREALTIME` (unemployment
   as first published) and says so, falling back to the revised computation with a note. The
   real-time-vs-revised question that stood here is **closed 2026-10-17**: the page is on the
   real-time series, which is the right one, and Sahm's endnote 8 says why it matters - "the
   real-time data trigger **a few months later** than would the fully revised data", so the page is
   on the slower and more conservative of the two. ALFRED is not needed.
   Remaining: consider the dollar and oil for sector context - **but read
   `research/2026-10-17-macro-backdrop.md` section 7 first.** Another series is another thing to
   state correctly, and the note's three defects were all sentences, not numbers; a new reading
   earns its place only if it answers a question the panel cannot, with a source that says what the
   sentence says.
8. **Valuation against its own history - BUILT 2026-10-09 (owner-run).** `valuation_history.py`:
   earnings and FCF yield at 60 month-ends from SEC filings (as first filed, split-adjusted), shown
   for ~456 stocks that pass a market-value self-check; logged as `_ctx_vh_ey_pct` / `_ctx_vh_fy_pct`
   and evaluated by `context_eval.py`. Next: EV/EBITDA on the same spine (needs debt and cash per
   month-end), and a look at the 47 not shown once a month of logs exists.
   `research/2026-10-09-valuation-vs-own-history.md`.

## Things not to do

- Do not add RSI/MACD "signals" or anything worded as a buy/sell trigger.
- Do not let a context field reach `raw`, `pct`, `cat_scores` or the composite.
- Do not show a list of current top-25 names as "the portfolio".
- Do not use the track record as evidence for a methodology change (rule 4).
