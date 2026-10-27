# Where the 02:00 data run's 27 minutes go - and what can be cut without changing a number

*2026-10-26 (owner-run night). Measurement and infrastructure, not methodology: nothing here
changes what is fetched, scored or published. `METHODOLOGY_CHANGELOG.md` is untouched because no
weight, metric, threshold or input changed.*

**Proposal status: DECIDED 2026-10-27 (06:00 session) - REFUSED as written**, on the measurement
this note nominated as its deciding evidence. The first `PHASE TIMING` line from a real 02:00 run
puts the step the proposal wants to overlap at **87.5 s, not the ~400 s section 3 inferred** - so
the saving is ~1.5 minutes, not 5-7, and it would be bought by threading CPU work into the one
phase that was rate-limited that morning. Section 5's ranking inverts with it: **item 2 now saves
more than item 1 and carries none of its risk.** Section 8 has the account.

## 1. The question

The 2026-10-26 data loop ran 02:00:01 - 02:27:17. `run_screener.py` was 22 minutes of it, and
`meta.json` said only `fetch_time: 1065` - one number covering the ticker fetch, two SEC passes,
the context layer and the metric computation. The loop's log keeps the screener's last 25 output
lines and nothing else. Nobody could say where the time went, so nobody could say what was safe
to cut. `meta.json` also said `cache_status: WARM, tickers_cache: 0` while `config.yaml` says
fundamentals refresh every 7 days.

## 2. What the `caching:` block actually governs

One thing: whether the **fully scored** `factor_scores` parquet from an earlier run may be reused
instead of running the pipeline at all. `factor_engine.factor_scores_cache_max_age_days` takes
`min(price_data_refresh_days, fundamental_data_refresh_days)` = **1 day**, and `cache_is_usable`
reads "1" as "only a cache written today". The 02:00 run always finds yesterday's file, so it is
`WARM` (a cache exists but is too old) and refetches everything. `tickers_cache` is set only on the
`HOT` path, to the universe size; there is **no per-ticker or per-statement cache of anything**.
`estimate_data_refresh_days` is read by nothing but `schemas.py`. This was already recorded as a
false page claim (`research/2026-10-19-methodology-claims-audit.md` row 2) and the page sentence
was fixed; the config block itself still reads like a fundamentals cache that does not exist.
**Every weekday run refetches every statement of every company.** Whether it should is section 5
item 3 - a data-freshness decision, not a speedup.

## 3. The measured breakdown, 2026-10-26 02:00

No phase timing existed, so this is reconstructed from evidence with timestamps: the run's own
`run.log` (artifact saves), `meta.json`, the per-ticker `fetch_time_ms` of all 503 tickers, the
modification times of every file the run wrote (to the 0.1 s), and the loop's log.

| Stage | Seconds | How it is known |
|---|---:|---|
| Universe, network probe (one full ticker fetch), risk-free rate, market returns | ~10 | subset runs tonight: 1.8 + 5.8 + 0.6 + 1.0 |
| **Yahoo ticker fetch** (`fetch_all_tickers`, 503 x 12 requests) | **452** | the run's 503 per-ticker times (sum 1,178 s) list-scheduled over 3 workers in 17 batches of 30 plus 16 x 3 s delays; section 3.1 |
| SEC ROE history (cached facts) | ~5 | timed tonight on the full universe: 5.3 s |
| Valuation vs own history (cached) | ~18 | timed tonight on the full universe: 17.9 s |
| **Context pass** (option quotes not served at 02:00, so: Yahoo insider rows only) | **178** | `meta.json` `context_pass.seconds` |
| **SEC Form 4 refresh** (~540 EDGAR requests) | **~400** | the remainder: 1,065 less every other fetch stage; section 3.2 |
| Rate sensitivity (one FRED series) | ~2 | `data/market/DGS10.csv` written 02:17:47 |
| Metrics, checks, scoring, diagnostics, artifacts | 2 | artifact saves 02:17:50.0 - 02:17:52.0 |
| Portfolio construction (price download) | 6.5 | `08_model_portfolio` saved 02:17:58.5 |
| Excel, data-quality log, meta.json, overview | 1.4 | file times |
| Market backdrop (11 FRED series) | 13 | `data/market/*.csv` 02:17:59.9 - 02:18:12.9 |
| Track record (price download) | 34 | `track_record.json` 02:18:46.7 |
| Signal log, context evaluation | 0.2 | file times |
| Dashboard generation (first time) | 25 | snapshot written 02:19:12 right after it |
| **Improvement snapshot** - forward returns + live IC | **181** | snapshot 02:19:12; `price_cache` 02:20:25 and 02:21:44; `performance_history.csv` 02:22:12.7 |
| *Screener total* | *1,330* | *02:00:04 - 02:22:14* |
| Health check | 13 | loop log |
| Dashboard generation (second time, `data-run.ps1`) | 26 | loop log |
| Published-claims gate | 165 | loop log (another agent is working on this tonight) |
| Signals, stage, commit, push | 12 | loop log - **live at 02:25:50** |
| `git gc --auto` + evidence report (after publishing) | 81 | loop log |
| Archive, prune, morning brief | 6 | loop log |

### 3.1 The Yahoo fetch: 12 requests per ticker, none duplicated

A request-level probe (every HTTP call yfinance made inside `_fetch_single_ticker_inner`, five
tickers serially) found **12 requests per ticker and no duplicate**: 6 statement timeseries
(annual and quarterly income, balance, cash flow), 3 for `.info` (quoteSummary, v7 quote, one
timeseries for the PEG ratio), the 13-month chart, `earningsHistory`, `earningsTrend`. Mean
request ~0.15-0.3 s; the rest of a ticker's ~2.3 s is parsing, which three threads share under
the GIL. 503 tickers is **6,036 Yahoo requests a day**.

Scheduling, from the per-ticker times of all nine runs retained (2026-10-16 to 10-26): the work is
1,044-1,264 s per run, i.e. 348-421 s at three workers if perfectly packed; the 30-ticker batches
add 48 s of inter-batch delay and 23-37 worker-seconds (8-12 s) of straggler wait. So the fetch is
**~87% request-and-parse time, ~11% delay, ~2% stragglers**. The "~500 s unaccounted" in the
brief for this work was not in the ticker fetch at all - it is the SEC Form 4 refresh and the
improvement snapshot, neither of which had a number.

### 3.2 The SEC Form 4 refresh

`insider_activity.refresh` requests every company's EDGAR `submissions` JSON every day (to find
new Form 4s) at a 0.13 s floor, and on 10-26 made 540 requests. Its share is the remainder above;
timed directly tonight on the full universe it took **335 s** (section 6). Tonight's new phase
line will give the direct number from the first 02:00 run.

## 4. What shipped, and the proof that each is output-identical

### 4.1 Phase timing (`phase_timing.py`)

`RunContext` owns a `PhaseClock`; `run_screener` laps it at the end of each of 27 stages, so the
phases are contiguous and add up to the run. At the end the run's `meta.json` gains
`phase_seconds`, `phase_detail` (for the ticker fetch: tickers, workers, batch size, delay, the
summed per-ticker seconds and **the count of rate-limited tickers**) and `phase_total_seconds`,
added by `RunContext.update_metadata`, which can add keys and cannot change one (so `start_time`,
`end_time` and `run_id`, which the health check and the generator key on, cannot move). The
screener's last output line is now e.g.

    PHASE TIMING (s): total 121 | universe 2 | fetch.probe 6 | fetch.tickers 60 | ... | dashboard 12 | other 4

which is inside the 25 lines `data-run.ps1` copies into `logs/datarun-*.log`. Provenance only:
`tests/test_phase_timing.py` (19 tests) fails if any scoring or publishing module - including
`generate_dashboard`, `write_signals`, `check_run_health`, `run_fingerprint` - reads it, if the
line stops being the last thing `main` prints, or if a stage loses its lap.

### 4.2 Forward returns: no download that cannot measure anything (`improvement_engine.py`)

`compute_forward_returns` downloaded ~500 tickers' prices for every snapshot with a due horizon,
then measured it only if the frame held a close on or after the horizon end. The frame never holds
a bar dated today (`_fetch_prices_for_returns` drops it, and a cached frame was cut the same way
on the day it was written). So for a horizon ending **today**, or on the **weekend** before a
Monday run, the download could not succeed: the code took the `will retry` branch, and the frame
was not cached either (that needs a close on or after the window end, later still). On 10-26,
**four of the six downloads were of this kind** (3m ending 10-26, 1m ending 10-24 and 10-25, 1w
ending 10-26). `_close_possible_since(end)` - true iff a weekday lies in [end, today) - now skips
them; holidays are weekdays, so the rule only ever errs towards downloading.

**Identity, on the real state.** The improvement state as it stood before the 10-26 run
(`performance_history.csv` and `live_ic_history.csv` from commit `3e366b0`, the snapshots, the
price cache less the two files that run added) was copied twice and `compute_forward_returns` +
`compute_live_ic` run on each, old code then new, back to back:

| | old | new |
|---|---:|---|
| downloads | 6 | 2 (the same two that the old code cached) |
| seconds (evening, machine shared with other agents) | 582 | 204 |
| new performance rows | 1,001 | 1,001 |
| `performance_history.csv`, `live_ic_history.csv` | | **byte-identical** |
| price cache | 93 files | **same 93 files, every frame equal** |

`tests/test_forward_return_download_skip.py` repeats this on synthetic snapshots placed so a
horizon ends today, on each recent day and on a weekend: skip off vs on gives byte-identical
outputs and strictly fewer downloads. **Expected saving at 02:00:** replaying the rule over the
committed snapshot dates for every weekday run from 2026-09-29 to 2026-10-30 (a horizon is
measured on the first run with a weekday between its end and the run day) gives **4 downloads a
day under the old code and 2 under the new** (3 -> 1 on Wednesdays, 5-6 -> 2 on Mondays; 133 ->
83 over the 25 runs). The replay reproduces 10-26 exactly (6 -> 2). At ~30 s a download at 02:00
that is **~1 minute a day, ~2 on a Monday**, about half the step.

### 4.3 The context pass: no Yahoo insider request the SEC rows overwrite (`context_fetch.py`, `run_screener.py`)

At 02:00 option quotes are not served, so the context pass's request for each stock fetched only
Yahoo's insider rows - 503 requests paced 0.35 s apart, 178 s - and the SEC Form 4 step that ran
next replaced `_ctx_insider` with the SEC's rows for every stock whose SEC record is fresh: **503 of
503** on 10-26. The SEC refresh now runs first; `enrich(..., insider_from_sec=...)` skips those
stocks when quotes are not live (with live quotes the same request fetches the option chain, so
nothing is skipped); the SEC rows are assigned **after** the pass exactly as before, so every record
gains the same keys in the same order. A stock the SEC does not cover still gets Yahoo's rows.

**Identity, end to end.** The real pipeline (`run_screener.main`, 30-ticker subset, option quotes
forced to "not served" as at 02:00) was run twice back to back, skip disabled then enabled:

* `00_raw_fetch.parquet`: same 174 columns **in the same order**; `_ctx_insider` and
  `_ctx_insider_src` equal for all 30 stocks (all `sec`); every `_ctx_opt_*` field equal;
* `dashboard_context.js`: 0 insider fields differ;
* the fields that did differ are the ones two fetches minutes apart legitimately disagree on, all
  from the price history fetched *before* the context pass: `price_12m_ago`, `volatility_1y`,
  `_vol_daily_sd`, `price_6m_ago`, `_ctx_sma200(_prev)`, `_ctx_high/low_52w`, `_ctx_rate_beta/r2`
  (largest relative difference 2.5e-6 - Yahoo re-deriving dividend-adjusted closes), and
  `_fetch_time_ms`;
* Yahoo requests in the pass: 30 -> 0.

`tests/test_insider_request_skip.py` pins the skip, its limits and the ordering. **Expected saving
at 02:00: ~178 s**, and 503 fewer Yahoo requests in the hour Yahoo has been seen to throttle.

### 4.4 What runs for the first time at 02:00 tonight

All three. Gate 2 (`run_screener.py --dry-run`) passes; two real `--tickers` subset runs of the
changed code completed end to end (one with live quotes, one with quotes forced off); the run
directories were deleted and every tracked file they rewrote restored. Neither speedup touches
`factor_engine.py`, so the research engine's ledger is not marked stale.

## 5. Recommendations, ranked by minutes saved

> **Superseded 2026-10-27 by section 8.** Items 1 and 2 below are ranked on an inferred ~400 s for
> the Form 4 refresh. Measured at 02:00 it is **87.5 s**, so item 1's ceiling is ~1.5 min and
> **item 2 saves more than item 1**. Item 6's "no 02:00 data run in the record was rate-limited"
> is **no longer true** - 2026-10-27 recorded 39. Read section 8 before acting on this list.

1. **Overlap the SEC passes with the Yahoo fetch (~5-7 min).** The SEC Form 4 refresh (~400 s),
   the weekly companyfacts refresh (~225 s one day a week - 10-23's run was 1,300 s against
   ~1,075 s), and the FRED backdrop talk to different hosts from Yahoo and need nothing the ticker
   fetch produces except the list of tickers that did not fail. Run them in a thread while
   `fetch_all_tickers` runs and the wall time becomes roughly max(452, 400) instead of the sum.
   *Why not tonight:* the refresh list today excludes tickers whose Yahoo fetch failed, so
   overlapping it changes which cache entries are refreshed for a failed ticker (a ticker that is
   never scored, but the cache is shared state), and Form 4 XML parsing would compete for the GIL
   with the Yahoo parse. *Evidence that would justify it:* a subset A/B like 4.3 showing identical
   records, and one full run's `PHASE TIMING` showing the fetch phase did not slow. Risk: low; no
   change to Yahoo's request rate.
2. **Fetch Form 4 submissions only for companies that filed (~5 min if not overlapped).** EDGAR's
   daily form index lists every Form 4 by issuer CIK; one or two index requests could replace ~500
   `submissions` requests. *Evidence:* two weeks of shadow runs finding exactly the same new
   filings as the per-company scan. Risk: an index lag drops a filing for a day.
3. **Refetch statements only when a company has filed (~3-4 min, and half of Yahoo's daily
   requests).** 6 of each ticker's 12 requests are financial statements that change when a company
   files, and the SEC submissions the loop already downloads say when each company last filed a
   10-Q/10-K. *This is a data-freshness change, not a speedup:* Yahoo revises statements without
   a filing (restatements, reclassifications), and a cached statement would miss them. *Evidence:*
   a shadow comparison of daily statement fetches for 2-3 weeks counting fields that change with no
   filing; adopt only if that is ~0, and say so in the changelog.
4. **Generate the dashboard once (~25 s).** `run_screener` generates it before the improvement
   snapshot; `data-run.ps1` regenerates it after the health check. Check first whether the second
   build differs (it runs after the snapshot, so evidence-dependent sections may); if not, the loop
   can skip one. Needs a `data-run.ps1` change, not made tonight by instruction.
5. **The remaining forward-return downloads (~1 min).** The two real downloads fetch ~500 tickers
   for one window each; one download over the union window would do. Modest; prove identity as 4.2.
6. **Concurrency and delays: leave them.** Evidence from every log in `logs/` and every retained
   run: **no 02:00 data run in the record was rate-limited** - their `fetch_time` is 1,065-1,105 s
   on seven runs (1,300 s on 10-23 is the weekly SEC facts refresh), where a single 30 s + 60 s
   backoff and the retry pass would add 2-4 minutes. **Every throttling episode was a second or
   later full run within hours of another**: 2026-10-20 at 06:16 (four hours after the 02:00 run:
   backoffs at batches 13-14, 46 tickers limited, all recovered by the retry pass) and at 13:21
   (41 limited); 2026-10-09 evening rehearsals (36-136 limited, up to five backoffs). That pattern
   says Yahoo's limit is a rolling request budget, not a concurrency ceiling - so more workers
   would send the same 6,036 requests faster and spend that budget sooner. The simulation (same
   per-ticker times, which more threads would not keep) puts 4 workers at ~356 s and 5 at ~295 s,
   i.e. at most 1.5-2.5 minutes, against the risk of a throttled 02:00 run failing its publish
   gates. *Evidence that would justify a change:* the new `phase_detail.fetch.tickers.rate_limited`
   count at 0 on every 02:00 run for a month, then a one-step trial (4 workers) on a weekend owner
   run with no full run in the previous 12 hours. Cutting requests (items 2-3) buys more headroom
   than adding workers does.

Together the two shipped skips should take **~4 minutes** off the 02:00 run (~178 s + ~60-120 s)
- from ~27 to ~23 minutes, and from ~25:50 to ~21:50 until the page is live. Item 1 would take
another 5-7.

## 6. Full-universe SEC Form 4 refresh, timed tonight

`insider_activity.refresh` on all 503 tickers, every EDGAR request timed (21:27-21:35 CT, the
worktree's own copy of the cache, the SEC's 0.13 s floor unchanged):

| | requests | total s | median s | p95 s | median size | downloaded |
|---|---:|---:|---:|---:|---:|---:|
| `submissions` JSON (one per company, to find new Form 4s) | 503 | 290 | 0.50 | 1.26 | 156 KB | 92 MB |
| Form 4 documents (the new filings) | 57 | 42 | 0.71 | 1.36 | 8 KB | 0.5 MB |
| **wall** | 560 | **335** | | | | |

So the step is **~5.5 minutes**, consistent with the ~400 s remainder in section 3 at 02:00, and
**87% of it is downloading 92 MB of filing indexes to find 57 new documents.** That is what makes
section 5 items 1 and 2 the largest remaining savings.

## 7. Reproducing

Scratch scripts used tonight (not committed): a yfinance request probe wrapping
`YfData._make_request`; a list-scheduling replay of `fetch_all_tickers` from each run's
`run.log`; the forward-return sandbox replay; an off-hours `run_screener.main` driver forcing
`options_cache.quotes_are_live` to False with and without the skip; a field-by-field comparison of
two runs' `00_raw_fetch.parquet` and `dashboard_context.js`. From tonight on, `meta.json`'s
`phase_seconds` makes most of section 3 a lookup.

## 8. The decision, 2026-10-27 (06:00 session): item 1 refused, and the ranking inverts

This note left one proposal open and named the evidence that should decide it: *"read the first
`PHASE TIMING` line in `logs/datarun-*.log` before deciding."* That line now exists. It refuses
the proposal, and it corrects two other numbers here.

### 8.1 The deciding number

`logs/datarun-2026-10-27_020002.log`, the first 02:00 run with phase timing:

    PHASE TIMING (s): total 1118 | universe 1 | fetch.probe 3 | fetch.tickers 659 | fetch.retry 125
      | fetch.sec_roe_history 5 | context.valuation_history 14 | context.sec_insider 88
      | context.options_pass 1 | context.rate_sensitivity 4 | portfolio 3 | excel 1
      | context.market_backdrop 16 | context.track_record 35 | dashboard 24 | snapshot 136 | other 3

**`context.sec_insider` - the SEC Form 4 refresh, the step item 1 exists to overlap - is 87.5 s.**
Section 3 put it at ~400 s and section 6 timed it at 335 s. It is **4.5x smaller** than the
number the proposal was ranked on.

It was not a short run. `data/insider/filings.json` `_meta` for that run: **503 tickers checked,
566 requests, 63 new filings, 0 failures, `complete: true`** - the same full-universe scan section
6 timed, finishing in a quarter of the time.

### 8.2 Why section 6's 335 s did not transfer to 02:00

Section 6 timed EDGAR at **21:27-21:35 CT** and measured a median `submissions` response of
0.50 s. `insider_activity.MIN_INTERVAL` is 0.13 s, so at 02:00 the step is **floor-bound, not
latency-bound**:

| | requests | floor we impose (n x 0.13 s) | measured | EDGAR's share |
|---|---:|---:|---:|---:|
| 21:30, section 6 | 560 | 72.8 s | 335 s | ~262 s |
| **02:00, measured** | **566** | **73.6 s** | **87.5 s** | **~14 s** |

At 02:00 the refresh runs **19% above a politeness floor this project chose itself**. There is
almost nothing there to overlap: even an infinitely fast EDGAR leaves 74 s.

Section 3's ~400 s was a *remainder* - `fetch_time` less every other fetch stage - so it absorbed
whatever the list-scheduling model under-counted. It did: the model put the Yahoo fetch at 452 s
and the measured `fetch.tickers` is **659 s**, with `sum_ticker_seconds` 1,032 over 3 workers.
The missing ~500 s section 3.1 went looking for was mostly the ticker fetch itself.

### 8.3 What item 1 could actually save, and what it would cost

Overlappable against the Yahoo fetch, measured: `context.sec_insider` 87.5 s plus
`fetch.sec_roe_history` 4.6 s - **~92 s, about 1.5 minutes.** The other passes the proposal
gestures at are not free: `context.rate_sensitivity` (4.4 s) needs each stock's daily returns from
the fetch, and `context.market_backdrop` (16.2 s) runs after scoring. So ~1.5 min is the ceiling,
against the 5-7 min claimed.

The costs are the ones this note already listed, unchanged - Form 4 XML parsing competing for the
GIL with three Yahoo-parsing threads, and a refresh list that today excludes tickers whose Yahoo
fetch failed, so overlapping changes which cache entries are refreshed for a failed ticker.

What is new is where that cost lands. The same run recorded
`phase_detail.fetch.tickers.rate_limited: 39` and a **124.7 s `fetch.retry` pass**. The fetch is
both the longest phase and the one with a demonstrated failure mode. **Threading extra CPU work
into it to buy 92 s is the wrong trade**, and the proposal's own acceptance test - "one full run's
`PHASE TIMING` showing the fetch phase did not slow" - would be measured against a phase whose
run-to-run spread is already larger than the prize.

### 8.4 The ranking inverts: item 2 dominates item 1

Item 2 replaces ~566 per-company `submissions` requests with one or two EDGAR daily-index
requests. On the measured number that takes the step from **87.5 s to ~2 s - a larger saving than
overlapping can achieve at all** - with no concurrency, no GIL contention, no shared-cache
question, and it drops the 92 MB of filing indexes section 6 found are downloaded daily to find 57
documents. Its own risk (an index lag dropping a filing for a day) and its evidence bar (two weeks
of shadow runs finding the same filings) are unchanged and are the right next step in this area.

**So: item 1 is refused as written. Item 2 is the one worth building**, and whoever builds it
should take its saving as ~85 s, not the ~5 min item 2's own line claims - that figure was also
derived from the 400 s.

### 8.5 Item 6's premise is now false, and its tripwire has fired

Item 6 states: *"no 02:00 data run in the record was rate-limited"*, and proposes raising worker
count only after *"the new `phase_detail.fetch.tickers.rate_limited` count at 0 on every 02:00 run
for a month."* The first 02:00 run that could report that count reported **39**, with a 124.7 s
retry pass. The sentence is false as written and the pre-registered condition failed on day one.

**Honestly read, this is not a clean refutation of item 6's theory.** The owner-run night of
2026-10-26 merged at 22:38 and included two real subset `run_screener.main` runs and a yfinance
request probe; 02:00 is 3h22m later, which is inside the "within hours of another run" window the
theory names. So today is *consistent with* a rolling request budget rather than evidence against
it. The conclusion item 6 reaches - **leave concurrency and delays alone** - stands either way,
and now rests on a direct observation that the 02:00 slot is not immune rather than on an
inference that it was.

What should change is the claim, not the policy. The count is worth watching for the month item 6
asked for; a second rate-limited 02:00 run on a night with no preceding owner session would
distinguish the two readings.

### 8.6 What the two shipped skips actually delivered

Section 4 predicted ~4 minutes. Measured on `meta.json` `total_time_seconds`, the same stage
boundary on both runs: **1,075.1 s on 2026-10-26, 906.3 s on 2026-10-27 - 169 s, ~2.8 minutes**,
with the shortfall explained by the fetch phase running long and retrying.
`context.options_pass` is **1.0 s**, against the 178 s section 4.3 removed, so that skip landed
exactly as claimed.
