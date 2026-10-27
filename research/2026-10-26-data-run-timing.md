# Where the 02:00 data run's 27 minutes go - and what can be cut without changing a number

*2026-10-26 (owner-run night). Measurement and infrastructure, not methodology: nothing here
changes what is fetched, scored or published. `METHODOLOGY_CHANGELOG.md` is untouched because no
weight, metric, threshold or input changed.*

**Proposal status: OPEN** - overlap the SEC Form 4 refresh (and the other SEC/FRED passes) with
the Yahoo ticker fetch, section 5 item 1; read the first `PHASE TIMING` line in
`logs/datarun-*.log` before deciding.

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
