# Methodology Changelog

Every change to **how the screener scores stocks** - factor weights, category
weights, metric weights, metric definitions, trap thresholds, scoring formulas,
neutralization, portfolio construction rules.

Changes here are **applied**, not proposed. Autonomous morning sessions may make
them directly. The obligation is not approval; it is **evidence**.

This file is the audit trail. If someone asks "why is Valuation weighted 22%?",
the answer must be findable here. A methodology change without an entry is a
bug, whether or not the code works.

## Entry format

```
## YYYY-MM-DD - Short title
**Area:** factor_weights / trap thresholds / metric definition / ...
**Changed:** exactly what, from what, to what
**Evidence:** citation, backtest result, IC measurement - the thing that
              justified it. Include effect sizes and the conditions.
**Expected effect:** what should move, and roughly how much
**Validated by:** the test/backtest that confirmed it, with the number
**Applied by:** improvement engine (auto) | morning session (manual)
**Rollback:** the tag or commit to revert to if this proves wrong
```

Entries by the improvement engine should also reference the IC observations and
information ratio that cleared its significance gates.

---

## 2026-08-05 - Enabled autonomous methodology evolution

**Area:** governance
**Changed:** `improvement.allow_auto_apply` false -> true in `config.yaml`.
Removed the human-approval requirement for methodology changes. Morning
sessions may now change scoring directly, and `improvement_engine.py` may write
weight changes once its statistical gates are satisfied.

**Evidence:** owner decision (2026-08-05) to run the project autonomously, with
evidence-backing rather than human review as the control.

**What did NOT change - deliberately:** the engine's statistical gates remain
exactly as they were:

| Gate | Value | Purpose |
|---|---|---|
| `min_observations_for_proposal` | 8 | no acting on noise |
| `min_ic_ir_for_auto_apply` | 0.5 | signal must be statistically real |
| `max_change_per_cycle` | 3.0% | no lurching |
| `shrinkage` | 0.5 | pull toward incumbent weights |
| `regime_scale_factor` | 0.0 | regime adjustment stays off until validated |

These gates *are* the safety mechanism now that human review is gone. Weakening
one requires its own changelog entry with a better argument than "it wasn't
firing."

**Expected effect:** none immediately. The engine has 3 live IC observations
and needs 8. First engine-applied change is realistically 2-3 weeks out, once
the data loop has accumulated evidence.

**Validated by:** `python -m pytest tests/ test_screener.py -q` -> 492 passed.
`tests/test_governance.py` covers the auto-apply gating.

**Applied by:** setup session (manual)
**Rollback:** set `allow_auto_apply: false` in `config.yaml`

---

## 2026-08-05 - Note: the learning loop had been inert since February

Not a methodology change; recorded because it explains the state of the
evidence base.

`improvement_engine.py` learns from snapshots recorded when the screener runs.
Between 2026-02-22 and 2026-08-05 the screener was not being run on a schedule,
so only **3 live IC observations** exist against a minimum of 8. The
self-improvement machinery has been present but starved.

`scripts/data-run.ps1` (Mon/Wed/Fri, 2:00 AM) now runs the screener and records
a snapshot each time. Evidence should clear the 8-observation gate in roughly
3 weeks.

**Implication for anyone reading the weights:** current factor weights are the
*designed* values from `config.yaml`, unchanged by live evidence. They have not
yet been validated against realized forward returns by this system.

---

## Open question that blocks trusting any of this

`backtest.py` carries two acknowledged biases, in its own docstring:
**survivorship bias** (uses today's S&P 500 constituents throughout history)
and **look-ahead bias** (fundamental scores held constant from a single
snapshot; only momentum and risk are recomputed).

A backtest with those properties **cannot honestly validate a methodology
change** - it will tend to flatter any strategy tilted toward stocks that
happen to be in the index today. Until this is fixed, "validated by backtest"
in this file should be read with suspicion, and IC measurements from the live
data loop are the more trustworthy evidence.

See `plan/backtest-v2.md`. This is priority 2 in `CLAUDE.md` for a
reason.

---

## 2026-08-11 - Research becomes the basis for methodology; backtest benched until 2027-02-11

**Area:** governance / evidence standard
**Applied by:** owner direction, recorded by setup session

**Changed.** Two related decisions:

1. **Published research and documented professional practice now carry equal
   weight to measured results** as justification for a methodology change.
   A well-sourced change no longer waits for a number.

2. **`backtest.py` output does not decide anything until 2027-02-11.** It may
   be run and reported as supporting colour. It may not justify a change, keep
   or revert one, appear under **Evidence** in this file, or act as a
   validation gate in the improvement engine.

**Evidence.** The backtest documents two biases in its own docstring:
survivorship (today's constituents applied across all history) and look-ahead
(fundamentals held constant from one snapshot). A result carrying both is not
weak evidence - its direction is unknown, and it systematically flatters
strategies tilted toward companies that still exist today, which is precisely
where this screener's valuation weighting sits.

Separately, `research/2026-08-10-ic-evidence-independence.md` established that
live IC evidence accrues far more slowly than the raw row count suggests -
at most 2 non-overlapping 30-day windows among 11 backfillable dates. Requiring
measured proof before any methodology change would therefore freeze the project
for roughly six months.

**Expected effect.** Methodology work proceeds on research grounds during a
period when no trustworthy measurement exists. Risk accepted knowingly: changes
made on literature and practitioner grounds are not yet confirmed by this
system's own data.

**How this is mitigated.** The written-argument requirement is unchanged - each
change still needs sources a sceptical reader can follow. Changes remain
individually reverted via their changelog entry, and `improvement_engine.py`
continues to learn from *live* forward returns, which are genuinely
out-of-sample and unaffected by the backtest's biases.

**Backtest observation:** none - deliberately.

**Validated by:** not yet. That is the point: revisit each entry made in this
period once backtest v2 exists or enough independent IC observations accrue.

**Rollback:** restore the prior standard by reverting CLAUDE.md rule 5 and the
"Your mandate" section. The date is the owner's to move.

---

## 2026-08-13 - The screener scored stale prices: factor_scores cache bounded by the wrong tier

**Area:** data freshness / what the published scores are computed from

**Changed:** `run_screener.py` bounded reuse of the `factor_scores` cache by
`caching.fundamental_data_refresh_days` (**7 days**). It now uses
`factor_scores_cache_max_age_days()`, which takes
`min(price_data_refresh_days, fundamental_data_refresh_days)` = **1 day**.

The comparison also changed from `age_days <= fresh_days` to a strict
`age_days < max_age_days` (`factor_engine.cache_is_usable`). Cache dates are
parsed from the filename and so are midnight-anchored; under `<=`, a cache
dated 2026-08-12 was still "fresh" at 02:00 on 2026-08-13 even with
`fresh_days = 1`. The rule is now stated explicitly and is teachable:
**`<tier>_refresh_days: N` means the cache is reusable for N calendar days
starting with the day it was written.** So `price_data_refresh_days: 1` means
"refetch unless the cache is from today".

**Why this is a methodology change, not a performance tweak.** `factor_scores`
is the *fully scored* dataset. **18 of the 44 metrics in `METRIC_COLS` move
with the daily close**, spanning five of the eight categories:

| Category | Price-driven metrics |
|---|---|
| Valuation | `ev_ebitda`, `fcf_yield`, `earnings_yield`, `ev_sales`, `pb_ratio`, `peg_ratio`, `dividend_yield` |
| Momentum | `return_12_1`, `return_6m`, `proximity_52w_high` |
| Risk | `volatility`, `beta`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown_1y`, `jensens_alpha` |
| Revisions | `price_target_upside` |
| Size | `size_log_mcap` |

Every valuation ratio has price or market cap in its numerator or denominator,
so "the fundamentals haven't changed" does not make a *valuation score* current.
Bounding this cache by the *fundamental* refresh window meant the published
Valuation, Momentum, Risk, Revisions and Size scores could be computed from a
close up to eight days old while presented as current. A cache is only as fresh
as its fastest-moving contents.

**Evidence - a documented, reproducible failure.**

1. *Direct arithmetic on the real artifacts.* The live cache is
   `cache/factor_scores_19c853468405_20260812.parquet`; the live config hash is
   `19c853468405`. At the 02:00 run on 2026-08-13, `age_days = 1`. Old rule:
   `1 <= 7` -> reuse. New rule: `1 < 1` is false -> fetch. Verified against the
   real on-disk cache and real config, not a fixture.

2. *It compounds, because a warm start never advances the cache date.* The
   warm-start path returns at `run_screener.py:1011`, before
   `write_scores_parquet` at `:1502`. So a warm-started run lays down no new
   cache file. One real fetch therefore suppressed the next seven days of
   fetches - **one real observation per eight daily runs**. Pinned by
   `test_old_bound_would_have_fetched_only_once` (measured: 1 fetch across 8
   consecutive daily runs) and `test_eight_consecutive_daily_runs_all_fetch`
   (after: 8 of 8).

3. *Three shipped incidents.* 2026-08-07 and 2026-08-10 published a dashboard
   in which 0 of 503 stocks had a price or analyst target and every category's
   dispersion collapsed 25-36% (`NIGHTLY_LOG.md` 2026-08-10). 2026-08-13 was
   caught by `scripts/check_run_health.py` and discarded - correctly, but the
   day's evidence was still lost. The detector added on 2026-08-10 was working;
   the underlying cause had never been found.

**Expected effect:** the 02:00 data loop performs a real fetch every weekday
instead of roughly one weekday in eight. The 18 price-driven metrics above are
computed from the most recent close rather than from a close up to eight days
old. Evidence accrual for `improvement_engine.py` speeds up by roughly 8x in
*calendar* terms - though note this does **not** speed up *independent* monthly
observations, which still accrue at about one a month
(`research/2026-08-10-ic-evidence-independence.md`). More runs is not more
independent evidence, and the engine's significance gates must keep counting
effective observations, not rows.

Runs will get slower (a real fetch instead of a 4.5s cache load) and will pick
up Yahoo's usual 10-25% ticker failure rate. Both are the intended cost of
fetching; `scripts/check_run_health.py` and the 40% failure-rate gate in
`scripts/data-run.ps1` remain the guards.

**Backtest observation:** none - not run, and it could not speak to this.

**Validated by:** `tests/test_cache_freshness.py`, 21 new tests, written red
before the fix (they failed with `ImportError` on the missing helpers, then
pinned the old numbers). Full suite `python -m pytest tests/ test_screener.py
-q`: **530 passed before -> 551 passed after, 0 failures**.
`python run_screener.py --dry-run` exits 0.

Real confirmation is the 2026-08-14 02:00 run: it must show a live fetch,
`00_raw_fetch.parquet` present, and pass the health check. If it warm-starts
again, this entry is wrong and should be revisited.

**Applied by:** morning session (manual)
**Rollback:** revert this commit; or set `caching.price_data_refresh_days: 7`
in `config.yaml` to restore the old effective window without touching code.

---

## 2026-08-20 - Weight changes come from research, not from the return series

**Area:** governance / evidence standard
**Applied by:** owner direction, recorded by setup session

**Changed:** `improvement.allow_auto_apply` **true -> false**. The engine still
records snapshots, computes forward returns and reports proposals; it may no
longer *write* a weight change. `CLAUDE.md` rule 4 was rewritten to match, and
the nightly prompt no longer lists an IC measurement as acceptable evidence.

This reverses the 2026-08-05 entry, deliberately.

**Evidence.** Three things, none of which is a backtest number:

1. **The evidence base is three observations.** `live_ic_history.csv` holds 3
   rows, all at the `1w` horizon, all from February 2026. The configured
   `optimization_horizon` is `1m`, which has none.
2. **The significance test overstates independence.**
   `research/2026-08-10-ic-evidence-independence.md` shows
   `_ir_to_one_sided_pvalue()` computes `t = IR * sqrt(n)` from raw row count.
   The 11 backfillable dates contain at most 2 non-overlapping 30-day windows,
   inflating t by ~2.35x and moving a borderline IR of 0.5 from p=0.24 to
   p=0.049 - through the gate on arithmetic alone.
3. **That correction has not shipped.** Verified 2026-08-20: no
   effective-observation counting exists in `improvement_engine.py`. So the
   moment the priority-0 backfill lands, the engine would have been armed to
   rewrite weights on inflated confidence.

The research note's own recommendation was explicit: ship the dedup fix *only*
alongside an independence correction, "or with `allow_auto_apply` temporarily
set back to `false`." The correction has not shipped, so this is that.

**Expected effect.** No change to current weights - the engine had never fired.
What changes is what happens *next*: methodology moves on research grounds, and
the backfill can now land safely without arming an under-powered gate.

**What this does not mean.** Weights are not frozen. Changing a factor weight
because the literature or documented practice says a factor is worth more or
less - with the argument written down - is legitimate and expected. What is
ruled out is changing one because a three-point return series drifted.

**Re-enable when both hold:**
1. `improvement_engine.py` counts effective (non-overlapping) observations in
   its significance test, per item 3 of the research note.
2. The history holds substantially more genuinely independent observations than
   the 8 raw rows the current gate asks for.

**Backtest observation:** none - benched until 2027-02-11 per rule 5.

**Validated by:** `python -m pytest tests/ test_screener.py -q`.
`tests/test_governance.py` covers auto-apply gating in both states.

**Rollback:** set `allow_auto_apply: true` in `config.yaml` and revert rule 4.

---

## 2026-08-24 - The evidence base: five defects that made the observation count meaningless

**Area:** improvement engine - forward-return accrual, IC computation,
statistical significance. No factor weight, metric or threshold changed.

**Changed.** `improvement_engine.py`, five linked defects (CLAUDE.md priority 0):

1. **Horizon-aware reprocessing.** `compute_forward_returns()` skipped any date
   already in `performance_history.csv`. A snapshot was processed once, at 7
   days old, when only the 1-week return existed; `fwd_return_1m` was written
   `NaN` and the date was never revisited. Eligibility is now tracked per
   `(run_date, horizon)`, so a date is reprocessed as it ages into the next
   horizon.
2. **One snapshot per run date.** Every snapshot file was processed, so a day
   with thirteen runs appended the same ticker-date rows thirteen times.
3. **Effective observation counting.** `_effective_observations()` counts
   non-overlapping return windows by greedy interval scan. Every gate now reads
   this number instead of the raw row count.
4. **Weekend run dates excluded.** A Saturday snapshot prices off Friday's
   close and its "one week later" price is the following Friday's close - the
   Friday observation counted twice.
5. **The data loop now computes live IC.** `record_run_snapshot()` called only
   `compute_dispersion()` and `compute_forward_returns()`, never
   `compute_live_ic()`.

Plus one defect found while fixing these: the price cache is keyed on
`(start, end)` and `end` was the *current* date, so every revisited snapshot
would have become a fresh full-universe yfinance download. The fetch window is
now bounded by the horizon being measured.

**Evidence - a documented failure, demonstrated.** All five are measured facts
about the live files, not inferences:

| Claim | Measurement |
|---|---|
| Duplicate rows | `performance_history.csv` held 20,057 rows for 8,020 unique `(run_date, ticker)` pairs - **60% duplicates** |
| Absurd IC inputs | `live_ic_history.csv` recorded **6,539 "tickers"** for 2026-02-21 in an S&P 500 screener |
| Weekend dates | 5 of 16 run dates were Saturdays or Sundays |
| 1-month returns never accrued | **1 of 16** dates carried a `fwd_return_1m`, while `optimization_horizon` is `'1m'` |
| IC series frozen | 3 rows, all `1w`, all February 2026 - unchanged for **183 days** while the data loop ran successfully every weekday |

The independence correction implements item 3 of
`research/2026-08-10-ic-evidence-independence.md`, which predicted the raw count
overstates independence by ~2.35x. `tests/test_evidence_integrity.py` now
demonstrates the consequence directly rather than arguing it: against the
pre-fix code, `propose_weight_changes()` returns **`proposal_ready`** on eleven
IC rows that are two independent observations. That was the failure mode
CLAUDE.md's priority-0 "STOP" warned about, and it is now a failing test.

**Effect on the evidence base** (`scripts/repair_evidence_base.py`, idempotent):

| | Before | After |
|---|---|---|
| `performance_history.csv` | 20,057 rows, 16 dates | 5,528 rows, 11 dates |
| `live_ic_history.csv` | 3 rows, newest 2026-02-22 | **23 rows, newest 2026-08-14** |
| Observations at `1m` (the optimization horizon) | **0** | 6 raw, **2 effective** |
| `n_tickers` per IC row | 1,006-6,539 | 499-511 |

**Expected effect:** no ranking or score changes - nothing in the scoring path
was touched. What changes is what the engine can see and what it will act on.
The engine still correctly refuses to propose: 2 effective 1-month observations
against a gate of 8.

**On the honest rate of accrual.** This does not clear the gate soon, and the
fix makes that *more* visible rather than less. Genuinely independent 1-month
observations accrue at about one a month, so 8 of them is roughly six more
months of daily running. The previous behaviour would have reached "8
observations" much sooner and been wrong.

**`allow_auto_apply` stays `false`.** Condition (a) in the `config.yaml`
comment - effective-observation counting - is now met. Condition (b), a history
with substantially more independent observations than the gate asks for, is
not: there are 2. Rule 4 stands and this entry does not relax it.

**Validated by:** `python -m pytest tests/ test_screener.py -q` -> **590
passed, 0 failed** (baseline at session start: 560 passed, 0 failed).
`tests/test_evidence_integrity.py` adds 30 tests; **24 of them fail against the
pre-fix code**, verified by checking the old file out and re-running.

Three pre-existing fixtures had to be corrected, and the correction is itself
part of the finding: `tests/test_governance.py::_write_ic_history` generated
dates as `(i % 28) + 1`, so "n=60 observations" was 28 distinct January dates
*repeated twice* - a single overlapping cluster. `test_improvement_engine.py`
and `test_metric_evolution.py` used consecutive calendar dates. All three now
space dates 35 days apart so that a fixture claiming n observations constructs
n independent ones. **No assertion was weakened**; the fixtures were made to
build the evidence their assertions always claimed.

**Backtest observation:** none - benched until 2027-02-11 per rule 5.

**Applied by:** morning session (manual)
**Rollback:** `good/2026-08-21-0616`. Note that reverting restores the inert
engine *and* the inflated history; `scripts/repair_evidence_base.py` is
idempotent and can be re-run afterwards.

---

## 2026-08-24 (evening) - The screener refuses to fabricate data

**Area:** data integrity / what may be published
**Applied by:** owner-run session

**Changed:** `run_factor_engine()` in `run_screener.py` caught a failed network
probe, set `USE_SAMPLE = True`, and generated "sector-realistic sample values"
for the entire universe. It now **exits 2 with an explanation** unless the new
`--allow-synthetic` flag is passed. The opt-in path additionally prints
`*** THIS OUTPUT IS FABRICATED. DO NOT PUBLISH IT. ***`.

**Evidence - a documented failure, not a citation.** On 2026-08-06 the 02:00
data run executed with no network. It fabricated all 503 tickers and produced a
normal-looking 2.6 MB dashboard payload reporting `stocks_scored: 503,
avg_composite: 50.5`. `scripts/data-run.ps1` committed it. The only reason
invented stock scores did not reach the public site is that the push failed on
the same dead network.

Nothing in the output distinguished it from a real run: the payload was a
normal size, the run summary reported no issues, and the single tell was
`validation/data_quality_log.csv` reading "Network unavailable - using
synthetic data" 503 times. A caller who did not read that file had no way to
know.

**Why this is a methodology matter.** `CLAUDE.md` opens with "its credibility
is the product". A screener that silently emits fiction when its data source is
down is not robust, it is dishonest - and the failure is invisible precisely
when it matters most. Refusing is the correct behaviour; a missing run is
recoverable, a fabricated one that gets believed is not.

**What did NOT change:** no factor weight, metric, threshold or scoring formula.
`_generate_sample_data()` itself is untouched and still available for pipeline
testing, which is what it was written for - it is only no longer reachable by
accident.

**Expected effect:** none on any healthy run. A run with no network now fails
loudly instead of publishing invented numbers.

**Validated by:** `tests/test_no_synthetic_by_default.py`, 6 tests. Verified
red against the pre-fix file: all four content assertions fail on
`git show HEAD:run_screener.py`. Full suite 595 -> 596 passing; gates 2, 3 and
4 re-run by hand.

**Backtest observation:** none - benched until 2027-02-11 per rule 5.

**Rollback:** revert the guard in `run_factor_engine()`; the flag can stay.

**Noticed while doing this, not fixed:** `cli.py` defines a near-identical
argument parser that **nothing imports except `tests/test_cli.py`**.
`run_screener.py` has its own `parse_args()` at line ~100, and that is the one
that runs. The first version of this change added the flag only to `cli.py`,
where it was completely inert - caught because `--help` did not list it. Two
parsers that drift apart, one of them tested and dead, is a trap; they should
be reconciled.

---

## 2026-08-25 - Which runs are comparable to each other: the history gate

**Area:** run comparability / display thresholds (no scoring change)

**What changed.** New module `history.py`, consumed by `generate_dashboard.py`
as a `history` block in the payload. It builds the dashboard's first time
dimension from the snapshots the data loop already writes, and it introduces
two thresholds a reader is entitled to check:

| Constant | Value | What it decides |
|---|---|---|
| `MIN_RANK_CONTINUITY` | 0.50 | whether a run may join the history at all |
| materiality | measured p95 of run-to-run abs(rank change), 54 today | whether a move is shown as a mover |

**What did NOT change:** no factor weight, category weight, metric definition,
trap threshold or scoring formula. Composites and ranks are exactly as before;
this decides only which past runs are placed beside each other, and which
differences are large enough to surface.

**Evidence - the exclusion rule.** The snapshot directory contains a run,
`2026-07-28`, whose ranking bears no relation to its neighbours: Spearman
**0.016** against the preceding run and **-0.020** against the following one,
with valuation dispersion 17.2 against a trailing median of 23.9. It predates
`scripts/check_run_health.py`, so nothing blocked it. Diffed naively it reports
**411 of 501 stocks (82%) moving more than 50 ranks**. A "biggest movers" panel
built without a gate would have led with pure artifact.

Measured over all 19 consecutive pairs in the directory, the 17 clean pairs
span **0.882 to 1.000** - the lowest being a 12-day gap, with a 29-day gap
still at 0.951 - and the only two breaks are the pair either side of
`2026-07-28`. **Any threshold between 0.05 and 0.87 classifies every observed
run identically**, so 0.50 sits in the middle of an empty region rather than
being fitted to one run.

**Evidence - why not the existing dispersion rule.** The first implementation
reused `check_run_health`'s "dispersion >20% below the trailing median". On the
real directory it excluded **16 of 20 runs**. Two causes, both worth recording:
risk-score dispersion has drifted legitimately from 26.7 (February) to 19.5
(August), and a history that baselines only on *kept* runs freezes its own
reference, so one exclusion cascades into excluding everything after it.
Dispersion remains correct at publish time, where the pipeline maintains a
baseline over every run. It is the wrong gate for judging comparability
*between* runs.

**Evidence - the materiality threshold.** Pooled over 13 consecutive clean
pairs (6,515 ticker-pairs), the distribution of absolute rank change between
runs is p50 **7**, p90 **36**, p95 **54**. Moves below that are ordinary
variation, so the panel surfaces only moves beyond p95 and says so on screen
with the sample size. The number is recomputed from the history at each build
rather than frozen.

**Evidence - round-trip flagging.** MNST's `return_12_1` percentile read 97.1
on 08-20, **2.9** on 08-21 and 08-24, then 97.1 again on 08-25, while its price
went 47.5 -> 48.9. A momentum score cannot cross 94 percentile points and back
on a 3% price move; the twelve-month return failed to compute for two runs.
Crucially it is **not** NaN - `factor_engine` correctly excludes missing
metrics via `na_option="keep"` and the `has_data` mask - so it is a *computed*
value from bad price history, and nothing downstream can distinguish it from a
real collapse.

On the 2026-08-25 run, **all 10** material one-day movers were excursions that
returned to base; over ~1 month, **169 of 193** were genuine trends. Movers
matching that signature are labelled `round-trip` rather than hidden, and the
default comparison is the ~1-month window rather than the previous run.

**Expected effect.** No stock's score or rank changes. The dashboard gains a
movers panel, a rank-delta column and a per-stock rank history. Two of 20
stored runs are excluded from the history, and both exclusions are printed on
the page with their reason.

**Validated by:** `tests/test_history.py` (31 tests), including a regression
that fails if `2026-07-28` ever rejoins the series and one that fails if the
gate becomes over-eager and rejects most real runs - the failure mode the first
implementation actually had. Plus `tests/test_dashboard_js.py` (12 tests).
Suite 596 -> 627, no pre-existing failures.

**Backtest observation:** none - benched until 2027-02-11 per rule 5. None of
the numbers above are forward returns or ICs; they are properties of the stored
snapshots, so rule 4 does not apply either.

**Rollback:** delete the `history` key from `dashboard_json` and the
`sec-changed` section; nothing else reads `history.py`.

**Open defect found while doing this, not fixed.** The MNST and FCX round-trips
are a real data-quality bug: a metric whose inputs fail transiently is scored
at an extreme percentile rather than being treated as missing. That silently
moves a stock ~100 ranks and, unlike a NaN, is invisible to every existing
check. Worth its own session - the movers panel is now the instrument that
makes it visible.

> **Diagnosed and fixed 2026-08-26 (entry below).** The cause was not a
> transient failure: Yahoo's MNST series mixes pre- and post-split prices.
> Note also that this entry has the direction backwards - the 97.1 reading
> was the artifact, not the 2.9 one.

---

## 2026-08-26 - A price series that mixes two split scales is refused, not scored

**Area:** metric definitions (momentum, risk) / data integrity

**Changed.** `factor_engine.check_price_series_integrity()` is new and runs on
every ticker's 13-month price history. When the series is internally
inconsistent across a declared stock split, the eight metrics derived from it
are withheld (set NaN) rather than computed:

| Category | Withheld | Kept |
|---|---|---|
| momentum | `return_12_1`, `return_6m`, `jensens_alpha` | `proximity_52w_high` |
| risk | `volatility`, `beta`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown_1y` | - |

Also withheld: `avg_daily_dollar_volume`, so the name drops out of the model
portfolio's liquidity filter. `price_latest` is deliberately **kept** - it is a
single point from the most recent bar, `info["currentPrice"]` takes precedence
over it everywhere it is used, and the defect is in relationships *between*
prices at different dates, which is exactly what the withheld metrics measure.

No weight, threshold or scoring formula changed. Every stock with a sound price
series scores identically to yesterday.

**Evidence - the documented failure.** Yahoo's 13-month series for MNST
alternates between pre- and post-split prices across its 2026-08-11 2:1 split:

```
2026-08-05    94.46      <- unadjusted
2026-08-06    47.08      <- adjusted
2026-08-07    90.36      <- unadjusted
2026-08-11    45.53      <- split date
```

`auto_adjust=True` and `auto_adjust=False` return **byte-identical** values, so
no adjustment was ever applied. From `runs/83c9e2e2dd48/00_raw_fetch.parquet`
(today's live run) the pipeline read `price_1m_ago = 93.49` (an unadjusted July
close) and `price_12m_ago = 62.30` (an adjusted 2025 close), giving

    return_12_1 = (93.49 - 62.30) / 62.30 = +0.5006     -> 97th percentile

against a true split-adjusted value of

    return_12_1 = (46.74 - 62.30) / 62.30 = -0.2497     -> 3rd percentile

MNST was published at momentum 71.5 and rank 360 on that basis. The error is
worth roughly **110 composite ranks**, and it was live on the public site.

**Evidence - calibration, measured 2026-08-26.** The check is exact rather than
heuristic: it uses the split ratio Yahoo itself reports, and asks whether any
day's close-to-close price ratio sits near `1/k` or `k`. Two numbers set it:

- **Arming floor, 25%.** Over **137,313 ticker-days** (503 S&P 500 names, 13
  months) p99.9 of |daily return| is **17.2%** and only **21 days in the whole
  sample** exceed 30%. A ratio implying a jump smaller than 25% cannot be told
  apart from ordinary trading, so it is left alone. This is what stops the
  small spin-off "ratios" Yahoo also reports as splits (SPGI 1.057, HON 1.061,
  CMCSA 1.067, FDX 1.241, BDX 1.272) from flagging every routine down day.
- **False-positive rate, zero.** Run against **all 17 real S&P 500 split events
  of the previous 13 months**: 11 were large enough to arm the check, and it
  fired on exactly one - MNST - passing AMCR, BDX, BKNG, CVNA, CMCSA, CRWD, DD
  (twice), FDX, HON (twice), KLAC, NFLX, SPGI, NOW and TPL, plus volatile
  controls including MRNA's genuine +177% single-day move.

**Why withhold rather than repair.** MNST's series flips scale on **seven**
separate days (2026-07-20, 07-23, 07-31, 08-03, 08-06, 08-07, 08-11), so there
is no single factor that puts it right. Withholding routes the problem into
machinery that already exists and is already trusted: `na_option="keep"` plus
the `has_data` mask in `compute_category_scores` renormalises the surviving
weights, so a missing category is neutral - the stock neither gains nor loses
from it.

**The synthesis finding - what this says about the screener as a whole.** The
eight categories are not eight independent bets. **Momentum and risk together
are 23% of composite weight (13 + 10), and every metric in both is derived from
one `Ticker.history()` call per stock.** Nothing checked that call's output for
internal consistency, so a single upstream defect could - and did - corrupt
almost a quarter of the composite for a name while every existing guard passed
it: `check_run_health` saw 100% price coverage and normal dispersion, and
winsorization *hid* the severity rather than catching it (MNST's raw
`volatility_1y` was **1.77**, capped to 0.845, which merely made Monster
Beverage look as volatile as SMCI).

A second, smaller coherence finding, now pinned by a test: momentum's only
non-price metric, `proximity_52w_high`, carries
`metric_weights.momentum.proximity_52w_high: 0` as a Phase 11 candidate. So on
paper a rejected series costs momentum 3/4 of its inputs; in practice the
renormalised weight sum is zero and the category goes NaN. **A rejected price
series costs a stock two entire categories, not one and a fraction.**

**Blast-radius guard.** Because withholding is now possible, a Yahoo-side change
that rejected the universe would publish a screener with 23% of the composite
blank, and dispersion could not catch it (with most stocks NaN it is computed
over whatever survives). `check_run_health.py` gains
`MIN_CATEGORY_COVERAGE = 0.90`: a run fails if under 90% of stocks have a
momentum or a risk score. One rejected name in 502 passes; fifty do not.

**Correction to the record.** `NIGHTLY_LOG.md` 2026-08-25, `history.py`'s
`round_trip_tickers` docstring and priority 1.5 in `CLAUDE.md` all recorded
this defect the other way round - that MNST's 2.9 percentile reading on 08-21
and 08-24 was the artifact and 97.1 was correct. **It is the reverse:** 08-21
and 08-24 were the two runs that got MNST right. The round-trip detector
shipped on 08-25 was nonetheless correct to flag it, and correct about why - a
round trip in the ranking is evidence of a data artifact somewhere, whichever
end of it is wrong. `history.py` and `CLAUDE.md` are corrected in this commit.

**Expected effect.** One stock of 502 (MNST) loses its momentum and risk scores
until Yahoo's series is repaired, and drops out of the model portfolio's
liquidity filter. Its `Composite_Confidence` falls - measured on a synthetic
universe, 80.0 -> 61.8 for the same stock with and without the eight metrics -
so the loss is visible to a dashboard user without any new UI. Its composite
moves by a couple of points, not tens, because renormalisation is neutral by
construction. No other stock is affected. Expect roughly **one name a year**:
17 split events per year in this universe, of which this is the first observed
failure.

**Validated by:** `tests/test_price_series_integrity.py` (21 tests) and six new
tests in `tests/test_run_health.py`. Suite **647 -> 668**, no pre-existing
failures and none introduced. End-to-end against the live feed: MNST's eight
metrics come back NaN with `proximity_52w_high` (0.971), `ev_ebitda` (32.24)
and `roic` (0.248) intact, while KO as a control is unchanged.

**Backtest observation:** none - benched until 2027-02-11 per rule 5. Nothing
above is a forward return or an IC, so rule 4 does not apply either; the
evidence is a demonstrated failure plus a distributional measurement over
stored and live price data. MNST's forward returns in
`improvement/performance_history.csv` were checked and are **not** polluted
(max 15.9%), so the evidence base needed no repair.

**FCX was not the same defect - and was not a defect.** The 08-25 entry cited
FCX's growth score (68.3 -> 42.5 -> 68.3) alongside MNST as the same bug. It is
not. On 2026-08-24 FCX's `forward_eps_growth` and `peg_ratio` were genuinely
**NaN** - the fetch did not return them - and `compute_category_scores`
correctly renormalised growth over the remaining three metrics, giving 42.5.
That is the missing-data path working exactly as designed, and the honest
number for that day. So priority 1.5 cited two cases: one real scoring bug,
fixed here, and one instance of correct behaviour.

What FCX does show is a **presentation** gap rather than a scoring one: a stock
whose category score moves 26 points because two of five inputs went missing
appears in the movers panel indistinguishably from one that moved on new
information. `Composite_Confidence` already falls, so the information is
present but not adjacent to the move. That is a product question for a Tuesday,
not a defect, and it is recorded here so the next session does not go looking
for a bug that is not there.

**Known limit, not fixed.** The check only speaks about splits Yahoo declares.
A corrupted series whose split record is missing entirely would pass. The
obvious generic detector - "more than one +-30% day in 13 months", which in
this sample separates MNST (7 days) from every other name (at most 1) - was
**not** shipped as a gate, because 13 months cannot rule out a genuine crash
producing repeated 30% days. It is recorded here so a future session can test
it against a wider window rather than rediscover it.

**Rollback:** `good/2026-08-25-0625`. Removing `check_price_series_integrity`'s
call site in `factor_engine.py` restores the previous behaviour exactly.

**Applied by:** morning session (manual)
