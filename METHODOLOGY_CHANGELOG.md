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
with the daily close** *(19 of 45 as of 2026-09-10 - `fy1_revision_3m` scales
by price too; the claim is unchanged in substance and slightly stronger)*,
spanning five of the eight categories:

| Category | Price-driven metrics |
|---|---|
| Valuation | `ev_ebitda`, `fcf_yield`, `earnings_yield`, `ev_sales`, `pb_ratio`, `peg_ratio`, `dividend_yield` |
| Momentum | `return_12_1`, `return_6m`, `proximity_52w_high` |
| Risk | `volatility`, `beta`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown_1y`, `jensens_alpha` |
| Revisions | `price_target_upside`, `fy1_revision_3m` *(added 2026-09-10)* |
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

---

## 2026-08-26 (evening) - The model portfolio leaves the dashboard; stocks gain a plain-English "about"

**Area:** dashboard surfaces / payload composition. **No scoring change.** No
weight, threshold, metric definition, or trap rule moved. Composite scores and
ranks are byte-identical before and after; this entry exists because the
Methodology text and the published payload both changed, and because a future
session must be able to find out why a surface disappeared.

**Applied by:** owner-directed session (interactive), owner request 2026-08-26.

### Changed - 1. The Model Portfolio surface is gone

Removed from the dashboard: the `Model Portfolio` section, the `Portfolio
Sector Allocation vs S&P 500` chart, `renderPortfolio()`,
`renderSectorAlloc()`, the `portfolio` payload key, and the `spx_weights` key.
Methodology text and AI-chat context updated to speak of the *ranking* rather
than a portfolio. `How Stable Is the Portfolio?` is now `How Stable Is the
Ranking?` - which is what that chart always measured (top-20 by composite via
`run_weight_sensitivity(..., top_n=20)`, never the constructed portfolio).

**Kept deliberately:** `portfolio_constructor.py`, the `08_model_portfolio`
artifact, the `ModelPortfolio` Excel sheet, and the `in_portfolio` snapshot
column. `plan/dashboard-inventory.md` warned that
`improvement_engine.record_run_snapshot()` computes **turnover** from
`in_portfolio`, so deleting construction outright would have quietly damaged
the evidence base. That warning was checked and found correct.

**Evidence.** Three findings, each measured rather than asserted:

1. **The panel carried no information `table_data` did not already hold.**
   Every field in a holding (`ticker`, `company`, `sector`, `composite`, the
   eight category scores, `vt`, `gt`) exists in `table_data` under a different
   case. The set difference is empty. It was a renamed, row-filtered copy.

2. **It did not answer "how much".** `plan/dashboard-north-star.md` names four
   questions, the fourth being position sizing, so removing the only
   portfolio-shaped surface looked like it might cost the tool an answer. It
   does not: the holdings payload **carries no position weights at all**. The
   sizing logic exists in `portfolio_constructor.py` and the Excel sheet, and
   was never exposed to the dashboard. Question 4 was already unanswered there.

3. **The owner's stated reason - that it wasted space - is not true, and the
   real reason is better.** Measured: `portfolio` was 9,681 bytes of a
   3,373,395-byte payload, **0.29%**. Removing it saves nothing. It was removed
   because a fixed 25-name sector-capped list published on a public site is the
   closest this tool came to emitting a recommendation, which contradicts the
   governing line in `CLAUDE.md`: *decision support, not a recommendation
   engine*. A ranking a reader sorts and filters is a screen. A named portfolio
   is advice.

**Top 5 was verified, not assumed.** It had been reading
`D.portfolio.holdings.slice(0, 5)`. It now filters `table_data` for trap-free
names and sorts by rank. On the live 2026-08-26 data both paths give
`HST, EXPE, APA, EIX, CF` - the sector cap is 8-of-25 and cannot bind on five
rows. The trap exclusion was preserved; dropping it would have promoted a
flagged name into the headline five.

### Changed - 2. Business descriptions in the stock drilldown

`factor_engine._fetch_single_ticker_inner()` now captures
`longBusinessSummary`, and the drilldown renders it under the score cards as an
`About` block with the company's specific industry, a 4-line clamp with a
"Show more" toggle, and an attribution line.

**Evidence.** The dashboard could score a company across 44 metrics but could
not tell a reader what it sold. For the investment-club audience that is a
teachability gap, not a polish item: a student looking at `APA` at rank 3 has
no way to learn it is oil and gas exploration without leaving the tool.

**Cost, measured rather than estimated:**

- **API: zero.** The field rides the `.info` dict `_fetch_single_ticker_inner`
  already pulls. A second endpoint would have multiplied per-ticker requests on
  a loop already losing 10-25% of tickers to Yahoo rate limits - that would
  have traded evidence accrual for prose and was not acceptable.
- **Payload: +0.71 MB raw (~+21%), ~+60 KB gzipped (~+8%).** Sampled across 8
  tickers the summaries average 1,421 characters; prose gzips ~11.6x against
  the payload's overall 4.2x. Pages serves gzip, so raw size overstates the
  cost by an order of magnitude here.

**It is display-only and must stay that way.** Never scored, ranked, or fed to
a metric. `test_description_is_not_scored` asserts it never appears in `raw` or
`pct`. It is provider text rendered verbatim with its source named, for the
same reason every other number on the page shows its provenance.

**Expected effect:** none on ranking. On the product: the drilldown answers
"what is this company?" without leaving the page. The `about` field is empty
for every stock until the next data run fetches it - the field did not exist in
the raw parquet before tonight, so it populates at the 02:00 run on 2026-08-27.
`industry` populated immediately (501/502) because it was already being
fetched and merely unused.

**Validated by:** `tests/test_dashboard_surfaces.py`, 30 tests. **29 of the 30
fail against the pre-change code** (run in a detached worktree at `9bed64f` to
confirm); the one that passes both ways is the guard asserting the
defensibility section survived. Full suite 706 passed, up from 676, no
regressions. Rendered and driven in a browser: no console errors, portfolio
section absent from the DOM, Top 5 renders five cards, and the About block
verified across all three data shapes (long text - block and toggle shown;
short text - block shown, toggle hidden; missing - block hidden).

**One real bug was found by rendering it rather than reading it.** The first
cut measured `scrollHeight > clientHeight` inside `renderAbout()`, which runs
while the modal is still `display:none`. Both heights read 0, so "Show more"
was hidden on every stock and long descriptions were permanently truncated with
no way to expand. Fixed by deferring the measurement to
`requestAnimationFrame`; `test_about_overflow_is_measured_after_layout` pins
it. A static read of that code looks correct, which is the point.

**Rollback:** `good/2026-08-26`. The two changes are independent and can be
reverted separately: the About block is confined to `renderAbout`/`toggleAbout`
plus the `_about`/`_industry` merge, and the portfolio removal is confined to
`generate_dashboard.py` - no pipeline or scoring code was touched by either.

## 2026-08-28 - The dashboard showed weights the scores were never multiplied by

**Area:** methodology reporting - `effective_weights.json`, the stock drilldown,
`SCREENER_OVERVIEW.md`. **No scoring change.** No weight, threshold, metric or
formula moved; every stock ranks exactly as it did before this entry. What
changed is what the tool *says* it did.

**Changed:**

1. `run_factor_engine()` now hands the regime-adjusted factor weights back to
   `main()`, and `RunContext.save_effective_weights()` records them - plus
   `base_factor_weights` and a `factor_weights_adjusted` flag.
2. `generate_dashboard.prepare_dashboard_data()` reconciles the recorded
   weights against the published contributions before publishing, and refuses
   to publish weights that do not reproduce them.
3. The drilldown displays **per-stock** effective weights - after both the
   run-level regime adjustment and the per-stock renormalisation - and
   explains, in prose, any gap against the Methodology page.
4. `SCREENER_OVERVIEW.md` now states that its printed weights are configured
   defaults, names the two rules that move them, and says where to see what a
   run actually used.

**Evidence:** a demonstrated, live, user-facing failure - not a citation, and
not this system's own IC series. Rules 4 and 5 do not bite: nothing here was
justified by a return number or a backtest.

The drilldown prints its arithmetic to the reader: `Score: 65.3/100 x 13% =
9.76 pts`. Measured against `dashboard_data.js` as served from `main` on the
morning of 2026-08-28, that equation was false. 65.3 x 13% is 8.49.

Solving `contrib / score` across the 491 stocks with all eight categories
populated recovers the weights the composite was really built from:

| Category | Published | Actually used |
|---|---|---|
| Valuation | 22 | **20.05** |
| Momentum | 13 | **14.95** |
| the other six | unchanged | unchanged |

Those are exactly a LOW VOL regime: `13 x 1.15 = 14.95`, the 1.95pp funded out
of Valuation, per `adjust_momentum_weight()`. The implied weights sum to
100.000.

**Root cause.** `adjust_momentum_weight()` returns a deep copy of the config.
`run_factor_engine` does `cfg = adjust_momentum_weight(...)`, rebinding a
*local* name, so the adjustment never reached `main()` - and
`ctx.save_effective_weights(cfg)` is called from `main()`. The
revisions/investment auto-disables assign into the shared dict
(`cfg["factor_weights"] = ...`) and therefore did propagate, which is precisely
why only momentum and valuation were wrong while the other six were right. The
file has been named `effective_weights.json`, with the docstring "the effective
weights", the whole time.

**Blast radius, counted rather than estimated:** of 4,016 (stock, category)
cells in the live payload, **1,051 displayed arithmetic that did not hold** -
momentum wrong for 498 of 502 stocks, valuation for 501 of 502, plus 52 cells
across 11 stocks from the second cause below. After the fix: **0 of 4,002.**

**The second cause, found alongside it.** When a category cannot be scored for
a stock, `compute_factor_contributions` drops it and renormalises the
survivors. The page showed the universe weight regardless. MNST - whose price
series the 2026-08-26 split-integrity check rejects, removing Momentum and Risk
- displayed "22% weight -> 20.64 pts" against a quality score of 70.43. The
drilldown now shows MNST's own ~28.4% and says why.

**Expected effect:** no ranking movement of any kind. The published
`weights.factor_weights` changes from the configured defaults to the run's real
weights, so the figure a reader sees next to Momentum moves 13% -> 15.0% in a
low-vol regime. The AI chat's system prompt reads the same key and stops
telling users the wrong number.

**Why this matters more than a display bug.** The tool's claim is not that its
numbers are good; it is that they are *checkable*. "Decision support, not a
recommendation engine - show why, with sources and uncertainty visible."
A student who checked the one worked example on the page found it did not add
up, and had no way to tell whether the weight or the score was wrong. That is
the credibility product failing in the exact place it is most on display.

**Validated by:** `tests/test_weight_transparency.py`, 34 tests. The
reconciliation guard reproduces the live case: given contributions built at
14.95% and weights recorded as 13%, it recovers 14.95 and 20.05 to within
0.02pp and flags the run. `TestPublishedPayloadAddsUp` asserts `score x
published weight = published contribution` for every stock and category,
including a stock with two categories withheld. End-to-end against the real
2026-08-28 run: the guard fired, printed both corrections, and the republished
payload reconciles on all 4,002 cells.

**A note on the fallback.** The reconciliation *derives* weights from the
scored rows when the recorded ones disagree. That is a repair path, not the
design - the fix is at source, in the pipeline. It exists so that old run
directories, every one of which records 22/13, republish truthfully, and so
that a future divergence is loud rather than silent. It declines to guess on a
universe under 20 rows and leaves the recorded weights alone.

**Applied by:** morning session (manual).

**Rollback:** `good/2026-08-27`. Self-contained: `run_screener.py` (the
handback plus the overview wording), `run_context.py` (the recorder), and
`generate_dashboard.py` (the guard plus the display). No scoring code was
touched, so a revert changes no rank.

## 2026-09-01 - The screener clipped values it then ranked, and published the clipped number

**Area:** scoring pipeline (pre-rank data treatment); public metric display.

**Changed:** `winsorize_metrics()` is removed. It ran immediately before
`compute_sector_percentiles()` in **four** places - `run_screener.py:1339`,
`factor_engine.py:3429` (the module's own end-to-end path), `backtest.py:309`
and `run_audit.py:221` - and clipped the top and bottom 1% of every one of the
44 metrics onto a single boundary value. It is replaced by
`factor_engine.flag_metric_outliers()`, which reports the same tails into the
data-quality log and **does not modify the frame**. Config key
`data_quality.winsorize_percentiles` is renamed `outlier_report_percentiles`
(the old name is still read as a fallback, and still accepted by `schemas.py`,
so an older `config.yaml` keeps working).

**Evidence.** Three independent lines, none of them a backtest or an IC.

1. **It could not have helped - this is a proof, not an estimate.**
   `compute_sector_percentiles()` is `Series.rank(pct=True)`. A rank transform
   is invariant under *any* monotone transform of its input. Winsorizing is
   monotone, so it cannot change a single ordering. It is not monotone
   *injective* - it maps the whole tail to one number - and that is the only
   effect it can have: distinct values become ties, which `rank` then resolves
   to a shared average rank. Locked down by
   `tests/test_no_winsorization.py::TestRankIsInvariantToMonotoneTransforms`,
   which shows an exponential rescale of the inputs leaves every percentile
   identical, while clipping four tail values collapses four distinct ranks
   onto one and leaves the rest of the distribution untouched.

2. **A demonstrated, user-facing failure on the live public site.** Measured on
   the published `dashboard_data.js` from the 2026-09-01 02:00 run: **301
   (stock, metric) cells across 33 continuous metrics, touching 159 of 502
   stocks**, carried a clipped value rather than the fetched one - and `raw` is
   what the drilldown shows the reader. The flagship case, with true figures
   fetched from the same source the screener uses:

   | Ticker | Published | True | Understated by |
   |---|---|---|---|
   | NVDA | $2,802.0B | $5,331.2B | $2,529B (47%) |
   | AAPL | $2,802.0B | $4,624.2B | $1,822B (39%) |
   | GOOGL | $2,802.0B | $4,150.2B | $1,348B |
   | GOOG | $2,802.0B | $4,102.0B | $1,300B |
   | MSFT | $2,802.0B | $3,766.9B | $965B |
   | AMZN | $2,802.0B | $2,802.0B | - (the survivor clipped onto) |

   Six of the largest companies in the world were published with one identical
   market capitalisation. For a tool whose credibility is the product, and
   whose second audience is students, that is not a rounding issue.

3. **It hid data errors - already documented in this file.** Changelog
   2026-08-26: MNST's corrupt `volatility_1y` of **1.77** was clipped to
   **0.845**, "which merely made Monster Beverage look as volatile as SMCI".
   An implausible number is the signal that a feed has broken; clipping deleted
   exactly that signal. Reporting the tails preserves it, which is why the
   replacement logs rather than discards.

**Scoring effect, measured, and deliberately not overstated.** Because ranking
is invariant, no *ordering* changes; what changes is the resolution of the ties
winsorization manufactured. On the same published run, **58 (metric, sector)
tie groups collapsed two or more stocks onto a single percentile**. The largest:
4 Energy stocks shared one `beta` rank in a 21-stock sector, spanning **14.3
percentile points**; 5 Utilities shared one `volatility` rank (12.9 pp); 6
Information Technology names shared one `return_6m` rank and one `beta` rank
(6.8 pp each). Those stocks now receive their own ranks. Composite movement is
correspondingly small - a single metric inside a 5-22% category weight - so the
top of the ranking is not expected to reorder much. The defect being fixed is
mostly one of *published truthfulness*, and secondarily of rank fidelity.

**What this does NOT change, so it is not assumed away later:** `metric_clamps`
stays exactly as it is. That is a different mechanism with a different argument
- a domain judgement that a forward EPS growth above 150% is not a credible
input - and on this run it is not even binding (observed maxima 0.978 against a
1.50 bound, 0.583 against 1.00). Whether a non-credible value should be clamped
or withheld as NaN is a real question and is left open, not answered here.

**Expected effect:** every published `raw` value equals the value fetched.
Ranks are unchanged except inside the 58 collapsed tie groups, whose members
separate. No category weight, metric weight or threshold moved.

**Validated by:** `tests/test_no_winsorization.py`, 18 tests, including the
exact regression - the six largest US companies must keep six distinct market
caps and six distinct size percentiles. Two of those tests are structural
guards: an AST walk asserting no module in the scoring path *calls* a
winsorizing function, and a second asserting no non-docstring string in those
modules still *tells the reader* it winsorizes. Both fail against the
pre-change code (5 offending calls; 32 offending user-facing strings), and the
second is what caught 19 stale per-metric "Winsorize 1/99 pctile" descriptions
in `run_audit.py` and the fourth call site in `factor_engine.py` that the first
pass of this change had missed. Full suite **872 passed**, up from a session
baseline of 853, no failures.

**Docs corrected in the same commit,** because they asserted the false
rationale: `SCREENER_OVERVIEW.md` Step 2 (regenerated from `run_screener.py`,
which is its source) previously read "Extreme outliers can distort rankings" -
they cannot, for a rank-based screen; `Multi-Factor-Screener-Blueprint.md`
(volatility/beta rows and the "Winsorization Applied" note); `README.md`;
`config.yaml`. The dated audit reports (`FORENSIC_AUDIT_REPORT.md`,
`INSTITUTIONAL_AUDIT_REPORT*.md`, `HARDENING_REPORT.md`,
`HEDGE_FUND_REVIEW_FINDINGS.md`, `IMPLEMENTATION_PLAN.md`) were left alone on
purpose: they are records of what was true when written.

**One consequence to expect.** `cache/factor_scores_*.parquet` stores the fully
*scored* frame, so every cached file written before today still holds clipped
values, and a warm cache hit returns them without re-scoring. Two independent
mechanisms guarantee the next data run does not reuse one, so the fix reaches
the live site on 2026-09-02:

- The cache key includes a hash of `data_quality`, and renaming the config key
  moved it from `19c853468405` to `2bde439e06ad`. `_find_latest_cache()` filters
  on that hash, so the pre-fix files are unreachable - a cold start regardless
  of age. This is the binding one.
- Independently, `factor_scores` is bounded by the price tier
  (`price_data_refresh_days: 1`) and `cache_is_usable()` is exclusive
  (`age < max_age`), so only a cache written *today* is reusable anyway.

If the published megacap market caps are still identical after the 2026-09-02
run, both of those failed and that is the thing to investigate.

**Applied by:** morning session (manual).

**Rollback:** `good/2026-08-31-0617`. The change is confined to
`factor_engine.py`, `run_screener.py`, `backtest.py`, `run_audit.py`,
`config.yaml`, `schemas.py`, the docs above, and the test suite.

---

## 2026-09-02 - The risk category was 30% momentum wearing a risk label

**Area:** metric weights (`risk` category)

**Changed:** `config.yaml -> metric_weights.risk` and the matching defaults in
`schemas.py`:

| Metric | Was | Now | What it measures |
|---|---|---|---|
| `volatility` | 30 | **42.86** | dispersion (total risk) |
| `beta` | 20 | **28.57** | dispersion (systematic risk) |
| `max_drawdown_1y` | 20 | **28.57** | dispersion (tail risk) |
| `sharpe_ratio` | 15 | **0** | (return - rf) / volatility |
| `sortino_ratio` | 15 | **0** | (return - rf) / downside deviation |

The three survivors are the old 30/20/20 renormalised over 70, so their
relative emphasis is **unchanged**. Rebalancing among the dispersion metrics
would be a second claim this change does not make and did not research.

Sharpe and Sortino are **not deleted**. They remain in `METRIC_COLS` and in
`CAT_METRICS["risk"]`, are still computed, and still appear on each stock's
detail page - weight-0 candidates, the same treatment `proximity_52w_high`
and `peg_ratio` already get. They are informative; they are not risk.

**Evidence:**

*1. Measured on this screener's own published output.* Both ratios are built
in `factor_engine.py` from the same numerator - `sharpe_ratio` at :1923 is
`(return_12m - rf) / volatility`, `sortino_ratio` at :1946 is
`(return_12m - rf) / downside_deviation`. Across the S&P 500 the
cross-sectional spread in trailing returns is far wider than the spread in
dispersion, so that shared numerator dominates both. Spearman correlations on
the metric percentiles published on `main` (N = 498-499):

| Pair | 2026-08-31 | 2026-09-01 | 2026-09-02 |
|---|---|---|---|
| `sharpe_ratio` ~ `sortino_ratio` | +0.993 | +0.994 | +0.993 |
| `sharpe_ratio` ~ `return_12_1` | +0.940 | +0.940 | +0.944 |
| `sortino_ratio` ~ `return_12_1` | +0.936 | +0.933 | +0.940 |
| **`sharpe_ratio` ~ `volatility`** | **+0.029** | **+0.032** | **+0.025** |

Three consecutive runs, essentially identical, and mechanically necessary
rather than incidental. The category scored five metrics that were three
distinct things: two of the five were each other (+0.993), and both were the
momentum signal (+0.94) rather than a risk measure (+0.03).

At category level this made **momentum ~ risk = +0.516**, the largest of the
28 pairs in the 8x8 category-score matrix - larger than valuation~growth
(-0.349) or size~valuation (+0.337). Recomputing the risk score on dispersion
alone takes it to **+0.150**.

*2. A demonstrable user-facing consequence.* On the 2026-09-02 run **SNDK
published a risk score of 31.1 alongside a momentum score of 94.2**. Scored on
dispersion alone its risk score is **1.6**. The same pattern held for MRNA
(34.7 -> 6.8), VRT (29.1 -> 3.2), FIX (37.5 -> 12.0), MU (41.6 -> 17.2) and
WDC (39.9 -> 15.7) - every one a high-momentum name. The public site was
telling a student that a violently volatile stock was mid-pack on risk,
*because it had gone up*. For a tool whose stated purpose is to be teachable,
a category that does not mean what its name says is the defect.

*3. The documentation asserted the opposite, and was wrong.*
`SCREENER_OVERVIEW.md` justified the design with "Five metrics give a more
complete risk picture than two." A +0.993 correlation between two of the five
refutes that directly. The generator text in `run_screener.py` has been
rewritten rather than softened.

*4. Documented professional practice.* Institutional risk models measure risk
with dispersion, never with return/risk ratios. The **Barra US Equity Model
(USE4)** builds its Residual Volatility style factor from daily standard
deviation, cumulative range and residual sigma; Beta is its own descriptor.
**MSCI Minimum Volatility** indexes optimise against those Barra BETA and
RESVOL exposures, leaving them unconstrained while constraining every other
style factor to +/-0.25 sd. No index provider selects for low risk with a
Sharpe ratio. (The specific USE4 descriptor weights - 0.74 DASTD + 0.16 CMRA
+ 0.10 HSIGMA - come from a **secondary** source; the primary MSCI PDF was not
text-extractable in this session, so treat those decimals as indicative. The
substantive point, that the descriptors are all dispersion measures, is not in
doubt.)

*5. Published literature.* The cross-sectional risk effects are documented on
dispersion measures: **Ang, Hodrick, Xing and Zhang (2006), "The Cross-Section
of Volatility and Expected Returns," *Journal of Finance* 61(1), 259-299** -
idiosyncratic volatility, quintile 1-minus-5 spread over **1%/month**, robust
at **-0.63%/month, t = -3.30** excluding the smallest growth firms; and
**Frazzini and Pedersen (2014), "Betting Against Beta," *Journal of Financial
Economics* 111(1), 1-25** - selection on **beta**, BAB factor Sharpe **0.78**
(1926 - March 2012). Note what Frazzini and Pedersen do with the Sharpe ratio:
they use it to *evaluate the resulting portfolio*, not to rank the
cross-section. That is the correct use of the statistic, and it is the use this
screener was not making of it.

**Expected effect:** momentum ~ risk category correlation +0.516 -> +0.150.
Composite ranking Spearman **0.990** against the old ranking; median absolute
rank change **10 places**, p90 **35**, max **83**; **3 of the top 50 change**
(out: DELL, FOX, STLD; in: ADBE, CB, EXE). High-momentum, high-volatility
names fall in the risk category and therefore slightly in composite; genuinely
low-dispersion names rise. Nominal category weights are untouched - but the
*realised* exposure moves, because roughly 3% of composite that was labelled
risk was behaving as momentum. Momentum's true weight falls back toward its
stated 13-15% and risk's rises toward its stated 10%.

**What deliberately did NOT change:** the eight category weights. The
measurement says the risk category was mismeasuring risk, not that risk
deserves more or less of the composite. Re-deciding the category weights is a
separate question needing its own research, and doing both at once would make
neither attributable.

**Validated by:** `tests/test_risk_category_independence.py`, 13 tests. Three
of them (`test_return_over_risk_ratios_carry_no_scoring_weight`,
`test_only_dispersion_metrics_are_scored`,
`test_overview_no_longer_claims_five_risk_metrics`) were confirmed to **fail
against the pre-change config** before being trusted. Four more reproduce the
mechanism deterministically on a synthetic cross-section built so volatility
is independent of return by construction, so the finding does not depend on
one day's live data. Full suite 882 -> 895 passing, same single pre-existing
failure (`test_parquet_roundtrip`).

`tests/fixtures/golden_scores.parquet` was regenerated. Before regenerating,
the golden diff was inspected to confirm it was confined to `risk_score` and
downstream: all 50 preceding columns (Ticker, Sector and every one of the 44
metrics) compared **equal**. No raw metric value moved.

**No backtest number and no figure from `live_ic_history.csv` appears in this
entry** (rules 4 and 5). The `1m` optimization horizon currently holds **3
effective observations** against a gate of 8; it will be a long time before it
can speak to this, and the argument does not need it.

**Applied by:** morning session (manual), synthesis day.

**Rollback:** `good/2026-09-01-evening`. The change is confined to
`config.yaml`, `schemas.py`, the two documentation blocks in
`run_screener.py`, `tests/fixtures/golden_scores.parquet`, and the new test
module.

### Confirmed 2026-09-09 — the predicted effect landed, and overshot in the right direction

Seven days of live running later, measured on the 2026-09-09 02:00 run
(N = 501, `improvement/snapshots/2026-09-09_8d92af6b7208.parquet`):

| Quantity | Before (2026-09-02) | Predicted | **Measured 2026-09-09** |
|---|---|---|---|
| `momentum_score` ~ `risk_score` | +0.516 | +0.150 | **+0.084** |
| Its rank among the 28 category pairs | **1st (largest)** | — | **20th** |

The pair went from the largest correlation in the 8x8 category matrix to the
20th of 28. The mechanism is confirmed too, not just the outcome: on the same
run `sharpe_ratio` and `sortino_ratio` correlate **+0.925 / +0.917 with
momentum** and **+0.119 / +0.120 with risk** — which is precisely the
diagnosis ("two of the five were the momentum signal rather than a risk
measure") that justified moving them to weight 0. They are still computed and
still shown on each stock's detail page.

Recorded per `CLAUDE.md`'s "validation is continuous" rule: when evidence
accumulates, go back and check the entry that made the change. This is a
structural correlation measured on published output, **not** a return or IC
measurement — rules 4 and 5 are untouched, and nothing here is offered as
evidence for a further change.

Measured during the 2026-09-09 synthesis session; full context in
`research/2026-09-07-revisions-category-has-no-revisions.md` §8.8.

---

## 2026-09-08 - The AI chat leaves the dashboard; every stock gains a deterministic "Why it ranks here"

**Area:** dashboard surfaces / payload composition. **No scoring change.** No
weight, threshold, metric definition, trap rule or scoring formula moved.
Composite scores and ranks are byte-identical before and after. This entry
exists because a published surface was removed and another added, and because a
future session must be able to find out why - the same reason the 2026-08-26
(evening) model-portfolio entry exists.

**Applied by:** morning session (manual), product day. Owner directive
2026-08-10, priority 4 in `CLAUDE.md`, specified in
`plan/dashboard-north-star.md` ("Replace the chatbot with generated summaries").
Open **29 days**.

### Changed - 1. The "Screener AI" chat is gone

Removed from the generator: the chat FAB and panel, the Chat Settings dialog
with its API-key field and model picker, 27 JavaScript functions
(`sendMessage`, `callClaude`, `buildSystemPrompt`, `parseChatMd`,
`initChatResize`, ...), the `chatSlideUp`/`chatFabPulse`/`chatDotBlink`
keyframes and the whole `AI CHAT PANEL` stylesheet block. **891 lines**: 80
HTML, 583 JS, 228 CSS. `generate_dashboard.py` is 921 lines shorter.

The `config_traps` payload key went with it. It carried the four trap
thresholds solely so the chat could put them in its system prompt; nothing
rendered them, and the same thresholds are already published in the Methodology
section, which `run_screener.generate_screener_overview()` templates from
`config.yaml`. Same reasoning that retired `spx_weights` on 2026-08-26: a
payload key whose only consumer has gone is dead weight that reads like a
feature.

**Evidence - a documented user-facing failure, not a preference.** The chat
required each visitor to paste their own Anthropic API key into `localStorage`
and then called `api.anthropic.com` **from the browser**. Four consequences,
all demonstrable from the shipped code rather than argued:

1. **It was unusable for the stated audience.** Every student in a college
   investment club would need their own paid API account. Most do not have one,
   so for most visitors the feature was a button that opened a form asking for a
   credential they cannot obtain.
2. **It taught a bad habit.** A public web page with a password field labelled
   "Anthropic API Key" is the exact shape of a credential-phishing form. That is
   a poor thing for a teaching tool to normalise.
3. **It cost the reader money per question**, on a tool whose premise is that it
   is free and reproducible.
4. **It was un-reproducible, and that is the one that decides it.** Two students
   asking the same question got different answers, and neither answer was
   recorded anywhere. `plan/dashboard-north-star.md` puts the standard plainly:
   *"decision support, not a recommendation engine... show why, with sources and
   uncertainty visible."* An explainer that cannot be audited works directly
   against the property the rest of this tool is built to have. The screener
   refuses to publish a run whose price coverage is below 90%, while shipping an
   explanation layer with no provenance at all.

### Changed - 2. "Why it ranks here" replaces it

`stock_summary.py` (new) builds an ordered list of factual sentences per stock
from fields the payload already carried - `contrib`, `cat_scores`, `pct`, `raw`,
`peers`, `flags`, the analyst targets, `metric_count`/`metric_total` and the
`history` spine. It renders as the first block of the stock drilldown. Worked
example, HST on the 2026-09-08 run:

> Ranks 1st of 502. Its composite of 74.7 is a percentile: it scores above 75%
> of the universe. Most of that composite comes from Valuation (category score
> 96, 21.1 points) and Quality (category score 83, 18.4 points) - 39.4 of its
> 74.7 points. Its weakest scored category is Risk at 38 out of 100,
> contributing 3.8 points. A category score near 50 is the sector median. Inside
> Valuation it sits in the 97th sector percentile on EV/EBITDA (8.99) and the
> 97th on Earnings Yield (6.7%). Its lowest-ranked weighted input is Beta
> (0.70), in the 17th sector percentile. Since the run of 2026-08-10 (29 days
> ago) it has held its rank, with the composite down 3.3. It last traded at
> $22.05 against an analyst price target of $25.14, 14.0% above the current
> price, the mean of 20 analyst estimates. Among the 5 closest Real Estate names
> by market cap plus itself, it ranks 1st of 6 on composite (best peer: REG at
> 57.4). Flagged as a channel-stuffing risk (receivables growing faster than
> revenue). The score rests on 18 of 18 metrics.

Every figure is arithmetic on the run's own output. It is **generated at build
time and baked into the payload**, not computed in the browser, so what a reader
diffs is what shipped and two readers cannot see different text.

**Three constraints are enforced by tests, not by care:**

- **It explains; it never advises.** `BANNED_TERMS` in `stock_summary.py` is the
  machine-checkable form of the north-star line ("good: *ranks 1st, driven by
  valuation and momentum*; bad: *attractive entry point*"). All 502 summaries on
  the live run contain **zero** matches, and the detector is itself tested
  against the plan's own bad examples, so a clean sweep means something.
- **Metric percentiles are labelled sector-relative**, because they are
  (`factor_engine.compute_sector_percentiles`). Calling them plain percentiles
  would publish a false claim about how the number was computed.
- **A fact that cannot be stated exactly is omitted, never approximated.** A
  stock with no history, no analyst coverage or a withheld category gets a
  shorter summary, not a hedged one. FDXF reads *"The score rests on 12 of 18
  metrics. Momentum, Risk and Investment could not be scored for this stock, so
  the remaining categories were reweighted to fill the gap. Its filings are
  flagged stale (282 days old)."* - which is north-star gap 5 (per-stock
  confidence made legible) arriving as a side effect.

**Scope: all 502 stocks, departing from the plan's "top ~25 first".** That
guidance was written before anyone measured the cost; the measurement is **+665
KB raw, +101 KB gzipped** (payload 1,078 -> 1,179 KB on the wire). Removing the
chat gives back 11 KB of page (`index.html` 281,678 -> 237,952 chars; 66 -> 55
KB gzipped), so the net is **+90 KB on the wire, +8%**. Against that, the
drilldown is the surface that answers *should I buy this one*, and it is
reachable for every name in the universe - a summary that appears only for names
a reader already knows is missing exactly where it helps most.
`plan/dashboard-north-star.md` now carries the measurement, so the next session
inherits the number rather than the guess.

**Expected effect:** none on any score, rank, weight or artifact other than
`dashboard.html` / `index.html` / `dashboard_data.js`. The drilldown gains a
lede; the page loses a credential form.

**Validated by:** full suite **965 -> 1,117 passing**, no failures before or
after. Two new modules: `tests/test_stock_summary.py` (77 tests) and
`tests/test_ai_chat_removed.py` (75 tests). **58 of those 75 fail against the
pre-change generator**, verified by swapping in `HEAD:generate_dashboard.py` and
re-running; the working copy was restored and checked byte-identical by SHA-256
afterwards.

The removal tests assert **both halves**. A partial swap is the dangerous state:
an `onclick="toggleChat()"` surviving its deleted function throws at click time,
and a dangling identifier in the script body blanks the whole page with all four
ship gates green - the failure `tests/test_dashboard_js.py` and the 2026-08-26
portfolio removal were both written about. So the module checks that no chat
symbol, element id, model id, `localStorage` key or provider URL survives, **and**
that `renderSummary` exists, is wired into `openStockDetail`, escapes its text,
and that every stock carries a non-empty summary. The emitted script is
additionally parsed with `node --check`.

**A parser-free brace-balance backstop was written for that last check and
removed the same session.** JavaScript regex literals make it unsound:
`escapeHtml` contains `.replace(/'/g, '&#39;')`, and a scanner without
regex-literal support reads that apostrophe as a string delimiter and
desynchronises. It reported the page unbalanced while `node --check` passed. A
check that fires on healthy code is the failure shape the 2026-09-01
bank-metrics fix was about - it trains a reader to ignore it, which is exactly
when the real defect gets through. Where `node` is absent the syntax test skips
visibly rather than pretending to cover.

**No backtest number and no figure from `live_ic_history.csv` appears in this
entry** (rules 4 and 5), and neither would be relevant: nothing here touches
scoring.

**Rollback:** `good/2026-09-07`. The change is confined to
`generate_dashboard.py`, the new `stock_summary.py`, the two new test modules,
the regenerated artifacts, and documentation.

---

## 2026-09-10 - The revisions category gets an actual revision, and stops being 78% earnings surprise

**Area:** metric definition (new metric) + `metric_weights.revisions`

**Changed:** two halves of one change. Splitting them would leave the
category's two heaviest metrics correlating at +0.401, so they ship together.

*(a) New metric `fy1_revision_3m`* - the 90-day change in FY1 consensus EPS,
scaled by price:

```
fy1_revision_3m = (eps_trend['0y','current'] - eps_trend['0y','90daysAgo']) / price
```

Source is `Ticker.eps_trend`, +1 HTTP request per ticker. Direction: higher is
better. Coverage measured on the full universe: **500/502 = 99.6%**, which is
*better* than the `analyst_surprise` it takes weight from (99.2%).

*(b) Within-category reweight.* The category's **10% share of the composite is
unchanged** - only the split inside it moved:

| Metric | Was | Now |
|---|---|---|
| `fy1_revision_3m` | - | **35** |
| `analyst_surprise` | 38 | **15** |
| `consecutive_beat_streak` | 20 | **10** |
| `earnings_acceleration` | 20 | 20 |
| `price_target_upside` | 12 | **10** |
| `short_interest_ratio` | 10 | 10 |

**The defect this fixes.** The category was named "revisions" and contained
none. 78 of its 100 points sat on the earnings-**surprise** family
(`analyst_surprise` + `consecutive_beat_streak`). The `config.yaml` comment
asserting that a revision metric was unavailable without FactSet or Refinitiv
I/B/E/S was **false** - it is one field on the data source already in use.

**Evidence:**

- **Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies", J. Finance
  51(5).** Of the three earnings-momentum legs, the analyst-revision measure
  (REV6) was the strongest: **+7.7% six-month decile spread**, IBES universe
  1977-93. Replicated by **Stickel (1991)** at **+7.07%**. Price scaling by the
  prior close is CJL's own construction, not a choice invented here.
- **Martineau (2022), "Rest in Peace Post-Earnings Announcement Drift",
  Critical Finance Review 11(4).** PEAD - the effect the *surprise* metrics
  rely on - is documented as **non-existent since 2006** for all but microcaps,
  with a **significantly negative** 2016-19 coefficient. This is an S&P 500
  screener, i.e. exactly the population where the effect is gone. That is the
  case for cutting surprise from 38 to 15, and it is a claim about the
  incumbent metrics rather than about the new one.
- **Novy-Marx (2015), "Fundamentally, Momentum is Fundamental Momentum",
  NBER w20984.** Earnings-momentum alpha is strongest in large caps
  (SUE **t = 2.83** in the top quintile, where price momentum is insignificant
  at t = 1.48). This is why the measured overlap with price momentum below is
  read as an economic fact rather than a construction error.
- **Documented practice.** Barra's USFAST `Sentiment` factor is built from
  analyst **revision** descriptors; "surprise" appears **zero** times in the
  datasheet. The Zacks Rank has four components (Agreement, Magnitude, Upside,
  Surprise); this screener implemented **only Surprise**, the one practitioners
  weight least.
- **Measured on the full 502-name published payload, 2026-09-09.** Coverage
  99.6%; **0.0% ties** price-scaled; overlap with `forward_eps_growth` (the
  same FY1 consensus line, 45% of the growth category) only **+0.152**, so a
  level and a change in that level are confirmed to be distinct objects;
  **71.6% unspanned** by all eight existing categories (R2 0.284).
- Full working: `research/2026-09-07-revisions-category-has-no-revisions.md`,
  sections 8.0-8.7.

**What this entry does NOT claim.** It does **not** claim the ranking improves.
The research pre-registered a materiality bar - "if the change moves fewer
names than deleting the category outright, argue it on explainability alone" -
and that bar **fired**: deleting the category moves 7 of the top 50, this
change moves 3. The bar is asymmetric (a 3.3%-of-composite within-category
reweight measured against deleting a 10% slot), but a threshold set in advance
and explained away the moment it fires is not a threshold. **The case is
construct validity: a category named for revisions now measures revisions.**
No backtest figure and no `live_ic_history.csv` number appears anywhere in this
entry (rules 4 and 5); the `1m` horizon holds **3 effective observations**
against a gate of 8.

**Expected effect:** 3 of the top 50 change; median |delta rank| 9; 218 names
move more than 10 ranks; max 109. `revisions ~ momentum` category correlation
**+0.171 -> +0.317** (4th largest of the 28 pairs). Revisions unspanned
**91.1% -> 85.2%**. That last pair is independence **spent deliberately** to
buy construct validity, and is named as a cost rather than left unremarked.

**Observed 2026-09-11 (first live run scored with the new weights).** Top-50
turnover 2026-09-10 -> 2026-09-11 was **exactly 3** (`AMCR`, `MAS`, `STLD` in;
`ADSK`, `CVS`, `HIG` out), matching the pre-registered figure, with rank
Spearman 0.9841.

**But the check is underpowered and does not confirm anything.** Ordinary
day-over-day top-50 churn across the prior ten run transitions (2026-08-28
onward) is **median 3, range 0-7**, so a no-op day produces the same number.
Spearman 0.9841 is the second-lowest of those eleven transitions - marginally
more movement than typical, well inside the range. The prediction was not
falsified; it was also not tested. Separating the weight change from one day of
price movement needs the same run scored both ways, which the pipeline does not
currently support. Recorded here rather than claimed as confirmation.

**Validated by:** `tests/test_fy1_revision.py`, **44 tests** - formula, sign,
price scaling, the change-vs-level property, every missing-input path, the
`eps_trend` extraction against a mock frame (including that a broken
`eps_trend` costs one metric and not the whole ticker), the three registries,
the exact reweight, both display formatters, and end-to-end through the scoring
pipeline. Full suite **1117 -> 1161 passed, 0 failed**. Golden fixture
regenerated; its mock data now carries FY1 endpoints spanning both signs, plus
a loss-making name with an upward revision and one name with no data at all.

**Live verification:** fetched 8 real tickers end-to-end - 8/8 coverage, values
+1.9 to +52.7 bp, inside the measured full-universe distribution
(p10 -19.5, median +5.9, p90 +52.5 bp).

**Second-order effect, flagged as unmeasured by the research and now
measured.** The coverage-discount denominator in `compute_composite()` is
`METRIC_COLS`, so a 45th metric shifts every stock's coverage ratio. On the
live payload: max change in the discount **0.0019** (about 0.1 point of
composite), and **zero** names cross the 0.80 threshold in either direction.
For the ~2 names that lack the metric the discount rises by at most 0.0029 -
the mechanism working as intended, since they genuinely do have less data.

**Three things a future session must not undo:**

1. **Scale by price, and know why.** Not to reduce the momentum overlap - it
   does not, at all - but because the estimate-scaled denominator is
   **undefined** on this universe (a name whose 90-day-ago FY1 consensus rounds
   to 0.00 makes the mean literally +inf).
2. **The ~+0.42 correlation with momentum is economic, not mechanical.** Five
   reconstructions were measured, including a **sign-only** variant with no
   denominator at all; every one carries +0.32 to +0.43, while `1/price ~
   momentum` is -0.128, the *wrong sign* for the artifact explanation. Do not
   "fix" it with a cleverer denominator without re-running section 8.3.
3. **The revisions category has spent its independence budget.** Adding any
   further momentum-adjacent metric here requires re-running the section 8.4
   spanning regression first, not just a citation.

**Rejected, and why:** diffusion (`eps_revisions`, the Zacks "Agreement"
component) - **75.8% ties** on the full universe, 37.3% of names pinned at
exactly +1.0; three names in eight would form one indistinguishable rank block.
Also the conservative `fy1_revision_3m = 20` variant: at 20 the surprise family
is still 63% of a category named for revisions, leaving the defect largely in
place.

**Display note.** A `bp` (basis points of price) format was added to both the
Python and JS metric formatters. Under the existing `pct` format at one
decimal, the measured p10/median/p90 (-0.00195 / +0.00059 / +0.00525) collapse
onto two strings, manufacturing visible ties in a metric with 0.0% actual ties.
The two formatters are asserted to agree, because the drilldown's prose and its
metric table are rendered by different code paths.

**Applied by:** morning session (manual, research-led per rule 4).

**Rollback:** `good/2026-09-09`. The change touches `factor_engine.py`
(fetch + metric + three registries), `config.yaml`, `schemas.py`,
`generate_dashboard.py`, `stock_summary.py`, the golden fixture, three existing
test modules whose pinned counts moved, and documentation.

---

## 2026-09-11 - The published percentile means "best", not "largest", and the page now says so

**Area:** presentation of scored data (no scoring change)

**Changed:** three surfaces that publish a direction-adjusted percentile without
stating the convention.

1. `metric_meta[m]["dir"]` added for all 37 published metrics, **derived from
   `factor_engine.METRIC_DIR`** rather than written out, so the page cannot
   drift from the ranking it describes.
2. The drilldown's metric table: the column header `Percentile Rank` becomes
   `Sector Percentile - 100 = best`, a convention note is rendered beneath it,
   and each metric name carries a `↓ better` / `↑ better` chip with a tooltip.
3. The "Why it ranks here" prose appends `, lower is better` for the 13
   inverted metrics only - `_label_and_value()` in `stock_summary.py`.

Also: the eight category columns in the universe table (`Val`, `Qual`, `Grow`,
`Mom`, `Risk`, `Rev`, `Size`, `Inv`) gained definition tooltips naming their
scored metrics and weights, the bank carve-out, and - for `Risk` - the fact
that a **high** score means **low** risk.

**Nothing about how a stock is scored or ranked changed.** No weight, metric,
threshold or formula moved; `Composite` and `Rank` are unaffected.

**Evidence - a demonstrable user-facing failure, measured on the live payload.**
`compute_sector_percentiles()` does `ranks = 100 - ranks` wherever `METRIC_DIR`
is `False`, which is **13 of the 37 published metrics**. The percentile
therefore always means "better than this share of its sector" and never "larger
than". On the 2026-09-11 published payload:

| Stock | EV/EBITDA | Published percentile |
|---|---|---|
| HON | 6.95 | **99** |
| AXON | 98.61 | **0** |

and equivalently RSG beta -0.37 at the 99th against CVNA 2.35 at the 0th, and
UAL PEG 0.24 at the 99th against KMI 28.34 at the 0th.

Nothing on the page stated this. A reader seeing `EV/EBITDA 6.95` beside "99th
percentile" had no way to distinguish it from a raw rank, and the natural
reading - "this company's EV/EBITDA is high for its sector" - is exactly
backwards. The same contradiction appeared in prose: *"the 97th sector
percentile on EV/EBITDA (9.02)"* (HST, live).

This is the surface whose entire purpose is explaining **why** a stock ranks
where it does (priority 4, owner directive 2026-08-10). A correct number
presented so that its obvious reading is inverted is a comprehension defect, and
for the investment-club audience it is the expensive kind: a student who learns
the convention backwards misreads every valuation and risk metric on the site.

**Two errors caught before shipping, by checking rather than recalling.** Draft
tooltips named **P/B** under Valuation and **PEG** under Growth. Both carry
**zero** weight for non-banks - `config.yaml` marks P/B "Bank-only" and PEG
"Removed: P/E / growth double-counts valuation". Both would have taught
something false about how the score is built. The active weights were read from
`config.yaml` directly; `tests/test_percentile_direction.py` pins the
correction so the claim cannot silently return.

**Expected effect:** no change to any score, rank, or scored output file.
Payload cost measured at **+1.2 KB gzipped (+0.1%)** on a 1,200 KB wire payload
- 515 summary sentences gained the qualifier, and 37 metrics gained a short
`dir` string.

**Validated by:** `tests/test_percentile_direction.py`, **32 tests**, of which
**27 fail against the pre-change code** (26 of the generator's, plus the prose
test). The agreement test is the load-bearing one: it compares every published
`dir` against `METRIC_DIR` itself, so a future change to one and not the other
fails rather than silently publishing a false direction. Full suite
1161 -> 1193, zero failures either side.

**Not done, deliberately:** the qualifier was **not** added to the prose for
higher-is-better metrics. The ambiguity exists only where the percentile and the
raw value point opposite ways, and 502 stocks of prose is ~101 KB gzipped - a
phrase on every metric in every sentence is not free. The per-metric chip in the
drilldown covers all 37.

**Applied by:** morning session (manual).

**Rollback:** `good/2026-09-10`. Touches `generate_dashboard.py` and
`stock_summary.py` only; no scoring code, no config.

---

## 2026-09-15 - The dashboard gains a sell-side surface, built to the shape the evidence allows

**Area:** dashboard surfaces (no scoring change); per-stock summary prose

**Changed:** three things - one product, two supporting.

1. **New "My Holdings" section** in `generate_dashboard.py`, between Top 5 and
   What Changed. A client-side list (`localStorage`, key
   `screener_holdings_v1`, **tickers only**) that renders every saved name on
   every run with its rank, composite, a per-category score-and-delta strip,
   and the review sentences already baked into `stock_detail[t]["summary"]`.
   Plus a concentration line: names, sectors, largest sector share, how many
   sit inside the top 25 and top 100, how many carry a trap flag.
2. **New summary fact kind `change_driver`** in `stock_summary.py`: the
   category that moved furthest since the history baseline, the direction of
   its *score*, and what that category now contributes to the composite. It
   appears in every stock's drilldown, not only on holdings.
3. **A grammar fix in the existing `change` sentence** - "moved up 1
   **places**" had been live for every one-rank mover, which is the *median*
   move for a top-25 name.

**No weight, metric, threshold, formula or scoring path was touched.** This
entry exists because the *shape* of the surface is a methodology decision:
three of its properties are constraints taken from the literature, and without
this record they would read as arbitrary omissions to whoever finds them next.

**Evidence** - all from `research/2026-09-14-sell-discipline-and-hold-bands.md`,
which read each source directly:

- **Why the list is never filtered or sorted by size of move.**
  Akepanidtaworn, Di Mascio, Imas & Schmidt (2023), *Journal of Finance* 78(6),
  3055-3098: 783 institutional portfolios averaging $573M, 2000-2016, 4.4M
  trades. Sells underperform a factor-neutral random-sell counterfactual by
  **-80 bp/year** while buys beat theirs by **over +100 bp/year**. The
  mechanism is an attention failure: positions extreme on prior returns - best
  *and* worst - are sold at rates **more than 50% higher** than middling ones,
  surviving stock-date fixed effects. The proof that it is attention rather
  than ability is the earnings-day natural experiment, where sells beat
  non-announcement-day sells by **+150 bp/year**. The deficit is *worst* among
  fundamentals-oriented concentrated high-tracking-error managers - this
  screener's exact shape. A review queue ranked by size of move is that
  heuristic automated, so the surface lists everything, ordered by rank.
- **Why it never asks what you paid.** Odean (1998), *JF* 53(5), 1775-1798:
  PGR **0.233** against PLR **0.155**, a **1.50x** ratio at **t = -32**; the
  winners sold beat the losers held by **+1.03% over 84 days (p=0.002)** and
  **+3.41% over a year (p=0.001)**. The disposition effect is defined relative
  to the purchase price, so a cost basis *is* the reference point that produces
  it. No cost basis, share count or P&L field exists in the code or in what it
  stores, and a parametrised test asserts 23 such terms are absent.
- **Why there is no exit signal.** Novy-Marx & Velikov (2016), *RFS* 29(1),
  104-147: a buy/hold spread is "the single most effective simple cost
  mitigation strategy"; their hysteresis momentum factor nets **0.51%/month
  (net FF4 alpha 0.33, t=8.81)** against **0.31%/month (alpha 0.17, t=3.06)**
  for restricting to a low-cost universe. MSCI Momentum buffers between rank
  250 and 750 against a 500-name target; S&P DJI states the principle outright:
  "the addition criteria are for addition to an index, not for continued
  membership." This screener has one test (`portfolio.num_stocks: 25`). The
  second threshold is **deliberately not set here**: §6.3 of the note measured
  a 25/50 band firing **zero** times across 18 runs, and the strict top-25 rule
  producing sells that **round-trip 71% of the time** (22 of 31 back inside the
  top 25 within five runs). §9 asks for 60+ comparable runs before committing
  to a width; there are 32. Shipping a band today would have been shipping a
  guess dressed as a rule.
- **Why the movers panel could not simply be reused.** Measured on this repo's
  32 comparable runs: a top-25 name's median absolute rank change between runs
  is **1** (p95 **10**, n=675 holding-days) against a universe median of **7**
  (p95 **43**, n=13,542). The movers panel's material threshold *is* that
  universe p95, so it fires for a top-25 name **1 time in 675 - 0.15%**.
- **Why `change_driver` exists.** Same paper, earnings-day result: sells
  anchored to *information* do well, sells anchored to the *size* of a move do
  not. Until today the dashboard could state how far a stock had moved and not
  what moved it.

**Measured while building, and worth carrying forward:** across the 500 stocks
with a one-month category delta, the named largest mover is **Risk 34.0%,
Revisions 29.0%, Momentum 26.4%** - and **Quality 0.2%, one stock in 500**.
Roughly 90% of one-month category movement comes from the three categories fed
by daily prices and estimates; the fundamentals categories barely move between
quarterly filings. **Any future deterioration trigger keyed to Quality or Growth
would essentially never fire at monthly cadence.** That is a direct input to
question 2 of the research note's §8.

**Expected effect:** no change to any score, rank, or scored output file.
Payload cost measured: `change_driver` adds **+99 KB raw / +10.5 KB gzipped
(+0.89%)** across 502 stocks; the page adds **+25.9 KB raw / +6.4 KB gzipped**.
Total **+16.9 KB on the wire, about +1.4%**. The holdings panel reads only
fields the payload already carried, so it costs nothing beyond its own markup.

**Validated by:** `tests/test_holdings_panel.py`, **61 tests**, of which **60
fail against the pre-change generator**. Nine drive the real emitted script
under Node against a stubbed DOM and assert on rendered output, because "the
string is in the file" is a weak check for a panel whose entire contract is
what it renders: that a rank-300 name which moved not at all still appears, and
appears last; that a name new to the universe with a withheld category still
renders; that the drilldown's other sentences do not leak onto a review row;
and that a `localStorage` key hand-edited to hold `{ticker, shares, cost}` is
read for its ticker and written back clean. Plus **13 tests** for
`change_driver` and the plural fix in `tests/test_stock_summary.py`. Full suite
**1193 -> 1264, zero failures** either side. Also rendered against the live
502-stock payload with six real holdings and read through.

**Applied by:** morning session (manual).

**Rollback:** `good/2026-09-14`. Touches `generate_dashboard.py` and
`stock_summary.py` only; no scoring code, no config, no data artifact.

---

## 2026-09-16 - No hold band, and the rule for when one may be chosen

**Area:** portfolio construction rules (`config.yaml -> portfolio.num_stocks`)

**Changed:** **nothing.** No weight, metric, threshold, formula or scoring path
was touched. This entry exists because a *decision not to change* needs to be as
findable as a change, and because it pre-registers a condition that binds future
sessions.

`portfolio.num_stocks: 25` remains the screener's single membership test. Three
independent sources - Novy-Marx & Velikov (2016), MSCI Momentum, S&P DJI - say a
screen should use a **wider test for continued membership than for entry**, at
1.5x-3x the buy band. The 2026-09-14 research note recommended one. **It was not
adopted, and this records why**, so the next session neither adopts it by default
nor relitigates it from scratch.

**Evidence that a band is warranted (accepted):** of the names breaching a strict
top-25 boundary, the share back inside the top 25 at the very next review is
**37.5% at daily cadence, 47.4% at 2-4 days, 31.0% weekly**. The only rule this
screener has wastes between a third and a half of the trades it implies, and
widening the band monotonically reduces that at every cadence measured.
Directional, and it replicates three times out of three.

**Evidence that fixes the width (absent):** at a 1.4x band the same three
cadences report **0.0%, 31.2% and 5.9%** wasted - resting on **9, 7 and 3** fully
disjoint triples. No width in the 1.5x-3x range is distinguishable from any other
on this data.

**The measurement correction that produced this, and it generalises.** The note's
§6.3 reported that a 2x band "produced zero signals". That was an artifact of
walking **one path** through 18 runs. Over all comparable pairs it fires at 1.65%
of weekly holding-looks - but those pairs **overlap almost completely** (34 runs
yield 72 pairs at a fortnight's spacing), which is the same independence trap
`research/2026-08-10-ic-evidence-independence.md` found in the IC series and
`improvement_engine._effective_observations()` guards against. Restricting to
non-overlapping pairs raises wide-band breach rates by **2-3x** (weekly B=50:
1.65% -> 4.50%) and reduces the honest sample to **8 weekly, 4 fortnightly and 2
monthly** independent looks. Rank-migration statistics need the same treatment as
ICs; nobody had noticed.

**Pre-registered condition for revisiting - the part that binds:**

> Do not commit to a hold-band width until there are **>= 8 disjoint observation
> windows at the review cadence the band will govern**. Today there are **2
> monthly**. The note's original "60+ comparable runs" is the wrong unit - 60
> runs of a daily series is still 2-3 independent monthly looks. At one per month
> of continuous running this is roughly **2027-04**, within a month of when the
> improvement engine reaches its own 8-observation gate, for the same reason.
>
> At that point: choose the **narrowest** band whose wasted-trade rate is below
> half the strict rule's and whose implied monthly one-sided turnover is under
> Novy-Marx & Velikov's 50%. If two qualify, take the narrower - signal given up
> is a real cost, and the practice range's 3x upper end comes from 500-name
> quarterly indices, not 25-name screens.

**Do not pick 50 because MSCI doubles.** That is borrowing a parameter from a
different instrument, and it is the specific shortcut this entry exists to block.

**What the evidence *did* endorse, deferred to the build day:** the screener
records a **quarterly** rebalance cadence (`config.yaml` line 221) while the
dashboard regenerates **every weekday and states no cadence anywhere**. Strict
top-25 turnover is **121.8% monthly one-sided at daily review against 24.0% at
monthly** - NMV find few anomalies survive costs above ~50%, so daily action on
this surface sits 2.4x outside the survivable region, a conclusion that tolerates
a 2.4x error in the estimate before it changes. That gap needs no threshold and
no new data.

**Expected effect:** none on any published score. `portfolio.num_stocks: 25` and
every category weight are byte-identical.

**Validated by:** `research/measurements/2026-09-16-hold-band-and-input-churn.py`,
which reproduces every number above from `improvement/snapshots/` through the
same comparability gate `history.py` uses, and prints the overlapping and
disjoint estimators side by side so the difference cannot be overlooked again.
Full suite **1264 -> 1264**, unchanged, as expected for a session that shipped no
production code.

**Applied by:** morning session (manual) - synthesis day.

**Rollback:** not applicable; no code or config changed. Documentation only.

---

## 2026-09-17 - The tool states the cadence it is built for, and flags when a rank move is the inputs changing

**Area:** portfolio construction rules (`config.yaml -> portfolio.review_cadence`,
new key); dashboard decision surfaces (holdings panel, What Changed panel,
per-stock summaries)

**Changed:** no weight, metric, threshold, formula or scoring path. Every
category weight, `portfolio.num_stocks: 25` and all 45 metric definitions are
byte-identical, and no published score moves. Two things a reader sees changed:

1. **`portfolio.review_cadence: 'quarterly'` is now a config key** rather than a
   bare comment, and the dashboard states it on the two surfaces that show rank
   movement. The value is unchanged - `config.yaml` has recorded a quarterly
   rebalance cadence since launch. What changed is that the generator can now
   read it, so the page can say it.
2. **A per-stock caveat fires when >= 2 metric percentiles changed availability**
   between the run being shown and its comparison baseline, on the drilldown and
   on every holdings row. New `input_churn` fact in `stock_summary.py`; new
   `ch: [lost, gained]` key in the `history.py` delta payload.

Both were specified as items 1 and 2 of §8.7 of
`research/2026-09-14-sell-discipline-and-hold-bands.md`, written 2026-09-16.

**Evidence, change 1 - the cadence.** Novy-Marx & Velikov (2016, *Review of
Financial Studies* 29(1) 104-147) find anomalies under roughly **50% monthly
one-sided turnover** mostly survive trading costs and few above it do, and name a
buy/hold spread "the single most effective simple cost mitigation strategy".
Measured on this repo's own snapshots, acting on the strict top-25 rule at every
run implies **121.8%** monthly one-sided turnover against **24.0%** reviewing the
same rule monthly - so daily action sits **2.4x outside** the region NMV find
survivable, and that conclusion tolerates a 2.4x error in the estimate before it
reverses. Barber & Odean (2000, *JF* 55(2)) supply the household-level version
already quoted in the panel: the most active quintile earned 11.4% a year against
a 17.9% market return.

The defect this corrects is a **product** one, not a methodology one. The
methodology has always said quarterly; the dashboard regenerated every weekday
and said nothing, which implicitly invites a reader to act on every redraw. A
grep of `generate_dashboard.py` for "quarterly" before this change returned one
unrelated data-source label.

**It is a sentence, not a lock.** The tool does not know what a reader is doing
and must not pretend to. Naming the cadence it was built for is decision support;
withholding a number until a date would not be, and would also break the data
loop's own reason for running daily.

**Evidence, change 2 - input churn.** The mechanism is arithmetic and needs no
significance test: when a metric percentile flips between present and absent, its
category renormalises over a different metric set (`factor_engine`'s `has_data`
mask, working as designed), so the score moves without the company moving. This
is `CLAUDE.md` priority 1.5's FCX case - growth 68.3 -> 42.5 -> 68.3 across three
runs - which was investigated as a suspected defect and turned out to be correct
behaviour that nothing downstream could distinguish from a real collapse.

Only the **size** was ever in question, and it was measured over 12,044
ticker-transitions across 24 run-pairs:

| input churn | n | median &#124;rank change&#124; | share worsening |
|---|---|---|---|
| none | 11,450 | **6** | 44.6% |
| 1 metric | 447 | 7 | 47.0% |
| 2-3 metrics | 143 | **21** | 52.4% |

**The threshold is 2, and that is the finding.** One changed metric is
indistinguishable from ordinary run-to-run noise (median 7 against 6) and firing
on it would mark 4.93% of transitions in order to say nothing; two or more
triples the median move and marks 1.22%. A caveat that fires four times as often
as it means anything trains a reader to ignore it - the same failure mode that
made the permanent bank-only "High severity" alarm worthless (fixed 2026-09-01).

**It is worded as a caveat on the comparison, never as a reason to act**, and
that is also measured rather than stylistic: churn >= 2 leaves **52.4%** of names
worse off against a **44.6%** base rate. It scatters ranks; it does not push them
down. A surface presenting "the measurement got noisier" as deterioration would
manufacture exactly the kind of sell trigger Akepanidtaworn, Di Mascio, Imas &
Schmidt (2023, *JF* 78(6) 3055-3098) find costs institutional managers **80
bp/year**. A test asserts the sentence contains none of "deteriorat", "worse",
"warning", "risk", "concern", "weaken" or "decline", and that it reads
identically whether the stock rose or fell.

**Expected effect:** no published score moves. On the 2026-09-17 run, **27 of 502
stocks** carry the churn caveat against the ~1-month baseline (5.4%). That is
higher than the 1.22% measured rate because the measurement used consecutive runs
<= 7 days apart while the drilldown's preferred baseline is ~28 days, over which
more availability changes accumulate - expected, and worth stating so a future
session does not read it as the threshold misfiring. Payload cost is one optional
two-integer key on ~5% of delta entries.

**Two construction rules that must not be "tidied":**

- **Only metric columns *both* runs carry are compared.** The snapshot schema has
  grown over time (`fy1_revision_3m_pct` appears part-way through the directory).
  Counting a column that did not exist yet as a metric that went missing would
  flag the whole universe on the day a metric was added.
- **"Cannot tell" must not render as "nothing changed."** Snapshots before
  2026-03-09 carry 15 columns and no percentiles at all, so `input_churn()`
  returns `None` rather than `(0, 0)` for them, and the caveat is simply absent.

**Validated by:** `tests/test_input_churn.py` (58 tests, **52 of which fail**
against the pre-change code) and `tests/test_review_cadence.py` (40 tests, **37
of which fail** against the pre-change code), the latter driving the real emitted
JavaScript under Node against a stubbed DOM rather than grepping the artifact.
The churn wiring is additionally checked against the live snapshot directory: it
must fire on a real minority of names, neither zero (dead wiring) nor most of the
universe (wrong schema rule).

The turnover and churn figures are reproduced by
`research/measurements/2026-09-16-hold-band-and-input-churn.py`, and a test pins
the two excluded non-metric `_pct` columns against that script so the shipped
flag and its published justification cannot drift apart.

**Not decision-grade, and not used:** no backtest number and no IC observation
appears above (rules 4 and 5). Every measured figure is a descriptive statistic
on published scores and ranks - no forward returns anywhere.

**Applied by:** morning session (manual) - build day, implementing §8.7 items 1
and 2 of the 2026-09-14 research note.

**Rollback:** `good/2026-09-16`. Reverting restores a dashboard that states no
cadence and cannot distinguish a rank move from an input going missing; it does
not change any score.

---

## 2026-09-22 - The dashboard answers "how much?" with inputs, and never with a weight

**Area:** dashboard surfaces (no scoring change)

**Changed:** one new block, `holdingsConcentration(rows)` in
`generate_dashboard.py`, rendered on My Holdings below the existing fit line.
North-star question 4 - *how much / does it fit?* - had **no surface at all**
between the Model Portfolio's removal on 2026-08-26 and today. Three lines:

1. **The name count against the published counts for a diversified portfolio**,
   with a computed "below all three / above N of the three".
2. **The equal-split slice** (100/N), with the published caps on a single
   holding quoted for scale.
3. **The widest risk gap on the list**, in raw annualised volatility, with the
   equal-dollar arithmetic stated.

Two footnote paragraphs source the refusal to emit a weight, and the choice of
raw volatility over the percentile.

**No weight, metric, threshold, formula or scoring path was touched.** It is a
*view* over fields `stock_detail` already carried: regenerating the dashboard
left `dashboard_data.js` **byte-identical**, so the payload cost is zero.

**Evidence:**

- **DeMiguel, Garlappi & Uppal (2009), *RFS* 22(5), 1915-1953.** 14 optimisation
  models across 7 datasets; **none consistently beat 1/N** on Sharpe, CEQ or
  turnover. Reliably beating 1/N for 25 assets would need an estimation window
  of roughly **3,000 months**. Conditions: US equity, monthly rebalance,
  comparison over the same asset set - i.e. exactly this question, since
  selection has already happened.
- **Chopra & Ziemba (1993)**, via **Ziemba & MacLean (2011)**, *Stochastic
  Optimization Methods in Finance and Energy*, Springer ISOR 163, ch.1: errors
  in the **means do ~20x** the damage of covariance errors (~2x for variances),
  worsening to **~100:3:1** near zero risk aversion. A conviction-proportional
  weight is a mean-return estimate, which is the worst place to put estimation
  error. This is why the block reports facts and leaves the number to the reader.
- **Statman (1987) *JFQA*** (30 borrowing / 40 lending); **Campbell, Lettau,
  Malkiel & Xu (2001) *JF*** (~50, as idiosyncratic volatility rose over
  1962-1997); **Domian, Louton & Racine (2007) *Financial Review* 42(4),
  557-570** (on **shortfall risk** over 20 years: **63** names for 10%, 93 for
  5%, 164 for 1%). **Condition, and it is stated on the page:** all three
  measure *randomly selected* portfolios, so for a pre-screened large-cap list
  they bound the question rather than settle it.
- **Documented practice - caps, not targets.** US RIC Subchapter M **25/5/50**;
  UCITS **5/10/40** (ESMA, UCITS Directive Art. 52); S&P Dow Jones Select Sector
  indices re-cap a constituent above **24%** (and the >4.8% group above 50% of
  index weight). In every documented institutional scheme - equal, cap,
  inverse-volatility or optimiser weight - **the alpha signal drives selection
  and weighting is a separate, risk-driven decision.** Nobody sizes long-only
  equity in proportion to a bounded composite score.

Full sourcing in `research/2026-09-21-position-sizing-and-how-much.md`, §3.4,
§4.1, §4.2, §4.4 and §8.4.

**The measurement that decided the risk line's input.** §8.4 of the research
note proposed showing "the volatility percentile of each holding". That
percentile **cannot carry the claim**, and this is the substantive correction
made today. Percentiles here are **sector-relative**
(`factor_engine.compute_sector_percentiles` groups by `Sector`) and
**direction-inverted** (`METRIC_DIR['volatility']` is `False`), so a high value
means "calm *for its sector*". Measured on the 2026-09-22 run by
`research/measurements/2026-09-22-holdings-risk-comparability.py` over all
**111,417** cross-sector pairs of the 501 names carrying both figures:

| | |
|---|---|
| Pairs the percentile orders **backwards** vs raw volatility | **26,581 (23.9%)** |
| Worst case | LITE (IT) pct 1.4, raw 0.947 vs ARE (Real Estate) pct 0.0, raw 0.472 |
| | LITE reads as the safer holding at **2.00x** the volatility |
| Raw volatility p10 -> p90 | 0.205 -> 0.524, a **2.55x** spread |

So the block uses **raw annualised volatility**, which is directly comparable
between any two names, and the footnote explains why. These are descriptive
statistics on published percentiles and realised price volatility - no forward
returns anywhere, so rules 4 and 5 do not bite.

**Expected effect:** no score, rank or portfolio moves. A reader holding six
names now sees where six sits against 30/50/63, what an equal split of their own
list is against published caps, and which holding carries the most price
variability - none of which the page previously said.

**Validated by:** `tests/test_holdings_concentration.py`, **33 tests, all 33
failing against the pre-change generator**; **17** drive the real emitted script
under Node against a stubbed DOM. The load-bearing one is
`test_risk_line_follows_raw_volatility_not_the_sector_percentile`, which builds
a holding pair whose percentile and raw volatility disagree and asserts the
block follows the raw number - the 23.9% finding as a regression. Rendering was
also checked against the live payload for a 1-, 3- and 6-name list: a
tech-heavy three (NVDA 37% vs AAPL 25%, 1.5x) and a mixed six (NVDA 37% vs JNJ
19%, 2.0x).

**Not decision-grade, and not used:** no backtest number and no IC observation
appears above (rules 4 and 5).

**Applied by:** morning session (manual) - product day, implementing §8.4 of the
2026-09-21 research note.

**Rollback:** `good/2026-09-21`. Reverting restores a holdings panel with no
concentration block; it does not change any score.

---

## 2026-09-23 - Position weighting moves to equal, and the published methodology stops describing a scheme the tool does not use

**Area:** portfolio construction (`portfolio.weighting`), and the public
disclosure of it in `SCREENER_OVERVIEW.md` / the embedded methodology panel

**Changed:** three things - one methodology, two disclosure.

1. **`config.yaml` `portfolio.weighting`: `'score'` -> `'equal'`.** Position
   weights are now `100 / num_stocks` rather than composite-score proportional.
2. **`run_screener.py` now maps every weighting scheme to its own description**
   (`weighting_description()`), replacing a two-branch ternary that read
   "equal, else risk-parity" over a **four**-option setting.
3. **`max_position_pct` is disclosed as non-binding where arithmetic forbids it
   from firing** (`_max_pos_note()`). The value is unchanged at 5.0 and was
   deliberately not removed.

**The disclosure defect, which is the more serious half.** Because the ternary
had no `score` branch, it took its `else` on every run the tool has ever made,
and `SCREENER_OVERVIEW.md` - the canonical public methodology reference, which
`generate_dashboard.py` embeds verbatim into `index.html` - stated:

> **Weighting:** Risk-parity (inverse-volatility weighting - lower-volatility
> stocks get more weight)

The portfolio was composite-score weighted, which tilts the **opposite** way:
toward the highest-scoring names, not the calmest ones. That sentence was live
on the public site on 2026-09-23 and in every published overview before it. A
second site misdescribed the same thing: limitation 7 listed `score` under "the
default weighting uses single-name volatility only", and score weighting takes
no volatility input at all. Both are corrected, and the correction is pinned to
the *live config* rather than to a string, so the artifacts cannot drift from
the setting again without failing a test.

**Evidence** (research and documented practice; per rules 4 and 5, no backtest
number and no IC observation is used here):

- **Chopra, V.K. & Ziemba, W.T. (1993)**, read via **Ziemba, W.T. & MacLean,
  L.C. (2011)**, "Using the Kelly Criterion for Investing", ch.1 of *Stochastic
  Optimization Methods in Finance and Energy*, Springer ISOR 163: errors in
  **expected returns** do roughly **20x** the damage of errors in covariances
  (variance errors ~2x covariance errors), worsening to about **100:3:1** for
  an investor near zero risk aversion. Conditions: a mean-variance investor,
  cash-equivalent-loss metric, with the ratio rising as risk tolerance rises.
  **Score-proportional weighting is sizing by an expected-return estimate** -
  the single most error-sensitive input in the problem - using a composite
  whose predictive accuracy this system has measured at **3 effective
  (non-overlapping) observations** at the `1m` horizon.
- **DeMiguel, V., Garlappi, L. & Uppal, R. (2009), *Review of Financial
  Studies* 22(5), 1915-1953.** Across **14 optimisation models** (sample
  mean-variance, Bayes-Stein, minimum-variance and shrinkage variants) and
  **7 datasets**, none consistently beat **1/N** on Sharpe ratio, certainty
  equivalent or turnover. For sample-based mean-variance to beat 1/N reliably
  requires an estimation window of roughly **3,000 months for 25 assets** and
  **6,000 for 50**. Conditions: US equity calibration, monthly rebalancing,
  comparison over the *same* asset set - which is exactly this decision, since
  selection has already happened by the time weights are assigned.
- **Documented practice.** RIC Subchapter M diversification (25/5/50); UCITS
  Art. 52 (5/10/40); S&P DJI Select Sector capping (4.8% group / 24% single
  name, with the capping mechanism revised 2024-09-23 from clipping the
  smallest breacher to reducing all breachers proportionately); S&P 500 Equal
  Weight resetting every constituent to a fixed 0.2% quarterly; quant managers
  running constrained optimisers against Barra or Axioma risk models. **No
  documented institutional scheme sizes long-only equity in proportion to a
  bounded composite score.** In every one of them the alpha signal drives
  *selection* and weighting is a separate, risk-driven decision.
- **The disagreement, and why it resolves against us.** Practitioners optimise
  anyway, against commercial covariance models with far more structure than a
  sample covariance matrix, and under mandates that require explicit risk
  control. DeMiguel's critique targets *sample-based* estimation, which is
  precisely the setup this repo has and is not going to replace. See
  `research/2026-09-21-position-sizing-and-how-much.md` section 5.

**Measured, on this repo's own construction arithmetic** - "what weights does
this rule emit given these scores", not a backtest and not a return
measurement, so rules 4 and 5 do not reach it. Re-run this session over **41
run dates** (2026-02-20 .. 2026-09-23, one snapshot per date, two degraded
3-row February files excluded) with
`research/measurements/2026-09-21-position-sizing-dispersion.py`:

| | |
|---|---|
| Equal weight, 25 names | 4.000% |
| Score weight, observed span | **3.771% .. 4.569%** |
| Max deviation from equal weight | **0.569 pp** (median 0.412) |
| Active share vs equal weight, same names | median **1.30%**, max 1.98% |
| Heaviest/lightest ratio | **1.206x** |
| Positions ever hitting the 5% cap | **0** |

Composite scores are level-bounded 0-100 and the selected top 25 of 502 sit in
a narrow band (2026-09-23: 64.94-73.67), so a ~13% spread in level becomes a
~13% spread in weight around 4%. **`weighting: 'score'` was equal weight with
noise**, and could not be anything else under this construction.

**Expected effect:** **near zero on the portfolio, by the measurement above** -
at most 0.57 pp per position, median active share 1.30%, and no change to which
stocks are selected, since `weighting` is applied after selection and no
scoring path reads it. The gain is that **the tool now does what it says**,
which is the trade `CLAUDE.md` asks for explicitly: a change that improves a
number but makes the tool harder to explain is a bad trade, and this is its
mirror image.

**Why not inverse-vol, which would look more sophisticated.** **Moreira, A. &
Muir, T. (2017), *JF* 72(4)** report large alphas from scaling exposure by
inverse prior realised variance, but **Cederburg, S., O'Doherty, M.S., Wang, F.
& Yan, X.S. (2020), *JFE* 138(1)** test **103 strategies** and find
vol-managed portfolios do not systematically outperform; implementable
out-of-sample versions earn **lower** certainty equivalent and Sharpe than the
unmanaged originals, from structural instability in the spanning regressions.
The gains concentrate in momentum, profitability and BAB. `inverse_vol` stays
selectable and is now described for what it is - equalising risk contribution,
not improving expected return.

**Why not Kelly or fractional Kelly.** It requires a calibrated probability
distribution; this screener emits a cross-sectional rank with no probability
attached and no calibrated score-to-return mapping, so building it would mean
inventing the input. The overbetting penalty is also asymmetric and severe -
Ziemba & MacLean note 2x Kelly drives the long-run growth rate to zero.

**On `max_position_pct`, which is inert and stays.** It never bound in 41 run
dates under `'score'`, and under equal weighting of 25 names every position is
exactly 4.00%, so it binds only **below 20 holdings**. A parameter that reads
as a live safety control and cannot fire is the same failure shape as the
always-firing bank-metric alarm fixed 2026-09-01 - it trains a reader to stop
looking. The fix is to **label** it, not to delete it: it becomes live the
moment `num_stocks` falls or a dispersed scheme is selected, and the overview
now states both the current 4.00% slice and the threshold at which the cap
starts to matter. All three weight columns are still capped in
`portfolio_constructor.py` regardless of the active scheme, so the Excel
sheet's `InvVol` and `Score` columns are unaffected.

**How this fits the rest of the screener (the coherence question).** Sizing is
the one place where the eight categories deliberately do **not** apply, and
separating them is what every documented practitioner scheme does. It also
insulates the sizing decision from the category-overlap problem in
`research/2026-09-02-category-independence-synthesis.md`: score weighting
propagated whatever double-counting exists among the categories straight into
position size, where Chopra & Ziemba say the damage is 20x. Note also that
equal weight's historical excess return over cap weight is **more than half a
size tilt**, and this screener already runs an explicit `size` category - so
the change is justified on **estimation-error and explainability** grounds and
explicitly **not** on equal weight's return history, which would be betting the
same way twice and calling it two things.

**Validated by:** `tests/test_weighting_disclosure.py`, **28 tests**. Verified
against the pre-change tree: `test_live_page_states_the_configured_scheme`
fails on the shipped `index.html` with exactly the assertion this entry
describes. The suite pins the mapping (every scheme the schema accepts has its
own branch; only the inverse-vol branch may say "inverse-volatility"; an
unrecognised scheme names itself rather than borrowing another's description),
the artifact-vs-config agreement for both `SCREENER_OVERVIEW.md` and
`index.html`, and the cap note's threshold arithmetic. One test reads the
accepted set out of `schemas.py`, so adding a fifth scheme without a
description branch fails. Full suite **1411 -> 1439, no failures**.
`dashboard_data.js` regenerated **byte-identical**: zero payload cost.

**Not decision-grade, and not used:** no backtest number and no IC observation
appears above (rules 4 and 5). The "3 effective observations" figure is quoted
as a statement about *how little is known* about the composite's accuracy - the
argument against sizing by it - not as evidence for any weight.

**Applied by:** morning session (manual) - synthesis day, implementing sections
8.1 and 9 of `research/2026-09-21-position-sizing-and-how-much.md`.

**Rollback:** `good/2026-09-22`. Reverting restores `weighting: 'score'` and
the false risk-parity sentence on the public site; the portfolio would move by
at most 0.57 pp per position.

---

## 2026-09-25 - Two-source metric inputs can now actually use their second source

**Area:** metric definitions - the data-availability rule for nine numeric
inputs to `compute_metrics()`, and the documented intent of three of them

**Changed:** nine inputs that have two possible sources were all written as the
nested form `d.get(A, d.get(B, np.nan))`. That reaches `B` only when key `A` is
*absent*. `_fetch_single_ticker_inner()` writes every one of those keys
unconditionally - `_safe()` and `_stmt_val()` both return NaN rather than
omitting the key - so the fallback could only ever fire on an exception path,
never on the missing-data path it was written for. Replaced with a single
`_coalesce(d, *keys)` helper that skips a present-but-NaN or `None` value:

| input | preferred | falls back to | feeds |
|---|---|---|---|
| `_debt_info` | `.info` `totalDebt` | `totalDebt_bs` | enterprise value |
| `_debt_bs` | `totalDebt_bs` | `.info` `totalDebt` | invested capital |
| `_cash_ev` | `.info` `totalCash` | `cash_bs` | enterprise value |
| `_cash_bs` | `cash_bs` | `.info` `totalCash` | invested capital |
| `_ebit_curr` | `ebit_annual` | `ebit` | `operating_leverage` |
| `_price`, `_cur_price`, `_cur_price_c`, `_current_price`, bank `price` | `.info` `currentPrice` | `price_latest` | `peg_ratio`, `price_target_upside`, `proximity_52w_high`, `pb_ratio`, display |

Two deliberate exceptions:

- **`return_12m` loses its fallback rather than gaining one.** It is the only
  price site whose order ran the other way (`price_latest` first), and the
  `currentPrice` leg is **deleted**. It could never fire - `price_latest` and
  `price_12m_ago` are written by the same `len(hist) >= 10` block, so whenever
  the near endpoint is missing the far one is too - and if it ever did it would
  divide an unadjusted `.info` price by an adjusted `price_12m_ago`. That is
  exactly the cross-scale division that published MNST at momentum 71.5 when
  its true 12-1 return was the 3rd percentile (fixed 2026-08-26).
- **`fy1_revision_3m` is unchanged in behaviour.** It was guarded by hand on
  2026-09-10; it now shares the helper instead of its own inline check.

**Evidence:** a concrete, live, measured defect - not a citation, and none is
needed for a data-availability bug. Reproduced by
`research/measurements/2026-09-25-dead-two-source-fallbacks.py` against the
four retained raw fetches in `runs/*/00_raw_fetch.parquet` (503 names each,
identical counts on all four):

| first source NaN, backup usable | names |
|---|---|
| `totalDebt` -> `totalDebt_bs` | 1 (FISV) |
| `totalDebt_bs` -> `.info totalDebt` | 3 (ANET, ERIE, ISRG) |
| `totalCash` -> `cash_bs` | 1 (FISV) |
| `cash_bs`, `ebit_annual`, `currentPrice` | 0 |

Part 2 of the script is an exact A/B of the whole universe against the
pre-change module, same market series both sides. **Metrics gained, none lost**,
and no other numeric column changes except `_metric_count`:

| name | regains | weight restored |
|---|---|---|
| ANET | `roic`, `net_debt_to_ebitda` | **45 of 100 quality** |
| ISRG | `roic`, `net_debt_to_ebitda` | **45 of 100 quality** |
| FISV | `fcf_yield`, `ev_ebitda`, `ev_sales` | **80 of 100 valuation** |

ANET and ISRG report `.info` total debt of exactly **0.0** with no "Total Debt"
line on the quarterly balance sheet - the reading a debt-free company should
get, which the dead fallback discarded. FISV is the reverse: `.info` returned
nothing usable while the balance sheet carried 28.034B of debt and 245M of
cash, both filed 2026-06-30.

ERIE is rescuable on the same input and **regains nothing** - its EBIT is
missing too, so ROIC stays NaN for an unrelated reason. Reported because the
incidence count says 3 names and the outcome is 2.

**The price sites have measured incidence zero** and are therefore a safety net
rather than a live fix. They are still worth making real: `_current_price` is
what `improvement_engine.record_run_snapshot()` writes as `price_at_scoring`,
and a name with no baseline price contributes no forward return at all - so an
`.info` gap would quietly shrink the evidence base the project is bottlenecked
on, rather than showing up as a visible hole.

**The cost, stated rather than hidden:** ROIC's comment claimed all three
invested-capital components come from the same filing "for temporal
consistency". With the fallback live, debt may come from `.info` instead. That
claim is now qualified at the site. The error is bounded by construction - a
filing that omits Total Debt is a filing with little or no debt - and measured:
on the three affected names `.info` debt differs from the annual filing's
long-term debt by **<= 1.2% of invested capital**, against losing 45% of the
quality category outright.

**Expected effect:** three names move, and the next data run can check it.
Against the 2026-09-25 raw fetch: **ANET rank -79** (composite +4.12),
**ISRG -22** (+2.98), and **FISV +4** (-0.75) - gaining data made FISV score
slightly *worse*, which is the honest outcome and worth stating. 183 of 503
names shift by a median of **1** rank from the percentile-cohort ripple, and
**top-50 membership does not change at all** (0 in, 0 out). Absolute ranks in
that A/B are not live ranks - `_daily_returns` does not survive the parquet
round-trip, so four risk/momentum metrics are absent on both sides; the deltas
are valid because the treatment is identical. ANET should appear in the What
Changed movers panel after the next 02:00 run. If it does not, this line is
wrong and the correction belongs against this entry.

**Validated by:** `tests/test_nan_source_fallback.py`, **37 tests**, of which
**28 fail against the pre-change tree** (verified by running the new module
against it). They cover the helper's semantics including `0.0`-is-a-value, the
three real record shapes above, the "no fabrication" cases where both sources
are NaN, `return_12m`'s refusal to cross sources, and a source-level guard that
bars the nested-`get` idiom from `factor_engine.py` so this cannot recur
silently. Full suite 1498 -> **1535, no pre-existing failures**.

**Applied by:** morning session (manual) - harden-and-teach day, closing the
`currentPrice` fallback decision left open on 2026-09-11 and carried by nine
subsequent sessions. The investigation widened it: the item was recorded as six
price sites with zero incidence, and the same root cause turned out to be
costing real metrics at the debt and cash sites in the same four lines.

**Rollback:** `good/2026-09-24`. Reverting restores the dead fallbacks, which
means ANET and ISRG lose 45% of their quality weight and FISV 80% of its
valuation weight again.

---

## 2026-09-25 - The public methodology page stops describing a Revisions category from before 2026-09-10, and a fetch-failure rate from before the fixes

**Area:** public disclosure of methodology - `SCREENER_OVERVIEW.md`, which
`generate_dashboard.py` embeds verbatim into the `index.html` GitHub Pages
serves. No score changes in this entry.

**Changed:** four statements on the live page, three of them created by the
2026-09-10 session, which changed the Revisions weights and left every piece of
prose describing the category behind.

1. **`fy1_revision_3m` had an empty "What It Measures" cell** - the heaviest
   metric in the category at **35%**, undescribed, and the only row in the
   whole document still labelled with a raw snake_case identifier rather than a
   plain-English name. Now named **FY1 EPS Revision (3-month)** and described:
   the 90-day change in consensus current-fiscal-year EPS divided by price, so
   it reads in basis points and compares across a $20 and a $400 stock.
2. **"Analyst Surprise gets the highest weight"** - it does not, and has not
   since 2026-09-10. It is **15%** against the revision metric's **35%**,
   contradicted by the table two lines above the sentence. The paragraph now
   explains the actual ordering *and why*, which the old text never did for
   either arrangement.
3. **Limitation 5 and a trailing note both said a forward-EPS-consensus-change
   metric was "not feasible with yfinance"** and listed it as a future
   enhancement requiring FactSet or Refinitiv I/B/E/S - for a metric that had
   been live, and the category's heaviest, for 15 days. Replaced with the
   limitation that is actually true: the estimate history reaches back about
   **90 days**, so a recent revision is visible but *persistence* over CJL's
   six-month window is not.
4. **"Approximately 10-25% of tickers may fail to fetch on a given run due to
   Yahoo Finance rate limiting (HTTP 429)"** - unrelated and older. Replaced
   with the measured rate.

Step 1 also now states the two-source input fallback that the other 2026-09-25
entry made real, since it can change a published score.

**Evidence:**

- Claims 1-3 are contradicted by `config.yaml` on the same repository: the
  Revisions table's own weight column, and `metric_weights.revisions`, both
  give `fy1_revision_3m` 35% against `analyst_surprise` 15%.
- Claim 4 is contradicted by measurement: the **18** scheduled data runs from
  2026-09-02 to 2026-09-25 report **0 fetch failures across 9,036
  ticker-fetches**, counted from `logs/datarun-*.log`. `CLAUDE.md` priority 1
  recorded on 2026-09-01 that this figure was stale and asked for periodic
  re-verification; this is that re-verification, and it extends the clean run
  from 15 logs to 18.
- The replacement prose for claim 2 cites the sources the reweight was actually
  made on, which the page did not carry at all: **Chan, Jegadeesh & Lakonishok
  (1996), *JF* 51(5)** - the analyst-revision leg was the strongest of the
  three earnings-momentum measures tested, **+7.7% six-month decile spread**,
  IBES 1977-93 - and **Martineau (2022), *Critical Finance Review* 11(4)** -
  post-earnings-announcement drift **absent in large caps since 2006**, with a
  significantly negative coefficient over 2016-19, which is why the
  backward-looking surprise family was cut from 78% of the category to 45%.

**Expected effect:** none on any score. `dashboard_data.js` came back
**byte-identical** (sha256 verified against the published run), so zero payload
cost; `index.html` grows 287,316 -> 290,383 bytes for the corrected text. What
changes is that a student reading the heaviest metric in a category now finds
out what it measures, and is not told that a quarter of the universe may be
missing when the measured rate is zero.

**What was deliberately NOT done:** the fetch-reliability correction does not
claim fetches never fail. It is a free unofficial API; rate limiting stays
listed, and the sentence now points at `scripts/check_run_health.py` as the
reason a published number can be relied on - a run with price coverage under
90%, analyst-target coverage under 50%, or dispersion 20% below its trailing
median is discarded before publication. Replacing an overstated risk with an
understated one would be the same failure in the other direction.

**Validated by:** `tests/test_overview_claims.py`, **14 tests**, of which
**12 fail against the pre-change `SCREENER_OVERVIEW.md` and `index.html`**
(verified by checking both files out at `HEAD` and re-running). They are derived
from `config.yaml` rather than pinned to today's numbers: the weight multiset in
the Revisions table must equal the configured non-zero weights, the metric named
as highest-weighted must be the one that is, and a metric with non-zero weight
may not be described as infeasible. Two are general guards that would have
caught defect 1 the day it shipped - **no table cell anywhere in the document
may be empty**, and **no metric row may be labelled with a raw config key**.
Four more assert `index.html` carries the corrections, because editing the
markdown without regenerating is how all four stayed visible.

**Applied by:** morning session (manual) - harden-and-teach day. Found while
checking whether the ROIC temporal-consistency claim I had just qualified in
code was also made on the public page. It was not; these were.

**Rollback:** `good/2026-09-24`. Reverting restores the undescribed 35% metric,
the contradicted "highest weight" sentence, the claim that the live metric is
impossible, and the 10-25% fetch-failure figure.
