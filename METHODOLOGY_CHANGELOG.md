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

See `.claude/plan/backtest-v2.md`. This is priority 2 in `CLAUDE.md` for a
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
