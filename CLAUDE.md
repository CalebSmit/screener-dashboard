# Screener Dashboard - Project Context

## What this is

A multi-factor S&P 500 stock screener: a Python scoring engine plus a static
HTML/JS dashboard deployed via GitHub Pages at
https://calebsmit.github.io/screener-dashboard/

**Two audiences.** The owner uses it for his own research. It is also being
built into something college investment clubs can use — which means it must be
*teachable and explainable*, not merely accurate. A change that improves a
number but makes the tool harder to explain is usually a bad trade.

It is **not investment advice**. Its credibility is the product.

## Your mandate

You have full authority to change anything in this repo: scoring, weights,
metrics, tests, architecture, docs, the improvement engine, all of it. The
owner is not reviewing PRs. You push to `main` yourself.

The goal is simple: **the tool should be measurably better every morning than
it was the night before.**

That authority comes with exactly one obligation:

> **Every change must be justified by evidence, and the evidence must be
> written down where someone else can check it.**

"I think this is better" is not evidence. Evidence is:

- **published research** - a paper, with effect sizes and the conditions under
  which the effect held
- **documented professional practice** - how quant shops, institutional
  managers and serious practitioners actually build screens, and why
- **measured results** - a backtest, an information coefficient, a regression
  test, a profiling number
- **a documented failure** - a real user-facing problem you can demonstrate

**Owner direction, 2026-08-11: the first two carry as much weight as the
third.** Do not stall a well-reasoned, well-sourced methodology change because
it cannot be backtested yet. The backtest is known-broken (see
`.claude/plan/backtest-v2.md`) and genuinely independent IC observations accrue
at roughly one a month, so demanding measured proof up front would freeze the
project for half a year. Measurement is how a change is *confirmed over time*,
not the gate it must pass to be made.

What this does **not** license: changing something because it seems better.
The bar is a written argument a sceptical reader could follow to its sources.
If you cannot produce that, spend the session finding it instead. **A night
that produces one well-sourced finding and no code is a good night.**

### Build a coherent screener, not a pile of good ideas

The point is not to accumulate individually-defensible tweaks. It is to
understand how the pieces **fit together**: how factors interact and overlap,
where two metrics measure the same thing, which combinations professionals
actually use and which they avoid, and what the whole system is implicitly
betting on.

A change that improves one factor while quietly duplicating another, or that
raises a score while making the tool harder to explain, is a bad change even
with a citation attached. Ask what the screener is *for* and whether the change
makes it better at that - then write down the reasoning.

## Non-negotiable rules

These are not about permission. They are about not destroying the thing you are
improving.

1. **The ship gates are absolute.** You may only push to `main` when all of them
   pass (see "Ship gates" below). `main` is served live to the public by GitHub
   Pages — a broken push is a broken public site with nobody watching. If a gate
   fails, leave the work on the branch, write up why, and stop.

2. **Never rewrite history on `main`.** No force-push, no `git reset --hard` on
   `main`, no amending pushed commits. The tagged history is the rollback path;
   if you destroy it, there is no recovery.

3. **Every methodology change is recorded in `METHODOLOGY_CHANGELOG.md`** before
   it ships: what changed, the evidence, the expected effect, and how you
   validated it. A weight or threshold that changes without a changelog entry is
   an unexplainable tool, which defeats the purpose.

4. **Methodology is research-led. Performance history does not drive it yet.**
   Owner direction, 2026-08-20. Until the evidence base is much larger:

   - `allow_auto_apply` is **false**. The engine still records snapshots,
     computes forward returns, and *reports* proposals - it may not write a
     weight change. Do not flip it back without meeting both conditions in the
     `config.yaml` comment.
   - **Do not justify a methodology change with the IC series or
     `performance_history.csv` either.** There are 3 observations, all at the
     `1w` horizon, all from February. The significance test counts raw rows,
     which `research/2026-08-10-ic-evidence-independence.md` shows overstates
     independence by ~2.35x. Numbers off that base are not yet evidence.
   - **What does justify a change:** published research and documented
     professional practice, per "Your mandate" above, plus a clear account of
     how the change fits the screener as a whole.

   This is not a reason to avoid weights. It is a reason to change them from
   *research* - what the literature and practitioners say a factor is worth and
   why - rather than from a thin return series. Say so in the changelog entry.

   Meanwhile, keep improving the engine and its evidence base: fix observation
   independence, make forward returns accrue correctly, widen coverage. When
   the history is deep enough it becomes the better basis for weights, and the
   engine is how it gets applied - just not yet.

5. **The backtest does not decide anything until 2027-02-11.** Owner direction,
   2026-08-11. Until that date `backtest.py` output is **supporting colour
   only**. You may run it, report it, and use it to notice something worth
   researching. You may **not**:
   - cite a backtest number as the justification for a methodology change
   - revert or keep a change because of what the backtest said
   - quote it in `METHODOLOGY_CHANGELOG.md` under **Evidence**
     (put it under a separate *Backtest observation* line, clearly marked
     as not decision-grade)
   - feed it into the improvement engine as a validation gate

   The reason is not squeamishness: the current backtest has documented
   survivorship and look-ahead bias, so a number from it is not weak evidence,
   it is evidence pointing in an unknown direction. **Until then, methodology
   decisions rest on research** - published literature and documented
   professional practice - per "Your mandate" above.

   `improvement_engine.py` is a separate matter and is *also* benched for now -
   see rule 4. It learns from live forward returns, which are genuinely
   out-of-sample and unaffected by the backtest's biases, so it remains the
   right mechanism for weight changes *eventually*. It is switched off today
   because the history is too thin, not because the approach is wrong.

   If backtest v2 lands early and honestly fixes both biases, that is worth
   raising in the log - but the date stands until the owner moves it.

6. **Never commit secrets, credentials, or the `.certs/` bundle.**

7. **Preserve the defensibility features** - weight sensitivity, factor
   correlation, run provenance/snapshots, trap detection. You may redesign or
   replace them with something better; you may not quietly drop them. They are
   why the tool is credible.

8. **Update `NIGHTLY_LOG.md` every session.** It is the only memory that carries
   between sessions. Write it for a reader with no other context.

   **Every entry starts with these four health numbers**, whatever the focus:

   | Check | Where | Healthy |
   |---|---|---|
   | Did the last code session actually run? | newest `logs/nightly-*.log` | ends "shipped to main" or "no changes" - **not** "SESSION DID NOT RUN" |
   | Did the data loop publish? | newest `logs/datarun-*.log` | ends "Data loop complete", HEALTH: PASS |
   | Evidence base | `improvement/live_ic_history.csv` | **row count and newest date, as two literal numbers** |
   | Priority 0 | below | fixed, or still top of the queue |

   The evidence-base line would have read "3 rows, newest 2026-02-22" on every
   session from February to 2026-08-21, while the data loop ran successfully
   every weekday. Nobody wrote it down, so nobody noticed it had stopped
   moving. **If those two numbers have not moved in three consecutive sessions,
   making them move is that session's work, whatever the rotation says.**

9. **Don't hand-edit generated files.** `dashboard.html`, `index.html`, and
   `dashboard_data.js` are outputs of `generate_dashboard.py`. Edit the
   generator. `index.html` is what Pages serves.

## Ship gates

Run these before pushing to `main`. All must pass. The runner script re-checks
them independently and will refuse the push if you got it wrong.

| # | Gate | Command |
|---|---|---|
| 1 | Full suite passes, no new failures vs the baseline you took at session start | `python -m pytest tests/ test_screener.py -q` |
| 2 | Pipeline wiring intact | `python run_screener.py --dry-run` |
| 3 | Dashboard artifacts intact | `index.html` non-trivial and `dashboard_data.js` parses |
| 4 | No stray uncommitted files | `git status --porcelain` |

On success the runner tags the commit `good/YYYY-MM-DD`. That tag is the
rollback point - see `ROLLBACK.md`.

## The weekly cycle

Research-weighted by design: understand before building, and prove it after.

| Day | Focus | Output |
|---|---|---|
| **Mon** | **Research.** One specific thing - a factor, a metric, a threshold, a construction rule - learned properly, from *both* the literature and documented practice. Where they disagree, say so and say why. | A dated note in `research/`, complete in one session: citations with effect sizes, plus how practitioners actually do it |
| **Tue** | **Product.** Open the live dashboard as a user would. Does it answer *what should I look at / should I buy this / should I sell what I hold / how much*? Fix or build what it can't. | A dashboard change, or a written account of what it cannot answer and why |
| **Wed** | **Synthesis.** How does Monday's research fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. | A design section on Monday's note, plus any `METHODOLOGY_CHANGELOG.md` entry |
| **Thu** | **Build.** Implement what the week justified. Write tests alongside. | Working, tested code |
| **Fri** | **Harden and teach.** Tests, docs, methodology page, error handling, the investment-club experience. | A tool someone else can pick up and understand |

**Monday must produce a complete note.** It used to be split across Monday
(literature) and Tuesday (practice). The 2026-08-21 retrospective found that in
16 days the rotation produced **one** research note and **zero** practitioner
appendices - and that ~64% of scheduled sessions never started at all, which
makes any two-day chain fragile by construction: a lost Monday left Tuesday
with nothing to append to. One self-contained day is the honest unit.

**Tuesday is the product day**, and it is new. The owner's standing directive is
that the dashboard becomes the single place to look before buying or selling.
Between 2026-07-29 and 2026-08-21 not one line of `generate_dashboard.py`
changed, because no day in the rotation pointed at it and every session that
could run was spent on data-pipeline defects. Firefighting will always win a
fair fight against product work; this day exists to stop the fight being fair.

**Validation is continuous, not a weekly gate.** When the data loop has
accumulated enough evidence to test something, test it and record the result -
in `METHODOLOGY_CHANGELOG.md` against the entry that made the change. If a
change turns out to be wrong, revert it and say so. But do not wait for proof
before making a well-sourced change, and do not manufacture a backtest number
from a backtest you know is biased.

**Every other Friday is a RETROSPECTIVE instead** (even ISO week numbers). The
runner swaps in `.claude/prompts/retrospective.md` automatically. That session
does not work on the screener - it evaluates whether this routine is actually
producing value and rewrites its own process: these rules, the daily prompt,
the rotation, the runner scripts, even the retrospective prompt itself.

Two things a retrospective may never do: **weaken or remove the four ship
gates**, and **remove the evidence requirement, the rollback tagging, or that
restriction**. It may make them stricter. If it believes a gate is wrong, it
argues the case in the log and leaves it for the owner. A process able to
quietly relax its own standards eventually will.

If a day's focus has genuinely nothing valuable left, **do not manufacture
work** - move to the next most valuable thing and record the swap in the log.
An honest "UI is fine; spent the session on fetch resilience instead" is a good
outcome.

## Where things live

### Scoring engine
- `run_screener.py` - pipeline entry point
- `factor_engine.py` - metric registry (`METRIC_COLS`, 44 entries), scoring
- `portfolio_constructor.py` - sector-constrained portfolio construction
- `improvement_engine.py` - **the methodology learning loop** (see below)
- `backtest.py` - decile backtest + IC validation. Known-weak; see `.claude/plan/backtest-v2.md`
- `presets.py` - weighting presets (balanced/value/growth/momentum)
- `config.yaml` - all tuneable parameters
- `schemas.py`, `cli.py`, `run_context.py`, `instrumentation.py`

### Front-end
- `generate_dashboard.py` - **source of truth**; writes `dashboard.html` + `dashboard_data.js`
- `dashboard_data.js` - `window.SCREENER_DATA`. `table_data` holds all ~500 stocks
  with all 8 category scores; `stock_detail` covers **all ~500 stocks** (raw
  values, percentiles, per-category `contrib` attribution, peers, price
  targets, financials, provenance) and is ~90% of the payload weight. Check
  `.claude/plan/dashboard-inventory.md` before building anything "new".

### Docs (public-facing - keep truthful)
- `SCREENER_OVERVIEW.md` - canonical methodology reference
- `METHODOLOGY_CHANGELOG.md` - every methodology change, with evidence
- `Multi-Factor-Screener-Blueprint.md`, `SCREENER_DEFENSIBILITY_SPEC.md`, `README.md`

### Tests
- `tests/` (24 modules) + `test_screener.py`. Run: `python -m pytest tests/ test_screener.py -q`
- `conftest.py` at root protects published artifacts from test side effects.
  **Deeper fix still open:** point offending tests at `tmp_path` and stub the
  network call in `get_sp500_tickers`.

### Routine
- `.claude/prompts/nightly.md`, `scripts/nightly-screener.ps1` (6:00 AM code loop)
- `scripts/data-run.ps1` (2:00 AM data loop)
- `NIGHTLY_LOG.md`, `research/`, `ROLLBACK.md`, `logs/` (gitignored)

## The two loops

The tool only improves if **both** run. This was broken before 2026-08-05:
the improvement engine had 3 IC observations since February because nothing
was running the screener.

**Data loop (2:00 AM, Mon-Fri)** - `scripts/data-run.ps1` runs the screener
live, regenerates the dashboard, records an improvement-engine snapshot, and
pushes. Daily on weekdays to accumulate evidence as fast as possible.
**Watch repo growth:** `dashboard_data.js` is ~3 MB and changes every run, so
this adds roughly 60 MB/month of poorly delta-compressing JSON to git history.
If that becomes a problem, options are downsampling the payload, committing
data less often than the dashboard refreshes, or squashing history
periodically - raise it in the log rather than silently letting the repo bloat. This is what accumulates the evidence: forward returns, live ICs,
dispersion history. Without it, methodology can never learn.

**Code loop (6:00 AM, Mon-Fri)** - your session. Improves the system that
produces and uses that evidence.

If you find the data loop has not run or is failing, **fixing it is the highest
priority work available**, ahead of any feature. A stalled data loop means the
tool stops getting better in the way that matters most.

## Improvement engine

`config.yaml -> improvement:` gates weight changes on statistical significance:
`min_observations_for_proposal: 8`, `min_ic_ir_for_auto_apply: 0.5`,
`max_change_per_cycle: 3.0`, `shrinkage: 0.5`.

`allow_auto_apply` is **false** (owner direction 2026-08-20 - see rule 4 and
`METHODOLOGY_CHANGELOG.md`). The engine records snapshots, computes forward
returns and reports proposals; it may not write a weight change. Re-enable only
when both conditions in the `config.yaml` comment are met. Those statistical
gates are the safety mechanism - if you weaken them, you must justify it in
`METHODOLOGY_CHANGELOG.md` with a reason better than "it wasn't firing."

**The evidence base is not growing.** `live_ic_history.csv` has held 3 rows,
all horizon `1w`, all February 2026, through every successful data run since -
180 days of no progress as of 2026-08-21. `record_run_snapshot()` never calls
`compute_live_ic()` (step 5 of priority 0). Until that is fixed, nothing else
here matters.

Good work here: more/faster evidence, better IC estimation, regime handling,
smarter proposals. Every engine-applied change should also land in the
changelog.

## The dashboard is the product

Owner directive, 2026-08-05: **the dashboard should become the single place to
look when considering buying or selling a stock.** You have authority to add
and to delete features. Research before building.

**Read `.claude/plan/dashboard-inventory.md` before touching the dashboard.**
There is much more in it than a first look suggests - 501-stock detail
payloads, contribution attribution, sector peers, price targets, provenance,
and a very large embedded methodology document. The most likely failure mode is
rebuilding something that already exists. The governing plan is
`.claude/plan/dashboard-north-star.md`. In short: every element must help answer one of *what
should I look at / should I buy this / should I sell what I hold / how much*.
The dashboard is already strong at evaluating a single name; its real gaps are
the **absence of any time dimension** and the **absence of a sell-side
workflow**.

And the line that keeps it defensible: **decision support, not a
recommendation engine.** Show why, with sources and uncertainty visible. Never
emit a bare "buy". That distinction is what makes it appropriate for a college
investment club rather than a liability.

## Current priorities (rewrite this section as things land)

**-1. Sessions do not start. This is the dominant failure mode.**
Measured by the 2026-08-21 retrospective: of the 11 scheduled code-loop slots
from 2026-08-06 to 2026-08-20, **2 produced anything, 4 started and produced
nothing, and 5 never fired at all.** Causes, each fixed reactively after it had
already cost a day: `gh` false negative (08-06), no network (08-07), untrusted
workspace (08-10, 08-11), logged-out reboot (08-12), API weekly limit (08-14),
machine off or logged out for four consecutive weekdays (08-17 to 08-20).

**Zero ship-gate failures in that whole period** - not because the gates are
lenient, but because only one session ever produced code to gate.

Two things remain open:

- **The scheduled-task definitions are not in version control.**
  `grep -rn "Register-ScheduledTask" .` finds nothing. The 02:00 and 06:00
  triggers exist only as hand-made Task Scheduler entries on one machine.
  Committing a registration script is the single highest-value infrastructure
  work available.
- **Nothing watches whether the loop is running.** A run that never fires
  writes no log, so its absence is invisible until someone counts files.
  The 2026-08-21 fix makes a *failed* session loud; a *missing* one is still
  silent.

Workspace trust (resolved 2026-08-13) regresses as: `python --version` works
but everything else is denied. Fix in `scripts/fix-trust.ps1`; the runner now
fails fast with instructions.

**0. THE FIRST SESSION MUST FIX THIS, ahead of any rotation focus.**
`compute_forward_returns()` in `improvement_engine.py` (~line 250) skips any
snapshot whose date already appears in `performance_history.csv`:

```python
if snap_date_str in existing_dates:
    continue
```

Snapshots get processed as soon as they are 7 days old, when only the **1-week**
return is computable. `fwd_return_1m` is written `NaN`, the date joins
`existing_dates`, and **the date is never revisited** - so the 1-month return is
never filled in, however much time passes.

Consequences, re-verified 2026-08-10:
- 14 distinct snapshot dates exist (2026-02-20 through 2026-08-07)
- `performance_history.csv` now *does* have `fwd_return_1m` / `fwd_return_3m`
  columns (the 2026-08-05 note that it has only `fwd_return_1w` is stale), but
  only **2 of 14 dates** carry a 1m value and **1 of 14** a 3m value - those
  dates happened to sit unprocessed past the horizon by accident, not by design
- `live_ic_history.csv` has 3 rows, **all horizon `1w`**
- `config.yaml` sets `optimization_horizon: '1m'` and documents that it will
  *refuse to propose* if that horizon has no data

**Net effect: the improvement engine could never have proposed anything, and
never will, until this is fixed.** The entire self-improvement premise depends
on it.

> **STOP - do not ship the obvious fix on its own.**
> Research 2026-08-10 (`research/2026-08-10-ic-evidence-independence.md`) found
> that the backfill this unblocks is **not 11 independent observations**. All 11
> backfillable dates fall inside a 53-day window, so at most **2 non-overlapping
> 30-day return windows** fit among them; six sit in a single 9-day stretch.
> `_ir_to_one_sided_pvalue()` computes `t = IR * sqrt(n_obs)` from the raw row
> count, so the backfill inflates t by **2.35x** and moves a borderline IC-IR of
> 0.5 from p=0.24 to p=0.049 - straight through the `min_ic_ir_for_auto_apply`
> gate. With `allow_auto_apply: true` the engine would then start rewriting
> weights. **That is worse than the current inert state**, because the output
> would look authoritative.

The fix is therefore a package, not a one-liner:

1. Make the dedup key `(run_date, horizon)`-aware so a date is reprocessed when
   it becomes old enough for the next horizon.
2. Deduplicate `(run_date, ticker)` on append. Several snapshot files share a
   date and are all appended in one pass, so `performance_history.csv` is ~75%
   duplicate rows and `live_ic_history.csv` records **6,539 "tickers"** for
   2026-02-21 in an S&P 500 screener.
3. Count *effective* (non-overlapping) observations for the significance test
   rather than raw rows. On today's data that turns n=11 into n=2 and correctly
   refuses to propose.
4. Exclude weekend run dates - 5 of the 14 are Saturdays/Sundays, which have no
   market close.
5. Make the data loop actually call `compute_live_ic()`. It currently calls only
   `compute_dispersion()` and `compute_forward_returns()`, so the IC series
   never advances on its own.

**If step 3 cannot land in the same session as step 1, set
`allow_auto_apply: false` before shipping step 1** and record it in
`METHODOLOGY_CHANGELOG.md`.

Write a regression test that fails on the current behaviour. Record the fix in
`METHODOLOGY_CHANGELOG.md` - it changes what evidence the engine acts on.

**Start with step 5.** It is the smallest of the five, it is independent of the
other four, and it is the one whose absence is currently doing the most damage:
`live_ic_history.csv` has not gained a row in **180 days** (3 rows, all `1w`,
newest 2026-02-22) despite the data loop running successfully every weekday
since 08-13. Steps 1-4 improve evidence the engine will eventually act on;
step 5 is why there is no evidence at all. Note that step 3 must still land
before `allow_auto_apply` goes back to `true`.

Realistic expectation: this does **not** clear the 8-observation gate in days.
Genuinely independent monthly observations accrue at about one per month, so
honestly ~8 months from the start of the scheduled loop (2026-07-28) unless the
optimization horizon is reconsidered. Say so plainly rather than engineering
around it.

**0.5 / 0.7. DONE - do not weaken these.** `scripts/check_run_health.py`
(2026-08-10) discards a run before publishing on: missing fetch evidence, price
coverage <90%, analyst-target coverage <50%, or category dispersion >20% below
the trailing median. 14 tests in `tests/test_run_health.py`. The stale-cache
root cause behind those degraded runs was fixed 2026-08-13
(`factor_scores_cache_max_age_days()`, 21 tests in
`tests/test_cache_freshness.py`, changelog 2026-08-13) and **confirmed working**
by the 08-14 and 08-21 runs: live fetch, 100% price coverage, HEALTH: PASS.
Detail in `NIGHTLY_LOG.md` 2026-08-10 (evening) and 2026-08-13. Each threshold
exists because that exact failure shipped to the public site.

**0.6. Do not record an improvement-engine snapshot when the run did not
fetch.** Found 2026-08-11: a warm-started run still writes a snapshot, so a day
with two cached runs produced three "observations" of one real data point, all
byte-identical in composite. The engine gates on an observation *count*, so
duplicates directly inflate its confidence - the same evidence-inflation
failure `research/2026-08-10-ic-evidence-independence.md` identified via
overlapping return windows, arriving by another route. Either skip the snapshot
on a warm-start, or deduplicate on `(run_date, content hash)` before the engine
reads them.

1. **Keep the data loop healthy.** Currently ~10-25% of tickers fail per run
   (Yahoo rate limits). Every failed ticker is lost evidence, and evidence now
   drives both methodology and the historical spine below.

   **Related open defect (found 2026-08-06):** when a fetch fails, the pipeline
   silently substitutes *synthetic* values - "Generated sector-realistic sample
   values" - and produces output indistinguishable from a real run. With no
   network it fabricated all 503 tickers and still emitted a normal-looking
   2.6 MB payload. `data-run.ps1` now gates on this, but the fabrication
   happens upstream in `run_screener.py` / `factor_engine.py` and should be
   fixed at source: refuse, or exit non-zero, rather than emitting fiction that
   looks like analysis. This is a credibility bug, not a robustness nicety.
2. **Give the dashboard a time dimension.** Rank/score deltas between runs,
   per-category trends, biggest movers. Unlocks the sell-side workflow and
   time-series valuation. Buildable from `improvement/snapshots/` as the data
   loop accumulates. See the north-star plan.
3. **Backtest v2** - the current one has survivorship bias and holds
   fundamentals constant (look-ahead). It cannot honestly validate a
   methodology change, and now that the system validates *itself*, that bias
   steers the learning loop. `.claude/plan/backtest-v2.md`.
4. **Replace the AI chat with generated per-stock summaries** (owner directive
   2026-08-10). The chat needs each visitor to paste their own Anthropic API
   key, which is unusable for an investment club and un-reproducible. Build the
   deterministic template version first - the payload already has `contrib`,
   `pct`, `peers` and price targets, so summaries can be exact, free, identical
   for every viewer, and impossible to hallucinate. Explains *why it ranks
   there*, never *whether to buy*. Plus a run-level overview of what changed.
   See `.claude/plan/dashboard-north-star.md`.
5. **Sell-side workflow** - client-side watchlist/holdings, deterioration
   flags, review queue. Question 3 is currently unanswerable.
6. **Investor profile selector** - `.claude/plan/investor-profiles.md`.
   Reconcile with `presets.py` first. Note `contrib` is Balanced-only.
7. **Investment-club readiness** - can a student open this on a phone and
   understand what they're looking at?
8. **Test isolation** - remove the need for the `conftest.py` guard.
