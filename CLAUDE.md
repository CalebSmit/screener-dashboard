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

"I think this is better" is not evidence. Evidence is: a citation to published
research, a backtest result, an information-coefficient measurement, a
regression test, a profiling number, a documented user-facing failure.

If you cannot produce evidence for a change, do not make the change. Spend the
session finding the evidence instead. **A night that produces one
well-evidenced finding and no code is a good night.**

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

4. **Prefer the improvement engine over your own judgment on weights.**
   `improvement_engine.py` changes factor weights from measured live
   information coefficients, with shrinkage, per-cycle caps, and significance
   gates. That is a far better basis for a weight change than an LLM's opinion.
   Improve *the engine* and feed it *more data*; do not hand-tune weights around
   it. Hand-tuning is a last resort and needs a written argument for why the
   engine's evidence was insufficient.

5. **Never commit secrets, credentials, or the `.certs/` bundle.**

6. **Preserve the defensibility features** - weight sensitivity, factor
   correlation, run provenance/snapshots, trap detection. You may redesign or
   replace them with something better; you may not quietly drop them. They are
   why the tool is credible.

7. **Update `NIGHTLY_LOG.md` every session.** It is the only memory that carries
   between sessions. Write it for a reader with no other context.

8. **Don't hand-edit generated files.** `dashboard.html`, `index.html`, and
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
| **Mon** | **Research.** Pick one open question from the priorities list. Read actual literature and peer tools. | A dated note in `research/` with citations and a recommendation |
| **Tue** | **Research to design.** Turn the strongest evidence into a concrete, falsifiable design. State the hypothesis and how a backtest would refute it. | A design section appended to the research note |
| **Wed** | **Build.** Implement the highest-value designed item. | Working, tested code |
| **Thu** | **Validate.** Backtest / IC / regression the change. Does it actually beat baseline? **If it doesn't, revert it and say so.** | A validation result, merged or reverted |
| **Fri** | **Harden and teach.** Tests, docs, methodology page, error handling, the investment-club experience. | A tool someone else can pick up and understand |

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
  with all 8 category scores; `stock_detail` covers portfolio holdings only

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

`allow_auto_apply` is now **true**, so the engine may write weight changes once
its evidence gates are satisfied. Those gates are the safety mechanism - if you
weaken them, you must justify it in `METHODOLOGY_CHANGELOG.md` with a reason
better than "it wasn't firing."

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

Consequences, all verified 2026-08-05:
- 13 distinct snapshot dates exist (2026-02-20 through 2026-07-29)
- `performance_history.csv` has **only** a `fwd_return_1w` column
- `live_ic_history.csv` has 3 rows, **all horizon `1w`**
- `config.yaml` sets `optimization_horizon: '1m'` and documents that it will
  *refuse to propose* if that horizon has no data

**Net effect: the improvement engine could never have proposed anything, and
never will, until this is fixed.** The entire self-improvement premise depends
on it.

The fix: make the dedup key account for which horizons are still unfilled, so a
date is reprocessed when it becomes old enough for the next horizon. Then
backfill - **11 of the 13 existing snapshot dates are already >30 days old, so
their 1-month returns are computable from historical prices immediately.** Done
right, this clears the 8-observation gate in days rather than months.

Write a regression test that fails on the current behaviour. Record the fix in
`METHODOLOGY_CHANGELOG.md` - it changes what evidence the engine acts on.

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
4. **Sell-side workflow** - client-side watchlist/holdings, deterioration
   flags, review queue. Question 3 is currently unanswerable.
5. **Investor profile selector** - `.claude/plan/investor-profiles.md`.
   Reconcile with `presets.py` first. Note `contrib` is Balanced-only.
6. **Investment-club readiness** - can a student open this on a phone and
   understand what they're looking at?
7. **Test isolation** - remove the need for the `conftest.py` guard.
