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
`plan/backtest-v2.md`) and genuinely independent IC observations accrue
at roughly one a month, so demanding measured proof up front would freeze the
project for half a year. Measurement is how a change is *confirmed over time*,
not the gate it must pass to be made.

What this does **not** license: changing something because it seems better.
The bar is a written argument a sceptical reader could follow to its sources.
If you cannot produce that, spend the session finding it instead. **A night
that produces one well-sourced finding and no code is a good night.**

### How the owner directs this: `OWNER_FOCUS.md`

The owner is not reviewing PRs and does not write tickets. `OWNER_FOCUS.md` is
the one place he says what he wants, in plain English. **Read it before the
rotation, every session.** Open items there outrank the day's nominal focus.

Only two things outrank *it*: a stalled data loop and the ship gates. If you
defer an owner item for either, say so in `NIGHTLY_LOG.md` - an unmentioned
deferral is indistinguishable from an ignored request.

When you finish an item, move it to **Done** in that file with the date and
what shipped. If an item is a bad idea, do not quietly skip it: do the sound
part, leave it open, and write down the argument. He can then overrule you,
which is the point of him having a queue at all.

Items will usually be about the *product* - a surface that does not help, a
question the dashboard cannot answer. That is the half of this work only he can
see. Methodology stays research-led per the mandate above, unless he
specifically asks for a factor to be researched, which is a legitimate item.

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
   | Evidence base | `improvement/live_ic_history.csv` | **row count, newest date, and effective observations at `1m`, as three literal numbers** |
   | Priority 0 | below | fixed, or still top of the queue |

   The evidence-base line would have read "3 rows, newest 2026-02-22" on every
   session from February to 2026-08-21, while the data loop ran successfully
   every weekday. Nobody wrote it down, so nobody noticed it had stopped
   moving. **If those two numbers have not moved in three consecutive sessions,
   making them move is that session's work, whatever the rotation says.**

9. **You can edit your own instructions and plans - so keep them true.**
   `prompts/` and `plan/` are ordinary tracked directories, deliberately *not*
   under `.claude/`, because files there are blocked as sensitive and twice a
   session was unable to correct its own documentation: the 2026-08-21
   retrospective could not fix a rotation description it had just replaced, and
   the 2026-08-25 session could not update `plan/dashboard-inventory.md` after
   shipping the feature that made it wrong.

   So there is no longer an excuse for stale process docs. If you change the
   rotation, update `prompts/nightly.md`. If you change the dashboard, update
   `plan/dashboard-inventory.md` in the same session - the Tuesday focus tells
   the next session to trust that file, and a wrong inventory sends it to
   rebuild something that already exists.

10. **Don't hand-edit generated files.** `dashboard.html`, `index.html`, and
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
runner swaps in `prompts/retrospective.md` automatically. That session
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
- `backtest.py` - decile backtest + IC validation. Known-weak; see `plan/backtest-v2.md`
- `presets.py` - weighting presets (balanced/value/growth/momentum)
- `config.yaml` - all tuneable parameters
- `schemas.py`, `cli.py`, `run_context.py`, `instrumentation.py`

### Front-end
- `generate_dashboard.py` - **source of truth**; writes `dashboard.html` + `dashboard_data.js`
- `dashboard_data.js` - `window.SCREENER_DATA`. `table_data` holds all ~500 stocks
  with all 8 category scores; `stock_detail` covers **all ~500 stocks** (raw
  values, percentiles, per-category `contrib` attribution, peers, price
  targets, financials, provenance) and is ~90% of the payload weight. Check
  `plan/dashboard-inventory.md` before building anything "new".

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
- `prompts/nightly.md`, `scripts/nightly-screener.ps1` (6:00 AM code loop)
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

**The evidence base grows again as of 2026-08-24.** `live_ic_history.csv` had
held 3 rows, all horizon `1w`, all February 2026, through every successful data
run for 183 days, because `record_run_snapshot()` never called
`compute_live_ic()`. It now does, for all three horizons, and the history is 23
rows across `1w`/`1m`/`3m`. See priority 0 above.

**Read the count that matters.** The gates read *effective* (non-overlapping)
observations, not rows. `analyze_ic_trends()` returns both: `_n_observations`
is the effective count, `_n_raw_observations` the row count. At the `1m`
optimization horizon there are currently **6 rows but 2 effective
observations**. Quote the effective number in the log; quoting the row count is
how this went wrong the first time.

Good work here: more/faster evidence, better IC estimation, regime handling,
smarter proposals. Every engine-applied change should also land in the
changelog.

## The dashboard is the product

Owner directive, 2026-08-05: **the dashboard should become the single place to
look when considering buying or selling a stock.** You have authority to add
and to delete features. Research before building.

**Read `plan/dashboard-inventory.md` before touching the dashboard.**
There is much more in it than a first look suggests - 501-stock detail
payloads, contribution attribution, sector peers, price targets, provenance,
and a very large embedded methodology document. The most likely failure mode is
rebuilding something that already exists. The governing plan is
`plan/dashboard-north-star.md`. In short: every element must help answer one of *what
should I look at / should I buy this / should I sell what I hold / how much*.
The dashboard is already strong at evaluating a single name; its real gaps are
the **absence of any time dimension** and the **absence of a sell-side
workflow**.

And the line that keeps it defensible: **decision support, not a
recommendation engine.** Show why, with sources and uncertainty visible. Never
emit a bare "buy". That distinction is what makes it appropriate for a college
investment club rather than a liability.

**That line has teeth, and it cost a feature.** The Model Portfolio panel was
removed 2026-08-26 because a fixed 25-name sector-capped list on a public site
is the closest this tool came to emitting a recommendation. It also turned out
to carry no column `table_data` did not already have, and no position weights -
so it did not answer "how much" either. The construction engine stays (the
Excel sheet, and `in_portfolio` feeds turnover in the snapshots); only the
dashboard surface went. Changelog 2026-08-26 (evening);
`tests/test_dashboard_surfaces.py`.

**Display-only fields are legitimate and must stay display-only.** The same
session added each stock's business description and industry to the drilldown,
because the tool could score a company on 44 metrics without telling a student
what it sold. They ride the `.info` dict the fetch already pulls, so they cost
no API calls, and a test asserts they never enter `raw`/`pct`. Prose is never
scored.

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

**On the 08-14 quota failure - correction, owner 2026-08-21.** The retrospective
recorded this as "~$6/session". That is wrong and should not be repeated: the
owner runs **Claude Max**, so sessions draw on included subscription usage, not
per-session billing. There is no dollar cost per run to optimise.

The finding underneath it stands: a **weekly usage ceiling** exists and it
silently killed 08-14 with a 429. What follows from that is not "spend less
money" but "do not let one loop exhaust the week". The 06:00 code loop is the
only thing here consuming that quota - the 02:00 data loop is pure Python and
git and uses none of it. So a data run is always affordable; a code session is
the scarce resource, which is an argument for sessions that do one thing well
rather than many shallow things.

Both of the items that used to sit here are now **DONE - do not undo them.**

- **The scheduled-task definitions are in version control.** Shipped
  2026-08-21 as `scripts/register-tasks.ps1`, which registers both tasks
  idempotently with an at-logon catch-up trigger. This entry went on claiming
  `grep -rn "Register-ScheduledTask" .` "finds nothing" for six days after the
  script landed; it finds it at `scripts/register-tasks.ps1:111`. Rule 9.
- **Something watches whether the loop is running.** Shipped 2026-08-27:
  `scripts/check_loop_health.py` plus `.github/workflows/loop-watchdog.yml`.
  43 tests in `tests/test_loop_watchdog.py`.

  **The point is where it runs.** `write_brief.py` already printed a "THE
  ROUTINE IS NOT RUNNING" banner and structurally could not cover this case:
  it is invoked only from `data-run.ps1` and `nightly-screener.ps1`, so when
  neither loop fires the brief is never regenerated and the banner never
  appears. The watchdog was living inside the thing it was watching. The new
  one runs on GitHub Actions, where a PC that is off, asleep or logged out
  cannot silence it, and opens a single reused issue when a loop goes quiet.

  The heartbeat is the commit each loop pushes to `main` from a `finally`
  block, so it lands whether the run succeeded or failed - `brief: data run
  <date>` and `brief: code session <date>`. A session that failed still
  counts: this answers "did the task fire", which is a different question from
  "did it do anything useful", and the brief already covers the second.

  **Do not make it alarm faster.** Two *consecutive* missed weekdays is the
  threshold, weekends are excluded, and a day is not judged until its deadline
  (12:00 data / 16:00 code) has passed - late enough that the at-logon
  catch-up has had its chance. Replayed against the real 08-17..08-20 outage
  it would have raised the alarm on **08-18**, two days in rather than six. A
  watchdog that cries wolf gets muted, and a muted watchdog is worse than none
  because it still looks like coverage.

Workspace trust (resolved 2026-08-13) regresses as: `python --version` works
but everything else is denied. Fix in `scripts/fix-trust.ps1`; the runner now
fails fast with instructions.

**0. DONE 2026-08-24 - do not weaken these.** All five steps shipped together,
plus a sixth defect found while fixing them. See `METHODOLOGY_CHANGELOG.md`
2026-08-24 and `tests/test_evidence_integrity.py` (30 tests; 24 of them fail
against the pre-fix code).

| # | Was | Now |
|---|---|---|
| 1 | A date was processed once at 7 days old and never revisited, so `fwd_return_1m` stayed `NaN` forever - and `optimization_horizon` is `'1m'` | Eligibility tracked per `(run_date, horizon)`; a date is reprocessed as it ages |
| 2 | Every snapshot file processed, so a day with 13 runs appended the same rows 13 times | One snapshot per run date; `_normalize_performance_history()` on every read and write |
| 3 | `t = IR * sqrt(raw row count)` | `_effective_observations()` - non-overlapping windows only; every gate reads it |
| 4 | Weekend run dates counted as separate observations from the adjacent Friday | Excluded at generation and at IC time |
| 5 | Nothing ever called `compute_live_ic()` | `record_run_snapshot()` calls it for all three horizons |
| 6 | Price cache keyed on the *current* date, so each revisited snapshot meant a fresh full-universe yfinance download | Fetch window bounded by the horizon being measured |

Measured effect: `performance_history.csv` 20,057 rows -> 5,528 (60% were
duplicates); `live_ic_history.csv` **3 rows -> 23**, newest 2026-02-22 ->
2026-08-14; observations at the `1m` optimization horizon **0 -> 6 raw, 2
effective**; `n_tickers` per IC row 1,006-6,539 -> 499-511.
`scripts/repair_evidence_base.py` performs the one-time repair and is
idempotent.

**What has NOT changed, and must not be quietly assumed away:**

- **`allow_auto_apply` stays `false`.** Condition (a) in the `config.yaml`
  comment (effective-observation counting) is now met. Condition (b) - a
  history with substantially more independent observations than the gate asks
  for - is **not**: there are **2**. Rule 4 stands.
- **The engine still cannot propose, and that is correct.** 2 effective
  1-month observations against a gate of 8.
- **Accrual is genuinely slow.** Independent 1-month observations arrive about
  one a month, so the 8-observation gate is roughly **six more months** of
  daily running. The old behaviour would have reached "8 observations" much
  sooner and been wrong. Do not engineer around this; say it plainly.

The `_effective_observations()` guard is now the load-bearing safety mechanism.
Against the pre-fix code, `propose_weight_changes()` returns `proposal_ready`
on eleven IC rows that are two independent observations - that is a failing
test now, not a hypothetical. Weakening it needs its own changelog entry and a
better argument than "it wasn't firing."

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
1.5. **DONE 2026-08-26 - and the 08-25 diagnosis was backwards.** Found
   2026-08-25 by the movers panel; root-caused and fixed 2026-08-26. Changelog
   2026-08-26; `tests/test_price_series_integrity.py`, 21 tests.

   The cause was not a transient failure. Yahoo's 13-month series for MNST
   **alternates between pre- and post-split prices** across its 2026-08-11 2:1
   split (94.46 / 47.08 / 90.36 on consecutive days), and `auto_adjust=False`
   returns byte-identical numbers, so no adjustment was ever applied. The
   pipeline divided an unadjusted July close (93.49) by an adjusted 2025 close
   (62.30) and got `return_12_1 = +0.50`, the 97th percentile, against a true
   split-adjusted **-0.25**, the 3rd percentile.

   **So 97.1 was the artifact and 2.9 was correct** - the reverse of what
   `NIGHTLY_LOG.md` 2026-08-25 and this entry originally said. MNST was live on
   the public site at momentum 71.5 / rank 360, roughly 110 ranks too high.

   Fixed at source: `factor_engine.check_price_series_integrity()` refuses a
   series that mixes two split scales and withholds the eight metrics derived
   from it, which the existing `has_data` renormalisation already handles.
   Verified against all 17 S&P 500 split events of the prior 13 months - one
   true positive, zero false positives.

   **Two things not to undo.** The 25% arming floor is measured (p99.9 of
   |daily return| is 17.2% over 137,313 ticker-days); below it a "split ratio"
   cannot be told apart from an ordinary down day, which is what keeps the
   small spin-off ratios (SPGI 1.057, HON 1.061) from flagging everything. And
   `check_run_health`'s `MIN_CATEGORY_COVERAGE = 0.90` bounds the blast radius:
   withholding one name in 502 is the mechanism working, withholding the
   universe is a feed change that must not publish.

   **The coherence finding worth carrying forward:** the eight categories are
   not eight independent bets. Momentum and risk are **23% of composite weight
   and 100% derived from one `Ticker.history()` call** - momentum's only
   non-price metric, `proximity_52w_high`, carries weight 0, so a rejected
   series costs a stock two entire categories.

   **FCX, the other case cited on 08-25, was never a bug.** Its growth score
   moved 68.3 -> 42.5 -> 68.3 because on 08-24 `forward_eps_growth` and
   `peg_ratio` were genuinely NaN and growth correctly renormalised over the
   remaining three metrics. Do not go looking for a defect there. What it does
   expose is a *product* gap for a Tuesday: the movers panel cannot distinguish
   "moved on new information" from "moved because two inputs went missing",
   even though `Composite_Confidence` already carries that fact.

2. **Give the dashboard a time dimension.** **DONE 2026-08-25** - shipped as
   `history.py` plus three surfaces: a "What Changed" movers panel, a sortable
   Δ column in the universe table, and a per-stock "Rank History" block.
   Changelog 2026-08-25; `tests/test_history.py`, 31 tests.

   **Do not weaken the comparability gate.** Runs enter the history only if
   their ranking correlates with the last accepted run at Spearman >= 0.50.
   `2026-07-28` is a degraded run in `improvement/snapshots/` that correlates
   with its neighbours at 0.016 and -0.020; ungated, it reports 82% of the
   universe as material movers. A regression test fails if it rejoins the
   series. Note also that reusing `check_run_health`'s dispersion rule here was
   tried and **excluded 16 of 20 real runs** - it is the right gate at publish
   time and the wrong one for comparing runs. Reasons in the changelog.

   **What is still missing:** per-category trend lines over the full history
   (only the two comparison baselines carry category deltas today, to keep the
   payload at +8%), and time-series valuation percentiles (north-star gap 3),
   which need the same spine and are now cheap to build.
3. **Backtest v2** - the current one has survivorship bias and holds
   fundamentals constant (look-ahead). It cannot honestly validate a
   methodology change, and now that the system validates *itself*, that bias
   steers the learning loop. `plan/backtest-v2.md`.
4. **Replace the AI chat with generated per-stock summaries** (owner directive
   2026-08-10). The chat needs each visitor to paste their own Anthropic API
   key, which is unusable for an investment club and un-reproducible. Build the
   deterministic template version first - the payload already has `contrib`,
   `pct`, `peers` and price targets, so summaries can be exact, free, identical
   for every viewer, and impossible to hallucinate. Explains *why it ranks
   there*, never *whether to buy*. Plus a run-level overview of what changed.
   See `plan/dashboard-north-star.md`.
5. **Sell-side workflow** - client-side watchlist/holdings, deterioration
   flags, review queue. Question 3 is currently unanswerable.
6. **Investor profile selector** - `plan/investor-profiles.md`.
   Reconcile with `presets.py` first. Note `contrib` is Balanced-only.
7. **Investment-club readiness** - can a student open this on a phone and
   understand what they're looking at?
8. **Test isolation** - remove the need for the `conftest.py` guard.
