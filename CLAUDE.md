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
owner is not reviewing PRs. Nobody reads your work before it is public - you
push a branch and the runner merges it, gates permitting (see "Ship gates").

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

   **Every entry starts with these five health numbers**, whatever the focus:

   | Check | Where | Healthy |
   |---|---|---|
   | Did the last code session actually run? | newest `logs/nightly-*.log` | ends "shipped to main" or "no changes" - **not** "SESSION DID NOT RUN" |
   | Did the data loop publish? | newest `logs/datarun-*.log` | ends "Data loop complete", HEALTH: PASS |
   | Evidence base | `improvement/live_ic_history.csv` | **at horizon `1m` only: row count, newest `run_date`, and effective observations - three literal numbers** |
   | Priority 0 | below | fixed, or still top of the queue |
   | Top open roadmap item | "Current priorities" below | **name it and give its age in days, as a literal number** |

   The evidence-base line would have read "3 rows, newest 2026-02-22" on every
   session from February to 2026-08-21, while the data loop ran successfully
   every weekday. Nobody wrote it down, so nobody noticed it had stopped
   moving.

   **Read it at `1m` and nowhere else** - corrected 2026-09-18, because the
   tripwire had quietly stopped being able to fire. Sessions were reporting the
   whole file: "44 rows, newest 2026-09-10". Both numbers rise every single
   weekday, because the `1w` horizon gains a row per run date as it ages. But
   `1m` is the optimization horizon and the one the engine's gate reads, and on
   2026-09-18 its newest `run_date` was **2026-08-14** - unmoved for six
   consecutive sessions, every one of which logged a newer date. A tripwire
   wired to a number that cannot stand still is decoration.

   **The condition that replaces "has it moved":** the newest `1m` `run_date`
   must be **within 40 days of today**. In steady state the lag is 30-33 days -
   the 30-day horizon plus a weekend - so 40 tolerates a week of missed runs
   and fires on anything worse. It reads 35 days today, which is the
   2026-08-15..08-19 outage still working through the pipe, not a fault.
   **If it exceeds 40, finding out why is that session's work, whatever the
   rotation says.**

   The roadmap line is there for the same reason, added 2026-09-04. Between
   2026-08-25 and 2026-09-03 nine sessions all produced real work and **not one
   north-star item shipped** - seven correctness fixes, a research note and two
   pieces of infrastructure, every one of them defensible on the day. Priority 4
   is an owner directive from 2026-08-10 and nobody had written down how long it
   had been sitting there. Defect-fixing will always look more urgent than
   product work; writing the age down is what makes the trade visible at the
   moment it is being made.

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

   **`SCREENER_OVERVIEW.md` is also generated** - added 2026-09-28, because
   this list not naming it cost the project a live regression. It is written by
   `build_screener_overview()` in `run_screener.py` (step 11) on **every full
   run**, so the 02:00 data loop overwrites it five mornings a week. On
   2026-09-25 a session corrected four false statements on the public
   methodology page by editing the markdown, shipped 14 passing tests that read
   the markdown, and the 2026-09-28 data run reverted all four and published
   them - including a fetch-failure rate the same session had measured at **0 of
   9,036** and the page put at "10-25% of tickers". Edit the generator, then
   regenerate.

   **Assert claims against the generator's output, not the committed file.**
   That is the general lesson and it is what made the 09-25 tests unable to see
   this: a test reading a generated artifact passes on a hand-edit and says
   nothing about what the next run will publish.
   `tests/test_overview_is_generated.py` does it the other way round and
   includes a tripwire that fails if the committed file drifts from the
   generator at all.

11. **Finish what you find - apply the fix and verify it, don't leave a
    command for the owner to run.** Owner direction, 2026-08-29: the routine
    is supposed to be self-improving, which means machine-state changes are
    the session's job too, not just repo changes. If a fix needs something
    Task Scheduler or Windows has to apply - re-registering a task, a config
    change outside git - do it in the same session and confirm it took effect
    the same way you'd verify any other change, before you finish.

    The 2026-08-29 session found and fixed the logon-trigger collision, then
    stopped short: it registered a stagger in the versioned script but left
    `powershell -ExecutionPolicy Bypass -File scripts\register-tasks.ps1`
    for the owner to run by hand, because `register-tasks.ps1` unregisters
    both tasks before re-adding them and a failure partway through would have
    left the machine with neither. That caution was reasonable but the
    conclusion was wrong: run it and immediately re-read both tasks' triggers
    with `Get-ScheduledTask`/`Get-ScheduledTaskInfo` to confirm the change
    took, the same as re-running the test suite after a code change. It is
    idempotent, so a bad outcome is fixed by running it again, not by asking
    someone else to.

    The only time it is legitimate to leave something outstanding is when
    verifying it genuinely requires something outside the session's reach -
    and even then, the next scheduled session inherits it, not the owner.

## Ship gates

Run these before you finish. All must pass.

**You push your branch. `nightly-screener.ps1` merges to `main`, and only after
re-running all four gates itself** (changed by the 2026-09-04 retrospective).
Until then the session merged and pushed `main` as its own last step and the
runner re-checked *afterwards*, which made the gates an audit rather than a
precondition: on 2026-09-02 a session published at 06:24:22, the independent
test run failed two minutes later, and `scripts/revert-bad-merge.ps1` had to be
written to undo a commit already live on the public site. That script stays as
the second line of defence. It should now never have to fire.

Still run the gates yourself - they tell you whether the work is fit to ship,
and a session that hands the runner a broken tree has wasted the day.
`tests/test_gate_ordering.py`, 8 tests.

| # | Gate | Command |
|---|---|---|
| 1 | Full suite passes, no new failures vs the baseline you took at session start | `python -m pytest tests/ test_screener.py -q` |
| 2 | Pipeline wiring intact | `python run_screener.py --dry-run` |
| 3 | Dashboard artifacts intact | `index.html` non-trivial and `dashboard_data.js` parses (`node --check`) |
| 4 | No stray uncommitted files | `git status --porcelain` |

On success the runner tags the commit `good/YYYY-MM-DD`. That tag is the
rollback point - see `ROLLBACK.md`.

**Gate 3 became a real parse on 2026-09-18, on both publish paths** - it had
claimed "`dashboard_data.js` parses" while only regex-matching the first line
against a size floor, and a payload truncated to 50% is 2.5 MB, opens with the
expected assignment, clears every check both loops had, and renders a blank page.
`tests/test_payload_parse_gate.py`, 16 tests.

### The data loop's gates, and the asymmetry to keep checking

**The four gates above guard the code loop. The data loop publishes to the live
site five mornings a week and has its own, weaker, set** - and twice now a
retrospective has found the real hole there rather than in the code loop, for the
same structural reason: *the strongest checks guard the path that publishes least
often.* When you compare the two runners, **the weaker one is usually the bug.**

`data-run.ps1` refuses to publish on: synthetic data, a fetch-failure rate over
40%, `check_run_health.py` reporting degraded, dashboard generation failing, a
size floor, `node --check`, the payload's opening assignment, and
`scripts/check_published_claims.py`.

**That last one closed gate 1's missing counterpart, 2026-10-02.** The data loop
ran **no tests at all** while regenerating and publishing `SCREENER_OVERVIEW.md`,
the public methodology page. On 2026-09-28 it reverted four corrections shipped
three days earlier and published them - including a fetch-failure rate the same
session had measured at 0 of 9,036 and the page put at "10-25% of tickers" - and
left the tree red, so the next session's gate 1 failed at baseline. Measured at
that exact commit: `pytest tests/test_overview_claims.py` reports **12 failed, 2
passed in 0.36s.** The evidence to refuse the publish was already committed; the
loop publishing never asked for it.

The check is **deliberately narrow** - the modules that speak for the artifacts
that run rewrote, 237 tests in ~15s against the full suite's 124s. Running
everything would mean one unrelated red test stops the *evidence* loop as well as
the code loop, and a red `main` already blocks merges. A failure **discards the
run**, like every gate above it, so the live site keeps the last good version.
Where pytest cannot run at all it **warns and publishes** - the `node` fallback
precedent, because a gate that can only fail closed jams an unattended loop.
`tests/test_published_claims_gate.py`, 22 tests.

## The weekly cycle

Research-weighted by design: understand before building, and prove it after.

| Day | Focus | Output |
|---|---|---|
| **Mon** | **Research.** One specific thing - a factor, a metric, a threshold, a construction rule - learned properly, from *both* the literature and documented practice. Where they disagree, say so and say why. | A dated note in `research/`, complete in one session: citations with effect sizes, plus how practitioners actually do it |
| **Tue** | **Product.** Open the live dashboard as a user would. Does it answer *what should I look at / should I buy this / should I sell what I hold / how much*? Fix or build what it can't. | A dashboard change, or a written account of what it cannot answer and why |
| **Wed** | **Synthesis.** How does Monday's research fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. | A design section on Monday's note, plus any `METHODOLOGY_CHANGELOG.md` entry |
| **Thu** | **Build.** Implement what the week justified - or, if it justified no change, the top open item in "Current priorities". | Working, tested code |
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
- `factor_engine.py` - metric registry (`METRIC_COLS`, 46 entries), scoring
- `portfolio_constructor.py` - sector-constrained portfolio construction
- `improvement_engine.py` - **the methodology learning loop** (see below)
- `backtest.py` - decile backtest + IC validation. Known-weak; see `plan/backtest-v2.md`
- `universe_history.py` - point-in-time S&P 500 membership (2026-09-24). Built, **not wired into `backtest.py`** on purpose
- `lookahead.py` - **diagnostic**: classifies every weighted metric by what making
  it point-in-time would cost, and restates the price-dependent ones at a
  historical price. Measures `backtest.py`'s look-ahead bias; **must never be
  imported by production code** (2026-10-01, tests enforce it)
- `presets.py` - weighting presets (balanced/value/growth/momentum)
- `config.yaml` - all tuneable parameters
- `schemas.py`, `cli.py`, `run_context.py`, `instrumentation.py`

### Front-end
- `generate_dashboard.py` - **source of truth**; writes `dashboard.html` + `dashboard_data.js`
- `stock_summary.py` - the deterministic "Why it ranks here" sentences, built at
  run time into `stock_detail[t]["summary"]`. Never advises; see priority 4 below
- `dashboard_data.js` - `window.SCREENER_DATA`. `table_data` holds all ~500 stocks
  with all 8 category scores; `stock_detail` covers **all ~500 stocks** (raw
  values, percentiles, per-category `contrib` attribution, peers, price
  targets, financials, provenance) and is ~90% of the payload weight. Check
  `plan/dashboard-inventory.md` before building anything "new".

### Docs (public-facing - keep truthful)
- `SCREENER_OVERVIEW.md` - canonical methodology reference. **Generated** by
  `build_screener_overview()` in `run_screener.py`, rewritten every full run -
  edit the generator, never this file (rule 10)
- `METHODOLOGY_CHANGELOG.md` - every methodology change, with evidence
- `Multi-Factor-Screener-Blueprint.md`, `SCREENER_DEFENSIBILITY_SPEC.md`, `README.md`

### Internal record
- `DECISIONS.md` - the full reasoning behind every **settled** decision, moved out
  of this file 2026-10-02. "Current priorities" keeps the constraint and the test
  that enforces it; the archaeology lives there. Read the entry before you
  weaken a row.

### Tests
- `tests/` (70+ modules) + `test_screener.py`. Run: `python -m pytest tests/ test_screener.py -q`
- `conftest.py` at root is a **tripwire**: it restores published artifacts and fails the
  session if any test wrote one (isolation landed 2026-10-09; priority 8).

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
**Repo growth was measured 2026-09-18 and is a non-issue - do not "fix" it.**
This paragraph used to warn that a ~3 MB payload changing every run adds
"roughly 60 MB/month of poorly delta-compressing JSON", and suggested
downsampling the payload or committing data less often. Nobody had checked it
in the six weeks it stood. Measured: 40 versions of `dashboard_data.js`
totalling **151 MB raw** cost **14.3 MB in-pack** - **0.36 MB per version**,
about a 10x delta ratio - and the entire repository packs to **34.6 MiB**. At
21 weekday runs that is **~7.6 MB/month, not 60.** Acting on the old number
would have cut the payload a reader depends on to save nothing.

What *was* real: git had never repacked, so 1,301 loose objects and 4 packs
occupied **99.5 MiB** on disk against 34.6 MiB of content. Loose objects carry
no delta and git's own `gc.auto` threshold of 6,700 was most of a year away at
~30 objects a run. `data-run.ps1` now runs a non-fatal `git gc --auto` at
`gc.auto=200` after publishing. Re-measure with `git count-objects -vH` before
believing any future claim here. This is what accumulates the evidence:
forward returns, live ICs, dispersion history. Without it, methodology can
never learn.

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

**How it looks and feels is part of whether it is correct** - owner direction,
2026-10-05: *"It looks like AI slop... make it look premium and expensive... it
doesn't feel good to move around in it either."* Measured that day: GitHub's dark
palette verbatim, a hue per category, ~2,560 elements set in monospace, 12px as the
dominant size, 502 table rows in a nested scroll box, ~9,900 DOM nodes. The full
brief is the open item in `OWNER_FOCUS.md`. The standing rules that outlive it:
**look at the live page at desktop and 375px before and after any dashboard change
and say what you saw** (judging from the CSS is how it got here); colour is for
meaning, not decoration; every figure uses tabular numerals; and presentation work
must leave every number, rank and sentence byte-identical, which you verify by
diffing the rebuilt payload against the live one.

**Every number on the page must be checkable from the page** - owner direction,
2026-10-06: *"we can see how things score, but we don't actually see the numbers
going into any calculations... This will add trust to the screener."* Writing the plan
for it turned up three defects on the live site (`plan/calculation-transparency.md`):
the drilldown printed the generic metric weight for **276 of 502 stocks** where the
engine had used bank, Piotroski-conditional or renormalised weights (so JPM's
Valuation panel showed a score no arithmetic on screen produces); the composite line
omitted the coverage discount (2 stocks); and the first sentence of every drilldown
called the composite a percentile ("scores above 74%" for the stock ranked **1st**),
wrong by a median of 19.6 points, because the composite has been cardinal since
Phase 13. **All three were fixed on 2026-10-07** (the sentence by the 06:00 session; the weights and the composite line by an owner-run session). The standing rules that outlive the plan: **the page shows the engine's
numbers and never its own re-derivation of them** (one weight-resolution function
feeds both scoring and display); **a sentence that says how a number is computed
needs a registered check against the code that makes it true**; and **a build that
cannot reproduce its own published scores from its own payload does not publish.**

## Current priorities (rewrite this section as things land)

**Restructured 2026-10-02.** This section had reached **551 lines - 52% of this
file** - and most of it was narrative about work already finished, sitting in
front of the few items actually open. The reasoning moved verbatim to
`DECISIONS.md`. What stays here is the **constraint** each settled decision
imposes and the test module that enforces it, because a session reads this file
before it starts work and the constraint is what it needs in front of it.

**Before you undo, weaken or "simplify" anything in the table below, read its
`DECISIONS.md` entry.** Every row is there because the failure it prevents
already shipped to the live public site.

### Settled - do not weaken

| # | The constraint | Enforced by |
|---|---|---|
| -1 | Scheduled-task definitions stay in version control (`scripts/register-tasks.ps1`). The two loops keep **different** logon delays (data `PT3M`, code `PT20M`), and the loser of the shared repo lock **waits** - it must never exit instead. The third task, `Screener Option Quotes` (weekdays 20:00), **publishes nothing**: no repo lock, no mutating git command, writes only the gitignored `data/options/`, no logon trigger - that is what makes a third task safe without gates of its own | `tests/test_loop_mutual_exclusion.py`, `tests/test_option_quote_refresh_task.py` |
| -1 | The watchdog runs **outside** the machine it watches, and must not alarm faster than two *consecutive* missed weekdays. A watchdog that cries wolf gets muted, and a muted watchdog still looks like coverage | `tests/test_loop_watchdog.py` |
| -1 | The morning brief is published as a single-file commit built on `origin/main`. **Never go back to pushing a local ref** - from `finally`, that publishes work the gates just refused | `tests/test_brief_publish_safety.py` |
| -1 | The `nightly/*` branch sweep keeps its `--merged origin/main` filter, and a failed delete stays non-fatal | `tests/test_branch_sweep.py` |
| 0 | `_effective_observations()` gates every proposal; `allow_auto_apply` stays **`false`**; quote `_n_observations`, never `_n_raw_observations`. Independent 1-month observations accrue about **one a month** - say that plainly rather than engineering around it | `tests/test_evidence_integrity.py` |
| 0.5/0.7 | `check_run_health.py` discards a run **before publishing** on: missing fetch evidence, price coverage <90%, analyst-target coverage <50%, or category dispersion >20% below the trailing median | `tests/test_run_health.py`, `tests/test_cache_freshness.py` |
| 1 | The pipeline **never fabricates data**. Both entry points refuse without `--allow-synthetic`. Metric-drift severity is scored against the population a metric applies to, not the whole universe | `tests/test_no_synthetic_by_default.py`, `tests/test_metric_drift_scoping.py` |
| 1.5 | The split-scale guard keeps its **measured** 25% arming floor, and `MIN_CATEGORY_COVERAGE = 0.90` bounds the blast radius. Input churn arms at **>= 2** metrics, is worded as a caveat and never as deterioration, and compares availability **sets** | `tests/test_price_series_integrity.py`, `tests/test_input_churn.py` |
| 2 | A run enters the rank history only at Spearman **>= 0.50** against the last accepted run. Do **not** reuse `check_run_health`'s dispersion rule here - it excluded 16 of 20 real runs | `tests/test_history.py` |
| 4 | Per-stock summaries are baked **at build time**, advice language is blocked by `BANNED_TERMS` / `advice_terms_in()`, and metric percentiles are labelled **sector**-relative because that is what they are | `tests/test_stock_summary.py`, `tests/test_ai_chat_removed.py` |
| 5 | My Holdings renders **every** saved name every time, ordered **by rank and never by size of move**, and stores **no cost basis, share count or P&L**. The review cadence is read from the **run's own** config snapshot and stated, never enforced | `tests/test_holdings_panel.py`, `tests/test_review_cadence.py` |
| 0.8 | **The page shows the engine's numbers and refuses to publish if they do not add up.** Metric weights are resolved in one place (`factor_engine.metric_weight_profiles`) and published, never re-derived; `generate_dashboard` rebuilds every score from its own payload (`calc_trace`) and raises `CalculationMismatch`; an equation is shown only if it reproduces (`metric_lineage.RECOMPUTE`); `scoring_schema` keys the scored cache. Do not widen the tolerances to pass a build | `tests/test_calculation_reproducibility.py`, `tests/test_metric_lineage.py`, `tests/test_dashboard_browser.py` |
| 0.8c | **A scoring change ships with its frontend in the same commit** (owner, 2026-10-07). Opening any metric row shows the stock's figures through the formula (`metric_lineage.EQUATIONS`), and an unscored metric says why (`NOT_USED_BECAUSE`); the suite evaluates each exact equation against every stock's scored value at a 99% bar, and a weighted metric with no line fails. Change a formula -> update `LINEAGE`, `RECOMPUTE` and `EQUATIONS` together. Never satisfy it by loosening the bar or marking a template `exact=False` | `tests/test_metric_lineage.py` (equation tests), `tests/test_dashboard_navigation.py::test_every_metric_row_shows_its_own_numbers`; `prompts/nightly.md` section 3 |
| ctx | **Context is shown beside the score, never in it** (owner, 2026-10-08: "context only, track them"). Trend, options, insider, rate-sensitivity, macro, own-history valuation and track-record data live in `context_signals` / `context_fetch` / `insider_activity` / `market_context` / `valuation_history` / `track_record` and the payload's `ctx`, `market`, `track`; none may reach `raw`, `pct`, a category score or the composite, and no scoring module may read a `_ctx_` field. Option and insider requests run in a separate, time-budgeted pass **after** the core fetch (inside it they tripped the rate limiter). Every signal is logged per run in `data/context_log/`; one joins the score only with a research note **and** a measured `1m` record meeting the engine's own observation gate, then a changelog entry. The track record lists no current holdings (the Model Portfolio decision) and is not methodology evidence (rule 4) | `tests/test_context_layer.py`; `plan/context-layer.md` |
| gates | Both publish paths parse the payload (`node --check`) and check its opening assignment; the data loop also verifies the **claims** in what it is about to publish (`scripts/check_published_claims.py`). Where node or pytest is absent both fall back with a `WARN` rather than jamming | `tests/test_payload_parse_gate.py`, `tests/test_published_claims_gate.py` |

**One constraint with no test, kept here because it governs how you spend the
session:** a weekly usage ceiling exists and silently killed the 2026-08-14
session with a 429. There is no per-session dollar cost to optimise - the owner
runs Claude Max, so this is included subscription usage (owner correction
2026-08-21; do not reintroduce the "~$6/session" figure). The 06:00 code loop is
the only thing drawing on that quota, so **a code session is the scarce resource
and a data run is always affordable.** That is the argument for one thing done
properly over three done shallowly.

### Open

**0.8. Calculation transparency and the premium redesign - CLOSED 2026-10-07.** Owner:
*"I actually decided that I want you to finish up all UI/UX work in here tonight. And make the nightly sessions just focus on what it was previously focusing on, making sure the methodology is sound. And improving this tool overall every single night."* Finished that night across five owner-run passes; what shipped is in
`plan/dashboard-inventory.md` and `NIGHTLY_LOG.md` 2026-10-07. **Nightly sessions do not take
on design or fine-tuning work**; the page's standing correctness rules (row 0.8 and 0.8c above)
still bind. The paragraph below is the record of how it got there.

*Record:* first version of every stage shipped 2026-10-07 by an owner-run session
(`OWNER_FOCUS.md`, two items). `plan/calculation-transparency.md` and
`plan/dashboard-redesign-master.md` each carry a **"Status, 2026-10-07 evening"** table of what
is built and, per stage, what the deep pass should do. Highlights: the page now prints the
weights the engine used; every metric opens to its formula, inputs and peer rank; **4,010/4,010
category scores and 502/502 composites rebuild from the payload and the build refuses to publish
otherwise**; `scripts/audit_stock.py --all` independently reproduces 502/502; DOM nodes
9,928 -> 2,779, transitioned elements 662 -> 99, row click to paint 70 ms, payload gzipped flat.
**Be critical in the deep pass:** a first version built in one sitting is not "perfect", and
the owner asked for perfect. Look at every surface at 1440 and 375px, say what you saw, and fix
what is still not premium. No scoring change; explanation and presentation only.
**Pass 2 shipped the same night** (owner: *"be creative"*): a search palette (Ctrl/Cmd+K),
`#stock=TICKER` links, J/K stepping, Compare with the composite gap decomposed (claim
`compare.composite_gap`), a workings CSV, Add to Holdings from the sheet, analytics rows that
filter the table, Peers without green/red verdicts, and Chart.js removed (nothing used it). **The
nightly fine-tuning list is `plan/dashboard-redesign-master.md` "Status, pass 2"** - it supersedes
the evening table where they overlap; start by testing what pass 2 shipped on the live site. The
navigation layer **wraps** `openStockDetail`/`closeModal` (`_js_ux()`); keep it that way, so every
way of opening a stock keeps links and stepping.

**0.9. Four methodology questions the transparency build surfaced - ANSWERED 2026-10-09
(owner-run).** Each has a research note or a documented-defect entry in
`METHODOLOGY_CHANGELOG.md` 2026-10-09: **(a)** `operating_leverage` to weight 0 (85 of its 95
negatives were margin squeezes ranked best; the literature finds higher operating leverage earns
*more*; MSCI and AQR quality do not use it) - `research/2026-10-09-operating-leverage.md`;
**(b)** `revenue_growth` and Piotroski signals 3/8/9 now compare periods exactly a year apart
(the old comparison spanned 12-23 months by fiscal calendar) -
`research/2026-10-09-revenue-growth-window.md`; **(c)** one EBITDA for both ratios and the
snapshot; **(d)** labels "6-1M Return" and "Max Drawdown (13M)". **Found the same day by the same lens:**
`forward_eps_growth` (45% of Growth) spanned 13-24 months and mixed GAAP with adjusted EPS; it is now
MSCI's 12-month forward construction (`research/2026-10-09-forward-eps-growth.md`). **Their measured effects on the
first live run are recorded under those entries by the 2026-10-09 evening run** - read them before
touching any of the four again.

**T0a's standing constraint:** `claims.py` registers every sentence that says how a
number is computed, with the code that makes it true and the test that checks it.
**A new `_sentence_*` in `stock_summary.py` fails the suite until it is registered**,
and `FORBIDDEN` keeps the four false statements fixed that day from being republished.
Defect 3 survived for months because **a test asserted it** - when you correct a
published claim, grep the tests as well as the prose. `tests/test_claims_register.py`,
24 tests, in the data loop's publish gate.

**T0b's fourth defect is FIXED** (the "N of 18" badge reads the engine's applicable coverage).
**Since 2026-10-09 that coverage counts only metrics that carry weight** (28 generic, 25 for
banks; `factor_engine.weighted_metric_sets`), so a weight-0 candidate cannot move a composite.
The *coverage filter* that excludes thin stocks reads the same weighted coverage (aligned the same
day; measured: no stock's inclusion changed).

**0.11. The context layer - first draft shipped 2026-10-08 (owner-run); the nightly sessions own
it now (opened 2026-10-08, age 1 day).** Technicals, options, insider trades, a market backdrop and
the ranking's track record, all context-only (settled row "ctx"). `OWNER_FOCUS.md` has the open item
and `plan/context-layer.md` the ordered queue. Queue items 1 and 3 are done; **item 2 was measured
and fixed in code 2026-10-09** - the options panel was empty for every stock on every scheduled run
(`ok` for **0 of 503** at the 02:00 loop's hour against **394 of 503** at 21:27 ET, because the
source serves the chain overnight with bid, ask and implied volatility all zero), fixed by an
after-close quote cache plus a probe instead of a guessed hour.

**`Screener Option Quotes` IS REGISTERED** (2026-10-09, owner-run; `Get-ScheduledTask` read back:
one weekly 20:00 trigger, no logon trigger; the loops still PT3M / PT20M). Smoke-tested live at
08:46 CT: quotes live, cache written. **Item 5, the evaluation harness, is built**
(`context_eval.py`, one line in the morning brief; first one-month window closes 2026-11-07).
Item 4 step one (every research sentence on the panel checked against its source) is done -
`research/2026-10-09-context-panel-claims.md`. Next: re-measure the options `ok` share from the
first 02:00 log after a 20:00 refresh, then item 4's per-signal notes.

**0.12. Metric audit residuals - research, do not patch (opened 2026-10-09, age 0 days).**
Two audit passes over every weighted metric fixed nine defects the same day
(`research/2026-10-09-metric-audit.md`, changelog "metric audit" entries). Four findings are
judgement calls and need a Monday-style note before any change: **(a)** which financials are
"bank-like" - 26 of 59 reach the bank metric set only by the default for an unlisted industry
(asset managers, insurance brokers, broker-dealers; TROW 100th percentile on equity ratio, AON 8th on
P/B); **(b) DONE 2026-10-09 - weight 0, points to the non-surprise metrics:** `earnings_acceleration` (20% of Revisions) correlated -0.54 with the prior quarter's
surprise, so it penalises a stock that beat last quarter (surprises autocorrelate - Bernard & Thomas
1990); **(c)** accruals counted three times within Quality (the metric, Piotroski 4, Beneish TATA);
**(d) DONE 2026-10-09:** net debt now nets cash and short-term investments, as enterprise value does. Also: the
momentum regime rule is **off** (its input did not measure volatility); rebuilding it on a real
volatility measure (Barroso & Santa-Clara 2015; Daniel & Moskowitz 2016) is its own research item.

**0.10. Keep the inputs the 11 history-based metrics need, at fetch (opened 2026-10-07, age 0
days).** A data change, not design - the residual of the transparency work. Every metric's
calculation is one click away on the page, but for volatility, beta, drawdown, Jensen's alpha,
Sharpe/Sortino and the four analyst-history metrics the page can only describe the inputs
(`metric_lineage.SOURCES`), because the run does not keep them: the per-quarter EPS actuals and
estimates, the risk-free rate and the market's 12-month return. Keep them in the raw fetch,
publish them as inputs, and give each an `EQUATIONS` entry (exact where the arithmetic allows) -
the equation tests then hold them to the engine like the other 25.

*Measured 2026-10-08, re-measure rather than trusting it
(`research/measurements/2026-10-08-equation-coverage.py`): of 29 weighted metrics **15 carry an
exact equation (58.0% of composite weight), 5 an inexact one (19.5%) and 9 none at all
(22.6%)**. Of that 22.6%, Piotroski (3.30%) and Beneish (1.54%) already list their signals and
indices, so the genuine gap is the seven history-based metrics - 17.8%.*

**`max_drawdown_1y` is done (2026-10-08) and is the pattern to copy:** the two closes the
fall was measured between are computed **where the metric is computed**, stored as
`_mdd_peak` / `_mdd_trough`, published through `ENGINE_KEYS`, and divided on the row. One
computation, published - never a second one in the generator or in JS (row 0.8). Writing that
equation is also what exposed the log-return compounding error, which is the argument for
doing the rest: *you cannot write a metric's arithmetic down without checking it.*

**(a)-(d) ALL BUILT 2026-10-09 (owner-run)** and smoke-tested live (AAPL, JPM, NVDA reproduce to
4dp): Jensen's alpha prints its four CAPM terms, beta its covariance / variance, volatility its daily
standard deviation x 252^0.5 (all engine-published `_ja_*`, `_beta_cov/_var`, `_vol_sd`), and the
three surprise metrics show their four quarters (`eq4`) with a test rebuilding each from them.
`scoring_schema` is 4. **What remains is Sharpe and Sortino** (weight 0, shown for reference) - low
value; do them only if a session has nothing better.

**0.6. One observation per run date - CLOSED 2026-10-08.** Open since 2026-08-11. The
item asked for deduplication and a test, and both landed: `record_dispersion` now
**replaces** a date's row instead of appending, and `record_run_snapshot` removes
superseded files for its date - which is the rule `compute_forward_returns` already
applied when reading them, so the three now agree. Measured before the fix:
`dispersion_history.csv` held **52 rows for 45 distinct dates** (2026-04-14 four
times) and `improvement/snapshots` **79 files for 52 dates**;
`scripts/repair_one_observation_per_date.py` collapsed both, and
`live_ic_history.csv` / `performance_history.csv` came out byte-identical, so this
removed a latent distortion of `check_run_health`'s trailing dispersion median
rather than correcting a live one.

**A third file had the same defect and was found the same day, by the session's own rescoring
run adding a row to it: `factor_vol_history.csv`, 72 rows for 52 dates, 2026-02-21 appearing
nine times.** That one feeds **scoring** - `adjust_momentum_weight` ranks the run's momentum
dispersion against every row and the percentile sets the momentum weight. Deduplicating moved
the distribution (p75 26.59 -> 26.27) without flipping the regime, so the published run was
unaffected; it was capable of it. The constraint that stays: **every append-once-per-run file
holds one row per run date** - `dispersion_history.csv`, `improvement/snapshots/` and
`factor_vol_history.csv` - with a tripwire on the real files.
`tests/test_one_observation_per_run_date.py`, 11 tests. A fourth such file gets its tripwire
in the same module.

**3. Backtest v2 - step 1 is DONE on both halves; step 3 is next.**
`plan/backtest-v2.md` governs. The two measurements, both reproducible from
committed inputs:

| Bias | Size, on the name-month panel | Source |
|---|---|---|
| Look-ahead | **>= 63.2%** (30.3% move >= 2 deciles; only 59.5% of v1's top decile belongs there) | `research/2026-10-01-lookahead-bias-size.md` |
| Survivorship | **11.4%**; 4.89% would remain after a free-data v2 | `research/2026-09-30-delisted-price-source-cost.md` |

Look-ahead is **5.5x bigger on the same unit**, which reversed the plan's own
sequencing: **point-in-time fundamentals now outranks the point-in-time
universe.** `lookahead.weight_buckets()` derives the decomposition from
`config.yaml` so it cannot go stale - 16.9% of composite weight is honestly
recomputed per rebalance, **34.1 points of the 83.1 held-constant points are free
to fix** (28.0 restated exactly by one month-end price, 6.1 recomputed from a
price history), and 49.0 genuinely needs point-in-time filings and estimates.
*(Re-derived 2026-10-09 after the Revisions reweight: 34.45 free = 28.35 + 6.1; 48.65 needs
point-in-time data. `lookahead.weight_buckets(cfg)` is the live figure.)*

*Next:* the 34.1 free points first. Then size the 49.0pp fundamentals half via
SEC EDGAR's XBRL `companyconcept` endpoint, which is free and carries a `filed`
date per fact; expect the 9.0pp of analyst-estimate metrics to have no free
retrospective source and to be reportable only as permanently unmeasurable.
**Do not buy the $199/yr delisted-price feed first** - spending it to cut an
11.4% bias while a >= 63.2% one is untouched buys nothing a reader can use.

*Four things not to undo:* `lookahead.py` is a **diagnostic** and must never be
imported by production code (a half-fixed backtest is what the plan forbids);
`price_target_upside` is rebuilt from `pt_mean`, never by inverting the clamped
metric; enterprise value is restated as `ev + mc*(r-1)`, never `ev*r`; and
`universe_history.validate_membership` refuses a universe outside 495-515.
Availability of a delisted price must be tested by **coverage of the months the
caller will read**, never by row count - `INFO`'s series now begins in October
2024, two years after IHS Markit was absorbed. `tests/test_lookahead.py` (43),
`tests/test_universe_history.py` (59).

*Two defects in `backtest.py`, recorded and deliberately not patched* (same
reason): `dynamic_cols` recomputes **four** metrics where momentum and risk carry
**six**, so `jensens_alpha` and `max_drawdown_1y` are frozen; and v1 omits
`adjust_momentum_weight()`, so it backtests a weighting the site does not
publish. v2 must apply it **per rebalance month from that month's regime** -
reading the current run's regime is a third look-ahead vector, inside the weights
rather than the metrics.

**4's residual - DONE 2026-10-09 (owner-run).** The What Changed panel opens with a run-level
overview (`run_overview.py`, claim `history.run_overview`): who moved past the noise threshold,
which categories the movement came from (weighted, direction-free), how many of the top 25 held. Its
pair, Reporting Soon, shipped 2026-10-08. The 2026-08-10 owner directive's pieces are all built.

**5's residual: the hold band - settled 2026-09-16 as "not yet", with a date.**
The strict top-25 rule wastes **31-47%** of the trades it implies at every
cadence measured, so a band is warranted; its **width is not determinable** yet -
at 1.4x three cadences report 0.0%, 31.2% and 5.9% wasted. **The pre-registered
rule binds:** do not commit to a width until there are **>= 8 disjoint
observation windows at the review cadence the band will govern.** Today **2
monthly**; roughly **2027-04**. Then take the narrowest band clearing half the
strict rule's waste and Novy-Marx & Velikov's 50% turnover bound. Do **not** pick
50 because MSCI doubles, and do **not** reuse the movers panel's threshold (it
fires for a top-25 name 0.15% of the time). `review_cadence` is a **sentence, not
a lock** - do not gate, hide or delay a number behind it.
`research/2026-09-14-sell-discipline-and-hold-bands.md`.

**One measurement to carry forward:** the largest one-month category move is Risk
34%, Revisions 29%, Momentum 26% and **Quality 0.2% - one stock in 500**. A
deterioration trigger keyed to the fundamentals categories would essentially
never fire at monthly cadence.

**2's residual:** per-category trend lines over the full history (only the two
comparison baselines carry category deltas today, to keep the payload at +8%).
Time-series valuation (north-star gap 3) **SHIPPED 2026-10-09** as context:
`valuation_history.py`, "Against its own five years" in Before you decide.

**1's standing instruction:** the fetch-failure rate is **measured, not
assumed** - 0 across 9,036 ticker-fetches over 18 logs to 2026-09-25, against a
"10-25%" figure that was stale for months. Re-verify periodically and update the
record here in either direction.

**6. Investor profile selector - M1-M3 SHIPPED 2026-10-09 (owner-run).** A Weighting selector on
the rankings table; profiles from `presets.py`, composites computed at build time by the engine with
the run's own regime (`apply_momentum_regime`), withheld unless Balanced reproduces the published
ranking. M4 (custom sliders) is open and must first answer whether any client-side reweight can meet
row 0.8. `plan/investor-profiles.md`.

**7. Investment-club readiness** - can a student open this on a phone and
understand what they're looking at?

**8. Test isolation - DONE 2026-10-09 (owner-run).** A per-test write detector found exactly three
tests writing published files; all three now use `tmp_path` or the committed universe, and
`conftest.py` is a tripwire that restores the files and **fails the session** if any test writes
one.