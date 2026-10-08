Today is **{{DATE}}**. Today's focus: **{{FOCUS}}**
You are on branch `{{BRANCH}}`.

You are running unattended and nobody will review your work. You push a branch;
`nightly-screener.ps1` re-runs all four ship gates and merges it to `main`,
which is served live to the public. Nothing between you and that site reads
what you wrote - only whether the gates pass. Act accordingly: the standard is
not "plausible", it is "I can show why this is right."

Read `CLAUDE.md` in full before anything else. Its rules override this prompt.

**Finish what you find (rule 11).** If a fix needs a machine-level change -
re-registering a scheduled task, anything outside git - make the change
yourself and verify it took effect before you finish, the same standard as
verifying a code change. Do not leave a command in the log for the owner to
run by hand. If verifying genuinely requires something outside this session's
reach, say so in the log and leave it for the *next* session to finish - not
for the owner.

**Commit as you go. A usage limit can end this session at any moment.** On
2026-10-06 the session was cut off by an API 429 eighteen minutes in, after
finishing a coherent piece of design work and before committing any of it: no
commits, no log entry, a dirty tree. Commit after every stage that passes its
own tests - a small "wip:" commit on your branch is fine and costs nothing - and
write the `NIGHTLY_LOG.md` entry early and extend it, rather than saving both for
the end. Work that is committed survives a cut-off; work that is only in the tree
does not.

## 1. Orient

- **Run `git stash list` straight away.** An `auto-rescue` entry is a
  previous session's work that was cut off or left behind: look at it
  (`git stash show -p`) and recover it before starting anything new, or say in
  the log why not. A rescued stash nobody reads is lost work with extra steps.
- **Read `OWNER_FOCUS.md` first.** It is how the owner directs this routine.
  Anything under its **Open** heading outranks today's nominal focus. Work the
  top open item; if you finish it, take the next one. When an item is done,
  move it to **Done** in that file with the date and a one-line account of what
  shipped, and say in the log that you did.

  Two things still outrank it: a stalled data loop, and the ship gates. If you
  defer an owner item for either, say so explicitly in the log - a deferred
  item that nobody mentions looks identical to an ignored one.

  If an owner item is a bad idea, do not silently skip it. Say why in the log,
  do the part that is sound, and leave the item open with your reasoning.

  **When an item names a plan under `plan/`, that plan is the brief.** Read it in
  full, work the stage its Progress line says is *next* - one stage per session,
  split at a step boundary if the session is short - and do not start the stage
  after it. Look at the page before and after (`scripts/shot_dashboard.py`, desktop
  and 375px) and say what you saw. Re-measure any defect the plan cites before
  fixing it, and if the number has moved, say so rather than trusting the plan.
  Finish by updating the item's Progress line with what shipped and what is next,
  so the following session does not have to infer it. As of 2026-10-06 the two
  open items are `plan/calculation-transparency.md` and
  `plan/dashboard-redesign-master.md`; their order is fixed by the first of them
  (T0a, then T0b, then D2).

- Read the last 3 entries of `NIGHTLY_LOG.md`. What was in progress? What did
  the last session say to do next? What did it flag as broken?
- **Check the priorities section in `CLAUDE.md`.** Read what is at the top of
  the **Open** queue *now*. The **Settled** table above it is constraints, not
  work - if you are about to weaken a row, read its `DECISIONS.md` entry first.
- **The deferral-streak rule.** If the top open roadmap item has been deferred by
  **three consecutive sessions**, the next session carrying no owner item and no
  broken loop must **resolve** it: advance it, or demote it in "Current
  priorities" with the argument for what outranks it. Either is a result; a
  fourth "deferred, defensibly" is not. Measured 2026-10-02: priority 3 was
  deferred by **eight of ten** sessions, each time for defensible smaller work,
  and moved only once two sessions in a row had nominated it in writing. The
  roadmap-age line makes the trade visible; this is what closes it.
- Check the data loop: has `scripts/data-run.ps1` run recently? Look at
  `logs/`, `improvement/live_ic_history.csv`, and the newest files in
  `improvement/snapshots/`. **If the data loop is stalled or failing, fixing it
  is today's work regardless of the nominal focus.** Say so in the log and get
  on with it.
- Sanity-check the evidence base **at horizon `1m`**, filtering the file rather
  than reading its totals. The whole-file row count and newest date rise every
  weekday no matter what, because `1w` gains a row per run date - which is how
  the `1m` horizon sat frozen at 2026-08-14 for six sessions that each logged a
  newer number (fixed 2026-09-18). The `1m` newest `run_date` should trail
  today by 30-33 days. **Past 40, something is broken - investigate before
  doing anything else.**

## 2. Baseline

Record these before touching anything, so nothing gets misattributed to you:

```
python -m pytest tests/ test_screener.py -q
python run_screener.py --dry-run
```

Note pass/fail counts and any pre-existing failures.

## 3. Work the focus

Pick **one thing that matters**, not three that don't. Depth over breadth.

The bar for any change is evidence. Before you write code, be able to finish
this sentence: *"I know this is an improvement because ___."*

Acceptable endings **today**: a citation from the literature, documented
professional practice, a failing test that now passes, a profiling result, or a
concrete user-facing failure you can demonstrate.

**Not yet acceptable:** a backtest number (benched until 2027-02-11) or an IC
measurement from this system's own history. The history restarted growing on
2026-08-24, but at the `1m` optimization horizon it still holds a low
single-digit number of *effective* (non-overlapping) observations against a
gate of 8. Quote `_n_observations`, never `_n_raw_observations`. Both look like
evidence and are not.

Never acceptable: "it's cleaner", "it's more modern", "best practice".

**Monday - research.** Take one specific thing: a factor, a metric, a
threshold, a construction rule. Learn it properly in this one session, from
*both* sides:

- **The literature.** Real citations - author, title, year, what the finding
  actually was, the effect size, and the conditions it held under.
- **Documented practice.** How quant shops, institutional screens and index
  providers actually handle it. This is first-class evidence, not a footnote.

Where academia and practice disagree, say so and say why. Note where the
evidence contradicts what this screener currently does. The note must be
**complete today** - it is not a first half that Tuesday finishes.
**No production code.**

**Tuesday - product.** Open the live dashboard as a user would and ask whether
it answers *what should I look at / should I buy this / should I sell what I
hold / how much*. **Read `plan/dashboard-inventory.md` first** - the
most likely failure here is rebuilding something that already exists. Ship a
dashboard change, or write down precisely what it cannot answer and why.

**Wednesday - synthesis.** How does Monday's research fit the *rest* of the
screener? What does it overlap with or make redundant? What does it imply for
the other seven categories? Is the screener coherent after the change, or just
differently arranged? Record any methodology change in
`METHODOLOGY_CHANGELOG.md` with its sources.

**Thursday - build.** Implement what the week justified. Tests alongside, not
after.

If the week's research concluded *no change warranted* - a successful research
outcome, not a failed one - do **not** invent a methodology change so there is
something to build. Take the top open item in `CLAUDE.md`'s "Current
priorities" instead, and say in the log which of the two this was.

**Friday - harden and teach.** Tests, docs, error handling, and the
investment-club experience. Would a finance student understand what they're
looking at?

**On validation.** You do *not* need a number to make a well-sourced
methodology change; waiting for proof would freeze the project. Measurement
confirms a change over time. When evidence does accumulate, go back and check,
and record the result against the original changelog entry - if it turns out
wrong, revert it and say so. What you may not use as the justification is a
backtest figure (rule 5, benched until 2027-02-11, and never under **Evidence**)
or this system's own IC history (rule 4). Both look authoritative and are not.

**Methodology changes** are allowed and expected. Every one gets an entry in
`METHODOLOGY_CHANGELOG.md` *before* it ships, with evidence and expected
effect.

**Justify them from research, not from this system's own numbers** - see the
evidence rules above and rules 4 and 5 in `CLAUDE.md`.

That includes weights. Changing a factor weight because the research says a
factor is worth more or less - and explaining why - is legitimate work. Doing
it because a 3-point return series drifted is not.

**A scoring change is not finished until the page shows it** (owner direction,
2026-10-07: *"make sure ... if something is changed with scoring, that it also
updates on there, the frontend side of things to match the backend"*). The
drilldown prints, on every metric row, the stock's own figures put through the
formula, its rank among peers and percentile x weight. In the **same commit** as
any change to how a metric, weight or category is computed:

- **Weights** need nothing extra - the page reads the engine's own tables
  (`factor_engine.metric_weight_profiles`), and the build refuses to publish if
  it cannot rebuild every score (`calc_trace`).
- **A metric's formula or inputs** - update its entry in `metric_lineage.py`:
  `LINEAGE` (formula text, inputs, caveat), `RECOMPUTE` (the checker) and
  `EQUATIONS` (the line on the row). The suite evaluates every exact equation
  against every stock's scored value and **fails if they disagree** - do not
  loosen the 99% bar or flip a template to `exact=False` to get past it; fix
  the template.
- **A new metric** fails `test_every_weighted_metric_has_a_line_on_the_row`
  until it has an `EQUATIONS` or `SOURCES` entry, and
  `test_every_weighted_metric_has_a_lineage_entry` until it has a `LINEAGE`
  entry. If it needs a fetch field the page does not yet receive, add it to the
  entry's `inputs` (that publishes it).
- **Any sentence about how a number is computed** - register it in `claims.py`.

Then open the drilldown for a stock the change affects, at 1440 and 375px, and
check the row reads correctly. Say in the log what you saw.

## 4. Ship gates

All four must pass before you push to `main`:

1. `python -m pytest tests/ test_screener.py -q` - no new failures vs baseline
2. `python run_screener.py --dry-run` - exits 0
3. `index.html` still substantial, `dashboard_data.js` still parses
4. `git status --porcelain` - nothing unexpected left behind

Never weaken a test to pass a gate. If a gate fails and you cannot fix it
cleanly, revert your change - the gate is doing its job.

**Do not commit** `validation/data_quality_log.csv`, `factor_output.xlsx`, or
`sp500_tickers.json` unless changing them *is* the work. Stage files explicitly;
never `git add -A`.

## 5. Log and ship

Append to `NIGHTLY_LOG.md`:

```
## {{DATE}} - {{FOCUS}}

**Health (rule 8, all five):** last code session ran? | data loop published? |
evidence base at `1m` = R rows, newest YYYY-MM-DD (N days ago, bound 40),
E effective | priority 0 | top open roadmap item + its age in days
**Tests:** before N/M, after N/M
**Owner queue / rotation:** what you took, and anything deferred

### Did
- <what, and the evidence that it's an improvement>

### Evidence / research
- <citations: author, year, finding, effect size, conditions. Or "none - see why below">

### Methodology changed
- <changelog entries made, or "none">

### Tried and rejected
- <an idea the research did not support, and the source that ruled it out>

### Next
- <the single most valuable thing for the next session>
```

Then: commit in small scoped commits and **push the branch. Stop there.**

**The session does not merge or push `main`** (changed 2026-09-04). Run the
gates yourself - they tell you whether the work is fit to ship, and a session
that ignores them wastes the day - but `nightly-screener.ps1` re-runs all four
independently and merges only if they pass. Publishing yourself puts your push
*before* that check, which is how a failing commit reached the public site on
2026-09-02. Leave the merge to the runner and the gates become a precondition
instead of an audit.

Nothing is lost if a gate fails: your branch is on origin, and the next session
picks it up.

## If there is nothing worth doing

Say so and stop. Write a short log entry explaining why the focus area is
exhausted and what should replace it in the rotation. Do not invent
refactors to justify the session - churn on a mature codebase is a net
negative, and you are the only one watching for it.
