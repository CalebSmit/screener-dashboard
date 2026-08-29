Today is **{{DATE}}**. Today's focus: **{{FOCUS}}**
You are on branch `{{BRANCH}}`.

You are running unattended and nobody will review your work. You push to `main`
yourself, and `main` is served live to the public. Act accordingly: the standard
is not "plausible", it is "I can show why this is right."

Read `CLAUDE.md` in full before anything else. Its rules override this prompt.

**Finish what you find (rule 11).** If a fix needs a machine-level change -
re-registering a scheduled task, anything outside git - make the change
yourself and verify it took effect before you finish, the same standard as
verifying a code change. Do not leave a command in the log for the owner to
run by hand. If verifying genuinely requires something outside this session's
reach, say so in the log and leave it for the *next* session to finish - not
for the owner.

## 1. Orient

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

- Read the last 3 entries of `NIGHTLY_LOG.md`. What was in progress? What did
  the last session say to do next? What did it flag as broken?
- **Check the priorities section in `CLAUDE.md`.** The forward-return horizon
  bug that used to sit at priority 0 shipped 2026-08-24; do not go looking for
  it. Read what is at the top of the queue *now*.
- Check the data loop: has `scripts/data-run.ps1` run recently? Look at
  `logs/`, `improvement/live_ic_history.csv`, and the newest files in
  `improvement/snapshots/`. **If the data loop is stalled or failing, fixing it
  is today's work regardless of the nominal focus.** Say so in the log and get
  on with it.
- Sanity-check the evidence base: `live_ic_history.csv` should be gaining rows
  at the `1m` horizon once priority 0 lands. If it is not growing, something is
  broken - investigate before doing anything else.

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

This day exists because the dashboard had a standing owner directive and 23
days of zero progress: every session that could run was spent on data-pipeline
defects, correctly, and nothing in the rotation protected product work from
firefighting. Firefighting will always win that fight unless a day is reserved.

**Wednesday - synthesis.** How does Monday's research fit the *rest* of the
screener? What does it overlap with or make redundant? What does it imply for
the other seven categories? Is the screener coherent after the change, or just
differently arranged? Record any methodology change in
`METHODOLOGY_CHANGELOG.md` with its sources.

**Thursday - build.** Implement what the week justified. Tests alongside, not
after.

**Friday - harden and teach.** Tests, docs, error handling, and the
investment-club experience. Would a finance student understand what they're
looking at?

**On validation.** You do *not* need a backtest number to make a well-sourced
methodology change - the backtest is known-biased and independent IC
observations accrue about monthly, so waiting for proof would freeze the
project. Measurement confirms a change over time; it is not the gate it must
pass to be made. When evidence does accumulate, go back and check, and record
the result against the original changelog entry. If it turns out wrong, revert
it and say so.

**The backtest decides nothing until 2027-02-11** (`CLAUDE.md` rule 5). Run it
and report it if it is interesting, but never use a number from it to justify,
keep or revert a methodology change, and never file it under **Evidence** in
`METHODOLOGY_CHANGELOG.md`. It carries survivorship and look-ahead bias, so its
direction is unknown - that is worse than having no number at all, because it
looks authoritative. Until that date, methodology rests on research.

**Methodology changes** are allowed and expected. Every one gets an entry in
`METHODOLOGY_CHANGELOG.md` *before* it ships, with evidence and expected
effect.

**Justify them from research, not from this system's own numbers.** The IC
series has 3 observations, all `1w`, all from February, and the significance
test counts raw rows in a way that overstates independence ~2.35x. The backtest
is benched until 2027-02-11. So neither is evidence yet - see rules 4 and 5 in
`CLAUDE.md`. What counts today is published literature, documented
professional practice, and a clear account of how the change fits the screener
as a whole.

That includes weights. Changing a factor weight because the research says a
factor is worth more or less - and explaining why - is legitimate work. Doing
it because a 3-point return series drifted is not.

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

**Tests:** before N/M, after N/M
**Data loop:** healthy | stalled | fixed - <evidence>

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

Then: commit in small scoped commits, push the branch, merge it into `main`
(fast-forward or a clean merge commit - no history rewriting), and push `main`.

If any ship gate failed: push the branch, do **not** merge, and make the log
entry say clearly what is broken and what you'd try next.

## If there is nothing worth doing

Say so and stop. Write a short log entry explaining why the focus area is
exhausted and what should replace it in the rotation. Do not invent
refactors to justify the session - churn on a mature codebase is a net
negative, and you are the only one watching for it.
