Today is **{{DATE}}**. Today's focus: **{{FOCUS}}**
You are on branch `{{BRANCH}}`.

You are running unattended and nobody will review your work. You push to `main`
yourself, and `main` is served live to the public. Act accordingly: the standard
is not "plausible", it is "I can show why this is right."

Read `CLAUDE.md` in full before anything else. Its rules override this prompt.

## 1. Orient

- Read the last 3 entries of `NIGHTLY_LOG.md`. What was in progress? What did
  the last session say to do next? What did it flag as broken?
- **Check priority 0 in `CLAUDE.md` first.** If the forward-return horizon bug
  is still unfixed, that is today's work regardless of the nominal focus - the
  improvement engine cannot propose anything until it is done, so every other
  methodology task is blocked behind it.
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
measurement from this system's own history (3 observations, all `1w`, all
February). Both look like evidence and are not.

Never acceptable: "it's cleaner", "it's more modern", "best practice".

**Monday - component research.** Pick one specific thing: a factor, a metric,
a threshold, a construction rule. Learn it properly. Real citations - author,
title, year, what the finding actually was, the effect size, and the conditions
it held under. Note where the evidence contradicts what this screener currently
does. **No production code.**

**Tuesday - practitioner research.** How do people who do this for a living
actually handle it? Institutional screens, quant shop methodology, peer tools,
published factor definitions from index providers. Academia and practice often
disagree - where they do, say so and say why. This is first-class evidence, not
a footnote to the papers. **Still no production code.**

**Wednesday - synthesis.** The important day. How does this fit the *rest* of
the screener? What does it overlap with or make redundant? What does it imply
for the other seven categories? Is the screener as a whole coherent after the
change, or just differently arranged? Design the whole, then record any
methodology change in `METHODOLOGY_CHANGELOG.md` with its sources.

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
