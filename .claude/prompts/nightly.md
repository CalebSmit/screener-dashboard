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
this sentence: *"I know this is an improvement because ___."* Acceptable
endings are a citation, a backtest number, an IC measurement, a failing test
that now passes, a profiling result, or a concrete user-facing failure you can
demonstrate. Unacceptable: "it's cleaner", "it's more modern", "best practice".

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

Never quote a number from the current `backtest.py` as if it settled anything -
see `.claude/plan/backtest-v2.md` for why.

**Methodology changes** are allowed and expected. Every one gets an entry in
`METHODOLOGY_CHANGELOG.md` *before* it ships, with evidence and expected
effect. For factor weights specifically, prefer improving
`improvement_engine.py` and its evidence base over hand-tuning - see rule 4 in
`CLAUDE.md`.

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
- <citations, backtest numbers, IC deltas, or "none - see why below">

### Methodology changed
- <changelog entries made, or "none">

### Tried and rejected
- <what didn't survive validation, and the number that killed it>

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
