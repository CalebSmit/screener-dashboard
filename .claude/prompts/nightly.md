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

**Research days (Mon/Tue):** the deliverable is a dated note in `research/`,
with real citations - author, title, year, and what the finding actually was,
including effect sizes and the conditions under which it held. Note where the
evidence contradicts what this screener currently does. End with a concrete
recommendation and, on Tuesday, a design plus the hypothesis a backtest could
refute. **Do not write code on research days** beyond throwaway analysis
scripts.

**Build day (Wed):** implement what the week's research justified. Write tests
alongside, not after.

**Validation day (Thu):** measure whether the week's change actually helped.
Be genuinely willing to conclude that it did not, and revert it if so -
that is a successful Thursday, not a failed one. Record the number either way.

**Harden day (Fri):** tests, docs, error handling, and the investment-club
experience. Would a finance student understand what they're looking at?

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
