Today is **{{DATE}}**. This is a **RETROSPECTIVE** session, not a normal one.
You are on branch `{{BRANCH}}`.

Today you improve **how this project works on itself**, rather than working on
the project. The routine has been running for a while. Your job is to find out
whether it is actually producing value, and change the process where it is not.

Read `CLAUDE.md` first. Its rules still apply.

## 1. Gather the evidence

Do not rely on impressions. Go and look:

- **`NIGHTLY_LOG.md`** - read every entry since the last retrospective. For
  each: did it produce something real, or was it churn?
- **`git log --oneline --since="4 weeks ago"`** - what actually shipped? How
  many merges, and how substantial?
- **`logs/*.log`** - how many runs failed their ship gates, and which gate?
  How long do sessions take? Any that hit the 4-hour limit?
- **`METHODOLOGY_CHANGELOG.md`** - are methodology changes actually being
  evidenced, or are entries getting thin and formulaic?
- **`research/`** - are the notes real research with citations, or
  plausible-sounding filler? Be harsh here; this is the easiest thing to fake.
- **`improvement/live_ic_history.csv`** - is the evidence base growing? At
  which horizons?
- **`git log --stat`** on a few nightly merges - is the code getting better, or
  just getting *changed*?

## 2. Answer these honestly

Write the answers into the log. Short and specific, no hedging.

1. **What fraction of sessions produced something genuinely valuable?** Name
   the good ones and the wasted ones.
2. **Which rotation day earns its place, and which does not?** If a day
   consistently produces churn, it should change or go.
3. **Is the evidence standard holding?** Are changes really backed by
   citations, backtests and IC numbers - or has "evidence" quietly degraded
   into assertion?
4. **What keeps going wrong?** Repeated gate failures, recurring bugs,
   instructions that get misread the same way every time.
5. **Is the tool actually closer to being the place you'd look before buying or
   selling a stock?** If not, what is the honest blocker?
6. **What is the routine systematically blind to?** What has nobody looked at
   because no day in the rotation points there?

## 3. Change the process

You may edit any of these:

- `CLAUDE.md` - rules, priorities, the rotation itself
- `prompts/nightly.md` - the daily instructions
- `prompts/retrospective.md` - this file, including these questions
- `scripts/nightly-screener.ps1` and `scripts/data-run.ps1` - the runners
- The rotation table and the focus strings in the runner
- Retrospective frequency, if fortnightly is wrong

Make the changes concrete. "Be more rigorous" is not a process change. "Monday
research notes must include at least two primary sources published in a
peer-reviewed journal, and Tuesday must cite the note by filename" is.

**Prefer deleting to adding.** A prompt that grows every fortnight becomes a
prompt nobody follows. If you add an instruction, look for one to remove.

## 4. Two things you may not do

These exist because you are modifying the thing that constrains you, and
nobody is reviewing the result.

1. **Do not weaken or remove the four ship gates** (tests, dry-run, dashboard
   artifacts, clean tree). You may make them *stricter* or add new ones. If you
   genuinely believe a gate is wrong, write the argument in the log and leave
   it in place for the owner to decide - do not remove it yourself.
2. **Do not remove the evidence requirement, the rollback tagging, or this
   restriction.** A process that can quietly relax its own standards will,
   given enough fortnights.

Everything else is fair game.

## 5. Ship gates and log

Same four gates as any session - run them, and do not merge if any fails.

Log the retrospective under the normal format, with an extra section:

```
### Retrospective findings
- Sessions reviewed: N (from YYYY-MM-DD to YYYY-MM-DD)
- Genuinely valuable: N | Churn: N | Failed gates: N
- <the honest answers to the six questions>

### Process changes made
- <what you changed about how this works, and why>

### Flagged for the owner
- <anything you think should change but chose not to change yourself>
```

Then commit, merge, push - as normal.
