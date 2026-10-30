Today is **{{DATE}}**. This is a **RETROSPECTIVE** session, not a normal one.
You are on branch `{{BRANCH}}`.

Today you improve **how this project works on itself**, rather than working on
the project. The routine has been running for a while. Your job is to find out
whether it is actually producing value, and change the process where it is not.

Read `CLAUDE.md` first. Its rules still apply.

**You run under `claude --print`: a reply with no tool call ends the session.** A background
command's "you will be notified" never arrives and the command dies with the session - that is
how 2026-10-18 and 2026-10-19 ended dirty and failed gates 1 and 4
(`research/2026-10-19-session-ended-on-a-background-job.md`). Wait in the foreground with
`python scripts/wait_for.py`, and write your final reply only after `git status --porcelain` is
empty and the branch is pushed - by {{DEADLINE}} at the latest.

## 1. Gather the evidence

Do not rely on impressions. Go and look:

- **`NIGHTLY_LOG.md`** - read every entry since the last retrospective. For
  each: did it produce something real, or was it churn?
- **`git log --oneline --since="4 weeks ago"`** - what actually shipped? How
  many merges, and how substantial?
- **`logs/*.log`** - how many runs failed their ship gates, and which gate? Not
  session duration: measured three times now, the range is 14-35 minutes against
  a 4-hour limit and the worst ever was 93. Stop checking it. **Also count the
  sessions that never got to start** - a runner can lose a day before invoking
  anything (2026-10-28 06:00: "Could not stash the leftovers", a refused data run
  having left three untracked files behind), and those logs are short enough to
  skim past.
- **`BASELINE` in the newest session logs** (added 2026-10-30) - was `main` red
  when a session arrived, and did the session notice? The 02:00 data run can
  redden it with no commit touching code, and a session that spends its day on
  its focus instead gets its work refused by gate 1.
- **`METHODOLOGY_CHANGELOG.md`** - are methodology changes actually being
  evidenced, or are entries getting thin and formulaic?
- **`research/`** - are the notes real research with citations, or
  plausible-sounding filler? Be harsh here; this is the easiest thing to fake.
- **`improvement/live_ic_history.csv`** - **filter to horizon `1m`** and check
  its newest `run_date` against rule 8's 40-day bound. The whole-file totals
  rise every weekday whatever happens, which is how a frozen `1m` horizon went
  six sessions unremarked before 2026-09-18.
- **`scripts/nightly-screener.ps1` and `scripts/data-run.ps1`** - read every
  path that touches `main`: what publishes, in what order, what it is checked
  against first, and what runs from `finally`. Nothing else in the rotation ever
  points here, and **all four retrospectives that have looked found a live hole
  on the first read.**

  **Do this as an enumeration, not a read-through.** So:

  1. **List every artifact `data-run.ps1` stages** (`$DataArtifacts`) and, for
     each, name the check that stands between it and the public site **on both
     paths**. An artifact with no check is the finding. Do the converse too: an
     artifact named in a gate's scope that the loop no longer stages.
     **`python scripts/retro_artifact_coverage.py`** prints the crude version - every test
     module naming each artifact, and which of them the publish gate runs. It is a
     prompt for judgement, not a verdict (a module about the runners names every
     artifact); reading its `improvement` and `factor_vol_history.csv` rows is how
     2026-10-30 found its hole.
  2. **List every generated thing a session can change** and ask which gate would
     see it. Gates 1 and 3 read *committed* artifacts; only a full run rewrites
     them (this is why `scripts/rehearse_publish.py` exists).
  3. **Check `claims.py` against `scripts/check_published_claims.py`** - every
     `checked_by` module must be inside the gate. It is derived now, so this
     should be a one-line confirmation rather than an audit. **Then ask the
     question derivation cannot:** `claims.py` registers published *sentences*, so
     a staged artifact that carries no sentence is invisible to it. Which ones?

  Keep score rather than re-finding them:

  | Found | The asymmetry | Closed by |
  |---|---|---|
  | 2026-09-04 | the brief published with `git push origin HEAD:main` from `finally`, so a gate failure published the refused branch | `scripts/publish-brief.ps1` |
  | 2026-09-18 | the data loop published the payload behind a size floor where the code loop had gate 3 | `node --check` on both paths |
  | 2026-10-02 | the data loop published regenerated public docs having run **no tests**, where the code loop had gate 1 | `scripts/check_published_claims.py` |
  | 2026-10-16 | `dashboard_context.js` - a second published payload file since 2026-10-08 - had no floor, no parse and no header check on **either** path; and 15 modules holding 64 of `claims.py`'s own checks were outside the claims gate | both runners' spec lists; the gate's membership derived from `claims.py` |
  | 2026-10-30 | the three **evidence** files the loop appends to (`dispersion_history.csv`, `improvement/snapshots/`, `factor_vol_history.csv` - the last read by `adjust_momentum_weight`, so it sets the *next* run's weight) had no gate on either path, and the test that excluded them asserted they were "guarded by `check_run_health.py`", which does not look at them | `tests/test_one_observation_per_run_date.py` in the publish gate |

  The first four were one shape - *the strongest check guards the path that
  publishes least often*. **The fifth was a different one and is the more useful
  lesson: a derived membership rule is only as wide as the thing it derives from.**
  Deriving the gate from `claims.py` made a missing *page sentence* impossible and
  a missing *evidence file* permanent. When a list is replaced by a derivation,
  ask what the derivation cannot reach.

  Gates 1, 3 and the brief have counterparts on both paths. **Gate 2
  (`--dry-run`) and gate 4 (clean tree) deliberately do not**, and that was
  checked on 2026-10-16: the data loop *runs the real pipeline*, which strictly
  dominates a dry-run, and it stashes a dirty tree before starting and stages
  only its own artifact list, which is stronger than a clean-tree check in the
  direction that matters. Do not "close" either.

- **`CLAUDE.md`'s own length.** 860 -> 897 -> 1058 -> 970 -> 1036 -> 959 lines
  across five retrospectives. It grows back every fortnight by the same mechanism:
  an item closes and its record stays in "Open", in the one file every session
  must read before it starts. **Expect to move 100+ lines every time, and do it
  whatever the number says** - the test is not "is it longer than last time" but
  "is anything under Open actually closed". A closed item's reasoning belongs in
  `DECISIONS.md`, its constraint in `CLAUDE.md`, its discovery in
  `NIGHTLY_LOG.md`. The two biggest sources, both times: a per-item chronology
  that should be a table in `DECISIONS.md`, and a *refused* proposal whose
  full argument sits under Open when one line plus "read section N before
  reopening" would do.

- **Who is doing the work.** Count `NIGHTLY_LOG.md` entries since the last
  retrospective, split by scheduled versus owner-run. It was **11 scheduled / 44
  owner-run** on 2026-10-16 and **24 / 14** on 2026-10-30 - so the ratio is not a
  stable fact and the rotation's job changes with it. Say what the current split
  implies: at 11/44 the scheduled session's distinctive contribution was *deciding
  what the owner's sessions left open*; at 24/14 it is most of the throughput and
  the rotation has to carry the roadmap itself.

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
- `scripts/nightly-screener.ps1` and `scripts/data-run.ps1` - the runners, and
  the rotation table and focus strings inside them
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

Same four gates as any session. Run them yourself; `nightly-screener.ps1` runs
them again and does the merge.

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

Then commit in scoped commits and push the branch. **Do not merge or push
`main`** - the runner does that, after its own independent gate run. A
retrospective that exempted itself from the rule it had just written would be
the first step in the drift section 4 exists to prevent.
