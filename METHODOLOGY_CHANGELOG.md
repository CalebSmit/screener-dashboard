# Methodology Changelog

Every change to **how the screener scores stocks** - factor weights, category
weights, metric weights, metric definitions, trap thresholds, scoring formulas,
neutralization, portfolio construction rules.

Changes here are **applied**, not proposed. Autonomous morning sessions may make
them directly. The obligation is not approval; it is **evidence**.

This file is the audit trail. If someone asks "why is Valuation weighted 22%?",
the answer must be findable here. A methodology change without an entry is a
bug, whether or not the code works.

## Entry format

```
## YYYY-MM-DD - Short title
**Area:** factor_weights / trap thresholds / metric definition / ...
**Changed:** exactly what, from what, to what
**Evidence:** citation, backtest result, IC measurement - the thing that
              justified it. Include effect sizes and the conditions.
**Expected effect:** what should move, and roughly how much
**Validated by:** the test/backtest that confirmed it, with the number
**Applied by:** improvement engine (auto) | morning session (manual)
**Rollback:** the tag or commit to revert to if this proves wrong
```

Entries by the improvement engine should also reference the IC observations and
information ratio that cleared its significance gates.

---

## 2026-08-05 - Enabled autonomous methodology evolution

**Area:** governance
**Changed:** `improvement.allow_auto_apply` false -> true in `config.yaml`.
Removed the human-approval requirement for methodology changes. Morning
sessions may now change scoring directly, and `improvement_engine.py` may write
weight changes once its statistical gates are satisfied.

**Evidence:** owner decision (2026-08-05) to run the project autonomously, with
evidence-backing rather than human review as the control.

**What did NOT change - deliberately:** the engine's statistical gates remain
exactly as they were:

| Gate | Value | Purpose |
|---|---|---|
| `min_observations_for_proposal` | 8 | no acting on noise |
| `min_ic_ir_for_auto_apply` | 0.5 | signal must be statistically real |
| `max_change_per_cycle` | 3.0% | no lurching |
| `shrinkage` | 0.5 | pull toward incumbent weights |
| `regime_scale_factor` | 0.0 | regime adjustment stays off until validated |

These gates *are* the safety mechanism now that human review is gone. Weakening
one requires its own changelog entry with a better argument than "it wasn't
firing."

**Expected effect:** none immediately. The engine has 3 live IC observations
and needs 8. First engine-applied change is realistically 2-3 weeks out, once
the data loop has accumulated evidence.

**Validated by:** `python -m pytest tests/ test_screener.py -q` -> 492 passed.
`tests/test_governance.py` covers the auto-apply gating.

**Applied by:** setup session (manual)
**Rollback:** set `allow_auto_apply: false` in `config.yaml`

---

## 2026-08-05 - Note: the learning loop had been inert since February

Not a methodology change; recorded because it explains the state of the
evidence base.

`improvement_engine.py` learns from snapshots recorded when the screener runs.
Between 2026-02-22 and 2026-08-05 the screener was not being run on a schedule,
so only **3 live IC observations** exist against a minimum of 8. The
self-improvement machinery has been present but starved.

`scripts/data-run.ps1` (Mon/Wed/Fri, 2:00 AM) now runs the screener and records
a snapshot each time. Evidence should clear the 8-observation gate in roughly
3 weeks.

**Implication for anyone reading the weights:** current factor weights are the
*designed* values from `config.yaml`, unchanged by live evidence. They have not
yet been validated against realized forward returns by this system.

---

## Open question that blocks trusting any of this

`backtest.py` carries two acknowledged biases, in its own docstring:
**survivorship bias** (uses today's S&P 500 constituents throughout history)
and **look-ahead bias** (fundamental scores held constant from a single
snapshot; only momentum and risk are recomputed).

A backtest with those properties **cannot honestly validate a methodology
change** - it will tend to flatter any strategy tilted toward stocks that
happen to be in the index today. Until this is fixed, "validated by backtest"
in this file should be read with suspicion, and IC measurements from the live
data loop are the more trustworthy evidence.

See `.claude/plan/backtest-v2.md`. This is priority 2 in `CLAUDE.md` for a
reason.

---

## 2026-08-11 - Research becomes the basis for methodology; backtest benched until 2027-02-11

**Area:** governance / evidence standard
**Applied by:** owner direction, recorded by setup session

**Changed.** Two related decisions:

1. **Published research and documented professional practice now carry equal
   weight to measured results** as justification for a methodology change.
   A well-sourced change no longer waits for a number.

2. **`backtest.py` output does not decide anything until 2027-02-11.** It may
   be run and reported as supporting colour. It may not justify a change, keep
   or revert one, appear under **Evidence** in this file, or act as a
   validation gate in the improvement engine.

**Evidence.** The backtest documents two biases in its own docstring:
survivorship (today's constituents applied across all history) and look-ahead
(fundamentals held constant from one snapshot). A result carrying both is not
weak evidence - its direction is unknown, and it systematically flatters
strategies tilted toward companies that still exist today, which is precisely
where this screener's valuation weighting sits.

Separately, `research/2026-08-10-ic-evidence-independence.md` established that
live IC evidence accrues far more slowly than the raw row count suggests -
at most 2 non-overlapping 30-day windows among 11 backfillable dates. Requiring
measured proof before any methodology change would therefore freeze the project
for roughly six months.

**Expected effect.** Methodology work proceeds on research grounds during a
period when no trustworthy measurement exists. Risk accepted knowingly: changes
made on literature and practitioner grounds are not yet confirmed by this
system's own data.

**How this is mitigated.** The written-argument requirement is unchanged - each
change still needs sources a sceptical reader can follow. Changes remain
individually reverted via their changelog entry, and `improvement_engine.py`
continues to learn from *live* forward returns, which are genuinely
out-of-sample and unaffected by the backtest's biases.

**Backtest observation:** none - deliberately.

**Validated by:** not yet. That is the point: revisit each entry made in this
period once backtest v2 exists or enough independent IC observations accrue.

**Rollback:** restore the prior standard by reverting CLAUDE.md rule 5 and the
"Your mandate" section. The date is the owner's to move.
