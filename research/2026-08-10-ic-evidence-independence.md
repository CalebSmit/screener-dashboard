# Are the improvement engine's IC observations independent enough to change weights?

**Date:** 2026-08-10
**Session:** code loop, research day (ISO week 33)
**Status:** finding — **blocks the naive priority-0 fix**

---

## Bottom line

Priority 0 in `CLAUDE.md` is real and still unfixed. But the fix as described
there — reprocess old snapshot dates so their 1-month returns get filled in,
clearing the 8-observation gate "in days rather than months" — would take the
improvement engine from *"can never propose a weight change"* to
*"auto-applies weight changes on roughly two effectively independent
observations, drawn from an 8-week window, from snapshots that are mostly
development artifacts rather than scheduled market runs."*

`allow_auto_apply` is `true`. So the naive fix does not merely unblock
learning — it arms it, on evidence that cannot support the conclusion.

**Recommendation: do not ship the priority-0 dedup fix on its own.** It must
ship together with an observation-independence correction, or with
`allow_auto_apply` temporarily set back to `false`. Details in
[Recommendation](#recommendation).

This is a *good* outcome for the bug: the fix is still right, the backfill is
still worth doing, and the resulting 1-month IC series is still the most
honest evidence this system has. It just must not be counted as 11 independent
draws.

---

## Method

Everything below is reproducible from the repo with read-only shell commands.
No network and no Python were available this session (see
[Environment blocker](#environment-blocker)), so every number here comes from
the committed data files, and every command is given so it can be re-run.

---

## Findings

### F1 — Priority 0 confirmed, and the mechanism is slightly different from the CLAUDE.md description

`improvement_engine.py:253` still contains the skip:

```python
if snap_date_str in existing_dates:
    continue
```

`CLAUDE.md` states that `performance_history.csv` "has **only** a
`fwd_return_1w` column". That was true on 2026-08-05; it is **no longer
accurate**. The file now has `fwd_return_1m` and `fwd_return_3m` columns, and
they are partially populated:

```sh
cut -d, -f1,15 improvement/performance_history.csv | grep -E ",[-0-9]" | cut -d, -f1 | sort | uniq -c
#  503 2026-03-15
# 2000 2026-04-14

cut -d, -f1,16 improvement/performance_history.csv | grep -E ",[-0-9]" | cut -d, -f1 | sort | uniq -c
# 2000 2026-04-14
```

So **2 of 13 dates have a 1-month return; 1 of 13 has a 3-month return.**

The refined mechanism: a date's returns are frozen at whatever horizons were
computable **the first time that date was ever processed**. Dates first seen at
7–9 days old get `fwd_return_1w` and nothing else, forever. `2026-03-15` and
`2026-04-14` have longer-horizon data only because they happened to sit
unprocessed until they were already old enough — an accident, not a design.

The bug is exactly as diagnosed. The consequence is slightly less total than
described (the 1m column exists and is not entirely empty), which matters
because it changes what the backfill will actually produce.

### F2 — `performance_history.csv` contains large-scale exact duplication

```sh
ls improvement/snapshots/ | cut -d_ -f1 | sort | uniq -c
#  4 2026-02-20   13 2026-02-21    2 2026-02-22    1 2026-02-23
#  5 2026-02-24    1 2026-02-28    1 2026-03-01    1 2026-03-09
#  1 2026-03-15    2 2026-03-16    4 2026-04-14    3 2026-07-28
#  1 2026-07-29    1 2026-08-07

cut -d, -f1,2 improvement/performance_history.csv | sort | uniq -c | sort -rn | head -3
# 13 2026-02-21,ZTS
# 13 2026-02-21,ZBRA
# 13 2026-02-21,ZBH
```

`compute_forward_returns()` loads `existing_dates` **once**, then iterates over
every snapshot *file*. Multiple files sharing a date are all appended in the
same pass, so each ticker appears once per snapshot file for that date.

Effect: the file is 8.1 MB and 18,552 data rows for what is really ~4,500
distinct ticker-date pairs — roughly **75% redundant**.

`compute_live_ic()` groups by `run_date` and reports `len(group)` as
`n_tickers`. This flows straight into the published record:

```
improvement/live_ic_history.csv
2026-02-21,1w,6539,...
```

**6,539 "tickers" for an S&P 500 screener.** That number is on the record and
is wrong by 13×. It does not currently corrupt the significance test (which
uses the count of *dates*, not tickers — see F7), but it is a visible
credibility defect in an artifact whose whole purpose is to be checkable.

### F3 — The observations are not independent: 11 dates, at most 2 non-overlapping 30-day windows

These are the 13 snapshot dates:

```
2026-02-20, 02-21, 02-22, 02-23, 02-24, 02-28,
2026-03-01, 03-09, 03-15, 03-16,
2026-04-14,
2026-07-28, 07-29,
2026-08-07
```

The 11 dates that are ≥30 days old as of today — the ones the backfill would
light up — span **2026-02-20 to 2026-04-14, a 53-day window**.

A 1-month forward return from date *D* covers *D* to *D+30*. Two observations
are non-overlapping only if their dates are ≥30 days apart. Within a 53-day
span, the maximum number of mutually non-overlapping 30-day windows is:

> floor(53 / 30) + 1 = **2**

Six of the eleven dates fall inside a single 9-day stretch (02-20 → 02-28).
Their 1-month forward returns overlap by 21–29 of 30 days. They are, to a good
approximation, **the same observation measured six times**.

So the backfill yields 11 rows in `live_ic_history.csv` that represent roughly
**2 independent draws**.

### F4 — All of that evidence sits in a single 8-week regime

Every backfillable date is between 2026-02-20 and 2026-04-14. Whatever the
market did in those eight weeks is the entire basis on which factor weights
would be re-estimated. There is no second regime to check against, and the
engine has no mechanism that would notice.

### F5 — Most of these snapshots are development artifacts, not scheduled market runs

```sh
date -d 2026-02-21 +%A   # Saturday
date -d 2026-02-22 +%A   # Sunday
date -d 2026-02-28 +%A   # Saturday
date -d 2026-03-01 +%A   # Sunday
date -d 2026-03-15 +%A   # Sunday
```

**5 of the 13 snapshot dates are weekends.** There is no market data on a
Saturday; `_nearest_price()` silently substitutes an adjacent trading day, so
these rows are not wrong exactly, but they are not what they claim to be
either.

The scheduled data loop did not exist until 2026-08-05. Everything before
2026-07-28 is a by-product of development and testing — 13 snapshots on
Saturday 2026-02-21 alone is the signature of a script being run repeatedly in
a single sitting, not of a market being observed.

Worth stating plainly: **the entire current 1-month evidence base is two dates,
one of which is a Sunday dev run** (F1).

### F6 — The data loop never recomputes live IC

`record_run_snapshot()` calls `compute_dispersion()` and
`compute_forward_returns()`. It does **not** call `compute_live_ic()`.

Evidence — after the successful 2026-08-07 data run:

```sh
ls -la improvement/
# performance_history.csv   Aug  7 02:12   <- updated
# dispersion_history.csv    Aug  7 02:11   <- updated
# live_ic_history.csv       Aug  5 21:05   <- NOT updated
```

`live_ic_history.csv` only grows when the improvement *report* is generated.
The data-loop log line "Improvement engine now has 3 live IC observation(s)"
is just reading the stale file. So even once priority 0 is fixed, the IC series
will not advance on its own from the data loop.

### F7 — The arithmetic: the gate passes on the pseudo-evidence, and fails on the real evidence

`improvement_engine.py:1347`:

```python
t = float(ic_ir) * math.sqrt(n_obs)
```

where `n_obs = len(series)` — the number of **run dates** in
`live_ic_history.csv`. Independence is assumed, never checked.

`config.yaml` sets `min_ic_ir_for_auto_apply: 0.5` and
`min_observations_for_proposal: 8`. Take a category sitting exactly on the bar,
IC-IR = 0.5:

| Observation count used | t = IR·√n | one-sided p | Passes? |
|---|---|---|---|
| n = 11 (post-backfill, as counted today) | 0.5 × √11 = **1.66** | **0.049** | **yes** |
| n = 2 (non-overlapping windows, F3) | 0.5 × √2 = **0.71** | **0.240** | no |

The inflation factor is √(11/2) = **2.35×** on the t-statistic.

So the naive backfill does not just clear the observation-count gate — it
clears the *significance* gate too, at a p-value that is wrong by roughly an
order of magnitude. With `allow_auto_apply: true`, the engine would then write
weight changes to `config.yaml`.

**This is the finding.** The gates in `config.yaml` are described in
`CLAUDE.md` as "the safety mechanism". They are computed from an observation
count that the priority-0 fix is about to inflate 5-fold with overlapping data.

---

## Why this matters more than the bug itself

Today the engine is inert: it has never proposed a weight change. That is a
failure, but a *safe* one — the tool's published weights are the ones a human
chose and documented.

After a naive priority-0 fix, the engine becomes confident and wrong. It would
begin rewriting factor weights, and every change would arrive with a p-value
attached and an entry in the changelog saying the evidence gates were
satisfied. The tool's credibility is the product; a defensibility feature that
produces authoritative-looking numbers from two overlapping observations is
worse than no feature.

---

## Recommendation

Ship the priority-0 dedup fix **only as part of a package that also fixes how
observations are counted.** Concretely, in priority order:

1. **Fix the dedup key** (the original priority 0). Make it
   `(run_date, horizon)`-aware so a date is reprocessed when it becomes old
   enough for the next horizon, instead of being frozen at first sight.
   Regression test: a snapshot first processed at 7 days old must gain a
   non-NaN `fwd_return_1m` when `compute_forward_returns()` is next called at
   ≥30 days old.

2. **Deduplicate `(run_date, ticker)`** when appending to
   `performance_history.csv`, keeping one row per pair. Fixes the 13× inflated
   `n_tickers` (F2) and shrinks the file ~75%.

3. **Count effective observations, not rows.** Before any significance test,
   reduce the IC series to a non-overlapping subset — greedily take the
   earliest date, then the next date ≥ *horizon* days later, and so on. Use
   that count in `_ir_to_one_sided_pvalue()`. On today's data that turns
   n=11 into n=2 and correctly refuses to propose.

   This is the minimum honest change. A Newey–West / Hansen–Hodrick style
   autocorrelation-consistent standard error would be the more standard
   treatment and keeps more of the data, but it is materially more work and
   the simple non-overlapping subset is both defensible and *teachable* —
   which `CLAUDE.md` weights heavily.

4. **Exclude weekend run dates** from the IC series, or at minimum flag them.
   A scoring date with no market close is not an observation.

5. **Have the data loop call `compute_live_ic()`** so the series actually
   advances (F6).

6. **Consider quarantining the pre-2026-07-28 snapshots** from the evidence
   base entirely. They are dev artifacts (F5), clustered, weekend-heavy, and
   duplicated. The temptation is to keep them because they are the only
   history there is — but "the only data we have" is not the same as "data
   that supports a conclusion." A defensible position is: the evidence base
   starts 2026-07-28, when the scheduled loop began, and the engine stays
   inert until it has genuinely independent monthly observations. At one
   snapshot per weekday, 8 non-overlapping monthly observations take roughly
   **8 months**, not days.

   That is a real and unwelcome answer. It should be stated in the log and to
   the owner rather than engineered around.

**If (3) cannot be delivered in the same session as (1), set
`allow_auto_apply: false` before shipping (1)**, and record it in
`METHODOLOGY_CHANGELOG.md`. An inert engine is safe; a confidently wrong one is
not.

---

## Relevant literature

**These citations could not be verified this session** — WebSearch and
WebFetch were both unavailable (see below). They are recorded from model
knowledge as leads for Tuesday's design session and **must be checked before
any of them is cited in public-facing docs**. The evidence for today's finding
does not rest on them; it rests on the repo data in F1–F7 and on the arithmetic
in F3 and F7, all of which are reproducible.

- **Hansen & Hodrick (1980)**, "Forward Exchange Rates as Optimal Predictors of
  Future Spot Rates", *Journal of Political Economy* 88(5). The origin of the
  overlapping-observations correction; overlapping forecast horizons induce
  serial correlation in the error term and standard OLS errors are badly
  understated.
- **Newey & West (1987)**, "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
  *Econometrica* 55(3). The standard HAC estimator; the usual fix for (3)
  above if the non-overlapping-subset approach is judged too lossy.
- **Boudoukh, Richardson & Whitelaw (2008)**, "The Myth of Long-Horizon
  Predictability", *Review of Financial Studies*. Argues that much apparent
  long-horizon return predictability is an artifact of overlapping-observation
  inference. Directly on point.
- **Harvey, Liu & Zhu (2016)**, "…and the Cross-Section of Expected Returns",
  *Review of Financial Studies* 29(1). Argues that with hundreds of factors
  tested, a t-statistic hurdle around 3.0 is more appropriate than 2.0.
  Relevant because this engine tests 8 categories and 44 metrics; the
  Benjamini–Hochberg control at `improvement_engine.py:1324` is the right
  instinct, but it is fed the same inflated p-values as everything else.
- **Grinold (1989)**, "The Fundamental Law of Active Management", *Journal of
  Portfolio Management*. IR ≈ IC · √breadth, where breadth is the number of
  **independent** bets. The independence requirement is precisely what F3
  violates.
- **Asness (2016)**, "The Siren Song of Factor Timing", *Journal of Portfolio
  Management*. Skeptical of adjusting factor exposures on short-horizon
  evidence — the exact activity this engine automates. Worth reading properly
  before defending `allow_auto_apply: true` at all.

---

## For Tuesday's design session

The hypothesis to state and make falsifiable:

> *Restricting the IC series to non-overlapping observations at the
> optimization horizon changes the engine's proposals from "significant" to
> "insufficient evidence" on the current data, and continues to refuse until
> genuinely independent observations accumulate.*

Refutable by: building the corrected estimator, running it against the
backfilled series, and checking whether it still proposes. If it proposes
anyway, the diagnosis in F3/F7 is wrong and should be re-derived.

Open questions to settle Tuesday:

1. Non-overlapping subset vs. Newey–West. Simpler and teachable, or standard
   and data-efficient?
2. Does the 1-month optimization horizon make sense at all given the data
   arrival rate? A 1-week horizon gives ~5× the independent observations. The
   Phase 13 governance note ("a 1-week signal must never optimize a monthly
   strategy") is correct in principle — but it may be worth asking whether the
   *strategy* should be weekly instead.
3. Should pre-2026-07-28 snapshots be quarantined (recommendation 6)?

---

## Environment blocker

**No autonomous session can currently pass its own ship gates.**

This session could not execute Python at all. `python --version` succeeds;
anything that runs code does not:

```
python -c "print('hello')"                  -> denied
python -m pytest tests/ test_screener.py -q -> denied
python run_screener.py --dry-run            -> denied
```

`WebSearch` and `WebFetch` were denied too. All of these are explicitly listed
in `.claude/settings.json` → `permissions.allow`, which means **the project's
permission settings are not being applied to the unattended run**.

That is the exact failure mode `scripts/fix-trust.ps1` was written for: Claude
Code keys folder trust by path in `%USERPROFILE%\.claude.json`, the desktop app
writes it with backslashes and the CLI reads it with forward slashes, and an
untrusted workspace "ignores its permission settings". The header of that
script predicts this symptom precisely.

Consequences:

- **Ship gate 1** (`pytest`) — cannot be run.
- **Ship gate 2** (`run_screener.py --dry-run`) — cannot be run.
- Therefore nothing may be merged to `main` this session, per `CLAUDE.md` rule 1.
- Research days cannot do literature research, which is the entire deliverable.

**Fix (one command, run once, interactively):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fix-trust.ps1
```

It is idempotent, backs up `.claude.json` first, self-verifies, and restores
the backup on failure.

This was not attempted from inside the session: it writes to
`%USERPROFILE%\.claude.json`, outside the working directory, and PowerShell
execution was itself denied.

**Secondary:** the scheduled-task definitions are not in version control.
`grep -rn "Register-ScheduledTask" .` returns nothing. The 2:00 AM and 6:00 AM
triggers — the thing the entire two-loop premise depends on — exist only as
hand-made entries in Windows Task Scheduler on one machine, undocumented and
unreproducible. Worth committing a registration script.

> **Resolved since this note was written** (added 2026-08-27; the paragraph
> above is left as it stood on 2026-08-10). `scripts/register-tasks.ps1`
> shipped 2026-08-21 and the grep now finds it. The related gap this note
> gestures at — that a loop which never fires leaves no trace — was closed
> 2026-08-27 by `scripts/check_loop_health.py` and
> `.github/workflows/loop-watchdog.yml`, which read the heartbeat from GitHub
> rather than from the machine that may be the thing that is off.

**Also observed:** the data loop did not run today (Monday 2026-08-10). Logs
exist for `datarun-2026-08-06` and `datarun-2026-08-07` and for
`nightly-2026-08-10_060040`, but there is no `datarun-2026-08-10`. The 6:00 AM
code loop fired and the 2:00 AM data loop did not, on a day both were
scheduled. The 2026-08-07 run started at 02:11:38 — 11 minutes late — which is
consistent with a wake-from-sleep delay. A missing `WakeToRun` /
`StartWhenAvailable` on the data-loop task is the likely cause and would be
fixed by the registration script above. Task Scheduler could not be queried
from this session (`schtasks` denied).
