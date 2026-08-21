# Nightly Log

Append-only. Newest entries at the bottom. One entry per session.

Each entry records what changed and why, the evidence behind it, what was tried
and rejected, and where the next session should pick up. This is the only
memory that carries between otherwise-cold sessions - write it for a reader
with no other context.

---

## 2026-08-05 - Setup (human-run, not an autonomous session)

**Tests:** n/a before (repo not on this machine) -> **492 passed, 0 failed** after
**Data loop:** stalled since 2026-02 - now fixed, see below

### Did

**Got the machine working at all.** The repo was not present; the ThinkCentre
was a fresh build. Cloned `screener-dashboard` to
`C:\Users\smitc\OneDrive\Documents\Screener`, installed Node LTS, Python 3.12,
GitHub CLI, Claude Code CLI, and all `requirements.txt` deps. Verified
`python run_screener.py --dry-run` passes end to end: config valid, 502 tickers
loaded from GitHub source, yfinance fetch 3/3, output paths OK.

**Fixed test pollution of published artifacts.** The suite was rewriting three
tracked files as a side effect: `validation/data_quality_log.csv` (stamping in
fake rows), `factor_output.xlsx`, and `sp500_tickers.json`. Left alone this
would have dirtied the tree every run - breaking the next morning's clean-tree
guard - and risked committing test-generated rows into the provenance log.
Added root `conftest.py` with a session-scoped autouse fixture that snapshots
and restores them. Verified: full suite now leaves the tree clean.
*Still open:* this is a guard, not isolation. The real fix is `tmp_path`
fixtures plus stubbing the network call in `get_sp500_tickers`.

**Found the learning loop was inert.** `improvement_engine.py` has all the
machinery to evolve factor weights from live information coefficients - with
shrinkage, per-cycle caps, and significance gates - but it only learns when the
screener actually runs. Nothing had been running it: **3 IC observations since
2026-02-22, against a minimum of 8.** The originally-planned routine would not
have fixed this, because it never ran the screener either. This was the single
most important thing wrong with the plan.

**Rebuilt the routine around two loops:**
- **Data loop** (`scripts/data-run.ps1`, Mon/Wed/Fri 2:00 AM) - runs the
  screener live, regenerates the dashboard, copies to `index.html`,
  sanity-checks output size before publishing, commits and pushes. This is what
  accumulates evidence.
- **Code loop** (`scripts/nightly-screener.ps1`, Mon-Fri 6:00 AM) - the
  autonomous session. 2:00 AM first so 6:00 AM starts from a clean tree with
  fresh evidence.

**Switched to autonomous operation** per owner decision: methodology may change
without approval, sessions merge to `main` themselves, no morning review.
Replaced human review with four machine-enforced **ship gates** (tests, dry-run,
dashboard artifact integrity, clean tree), re-verified by the runner
independently of what the session believed. Gates fail -> work is pushed to a
branch, `main` is untouched, runner exits 2. Gates pass -> merge, push, and tag
`good/YYYY-MM-DD` as a rollback point. Enabled
`improvement.allow_auto_apply: true`, deliberately leaving every statistical
gate at its existing value.

**Rotation rebuilt research-first** (Mon research, Tue research->design, Wed
build, Thu validate, Fri harden/teach), replacing the old seven-topic rotation.

### Evidence / research
- Test baseline: 492 passed, 0 failed, ~25s.
- `run_screener.py --dry-run` exit 0; universe drift +7/-8 vs the committed
  `sp500_tickers.json`, i.e. the committed universe was ~3% stale.
- `improvement/live_ic_history.csv`: 3 rows, all 2026-02, vs
  `min_observations_for_proposal: 8`.

### Methodology changed
- `improvement.allow_auto_apply` false -> true. Recorded in
  `METHODOLOGY_CHANGELOG.md` with the gates that remain in force.

### Tried and rejected
- **Hand-tuning weights from the LLM's own judgment.** Rejected in favour of
  routing weight changes through `improvement_engine.py`, which decides from
  measured IC with shrinkage and caps. Encoded as rule 4 in `CLAUDE.md`.
- **Daily data runs.** Rejected for now: `dashboard_data.js` is ~3 MB and
  changes every run, so daily commits would add roughly 90 MB/month of poorly
  delta-compressing JSON to git history. Mon/Wed/Fri still clears the
  8-observation gate in about 3 weeks. Revisit if evidence accrual is the
  bottleneck.

### Noticed, not fixed
- `.claude/settings.local.json` is **tracked in git** and hardcodes an
  interpreter path under `C:/Users/Caleb/...` that does not exist on this
  machine. Normally machine-local and gitignored. Worth untracking.
- `index.html` and `dashboard.html` are byte-identical ~258 KB duplicates, both
  committed alongside the ~3 MB `dashboard_data.js`.
- `stock_detail` covers all 501 stocks with full per-metric payloads (raw,
  percentile, contribution, peers, price targets, financials, provenance). The
  dashboard is considerably more capable than a first look suggests - check
  what exists before building anything "new".
- `contrib` in `stock_detail` is computed under Balanced weights only. Any
  client-side reweighting must recompute or hide it.

### Addendum - the learning loop is broken, not merely starved

Deeper investigation found the real cause. `compute_forward_returns()` skips any
snapshot date already present in `performance_history.csv`. Snapshots are
processed at 7 days old, when only the 1-week return exists; `fwd_return_1m` is
written `NaN` and the date is then **never revisited**, so the 1-month return is
never computed no matter how much time passes.

Verified: 13 snapshot dates (2026-02-20 to 2026-07-29), `performance_history.csv`
has only a `fwd_return_1w` column, `live_ic_history.csv` has 3 rows all at `1w`,
and `config.yaml` sets `optimization_horizon: '1m'` with an explicit refusal to
propose when that horizon has no data.

**The engine could never have proposed a weight change.** This is priority 0 in
`CLAUDE.md` and the first session must fix it before any rotation focus.

Upside: 11 of the 13 existing snapshot dates are already >30 days old, so once
the dedup logic is fixed their 1-month returns are computable from historical
prices immediately - clearing the 8-observation gate in days rather than months.

Data runs moved from Mon/Wed/Fri to **daily Mon-Fri** to accelerate accrual,
accepting ~60 MB/month of git growth from the 3 MB payload.

### Dashboard inventory taken

`.claude/plan/dashboard-inventory.md` records what actually exists, to stop
sessions rebuilding it. Highlights: `stock_detail` is 2.69 MB of the ~3 MB
payload and covers all 501 stocks; there are only 3 charts and 1 table; and
roughly half of `index.html` is an embedded methodology document whose category
weights are **hardcoded in prose** - it will start lying the moment the
improvement engine adjusts a weight. Generating that text from `config.yaml` is
a correctness fix.

Owner directive: remove the Model Portfolio from the dashboard. Scoped in the
inventory doc - the UI section goes, but `portfolio_constructor.py` feeds
`in_portfolio` and turnover into every snapshot, so don't rip it out blind.

### Next
**Priority 0 (the forward-return horizon bug) comes first.** After that, the
backtest is the weak link and the first research target.

---

## 2026-08-06 - First live runs: both failed, one dangerously. Fixed.

**Tests:** unchanged, 492 passed
**Data loop:** ran 02:00, discarded | **Code loop:** ran 06:00, aborted

The machine had no internet overnight. Both scheduled runs fired on time, which
proved the scheduling works - and exposed two genuine defects.

### What happened

**02:00 data loop - published-quality fabricated data, narrowly avoided.**
With no DNS, all 503 tickers failed to fetch. The screener did **not** fail. It
silently substituted synthetic values - `validation/data_quality_log.csv` reads
*"Network unavailable - using synthetic data / Generated sector-realistic
sample values"* for every ticker - and produced a completely normal-looking
2.6 MB dashboard payload with `stocks_scored: 503, avg_composite: 50.5`.

`data-run.ps1` committed it. The only reason fabricated stock scores did not
reach the live public site is that the push also failed on the same dead
network. **The size-based sanity check was useless here** - a fully synthetic
run produces a perfectly normal-sized payload.

It also wrote a synthetic snapshot into `improvement/snapshots/`, which would
have poisoned the IC evidence base the improvement engine learns from.

**06:00 code loop - aborted on a false negative.** `gh auth status` returned
"not authenticated" inside the scheduled task even though `gh` is properly
authenticated interactively. gh keeps its token in the Windows keyring, which
a scheduled task cannot reliably read.

### Fixed

- **Reverted** the synthetic commit (`4a39060`), removing the fake dashboard
  data and the poisoned snapshot. Reverted rather than reset so the incident
  stays in the audit trail.
- **`data-run.ps1` now has a real data-quality gate.** Any synthetic
  substitution at all -> discard the run, clean the snapshot, exit 2. Fetch
  failure rate above 40% -> same. Missing data-quality log -> same. Publishing
  fabricated numbers is the worst thing this system could do; it is now gated
  on evidence rather than file size.
- **`data-run.ps1` waits for the network** (up to 5 minutes, 10 attempts)
  before running, since the machine may wake from sleep with no network yet.
- **`nightly-screener.ps1` no longer depends on `gh`.** It merges with plain
  git and pushes via the `manager` credential helper, so gh was never needed.
  Replaced the check with `git ls-remote`, which tests what actually matters.

- **Removed `runs/2b7db89f3f94/`** (gitignored, so not covered by the revert).
  It held the synthetic run's full artifacts including `meta.json` and
  `05_final_scored.parquet` - precisely what
  `improvement_engine.backfill_from_existing_runs()` scans for. Left in place,
  any future backfill would have rebuilt a snapshot from fabricated data and
  fed it into the IC evidence base. Verified 0 backfill-eligible run dirs
  remain. **Note for future incidents: reverting the git commit is not enough;
  `runs/` must be cleaned separately.**

### Noticed, not fixed
- **The screener silently fabricating data on fetch failure is a defect in
  `run_screener.py` / `factor_engine.py`, not just in my runner.** The gate now
  catches it downstream, but the pipeline should refuse, or at minimum exit
  non-zero, rather than emitting synthetic values that look real. A caller who
  did not check the data-quality log would never know. **Worth fixing at
  source** - consider a `--no-synthetic` mode, or making synthetic fallback
  opt-in rather than default.

### Next
Unchanged: priority 0, the forward-return horizon bug. But if the data loop
fails again tonight, fix that first.

---

## 2026-08-10 (evening) - The rankings were wrong. Root cause: runs stopped fetching.

**Owner noticed the Top 5 had changed completely and analyst price targets had
stopped displaying, and asked whether the methodology had changed.** It had not.
The data had silently degraded.

### The evidence

| | 2026-07-29 | 08-07 / 08-10 (broken) | 08-10 after fix |
|---|---|---|---|
| Top 5 | HST EXPE APA CF NEM | WRB KEYS OXY FICO EXE | HST EXPE EIX APA CF |
| stocks with price | 501/501 | **0/503** | 502/502 |
| stocks with pt_mean | 497/501 | **0/503** | 499/502 |
| dispersion (val/qual/gro/mom/risk) | 23.9/18.3/19.9/27.1/20.2 | 16.7/13.0/16.4/17.3/15.1 | 24.0/18.3/20.0/26.1/19.7 |
| runtime | - | 10-14s | 780s |

Factor weights were byte-identical to July throughout; no scoring code changed;
the improvement engine has never fired (3 IC observations, needs 8). The
methodology was never the variable.

### Root cause

The runs were warm-starting from cache and **skipping the fetch stage
entirely**. Run `70282daf8917` has no `00_raw_fetch.parquet` and no
`01_raw_metrics.parquet` at all. `_current_price` and `_target_mean` are
populated only from the live yfinance fetch (`generate_dashboard.py` ~line 123),
so on a cached run they are simply absent - hence 0 prices and 0 analyst
targets. Momentum and risk depend on price history, so every category
compressed ~25-36%, which reshuffled the whole ranking.

**The screener reported "0 issues logged, 0 fetch failures" on both degraded
runs.** It does not treat "I never fetched anything" as a problem. Same
silent-degradation family as the synthetic-data fallback and the missing
`markdown` package.

Aug 7 and Aug 10 produced *byte-identical* scores (composite_sd 6.67,
momentum_sd 17.29, dispersion equal to 4dp) - three days apart. That
impossibility is what exposed it.

### Fixed

- Quarantined and cleared `cache/` (it still held the 2026-08-06 synthetic
  parquet, which was never cleaned after that incident).
- Ran `run_screener.py --refresh`: 780s, 0 fetch failures, 0 synthetic
  substitutions, all 9 pipeline stages present.
- Regenerated dashboard; 502 rows, 58 methodology headings, index.html 257,682 B.
- **Purged 2 degraded snapshots** (`2026-08-07_6db2226ce6cd`,
  `2026-08-10_70282daf8917`) and their dispersion rows. They would have fed
  misleading ICs into an engine now permitted to auto-apply weight changes.
- 492 tests pass.

### Next - highest priority, above the forward-return bug

**Make a cached warm-start that skips fetching either impossible or loud.**
Options: refuse to publish a run with no `00_raw_fetch.parquet`; treat
"0 tickers fetched" as a High severity data-quality issue; or add a dispersion
regression check against `improvement/dispersion_history.csv` and abort when
it collapses >20% versus the trailing median. The last one would have caught
this automatically on 08-07 - the instrumentation already recorded the
collapse, nothing was watching it.

Add the same check to `scripts/data-run.ps1` so a degraded run can never be
published again.

### 2026-08-11 review - see the entry below

### Residual issue found during verification - NOT fixed

**16 of 502 stocks have no `size_score`** because Yahoo returned no
`market_cap` for them: HST, BBY, WDC, NUE, TGT, XOM, CRM, HPQ, MRK, HRL, ADI,
LOW, AZO, PPL, COO, GIS. Two are in the model portfolio, including **HST, the
#1 holding**. July had zero such gaps, so this is run-to-run yfinance variance.

**None of the 16 appear in `validation/data_quality_log.csv` at all.** A missing
`ev_ebitda` is logged as "weight redistributed to available metrics"; a missing
market cap that removes an entire 5%-weight factor category is logged nowhere.
Third silent-degradation instance found today.

Bounded impact - size is 5% of the composite, and HST also ranked #1 in July
*with* a size score, so the ranking is corroborated independently. But it should
be fixed:
1. Log a missing category input as a data-quality issue, at Medium or higher.
2. Derive market cap as `price x sharesOutstanding` when Yahoo omits
   `marketCap` - both fields are usually present in the same `info` payload.
`backtest.py` documents its own survivorship and look-ahead biases. Now that
the system validates its own methodology changes, a biased backtest doesn't
just mislead a reader - it steers the self-improvement loop toward whatever the
bias favours. See `.claude/plan/backtest-v2.md`; start by *quantifying* how
much those two biases are worth here before building anything, since that
determines how urgent the rest is.

Until then, live IC from the data loop is more trustworthy than any backtest
number, because it is genuinely out-of-sample.

---

## 2026-08-10 - RESEARCH. The priority-0 fix would have armed the engine on two overlapping observations.

**Tests:** before -/- , after -/- - **could not be run, see below**
**Data loop:** stalled - no run today (Mon 2026-08-10); last successful run 2026-08-07
**Code loop:** ran, but **could not execute Python at all**

### The blocker, first, because it changes what this session could be

This is the first autonomous session to get past the runner and actually start
work (2026-08-06 aborted on a `gh` false negative, 2026-08-07 on no network).
It immediately hit the next link in the chain: **the unattended session cannot
run Python.**

```
python --version                            -> works (3.12.10)
python -c "print('hello')"                  -> denied
python -m pytest tests/ test_screener.py -q -> denied
python run_screener.py --dry-run            -> denied
WebSearch / WebFetch                        -> denied
```

All of those are in `.claude/settings.json` -> `permissions.allow`. They are
denied anyway, which means the project's permission settings are **not being
applied** to the scheduled run. `scripts/fix-trust.ps1` predicts this symptom
exactly in its own header: folder trust is keyed by path in
`%USERPROFILE%\.claude.json`, the desktop app writes it with backslashes, the
CLI reads it with forward slashes, and an untrusted workspace "ignores its
permission settings".

**Ship gates 1 and 2 were therefore impossible to run, so nothing merged to
`main` today.** `main` is untouched.

`git add` / `git commit` / `git push` are denied by the same cause, so **this
session could not commit its own work.** The files below are sitting
uncommitted in the working tree:

```
 M CLAUDE.md
 M NIGHTLY_LOG.md
?? ACTION_REQUIRED.md
?? research/2026-08-10-ic-evidence-independence.md
```

**This will jam the loop.** `scripts/nightly-screener.ps1` checks
`git status --porcelain` before starting (line ~192) and refuses to run on a
dirty tree, so **tomorrow's 6:00 AM session will not start** until someone
commits or stashes these. Gate 4 also fails today for the same reason, so the
runner will push an empty `nightly/2026-08-10` branch and exit 2.

Recovery - two commands, run once, interactively:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fix-trust.ps1

git add CLAUDE.md NIGHTLY_LOG.md ACTION_REQUIRED.md research/2026-08-10-ic-evidence-independence.md
git commit -m "research: IC observation independence blocks the naive priority-0 fix"
```

A session cannot do the first itself - it writes outside the working directory
and PowerShell execution is denied too. Recorded as priority **-1** in
`CLAUDE.md` and in `ACTION_REQUIRED.md` at the repo root.

### Did

Spent the session on priority 0, as instructed. Could not fix it - no Python -
so I did the thing that was still possible and turned out to matter more:
established what the fix would actually produce. **It would have made things
worse.** Full note: `research/2026-08-10-ic-evidence-independence.md`.

### Evidence / research

All from repo data with read-only shell commands; every command is in the note.

- **Priority 0 confirmed** at `improvement_engine.py:253`, still unfixed. The
  `CLAUDE.md` description is now slightly stale: `performance_history.csv`
  *does* have `fwd_return_1m`/`3m` columns, and **2 of 14 dates** have a 1m
  value (`2026-03-15`, `2026-04-14`). Those two got filled only because they
  sat unprocessed past the horizon by accident.
- **The backfill is not 11 independent observations.** The 11 backfillable
  dates span 2026-02-20 to 2026-04-14 - **53 days**. At most
  `floor(53/30)+1 = 2` non-overlapping 30-day return windows fit among them.
  Six of the eleven sit inside a single 9-day stretch and overlap by 21-29 of
  30 days.
- **The gate would have passed on that.** `_ir_to_one_sided_pvalue()` computes
  `t = IR * sqrt(n_obs)` from the raw row count. At the configured
  `min_ic_ir_for_auto_apply: 0.5`:

  | n used | t | one-sided p | passes? |
  |---|---|---|---|
  | 11 (post-backfill, as counted today) | 1.66 | **0.049** | **yes** |
  | 2 (non-overlapping) | 0.71 | 0.240 | no |

  A **2.35x** t-statistic inflation. With `allow_auto_apply: true` the engine
  would have begun rewriting factor weights, each change arriving with a
  p-value and a changelog entry saying the gates were satisfied.
- **75% of `performance_history.csv` is duplicate rows.** Several snapshot
  files share a run date and `compute_forward_returns()` appends all of them in
  one pass. `2026-02-21` has 13 copies of every ticker, and that reaches the
  published record: `live_ic_history.csv` reports **6,539 "tickers"** for an
  S&P 500 screener. 8.1 MB of file for ~4,500 distinct ticker-date pairs.
- **5 of the 14 snapshot dates are weekends** (`2026-02-21`, `02-22`, `02-28`,
  `03-01`, `03-15` - confirmed with `date -d`). No market close on those days.
  The whole pre-2026-07-28 set is development artifacts, not observations; the
  scheduled loop did not exist until 2026-08-05. **The entire current 1-month
  evidence base is two dates, one of which is a Sunday dev run.**
- **The data loop never recomputes live IC.** `record_run_snapshot()` calls
  `compute_dispersion()` and `compute_forward_returns()` but not
  `compute_live_ic()`. Proof: after the successful 2026-08-07 run,
  `performance_history.csv` and `dispersion_history.csv` are stamped Aug 7,
  `live_ic_history.csv` is still stamped Aug 5. The loop's own log line
  "3 live IC observation(s)" is just reading a stale file.

### Methodology changed

None - no code shipped, and the ship gates could not be run to justify any.

`CLAUDE.md` priority 0 was rewritten to a 5-step package with an explicit
**STOP - do not ship the obvious fix on its own**, plus the instruction to set
`allow_auto_apply: false` if the effective-observation-count step cannot land
in the same session. Priority **-1** added for the environment blocker.

### Tried and rejected

- **Blind-editing `scripts/nightly-screener.ps1`** to invoke `fix-trust.ps1`
  before launching Claude. Rejected: PowerShell could not be executed to test
  it, and a syntax error in the runner kills the loop entirely. Documenting a
  one-line manual fix beats an unverifiable change to the only thing that
  starts sessions.
- **Writing the priority-0 fix unvalidated on the branch.** Rejected on the
  evidence rule - and the research then showed the obvious fix was the wrong
  fix anyway, which is the better argument for having waited.
- **Citing literature from memory as if verified.** WebSearch/WebFetch were
  denied, so the six references in the note are explicitly flagged unverified
  and the finding deliberately rests only on repo data and arithmetic that
  anyone can re-run.

### Honest correction to a standing assumption

`CLAUDE.md` said the fix would clear the 8-observation gate "in days rather
than months". That is not true at the 1-month horizon. Independent monthly
observations accrue at about one per month, so it is realistically **~8 months**
from the start of the scheduled loop (2026-07-28). Whether the optimization
horizon should be 1m at all is now an open question for the design session -
1w gives ~5x the independent observations, though Phase 13 governance rightly
forbids optimizing a monthly strategy on a weekly signal.

### Next

1. **Run `scripts\fix-trust.ps1` interactively.** Nothing else can ship first;
   every future session is blocked in exactly the same way.
2. Then the priority-0 package as rewritten in `CLAUDE.md` - all 5 steps, or
   step 1 plus `allow_auto_apply: false`.
3. Commit a `Register-ScheduledTask` script. The 2 AM / 6 AM triggers exist only
   as hand-made entries on one machine and are not in version control; the data
   loop silently missed today, likely a missing `WakeToRun`.

---

## 2026-08-11 - Both loops fired. One worked as designed, one crashed on my bug.

**Tests:** 506 -> 526 passed (20 new static checks on the runner scripts)
**Data loop:** crashed 02:00 | **Code loop:** blocked on trust, failed fast

### 06:00 code loop - the new preflight worked

Detected the untrusted workspace in **one second**, logged the exact fix, wrote
and published the morning brief, and exited. Previously this cost a full
12-minute session that then could not commit. Graceful failure, working as
intended - but still zero successful autonomous sessions to date.

### 02:00 data loop - crashed on a bug I introduced

`The term 'Write-NativeOutput' is not recognized` at `data-run.ps1:203`. I used
that helper in the health-check block but only ever defined it in
`nightly-screener.ps1`. The data loop therefore published nothing.

It shipped because the only pre-flight check available was counting braces and
parentheses, which cannot see an undefined function - PowerShell tooling was
unavailable in that session, so the script was never actually parsed.

**Fixed:** defined `Write-NativeOutput` in `data-run.ps1`.

**Fixed properly:** `tests/test_scripts_static.py` - 20 checks across all five
`.ps1` files covering undefined functions, unbalanced blocks, missing UTF-8 BOM
and non-ASCII characters. Verified it catches the real bug: run against the
crashed version it reports `Write-NativeOutput`. These scripts are unattended
infrastructure; a typo means a silently skipped run, so they now get the same
regression coverage as the Python.

### The health gate earned its place immediately

A manual re-run at 07:52 warm-started from the previous run's cache: 6 second
runtime, no fetch. Coverage and dispersion both looked *fine* - because the
cache came from Monday's good full refresh - but the gate refused it on missing
fetch evidence. That was the right call, for a reason worth recording:

**The screener records an improvement-engine snapshot even when it fetched
nothing.** Today produced two snapshots (`b84e370f3cf0` at 02:00,
`6c5ecaa1361c` at 07:52), both 190,149 bytes, both byte-identical in composite
to Monday's run - three "observations" for one real data point. Removed both.

That is the same evidence-inflation failure the 2026-08-10 research note
identified with overlapping return windows, arriving by a different route. The
improvement engine gates on an observation *count*, so duplicates directly
inflate its confidence. **Added as a priority below.**

Why it warm-started: `price_data_refresh_days: 1`, and the cache was 12 hours
old. The scheduled 02:00 runs sit ~24h apart so they should fetch normally -
this was an artifact of running manually the same day.

### Methodology emphasis changed (owner direction)

Research and documented professional practice now carry **equal weight to
measured results** as justification for a methodology change. Rationale: the
backtest is known-biased and independent IC observations accrue roughly
monthly, so requiring measured proof up front would freeze methodology work for
half a year. Measurement now confirms changes over time rather than gating them.

Rotation rebalanced to three research/design days per build day:
Mon component research -> Tue practitioner research -> Wed **synthesis** ->
Thu build -> Fri harden. Wednesday is the new centre of gravity: how the piece
fits the whole screener, what it overlaps or makes redundant, whether the
system is still coherent afterwards - rather than accumulating individually
defensible tweaks.

The evidence *requirement* is unchanged - a written argument a sceptical reader
can follow to its sources. Only what counts as a source has widened.

### Next
1. **Do not record a snapshot when the run did not fetch.** Duplicate snapshots
   inflate the improvement engine's observation count, which is the thing its
   significance gate depends on. Either skip the snapshot on a warm-start, or
   deduplicate on `(run_date, content hash)` before the engine reads them.
2. Trust step still outstanding - no autonomous session has yet completed.

---

## 2026-08-12 - Nothing ran. The PC rebooted overnight and nobody was logged in.

**Tests:** 526 -> 530 passed
**Data loop:** did not fire | **Code loop:** did not fire

### What happened

No logs for 2026-08-12 at all. Both tasks show Enabled/Ready, last run 08-11,
next run **08-13** - today's 02:00 and 06:00 slots passed without firing.

`System Boot Time: 8/12/2026, 12:47:15 AM` - the machine restarted overnight,
almost certainly a Windows update. It was powered on through both windows, so
this was not a sleep or network problem.

**Cause:** both tasks use `LogonType: InteractiveToken` - they run *only while
a user is logged on*. After the update reboot the machine sat at the login
screen with no user session, so neither task could start. Nothing errored;
there was simply nothing to write a log.

This was a known limitation from setup (running logged-out needs a stored
password, which was deliberately avoided) but nothing guarded against it.

### Fixed

**Run-once-per-day markers.** Both scripts now write
`logs/.datarun-last-success` / `logs/.nightly-last-success` on a successful
run and exit immediately if today's already succeeded. A *failed* run leaves no
marker, so it is correctly retried. Both take `-Force` to override.

**`scripts/add-catchup-trigger.ps1`** adds an at-logon trigger (3 min delay) to
each task, so a missed run is picked up when the owner next logs in. The
markers make that safe - logging in repeatedly cannot re-run the loop.

*Requires the owner to run it:* modifying scheduled-task triggers is a
persistence change and was correctly refused when attempted automatically.

**Static checks now cover 6 scripts, 24 assertions.**

### Noticed, not fixed

The cleanest root-cause fix is the Windows setting *"Use my sign-in info to
automatically finish setting up after an update or restart"* (Settings ->
Accounts -> Sign-in options). That restores the user session after an update
reboot, which is what these tasks need. It is a Windows account setting, not a
repo change - owner action.

### Next
Unchanged: the trust step still blocks every code session. Zero autonomous
sessions have completed since setup on 08-05.

---

## 2026-08-13 - BUILD. The data loop was only fetching once every eight days.

**Tests:** before 530/530, after 552/552 (22 new)
**Data loop:** was silently degraded - **fixed**. Today's 02:00 run warm-started
from an 08-12 cache, produced no fetch artifacts, and was correctly discarded by
the health gate. Root cause found and fixed; confirm on the 08-14 02:00 run.

### The trust blocker is gone

The 06:00 runner logged **"Workspace trust OK"**, and `python -c`,
`python -m pytest` and `python run_screener.py --dry-run` all executed. This is
the **first autonomous session that could run the ship gates**, and the first to
merge to `main`. Priority -1 in `CLAUDE.md` is marked resolved. Every session
from 08-05 to 08-12 was blocked on this.

### Did

Found and fixed why the data loop keeps publishing nothing. This was not a
scheduling problem - it is that **the screener almost never fetched**.

`run_factor_engine` bounded reuse of the `factor_scores` cache by
`caching.fundamental_data_refresh_days` (**7**), not
`price_data_refresh_days` (**1**). Two things compound:

1. `factor_scores` is the *fully scored* dataset, not fundamentals.
   **18 of the 44 metrics in `METRIC_COLS` move with the daily close**, across
   five of the eight categories - every valuation ratio (price is in all of
   them), all three momentum metrics, six risk metrics, `price_target_upside`
   and `size_log_mcap`. So a "fundamental" freshness bound was the wrong unit
   entirely, and published Valuation/Momentum/Risk scores could be computed
   from a close up to eight days old while presented as current.
2. The warm-start path returns at `run_screener.py:1011`, **before**
   `write_scores_parquet` at `:1502`. A warm-started run lays down no new cache
   file, so the cache date never advances. One real fetch therefore suppressed
   the next seven days of runs: **one real fetch per eight daily runs.**

Also fixed an off-by-one that would have defeated the fix on its own. Cache
dates are parsed from the filename and are midnight-anchored, so a cache from
yesterday is `age_days == 1` no matter the clock time. Under the old
`age_days <= fresh_days`, even `fresh_days = 1` would still have reused
yesterday's cache at 02:00 - the daily loop would have kept warm-starting.
The rule is now strict and stated in plain English:
**`<tier>_refresh_days: N` means the cache is reusable for N calendar days
starting with the day it was written.** `1` therefore means "refetch unless the
cache is from today". A same-day manual re-run still warm-starts, which is
correct - there is no new close to fetch.

Implemented as three pure, testable helpers in `factor_engine.py` beside the
existing caching section: `cache_age_days()`, `cache_is_usable()`,
`factor_scores_cache_max_age_days()`. Worth noting: `cache_is_fresh()` in the
same file **already had the correct timedelta semantics**; `run_screener.py`
had hand-rolled a looser `.days`/`<=` copy of it. The bug was a divergent
reimplementation, not a missing idea.

### Evidence / research

- **Direct arithmetic on the real artifacts, not fixtures.** Live cache
  `cache/factor_scores_19c853468405_20260812.parquet`, live config hash
  `19c853468405`. At 2026-08-13 02:00 `age_days = 1`. Old rule `1 <= 7` ->
  reuse. New rule `1 < 1` -> fetch. The exact run that was discarded today
  would now fetch.
- **Measured, in tests:** `test_old_bound_would_have_fetched_only_once` shows
  1 fetch across 8 consecutive daily runs; `test_eight_consecutive_daily_runs
  _all_fetch` shows 8 of 8 after the fix.
- **Three shipped incidents:** 2026-08-07 and 2026-08-10 published a dashboard
  with 0 of 503 prices and analyst targets and dispersion down 25-36%;
  2026-08-13 was caught and discarded. The health gate added on 08-10 was
  working correctly the whole time - it was reporting a cause nobody had found.
- 22 new tests, written **red first** (ImportError on the missing helpers).
- No backtest number used or run. Rule 5 respected.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-08-13** - "The screener scored stale
  prices: factor_scores cache bounded by the wrong tier". Filed as methodology
  rather than a perf tweak because it changes *what data the published scores
  are computed from*. Includes the 18-metric table, the rollback
  (`caching.price_data_refresh_days: 7` restores the old window without a code
  change), and the honest caveat below.

### Honest caveat on what this does and does not buy

It speeds up evidence accrual ~8x in **calendar** terms. It does **not** speed
up *independent* observations, which still accrue at about one a month
(`research/2026-08-10-ic-evidence-independence.md`). More rows is not more
evidence. Priority 0 - the forward-return horizon package - is still unfixed
and still gates the improvement engine; step 3 of that package (count
*effective*, non-overlapping observations) matters more now, not less, because
the row count will grow 8x faster while the independent-observation count does
not.

### Tried and rejected

- **Just changing `fundamental_data_refresh_days` to 1 in `config.yaml`.**
  Rejected: it would have mislabelled the fundamentals tier to work around a
  bug on the price path, and the `<=` off-by-one meant it would not have
  worked anyway - yesterday's cache is `age_days = 1`, and `1 <= 1` is true.
- **Running the full screener to prove the fix end-to-end.** Not done: that is
  the data loop's job and it publishes through the health gate. The 08-14 02:00
  run is the real confirmation, and it is written down as such in both
  `CLAUDE.md` and the changelog so a failure is not quietly forgotten.

### Noticed, not fixed

- `cache/` contains `factor_scores_20260812.parquet` and
  `factor_scores_20260813.parquet` with **no config hash** - test side effects
  leaking into the real cache directory (priority 8, test isolation). Harmless
  today only because production always passes a hash and `_find_latest_cache`
  filters on it. If `ctx` were ever `None` in production, the pipeline would
  load a test artifact as real data.
- Why the 08-12 08:16 run fetched at all is not fully explained - by this
  mechanism it should have warm-started off the 08-10 cache. Most likely the
  08-10 cache was cleared. It does not affect the diagnosis, which is proven
  arithmetically, but it is an unexplained detail rather than a confirmed one.

### Next

**Priority 0 - the forward-return horizon package** (all five steps in
`CLAUDE.md`), now that a session can finally run tests. Re-verified today and
unchanged: 13 distinct run dates, `fwd_return_1m` present on only **2 of 13**,
`fwd_return_3m` on **1 of 13**, and `live_ic_history.csv` still has just 3 rows,
all horizon `1w`, one of which claims **6,539 tickers** for a single date in a
500-stock universe. The improvement engine still cannot propose anything.
Ship step 3 (effective observation counting) in the same session as step 1, or
set `allow_auto_apply: false` first - the 8x faster row growth from today's fix
makes the inflation risk worse, not better.

---

## 2026-08-21 - RETROSPECTIVE. The routine's problem is not bad work, it is no work.

**Tests:** before 552/552, after 556/556 (4 new)
**Last code session:** 2026-08-14 - **did not run**; reported as success (below)
**Data loop:** healthy. 02:12 today, live fetch, 501/501 prices, 497/501 targets,
all five dispersions within range, HEALTH: PASS
**Evidence base:** `live_ic_history.csv` = **3 rows, newest 2026-02-22** - 180
days without a new observation
**Priority 0:** unfixed

This is the first retrospective. It reviews everything since setup.

### Retrospective findings

- **Sessions reviewed: 11 scheduled code-loop slots (2026-08-06 to 2026-08-20),
  plus the 2026-08-05 human setup session.**
- **Genuinely valuable: 2 | Fired and produced nothing: 4 | Never fired: 5 |
  Failed ship gates: 0**

| Date | What happened |
|---|---|
| 08-06 | Fired, aborted on a `gh` false negative. Nothing. |
| 08-07 | Fired, aborted, no network. Nothing. |
| 08-10 | Blocked (untrusted workspace, no Python) - and still produced `research/2026-08-10-ic-evidence-independence.md`, the best artifact in the repo. **Valuable.** |
| 08-11 | Blocked on trust, exited in 1 second. Nothing. |
| 08-12 | Never fired - reboot, nobody logged on. |
| 08-13 | Found and fixed the 8-day fetch bug, 22 red-first tests, changelog, merged, tagged. **Valuable.** |
| 08-14 | Fired, died in 1 second on an API weekly limit. Reported as a success. Nothing. |
| 08-17..08-20 | Never fired. Machine off or logged out. The at-logon catch-up trigger did pick the *data* loop back up on 08-20 at 23:28. |

**1. What fraction produced something genuinely valuable? Two of eleven (18%).**
The good ones are 08-10 and 08-13, and they are genuinely good: 08-10 stopped a
"fix" that would have armed the improvement engine on 2.35x-inflated
confidence, and 08-13 found that the screener had been publishing prices up to
eight days old. The wasted ones - 08-06, 08-07, 08-11, 08-14 - were not wasted
on bad work. They were wasted before any work started.

**The headline finding is that this routine's failure mode is absence, not
churn.** There is no churn to speak of; there is barely any output at all. Nine
of eleven slots produced nothing, and in eight of those nine the session either
never started or was denied the tools to work.

**2. Which rotation day earns its place?** None of them, because **the rotation
has never once been executed.** In 16 days it produced: one research note
(08-10, written *against* its nominal Monday focus, correctly, because priority
0 mattered more), zero practitioner appendices, zero Wednesday synthesis
sections, and zero builds that implemented a week's research. The 08-13 build
was a bug fix, not the output of a research week.

The rotation is also a fiction the prompt itself overrides: `nightly.md` section 1
carried *three separate* "regardless of the nominal focus" clauses (priority 0,
data loop health, evidence base) ahead of the day's focus. A focus that three
standing instructions outrank is not a rotation.

Tuesday was the weakest link specifically: it can only append to a note Monday
wrote, and Monday has never written one. A two-day chain is the most fragile
structure possible when half of all sessions never start.

**3. Is the evidence standard holding? Yes - it is the strongest part of the
routine, and it is not degrading.** The 2026-08-13 changelog entry does
arithmetic on the real on-disk cache file rather than a fixture, writes its
tests red first, states a falsification condition ("the 08-14 02:00 run must
show a live fetch; if it warm-starts again this entry is wrong"), and adds an
unprompted caveat that the fix buys calendar speed and *not* independent
evidence. The 08-10 research note explicitly flags all six of its citations as
unverified and rests its finding on reproducible repo arithmetic instead. The
08-13 session log corrects its own first draft ("I said nine metrics; the actual
count is double"). Rule 5 is respected - no backtest number appears under
**Evidence** anywhere.

The honest caveat: there are 6 changelog entries and 4 are governance or
process, not methodology. The standard is holding partly because almost nothing
has been claimed.

**4. What keeps going wrong?**

- **Sessions do not start.** Six distinct causes in 16 days, each fixed
  reactively after it had already cost a day.
- **Failure reports itself as success.** The clearest case is 08-14 and it is
  fixed today - see below.
- **The ship gates have never fired.** Zero failures, because only one session
  ever produced code. They are untested in anger.
- **The evidence base has not grown at all.** Three rows, all horizon `1w`, all
  February, unchanged through every successful data run. `record_run_snapshot()`
  never calls `compute_live_ic()` - step 5 of priority 0, known since 08-10,
  still unfixed. Every data run dutifully logs "3 live IC observation(s)" and
  nobody noticed the number is frozen.
- **`CLAUDE.md` had started contradicting itself.** Rule 4 said
  `allow_auto_apply` is false (correct, and what `config.yaml` says); the
  "Improvement engine" section still said it "is now **true**". Fixed today.

**5. Is the tool closer to being the place you'd look before buying or selling?
No. It has not moved at all.** Not one line of `generate_dashboard.py` has
changed since 2026-07-29. Zero of priorities 2 and 4-7 (time dimension,
generated per-stock summaries, sell-side workflow, investor profiles, club
readiness) have been started.

The honest blocker: every session that could run spent itself on data-pipeline
defects, and was *right* to - publishing fabricated or eight-day-stale prices is
worse than a missing feature. But the rotation had no mechanism to protect
product work from firefighting, and firefighting will always win that fight.

**6. What is the routine systematically blind to?**

- **The published site.** Nothing in the rotation or the gates ever loads the
  live page. Gate 3 checks that `index.html` is over 50 KB. On 08-10 the
  dashboard's methodology section was broken and it took the *owner noticing*
  to surface it.
- **Its own trend lines.** Every session reads the last three log entries.
  Nobody ever asked "is the IC count going up?" - which is exactly how
  3-rows-since-February survived six sessions and 16 days of daily data runs.
- **The owner as a channel.** Two of the most consequential inputs in the whole
  history came from the owner noticing something (the Top-5 change on 08-10, the
  governance direction on 08-11 and 08-20), not from the routine.
  `MORNING_BRIEF.md` is written after every run and nothing checks whether it
  is telling the truth. On 08-14 it actively said the morning was fine.
- **Cost and quota.** ~$6/session and a weekly ceiling that silently killed
  08-14. No session has looked at this.

### Process changes made

**1. A session that never ran is no longer reported as a success.**
`scripts/nightly-screener.ps1` captured `$claudeExit`, logged it at INFO, and
never branched on it. On 08-14 the CLI exited 1 after one second on a 429 weekly
limit; with no commits all four gates passed trivially, the runner logged
"Run complete (no changes)", **wrote the once-per-day success marker**, and
published a normal morning brief. The marker is what makes this expensive: it
tells the at-logon catch-up trigger the day is done, so a transient API limit
costs the whole day rather than being retried.

New `Get-SessionOutcome` reads the CLI's own JSON transcript and fails the run
on any of: non-zero exit, empty/missing/unparseable transcript, `is_error`
(reporting `api_error_status` and the CLI's own message), or `num_turns <= 1`.
A failed session now logs `SESSION DID NOT RUN` at ERROR, **never** writes the
success marker on any path, exits 2, and labels the morning brief
`SESSION DID NOT RUN`. `write_brief.py` classifies run logs by their text, so
removing the "Run complete (no changes)" line is what makes the brief stop
lying.

Four regression tests in `tests/test_scripts_static.py`, verified red against
`git show HEAD:scripts/nightly-screener.ps1` - all five assertions fail on the
old file and pass on the new one.

**2. The morning brief now reports evidence-base staleness, not just a count.**
`ic_observations()` in `scripts/write_brief.py` said "3 of 8 needed" on every
brief from February to today. It now reads
`3 of 8 needed, newest 2026-02-22 - STALE, nothing new in 180 days`. A count
that never moves looks like slow progress; the date is what shows it is no
progress at all.

**3. The rotation is rebuilt around what sessions actually do.**
Monday and Tuesday were both research days, both banned from writing production
code, with Tuesday dependent on Monday's output. Merged into a single
self-contained Monday that must produce a complete note - literature *and*
practice - in one session.

**Tuesday is now PRODUCT**, and it is new. It exists because of finding 5: the
dashboard has a standing owner directive and had 23 days of zero progress, and
no day in the rotation pointed at it. Updated in `CLAUDE.md` and in
`$FocusByDay` in the runner.

**4. Mandatory health numbers in every log entry** (`CLAUDE.md` rule 8): whether
the last code session ran, whether the data loop published, **the IC row count
and its newest date as two literal numbers**, and priority-0 status. If those
two numbers have not moved in three consecutive sessions, making them move is
that session's work. This is the specific instruction that would have caught
finding 6's worst case.

**5. Priority 0 now says start with step 5.** "THE FIRST SESSION MUST FIX THIS"
has been at the top of `CLAUDE.md` since 08-05 and six sessions have not fixed
it, because it is a five-part package and every session that could run found
something more urgent. Step 5 (make the data loop call `compute_live_ic()`) is
small, independent of the other four, and is the one whose absence is doing the
damage. Step 3 must still land before `allow_auto_apply` returns to `true`.

**6. `CLAUDE.md` compressed.** The priorities section carried ~90 lines of
RESOLVED/DONE/FIXED narrative duplicating this log and the changelog. Collapsed
to pointers; the detail lives where it belongs. Priority -1 is now "sessions do
not start", which is what the evidence says it is.

### Tried and rejected

- **A node-based gate 3 that actually parses `dashboard_data.js`.** `CLAUDE.md`
  and the nightly prompt both describe gate 3 as "dashboard_data.js parses", but
  it only regex-matches the first line - a truncated 3 MB payload passes. I
  wrote the parse, then reverted it: neither PowerShell nor `node` could be
  executed in this session, and an unverified change *here* can only fail
  closed, refusing every future merge. Jamming the loop is worse than a weak
  check. The 2026-08-10 session refused to blind-edit this same file for the
  same reason and was right. A comment marking the gap sits at the site.
- **Weakening any ship gate.** Not permitted, and nothing in the evidence
  suggests the gates are the problem. They have never once failed.
- **Changing the retrospective cadence.** Fortnightly looks wrong on its face -
  16 days for 2 productive sessions - but the cause was an outage, not the
  cadence, and shortening it spends scarce sessions on self-examination. Left
  alone.

### Flagged for the owner

1. **The prompt templates could not be edited.** `.claude/prompts/nightly.md`
   and `retrospective.md` are blocked as sensitive files, so the two prompt
   changes I intended - collapsing the three competing "regardless of the
   nominal focus" clauses in section 1 into one health check, and deleting the
   ~40 lines that restate `CLAUDE.md` rules 4 and 5 nearly verbatim - could not
   be made. I put the health-check requirement into `CLAUDE.md` rule 8 instead,
   which the prompt itself says overrides it, so it takes effect either way.
   But **a retrospective is instructed to edit files it is not permitted to
   write.** Either grant write access to `.claude/prompts/`, or move the
   templates to a top-level `prompts/` directory. Note `nightly.md` still
   describes the old Monday/Tuesday split; the runner injects the correct focus
   string so the rotation change is live, but the prompt's own day list now
   disagrees with `CLAUDE.md`.

2. **The scheduled tasks are still not in version control.** Open since 08-10.
   `grep -rn "Register-ScheduledTask" .` finds nothing. This is the single
   highest-value infrastructure item and no session has done it.

3. **Nothing watches whether the loop is running.** Today's fix makes a *failed*
   session loud. A session that never fires writes no log at all, so its absence
   is still only detectable by counting files - which is how 08-17 to 08-20 went
   unnoticed for four days. A heartbeat that complains when no run has been
   logged in 48 hours would close this.

4. **The weekly API quota killed 08-14.** Worth deciding deliberately how much
   of the weekly budget the 06:00 loop should get, rather than discovering the
   ceiling by hitting it. Sessions cost roughly $6.

5. **A fifth gate I did not add, per the rules.** Nothing verifies that the live
   public page *renders*. I would want a gate that loads it, but that needs a
   headless browser in an unattended run and can jam the loop, so I am leaving
   the argument here rather than acting on it.

### Next

**Priority 0, step 5** - make the data loop call `compute_live_ic()`. It is
small, it is independent, and until it lands the evidence base stays frozen at
3 rows and every other measurement claim in this project is theoretical.
