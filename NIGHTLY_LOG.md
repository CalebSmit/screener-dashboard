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

`plan/dashboard-inventory.md` records what actually exists, to stop
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
bias favours. See `plan/backtest-v2.md`; start by *quantifying* how
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

---

## 2026-08-21 (later) - Acting on the retrospective's owner-flagged items

Owner-run session, not a scheduled one. The 2026-08-21 retrospective ended with
five items it could not action itself. Three are now done.

**Health numbers:** last code session **ran and shipped** (06:16, `good/2026-08-21-0616`);
data loop **published** 02:12; `live_ic_history.csv` = **3 rows, newest
2026-02-22**; priority 0 **unfixed**.

### Did

**1. Prompts moved out of `.claude/`.** `prompts/nightly.md` and
`prompts/retrospective.md`. The retrospective could not edit its own templates -
`.claude/` is treated as sensitive - so the two prompt changes it wanted were
left undone and `nightly.md` kept describing a rotation that no longer existed.
A self-improving routine that cannot edit its own instructions is only half a
loop. Runner updated to `prompts\$TemplateName`.

**2. `prompts/nightly.md` rotation synced to `CLAUDE.md`.** It still described
Monday component-research / Tuesday practitioner-research. The runner injects
the focus string so behaviour was already correct, but the prompt body
contradicted it - a session would read one thing in its instructions and
another in `CLAUDE.md`. Now: Mon research (both halves, one session), Tue
product, Wed synthesis, Thu build, Fri harden.

**3. `scripts/register-tasks.ps1`** - the scheduled tasks are in version control
at last. Open since 08-10, called the highest-value infrastructure item. Creates
both tasks from a single definition, idempotent, with the at-logon catch-up
included, and documents *why* `LogonType` is `InteractiveToken` (the alternative
needs a stored password, which this project will not do). Supersedes
`add-catchup-trigger.ps1`.

**4. The brief now shouts when a loop stops.** `write_brief.py` read the newest
log and reported its outcome as if current - so a loop that stopped firing
entirely kept the status page reporting the last successful run. That is how
08-17..20 passed unnoticed and how the machine sat at a login screen for six
days while the brief said everything was fine. Each loop now shows *when it
last ran*, and a `THE ROUTINE IS NOT RUNNING` banner appears above everything
else when either has been quiet for 2+ days. Verified against a synthetic
5-day-old log and a missing log; does not fire for today's runs.

### Corrected

**The retrospective's "~$6/session" is wrong** and is now flagged as such in
`CLAUDE.md`. The owner runs Claude Max: sessions draw on included subscription
usage, not per-session billing. The finding underneath survives - a **weekly
usage ceiling** exists and silently killed 08-14 - but the conclusion is not
"spend less", it is "do not let one loop exhaust the week". Note the 02:00 data
loop consumes none of this quota; only the 06:00 session does.

### Still open from the retrospective

- **A heartbeat that complains when no run has been logged in 48 hours.** #4
  above only helps if *something* runs to write the brief. Genuine absence
  detection needs a watcher outside both loops.
- **A gate that verifies the live page renders.** Argued for, deliberately not
  added: it needs a headless browser in an unattended run and can jam the loop.

### Next
Unchanged: **priority 0, step 5** - make the data loop call `compute_live_ic()`.
Three rows, newest 2026-02-22, 180 days frozen. Every measurement claim in this
project is theoretical until that number moves.

---

## 2026-08-24 - PRIORITY 0, all five steps. The number finally moved: 3 rows -> 23.

**Health numbers:** last code session **ran and shipped** (2026-08-21 06:16,
`good/2026-08-21-0616`); data loop **published** 02:11 today, HEALTH: PASS, 501
scored, 100% price coverage; evidence base **3 rows, newest 2026-02-22, 0
effective observations at the `1m` optimization horizon** -> now **23 rows,
newest 2026-08-14, 2 effective at `1m`**; priority 0 **FIXED**.

**Tests:** before 560/560, after **590/590**
**Data loop:** healthy - `logs/datarun-2026-08-24_020001.log` ends "Data loop
complete", HEALTH: PASS, 0 fetch failures, 0 synthetic substitutions.

### Swapped the rotation focus, deliberately

Today was scheduled as **research**. I did not do research. `CLAUDE.md` priority
0 says the first session must fix the forward-return bug ahead of any rotation
focus, the nightly prompt repeats that override, rule 8's three-session trigger
had fired (the evidence numbers had not moved on 08-13, 08-21 or 08-21-later),
and the previous session named step 5 as the single next thing. Three separate
rules pointed at the same work. **No research note today**; Monday's slot is
owed one, and the next session should take it.

### Did

**All five priority-0 steps, plus a sixth defect found while fixing them.**
Full detail in `METHODOLOGY_CHANGELOG.md` 2026-08-24. In short:

1. `compute_forward_returns()` tracks eligibility per `(run_date, horizon)`, so
   a date is revisited as it ages instead of being frozen at its 7-day state.
2. One snapshot per run date, not one per file.
3. `_effective_observations()` - non-overlapping windows - now feeds every gate.
4. Weekend run dates excluded.
5. `record_run_snapshot()` calls `compute_live_ic()` for all three horizons.
6. **New, found while fixing 1:** the price cache is keyed `(start, end)` and
   `end` was the *current* date. My horizon fix would have turned every
   revisited snapshot into a fresh full-universe yfinance download - about ten
   of them in tomorrow's 02:00 run, against the rate limits that already cost
   the loop 10-25% of its tickers. The fetch window is now bounded by the
   horizon being measured, so the key is stable.

**Ran the real backfill rather than leaving it to discover itself overnight.**
2,495 new rows, 4 tickers failed out of ~500 (HOLX, CTRA, BK, EA - Yahoo
"possibly delisted"). This is why the `1m` horizon has observations at all now.

**Fixed both places that report the evidence count to the owner.** This was a
bug I was about to introduce: the brief and `data-run.ps1` printed the raw row
count, so after the repair they would have said **"23 of 8 needed"** - a
cleared gate that is nowhere near cleared. Both now report effective
observations at the optimization horizon. The brief reads: *"2 of 8 needed at
the 1m horizon (6 rows, but overlapping windows are not independent; 23 rows
across all horizons), newest 2026-08-14"*. Verified by running both.

### Evidence / research

Not literature - a **documented failure**, measured on the live files:

| Claim | Measurement |
|---|---|
| Duplicate rows | 20,057 rows for 8,020 unique `(run_date, ticker)` - 60% duplicates (CLAUDE.md estimated ~75%) |
| Absurd IC inputs | 6,539 "tickers" recorded for 2026-02-21 in an S&P 500 screener |
| Weekend dates | 5 of 16 run dates were Sat/Sun |
| 1m never accrued | 1 of 16 dates carried `fwd_return_1m`, while `optimization_horizon` is `'1m'` |
| IC frozen | 3 rows, all `1w`, all Feb 2026 - 183 days while the loop ran every weekday |

The independence correction implements item 3 of
`research/2026-08-10-ic-evidence-independence.md`. **That note's central claim
is now a test rather than an argument:** against the pre-fix code,
`propose_weight_changes()` returns `proposal_ready` on eleven IC rows that are
two independent observations. `tests/test_evidence_integrity.py` has 30 tests;
**24 fail against the pre-fix file**, verified by checking it out and re-running.

### Methodology changed

`METHODOLOGY_CHANGELOG.md` 2026-08-24. No factor weight, metric or threshold
moved - nothing in the scoring path was touched. What changed is what evidence
the engine can see and act on.

**`allow_auto_apply` stays `false`.** Condition (a) in the `config.yaml`
comment - effective-observation counting - is now met. Condition (b) is not:
there are **2** independent observations against a gate of 8. The engine still
correctly refuses to propose.

### Tried and rejected

**Making the three failing pre-existing fixtures pass by relaxing their
assertions.** Five tests broke after the fix. The temptation was to lower
`assert result["_n_observations"] >= 6`. Looking at the fixtures instead
revealed the actual finding: `tests/test_governance.py::_write_ic_history`
built dates as `(i % 28) + 1`, so `n=60` was 28 distinct January dates
**repeated twice** - one overlapping cluster asserting it was sixty
observations. The other two used consecutive calendar dates. All three now
space dates 35 days apart. **No assertion was weakened**; the fixtures were
made to construct the evidence they always claimed. The old fixtures passed
only because the code counted raw rows - the test suite shared the bug.

**Deleting the 11 now-dead price-cache files** (old key format, never hit
again, ~3 MB tracked in git). Left alone: deleting tracked data files
unattended is not worth 3 MB. Worth doing in a session that is looking at repo
growth anyway.

### Next

**Monday's research note is owed** - the rotation lost its research day to
priority 0. Take one specific thing and do it properly.

Then, in priority order:
- **Watch tomorrow's 02:00 run.** It is the first to exercise the new path
  end-to-end unattended. Expect the log line "Improvement engine evidence: 2
  effective (6 rows) at the 1m horizon; 23 rows across all horizons". If it
  says something else, the bounded-fetch change is where to look.
- **A heartbeat that complains when no run has been logged in 48 hours** -
  still open from the retrospective, and still the dominant failure mode
  (priority -1: 5 of 11 scheduled sessions never fired).
- **The synthetic-data fabrication defect** (priority 1) - a failed fetch still
  silently substitutes sector-realistic fiction upstream in `run_screener.py`.
  `data-run.ps1` gates on it, but it should refuse at source.

**Honest expectation, unchanged:** 8 independent 1-month observations is about
**six more months** of daily running. The fix makes that visible rather than
fixing it. The old behaviour would have reached "8 observations" much sooner
and been wrong.

---

## 2026-08-24 (evening) - Priority 1 closed: the screener no longer fabricates data

Owner-run session. Asked to make the routine work as well as it can going
forward, on the evening after priority 0 landed.

**Health numbers:** last code session **ran and shipped** (06:21 today,
`good/2026-08-24`); data loop **published** 02:11, HEALTH: PASS; evidence base
**23 rows, newest 2026-08-14, 2 effective at the `1m` horizon**; priority 0
**DONE**.

**Tests:** before 590/590, after **596/596**

### Did

**1. De-risked tomorrow's 02:00 run before it happens.** This morning's session
flagged that tonight is the first unattended exercise of the new evidence path,
and named the exact log line to expect. Ran both reporting paths by hand:

- `data-run.ps1`'s inline snippet prints *"2 effective (6 rows) at the 1m
  horizon; 23 rows across all horizons"* - exactly as predicted.
- `write_brief.py` prints *"2 of 8 needed at the 1m horizon (6 rows, but
  overlapping windows are not independent...)"*.
- `compute_forward_returns()` with today's date: **0 new rows in 0.1s, no
  fetch triggered.** This was the flagged risk - the bounded-fetch change could
  have turned every revisited snapshot into a full-universe download. It does
  not.

**2. Priority 1 - the synthetic-data fabrication defect, closed at source.**
Detail in `METHODOLOGY_CHANGELOG.md` 2026-08-24 (evening). `run_factor_engine()`
now exits 2 rather than generating fiction when the network probe fails;
`--allow-synthetic` is the deliberate opt-in and labels its own output.

Open since 08-06 and gated only downstream, so the scheduled loop was protected
and nothing else was.

### Tried and rejected

**Adding the flag to `cli.py` alone.** That was the first attempt and it was
**inert** - `run_screener.py` defines its own `parse_args()` and never imports
`cli.py`. `args.allow_synthetic` would not have existed, the `getattr` default
would have refused *every* run including the intended opt-in, and the 02:00
loop would have failed tomorrow. Caught only because `--help` did not list the
new flag. The regression test now asserts the flag on the **live** parser and
parses an empty argv through it.

That near-miss is the same shape as the 08-11 `Write-NativeOutput` bug: a change
that looked right, passed a shallow check, and would have broken the unattended
run. Verifying through the actual entry point is what caught both.

### Noticed, not fixed

- **`cli.py` is dead code with a live-looking test.** Near-identical parser,
  imported only by `tests/test_cli.py`. Two parsers that can drift, one of them
  tested and unused. Reconcile them.
- **No at-logon catch-up triggers are installed.** Checked tonight: both tasks
  have exactly one trigger. `scripts/register-tasks.ps1` has never been run, so
  the dominant failure mode (priority -1: 5 of 11 sessions never fired) is
  still completely unmitigated. This is the single highest-value thing the
  owner could do and it is one command.

### Next

Unchanged and now genuinely next: **Monday's research note is still owed** -
the rotation lost its research day to priority 0 and this session did not
reclaim it.

Then the heartbeat (no run logged in 48 hours), which remains open from the
2026-08-21 retrospective and is the other half of the absence problem.

## 2026-08-25 - PRODUCT. The dashboard gets a time dimension.

**Health numbers:** last code session **ran and shipped** (2026-08-24,
`good/2026-08-24`); data loop **published** 02:11 today, HEALTH: PASS, 501/501
price coverage, 0 fetch failures, 0 synthetic substitutions; evidence base
**23 rows, newest 2026-08-14, 2 effective observations at the `1m` horizon**;
priority 0 **DONE** (unchanged since 08-24).

**Tests:** before **596/596**, after **627/627** (+31, no pre-existing
failures)
**Data loop:** healthy - see health numbers above.

### Did

Shipped the dashboard's first **time dimension**. This was gap 1 in
`plan/dashboard-north-star.md` ("There is no time dimension. This is
the biggest one... **Start here.**") and priority 2 in `CLAUDE.md`. Three
surfaces, all fed by a new `history.py` built from the snapshots the data loop
has been writing all along:

1. **A "What Changed" section** below the KPI row - biggest rank movers in each
   direction, each with an inline sparkline, the category that moved furthest,
   and the current rank.
2. **A Δ column** in the Full Universe table, sortable, showing the rank change
   since the previous comparable run.
3. **A "Rank History" block** in the per-stock drill-down - the full rank path
   plus the four categories that moved most, since the last run and since ~1
   month.

**The hard part was not the arithmetic. It was deciding which runs are
comparable to each other**, and getting that wrong would have been worse than
shipping nothing.

`2026-07-28` is a degraded run sitting in `improvement/snapshots/`. It predates
`check_run_health.py`, so nothing ever blocked it. Its ranks correlate with the
run before at Spearman **0.016** and with the run after at **-0.020**. Diffed
naively it reports **411 of 501 stocks (82%) moving more than 50 ranks**. The
flagship new panel would have opened with 15 fictional movers.

**First attempt was wrong and the real data caught it.** I reused
`check_run_health`'s dispersion rule (>20% below trailing median) on the theory
that reusing an already-justified threshold beats inventing one. It excluded
**16 of 20 runs**. Risk dispersion has drifted legitimately from 26.7 (Feb) to
19.5 (Aug), and because my baseline only recorded *kept* runs it froze in
February and every later run failed against it - one exclusion cascading into
all of them. Dispersion is the right gate at publish time and the wrong one
here.

The replacement gates on the property the feature actually needs - that a run's
ranking is comparable to its neighbours' - via Spearman correlation against the
last accepted run. Over all 19 consecutive pairs, the 17 clean ones span
**0.882-1.000** (lowest is a 12-day gap; a 29-day gap still scores 0.951) and
the only two breaks are either side of `2026-07-28`. **Any threshold in
[0.05, 0.87] classifies every observed run identically**, so 0.50 is not tuned
to a run - it sits in an empty region. Result: 18 runs kept, 2 excluded
(`2026-07-28` discontinuous, `2026-03-01` a byte-identical warm-start re-run),
both printed on the page with their reason.

### Evidence / research

Measured from this repo's own stored snapshots. None of it is a forward return,
an IC or a backtest number, so rules 4 and 5 do not bite.

- **Noise floor of a rank change.** Pooled over 13 consecutive clean pairs
  (6,515 ticker-pairs): p50 **7**, p90 **36**, p95 **54**. The panel surfaces
  only moves past p95 and states the sample size on screen. Recomputed each
  build, not frozen.
- **The degraded-run contrast.** Median abs(rank change) across the
  `2026-07-28` boundary is **151** against a normal 4-23.
- **Round-trips dominate daily movement.** On today's run **all 10** material
  one-day movers were excursions that returned to base. Over ~1 month, **169 of
  193** were genuine trends (GILD 44 -> 408 and staying there; BDX 253 -> 427).
  **So the default comparison is the ~1-month window, not "since last run".**
  That is a measurement, not a preference - and it lines up with research
  question 2 in the north-star plan, which worries about a dashboard that
  provokes churn.
- **Applied `dataviz` skill guidance**: single series so no legend; direction
  carried by a glyph and a signed number as well as colour, so the panel is
  readable without colour vision; endpoint dot coloured only when the move
  clears the materiality floor; the sortable Δ column is the table view.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-08-25** - "Which runs are comparable to
  each other: the history gate". No factor weight, metric, threshold or scoring
  formula changed; ranks and composites are byte-identical. What the entry
  records is the two *display* thresholds and the evidence behind each.

### Tried and rejected

- **Reusing the dispersion gate for history selection** - rejected by
  measurement: 16 of 20 real runs excluded. Written up in the changelog and in
  `history.py`'s docstring so the next session does not retry it.
- **Presenting daily movers as the headline** - rejected: every one of today's
  was an artifact.
- **Hiding round-trip movers** - rejected as the wrong instinct. They are
  labelled instead, because the honest claim is "this looks like a data
  artifact, here is the path", and suppressing them would have hidden the
  data-quality bug below.

### Found, not fixed - worth its own session

**A transiently-failing metric is scored at an extreme percentile instead of
being treated as missing.** MNST's `return_12_1` percentile read 97.1 on 08-20,
**2.9** on 08-21 and 08-24, then 97.1 again on 08-25 - while the price went
47.5 -> 48.9. FCX's growth score did the same (68.3 -> 42.5 -> 68.3). It is
**not** a NaN: `factor_engine` handles missing metrics correctly
(`na_option="keep"` plus the `has_data` mask). It is a *computed* value from bad
price history, which is why nothing catches it - it moves a stock ~100 ranks
and looks exactly like a real collapse. The movers panel is now the instrument
that makes this visible; it found two cases on its first run.

**`plan/dashboard-inventory.md` is now stale** - it still says "as of
2026-08-05", lists 3 charts and 1 table, and does not mention the What Changed
section, the Δ column or the Rank History block. I could not update it: edits
under `.claude/` are blocked as sensitive in this session. Someone with write
access should refresh it, or the next session will "discover" a gap that is
now filled.

### Also shipped

**`tests/test_dashboard_js.py`** - the emitted dashboard script is ~2,000 lines
of JS built inside a Python f-string, where one un-doubled brace blanks the
entire public page while all four ship gates still pass (gate 3 checks that
`dashboard_data.js` parses; nothing checked the script consuming it). Now
syntax-checked with `node --check`, and I verified the check actually fails on
broken input rather than trusting it. Skips cleanly where node is absent.

### Next

**The transient-metric defect above.** It is a scoring-integrity bug on the
same footing as the 08-24 fabrication fix: the output is indistinguishable from
analysis and is wrong. Two demonstrated cases, a reproduction path
(`improvement/snapshots/`, compare `return_12_1_pct` against
`price_at_scoring`), and a natural home in the same place the NaN handling
already lives.

Still owed and still slipping: **Monday's research note**, now missed twice.

---

## 2026-08-25 (evening) - Removing the last things that needed a human

Owner-run. Brief: make it run smoothly without being asked daily whether it
ran, and without needing me to make updates.

**Health numbers:** last code session **ran and shipped** (06:25 today,
`good/2026-08-25-0625`); data loop **published** 02:11, HEALTH: PASS; evidence
base **23 rows, newest 2026-08-14, 2 effective at the `1m` horizon**; priority 0
**DONE**.

**Tests:** before 639/639, after **647/647**

### Did

**1. `plan/` moved out of `.claude/`.** Eight plan files sessions work from
daily. `.claude/` is blocked as sensitive, so a session could read them and not
correct them - which is exactly what happened this morning: the session shipped
the time dimension and then could not mark it shipped in
`plan/dashboard-inventory.md`. Second occurrence of this shape; `prompts/` was
moved for the same reason on 08-21. All references updated across nine files.

**2. Refreshed `plan/dashboard-inventory.md`** from the live artifacts, which
this morning's session was blocked from doing. It claimed "as of 2026-08-05",
252,191 chars and no time dimension; reality is 270,427 chars, a `history` key
of 0.25 MB over 18 accepted run dates, and gap 1 closed. Also recorded the two
things not to undo: the Spearman >= 0.50 comparability gate, and the ~1-month
default comparison window.

**3. `CLAUDE.md` rule 9 - keep your own docs true.** Now that `prompts/` and
`plan/` are editable there is no excuse for stale process docs, and the Tuesday
focus tells the next session to *trust* the inventory. A wrong inventory sends
it to rebuild something that exists.

**4. `scripts/prune_artifacts.py` + wired into the data loop.** `runs/` and
`logs/` are gitignored working directories nothing ever removed. Measured
today: **44 directories, 62 MB**, three weeks in, growing ~1.4 MB per run. That
is a disk-space failure some months out whose first symptom would be a failed
run. Keeps the newest 20 runs and 60 logs; **never touches `improvement/`,
`cache/` or `validation/`** - the evidence base gets *more* valuable with age,
and `cache/` freshness rules are load-bearing. 8 tests.

### Tried and rejected

Nothing rejected - but the pruner took **three wrong diagnoses** before it
worked, and all three are now pinned by tests:

- **Husks displaced real runs.** Emptying a directory gives it a fresh mtime,
  so a newest-first sort ranked husks above populated runs and the second pass
  deleted the 20 directories the retention count had just protected. Caught by
  checking the directory count afterwards rather than trusting the script's own
  "removed 24" line. Nothing of value lost - `runs/` is gitignored scratch and
  the evidence base was verified intact - but the logic was inverted.
- **Blamed OneDrive for holding handles.** It was not.
- **Blamed `Path.iterdir()` leaving a scandir handle open.** Also not.
- **Actual cause:** OneDrive marks synced directories **read-only**. `os.rmdir`
  honours that and fails WinError 5 on an already-empty directory, while
  `rmdir` from Git Bash succeeds because the POSIX layer clears the attribute
  first. Found by printing the errno instead of guessing a fourth time.

The lesson is the recurring one in this project: the script reported success
while leaving 44 husks behind. Verifying the *effect* rather than the *report*
is what caught it, twice.

### Still needs the owner - and only these

1. **Nothing pushes a notification.** The brief is written and pushed after
   every run and shows staleness prominently, but it must be *looked at*. The
   zero-credential fix is GitHub -> Watch -> All Activity, which emails on
   every push; commit subjects already carry the headline numbers. An
   unattended script cannot send mail without a stored password.
2. **Stay logged in.** The at-logon catch-up (installed 08-24) covers a reboot,
   but the tasks are `InteractiveToken` and cannot run with nobody signed in.
   Settings > Accounts > Sign-in options > "Use my sign-in info to
   automatically finish setting up after an update or restart" closes this.

### Next

Unchanged: **Monday's research note is still owed** - two sessions have now
skipped it for higher-priority work, correctly both times, but the debt is real.

Then the heartbeat that complains when no run has been logged in 48 hours,
still open from the 2026-08-21 retrospective.

---

## 2026-08-26 - SYNTHESIS. How does this fit the rest of the screener?

**Health numbers:** last code session **ran and shipped** (06:25 on 08-25,
`good/2026-08-25-0625`); data loop **published** 02:11 today, HEALTH: PASS,
502/502 price coverage; evidence base **23 rows, newest 2026-08-14, 2 effective
observations at the `1m` horizon**; priority 0 **DONE**, priority 1.5 **closed
today**.

**Tests:** before 647/647, after **676/676**
**Data loop:** healthy - `logs/datarun-2026-08-26_020001.log` ends "Data loop
complete", HEALTH: PASS, 0 fetch failures, 0 synthetic substitutions.

**On the evidence base not moving.** 23 rows / 2026-08-14 / 2 effective is
identical to 08-25, which is two consecutive sessions. That is **expected
latency, not a stall**: the newest snapshot old enough for a `1w` IC is
2026-08-20, which becomes eligible on 08-27. Snapshots exist for 08-20, 08-21,
08-24, 08-25 and 08-26 and are queued. If the row count has not moved by the
08-27 session, rule 8 bites and that becomes the work.

### Did

**Root-caused and fixed priority 1.5 - and the 08-25 diagnosis of it was
backwards.**

MNST's `return_12_1` percentile round-trip (97.1 -> 2.9 -> 97.1) was not a
transiently-failing metric. Yahoo's 13-month series for MNST **alternates
between pre- and post-split prices** across its 2026-08-11 2:1 split:

```
2026-08-05    94.46      <- unadjusted
2026-08-06    47.08      <- adjusted
2026-08-07    90.36      <- unadjusted
2026-08-11    45.53      <- split date
```

`auto_adjust=False` returns byte-identical numbers, so no adjustment was ever
applied. From today's live `runs/83c9e2e2dd48/00_raw_fetch.parquet` the pipeline
divided an unadjusted July close (93.49) by an adjusted 2025 close (62.30):

    published   return_12_1 = +0.5006  -> 97th percentile
    correct     return_12_1 = -0.2497  ->  3rd percentile

**So 97.1 was the artifact and 2.9 was right** - the reverse of what
`NIGHTLY_LOG.md` 08-25, `history.py` and `CLAUDE.md` priority 1.5 all said.
MNST was live on the public site at momentum 71.5, rank 360, roughly **110
ranks too high**. All three records are corrected in this commit.

Fixed at source: `factor_engine.check_price_series_integrity()` refuses a series
that mixes two split scales, and the eight metrics derived from it are withheld
rather than computed. Withholding routes into machinery that already exists -
`na_option="keep"` plus the `has_data` mask renormalise the surviving weights,
so a missing category is neutral. Repair was rejected: MNST's series flips scale
on **seven** separate days, so no single factor puts it right.

### Evidence / research

All measured today; none of it is a forward return, an IC or a backtest number,
so rules 4 and 5 do not bite.

- **The failure itself**, arithmetically exact against the live run artifact
  and reproducible from `cache/factor_scores_19c853468405_*.parquet`.
- **Arming floor, 25%.** Over **137,313 ticker-days** (503 names, 13 months)
  p99.9 of |daily return| is **17.2%** and only **21 days in the entire sample**
  exceed 30%. Below a 25% implied jump a "split ratio" cannot be told apart from
  an ordinary down day - which is what keeps the small spin-off ratios Yahoo
  also reports as splits (SPGI 1.057, HON 1.061, CMCSA 1.067, FDX 1.241,
  BDX 1.272) from flagging everything.
- **False positives: zero.** Run against **all 17 real S&P 500 split events of
  the prior 13 months** - 11 armed the check, it fired on exactly one (MNST) -
  plus volatile controls including MRNA's genuine +177% single-day move.
- **Frequency:** ~17 split events a year in this universe, so expect roughly
  **one affected name a year**. This is the first observed failure.

### The synthesis - what this says about the screener as a whole

**The eight categories are not eight independent bets.** Momentum and risk
together are **23% of composite weight (13 + 10), and every metric in both is
derived from one `Ticker.history()` call per stock.** Nothing checked that
call's output for internal consistency, so one upstream defect corrupted almost
a quarter of a stock's composite while every guard passed it:
`check_run_health` saw 100% price coverage and normal dispersion, and
**winsorization hid the severity rather than catching it** - MNST's raw
`volatility_1y` was **1.77**, capped to 0.845, which merely made Monster
Beverage look as volatile as SMCI.

Second finding, now pinned by a test: momentum's only non-price metric,
`proximity_52w_high`, carries weight **0** as a Phase 11 candidate. So on paper
a rejected series costs momentum 3 of 4 inputs; in practice the renormalised
weight sum is zero and the category goes NaN. **A rejected series costs a stock
two entire categories, not one and a fraction.**

Because withholding is now possible, `check_run_health.py` gains
`MIN_CATEGORY_COVERAGE = 0.90` - one rejected name in 502 is the mechanism
working, fifty is a feed change that must not publish. Dispersion could not
catch that case: with most stocks NaN it is computed over whatever survives.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-08-26** - "A price series that mixes two
  split scales is refused, not scored". No weight, threshold or scoring formula
  changed; every stock with a sound price series scores identically to
  yesterday.

### Tried and rejected

- **Repairing the series by back-adjusting pre-split prices** - rejected by the
  data: MNST flips scale on seven days, not once, so there is no single factor
  that fixes it.
- **A generic "more than one +-30% day in 13 months" detector** - it separates
  MNST (7 days) from every other name in the universe (at most 1), which is a
  clean empty region, but 13 months cannot rule out a genuine crash producing
  repeated 30% days. Recorded in the changelog rather than shipped as a gate,
  so a future session can test it against a wider window instead of
  rediscovering it.
- **Nulling `price_latest` along with the rest** - rejected: it is a single
  point from the most recent bar, `info["currentPrice"]` takes precedence over
  it everywhere, and dropping it would disable valuation metrics that have
  nothing to do with the defect.

### Corrected, not found

**FCX was never a bug.** The 08-25 entry cited FCX's growth score
(68.3 -> 42.5 -> 68.3) as the same defect as MNST. It is not: on 08-24
`forward_eps_growth` and `peg_ratio` were genuinely **NaN** and
`compute_category_scores` correctly renormalised growth over the remaining
three metrics. That is the missing-data path working as designed, and 42.5 was
the honest number for that day.

What it does expose is a **product** gap, not a scoring one: the movers panel
cannot distinguish "moved on new information" from "moved because two inputs
went missing", even though `Composite_Confidence` already carries that fact.
That is a Tuesday question.

### Not done, deliberately

**The published dashboard still shows the old MNST numbers.** I did not
regenerate it - a code session republishing data would create a second snapshot
for a date the 02:00 loop already covered, for no gain. The fix takes effect on
tomorrow's 02:00 data run, after which MNST should show blank momentum and risk
and a lower `Composite_Confidence`. **Worth checking that it does.**

### Next

**Verify the fix landed on the live site** in tomorrow's run: MNST's momentum
and risk blank, health check still PASS, and the new `price_series_rejected`
line in `validation/data_quality_log.csv`. That is a five-minute check, not a
session.

Then the highest-value work is still infrastructure, unchanged since the
08-21 retrospective and now the oldest open item: **the scheduled-task
definitions are not in version control** (`Register-ScheduledTask` appears
nowhere in the repo), and **nothing watches whether the loop is running** - a
run that never fires writes no log, so its absence stays invisible.

Still owed: **Monday's research note**, now missed three times. Each session
skipped it for a demonstrable data-integrity defect, correctly, but the debt is
real and the rotation is not producing the thing it was designed around.

---

## 2026-08-26 (evening) - Owner-directed: the model portfolio leaves the dashboard, stocks gain an "about"

**Not a scheduled session.** The owner asked for two specific changes in an
interactive session and, separately, asked *how he is supposed to tell this
routine what to focus on*. That question turned out to be the most important
part of the evening: until tonight there was no answer. See "The channel" below.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-08-26_060001.log` - ran, shipped to main, tagged `good/2026-08-26` |
| Data loop published? | `logs/datarun-2026-08-26_020001.log` - HEALTH: PASS, 0 fetch failures, 502 scored, published |
| Evidence base | **23 rows, newest 2026-08-14, 2 effective observations at `1m`** (6 raw) |
| Priority 0 | Fixed 2026-08-24, still holding |

The evidence base has not moved since 08-24 by row count. The 08-26 morning
session recorded why: the next eligible snapshot becomes computable on
**2026-08-27** with five queued behind it, and wrote itself a tripwire - if the
count has not moved by tomorrow's session, rule 8 bites and that becomes the
work regardless of rotation. **That tripwire is still armed and this session
did not touch it.** Tomorrow: check `live_ic_history.csv` first.

### The channel (the part worth keeping)

The owner had no way to direct this routine. `CLAUDE.md` priorities are written
*by sessions, for sessions*; the weekly rotation is fixed; and he is explicitly
not reading diffs. So a request like tonight's could only ever reach the system
by him opening a chat and asking - which does not scale and leaves no record.

`OWNER_FOCUS.md` is now that channel: plain English, **Open** and **Done**
headings, read during Orient *before* the rotation is consulted. Open items
outrank the day's nominal focus. Only two things outrank an owner item - a
stalled data loop and the ship gates - and the prompt now requires a session
that defers one to *say so in the log*, because an unmentioned deferral is
indistinguishable from an ignored request.

Wired into `prompts/nightly.md` (step 1) and `CLAUDE.md`. Pinned by
`tests/test_owner_focus.py` (7 tests) - including that the reference appears
between "## 1. Orient" and "## 2. Baseline", so it cannot drift to a position
after the work is already chosen. **A silent channel looks exactly like an
empty one**, which is the same shape as the evidence base sitting at 3 rows for
183 days while every run reported success.

While in `prompts/nightly.md` I also corrected two stale claims it was still
making: that the forward-return horizon bug is unfixed (it shipped 08-24) and
that the IC history holds "3 observations, all `1w`, all February" (23 rows;
the number that matters is 2 effective at `1m`). Rule 9.

### Did - 1. Removed the Model Portfolio from the dashboard

Owner's stated reasons: it serves no genuine purpose, and it wastes payload.
**The second reason is false and I checked before acting** - `portfolio` was
9,681 bytes of a 3,373,395-byte payload, 0.29%. Removing it saves nothing.

The first reason is right, and stronger than stated. Two findings:

- **It carried no column `table_data` did not already have.** Holdings held
  `ticker/company/sector/composite`, the eight category scores, `vt`, `gt` -
  every one already present under a different case. A renamed, row-filtered copy.
- **It did not answer "how much" either.** The north star names position sizing
  as question 4, so this looked like it might cost an answer. It does not: the
  holdings payload **carries no weights at all**. The sizing logic lives in
  `portfolio_constructor.py` and the Excel sheet and was never exposed.

So the real justification is the governing line in `CLAUDE.md`: a fixed 25-name
sector-capped list published to a public site is the closest this tool came to
emitting a recommendation. A ranking a reader sorts is a screen; a named
portfolio is advice.

**`plan/dashboard-inventory.md` had already worked this out** (owner directive
2026-08-05) and warned: do not blindly delete `portfolio_constructor.py`,
because `improvement_engine.record_run_snapshot()` computes **turnover** from
`in_portfolio`. Checked - correct. The engine, artifact, Excel sheet and
snapshot column all stay; only the dashboard surface went. That file is now
marked DONE with what shipped.

Also removed `spx_weights`: the portfolio-vs-SPX chart was its only consumer,
and a sector split of the S&P 500 against itself is a tautology.

**Top 5 was verified, not assumed.** It read `D.portfolio.holdings.slice(0,5)`.
It now filters `table_data` for trap-free names and sorts by rank. Both paths
give `HST, EXPE, APA, EIX, CF` on live data - the sector cap is 8-of-25 and
cannot bind on five rows. The trap exclusion was kept deliberately; dropping it
would promote a flagged name into the headline five.

### Did - 2. "About" sections in the stock drilldown

The tool could score a company on 44 metrics and not say what it sold. For the
investment-club audience that is a teachability gap, not polish: a student
looking at APA at rank 3 cannot learn it is oil-and-gas exploration without
leaving the tool.

`longBusinessSummary` now comes off the `.info` dict the fetch **already
pulls**, so the API cost is zero - important, because the data loop is already
losing tickers to Yahoo rate limits and buying prose with evidence would have
been a bad trade. Rendered under the score cards with the specific industry, a
4-line clamp, a "Show more" toggle, and an attribution line.

Payload: **+0.71 MB raw (~+21%), ~+60 KB gzipped**. Prose gzips ~11.6x against
the payload's overall 4.2x, and Pages serves gzip, so raw size overstates this
by an order of magnitude. Worth writing down generally: **measure gzip before
calling a payload change expensive.**

Display-only, and a test enforces it - `about` must never appear in `raw`/`pct`.
The screener does not rank prose.

**`about` is empty on the site until the 02:00 run on 2026-08-27.** The field
did not exist in the raw parquet before tonight. `industry` populated
immediately (501/502) because it was already being fetched and simply unused -
which is also how I confirmed the merge path works before trusting it.

### Found by looking, not by reading

The first cut measured `scrollHeight > clientHeight` inside `renderAbout()`,
which runs while the modal is still `display:none`. Both heights read 0, so
**"Show more" was hidden on every stock** and long descriptions were
permanently truncated with no way to expand them. The source reads as correct.
It was caught by rendering the page in a browser and driving it.

Fixed with `requestAnimationFrame`; pinned by
`test_about_overflow_is_measured_after_layout`. `.claude/launch.json` now
serves the repo root on :8931 so the next session can do the same thing cheaply
- there is now a standing way to *look at* the dashboard before shipping it,
which this repo did not have.

### Tried and rejected

- **Deleting `portfolio_constructor.py` outright.** Would have broken turnover
  in the evidence base. The inventory file predicted this; I verified rather
  than trusting it.
- **Truncating summaries to ~2 sentences to save payload.** Unnecessary once
  gzip was measured, and it would have cut mid-thought for the diversified
  names that most need explaining.
- **Justifying the removal on payload size.** It is 0.29%. Shipping that
  reasoning into the changelog would have put a false number in the audit trail.

### Verification

- **706 passed**, up from 676. No regressions; the baseline was clean both
  before and after.
- The 30 new dashboard tests were run against `9bed64f` in a detached worktree:
  **29 of 30 fail** there. The one that passes both ways is the guard asserting
  the defensibility section survived - correct behaviour for a "do not break
  this" test.
- Browser-driven: no console errors, `sec-portfolio` absent from the DOM, Top 5
  renders five cards, About verified across all three data shapes (long -
  block and toggle; short - block only; missing - no block).
- Published artifacts regenerated from `runs/83c9e2e2dd48`, the same run
  already live, so the data is unchanged and only the surfaces differ.

### Noticed, not fixed

**The publish path writes the root artifacts twice, by two different routes.**
`run_screener.py` step 12 generates into the run dir and copies to root;
`data-run.ps1` then regenerates into the run dir and copies only
`dashboard.html` to `index.html`. Root `dashboard_data.js` is never re-copied -
it stays correct today only because the second generation is byte-identical to
the first. If anything makes those two generations differ (a changed
`SCREENER_OVERVIEW.md` between them would do it), `index.html` and
`dashboard_data.js` could ship out of step, and the embedded `data_version`
hash would disagree with the data actually loaded. Not urgent, not touched
tonight - but it is a real trap for a future session.

### Next

1. **Check `improvement/live_ic_history.csv` first.** The 08-26 tripwire is
   armed: if the row count has not moved past 23, that is the session's work.
2. **Read `OWNER_FOCUS.md`.** It is empty now, but it is the first thing to
   check from here on.
3. Confirm the 02:00 run populated `about` - the drilldown should show real
   descriptions from 2026-08-27. If it does not, the merge in
   `generate_dashboard.load_run_data` is the place to look.
4. Monday's research note is still owed. Two sessions have skipped it.

### Follow-up, same evening - layout tweaks (owner request)

Three changes to the landing view, all owner-directed:

- **"What Changed" moved below "Top 5 Stocks."** Section order in the emitted
  HTML *is* the reading order - there is no ordering layer - so this is a
  literal move of the markup block.
- **"What Changed" and "Factor Analytics" now collapsed by default.** The
  landing view is Top 5 plus the full universe table; everything else is one
  click away rather than scrolled past. `sec-defensibility` was already
  collapsed.

**A worry that turned out to be unfounded, checked rather than assumed.**
Collapsing `sec-analytics` puts two Chart.js canvases inside a container with
`max-height: 0`, and a chart that initialises at zero size normally stays
broken after expansion. Measured in the browser: both canvases are 856x320 and
428x320 *while collapsed*, unchanged after expanding. `overflow: hidden` with
`max-height: 0` preserves layout width and the canvases keep their explicit
height, so Chart.js sizes correctly. No workaround needed - and no
`requestAnimationFrame` hack added on spec.

Pinned by 8 more tests in `tests/test_dashboard_surfaces.py`: section order,
collapse state per section, and that `renderChanged()` - which un-hides the
section via `style.display` when history exists - does not also clear
`collapsed` and silently undo the default. Suite 713 -> 721.

### The fresh run, and a bug the fresh run exposed

The owner asked for a full run so the new `about` field would populate.

**It worked: 501/502 stocks now carry a real business description**, run
`cc84fe992a17`, HEALTH: PASS, 502/502 price coverage, 498/502 analyst targets,
all five dispersion checks within tolerance of the trailing median.

**First attempt failed correctly, and that is worth recording.** A plain
`data-run.ps1 -Force` finished in **5.6 seconds** and was refused by
`check_run_health` with "no evidence of a live fetch". The `factor_scores`
cache from the 02:00 run was still inside its 1-day window, so the run
warm-started. That is the intended behaviour - but it means **a cached run can
never populate a newly added fetch field**, because the cache predates the
column. Added a `-Refresh` switch to `data-run.ps1` that passes `--refresh`
through, rather than hand-running the pipeline and skipping its health gate.
The scheduled 02:00 run must not set it; a warm start is the normal cheap path.

**A cache worry that turned out to be unfounded.** I expected tonight's cache
(written ~19:50) to still be inside the 1-day window at 02:00 tomorrow and so
suppress the scheduled fetch. It will not: `_find_latest_cache` derives the
cache timestamp from the **date suffix in the filename**, i.e. midnight of that
calendar day, not the file mtime. At 02:00 on 08-27 the cache reads as 26 hours
old and the run fetches. No cleanup was needed and none was done. This also
explains the 19:37 warm start (19.6h) and why 02:00 runs fetch every day.

### The bug: my portfolio removal made every future data commit lie

The 19:38 run committed **`data: screener run 2026-08-26 - 502 scored, top: MAA
DOC KIM REG UDR`**. The real top five were `HST EXPE APA EIX CF` - unchanged
from the morning run, Spearman **0.9966** between the two composites, median
absolute rank move 4.

`data-run.ps1` built that subject by regex over the raw payload: first five
matches of `"ticker": "XXX"`. The lowercase key belonged to the **model
portfolio holdings**, which happened to be serialised in rank order. I removed
that surface earlier tonight, so the same regex began matching the first
stock's **sector peers** instead. `stock_detail` starts at HST, HST is a REIT,
its peers are REITs - so the wrong answer read as a plausible all-REIT top five
and I nearly accepted it as a market move.

Nothing failed. Health passed, the push succeeded, the payload was correct.
Only the audit trail lied. **The same failure shape this repo keeps
rediscovering: the system reporting success while producing garbage** - and
this time I introduced it, and caught it only because the committed headline
disagreed with what I already knew the top five to be.

Replaced with `scripts/commit_subject.py`, which loads the payload and sorts
`table_data` by `Rank`. It cannot be fooled by key casing, serialisation order,
or a lowercase `ticker` key appearing elsewhere, and on any error it prints an
*uninformative* subject rather than a wrong one - a crash there must never stop
a healthy run publishing. 9 tests in `tests/test_commit_subject.py`, including
a fixture whose peer block reproduces the exact trap and a check that
`data-run.ps1` never scrapes `"ticker":` again.

**Commit `d6074a9` keeps its wrong subject.** Rule 2 - history on `main` is
never rewritten - and the correction lives here and in the changelog instead.

**The general lesson, worth carrying:** removing a payload key is not a
self-contained change. Anything that *pattern-matches* the payload rather than
parsing it can silently re-aim at a different key of the same name. Grep for
consumers outside the front end - shell scripts included - before deleting a
key.

Suite 721 -> 733.

### Scheduled-run audit (end of session)

Owner asked for confirmation that future runs fire cleanly. Everything below
was checked against the live machine, not inferred from the scripts.

| Check | Result |
|---|---|
| Task Scheduler entries | `Screener Data Run` and `Nightly Screener Improvement`, both **Ready** |
| Last result | both ran 2026-08-26, `LastTaskResult=0` |
| Next run | 2026-08-27 02:00 and 06:00 |
| Triggers | 2 each: weekly `DaysOfWeek=62` (Mon-Fri) + logon catch-up, both enabled |
| Settings | `StartWhenAvailable=True`, `WakeToRun=True`, `Enabled=True` |
| Registered vs version control | actions, times and days match `scripts/register-tasks.ps1` exactly - no drift |
| Neither task passes `-Refresh` | correct; a warm start is the normal cheap path |
| Success markers | `.datarun-last-success` and `.nightly-last-success` both `2026-08-26`, **no BOM**, so tomorrow's date differs and neither run-once guard blocks |
| Lock files | none left behind |
| Script syntax | `data-run.ps1`, `nightly-screener.ps1`, `register-tasks.ps1`, `fix-trust.ps1` all parse via `Parser::ParseFile` |
| Nightly preflight | `git`, `python`, `claude` all resolve on PATH; both prompt templates present; folder trust `hasTrustDialogAccepted=true` for both path spellings in `.claude.json` |
| Next sessions | 08-27 Thu = BUILD, 08-28 Fri = normal (ISO week 35 is odd, so not a retrospective), 08-31 Mon = RESEARCH |

**One hardening made.** The new commit-subject call reads `Invoke-Native`'s
`.Output`, which merges stderr - so an element can be an `ErrorRecord`, and
calling `.Trim()` straight on one throws. Now goes through `.ToString()` first.
Verified both paths in a real PowerShell process: the good path returns
`data: screener run 2026-08-27 - 502 scored, top: HST EXPE APA EIX CF`, and a
deliberately missing script falls back to the plain subject at exit 2 without
throwing.

**Two things checked and found to be non-issues**, recorded so nobody re-opens
them: `output/` is gitignored and empty, so the `Permission denied` seen during
a `git stash -u` cannot affect the runners (data-run stages explicit paths, and
`git checkout -- .` only touches tracked files); and `SCREENER_OVERVIEW.md` is
already in `$DataArtifacts`, so the overview being regenerated every run does
not leave a dirty tree - tonight's run confirmed it empirically.

The local preview server on :8931 was stopped and the port confirmed free.

---

## 2026-08-27 - BUILD. The watchdog moves outside the thing it was watching.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-08-26_060001.log` - "Run complete: shipped to main" |
| Data loop published? | `logs/datarun-2026-08-27_020001.log` - "Data loop complete", HEALTH: PASS, 0 fetch failures, 502/502 price coverage |
| Evidence base | **25 rows, newest 2026-08-20, 3 effective observations at `1m`** (7 raw) |
| Priority 0 | Fixed 2026-08-24, holding |

**Tests:** before 733/733, after **791/791**
**Data loop:** healthy

**The 08-26 tripwire is released.** The previous two sessions read "23 rows,
newest 2026-08-14, 2 effective at `1m`" and armed rule 8: if it had not moved
today, making it move was today's work regardless of rotation. It moved -
**23 -> 25 rows, 2 -> 3 effective at `1m`** - so the 08-24 fix is accruing
evidence as predicted and the rotation stood. Independent 1-month observations
still arrive about one a month; 3 of 8 is roughly five more months.

### First, the five-minute check the last session asked for

**The MNST price-series fix landed on the live site.** Confirmed in today's
02:00 run rather than assumed:

- `validation/data_quality_log.csv` carries the first-ever
  `price_series_rejected` row - MNST, "series mixes pre- and post-split prices
  across a 2:1 split - 4 day(s) move by ~0.5x".
- The published payload has `momentum_score: null` and `risk_score: null` for
  MNST, and `return_12_1`/`return_6m`/`volatility`/`beta` all null in `raw`.
- MNST moved rank **360 -> 370**, in the direction the correction implies.
- `check_run_health` still PASS: one withheld name in 502, which is
  `MIN_CATEGORY_COVERAGE = 0.90` doing its job rather than tripping.

### Did - 1. A watchdog that is not inside the thing it watches

`CLAUDE.md` priority -1 listed two open infrastructure items. **One of them was
already done and the file did not know.** `scripts/register-tasks.ps1` shipped
2026-08-21 (commit `33f3ca7`), yet the priority section, `ACTION_REQUIRED.md`
and a research note all still asserted that `grep -rn "Register-ScheduledTask"
.` "finds nothing". It finds it at `scripts/register-tasks.ps1:111`. Corrected
in all three places - rule 9.

The second item was real, and is the session's work: **nothing watched whether
the loops were running.**

**Why the existing detector could not have caught it.** `write_brief.py`
already prints a "THE ROUTINE IS NOT RUNNING" banner when a loop has been quiet
for two days. It is structurally incapable of covering the case that matters,
and this is the finding worth keeping: **the watchdog was living inside the
thing it was watching.** `write_brief.py` is invoked only from `data-run.ps1`
(line 90) and `nightly-screener.ps1` (line 107). If neither loop fires, the
brief is never regenerated, so the banner never renders and `MORNING_BRIEF.md`
goes on describing the last run that *did* happen. That banner can only ever
catch "one loop died while the other lived" - never "the machine was off",
which is the documented dominant failure mode.

Shipped:

- **`scripts/check_loop_health.py`** - the decision logic. Stdlib-only, for the
  same reason `write_brief.py` is: it is what reports that everything else is
  broken, so it must not break with it.
- **`.github/workflows/loop-watchdog.yml`** - the external observer. Runs on
  GitHub Actions at 23:00 UTC on weekdays, where a PC that is off, asleep or
  logged out cannot silence it. Opens **one** reused issue when a loop stalls,
  updates it in place, and closes it on recovery.
- **`tests/test_loop_watchdog.py`** - 43 tests.

**The heartbeat.** Both loops push a commit to `main` from a `finally` block,
so it lands whether the run succeeded, was discarded by a gate, or crashed:
`brief: data run <date>` and `brief: code session <date>`. A failed session's
`- SESSION DID NOT RUN` suffix still counts as a heartbeat, deliberately: this
answers "did the task fire", which is a different question from "did it do
anything useful", and `write_brief.py` already answers the second.
`brief: evening session <date>` is deliberately *not* matched - owner-initiated
interactive work proves a human was present, not that the 06:00 task fired.

**Most of the design is about staying quiet.** An alarm that fires on noise
gets muted, and a muted alarm is worse than none because it still looks like
coverage. So: weekends are excluded; a day is not judged until its deadline
(12:00 data, 16:00 code) has passed, which is late enough that the at-logon
catch-up in `register-tasks.ps1` has had its chance; one missed weekday is a
WARN with no issue; two *consecutive* missed weekdays is the alarm.

### Evidence / research

Not a backtest, not an IC number, so rules 4 and 5 do not bite. The evidence is
a documented failure and a replay of it.

- **The failure.** 2026-08-21 retrospective, quoted in `CLAUDE.md` priority -1:
  of 11 scheduled code-loop slots from 2026-08-06 to 2026-08-20, **5 never
  fired at all**, four of them consecutive weekdays (08-17..08-20) while the
  machine sat logged out. It went unnoticed for six days because a run that
  never fires writes no log.
- **The detector reproduces that outage from live history, unprompted.** Run
  against this repo's real `main`, `check_loop_health.py` reports the code loop
  missing exactly `2026-08-17, 08-18, 08-19, 08-20` and the data loop missing
  `08-17, 08-18, 08-19`. The one-day difference is correct and was verified:
  `data: screener run 2026-08-20` was committed at **23:28 on 08-20**, the
  logon catch-up firing when the owner signed back in. A UTC-based watchdog
  that used the observer's date would have mis-filed that commit to 08-21;
  pinned by `test_the_commits_own_date_is_used_not_the_observers`.
- **When it would have spoken: 2026-08-18, two days in rather than six.**
  `TestRealOutage` replays the outage day by day and asserts WARN on 08-17,
  STALLED on 08-18, and - importantly - OK on 08-14, the Friday before, so the
  alarm would not have been lost in prior noise.

### Did - 2. The morning brief had silently lost its Top 5

Found while reading `write_brief.py` for the above, and confirmed against the
published artifact rather than inferred.

`dashboard_facts()` read `d["portfolio"]["holdings"]`. The 2026-08-26 evening
session removed the Model Portfolio and its payload key. Every lookup on that
path uses `.get()` with a default, so `d.get("portfolio", {})` returned `{}`,
`top5` became `[]`, and `if facts.get("top5")` dropped the row. **No exception,
no log line, no empty row - the single most decision-relevant line on the
owner's daily page just stopped being emitted, and had been missing from every
brief since.** Verified by importing the pre-fix module from a detached
worktree at `c52abb3`: `top5 -> []` against the same payload that the live
dashboard renders as `HST EXPE EIX APA CF`.

Fixed by computing it the way the dashboard does - trap-flagged names excluded,
then rank order, mirroring `renderTop5()`. Brief and dashboard now agree
exactly, and both agree with today's data-run commit subject.

**And the bug class, not just the bug.** `PAYLOAD_KEYS` names the payload keys
the brief depends on; `dashboard_facts()` reports any that have gone missing,
and `main()` surfaces them under "Things that needed attention". A future
payload change that breaks the brief now says so on the brief itself instead of
quietly shortening it. `tests/test_morning_brief.py`, 15 tests - the file had
**no tests at all** before today, which is precisely how this shipped.

**12 of those 15 fail against `c52abb3`**, the two headline ones with the exact
live symptom: `assert [] == ['HST', 'EXPE', 'EIX', 'APA', 'CF']`.

### Methodology changed

**None.** No weight, threshold, metric or scoring formula moved, so there is no
`METHODOLOGY_CHANGELOG.md` entry - this was infrastructure and a reporting
defect. Every stock scores today exactly as it did yesterday.

### Tried and rejected

- **A third scheduled task on the same machine as the watchdog.** It would
  share the failure mode it exists to detect: nobody logged on means the
  watchdog does not run either. The observer has to be off-box, which is what
  forced the GitHub Actions design.
- **Alarming on a single missed weekday.** Rejected against the record: the
  documented outages ran 2+ consecutive days, while single misses have
  ordinary transient causes (one reboot, one network drop). Firing on those
  trains the owner to ignore the alarm.
- **Reusing `write_brief.py`'s "2 days since last run" rule.** It is calendar
  days, so it alarms every Monday - the last run was Friday, three calendar
  days and zero missed weekday slots.
  `test_a_weekend_gap_is_not_a_stall` pins this.
- **`zoneinfo` for the scheduling timezone.** Needs the `tzdata` package on
  Windows, and this script must not acquire a dependency. The offset is taken
  from the newest heartbeat commit instead, which is self-calibrating. Across a
  DST boundary it can be an hour out, immaterial against deadlines in hours.
- **Deleting `ACTION_REQUIRED.md`.** Its own header invites deletion and its
  premise is now false, but it documents a trust-regression path worth keeping.
  Marked RESOLVED with dates instead; deleting it is the owner's call.

### Noticed, not fixed

- **A `.git/worktrees/prefix-check` admin directory could not be deleted**
  (Permission denied, presumably a OneDrive or AV lock after pytest ran there).
  Git no longer lists it as a worktree and `git status --porcelain` is clean of
  it, so no gate is affected; it is untracked internal metadata that will prune
  when the lock releases.
- The double-publish trap flagged on 08-26 is untouched and still real.

### Verified after merging, not left to the next session

The workflow could not run until it was on `main` - GitHub only runs scheduled
workflows from the default branch - so it was merged and then **dispatched
manually**. Run
[33066665935](https://github.com/CalebSmit/screener-dashboard/actions/runs/33066665935)
is green in 11s, all five steps passing, and no issue was opened, which is the
correct behaviour at verdict `ok`.

**The part local tests could not prove.** GitHub's runner is UTC and fired at
`11:16Z`; the checker reported *"Loop health as of 2026-08-27 06:16 (scheduling
timezone)"* and found both heartbeats. The offset self-calibration from the
newest commit works in CI, and `fetch-depth: 0` gives it the history it needs -
the two things most likely to have been wrong in an environment I cannot run
locally. CI output is identical to local output.

The run raised a Node 20 deprecation annotation, so the three actions were
bumped to their current majors (`checkout@v7`, `setup-python@v7`,
`github-script@v9`) and re-verified rather than left to fail later.

### Next

1. **Monday's research note, now missed four times.** Every skip has been for a
   demonstrable defect and each was the right call in isolation, but the
   rotation is not producing the thing it was designed around. This is the
   oldest real debt in the project and it should outrank a fifth firefight.
2. Per-category trend lines over the full history (priority 2's remainder).
3. Confirm the watchdog's first *scheduled* firing (23:00 UTC weekdays) also
   went green - the manual dispatch proves the job, not the cron.

## 2026-08-28 - HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-08-27_060001.log` - "Run complete: shipped to main" |
| Data loop published? | `logs/datarun-2026-08-28_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **27 rows, newest 2026-08-21, 3 effective observations at `1m`** (8 raw) |
| Priority 0 | Fixed 2026-08-24, holding |

**Tests:** before 791/791, after **825/825**
**Data loop:** healthy
**Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`, so the
rotation stood. Nothing was deferred.
**Rotation:** ISO week 35 is odd, so this was a normal Friday, not a
retrospective.

**Last session's item 3 is closed first, because it was cheap.** The watchdog's
first *scheduled* firing (run
[33148005073](https://github.com/CalebSmit/screener-dashboard/actions/runs/33148005073),
`schedule` trigger, 10s, success) is green. The cron works, not just the manual
dispatch.

### The question this day asks, asked literally

"Would a finance student understand what they are looking at?" The most
teachable surface in the tool is the drilldown's contribution panel, because it
does not just show a score - it shows the working:

```
Momentum   13% weight
Score: 65.3/100  [Average]  x 13% = 9.76 pts
```

So I checked the arithmetic against the payload that was live on `main` this
morning. **65.3 x 13% is 8.49, not 9.76.** The one worked example on the site
did not add up.

### Did - the weights shown were not the weights used

Solving `contrib / score` over the 491 stocks with all eight categories
populated recovers what the composite was really built from: **valuation 20.05,
momentum 14.95**, the other six unchanged, summing to 100.000. Those are
exactly a LOW VOL regime - `13 x 1.15 = 14.95`, the 1.95pp taken out of
valuation, per `adjust_momentum_weight()`.

**Root cause, and why it hit only two categories.** `adjust_momentum_weight()`
returns a deep copy. `run_factor_engine` does `cfg = adjust_momentum_weight(...)`,
which rebinds a *local* name, so the adjustment never reached `main()` - and
`ctx.save_effective_weights(cfg)` is called from `main()`. The
revisions/investment auto-disables assign into the shared dict
(`cfg["factor_weights"] = ...`) and so did propagate. That asymmetry is the
whole bug, and it is why six categories were right and two were wrong. The file
has been called `effective_weights.json`, docstring "the effective weights",
the entire time.

**A second cause, found alongside.** When a category cannot be scored for a
stock, `compute_factor_contributions` drops it and renormalises the survivors;
the page showed the universe weight anyway. MNST - the name whose price series
the 2026-08-26 split check rejects, removing Momentum and Risk - displayed
"22% weight -> 20.64 pts" against a quality score of 70.43.

**Blast radius, counted:** of 4,016 (stock, category) cells in the live
payload, **1,051 showed arithmetic that did not hold** - momentum wrong for 498
of 502 stocks, valuation for 501 of 502, plus 52 cells across 11 stocks from
the renormalisation. **After the fix: 0 of 4,002.**

Shipped:

- **`run_screener.py`** hands the regime-adjusted weights back through `stats`;
  **`run_context.py`** records them, plus `base_factor_weights` and a
  `factor_weights_adjusted` flag. That is the fix at source.
- **`generate_dashboard.py`** reconciles recorded weights against published
  contributions on every build and **will not publish weights that fail to
  reproduce them**. On the real 2026-08-28 run it fired, printed
  `valuation: recorded 22 -> actual 20.05` and `momentum: recorded 13 -> actual
  14.95`, and republished truthfully. This is the guard that would have caught
  the original bug; nothing was checking that the sum added up.
- **The drilldown shows per-stock effective weights** across all three
  surfaces, and `weightNote()` explains any gap in prose - which regime rule
  moved it, or which category was withheld and where its weight went. A
  withheld category keeps its row, marked "no data", rather than vanishing;
  hiding it would leave the reader unable to see why the rest total more than
  the defaults.
- **`SCREENER_OVERVIEW.md`** (generated - I edited the generator, rule 10) now
  says its printed weights are configured defaults, names both rules that move
  them, and points at `effective_weights.json`.

**The live site is already corrected** - regenerated from today's run and
republished, not left for Monday's 02:00 loop.

### Evidence / research

A demonstrated user-facing failure, measured on the published artifact. No
citation, no backtest, no IC number - rules 4 and 5 do not bite, because
nothing here was justified by a return.

The measurement is reproducible from the payload alone: for each stock, predict
`cat_score x eff_weight / sum(eff_weights over categories with data)` and
compare to the published `contrib`. Before: 1,051 of 4,016 cells disagree by
more than 0.011. After: 0 of 4,002.

### Methodology changed

`METHODOLOGY_CHANGELOG.md` 2026-08-28. Filed there deliberately even though
**no stock's score or rank moves by a single place** - no weight, threshold,
metric or formula changed. What changed is what the tool asserts about how it
scored, which is exactly what that file exists to keep honest. The published
`weights.factor_weights` now reads 20.05/14.95 rather than 22/13.

### Tried and rejected

- **Generating the methodology prose from `config.yaml`.** `plan/dashboard-inventory.md`
  proposed this as "a genuine correctness fix" because the weights were
  "hardcoded into the prose". They are not - `generate_screener_overview(cfg)`
  has been templating the whole document from config all along. I checked all 8
  category weights and ~40 metric weights against `config.yaml`: every one
  matched. The inventory was wrong and is corrected. The real defect was one
  level down and the opposite shape: the *document* was faithful to config
  while the *screener* was not.
- **Publishing per-stock effective weights in the payload.** 8 floats x 502
  stocks, when the renormalisation is a pure function of which categories have
  data - which the payload already carries as nulls in `cat_scores`. The JS
  recomputes it instead, mirroring `compute_factor_contributions`. No payload
  growth.
- **Making the reconciliation a hard build failure.** Tempting, and wrong:
  every run directory on disk records 22/13, so a build that refuses would have
  taken the data loop down on Monday morning rather than fixing anything. It
  corrects, says loudly what it corrected, and flags the payload
  `factor_weights_derived`.
- **Trusting the derived weights as the design.** The derivation is a repair
  path for old runs and a tripwire for future divergence. The fix is the
  pipeline handback. It declines to guess on a universe under 20 rows.

### Noticed, not fixed

- **`runs/` holds test-created directories** (`test_hash_1`, `test_meta`,
  `test_artifact`, `test_save_cfg`, `test_git_sha`, ...) alongside real runs.
  Harmless today - `_find_latest_run()` picks by artifact presence and got the
  right one - but it is priority 8 (test isolation) leaving litter in a
  directory the dashboard reads from. My own new tests deliberately avoid it by
  calling `save_effective_weights` on a stub rather than constructing a
  `RunContext`.
- **`data-run.ps1` logs the raw IC row count** - "27 raw IC row(s) (effective
  count unavailable)". Honest about its own limitation, so not misleading, but
  the effective count is the one that matters (`CLAUDE.md` rule 8) and the
  script could compute it.
- **`data-run.ps1` regenerates the dashboard into the run directory, not the
  repo root.** The root artifacts come from `run_screener.py` earlier in the
  same run, so the standalone `generate_dashboard.py` call at line 253 has no
  effect on what gets published. It is redundant rather than broken, but it
  means a dashboard-only change does not reach the site by re-running that
  script alone.

### Next

1. **Monday's research note, now missed five times.** Unchanged from last
   session and now a day older. Every skip has been for a demonstrable defect,
   including today's, and each was right in isolation - but the rotation has
   produced one research note in a month. The next Monday should be spent on it
   even if something else is broken, unless the data loop itself is down.
2. Per-category trend lines over the full history (priority 2's remainder).
3. The `runs/` test-litter above - small, and it is the visible edge of
   priority 8.

## 2026-08-29 - CATCH-UP. Not normally scheduled. Work the single highest-value item from the priorities list.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-08-28_060001.log` - "Run complete: shipped to main" |
| Data loop published? | **NO - `logs/datarun-2026-08-29_121115.log` died at "Could not check out main." That is today's work.** Last good run: `datarun-2026-08-28_020001.log`, "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **27 rows, newest 2026-08-21, 3 effective observations at `1m`** (8 raw) |
| Priority 0 | Fixed 2026-08-24, holding |

**Tests:** before 825/825, after **853/853**
**Data loop:** **was broken this morning - fixed**
**Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
**Rotation:** Saturday catch-up, so there was no nominal focus to defer. The
data-loop failure would have outranked one anyway (`CLAUDE.md`: "fixing it is
the highest priority work available, ahead of any feature").

### Did - the catch-up trigger killed the data run it exists to protect

Both logs are stamped the same second:

```
logs/datarun-2026-08-29_121115.log   [12:11:15] === Data loop 2026-08-29 ===
logs/nightly-2026-08-29_121115.log   [12:11:15] === Code loop 2026-08-29 ===
```

One second later the data loop was dead:

```
[12:11:16] [ERROR] Could not check out main.
```

**Root cause.** `register-tasks.ps1` gave both scheduled tasks an at-logon
catch-up trigger with the *same* `PT3M` delay, so on the first logon of a day
they start together. They share one working tree and git serialises nothing for
them. At 12:11:16 the data loop ran `git checkout main` while the code loop was
inside `Restore-Artifacts` running `git status` - which takes `.git/index.lock`
to refresh the index. The data loop treated a transient lock as fatal and
stopped **before running the screener at all**.

The at-logon trigger is the fix for priority -1, the dominant failure mode
("sessions do not start"). It had become a way to lose a run.

**Why nothing caught it.** Each script has a single-instance lock
(`.datarun.lock`, `.nightly.lock`). Those stop a loop racing *itself*; nothing
stopped the two loops racing *each other*. The watchdog meanwhile reported
healthy, correctly: the data loop pushed `brief: data run 2026-08-29` from its
`finally` block, which is the heartbeat, and by design that answers "did the
task fire", not "did it do anything". Only the log said otherwise.

Shipped:

- **`scripts/repo-lock.ps1`** - a shared lock both loops take before their
  first git command and release *after* `Publish-Brief` (which itself runs
  git add/commit/push). Acquisition uses `FileMode::CreateNew`, which is
  atomic; `Test-Path` then `Set-Content` is not, and two processes three
  seconds apart both pass that test.
- **The loser waits, it does not die.** Measured over 2026-08-21..28, data runs
  take 11.8-13.6 min and code sessions 16.9-25.9 min, against task execution
  limits of 3h and 4h. The 60-minute default wait is over twice the longest
  observed hold. A lock whose owning process is gone is reclaimed at once, with
  a hard 6h age ceiling to cover PID reuse.
- **Staggered logon delays** - data `PT3M`, code `PT20M`. The lock makes a
  collision *safe*; the stagger makes the order *deterministic*, and the right
  order is evidence first, then the session that reads it.
- **`data-run.ps1` reports git's own words, and retries.** The fatal path said
  only "Could not check out main." and discarded git's stderr, which is why
  diagnosing this needed a reproduction rather than a read of the log. It now
  logs the stderr and retries 5x/10s first: the repo lock excludes the other
  loop, but an editor or a stray `gh` can still hold the index briefly, and
  that must not cost a day.
- **`add-catchup-trigger.ps1` now delegates to `register-tasks.ps1`.** It wrote
  its own `PT3M`-for-both triggers, so running it once would have silently
  restored the collision. Two scripts writing the same triggers is exactly the
  drift the 2026-08-21 retrospective shipped `register-tasks.ps1` to end.

### Evidence / research

A demonstrated failure in the published logs, plus a reproduction. No citation,
no backtest, no IC number - rules 4 and 5 do not bite, because nothing here was
justified by a return.

**The reproduction.** In a scratch repo, with `.git/index.lock` present and
nothing else wrong:

```
no contention   -> exit 0   | Already on 'main'
index.lock held -> exit 128 | fatal: Unable to create '.../.git/index.lock':
                              File exists.
```

Exit 128 is exactly the `$co.ExitCode -ne 0` branch that logged "Could not
check out main."

**15 new tests** in `tests/test_loop_mutual_exclusion.py` (853 in the suite
overall, up from 825 - the other 13 are the parser check below applied to each
script, plus the existing static checks picking up the new file). Five of the
15 drive real PowerShell processes against the real lock, two of them
concurrently: a held lock cannot be taken; a released one can; a lock left by a
dead process is reclaimed; releasing does not drop a lock that now belongs to
someone else; and of six processes started together, **exactly one** holds the
repo. Eleven of the static assertions were re-applied to the pre-fix scripts
pulled from `git show HEAD:` - **all eleven fail** against them.

One of those tests failed first time for the right reason and is worth
recording: six racers all "won", because each acquired and exited without
holding, so every contender found a lock owned by a dead process and correctly
reclaimed it. The lock was right and the test was wrong. The racers now hold
for 15s, which is the case the test is actually about.

Also added, because these scripts are unattended infrastructure: every
`scripts/*.ps1` is now handed to PowerShell's own parser
(`test_parses_as_powershell`). The module previously counted braces as a proxy
for this. All 8 scripts parse.

### Methodology changed

**None, deliberately.** No weight, threshold, metric or formula changed, and
nothing changed about what the tool asserts about how it scored. This is runner
infrastructure, which by precedent (`register-tasks.ps1` 2026-08-21, the
watchdog 2026-08-27) lives in this log and not in
`METHODOLOGY_CHANGELOG.md`. `CLAUDE.md` priority -1 is updated per rule 9.

### Did not do, and why

- **Did not re-run the data loop to recover today.** Nothing was lost.
  `_normalize_performance_history()` drops weekend run dates at generation
  (`improvement_engine.py:120`, priority 0 item 4), so a Saturday snapshot
  contributes **no** evidence. Re-running would have added ~3 MB of
  poorly-compressing JSON to git history for zero observations. The evidence
  base is unmoved for the second consecutive session at 3 effective `1m`
  observations - rule 8's three-session trigger is not met, and the cause is
  structural rather than a defect: `1m` rows mature only as older run dates
  age, and the newest `1w` row (2026-08-21) is exactly what an 08-28 run could
  reach.

### Tried and rejected

- **Making the loser exit rather than wait.** Simpler, and it converts a
  collision straight back into a lost day - the thing being fixed. Waiting
  costs at most one loop's duration out of a 3-4h limit.
- **Staggering the logon delays and stopping there.** It does not fix anything:
  a data run lasts ~13 minutes, so no plausible stagger prevents an overlap,
  and the two tasks can still be started by hand or by different triggers. The
  lock is the fix; the stagger only fixes the ordering.
- **Gating the at-logon catch-up to weekdays.** A weekend data run provably
  accrues no evidence (above), so today's firing was pure cost. But the *code*
  session it also started is this one, which is doing useful work, and
  narrowing a recovery mechanism that priority -1 calls load-bearing is not
  something to do off one Saturday's observation. Left open; see Next.
- **Duplicating the lock functions into both scripts** instead of a
  dot-sourced module. Two copies of an unattended-infrastructure primitive will
  drift, which is exactly what `add-catchup-trigger.ps1` did. Instead
  `test_scripts_static.py` now resolves dot-sourced files when checking for
  undefined functions - narrower than adding names to `KNOWN`, and it still
  catches the 2026-08-11 bug that module was written for.

### Noticed, not fixed

- **`history.py` does not exclude weekend run dates**, though
  `improvement_engine.py` does. A weekend catch-up data run would therefore
  enter the dashboard's rank-history spine as its own day, computed against
  Friday's closes. It would pass the Spearman >= 0.50 comparability gate easily
  (it correlates ~1.0 with Friday), so the visible effect is a spurious extra
  point on the sparkline rather than a wrong comparison. Small, but it is the
  same weekend question as above and the two should be decided together.
- **Verification limits, stated plainly.** The lock primitive is tested with
  real concurrent processes, and both scripts are checked statically and
  parsed. I did **not** run `data-run.ps1` or `nightly-screener.ps1` end to
  end: the first would publish a weekend run, and the second would recursively
  invoke a session inside this one. The first real proof is Monday's 02:00 run.
- The `runs/` test-litter noted on 2026-08-28 is untouched and still real.

### Next

1. **Monday's research note, now missed six times.** Unchanged and a day older.
   Today was a data-loop failure, which outranks everything by rule - but that
   is the sixth consecutive defensible skip, and the rotation has produced one
   research note in a month. Monday should be spent on it unless the data loop
   is actually down.
2. Confirm Monday's 02:00 data run publishes normally with the repo lock in
   place, and that `live_ic_history.csv` gains its 2026-08-24 `1w` row.
3. Decide the weekend question once: should the at-logon catch-up fire on
   weekends at all, and should `history.py` exclude weekend run dates?
