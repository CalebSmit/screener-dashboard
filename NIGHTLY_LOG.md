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
- **The stagger is committed but not live.** Checked after pushing: both
  scheduled tasks still report `MSFT_TaskLogonTrigger | delay=PT3M`. Trigger
  definitions are machine state, and `register-tasks.ps1` unregisters before it
  registers - a failure partway through would leave the machine with no loops
  at all, which is worse than the defect being fixed, so I did not run it
  unattended. **Nothing is broken meanwhile:** the lock makes a simultaneous
  start safe, so the only cost is that the winner is arbitrary rather than
  data-first. To apply it:
  `powershell -ExecutionPolicy Bypass -File scripts\register-tasks.ps1`
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

---

## 2026-08-29 (evening) - Owner-run: the stagger goes live, and a standing rule changes

The morning's CATCH-UP session fixed the logon-trigger collision (see above)
but left `register-tasks.ps1` for the owner to run by hand, reasoning that the
script unregisters both tasks before re-adding them and a failure partway
through would leave the machine with neither.

Told this, the owner's answer was direct: *"never have it leave things for me
to do, it should figure it out on its own, after all, it should be self
improving."*

### Did

**Ran it and verified, in that order.** `powershell -ExecutionPolicy Bypass
-File scripts/register-tasks.ps1` re-registered both tasks; immediately
confirmed with `Get-ScheduledTask` / `Get-ScheduledTaskInfo` rather than
trusting the script's own summary line - `Screener Data Run` triggers at
`delay=PT3M`, `Nightly Screener Improvement` at `delay=PT20M`, both `Ready`,
`StartWhenAvailable`/`WakeToRun` intact. The morning session's caution about a
partial failure was reasonable; the missing step was verifying afterward, not
declining to run it - the script is idempotent, so a bad outcome is fixed by
running it again, not by asking someone else to.

**The more durable change is rule 11.** Added to `CLAUDE.md`: apply a
machine-level fix and verify it in the same session, rather than leaving a
command for the owner. Also added to `prompts/nightly.md`, read at the very
top before Orient, so it is standing operating instruction for every future
session - not a note that only helps because a human happened to be in the
loop tonight. The one exception written into both: if verification genuinely
needs something outside the session's reach, the *next* session inherits it,
never the owner.

### Also confirmed, unrelated to the above

Re-verified all four ship gates independently rather than trusting the
morning's `good/2026-08-29-1231` tag: **853/853 tests, tree clean**. The
morning session's own nightly log (`logs/nightly-2026-08-29_121115.log`) is
truncated after "Invoking Claude Code..." - almost certainly because an
interactive `tail -f` watch on that exact file (mine, checking on the session's
progress) held it open the whole time PowerShell's `Add-Content` tried to write
to it, the same non-fatal `IOException` seen once before on 2026-08-26. The
session's actual work - five commits, tests, the tag - was unaffected, since
`$ErrorActionPreference = 'Continue'` makes a failed log write non-terminating.
Noted so it isn't mistaken for a hang next time: **do not hold a scheduled
run's own log file open with a live tail while it may still be writing to it.**

### Tests

733 -> 853 (morning session's own work; nothing added this evening).

### Next

Everything from this morning's entry still stands - Monday's research note
(five skips now), per-category trend lines, confirming the watchdog's first
real scheduled firing.

## 2026-08-31 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-08-29_121115.log` - catch-up session, shipped to main |
| Data loop published? | `logs/datarun-2026-08-31_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **28 rows, newest 2026-08-24, 3 effective observations at `1m`** (8 raw) |
| Priority 0 | Fixed 2026-08-24, holding |

**Tests:** before 853/853, after 853/853
**Data loop:** healthy - ran 02:00, all coverage/dispersion checks passed, published to main
**Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`, so the
rotation stood. Nothing was deferred.
**Rotation:** ISO week 36, Monday. Research day.

**The research note is written. That is the whole session.** It had been skipped
five consecutive times, each skip for a real defect and each defensible in
isolation, but the rotation had produced exactly **one** note in a month. Today
nothing was broken, so there was no excuse.

### Did

Wrote `research/2026-08-31-size-factor-in-a-large-cap-universe.md`. The question:
the screener spends 5% of composite on `size_log_mcap` = `-ln(mcap)` inside the
S&P 500, a universe with no small caps. Does a size tilt belong here, and is this
the right way to build one?

**Answer: keep it at 5%, and the reason is better than the one it had.** Four
findings, all measured on the published payload from today's 02:00 run:

- **The tilt is junk-seeking on average, exactly as the literature predicts.**
  The 50 names it promotes most vs the 50 it demotes: median cap $15.0B vs
  $264.1B, quality 49.3 vs 55.6, risk 44.9 vs 60.1, volatility 0.33 vs 0.30.
  That is the pattern Asness et al. (2018) identify as the reason raw SMB fails,
  and that MSCI concedes in its own Low Size brochure.
- **But the composite already controls for the junk where the product points.**
  Comparing the top 50 with and without the size category: quality moves
  **-0.20**, risk **-0.47**, median cap $38.5B -> $30.4B. The 22% quality weight
  removes the junk before size can promote it. Combined with the S&P 500's own
  GAAP-profitability entry gate, this screener is running much closer to the
  quality-controlled version of the factor (t = 4.89) than the raw one
  (t = 1.23). **That defence did not exist before today; the weight was
  previously unexamined.**
- **The `log` in `size_log_mcap` is provably inert.** The pipeline ranks the
  metric one step later, and percentile ranking is invariant to any monotone
  transform: `max |rank(-log mcap) - rank(-mcap)| = 0.0000000000`. The log is
  the entire mechanism by which MSCI keeps its size tilt gentle - their worked
  example turns a 9x cap gap into a **4.6pp** weight gap. This screener turns the
  same gap into a **15-57 point** score gap (CAT $368B score 1 vs UAL $36B score
  59). Not a defect; the tilt is simply far more aggressive than the
  practitioner standard it resembles, and nobody chose that.
- **Size is a genuine independent bet**, not a restatement: R^2 = 0.279 when
  regressed on the other seven categories, so 72% is unspanned.

### Evidence / research

- **Asness, Frazzini, Israel, Moskowitz & Pedersen (2018)**, "Size matters, if
  you control your junk", *JFE* 129(3), 479-509. Raw SMB 1926-2012: 23 bps/month,
  **t = 2.27**; insignificant over Banz's own sample; CAPM alpha 12 bps
  (t = 1.12); FF3+UMD alpha 14 bps (t = 1.23). **Adding QMJ: 49 bps, t = 4.89.**
  Outside January the raw effect is **-0.04%, t = -0.32** - no size effect for
  eleven months of the year - which QMJ restores to +38 bps (t = 3.62). Within
  quality quintiles: 50 bps, t = 3.18, but **among the junkiest quintile the
  relation is not monotonic and is insignificant**. Adding RMW/CMA doubles alpha
  to 33 bps (t = 2.81). *Numbers are from the Jan-2015 working paper, which is
  the version I could read in full; flagged as such in the note.*
- **Harvey, Liu & Zhu (2016)**, *RFS* 29(1), 5-68 - given factor data-mining,
  **t > 3.0** is the appropriate bar. Raw SMB (2.27) fails it; quality-controlled
  SMB (4.89) clears it. That is the whole argument in one line.
- **Banz (1981)**, *JFE* 9(1), 3-18 - the original. 1936-1975 quintile spread
  ~7.19%/yr equal-weighted vs ~2.73%/yr value-weighted (secondary source, flagged
  in the note). The 2.6x gap is itself the warning: the effect lives in the
  smallest names.
- **MSCI Low Size Indexes brochure** (read in full) - weights in proportion to
  **1/ln(mcap)**, applied to large+mid cap, reweighting not excluding. Worked
  example: $90B vs $10B -> 47.7% / 52.3%. MSCI states the log is chosen because
  it "minimized the impact of the largest values", and concedes that size
  strategies come "at the expense of relatively poorer quality and more volatile
  stocks".
- **Barra USE4** - Size is a style **risk** factor, standardly neutralised rather
  than harvested. This is the real academia/practice split and the note explains
  why it is genuine: the academic premium is long-short, quality-controlled and
  monthly-rebalanced, i.e. not directly investable by a long-only large-cap
  manager.
- **S&P DJI** - S&P 500 eligibility requires positive GAAP earnings in the most
  recent quarter *and* over the trailing four quarters. A real junk screen at the
  universe boundary, and a point in this screener's favour the literature alone
  would not surface.
- **S&P 500 Equal Weight** - the closest live analogue to a within-S&P-500 size
  tilt: **+63 bps/yr since 1990**, attributed to smaller size, value orientation
  and an anti-momentum bias (three of which match the tilt profile I measured).
  But cap weight beat it by **~32% over 2023-2025**. A 63bp edge that can lose
  32% over three years is not something a 5% weight delivers reliably.

**The gap I could not close, stated plainly in the note:** no cited source
establishes a size premium *within* the top two US market-cap deciles, which is
the entire S&P 500. Asness et al.'s "not concentrated in microcaps" means the
effect is not *exclusively* microcap - not that it survives among the 500 largest
US companies. The screener is extrapolating, and the note labels it as such
rather than papering over it.

### Methodology changed

**None.** No weight, threshold, metric or formula moved, and no production code
was touched - correct for a research day. The note's recommendation is explicitly
"keep 5%, and here is the argument it was missing".

### Tried and rejected

- **Raising the size weight on Asness et al.'s 49 bps / t = 4.89.** That premium
  is measured on a long-short, quality-controlled, full-cross-section portfolio.
  This screener is long-only, large-cap-only, and quality-controlled only
  incidentally. Importing the effect size would be precisely the "pile of good
  ideas" failure `CLAUDE.md` warns against.
- **Cutting size to zero on the "premium died after 1981" literature.** Defeated
  by Finding A - measured on the live payload, the junk mechanism that kills the
  raw premium is neutralised in this screener's top 50.

### Found while researching, NOT fixed (needs its own session)

**Winsorising before ranking destroys information and cannot add any.**
`winsorize_metrics()` runs at `factor_engine.py:3429`, four lines before
`compute_sector_percentiles` at `:3433`. Ranks are already immune to outliers, so
clipping the extreme 1% first cannot change any ordering - it can only create
ties. Measured: **27 continuous metrics show the 1%/99% tie signature, collapsing
282 (stock, metric) cells** (discrete metrics like `piotroski_f_score` excluded -
their ties are genuine).

It has a **user-facing consequence**: the winsorized values are what the dashboard
publishes as `raw`, so the live site currently shows **AAPL, NVDA, GOOG, GOOGL,
MSFT and AMZN with an identical market cap of $2,873.8B**. Same six-way tie at the
bottom (TTD, AOS, TAP, NCLH, BLDR, MOS).

Deliberately left alone today: it is outside the size question, it touches all 44
metrics, and it needs its own changelog entry and a test that published market
caps are distinct. Not left for the owner - it is the next session's work.

### Also did

Corrected `research/README.md` (rule 9). It still described the old split
rotation - "Monday and Tuesday sessions produce notes here" and "Tuesday's design
section" - which the 2026-08-21 retrospective replaced with one self-contained
Monday, Tuesday being the product day. A note written to that spec would have
handed its design section to the wrong day.

### Next

1. **The winsorise-before-rank defect above.** It is the only concrete,
   demonstrable, user-facing problem this session found, it has a measured blast
   radius (282 cells, 6 wrong market caps on the live site), and it is cheap.
2. **Wednesday's synthesis has a real question waiting**, which is the first time
   in weeks that has been true: are `size` and `investment` two 5% bets or one
   10% bet? They correlate +0.281, they are one metric each, and Asness et al.
   show CMA absorbs part of SMB's alpha.
3. Per-category trend lines over the full history (priority 2's remainder).

## 2026-09-01 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-08-31_060001.log` - "Run complete: shipped to main" |
| Data loop published? | `logs/datarun-2026-09-01_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **29 rows, newest 2026-08-25, 3 effective observations at `1m`** (8 raw) |
| Priority 0 | Fixed 2026-08-24, holding |

**Tests:** before 853/853, after **872/872**
**Data loop:** healthy - ran 02:00, all coverage and dispersion checks passed, published to main
**Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
**Rotation:** ISO week 36, Tuesday. Product day - and the work was a product defect,
so no swap was needed.

### Did

**Fixed the live site publishing false numbers.** The dashboard showed AAPL,
NVDA, MSFT, GOOG, GOOGL and AMZN with an identical market capitalisation of
**$2,802.0B**. Nvidia's true figure is **$5,331.2B** - understated by 47%, or
$2.5 trillion. This is a product-day finding in the most literal sense: it is
what a user sees, and it is wrong.

The cause was `winsorize_metrics()`, which clipped the top and bottom 1% of
every metric onto one boundary value four lines before the ranking step. The
clipped number was then published as the stock's `raw` value. Removed; replaced
by `flag_metric_outliers()`, which reports the same tails into the data-quality
log and does not touch the frame. Changelog 2026-09-01;
`tests/test_no_winsorization.py`, 18 tests.

This was the item the 2026-08-31 session identified as the next session's work.
It is done, and the diagnosis it left was right in substance - though its two
headline numbers were slightly off and are corrected here: the tie value is
**$2,802.0B**, not $2,873.8B, and on the current payload the signature covers
**33 continuous metrics / 301 collapsed cells**, not 27 / 282.

**Measured blast radius**, all on the published `dashboard_data.js` from today's
02:00 run:

- **301 (stock, metric) cells across 33 continuous metrics, on 159 of 502
  stocks**, carried a clipped value instead of the fetched one.
- **58 (metric, sector) tie groups** collapsed two or more stocks onto a single
  percentile. Worst: 4 Energy names shared one `beta` rank in a 21-stock sector,
  spanning **14.3 percentile points**; 5 Utilities shared one `volatility` rank
  (12.9 pp); 6 Information Technology names shared one `return_6m` rank (6.8 pp).

**Four call sites, not one.** `run_screener.py`, `backtest.py`, `run_audit.py`
and - missed on the first pass - `factor_engine.py`'s own end-to-end path. The
AST guard written for this change is what found the fourth, and it also found
19 stale per-metric "Winsorize 1/99 pctile" descriptions in `run_audit.py`. Both
guards fail against the pre-change code (5 offending calls, 32 offending
user-facing strings), so they are not vacuous.

**Verified end-to-end on the real universe, not just on fixtures.** Took the
502-row cached frame, restored the six true market caps, and pushed it through
the production `flag_metric_outliers` -> `compute_sector_percentiles` path: six
distinct values, six distinct percentiles. AAPL/NVDA/MSFT separate from a shared
2.74 into 2.74 / 1.37 / 4.11; GOOG/GOOGL from a shared 6.25 into 8.33 / 4.17.

### Evidence / research

Three independent lines. None is a backtest and none is an IC from this system,
per rules 4 and 5.

- **A proof, not an estimate.** `compute_sector_percentiles()` is
  `Series.rank(pct=True)`. A rank transform is invariant under *any* monotone
  transform of its input, so clipping the tails cannot change one ordering.
  Winsorizing is monotone but not injective - it maps a whole tail to one number
  - and manufacturing ties is therefore the *only* effect it can have. Both
  halves are locked down by tests: an exponential rescale of the inputs leaves
  every percentile identical; clipping four tail values collapses four distinct
  ranks onto one and leaves the rest of the distribution untouched.
- **A documented user-facing failure**, per the mandate's fourth category:
  the six market caps above, with the true values fetched from the same source
  the screener uses.
- **In-repo evidence that it actively hid data errors.**
  `METHODOLOGY_CHANGELOG.md` 2026-08-26 records that MNST's corrupt
  `volatility_1y` of 1.77 was clipped to 0.845, "which merely made Monster
  Beverage look as volatile as SMCI". An implausible value is the signal a feed
  has broken; clipping deleted exactly that signal. That is why the replacement
  logs the tails instead of discarding them.

The stated rationale in the docs was simply false, and is corrected rather than
softened. `SCREENER_OVERVIEW.md` Step 2 read "Extreme outliers can distort
rankings" - they cannot, for a rank-based screen.
`Multi-Factor-Screener-Blueprint.md` said volatility and beta were winsorized
"to prevent extreme outliers from distorting the distribution". Both now explain
why that is wrong, which matters more for the investment-club audience than the
fix itself does.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-09-01** - winsorization removed from the
  scoring path; `flag_metric_outliers()` added; config key
  `data_quality.winsorize_percentiles` renamed `outlier_report_percentiles`
  (old name still read as a fallback, still accepted by `schemas.py`).
- No category weight, metric weight or threshold moved.
- Docs corrected in the same commit: `SCREENER_OVERVIEW.md` (regenerated from
  `run_screener.py`, its source), `Multi-Factor-Screener-Blueprint.md`,
  `README.md`, `config.yaml`, `plan/dashboard-inventory.md` (rule 9).

### Tried and rejected

- **Fixing only the display - publish unwinsorized `raw` but keep clipping for
  scoring.** Smaller and purely a product change, but worse: it leaves the 58
  manufactured tie groups in the ranking and makes the tool harder to explain,
  because the number shown would no longer be the number scored. The coherent
  fix was the one the maths already licensed.
- **Removing `metric_clamps` in the same pass.** Superficially the same shape,
  but a different argument - a domain judgement that a forward EPS growth above
  150% is not a credible input, rather than a claim about outliers distorting
  ranks. On this run the clamps are not even binding (observed maxima 0.978
  against a 1.50 bound; 0.583 against 1.00), so nothing was gained by touching
  them. Left alone and flagged in the changelog. Whether a non-credible value
  should be clamped or withheld as NaN is a real open question.
- **Rewriting the dated audit reports** (`FORENSIC_AUDIT_REPORT.md`,
  `INSTITUTIONAL_AUDIT_REPORT*.md`, `HARDENING_REPORT.md`,
  `HEDGE_FUND_REVIEW_FINDINGS.md`, `IMPLEMENTATION_PLAN.md`), which all describe
  winsorization approvingly. They are records of what was true when written;
  editing them would be rewriting history rather than correcting documentation.

### Not done, and why - the fix is not on the live site yet

The public site still shows the wrong market caps as of this commit. I did
**not** force a second full-universe fetch to republish today, and that is a
judgement call worth stating plainly rather than burying:

1. The 02:00 data run already fetched the universe today. A second full fetch
   the same morning is the condition that produces the 10-25% ticker failures
   noted at priority 1, and publishing a degraded run to the public site would
   be a worse outcome than the defect it fixes.
2. The health gates that protect publication live in `data-run.ps1`, which also
   takes the shared repo lock and pushes to `main` - running it mid-session from
   a branch would fight this session's own git state.

This is not left for the owner (rule 11): the 2026-09-02 02:00 data run does it,
and **two independent mechanisms guarantee it cannot reuse the clipped cache**.
The cache key hashes `data_quality`, and renaming the key moved it from
`19c853468405` to `2bde439e06ad`, so `_find_latest_cache()` cannot see the old
files at all - verified this session. Independently, `factor_scores` is bounded
by the price tier (1 day) and `cache_is_usable()` is exclusive, so only a cache
written today is reusable anyway.

**Verification for the next session, and it is one command:** after the 09-02
data run, confirm the six megacaps carry six distinct market caps in
`dashboard_data.js`. If they are still identical, both mechanisms failed and
that is the thing to investigate before anything else.

### Also found, not fixed

`data-run.ps1` logs **"Improvement engine: 29 raw IC row(s) (effective count
unavailable)"**. That is the raw row count - the number `CLAUDE.md` says
explicitly is "how this went wrong the first time" - printed as the loop's daily
evidence-base readout, with the effective count it should be reporting marked
unavailable. `analyze_ic_trends()` returns `_n_observations` perfectly well when
called directly; this session read 3 effective at `1m` from it in one line. The
nightly readout is the one place the number is looked at without thinking, so it
is the worst place for it to be the misleading one. Cheap to fix, and it is a
data-loop instrumentation defect rather than a product one.

### Next

1. **Confirm the fix reached the live site** after the 09-02 02:00 data run -
   six distinct megacap market caps in `dashboard_data.js`. One check, and it
   closes this out.
2. **Make `data-run.ps1` report the effective observation count**, not the raw
   row count. See above.
3. Per-category trend lines over the full history (priority 2's remainder), and
   the movers panel still cannot distinguish "moved on new information" from
   "moved because inputs went missing" even though `Composite_Confidence`
   carries that fact - a real product gap for a future Tuesday.

---

## 2026-09-01 (evening) - Owner-run: a general audit for smoothness, three real findings

Owner's brief: "make all necessary updates for it to run as smooth and
effective moving forward." Interpreted as a health/reliability audit rather
than a specific feature - checked scheduled tasks, repo hygiene, and whether
CLAUDE.md's own priority claims still matched reality, then fixed what was
actually wrong rather than inventing scope.

### Did

**1. Corrected a stale claim before acting on it.** Priority 1 said "~10-25%
of tickers fail per run." Checked the last 15 data-run logs directly rather
than trust it: every single one, back to 2026-08-10, reports 0 fetch
failures. Updated CLAUDE.md to say so, with a note to re-verify periodically
rather than let the record drift stale in either direction again.

**2. Found and fixed a permanent false alarm.** `validation/data_quality_log.csv`
flagged four bank-only metrics as "High severity - missing >50%" on every run
since launch - 88.4%/88.2%, unchanging. Traced it: only ~58 of 502 stocks are
banks, and these metrics are correctly absent from every non-bank by design.
The drift check scored missing-% against the whole universe instead of the
population a metric applies to; the coverage filter a few hundred lines away
in the same function already did this correctly and the drift check simply
never matched it. Extracted `_metric_missing_pct()`, scoped by the existing
`_BANK_ONLY_METRICS`/`_NONBANK_ONLY_METRICS` sets. 7 tests.

Worth naming plainly: a permanent "High severity" alert that never means
anything is the same failure shape the loop watchdog was explicitly designed
to avoid (CLAUDE.md rule 7's reasoning) - it trains a reader to stop looking,
which is exactly when a real drift would go unnoticed.

**3. Found the synthetic-data refusal only covered one of two entry points.**
`run_screener.py` refuses to fabricate data on a failed fetch (fixed
2026-08-11). `factor_engine.py` has its own independent `main()` - unreachable
from the scheduled loops, reachable by anyone running it directly - and it
still had the exact pre-fix behaviour: unconditional fabrication, no flag, no
refusal. Fixed the same way. 4 new tests, confirmed to fail against the
pre-fix file before trusting them.

**4. Nightly branches were not actually self-cleaning.** `git branch` showed
`nightly/2026-08-10` and `nightly/2026-08-27` still present, both fully
merged weeks ago. The delete-after-merge call existed but piped its exit code
to `Out-Null`, so a rare failure (transient lock, interrupted run) left debris
with zero visibility. Fixed two ways: the delete now logs a `WARN` on
failure, and every run sweeps any local `nightly/*` branch already merged
into `main` at startup, so a miss self-heals on the next run instead of
accumulating. Did the one-time cleanup directly too - deleted both stray
branches and pruned a stale `prefix-check` worktree left over from
2026-08-27 (same `.git/worktrees/` permission issue noted that day; the
`os.chmod` + `shutil.rmtree(onerror=...)` workaround from `prune_artifacts.py`
cleared it).

### Methodology changed

None. All four fixes are reporting, refusal-gate, and repo-hygiene changes -
nothing in `raw`, `pct`, or `Composite` moved.

### Verified, not just reasoned about

- Manually swept the sweep logic against the real repo before committing it:
  found exactly the two known-stray branches, deleted them, re-ran to confirm
  idempotency (clean on the second pass).
- Ran the new factor_engine.py tests against the pre-fix file via `git show`
  before trusting them - all three key assertions correctly failed there.
- Full suite, dry-run, dashboard artifacts and tree all re-verified
  independently after merging, not just trusted from the commit.

### Tests

872 -> 883.

### Next

Everything from the 09-01 morning entry still stands, plus: watch
`validation/data_quality_log.csv` on the next run to confirm the four
bank-metric entries no longer appear as High severity.

---

## 2026-09-02 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.

**Tests:** before 882/883 (1 pre-existing failure), after 904/905 (same one)
**Data loop:** healthy - `logs/datarun-2026-09-02_020001.log` ends "Data loop
complete", HEALTH: PASS, 0 fetch failures, price coverage 502/502 (100%)
**Evidence base:** `improvement/live_ic_history.csv` holds **30 rows**, newest
**2026-08-26**, and **3 effective observations at the `1m` horizon** (8 raw)
against a gate of 8. Moving: 29 -> 30 rows and 2 -> 3 effective since 09-01.
**Priority 0:** DONE 2026-08-24, not reopened.
**Owner queue:** empty. Nothing deferred.

### Did

**1. Verified last session's market-cap fix reached the live site.** The one
check 09-01 asked for. The six megacaps now publish six distinct market caps -
NVDA $5,250B, AAPL $4,745B, GOOGL $4,097B, MSFT $3,720B, AMZN $2,750B, META
$1,474B - where they previously shared one winsorized value. Closed.

**2. The synthesis, and the change it produced: the risk category was 30%
momentum wearing a risk label.**

Monday's note (`research/2026-08-31-size-factor...`) left three candidates and
told Wednesday to decide whether they fit the screener as a whole. Doing that
properly meant dropping size and measuring the entire 8x8 category structure.
One number dominated it: **momentum ~ risk = +0.516**, the largest of the 28
pairs, half again the next largest. Spanning regressions agreed - risk (R^2
0.378) and momentum (0.352) were the *least* independent of the eight
categories, and they carry ~25% of composite weight between them.

The cause is not the shared `Ticker.history()` call `CLAUDE.md` already notes
- volatility and 12-month return share that source and correlate -0.013. It is
that two of the five scored risk metrics were not risk metrics.
`factor_engine.py` builds `sharpe_ratio` (:1923) and `sortino_ratio` (:1946)
from the **same numerator**, `(return_12m - rf)`, and across the S&P 500 the
spread in returns swamps the spread in dispersion. Measured on published
percentiles, identically across three consecutive runs:

| Pair | 08-31 | 09-01 | 09-02 |
|---|---|---|---|
| sharpe ~ sortino | +0.993 | +0.994 | +0.993 |
| sharpe ~ return_12_1 | +0.940 | +0.940 | +0.944 |
| **sharpe ~ volatility** | **+0.029** | **+0.032** | **+0.025** |

Five metrics that were three things: two of them were each other, and both
were momentum rather than risk.

**The user-facing consequence is what makes it a defect.** SNDK published a
**risk score of 31.1** next to a **momentum score of 94.2**; on dispersion
alone its risk score is **1.6**. Same for MRNA (34.7 -> 6.8), VRT (29.1 ->
3.2), FIX (37.5 -> 12.0), MU (41.6 -> 17.2), WDC (39.9 -> 15.7) - every one a
high-momentum name. The public site was telling a student that a violently
volatile stock was mid-pack on risk *because it had gone up*.

And `SCREENER_OVERVIEW.md` was actively asserting the opposite: "Five metrics
give a more complete risk picture than two." A +0.993 correlation between two
of the five refutes that sentence directly.

Shipped: `sharpe_ratio` and `sortino_ratio` to **weight 0** within risk;
30/20/20 renormalised over 70 to **42.86 / 28.57 / 28.57**, preserving relative
emphasis exactly. Both ratios stay computed and stay on the drilldown - the
same weight-0 treatment `proximity_52w_high` and `peg_ratio` already get. They
are informative; they are not risk. Changelog 2026-09-02;
`tests/test_risk_category_independence.py`, 13 tests.

**Effect on the whole:** risk R^2 0.378 -> **0.188**, momentum 0.352 ->
**0.115** - the two least-independent categories become two of the most, and
nothing else moves more than 0.012. Composite Spearman 0.990, median rank
change 10 places, 3 of the top 50 change (out DELL/FOX/STLD, in ADBE/CB/EXE).

**3. The daily evidence readout was printing the misleading number.** 09-01
flagged that `data-run.ps1` logs "30 raw IC row(s) (effective count
unavailable)". The correct code already existed - that was the *fallback*
firing, because the primary path was an inline `python -c` here-string that
failed silently. So the one number the loop prints without anyone thinking
about it was the raw row count, which `CLAUDE.md` rule 8 calls out by name as
"how this went wrong the first time". 30 reads as though the gate were long
cleared; the truth was 3 effective against 8.

Moved into `scripts/report_evidence.py`. The deeper point is not the quoting:
logic inside a here-string **cannot be unit-tested**, which is why nothing
caught it. It is now tested - `tests/test_evidence_readout.py`, 9 tests, three
confirmed to fail against the pre-fix script. The failure branch no longer
substitutes a raw count at all; it logs a `WARN` saying the number is
unavailable, because an authoritative-looking wrong number is worse than an
admitted gap.

### Evidence / research

- **Ang, Hodrick, Xing and Zhang (2006)**, "The Cross-Section of Volatility and
  Expected Returns," *JF* 61(1), 259-299. Sorts on **idiosyncratic
  volatility**; quintile 1-minus-5 spread over **1%/month**, robust at
  **-0.63%/month, t = -3.30** excluding the smallest growth firms.
- **Frazzini and Pedersen (2014)**, "Betting Against Beta," *JFE* 111(1), 1-25.
  Selects on **beta**; BAB factor Sharpe **0.78** (1926 - Mar 2012). Note what
  they do with the Sharpe ratio: **evaluate the resulting portfolio**, not rank
  the cross-section. That is the correct use of the statistic and precisely the
  use this screener was not making of it.
- **Practice:** Barra USE4 builds its Residual Volatility style factor from
  dispersion descriptors (daily standard deviation, cumulative range, residual
  sigma), Beta being its own; MSCI Minimum Volatility optimises against those
  Barra BETA/RESVOL exposures while constraining every *other* style factor to
  +/-0.25 sd. No index provider selects for low risk with a Sharpe ratio.
  (Descriptor *weights* 0.74/0.16/0.10 are from a **secondary** source - the
  primary MSCI PDF was not text-extractable this session, flagged as such in
  the note and changelog. That the descriptors are all dispersion measures is
  not in doubt.)
- **Unusually, academia and practice agree here.** Worth saying, because
  Monday's note found a genuine divergence on size. On how to measure risk
  cross-sectionally they do the same thing, and this screener was doing
  something else.
- Full write-up: `research/2026-09-02-category-independence-synthesis.md`,
  including the 8x8 matrix and spanning tables.

### Methodology changed

- `METHODOLOGY_CHANGELOG.md` **2026-09-02 - "The risk category was 30% momentum
  wearing a risk label."** Metric weights within `risk` only. **Category
  weights deliberately untouched** - the finding is that risk was mismeasuring
  risk, not that risk deserves more or less composite weight. Re-deciding those
  needs its own research, and doing both at once would make neither
  attributable.
- `tests/fixtures/golden_scores.parquet` regenerated. Before regenerating I
  inspected the diff to confirm it was confined to `risk_score` and downstream:
  all 50 preceding columns (Ticker, Sector, every one of the 44 metrics)
  compared **equal**. No raw metric value moved.
- `SCREENER_OVERVIEW.md` regenerated from live config so the public doc does
  not contradict the config for a day. Metric counts moved 34 -> 32 scored and
  10 -> 12 candidates automatically.

### Tried and rejected

- **Raising or lowering any of the eight category weights.** Tempting, since
  ~3% of composite that was labelled risk was behaving as momentum. Rejected:
  that is the categories *becoming what the documentation always said they
  were*, not a new bet, and bundling it would make neither change
  attributable.
- **Rebalancing among the three surviving dispersion metrics.** The 42.86 /
  28.57 / 28.57 is exactly the old 30/20/20 renormalised. Picking new relative
  weights would be a second claim I did not research today.
- **Deleting Sharpe and Sortino outright.** They are genuinely informative to a
  reader - they are just not risk. Weight 0 with continued display is the
  established house idiom and keeps the information.
- **An "effective number of independent bets" metric.** I computed it (entropy
  of the weighted correlation eigenspectrum): **5.18 -> 5.29 of 8**. It barely
  moves because it is dominated by weight concentration rather than
  correlation - quality and valuation alone are 42%. Wrong instrument.
  Reported in the note rather than quietly dropped, so no future reader expects
  a bigger number from it.
- **Monday's candidate 2, `size` + `investment` as one 10% bet.** Measured and
  **closed as two bets**: they correlate +0.280 but are 73% and 84% unspanned
  by the other seven. Merging would lose information.
- **Monday's candidate 1, the size tilt's aggressiveness.** Left **open,
  deliberately**, and recorded as such on the Monday note rather than dropped.
  A real question about one 5% category's transfer function, which lost today
  to a 25%-of-weight overlap. Its own refutation criterion (does compressing
  change the top 50 by fewer than ~2 names?) should be measured before any code
  is written.

### Not verified, and why

`scripts/data-run.ps1` changed, and PowerShell execution is blocked in this
session's sandbox, so I could not reproduce the original inline-invocation
failure or run the new one end to end. What I did instead: made the logic a
Python file that **is** directly executable and tested here (9 tests, run
green, and verified to fail against the pre-fix script), plus the existing
static PowerShell checks in `tests/test_scripts_static.py`. This is not left
for the owner - the **2026-09-03 02:00 data run** exercises it. If its log line
still reads "raw IC row(s)" or the new `WARN`, that is the next session's first
job.

### Next

1. **Confirm both changes on the live site after the 09-03 02:00 run.** Two
   checks: the published momentum/risk category correlation should read near
   **+0.15**, not +0.52; and the log's evidence line should read "**3
   effective (8 rows) at the 1m horizon**", not a raw count.
2. **Monday's candidate 1** - measure whether compressing the size tilt changes
   the top 50 by more than ~2 names *before* writing any code.
3. **`growth` ~ `investment` = -0.331 is an undocumented internal tension.**
   `investment` rewards low asset growth (CMA proxy); `growth` rewards fast
   revenue and EPS growth. Companies growing fast grow assets to do it, so the
   screener rewards growth with one hand and penalises how it is funded with
   the other. That may well be correct - it is close to what the five-factor
   model does - but it is currently an accident of construction rather than a
   documented choice, and it is 18% of composite weight. Worth a research day.

---

## 2026-09-02 (evening) - Owner-run: investigating last night's first-ever ship-gate failure

Owner asked how last night's run went. Last night's 6 AM session (Wednesday,
SYNTHESIS) logged `SHIP GATES FAILED: tests. Not merging.` - the first gate
failure in this project's history. What follows is what that actually meant,
found by checking directly rather than trusting the log's own conclusion.

### What actually happened, in order

1. The session did real work: found the risk category was scoring two metrics
   that were momentum, not risk, fixed it, wrote a synthesis note.
2. Per `prompts/nightly.md`'s own instructions, the session merged and pushed
   `main` itself, as its normal final step. That landed on `origin/main` at
   06:24:22 - confirmed from the commit's own timestamp.
3. `nightly-screener.ps1`'s independent re-verification then ran a fresh copy
   of the test suite two minutes later and hit `test_parquet_roundtrip` FAIL.
4. The wrapper's recovery path reset **local** `main` back to the last known
   good commit and logged "main is untouched." **That claim was wrong.** The
   session's own push in step 2 had already reached `origin/main` - the
   branch GitHub Pages actually serves - before this independent check ever
   ran. A local reset cannot undo a push it did not make.

### Checked, not assumed

- **The live public dashboard was never at risk.** Fetched it directly:
  serving correct data from the clean 02:00 run.
- **Last night's risk-category fix was legitimate**, not something that
  needed reverting. `git log origin/main` showed the merge had genuinely
  landed; re-running the exact same suite against that exact commit found no
  regression in it.
- **`test_parquet_roundtrip` is not flaky - it is a genuine, reproducible
  test-isolation bug**, and I only believed "flaky" for about ten minutes
  before it failed again, on demand, on this machine, tonight. Root cause:
  it calls the real `write_scores_parquet`/`_find_latest_cache` against the
  real, shared `cache/` directory. `_find_latest_cache("factor_scores")` with
  no hash filter globs every `factor_scores_*.parquet` for *any* date and
  reverse-sorts filenames; a hex hash starting with a-f sorts ahead of one
  starting with a digit, so a same-day real pipeline cache file
  (`factor_scores_2bde439e06ad_20260902.parquet`) beat the test's own
  `factor_scores_20260902.parquet` and the test silently read 502 real
  production rows instead of its own 503 synthetic ones. This is exactly the
  test-isolation gap CLAUDE.md priority 8 already named, just never traced to
  a specific test before. Fixed: isolated to `tmp_path` via `monkeypatch`,
  confirmed deterministic across 4 consecutive runs with the real same-day
  cache file present and untouched throughout.

### Did - the actual fix, tested before it ever touches a live repo

**`scripts/revert-bad-merge.ps1`.** If `origin/main` is still at the last
known-good commit, it is a verified no-op (the common case - most gate
failures really are caught before any push happens). If `origin/main` has
moved past it - the session's own push landed before the independent check
caught a problem - it reverts what changed there, as a **new commit, never a
rewrite** (rule 2), and only pushes after confirming the reverted tree is
**byte-identical** to the last known-good state. If that verification cannot
be made exact, it refuses to push and leaves the state for a human rather
than guessing - a smaller, clearer problem than pushing something unverified
to the same branch that just failed verification.

Built and proven against a real origin+clone git sandbox before it was wired
into `nightly-screener.ps1` at all: the no-op case, a merge-commit divergence
(the exact 09-02 shape), and a fast-forward divergence, each checked by
reading the actual resulting file content and diffing trees, not by trusting
exit codes. 7 dedicated tests plus 5 more from the existing `scripts/*.ps1`
auto-discovery in `tests/test_scripts_static.py`.

**Also fixed the false claim itself.** `nightly-screener.ps1` no longer prints
"main is untouched" unconditionally after a gate failure - it now says so only
after the revert script has actually confirmed it.

### Methodology changed

None. Everything tonight is process/test-infrastructure; last night's real
methodology change (risk category) already has its own changelog entry from
that session.

### Not done, noticed while looking

**Remote `nightly/*` branches are not being cleaned up.** Last session's
smoothness pass fixed *local* branch accumulation; `origin` still carries
`nightly/2026-08-10` through `nightly/2026-09-02`, several long since merged.
Low urgency, real debt - worth a `git push origin --delete` sweep of anything
`--merged main` on a future session.

### Verified after merging, not left to the next session

Reconciled local `main` (one unpushed brief commit) with `origin/main`
(last night's legitimate work) via a plain merge - no rebase, no rewrite.
Re-ran all four ship gates independently against the merged result before
pushing: 917/917, dry-run PASS, dashboard artifacts intact, tree clean.

### Tests

895 (local, before merge) + last night's 22 -> **917**.

### Next

- Sweep merged remote `nightly/*` branches.
- Confirm tonight's fix holds through a real gate failure someday (nothing to
  do now - it is tested, not merely reasoned about - but worth remembering
  this exists if `nightly-screener.ps1`'s log ever again claims "main is
  untouched" after a merge).
- Everything from last night's own entry still stands.

---

## 2026-09-03 - BUILD. Implement what the week's research justified. Write tests alongside the code.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-09-02_060001.log` - ran; gate failure investigated and resolved the same evening (see 09-02 evening entry) |
| Data loop published? | `logs/datarun-2026-09-03_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored, 0 fetch failures, price coverage 502/502 (100%) |
| Evidence base | **31 rows, newest 2026-08-27, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
| Priority 0 | DONE 2026-08-24, not reopened |

**Tests:** before 917/917, after **938/938** (+21, no pre-existing failures)
**Data loop:** healthy
**Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
**Rotation:** ISO week 36, Thursday. Build day.

### First: the two checks 09-02 asked for, both confirmed

1. **Wednesday's risk-category fix reached the live site.** On the 09-03
   published run, `momentum ~ risk` is **+0.100** (it was +0.516 before the fix,
   and 09-02 predicted "near +0.15"). `sharpe_ratio ~ volatility` is +0.012,
   confirming the two dropped metrics carried essentially no dispersion signal.
2. **The evidence readout prints the honest number.** The 02:00 log reads
   "3 effective (8 rows) at the 1m horizon" - the effective count, not the raw
   row count that `CLAUDE.md` rule 8 names as how this went wrong the first
   time. `scripts/report_evidence.py` works in production, which the 09-02
   session could not verify from its sandbox.

### Did

**1. Closed Monday's Candidate 1 - the size tilt - by measurement, and the
answer was "do not change the methodology; the documentation was false".**

This was the week's open thread: Monday found the tilt far more aggressive than
the practitioner standard it resembles, Wednesday deferred it to a
pre-registered measurement, today ran it. All numbers from the 09-03 published
run, recomputed through the **real** `factor_engine.compute_composite` (exact
reproduction of the published composite, err 0.0).

**The pre-registered criterion did not discriminate.** "Refuted if the
compressed version changes the top 50 by fewer than ~2 names" returns 1, 2, 3
or 5 names depending only on which steepness constant you pick - and there is no
evidence for any particular one. A criterion whose verdict is set by a free
parameter is not a criterion.

**The number that settles it:** deleting the size category outright - the
largest possible change to it - moves only **6** of the top 50 (Spearman 0.981).
Every compression variant sits *inside* that footprint, so none is
distinguishable at the decision surface from turning the category down or off.

**And the argument I think is genuinely new:** compressing toward MSCI is a
*disguised deletion*. Imported honestly onto this universe, MSCI Low Size's
1/ln(mcap) weighting turns a **798x** spread in market cap into a **1.295x**
spread in weight (0.1678%-0.2173% against an equal weight of 0.1992%). Rendered
as a 0-100 score that either stretches back out to fill the range (reproducing
the current tilt, achieving nothing) or stays flat (arithmetically ~ setting the
size weight to zero). So the question was never "which transfer function" but
"how much weight" - and burying a weight decision inside a formula is the
opposite of explainable. The screener already has an honest knob for that, and
Monday's research concluded it stays at 5%.

**What did change: `SCREENER_OVERVIEW.md` was making a false claim.** The
canonical public methodology page - read by the investment-club audience -
justified the size metric with:

> "Using the log transform compresses the enormous range of market caps ($2B to
> $3T+) into a more linear scale that ranks sensibly."

The *same document* contradicts this 44 lines later, where it correctly says
every metric is scored by rank and "a rank does not care how far away an outlier
is". Re-verified today against the real `compute_sector_percentiles`:
`rank(-log mcap)` is identical to `rank(-mcap)`, `rank(-sqrt mcap)`,
`rank(-cbrt mcap)` and `rank(-log10 mcap)` **to ten decimal places**. The quoted
cap range was wrong too - the live universe spans $6.8B to $5,419B.

The section now states what the log does and does not do, shows the evidence,
gives the MSCI comparison, names the tilt as equal-weight-style rather than
log-compressed, and labels the large-cap extrapolation as a known weakness.
Fixed in the generator (`run_screener.py`), not the generated file (rule 10).
`tests/test_size_tilt_is_documented_truthfully.py`, 11 tests - the 5
documentation assertions confirmed to fail against the pre-fix files, the 6
invariance assertions pin pipeline behaviour and hold either way.

**2. Merged `nightly/*` branches are now swept from `origin`, not just
locally.** The 09-02 evening session found 11 dead remote branches and wrote it
down as a sweep for "a future session" - i.e. manual work, which is what rule 11
exists to forbid. `nightly-screener.ps1` now sweeps them right after creating
the run's branch, so it self-heals instead of accumulating one ref per session
forever.

The delete list comes from `git branch -r --merged origin/main`, so every commit
on a swept branch is already reachable from `main` - deleting the ref discards
no history and is not a rewrite (rule 2). `$Branch` is excluded so a same-day
rerun cannot delete the branch it is about to push, and a failed delete is a
`WARN` retried next run rather than a lost session.

`tests/test_branch_sweep.py`, 10 tests. Because this is an unattended
`git push origin --delete` loop against the repo that serves the public site, the
`--merged` filter's semantics are asserted against a **real origin+clone git
sandbox** containing one merged and one unmerged branch, rather than assumed -
that filter is the only thing preventing it from destroying work that never
reached `main`. The 4 remote-sweep assertions fail against the pre-fix script.

### Evidence / research

- **MSCI Low Size methodology** (weights proportional to `1/ln(mcap)`),
  recomputed over this screener's actual 502-name universe: 798x cap spread ->
  1.295x weight spread. This is the measurement that converts Monday's
  qualitative "far more aggressive than the practitioner standard" into a
  decision, and it points the opposite way from what Monday's sketch assumed.
- **Asness, Frazzini, Israel, Moskowitz & Pedersen (2018)**, *JFE* 129(3) and
  **S&P 500 Equal Weight** (+63 bps/yr since 1990; -32% relative 2023-2025) -
  carried over from Monday's note, now quoted on the public methodology page as
  the size section's stated known weakness rather than left in `research/`.
- **No new literature was read today.** Today's work was measurement and
  disclosure against research already gathered this week, which is what a build
  day is for.
- **No backtest number and no figure from `live_ic_history.csv` was used**
  (rules 4 and 5).

### Methodology changed

**None - deliberately, and that is the finding.** No weight, threshold, metric
or formula moved. `METHODOLOGY_CHANGELOG.md` has no new entry because nothing
changed; the reasoning for *not* changing lives where a future reader will
actually hit it - the size section of `SCREENER_OVERVIEW.md` now explains the
tilt's real shape and links to the research note, whose "Disposition of
Candidate 1" section carries the full measurement.

All three of Monday's candidates are now resolved: 1 documented (today),
2 closed as two bets (09-02), 3 shipped (09-01).

### Tried and rejected

- **Applying the existing `percentile_transform` logistic to `size_log_mcap`** -
  Monday's own sketch. Rejected on its own pre-registered criterion once the
  criterion turned out to be parameter-dependent, and on the stronger ground
  that it is a weight change in disguise.
- **Special-casing size's transfer function at all.** Rank-scoring is the
  screener's universal mechanism across all 44 metrics and is what makes the
  eight categories commensurable. Making one 5% category different, to chase a
  shape indistinguishable from deleting it, is the "pile of good ideas" failure
  `CLAUDE.md` warns against.
- **Starting priority 4 (deterministic per-stock summaries) with the remaining
  session.** It is the top open product item and fully designed in
  `plan/dashboard-north-star.md`, but it is a new module plus generator wiring
  plus dashboard rendering plus tests. Beginning it here would have produced a
  half-built feature on a public site rather than a third finished thing.
  Recorded as next instead.

### Not done, and why - needs the next session, not the owner

**The 11 existing dead branches on `origin` are still there.** The automated
sweep ships and will remove them on the next scheduled run, but I could not do
the one-time cleanup by hand: this session's sandbox denied
`git push origin --delete`. The fix that matters - making it self-healing - is
committed and tested; only the manual catch-up is outstanding, and the machine
will do it at 06:00 tomorrow without anyone intervening.

**Next session: confirm it happened.** `git branch -r | grep nightly` should
show only recent branches, and the run log should carry
"Swept stale merged remote branch: origin/nightly/..." lines. If it does not,
the sweep block is wrong in a way static tests could not catch, and that is the
first thing to fix.

### Next

1. **Verify the remote branch sweep fired** (above). Cheap, and it is the only
   thing this session left unverified.
2. **Priority 4 - deterministic per-stock summaries.** Owner directive from
   2026-08-10, open 24 days, the top remaining north-star item, and fully
   specified in `plan/dashboard-north-star.md` down to the template and the
   scope (top ~25 plus holdings). The payload already carries `contrib`, `pct`,
   `peers`, price targets and `metric_count` - everything the template needs, so
   this is a build with no research dependency. It deserves a whole session.
3. **`growth ~ investment = -0.331`** - still the best research question on the
   board, carried from 09-02: the screener rewards fast growth with one hand and
   penalises how it is funded with the other, across 18% of composite weight,
   and nobody has written down whether that is intentional.

---

## 2026-09-04 - RETROSPECTIVE. Evaluate whether this routine is producing value, and change the process where it is not.

### Health numbers (rule 8)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-09-03_060001.log` - "Run complete: shipped to main" |
| Data loop published? | `logs/datarun-2026-09-04_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored, 0 fetch failures, price coverage 502/502 (100%) |
| Evidence base | **32 rows, newest 2026-08-28, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
| Priority 0 | DONE 2026-08-24, not reopened |
| Top open roadmap item | **Priority 4, deterministic per-stock summaries - owner directive 2026-08-10, open 25 days** (this row is new; see Process changes) |

**Tests:** before 938/938, after **965/965** (+27, no pre-existing failures)
**Data loop:** healthy
**Owner queue:** empty - nothing under **Open** in `OWNER_FOCUS.md`. Nothing deferred.
**Rotation:** ISO week 36, Friday, even week - retrospective.

Two carried-over checks first, both confirmed:

1. **The remote branch sweep fired.** `logs/nightly-2026-09-04_060001.log` shows
   12 "Swept stale merged remote branch" lines; `git branch -r` is down to
   `origin/main`, `origin/master` and `origin/HEAD`. The 09-03 session left this
   as the one thing it could not verify from its sandbox.
2. **The evidence base is moving.** 31 -> 32 rows, newest 08-27 -> 08-28. `1m`
   sits at 3 effective for the fifth session, which is structural rather than a
   defect: `1m` rows mature only as older run dates age past the horizon.

### Retrospective findings

- **Sessions reviewed: 9 scheduled** (2026-08-24 to 2026-09-03), plus 6
  owner-run evening/catch-up sessions.
- **Genuinely valuable: 9 | Churn: 0 | Failed gates: 1** (2026-09-02, gate 1).

**1. What fraction of sessions produced something genuinely valuable? All nine.**
The last retrospective measured 2 of 11 slots producing anything and 5 never
firing. This period: 9 of 9 fired and 9 of 9 shipped. Named - 08-24 the
evidence-base repair (5 defects, 3 IC rows -> 23), 08-25 `history.py` and the
dashboard's time dimension, 08-26 the split-scale price guard (MNST was live at
~110 ranks too high), 08-27 the GitHub Actions loop watchdog, 08-28 the
weight-transparency fix, 08-31 the size-factor research note, 09-01 the removal
of winsorization (six megacaps published at an identical false $2,802.0B), 09-02
the risk category shedding two momentum metrics, 09-03 the size tilt documented
truthfully plus the origin branch sweep. Not one is churn. Sessions run 13.6-22.4
minutes against a 4-hour limit; nothing came close.

**2. Which rotation day earns its place?** All five, on this sample, but the
justification differs. **Tuesday (product) is the highest-yield day** and both
its outings paid: 08-25 shipped the only new surface of the period, 09-01 found
the trillion-dollar display error by opening the dashboard as a user. **Monday
(research) works now that sessions start** - one complete note on 08-31 with real
citations, after six consecutive skips under the old regime. **Thursday (build)
is the weak one**, and specifically: when Monday's research concludes "no change
warranted" - a successful outcome `research/README.md` explicitly endorses, and
what 08-31 concluded - Thursday has nothing to build and improvises. 09-03
improvised well, but it improvised. Fixed below rather than removed.

**3. Is the evidence standard holding? Yes, and it has got stricter.** Every
changelog entry since 08-24 carries an explicit `Backtest observation: none -
benched until 2027-02-11 per rule 5`, and 09-03 states in terms that no figure
from `live_ic_history.csv` was used. The research notes are real: the 08-31 size
note carries Banz (1981), Asness/Frazzini/Israel/Moskowitz/Pedersen (2018) with
SMB alphas and t-stats by specification (23bps/2.27 raw, 49bps/4.89 after QMJ),
Alquist/Israel/Moskowitz (2018), and an MSCI Low Size reconstruction on this
screener's own 502 names. Best sign of health: **09-01 and 09-03 both corrected
published claims of this project's own that turned out to be false**, rather than
softening them.

**4. What keeps going wrong?** One thing, and it is structural rather than a
recurring bug: **the ship gates were an audit, not a precondition.** See below.
Otherwise the recurring pattern is that documentation goes stale faster than it
is corrected - three of this period's sessions spent effort fixing claims in
`CLAUDE.md`, `SCREENER_OVERVIEW.md` and `research/README.md` that were untrue.

**5. Is the tool closer to the place you would look before buying or selling?
On correctness, substantially. On surface, not at all.** A user today sees true
market caps instead of six identical fakes, momentum that is not corrupted by
split-scale price series, and a risk score that is not 30% momentum. Those are
real. But **no north-star surface has shipped since 2026-08-25**: questions 3
(should I sell what I hold) and 4 (how much) remain exactly as unanswerable as
they were. The honest blocker is not difficulty - priority 4 is fully specified
in `plan/dashboard-north-star.md` and needs no research - it is that **defect
discovery outruns feature work.** Nine sessions each found something smaller,
real, and more urgent, and each deferral was correct on the day.

**6. What is the routine systematically blind to? The runner scripts.** No day in
the rotation points at `scripts/*.ps1`, and every improvement to them in this
period was reactive, made only after a run had already been damaged. Looking
there today found the defect below on the first read.

### Did - the ship gates now actually gate

**The defect: the morning brief could publish the work the gates had just
refused.** Both runners publish the brief from a `finally` block, because it is
the watchdog heartbeat and must land whether the run succeeded or failed. They
did it with:

```
git push origin HEAD:main
```

On the happy path HEAD is `main`, so that is correct - and it is what ran on
every successful day, which is why it went unnoticed. On a **ship-gate failure**
HEAD is the nightly branch carrying the work the gates refused, and that command
fast-forwards `origin/main` onto every commit on it. `prompts/nightly.md` told a
session that fails its own gates to leave the work on the branch, so that is not
a hypothetical shape - it is the documented one.

Reproduced in a real origin+clone sandbox before writing anything: with HEAD on
`nightly/2026-09-04` carrying a file marked "BROKEN - gates refused this",
`git push origin HEAD:main` puts that file on `origin/main`. That reproduction is
now `tests/test_brief_publish_safety.py::test_pushing_head_to_main_publishes_the_whole_branch`.

It had not fired only by luck of ordering. On 2026-09-02, the one gate failure in
this project's history, the session had already merged onto local `main`, so the
recovery path reset local `main` first and the subsequent push was rejected as a
non-fast-forward - the log's "Brief committed locally; push failed." is that near
miss, recorded as a routine warning.

**Fixed: `scripts/publish-brief.ps1`.** Builds a single-file commit on top of
`origin/main` with plumbing (`hash-object` -> throwaway index -> `write-tree` ->
`commit-tree`) and pushes the **commit object**, not a ref:
`git push origin <sha>:refs/heads/main`. It never checks out, never merges local
work, and never names a local ref, so `MORNING_BRIEF.md` is the only path it can
change - by construction rather than by convention. It retries on a moved
`origin/main` (rebuilding on the new tip, so a concurrent data-run commit is not
clobbered), leaves the working tree clean so the next run's `git pull` is not
blocked, and does not disturb the repository index. Both runners dot-source the
one function, on the 2026-08-29 precedent that two copies of an unattended
primitive drift.

**14 tests** in `tests/test_brief_publish_safety.py`, driving real git and real
PowerShell. Four static assertions confirmed to fail against the pre-fix scripts
pulled from `git show HEAD:`.

**The deeper problem, and the actual process change: the session was publishing
`main` before the gates ran.** `prompts/nightly.md` had the session merge and
push `main` as its own final step; `nightly-screener.ps1` then re-verified all
four gates *afterwards*. That is what 09-02 was: pushed at 06:24:22, gate 1
failed at 06:26:53, and `scripts/revert-bad-merge.ps1` had to be written to undo
a commit already live on the public site. `CLAUDE.md` rule 1 says "You may only
push to `main` when all of them pass"; the routine did not implement that.

**The session now pushes its branch and stops. The runner merges, after its own
gate run.** This is strictly stricter, which is what section 4 of the
retrospective prompt permits.

`nightly-screener.ps1` already contains that merge path - but **it has never run
in production.** Every successful log line reads "HEAD is on 'main'", because the
session always merged first. So its git sequence is now exercised against an
origin+clone sandbox: the merge lands the branch on `origin/main` and leaves the
tree clean, and a conflicting merge aborts with `main` at exactly its previous
sha and the work preserved on origin. `tests/test_gate_ordering.py`, 8 tests,
including source-order assertions that the push to `main` sits downstream of the
gate decision and the `good/` rollback tag downstream of a successful push.

**This session is the first to follow the new rule**, deliberately. A
retrospective that exempted itself from the rule it had just written would be the
first step in the drift section 4 exists to prevent - so this work stops at
`nightly/2026-09-04` and the runner merges it. If that path is wrong, the failure
is fail-closed: the branch is pushed to origin and Monday's session inherits it.
`prompts/retrospective.md` now says so too.

### Process changes made

1. **The gates are a precondition, not an audit.** `prompts/nightly.md` section 5
   and `prompts/retrospective.md` section 5: the session pushes its branch and
   does not merge or push `main`. `CLAUDE.md`'s Ship gates section says the same
   and says why. `revert-bad-merge.ps1` stays as the second line of defence and
   should now never fire.
2. **The brief cannot carry anything but the brief** - `scripts/publish-brief.ps1`
   above, recorded in `CLAUDE.md` priority -1 as a do-not-undo.
3. **A fifth health number in rule 8: the top open roadmap item and its age in
   days.** Same mechanism that fixed the evidence base, applied to the finding in
   question 5. Nine sessions produced real work and no north-star item, and
   nobody had written down that priority 4 had been open for 25 days. Writing the
   age down makes the trade visible at the moment it is made; it does not force
   the choice.
4. **Thursday has a defined fallback.** If the week's research concluded "no
   change warranted", Thursday takes the top open item in Current priorities
   instead of inventing a methodology change to have something to build. The log
   says which of the two it was.
5. **The retrospective now reads the runner scripts** - added to section 1's
   evidence list, in place of the `git log --stat` bullet, which the last two
   retrospectives have not found informative next to the log entries themselves.

**Deleted, per "prefer deleting to adding":** `CLAUDE.md` priority -1 loses ~30
lines of narrative about outages that are fixed and recorded in this log;
`prompts/nightly.md` loses a duplicated paragraph on IC evidence (it repeated
section 3 and `CLAUDE.md` rules 4-5) and a stale sanity-check that still spoke of
"once priority 0 lands". Both prompts are net shorter than they were.

### Evidence / research

A demonstrated failure plus a reproduction, per the mandate's fourth category.
No citation, no backtest, no IC number - rules 4 and 5 do not bite, because
nothing here was justified by a return.

Session-rate measurement: 9 of 9 scheduled slots from 08-24 to 09-03 fired and
shipped, from `logs/nightly-*.json` (`num_turns` 82-154, `is_error` false,
13.6-22.4 min) cross-checked against `logs/nightly-*.log` and `git log`. Against
the last retrospective's 2 of 11.

### Methodology changed

**None.** No weight, threshold, metric or formula moved. Runner and process
infrastructure, which by precedent lives in this log rather than in
`METHODOLOGY_CHANGELOG.md`.

### Tried and rejected

- **Leaving the session's self-merge in place and relying on
  `revert-bad-merge.ps1`.** It works - it was built and proven against a sandbox
  on 09-02 - but it is a mechanism for undoing a public push that should never
  have happened. Ordering removes the class; recovery handles instances.
- **Fixing `Publish-Brief` by skipping publication when the gates fail.** The
  smallest change, and it breaks the watchdog: the heartbeat exists precisely to
  answer "did the task fire" on days the run failed. Suppressing it on failure
  would make a failed run look like a machine that never woke up.
- **Committing the brief locally and pushing the branch's tip.** A smaller edit
  than the plumbing route, but it still puts a local ref on the left of the
  refspec, which is the property that failed. The safety had to be structural.
- **Reserving a whole rotation day for north-star product work.** Tempting given
  question 5, but the rotation already reserves Tuesday for product and the
  problem is not the day - it is that a real defect found on that day rightly
  wins. A sixth day would be consumed the same way. The health-number line is the
  cheaper intervention and does not override a session's judgement.
- **Making the retrospective monthly instead of fortnightly.** Considered because
  the routine is now healthy. Rejected: this fortnight's finding was a live hole
  in the publish path, found only because a retrospective looks at the runner at
  all. Nothing else in the rotation does.

### Not done, and why

- **Priority 4 (deterministic per-stock summaries) is still not started**, now 25
  days old. A retrospective is the wrong session to start it in - it would leave
  a half-built feature on a public site - but it is the top open item and Monday
  or Thursday should take it. This is exactly the deferral the new health-number
  line is meant to make visible; it is recorded rather than hidden.
- **Gate 3 still only regex-matches `dashboard_data.js`'s first line**, not a
  real parse. Unchanged from the 08-21 retrospective's note. PowerShell *was*
  executable this session, so the objection recorded then no longer holds and a
  node parse is now buildable - a legitimate way to make a gate stricter. Left
  for a session that can test it end to end rather than bolted on at the close of
  this one.

### Flagged for the owner

- **Nothing needs your decision.** The queue in `OWNER_FOCUS.md` is empty and has
  been for the period; the routine has been running unattended and correctly. If
  you want the dashboard to move faster than the defect fixes allow, the single
  most useful thing you can do is put one line under **Open** naming the surface
  you want - an owner item outranks the rotation, which is the one lever that
  reliably beats firefighting.

### Next

1. **Confirm the runner merged this session** - `git log origin/main` should show
   `nightly 2026-09-04 - RETROSPECTIVE`, and the log should read "All gates
   passed. Merging to main." with "HEAD is on 'nightly/2026-09-04'" above it.
   That line has never appeared on a successful run; it is the first production
   exercise of the merge path. If it did not merge, the work is on
   `origin/nightly/2026-09-04` and recovering it is the first job.
2. **Priority 4 - deterministic per-stock summaries.** Fully specified, no
   research dependency, 25 days open.
3. `growth ~ investment = -0.331`, carried from 09-02 and still the best research
   question on the board.

---

## 2026-09-07 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.

### Health numbers (rule 8, all five)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-09-04_060001.log` - "Run complete: shipped to main" (09-05/09-06 were the weekend) |
| Data loop published? | `logs/datarun-2026-09-07_020001.log` - "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **33 rows, newest 2026-08-31, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
| Priority 0 | DONE 2026-08-24, not reopened |
| Top open roadmap item | **Priority 4, deterministic per-stock summaries - owner directive 2026-08-10, open 28 days** |

**Tests:** before 965/965, after 965/965 (no pre-existing failures; no tests added - research day)
**Data loop:** healthy. Evidence base moved 32 -> 33 rows, newest 08-28 -> 08-31.
**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty. Nothing deferred.
ISO week 37, Monday - research day, taken as the focus.

### Did

**Researched the Revisions category and found it contains no revisions.**
`research/2026-09-07-revisions-category-has-no-revisions.md`.

The category carries **10% of the composite**. All five scored metrics are past
earnings surprises, a price-target *level*, or short interest. The three
surprise metrics come from the same four rows of `Ticker.earnings_history`
(`factor_engine.py:1089-1130`) and are **78% of the category = 7.8% of the
composite**.

Three findings, in order of how much they should change what we do:

**1. The category's public rationale rests on an effect documented as absent in
this universe.** `SCREENER_OVERVIEW.md:149` justifies it with "When a company
consistently beats earnings estimates, the stock price usually follows - but
with a lag, which creates an opportunity." That is post-earnings announcement
drift. Martineau (2022, *Critical Finance Review* 11(3-4)) finds PEAD
**non-existent for all-but-microcap stocks since 2006**, with the 2016-2019
60-day coefficient significantly *negative*; the return moved to the
announcement date (large-stock BHAR[0,1] ~20bps in 1984-1990 -> ~120bps in
2016-2019). His surprise measure is the analyst-estimate kind this screener
computes, and every S&P 500 name is "all-but-microcap".

**2. The claim that a revisions metric is impossible is false, and it was on the
public methodology page.** `config.yaml` and `SCREENER_OVERVIEW.md` (twice) said
forward-EPS revisions "would require a paid data source like FactSet or
Refinitiv I/B/E/S". Measured today against the installed yfinance 0.2.66:
`Ticker.eps_trend` returns consensus EPS now vs **7/30/60/90 days ago** for
FY1/FY2, and `Ticker.eps_revisions` returns up/down analyst counts. Coverage on
a deterministic 84-name sample (`sp500_tickers.json[::6]`): **82/84 = 97.6%**,
95% Wilson CI [91.7%, 99.3%] - **identical to `analyst_surprise`**, the metric
already carrying 38% of the category. Measured cost, by instrumenting every
`YfData` network method: **+1 HTTP request per ticker**, after which
`eps_revisions` and `earnings_estimate` are free from the same cached response.

**3. The surprise signal is structurally stale and the metric does not know it.**
`analyst_surprise` is the median of the last *four* reported quarters, with no
time-since-announcement term anywhere. On the 84-name sample, days since last
report: median **69**, and **0% of names within 30 days**, 85.5% in the 60-90
day band. Reporting is clustered so this swings with the calendar - 09-07 sits
mid-quarter, the stale end - and the note says so explicitly rather than
claiming the number is representative. The point stands regardless: the metric
weights a 2-day-old beat and an 89-day-old beat identically.

**What is working and should not be shrunk.** The revisions category is the
screener's **most independent** - max |Spearman| against the other seven is
**0.192** (momentum), and `revisions ~ composite` is only +0.237. Against the
2026-08-26 finding that momentum and risk were 23% of composite weight off one
`Ticker.history()` call, this is the counter-example. So the recommendation is
to fix what is *in* the 10% slot, not to cut the slot.

**Also shipped: corrected the two false claims on the public methodology page**
(`SCREENER_OVERVIEW.md` lines 151 and 476) and the matching `config.yaml`
comment. Facts only - **no weight, threshold, metric or code path changed**, so
there is no `METHODOLOGY_CHANGELOG.md` entry. Precedent: 09-01 and 09-03 both
corrected published claims of this project's own that turned out to be false,
and CLAUDE.md requires public docs be kept truthful. Leaving a demonstrably
false "requires FactSet or Refinitiv" on a page an investment club reads, for
four more days, was the worse option.

### Evidence / research

- **Chan, Jegadeesh & Lakonishok (1996)**, *JF* 51(5) 1681-1713. IBES 1977-1993.
  REV6 = 6-month MA of (consensus revision / price). Top-vs-bottom **decile:
  +7.7% over 6 months**, +8.7% at 12. SUE over 1973-1993: **+7.5%**. Rank
  correlation SUE~revisions **0.440** - "do not reflect the same information".
  Via Jegadeesh (2001) *Momentum* survey section 6, which also carries **Stickel
  (1991)** (Zacks, 1981-1984, top/bottom 5%: **+7.07%** consensus revisions,
  +6.36% individual) and the judgement that the revision strategy is
  "remarkably robust... not sensitive to the specific definition... nor to the
  source of analyst forecasts".
- **Novy-Marx (2015)**, NBER WP 20984, US 1975-2012. Monthly excess returns by
  size quintile, **largest quintile (72% of market cap)**: price momentum
  **0.35 [t=1.48]**, CAR3 **0.20 [2.12]** with alpha **0.15 [1.66]** - both
  insignificant. **SUE survives at 0.26 [2.46], alpha 0.29 [2.83].** Reported in
  the note as cutting both ways: a properly standardised surprise did earn a
  large-cap alpha, on the thinnest margin in the paper.
- **Martineau (2022)**, *CFR* 11(3-4) 613-646. I/B/E/S 1984-2019, 312,462
  announcements. Surprise = (actual - median forecast)/price. See finding 1.
- **Bartov, Givoly & Hayn (2002)**, *JAE* 33(2) 173-204. Meet-or-beat premium
  ~3%. Read from the paper rather than the abstract: it is a **return over the
  quarter in which the MBE occurs**, not a forward return, and "leading
  indicator of future performance" means future *fundamentals* (their Table 9,
  "both of the years following the MBE year"). So it supports
  `consecutive_beat_streak` as a persistence signal, not as a return predictor.
- **Practice - MSCI Barra USFAST datasheet (March 2015)**, 24 style factors.
  `Sentiment` = "return differences between stocks based on sell-side analyst
  revisions and news sentiment", descriptors **Revision ratio / Change in
  analyst-predicted earnings-to-price / Change in analyst-predicted earnings per
  share** + news sentiment. Searched the full datasheet: the word **"surprise"
  appears zero times** across all 24 factors and their descriptors.
- **Practice - Zacks Rank**, four components: **Agreement** (revision breadth),
  **Magnitude** (change in consensus for current/next fiscal year), **Upside**
  (proprietary), **Surprise** ("a company's last few quarters' EPS surprises").
  This screener implements **only Surprise** - one of four, and the only one
  that is not a revision - in a category named for revisions.
- **Where they disagree:** academia says analyst-surprise drift is dead outside
  microcaps; Zacks still ships a surprise leg. The note takes the academic side
  on *surprise alone in large caps* (Martineau's test matches our universe,
  surprise definition and period) and the practitioner side on category shape
  (surprise stays as one leg of several, not 78%).
- **Measured on the live payload and a fresh sample**, not from the backtest or
  the IC history: `analyst_surprise ~ consecutive_beat_streak` **+0.497** (58
  points of category weight, largely one signal); `earnings_acceleration`
  independent at **-0.070**; 90-day revision vs `analyst_surprise` **+0.346**,
  *below* CJL's 0.440, so the duplication objection fails.

### Methodology changed

- **None.** No weight, threshold, metric or code path was touched. Doc
  truthfulness corrections only (section 7 of the note). The proposed metric and
  reweighting are argued in the note and left for Wednesday's synthesis and
  Thursday's build.

### Tried and rejected

- **Adding a recency / time-since-announcement weighting to the surprise
  metrics.** This was the obvious fix suggested by the 69-day staleness finding
  and I worked it through before rejecting it. Ruled out by Martineau (2022):
  conditioning on recency would sharpen a signal that does not exist in this
  universe - a better estimate of zero, bought with an extra term and a
  paragraph of explanation. It also makes every stock's score depend on its
  reporting calendar, so the movers panel would show names shifting for a
  reason a student cannot see.
- **Using the up/down diffusion index `(up-down)/(up+down)` as the primary
  revision metric.** Rejected on measurement, not preference: **62% of the
  sample is tied**, with many names pinned at exactly +1.0. In a rank-scored
  system that is a large tie block. `eps_trend` magnitude has **0% ties** and is
  the better primary; diffusion is at best secondary.
- **Scaling the revision by the estimate rather than by price.** The sampled
  ratio has mean **+19.9%** against median **+1.7%** and sd 1.55 - a fat right
  tail from small denominators, the same pathology `analyst_surprise` already
  guards against with its `max(|e|, 0.10)` floor. CJL scale by *price*; so
  should we.
- **Cutting the revisions category's 10% weight.** Tempting given finding 1, and
  rejected: at max |Spearman| 0.192 it is the most orthogonal category in the
  screener. The slot is worth keeping; the contents are the problem.

### Next

**Wednesday's synthesis, and it has one measurement that must come first:**
`fy1_revision_3m ~ forward_eps_growth`. `forward_eps_growth` is **45% of the
growth category** and is built on the *same* FY1 consensus number. A level and a
change in that level are different objects, but if they come back highly
correlated, adding the revision metric would spend a growth slot and a revisions
slot on one input - the 2026-08-26 momentum/risk failure in a new place. Measure
that, and `revisions ~ momentum` (currently +0.192; Novy-Marx's thesis is that
price momentum *is* earnings momentum), before writing any changelog entry.

Standing, unchanged: **Priority 4, per-stock summaries, 28 days open** - and the
note argues it gets easier, since "analysts raised their estimate 4% in three
months" is a sentence a student understands and "the median of its last four
quarterly EPS surprises is in the 71st percentile" is not.

---

## 2026-09-08 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.

### Health numbers (rule 8, all five)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-09-07_060001.log` - "Run complete: shipped to main" |
| Data loop published? | `logs/datarun-2026-09-08_020002.log` - "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **34 rows, newest 2026-09-01, 3 effective observations at `1m`** (8 raw) against a gate of 8 |
| Priority 0 | DONE 2026-08-24, not reopened |
| Top open roadmap item | **Priority 4, per-stock summaries - owner directive 2026-08-10, open 29 days. Taken and shipped today.** Next up is priority 5, sell-side workflow (north-star gap 2, 2026-08-05, **34 days**) |

**Tests:** before 965/965, after **1117/1117** (no pre-existing failures; +152 tests)
**Data loop:** healthy. Evidence base moved 33 -> 34 rows, newest 08-31 -> 09-01.
**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty. Nothing deferred.
ISO week 37, Tuesday - product day, taken as the focus. The top open roadmap
item happens to *be* a product item, so for once the rotation and the queue
pointed at the same work.

### Did

**Shipped priority 4: the "Screener AI" chat is gone and every stock's
drilldown now opens with a deterministic "Why it ranks here" block.** Owner
directive 2026-08-10, open 29 days. `METHODOLOGY_CHANGELOG.md` 2026-09-08.

This is the first north-star item to ship since 2026-08-25. The 2026-09-04
retrospective added the roadmap-age line to rule 8 precisely because nine
consecutive sessions had produced real work and no north-star item; writing the
age down is what made "29 days" visible at the moment the day's focus was being
chosen.

**Removed** (891 lines: 80 HTML, 583 JS, 228 CSS; `generate_dashboard.py` is
921 lines shorter): the chat FAB and panel, the Chat Settings dialog with its
API-key field and model picker, 27 JS functions, three keyframe blocks and the
`AI CHAT PANEL` stylesheet. The `config_traps` payload key went with it - it
carried the four trap thresholds solely so the chat could put them in its system
prompt, nothing rendered them, and the Methodology section already publishes
them from `config.yaml`. Same reasoning that retired `spx_weights` on
2026-08-26.

**Why it had to go** - four consequences, all readable off the shipped code
rather than argued. It required each visitor to paste an Anthropic API key into
`localStorage` and called `api.anthropic.com` from the browser, so: every
student in a college investment club needed a paid API account most do not have;
a public page carried a password field labelled "Anthropic API Key", which is
the shape of a phishing form; each question cost the reader money; and two
students asking the same question got different answers, recorded nowhere. The
last one decides it. This tool refuses to publish a run whose price coverage is
below 90% and was shipping an explanation layer with no provenance at all.

**Added:** `stock_summary.py` builds an ordered list of factual sentences per
stock from fields the payload already carried - `contrib`, `cat_scores`, `pct`,
`raw`, `peers`, `flags`, the analyst targets, `metric_count`/`metric_total` and
the `history` spine - and `renderSummary()` puts it at the top of the drilldown.
HST on this run:

> Ranks 1st of 502. Its composite of 74.7 is a percentile: it scores above 75%
> of the universe. Most of that composite comes from Valuation (category score
> 96, 21.1 points) and Quality (category score 83, 18.4 points) - 39.4 of its
> 74.7 points. Its weakest scored category is Risk at 38 out of 100,
> contributing 3.8 points. A category score near 50 is the sector median. [...]
> Since the run of 2026-08-10 (29 days ago) it has held its rank, with the
> composite down 3.3. [...] The score rests on 18 of 18 metrics.

**Built at run time and baked into the payload**, not computed in the browser -
that is what makes it diffable and identical for every reader, which is the
entire reason it replaced the chat. Three constraints are enforced by tests
rather than by care:

- **It explains; it never advises.** `BANNED_TERMS` / `advice_terms_in()` is the
  machine-checkable form of the north-star line. All 502 live summaries contain
  zero matches, and the detector is itself tested against the plan's own bad
  examples ("attractive entry point", "undervalued", "a strong buy") so a clean
  sweep means something.
- **Percentiles are labelled sector-relative**, because they are.
- **A fact that cannot be stated exactly is omitted, never approximated.**

**A side effect worth naming:** this closes most of north-star gap 5, per-stock
confidence made legible. FDXF now reads *"The score rests on 12 of 18 metrics.
Momentum, Risk and Investment could not be scored for this stock, so the
remaining categories were reweighted to fill the gap. Its filings are flagged
stale (282 days old)."* That is the 2026-08-26 coherence finding - momentum and
risk are 23% of composite weight off one `Ticker.history()` call - surfacing to
a reader in prose, for the first time, without anyone building a feature for it.

**Tests: 965 -> 1117.** `tests/test_stock_summary.py` (77) and
`tests/test_ai_chat_removed.py` (75). **58 of those 75 fail against
`HEAD:generate_dashboard.py`**, verified by swapping the file in, re-running,
restoring, and confirming the working copy byte-identical by SHA-256. The
removal module asserts **both halves** - no chat symbol, element id, model id,
`localStorage` key or provider URL survives, *and* `renderSummary` exists, is
wired into `openStockDetail`, and escapes its text. A partial swap is the
dangerous state: a dangling identifier blanks the page with all four gates
green. The emitted script is additionally parsed with `node --check`.

**Payload cost, measured before deciding scope:** the summaries add 665 KB raw
but compress only 6.6x, so **+101 KB gzipped**; deleting the chat gives back
11 KB of page (`index.html` 281,678 -> 237,952 chars, 66 -> 55 KB gzipped). Net
**+90 KB on the wire, +8%**. The plan said "top ~25 first"; that was written
before anyone measured, and on the measurement all 502 is the right call - the
drilldown is the surface that answers *should I buy this one*, and a summary
that only appears for names a reader already knows is missing exactly where it
helps most. The number is now in `plan/dashboard-north-star.md` and
`plan/dashboard-inventory.md` so the next session inherits it rather than the
guess, along with the cheap lever if payload weight ever binds (drop the `peers`
and `flags` sentences, which duplicate panels a few hundred pixels below).

**Docs kept true (rule 9):** `plan/dashboard-inventory.md` refreshed in the same
session - new section on the summary, the chat section rewritten as a removal
record, payload table and key list corrected, gap 5 updated.
`plan/dashboard-north-star.md` marks the directive shipped and records both
departures from what it specified. `plan/refresh-button-and-chatbot-websearch.md`
and `plan/dashboard-frontend-fixes.md` both described chat code that no longer
exists and now say so at the top. `CLAUDE.md` priority 4 rewritten as DONE with
the three things not to undo.

**`SCREENER_OVERVIEW.md` deliberately untouched.** It is generated from
`run_screener.generate_screener_overview()` and documents *scoring*; nothing in
it became false today. Adding the summary to its "Defensibility & Transparency
Features" table would be a fair improvement but it is a separate change to a
generator, not a truthfulness fix, and this session had one job.

### Evidence / research

- **A documented user-facing failure, which is the acceptable evidence class for
  a product change.** The chat's four defects above are all readable off the
  deleted code: `localStorage.getItem('screener_anthropic_api_key')`, a `fetch`
  to `api.anthropic.com` with `anthropic-dangerous-direct-browser-access`, and a
  model picker. No measurement was needed to establish that a browser-side LLM
  call is not reproducible.
- **The project's own standard, cited rather than invented.**
  `plan/dashboard-north-star.md`: *"decision support, not a recommendation
  engine... show why, with sources and uncertainty visible... never emit a bare
  'buy'."* The `BANNED_TERMS` list is that sentence turned into a test.
- **Measured, not assumed:** gzip cost of the summaries (665 KB -> 101 KB, 6.6x,
  against the ~11x the inventory had recorded for earlier prose); page size
  before and after; 58/75 tests failing against the pre-change generator;
  0 advice-term matches across 502 live summaries.
- **No backtest number and no figure from `live_ic_history.csv`** appears
  anywhere in today's work (rules 4 and 5), and neither would be relevant -
  nothing here touches scoring. Composites and ranks are byte-identical.

### Methodology changed

- **None in substance.** No weight, threshold, metric definition, trap rule or
  scoring formula moved; composites and ranks are byte-identical before and
  after. A `METHODOLOGY_CHANGELOG.md` entry was written anyway, for the same
  reason the 2026-08-26 (evening) model-portfolio entry exists: a published
  surface was removed and another added, and a future session has to be able to
  find out why.

### Tried and rejected

- **A parser-free brace-balance check on the emitted JS.** Written as a backstop
  for machines without `node`, and removed the same session because JavaScript
  regex literals make it unsound: `escapeHtml` contains
  `.replace(/'/g, '&#39;')`, and a scanner without regex-literal support reads
  that apostrophe as a string delimiter and desynchronises everything after it.
  It reported the page unbalanced while `node --check` passed. A check that
  fires on healthy code is the exact failure shape the 2026-09-01 bank-metrics
  fix was about - it trains a reader to ignore it, which is when the real defect
  gets through. The `node` test now skips visibly where node is absent rather
  than pretending to cover.
- **Scoping summaries to the top 25, as the plan specified.** Rejected on the
  measurement, not on preference - see the cost paragraph above. Recorded as a
  deliberate departure in the plan file rather than silently done.
- **Building the optional LLM gloss layer** the north star allows as a second
  step. The deterministic block already reads as plain English, so a generated
  gloss would restate it while giving back the reproducibility that justified
  removing the chat in the first place.
- **Rebuilding a run-level overview.** The inventory check did its job: the
  "What Changed" movers panel (2026-08-25) already covers most of what the
  directive's last paragraph asks for. What remains is genuinely narrow and is
  written down as such rather than being built twice.
- **Adding a recency weighting to trap severity, and three other drilldown
  ideas** that surfaced while reading the payload. Left alone: the session had
  one job and a weekly usage ceiling exists.

### Next

1. **Priority 5, the sell-side workflow** - client-side watchlist/holdings,
   deterioration flags, a review queue. Now the top open north-star item at
   **34 days** (gap 2, dated 2026-08-05), and question 3 of four is still
   completely unanswerable. The summary shipped today is a good foundation for
   it: `_sentence_change` already states what moved and `_sentence_confidence`
   already states what broke, so a deterioration flag is largely a matter of
   deciding the threshold and where to surface it.
2. **The run-level overview sentence**, the narrow remainder of priority 4.
   Cheap now that `history.movers` and the summary template both exist.
3. Carried from 09-07, still the best research question on the board:
   `fy1_revision_3m ~ forward_eps_growth`, before any revisions-category change
   is written up - a level and a change in the same FY1 consensus number could
   spend a growth slot and a revisions slot on one input.

---

## 2026-09-09 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.

### Health numbers (rule 8, all five)

| Check | Reading |
|---|---|
| Last code session ran? | `logs/nightly-2026-09-08_060001.log` - "Run complete: shipped to main" |
| Data loop published? | `logs/datarun-2026-09-09_020001.log` - "Data loop complete", **HEALTH: PASS**, 502 scored |
| Evidence base | **36 rows, newest 2026-09-02, 3 effective observations at `1m`** (9 raw) against a gate of 8 |
| Priority 0 | DONE 2026-08-24, not reopened |
| Top open roadmap item | **Priority 5, sell-side workflow** - north-star gap 2, dated 2026-08-05, **35 days open**. Not taken today; see "Owner queue / rotation" |

**Tests:** before 1117/1117, after **1117/1117** (no pre-existing failures; no
code changed today, so no test changed)
**Data loop:** healthy. Evidence base moved 34 -> 36 rows, newest 09-01 -> 09-02.
**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty. Nothing deferred
for a stalled loop or a failing gate. ISO week 37, Wednesday - synthesis day,
taken as the focus, with Monday's note
(`research/2026-09-07-revisions-category-has-no-revisions.md`) as its subject,
exactly as that note's §6 instructed. Priority 5 was **not** taken: Monday
deferred a specific measurement to Wednesday and a synthesis day that skips it
leaves Thursday building on an unverified design. Priority 5 remains the top
open north-star item and its age is written above so the trade stays visible.

### Did

**Answered the coherence question Monday deferred, on the full 502-name
universe, and settled the design for Thursday's build.** No code, weight,
threshold or scoring path changed today. Output is §8 of the research note
(~330 lines) plus a measured confirmation against the 2026-09-02 changelog
entry.

**First, the counterfactual was made trustworthy.** Before computing anything
hypothetical, the published composite was reproduced from its own inputs -
the eight category scores through `compute_composite`'s per-row
renormalisation, the revisions category rebuilt from its five published metric
percentiles, and the coverage discount reconstructed from per-stock metric
presence split by `_BANK_ONLY_METRICS`/`_NONBANK_ONLY_METRICS`. **Max absolute
error 0.000000000000 on all 502 names**, including the three where the
discount actually bites (FDXF at 62.5% coverage, FISV, L). A counterfactual is
only worth reading if the factual reproduces first.

*Trap for the next session, now written down:* the published `Composite` is
stored rounded to 2dp while category scores are stored at full precision, so a
naive comparison shows a spurious ~0.005 residual on **every** name and looks
like a real discrepancy. It is not. Round to 2dp before comparing.

**Monday's two named risks came out backwards.**

- **Risk 1, growth overlap - cleared.** Monday flagged this "measure this
  first": `forward_eps_growth` is 45% of the growth category and is built on
  the same FY1 consensus line as the proposed `fy1_revision_3m`. Measured:
  **+0.152** (n=397). A level and a change in that level really are different
  objects. `revisions ~ growth` moves only +0.039 -> +0.089. The 2026-08-26
  momentum/risk failure does not repeat here.
- **Risk 2, momentum overlap - real, larger than expected, and *economic
  rather than mechanical*.** `fy1_revision_3m ~ momentum_score = +0.417`;
  `~ return_12_1 = +0.416`. The candidate is **more correlated with momentum
  than with the category it would join** (+0.387).

**That second number got the hard look it deserved, and the answer is the most
useful thing today produced.** The obvious suspect was Monday's own scaling
choice: `Δ EPS / price` puts price in the denominator, and a stock that has
fallen has both a small denominator and probably a negative revision. If that
were the mechanism the whole +0.417 would be an artifact. It is not. Every
reconstruction carries it - `/ |estimate|` with no price term at all **+0.429**,
**sign-only (-1/0/+1) with no denominator whatsoever +0.324**, numerator alone
+0.408 - while `1/price ~ momentum_score` is only **-0.128, the wrong sign for
the mechanism**. So it is a fact about the market, not about the formula:
**Novy-Marx (2015) measured directly in this screener's own universe.** Any
future session that thinks it has found a cleverer denominator should re-run
that table before believing it, and it is written into §8.7 as one of three
things not to undo.

**The category-level cost stays under the bar Monday set before seeing the
number.** `revisions ~ momentum` +0.171 -> **+0.317**, against Monday's
pre-registered "materially above ~0.35 is a coherence cost". It would become
the 4th largest of the 28 category pairs. Spanning regression: the new metric
is **71.6% unspanned** by all eight existing categories (R2 0.284), and the
revisions category falls from 91.1% unspanned to **85.2%** - from roughly
tied-first to fourth, still above risk, valuation, growth and size. That is
independence being **spent deliberately to buy construct validity**, and §8.4
names it as such rather than letting it pass unremarked.

**Monday's own materiality test fired, and I let it.** Monday wrote: *"If the
change moves fewer names than deleting the category outright... it is a
presentational change and should be argued on explainability alone,
honestly."* Measured: deleting the revisions category moves **7 of the top
50** (median |Δrank| 17); the proposed reweight moves **3 of 50** (median 9,
218 names moving >10 ranks, max 109). **3 < 7.** So the changelog Thursday
writes may **not** claim the ranking improves. The bar is asymmetric - it
compares a 3.3%-of-composite within-category reweight against deleting a 10%
slot - and §8.5 says so, then accepts the conclusion anyway. A threshold set in
advance that gets explained away the moment it fires is not a threshold.

**Corrected one of Monday's arguments.** Monday ruled out duplication with
"90-day revision vs `analyst_surprise` = +0.346, lower than CJL's 0.440". On
the full universe it is **+0.401** (n=498), not +0.346 - the 84-name sample
understated it by 0.055. The conclusion survives (still under the 0.440 CJL
themselves called distinct) but the margin goes from 0.09 to 0.04. Both §4.4
and §6 of Monday's note now carry an inline correction so nobody quotes the old
figure. It also makes Monday's §5(c) reweighting *more* necessary: shipping the
new metric at 35 without cutting the surprise family from 78 to 45 would leave
the category's two heaviest metrics correlating at +0.401. **The two halves are
one change, not two.**

**Two of Monday's open construction questions are now closed on better data.**
Coverage on all 502 is **500/502 = 99.6%**, *better* than the `analyst_surprise`
it takes weight from (99.2%). And the estimate-scaled denominator is not merely
fat-tailed as Monday found on 84 names - on the full universe it has a **zero
denominator**, so its mean is literally `+inf` and its sd undefined. Price
scaling is not a refinement, it is the difference between a computable metric
and one needing a special case. Diffusion (`eps_revisions`) is dead: **75.8%
ties, 37.3% of names pinned at exactly +1.0** (Monday measured 62% on the
sample).

**Confirmed the 2026-09-02 risk-category change against its own prediction.**
`CLAUDE.md` asks that changes be re-checked as evidence accrues, recorded
against the entry that made them. That entry predicted `momentum ~ risk` would
fall from +0.516 to +0.150. Measured on the 09-09 run: **+0.084** - from the
**largest** of the 28 category pairs to the **20th**. The mechanism confirms
too: `sharpe_ratio`/`sortino_ratio` now sit at +0.925/+0.917 with momentum and
+0.119/+0.120 with risk, exactly the diagnosis that put them at weight 0. Added
as a **Confirmed 2026-09-09** subsection to the 2026-09-02 entry.

### Evidence / research

- **Published research**, carried from Monday and unchanged: Chan, Jegadeesh &
  Lakonishok (1996) REV6 +7.7% 6-month decile spread (IBES 1977-93) and the
  Stickel (1991) replication (+7.07%); Martineau (2022) - analyst-surprise PEAD
  "non-existent since 2006" for all-but-microcap, 2016-19 coefficient
  significantly negative; Novy-Marx (2015) size gradient, SUE alpha t=2.83 in
  the largest quintile where price momentum is insignificant (t=1.48).
- **Documented practice**, carried from Monday: Barra USFAST `Sentiment`
  descriptors are revisions ("surprise" appears zero times in the datasheet);
  the Zacks Rank's four components, of which this screener implements only
  Surprise.
- **Measured today, full universe, N=502** - all new, and the reason the
  session existed: exact composite reproduction (err 0.0); coverage 99.6%;
  0% ties price-scaled vs an undefined mean estimate-scaled; diffusion 75.8%
  ties; the growth/momentum/surprise correlation table; the five-variant
  mechanical-vs-economic test; spanning R2 for the metric and for all eight
  categories; top-50 turnover for three scenarios against the delete-the-
  category calibration bar.
- **No backtest number and no figure from `live_ic_history.csv`** appears in
  §8 or in the changelog addition (rules 4 and 5). The `1m` horizon holds **3
  effective observations** against a gate of 8; §8.8's confirmation is a
  structural correlation on published output, not a return measurement, and
  says so explicitly.

### Methodology changed

- **None.** No weight, threshold, metric definition or scoring path moved
  today; composites and ranks are untouched. The changelog's own opening line
  is "Changes here are **applied**, not proposed", so the settled design gets no
  entry until Thursday applies it - putting proposed weights in the audit trail
  would misdescribe what the tool currently does.
- **One addition to an existing entry:** a `Confirmed 2026-09-09` subsection on
  the 2026-09-02 risk-category entry, recording that its predicted effect landed
  and overshot in the right direction. That is a measurement of an applied
  change, which is what the file is for.

### Tried and rejected

- **Blaming the momentum overlap on the price denominator.** The tidy answer,
  and false - five reconstructions including sign-only all carry ~0.32-0.43,
  and the denominator alone correlates -0.128 with the wrong sign. Killed by
  the measurement rather than argued away, which is the only reason the +0.417
  can be reported as a market fact.
- **Softening Monday's materiality bar because it is asymmetric.** The
  asymmetry is real and is stated in §8.5; it is still not grounds to ignore a
  threshold set in advance the moment it fires. The change proceeds on construct
  validity instead, which is a weaker-sounding and more honest claim.
- **The conservative weighting (`fy1_revision_3m` at 20 rather than 35).**
  Measured and reported (revisions~momentum +0.272, 1 of top-50, median |Δrank|
  5) so a future reader can see the trade - but rejected: at 20 the surprise
  family is still 63% of a category named for revisions, which leaves the defect
  the change exists to fix largely in place.
- **Diffusion (`eps_revisions`, Zacks "Agreement") as a second new metric.**
  75.8% ties on the full universe. Three names in eight would be one
  indistinguishable rank block.
- **Building the metric today.** Wednesday is design; the note explicitly
  assigned (b) and (c) to Thursday. The build touches the fetch path
  (+1 request/ticker), `METRIC_COLS`/`METRIC_DIR`/`CAT_METRICS`, `config.yaml`,
  `schemas.py`, the golden fixture, `SCREENER_OVERVIEW.md`, `stock_summary.py`
  and a new test module - a full session's careful work, and half of it done
  today would be worse than all of it done tomorrow. A weekly usage ceiling
  exists and argues the same way.

### Next

1. **Thursday: build `fy1_revision_3m` and apply the §8.7 weights as one
   change.** The design is settled, the numbers for the changelog's *Expected
   effect* are already measured (3 of top 50, median |Δrank| 9,
   revisions~momentum +0.171 -> +0.317, unspanned 91.1% -> 85.2%), and §8.7
   states exactly what may and may not go under *Evidence*. Note the
   second-order effect not yet measured: adding a 45th metric to `METRIC_COLS`
   changes the coverage-discount denominator, so a stock near the 0.80 threshold
   shifts slightly. Also update the `config.yaml` comment and
   `SCREENER_OVERVIEW.md`, which the 09-07 session already corrected on the
   "requires FactSet" claim.
2. **Priority 5, the sell-side workflow** - still the top open north-star item
   at **35 days** and still untouched. Two Thursdays in a row will now have gone
   to something else. If Thursday's build lands cleanly, this is Friday's or
   next week's, and it should not slip again without the age being written down.
3. The run-level overview sentence, the narrow remainder of priority 4.

---

## 2026-09-10 - BUILD. Implement what the week's research justified. Write tests alongside the code.

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes** - `logs/nightly-2026-09-09_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-09` |
| Data loop published? | **Yes** - `logs/datarun-2026-09-10_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored |
| Evidence base | **37 rows, newest 2026-09-03, 3 effective observations at `1m`** (9 raw) against a gate of 8 |
| Priority 0 | Fixed 2026-08-24, not weakened today. `_effective_observations()` untouched |
| Top open roadmap item | **Priority 5, the sell-side workflow - 36 days old**, still untouched |

**Tests:** before 1117/1117, after **1161/1161**. Zero failures either side; the
44 new tests are `tests/test_fy1_revision.py`.

**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
deferred. Thursday taken as the focus, building exactly what §8.7 of Monday's
research note specified and Wednesday settled. Priority 5 was **not** taken -
see *Next*, and note its age is now written down for the third consecutive
session.

### Did

**Shipped `fy1_revision_3m` and the revisions reweight as one change.** The
category was named for revisions and contained none: 78 of its 100 points sat
on the earnings-**surprise** family, whose drift Martineau (2022) documents as
absent in large caps since 2006. It now leads with an actual revision, at
weight 35. The category's **10% share of the composite did not change** - only
the split inside it.

| Metric | Was | Now |
|---|---|---|
| `fy1_revision_3m` | - | **35** |
| `analyst_surprise` | 38 | **15** |
| `consecutive_beat_streak` | 20 | **10** |
| `earnings_acceleration` | 20 | 20 |
| `price_target_upside` | 12 | **10** |
| `short_interest_ratio` | 10 | 10 |

Touched: `factor_engine.py` (fetch block, metric, `METRIC_COLS` / `METRIC_DIR`
/ `CAT_METRICS`), `config.yaml`, `schemas.py`, `generate_dashboard.py`,
`stock_summary.py`, the golden fixture, three existing test modules whose
pinned counts genuinely moved, and five documents.

**Verified live, not just against mocks.** Fetched 8 real tickers end-to-end:
**8/8 coverage**, +1.9 to +52.7 bp, inside the full-universe distribution the
research measured (p10 -19.5, median +5.9, p90 +52.5 bp). A metric that passes
44 unit tests and has never touched the real feed is not finished.

**Closed the second-order effect §8.7 flagged and left unmeasured.** The
coverage-discount denominator in `compute_composite()` really is `METRIC_COLS`,
so a 45th metric does shift every stock's coverage ratio - the note was right to
flag it. Measured on the live payload: max change in the discount **0.0019**
(~0.1 point of composite) and **zero** names cross the 0.80 threshold. For the
~2 names lacking the metric it rises by at most 0.0029, which is the mechanism
working, since they do have less data.

**Found and fixed a defect while writing the tests.** The price-denominator
lookup followed the existing `d.get("currentPrice", d.get("price_latest"))`
idiom, which does **not** fall back when `currentPrice` is *present but NaN* -
`dict.get` returns the NaN and never reaches the default. A name in that state
would have lost the metric despite `price_latest` being available. The test was
written first, failed, and the code was fixed rather than the expectation
lowered. **Note for a future session: five other metrics still use the unguarded
idiom** (lines ~1802, 1909, 2064, 2090, 2170) and have the same latent hole.
Not fixed today - out of scope and each needs its own check - but written down.

**Two things deliberately not done, both to avoid fabricating data.**

- **The synthetic sample path leaves the metric `NaN`.** That generator emits
  finished metric values and carries no price field at all, so it cannot build
  this one. I started to add a price so it would compute, then measured the
  blast radius: it would have changed five unrelated metrics on that path.
  Inventing a consensus revision is also precisely the fabrication failure the
  2026-08-11 and 2026-09-01 fixes exist to prevent. NaN is honest, matches how
  `price_target_upside` and `proximity_52w_high` already behave there, and the
  `has_data` renormalisation absorbs it. `test_screener.py`'s "generator
  produces every `METRIC_COL`" guard now carries a **documented exception
  list**, asserted to stay at most 3 long, rather than being deleted.
- **The revisions coverage auto-disable guard was left alone.** It samples
  `analyst_surprise` and `price_target_upside` only, so it no longer includes
  the category's heaviest metric. Adding the new one would make the guard
  *less* likely to fire - a behaviour change with no evidence behind it, in a
  degenerate-case guard that has never fired. Flagged, not touched.

**Display needed a new format, which the research had not anticipated.** Under
the existing `pct` at one decimal, the measured p10/median/p90
(-0.00195 / +0.00059 / +0.00525) collapse onto two strings - manufacturing
*visible* ties in a metric measured at **0.0% actual ties**. Added a `bp`
(basis points of price) format to both the Python and the JS formatter, with a
test asserting they agree: the drilldown's prose and its metric table are
rendered by different code paths, and `_fmt_metric`'s docstring already promised
it mirrors the emitted JS.

**Corrected three stale documentation claims** found by following my own change
rather than by looking for them:

- `SCREENER_OVERVIEW.md` limitation 5 said a revision metric "would require a
  paid data source like FactSet or Refinitiv I/B/E/S". **False**, and now
  corrected in place with the correction visible rather than silently rewritten.
  The honest residual limitation is that the feed only reaches 90 days back, so
  revision *persistence* and CJL's 6-month window remain out of reach.
- The 2026-08-13 changelog entry pins "18 of the 44 metrics move with the daily
  close". `fy1_revision_3m` has price in its denominator, so it is **19 of 45**.
  The claim it supports gets slightly stronger. `tests/test_cache_freshness.py`
  pins the new numbers.
- `README.md` and `SCREENER_DEFENSIBILITY_SPEC.md` both described the registry
  split as **32 scored + 4 bank + 8 candidates**. The true split has been
  **28/4/12** since `sharpe_ratio` and `sortino_ratio` went to weight 0 on
  2026-09-02 - the two errors cancelled, so the total stayed a plausible 44 and
  nobody noticed. Now 29/4/12 of 45, counted from `METRIC_COLS` and
  `config.yaml` rather than incremented by hand.

### Evidence / research

- **Chan, Jegadeesh & Lakonishok (1996)**, J. Finance 51(5): of the three
  earnings-momentum legs the analyst-revision measure (REV6) was strongest,
  **+7.7% six-month decile spread**, IBES 1977-93; **Stickel (1991)**
  replication **+7.07%**. Price scaling is CJL's own construction.
- **Martineau (2022)**, Critical Finance Review 11(4): PEAD - what the surprise
  metrics rely on - **non-existent since 2006** outside microcaps, 2016-19
  coefficient significantly negative. This is an S&P 500 screener, so that is
  exactly this universe. The case for cutting surprise 38 → 15.
- **Novy-Marx (2015)**, NBER w20984: earnings-momentum alpha strongest in large
  caps (SUE **t = 2.83** top quintile, price momentum insignificant at 1.48).
  Why the momentum overlap is read as economic rather than as a bug.
- **Practice:** Barra USFAST `Sentiment` is built from revision descriptors -
  "surprise" appears **zero** times in the datasheet; the Zacks Rank has four
  components and this screener implemented only Surprise, the weakest.
- **Measured (full 502-name payload, 2026-09-09):** coverage 99.6% vs 99.2% for
  the metric it takes weight from; 0.0% ties; +0.152 with `forward_eps_growth`;
  71.6% unspanned by the other eight categories.
- **Measured today:** live 8-ticker fetch; the coverage-discount second-order
  effect.
- **No backtest figure and no `live_ic_history.csv` number** appears in the
  changelog entry (rules 4 and 5). The `1m` horizon holds **3 effective**
  observations against a gate of 8.

### Methodology changed

- **`METHODOLOGY_CHANGELOG.md` 2026-09-10** - the new metric and the reweight,
  as one entry. It states explicitly that it does **not** claim the ranking
  improves, and why: Monday's pre-registered materiality bar **fired** (deleting
  the category outright moves 7 of the top 50; this change moves 3). The bar is
  asymmetric and the entry says so, but a threshold set in advance and explained
  away the moment it fires is not a threshold. The case rests on **construct
  validity** - a category named for revisions now measures revisions - and on
  the literature, not on a claim of better ranking.
- The entry also records the cost in the open: `revisions ~ momentum`
  **+0.171 → +0.317**, 4th largest of the 28 category pairs, and revisions
  unspanned **91.1% → 85.2%**. Independence spent deliberately, named as a cost.
- **Inline correction** to the 2026-08-13 entry (18 of 44 → 19 of 45).

### Tried and rejected

- **Adding a price field to the synthetic generator** so the metric would
  compute there. Measured first: it would have activated five unrelated
  metrics on that path. Rejected as out of scope, and fabricating a consensus
  revision would be the exact failure two prior fixes exist to prevent.
- **Weakening the "generator produces every `METRIC_COL`" test** to make it
  pass. Replaced with a documented exception list that is itself bounded, so
  the guard still fails on any metric that was simply never wired up.
- **Diffusion (`eps_revisions`)** and the **conservative `fy1=20` variant** -
  both settled by Wednesday (75.8% ties; and at 20 the surprise family is still
  63% of the category). Not relitigated, per §8.7.
- **Touching the revisions auto-disable guard.** No evidence either way; a
  behaviour change smuggled in beside a justified one.

### Next

1. **Priority 5, the sell-side workflow** - top open north-star item, now **36
   days**. The week's research is shipped and Thursday's build landed, so there
   is no longer a research thread pulling against it. Question 3 ("should I sell
   what I hold?") is still unanswerable. **This should be the next session's
   work unless the data loop breaks.**
2. **The `dict.get` NaN-fallback hole in five other metrics** (`factor_engine.py`
   ~1802, 1909, 2064, 2090, 2170). Same shape as the one fixed today; each needs
   its own check of whether a present-but-NaN `currentPrice` actually occurs.
   Small, contained, and a good warm-up rather than a session.
3. Confirm the shipped change against the live run: tomorrow's 02:00 data run is
   the first to score with the new weights, so the What Changed panel should
   show roughly the predicted 3-of-top-50 turnover. If it shows far more, the
   *Expected effect* line is wrong and the changelog entry needs the correction
   recorded against it.

---

## 2026-09-11 - HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes** - `logs/nightly-2026-09-10_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-10` |
| Data loop published? | **Yes** - `logs/datarun-2026-09-11_020001.log` ends "Data loop complete", 502 scored, top EXPE HST APA VLO EIX |
| Evidence base | **39 rows, newest 2026-09-04, 3 effective observations at `1m`** (10 raw) against a gate of 8 |
| Priority 0 | Fixed 2026-08-24, not weakened today. Nothing in this session touches `_effective_observations()` or any scoring path |
| Top open roadmap item | **Priority 5, the sell-side workflow - 37 days old**, still untouched |

**Tests:** before 1161/1161, after **1193/1193**. Zero failures either side; the
32 new tests are `tests/test_percentile_direction.py`.

**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
deferred. Friday's focus taken as written. Priority 5 was **not** taken - see
*Next*, fourth consecutive session its age has been written down.

### Did

**The dashboard published a percentile whose obvious reading was backwards, and
now it doesn't.** `compute_sector_percentiles()` does `ranks = 100 - ranks`
wherever `METRIC_DIR` is `False`, so a published percentile always means "best
in its sector" and never "largest". That is **13 of the 37 published metrics**,
and nothing on the page said so. Measured on the live payload:

| Stock | EV/EBITDA | Published percentile |
|---|---|---|
| HON | 6.95 | **99** |
| AXON | 98.61 | **0** |

Same shape for beta (RSG -0.37 at the 99th vs CVNA 2.35 at the 0th) and PEG
(UAL 0.24 at the 99th vs KMI 28.34 at the 0th). In prose it read as a flat
contradiction: *"the 97th sector percentile on EV/EBITDA (9.02)"* (HST, live).

This is the surface whose entire purpose is explaining **why** a stock ranks
where it does. A student who learns the convention backwards misreads every
valuation and risk metric on the site - which is the exact question this day
exists to ask.

Three fixes, plus the category columns:

1. **`metric_meta[m]["dir"]`, derived from `factor_engine.METRIC_DIR`** rather
   than written out. This is the load-bearing decision: the page cannot claim a
   direction the scorer disagrees with, and a test compares the two on every
   build. Hand-writing 37 directions would have been a second source of truth,
   which is how three documentation claims went wrong before 2026-09-10.
2. **Drilldown metric table** - header `Percentile Rank` → `Sector Percentile -
   100 = best`, a convention note under it, and a `↓ better` / `↑ better` chip
   with a tooltip beside every metric name.
3. **Summary prose** - `_label_and_value()` appends `, lower is better` for the
   13 inverted metrics only.
4. **The eight category columns** (`Val`, `Qual`, `Grow`, `Mom`, `Risk`, `Rev`,
   `Size`, `Inv`) were bare abbreviations with no explanation anywhere on the
   page - only `Δ` had a tooltip. They now carry definitions naming their scored
   metrics and weights, the bank carve-out for Valuation and Quality, and, for
   `Risk`, the fact that a **high** score means **low** risk.

**Verified I changed nothing I said I didn't.** Rebuilt the payload and compared
cell-by-cell against the live one: **0 differing `Composite`/`Rank`/category
cells and 0 stocks with changed `raw`/`pct`**, across all 502. Cost measured at
**+1.2 KB gzipped (+0.1%)** on a 1,200 KB wire payload - measured after gzip,
because raw size is the wrong number here.

**Two false claims caught before shipping, by checking instead of recalling.**
My draft tooltips named **P/B** under Valuation and **PEG** under Growth. Both
carry **zero** weight for non-banks - `config.yaml` marks P/B "Bank-only" and
PEG "Removed: P/E ÷ growth double-counts valuation". I had written them from
`metric_meta`, which lists everything *displayed*; the scored set is in
`config.yaml`'s active weights, and `CAT_METRICS` is also wrong for this because
it includes zero-weight candidates. A test now pins the correction. Writing
tooltips from memory would have shipped a confident, specific, wrong account of
how the score is built - worse than the silence it replaced.

**Confirmed yesterday's change against the live run, and it confirms nothing.**
The 2026-09-11 run was the first scored with the new revisions weights. Top-50
turnover was **exactly 3**, matching the pre-registered figure. But baseline
day-over-day churn across the prior ten transitions is **median 3, range 0-7**,
so a no-op day yields the same number. The prediction was not falsified and was
also not tested. Recorded against the 2026-09-10 changelog entry as an
underpowered observation rather than as confirmation, because filing it as a
pass would make the next reader believe the change was validated when it wasn't.

### Evidence / research

- **A demonstrable user-facing failure**, which is the acceptable evidence type
  for a hardening day: the HON/AXON, RSG/CVNA and UAL/KMI pairs above, read off
  the live published payload, plus the HST summary sentence.
- **The mechanism, in this repo's own code:** `factor_engine.py` lines ~2416 and
  ~2435, `ranks = 100 - ranks` guarded by `METRIC_DIR`. 13 of 37 published
  metrics, counted rather than estimated.
- **27 of the 32 new tests fail against the pre-change code** (26 generator, 1
  prose), each confirmed by stashing the change and re-running.
- **Measured:** 0 score/rank/raw/pct cells changed; +1.2 KB gzipped; 515 summary
  sentences gained the qualifier; 0 advice-term breaches across all 502
  summaries, checked through `advice_terms_in()` rather than by eye.
- **No backtest figure and no `live_ic_history.csv` number** is used anywhere -
  neither would be relevant, since nothing about scoring changed.

### Methodology changed

- **`METHODOLOGY_CHANGELOG.md` 2026-09-11** - filed under presentation of scored
  data, and states plainly that **no weight, metric, threshold or formula
  moved**. It is in the changelog because it changes what a published number
  *means to a reader*, which is the part of methodology the audit trail is for.
- **Validation note appended to the 2026-09-10 entry** recording the
  3-of-top-50 observation and why it is underpowered.
- `plan/dashboard-inventory.md` updated in the same session (rule 9), including
  the warning to read active weights from `config.yaml` rather than
  `CAT_METRICS` when describing what a category scores.

### Tried and rejected

- **Adding the direction qualifier to higher-is-better metrics in the prose.**
  The ambiguity exists only where percentile and raw value point opposite ways;
  502 stocks of prose is ~101 KB gzipped, and a phrase in every sentence is not
  free. The drilldown chip covers all 37, so the general case is still taught.
- **Hand-writing `dir` into the 37 `metric_meta` literals.** Faster to read, but
  a second source of truth for a fact the scorer already owns. Derived instead,
  with a test asserting agreement.
- **A hardcoded "13 of 37" in the convention note.** Computed in JS from the
  payload, so it cannot go stale when a metric is added or reweighted.
- **The `dict.get` NaN-fallback hole in five other metrics**, left by the
  2026-09-10 session as the suggested warm-up. I checked it and it is **not
  worth fixing as described**: `rec["currentPrice"]` is set unconditionally by
  `_safe()`, so the key always exists and the `d.get("currentPrice",
  d.get("price_latest"))` fallback is **dead code at every site**, not just
  latent. But incidence is **zero** - all 502 names on the live payload carry a
  price. So it is a safety net that does not exist rather than a bug that is
  firing, and the honest fix is to make the fallback real *or* delete it and
  stop implying coverage. That is a decision, not a warm-up; it should not be
  bundled into a session about something else. Written up here so the next
  session inherits the finding rather than repeating the investigation.

### Next

1. **Priority 5, the sell-side workflow** - top open north-star item, now **37
   days**. Question 3 ("should I sell what I hold?") is still unanswerable.
   Nothing is now pulling against it: the week's research shipped, Thursday's
   build landed, and today's work is closed. **This should be the next session's
   work unless the data loop breaks.**
2. **Decide the `currentPrice` fallback** (see *Tried and rejected*): make it
   real at all six sites, or remove it and the comment at `factor_engine.py`
   line ~683 that promises it. Either is defensible; leaving a documented
   safety net that cannot fire is not.
3. The 2026-09-10 *Expected effect* line still has no discriminating test
   behind it. If run-level A/B scoring is ever cheap to add, it would make
   every future "expected effect" claim checkable instead of decorative.

---

## 2026-09-14 - RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes** - `logs/nightly-2026-09-11_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-11` |
| Data loop published? | **Yes** - `logs/datarun-2026-09-14_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored, top EXPE HST VLO APA CAH |
| Evidence base | **41 rows, newest 2026-09-07, 3 effective observations at `1m`** (11 raw) against a gate of 8 |
| Priority 0 | Fixed 2026-08-24, not weakened today. Research-only session; `_effective_observations()` and every scoring path untouched |
| Top open roadmap item | **Priority 5, the sell-side workflow - 40 days old.** Not built today, but this session is the research that unblocks it - see *Owner queue / rotation* |

**Tests:** before 1193/1193, after **1193/1193**. No production code changed;
the run is a no-new-failures check, not a claim of new coverage.

**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
deferred. Monday's focus taken as written.

**On priority 5, and why this counts as progress on it.** Its age has now been
written down for five consecutive sessions, each time as "still untouched".
Monday is research day and priority 5 is a build item, so the two do not compete
directly - but the topic was chosen so that they stop pulling against each
other. `plan/dashboard-north-star.md` parked "what sell disciplines have
evidence behind them?" as Monday research question 3, and priority 5 is the
build that question exists to inform. That question is now answered. Thursday
can build from a note instead of from intuition, which is the whole point of
having a research day ahead of a build day.

### Did

**One research note, complete today:
`research/2026-09-14-sell-discipline-and-hold-bands.md`.** Five papers and three
index-provider methodologies, read from primary sources rather than summaries -
`pypdf` against the downloaded PDFs, because `WebFetch` cannot read a PDF and
returns a confident "I cannot extract this" that is easy to mistake for "the
source does not say".

**The headline finding is that selling is the part of the process where
documented professional skill disappears.** Akepanidtaworn, Di Mascio, Imas &
Schmidt (2023, *JF* 78(6)) track 783 institutional portfolios averaging $573M,
2000-2016, 4.4M trades. Against counterfactuals built from the managers' own
holdings: buys beat a random-buy counterfactual by **over +100 bp/year**; sells
**underperform a factor-neutral random-sell counterfactual by -80 bp/year**.
That deficit is larger than the fee these managers charge.

**The mechanism is what constrains the build, and it cuts against the obvious
design.** The deficit is an attention failure, not a skill failure: PMs sell
positions that are extreme on prior returns - best *and* worst - at rates
**more than 50% higher** than middling positions, a pattern that survives
stock-date fixed effects. The proof that it is attention: on earnings-
announcement days, sells beat non-announcement-day sells by **+150 bp/year** and
actually beat the counterfactual, while buying performance is unchanged.

So **a review queue ranked by size of move is the documented error, automated**
- and that is exactly what `plan/dashboard-north-star.md` gap 2 currently
specifies ("a review queue of owned names whose scores dropped materially").
Amended in place today, rule 9.

**The one construction rule that is both well-evidenced and standard practice:
an asymmetric hold band.** Novy-Marx & Velikov (2016, *RFS* 29(1)) find a
buy/hold spread is "the single most effective simple cost mitigation strategy";
their momentum factor nets **0.51%/month (net FF4 alpha 0.33, t=8.81)** under
trading hysteresis against **0.31%/month (alpha 0.17, t=3.06)** restricting to a
low-cost universe. MSCI Momentum buffers at 50% of target count (buy rank 250,
hold to 750) and pointedly does *not* apply its turnover buffer to deletions.
S&P DJI states the principle outright: **"the addition criteria are for addition
to an index, not for continued membership."** Three sources, hold bands of
1.5x-3x the buy band, none symmetric. This screener has **one** test -
`config.yaml -> portfolio.num_stocks: 25`, a plain top-N cut.

**Measured on our own ranking - two findings I did not expect.** Descriptive
statistics only, from `improvement/snapshots/` through the existing Spearman
comparability gate; **no forward returns, no IC, no backtest**, so rules 4 and 5
do not bite.

- **The top of the ranking is far stickier than the universe.** Absolute rank
  change between consecutive runs: universe p50 **7**, p95 **43** (n=13,542);
  names in the top 25, p50 **1**, p95 **10** (n=675).
- **Therefore the movers panel is nearly blind to holdings.** Its "material
  mover" threshold is the universe 95th percentile, currently 43 ranks. Top-25
  names clear it **1 time in 675 holding-days - 0.15%**. A name can fall from
  rank 3 to rank 27 and never appear in "What Changed". Not a defect in that
  panel, which is a universe-discovery surface doing its job; a demonstration
  that **the cheapest way to build priority 5 - reuse the movers threshold -
  is the wrong one.**
- **A strict sell rule would mostly generate trades that undo themselves.** Over
  the dense 18-run weekday window 08-20 to 09-14, "sell when it leaves the top
  25" fires on **7.3%** of holding-days (31 of 425) and **71% of those (22 of
  31) are back inside the top 25 within five runs**. A 25/50 band produced zero
  signals - though see the caveat, which is in the note and repeated below.

### Evidence / research

- **Akepanidtaworn, Di Mascio, Imas & Schmidt (2023)**, *JF* 78(6) 3055-3098 /
  NBER w29076: buys +>100 bp/yr, sells **-80 bp/yr** vs random-sell; extremes
  sold at **>50%** higher rates; earnings-day sells **+150 bp/yr** better.
  Deficit **worst** among fundamentals-oriented concentrated high-tracking-error
  managers - which is precisely this screener's shape.
- **Odean (1998)**, *JF* 53(5): PGR **0.233** vs PLR **0.155** (1.50x, t=-32);
  winners sold beat losers held by **+1.03%/84d (p=0.002), +3.41%/252d
  (p=0.001), +3.58%/504d (p=0.014)**. Reverses in December (t=4.6).
- **Barber & Odean (2000)**, *JF* 55(2): 66,465 households 1991-96;
  highest-turnover **11.4%/yr** vs market **17.9%**; average household 16.4% at
  75% annual turnover. The number to show an investment club.
- **Novy-Marx & Velikov (2016)**, *RFS* 29(1) / NBER w20721: Table 5 above.
  Round-trip costs **>50 bp** value-weighted; costs cut realized spreads by
  **>1% of monthly one-sided turnover**; anomalies under **50%** monthly
  one-sided turnover mostly survive costs, few above do.
- **Kaminski & Lo (2014)**, *JFM* 18, 234-254: stopping premium **always
  negative** under a random walk (Proposition 1, analytic). Positive only under
  return persistence; the empirical result (+1.5% return, -5% vol, Sharpe +20%)
  is a **stocks-vs-bonds index-futures overlay at monthly frequency**, and they
  find **no value at short sampling frequencies**.
- **Practice, primary documents read directly:** MSCI Momentum Indexes
  Methodology (July 2025) §3.1.1 / §3.1.2 / Appendix III; S&P DJI Select
  Industry Methodology, "Turnover".
- **Measured here:** the three bullets above, from 32 comparable runs
  2026-02-20 to 2026-09-14.

### Methodology changed

- **None.** No weight, metric, threshold, formula or scoring path was touched,
  so there is no `METHODOLOGY_CHANGELOG.md` entry. Monday is research day and
  the prompt says no production code; the note's recommendations are explicitly
  left for Wednesday's synthesis (§8) to settle and Thursday to build.
- `plan/dashboard-north-star.md` updated in the same session (rule 9): research
  question 3 marked answered with its findings, and **gap 2's "review queue"
  wording amended in place**, because that sentence is what a future session
  would build from and the evidence now contradicts it.

### Tried and rejected

- **A per-stock stop-loss**, the first thing anyone reaches for and a genuine
  academia/practice disagreement. Ruled out by Kaminski & Lo (2014): the only
  rigorous study of stops proves the stopping premium is negative under a random
  walk, and its positive result is confined to an **asset-class overlay at
  monthly frequency** - not single names, not daily. Practice applies stops in
  exactly the regime the paper finds worthless. Second, independent reason: a
  stop fires on prior-return extremes, which is the heuristic that costs
  80 bp/yr.
- **Reviewing more often than quarterly.** `config.yaml` line 221 already
  records quarterly manual rebalancing. MSCI reviews quarterly; NMV's
  staggered-quarterly variant beats the low-cost-universe variant; Barber &
  Odean price the churn. **No change warranted** - written into the note so a
  future session does not relitigate it.
- **Treating the 25/50 band's zero signals as proof the band is right.** It is
  an 18-run, 25-day, fairly quiet window, and the worst next-run rank for any
  top-25 name in it was 46 - so "never breached" partly means "nothing bad
  happened". The robust numbers are the **71% round-trip rate** and the rank
  distributions, computed over the full 32-run series. A band that never fires
  is a dead feature, not a conservative one, and the synthesis has to settle
  that before anything ships. Recording this rather than quoting the clean zero.
- **Citing the S&P Quality 20% buffer as primary evidence.** spglobal.com
  returns HTTP 403 to automated fetches, so that parameterisation rests on two
  independent search retrievals, not the PDF. Flagged as such in the note
  instead of being presented at the same confidence as the seven sources I
  actually opened.

### Next

1. **Wednesday 2026-09-16 is the synthesis, and section 8 of the note lists the
   five questions it must settle** - band width above all, since a 2x band may
   be too wide to ever fire on this screener. Then **Thursday builds priority
   5**, at last, from a written argument rather than from intuition.
2. **The band width needs re-measuring at 60+ comparable runs.** There are 32
   today. This is the single measurement that would change the design, and it
   accrues on its own as the data loop runs - no work required, just don't
   commit to a width before it exists.
3. Still open from 2026-09-11: **decide the `currentPrice` fallback** - make it
   real at all six sites in `factor_engine.py`, or delete it and the comment at
   ~line 683 that promises it. Unchanged today; a documented safety net that
   cannot fire is still worse than none.

---

## 2026-09-15 - PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes** - `logs/nightly-2026-09-14_060001.log` ends "Run complete: shipped to main", tagged `good/2026-09-14` |
| Data loop published? | **Yes** - `logs/datarun-2026-09-15_020001.log` ends "Data loop complete", HEALTH: PASS, 502 scored, top EXPE HST VLO APA CAH |
| Evidence base | **42 rows, newest 2026-09-08, 3 effective observations at `1m`** (11 raw) against a gate of 8. Up from 41 rows yesterday - moving |
| Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call was touched today |
| Top open roadmap item | **Priority 5, the sell-side workflow - 41 days old. The list half shipped today.** The remaining half (a hold band) is blocked on measurement, not design - see below. Next unblocked item is **Priority 3, backtest v2**, whose plan file dates to 2026-08-25 - **21 days** |

**Tests:** before **1193/1193**, after **1264/1264**. 71 new tests, zero
failures either side.

**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
deferred. Tuesday's product focus taken as written, and it pointed at the same
place the roadmap did: Priority 5 is a *product* gap and Tuesday is the product
day, so for once the rotation and the north star wanted the same thing.

**On running ahead of Wednesday's synthesis.** Monday's note (2026-09-14) parked
five design questions for the 09-16 synthesis. I built anyway, and only the part
that none of those questions gate: the **list**, with no threshold of any kind.
Question 1 - band width - is untouched and still open, which is the point.
`CLAUDE.md` records nine consecutive sessions that produced real work and
shipped no north-star item because something smaller always looked more urgent;
deferring a 41-day-old product item on the product day, when the research it was
waiting for landed yesterday, would have been the tenth.

### Did

**Shipped the sell-side workflow's list half: a "My Holdings" panel.** Between
Top 5 and What Changed. A `localStorage` list under `screener_holdings_v1`
holding **tickers and nothing else**, rendering every saved name each run as a
card: rank, composite, an eight-category score-and-delta strip, and the review
sentences already baked into `stock_detail[t]["summary"]`. Above it, a
concentration line - names, sectors, largest sector share, how many sit inside
the top 25 and the top 100, how many carry a trap flag.

**It costs nothing in payload.** It is a *view* over fields `stock_detail`
already carried. `plan/dashboard-inventory.md` warns that the likeliest failure
here is rebuilding what exists; the useful version of heeding that was noticing
that `stock_summary.py` already produces build-time, advice-screened sentences
covering what changed, what is flagged and what the score rests on. The panel
renders those rather than composing its own prose in the browser, which keeps
the 2026-09-08 property that what shipped is what a reader can diff.

**Three of its properties are research constraints with tests behind them**, not
styling, and each is the opposite of the obvious design:

1. **Every saved name renders, every time** - never a filtered subset, because
   the documented institutional failure is a *restricted consideration set*.
2. **Ordered by rank, never by size of move.** The rank change is shown for
   context; it is not the sort key and not a filter.
3. **No cost basis, share count or P&L**, in the code or in storage. A key
   hand-edited to hold `{ticker, shares, cost}` is read for its ticker and
   written back clean. This is also why it works equally as a watchlist -
   nothing about it assumes you own the name.

**Added `change_driver` to the per-stock summary**, which every drilldown gets,
not just holdings. `change` said how far a stock moved; nothing said *what
moved it*. The new sentence names the category that moved furthest since the
history baseline, the direction of its **score** (said explicitly, because a
high Risk score means low risk and "Risk down 22 points" otherwise reads as an
improvement), and what that category now contributes to the composite. It and
`change` read their baseline from one helper so they cannot describe different
windows.

**Fixed a live grammar defect while in the file:** "moved up 1 **places**" was
on the public site for every one-rank mover - and a one-rank move is the
*median* for a top-25 name, so it was about to become the commonest sentence on
the surface I was shipping.

**Verified by rendering, not by grepping.** Nine of the new tests drive the real
emitted script under Node against a stubbed DOM and assert on output, because
"the string appears in the file" is a weak check for a panel whose entire
contract is what it renders. I also rendered the live 502-stock payload with six
real holdings and read the result through.

### Evidence / research

All from `research/2026-09-14-sell-discipline-and-hold-bands.md`, which read
each source directly.

- **Akepanidtaworn, Di Mascio, Imas & Schmidt (2023)**, *JF* 78(6) 3055-3098.
  783 institutional portfolios averaging $573M, 2000-2016, 4.4M trades. Sells
  underperform a factor-neutral random-sell counterfactual by **-80 bp/year**;
  buys beat theirs by **over +100 bp/year**. The mechanism is attention:
  positions extreme on prior returns - best *and* worst - are sold at rates
  **>50% higher** than middling ones, surviving stock-date fixed effects;
  earnings-day sells beat non-announcement-day sells by **+150 bp/year**. The
  deficit is *worst* among fundamentals-oriented concentrated
  high-tracking-error managers. → constraints 1 and 2, and `change_driver`.
- **Odean (1998)**, *JF* 53(5) 1775-1798. PGR **0.233** vs PLR **0.155**, a
  **1.50x** ratio at **t = -32**; winners sold beat losers held by **+1.03% /
  84 days (p=0.002)** and **+3.41% / year (p=0.001)**. The effect is defined
  relative to purchase price. → constraint 3.
- **Novy-Marx & Velikov (2016)**, *RFS* 29(1) 104-147. A buy/hold spread is
  "the single most effective simple cost mitigation strategy"; hysteresis nets
  **0.51%/month, net FF4 alpha 0.33 (t=8.81)** against **0.31%/month, alpha
  0.17 (t=3.06)** for a low-cost universe. With **MSCI Momentum** (buy 250,
  hold 750 against a 500 target) and **S&P DJI** ("the addition criteria are
  for addition to an index, not for continued membership"). → why a hold band
  belongs here eventually, and why it is a *different* test.
- **Barber & Odean (2000)**, *JF* 55(2) 773-806. Highest-turnover households
  **11.4%/yr** against a market **17.9%**. → the footnote's cost-of-churn line,
  which is the sentence an investment club should read.
- **Measured here, new today:** across the 500 live stocks with a one-month
  category delta, the largest mover is **Risk 34.0%, Revisions 29.0%, Momentum
  26.4%, Valuation 3.2%, Investment 3.2%, Growth 2.8%, Size 1.2%, Quality 0.2%
  - one stock in 500.** ~90% of one-month category movement comes from the
  three categories fed by daily prices and estimates. **A deterioration trigger
  keyed to Quality or Growth would essentially never fire at monthly cadence**,
  which rules out the most intuitive reading of "fundamental deterioration" and
  is a direct input to §8 question 2. Descriptive statistics on published
  scores - no forward returns, no IC, no backtest.
- **Measured here, costs:** `change_driver` **+99 KB raw / +10.5 KB gzipped
  (+0.89%)** across 502 stocks; the page **+25.9 KB raw / +6.4 KB gzipped**.
  Total **+16.9 KB on the wire, ~+1.4%**.

### Methodology changed

- **`METHODOLOGY_CHANGELOG.md` 2026-09-15.** No weight, metric, threshold,
  formula or scoring path changed. The entry exists because the *shape* of the
  surface is a methodology decision - three of its properties are omissions
  taken from the literature, and without the record they read as arbitrary.
- Rule 9 updates in the same session: `plan/dashboard-inventory.md` (new
  section, eleven summary kinds, payload table, "genuinely missing" item 2),
  `plan/dashboard-north-star.md` (gap 2 marked shipped-in-part with what is
  left), `CLAUDE.md` priority 5, and an update block on §8 of the research note
  recording which of its five questions moved.

### Tried and rejected

- **A hold band, which is the best-evidenced rule in the whole note.** Three
  independent sources put it at 1.5x-3x the buy band and two of them are live
  index products. I did not ship one. §6.3 measured a 25/50 band firing **zero**
  times across the 18-run window and the strict top-25 rule producing sells that
  **round-trip 71% of the time** (22 of 31 back inside the top 25 within five
  runs); §9 asks for **60+ comparable runs** before committing to a width and
  there are **32**. A band that never fires is a dead feature, not a
  conservative one, and picking 50 because MSCI doubles would be borrowing a
  number from a 500-name quarterly index for a 25-name daily screen. It accrues
  on its own - **do not guess a width.**
- **Reusing the movers panel's threshold**, which is the cheapest way to build
  this. Measured: 43 ranks is the universe p95, and a top-25 name clears it
  **1 time in 675 holding-days - 0.15%**. A name can fall from rank 3 to rank 27
  and never appear. Not a defect in that panel; a demonstration that it is the
  wrong instrument here.
- **A "review queue of names whose scores dropped materially"**, which is what
  `plan/dashboard-north-star.md` originally specified and what anyone would
  build. That is a queue ranked by size of move, i.e. the -80 bp/year heuristic
  automated and presented as a feature. The plan was amended yesterday; today
  the code makes the amendment real, and both the plan and the inventory now
  keep the argument rather than just the conclusion, because the shortcut will
  look reasonable again to the next reader.
- **A gain/loss column.** Every retail portfolio tracker has one. Odean (1998)
  is the reason not to, the cost of honouring it is zero, and retrofitting it
  later would be expensive - so it was settled before the first line of code.
- **Putting the review prose in the browser.** It would have been simpler, and
  it would have given up exactly the diffability and per-reader identity that
  justified deleting the chat on 2026-09-08.

### Next

1. **Wednesday 2026-09-16, the synthesis (§8 of the note).** Questions 1 (band
   width), 4 (earnings dates) and 5 (implied turnover) are untouched. Question 2
   now has the category-movement distribution above to work from, and question 3
   is answered in shipped code. Question 4 looks like the strongest remaining
   thread: earnings-date proximity is north-star gap 4 *and* the one
   information anchor the evidence positively endorses (+150 bp/year), and it is
   cheap - the fetch already touches the provider response that carries it.
2. **The band still needs 60+ comparable runs** (32 today, ~1 per weekday). No
   work required; just do not commit to a width before it exists.
3. Still open from 2026-09-11: **decide the `currentPrice` fallback** in
   `factor_engine.py` - make it real at all six sites or delete it and the
   comment at ~line 683 that promises it. Unchanged again today.

---

## 2026-09-16 - SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes** - `logs/nightly-2026-09-15_060000.log` ends "Run complete: shipped to main", tagged `good/2026-09-15` |
| Data loop published? | **Yes** - `logs/datarun-2026-09-16_020000.log` ends "Data loop complete", HEALTH: PASS, 502 scored, top EXPE HST VLO CAH BBY |
| Evidence base | **43 rows, newest 2026-09-09, 3 effective observations at `1m`** (11 raw) against a gate of 8. Up from 42 yesterday - moving |
| Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call touched. Today **extended its lesson** to a third place - see below |
| Top open roadmap item | **Priority 5, the sell-side workflow - 42 days old.** Its remaining half (the hold band) is now **settled as "not yet, until ~2027-04"** with a pre-registered rule rather than left vague. Next unblocked item is **Priority 3, backtest v2**, plan file dated 2026-08-25 - **22 days** |

**Tests:** before **1264/1264**, after **1264/1264**. No production code changed;
this is a no-new-failures check, not a claim of new coverage.

**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
deferred. Wednesday's synthesis taken as written, working §8 of Monday's note.

### Did

**Wrote §8, the design section Monday's note was left open for - and in doing so
found that two of that note's headline numbers were wrong.** The deliverable is
the design; the correction is the more important half.

**1. §6.3's "a 25/50 band never fires" was an estimator artifact.** Monday walked
**one path** through 18 runs and got zero. Taking instead *every ordered pair* of
comparable runs at a given calendar spacing - 34 runs now - a 2x band fires at
**1.65%** of weekly holding-looks. It is rare, not dead.

**2. But the pairwise estimator has the project's own oldest defect, and this is
the finding of the day.** **34 runs yield 72 pairs at a fortnight's spacing**, so
each run feeds many pairs and the observations are nowhere near independent -
the identical trap `research/2026-08-10-ic-evidence-independence.md` found in the
IC series and `improvement_engine._effective_observations()` was built to guard
against. **Nobody had noticed it applies to rank statistics too.** Restricting to
non-overlapping pairs:

| spacing | pairs all → disjoint | B=50 breach all → disjoint |
|---|---|---|
| 5-9 days | 68 → **8** | 1.65% → **4.50%** |
| 12-18 days | 72 → **4** | 3.99% → **10.89%** |
| 25-35 days | 37 → **2** | 1.93% → **4.00%** |

The overlapping estimator **understates wide-band breach rates by 2-3x** - the
same order as the ~2.35x the IC note measured - and the honest sample is **8
weekly, 4 fortnightly, 2 monthly independent looks.** So the true position was
never "a 2x band never fires"; it is **"we have two independent monthly looks"**,
which is the same evidential state the improvement engine is in, reached
independently, and it earns the same answer: do not act, and say so.

**3. Settled the band as "not yet", with a date and a pre-registered rule.** A
band *is* warranted - the strict top-25 rule wastes **31-47% of the trades it
implies** (37.5% daily / 47.4% at 2-4 days / 31.0% weekly), and that replicates
three times out of three. Its **width is not determinable**: at 1.4x the three
cadences report 0.0%, 31.2% and 5.9% wasted, on 9/7/3 disjoint triples. §9's
"60+ comparable runs" is the **wrong unit** - 60 runs of a daily series is still
2-3 independent monthly looks. Replaced with: **>= 8 disjoint windows at the
review cadence the band will govern**, ~**2027-04**, within a month of when the
improvement engine hits its own gate, for the same reason. Plus an explicit
"do not pick 50 because MSCI doubles".

**4. Found a real coherence gap the evidence *does* settle, needing no
threshold.** `config.yaml` line 221 records a **quarterly** rebalance cadence.
The dashboard regenerates **every weekday and states no cadence anywhere**
(grep confirms the only "quarterly" in `generate_dashboard.py` is an unrelated
data-source label). Strict top-25 turnover is **121.8% monthly one-sided at
daily review against 24.0% at monthly**, where Novy-Marx & Velikov find few
anomalies survive costs above ~50%. **A surface that redraws a rank every
morning invites daily action the methodology prices at 2.4x outside the
survivable region.** That conclusion tolerates a 2.4x error before it changes.

**5. Answered "what should the trigger key on" with a measurement, not a
preference.** When a metric's availability changes, its category renormalises
over a different metric set and the score moves because the *measurement*
changed - the FCX case in priority 1.5, now counted over 24 run-pairs:

| input churn | n | median &#124;rank change&#124; | share worsening |
|---|---|---|---|
| none | 11,450 | **6** | 44.6% |
| 1 metric | 447 | 7 | 47.0% |
| 2-3 metrics | 143 | **21** | 52.4% |

One lost metric is indistinguishable from noise; **two or more triples the
median rank move**, so a flag should arm at >=2. And it is **nearly
direction-neutral** - it scatters ranks rather than pushing them down, which is
exactly the signal a sell surface must not present as a reason to act. Among
top-25 exits, **6 of 43 (14.0%) coincided with input churn against 3.8% of
stayers**; that ratio is indicative only, for the independence reason above, and
the note says so. The mechanism needs no significance test - renormalisation
moving a score is arithmetic. `_sentence_confidence()` already states the
coverage **level** and never the **change**; `history.py` carries no metric
count. One missing quantity is the whole gap.

**6. Verified rather than repeated Monday's earnings-date claim.** `.info` -
already fetched at `factor_engine.py:744` - does carry `earningsTimestampStart`,
`earningsTimestampEnd` and **`isEarningsDateEstimate`**, checked live against
AAPL/HST/EXPE. Nothing in the repo reads any of them. So it is genuinely
zero-API-cost, and the estimate flag must be **shown**: EXPE's next date is
flagged estimated today.

**7. Answered the day's actual question - what it implies for the other seven
categories.** ~90% of one-month category movement comes from Risk, Revisions and
Momentum, and priority 1.5 records that momentum and risk are 23% of composite
weight from a single `Ticker.history()` call. **So a hold band on composite rank
is, at monthly cadence, mostly a band on price wearing eight categories as a
costume.** Not a stop-loss, but it shares the input Kaminski & Lo found
unsupported at single-name frequency. Two consequences: `change_driver` is
**load-bearing, not decorative** - it is the only thing distinguishing "the
business changed" (essentially never, monthly) from "the price moved" (usually);
and any future deterioration trigger must be **category-aware**, because a
composite-rank band weights a rare Quality move identically to a routine
Momentum one.

**8. Made the numbers re-runnable, which they were not.**
`research/measurements/2026-09-16-hold-band-and-input-churn.py` reproduces every
figure above from `improvement/snapshots/` through the same comparability gate
`history.py` uses, printing overlapping and disjoint estimators side by side so
the difference cannot be overlooked again. Monday's note asked a later session to
re-measure and left no script to re-run - which is precisely how two of its
numbers survived as long as they did. New convention documented in
`research/README.md`.

### Evidence / research

- **Novy-Marx & Velikov (2016)**, *RFS* 29(1) 104-147. Anomalies under ~**50%
  monthly one-sided turnover** mostly survive trading costs; few above do. A
  buy/hold spread is "the single most effective simple cost mitigation
  strategy". → the turnover yardstick in §8.2 and the pre-registered rule.
- **Akepanidtaworn, Di Mascio, Imas & Schmidt (2023)**, *JF* 78(6) 3055-3098.
  Sells underperform a factor-neutral counterfactual by **-80 bp/year**;
  earnings-day sells beat non-announcement-day sells by **+150 bp/year** and are
  the only sells that beat their counterfactual. → §8.4, why earnings dates are
  the endorsed anchor.
- **Kaminski & Lo (2014)**, *JFM* 18, 234-254. Stopping premium negative under a
  random walk by proposition; positive result confined to asset-class overlays at
  monthly frequency. → §8.6, the tension in a price-driven rank band.
- **MSCI Momentum Indexes Methodology (July 2025)** §3.1.1-3.1.2; **S&P DJI
  Select Industry Methodology**, "Turnover". → the 1.5x-3x practice range, and
  the reason not to borrow 50 from a 500-name quarterly index.
- **Measured here today**, all from the script above, descriptive statistics on
  published scores - no forward returns, no IC, no backtest, so rules 4 and 5 do
  not bite: the disjoint-vs-overlapping breach table; wasted-trade rates
  31-47% at B=25 falling monotonically with width; 121.8% vs 24.0% monthly
  one-sided turnover; the input-churn table and the 14.0%/3.8% exit contrast.
- **Verified live today:** `.info` earnings fields on AAPL/HST/EXPE.

### Methodology changed

- **`METHODOLOGY_CHANGELOG.md` 2026-09-16 - "No hold band, and the rule for when
  one may be chosen".** No weight, metric, threshold, formula or scoring path
  changed; `portfolio.num_stocks: 25` and every category weight are
  byte-identical. The entry exists because a **decision not to change** needs to
  be as findable as a change, the changelog explicitly covers portfolio
  construction rules, and the most likely future failure is a session picking 50
  because MSCI doubles.
- Rule 9 updates in the same session, all correcting things this session proved
  wrong: `research/2026-09-14-...md` (§8 written, §9's stopping rule struck
  through and restated in the right unit), `plan/dashboard-north-star.md` (gap
  2's superseded numbers corrected, the "buy 25 / hold ~50" width struck, gap 4
  marked verified), `CLAUDE.md` priority 5, and `research/README.md` (new
  `measurements/` convention plus a Standards entry on counting independent
  observations).

### Tried and rejected

- **Committing to a band width of 35**, which the weekly wasted-trade curve
  (31.0 → 20.8 → 5.9 → 3.4 → 0.0%) makes look like a clean knee. It is not: the
  2-4 day cadence puts B=35 at **31.2%** and daily puts it at 0.0% on **four
  breach events**. Three cadences, three answers, 9/7/3 disjoint triples.
  Picking 35 because one row looked tidy is the failure the project's evidence
  rules exist to prevent, and §9 had pre-registered a stopping condition
  precisely so this moment would not be a judgement call.
- **Overriding §9's stopping rule because today's numbers came out
  interesting.** Honouring a pre-registered threshold when the data looks
  tempting is the whole point of pre-registering it. What I did instead was
  correct its **unit**, which is a different act and is argued in §8.1.
- **Quoting n=1,803 observations** for the fortnightly breach rate, which is what
  the natural estimator hands you and what I would have written down had I not
  checked. It is **4 independent looks**. The project has now hit this trap three
  times; it is in `research/README.md` Standards so the fourth time is someone
  ignoring a written warning rather than rediscovering it.
- **Concluding that the band is redundant with the quarterly cadence**, which was
  my working hypothesis for most of the session and is wrong. Cadence controls
  *how often* you may trade; the band controls *whether the trade is worth
  making*. Lengthening cadence barely moves the wasted-trade rate (37.5% daily →
  31.0% weekly), so it does not substitute for a band. They are complementary,
  not overlapping - the opposite of what I expected to find.
- **Building any of it.** Wednesday is synthesis and Thursday is build; yesterday
  already ran one day ahead, and the value of writing the design first is
  precisely what §8.0 demonstrates - the obvious build would have shipped a band
  width chosen from an artifact.

### Next

1. **Thursday 2026-09-17 builds §8.7, in order:** (a) **say what cadence the tool
   is for** - no threshold, no new data, the gap the evidence most clearly
   supports; (b) **flag input-availability change on the holdings surface**,
   armed at >=2 metrics, worded as a caveat on the move rather than a reason to
   act, which needs `history.py` to carry a per-ticker metric count - note the
   pre-2026-03-09 snapshots have **no percentile columns at all** (15 columns),
   so handle the absence rather than assuming the schema; (c) **earnings dates,
   display-only, with the estimate flag shown**.
2. **The band is on a timer, not a queue.** ~2027-04, at >=8 disjoint monthly
   windows. Re-run the measurement script and read the **DISJOINT** column. No
   work required before then; do not guess a width.
3. Still open from 2026-09-11: **decide the `currentPrice` fallback** in
   `factor_engine.py` - make it real at all six sites or delete it and the
   comment at ~line 683 that promises it. Unchanged again today, third session
   running.

---

## 2026-09-17 - BUILD. Implement what the week's research justified. Write tests alongside the code.

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes.** `logs/nightly-2026-09-16_060000.log` ends "Run complete: shipped to main", tagged `good/2026-09-16` |
| Data loop published? | **Yes.** `logs/datarun-2026-09-17_020000.log` ends "Data loop complete", 502 scored, HEALTH: PASS, 0 fetch failures |
| Evidence base | **44 rows, newest 2026-09-10, 3 effective observations at `1m`** (11 raw) against a gate of 8. Up from 43 rows / 2026-09-09 yesterday - moving. Effective count unchanged at 3, as expected: 1-month observations accrue about one a month |
| Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call touched today |
| Top open roadmap item | **Priority 5, the sell-side workflow - 43 days old.** Two of its three remaining build items shipped today. Next unblocked item is **Priority 3, backtest v2**, plan file dated 2026-08-25 - **23 days** |

**Tests:** before **1264/1264**, after **1362/1362** (+98: 58 new in
`test_input_churn.py`, 40 in `test_review_cadence.py`).

**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty, so nothing was
deferred. Thursday's build taken as written, implementing §8.7 items 1 and 2 of
the 2026-09-14 research note - the rotation's intended path, not a swap.

### Did

**Shipped the two §8.7 items that share a single argument: the tool now says how
often it is meant to be acted on, and says when a move is the measurement
changing rather than the company.** Both are about not inviting action the
evidence cannot justify, which is why they went together rather than one per
session.

**1. The cadence the tool is built for is now stated on the surfaces that move.**
`config.yaml` has recorded a quarterly rebalance cadence since launch - as a bare
*comment*, which the generator could not read. The site regenerates every weekday
and said nothing: a grep of `generate_dashboard.py` for "quarterly" returned one
unrelated data-source label. A surface that redraws a rank every morning
implicitly invites acting on it every morning.

`portfolio.review_cadence` is now a real key, surfaced as `D.cadence` and
rendered by `cadenceText(long)` in three places - the holdings panel (short form,
both empty and populated states), the What Changed footnote (short form), and the
holdings footnote (long form with the numbers). Read from the **run's own** config
snapshot, not the working tree, so a republished old run states what it was
configured for; `configured: false` marks the fallback so it cannot be mistaken
for a real setting. Both paths are exercised today: the published 2026-09-17 run
predates the key and falls back cleanly; tonight's data run will carry it.

**It is a sentence, not a lock**, and that was a deliberate call. The tool does
not know what a reader is doing. Naming the cadence is decision support;
withholding a number until a date would not be, and would also defeat the data
loop's reason for running daily.

**2. A rank move that is really an input going missing now says so.** When a
metric percentile flips between present and absent, its category renormalises
over a different metric set and the score moves as **arithmetic** - no company
event. This is priority 1.5's FCX case (growth 68.3 -> 42.5 -> 68.3), which was
investigated as a suspected defect and turned out to be correct behaviour that
nothing downstream could tell apart from a real collapse. `CLAUDE.md` priority
1.5 has recorded it as an open product gap since 2026-08-26.

`history.py` now carries per-ticker metric availability and emits
`ch: [lost, gained]`; `stock_summary._sentence_input_churn` turns >= 2 into a
caveat on the drilldown and on every holdings row. **Fires for 27 of 502 stocks**
on today's run against the ~1-month baseline.

**Three things I got from building it that the design did not anticipate:**

- **A metric *count* is not enough.** A net count difference of zero hides one
  metric dropping out as another returns - exactly the case the flag exists for.
  The implementation compares availability **sets**.
- **Only columns both runs carry can be compared.** The snapshot schema has grown
  (`fy1_revision_3m_pct` appears part-way through the directory). Counting a
  column that did not exist yet as a metric that went missing would flag the
  entire universe on the day a metric was added. There is a test for this.
- **27 of 502 is 5.4%, against §8.3's measured 1.22%, and that is expected
  rather than a misfire.** §8.3 measured consecutive runs <= 7 days apart; the
  drilldown's preferred baseline is ~28 days, over which more availability
  changes accumulate. Written into the changelog so a future session does not
  read it as the threshold being wrong.

**3. Corrected `plan/dashboard-inventory.md`, which still carried both numbers
the 2026-09-16 session disproved.** It read "32 comparable runs is short of the
60+ the research asks for" and "a 25/50 band fired zero times". The 09-16 session
corrected `plan/dashboard-north-star.md`, `CLAUDE.md` and the research note, and
missed this one - so the file the Tuesday focus tells the next session to trust
was the last place the superseded numbers survived. Rule 9 territory; found while
updating the same file for today's work.

### Evidence / research

- **Novy-Marx & Velikov (2016)**, *RFS* 29(1) 104-147. Anomalies under roughly
  **50% monthly one-sided turnover** mostly survive trading costs; few above it
  do. → the cadence line. Measured against it on this repo's own snapshots:
  **121.8%** at daily review vs **24.0%** monthly, so daily action sits **2.4x
  outside** the survivable region - a conclusion that tolerates a 2.4x error in
  the estimate before it reverses.
- **Barber & Odean (2000)**, *JF* 55(2). Most active household quintile earned
  **11.4%/yr against a 17.9% market return**. → already in the panel copy; the
  household-level version of the same point.
- **Akepanidtaworn, Di Mascio, Imas & Schmidt (2023)**, *JF* 78(6) 3055-3098.
  Institutional sells trail a factor-neutral counterfactual by **-80 bp/year**.
  → why the churn caveat is worded as a caveat and never as deterioration.
- **Measured (§8.3, 12,044 ticker-transitions over 24 run-pairs):** median
  |rank change| **6** with no churn, **7** with one metric changed, **21** with
  two or three; churn >= 2 leaves **52.4%** worse off against a **44.6%** base
  rate. → the threshold of 2, and the near-symmetry that forbids calling it bad
  news. All descriptive statistics on published scores - no forward returns, no
  IC, no backtest, so rules 4 and 5 do not bite.
- **Verified today:** 52 of 58 churn tests and 37 of 40 cadence tests **fail
  against the pre-change code**, checked by stashing the four source files.

### Methodology changed

- **`METHODOLOGY_CHANGELOG.md` 2026-09-17** - "The tool states the cadence it is
  built for, and flags when a rank move is the inputs changing". No weight,
  metric, threshold, formula or scoring path changed; every category weight,
  `portfolio.num_stocks: 25` and all 45 metric definitions are byte-identical and
  no published score moves. The entry exists because `review_cadence` is a new
  portfolio-construction key and because the two display rules (arm at 2; never
  word it as deterioration) are research constraints a future session would
  otherwise "tidy".
- Rule 9 updates: `plan/dashboard-inventory.md` (two new sections, plus the
  superseded hold-band numbers corrected) and §8.7 of
  `research/2026-09-14-sell-discipline-and-hold-bands.md` marked built, with the
  set-vs-count design point recorded.

### Tried and rejected

- **Storing the available metric set per ticker per run.** The obvious shape, and
  it would make `history.py` the largest thing the payload builder holds -
  ~45 names x ~500 tickers x ~60 kept runs, for a quantity two sentences consume.
  Stores the **missing** set instead, which is empty for ~95% of tickers.
- **Wording the churn caveat as a data-quality warning**, which is what it looks
  like at first glance. Ruled out by the measurement: 52.4% worse off against a
  44.6% base rate is near-symmetric, so churn *scatters* ranks rather than
  pushing them down. A surface presenting it as deterioration would manufacture
  the exact sell trigger Akepanidtaworn et al. find costs 80 bp/year. There is
  now a test asserting the sentence contains none of "deteriorat", "worse",
  "warning", "risk", "concern", "weaken", "decline", and that it reads
  identically whether the stock rose or fell.
- **Arming the flag at >= 1 metric.** Median rank move 7 against a baseline of 6
  - indistinguishable from noise, and it would mark 4.93% of transitions instead
  of 1.22%. A caveat that fires four times as often as it means anything trains a
  reader to ignore it, which is how the permanent bank-only "High severity" alarm
  became worthless (fixed 2026-09-01).
- **Gating or hiding the daily refresh behind the quarterly cadence.** Considered
  and rejected: the tool does not know what a reader is doing, and the data loop
  running daily is what accrues the evidence base. Stating the cadence is
  decision support; enforcing it would not be.
- **Building §8.7 item 3 (earnings dates) as well.** Separable - it touches the
  fetch layer rather than the history spine and needs a full refetch to populate
  - and the weekly usage ceiling argues for one coherent thing done properly.
  Left open on the merits of scope, not of evidence; §8.4 still endorses it.
- **Committing `index.html` / `dashboard_data.js`.** Generated artifacts, only
  ever committed by the 02:00 data run (rule 10). The generator change ships;
  tonight's run publishes it.

### Next

1. **§8.7 item 3: earnings dates, display-only, with the estimate flag shown.**
   `.info` already carries `earningsTimestampStart/End` and
   `isEarningsDateEstimate` (verified live 2026-09-16 on AAPL/HST/EXPE) and
   nothing in the repo reads any of them, so it is zero-API-cost. It is the one
   place the literature positively endorses spending attention: earnings-day
   sells beat non-announcement-day sells by **+150 bp/year** and are the only
   sells in Akepanidtaworn et al. that beat their counterfactual. **Show the
   estimate flag** - EXPE's next date is flagged estimated.
2. **Then Priority 3, backtest v2** - 23 days old and the top unblocked
   north-star item now that priority 5's buildable half is done.
3. **The hold band is on a timer, not a queue.** ~2027-04, at >= 8 disjoint
   monthly windows. Re-run
   `research/measurements/2026-09-16-hold-band-and-input-churn.py` and read the
   **DISJOINT** column. Do not guess a width.
4. Still open from 2026-09-11, **fourth session running**: decide the
   `currentPrice` fallback in `factor_engine.py` - make it real at all six sites
   or delete it and the comment at ~line 683 that promises it. It is small; it
   keeps losing to larger work. Worth doing next time it is the cheapest thing
   available rather than carrying it a fifth time.

---

## 2026-09-18 - RETROSPECTIVE. Evaluate whether this routine is producing value, and change the process where it is not.

**Health (rule 8, all five):**

| Check | Reading |
|---|---|
| Last code session ran? | **Yes.** `logs/nightly-2026-09-17_060000.log` ends "Run complete: shipped to main", tagged `good/2026-09-17` |
| Data loop published? | **Yes.** `logs/datarun-2026-09-18_020000.log` ends "Data loop complete", HEALTH: PASS, 502 scored, 0 fetch failures, price coverage 502/502 |
| Evidence base | **At `1m`: 11 rows, newest 2026-08-14, 3 effective observations** against a gate of 8. Read at `1m` for the first time today - see finding 1. Lag **35 days**, inside the new 40-day bound; it is the 08-15..08-19 outage still working through the pipe, and resumes 09-21 |
| Priority 0 | Fixed 2026-08-24, not weakened. No scoring path, weight, threshold or `_effective_observations()` call touched |
| Top open roadmap item | **Priority 3, backtest v2** - `plan/backtest-v2.md` dated 2026-08-25, **24 days**. Not taken: a retrospective does not work on the screener |

**Tests:** before **1362/1362**, after **1378/1378** (+16, `tests/test_payload_parse_gate.py`; 11 of them fail against the pre-change runners).
**Owner queue / rotation:** `OWNER_FOCUS.md` **Open** is empty; nothing deferred. ISO week 38, Friday, even week - retrospective, per the rotation.

### Retrospective findings

- **Sessions reviewed: 9 scheduled** (2026-09-07 to 2026-09-17). No owner-run
  sessions this period - the first fortnight with none.
- **Genuinely valuable: 9 | Churn: 0 | Failed gates: 0.**

**1. What fraction produced something genuinely valuable? All nine, and there
is a merge commit per session to check it against.** 09-07 found the Revisions
category contains no revisions; 09-08 shipped priority 4, replacing the AI chat
with deterministic per-stock summaries; 09-09 settled the design and caught two
of Monday's own numbers being wrong; 09-10 built `fy1_revision_3m` and the
reweight; 09-11 fixed a published percentile whose obvious reading was
backwards on 13 of 37 metrics; 09-14 researched sell discipline from five
papers and three index methodologies; 09-15 shipped My Holdings; 09-16 found
the project's independence trap for the third time, in rank statistics, and
settled the hold band as "not yet" with a date; 09-17 shipped the cadence
statement and the input-churn flag. Sessions ran **13.0-22.2 API-minutes**
against a 4-hour limit; none came close.

**The quality signal worth naming: five of the nine corrected this project's
own published claims** rather than defending them - 09-09 corrected Monday's
+0.346 to +0.401, 09-10 corrected three stale doc claims including a registry
split wrong since 09-02 where two errors cancelled, 09-11 caught two false
tooltips before shipping by checking `config.yaml` instead of recalling, 09-16
corrected two of Monday's headline numbers, 09-17 corrected the inventory's
superseded hold-band figures. That is the habit that makes the rest credible.

**2. Which rotation day earns its place? All five, and the Mon-Wed-Thu chain
is now demonstrably load-bearing rather than nominal.** 09-07 research →
09-09 synthesis → 09-10 build shipped one coherent methodology change across
three days, and the synthesis day paid for itself twice: it killed the
price-denominator explanation for the momentum overlap by measuring five
reconstructions, and it let Monday's pre-registered materiality bar fire
instead of explaining it away. Same shape 09-14 → 09-16 → 09-17, where
Wednesday's finding was that Monday's headline number was an estimator
artifact. **A build day fed by a synthesis day catches things a build day fed
by intuition cannot.** The 09-04 retrospective called Thursday "the weak one";
on this fortnight it was not, and the fallback rule added then was never needed.

**3. Is the evidence standard holding? Yes, and it tightened in a way I did
not expect.** Every changelog entry carries its sources; none are thin - the
five entries since 09-08 run 80-160 lines each. Three specific signs:
**09-10's entry states in terms that it does not claim the ranking improves**,
because the pre-registered materiality bar fired; **09-11 filed its own
validation observation as underpowered** rather than as a pass, on the ground
that baseline churn produces the same number; and **09-16 declined to pick a
band width** the data made tempting, honouring a stopping rule set before the
numbers were seen. The research notes are real - Martineau (2022) *CFR*
11(3-4), Akepanidtaworn et al. (2023) *JF* 78(6), Odean (1998), Novy-Marx &
Velikov (2016) *RFS* 29(1), Kaminski & Lo (2014) - read from primary PDFs, with
effect sizes and the conditions they held under. No backtest figure and no IC
number appears under **Evidence** anywhere in the period.

**4. What keeps going wrong? Documentation drifts out of true faster than
anyone corrects it, and the drift lands in the instructions themselves.** This
is the same answer the 09-04 retrospective gave, and it has not improved. Found
today, all shipped and all wrong: `CLAUDE.md` line 20 and `prompts/nightly.md`
line 4 both still said **"You push to `main` yourself"** - the first thing a
session reads, directly contradicting the change 09-04 made 190 lines later in
the same prompt; `nightly-screener.ps1`'s own synopsis said "the session is
autonomous and merges its own work"; and the repo-growth warning in `CLAUDE.md`
was wrong by **8x** (finding 3). Rule 9 says keep process docs true, and
sessions do update the docs they touch - but nobody re-reads the top of the
file they are given.

**5. Is the tool closer to the place you would look before buying or selling?
Yes, and for the first time the answer is about surfaces rather than
correctness.** Question 3 ("should I sell what I hold?") went from completely
unanswerable to a working My Holdings panel (09-15) that renders every saved
name with its category deltas and review sentences, plus a stated review
cadence and a caveat when a rank move is the inputs changing rather than the
company (09-17). Question 2 gained a deterministic "Why it ranks here" on all
502 names (09-08). **The honest remaining blocker is question 4, "how much"**,
which nothing addresses and no open item claims - position sizing was removed
with the model portfolio in August and never replaced by anything that answers
it. The hold band, the other half of question 3, is correctly parked on a
measurement until ~2027-04.

**6. What is the routine systematically blind to? The data loop's publish
path - and that is where today's defect was.** The 09-04 retrospective added
"read the runner scripts" to this prompt, and I did; but it framed them around
`finally` and the brief, so I read `nightly-screener.ps1` first and found it
sound. The hole was in the *other* runner, in the ordinary path, and it is
structural rather than incidental: **the code loop's ship gates are the
project's strongest check and run on the path that publishes a handful of times
a week, while the data loop publishes the actual payload to GitHub Pages five
mornings a week behind a `> 100000 bytes` size floor.** Nothing in the rotation
compares the two. Fixed below, and the comparison is now an instruction.

Second blind spot, smaller: **nobody had ever run `git count-objects`**, despite
`CLAUDE.md` explicitly asking for repo growth to be raised in the log. Zero
mentions in the whole of `NIGHTLY_LOG.md`.

### Did - three process defects, found by reading rather than by an incident

**1. The evidence-base tripwire had stopped being able to fire.** Rule 8 asks
for "row count, newest date, and effective observations at `1m`". Sessions read
that as the whole file, and logged "44 rows, newest 2026-09-10" - both numbers
rising every weekday, because `1w` gains a row per run date as it ages. But
`1m` is the optimization horizon and the only one the engine's gate reads, and
**its newest `run_date` has been 2026-08-14 since 2026-09-07 - six consecutive
sessions, each of which logged a newer date.**

The freeze is legitimate: there are no snapshots for 08-15..08-19 (the
documented outage), and 08-20 + 30 days is 09-19, so `1m` resumes on 09-21.
That is the point. **The tripwire was written to catch a stall, and it was
wired to a quantity that cannot stand still.** Rule 8's own text says "if those
two numbers have not moved in three consecutive sessions, making them move is
that session's work" - unfireable as instrumented, and it would have fired
spuriously every month if it had been read at `1m` instead.

So the condition is replaced rather than re-pointed: **the newest `1m`
`run_date` must be within 40 days of today.** In steady state the lag is 30-33
(the horizon plus a weekend), so 40 tolerates a week of missed runs and fires
on anything worse. It reads **35** today. Rule 8, `prompts/nightly.md`
sections 1 and 5, and `prompts/retrospective.md` section 1 all now say `1m`
explicitly and carry the bound.

**2. The data loop publishes the payload to the public site behind a weaker
check than the code loop applies.** `data-run.ps1`'s pre-publish sanity block
checked `index.html > 50000` and `dashboard_data.js > 100000` bytes. That is
it. `nightly-screener.ps1`'s gate 3 added a first-line regex - also not a
parse, which the 08-21 retrospective identified and deliberately declined to
fix, correctly, because neither PowerShell nor node could be executed in that
session and a gate that can only fail closed jams the loop.

Both run now (node v24.19.0, PowerShell verified this session), so it is
verified rather than deferred a third time. **Measured against the live 5 MB
payload before writing anything:**

| payload | `node --check` | passed the old checks |
|---|---|---|
| full (5,019,178 bytes) | 0 | yes |
| truncated to 50% (2.5 MB) | **1** | **yes** |
| truncated to 90% | **1** | **yes** |
| header only | 1 | no |

A half-written payload is 2.5 MB, opens with `window.SCREENER_DATA =`, clears
every check both loops had - and renders a blank page. The parse costs
**0.14s** on the full file.

Both runners now run `node --check` before they commit or merge. Where node is
absent both **fall back to the checks they already had and log a `WARN`** -
strictly stricter than before, and still unable to jam an unattended loop,
which is the property the 08-21 session was right to insist on.

**Verified by execution, not by assertion.** The real gate-3 block was lifted
verbatim out of `nightly-screener.ps1` and run against a truncated payload:
`GATE 3: dashboard_data.js does not parse as JavaScript`, gate failed. Same for
`data-run.ps1`'s block, in all four states - full/node → publishes;
truncated/node → `Refusing to publish a dashboard payload the browser cannot
read`, exit 2; full/no-node → publishes with the WARN; truncated/no-node →
publishes with the WARN, i.e. exactly the old behaviour and no worse. Both
runners re-parsed with `[Parser]::ParseFile` afterwards.

**16 tests** in `tests/test_payload_parse_gate.py`, **11 failing against the
pre-change scripts** (confirmed by swapping in `git show HEAD:` copies and
restoring, both verified byte-identical by SHA-256). They pin the ordering -
parse before commit, parse before merge - the skip-not-fail behaviour, the WARN,
and that `CLAUDE.md` still claims the payload parses, so the promise and the
implementation cannot drift apart again.

**3. The repo-growth warning was wrong by 8x and nobody had checked it.**
`CLAUDE.md` warned that the payload "adds roughly 60 MB/month of poorly
delta-compressing JSON", and offered downsampling the payload or committing
data less often as remedies. Measured: **40 versions of `dashboard_data.js`
totalling 151 MB raw cost 14.26 MB in-pack - 0.36 MB per version**, roughly a
10x delta ratio, and the whole repository packs to **34.55 MiB**. At 21 weekday
runs that is **~7.6 MB/month.** Acting on the old number would have cut a
payload readers depend on to save nothing.

What was real: git had never repacked. **1,301 loose objects and 4 packs
occupied 99.49 MiB against 34.55 MiB of content**, because loose objects carry
no delta and git's own `gc.auto` threshold of 6,700 was most of a year away at
~30 objects a run. Repacked by hand this session (rule 11), and `data-run.ps1`
now runs a non-fatal `git gc --auto` at `gc.auto=200` after publishing, so it
cannot recur and cannot cost a run.

### Evidence / research

- **A demonstrated failure with a reproduction**, the mandate's fourth
  category, for finding 2: the truncation table above, produced by running
  `node --check` and a reimplementation of the old checks against slices of the
  live payload, and re-confirmed by executing both runners' real blocks.
- **Measured, this session:** `1m` horizon frozen at 2026-08-14 for six
  sessions, against a whole-file newest date of 2026-09-11, read straight off
  `improvement/live_ic_history.csv`; 35-day current lag; 0.36 MB/version
  in-pack via `git verify-pack -v` across all 4 packs, 34.55 MiB total via
  `git count-objects -vH` before and after `git gc`; 9 of 9 merge commits, one
  per session; session durations 13.0-22.2 API-minutes from `logs/*.json`.
- **No backtest figure and no `live_ic_history.csv` number** is used to justify
  anything here (rules 4 and 5). The `1m` count is quoted as a health reading,
  which is what rule 8 asks for - not as support for a change. Nothing this
  session touches scoring.

### Methodology changed

- **None.** No weight, metric, threshold, formula or scoring path moved;
  composites and ranks are untouched, and no run was regenerated. Runner and
  process infrastructure, which by precedent (2026-09-04) lives in this log
  rather than in `METHODOLOGY_CHANGELOG.md`.
- Rule 9 updates in the same session: `CLAUDE.md` (rule 8's evidence-base line
  and its tripwire, the mandate's stale self-push claim, the ship-gates table,
  the repo-growth paragraph), `prompts/nightly.md` (opening paragraph, the
  orientation bullet, the log template), `prompts/retrospective.md` (section 1),
  and `nightly-screener.ps1`'s synopsis.

### Process changes made

1. **The evidence base is read at `1m` and bounded at 40 days.** Rule 8 and
   both prompts. Replaces a "has it moved in three sessions" tripwire that was
   wired to a number which cannot stand still - and would have false-alarmed
   monthly had it been pointed at `1m` unchanged.
2. **Gate 3 is a real parse, on both publish paths.** `node --check` in
   `nightly-screener.ps1`'s gate 3 and in `data-run.ps1`'s pre-publish block,
   falling back with a WARN where node is absent. Strictly stricter, which is
   what section 4 of the retrospective prompt permits; 16 tests.
3. **The retrospective now compares the two runners against each other**, not
   just reads them. Section 1: "where one is stricter than the other about the
   same artifact, the weaker one is usually the bug, and it is usually the one
   that publishes more often." That framing is what today's defect needed and
   what the 09-04 framing - built around `finally` and the brief - pointed away
   from.
4. **The data loop keeps its own object store packed**, non-fatally, after
   publishing.

**Deleted, per "prefer deleting to adding":** `prompts/nightly.md` loses two
paragraphs on validation and the backtest that restated `CLAUDE.md` rules 4 and
5 nearly verbatim while section 3 already covered both - eight lines replaced
by three. `CLAUDE.md` loses the ten-line 2026-08-29 stagger narrative, which is
fixed, verified, recorded in this log, and whose only durable lesson is already
rule 11. `prompts/nightly.md` is net shorter, 214 -> 212 lines. **`CLAUDE.md`
is not: 860 -> 897.** Most of that is the corrected repo-growth paragraph and
the gate-3 note, both of which exist to stop a future session acting on a
number that was wrong - but it is growth, and the next retrospective should
look at priority -1 and the 1.5 entry for narrative that has outlived its use.

### Tried and rejected

- **Adding a trailing-terminator check** to catch truncation without node. It
  would fire on a harmless condition: dropping the final `;` leaves valid
  JavaScript with complete data, and `node --check` accepts it - correctly,
  verified. A check that fires on healthy output is the failure shape fixed on
  2026-09-01 (the permanent bank-metrics "High severity" alarm) and avoided
  again on 2026-09-17 (arming input churn at 2 metrics, not 1). There is a test
  asserting this stays a non-goal.
- **Making a missing `node` fail the gate.** Tidier, and it would jam an
  unattended loop on any machine without node - the precise objection the
  08-21 retrospective raised to shipping this at all, which was right then and
  is right now. Skip-with-WARN is strictly stricter than the old behaviour
  without introducing a way to lose a day.
- **Re-pointing rule 8's "has it moved in three sessions" tripwire at `1m`
  unchanged.** The smallest edit, and wrong: independent 1-month observations
  accrue about one a month, so a correctly-read `1m` horizon stands still for
  three sessions routinely. It would have alarmed every month and been muted,
  which `CLAUDE.md` priority -1 already identifies as worse than no watchdog.
  A staleness bound fires on the condition that actually means something.
- **Making the retrospective monthly.** Considered again, rejected again, and
  more firmly than on 09-04: this is the second consecutive fortnight where
  reading the runners found a live publish-path hole on the first pass. Two for
  two is not a coincidence to thin out.
- **Reserving a rotation day for the runners.** The cheaper intervention is the
  one in change 3 - sharpen what the retrospective looks for - and a sixth day
  would be consumed by whatever defect that day found, exactly as the 09-04
  retrospective argued when rejecting a dedicated product day.
- **Touching the four ship gates in any loosening direction.** Not attempted.
  Gate 3 got stricter; gates 1, 2 and 4 are unchanged.

### Flagged for the owner

- **Nothing needs your decision, and the queue is empty.** One thing worth
  knowing: of the four questions the dashboard exists to answer, three now have
  a surface and **"how much" has none** - no position sizing has existed since
  the model portfolio was removed on your instruction in August, and no open
  priority claims it. Removing that surface was right (a fixed public 25-name
  list is the closest this tool came to a recommendation), but nothing replaced
  the sizing question it half-answered. If you want it back in a defensible
  form, one line under **Open** is the lever - an owner item outranks the
  rotation.

### Next

1. **Priority 3, backtest v2** - 24 days old and the top unblocked north-star
   item. `plan/backtest-v2.md`; it is the last thing standing between this
   project and being able to check its own methodology changes.
2. **§8.7 item 3: earnings dates**, display-only with the estimate flag shown.
   Zero API cost, verified live 2026-09-16, and the one place the literature
   positively endorses spending attention (+150 bp/year).
3. **The `1m` horizon resumes 2026-09-21.** If the newest `1m` `run_date` is
   still 2026-08-14 after that run, the 40-day bound is about to break and that
   is a real stall, not the outage - investigate before anything else.
4. Still open from 2026-09-11, **fifth session running**: decide the
   `currentPrice` fallback in `factor_engine.py` - make it real at all six
   sites or delete it and the comment at ~line 683 that promises it.

---

## 2026-09-21 - RESEARCH. One specific thing, learned properly from the literature AND documented practice, in one session. A dated note in research/, complete today. No production code.

**Health (rule 8, all five):** last code session ran? **yes** - `logs/nightly-2026-09-18_060001.log`
ends "Run complete: shipped to main", tagged `good/2026-09-18` | data loop published?
**yes** - `logs/datarun-2026-09-21_020000.log` ends "Data loop complete" | evidence base
at `1m` = **13 rows, newest 2026-08-21 (31 days ago, bound 40), 3 effective** - the
horizon moved (it read 2026-08-14 for six sessions) and the 08-15..08-19 outage has
cleared the pipe | priority 0 **fixed** (2026-08-24, untouched) | top open roadmap item:
**priority 3, backtest v2, 27 days old**
**Tests:** before 1378/1378, after 1378/1378
**Owner queue / rotation:** `OWNER_FOCUS.md` **Open is empty**, so the rotation governed.
Took Monday research. Nothing deferred.

### Did

One research note, complete in the session, on **position sizing - the "how much"
question**: `research/2026-09-21-position-sizing-and-how-much.md`, with its numbers as a
re-runnable script at `research/measurements/2026-09-21-position-sizing-dispersion.py`.

Why this topic: of the four questions `plan/dashboard-north-star.md` says the dashboard
exists to answer, **"how much / does it fit?" is the only one with no surface at all**,
and has had none since the Model Portfolio was removed on 2026-08-26. The 09-18 session
flagged exactly that to the owner. Research is the right first step because the obvious
implementation - print a recommended weight - is the thing that got the Model Portfolio
deleted.

**The note found a live defect in what this repo already does.** `config.yaml` has set
`portfolio.weighting: 'score'` since launch - composite-score-proportional position
sizing. Measured across **39 run dates (2026-02-20 .. 2026-09-21)**, one snapshot per
date, two degraded 3-row February files excluded:

| | |
|---|---|
| Equal weight, 25 names | 4.00% |
| Score weight, full span | **3.77% .. 4.56%** |
| Max deviation from equal weight | **0.57 pp** (median 0.42) |
| Active share vs equal weight, same names | median **1.31%** |
| Heaviest/lightest ratio | **1.21x** |
| Positions ever hitting the 5% cap | **0** |

So **`weighting: 'score'` is equal weight with noise**, and it cannot be anything else:
composite scores are level-bounded 0-100 and the top 25 of 502 sit in a narrow band
(today 64.75-73.48), so a 13% spread in level becomes a 13% spread in weight around 4%.
**`max_position_pct: 5.0` is inert** for the same reason - it would need a composite 25%
above the selected mean, which has never happened and cannot under this construction.
That is the same failure shape as the always-firing bank-metrics alarm fixed 2026-09-01,
inverted: a control that reads as a safety mechanism and can never fire.

This is a property of the construction arithmetic - "what weights does this rule emit
given these scores" - not a backtest and not a return measurement, so it is not gated by
rules 4 or 5. The note says so explicitly.

### Evidence / research

- **DeMiguel, Garlappi & Uppal (2009), *RFS* 22(5), 1915-1953.** 14 optimisation models
  (incl. Bayes-Stein and shrinkage) across 7 datasets; **none consistently beat 1/N** on
  Sharpe, CEQ or turnover. For sample mean-variance to beat 1/N reliably needs an
  estimation window of **~3,000 months for 25 assets, ~6,000 for 50**. Conditions: US
  equity calibration, monthly rebalance, comparison over the *same* asset set - i.e.
  exactly our question, since selection already happened.
- **Chopra & Ziemba (1993)**, read from **Ziemba & MacLean (2011), ch.1,
  *Stochastic Optimization Methods in Finance and Energy*, Springer ISOR 163**: errors
  in the **means matter ~20x** errors in covariances, with variance errors ~2x
  covariance errors - and **~100:3:1 for near-zero risk aversion**. "So log investors
  must estimate means well if they are to survive." This is the core finding: **score
  weighting sizes by a mean-return estimate**, the single most error-sensitive input,
  using a score whose accuracy this system has **3 effective observations** on.
- **Statman (1987) *JFQA*** 30-40 names; **Campbell, Lettau, Malkiel & Xu (2001) *JF***
  ~50 as idiosyncratic vol rose; **Domian, Louton & Racine (2007) *Financial Review*
  42(4)** - on **shortfall risk** over 20 years, **63 names for 10%, 93 for 5%, 164 for
  1%**. All assume random selection, so they overstate the need for a pre-screened
  large-cap 25 - but `num_stocks: 25` is below every one of them and the tool says so
  nowhere.
- **Plyakha, Uppal & Vilkov.** EW beats VW by **2.71%/yr**, **58% systematic / 42%
  alpha** - and the alpha "depends only on the monthly rebalancing and not on the choice
  of initial weights." The edge is the *rebalancing discipline*, not the weights.
- **Grinold (1989); Clarke, de Silva & Thorley (2002) *FAJ* 58(5).** IR ~ TC x IC x
  sqrt(Breadth). 25 of 502 keeps sqrt(25/502) ~ **0.22** of available IR before any TC
  penalty - but the law assumes *independent* bets, and
  `research/2026-09-02-category-independence-synthesis.md` already shows ours are not.
  Nominal breadth overstates real breadth, the same trap as raw IC row counts.
- **Goetzmann & Kumar (2008), *Review of Finance* 12(3).** US individual investors are
  materially under-diversified, worse among **younger, lower-income, less-educated,
  less-sophisticated** investors, correlated with overconfidence and with overweighting
  high-volatility/high-skew stocks. That is the investment-club demographic exactly: the
  audience's failure mode is too few and too correlated, **not** mis-weighting.
- **Practice (first-class evidence, and strikingly uniform):** RIC Subchapter M
  **25/5/50**; UCITS **5/10/40**; S&P DJI Select Sector **4.8% / 50% / 24%** thresholds,
  with the capping *mechanism* changed 2024-09-23 from clipping the smallest breacher to
  4.5% to reducing all breachers proportionately; S&P 500 Equal Weight resets every name
  to a fixed **0.2% quarterly**, historically +**1.05%/yr** through 2023 (over half of it
  a size tilt) and negative since. Quant shops run constrained optimisers against Barra
  or Axioma. **Nobody sizes long-only equity in proportion to a bounded composite score.
  In every documented scheme the alpha signal drives *selection* and weighting is a
  separate, risk-driven decision.**

### Methodology changed

**None - correctly.** Today is research; the rotation says no production code and the
note's recommendation (`weighting: 'score'` -> `'equal'`) is written up for Wednesday's
synthesis and Thursday's build, with the changelog entry it will need already drafted in
section 9. Shipping it today would have been a methodology change on a research day with
no synthesis pass.

### Tried and rejected

- **Inverse-vol as the new default.** The sophisticated-looking choice, and the evidence
  will not carry it. **Moreira & Muir (2017) *JF* 72(4)** find large alphas from scaling
  by inverse prior realised variance, but **Cederburg, O'Doherty, Wang & Yan (2020)
  *JFE* 138(1)** test **103 strategies** and find vol-managed portfolios do **not**
  systematically outperform; implementable out-of-sample versions **earn lower CEQ and
  Sharpe than the unmanaged originals**, from structural instability in the spanning
  regressions. It helps momentum, profitability and BAB, and nothing else. The only
  claim that survives both papers is descriptive - inverse-vol equalises *risk*
  contribution rather than dollar contribution - so that is all the note claims.
- **Kelly / fractional Kelly sizing.** Ruled out by its own literature: it needs a
  calibrated probability distribution, and this screener emits a cross-sectional rank
  with no probability attached and no calibrated score-to-return mapping. Building it
  means inventing the input. Ziemba & MacLean also note growth *and* security both fall
  beyond full Kelly, and **2x Kelly drives the growth rate to zero** - the overbetting
  penalty is asymmetric and severe.
- **Quoting the "half Kelly keeps ~75% of growth at ~50% of volatility" figure.** It is
  everywhere in secondary sources; I could not find it in the primary chapter and did
  not use it. `research/README.md`: a blog summarising a paper is a pointer, not a
  citation.
- **Justifying equal weight by its historical outperformance.** Over half of the S&P 500
  Equal Weight excess return is a size tilt, and this screener already runs an explicit
  `size` category. That would be betting the same way twice and calling it two things.
  The note justifies equal weight on **estimation-error and explainability** grounds
  instead.
- **Raising `num_stocks` to 30-50 to satisfy Statman/Campbell.** Those thresholds assume
  *randomly* selected portfolios; a pre-screened large-cap 25 carries less residual
  idiosyncratic risk, and the breadth argument cuts the other way. The defensible move is
  to *state* where 25 sits against the literature, not to move it on a number derived
  from a different portfolio.
- **Building a "how much" surface that prints a target weight.** That is the Model
  Portfolio again. The note's product recommendation is to answer the question with
  **inputs** - the reader's own concentration, name count against the literature,
  volatility percentiles, and published external caps - and never a per-stock weight.

### Next

1. **Wednesday (synthesis): take section 9 of the note.** Flip `portfolio.weighting` to
   `'equal'` with the changelog entry citing DeMiguel et al. and Chopra & Ziemba, and
   record that `max_position_pct` is currently **inert** so a later session does not
   mistake it for an active control. Do not delete it - it becomes live if `num_stocks`
   falls. Measured portfolio effect is ~0.5 pp per position; the gain is that the tool
   would do what it says.
2. **Then the Concentration block on My Holdings** - name count vs the 30/40/50/63
   thresholds, sector spread, per-holding volatility percentile. Zero payload cost; every
   field is already in `stock_detail`. The 2026-09-14 constraints still bind: no cost
   basis, no share count, no P&L, and no target weight.
3. **Priority 3, backtest v2 - now 27 days old** and still the top unblocked north-star
   item. It has now been deferred by three consecutive sessions, each for a defensible
   reason. That is the pattern the 2026-09-04 retrospective added the roadmap-age line to
   make visible.
4. Still open from 2026-09-11, **sixth session running**: decide the `currentPrice`
   fallback in `factor_engine.py` - make it real at all six sites or delete it and the
   comment at ~line 683 that promises it.
