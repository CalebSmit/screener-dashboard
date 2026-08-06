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
`backtest.py` documents its own survivorship and look-ahead biases. Now that
the system validates its own methodology changes, a biased backtest doesn't
just mislead a reader - it steers the self-improvement loop toward whatever the
bias favours. See `.claude/plan/backtest-v2.md`; start by *quantifying* how
much those two biases are worth here before building anything, since that
determines how urgent the rest is.

Until then, live IC from the data loop is more trustworthy than any backtest
number, because it is genuinely out-of-sample.
