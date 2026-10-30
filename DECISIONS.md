# Settled decisions and the reasoning behind them

**This is the archive `CLAUDE.md` points at, not a second place to look for
what to do next.** It holds the full reasoning for decisions that are *settled*:
why each was made, what failure it prevents, and what it measured. The
**constraint** each one imposes lives in `CLAUDE.md`'s "Settled" table, next to
the test module that enforces it, because that is the file every session reads
before it starts.

Read an entry here when you are about to undo, weaken or "simplify" the
corresponding row. Every one of them exists because the failure it prevents
already shipped to the live public site.

Created 2026-10-02 by the retrospective. `CLAUDE.md`'s "Current priorities"
section had reached **551 lines - 52% of the file** - and most of it was
narrative about work already finished, sitting in front of the handful of items
that were actually open. The 2026-09-18 retrospective asked the next one to look
for exactly that. Nothing was deleted: everything below is the section verbatim
as it stood at commit `501c254`, with only this header added.

Later retrospectives should keep doing this. A closed item's *reasoning* belongs
here; its *constraint* belongs in `CLAUDE.md`; its *discovery* belongs in
`NIGHTLY_LOG.md`. When those three start saying the same thing three times, the
first thing to go is the copy in `CLAUDE.md`.

---

## The record as of 2026-10-02

**-1. Sessions not starting was the dominant failure mode. It is fixed - keep
it fixed.** Measured by the 2026-08-21 retrospective: of 11 scheduled code-loop
slots from 08-06 to 08-20, **2 produced anything and 5 never fired at all.**
Measured again by the 2026-09-04 retrospective: of the 9 scheduled slots from
08-24 to 09-03, **9 ran and 9 produced work.** The causes are gone, not dormant,
and the machinery below is why. Do not undo any of it; the full history is in
`NIGHTLY_LOG.md` 2026-08-21 and 2026-09-04.

- **The scheduled-task definitions are in version control** -
  `scripts/register-tasks.ps1` (2026-08-21), which registers both tasks
  idempotently with an at-logon catch-up trigger.
- **Something outside the machine watches whether the loop is running** -
  `scripts/check_loop_health.py` + `.github/workflows/loop-watchdog.yml`
  (2026-08-27), 43 tests in `tests/test_loop_watchdog.py`. It runs on GitHub
  Actions specifically because a PC that is off, asleep or logged out cannot
  silence it - the previous watchdog lived inside the thing it was watching.

  The heartbeat is the `brief:` commit each loop pushes to `main` from a
  `finally` block, so it lands whether the run succeeded or failed. That
  answers "did the task fire", not "did it do anything useful"; the brief
  itself covers the second.

  **Do not make it alarm faster.** Two *consecutive* missed weekdays, weekends
  excluded, and a day is not judged until its deadline (12:00 data / 16:00
  code) has passed - late enough for the at-logon catch-up to have had its
  chance. Replayed against the real 08-17..08-20 outage it alarms on **08-18**,
  two days in rather than six. A watchdog that cries wolf gets muted, and a
  muted watchdog is worse than none because it still looks like coverage.

  **The heartbeat must never carry anything but the heartbeat**, fixed
  2026-09-04. Both runners published it with `git push origin HEAD:main` from
  `finally`. On the happy path HEAD is `main` and that is correct, which is why
  it ran unnoticed for a month. On a ship-gate failure HEAD is the nightly
  branch holding the work the gates just refused, and that command
  fast-forwards `origin/main` onto all of it - publishing to the live site,
  after the gates said no, from the one block that always runs.
  `scripts/publish-brief.ps1` now builds a single-file commit on top of
  `origin/main` with plumbing and pushes the commit object, so
  `MORNING_BRIEF.md` is the only path it can touch - by construction, not by
  convention. **Do not go back to pushing a local ref**, and keep the two
  runners on the one shared function so the safe one cannot drift.
  `tests/test_brief_publish_safety.py`, 14 tests; the first drives the old
  command in a sandbox and watches the refused work land on `main`.

**A weekly usage ceiling exists** and silently killed the 08-14 session with a
429. There is no per-session dollar cost to optimise - the owner runs Claude
Max, so this is included subscription usage (owner correction, 2026-08-21; do
not reintroduce the "~$6/session" figure). What follows is "do not let one loop
exhaust the week": the 06:00 code loop is the only thing consuming that quota,
so a data run is always affordable and a code session is the scarce resource.
That is an argument for sessions that do one thing well rather than many
shallow things.

**The catch-up trigger had a defect of its own, fixed 2026-08-29.** Both tasks
carried an at-logon trigger with the *same* `PT3M` delay, so on the first logon
of the day they started in the same second and raced for `.git/index.lock`. On
2026-08-29 the data loop's `git checkout main` hit the index while the code
loop's `Restore-Artifacts` was running `git status`; git exits 128, the data
loop treated that as fatal, and the run stopped before the screener ran at all.
The mechanism built to stop days being lost had become a way to lose one.

Fixed by `scripts/repo-lock.ps1`, a shared lock both loops take before their
first git command and release after `Publish-Brief`. The loser **waits** rather
than dying - data runs take 11.8-13.6 min and code sessions 16.9-25.9 min
against 3h/4h execution limits, so waiting is nearly free and giving up costs
the day. Logon delays are now staggered (data `PT3M`, code `PT20M`) so the
order is deterministic: evidence first, then the session that reads it.
`scripts/add-catchup-trigger.ps1` wrote its own `PT3M`-for-both triggers and
would have silently restored the collision, so it now delegates to
`register-tasks.ps1`. 15 tests in `tests/test_loop_mutual_exclusion.py`; 853
tests overall, up from 825.

**Do not give the two loops the same logon delay again**, and do not make the
loser exit instead of wait. Each script's own single-instance lock stops it
racing *itself*; only the shared lock stops the two racing *each other*.

The stagger went live and was verified the same evening: both tasks `Ready`,
`Get-ScheduledTask` confirming `PT3M` and `PT20M`. The session that found it had
left the command for the owner to run; rule 11 exists because of that, and
carries the reasoning. `NIGHTLY_LOG.md` 2026-08-29 (evening) has the rest.

Workspace trust (resolved 2026-08-13) regresses as: `python --version` works
but everything else is denied. Fix in `scripts/fix-trust.ps1`; the runner now
fails fast with instructions.

**Nightly branches now self-clean, fixed 2026-09-01.** `nightly-screener.ps1`
deletes its own working branch after a successful merge, but the delete's
exit code was piped to `Out-Null` and never checked, so a rare failure - a
transient git lock, an interrupted run - left the branch behind with no log
line. `nightly/2026-08-10` and `nightly/2026-08-27` were found this way,
weeks apart, both cleanly merged and never noticed until someone ran
`git branch`. Fixed two ways: the final delete now logs a `WARN` on failure
instead of swallowing it, and every run sweeps any local `nightly/*` branch
already merged into `main` at startup, so one missed delete self-heals on the
next run rather than accumulating silently. Rule 11 territory - self-healing,
not something to notice by hand.

**The same sweep now runs against `origin`, added 2026-09-03.** The 09-01 fix
was local-only, so `origin` still gained one dead ref per session: 11 by
2026-09-03 (`nightly/2026-08-10` .. `nightly/2026-09-02`), all fully merged.
The 09-02 evening session spotted them and wrote it down as a sweep for "a
future session" - manual work, which is what rule 11 forbids. It now runs in
`nightly-screener.ps1` right after the run's branch is created.

**Do not remove the `--merged origin/main` filter, and do not make a failed
delete fatal.** That filter is the only thing standing between an unattended
`git push origin --delete` loop and branches whose work never reached `main`;
its semantics are asserted against a real origin+clone sandbox in
`tests/test_branch_sweep.py` (10 tests) rather than assumed. `$Branch` is
excluded so a same-day rerun cannot delete the branch it is about to push.
A dead ref on origin is untidy, not dangerous - it is swept again next run.

**0. DONE 2026-08-24 - do not weaken these.** All five steps shipped together,
plus a sixth defect found while fixing them. See `METHODOLOGY_CHANGELOG.md`
2026-08-24 and `tests/test_evidence_integrity.py` (30 tests; 24 of them fail
against the pre-fix code).

| # | Was | Now |
|---|---|---|
| 1 | A date was processed once at 7 days old and never revisited, so `fwd_return_1m` stayed `NaN` forever - and `optimization_horizon` is `'1m'` | Eligibility tracked per `(run_date, horizon)`; a date is reprocessed as it ages |
| 2 | Every snapshot file processed, so a day with 13 runs appended the same rows 13 times | One snapshot per run date; `_normalize_performance_history()` on every read and write |
| 3 | `t = IR * sqrt(raw row count)` | `_effective_observations()` - non-overlapping windows only; every gate reads it |
| 4 | Weekend run dates counted as separate observations from the adjacent Friday | Excluded at generation and at IC time |
| 5 | Nothing ever called `compute_live_ic()` | `record_run_snapshot()` calls it for all three horizons |
| 6 | Price cache keyed on the *current* date, so each revisited snapshot meant a fresh full-universe yfinance download | Fetch window bounded by the horizon being measured |

Measured effect: `performance_history.csv` 20,057 rows -> 5,528 (60% were
duplicates); `live_ic_history.csv` **3 rows -> 23**, newest 2026-02-22 ->
2026-08-14; observations at the `1m` optimization horizon **0 -> 6 raw, 2
effective**; `n_tickers` per IC row 1,006-6,539 -> 499-511.
`scripts/repair_evidence_base.py` performs the one-time repair and is
idempotent.

**What has NOT changed, and must not be quietly assumed away:**

- **`allow_auto_apply` stays `false`.** Condition (a) in the `config.yaml`
  comment (effective-observation counting) is now met. Condition (b) - a
  history with substantially more independent observations than the gate asks
  for - is **not**: there are **2**. Rule 4 stands.
- **The engine still cannot propose, and that is correct.** 2 effective
  1-month observations against a gate of 8.
- **Accrual is genuinely slow.** Independent 1-month observations arrive about
  one a month, so the 8-observation gate is roughly **six more months** of
  daily running. The old behaviour would have reached "8 observations" much
  sooner and been wrong. Do not engineer around this; say it plainly.

The `_effective_observations()` guard is now the load-bearing safety mechanism.
Against the pre-fix code, `propose_weight_changes()` returns `proposal_ready`
on eleven IC rows that are two independent observations - that is a failing
test now, not a hypothetical. Weakening it needs its own changelog entry and a
better argument than "it wasn't firing."

**0.5 / 0.7. DONE - do not weaken these.** `scripts/check_run_health.py`
(2026-08-10) discards a run before publishing on: missing fetch evidence, price
coverage <90%, analyst-target coverage <50%, or category dispersion >20% below
the trailing median. 14 tests in `tests/test_run_health.py`. The stale-cache
root cause behind those degraded runs was fixed 2026-08-13
(`factor_scores_cache_max_age_days()`, 21 tests in
`tests/test_cache_freshness.py`, changelog 2026-08-13) and **confirmed working**
by the 08-14 and 08-21 runs: live fetch, 100% price coverage, HEALTH: PASS.
Detail in `NIGHTLY_LOG.md` 2026-08-10 (evening) and 2026-08-13. Each threshold
exists because that exact failure shipped to the public site.

**0.6. Do not record an improvement-engine snapshot when the run did not
fetch.** Found 2026-08-11: a warm-started run still writes a snapshot, so a day
with two cached runs produced three "observations" of one real data point, all
byte-identical in composite. The engine gates on an observation *count*, so
duplicates directly inflate its confidence - the same evidence-inflation
failure `research/2026-08-10-ic-evidence-independence.md` identified via
overlapping return windows, arriving by another route. Either skip the snapshot
on a warm-start, or deduplicate on `(run_date, content hash)` before the engine
reads them.

1. **Data loop health - corrected 2026-09-01, ~10-25% was stale.** That figure
   dated to the Yahoo rate-limit era around launch. Checked directly against
   the last 15 data-run logs (2026-08-10 through 2026-09-01): **every one
   reports 0 fetch failures.** Whatever combination of the cache-freshness fix
   (2026-08-13), the price-series-integrity guard (2026-08-26) and simple
   improved reliability on Yahoo's side resolved this, nobody had gone back to
   check. Left here as a reminder to re-verify periodically rather than assume
   either the old bad number or a permanent fix - and if a future run shows
   real fetch failures again, this is where to update the record, not silently
   let it drift stale in the other direction.

   **The related fabrication defect - DONE 2026-09-01, both entry points now.**
   Found 2026-08-06: a failed fetch made the pipeline silently substitute
   *synthetic* "sector-realistic sample values" and emit output indistinguishable
   from a real run. `run_screener.py` was fixed at source 2026-08-11 (refuses
   unless `--allow-synthetic` is passed explicitly). What had **not** been
   checked until 2026-09-01: `factor_engine.py` has its own independent `main()`
   (`python factor_engine.py`, used by nothing in the scheduled loops but
   reachable by anyone testing directly) that still had the exact pre-08-11
   behaviour - unconditional fabrication, no flag, no refusal. Fixed the same
   way: refuses unconditionally and points at the supported
   `run_screener.py --allow-synthetic` path for sample-data testing.
   `tests/test_no_synthetic_by_default.py`, 10 tests (4 new), the new ones
   confirmed to fail against the pre-fix file.

   **A permanent false "High severity" alarm in the same log - DONE
   2026-09-01.** `validation/data_quality_log.csv` had flagged four bank-only
   metrics (`pb_ratio`, `roe`, `roa`, `equity_ratio`) as "High severity -
   missing >50% threshold" on every single run since launch: 88.4%/88.2%
   missing, unchanging. Not a defect - only ~58 of 502 S&P 500 stocks are
   banks, and these metrics are correctly absent from every non-bank by
   design. The drift check scored the missing-% against the whole universe
   instead of the population a metric actually applies to; the coverage
   filter a few hundred lines away in the same function already scoped this
   correctly and the drift check simply never matched it. A permanent,
   always-firing "High severity" alert is the same failure shape the
   watchdog's own design explicitly guards against (rule 7) - it trains a
   reader to stop looking, which is exactly when a real drift goes unnoticed.
   Extracted into `_metric_missing_pct()` in `run_screener.py` and fixed to
   scope by `_BANK_ONLY_METRICS` / `_NONBANK_ONLY_METRICS`, mirroring the
   existing coverage-filter pattern. `tests/test_metric_drift_scoping.py`,
   7 tests, including one pinning the exact 88.4% figure the fix corrects.
1.5. **DONE 2026-08-26 - and the 08-25 diagnosis was backwards.** Found
   2026-08-25 by the movers panel; root-caused and fixed 2026-08-26. Changelog
   2026-08-26; `tests/test_price_series_integrity.py`, 21 tests.

   The cause was not a transient failure. Yahoo's 13-month series for MNST
   **alternates between pre- and post-split prices** across its 2026-08-11 2:1
   split (94.46 / 47.08 / 90.36 on consecutive days), and `auto_adjust=False`
   returns byte-identical numbers, so no adjustment was ever applied. The
   pipeline divided an unadjusted July close (93.49) by an adjusted 2025 close
   (62.30) and got `return_12_1 = +0.50`, the 97th percentile, against a true
   split-adjusted **-0.25**, the 3rd percentile.

   **So 97.1 was the artifact and 2.9 was correct** - the reverse of what
   `NIGHTLY_LOG.md` 2026-08-25 and this entry originally said. MNST was live on
   the public site at momentum 71.5 / rank 360, roughly 110 ranks too high.

   Fixed at source: `factor_engine.check_price_series_integrity()` refuses a
   series that mixes two split scales and withholds the eight metrics derived
   from it, which the existing `has_data` renormalisation already handles.
   Verified against all 17 S&P 500 split events of the prior 13 months - one
   true positive, zero false positives.

   **Two things not to undo.** The 25% arming floor is measured (p99.9 of
   |daily return| is 17.2% over 137,313 ticker-days); below it a "split ratio"
   cannot be told apart from an ordinary down day, which is what keeps the
   small spin-off ratios (SPGI 1.057, HON 1.061) from flagging everything. And
   `check_run_health`'s `MIN_CATEGORY_COVERAGE = 0.90` bounds the blast radius:
   withholding one name in 502 is the mechanism working, withholding the
   universe is a feed change that must not publish.

   **The coherence finding worth carrying forward:** the eight categories are
   not eight independent bets. Momentum and risk are **23% of composite weight
   and 100% derived from one `Ticker.history()` call** - momentum's only
   non-price metric, `proximity_52w_high`, carries weight 0, so a rejected
   series costs a stock two entire categories.

   **FCX, the other case cited on 08-25, was never a bug.** Its growth score
   moved 68.3 -> 42.5 -> 68.3 because on 08-24 `forward_eps_growth` and
   `peg_ratio` were genuinely NaN and growth correctly renormalised over the
   remaining three metrics. Do not go looking for a defect there. What it does
   expose is a *product* gap: the movers panel could not distinguish "moved on
   new information" from "moved because two inputs went missing".

   **That gap is CLOSED 2026-09-17 - do not weaken these.** `history.py` carries
   per-ticker metric availability and emits `ch: [lost, gained]`;
   `stock_summary._sentence_input_churn` states it as a caveat on the drilldown
   and on every holdings row. Fires for 27 of 502 on the 2026-09-17 run.
   Changelog 2026-09-17; `tests/test_input_churn.py`, 58 tests, 52 of which fail
   against the pre-change code.

   - **Arms at >= 2 metrics.** One changed metric moves the median rank by 7
     against a baseline of 6 - noise - and firing on it would mark 4.93% of
     transitions instead of 1.22%.
   - **Worded as a caveat, never as deterioration.** Churn leaves 52.4% of names
     worse off against a 44.6% base rate, so it scatters ranks rather than
     pushing them down. A test asserts the sentence reads identically whether the
     stock rose or fell.
   - **Compares availability *sets* over the columns both runs carry**, not a
     metric count. A net count of zero hides one metric dropping out as another
     returns, and counting a column the older schema never had would flag the
     whole universe the day a metric is added.

2. **Give the dashboard a time dimension.** **DONE 2026-08-25** - shipped as
   `history.py` plus three surfaces: a "What Changed" movers panel, a sortable
   Δ column in the universe table, and a per-stock "Rank History" block.
   Changelog 2026-08-25; `tests/test_history.py`, 31 tests.

   **Do not weaken the comparability gate.** Runs enter the history only if
   their ranking correlates with the last accepted run at Spearman >= 0.50.
   `2026-07-28` is a degraded run in `improvement/snapshots/` that correlates
   with its neighbours at 0.016 and -0.020; ungated, it reports 82% of the
   universe as material movers. A regression test fails if it rejoins the
   series. Note also that reusing `check_run_health`'s dispersion rule here was
   tried and **excluded 16 of 20 real runs** - it is the right gate at publish
   time and the wrong one for comparing runs. Reasons in the changelog.

   **What is still missing:** per-category trend lines over the full history
   (only the two comparison baselines carry category deltas today, to keep the
   payload at +8%), and time-series valuation percentiles (north-star gap 3),
   which need the same spine and are now cheap to build.
3. **Backtest v2** - the current one has survivorship bias and holds
   fundamentals constant (look-ahead). It cannot honestly validate a
   methodology change, and now that the system validates *itself*, that bias
   steers the learning loop. `plan/backtest-v2.md`.

   **Step 1 is DONE for the survivorship half, 2026-09-24 - and the answer
   settles the plan's own decision rule against v1.** The plan said 0.5%/yr
   means "usable with a caveat" and 4%/yr means "every existing validation
   claim needs retracting". Measured: **4.3%/yr**. Of the 2020-01-31 universe,
   **98 of 505 names (19.4%) are absent from every v1 backtest**, and across
   the window the index held **643 distinct names against the 503** v1 ever
   sees. Shipped: `universe_history.py`, an 80-month committed cache, and
   `research/measurements/2026-09-24-survivorship-gap.py`, which reproduces
   every figure. `tests/test_universe_history.py`, 59 tests.

   **Three things not to undo.**
   - **Renames are separated from exits by SEC CIK, not by ticker or name.**
     ANTM->ELV, FB->META, BK->BNY change symbol *and* company name together,
     so only the registrant id links them. 18 of the 116 gross "deletions"
     were renames; reporting the gross 23.0% as survivorship overstates it.
   - **`validate_membership` refuses a universe outside 495-515.** A short
     parse looks exactly like a real index contraction. The band is measured
     (501-505 across all 80 months), and the failure it guards is concrete:
     the same page carried a 269-row "selected changes" table.
   - **`backtest.py` is deliberately NOT wired to it.** Only **40%** of exited
     names have downloadable prices, and the missing 60% are the acquisitions
     and buyouts - the terminal outcomes survivorship bias is actually made
     of. A point-in-time universe that restores names but not their returns is
     differently wrong, not fixed.

   **The procurement half is DONE 2026-09-30, and the answer is that it was
   never the blocker.** `research/2026-09-30-delisted-price-source-cost.md`;
   `research/measurements/2026-09-30-delisted-price-requirement.py` and
   `-exit-reasons.py` reproduce every number. The item had been deferred by
   **eight of the ten sessions** before that date, every time for defensible
   smaller work, because it was a decision rather than code and so always lost to
   work that was.

   **Priced: $19 to download, $199/yr to keep** (Sharadar Prices, 10-year
   history). CRSP is the only surveyed source carrying a real delisting return
   and has no list price - institutional contracts only. **Decision: do not buy
   yet**, because this plan's own rule is that a v2 fixing survivorship but not
   look-ahead is not decision-grade, and the licence requires deleting the data
   30 days after the subscription ends - so buying before a consumer exists pays
   for a panel nothing reads.

   **Three measurements that supersede what this section said before.** The
   2026-09-24 figures came from a 30-name sample of one month's exits; these are
   a census of all 123.

   | | Was | Now |
   |---|---|---|
   | Rename/exit split | oldest month only | all 80 months by CIK: **142 absent = 19 renames + 123 exits + 0 unresolved** |
   | Free price coverage | "only 40% of exited names" | **54% of names, 57.2% of name-months** |
   | Survivorship, as a share of the panel | not computed | **11.4%**; **4.89%** would remain after a free-data v2 |

   **The free gap is the acquisitions, and that is why a cheap feed suffices.**
   Of 1,938 missing name-months, **69% belong to the 35 acquired names and free
   data supplies none of them**, while market-cap demotions - 62% of exits, which
   keep trading - are **90% covered free**. Per S&P DJI's own methodology an
   acquisition is removed *"at the close of the last day of trading or expiration
   of a tender offer"*, so the last traded close **is** the index's exit value,
   not an estimate of it. Performance delistings, the only case Shumway's
   -30%/-55% corrections apply to, are **3 of 116** exits.

   **Do not test price availability by row count.** INFO, LB and SBNY each clear
   200 rows and cover **none** of their membership months - `INFO`'s series now
   begins in **October 2024**, two years after IHS Markit was absorbed. Wiring on
   row count would insert a different company's prices under a former
   constituent's symbol. An availability check must assert coverage of the span
   the caller will read.

   **Step 1 is now DONE on both halves, 2026-10-01, and the answer reverses the
   plan's own sequencing.** `research/2026-10-01-lookahead-bias-size.md`;
   `research/measurements/2026-10-01-lookahead-price-component.py` reproduces every
   figure from the committed `dashboard_data.js`; `lookahead.py` plus 43 tests in
   `tests/test_lookahead.py`.

   **Look-ahead is >= 63.2% of the name-month panel against survivorship's 11.4%
   - 5.5x bigger on the same unit.** 30.3% of name-months move two or more
   deciles and only **59.5%** of v1's top decile belongs there. It is a *lower
   bound*: 49.0 points of composite weight stay frozen in both arms of the
   experiment. Like survivorship it decays monotonically toward the present
   (73.9% in 2020-01 to 2.0% in 2026-09, age-vs-error rank correlation 0.86),
   which is the signature that says the measurement is structural and not noise.

   **The decomposition is the usable part, and `lookahead.weight_buckets()`
   derives it from `config.yaml` so it cannot go stale:** 16.9% of composite
   weight is honestly recomputed per rebalance, **28.0% is held constant although
   one month-end price restates it exactly**, 6.1% is held constant although it is
   a pure function of a price history, and 49.0% genuinely needs point-in-time
   filings and estimates. **So 34.1 of the 83.1 held-constant points are free to
   fix** - no vendor, no licence, no permission.

   **Three things not to undo.**
   - **`lookahead.py` is a diagnostic and must not be wired in.** Restating a
     valuation ratio at a historical price while leaving its fundamental at
     today's value removes some look-ahead and leaves the rest, which is the
     half-fixed backtest `plan/backtest-v2.md` forbids. Four tests assert that
     `backtest.py`, `run_screener.py`, `factor_engine.py` and
     `generate_dashboard.py` do not import it.
   - **`price_target_upside` is rebuilt from `pt_mean`, never by inverting the
     metric.** The metric is clamped to `metric_clamps`, so dividing it by the
     price ratio invents an analyst target for every name on the bound. A test
     drives the clamped case and pins the honest answer against the inverted one.
   - **Enterprise value is restated as `ev + mc*(r-1)`, not `ev*r`.** Debt and
     cash do not move with the share price. Scaling EV wholesale would make a
     leveraged name a third cheaper than it was, and a test pins both numbers.

   **Two defects in `backtest.py` found while measuring, recorded and
   deliberately not patched** (same reason: no half-fixed backtest):
   - Its docstring claims *"Only Momentum and Risk metrics are recomputed from
     trailing prices"*. `simulate_monthly_scores`'s `dynamic_cols` holds **four**
     names and those two categories carry **six** weighted metrics -
     `jensens_alpha` and `max_drawdown_1y` are frozen.
     `test_recomputed_matches_backtests_dynamic_cols` parses that literal, so the
     classification fails loudly if the list moves.
   - **v1 backtests a weighting the site does not publish.** `run_screener.py`
     calls `adjust_momentum_weight()` between the category scores and the
     composite; `backtest.py` does not. On the 2026-10-01 run it moves momentum
     **13 -> 14.95** and valuation **22 -> 20.05**. Adding that one call to the
     reconstruction closed its gap against the published ranking from a median of
     8 rank places to 2 - which is how arm A was shown faithful to the thing being
     measured. v2 must apply it **per rebalance month from that month's regime**;
     reading the current run's regime is a third look-ahead vector, inside the
     weights rather than the metrics.

   **Next on this item: step 3 (point-in-time fundamentals), which now outranks
   step 2.** Within it, the 34.1 free points first. Then size the 49.0pp
   fundamentals half with SEC EDGAR's XBRL `companyconcept` endpoint, which is
   free and carries a `filed` date per fact; the 9.0pp of analyst-estimate metrics
   has no free retrospective source and may only be reportable as permanently
   unmeasurable. **Do not buy the delisted-price feed before that** - spending
   $199/yr to cut an 11.4% bias while a >= 63.2% one is untouched buys nothing a
   reader can use.

   **Nothing needed retracting** in `METHODOLOGY_CHANGELOG.md` - checked again
   2026-09-30: no entry cites a backtest under **Evidence**, because the
   2026-08-11 bench rule landed first.
4. **DONE 2026-09-08 - the AI chat is gone, replaced by deterministic per-stock
   summaries.** Owner directive 2026-08-10, open 29 days. Changelog 2026-09-08;
   `stock_summary.py`; `tests/test_stock_summary.py` (77) and
   `tests/test_ai_chat_removed.py` (75, of which 58 fail against the pre-change
   generator).

   Removed: the chat panel, the API-key dialog, the model picker, 27 JS
   functions, three keyframe blocks and the chat stylesheet - 891 lines, and the
   `config_traps` payload key whose only consumer was the chat's system prompt.
   Added: a **"Why it ranks here"** block opening every stock's drilldown, built
   at run time from `contrib`, `cat_scores`, `pct`, `raw`, `peers`, `flags`, the
   analyst targets and the history spine.

   **Three things not to undo.** The summary is built **at build time and baked
   into the payload** - moving it into the browser would give up the diffability
   and per-reader identity that were the whole reason the chat went. Advice
   language is blocked by `BANNED_TERMS` / `advice_terms_in()` and checked
   against all 502 live summaries; "explains why it ranks there, never whether
   to buy" is a constraint with a test, not a style note. And metric percentiles
   are labelled **sector**-relative because that is what they are.

   **Cost, measured:** +101 KB gzipped for the summaries, -11 KB from deleting
   the chat, net **+90 KB (+8%)** on a 1,078 KB wire payload. Scope was widened
   from the plan's "top ~25" to all 502 on that measurement; if payload weight
   ever binds, drop the `peers` and `flags` sentences before cutting coverage.

   **Still open from that directive:** the **run-level** overview - one or two
   sentences on what moved across the whole run. Most of it already exists as the
   What Changed movers panel (2026-08-25), so the remaining gap is narrow.
5. **Sell-side workflow - the list shipped 2026-09-15, the rule did not.**
   Owner directive via the north star, open 42 days. Changelog 2026-09-15;
   `tests/test_holdings_panel.py` (61 tests, 60 of which fail against the
   pre-change generator), plus `change_driver` in `stock_summary.py`.

   **My Holdings** is a `localStorage` list holding **tickers and nothing
   else**, rendering every saved name each run with rank, composite, an
   eight-category score-and-delta strip, and the baked review sentences. Zero
   payload cost - it is a view over fields `stock_detail` already carried.

   **Three properties are research constraints with tests, not styling.** Do
   not "tidy" them; the sources are in
   `research/2026-09-14-sell-discipline-and-hold-bands.md`:
   - **Every saved name renders, every time.** Akepanidtaworn et al. (2023,
     *JF* 78(6)) trace an **-80 bp/year** institutional selling deficit to a
     restricted consideration set: extremes on prior returns are sold at rates
     **>50% higher** than middling positions. A move-ranked review queue is
     that heuristic automated, which is what the north-star plan originally
     specified and what this does not do.
   - **Ordered by rank, never by size of move.** Same source. The rank change
     is shown for context; it is not the sort key and not a filter.
   - **No cost basis, share count or P&L**, in the code or in storage. Odean
     (1998): PGR/PLR **1.50, t = -32**; winners sold beat losers held by
     **+3.41%** over the following year. A key hand-edited to hold a position
     dict is read for its ticker and written back clean.

   **2026-10-16: a saved name outside this run is kept, not erased.** The
   first property had a hole from the day it shipped. `loadHoldings()` kept a
   stored ticker only if this run had scored it (`!D.stock_detail[t]`), and the
   next write - any add or remove - saved the filtered list, so a name that left
   the S&P 500, or failed one night's fetch, was gone from the reader's list for
   good, without a word. Measured that day against `improvement/snapshots/`:
   **30 tickers** ranked at some point since February are not in the current
   run (WBD, CTVA and PSKY last on 2026-10-08). A holding that leaves the index
   is exactly the one a reader most needs to see again, so this was the
   restricted-consideration-set failure the property exists to prevent, done
   by the storage layer instead of a sort.

   Now the read keeps any string matching `/^[A-Z0-9.\-]{1,10}$/` ("." stored
   as "-", the payload's form) and still drops everything else (dicts, numbers,
   free text, over-long strings), so the no-cost-basis property is unchanged:
   storage holds tickers and nothing else. A name this run did not score renders
   last, as a fact and not a verdict - "<T> is not in this run's ranking: it has
   left the S&P 500 or its data could not be fetched." - with its last rank and
   date where the payload's history still carries a series for it, and the same
   remove button. The fit and concentration lines describe scores, so they read
   the scored names only; the list count includes every name. Adding a ticker
   still requires it to be in the run (`addHolding`), because a name typed in
   that was never scored has nothing to show. Tests:
   `tests/test_holdings_panel.py::test_round_trips_through_storage_and_survives_junk`,
   `::test_a_name_this_run_did_not_score_still_renders_and_survives_edits`, and
   the browser check in `tests/test_dashboard_interaction.py`.

   **What is still open: the hold band - settled 2026-09-16 as "not yet", with
   a date.** The screener has one test (`portfolio.num_stocks: 25`) where
   Novy-Marx & Velikov (2016), MSCI and S&P DJI all use a wider, different test
   for continued membership.

   **The 2026-09-15 reasoning quoted here was wrong on both numbers and is
   corrected.** It said a 25/50 band "fired zero times" and that §9 wanted "60+
   comparable runs, there are 32". The zero came from walking **one path**
   through 18 runs; over all comparable pairs a 2x band fires at 1.65-4.50% of
   holding-looks. And run count is the **wrong unit**: pairs from a daily series
   overlap almost completely, so 34 runs are **2 independent monthly looks**.
   Changelog 2026-09-16; §8 of the research note;
   `research/measurements/2026-09-16-hold-band-and-input-churn.py` reproduces
   every number and prints overlapping vs disjoint estimators side by side.

   **What is established:** the strict top-25 rule wastes **31-47%** of the
   trades it implies, at every cadence measured (9/7/3 disjoint triples). A band
   is warranted. Its **width is not determinable** - at 1.4x the three cadences
   report 0.0%, 31.2% and 5.9% wasted.

   **The pre-registered rule, which binds:** do not commit to a width until
   there are **>= 8 disjoint observation windows at the review cadence the band
   will govern**. Today **2 monthly**; roughly **2027-04** at one a month. Then
   take the **narrowest** band clearing half the strict rule's waste and NMV's
   50% turnover bound. **Do not pick 50 because MSCI doubles** - that borrows a
   parameter from a 500-name quarterly index. And **do not reuse the movers
   panel's threshold** (measured: it fires for a top-25 name **0.15%** of the
   time).

   **The cadence half shipped 2026-09-17, and it was the other open item.**
   `config.yaml` had recorded a quarterly rebalance cadence since launch as a
   bare *comment*, which the generator could not read, so the site regenerated
   every weekday and stated no cadence anywhere. `portfolio.review_cadence` is
   now a key, surfaced as `D.cadence` and stated on the holdings panel, the What
   Changed footnote and the holdings footnote. Acting on the strict top-25 rule
   at every run implies **121.8%** monthly one-sided turnover against **24.0%**
   at monthly review, where NMV find few anomalies survive above ~50%.
   Changelog 2026-09-17; `tests/test_review_cadence.py`, 40 tests, 37 failing
   against the pre-change generator.

   **It is a sentence, not a lock, and must stay one.** The tool does not know
   what a reader is doing. Do not gate, hide or delay a number behind the
   cadence - stating it is decision support, enforcing it would not be, and the
   data loop running daily is what accrues the evidence base. It is read from
   the **run's own** config snapshot, not the working tree, so a republished old
   run states what it was configured for; `configured: false` marks the fallback.

   **The generalisable lesson:** rank-migration statistics need the same
   non-overlapping treatment as ICs. This is the third place the project has hit
   the independence trap - see `research/README.md` Standards.

   **One measurement to carry forward:** the largest one-month category move is
   Risk 34%, Revisions 29%, Momentum 26% and **Quality 0.2% - one stock in
   500**. A deterioration trigger keyed to the fundamentals categories would
   essentially never fire at monthly cadence.
6. **Investor profile selector** - `plan/investor-profiles.md`.
   Reconcile with `presets.py` first. Note `contrib` is Balanced-only.
7. **Investment-club readiness** - can a student open this on a phone and
   understand what they're looking at?
8. **Test isolation** - remove the need for the `conftest.py` guard.

---

## 0.8a. A claim with no registered check fails the build (settled 2026-10-07)

**The constraint:** every sentence on the site that states *how a number is
computed* has an entry in `claims.py` naming what it asserts, the code that makes
it true, and the test that checks it. Adding a `_sentence_*` function to
`stock_summary.py` fails the suite until it is registered.
`tests/test_claims_register.py` (24 tests), and in `scripts/check_published_claims.py`
so the 02:00 data loop cannot republish a false claim.

**Why, and read this before you weaken it.** On 2026-10-06 the first sentence of
all 502 drilldowns read *"Its composite of 73.7 is a percentile: it scores above
74% of the universe"* - for the stock ranked **1st of 502**. `Composite` has been
the cardinal weighted average since Phase 13 (F1) and `Composite_Pct` is the
percentile. Measured: wrong for **493 of 502** stocks, median error **19.6
percentage points**, more than 10 points for **74.9%**, worst **31.2**.

It had been live for months, and the mechanism is the part that matters:

1. **A test asserted it.** `test_composite_is_described_as_a_universe_percentile`
   pinned the false sentence and cited `SCREENER_OVERVIEW.md` as its authority -
   while the *same page's* Limitation 8 said the opposite. The page contradicted
   itself and the test pinned the wrong half, so correcting the sentence would
   have read as a regression.
2. **Nothing enumerated the set of claims.** `check_published_claims.py` checks
   specific claims someone thought to write down. An *unchecked* claim was
   therefore invisible - there was no list it could be missing from.

The register exists for (2), and `FORBIDDEN` for (1): the four false statements
are pinned as parametrised tests, so the guard cannot decay into a no-op while the
suite stays green.

**Evidence it is not decoration.** The register caught **11 of its own 26
`checked_by` references** as test names that did not exist, the first time it ran.
A negative control before shipping confirmed every tripwire fires on input it must
reject - the pre-fix payload, all four shipped false sentences, an unregistered
builder, an invented test name, a stale `detect` pattern - and none false-positives
on the corrected text. That control is the lesson of rule 8's `1m`-horizon
tripwire: a check wired to something that cannot move is decoration.

**Two generalisations worth keeping:**

- **When you correct a published claim, grep the tests, not just the prose.** The
  plan for this work named four sites; there were six. The two it missed were a
  stale correction note in `FORENSIC_AUDIT_REPORT.md` and the test above.
- **Say what cannot be shown rather than print an equation that does not
  reproduce.** The corrected sentence says the composite is *computed from* the
  category scores and weights, not that it *equals* their weighted average,
  because the coverage discount reduces it for 3 of 502 stocks and the payload
  does not carry the coverage figure that discount reads. The caveat is recorded
  on the claim and a test requires it to name the stage that closes it (T0b).

**Do not** re-implement any of this in the generator or in JS. Principle 1 of
`plan/calculation-transparency.md`: the page shows the engine's numbers, never its
own re-derivation. A second copy is how the metric-weight defect happened.


## 0.8b. The page shows the engine's numbers, and refuses to publish if they do not add up (settled 2026-10-07)

**The constraint.** Three things, each with a test:

1. **Metric weights are resolved in exactly one place** - `factor_engine.metric_weight_profiles()`.
   Scoring consumes its tables; the dashboard publishes the same tables and each stock's choice
   (`weights.profiles`, `wp`). Do not re-implement the bank / Piotroski / renormalise rules in
   the generator or in JS. That second copy is what let the page print the generic weight for
   275 of 502 stocks. (`tests/test_calculation_reproducibility.py`)
2. **A build that cannot rebuild its own published scores from its own payload does not publish.**
   `generate_dashboard._reconcile_scores` runs `calc_trace.verify_payload` and raises
   `CalculationMismatch`; the data loop treats a failed build as a discarded run, so the live site
   keeps the last good version. Tolerances are tied to the payload's stored precision (4dp scores,
   6dp weights, 2dp contributions): **do not widen them to make a build pass** - a mismatch
   means the page and the engine disagree, which is exactly the defect.
3. **An equation is shown only if it reproduces.** `metric_lineage.RECOMPUTE` holds a function per
   metric; a stock whose inputs do not rebuild its value is listed in `inp_bad` and the page says
   so. Series-based metrics (volatility, beta, drawdown, Sharpe, Sortino) and provider ratios show
   no equation and say why. (`tests/test_metric_lineage.py`)

**Why.** The page prints its own arithmetic and presents it as the thing to check. On 2026-10-06 it
was false for 55% of stocks, and no test could see it because nothing recomputed a score from the
payload. Two independent implementations agreeing (`calc_trace` and the engine, then
`scripts/audit_stock.py`) is the evidence; one implementation agreeing with itself is not.

**Things not to undo.**
- `run_context.config_hash` carries `scoring_schema`. The scored-data cache is keyed by config
  alone, so a code change that adds columns is invisible to it; bump the number when scoring
  output changes shape.
- `peers` ships tickers only. The full rows were ~195 KB gzipped of duplication; the browser
  rebuilds them from `stock_detail`.
- Inputs, `pio`, `bn`, `asof`, `wp` and `cov` are display-only and are asserted absent from
  `raw`/`pct`.
- The claims register covers the drilldown (`drilldown.*`); a new on-page claim about how a
  number is computed needs an entry and a test, like a new `_sentence_*` does.

**Cost, measured.** Payload 1,268,733 B gzipped against 1,278,885 B before (flat): inputs +142 KB,
peer slimming -170 KB. The data run's extra publish gates add ~2 s.

**Source.** `plan/calculation-transparency.md`; `METHODOLOGY_CHANGELOG.md` 2026-10-07 (owner-run).

## 0.8c. A scoring change ships with its frontend in the same commit (settled 2026-10-07)

**Owner, verbatim:** *"there is nothing like showing the actual numbers going into any scores ...
we see if it goes up or down, but we dont see any numbers anywhere for these companys ... make sure
the nightly sessions know that if something is changed with scoring, that it also updates on there,
the frontend side of things to match the backend."*

**What was wrong.** The inputs behind every metric were already in the payload (T2, same day), but
only behind a small disclosure triangle on each row. Looking at the workings, a reader saw a value,
a percentile bar and a direction chip - and reasonably concluded the numbers were not there.

**What shipped.** Under every metric row, without a click: the stock's reported figures put through
the formula to the value scored ("$4.5B free cash flow ÷ $31.0B enterprise value = 14.4%"), its rank
among the peers it was ranked against with the sector median, and percentile x weight = points.
The equation is a template in `metric_lineage.EQUATIONS`; the page fills it with published inputs
and **never computes the result itself** - the value after "=" is the engine's.

**Why the template is arithmetic.** An *exact* template is evaluated by
`metric_lineage.evaluate_template` for every stock in the payload and must give the scored value for
99% of them (it gave 100% for all 19 exact metrics on the day it shipped). So a scoring formula that
changes in `factor_engine` without its template changing fails the suite - measured by substitution:
a stale FCF-yield template (÷ market cap instead of ÷ EV) matched 34 of 466 stocks. That is what makes
"the frontend matches the backend" a property the build checks rather than a promise. Templates whose
calculation has steps a line cannot carry (ROIC's tax rate and floor, clamps, net-cash floor) are
`exact=False`, list the real inputs, and end in an arrow instead of "=".

**Constraint.** Change a formula -> update `LINEAGE`, `RECOMPUTE` and `EQUATIONS` together. Never get
past the test by lowering the bar or flipping a template to `exact=False`. `test_metric_lineage.py`
is in the data loop's publish gate, so the equations are also checked on every 02:00 publish.

---

# Moved out of `CLAUDE.md` by the 2026-10-16 retrospective

The "Open" section had filled up again with narrative about work already
finished - the same growth the 2026-10-02 retrospective moved 551 lines for,
and for the same reason: a session reads `CLAUDE.md` before it starts, and a
closed item's story sits in front of the ones that are open. Each entry below
is the text as it stood; `CLAUDE.md` keeps the **constraint** and names this
file.

## 0.8/0.9, closed - calculation transparency, the premium redesign, and the four methodology questions it surfaced

**0.8 - CLOSED 2026-10-07.** Owner: *"I actually decided that I want you to finish up all UI/UX
work in here tonight. And make the nightly sessions just focus on what it was previously focusing
on, making sure the methodology is sound. And improving this tool overall every single night."*
Finished that night across five owner-run passes; what shipped is in
`plan/dashboard-inventory.md` and `NIGHTLY_LOG.md` 2026-10-07.

*Record:* first version of every stage shipped 2026-10-07 by an owner-run session
(`OWNER_FOCUS.md`, two items). `plan/calculation-transparency.md` and
`plan/dashboard-redesign-master.md` each carry a **"Status, 2026-10-07 evening"** table of what
is built. Highlights: the page prints the weights the engine used; every metric opens to its
formula, inputs and peer rank; **4,010/4,010 category scores and 502/502 composites rebuild from
the payload and the build refuses to publish otherwise**; `scripts/audit_stock.py --all`
independently reproduces 502/502; DOM nodes 9,928 -> 2,779, transitioned elements 662 -> 99, row
click to paint 70 ms, payload gzipped flat. **Pass 2 the same night** (owner: *"be creative"*): a
search palette (Ctrl/Cmd+K), `#stock=TICKER` links, J/K stepping, Compare with the composite gap
decomposed (claim `compare.composite_gap`), a workings CSV, Add to Holdings from the sheet,
analytics rows that filter the table, Peers without green/red verdicts, and Chart.js removed
(nothing used it). The navigation layer **wraps** `openStockDetail`/`closeModal` (`_js_ux()`).

**0.9 - ANSWERED 2026-10-09 (owner-run).** Each has a research note or a documented-defect entry
in `METHODOLOGY_CHANGELOG.md` 2026-10-09: **(a)** `operating_leverage` to weight 0 (85 of its 95
negatives were margin squeezes ranked best; the literature finds higher operating leverage earns
*more*; MSCI and AQR quality do not use it) - `research/2026-10-09-operating-leverage.md`;
**(b)** `revenue_growth` and Piotroski signals 3/8/9 now compare periods exactly a year apart
(the old comparison spanned 12-23 months by fiscal calendar) -
`research/2026-10-09-revenue-growth-window.md`; **(c)** one EBITDA for both ratios and the
snapshot; **(d)** labels "6-1M Return" and "Max Drawdown (13M)". **Found the same day by the same
lens:** `forward_eps_growth` (45% of Growth) spanned 13-24 months and mixed GAAP with adjusted
EPS; it is now MSCI's 12-month forward construction
(`research/2026-10-09-forward-eps-growth.md`). Their combined measured effect on the first live
run is the changelog's "Measured: the day's changes" entry (2026-10-09 evening).

**Constraints kept in `CLAUDE.md`:** nightly sessions do not take on design or fine-tuning work;
`_js_ux()` keeps wrapping the open/close functions; read the 2026-10-09 evening changelog entry
before touching any of those five metrics.

## 0.11, closed - the context layer (2026-10-08 to 2026-10-20)

Moved verbatim from CLAUDE.md "Current priorities" on 2026-10-20, when the close-out audit
found every queue item done and fixed the three defects it had hidden (`plan/context-layer.md`
"CLOSED 2026-10-20", `METHODOLOGY_CHANGELOG.md` 2026-10-20 evening).

**0.11. The context layer - first draft shipped 2026-10-08 (owner-run); the nightly sessions own
it now (opened 2026-10-08).** Technicals, options, insider trades, a market backdrop and
the ranking's track record, all context-only (settled row "ctx"). `OWNER_FOCUS.md` has the open item
and `plan/context-layer.md` the ordered queue. Queue items 1 and 3 are done; **item 2 was measured
and fixed in code 2026-10-09** - the options panel was empty for every stock on every scheduled run
(`ok` for **0 of 503** at the 02:00 loop's hour against **394 of 503** at 21:27 ET, because the
source serves the chain overnight with bid, ask and implied volatility all zero), fixed by an
after-close quote cache plus a probe instead of a guessed hour.

**`Screener Option Quotes` IS REGISTERED** (2026-10-09, owner-run; `Get-ScheduledTask` read back:
one weekly 20:00 trigger, no logon trigger; the loops still PT3M / PT20M). Smoke-tested live at
08:46 CT: quotes live, cache written. **Item 5, the evaluation harness, is built**
(`context_eval.py`, one line in the morning brief; first one-month window closes 2026-11-07).
Item 4 step one (every research sentence on the panel checked against its source) is done -
`research/2026-10-09-context-panel-claims.md`.

**Item 2 is CLOSED 2026-10-12.** The first 02:00 run after a 20:00 refresh gave options `ok` for
**397 of 503** (plus 59 `partial`), against 0 of 503 before the fix. Closing it caught a live false
statement: the refresh stamped the UTC date at 21:00 ET, so every stamp named the session *after* its
quotes, and the page told readers of **455 of 501** stocks that the quotes closed on **Saturday
2026-10-10**. Fixed by `options_cache.market_session_date`, with `is_session_date` as **one predicate
shared by the cache and the generator**. The standing lesson: `context.expected_move` was a registered
claim whose checks proved only that *a* date was carried through - **a claim whose check tests the
plumbing instead of the assertion is an unchecked claim.**

**The options panel's per-signal note is done** (`research/2026-10-12-option-implied-signals.md`) and
settles a question the plan left open: **the option-implied signals are not candidates for the score,
as a finding rather than a holding position** - every supported one decays faster than a monthly
review cadence, which is also why index providers keep option-implied inputs out of factor selection
while derivatives desks use them daily. Read a future `1m` record from `context_eval.py` with that
prior. Three page-sentence corrections are left open in `plan/context-layer.md` item 4 (the skew
effect size is attributed to a construction the page does not compute; the post-publication decay is
omitted; put/call open interest is a bare *level* where the evidence supports the *change*).
Next: item 4's remaining per-signal notes (macro backdrop, track record), then item 6.

## 0.12, closed - the metric audit's four residual judgement calls

Two audit passes over every weighted metric fixed nine defects on 2026-10-09
(`research/2026-10-09-metric-audit.md`, changelog "metric audit" entries). The four judgement
calls:

- **(a) DONE 2026-10-09 - a GICS sub-industry rule** (`tests/test_bank_like.py`; 13 stocks moved
  to the generic set, 0 left on the default): 26 of 59 financials reached the bank metric set
  only by the default for an unlisted industry - asset managers, insurance brokers,
  broker-dealers; TROW sat at the 100th percentile on equity ratio, AON at the 8th on P/B.
- **(b) DONE 2026-10-09 - weight 0.** `earnings_acceleration` (20% of Revisions) correlated
  -0.54 with the prior quarter's surprise, so it penalised a stock that beat last quarter
  (surprises autocorrelate - Bernard & Thomas 1990). The weight points to the non-surprise
  metrics.
- **(c) MEASURED, no change.** Accruals counted three times within Quality - correlation 0.49
  with Beneish, 0.00 with Piotroski, 13 points in all.
- **(d) DONE 2026-10-09.** Net debt nets cash and short-term investments, as enterprise value
  does.

**The momentum regime rule was REBUILT 2026-10-12 (owner-run)** as pre-registered in
`research/2026-10-09-momentum-regime.md`: momentum's weight is cut x0.70 (freed weight half to
Quality, half to Valuation) only in Daniel & Moskowitz's panic state - S&P 500 total-return index
24-month return negative **and** 126-day volatility above a fixed 20.6% - and never raised
(`factor_engine.market_panic_state`; changelog 2026-10-12 evening). Not in force on the day, so
no score moved.

**Constraints kept in `CLAUDE.md`:** the three panic-state numbers are never re-fitted; the
triple-counted accruals are measured and deliberately unchanged.

## 0.6, closed - one observation per run date

Open since 2026-08-11, closed 2026-10-08. The item asked for deduplication and a test, and both
landed: `record_dispersion` now **replaces** a date's row instead of appending, and
`record_run_snapshot` removes superseded files for its date - which is the rule
`compute_forward_returns` already applied when reading them, so the three now agree. Measured
before the fix: `dispersion_history.csv` held **52 rows for 45 distinct dates** (2026-04-14 four
times) and `improvement/snapshots` **79 files for 52 dates**;
`scripts/repair_one_observation_per_date.py` collapsed both, and `live_ic_history.csv` /
`performance_history.csv` came out byte-identical, so this removed a latent distortion of
`check_run_health`'s trailing dispersion median rather than correcting a live one.

**A third file had the same defect and was found the same day, by the session's own rescoring run
adding a row to it: `factor_vol_history.csv`, 72 rows for 52 dates, 2026-02-21 appearing nine
times.** That one feeds **scoring** - `adjust_momentum_weight` ranks the run's momentum dispersion
against every row and the percentile sets the momentum weight. Deduplicating moved the
distribution (p75 26.59 -> 26.27) without flipping the regime, so the published run was
unaffected; it was capable of it.

**Constraint kept in `CLAUDE.md`:** every append-once-per-run file holds one row per run date.

## 3. Backtest v2 - the record behind the open item (moved out of `CLAUDE.md` 2026-10-30)

`CLAUDE.md` priority 3 keeps the constraints and the Next; `plan/backtest-v2.md` governs. This is
the measurement record a session should read before it reopens a sequencing decision.

**Why the sequence is fundamentals-first.** Both biases measured on the same unit, the name-month
panel, both reproducible from committed inputs:

| Bias | Size | Source |
|---|---|---|
| Look-ahead | **>= 63.2%** (30.3% move >= 2 deciles; only 59.5% of v1's top decile belongs there) | `research/2026-10-01-lookahead-bias-size.md` |
| Survivorship | **11.4%**; 4.89% would remain after a free-data v2 | `research/2026-09-30-delisted-price-source-cost.md` |

Look-ahead is **5.5x bigger**, which reversed the plan's own sequencing: point-in-time
fundamentals outranks the point-in-time universe. `lookahead.weight_buckets()` derives the
decomposition from `config.yaml` so it cannot go stale - 16.9% of composite weight is honestly
recomputed per rebalance, **34.1 of the 83.1 held-constant points are free to fix** (28.0 restated
exactly by one month-end price, 6.1 recomputed from a price history), and 49.0 genuinely needs
point-in-time filings and estimates. Re-derived 2026-10-09 after the Revisions reweight: 34.45
free = 28.35 + 6.1; 48.65 needs point-in-time data. **`lookahead.weight_buckets(cfg)` is the live
figure** - read it rather than quoting these.

**What `pit_record.py` measured, 2026-10-15** (`research/2026-10-15-pit-record-validation.md`). It
rebuilds the engine's raw record from SEC filings as of any date, so v2 scores history with the
engine's own `compute_metrics` rather than a second implementation of the formulas. Validated
against today at **Spearman 0.885**, and it put the fundamentals half of v1's look-ahead at
**72.5% of name-months** - larger than the price half (60.1%); 75.8% together.

**The research engine's first batch, not decision-grade** (`v2_experiments.py`,
`research/2026-10-15-research-engine-first-batch.md`). v2 runs 2012-2026, 176 months. Of 27
pre-registered experiments **none beats the current weighting, the baseline's IC is ~0 in both
eras, and the design can only see an IC difference of ~0.04-0.07.** `backtest_v2_returns.py`'s own
first reading: IC **-0.002 (+-0.027)** over 80 months, against **+0.050 (t 4.7)** for the same
pipeline fed today's filings - which is what look-ahead does to a result (section 5c).
**METHOD_VERSION 2** (2026-10-17) pairs each comparison on the month's *common stocks*; under
version 1 a narrow screen was measured over its own smaller set, so part of every difference was a
difference in who was ranked.

## 0.14, the three input definitions settled against the filings (moved out of `CLAUDE.md` 2026-10-30)

All three were proposals to change what a metric is computed from, all three were answered with a
measurement, and all three shipped a **display-only** row instead. `CLAUDE.md` keeps the verdict
and the section a reopening must answer; this is why.

**Capex: the lease-purchase subtraction was REFUSED, 2026-10-15.** Yahoo's capex row adds
intangibles and finance-arm lease assets, so it is different from the filings *by construction*.
The proposed rule would have kept a captive finance arm's operating cash flow and its debt while
dropping its capex - GM's own 10-K starts from *automotive* operating cash flow of **$18.7B**
against the consolidated **$26.87B**. Any such rule also puts **9.9 composite points** on the
weekly SEC cache, which scoring deliberately never reads. What shipped: the FCF-yield row
decomposes the capex figure from the company's own filings (claim `drilldown.capex_lease_note`,
`tests/test_capex_lease_note.py`). Reopening must answer section 5.7 of
`research/2026-10-14-capex-definition.md`.

**Operating income: keep the source's "Operating Income" row, 2026-10-15.** It is as-reported
less the source's "Special Income Charges", and the "Total Operating Income As Reported" row
equals the filings **374 of 374**. Rouen, So & Wang (2021) on persistence and MSCI's practice both
support the pre-special-items figure for ROIC and EV/EBITDA, so its disagreement with the filing
is the definition, not a defect. Do not "fix" it by switching rows or to the filings.

**Earnings yield: scoring it on "Normalized Income" was REFUSED, 2026-10-16.** The research is
good (Rouen, So & Wang 2021, persistence **0.810 against 0.637**) but the input cannot be put on
one basis. At the four-quarter window the engine sums, **23 companies have no normalized figure at
all** (Amazon, Bank of America, Berkshire, Boeing), which re-creates the fallback-cohort defect
`factor_engine.py:2094` records Phase 13 fixing *in this very metric*; and **8 of the 18 fallback
companies whose figures are known carry annual one-offs worth 10%+ of net income** (Boeing 434%,
General Mills 2,193%), so the change would remove the small distortions and keep the largest.
Reading "absent" as zero is false for 70 companies. What shipped: the earnings-yield row says what
one-off charge is inside the figure, from the company's own filings (47 of 501 stocks; **37 of
those 47 tag one in every one of their last four fiscal years**) - claim
`drilldown.special_item_note`, `tests/test_special_item_note.py`. Reopening must answer section
8.6 of `research/2026-10-15-earnings-definition.md`.

**The agreement measured 2026-10-14** (`research/2026-10-14-yahoo-vs-sec.md`), rebuilt from SEC
XBRL for the same quarter: within 0.5% for **99.8%** of companies on total assets, **98.8%** on
operating cash flow, **97.9%** on equity, **94.3%** on net income, **88.5%** on revenue. The
larger disagreements examined are definitional - bank revenue, minority interests, excise taxes.

## 0.13, experiments 14-17 (moved out of `CLAUDE.md` 2026-10-30)

Four more after the thirteen below, each with its note and its entry in `plan/factor-lab.md`.
Seventeen in all as of 2026-10-20, **145 of 217 figures within tolerance**.

| # | Paper | Result |
|---|---|---|
| 14 | Cooper, Gulen & Schill 2008, asset growth | 13 of 24; by market value it has earned about nothing since 2003 (`research/2026-10-19-lab-asset-growth.md`) |
| 15 | Jensen, Kelly & Pedersen 2023, their own 153 factors | 10 of 10. **Its data is CC BY-NC 4.0 - teaching page only, never anything that trades** (`research/2026-10-20-lab-jkp.md`) |
| 16 | size inside each profitability fifth | 10 of 17; Size's pre-registered rule **not refuted**, which is what kept the weight at 5% |
| 17 | Piotroski 2000 on Open Source Asset Pricing's published portfolios | 1 of 4, the miss explained by the decile cut of a ten-value score (`research/2026-10-20-lab-osap.md`) |

**Every new experiment needs a second reader before it counts as checked** (`OWNER_FOCUS.md`,
2026-10-20 item 2). The 2026-10-23 13:00 session did that for 16 and 17: all 21 reference figures
exact, every measured figure reproduced.

## The four weight questions the lab raised, and how each was answered

The reasoning, moved out of `CLAUDE.md` 2026-10-30. The *answers* and the tests that hold them
stay there; this is what a session should read before reopening one.

**Question 1 - Growth 13 -> 10 / Quality 22 -> 25: REFUSED 2026-10-13**
(`research/2026-10-12-growth-category.md` section 10). Net growth exposure is monotonic in the
weight, crosses zero at **13.5**, and is within 0.05 of neutral only over **11-16**, because
Growth is the one category running against the rest (**-0.388** against the mean of the other
seven). If growth earns nothing the supported exposure is zero, and 13 points is what delivers it:
**cutting an unrewarded-but-offsetting category's weight does not remove a position on growth, it
takes one.** The open Growth question is **composition, not weight** - `revenue_growth` against
`asset_growth`, and whether growth in profitability belongs in Quality (note section 11).

**Question 2 - Momentum's weight: KEPT**
(`research/2026-10-12-value-and-momentum-among-large-companies.md`). Large-company value was never
reliable on book value and is better supported on the earnings and cash-flow measures this
screener uses; large-company momentum is strong in every published sample and flat on public data
since 2000. Lab and literature disagree, so the weight stands.

**Question 4 - Size stays at 5%, 2026-10-20** (`research/2026-10-20-size-weight.md`, 703 lines,
13 sources). No unconditional size premium among large companies - Fama & French (1992) *JF*
47(2) Table II, Fama & French (2008) *JF* 63(4) Table IV, Alquist, Israel & Moskowitz (2018)
*JPM* 45(1) - but the within-sector small tilt beside Quality is documented practice (MSCI
Diversified Multiple-Factor May 2022 2.3-2.4; FTSE Russell Global Factor Ground Rules v8.8;
Dimensional), and it is one of the counterweights holding the ranking growth-neutral. The trigger
to reopen it is **lab experiment 4b**, with its refutation criterion written in the note.
Confirmed by experiment 16 and second-read 2026-10-23.

**Questions 3 and 5 are open Mondays.**

## 0.13, the lab's first thirteen experiments

Chronology as of 2026-10-15; `plan/factor-lab.md` is the live queue and `lab.html` the result.
Thirteen experiments, **111 of 162 figures within tolerance**.

| # | Paper | Result |
|---|---|---|
| 1 | Barroso & Santa-Clara 2015 | 13 of 14 within, one marked outside |
| 2 | Daniel & Moskowitz 2016 (behind the momentum regime rule) | 9 of 15; misses traceable to the paper building its own portfolios |
| 3 | Fama & French 2015, five factors | 20 of 21 |
| 4 | Fama & French 1992, size | 3 of 11 - the first that mostly **fails**, because the paper's sample is a different set of companies; published that way |
| 5-7 | volatility, earnings yield, market beta | generic sorted-portfolio layout (`factor_lab.SORTS`, `kind: "sort"`) - a new sort is a record and a file entry, no page work |
| 8 | the factors among large companies (`kind: split`) | value, investment and momentum about half-size there; profitability is the one that holds |
| 9 | Israel & Moskowitz 2013, the largest fifth | 5 of 12, the four largest-fifth figures all matching |
| 10 | Novy-Marx 2013, profitability | shipped 2026-10-15 |
| 11 | Lakonishok, Shleifer & Vishny 1994, cash-flow yield | 11 of 11 |
| 12 | Sloan 1996, accruals | 6 of 11; the direction reproduces at half the size |
| 13 | Asness, Moskowitz & Pedersen 2013, value and momentum together | 7 of 10, with a new `combo` layout |

**Read `research/2026-10-12-factor-lab-findings.md`:** the first three results weakened after
their samples ended, and value and investment are largely one bet. Neither is a reason to move a
weight; both are research questions for a Monday.

**Constraints kept in `CLAUDE.md`:** the lab never justifies a parameter by itself; tolerances
before the comparison; a failed replication is published; five things move together; the route to
a weight is the plan's "From the lab to the weights"; Growth's 13 and Momentum's weight both
stand, for the reasons in their notes.

## 0.10, 4, 8 - the short ones

**0.10's measured coverage, 2026-10-08** (`research/measurements/2026-10-08-equation-coverage.py`,
re-measure rather than trusting it): of 29 weighted metrics **15 carry an exact equation (58.0% of
composite weight), 5 an inexact one (19.5%) and 9 none at all (22.6%)**. Of that 22.6%, Piotroski
(3.30%) and Beneish (1.54%) already list their signals and indices, so the genuine gap is the
seven history-based metrics - 17.8%. Parts (a)-(d) were built 2026-10-09 (owner-run) and
smoke-tested live on AAPL, JPM and NVDA to 4dp, publishing `_ja_*`, `_beta_cov/_var`, `_vol_sd`
and the surprise metrics' four quarters (`eq4`) from the engine. `scoring_schema` is 4.

**4's residual - DONE 2026-10-09 (owner-run).** The What Changed panel opens with a run-level
overview (`run_overview.py`, claim `history.run_overview`): who moved past the noise threshold,
which categories the movement came from (weighted, direction-free), how many of the top 25 held.
Its pair, Reporting Soon, shipped 2026-10-08. The 2026-08-10 owner directive's pieces are all
built.

**8. Test isolation - DONE 2026-10-09 (owner-run).** A per-test write detector found exactly three
tests writing published files; all three now use `tmp_path` or the committed universe, and
`conftest.py` is a tripwire that restores the files and **fails the session** if any test writes
one.
