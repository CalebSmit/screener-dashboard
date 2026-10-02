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
