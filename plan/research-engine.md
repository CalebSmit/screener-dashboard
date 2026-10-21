# The research engine - plan and queue

**Opened 2026-10-15 (owner-run).** Owner: build the methodology the way automated research loops are
being built - many ideas from the literature tested against each other - so the screener can be
shown to people and trusted (a trading bot first, a fund's method later). First version built and
run the same night: `v2_experiments.py`, `research/experiments/` (registry, ledger, results),
`research/2026-10-15-research-engine-first-batch.md`. **Read that note before anything here.**

## What it is for, and what it is not

It turns "I think this weighting is better" into a pre-registered, held-out, multiple-testing-adjusted
comparison on point-in-time history (backtest v2, 2012-01 .. 2026-08). It is **not** a way to set
weights: under CLAUDE.md rules 4 and 5 nothing it reports moves a weight before 2027-02-11, and even
then a result is a research question that a note and a changelog entry must answer.

**Its first answer, which governs how to read every later one:** none of 27 ideas beats the current
weighting; the baseline's IC is ~0 in both eras; the design can see an IC difference of about 0.040
(2012-2019) to 0.070 (held-out months) and no smaller (Student's t since 2026-10-16); a live record needs about 7½ years to show an IC of
0.05 at t >= 3. "Not distinguishable" here means "not large enough to see", never "equal".

## The rules (enforced by `tests/test_v2_experiments.py` where code can enforce them)

1. **Register before running.** An entry in `research/experiments/registry.yaml` - hypothesis,
   source (a paper or documented practice), prediction - committed and pushed before its first run.
   `run` refuses an unregistered id. A prediction is never edited after a run; a changed idea is a
   new id. A broken experiment is marked `void:` with the reason, prediction untouched.
   **An idea whose result is already on file is not pre-registered** (2026-10-16): `run` refuses an
   id whose statistic appears in a committed `research/measurements/*.json` older than the id's
   registration commit, unless the entry discloses it with `outcome_known_at_registration`
   (`v2_experiments.refuse_known_outcomes`). Disclosure fields may be added to an entry after it ran;
   its hypothesis, prediction and transform may not change.
2. **Every run is kept** (`ledger.jsonl`, append-only, with the data version it used). The
   significance bar counts every experiment ever registered (p < 0.05/N, |t| >= 3).
3. **A data fix reruns everything.** Rows from different data versions do not compare; the report
   says so when they are mixed. **Run a rerun with `python scripts/rerun_experiments.py`**
   (2026-10-19): it runs only the rows not yet current, so a rerun that was cut off resumes instead
   of restarting; `--limit 4` (about 8 minutes) fits one foreground call, `--status` changes nothing.
   A full rerun is about 47 minutes - never start one in the background and end the reply, which is
   how 2026-10-18 and 2026-10-19 lost theirs (`research/2026-10-19-session-ended-on-a-background-job.md`).
4. **Never register a variation of a near-miss.** If size or momentum looked good, the next entry
   is not "size, but 70%". That is the forking path the rules exist to close. A new entry needs its
   own source.
5. **Periods are fixed in code**: discovery 2020-01..2023-06, held out 2023-07..2026-08, before the
   sample 2012-01..2019-12 (registered before it was built). Do not move them; add a period only by
   registering it, with its rule, before its data exists.

6. **The public page quotes the engine** (2026-10-15, claim `overview.research_engine`): the
   methodology page's point-in-time paragraph reads `research/experiments/summary.json`, which
   `python v2_experiments.py report` writes. After any rerun, run `report` and commit the ledger,
   `RESULTS.md` and `summary.json` together - the claims gate fails if the summary and the ledger
   disagree.

7. **A result names the code that produced it, or it is not a result** (2026-10-18, item 8b).
   `run` and `evaluate` refuse to start from an edited tree, over the files that can change a
   number (`v2_experiments.RESULT_PATHS`) - so commit first; the commit is also the
   pre-registration proof rule 1 rests on. Every row carries `code_version`, `code_hash`,
   `content_hash` and `base_config_hash`, and `report` flags a mixture of any of them the way it
   flags mixed data or method versions. **Do not add an override flag.** The one legitimate escape
   is the one already there: where git itself is absent the run proceeds and records `?`, because a
   gate that can only fail closed jams an unattended loop.
   **`code_version`'s `-dirty` marker is scoped to `RESULT_PATHS`, like the refusal, and is never
   asked of `git describe`** (fixed 2026-10-18 - see item 8(b)): the whole-tree version fires on
   the engine's own `ledger.jsonl` and `RESULTS.md`, so every second entry point in a rerun was
   branded unreproducible. The guard and the marker say the same thing about the same tree, or one
   of them is wrong.

## Queue (in order)

1. **DONE 2026-10-15 - and the guard refined** (note section 2: a mismatch is excluded only when today's
   filer was another member that month). Kept for the record: **Per-month CIKs for 2020-2026** (`ticker_ciks.json` has the revids): fetch each revision's CIK
   column into `ciks_by_month.json`, rebuild 2020-2026 (`python backtest_v2.py --shard k/4`, then
   `--merge 4`), rerun every experiment. The reuse guard covers 2014-2019 only today; IR in
   2020-01/02 is the known case it would catch. Expect no change in any verdict - which is the point
   of checking.
2. **DONE 2026-10-15: every comparison in `RESULTS.md` prints what it could see (`; sees`,
   `v2_experiments.detectable`).** Kept for the record: Add a column to `RESULTS.md`: the smallest IC difference
   each comparison could detect at the bar (from its own paired standard error). A reader should not
   be able to read "not distinguishable" without seeing what that excludes.
3. **DONE 2026-10-15 (note section 10): median lag-1 autocorrelation +0.002; no Newey-West t reaches
   3; `t_nw` and `ac1` are recorded on every run.** Kept for the record: The t-statistics assume months are independent. Measure the
   autocorrelation of the paired IC differences; if any lag-1 exceeds ~0.2, report Newey-West
   t-statistics beside the plain ones. A block bootstrap of the baseline's IC is the cross-check.
4. **DONE 2026-10-15 (note section 9): the bot's rule lagged the equal-weighted universe in every
   period.** Kept for the record: Register the trading bot's selection rule (read it from
   `CalebSmit/screener-bot`'s `STRATEGY_LOG.md` - do not change the bot) as an evaluation: its top-N,
   equal weight, monthly, against the equal-weighted universe, with a stated cost assumption, on
   both periods. Predicted before it runs. This answers "what should the owner expect from the bot"
   in the same honest frame. Nightly sessions read the bot repo; they never edit it.
5. **Survivorship** is v2's binding limit (58% of 2012 members present). **The owner decided
   2026-10-15: no paid data** - not the $199/yr feed, not any other. Free routes only: SEC filings
   are free for departed companies too (the fundamentals exist; it is prices that are missing), so
   look for free end-of-period prices of departed members (e.g. the last close a merger or
   delisting announcement states in an 8-K, or a deal price), and measure how much of the gap a
   free source closes before building anything on it. *Tried 2026-10-15:* Stooq's CSV download now
   answers every request (delisted or not) with a browser challenge page - not usable, and getting
   around a bot check is off-limits.
6. **New ideas come from Monday notes.** Each Monday research note that ends with a testable claim
   registers one experiment for it. That is the route by which the engine grows - one sourced idea
   at a time - and the multiple-testing bar tightens as it does, which is correct.

   **First one landed 2026-10-19: `drop-revenue-growth`**, from
   `research/2026-10-19-revenue-growth-vs-asset-growth.md` (sections 7, 9 and 10), registered and
   committed before it ran. Result: held-out difference **-0.007 at t -2.5**, pre-sample -0.001 at
   t -0.69 - **not confirmed**, and **the sign came out opposite to the registered prediction**.
   *(2026-10-19, after the `pit_record` record fixes and a full rebuild and rerun: -0.0072 at
   t -2.45; verdict unchanged - `research/2026-10-19-pit-fix-rebuild-effect.md`.)*
   The note reports that against itself rather than around it, which is what registering a
   direction is for. Nothing was changed on the strength of it (rule 5 bans citing it either way,
   including keeping a weight because of it).

   **Budget for this: registering one experiment costs a full rerun of all of them.**
   `research/experiments/registry.yaml` is in `RESULT_PATHS`, so adding an entry changes
   `code_hash`, and rule 3 plus `test_every_ledger_row_the_report_reads_names_its_code_and_its_baseline`
   then require every id's newest row to have been produced by that code - i.e. all of them rerun.
   On 2026-10-19 that was 30 runs (29 registry entries plus the evaluation), ~1.6 minutes each -
   and the 06:00 session that registered it ran out of time after 15, so its gates failed and an
   owner-run session finished the other 15. It is the right behaviour and should not be weakened: the
   multiple-testing bar moves for every row anyway when N changes (p < 0.05/27 -> 0.05/28, |t| >= 3.35
   -> 3.37), so a mixed ledger really would be incomparable. But a session that registers an
   experiment must plan for the rerun **and run `report` afterwards**, or it hands the runner a red
   gate 1 - and `research.html` cannot even represent a registered-but-unrun entry, because
   `research_page.display_label` labels anything carrying a `construction_note` and the page lists
   only rows the ledger has.
7. **DONE 2026-10-16 (audit of `research.html`): the page now claims only what the ledger supports.**
   Size's lead is stated as a survivorship hypothesis the data cannot separate from time (the
   sentence is generated, `research_page.size_coverage_relation`); the seven `only-*` entries whose
   2020-2026 outcome was committed before registration carry `outcome_known_at_registration` and a
   dagger, and `run` refuses an undisclosed case (`refuse_known_outcomes`); the pairing is described
   as by month; headline ICs carry 95% intervals and a pooled 176-month estimate; the chart separates
   the detectable band from the pass bar; the trading-rule sentence is generated from each period's
   sign and t; within-sector screens say so (`construction_note`); p-values and critical values are
   Student's t (no verdict changed). Tests: `tests/test_research_page.py`, `tests/test_v2_experiments.py`.
8. **Queued by the same audit, specified, NOT built.** In this order:

   **(a) DONE 2026-10-17 - method version 2.** Both sides of every comparison are now recomputed on
   each month's ticker intersection before the difference is taken, so a difference is a difference
   in ranking and not in which companies were ranked. `per_month(scores, fwd, baseline_scores=...)`
   returns `base_ic` / `base_top_minus_universe` / `n_common` beside the experiment's own figures;
   `summarise` differences only within one frame and has **no parameter for another configuration's
   frame**, which is what version 1 did. Every ledger row carries `method_version`,
   `n_common_by_month` and `base_ic_by_month`; `report` refuses to mix method versions
   (`method_note`) exactly as it does data versions, and `summary_for_page` drops a row of either
   mismatch. All 28 experiments and the one evaluation were rerun (rule 3) and the public page is
   regenerated. The record is `research/2026-10-15-research-engine-first-batch.md` section 11.
   *Kept for the record:* under version 1 each configuration's IC was computed on the stocks *it*
   ranked (`backtest_v2.rank_month` filters on coverage of that configuration's own weighted
   metrics), so a narrow screen was judged on fewer stocks - free-cash-flow yield alone ranks 348 a
   month against the baseline's 440 in 2020-2026, and 213 against 327 in 2012-2019
   (`research/measurements/2026-10-16-stocks-ranked-per-config.json`).

   **(b) DONE 2026-10-18 - provenance hardening.** Built by the 06:00 session, **finished by the
   13:00 one**, whose ship gates the unfinished half had failed - see the caution at the end of
   this item. Every ledger row carries `git describe --tags` plus a `-dirty` marker
   (`code_version`), the SHA-256 of each result-affecting file (`code_hashes`, hashed together as
   `code_hash`), `content_hash` and `base_config_hash`; `run` and `evaluate` refuse to start from
   an edited tree, scoped to `RESULT_PATHS`. `report` flags a mixed code version, baseline
   weighting or content hash the way it flags a mixed data version (`provenance_note`), and
   `summary_for_page` drops a row of any mismatch (`comparable_provenance`). All 28 experiments and
   the evaluation were rerun on a clean tree so every row the report reads can name its code; the
   ledger's method-1 and pre-8b rows stay in it (rule 2). `tests/test_v2_experiments.py`
   provenance block.
   **The rerun moved no number, measured rather than assumed**
   (`research/measurements/2026-10-18-provenance-rerun-effect.py`): 29 of 29 ids compared against
   their previous run, every leaf of the stored summary at full precision, **0 differing anywhere,
   largest numeric difference 0.0**. Identical was the prediction - neither the data nor the method
   changed - and the research page quotes these numbers, so it is worth having measured.
   **The rerun found a defect in the marker, now fixed:** `git describe --dirty` reads the **whole
   tree**, and a run appends to the tracked `ledger.jsonl` while `report` rewrites `RESULTS.md` -
   so by the time the *second* entry point starts, the tree is dirty **because of the run itself**.
   It stamped the evaluation row `-dirty` on a rerun where nothing but the engine's own output had
   changed, and there is no order of `run` and `evaluate` that escapes it, in one process or two.
   The marker now comes from `edited_result_files()`, the same `RESULT_PATHS` scope the refusal
   uses. **The guard and the marker must mean the same thing** - before the fix the guard permitted
   a run the marker then branded unreproducible - and an engine *output* is a result, which cannot
   change a number. Keep it that way: `test_the_engines_own_output_does_not_brand_a_row_unreproducible`
   fails if `--dirty` is ever asked of git again. The general form, worth remembering next time a
   provenance check is added anywhere: **a check whose subject includes the checker's own output
   fires in normal operation, and a marker that fires in normal operation gets ignored.**
   *A caution about this queue, from how (b) went:* the item above was written as DONE, in this
   file, while **2 of the 29 rows** had actually been rerun; the tests that enforce it were red and
   the ship gates refused the merge, which is the argument for the runner re-running them
   independently (CLAUDE.md "Ship gates", 2026-09-04) rather than for trusting a progress line.
   **A stage is done when its own tests pass on a clean tree, not when its description is written.**
   *Two deviations from the specification above, both deliberate:* `universe_history.py` is on
   `RESULT_PATHS` as well - it decides who was a member of each month, which is as much a
   result-affecting input as `pit_record.py`, which the spec did name; and the **base run id is
   `unrecorded` for the months on disk today**, because they were built on 2026-10-15 and nothing
   recorded it then. `backtest_v2.record_build` records it per month from now on, merging across
   shards. Inferring it from directory timestamps was considered and rejected: a guess printed in
   the place a record belongs is worse than an honest absence.
   *What `base_config_hash` is for, since it is easy to mistake for the existing field:*
   `config_hash` covers the configuration *after* the transform, so it differs between experiments
   by design and can never say whether two rows started from the same `config.yaml`.
   *Kept for the record:* the specified data fingerprint was the member list only, which a rebuild
   can leave byte-identical while every number inside the months moves.

   **(c) The positive control, in the ledger.** The look-ahead arm (today's filings for every
   month, IC +0.050, t 4.71 on 2020-2026) is the page's evidence that the test can see a signal, but
   it lives in a measurement file. Record it as a ledger row of `kind: control` on the current data
   version, rerun with everything else. **The synthetic calibration test is DONE 2026-10-20 (owner-run, `tests/test_engine_calibration.py`, 7 tests):** planted differences of 0.03, 0.05 and 0.10 come back within four standard errors in every period, 0.10 clears the bar on the 38 held-out months, no false alarm across ten null pairs, twice `detectable` is always caught, and a 70%-coverage screen is measured on the shared stocks. The ledger row of `kind: control` is still open. As specified: inject a known signal
   (a score equal to next month's return plus noise scaled to a target IC of 0.03, 0.05 and 0.10)
   into a copy of one era's months, and assert the engine's paired comparison reports a difference
   within its interval of the injected one and detects 0.10 at the bar.

   **(d) The free extended-universe rebuild - the survivorship test.** SEC filings exist for
   departed members; prices are the gap. Build the months with the departed members whose last
   prices can be reconstructed from free sources (8-K deal prices, final closes stated in filings),
   measure how much of the 2012-2019 coverage gap it closes, and rerun `only-size` against the
   baseline on it. **Pre-registerable prediction, to commit before the rebuild exists:** if
   survivorship drives Size's 2012-2019 lead, its difference from the baseline falls from **+0.028**
   to **below +0.014**; the survivorship explanation is **falsified if it stays at +0.020 or
   more**. Sources: Linnainmaa & Roberts (2018), RFS 31(7), on testing anomalies in the history
   before their samples; Shumway (1997), J. Finance 52(1), on the delisting-return bias in
   survivor-only data. No paid data (owner, 2026-10-15).

   **(e) Measure today's sector and industry taxonomy on historical months before changing it.**
   v2 assigns each historical member today's sector (the engine ranks within sector, so this sets
   every peer group). Measure, on the months where a point-in-time classification can be had free
   (SEC SIC codes by filing date), how many members change sector and how far their within-sector
   percentiles move. Change nothing until that number is in a note.

## Things not to do

- Do not quote an engine number as evidence in `METHODOLOGY_CHANGELOG.md` (rule 5) - a *Backtest
  observation* line, clearly marked, at most.
- Do not import `v2_experiments` from production code (tested).
- Do not loosen the bar, merge periods after the fact, or report the best of several runs.
- Do not let the engine's cache be written by tests (`tests/test_backtest_v2.py` redirects it; the
  engine reads month-ends only).
