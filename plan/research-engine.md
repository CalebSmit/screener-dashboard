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
weighting; the baseline's IC is ~0 in both eras; the design can see an IC difference of about 0.04
(2012-2019) to 0.07 (held-out months) and no smaller; a live record needs ~7 years to show an IC of
0.05 at t >= 3. "Not distinguishable" here means "not large enough to see", never "equal".

## The rules (enforced by `tests/test_v2_experiments.py` where code can enforce them)

1. **Register before running.** An entry in `research/experiments/registry.yaml` - hypothesis,
   source (a paper or documented practice), prediction - committed and pushed before its first run.
   `run` refuses an unregistered id. A prediction is never edited after a run; a changed idea is a
   new id. A broken experiment is marked `void:` with the reason, prediction untouched.
2. **Every run is kept** (`ledger.jsonl`, append-only, with the data version it used). The
   significance bar counts every experiment ever registered (p < 0.05/N, |t| >= 3).
3. **A data fix reruns everything.** Rows from different data versions do not compare; the report
   says so when they are mixed.
4. **Never register a variation of a near-miss.** If size or momentum looked good, the next entry
   is not "size, but 70%". That is the forking path the rules exist to close. A new entry needs its
   own source.
5. **Periods are fixed in code**: discovery 2020-01..2023-06, held out 2023-07..2026-08, before the
   sample 2012-01..2019-12 (registered before it was built). Do not move them; add a period only by
   registering it, with its rule, before its data exists.

## Queue (in order)

1. **Per-month CIKs for 2020-2026** (`ticker_ciks.json` has the revids): fetch each revision's CIK
   column into `ciks_by_month.json`, rebuild 2020-2026 (`python backtest_v2.py --shard k/4`, then
   `--merge 4`), rerun every experiment. The reuse guard covers 2014-2019 only today; IR in
   2020-01/02 is the known case it would catch. Expect no change in any verdict - which is the point
   of checking.
2. **Show power next to every verdict.** Add a column to `RESULTS.md`: the smallest IC difference
   each comparison could detect at the bar (from its own paired standard error). A reader should not
   be able to read "not distinguishable" without seeing what that excludes.
3. **Robustness of the statistics.** The t-statistics assume months are independent. Measure the
   autocorrelation of the paired IC differences; if any lag-1 exceeds ~0.2, report Newey-West
   t-statistics beside the plain ones. A block bootstrap of the baseline's IC is the cross-check.
4. **The bot's own rule.** Register the trading bot's selection rule (read it from
   `CalebSmit/screener-bot`'s `STRATEGY_LOG.md` - do not change the bot) as an evaluation: its top-N,
   equal weight, monthly, against the equal-weighted universe, with a stated cost assumption, on
   both periods. Predicted before it runs. This answers "what should the owner expect from the bot"
   in the same honest frame. Nightly sessions read the bot repo; they never edit it.
5. **Survivorship** is v2's binding limit (58% of 2012 members present). The decision on the
   $199/yr delisted-price feed (`research/2026-09-30-delisted-price-source-cost.md`) is the owner's -
   write up what it would now buy (the 2012-2019 test with the stocks that left) and leave it in
   `OWNER_FOCUS.md`; do not buy it.
6. **New ideas come from Monday notes.** Each Monday research note that ends with a testable claim
   registers one experiment for it. That is the route by which the engine grows - one sourced idea
   at a time - and the multiple-testing bar tightens as it does, which is correct.

## Things not to do

- Do not quote an engine number as evidence in `METHODOLOGY_CHANGELOG.md` (rule 5) - a *Backtest
  observation* line, clearly marked, at most.
- Do not import `v2_experiments` from production code (tested).
- Do not loosen the bar, merge periods after the fact, or report the best of several runs.
- Do not let the engine's cache be written by tests (`tests/test_backtest_v2.py` redirects it; the
  engine reads month-ends only).
