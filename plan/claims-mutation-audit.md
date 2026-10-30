# Claims mutation audit - would each check fail if its claim became false?

Opened 2026-10-20 (owner-run, night), from `OWNER_FOCUS.md` item 3, "Test the tests".
Owner of the queue below: the nightly sessions.

## Why

`claims.py` registers every public sentence that says how a number is computed, with the
code that makes it true (`made_true_by`) and the tests that check it (`checked_by`).
`tests/test_claims_register.py` proves each `checked_by` test *exists*. Nothing proved it
would *fail* if the sentence stopped being true. The 0.11 lesson says why that matters:
`context.expected_move` was registered, checked, and false, because its checks tested that
*a* date was carried through rather than *which* date - "a claim whose check tests the
plumbing instead of the assertion is an unchecked claim."

A mutation audit asks the question directly: make the smallest code change that makes the
claim false, run only that claim's `checked_by`, and see whether anything goes red.

## Method

For each claim:

1. Read its `asserts` and find, in the module `made_true_by` names (or the code it calls),
   the line that carries one arithmetic fact the sentence states - a window, a threshold,
   a denominator, a boundary, a rounding rule, a weight.
2. Make the **smallest plausible** change that makes the sentence false and leaves the code
   otherwise working: 252 -> 250, `<` -> `<=`, 0.05 -> 0.5, a proportional split -> an
   equal one, a floor halved. Prefer changes a careless edit could really make.
3. Run **only** that claim's `checked_by` (the gate does the same, so this is the coverage
   the register actually buys).
4. Record caught (some test failed) or **gap** (all passed). Revert with
   `git checkout -- <file>` and confirm `git status --porcelain` shows no source file
   before the next mutation. **Never commit a mutation.**
5. For a gap: write a test that asserts the *claim* against the code's *behaviour* - not
   that a string is present, not that a field exists, not that the payload agrees with
   itself. Show it fails under the mutation and passes without it, then add it to the
   claim's `checked_by`. The data loop's publish gate picks it up automatically
   (`scripts/check_published_claims.py` derives its modules from `checked_by`).

The helper used on 2026-10-20 was a 40-line script (apply one exact-string replacement that
must occur once, run the claim's `checked_by` via `claims.CLAIMS`, `git checkout` the file,
print CAUGHT or GAP). It lives in no tracked file on purpose: a mutation tool inside the repo
is one bad `finally` away from committing a mutant. Rewrite it in the scratchpad; it is short.

**Equivalent mutants.** A mutation that cannot change behaviour on today's data or config
(e.g. ignoring a config value that equals the code's default) is still a gap when the claim
says "the configured" value - the test should run the engine on a *different* config and
see it followed. Record it rather than skipping it.

## Results, 2026-10-20

22 claims, 38 mutations, all four surfaces. **15 mutations (11 claims) passed every
registered check.** All 15 now fail a registered check; each new test was shown to fail under
its mutation and pass without it.

| # | Claim | Mutation (file) | Registered checks | Now |
|---|---|---|---|---|
| 1 | `summary.target` | thin basis `analysts < 5` -> `<= 5` (stock_summary) | **gap** | fixed: `test_thin_basis_is_below_five_analysts_and_not_at_five` |
| 2 | `summary.target` | gap as (target - price) / target instead of / price | caught | - |
| 3 | `summary.change` | composite-move threshold 0.05 -> 0.5 | **gap** | fixed: `test_composite_move_is_stated_from_005_and_omitted_below` |
| 4 | `summary.input_churn` | `total < MIN_INPUT_CHURN` -> `<=` | caught | - |
| 5 | `summary.peers` | group size drops "plus itself" | caught | - |
| 6 | `summary.worst_input` | lowest input taken from the first two categories only | caught | - |
| 7 | `overview.percentile_rule`, `drilldown.peer_rank` | universe fallback `len(valid) < 10` -> `<= 10` (factor_engine) | **gap** | fixed: `test_a_sector_with_exactly_the_minimum_is_ranked_within_itself` |
| 8 | `overview.percentile_rule` | midpoint rank (r - 0.5)/n -> r/n | caught | - |
| 9 | `overview.coverage_filter` | `int()` -> `round()` in `coverage_minimum` | caught | - |
| 10 | `overview.auto_disable` | freed weight split equally, not in proportion (run_screener) | caught | - |
| 11 | `overview.weight_sensitivity` | +-5 points -> +-5% of itself | caught | - |
| 12 | `overview.coverage_discount`, `drilldown.composite_chain` | discount rate halved inside `compute_composite` | **gap** | fixed: `test_the_engine_applies_the_discount_the_page_states` |
| 13 | `drilldown.composite_chain` | discount starts 5 points below the stated threshold | **gap** | fixed: same test |
| 14 | `drilldown.ranking_audit` | tax 21% -> 25% (ranking_audit) | caught | - |
| 15 | `drilldown.ranking_audit` | net-income ratio bar 1.25 -> 1.5 | caught | - |
| 16 | `drilldown.ranking_audit` | thin-equity bar 10% -> 15% | **gap** | fixed: `test_the_notes_fire_either_side_of_the_stated_thresholds` |
| 17 | `drilldown.filing_check` | "agrees" at 3% instead of 2% (page JS) | caught | - |
| 18 | `drilldown.filing_check` | relative difference over the filing only, not the larger of the two (sec_crosscheck) | caught | - |
| 19 | `context.price_trend` | 52-week range over 250 closes, not 252 | caught | - |
| 20 | `context.price_trend` | 200-day slope lag 20 -> 21 days | caught | - |
| 21 | `context.price_trend` | uptrend `last >= sma200` -> `>` | caught | - |
| 22 | `context.price_trend` | 3-month lookback 91 -> 90 days | caught | - |
| 23 | `context.rate_sensitivity` | share explained = abs(corr), not corr squared | caught | - |
| 24 | `context.rate_sensitivity` | sector bars from 4 stocks, not 5 (generate_dashboard) | caught | - |
| 25 | `context.insider_trades` | window 90 -> 120 days (insider_activity) | **gap** | fixed: `test_the_window_is_the_last_ninety_days_and_no_more` |
| 26 | `context.track_luck` | years of 250 trading days, not 252 (track_record) | **gap** | fixed: `test_the_years_to_two_are_counted_in_years_of_252_trading_days` |
| 27 | `context.track_luck` | coin-toss tail "more than k" instead of "at least k" | caught | - |
| 28 | `history.run_overview` | rank 25 at the baseline not counted as top 25 (run_overview) | **gap** | fixed: `test_rank_25_is_in_the_top_25_on_both_dates_and_26_is_not` |
| 29 | `history.run_overview` | category shares without the category weight | caught | - |
| 30 | `context.macro_series` | ten-year percentile counts no ties instead of half | caught | - |
| 31 | `context.macro_series` | 1m change against 31 days back, not 30 | caught | - |
| 32 | `context.expected_move` | report window 60 -> 45 days (context_signals) | **gap** | fixed: `test_expiry_rule_holds_at_its_stated_edges` |
| 33 | `context.expected_move` | target expiry 30 -> 35 days (and -> 25) | **gap** | fixed: same test |
| 34 | `context.valuation_history` | market-value tolerance 15% -> 20% | caught | - |
| 35 | `overview.clamps_and_floors` | forward-EPS floor $1.00 -> $0.50 (factor_engine) | **gap** in the register; an unregistered test caught it | registered `tests/test_forward_eps_growth.py::test_a_small_base_uses_the_one_dollar_floor` |
| 36 | `overview.clamps_and_floors` | analyst-surprise floor $0.10 -> $0.05 (the fetch) | **gap** - no test in the whole suite | fixed: `test_the_fetch_applies_the_surprise_floor_the_page_states` |
| 37 | `overview.clamps_and_floors` | price-target clamp ceiling x1.5 | **gap** in the register; `tests/test_metrics.py::TestPriceTargetUpside::test_clamp_high` caught it | fixed: `test_the_engine_clamps_price_target_upside_where_the_page_says` |
| 38 | `overview.clamps_and_floors` | clamp read from a constant, not config (equivalent on today's config) | **gap** | fixed: same test, run on a different config |

## Results, 2026-10-25 (13:00 midday session)

The next six claims from the queue below, plus the three the same code paths pulled in.
**24 mutations, 11 claims. 15 caught; 4 real gaps (nothing in the suite saw them); 5 gaps in
the register (the suite had cover, `checked_by` did not name it); and one mutation that could
not be made, because the claim was already false.**

The harness gained one thing worth keeping: it now runs the claim's `checked_by` **and** the
whole of each module those tests live in, so it separates "nothing checks this" from "the
register does not know what checks this". They want different fixes and the 2026-10-20 pass had
to tell them apart by hand (its rows 35 and 37).

| # | Claim | Mutation (file) | Registered checks | Now |
|---|---|---|---|---|
| 39 | `summary.earnings` | same-day date dropped: `days < 0` -> `<= 0` (stock_summary) | **gap in the register** - `test_near_horizons_read_naturally` caught it | **the claim was false** - corrected; that test and `test_past_dates_are_dropped_never_relabelled` registered |
| 40 | `summary.weakest` | `LARGEST_SIZE_SCORE` 10 -> 11 (stock_summary) | **gap** | fixed: `test_the_largest_company_bar_holds_at_ten_and_not_above_it` |
| 41 | `summary.weakest` | the sector-median bar `n >= 50` -> `n > 50` | **gap** | fixed: `test_the_gloss_is_said_below_fifty_and_not_at_fifty` |
| 42 | `summary.drivers` | sorted by category score, not by contribution | caught | - |
| 43 | `summary.confidence` | configured stale-filing bar less 30 days (factor_engine) | **gap in the register** - `TestStalenessThreshold` caught it | registered both of its methods; `checked_by` taught `Class::method` |
| 44 | `overview.momentum_regime`, `context.macro_momentum_rule` | panic vol bar `vol > thr` -> `vol >= thr` (factor_engine) | **gap** | fixed: the strictness probe added to `test_the_threshold_is_strict_and_configurable` |
| 45 | same | vol threshold off config to a 0.19 constant | **gap in the register** | registered `test_the_readings_are_the_replays_definitions`, `test_the_threshold_is_strict_and_configurable`, `test_the_configured_thresholds_are_the_pre_registered_ones` |
| 46 | same | vol window 126 -> 120 trading days | **gap in the register** | same three |
| 47 | same | return window 24 -> 23 months | **gap in the register** | same three |
| 48 | same | PANIC scale 0.70 -> 0.75 | caught | - |
| 49 | same | freed weight split 2:1 to Quality, not equally (`apply_momentum_regime`) | caught | - |
| 50 | same | panic needs either condition, not both (`and` -> `or`) | caught | - |
| 51 | `overview.trap_rules`, `dashboard.trap_flag_key` | majority `>= 2` -> `>= 1` of three (factor_engine) | caught | - |
| 52 | same | value-trap cheap quantile shifted 5 points | caught | - |
| 53 | same | growth-trap quality floor off config, 35 -> 30 | caught | - |
| 54 | `overview.roic_invested_capital` | operating-cash share 2% -> 3% of revenue | caught | - |
| 55 | same | excess-cash cap 50% -> 60% of cash | caught | - |
| 56 | same | invested-capital floor 10% -> 12% of assets | caught | - |
| 57 | `drilldown.special_item_note` | materiality bar 10% -> 15% of net income (sec_crosscheck) | caught | - |
| 58 | `drilldown.capex_lease_note` | lease-share bar 10% -> 15% of capex | **gap in the register** - `test_the_threshold_is_the_registered_ten_percent` caught it | registered |
| 59 | `drilldown.capex_lease_note` | reconciliation tolerance 5% -> 8% | **gap** | fixed: `test_the_reconciliation_is_the_registered_five_percent` |
| 60 | `overview.bank_roe_roa` | bank ROE on ending equity, not the average of two | caught | - |
| 61 | same | bank ROA on ending assets, not the average | caught | - |
| 62 | same | `sustainable_growth`'s ROE on ending equity | caught | - |

### The claim that was already false (row 39)

`summary.earnings` asserted, and `_sentence_earnings`'s own docstring repeated, that **"a date
at or before the run date is dropped"**. No version of the code has done that: it drops
`days < 0` and renders `days == 0` as "the day of this run". The mutation that would make the
sentence true was therefore the one that *broke* the behaviour, and a test in the module -
`test_near_horizons_read_naturally["2026-09-29"]` - fails on it.

**The code is the right half, and the sentence was corrected.** A company reporting on the run
date genuinely is its next report, and by the paper the feature is built on (Akepanidtaworn,
Di Mascio, Imas & Schmidt, 2023, *Journal of Finance* 78(6)) announcement day is the one day on
which a sell beats its random-sell counterfactual, so dropping the date would withhold the
schedule exactly where the evidence says the schedule matters most. Nothing a reader sees
changed; what changed is that the register and the docstring now describe the code.

This is the route the method's rule 5 reserves for a false claim rather than a missing test, and
it is the second time the audit has found one: the first was 0.11's `context.expected_move`.

### What the 2026-10-25 gaps have in common

- **A tolerance read from the constant under test (59).** The only assertion about the capex
  note's 5% reconciliation was `entry["ppe"] + entry["purchases"] == approx(entry["capex"],
  rel=X.LEASE_RECONCILE_TOL)`. Widening the constant widens the assertion with it, so the test
  passes at any value. It is the self-consistency failure of rows 12-13 in a new shape: **a
  check that imports the number it is checking cannot see the number move.**
- **A test named for the property it does not test (44).**
  `test_the_threshold_is_strict_and_configurable` probed the volatility bar 0.001 above and
  0.001 below the reading and never *at* it - the single point that separates `>` from `>=`.
  The name made the gap invisible to a reader looking for one, which is worse than no test:
  three sessions could have read that name and concluded strictness was covered.
- **The register naming the weakest tests in a module (43, 45-47, 58).** Five of nine gaps were
  modules that *did* hold the right test. For `overview.momentum_regime` the three registered
  checks pinned 0.70, the equal split and the two-condition logic, while the 126-day window, the
  24-month window and the configured threshold - three of the four numbers the sentence names,
  and numbers CLAUDE.md 0.12 says may never be re-fitted - were held only by tests the register
  did not mention. The module runs in the publish gate either way (`module_paths()` is derived),
  so nothing was unprotected today; what was missing is the thing the register exists for -
  telling the *next* session which test is load-bearing for a public sentence before it edits it.
- **`Class::method` is now registrable.** The 2026-10-20 pass recorded the limitation and worked
  around it by writing module-level tests. Hitting it a second time (row 43: the two tests that
  pin the configured stale bar are the only ones in the suite that do, and both are methods)
  made fixing it cheaper than working around it. `_test_functions` in
  `tests/test_claims_register.py` now emits both forms; every other consumer of `checked_by`
  already takes `ref.split("::")[0]`. This also makes the good tests in `tests/test_metrics.py`
  registrable, which the last pass listed as a known cost.

### What the 2026-10-20 gaps had in common

- **Untested boundaries (8 of 15: #1, 3, 7, 16, 25, 28, 32, 33).** The test used one value far
  from the stated threshold - 2 and 20 analysts around a bar of 5, a report 27 days out
  against a 60-day window - so the threshold could move anywhere in a wide band. A claim that
  names a number needs a test on **both sides of that number**.
- **Checks that read the committed payload, not the code (#35-37).** The clamps-and-floors
  test compares the page with `dashboard_data.js`, which the code under test did not write. A
  code change passes gate 1, merges, and is first seen by the data loop's publish gate the
  next morning - which then refuses to publish every day until someone fixes it. The same
  shape as rule 10's "assert claims against the generator's output, not the committed file",
  one level down: **assert computations against the function, not the artifact it last wrote.**
- **Checks of self-consistency, not of the stated parameter (#12, 13).** The composite-chain
  tests rebuild each composite from the payload's own `disc` field, so they prove the page
  adds up but not that the discount is the one the page states. A payload-reproducibility
  test cannot see a wrong constant; it can only see a wrong sum.
- **Tolerance looser than the claim (#26).** `abs=0.1` on a figure near 1.5 years cannot see
  a 0.8% error. Pick a case where the stated arithmetic moves the published digits.
- **`checked_by` cannot name a class method.** `test_every_checked_by_test_exists` splits
  on the first `::`, so `tests/test_metrics.py::TestPriceTargetUpside::test_clamp_high` is
  unregistrable. The good tests in `tests/test_metrics.py` are invisible to the register and
  to the data loop's publish gate. Either teach the register `Class::method`, or write
  module-level tests for claims (done here).

## Results, 2026-10-26 (owner-run night): the two mechanical searches

Items 5 and 6 of the list below, done as searches rather than claim by claim. **45 mutations
across 25 claims. 24 caught; 10 real gaps (nothing in the claim's registered tests or their
modules saw them); 11 gaps in the register; and one register defect that stopped three registered
checks being run at all.** Every gap now fails a registered check: each new test was shown to
pass on the unmutated tree and fail under its mutation, and every mutation was re-run after the
fix and required to come back `CAUGHT` (the 2026-10-25 rule).

**Item 6, the row-59 shape.** An AST walk over `tests/` (not grep: it resolves each module
alias and `from m import X`, imports the module and keeps only attributes that are numbers)
listed every comparison or `approx(rel=/abs=)` in a *registered* test that reads a numeric
constant from the code under test: 20 assertions, 21 distinct constants once uses outside an
assertion are counted. Run over every test, it found 63, and 11 more constants that a registered
claim states. Each of the 32 constants was then mutated and the claim's `checked_by` run.
**5 real gaps, 4 gaps in the register, 23 caught.** The search over-reports by design: most
constants it finds are also pinned by a literal or a behavioural probe elsewhere in the
register, which only the mutation can tell. So the scan is not committed as a test-of-tests -
a rule "never compare against the module's constant" would fail on assertions that are
legitimate *beside* a literal pin (the capex reconciliation test checks the decomposition at the
code's own tolerance, correctly, and row 59's literal test sits next to it), and an allowlist
for those would be the next thing to drift. The question it raises has to be answered by a
mutation each time.

**Item 5, payload-only checks.** The same walk followed each registered test through the
module-level helpers and fixtures it reaches and marked it if any of them reads
`dashboard_data.js` / `dashboard_context.js`. **Six claims had no other kind of check:**
`compare.composite_gap`, `dashboard.score_key`, `drilldown.inputs_rebuild`,
`drilldown.piotroski_beneish_parts`, `overview.eps_basis` and `overview.data_source`. Five were
real (12 mutations, rows 71-73 and 76-84: 5 gaps, 7 gaps in the register); `overview.data_source` was not - its one check also
reads `factor_engine`'s source for the values the fetch writes, and renaming one fails gate 1
(row 88). Every new check runs the engine or the generator on fixtures, needs no network and
no run directory, and writes only to `tmp_path`, because registering it puts it in the data
loop's publish gate.

| # | Claim | Mutation (file) | Registered checks | Now |
|---|---|---|---|---|
| 63 | `drilldown.composite_chain`, `compare.composite_gap` | points-add-up tolerance `CONTRIB_SUM_TOL` 0.05 -> 0.5 (calc_trace) | **gap** - nothing in the suite; its one reader compared with the constant | fixed: `test_points_that_miss_by_more_than_rounding_are_caught` (0.04 accepted, 0.06 refused: eight 2dp roundings miss by at most 0.04) |
| 64 | `context.price_trend` | `SMA_LONG` 200 -> 150; `SMA_SHORT` 50 -> 40; `SLOPE_LAG` 20 -> 21; `VOL_SHORT` 20 -> 21; `VOL_LONG` 63 -> 60 (context_signals) | caught, all five | - |
| 65 | `context.put_skew` | `XZZ_PUT_TARGET` 0.95 -> 0.90; `XZZ_PUT_LO` 0.80 -> 0.85; `XZZ_PUT_HI` 0.95 -> 0.96; `XZZ_CALL_HI` 1.05 -> 1.10; `XZZ_DAYS_LO` 10 -> 7; `XZZ_DAYS_HI` 60 -> 90 | caught, all six | - |
| 66 | `overview.workbook_sheets` | `DASHBOARD_TOP_N` 25 -> 20 (portfolio_constructor) | **gap** - page text and row count both built from the constant | fixed: `test_the_workbook_the_run_writes_holds_the_top_25_and_validates_the_top_10` |
| 67 | same | `DATA_VALIDATION_TOP_N` 10 -> 12 | **gap** - the test passed its own `top_n` to the sheet writer | fixed: same test, through `write_full_excel` into `tmp_path`; `write_full_excel` passing a literal `top_n=12` is caught too (run after the fix only) |
| 68 | `overview.auto_disable` | `CATEGORY_AUTO_DISABLE_PCT` 30 -> 40 (run_screener) | **gap** - probes at 20% and 45% straddle 40 too | fixed: `test_the_auto_disable_bar_is_thirty_percent_and_thirty_itself_is_kept` (29.9% off, 30.0% kept, both categories); `pct >= threshold` -> `>` is now caught as well (not measured before the fix) |
| 69 | `overview.weight_sensitivity` | `SENSITIVITY_PERTURBATION_PTS` 5 -> 10 | caught | - |
| 70 | same | `SENSITIVITY_TOP_N` 20 -> 25 | **gap** | fixed: `test_weight_sensitivity_compares_the_top_twenty` |
| 71 | `overview.eps_basis` | fallback's ratio rule 2.0x -> 2.5x (factor_engine) | **gap** - its one check reads the payload | fixed: `test_the_engine_flags_outside_03_to_2x_and_drops_growth_only_on_the_fallback` |
| 72 | same | the ratio rule applied to the twelve-month construction as well | **gap** | fixed: same test |
| 73 | same | the EPS-basis flag's bar 2.0x -> 2.5x | **gap** | fixed: same test |
| 74 | `context.put_skew` | `XZZ_CALL_LO` 0.95 -> 0.90 | **gap in the register** - only `test_option_implied_claims.py::test_the_constants_are_the_papers_and_the_code_uses_them`, a literal comparison that tests no use | fixed: `test_skew_call_band_is_095_to_105_inclusive_and_nearest_one`; the literal test renamed `test_the_constants_are_the_papers` (rule below) and registered |
| 75 | same | `XZZ_CALL_TARGET` 1.0 -> 1.02 | **gap in the register** - the same literal test and `test_skew_picks_the_nearest_eligible_strike_after_filtering` | fixed: same test; the inclusive call band made exclusive is now caught too |
| 76 | `drilldown.piotroski_beneish_parts` | Beneish's SGI coefficient 0.892 -> 0.982 (factor_engine) | **gap** - both checks read the payload, and nothing else in `tests/` or `test_screener.py` went red (the whole suite was run under this mutation) | fixed: `test_the_engines_beneish_indices_rebuild_its_m_score_with_the_published_weights` |
| 77 | same | Piotroski score scaled to nine, `round(f * 9 / n_testable)` | **gap in the register** - `test_metrics.py::TestPiotroskiFScore::test_partial_data_not_inflated` | fixed: `test_the_engines_piotroski_signals_add_up_to_its_score`; `test_partial_data_not_inflated` registered |
| 78 | `dashboard.score_key` | `size_log_mcap = log(mc)`, the sign dropped | **gap in the register** - `test_metrics.py::TestSizeLogMcap` | fixed: `test_the_engine_gives_the_largest_company_in_a_sector_the_lowest_size_score` |
| 79 | same | Size's direction `True` -> `False` (`METRIC_DIR`) | **gap in the register** - only the golden-file snapshot saw it | fixed: same test |
| 80 | same | `kpis.median_composite` from `.median()` -> `.mean()` (generate_dashboard) | **gap** - no other test reads `median_composite` | fixed: `test_the_median_composite_the_key_quotes_is_the_median_not_the_mean` |
| 81 | `drilldown.inputs_rebuild` | earnings yield over enterprise value, not market cap (compute_metrics) | **gap in the register** - `TestEarningsYield` (test_metrics); in `test_metric_lineage.py` only the new test fails | fixed: `test_the_engines_inputs_rebuild_its_values_on_the_fixture_universe` (22 metrics on the ten-stock fixture) |
| 82 | same | accruals with the sign turned | **gap in the register** - `TestAccruals` | fixed: same test |
| 83 | same | asset growth over this year's assets | **gap in the register** - `TestAssetGrowth` | fixed: same test |
| 84 | `compare.composite_gap` | contributions not renormalised over the categories present (`w / 100`, `compute_factor_contributions`) | **gap in the register** - `test_composite_and_points_reproduce_including_the_coverage_discount`, registered only for `drilldown.composite_chain` | registered |
| 85 | `drilldown.special_item_note` | `SPECIAL_ITEM_SHARE_MIN` 0.10 -> 0.15 (sec_crosscheck) | **gap in the register** - `test_the_threshold_is_the_registered_ten_percent` was unregistered and the registered payload test skips without the SEC cache (row 57's catch exists only where the cache does) | registered |
| 86 | `overview.share_classes` | `CHALLENGE_MARGIN` 0.20 -> 0.25 (share_classes), and `challenge_margin` 0.20 -> 0.25 (config.yaml) | **could not be run**: three registrations named class methods by their bare name, which pytest cannot select; then **gap in the register** - `TestDeclaration::test_score_once_is_on_and_the_margin_is_spdji_s` | names fixed (and `overview.effective_weights`' two), the register tightened, the test registered; both mutations caught |
| 87 | re-probed at the constant | `ROIC_OPERATING_CASH_SHARE`, `ROIC_EXCESS_CASH_CAP`, `ROIC_IC_FLOOR_SHARE`; `SECTOR_MIN_PEERS` 10 -> 8; `run_overview.TOP_N` 25 -> 20; `LEASE_SHARE_MIN`, `LEASE_RECONCILE_TOL`; `THIN_ANALYST_COUNT` 5 -> 6; `MIN_INPUT_CHURN` 2 -> 3; `MCAP_TOLERANCE` 0.15 -> 0.20; `HLZ_T` 3.0 -> 2.5 | caught, all eleven | - |
| 88 | `overview.data_source` | `_data_source` "annual" -> "yearly" (factor_engine) | caught - not a gap; the payload scan's flag was a false positive | - |

### What the 2026-10-26 gaps have in common

- **The page and the probe built from one constant (66-68, 70).** Every one of these tests did
  the work - real behaviour, often both sides of the bar - but derived its expected page text
  *and* its probes from the constant: `thr = rs.CATEGORY_AUTO_DISABLE_PCT`, then "fewer than
  {thr}%" on the page and coverage 20% and 45% around it. The generator reads the same constant,
  so a moved constant moves the page, the code and the test together and all three agree. The
  public page stays *consistent* - which is why this shape survives review - but the methodology
  changes with no failing test and no changelog entry (rule 3), and the register's own sentence
  goes false. Writing the number once as a literal fixes all three.
- **A test that passes its own argument (67).** `write_data_validation_sheet(wb, df, top_n=n)`
  checks the sheet writer with the count the test chose; the run calls it from `write_full_excel`
  with whatever that function passes. Go in through the function the run calls.
- **Payload-only checks hide engine changes until the morning (71-73, 76, 80).** Same lesson as
  rows 35-37, now measured as a class: six claims, five real, 13 mutations, none seen at gate 1.
- **The register naming tests it cannot run (86).** `_test_functions` collected names with
  `ast.walk`, so a method's bare name "existed"; five registrations used that form and
  `pytest file::bare_name` reports "not found". The data loop's gate runs whole modules, so
  nothing went unprotected - but the register's promise that a named check can be run by its
  reference was false, and the mutation harness is the first thing that took it literally. The
  existence check now accepts only selectable names (proved: restoring one bare name fails
  `test_every_checked_by_test_exists`).

## Results, 2026-10-28 (13:00 midday session): the research surface, and the last three residuals

Items 4, then 1-3 of the list below. **17 mutations across 6 claims. 11 caught, 6 gaps - and
four of the six were caught by nothing anywhere in the suite.** Every gap now fails a registered
check, each shown to pass unmutated and fail under its mutation, and every mutation re-run
afterwards and required to come back `CAUGHT` (the 2026-10-25 rule). No production code changed.

**The shape of the research surface's gaps, which is new and worse than rows 71-73.** Those rows
found claims whose only checks read the committed payload, so an engine change was invisible
until the morning. Here the only checks that saw six of the eight research-surface gaps were two
*drift* tripwires - `test_the_committed_page_is_what_the_generator_makes_from_the_ledger` and
`test_the_methodology_page_quotes_the_engine_s_own_summary`, both of which assert that the
committed artifact equals a regeneration. **A drift check cannot fire in the state the change
actually ships in.** Measured rather than argued: each mutation was re-run with
`python v2_experiments.py report` in between - the way `research.html` and `summary.json` are
published - and **all seven gaps stayed green**, because the page, the summary and the test then
agree with each other and with the mutated code. That is CLAUDE.md rule 10 at the level of a
claim: *assert the claim against the generator's arithmetic, not against the artifact.*

**And the register's own weakest form, twice over.** `research.plain_answer`'s single check opened
with `every_zero = all(lo <= 0 <= hi for lo, hi in ci.values())` - the code's rule, restated - and
then asserted the page agreed with it; `overview.research_engine`'s read `published["better"]` out
of the same function that computes it. A check written that way is a consistency check between the
code and itself. It is row 59's rule ("a check may not import the number it is checking") applied
to a *rule* rather than a constant, and it is harder to see, because nothing in the test looks
like a constant.

**Four of the six gaps change no published figure today** (92, 97, 102, 103): the boundary at
zero, the `or` deciding what "cleared its bar" means, the Sahm bar at exactly 0.50, and the
window that makes the Sahm low a *prior*-12-month low. Each is equivalent on today's data and
false about the next value - which is the Method's "equivalent mutants" note turning out to be
the majority case, not the footnote. A fixture chosen to exercise the sentence rather than the
code is what separates them: every Sahm fixture in the register was a *rising* unemployment
series, where the prior-12 low and the inclusive-12 low are the same month.

| # | Claim | Mutation (file) | Registered checks | Now |
|---|---|---|---|---|
| 89 | `research.positive_control` | `detectable`'s 80%-power quantile, `t_critical(0.40, df)` -> `(0.20, df)` (v2_experiments) | **gap in the register** - the three checks assemble the sentence from the row's fields and never ask what the figure means; `test_the_bar_at_38_held_out_months...` (unregistered) sees it | registered that test, `test_the_detectable_figure_is_honest` and `test_every_verdict_says_how_small_a_difference_it_could_see`; `tests/test_engine_calibration.py` was named by **no** claim, so the publish gate had never run it |
| 90 | same | the power term dropped: `(pass_bar + power) * se` -> `pass_bar * se` | **gap in the register** - same three; caught by the calibration module's behavioural check (plant 2x, it must be seen) | same; also now pinned as arithmetic by row 93's new test (`pooled_detectable == 0.0029`, not 0.0023) |
| 91 | same | `df = (n - 1) if n > 1 else None` -> `df = None` (the normal curve for the bar and the power) | **gap in the register** | same registration |
| 92 | `research.plain_answer` | `all(lo <= 0 <= hi ...)` -> `all(lo < 0 < hi ...)` (research_page) | **gap - nothing in the suite**, and nothing after a regeneration either | fixed: `test_an_interval_that_only_just_includes_zero_still_includes_it` - endpoints exactly on zero both ways keep the line, +0.0001 ends it. The published interval is rounded to 4dp, so an endpoint *of exactly 0.0000* is a real state, not a contrivance |
| 93 | same | each era's interval at 10% instead of 5%: `t_critical(0.05, st["n"] - 1)` -> `(0.10, ...)` (summary_for_page) | **gap** - only the two drift tripwires, and only while the page is stale | fixed: `test_the_intervals_behind_the_plain_answer_are_95_percent_over_the_whole_record`, on a synthetic baseline against quantiles written as literals (2.0262 on 37 df, 1.9905 on 79, 1.9736 on 175) |
| 94 | same | the pooled interval at 10% | **gap** - as above | fixed: same test |
| 95 | `overview.research_engine` | `pass_bar` without Bonferroni: `t_critical(0.05 / max(n_tests, 1), ...)` -> `t_critical(0.05, ...)` | caught | - |
| 96 | same | `n = len([e for e in reg if kind != "baseline"])` -> `len(reg)` | caught | - |
| 97 | same | `cleared = v.endswith(...) or vp.endswith(...)` -> `and` | **gap - nothing in the suite.** No alternative clears either judgement today, so the mutation moves no figure; it changes what the first one to clear would be counted as | fixed: `test_an_alternative_that_cleared_the_bar_in_either_judgement_is_counted` (one row clearing by each route, both counted) and `test_a_comparison_that_cleared_nothing_is_counted_in_neither_column` |
| 98 | `research.plain_answer` | the pooled record's start, `PRESAMPLE[0]` -> `SAMPLE[0]` (176 months -> 80) | **gap** - the two drift tripwires only | fixed: row 93's test, with a month either side of the record that no interval may include |
| 99 | `context.put_skew` | the paper's window flag, `XZZ_DAYS_LO <= days <= XZZ_DAYS_HI` -> `<` both sides | caught | - |
| 100 | same | the nearest-target pick with `.abs()` dropped (`_xzz_leg`) | caught | - |
| 101 | same | the put leg's target moved to the band's midpoint, 0.95 -> 0.875, leaving the call leg alone | caught | - |
| 102 | `context.sahm` | the trigger, `sahm >= 0.5` -> `> 0.5` (market_context.readings) | **gap - nothing in the suite**: the only reading ever handed to `readings` is 0.6 | fixed: `test_the_sahm_trigger_is_half_a_point_and_half_a_point_triggers_it` (0.49 / 0.50 / 0.51) |
| 103 | same | `low12 = avg3.shift(1).rolling(12).min()` -> `avg3.rolling(12).min()` (the low stops being *prior*) | **gap - nothing in the suite.** Both fixtures for this claim - including the one in `test_club_readiness` that restates the window as `avg3.iloc[-13:-1]` - are rising series, where the two windows have the same minimum | fixed: `test_the_sahm_low_is_the_prior_twelve_months_and_not_this_one`, on a falling series: -0.1 against the mutation's flat 0.00, which is also the state in which the mutated code could never print a negative reading at all |
| 104 | `context.track_turnover` | `0.5 * sum(abs(...))` -> `sum(abs(...))` (one-way turnover doubled) | caught | - |
| 105 | same | the break-even's `1 - 2 c TO` -> `1 - c TO` (cost charged on one side) | caught | - |

## Results, 2026-10-29 (06:00 session): three sentences that were not claims at all

The 2026-10-28 pass found the research page's "How long would it take to prove an edge?" section
printing **three computational sentences that `claims.py` did not register** and left them as item
3 of its next list, with the work specified. They are registered now - `research.ic_swing`,
`research.months_to_prove`, `research.detectable_difference` - each with checks on both sides of
its number, and then mutated like any other claim.

**12 mutations, 12 caught.** That is the expected result for tests written the same day and is not
the measurement worth keeping. This is:

**Before today, three of these mutations - one per claim - pass the entire suite.** Not "pass the
claim's checks", which they had none of: mutate, run `python v2_experiments.py report` the way the
page is actually published, then run `tests/ test_screener.py` with only today's four new tests
deselected. **3,506 passed, 1 skipped, three times out of three.** So these were not
under-checked sentences; nothing anywhere in the project could see them change.

| # | Claim | Mutation (`research_page.py`) | Before today | Now |
|---|---|---|---|---|
| 106 | `research.ic_swing` | the swing taken over `presample` instead of `all` (0.10 -> 0.12, and all three month counts with it) | **blind - the whole suite, 3,506 passed** | caught: `test_the_monthly_swing_is_the_standard_deviation_of_the_2020_2026_record` |
| 107 | same | `se * math.sqrt(n)` -> `se / math.sqrt(n)` | no claim | caught: same test |
| 108 | same | `se * math.sqrt(n)` -> `se * n` | no claim | caught: same test |
| 109 | `research.months_to_prove` | `((z + 0.84) * ic_sd / ic) ** 2` -> `((z) * ...)`: the 80%-power term dropped | caught, by an **unregistered** test | caught: `test_the_months_to_prove_an_edge_are_that_swing_at_eighty_percent_power`, and the registered `test_signs_and_zero_are_printed_honestly` |
| 110 | same | `int(round(x))` -> `int(x)` | caught, by the same unregistered test | caught: same two |
| 111 | same | the 5%-level stat printing `months_needed(0.05, ic_sd, 3.0)` | no claim | caught: the stat/label pairing |
| 112 | same | the 0.03 stat printing `months_needed(0.05, ic_sd, 3.0)` | **blind - the whole suite, 3,506 passed** | caught: the stat/label pairing |
| 113 | `research.detectable_difference` | `sorted(rows, key=...["p"]["se"])[len // 2]` -> `rows[len // 2]`: the middle of the *mean* ordering | no claim | caught: both new tests |
| 114 | same | `[len(rows) // 2]` -> `[0]`: the smallest standard error, not the median | no claim | caught: both |
| 115 | same | `mid_h = mid_p`: the held-out figure built from the 2012-2019 comparisons | **blind - the whole suite, 3,506 passed** | caught: both |
| 116 | same | `detectable(mid_h["se"], n, mid_p.get("n"))`: the right standard error over the wrong month count | no claim | caught: the spy test |
| 117 | same | `detectable(mid_p["se"], 1, ...)`: no correction for the number of comparisons | no claim | caught: the literal-figure test |

### What this pass adds to the method

- **An unregistered test is not nothing, and it is not coverage either.** Rows 109 and 110 were
  *caught* before today, by `test_signs_and_zero_are_printed_honestly` - which no claim named, so
  the publish gate never ran it. The formula was watched; **which figure goes with which label was
  not** (111, 112), and that is the half a reader actually reads. Both halves are registered now.
- **The check that caught them was itself the row-59 shape.** Its assertion read
  `rp.months_needed(0.05, 0.12, 3.0) == round(((3.0 + 0.84) * 0.12 / 0.05) ** 2)` - the function's
  own expression copied - so it passes whatever that expression becomes. It survived mutations 109
  and 110 only because they changed the function *around* the copied line (`int(round(` and the
  `+ 0.84`) rather than the line itself. A mutation of `** 2` -> `** 2.0`, or of both sides
  together, would have gone straight through. It is literals now: 85 and 45.
- **`claims.RESEARCH` is a `detect` surface now**, and it is the **generated** page, not the
  committed one (`generated_research` in `tests/test_claims_register.py`). The register could not
  carry a pattern for a research-page sentence before today, because
  `test_every_detect_pattern_still_matches_its_surface` had text for only two surfaces and asserts
  `text is not None`; both pre-existing research claims legitimately have `detect=None` (each is
  one of three mutually exclusive states), so nothing had forced the question. A reword now goes
  stale on the session that makes it.
- **The fixture is the real generator with hand-chosen inputs.** Every alternative's comparison
  statistics are replaced in a deep copy of the committed ledger - a plateau of standard errors
  wide enough to cover the median index whatever the row count, two rows below it, two above, and
  a distinctive value on the row that sits in the middle of the *mean* ordering - and everything
  else on the page (the registry, the control, the bot rule, the coverage) is left real. So the
  tests exercise the function the data loop calls, and the median pick they assert cannot be
  satisfied by the mean, the smallest, the largest, or the row `build` happens to have at that
  index. The injected `detectable` returns a value encoding **both** its arguments, which is what
  makes "the wrong period" and "the wrong month count" separately visible (0.400/0.603 against
  0.600/0.403 and 0.403/0.600).

### What the 2026-10-28 gaps have in common

- **A check that restates the rule and then asks the page to agree (92-94, 97, 98).** Not a
  constant read into an `approx` (row 59) and not a page built from a constant (rows 66-70) - the
  *decision* recomputed. `every_zero = all(lo <= 0 <= hi ...)` in the test is the code's line
  copied; `published["better"]` is the function's own answer. Write the rule's inputs by hand and
  the expected outcome as a literal, or the test can only prove the page is self-consistent.
- **A drift tripwire is not a check of a claim (89-91, 93, 94, 98).** It fires on any change and
  on nothing in particular, and it is green in exactly the state the change ships in. Where a
  claim's surface is generated, at least one of its checks must call the generator with inputs the
  test chose.
- **The fixture, not the assertion, is what was missing (102, 103).** Both Sahm checks did real
  arithmetic, one of them restating the window independently, and both were blind - because every
  fixture was a rising series. When a claim names a window, a boundary or a direction, the fixture
  has to be the case where getting it wrong shows.
- **A module can hold the best evidence in the project and be outside the gate.**
  `tests/test_engine_calibration.py` - seven synthetic tests that are the reason anyone should
  believe the engine can measure what it reports - was named by no claim, so
  `check_published_claims.py` had never run it before publishing the page that quotes it. 23s,
  synthetic, fixed seed, no network.

## Results, 2026-10-29 (13:00 session): the sweep, and what it costs to find a sentence by reading

The three sentences registered this morning were found by **reading one section of one page**, and
the morning's own measurement was that each was invisible to the whole suite. That is not a method -
it is luck plus attention, and this audit cannot reach the class at all, because **mutation can only
mutate what the register names.** This pass builds the other direction, item 3 of the 2026-10-28
next list.

**`tests/test_claims_register.py` section 5 sweeps `research_page.py` the way
`test_every_sentence_builder_is_registered` sweeps `stock_summary.py`.** It enumerates every
sentence the generator prints that puts a number *this repo computed* into prose - 32 of them - and
requires each to carry a verdict: `claim:`, `checked:`, `reported:` or `open:`. A new or reworded
one fails the suite until it gets one.

**The scope rule, stated so it cannot drift.** A sentence is in scope when it interpolates a value
the generator formats as a number (a numeric format spec) or computes (arithmetic, or a call to one
of its own statistics helpers). A figure read verbatim out of the ledger or the registry with no
formatting is a *reported* number - the page-matches-its-source tests cover those, and `reported:`
records which. `checked:`/`reported:` mean **every** computed figure in the sentence is pinned by
the named test; one unpinned figure makes the whole row `open:`, because a row that reads as
coverage it does not have is the failure the register exists to catch. Every verdict was assigned by
reading the named assertion.

**Two implementation facts worth keeping**, both found by the sweep's own output being wrong:

- **Inline tags are inside a sentence; every other tag ends it.** A figure is wrapped in
  `<span class="num">`, so breaking on all tags separates every number from its words - and
  stripping all tags without breaking produced a single key running from the chart legend through
  the whole alternatives table to the next heading.
- **A stat card is a figure in one element and its caption in the next.** With block-level
  breaking, `<div class="v">{months_needed(...)}</div><div class="k">months of record to show
  ...</div>` leaves a figure with no words and a caption with no figure, so **all three month
  counts registered this morning vanished from the sweep.** A figure-only unit therefore takes the
  following unit's words.

**The verdicts: 12 of the 32 are `open:`** - at least one computed figure pinned by nothing. They
are listed by **literal slug** in `KNOWN_OPEN`, not by a comprehension over the table, which would
be the table compared with itself (row 59 applied to this file's own ratchet). The set only ever
shrinks.

**One closed here: the universe paragraph**, named as a residual by the 06:00 session.
`research.pairing_overlap`, five checks. Until today
`test_the_page_says_the_pairing_is_on_the_shared_stocks_and_states_the_universe_difference` pinned
one of its four figures (the baseline's stocks a month) and the words around the other three.

**9 mutations, 9 caught by today's tests, and 9 of 9 blind to everything else.** Same method as the
morning's: apply one exact single-occurrence replacement, regenerate `research.html` with
`python v2_experiments.py report` the way the page is actually published, then run the whole suite
with only today's eight new tests deselected. **3,515 passed, 1 skipped, eight times out of eight.**

| # | Claim | Mutation | Before today | Now |
|---|---|---|---|---|
| 118 | `research.pairing_overlap` | `only_narrow = m_n - c` -> `b0["mean_stocks"] - c`: the stocks unique to the screen become the stocks only the baseline can score | **blind - the whole suite, 3,515 passed** | caught: `test_a_near_subset_says_which_side_the_pairing_holds_and_what_it_gives_up` |
| 119 | same | the same two subtractions swapped the other way | **blind - 3,515 passed** | caught: same test |
| 120 | same | `c >= 0.99 * m_n` -> `c > 0.99 * m_n`: the near-subset branch turns strictly above 99% instead of at it | **blind - 3,515 passed** | caught: `test_the_near_subset_branch_turns_at_ninety_nine_percent_of_the_screens_universe` |
| 121 | same | `min(narrow)` -> `max(narrow)`: the page names the *widest* alternative as the narrowest | **blind - 3,515 passed** | caught: `test_the_universe_paragraph_names_the_narrowest_screen_and_both_its_own_figures` |
| 122 | same | the `X.SAMPLE` filter dropped: the overlap averaged over every month on file, not the record | **blind - 3,515 passed** | caught: `test_the_overlap_is_the_mean_of_the_screens_own_monthly_counts_inside_the_record` |
| 123 | same | `sum(common) / len(common)` -> `max(common)`: the overlap is the best month, not the mean | **blind - 3,515 passed** | caught: same test |
| 124 | the sweep itself | a **new** computed-number sentence appended to the published paragraph | **blind - 3,515 passed** | caught: `test_every_computed_sentence_on_the_research_page_has_a_verdict` |
| 125 | the sweep itself | a registered sentence **reworded** ("The two had" -> "The two shared") | **blind - 3,515 passed** | caught: same test |
| 126 | the sweep itself | `_INLINE = frozenset()`: the sweep stops matching and would pass vacuously | n/a | caught: the `>= 25` floor, with its own message |

**Row 124 is the finding.** Adding a brand-new sentence that states the arithmetic of a published
figure is, before today, invisible to 3,515 tests. That is the class this audit could not reach,
measured rather than argued, and it is now the one thing the sweep makes impossible.

**Row 125 is why the key is the sentence text with its expressions in place** rather than a `{}`
placeholder template. A reword is blind to everything else even after the page is regenerated - the
drift tripwire is green by then (the 2026-10-28 lesson) - and a placeholder key would also miss
swapping *which* expression is printed, which is exactly how row 106 took a figure off the wrong
era.

**What the 12 open rows are**, as a queue for later sessions (slug, then what is unpinned):

- `pass-bar-printed`, `pass-bar-under-chart` - the page's own `bar_p`/`bar_h`, printed twice.
  `test_the_bar_at_38_held_out_months...` pins `X.pass_bar` as a function, not that the page calls
  it with the median row's month count per era, and nothing pins that the two printings agree.
- `chart-axis-and-geometry` - the axis ticks and, in the same block, `_pos`. A sign error there puts
  every dot on the wrong side of zero with every printed number still correct.
- `size-row-mean-t-and-bar-word`, `size-coverage-percentages` - the Size row's own mean and t, the
  "short of"/"beyond" word at |t| = bar, and the two yearly coverage percentages.
  `test_the_size_sentence...` recomputes `r_time` and both within-era correlations, and not these.
- `pbo-all-era-prose`, `pbo-recent-era-prose`, `pbo-without-winner-prose`, `pbo-top-winner-lead`,
  `pbo-configuration-count` - the overfitting section's prose copies of its figures (the cards are
  pinned, the prose is not), the out-of-sample shares, the top winner's two era means, and
  `_pbo_words`, which is probed at 0.21/0.5/0.73 and **never at its own bars 0.45 and 0.55**.
- `bot-monthly-excess-figure` - `test_the_trading_rule_sentence_follows_each_period_s_sign_and_t`
  matches the percentage as `[0-9.]+%`, so the monthly figure is a wildcard while its sign, words
  and t are pinned.
- `bot-coverage-by-year` - the first and last year's mean coverage under the trading-rule table.

## Results, 2026-10-30 (13:00 session): the sweep over the public methodology page

Item 0 of the 2026-10-29 next list, one surface as that list requires. `SCREENER_OVERVIEW.md` is
the public methodology page and is **generated on every full run**, so the 02:00 loop publishes
whatever the generator says five mornings a week.

**`tests/test_claims_register.py` section 6 sweeps `run_screener.build_screener_overview`.**
Scoped to the *generator*, not the file: `run_screener.py` is the pipeline, and a file-wide sweep
enumerates progress lines as published sentences (`computed_sentences(..., within=...)`, with a
test that fails loudly if the generator is renamed rather than silently sweeping nothing).

**The yield is 11 sentences: 4 `checked:`, 7 `open:`.** Smaller than the research page's 32,
because most figures here are config constants printed verbatim - *reported* under the scope rule,
and already pinned against config by the overview-claims modules.

**Two markdown facts, both found by this sweep's own output being wrong, and both now in
`computed_sentences`' docstring:**

- **A newline is a block boundary in markdown where in html it is whitespace.** The html sweep
  breaks on block-level tags; markdown has none to break on, so the first run produced one key
  running from a three-row table straight through the sentence after it. This is safe only because
  a generator's newlines are the ones it *writes* - prose wrapped across source lines by implicit
  concatenation carries no newline into the output - so no sentence is split by the rule.
- **A markdown table row is not a sentence, and must not be merged into one.**
  `| Earnings yield (E/P) | **+{bank_net_ep:.1f}** |` has a figure and no three-word run (the
  "(E/P)" breaks `_PROSE`), so the stat-card merge rule treated it as a figure-only unit and
  chained the separator and the next two rows into a single "sentence". Table cells were already
  outside the sweep on the html side by the same reasoning, and **both surfaces now need the same
  decision about whether the register covers cells** - the bank coordinate table's three figures
  are in that class, and they *are* pinned, cell by cell, by
  `test_the_first_order_arithmetic_the_page_prints_comes_from_the_config`.

**The 7 open rows**, as a queue, with what each needs:

- `bank-five-metric-point-total`, `bank-pb-offset-by-roe-roa`, `bank-pb-residual-clause` - the
  bank P/B-ROE section's **prose** restatements of its table. The table's three cells are pinned
  against an independent recomputation from config; the prose sum of five weights, the
  "P/B's X offset by ROE's Y and ROA's Z" line and the residual clause are read by nothing. The
  clause also prints a **word** chosen on `bank_net_pb < 0`, and nothing probes that boundary -
  the same shape as `_pbo_words` never being probed at its own bars.
- `clamp-bullet-second-printing` - **the sharpest one.**
  `test_the_clamps_and_floors_on_the_page_are_the_ones_applied` says in its own docstring that
  "the metric rows hard-code their clamps; the Data quality bullet reads config. Both must be
  config's", and then matches only the metric-table rows. The bullet is the second printing of all
  four figures and nothing reads it, so the two disagreeing would publish. Same shape as
  `pass-bar-printed`, and here a docstring asserts the coverage it does not have.
- `coverage-discount-worked-example` - the threshold (both printings) and the rate *are* pinned,
  and well: read off this sentence and then applied through `compute_composite` on stocks either
  side of the bar. The **worked example** that follows is a second derivation of the same formula
  (`_ex_k` of `_n_gen`, its coverage percentage, the discount it loses) and so is the deepest-cut
  figure; none of those five is read.
- `coverage-sentence-metric-counts` - `test_confidence_metric_count_is_the_discount_coverage`
  pins both counts against `weighted_metric_sets`, but on the page only as
  `f"{bank}" in generated_overview` - a bare substring anywhere on a 1,000-line page - and it does
  not look for the other-financials count (`len(_w_gen) - 1`) at all.
- `share-class-switching-margin` - the margin is pinned in `config.yaml` and in
  `share_classes.CHALLENGE_MARGIN`, both as literals, and **no test in the suite reads the figure
  off this page**: `tests/test_share_classes.py` never calls `build_screener_overview`.

**4 mutations. The two that matter are the sweep's own teeth, and both are blind to everything
else.** Method as before: one exact single-occurrence replacement, then regenerate the way the
publish does, then the whole suite with only today's four new tests deselected.

| # | Claim | Mutation | Before today | Now |
|---|---|---|---|---|
| 127 | the sweep itself | a **new** computed-number sentence appended to the markdown `build_screener_overview` returns | **blind - the whole suite, 3,566 passed** | caught: `test_every_computed_sentence_on_the_methodology_page_has_a_verdict` |
| 128 | the sweep itself | a registered sentence **reworded** ("those five metrics" -> "these five metrics") | **blind - 3,566 passed** | caught: same test |
| 129 | the markdown rules | `markup="markdown"`'s newline break removed, and the table-row skip removed | n/a | caught: `test_the_markdown_sweep_breaks_on_blocks_and_skips_table_rows`, which asserts the exact key both ways |
| 130 | the sweep's scope | `within=` pointed at a renamed generator | n/a | caught: `test_the_sweeps_scope_is_the_generator_not_the_file`, plus the `>= 9` floor |

**What generalises, and it is a correction to this file's own method.** Row 128 was *first*
measured as **caught**, by `tests/test_rebuild_page.py::test_committed_page_is_what_the_generator_builds` -
and that was an artefact of the harness. The methodology page is embedded in `index.html`, so
regenerating `SCREENER_OVERVIEW.md` alone leaves the page drifted and a drift tripwire fires. On
the day the change actually ships the 02:00 run rewrites **both**, and the tripwire is green - the
2026-10-28 lesson, met again from the other direction. So:

> **Regenerate every artifact the publish rewrites, not just the one the generator is named for.**
> A mutation that reads as caught because a second artifact was left stale is a false negative for
> this audit: it says "something sees this" when nothing would on the day.

The harness now runs `scripts/rebuild_page.py` after rewriting the markdown, and restores
`index.html` and `dashboard.html` as well. Row 127 was re-run for the same reason: its first form
assigned the new sentence to an unused variable, so the sweep (which reads the AST) saw it while
the published page did not - which would have made "blind" trivially true. It is appended to the
returned markdown now.

**Nothing about scoring, no published text and no production code changed** - the diff is
`tests/test_claims_register.py`, this file and the log.

## Next claims (for the nightly sessions)

Take them in order; one session can do four to six. Record each in the table above with
the date, and move this list down.

**The 2026-10-20 list's items 1-8 and 12 are done** (2026-10-25, rows 39-62), **the two
mechanical searches are done** (2026-10-26, rows 63-88), **items 1-4 of the 2026-10-28 list are
done** (rows 89-105), **its item 3 - the research page's unregistered sentences - is done**
(2026-10-29 06:00, rows 106-117), **and item 3 below - the sweep for sentences that are not
registered at all - is done for `research_page.py`** (2026-10-29 13:00, rows 118-126). What is
left:

0. **`generate_dashboard.py` is the last of the three generated surfaces.**
   `build_screener_overview` is **DONE** (2026-10-30, rows 127-130: 11 sentences, 4 `checked:`,
   7 `open:`). `computed_sentences()` now takes `within=` and `markup=`; the dashboard is html, so
   `markup="html"` as the research page uses, and it is 13,485 lines, so expect the biggest table
   of the three - **decide up front whether to scope it with `within=` per generator function
   rather than sweeping the file**, which is how the overview stayed readable. The work is not the
   sweep, it is the verdicts: each one means reading the assertion that is supposed to pin the
   figure, not its name.
   **The nineteen `open:` rows are a queue in their own right** - twelve from the research page
   (2026-10-29 13:00) and seven from the methodology page (2026-10-30, above). The cheapest are
   `_pbo_words` at its own bars 0.45 and 0.55, and `share-class-switching-margin`, whose module
   has no page test at all. The two most likely to be wrong without anyone noticing are
   `bot-monthly-excess-figure` (a published percentage matched as `[0-9.]+%`) and
   `clamp-bullet-second-printing` (a second printing of four clamps that a test's docstring claims
   to cover and does not).

1. **The partly-payload claims.** The 2026-10-26 scan lists 21 claims whose checks *mix* payload
   reads with other tests. A claim is safe only if a non-payload check pins the specific number
   - the 2026-10-26 pass did not go through them. Re-run the scan (below) and mutate the numbers
   whose only reader is a payload test. **Widen it with what 2026-10-28 found:** a *drift* check
   against a generated artifact (`research.html`, `SCREENER_OVERVIEW.md`, `index.html`,
   `summary.json`) counts as a payload read for this purpose, because it is green once the
   artifact is regenerated. The scan looks for the payload file names; add the generated-page
   paths and `== _rebuilt()`-shaped assertions.
2. **A third mechanical search, from the 2026-10-28 shape:** a registered test that recomputes its
   expected value by calling the same function it is checking, or by copying a predicate out of
   the module under test. Rows 92-94, 97 and 98 all have it, and an `ast` walk can find the
   obvious form - a call to `made_true_by`'s own module whose result is then compared with the
   page. Same caveat as item 6 in 2026-10-26: it over-reports, and only a mutation decides.
3. **DONE for `research_page.py` (2026-10-29, 13:00; rows 118-126). The remaining two surfaces
   are item 0 above.** A fourth mechanical search: the sentences that are not registered at all.** This audit can only mutate what the register
   names, and the three it added today were found by reading one section of one page. The gap is
   structural and it is the one `claims.py`'s own docstring names ("nothing enumerated the *set*
   of computational claims the site makes"). Every *generated* surface should be swept the way
   `test_every_sentence_builder_is_registered` sweeps `stock_summary.py`: for each f-string in
   `research_page.py`, `run_screener.build_screener_overview` and `generate_dashboard.py` that
   interpolates a number into prose, decide whether its sentence states arithmetic, and require a
   claim or an explicit exemption. The three registered today were blind to the whole suite, so
   the expected yield of the sweep is higher than another hand-picked mutation. Start with
   `research_page.py`, which is small and whose numbers are all statistics.

   Of the three residuals named here while doing rows 106-117: the **overlap figures are
   closed** (`research.pairing_overlap`, rows 118-123). The chart's band and tick positions
   (`_pos`) are now an enumerated `open:` row, `chart-axis-and-geometry`. The bot table's 95%
   intervals (`_ex`) are **outside the sweep by construction** - "95%: X to Y" in a table cell is
   a published figure with no prose around it, so it is not a sentence; it needs a decision about
   whether the register covers table cells, and that decision is not a thing to make in the last
   hour of a session.

The two scans are not tracked (see "Method" on why harness code stays out of the repo); both
are about 100 lines - an `ast` walk over `tests/` that resolves `import m as x` / `from m import
X`, imports the module and keeps numeric attributes, and a closure over each registered test's
module-level helpers and fixtures looking for the payload file names.

Reserves: `overview.metric_alerts` (the missing-share bar), `drilldown.avg_metric_coverage`
(mean of weighted coverage), `drilldown.newer_filing` ("a quarter or more"),
`context.valuation_history` (percentile "share with a lower yield": `<` -> `<=`),
`overview.cache_window`, `context.track_record` (entry at the next close after a late run).

## Rules that carry forward

- A new claim that states a number ships with a test on both sides of that number.
- A claim whose test reads a committed artifact also needs one that runs the code.
- Never widen a tolerance to make a new test pass; if the published rounding hides the
  stated arithmetic, choose inputs where it does not.
- Nothing in this audit changes scoring behaviour or published text. A gap that turns out to
  be a *false claim* (the code is right and the sentence is wrong, or the reverse) is not a
  test fix - stop, write it up, and correct the claim through the usual changelog route.
- **A check may not import the number it is checking** (added 2026-10-25, row 59). Write the
  stated figure as a literal in the test and probe both sides of it. `rel=M.TOLERANCE` and
  `assert x == M.THRESHOLD` are not checks of the claim; they are restatements of the code.
- **A copied expression can still catch a mutation, and that is the trap** (added 2026-10-29,
  rows 109-112). `months_needed(...) == round(((3.0 + 0.84) * 0.12 / 0.05) ** 2)` caught two of
  the three mutations aimed at it, because they changed the code *around* the copied line rather
  than the line itself. So a row-59 check does not reliably read as a gap - it reads as coverage
  until the mutation happens to land on the part that was copied. Do not take "this mutation was
  caught" as evidence the check is sound; look at whether the expected value is a literal.
- **A test the register does not name is not coverage** (same rows). Two mutations here were
  caught before today by an unregistered test, so the publish gate - whose membership is derived
  from `checked_by` - never ran it. "Something in the suite fails" and "the loop that publishes
  the page would refuse" are different facts, and only the second one protects a reader.
- **A test's name is read far more often than its body.** If it is called
  `..._is_strict_and_configurable`, the next session will not re-derive whether it tests
  strictness (row 44). Either make the name true or make the body match it.
- **Finishing a gap means registering the fix too.** A new test that `checked_by` does not name
  reads as a gap on the next pass and is outside the publish gate's derivation. The 2026-10-25
  pass re-ran its own mutations after fixing them and two came back
  `GAP-IN-REGISTER` for exactly this reason - re-run the mutation after the fix, and require
  `CAUGHT`, not merely a red test somewhere.
- **Build the expected text and the probe from the literal, not from the constant** (added
  2026-10-26, rows 66-70). Reading the constant into both the page sentence the test expects and
  the inputs it probes with means a moved constant moves page, code and test together. It is the
  row-59 rule one step removed - the constant is not in the `approx`, it is upstream of it.
- **Call the function the run calls, with the arguments the run uses** (row 67). A test that
  passes its own `top_n` to a helper checks the helper, not the run.
- **A registered check must be selectable by its reference** (row 86). `checked_by` entries are
  `file::function` or `file::Class::method`, never a method's bare name;
  `test_every_checked_by_test_exists` enforces it.
- **The mutation harness reverts with `git checkout -- <file>`, which reverts to the last
  commit.** Commit, or copy aside, any uncommitted work in a file before mutating it - the
  2026-10-26 pass lost an uncommitted `claims.py` edit that way and restored it from a copy.
- **Regenerate every artifact the publish rewrites, not just the one the generator is named for**
  (added 2026-10-30, row 128). The methodology page is embedded in `index.html`, so rewriting
  `SCREENER_OVERVIEW.md` alone leaves the page drifted and a drift tripwire fires - on a day when
  the real 02:00 run would have rewritten both and been green. A mutation that reads as *caught*
  because a second artifact was left stale is a **false negative for this audit**: it says
  something sees the change when nothing would.
- **A mutation must change the published output, not only the source the checker reads.** Row 127's
  first form assigned its new sentence to an unused variable: the AST sweep saw it, the page did
  not, and "blind to everything else" would have been trivially true. Check the regenerated
  artifact actually differs before believing either verdict.
