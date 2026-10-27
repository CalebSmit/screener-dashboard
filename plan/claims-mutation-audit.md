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
| 76 | `drilldown.piotroski_beneish_parts` | Beneish's SGI coefficient 0.892 -> 0.982 (factor_engine) | **gap** - both checks read the payload; across all of `tests/` only `test_investor_profiles.py::test_each_profile_is_what_the_preset_would_publish` went red, indirectly, by comparing a rerun with the committed payload | fixed: `test_the_engines_beneish_indices_rebuild_its_m_score_with_the_published_weights` |
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

## Next claims (for the nightly sessions)

Take them in order; one session can do four to six. Record each in the table above with
the date, and move this list down.

**The 2026-10-20 list's items 1-8 and 12 are done** (2026-10-25, rows 39-62), **and the two
mechanical searches are done** (2026-10-26, rows 63-88; item 1's bands and window were covered
there, rows 65 and 74-75). What is left:

1. `context.put_skew` - what is left of it: the *put* leg's "nearest 0.95" with two eligible
   puts either side of 0.95 (row 65 moved the target to 0.90, which the 0.90-strike fixture
   happens to sit on), and the 10-60 day flag *at* 10 and 60 (`<=` -> `<`).
2. `context.sahm` - 3-month average minus the prior-12-month low of the 3-month average; the
   0.50 trigger (`>=` -> `>`).
3. `context.track_turnover` - half the sum of absolute weight changes; the break-even
   formula's factor of 2.
4. **The `research.*` surface, which no pass has touched.** `research.html` became a claims
   surface on 2026-10-25 (claim `research.positive_control`); its sentences quote intervals,
   t-statistics and a detectable-difference figure, which is exactly the shape rows 26 and 59
   show is easiest to get wrong - a tolerance or a divisor that no test pins independently.
   (Row 87 moved `HLZ_T` and was caught; the intervals and the detectable difference were not
   probed.)
5. **The partly-payload claims.** Item 5's scan also lists 21 claims whose checks *mix* payload
   reads with other tests. A claim is safe only if a non-payload check pins the specific number
   - the 2026-10-26 pass did not go through them. Re-run the scan (below) and mutate the numbers
   whose only reader is a payload test.

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
