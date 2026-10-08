# Calculation transparency - show the numbers that go into every score

**Created 2026-10-06.** Owner direction, same day: *"we can see how things
score, but we don't actually see the numbers going into any calculations in the
breakout details for each company. This will add trust to the screener."*

This is the governing plan for that request. `OWNER_FOCUS.md` holds the queue
entry; this file holds the argument, the measurements, the stages and the tests.
It is **not** the visual redesign (`plan/dashboard-redesign-master.md`), but the
two meet at the stock drilldown and are sequenced together below.

## The goal, in one sentence

**For any stock, a reader can follow one number all the way down - composite,
to category score, to metric percentile, to raw value, to the company's own
reported inputs - see every weight and every peer count used on the way, and
recompute it with a calculator and get the same answer.**

Credibility is the product (CLAUDE.md). A score a student cannot check is a
claim; a score they can check is evidence. This is also the investment-club
requirement: *teachable and explainable*.

## Status, 2026-10-07 late (pass 3) - the numbers are on the row

The owner looked at the workings and saw no numbers: the inputs were there, but behind a
disclosure triangle. Every metric row now prints its own figures through the formula to the
value scored, its rank and sector median, and percentile x weight = points
(`metric_lineage.EQUATIONS` / `SOURCES`; since pass 4 behind a click on the row, with reasons for unused metrics). Exact templates are evaluated against
every stock by `tests/test_metric_lineage.py` (19 of 19 reproduce 100%), so a scoring formula cannot
change without the page - see `DECISIONS.md` 0.8c. The CSV ("download this stock's workings")
shipped in pass 2. Still open from the table below: inputs at fetch for the 11 history-based
metrics, the 0.9 research items, `audit_stock.py --sample 25` in the morning brief.

## Status, 2026-10-07 evening - built, and ready for a deeper pass

The owner asked for all of it to be built in one owner-run session, with the nightly sessions then
going deeper. **Every stage below is shipped in a first version.** What each one left for the
nightly sessions is the work to do next, in this order.

| Stage | State | Left for the deep pass |
|---|---|---|
| T0a false sentences + claims register | **Done** (06:00 session) | - |
| T0b true weights, reproducibility, refuse-to-publish | **Done.** 4,010/4,010 category scores and 502/502 composites rebuild from the payload; the build refuses otherwise | Promote `research/measurements/2026-10-06-calculation-reproducibility.py` into the dated record of the *original* defect and stop re-running it as a live check (it measures the pre-fix reading by design) |
| T1 lineage registry | **Done** (`metric_lineage.py`, 36 metrics with formula, inputs, caveat) | The 8 zero-weight candidate metrics have no entry; add them when one is activated |
| T2 inputs + "how it was computed" | **Done** for 24 of 36 metrics (equation rebuilt for every stock, 99.5-100%); Piotroski and Beneish show their components | The 12 others say why they show no equation. **Add the missing inputs at fetch** for the analyst metrics (per-quarter EPS actual/estimate are not retained) and the shared risk-free rate / market return behind Jensen's alpha, Sharpe and Sortino; then add `sustainable_growth` to `RECOMPUTE` |
| T3 percentile context | **Done** ("Ranked 3rd of 47 in Industrials", sector median and quartiles, universe fallback) | A small distribution strip per metric (where this stock sits between the sector's quartiles) |
| T4 the equation view in the drilldown | **Done** except the download | **"Download this stock's workings" (client-side CSV)** - the most direct form of "don't trust us, check" |
| T5 independent checker + gate | **Done** (`scripts/audit_stock.py`, two new modules in the data loop's publish gate) | Run `--sample 25` in the morning brief and record the pass count |
| T6 per-input provenance | **Partly**: statement dates, source, what the page cannot vouch for | Per-input staleness (filing age is shown once per stock, not per figure); name which metrics used the annual fallback |

### Findings the build produced that need *research*, not a patch (rule 4)

All four were read from the code by the lineage audit and **verified against the live payload**.
The page now states each in plain words next to the metric; none is changed. Each deserves a
dated note in `research/` with the literature and practice, then a changelog entry if it changes:

1. **`operating_leverage` ranks backwards for negative values.** 95 of 393 are negative and
   average the 84th sector percentile against 37th for the rest. 8% of non-bank Quality.
   (Q: how do practitioners treat a negative degree of operating leverage - exclude, floor, or
   drop the metric?)
2. **"Year-over-year" growth is 12-21 months.** The prior figure is the fiscal year before the
   latest completed one. Affects `revenue_growth`, Piotroski signals 3, 8 and 9 and the Company
   Snapshot's YoY figures. (Q: what window does the literature assume; can a true prior-year
   quarter-aligned figure be built from the five quarters Yahoo returns?)
3. **Two EBITDA definitions** across `ev_ebitda` and `net_debt_to_ebitda`, neither equal to the
   Company Snapshot figure.
4. **Labels that disagree with the code**: `return_6m` is a 6-1 return; "1Y" risk metrics use about
   13 months; `consecutive_beat_streak` is a recency-weighted count; Sortino's denominator.

## What the drilldown shows today, and what it does not

Today, for each of the eight categories, the metric table shows: raw value,
sector percentile (with a direction chip), and **a weight**. Above it,
"Score Contribution Breakdown" shows `score x category weight = points`. That
already goes further than most screens, and it is why the gaps below are worth
closing rather than starting over.

What is **missing**:

| Layer | Missing |
|---|---|
| Inputs to a raw metric | "FCF yield 14.4%" never shows FCF or market cap. 45 metrics, almost none show their components |
| Compound metrics | Piotroski (9 pass/fail signals), Beneish (8 indices), Jensen's alpha (beta, risk-free, market return), momentum (the two prices and dates), volatility (how many days) are shown as one opaque number |
| Raw -> percentile | The reader sees "63%" but not *63% of whom*: which sector, how many peers, the sector's median and quartiles, or that a sector with <10 valid values falls back to the whole universe |
| Percentile -> category | Whether a metric was even in the average for this stock, and what it was weighted at (see defect 1) |
| Category -> composite | The coverage discount (defect 2) |
| Provenance of each input | Filing date / staleness per input, quarterly vs annual (`data_source` is stored but not legible - inventory gap 5), fetch time |
| Reproducibility | No way to take a stock's workings away and check them elsewhere |

## Three defects found while writing this plan (measured 2026-10-06)

Measured against the live `dashboard_data.js` by recomputing each score from the
payload alone (`research/measurements/2026-10-06-calculation-reproducibility.py`, to be promoted to a test in T0b).
These are the "documented failure" the evidence rule asks for, and they are the
same class as the 2026-08-28 factor-weight bug (the page printed 13% where
14.95% had been used, for 498 of 502 stocks).

**Defect 1 - the per-metric weights on the page are not the weights used, for
276 of 502 stocks.** The drilldown prints `D.weights.metric_weights[cat][metric]`,
the *generic configured* weight, for every stock. The engine
(`factor_engine.compute_category_scores`, lines 2590-2706) uses a different weight
set in three situations the page never mentions:

1. **Bank-like stocks** use `bank_metric_weights` (ROE/P/B in, ROIC/EV-based out).
2. **Piotroski conditional weighting** halves the Piotroski weight for non-bank
   stocks whose valuation score is below 50, and shares the freed weight across
   ROIC and gross-profit/assets - and a growth-trap variant does the same for
   high-growth, low-quality names.
3. **Missing metrics**: a metric with no data drops out and the rest are
   renormalised by that stock's own weight sum.

Result: recomputing `sum(pct x generic weight) / sum(generic weight with data)`
reproduces the published category score for **3,678 of 4,012** stock-category
pairs. **334 do not** - 59 in Valuation, 275 in Quality - touching **276 of 502
stocks (55%)**. The score is right; the *explanation printed beside it is false*.
Nothing in CI could see it, because nothing recomputes a score from the payload.

**The worst case is visible on JPM's drilldown.** Valuation shows EV/EBITDA, FCF
Yield and EV/Sales as **N/A with weights of 25%, 45% and 10%**; the only metric with
data is Earnings Yield (7.4%, 54th percentile, weight 20%); the same panel marks P/B
as **"Inactive (0% weight)"** - it is the bank's *main* valuation metric at 60% - and
the category score printed beside all of it is **39.9**, which no arrangement of the
numbers on screen produces. The Quality block repeats it (ROE, ROA and Equity Ratio
listed as inactive for a bank, in raw `snake_case`). A student checking the arithmetic
would conclude the tool is broken. It is not; the page is describing the wrong
weights.

**Defect 2 - "Composite = sum of the points above" is false for 2 stocks.**
`compute_composite` applies a coverage discount after the weighted average (stocks
below 80% metric coverage). The drilldown's composite line is the plain sum of the
contributions. FDXF shows 42.04 against a composite of 41.15, L 51.68 against
51.45. Small, but it is the page's central arithmetic claim, and the discount is
invisible.

**Defect 3 - the first sentence of every drilldown misdescribes the composite, and
the public methodology page contradicts itself about it.** `stock_summary._sentence_rank`
(line 179) says *"Its composite of 73.7 is a percentile: it scores above 74% of the
universe"* - for **the stock ranked 1st of 502**, which by definition scores above
100% of it. Measured on all 502 summaries
(`research/measurements/2026-10-06-rank-sentence-claim.py`): the
claimed percentage differs from the share of the universe the stock actually ranks
ahead of by a **median of 19.6 points**, by **more than 10 points for 75.1%** of
stocks, and by **31 points** at worst (NTAP: "scores above 64%", ranks 26th of 502).
The cause is a stale idea: since "Phase 13 (F1)" `Composite` is the **cardinal**
weighted average of category scores (`compute_composite`, factor_engine.py:2927-2952,
which keeps magnitude as the ranking key) and the percentile is a *separate*
`Composite_Pct`. Four places still say otherwise: that sentence, `README.md` line 54
("a score of 95 means better than 95% of the universe"), `plan/investor-profiles.md`
line 58, and **Step 5 of the generated methodology page** (`SCREENER_OVERVIEW.md`
line 264, produced by `generate_screener_overview`), while the **same page's
Limitation 8** (line 491) correctly says *"Do not read the cardinal Composite as a
percentile."* The page therefore contradicts itself, and Step 5's formula also omits
the coverage discount that Data Quality (line 289) describes as "currently enabled".
`check_published_claims.py` did not catch it because it tests specific claims people
thought to write down, not "every sentence that says how a number is computed".

## Design principles (constraints, not preferences)

1. **The page shows the engine's numbers, never its own re-derivation.** The
   weight a metric was multiplied by must be *emitted by the engine from the same
   code path that used it*. Re-implementing the bank / Piotroski / renormalise
   rules in the generator or in JS creates a second copy that drifts - that is
   precisely how defect 1 happened. **Refactor `compute_category_scores` so weight
   resolution is one function used by both scoring and export.**
2. **Reproducibility is enforced, not promised.** A build-time reconciliation, in
   the style of `_reconcile_factor_weights`, recomputes every stock's category
   scores and composite from the payload alone and **refuses to publish** if any
   pair is off by more than rounding (0.01). It joins the data loop's publish
   gates (`scripts/check_published_claims.py`) so the 02:00 run cannot ship a page
   whose arithmetic does not add up.
3. **Show the inputs the company reported, as fetched.** Same rule as `raw` since
   2026-09-01 (no winsorisation). Never clip, round early or impute for display.
4. **Display-only, never scored.** Inputs are provenance for a number that already
   exists. They enter no metric, no weight, no rank. Tests assert it, as for
   `about` and `earn`.
5. **Say plainly what cannot be shown.** If a metric's raw value cannot be
   recomputed from the inputs displayed (a provider-supplied ratio, an analyst
   consensus, a value built from a long price history), the page says
   *"provider-supplied"* or *"computed from 252 daily closes"* rather than showing
   a formula that does not reproduce. A false equation is worse than none.
6. **Decision support, not advice.** `BANNED_TERMS` applies to every new sentence.
7. **Payload discipline.** `stock_detail` is already ~90% of the payload and a
   phone on 4G is the budget. Measure gzipped before and after every stage. Target
   **<= +150 KB gzipped inline**; beyond that, ship the inputs as per-ticker shards
   fetched on drilldown open (still static files on Pages). The inline path is
   preferred because it keeps the page working offline once loaded - say in the log
   which path you took and what the measurement was.
8. **Scores, ranks and sentences are untouched** by every stage except T0a's
   sentence fixes and T0b's *displayed* weights. Verify with the payload diff (see "Verification" below).

## The stages

Each stage is one session's work, ends committed, and is independently shippable.
**Commit after every stage** - the 2026-10-06 session was lost to a usage limit
with nothing committed.

### T0a - Fix the false sentences and build the claims register  *(defect; first)*

> **DONE 2026-10-07.** `claims.py`, `tests/test_claims_register.py` (24 tests),
> `scripts/diff_payload.py` (brought forward from T0b - every later stage needs
> it). Four corrections from this stage that the rest of the plan should carry:
>
> 1. **There were six sites, not four** - also `FORENSIC_AUDIT_REPORT.md` and
>    `tests/test_stock_summary.py`, which *asserted* the false claim. Expect the
>    same when fixing defects 1 and 2: grep the tests, not just the prose.
> 2. **Defect 2 is 3 stocks, not 2** (FDXF -0.90, PSKY -0.24, L -0.23) and
>    defect 1 re-measures to **333 of 4,010 pairs across 275 of 502 stocks**,
>    not 334/4,012/276. Re-run the scripts; the numbers move with each data run.
> 3. **A fourth defect, now T0b's:** the drilldown's "rests on N of 18 metrics"
>    and its provenance badge's 60/80% colours use `factor_engine`'s hard-coded
>    `_metric_keys` list, **not** the applicable-metric coverage the composite's
>    coverage discount reads (`METRIC_COLS` less the stock type's exclusions -
>    35 for a bank-like stock, 41 otherwise). 62 stocks read under 80% on that
>    badge; 3 were discounted. T0b must emit applicable coverage from the engine,
>    which is also what the composite line needs for defect 2.
> 4. **The rank sentence does not claim `composite == sum of contributions`**,
>    on purpose - the payload cannot yet show that chain honestly. When T0b adds
>    applicable coverage, revisit `claims.claim("summary.rank").caveat` and
>    `summary.drivers`'s caveat; both name T0b and a test enforces that.
>
> Verified: 181,446 payload leaves, 502 changed (all `summary[0].t`), 0 added,
> 0 removed; +309 bytes gzipped; suite 1716 passed (baseline 1692).

Defect 3, and the general fix for its cause. Small, text-level, no scoring change.

- **Fix the four places** that call the composite a percentile: replace
  `_sentence_rank`'s second sentence with one that is true of every stock - the
  rank-derived share ("ahead of 100% of the universe" for rank 1: `(N - rank) /
  (N - 1)`) and an honest description of the composite ("a weighted average of its
  eight category scores, each 0-100"). Correct `README.md`, `plan/investor-profiles.md`
  and **Step 5 of the generated methodology** (edit `generate_screener_overview`, then
  regenerate - rule 10), and add the coverage discount to Step 5's formula. Every new
  sentence passes `BANNED_TERMS`.
- **Build the claims register**: a single table (`claims.py` or a section of the lineage
  registry) of every sentence-template on the site and in the generated docs that states
  *how a number is computed or what it means* - each with the **code that makes it true**
  and a test that asserts it against a build, in the spirit of
  `tests/test_overview_claims.py` but driven by the register so a *new* sentence cannot
  ship unregistered. Seed it by grepping `stock_summary.py`, `generate_dashboard.py`'s
  static copy, and `run_screener.generate_screener_overview` for words like "percentile",
  "weighted", "score", "composite", "rank", "average", "median", "discount".
  **A claim with no registered check fails a test.**
- Add the module to the data loop's publish gates (`scripts/check_published_claims.py`)
  so a false claim cannot be republished by the 02:00 run.
- **New tests**: for all 502 stocks the rank sentence's percentage equals the rank-derived
  share to the printed precision (it fails against today's tree for 75% of stocks);
  the methodology page no longer contains both "converted to a cross-sectional percentile
  rank" and "Do not read the cardinal Composite as a percentile".
- Update `METHODOLOGY_CHANGELOG.md` (a description fix, not a scoring change - say so),
  `DECISIONS.md`, and the inventory.

### T0b - Fix the false weights and the composite line  *(defect)*

This is a correctness fix of something false on the public site, so it outranks
every presentation stage.

- Refactor `compute_category_scores` so per-row, per-metric effective weight
  resolution lives in **one function**; scoring consumes it and so does the
  exporter. Scores must come out **bit-identical** (assert against the current
  `factor_scores` parquet / golden tests: `tests/test_golden.py`, `test_scoring.py`).
- Carry each stock's effective metric weights into the payload compactly: emit a
  **weight profile id** per stock-category (`generic` / `bank` / `pio_lowval` /
  `pio_growthtrap`) with the profile tables once under `D.weights.profiles`, and let
  the drilldown renormalise over metrics with data (that last step is pure
  arithmetic on published numbers, so it cannot disagree with the engine - and the
  reconciliation proves it). If profiles prove insufficient, emit per-stock weight
  overrides for the deviating stocks only. Measure the cost.
- Drilldown metric table prints the weight **actually used** and, when it differs
  from the generic configured weight, says why in plain words ("bank weighting",
  "Piotroski halved: valuation score below 50", "re-weighted: 2 inputs missing").
  Extend `weightNote()`'s idea down one level.
- Composite line: add the coverage-discount step when it applies ("coverage 74%,
  below the 80% threshold: composite reduced by 0.9%"), so the displayed chain sums
  to the displayed composite for every stock.
- **New `tests/test_calculation_reproducibility.py`**: from the payload alone,
  recompute all 502 x 8 category scores (+-0.01) and all 502 composites (+-0.01).
  It must **fail against the current tree** (334 pairs, 2 composites - record the
  counts in the test docstring). Also assert every metric with weight > 0 in any
  profile is published in `raw` and `pct` (CAT_METRICS has 45 entries, `raw`
  carries 37 - confirm the 8 absent are zero-weight and pin it).
- Add the reconciliation to `prepare_dashboard_data()` and the new module to
  `scripts/check_published_claims.py`'s set.
- Update `METHODOLOGY_CHANGELOG.md` (this changes what is *displayed*, not scored -
  say so), `plan/dashboard-inventory.md`, and add a `DECISIONS.md` entry.

### T1 - Metric lineage registry  *(no visible change; small)*

One structured table, the single source for "what is this number":
for every metric with weight in any profile - **formula**, **input fields with
their yfinance source (statement line / `.info` key)**, **period** (TTM sum of 4
quarters, last annual, point-in-time), **direction**, **missing/negative handling**
(what makes it NaN), and whether it is **recomputable from displayed inputs**.
Seed it by reading `compute_metrics` (factor_engine.py:1506-2402) and the fetch
(`_fetch_single_ticker_inner`, :779). Feed the Methodology page's metric
descriptions from it so prose and code cannot disagree (rule 10's lesson).
Test: every weighted metric has an entry; every named input exists in the fetch
dict; formula text is generated, not hand-typed, where a callable exists.

### T2 - Inputs in the payload and the "How this was computed" row

- Add per-stock inputs for the metrics, deduplicated against what `financials`
  already carries (market cap, EV, revenue, net income, EBITDA, FCF, debt, cash,
  shares, EPS are there). Likely additions: EBIT/operating income, invested capital,
  total assets (and prior-year), equity, gross profit, operating cash flow, interest
  expense, the Piotroski signals, the Beneish indices, beta + the rf and market
  return used, the two prices and dates behind 12-1 / 6m momentum, the day count
  behind volatility/Sharpe/Sortino, analyst counts.
- Each metric row in the drilldown gets a disclosure: **inputs -> operation ->
  result**, e.g. `FCF yield = FCF $4.46B / market cap $31.2B = 14.4%`, using the
  same number formatting as the rest of the page. Compound metrics expand to their
  parts (Piotroski: nine labelled pass/fail rows summing to the score).
- **Reproduction test**: for every metric flagged recomputable in the registry,
  recompute `raw` from the displayed inputs and match the published value to the
  display precision, across all 502 stocks. A metric that fails is either fixed or
  re-flagged "provider-supplied" - never shown with a formula that does not hold.
- This will find real discrepancies (a TTM vs annual mismatch, an EPS-basis
  difference). **Report each in the log and the changelog; do not paper over one.**
  That is the point of the exercise.

### T3 - Make the percentile step visible

For each metric: *"Ranked 12th of 63 in Industrials (sector median 11.2, interquartile
9.8-14.1) - higher is better, so a lower value earns a higher percentile."* Source:
per-sector, per-metric `n`, median, quartiles (11 sectors x ~37 metrics x ~4 numbers
is small); flag the `<10 valid values -> whole universe` fallback when it applies,
and say when the stock has no value (NaN, excluded - not imputed to 50). The numbers
must come from the same run that produced the percentiles, not a recomputation.

### T4 - The equation view, inside the redesigned drilldown

Compose T0a-T3 into one coherent top-to-bottom reading order, built **with** the
drilldown redesign (design stage D3), not styled twice:
**composite -> eight category rows (score, weight, points) -> expand a category ->
its metrics (raw, percentile, weight used, points) -> expand a metric -> inputs and
peer context.** At every level the arithmetic is on screen and sums.
Add **"Download this stock's workings"** (client-side CSV, built from the payload)
so a student can check it in a spreadsheet - the most direct form of "don't trust
us, check".

### T5 - An independent checker, and a gate

`scripts/audit_stock.py TICKER`: prints the full trace from the payload and
recomputes it with a **separate implementation** (deliberately not importing
`factor_engine`'s scoring), failing loudly on any difference. Run it on a rotating
sample in CI and in the data loop. Two implementations agreeing is the evidence.
Also wire the T0b reproducibility module into `data-run.ps1`'s publish gates (already
via `check_published_claims.py`) and record the pass count in the morning brief.

### T6 - Provenance per input

Per-input as-of date (filing date / period end), age in days with the existing
stale-filing logic, `data_source` made legible (quarterly vs annual - closes
inventory gap 5), and fetch time. Extend the existing Data Provenance block rather
than adding another.

## Order, relative to the visual redesign

`T0a` then `T0b` first (defects, small, no design dependency; T0a is the smaller and
can ship alone). Then `D2` (rankings table) and
`T1` (registry) in either order. `T2` and `T3` are payload work and can land before
the drilldown is rebuilt; `D3` + `T4` are **one stage built together**. Then
`T5`, `T6`. The ordering is deliberate: restyling the old category tables and then
replacing them would be paying for the same surface twice.

## Verification, every stage

1. **Recompute from the payload** with `tests/test_calculation_reproducibility.py`
   - this is the integrity metric. Record the counts (pairs reproduced of 4,012;
   composites of 502) in the `NIGHTLY_LOG.md` health entry.
2. **Payload diff against the live one**, classifying keys as unchanged / added /
   changed. For this work the rule is: **every pre-existing score, rank, raw value,
   percentile and sentence is byte-identical**; only added keys (and, in T0a/T0b, the
   corrected sentences and displayed weights) differ. `scripts/diff_payload.py` does not exist yet - **the
   T0b session builds it** (small: load both payloads, walk both, bucket the
   differences) because every later stage needs it.
3. **Gzipped payload size**, before and after.
4. **Look at it** - the drilldown for one bank (e.g. a regional bank), one
   Piotroski-adjusted name, one thin-coverage name (FDXF, L), one ordinary name, at
   desktop and 375px - and say what you saw. Choosing those four on purpose is what
   exercises the paths that were wrong.

## Sources and where to look (the evidence the rule asks for)

- **The documented failure** - the two measurements above, and the 2026-08-28
  precedent (`METHODOLOGY_CHANGELOG.md`, `tests/test_weight_transparency.py`).
- **Replication as the standard for credibility.** Hou, Xue & Zhang (2020),
  "Replicating Anomalies", *Review of Financial Studies* 33(5): a large share of
  published anomalies fail to replicate when construction details differ - the
  case for publishing construction details. The session should read it and quote the
  actual numbers rather than rely on this paragraph.
- **Documented practice:** index and factor providers (MSCI, S&P Dow Jones, FTSE
  Russell) publish factor-index methodology documents specifying each metric's
  formula, input definitions, winsorisation/standardisation steps and treatment of
  missing data, and worked constituent-level examples. The Monday research note for
  this work should cite specific documents, say what each exposes that this tool
  does not, and where **they** are less transparent than this tool can be.
- **Additive attribution as a design pattern:** contributions that sum exactly to
  the total (as `compute_factor_contributions` already does for categories) are the
  property that makes an explanation checkable. Extend it one level down - metric
  points that sum to the category score - and cite the attribution literature
  (e.g. Shapley-value / additive feature-attribution work) only for the claim that
  *exact additivity* is what makes an explanation auditable, not as a method to
  import.

## What this plan deliberately does not do

- **No change to any weight, metric, percentile rule or score.** Showing the math
  must not become a reason to alter it, and nothing here may be justified by the
  backtest (rule 5) or the thin IC series (rule 4).
- **No client-side re-derivation of engine logic** (principle 1).
- **No raw filings or financial-statement browser.** Inputs are exactly those
  behind a score, no more - the dashboard is not a data terminal.
- **No claim of independent verification of the provider's data.** The page shows
  what Yahoo Finance reported and how it was used; it can say a number was
  *derived consistently*, not that it is *true*. Say so on the page (T6).

## Done means

An investment-club student picks any stock, opens it, and can answer *"where did
this 63 come from?"* three levels down without leaving the panel; downloads the
workings; and a spreadsheet agrees with the page to the displayed precision. And
the data loop will refuse to publish a day on which that stops being true.
