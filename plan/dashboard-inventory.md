# Dashboard inventory (as of 2026-08-28)

**Read this before changing the dashboard.** There is far more in it than a
first look suggests, and the most common failure mode will be rebuilding
something that already exists. Refresh this doc when you materially change the
layout.

Generated from `index.html` (270,427 chars) and `dashboard_data.js`.

**Refresh this file whenever you change the layout.** It went stale on
2026-08-25 because the session that shipped the time dimension could not
edit it - `.claude/` was blocked as sensitive. The plans now live in
`plan/` precisely so that cannot happen again; there is no excuse for
leaving it wrong.

## Top-level views

| Section | Contents |
|---|---|
| **Top 5 Stocks** | Highest-composite names, card layout. Reads `table_data` directly, excluding trap-flagged names |
| **Factor Analytics** | `Factor Scores by Sector`, `Trap Rate by Sector` |
| **Defensibility & Diagnostics** | `How Stable Is the Ranking?` (weight sensitivity), `Are the Factors Independent?` (factor correlation) |
| **Full Universe Rankings** | The 501-row sortable table - the workhorse view |
| **Methodology** | A very large embedded explainer (~30 headings) |

Interactive elements are sparse: **2 charts** (`sector-dist-chart`,
`vt-chart`) and **1 table** (`universe-table`). There is
also a "Screener AI" chat with a Chat Settings panel.

## The Methodology section is roughly half the file

It embeds a full document: What Is This, Where Does the Data Come From, all 8
factor categories with weights, Bank-Specific Scoring, a 6-step score
calculation walkthrough, Piotroski Conditional Weighting, Data Quality
Safeguards, Value/Growth Trap Detection, Portfolio Construction, What Gets
Output (all 6 Excel sheets), Factor-Exposure Diagnostics, Reproducibility,
Defensibility & Transparency Features, Key Design Decisions, Limitations,
Quick Start, Summary.

This is **an asset for the investment-club audience and the main reason the
tool is defensible** - do not delete it.

**Correction, 2026-08-28: it is already generated from config.** The previous
version of this section warned that the weights were "hardcoded into the
prose" and would go stale. They are not: `run_screener.generate_screener_overview(cfg)`
templates the whole document out of `config.yaml` on every run, and all 8
category weights and ~40 metric weights were checked against `config.yaml`
that morning and matched exactly. **Edit the generator, not the file**
(rule 10).

The real failure was one level down, and worse: the document was faithful to
`config.yaml` while the *screener* was not. A run's momentum weight is scaled
by the volatility regime, so the composite was built at 14.95% while every
surface printed 13%. See below and `METHODOLOGY_CHANGELOG.md` 2026-08-28.

## Payload weight

| Key | Size (MB) | Notes |
|---|---|---|
| `stock_detail` | 2.81 -> ~3.5 | **~83% of the payload.** All 502 stocks. Grew 2026-08-26 with `about` |
| `history` | 0.27 | Added 2026-08-25. 18 accepted run dates, 2 excluded |
| `table_data` | 0.26 | 502 rows, 8 category scores + composite/rank/flags |
| everything else | <0.02 | `portfolio` (0.010) and `spx_weights` removed 2026-08-26 |

**Raw size is the wrong number to optimise.** Pages serves gzip, and the
payload compresses 4.2x overall - prose closer to 11x. The business
descriptions add ~0.71 MB raw but only ~60 KB on the wire. Measure gzip before
calling anything expensive.

`stock_detail` dominates. Per stock: `raw`, `pct`, `cat_scores`, `contrib`,
`composite`, `rank`, `sector`, `company`, `industry`, `about`, `vt`/`gt`,
`price`, `pt_mean/high/low`, `num_analysts`, `eps_mismatch`, `eps_ratio`,
`data_source`, `metric_count`/`metric_total`, `financials`, `flags`, `peers`,
`self_metrics`.

`industry` and `about` were added 2026-08-26 (owner request). Both are
**display-only** - `about` is the provider's `longBusinessSummary`, rendered
verbatim in a clamped block under the score cards with a "Show more" toggle and
an attribution line. Neither is scored, ranked, or fed to a metric, and
`tests/test_dashboard_surfaces.py` asserts they never appear in `raw`/`pct`.
Both ride the `.info` dict the fetch already pulls, so they cost no API calls.

Adding history will grow this fast. Lazy-load or downsample - do not ship a
10 MB payload to a phone.

## Weights: what the drilldown shows - DONE 2026-08-28

The stock drilldown shows **per-stock effective weights**, not the configured
defaults. Three surfaces read them (`effWeights()` in the emitted JS): the
score cards, the contribution bars, and the category-detail badges.

Two things move a weight away from the Methodology page's number, and both are
now stated on the page by `weightNote()`:

1. **The volatility-regime adjustment**, run-level. Baked into
   `weights.factor_weights`, with `weights.base_factor_weights` kept alongside
   so the page can show `13% -> 15.0%` rather than just asserting 15.0%.
2. **Per-stock renormalisation**, when a category could not be scored. The
   category keeps its row, marked "no data", instead of disappearing - hiding
   it would leave the reader unable to see why the rest add to more than the
   defaults.

**Do not revert these to `D.weights.factor_weights[c]`.** That is the bug:
between February and 2026-08-28 the page printed `Score x 13% = 9.76 pts`,
which is false, for 498 of 502 stocks. `prepare_dashboard_data()` now
reconciles the recorded weights against the published contributions on every
build and will not publish weights that fail to reproduce them.
`tests/test_weight_transparency.py`, 34 tests.

## Other payload keys

`kpis`, `weights` (factor + metric, plus `base_factor_weights` /
`factor_weights_adjusted` / `factor_weights_derived` since 2026-08-28),
`metric_meta` (36 metrics),
`sectors` (11), `sector_composition`, `histogram`, `vt_by_sector`,
`gt_by_sector`, `sector_distributions`, `factor_correlation`,
`weight_sensitivity` (8), `data_quality`, `config_traps`, `history`.

`portfolio` and `spx_weights` were removed 2026-08-26 - see below.

## What is genuinely missing

Confirmed against the above, not guessed:

1. ~~Any time dimension.~~ **SHIPPED 2026-08-25.** `history.py` builds a
   quality-gated spine from `improvement/snapshots/`, surfaced as: a
   **What Changed** panel under the KPI row (biggest movers each way, inline
   sparklines, category that moved most), a sortable **delta column** in the
   universe table, and a **Rank History** block in each stock's drill-down.
   Payload key `history` (~0.25 MB): `dates`, `series`, `delta`, `movers`,
   `noise`, `compare`, `excluded`, `available`.

   Two things not to undo. **Runs enter the history only if their ranking
   correlates with the previous accepted run at Spearman >= 0.50** - the
   `2026-07-28` degraded run correlates at 0.016/-0.020 and, ungated, reports
   82% of the universe as material movers. And **the default comparison is the
   ~1-month window, not the previous run**: measured on this repo's snapshots,
   every material one-day mover on 2026-08-25 was a round-trip, while 169 of
   193 one-month moves were genuine trends.
2. **Any sell-side workflow.** No watchlist, no holdings, no deterioration
   alerts.
3. **Time-series valuation context.** `pct` is cross-sectional only.
4. **Catalyst/earnings-date proximity.**
5. **Per-stock confidence surfaced.** `metric_count/metric_total` and
   `num_analysts` are in the payload but not made legible.
6. **Charting breadth.** Three charts for a 3 MB payload is thin - though add
   charts only where they beat a table, not for decoration.

## Model Portfolio removal - DONE 2026-08-26

Owner directive 2026-08-05, restated as a priority 2026-08-26 and shipped the
same evening. The recommendation this file made was followed exactly: **the
dashboard surface went, the construction engine stayed.**

**Removed:** the `Model Portfolio` section, the `Portfolio Sector Allocation vs
S&P 500` chart, `renderPortfolio()`, `renderSectorAlloc()`, the `portfolio`
payload key, and the `spx_weights` key - the latter because that chart was its
only consumer, and a sector split of the S&P 500 against itself says nothing.

**Kept, deliberately:** `portfolio_constructor.py`, the `08_model_portfolio`
artifact, the `ModelPortfolio` Excel sheet, and the `in_portfolio` snapshot
column. The warning in the previous version of this section was correct -
`improvement_engine.record_run_snapshot()` writes `in_portfolio` into every
snapshot and computes **turnover** from it, so deleting construction outright
would have silently damaged the evidence base. Three test modules cover it too.

**Top 5 was checked, not assumed.** It had been reading the portfolio holdings.
It now filters `table_data` for trap-free names and sorts by rank. Verified
identical on live data (`HST, EXPE, APA, EIX, CF` both ways): the sector cap is
8-of-25 and cannot bind on five rows.

Pinned by `tests/test_dashboard_surfaces.py` (30 tests, 29 of which fail
against the pre-removal code). A *partial* removal is the dangerous state -
`D.portfolio` undefined at render time takes the whole script down and the page
goes blank with every ship gate still green.

If the owner later wants the construction engine gone too, the open questions
are whether turnover is actually used by anything and whether the Excel sheet
is still wanted.
