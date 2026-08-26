# Dashboard inventory (as of 2026-08-25)

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
| **Top 5 Stocks** | Highest-composite names, card layout |
| **Model Portfolio** | Holdings + `Portfolio Sector Allocation vs S&P 500` chart. **Slated for removal - see below** |
| **Factor Analytics** | `Factor Scores by Sector`, `Trap Rate by Sector` |
| **Defensibility & Diagnostics** | `How Stable Is the Portfolio?` (weight sensitivity), `Are the Factors Independent?` (factor correlation) |
| **Full Universe Rankings** | The 501-row sortable table - the workhorse view |
| **Methodology** | A very large embedded explainer (~30 headings) |

Interactive elements are sparse: **3 charts** (`sector-alloc-chart`,
`sector-dist-chart`, `vt-chart`) and **1 table** (`universe-table`). There is
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
tool is defensible** - do not delete it. But it is reference material sitting
in the same document as the decision surface, and the category weights are
hardcoded into the prose ("1. Valuation (22% of final score)"). Once the
improvement engine starts adjusting weights, **this text will silently go
stale and start lying.** Generating it from `config.yaml` is a genuine
correctness fix, not a polish task.

## Payload weight

| Key | Size (MB) | Notes |
|---|---|---|
| `stock_detail` | 2.682 | **~85% of the payload.** All 501 stocks |
| `history` | 0.251 | Added 2026-08-25. 18 accepted run dates, 2 excluded |
| `table_data` | 0.244 | 501 rows, 8 category scores + composite/rank/flags |
| everything else | <0.03 | |

`stock_detail` dominates. Per stock: `raw`, `pct`, `cat_scores`, `contrib`,
`composite`, `rank`, `sector`, `company`, `vt`/`gt`, `price`,
`pt_mean/high/low`, `num_analysts`, `eps_mismatch`, `eps_ratio`, `data_source`,
`metric_count`/`metric_total`, `financials`, `flags`, `peers`, `self_metrics`.

Adding history will grow this fast. Lazy-load or downsample - do not ship a
10 MB payload to a phone.

## Other payload keys

`kpis`, `portfolio`, `weights` (factor + metric), `metric_meta` (36 metrics),
`sectors` (11), `sector_composition`, `histogram`, `vt_by_sector`,
`gt_by_sector`, `sector_distributions`, `spx_weights`, `factor_correlation`,
`weight_sensitivity` (8), `data_quality`, `config_traps`.

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

## Model Portfolio removal (owner directive 2026-08-05)

The owner does not want the Model Portfolio in the dashboard; it does not serve
the buy/sell decision goal.

**Remove:** the `Model Portfolio` section and its `Portfolio Sector Allocation
vs S&P 500` chart from the dashboard UI, plus the `Top 5` framing if it is
merely a portfolio preview rather than a screening result.

**Do not blindly delete `portfolio_constructor.py`.** Check these first:

- `improvement_engine.record_run_snapshot()` takes `portfolio_df` and writes an
  `in_portfolio` column into every snapshot, and uses portfolio membership to
  compute **turnover**. Removing portfolio construction outright would break
  turnover tracking, which is part of the evidence base.
- The Excel workbook has a `ModelPortfolio` sheet, and the Methodology section
  documents it.
- `tests/test_portfolio_weighting.py`, `tests/test_portfolio_risk.py`, and
  parts of `tests/test_defensibility_improvements.py` cover it.

**Recommended:** remove it from the dashboard UI and the Methodology text, keep
the construction logic feeding snapshots/turnover/Excel. If a later session
shows turnover is unused, revisit. Record the decision in `NIGHTLY_LOG.md`.
