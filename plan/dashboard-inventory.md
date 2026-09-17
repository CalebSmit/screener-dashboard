# Dashboard inventory (as of 2026-09-15)

**Read this before changing the dashboard.** There is far more in it than a
first look suggests, and the most common failure mode will be rebuilding
something that already exists. Refresh this doc when you materially change the
layout.

Generated from `index.html` (237,952 chars) and `dashboard_data.js`.

**Refresh this file whenever you change the layout.** It went stale on
2026-08-25 because the session that shipped the time dimension could not
edit it - `.claude/` was blocked as sensitive. The plans now live in
`plan/` precisely so that cannot happen again; there is no excuse for
leaving it wrong.

## Top-level views

| Section | Contents |
|---|---|
| **Top 5 Stocks** | Highest-composite names, card layout. Reads `table_data` directly, excluding trap-flagged names |
| **My Holdings** | The sell-side surface, added 2026-09-15. Client-side list, one card per saved name |
| **Factor Analytics** | `Factor Scores by Sector`, `Trap Rate by Sector` |
| **Defensibility & Diagnostics** | `How Stable Is the Ranking?` (weight sensitivity), `Are the Factors Independent?` (factor correlation) |
| **Full Universe Rankings** | The 501-row sortable table - the workhorse view |
| **Methodology** | A very large embedded explainer (~30 headings) |

Interactive elements are sparse: **2 charts** (`sector-dist-chart`,
`vt-chart`) and **1 table** (`universe-table`).

The **stock drilldown** (`openStockDetail`) is where most of the surface area
actually is, in this order: **Why It Ranks Here**, the score-card row, About,
Rank History, Analyst Price Targets, Company Snapshot, Sector Peers, Data
Provenance, Score Contribution Breakdown, and the eight category-detail
sections with their metric tables.

## My Holdings - the sell-side surface (2026-09-15)

Priority 5 / north-star gap 2. A `localStorage` list under
`screener_holdings_v1` holding **tickers and nothing else**, rendered as one
card per name: rank, composite, an eight-category score-and-delta strip, and
the `change` / `change_driver` / `flags` / `confidence` sentences lifted from
`stock_detail[t]["summary"]`. Above it, a concentration line (names, sectors,
largest sector share, count inside the top 25 and top 100, trap-flag count) -
the honest half of "how much / does it fit", since the list holds no weights.

**Added 2026-09-17:** the `input_churn` sentence joined `HOLDINGS_FACTS`, and a
cadence note now sits above the concentration line (see the two sections below).

**Zero payload cost.** It is a *view* over fields the payload already carried;
nothing was added to `stock_detail` for it.

**Three properties are research constraints, not styling.** All three have
tests, and the sources are in `research/2026-09-14-sell-discipline-and-hold-bands.md`
and `METHODOLOGY_CHANGELOG.md` 2026-09-15:

1. **Every saved name renders, every time** - never a filtered subset.
   Akepanidtaworn et al. (2023, *JF* 78(6)) trace an 80 bp/year institutional
   selling deficit to a restricted consideration set.
2. **Ordered by rank, never by size of move.** The rank change is shown for
   context; it is not the sort key and not a filter. Same source.
3. **No cost basis, share count or P&L anywhere**, in the code or in storage -
   Odean (1998). A hand-edited key containing a position dict is read for its
   ticker and written back clean. This is also why the panel works equally as a
   watchlist.

**There is no exit rule and no hold band, deliberately - and the reason written
here until 2026-09-17 was wrong on both numbers.** Corrected per
`METHODOLOGY_CHANGELOG.md` 2026-09-16 and §8 of the research note, which is where
the working is:

- "A 25/50 band fired zero times" came from walking **one path** through 18 runs.
  Over all comparable pairs a 2x band fires at **1.65-4.50%** of holding-looks.
- "60+ comparable runs" is the **wrong unit**. Pairs from a daily series overlap
  almost completely, so 34 runs are **2 independent monthly looks**, not 34
  observations - the same independence trap as the IC series.

**What is established:** a band *is* warranted (the strict top-25 rule wastes
**31-47%** of the trades it implies, at every cadence measured), but its **width
is not determinable** - at 1.4x the three cadences report 0.0%, 31.2% and 5.9%.
**The pre-registered rule binds:** no width until **>= 8 disjoint observation
windows at the review cadence the band will govern**; today **2 monthly**,
roughly **2027-04**. Do not pick 50 because MSCI doubles, and **do not reuse the
movers panel's threshold** - measured, it fires for a top-25 name 0.15% of the
time.

The empty panel ships **collapsed**; a saved list auto-expands it.
`tests/test_holdings_panel.py`, 61 tests (60 fail against the pre-change
generator); nine drive the emitted script under Node against a stubbed DOM.

## Review cadence, stated on the surfaces that move (2026-09-17)

`config.yaml -> portfolio.review_cadence` (a real key since 2026-09-17; it was a
bare comment before, which is why the page could not state it). Surfaced as
`D.cadence` and rendered by `cadenceText(long)` / `cadenceLine(long)` in three
places: the holdings panel above the concentration line (short form, both empty
and populated states), the What Changed footnote (short form), and the holdings
footnote (long form, with the turnover numbers and the Novy-Marx & Velikov
citation).

**The gap it closes:** the site is rebuilt every weekday and, until this landed,
said nothing about how often acting on it was intended. Acting on the strict
top-25 rule at every run implies **121.8%** monthly one-sided turnover against
**24.0%** at monthly review, where NMV (2016) find few anomalies survive costs
above **~50%**.

**It is a sentence, not a lock** - the tool does not know what a reader is doing.
Read from the *run's own* config snapshot, not the working tree, so a republished
old run states what it was configured for; `configured: false` marks the
quarterly fallback so it is distinguishable from a real setting.
`tests/test_review_cadence.py`, 40 tests (37 fail against the pre-change
generator); eleven drive the emitted script under Node.

## Input-availability churn - "is this move information?" (2026-09-17)

`history.py` records per-ticker metric availability per run and emits
`ch: [lost, gained]` on a delta entry when it changed; `stock_summary.py` turns
`>= 2` into an `input_churn` sentence, shown on the drilldown and on every
holdings row (amber left rule, `.holding-note-input_churn`).

**Why it exists:** when a metric percentile flips between present and absent, its
category renormalises over a different metric set and the score moves as
arithmetic, with no company event. That is `CLAUDE.md` priority 1.5's FCX case,
and it closes the product gap recorded there - the movers panel could not
distinguish "moved on new information" from "moved because two inputs went
missing".

**Three things not to tidy**, all with tests in `tests/test_input_churn.py`
(58 tests, 52 fail against the pre-change code):

1. **Arms at 2, not 1.** One changed metric moves the median rank by 7 against a
   baseline of 6 - noise. Two or more triples it to 21.
2. **Worded as a caveat, never as deterioration.** Churn leaves 52.4% of names
   worse off against a 44.6% base rate; it scatters ranks rather than pushing
   them down, and the sentence reads identically whether the stock rose or fell.
3. **Only columns both runs carry are compared**, and pre-2026-03-09 snapshots
   (15 columns, no percentiles) yield `None` rather than `(0, 0)` - "cannot tell"
   must not render as "nothing changed".

On the 2026-09-17 run it fires for **27 of 502** stocks against the ~1-month
baseline. Payload cost is one optional two-integer key on ~5% of delta entries.

## Why It Ranks Here - the deterministic summary (2026-09-08)

Built by `stock_summary.py` **at run time**, stored per stock as
`stock_detail[t]["summary"]` = `[{"k": kind, "t": sentence}, ...]`, rendered by
`renderSummary()` as the first block of the drilldown. **Eleven kinds**: `rank`,
`drivers`, `weakest`, `best_inputs`, `worst_input`, `change`, `change_driver`,
`target`, `peers`, `flags`, `confidence`. A kind is omitted when it cannot be
stated exactly, so a thin stock gets a shorter summary rather than a hedged one.

**`change_driver` was added 2026-09-15** and says *why* a stock moved, not how
far: the category that moved furthest since the baseline, the direction of its
**score** (stated explicitly, because a high Risk score means low risk), and
what that category now contributes. It and `change` read their baseline from
one place (`_pick_comparison`) so they cannot end up describing different
windows. Measured across the live payload it names Risk 34%, Revisions 29%,
Momentum 26%, Quality 0.2% - the fundamentals categories barely move between
quarterly filings. Cost: **+10.5 KB gzipped (+0.89%)**.

**Do not move this into the browser.** Building it here is what makes it
diffable and identical for every reader, which is the entire reason it replaced
the chat.

**Do not let advice language in.** `stock_summary.BANNED_TERMS` +
`advice_terms_in()` are checked against all 502 live summaries by
`tests/test_ai_chat_removed.py`. "Explains why it ranks there, never whether to
buy" is a north-star constraint with teeth, not a style note.

**It says "sector percentile" deliberately** - the percentiles are
sector-relative (`factor_engine.compute_sector_percentiles`), and dropping the
qualifier would publish a false claim about how the number was computed.

## Percentiles are direction-adjusted, and the page says so - 2026-09-11

**Read this before touching any percentile surface.** `compute_sector_percentiles()`
does `ranks = 100 - ranks` wherever `METRIC_DIR` is `False`, so a published
percentile **always means "best in its sector", never "largest"**. That is true
of 13 of the 37 metrics in `metric_meta` - on the live payload HON's EV/EBITDA
of 6.95 is the 99th percentile and AXON's 98.61 is the 0th.

Until 2026-09-11 nothing said so, and the natural reading of the drilldown was
exactly backwards. Three things now state it, and all three must stay:

1. **`metric_meta[m]["dir"]`** - `"higher"` or `"lower"`, **derived in
   `prepare_dashboard_data()` from `factor_engine.METRIC_DIR`.** Do not
   hand-write these. The derivation is what makes it impossible for the page to
   claim a direction the scorer disagrees with, and
   `tests/test_percentile_direction.py` compares the two on every build.
2. **The drilldown metric table** - header reads `Sector Percentile - 100 =
   best`, a `.pctile-convention-note` sits under it, and `dirChip()` renders a
   `↓ better` / `↑ better` chip beside every metric name. The note's "13 of 37"
   count is computed in JS from the payload, not written as a literal.
3. **The summary prose** - `_label_and_value()` appends `, lower is better` for
   inverted metrics only. Higher-is-better metrics are deliberately left plain:
   the ambiguity only exists where percentile and raw value point opposite ways,
   and the prose is ~101 KB gzipped across 502 stocks.

Total cost measured at **+1.2 KB gzipped (+0.1%)**. No score, rank, `raw` or
`pct` value changed - verified cell-by-cell across all 502 stocks.

**The eight category columns also carry definitions now** (`title` on each
`<th>`), naming their scored metrics with weights, the bank carve-out for
Valuation/Quality, and the fact that a **high `Risk` score means low risk**.

**Name only metrics that actually carry weight.** Draft tooltips listed P/B
under Valuation and PEG under Growth; both are weight 0 for non-banks. Read
`config.yaml`'s active weights rather than `CAT_METRICS`, which includes
zero-weight candidates. A test pins this.

## The "Screener AI" chat is gone - DONE 2026-09-08

Owner directive 2026-08-10, priority 4, open 29 days. Removed: the chat FAB and
panel, the Chat Settings dialog (API-key field + model picker), 27 JS functions,
three keyframe blocks and the `AI CHAT PANEL` stylesheet - 891 lines, and 921
lines off `generate_dashboard.py`. The `config_traps` payload key went with it:
its only consumer was the chat's system prompt, and the same thresholds are
already in the Methodology section.

It required each visitor to paste an Anthropic API key into `localStorage` and
called `api.anthropic.com` from the browser - unusable for a student club,
costly per question, a credential-phishing-shaped form on a public page, and
un-reproducible, which is the one that decided it. Changelog 2026-09-08.

`tests/test_ai_chat_removed.py` (75 tests, 58 failing against the pre-change
generator) pins **both halves**: no chat symbol, element id, model id,
`localStorage` key or provider URL survives, *and* the summary block renders. A
partial swap is the dangerous state - a dangling identifier blanks the page with
all four ship gates green.

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
| `stock_detail` | 2.81 -> ~4.3 | **~88% of the payload.** All 502 stocks. Grew 2026-08-26 with `about`, 2026-09-08 with `summary` (+0.67 raw), 2026-09-15 with `change_driver` (+0.10 raw / +10.5 KB gz) |
| `history` | 0.27 | Added 2026-08-25. 18 accepted run dates, 2 excluded |
| `table_data` | 0.26 | 502 rows, 8 category scores + composite/rank/flags |
| everything else | <0.02 | `portfolio` (0.010) and `spx_weights` removed 2026-08-26 |

**Raw size is the wrong number to optimise.** Pages serves gzip, and the
payload compresses ~4x overall. The business descriptions add ~0.71 MB raw but
only ~60 KB on the wire. Measure gzip before calling anything expensive.

**But do measure.** The 2026-09-08 summaries compress only **6.6x** (665 KB raw
-> 101 KB gzipped), well short of the ~11x the earlier prose achieved, because
every stock's sentences carry different numbers and gzip's 32 KB window cannot
match far back. Wire payload went **1,078 -> 1,179 KB**; removing the chat gave
back 11 KB of page, so the net was **+90 KB (+8%)**. That was judged worth it -
see `plan/dashboard-north-star.md` - but it is the largest single addition since
`about`, and the next thing added should be weighed against a phone on 4G.

`stock_detail` dominates. Per stock: `raw`, `pct`, `cat_scores`, `contrib`,
`composite`, `rank`, `sector`, `company`, `industry`, `about`, `summary`,
`vt`/`gt`, `price`, `pt_mean/high/low`, `num_analysts`, `eps_mismatch`,
`eps_ratio`, `data_source`, `metric_count`/`metric_total`, `financials`,
`flags`, `peers`, `self_metrics`.

**`raw` is the value as fetched, from 2026-09-01.** Until then the pipeline
winsorized every metric at the 1st/99th percentiles immediately before ranking
it, and `raw` carried the *clipped* number — so the site published AAPL, NVDA,
MSFT, GOOG, GOOGL and AMZN with one identical market cap of $2,802.0B against a
true $5,331.2B for NVDA. 301 cells across 33 metrics on 159 stocks were wrong
the same way. Clipping could never have helped, because `pct` is a rank and a
rank is invariant under monotone transforms. Removed; changelog 2026-09-01,
`tests/test_no_winsorization.py`. If a metric ever again shows a cluster of
identical values at its extremes, that is the defect returning.

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
`weight_sensitivity` (8), `data_quality`, `history`.

`portfolio` and `spx_weights` were removed 2026-08-26 - see below.
`config_traps` was removed 2026-09-08 with the chat that was its only consumer.

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
2. ~~Any sell-side workflow.~~ **MOSTLY SHIPPED 2026-09-15** - see the My
   Holdings section above. What remains is the **hold band**: the screener
   still has one test (top 25) where the evidence says entry and continued
   holding should use different, asymmetric tests. Blocked on measurement, not
   on design - re-measure the band at 60+ comparable runs (32 today).
3. **Time-series valuation context.** `pct` is cross-sectional only.
4. **Catalyst/earnings-date proximity.**
5. ~~Per-stock confidence surfaced.~~ **MOSTLY SHIPPED 2026-09-08.** The
   summary's `confidence` and `target` sentences state metric coverage ("rests
   on 12 of 18 metrics"), name any withheld category and the reweighting it
   caused, flag stale filings with their age, flag an EPS-basis mismatch, and
   call out a thin analyst base (<5). What is still not legible anywhere is
   `data_source` (quarterly vs annual).
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
