# Dashboard north star - the one place you look before buying or selling

**Owner directive (2026-08-05):** make this the single place to look when
considering buying or selling a stock. Add what helps, remove what doesn't,
research first. You have authority to delete existing features.

## The decision it must support

A user arrives with one of four questions. Every element on the dashboard
should earn its place against at least one:

1. **What should I look at?** - surface candidates worth attention
2. **Should I buy this one?** - evaluate a specific name on the merits
3. **Should I sell what I hold?** - detect deterioration in something owned
4. **How much / does it fit?** - position sizing and portfolio fit

Anything that answers none of these is reference material. Reference material
is not worthless - it is what makes the tool defensible and teachable - but it
belongs behind a tab, not in the decision path.

## The boundary that keeps this defensible

This is **decision support, not a recommendation engine.** The distinction is
not legal throat-clearing; it is the product's whole credibility, and it is
what makes it appropriate for a college investment club.

- Show *why* a stock ranks where it does, with the inputs visible and
  attributable. Never emit a bare "BUY".
- Every number traceable to its source and as-of date.
- Uncertainty shown, not hidden - coverage gaps, stale data, thin analyst
  counts, low-confidence scores.
- The user reaches the conclusion. The tool supplies the evidence.

A screener that says "buy this" is unfalsifiable and indefensible. A screener
that says "this ranks 4th on valuation because FCF yield is in the 91st
percentile, computed from quarterly data as of 12 days ago, and here are its
five closest sector peers" is a research tool. Build the second one.

## What already exists

Check before building. `stock_detail` covers **all 501 stocks** (not just
holdings) with: `raw` metric values, `pct` percentiles, `cat_scores`,
per-category `contrib` attribution, `peers`, analyst price targets
(`pt_mean/high/low`, `num_analysts`), `financials`, `flags`, `data_source`
provenance, and `metric_count/metric_total` coverage.

Rendered sections include: Full Universe Rankings, Model Portfolio, Top 5,
Company Snapshot, Score Contribution Breakdown, Sector Peers, Analyst Price
Targets, Data Provenance, Factor Correlation Matrix, Weight Sensitivity,
Factor Scores by Sector, Trap Rate by Sector, Factor-Exposure Diagnostics, a
large Methodology section, and a "Screener AI" chat.

**The dashboard is already strong on question 2.** Its real gaps are elsewhere.

## The gaps that actually matter

**1. There is no time dimension. This is the biggest one.**
Every view is a snapshot. Buy and sell decisions are overwhelmingly about
*change*: "this fell 60 ranks since last month - what broke?", "quality has
been deteriorating three runs running", "revisions just turned positive".
Nothing on the dashboard can answer that today.

This is now buildable: the data loop writes a snapshot every run into
`improvement/snapshots/`, and `improvement/performance_history.csv` already
exists. Rank/score deltas between runs, sparklines per category, and a
"biggest movers" view are all reachable from data you will be accumulating
from day one. **Start here.**

**2. There is no sell-side workflow.** The tool is built to find candidates,
not to monitor what you own. Question 3 is currently unanswerable. Consider: a
client-side watchlist/holdings list (localStorage, no backend), deterioration
flags against it, and a "review queue" of owned names whose scores dropped
materially.

**3. Valuation is only cross-sectional.** `pct` says cheap *versus peers*. It
never says cheap *versus its own history*. A stock in the 90th percentile on
EV/EBITDA across the S&P may still be at the top of its own 5-year range.
Time-series percentiles would be a genuine analytical addition - and they need
the same historical spine as gap 1.

**4. Nothing about timing or catalysts.** Earnings date proximity materially
changes whether today is the day to act. Cheap to add, high decision value.

**5. Confidence is uneven and mostly invisible.** `metric_count/metric_total`
and `num_analysts` exist per stock but a user cannot easily see "this score
rests on 11 of 18 metrics and 2 analysts". Surfacing per-stock confidence is
both a decision aid and a defensibility win.

## Removal criteria

You may delete things. Justify removal in `NIGHTLY_LOG.md` against these:

- **Redundant** - another element answers the same question better
- **Un-actionable** - interesting but changes no decision
- **Un-maintained** - drifted out of sync with actual behaviour
- **Cognitively expensive** - its screen space costs more than it returns

Two explicit cautions. First, **do not delete the defensibility features**
(weight sensitivity, factor correlation, provenance, trap detection) - rule 6
in `CLAUDE.md`. They may look like clutter on a decision surface; they are the
reason anyone should trust the decision surface. Move or restyle them, don't
remove them. Second, evaluate the "Screener AI" chat honestly against the
criteria above - if it cannot ground its answers in the payload and cite them,
it is a liability on a tool whose selling point is auditability.

## Replace the chatbot with generated summaries (owner directive 2026-08-10)

**Remove the "Screener AI" chat. Put per-stock summaries in its place.**

### Why the chat has to go

It requires each visitor to paste their own Anthropic API key into
`localStorage`, then calls `api.anthropic.com` from the browser. For the
investment-club audience that means: every student needs a paid API account
(most won't), it teaches people to paste credentials into web pages, it costs
them money per question, and two students asking the same question get
different answers. That last one is the killer - the tool's claim is
auditability, and an un-reproducible explainer works against it.

### What replaces it

**Build the deterministic version first. It may be all you need.**

The payload already carries everything required: `contrib` (per-category
contribution to the composite), `pct` (metric percentiles), `peers`,
`financials`, `flags`, `pt_mean/high/low`, `num_analysts`,
`metric_count/metric_total`. A template over those fields produces something
like:

> **HST - ranks 1 of 502.** Driven by Valuation (94th pct, contributing 22.4 of
> its 78.0) and Momentum (98th pct, 8.9). Weakest: Growth (49th pct, 6.4).
> No trap flags. $22.67 against a mean analyst target of $25.07 (+10.5%, 20
> analysts). Cheaper than its Real Estate peers on EV/EBITDA (10.2x vs 14.1x
> median). Score rests on 18 of 18 metrics.

Every number traceable, identical for every viewer, free, instant, and it
cannot hallucinate. That is decision support.

**Optional second layer:** a short plain-English gloss written by the nightly
session at build time and baked into the payload - useful for the teaching
audience ("what does a 94th percentile FCF yield actually mean?"). If added, it
must be constrained to restating facts already in the deterministic block, and
generated **at build time, not in the browser**, so it ships reviewed and
identical for everyone.

### The line that must not be crossed

A summary explains **why a stock ranks where it does**. It never says whether
to buy it.

- Good: "ranks 1st, driven by valuation and momentum; growth is its weak point"
- Bad: "attractive entry point", "undervalued", "a strong buy"

The second kind is a recommendation, is unfalsifiable, and would make this a
liability for a student investment club rather than a teaching tool. See
"The boundary that keeps this defensible" above.

### Scope

Summaries for the **top ~25 and any portfolio holding** first, not all 502 -
that is where attention actually goes, and it keeps the payload from growing.
Extend later if it proves useful.

Also worth adding, and related to the time-dimension gap: a **run-level
overview** at the top of the page - what changed since the last run, biggest
rank movers, anything newly flagged. That is the single most decision-relevant
thing the dashboard could gain.

## Candidate improvements (not yet prioritised - pick by value, not order)

Recorded 2026-08-10. Each has been checked against what the payload and fetch
already provide, so the cost estimates are real rather than guessed.

### Cheap - the data is already there

**1. Business description in the drill-down.** (owner suggestion)
`factor_engine.py` already calls yfinance `info` and pulls `industry` from it
(line ~617). `longBusinessSummary` is in that *same* response, so capturing it
costs **zero extra API calls**. "What does this company actually do?" is the
first question a student asks and the dashboard cannot currently answer it.
*Caveat:* full summaries average ~1.2 KB, so 502 of them add ~600 KB to a
3 MB payload. Truncate to the first two sentences (~300 chars) or lazy-load
per stock.

**2. Surface `industry`.** Already fetched, used only for bank-like detection,
never shown. Sector is too coarse - "Information Technology" spans a chip
fabricator and a payroll processor. ~20 bytes per stock.

**3. A metric glossary.** `metric_meta` carries `label`, `fmt` and `category`
for 36 metrics but **no explanation**. Add a one-line plain-English definition
and a note on why it is in the model - roughly 5 KB of static text for the
whole registry, shown on hover or tap. This is the single cheapest thing that
would make the tool teachable, and it directly serves the investment-club
audience.

### High decision value

**4. "Why is this NOT in the portfolio?"** A stock can rank in the top 20 and
still be excluded - by a sector cap, a position cap, or a trap flag - and the
dashboard shows no reason. Fully deterministic from data already present, and
it answers a question a user will definitely ask.

**5. "What would have to change?"** For a holding, how far would a category
score need to fall before it dropped out of the top 25? Computable exactly from
the weights and the current distribution. Turns a static ranking into something
that supports a *sell* decision, which is the workflow the tool most lacks.

**6. Per-stock confidence, made legible.** `metric_count/metric_total`,
`num_analysts`, `data_source` (quarterly vs annual) and `eps_mismatch` are all
in the payload and effectively invisible. "This score rests on 11 of 18 metrics
and 2 analysts" changes how much weight a reader should put on it. Ties to the
16 stocks currently missing a size score - a gap that today is silent.

### Investment-club specific

**7. A printable one-page tear sheet.** At a club meeting somebody presents a
stock. A clean print/PDF view per name - scores, contributions, peers, targets,
the business description - would get real use and is mostly CSS `@media print`.

**8. Mobile layout.** A 502-row table is unusable on a phone, and "a student
checking before a club meeting" is the stated use case. Card view under a
breakpoint, with the full table on desktop.

## Research questions for Monday sessions

Roughly in priority order. One per week is plenty.

1. **What do practitioners actually look at before a buy/sell decision?**
   Study real research terminals and tear sheets. What is on the first screen?
   What did they choose to leave off, and why?
2. **How should score changes over time be presented** so they inform rather
   than encourage overtrading? There is real literature on turnover, tracking
   error, and rebalancing costs - a dashboard that provokes churn actively
   loses money.
3. **What sell disciplines have evidence behind them?** Momentum-based exits,
   fundamental deterioration triggers, valuation-based trims. What actually
   survives out-of-sample?
4. **Time-series vs cross-sectional valuation** - which better predicts forward
   returns in US large caps, and does combining them add anything?
5. **How do you present uncertainty to a non-expert** without either hiding it
   or paralysing the user? Directly relevant to the investment-club audience.

## Constraints

- Front-end changes go in `generate_dashboard.py`, never in generated files.
- No backend. Static hosting on GitHub Pages. Anything stateful is
  `localStorage`, and say so plainly - a student should not think their
  watchlist is stored somewhere safe.
- Payload is already ~3 MB. Historical data will grow it. Watch load time on
  a phone; lazy-load or downsample rather than shipping everything eagerly.
- Mobile matters. A student checking before a club meeting is on a phone.
