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
