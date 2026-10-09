# Valuation against the stock's own history (north-star gap 3) - design notes, 2026-10-09

**Status: BUILT 2026-10-09 (owner-run, later the same day)** - `valuation_history.py`, route 1's
spirit with diluted weighted-average shares (split-adjusted by filing date) instead of per-share EPS.
Evidence, construction and validation: `research/2026-10-09-valuation-vs-own-history.md`. The notes
below are the design as first written.

**The question it answers:** "is this stock's earnings yield (or FCF yield, EV/EBITDA) high or low
*against its own past five years*?" - `pct` is cross-sectional only, so today the page can say a
stock is cheaper than its sector but not cheaper than it usually is. Display-only context, like the
"Before you decide" panel; never scored without its own note and changelog entry.

**Data now available:**
- Five-plus years of fundamentals *as filed* per month: `pit_fundamentals.PointInTime` over the
  weekly SEC `companyfacts` cache (`data/sec/pit/facts.parquet`) - net income, revenue, OCF, capex,
  debt, cash, D&A; ~95% coverage for the core inputs.
- Monthly prices: one batched yfinance download for ~500 names x 5 years, cacheable like
  `data/track/prices.parquet`.

**The trap - stock splits.** A market value history needs price x shares at each month. yfinance
prices are split-adjusted; SEC share counts are as reported. Multiplying an adjusted price by a
pre-split share count overstates past market value by the split ratio (NVDA's 10-for-1 in 2024 would
make its 2023 earnings yield look ten times too low). Three safe routes, in order of preference:
1. Work **per share on both sides**: EPS as filed (`EarningsPerShareDiluted`, unit USD/shares - add
   it to `FACT_CONCEPTS`, which today keeps USD units only) divided by an *unadjusted* price, and
   adjust both for splits using yfinance's `Ticker.splits` history.
2. Use **current share count with adjusted prices** and say so - simple, but it ignores buybacks
   (Apple's count fell ~20% in five years, which biases its past yields low).
3. Skip market value entirely and show **fundamental trends** (margins, ROE) against their own
   history, which needs no price at all.

**Wording constraint:** `BANNED_TERMS` forbids "cheap"/"expensive"; write "a higher earnings yield
than at 72% of month-ends in the past five years".
