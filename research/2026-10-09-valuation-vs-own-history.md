# Valuation against the stock's own history - 2026-10-09 (owner-run)

**Question.** The screener's valuation percentiles are cross-sectional: they say a stock's earnings
yield is higher than most of its sector's. A student deciding whether to look at a stock also asks
the other question - *is it higher than this stock usually is?* - and the page could not answer it
(north-star gap 3, `plan/dashboard-north-star.md`). This note says what the evidence supports, how the
answer is built, and how well the build reproduces an independent source.

**Decision.** Built as **context, not score** (CLAUDE.md settled row "ctx"): earnings yield and
free-cash-flow yield at each of the past 60 month-ends, from SEC filings, with today's place in that
range. Recorded every run in the context log so it builds its own out-of-sample record.

## What the literature supports, and what it does not

* **Most of a valuation gap between two companies is persistent.** Cohen, Polk & Vuolteenaho (2003,
  *Journal of Finance* 58(2), "The Value Spread") decompose the cross-sectional variance of
  book-to-market and attribute only **20-25%** to transitory differences in expected returns; the rest
  is expected profitability and the persistence of valuation levels. That is the case for showing a
  stock against itself: a large part of what the cross-sectional percentile measures is a lasting
  trait of the company, which its own range nets out.
* **Within-firm (time-series) valuation as a return predictor is not established.** Lewellen (1999,
  *JFE*, "The time-series relations among expected return, risk, and book-to-market") finds that
  book-to-market predicts time-variation in expected returns of portfolios, **but adds nothing once
  risk is controlled for**. Firm-level earnings-yield predictability has support in some samples and
  none in others (Ang & Bekaert 2007 find the market earnings yield forecasts cash flows, not
  returns). A search for a direct test of "a firm's yield relative to its own five-year range predicts
  its return" found none.
* **The obvious failure mode** (also the practitioner caveat, e.g. YCharts' "valuation from historical
  multiples"): a yield sits high in its own range when the market expects earnings to fall. Own-history
  comparison works for temporary price shocks and misleads for permanent ones.

**What this means for the screener.** Show it, say plainly that the predictive claim is unproven,
and do not score it. Its evaluation is the same as every context signal's: two log columns
(`_ctx_vh_ey_pct`, `_ctx_vh_fy_pct`) feed `context_eval.py`, and the first one-month window closes
2026-11-07. A score change needs a note *and* a measured record meeting the engine's own gate.

**Overlap with what is scored.** The Valuation category already carries earnings yield (20% of it) and
FCF yield (45%) cross-sectionally. This adds no new quantity, only a second reference point for the
same two, which is why it belongs beside the score rather than in it - scoring it would double-count
Valuation's inputs with a different denominator.

## Construction

`valuation_history.py`. Three rules, each the answer to a way this goes wrong:

1. **As first filed, and only what had been filed.** Each month-end uses facts with an SEC `filed` date
   on or before it, at their first-reported value (a later restatement does not rewrite history).
2. **Trailing twelve months** = latest fiscal year + current year-to-date - the same year-to-date a
   year earlier. Cash-flow statements are reported only year-to-date, so the construction works from
   YTD periods throughout. A TTM whose period ended more than 460 days earlier is not used.
3. **Splits.** Yahoo's monthly closes are split-adjusted; share counts are as reported. A share count
   filed before a split is multiplied by that split's ratio (exact split dates from Yahoo's split
   history). Market value = split-adjusted price x diluted weighted-average shares.

Inputs: the weekly SEC `companyfacts` cache (`sec_fundamentals.refresh_companyfacts`, which now also
keeps `WeightedAverageNumberOfDilutedSharesOutstanding` - else basic-and-diluted, else basic - and
`PaymentsToAcquireProductiveAssets`, the capex tag NVDA, AMZN and V use), and one batched monthly
price download cached in `data/valhist/` and refetched once a month. Run time: ~13 s for 503 stocks
after the cache is warm.

**The self-check that decides whether a stock is shown:** today's price x filed share count must
reproduce Yahoo's market capitalisation within **15%**. A company whose filings do not reproduce its
market value today cannot be trusted for the past. Banks and insurers get no FCF yield (operating cash
flow moves with loans, deposits and claims - the same `_is_bank_like` rule scoring uses).

## Measured, 2026-10-09 (this morning's 501-stock run)

| | |
|---|---|
| Stocks with a history | **456 of 503** (453 earnings yield, 327 FCF yield) |
| Today's market value vs Yahoo's | median gap **0.98%**, 90th percentile 3.7% (shown stocks) |
| Our earnings yield vs Yahoo's trailing EPS / price | Spearman **0.991**, median gap **0.04pp**, 88% within 0.5pp, 95% within 1pp (n=452) |
| Our earnings yield vs Yahoo's net income / market cap | Spearman **0.992**, 94% within 0.5pp (n=453) |
| Our FCF yield vs Yahoo's (OCF - capex) / market cap | Spearman **0.963**, 85% within 0.5pp (n=323) |
| Splits | NVDA 10:1 (2024-06-10): January 2023 reads 1.22% (~$6B TTM net income over ~$480B), no jump at the split. WMT 3:1 (2024-02-26) continuous across it |

**Not shown, and why (47):** 14 fail the self-check - share counts reported per class (GOOGL/GOOG
have too little diluted history; IBKR -74%, TKO -60% are one class of several) or mis-scaled in the
filing (MCD -100%, WAT +999x); 9 carry no share-count tag at all (BRK-B, V, XOM, KKR, ARES, AJG, STZ,
HONA, VYLR); 7 stopped tagging diluted shares years ago (HSY 2015, LYB 2019, BKR, ERIE, REG, SJM,
TTWO); 17 have under 36 months of history as a filer (spin-offs and new listings: GEV, KVUE, SOLV,
VLTO, SW, GEHC, Q, SNDK, CRH, FERG, BLK's new holding company, ...). Each gap is the honest answer
rather than a figure built from a guess.

**Largest disagreements with Yahoo** (ours vs trailing EPS/price): VTRS -17.3% vs -2.1%, GPN -4.2% vs
+2.6%, DOW, OXY - names with large one-off charges, where a one-quarter difference in the two
sources' trailing windows moves the yield a lot. Not investigated one by one; the page labels the
figure as built from the filings, not as Yahoo's.

## Sources

* Cohen, R., Polk, C. & Vuolteenaho, T. (2003). The Value Spread. *Journal of Finance* 58(2), 609-641.
  https://personal.lse.ac.uk/polk/research/jofi_5802005.pdf
* Lewellen, J. (1999). The time-series relations among expected return, risk, and book-to-market.
  *Journal of Financial Economics* 54(1), 5-43.
* Ang, A. & Bekaert, G. (2007). Stock Return Predictability: Is it There? *Review of Financial Studies* 20(3).
* YCharts, "The Valuation from Historical Multiples" (practitioner method and its caveat).
  https://go.ycharts.com/knowledge-base/the-valuation-from-historical-multiples
