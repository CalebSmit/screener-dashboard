# The fundamental reporting lag: when accounting data becomes usable

**Date:** 2026-10-05 (Monday research session)
**Supports:** `plan/backtest-v2.md` step 3 — the 49.0pp of composite weight that
needs point-in-time filings (`research/2026-10-01-lookahead-bias-size.md`)
**Measurement:** `research/measurements/2026-10-05-edgar-reporting-lag.py`
(output JSON committed beside it)

## The question

When a fiscal period ends, how long until its financial statements are public —
and therefore what alignment rule should backtest v2 use between fundamentals
and prices? One sentence earlier than that: *is a fixed-lag convention (Fama &
French's 6 months, Hou-Xue-Zhang's 4) the right tool here, or should v2 use
actual filing dates?* This decides how the largest remaining bias bucket
(49.0pp of weight, part of the >= 63.2% look-ahead) gets fixed.

## What the literature actually says

**1. Fama, E. F. & French, K. R. (1992), "The Cross-Section of Expected Stock
Returns", *Journal of Finance* 47(2), 427-465.** The canonical alignment rule:
accounting data for fiscal years ending in calendar year *t-1* is matched to
returns from **July of year *t*** — a minimum six-month gap, so that every
filing used was already public at portfolio formation. Conditions: NYSE/Amex/
NASDAQ non-financials, 1962-1989, annual COMPUSTAT data. The six months was a
**conservative guess made when machine-readable filing dates did not exist**;
the paper itself calls the choice conservative. It is the convention most of
the subsequent factor literature inherited.

**2. Hou, K., Xue, C. & Zhang, L. (2020), "Replicating Anomalies", *Review of
Financial Studies* 33(5), 2019-2133**, and the q-factor technical document
(global-q.org, Feb 2024 revision). The modern refinement, on US all-but-
microcap data 1967-2016. Two different rules by data type:

- **Earnings items**: usable from the **earnings announcement date** (Compustat
  quarterly item RDQ) — because earnings are public at the press release,
  weeks before the SEC filing.
- **Other quarterly items** (balance-sheet, cash-flow): a **4-month lag** from
  fiscal quarter end, because comprehensive statement data arrives only with
  the filing — they quote a median reporting lag of **46 days (NYSE/Amex) and
  52 days (NASDAQ)**, and that only **~37% of quarterly earnings announcements
  include balance sheet information** (citing Chen, DeFond & Park 2002, *JAE*).

So the state of the art in academia is already two-tier: *announcement date
where you have it, deliberate over-lag where you don't.*

**3. Asness, C. & Frazzini, A. (2013), "The Devil in HML's Details", *Journal
of Portfolio Management* 39(4), 49-68.** The other side of the same ratio:
standard annually-rebalanced HML holds **both** book value and price fixed, so
by the time the portfolio is reformed the price inside the "value" ratio is up
to **18 months old**. Lagging the book value (necessary — it is not public
sooner) while updating the **price** monthly ("HML Devil") produces B/P ratios
that better forecast the true unobservable ratio, and value portfolios on the
timely measure earn **305-378 bps/yr of alpha against five-factor models**
(US + 19 international markets, 1950s-2011 depending on sample). The
construction principle: **lag the fundamental because you must; never lag the
price, because you don't have to.** This is precisely the restatement backtest
v2 already plans for the 28.0pp price-restatable bucket — the plan's next step
is not just a bias fix, it is the construction the literature recommends on
its own merits.

**4. Ljungqvist, A., Malloy, C. & Marston, F. (2009), "Rewriting History",
*Journal of Finance* 64(4), 1935-1960.** Bears on the 9.0pp analyst-estimate
bucket. Across seven downloads of the I/B/E/S recommendations history taken
2000-2007, **between 1.6% and 21.7% of matched observations differed from one
download to the next** — recommendations altered, records added and deleted,
analyst names removed. A vendor's historical estimates file is not a stable
record of what was believed at the time. Implication: even a *paid*
retrospective source for `forward_eps_growth` / `fy1_revision_3m` /
`analyst_surprise` history would need its own point-in-time audit before a
backtest could trust it; a free one does not exist. Expect this bucket to be
reported as **permanently unmeasurable** rather than approximated.

## Documented practice

- **SEC deadlines are the binding upper bound, and for this universe they are
  tight.** Large accelerated filers (public float >= $700M — effectively every
  S&P 500 constituent) must file the 10-K within **60 days** of fiscal year end
  and the 10-Q within **40 days** of quarter end (Exchange Act Rules 13a-13/
  15d-13; deadline calendars published by Mayer Brown and others). The 6-month
  and 4-month academic conventions are 3-6x looser than the legal deadline for
  these firms.
- **The practitioner product is filing-date data, not a lag convention.** S&P
  Global sells Compustat Point-in-Time snapshots "as they appeared at the end
  of any month" explicitly to avoid look-ahead (cited with URLs in
  `research/2026-10-01-lookahead-bias-size.md`). Practitioners who pay, pay
  for *dates*, not for a smarter lag.
- **SEC EDGAR gives the dates away free for this universe.** The XBRL
  `companyconcept` / `companyfacts` APIs return every disclosed fact with
  `accn`, `end`, `filed`, `form`, `fp`, `fy`, `val`. Verified today with a
  live call (`CIK0000320193/us-gaap/Assets.json`): **146 USD facts, every one
  carrying both `filed` and `end`** — the JSON check is in the measurement
  output. This is the free equivalent of the thing practitioners buy, for
  as-first-reported values with their publication dates.

## The measurement: what the lag actually is for this universe

`research/measurements/2026-10-05-edgar-reporting-lag.py` — deterministic
sample (sorted tickers, every 8th: 63 of 503 names, all resolved to CIKs),
all 10-K/10-Q filings with period end >= 2019-06-30, lag = filingDate −
reportDate from EDGAR's submissions API:

| Form | filings | median | p10 | p90 | max | share past SEC deadline |
|---|---|---|---|---|---|---|
| 10-K | 386 | **48 days** | 36 | 58 | 91 | 1.6% (of 60) |
| 10-Q | 1,201 | **32 days** | 23 | 39 | 49 | 1.0% (of 40) |

**Independent observations, not rows** (the standard this project keeps
re-learning): filings from one company share its filing habits, so the honest
unit is the company. Per-company medians: 10-K **median-of-medians 47.5 days**
across 62 companies (slowest company median 59 — STE); 10-Q **33.0 days**
across 63 (slowest 42). The cross-company spread is narrow: p10-p90 of company
medians is 36-57 (10-K) and 25-37 (10-Q).

**Where the measurement disagrees with the literature's numbers:** HXZ's 46-52
day median is for *earnings announcements*, economy-wide, over 1972-2016
including small caps. Today's S&P 500 files the **complete 10-Q at a median of
32 days** — faster than the old economy-wide earnings announcement itself. A
4-month lag applied to this universe would discard a month-plus of genuinely
public data on every quarterly fact; a 6-month annual lag discards even more.
The conventions are not wrong — they were designed for universes and eras
without reliable filing dates — but they are the wrong tool when the actual
date is free.

## Where the evidence contradicts what we currently do

- **`backtest.py` v1 applies a *negative* lag of up to 80 months** — measured
  2026-10-01 (>= 63.2% of name-months change decile). This note does not
  re-argue that; it fixes the design of the repair.
- **The live screener is correct as-is**: it scores on vendor-current data at
  each run, which is the Asness-Frazzini construction (current price, latest
  public fundamental). No change to the live path is implied by anything here.
- **`plan/backtest-v2.md` step 3 needs a lag rule and this note supplies it**
  (see Recommendation). Nothing in config or code currently encodes one.

## What would change our mind

- If EDGAR XBRL coverage for the specific concepts behind the filings-fed
  metrics turns out spotty before ~2021 (XBRL phase-in is complete for
  large filers well before 2019, but concept-level tagging varies), the
  deadline-based fallback does more work than expected and its share should be
  reported, not hidden.
- If late filings cluster in exactly the stress months a backtest cares about
  (the 1.6%/1.0% deadline misses were not checked for time-clustering), the
  fallback could inject look-ahead in the worst months. The re-run should
  group `share_over_deadline` by year before the fallback is trusted.
- If a re-measure on the full 503 names moves the medians materially (the
  stride-8 sample is deterministic but 1/8th of the universe), widen the
  sample. The script takes ~1 minute at 63 names; the full universe is ~8.

## Recommendation (for Wednesday's synthesis and Thursday's build)

1. **Backtest v2 aligns filings-fed fundamentals by per-fact `filed` dates
   from EDGAR `companyconcept`**: a fact is usable at rebalance month *m* iff
   `filed` <= month-end(*m*). No fixed-lag convention. This is both stricter
   and more faithful than FF-6m/HXZ-4m — zero look-ahead, zero avoidable
   staleness, and it matches what the live screener does (use data as soon as
   it is public), so v2 backtests the construction the site actually runs.
2. **Fallback where a `filed` date is missing**: the large-accelerated-filer
   deadline — 60 days for annual facts, 40 for quarterly. Measured today, that
   covers ~98.5% of this universe's filings honestly; report the fallback's
   usage share in v2 output so a reader can see how much rests on it.
3. **Income-statement items via filed dates are mildly conservative** (public
   at the press release ~2-4 weeks earlier — the RDQ point). Accept it: the
   error direction is safe (staleness, not look-ahead), the size is bounded by
   the press-release-to-filing gap, and it avoids needing RDQ, which EDGAR
   does not carry.
4. **The 28.0pp price restatement keeps its priority and gains a second
   justification**: it is not only the removal of 34.1 free bias points, it is
   the Asness-Frazzini timely-price construction with documented alpha of
   305-378 bps/yr in value portfolios. Lag the fundamental, never the price.
5. **Plan for the 9.0pp estimates bucket to be reported as unmeasurable**, per
   Ljungqvist et al. — and say in v2's output that its weight is carried at
   snapshot values, clearly marked, rather than silently frozen.

### Wednesday design section (hypothesis / sketch / refutation)

- **Hypothesis**: aligning the Quality/Growth/Investment inputs by `filed`
  date changes historical decile assignments materially less than the price
  restatement did (the fundamentals drift slowly — quarterly, with ~32-48 day
  lags — while prices moved the panel by 63.2%), but the *reporting lag
  itself* is not the main event: the main event is restating the fundamental
  *values* to their as-of-month levels, for which the `filed` date is the
  join key.
- **Sketch**: `lookahead.NEEDS_POINT_IN_TIME` holds **27 metrics** (checked
  against the module today), and they are not all filings-fed: the bucket also
  carries the 3 analyst-estimate metrics plus `analyst_rating`, two
  short-interest metrics (exchange data, not filings) and `insider_ownership`
  (Form 4/proxy, not XBRL). The first task of the sizing step is the honest
  split of that list by *source*, with weights from `weight_buckets()`. For
  each genuinely filings-fed metric, map to its us-gaap
  concept(s); pull `companyconcept` per name (one request per concept per
  name, cacheable and rate-limited exactly like today's script); build a
  (name, concept, end, filed, val) panel; at each rebalance month select the
  latest fact with `filed` <= month-end. The 2026-10-01 measurement harness
  (arm A/arm B, null + fidelity checks) is the template for sizing what this
  changes.
- **Refutation**: if the null check (facts selected at the snapshot month
  reproduce the live values) fails, the concept mapping is wrong — yfinance's
  definitions and us-gaap tags will not agree everywhere (e.g. EBITDA is not
  an XBRL concept; it must be composed). Expect the mapping, not the dates,
  to be where this gets hard. If more than ~a third of the 49.0pp cannot be
  mapped to tagged concepts, the honest report is a smaller measurable share,
  not a forced mapping.

## What this note does *not* rest on

No backtest number and no IC figure justifies anything here (rules 4 and 5).
The `1m` horizon holds **4 effective** observations against a gate of 8. Every
number in this note is either a published finding (cited), a regulatory
deadline (public record), or a property of EDGAR filing metadata measured by
the committed script. No production code changed; no methodology changed; no
stock's score moves.

## Sources (accessed 2026-10-05)

- Fama & French 1992: *Journal of Finance* 47(2), 427-465.
- Hou, Xue & Zhang 2020: *RFS* 33(5), 2019-2133; technical doc:
  http://global-q.org/uploads/1/2/2/6/122679606/factorstd_2024feb.pdf
- Asness & Frazzini 2013: *JPM* 39(4), 49-68;
  https://www.aqr.com/Insights/Research/Journal-Article/The-Devil-in-HMLs-Details
  (alpha figures 305-378 bps/yr from the paper's abstract/results as
  summarised on the publisher page https://www.pm-research.com/content/iijpormgmt/39/4/49)
- Ljungqvist, Malloy & Marston 2009: *Journal of Finance* 64(4), 1935-1960;
  https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2009.01484.x
- SEC deadlines: Exchange Act Rules 13a-13/15d-13; e.g.
  https://www.mayerbrown.com/-/media/files/perspectives-events/publications/2025/12/2026-sec-filing-deadlines-and-financial-statement-staleness-dates.pdf
- EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- This repo: `research/2026-10-01-lookahead-bias-size.md`,
  `research/measurements/2026-10-05-edgar-reporting-lag.py` + `.json`.
